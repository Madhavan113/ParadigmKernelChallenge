# okc: problem=muse_attention_decode
"""Gated attention decode for Muse Spark Open, as one Triton kernel per step.

Grid: (batch x kv_heads) x SPLITS. A CTA owns one request, one KV head (so the 8 query heads
of its group, times the block's L tokens: 8 or 64 rows) and one slice of that request's keys:

  * it normalizes and (on sliding layers) rotates its query rows and the block's new keys,
    quantizes the new keys/values to E4M3 (split 0 writes them to the cache), and attends
    over its key slice straight from the paged FP8 cache with an online softmax
    (tensor-core dots in BF16, fp32 statistics);
  * split 0 also attends over the block's own new tokens (causal inside the block);
  * the partial (max, sum, acc) rows go to a workspace and the LAST split to finish (an
    acq_rel ticket at gpu scope) merges them, applies the per-head RMSNorm x sigmoid(gate)
    and writes the output. No second launch, no memset: the ticket resets itself.

Keys are read exactly once per step; nothing is skipped: every key in the window (sliding)
or the whole context (full) enters the softmax.
"""

import math
from typing import TypedDict

import torch
import triton
import triton.language as tl


class Config(TypedDict):
    batch: int
    q_len: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    window: int
    rope_theta: float
    qk_scale_factor: float
    sm_scale: float
    kv_scale: float


class Inputs(TypedDict):
    qkv: torch.Tensor
    positions: torch.Tensor
    cache_loc: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    scheduler_metadata: torch.Tensor | None
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    config: Config


# == BEGIN SHARED attn ==
RMS_EPS = tl.constexpr(1e-5)
OUT_DTYPE = torch.bfloat16
ACT_DTYPE = tl.float16    # the dots' operand dtype: exact for the E4M3 cache values and finer than BF16
FP8_ASM = True            # hardware e4m3x2 -> f16x2 conversion (False: generic cast, for the interpreter)
QK_FP8_ROWS = True        # 64-row (verify) tiles: E4M3 queries and keys straight into the QK^T MMA
NEG = tl.constexpr(-1.0e30)  # finite "minus infinity" for masked logits (keeps every exp2 finite)
LOG2E = 1.4426950408889634


@triton.jit
def _cache_to_act(x8, ACT: tl.constexpr, FP8_ASM: tl.constexpr):
    """FP8 E4M3 cache bytes -> the dots' operand dtype. On the GPU, one cvt.f16x2.e4m3x2 per
    pair (exact); Triton's generic cast would spend three conversions per element."""
    if FP8_ASM:
        b = x8.to(tl.uint8, bitcast=True)
        # 4 bytes in one register -> 4 fp16 in two (elements 0,1 then 2,3): no repacking
        return tl.inline_asm_elementwise(
            asm="{ .reg .b16 lo, hi; mov.b32 {lo, hi}, $2; cvt.rn.f16x2.e4m3x2 $0, lo; cvt.rn.f16x2.e4m3x2 $1, hi; }",
            constraints="=r,=r,r", args=[b], dtype=tl.float16, is_pure=True, pack=4)
    else:
        return x8.to(ACT)


@triton.jit
def _rms_rope(xe, xo, rmask, pos, inv_freq, scale, ROPE: tl.constexpr, D: tl.constexpr):
    """Per-row RMSNorm of the 64-wide row split into even/odd halves [R, 32], scaled, then
    interleaved RoPE (pairs (2i, 2i+1) rotate by pos * theta^(-2i/64)) when ROPE."""
    ss = (tl.sum(xe * xe, 1) + tl.sum(xo * xo, 1)) / D
    inv = tl.rsqrt(ss + RMS_EPS) * scale
    xe = xe * inv[:, None]
    xo = xo * inv[:, None]
    if ROPE:
        ang = pos.to(tl.float32)[:, None] * inv_freq[None, :]
        c = tl.cos(ang)
        s = tl.sin(ang)
        xe, xo = xe * c - xo * s, xe * s + xo * c
    return xe, xo


@triton.jit
def _attn_decode_kernel(
    qkv_ptr, loc_ptr, seq_ptr, pt_ptr, kc_ptr, vc_ptr, kscale_ptr, vscale_ptr,
    out_ptr, part_ptr, ml_ptr, cnt_ptr,
    max_pages, qk_scale, sm_scale_log2, log2_theta,
    L: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
    PS: tl.constexpr, WINDOW: tl.constexpr, ROPE: tl.constexpr, SPLITS: tl.constexpr,
    RB: tl.constexpr, LB: tl.constexpr, BLOCK_T: tl.constexpr, ACT: tl.constexpr, FP8_ASM: tl.constexpr,
    CH: tl.constexpr, QK_FP8: tl.constexpr,
):
    GROUP: tl.constexpr = HQ // HKV
    R: tl.constexpr = L * GROUP
    QKV_W: tl.constexpr = (2 * HQ + 2 * HKV) * D
    ROW_W: tl.constexpr = HKV * D   # one cache row (all KV heads) in elements

    bh = tl.program_id(0)
    s = tl.program_id(1)
    b = bh // HKV
    h = bh % HKV
    seq = tl.load(seq_ptr + b)   # cached tokens; the block sits at positions seq .. seq+L-1
    k_scale = tl.load(kscale_ptr)
    v_scale = tl.load(vscale_ptr)

    rows = tl.arange(0, RB)
    j = rows // GROUP            # block token of the row
    g = rows % GROUP             # query head within the group
    rmask = rows < R
    qh = h * GROUP + g           # query head index
    tok = b * L + j
    dh = tl.arange(0, D // 2)    # pair index i: dims 2i (even) and 2i+1 (odd)
    d = tl.arange(0, D)
    inv_freq = tl.exp2(-(2.0 * dh.to(tl.float32) / D) * log2_theta)

    # ---- queries: RMSNorm, x qk_scale, RoPE, rounded to BF16 (as the reference does) ----
    qp = qkv_ptr + tok[:, None] * QKV_W + qh[:, None] * D + 2 * dh[None, :]
    qe = tl.load(qp, mask=rmask[:, None], other=0.0).to(tl.float32)
    qo = tl.load(qp + 1, mask=rmask[:, None], other=0.0).to(tl.float32)
    if QK_FP8:
        # FA3's FP8 path: the normalized query is rounded to E4M3; the 5.473 factor and the
        # cache scale move into the softmax scale
        qe, qo = _rms_rope(qe, qo, rmask, seq + j, inv_freq, 1.0, ROPE, D)
        q = tl.interleave(qe, qo).to(kc_ptr.dtype.element_ty)
        qk_mult = sm_scale_log2 * qk_scale * k_scale
    else:
        qe, qo = _rms_rope(qe, qo, rmask, seq + j, inv_freq, qk_scale, ROPE, D)
        # keys are used as their raw E4M3 values: the cache scale multiplies q instead
        q = (tl.interleave(qe, qo) * k_scale).to(ACT)   # [RB, D]
        qk_mult = sm_scale_log2

    # ---- the block's new keys and values for this KV head: [LB, D] (rows >= L masked) ----
    kj = tl.arange(0, LB)
    kmask = kj < L
    ktok = b * L + kj
    kp = qkv_ptr + ktok[:, None] * QKV_W + (HQ + h) * D + 2 * dh[None, :]
    ke = tl.load(kp, mask=kmask[:, None], other=0.0).to(tl.float32)
    ko = tl.load(kp + 1, mask=kmask[:, None], other=0.0).to(tl.float32)
    ke, ko = _rms_rope(ke, ko, kmask, seq + kj, inv_freq, 1.0, ROPE, D)
    k_new8 = (tl.interleave(ke, ko) / k_scale).to(kc_ptr.dtype.element_ty)
    vp = qkv_ptr + ktok[:, None] * QKV_W + (HQ + HKV + h) * D + d[None, :]
    v_new = tl.load(vp, mask=kmask[:, None], other=0.0).to(tl.float32)
    v_new8 = (v_new / v_scale).to(vc_ptr.dtype.element_ty)
    if s == 0:
        loc = tl.load(loc_ptr + ktok, mask=kmask, other=0)
        cp = loc[:, None] * ROW_W + h * D + d[None, :]
        tl.store(kc_ptr + cp, k_new8, mask=kmask[:, None])
        tl.store(vc_ptr + cp, v_new8, mask=kmask[:, None])
    # Attend to the entries as stored (what the reference checks against), raw E4M3 values.
    if QK_FP8:
        k_new = k_new8
    else:
        k_new = _cache_to_act(k_new8, ACT, FP8_ASM)    # [LB, D]
    v_newb = _cache_to_act(v_new8, ACT, FP8_ASM)   # [LB, D]

    # ---- online softmax over this split's slice of the past keys ----
    m_i = tl.full((RB,), NEG, tl.float32)
    l_i = tl.zeros((RB,), tl.float32)
    acc = tl.zeros((RB, D), tl.float32)
    if WINDOW > 0:
        lo = tl.maximum(seq - (WINDOW - 1), 0)
    else:
        lo = seq * 0
    n_keys = seq - lo
    chunk = (n_keys + SPLITS - 1) // SPLITS
    start = lo + s * chunk
    end = tl.minimum(seq, start + chunk)
    first = (seq + j) - (WINDOW - 1)   # each row's oldest attended position (sliding)
    for t0 in range(start, end, BLOCK_T):
        t = t0 + tl.arange(0, BLOCK_T)
        valid = t < end
        page = tl.load(pt_ptr + b * max_pages + t // PS, mask=valid, other=0)
        row = page.to(tl.int64) * PS + (t % PS).to(tl.int64)
        cp = row[:, None] * ROW_W + h * D + d[None, :]
        k8 = tl.load(kc_ptr + cp, mask=valid[:, None], other=0.0)
        if QK_FP8:
            sc = tl.dot(q, tl.trans(k8)) * qk_mult                # e4m3 x e4m3, fp32 accumulate
        else:
            kt = _cache_to_act(k8, ACT, FP8_ASM)                  # [BLOCK_T, D]
            sc = tl.dot(q, tl.trans(kt)) * qk_mult                # [RB, BLOCK_T]
        if WINDOW > 0:
            ok = valid[None, :] & (t[None, :] >= first[:, None])
        else:
            ok = valid[None, :] & (rows[:, None] >= 0)
        sc = tl.where(ok, sc, NEG)
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(sc - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v8 = tl.load(vc_ptr + cp, mask=valid[:, None], other=0.0)
        vt = _cache_to_act(v8, ACT, FP8_ASM)
        acc = acc * alpha[:, None] + tl.dot(p.to(ACT), vt)
        m_i = m_new

    # ---- split 0: the block's own tokens (causal inside the block) ----
    if s == 0:
        sc = tl.dot(q, tl.trans(k_new)) * qk_mult                  # [RB, LB]
        ok = kmask[None, :] & (kj[None, :] <= j[:, None])
        sc = tl.where(ok, sc, NEG)
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(sc - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(ACT), v_newb)
        m_i = m_new

    acc = acc * v_scale   # values were accumulated as raw E4M3

    # ---- merge the splits (last CTA of this (b, h)), then the gated norm ----
    if SPLITS > 1:
        base = (bh * SPLITS + s) * RB
        tl.store(part_ptr + (base + rows)[:, None] * D + d[None, :], acc.to(part_ptr.dtype.element_ty))
        tl.store(ml_ptr + (base + rows) * 2, m_i)
        tl.store(ml_ptr + (base + rows) * 2 + 1, l_i)
        tl.debug_barrier()
        ticket = tl.atomic_add(cnt_ptr + bh, 1, sem="acq_rel", scope="gpu")
        tl.debug_barrier()
        if ticket == SPLITS - 1:
            # merge: max over splits, then weighted sums, CH splits per step (independent loads)
            m_all = tl.full((RB,), NEG, tl.float32)
            for s0 in tl.static_range(0, SPLITS, CH):
                sidx = s0 + tl.arange(0, CH)
                b2 = (bh * SPLITS + sidx) * RB                                  # [CH]
                m2 = tl.load(ml_ptr + (b2[:, None] + rows[None, :]) * 2, cache_modifier=".cg")   # [CH, RB]
                m_all = tl.maximum(m_all, tl.max(m2, 0))
            l_all = tl.zeros((RB,), tl.float32)
            o = tl.zeros((RB, D), tl.float32)
            for s0 in tl.static_range(0, SPLITS, CH):
                sidx = s0 + tl.arange(0, CH)
                b2 = (bh * SPLITS + sidx) * RB
                m2 = tl.load(ml_ptr + (b2[:, None] + rows[None, :]) * 2, cache_modifier=".cg")
                l2 = tl.load(ml_ptr + (b2[:, None] + rows[None, :]) * 2 + 1, cache_modifier=".cg")
                a2 = tl.load(part_ptr + (b2[:, None, None] + rows[None, :, None]) * D + d[None, None, :],
                             cache_modifier=".cg").to(tl.float32)                  # [CH, RB, D]
                w = tl.exp2(m2 - m_all[None, :])                                    # [CH, RB]
                l_all += tl.sum(l2 * w, 0)
                o += tl.sum(a2 * w[:, :, None], 0)
            o = o / l_all[:, None]
            ss = tl.sum(o * o, 1) / D
            o = o * tl.rsqrt(ss + RMS_EPS)[:, None]
            gp = qkv_ptr + tok[:, None] * QKV_W + (HQ + 2 * HKV + qh)[:, None] * D + d[None, :]
            gate = tl.load(gp, mask=rmask[:, None], other=0.0).to(tl.float32)
            o = o * tl.sigmoid(gate)
            op = out_ptr + tok[:, None] * (HQ * D) + qh[:, None] * D + d[None, :]
            tl.store(op, o.to(out_ptr.dtype.element_ty), mask=rmask[:, None])
            tl.atomic_xchg(cnt_ptr + bh, 0)
    else:
        o = acc / l_i[:, None]
        ss = tl.sum(o * o, 1) / D
        o = o * tl.rsqrt(ss + RMS_EPS)[:, None]
        gp = qkv_ptr + tok[:, None] * QKV_W + (HQ + 2 * HKV + qh)[:, None] * D + d[None, :]
        gate = tl.load(gp, mask=rmask[:, None], other=0.0).to(tl.float32)
        o = o * tl.sigmoid(gate)
        op = out_ptr + tok[:, None] * (HQ * D) + qh[:, None] * D + d[None, :]
        tl.store(op, o.to(out_ptr.dtype.element_ty), mask=rmask[:, None])


# ----------------------------------------------------------------------------------------------
# host side
# ----------------------------------------------------------------------------------------------

_WS = {}


def _workspace(device, n_part: int, n_ml: int, n_cnt: int):
    """Split-KV partials and the self-resetting tickets, allocated once per device/size.
    Nothing here depends on the inputs' values."""
    key = (str(device), n_part, n_ml, n_cnt)
    ws = _WS.get(key)
    if ws is None:
        ws = (torch.empty((n_part,), dtype=torch.float16, device=device),   # acc partials (renormalized later)
              torch.empty((n_ml,), dtype=torch.float32, device=device),
              torch.zeros((n_cnt,), dtype=torch.int32, device=device))
        _WS[key] = ws
    return ws


def _plan(cfg):
    """Splits and tile size from the shape only."""
    b, hkv, L = cfg["batch"], cfg["num_kv_heads"], cfg["q_len"]
    ctas = b * hkv
    rb = max(16, triton.next_power_of_2(L * (cfg["num_q_heads"] // hkv)))   # dots need >= 16 rows
    # measured on the H200 (attention run 2026-10-23): 64-row tiles want fewer splits
    if rb <= 16:
        splits = max(1, min(16 if cfg["window"] else 64, (512 if cfg["window"] else 1024) // ctas))
    else:
        cap = 16 if cfg["window"] else (64 if ctas <= 4 else 16)
        splits = max(1, min(cap, (256 if cfg["window"] else 512) // ctas))
    splits = 1 << (splits.bit_length() - 1)   # a power of two: the combine merges CH splits at a time
    block_t = 128
    lb = 16
    return splits, block_t, rb, lb


_PLAN_OVERRIDE = None   # (splits, block_t, warps, stages) during benchmarking
ATTN_BENCH = True       # one on-device sweep in the first eager call (never under graph capture)
_BENCH_DONE = False


def _attn_bench(data):
    """Times kernel configurations on synthetic inputs shaped like the scored cases; prints."""
    import time
    global _PLAN_OVERRIDE
    t0 = time.time()
    dev = data["qkv"].device
    cfg0 = dict(data["config"])
    HQ, HKV, D = cfg0["num_q_heads"], cfg0["num_kv_heads"], cfg0["head_dim"]
    g = torch.Generator(device=dev).manual_seed(0)

    def make(B, L, window, ctx=80000):
        pages = B * (ctx + 8)
        k_cache = torch.randn((pages, 1, HKV, D), generator=g, device=dev).to(torch.float8_e4m3fn)
        v_cache = torch.randn((pages, 1, HKV, D), generator=g, device=dev).to(torch.float8_e4m3fn)
        seq = torch.full((B,), ctx, dtype=torch.int32, device=dev)
        page_table = torch.randperm(pages, generator=g, device=dev).to(torch.int32).view(B, ctx + 8)
        pos = (seq.long()[:, None] + torch.arange(L, device=dev)).reshape(-1)
        loc = torch.gather(page_table.long(), 1, pos.view(B, L)).reshape(-1)
        qkv = torch.randn((B, L, (2 * HQ + 2 * HKV) * D), generator=g, device=dev).to(torch.bfloat16)
        c = dict(cfg0, batch=B, q_len=L, page_size=1, window=window, rope_theta=500000.0 if window else 0.0)
        return {"qkv": qkv, "cache_loc": loc, "seq_lens": seq, "page_table": page_table, "k_cache": k_cache,
                "v_cache": v_cache, "k_scale": data["k_scale"], "v_scale": data["v_scale"], "config": c}

    def timeit(fn, reps=5):
        fn(); fn()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(reps):
            fn()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) * 1000.0 / reps

    print("BENCH attention: us per call (warm L2); config = (splits, block_t, warps, stages)")
    cases = [("B16 L1 full", make(16, 1, 0)), ("B16 L8 full", make(16, 8, 0)), ("B32 L8 sliding", make(32, 8, 2048)),
             ("B1 L8 full", make(1, 8, 0))]
    plans = {
        "B16 L1 full": [(32, 128, 4, 3, 0), (32, 128, 4, 3, 1)],
        "B16 L8 full": [(16, 128, 4, 3, 0), (16, 128, 4, 3, 1), (16, 256, 4, 2, 1), (32, 128, 4, 3, 1), (16, 128, 8, 2, 1)],
        "B32 L8 sliding": [(4, 128, 4, 3, 0), (4, 128, 4, 3, 1), (4, 128, 8, 2, 1), (8, 128, 4, 3, 1)],
        "B1 L8 full": [(64, 128, 8, 3, 0), (64, 128, 8, 3, 1), (32, 128, 8, 3, 1), (64, 128, 4, 3, 1), (128, 128, 8, 3, 1)],
    }
    try:
        fa3 = _fa3_forward
        for name, d in cases:
            print(f"  {name:15s} FA3 (stock path)      {timeit(lambda: fa3(d)):9.1f} us")
    except Exception as e:
        print("  FA3 timing unavailable:", str(e)[:100])
    for name, d in cases:
        for plan in plans[name]:
            if time.time() - t0 > 150:
                print("BENCH budget reached"); break
            _PLAN_OVERRIDE = plan
            try:
                us = timeit(lambda: _attention_forward(d))
                print(f"  {name:15s} {str(plan):20s} {us:9.1f} us")
            except Exception as e:
                print(f"  {name:15s} {str(plan):20s} failed: {str(e)[:90]}")
        _PLAN_OVERRIDE = None
    _PLAN_OVERRIDE = None
    print(f"BENCH attention done in {time.time() - t0:.0f}s")


_FA3 = {}


def _fa3_forward(data):
    """SGLang's stock attention for the step (FusedQKNorm, store_cache, FA3, rms_norm_gate), for
    comparison and as a per-shape fallback."""
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    from sglang.kernels.ops.kvcache.kvcache import store_cache
    from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
    from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    device = qkv.device
    key = ("norm", hq, hkv, d)
    if key not in _FA3:
        _FA3[key] = FusedQKNorm(head_dim=d, num_q_heads=hq, num_kv_heads=hkv, eps=1e-5)
    norm = _FA3[key]
    theta = cfg["rope_theta"]
    if theta:
        ck = ("cos_sin", theta)
        if ck not in _FA3:
            inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
            freqs = torch.einsum("i,j -> ij", torch.arange(1 << 21, dtype=torch.float, device=device), inv_freq)
            _FA3[ck] = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
        q, k = norm(qkv, data["positions"] if "positions" in data else
                    (data["seq_lens"].long()[:, None] + torch.arange(length, device=device)).reshape(-1), _FA3[ck])
    else:
        q, k = norm(qkv)
    v = qkv[:, (hq + hkv) * d:(hq + 2 * hkv) * d]
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    k = k.reshape(tokens, hkv * d)
    if scaled:
        k, v = k / data["k_scale"], v / data["v_scale"]
    fp8 = torch.float8_e4m3fn
    store_cache(k.to(fp8).view(torch.uint8), v.to(fp8).view(torch.uint8),
                k_cache.view(-1, hkv * d).view(torch.uint8), v_cache.view(-1, hkv * d).view(torch.uint8),
                data["cache_loc"], reserved_skip_index=-1)
    window = cfg["window"]
    descale = {}
    if scaled:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    cache_seqlens = data["cache_seqlens"] if "cache_seqlens" in data else (data["seq_lens"] + length).to(torch.int32)
    cu = data["cu_seqlens_q"] if "cu_seqlens_q" in data else torch.arange(0, tokens + 1, length, dtype=torch.int32, device=device)
    out = flash_attn_with_kvcache(
        q=q.view(tokens, hq, d).to(fp8), k_cache=k_cache, v_cache=v_cache, page_table=data["page_table"],
        cache_seqlens=cache_seqlens, cu_seqlens_q=cu, max_seqlen_q=length,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
        window_size=(window - 1, 0) if window else (-1, -1), scheduler_metadata=data.get("scheduler_metadata"),
        **descale)
    if isinstance(out, tuple):
        out = out[0]
    gate = qkv[:, (hq + 2 * hkv) * d:]
    o = rms_norm_gate(out.reshape(tokens, hq * d).to(torch.bfloat16), gate, None, head_dim=d, eps=1e-5, has_norm=True)
    return o.view(batch, length, hq * d)


def _attention_forward(data) -> torch.Tensor:
    global _BENCH_DONE
    if ATTN_BENCH and not _BENCH_DONE and torch.cuda.is_available() and not torch.cuda.is_current_stream_capturing():
        _BENCH_DONE = True
        try:
            _attn_bench(data)
        except Exception as e:
            print("BENCH failed:", repr(e))
    cfg = data["config"]
    B, L = cfg["batch"], cfg["q_len"]
    HQ, HKV, D = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    qkv = data["qkv"]
    if qkv.stride(-1) != 1:
        qkv = qkv.contiguous()
    device = qkv.device
    out = torch.empty((B, L, HQ * D), dtype=OUT_DTYPE, device=device)

    splits, block_t, rb, lb = _plan(cfg)
    nw, ns = 4, 3
    if rb > 16 and B * HKV <= 8:
        nw = 8   # latency-bound tiny batches: spread the 64 rows over more warps
    qk_fp8 = QK_FP8_ROWS and rb > 16
    if _PLAN_OVERRIDE is not None:
        splits, block_t, nw, ns = _PLAN_OVERRIDE[:4]
        if len(_PLAN_OVERRIDE) > 4:
            qk_fp8 = bool(_PLAN_OVERRIDE[4])
        splits = max(1, min(splits, 1 << (max(1, 1024 // (B * HKV)).bit_length() - 1)))
    ctas = B * HKV
    part, ml, cnt = _workspace(device, ctas * splits * rb * D, ctas * splits * rb * 2, max(ctas, 1024))
    page_table = data["page_table"]
    theta = cfg["rope_theta"]
    grid = (ctas, splits)
    ch = min(splits, 8 if rb <= 16 else 4)        # splits merged per load group
    _attn_decode_kernel[grid](
        qkv, data["cache_loc"], data["seq_lens"], page_table,
        data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"],
        out, part, ml, cnt,
        page_table.shape[1], cfg["qk_scale_factor"], cfg["sm_scale"] * LOG2E,
        math.log2(theta) if theta else 0.0,
        L=L, HQ=HQ, HKV=HKV, D=D, PS=cfg["page_size"], WINDOW=cfg["window"],
        ROPE=bool(theta), SPLITS=splits, RB=rb, LB=lb, BLOCK_T=block_t, ACT=ACT_DTYPE, FP8_ASM=FP8_ASM,
        CH=ch, QK_FP8=qk_fp8, num_warps=nw, num_stages=ns,
    )
    return out
# == END SHARED attn ==


def kernel_fn(data: Inputs) -> torch.Tensor:
    return _attention_forward(data)
