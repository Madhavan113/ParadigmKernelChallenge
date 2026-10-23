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


RMS_EPS = tl.constexpr(1e-5)
OUT_DTYPE = torch.bfloat16
ACT_DTYPE = tl.float16    # the dots' operand dtype: exact for the E4M3 cache values and finer than BF16
FP8_ASM = True            # hardware e4m3x2 -> f16x2 conversion (False: generic cast, for the interpreter)
NEG = tl.constexpr(-1.0e30)  # finite "minus infinity" for masked logits (keeps every exp2 finite)
LOG2E = 1.4426950408889634


@triton.jit
def _cache_to_act(x8, ACT: tl.constexpr, FP8_ASM: tl.constexpr):
    """FP8 E4M3 cache bytes -> the dots' operand dtype. On the GPU, one cvt.f16x2.e4m3x2 per
    pair (exact); Triton's generic cast would spend three conversions per element."""
    if FP8_ASM:
        b = x8.to(tl.uint8, bitcast=True)
        return tl.inline_asm_elementwise(
            asm="cvt.rn.f16x2.e4m3x2 $0, $1;", constraints="=r,h", args=[b],
            dtype=tl.float16, is_pure=True, pack=2)
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
    qe, qo = _rms_rope(qe, qo, rmask, seq + j, inv_freq, qk_scale, ROPE, D)
    # keys are used as their raw E4M3 values: the cache scale multiplies q instead
    q = (tl.interleave(qe, qo) * k_scale).to(ACT)   # [RB, D]

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
        kt = _cache_to_act(k8, ACT, FP8_ASM)                      # [BLOCK_T, D]
        sc = tl.dot(q, tl.trans(kt)) * sm_scale_log2              # [RB, BLOCK_T]
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
        sc = tl.dot(q, tl.trans(k_new)) * sm_scale_log2            # [RB, LB]
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
        tl.store(part_ptr + (base + rows)[:, None] * D + d[None, :], acc)
        tl.store(ml_ptr + (base + rows) * 2, m_i)
        tl.store(ml_ptr + (base + rows) * 2 + 1, l_i)
        tl.debug_barrier()
        ticket = tl.atomic_add(cnt_ptr + bh, 1, sem="acq_rel", scope="gpu")
        tl.debug_barrier()
        if ticket == SPLITS - 1:
            m_all = tl.full((RB,), NEG, tl.float32)
            for s2 in tl.static_range(SPLITS):
                b2 = (bh * SPLITS + s2) * RB
                m_all = tl.maximum(m_all, tl.load(ml_ptr + (b2 + rows) * 2, cache_modifier=".cg"))
            l_all = tl.zeros((RB,), tl.float32)
            o = tl.zeros((RB, D), tl.float32)
            for s2 in tl.static_range(SPLITS):
                b2 = (bh * SPLITS + s2) * RB
                m2 = tl.load(ml_ptr + (b2 + rows) * 2, cache_modifier=".cg")
                l2 = tl.load(ml_ptr + (b2 + rows) * 2 + 1, cache_modifier=".cg")
                a2 = tl.load(part_ptr + (b2 + rows)[:, None] * D + d[None, :], cache_modifier=".cg")
                w = tl.exp2(m2 - m_all)
                l_all += l2 * w
                o += a2 * w[:, None]
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
        ws = (torch.empty((n_part,), dtype=torch.float32, device=device),
              torch.empty((n_ml,), dtype=torch.float32, device=device),
              torch.zeros((n_cnt,), dtype=torch.int32, device=device))
        _WS[key] = ws
    return ws


def _plan(cfg):
    """Splits and tile size from the shape only."""
    b, hkv, L = cfg["batch"], cfg["num_kv_heads"], cfg["q_len"]
    ctas = b * hkv
    rb = max(16, triton.next_power_of_2(L * (cfg["num_q_heads"] // hkv)))   # dots need >= 16 rows
    cap = 128 if rb <= 16 else 32
    if cfg["window"]:
        splits = max(1, min(16 if rb <= 16 else 8, 512 // ctas))
        block_t = 64 if cfg["window"] <= 2048 else 128
    else:
        splits = max(1, min(cap, 1024 // ctas))
        block_t = 128
    lb = 16
    return splits, block_t, rb, lb


def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg = data["config"]
    B, L = cfg["batch"], cfg["q_len"]
    HQ, HKV, D = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    qkv = data["qkv"]
    if qkv.stride(-1) != 1:
        qkv = qkv.contiguous()
    device = qkv.device
    out = torch.empty((B, L, HQ * D), dtype=OUT_DTYPE, device=device)

    splits, block_t, rb, lb = _plan(cfg)
    ctas = B * HKV
    part, ml, cnt = _workspace(device, ctas * splits * rb * D, ctas * splits * rb * 2, max(ctas, 1024))
    page_table = data["page_table"]
    theta = cfg["rope_theta"]
    grid = (ctas, splits)
    _attn_decode_kernel[grid](
        qkv, data["cache_loc"], data["seq_lens"], page_table,
        data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"],
        out, part, ml, cnt,
        page_table.shape[1], cfg["qk_scale_factor"], cfg["sm_scale"] * LOG2E,
        math.log2(theta) if theta else 0.0,
        L=L, HQ=HQ, HKV=HKV, D=D, PS=cfg["page_size"], WINDOW=cfg["window"],
        ROPE=bool(theta), SPLITS=splits, RB=rb, LB=lb, BLOCK_T=block_t, ACT=ACT_DTYPE, FP8_ASM=FP8_ASM,
        num_warps=4, num_stages=3,
    )
    return out
