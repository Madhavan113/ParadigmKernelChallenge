# okc: problem=muse_decode_layer
# Derived work. Base: https://github.com/vjeux/okc-2026 (muse_decode_layer.py), Copyright (c) 2026 vjeux,
# licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
# Modifications Copyright (c) 2026 Madhavan113, same license. See NOTICE in the repository root.
"""Muse Spark decode layer: one CUDA-graph-capturable kernel_fn composed from three kernel modules and this glue.

GENERATED FILE: tools/compose_decode_layer.py embeds the three component files whole (every module-level name of a
component prefixed att_ / moe_ / bfp_) into muse_decode_layer/layer_glue.py. Edit those sources, not this file.

Provenance (all derived from vjeux/okc-2026, Apache-2.0; see the headers of the component sections):
  * attention (prefix att_, from muse_attention_decode/submission.py): the fused decode kernel (QK-norm, RoPE, FP8
    cache write, split-KV attention, merge, per-head RMSNorm * sigmoid(gate)), driven by its own launcher, plan and
    workspace allocator (att__launch, att__plan, att__workspace, att__supported); this layer builds it with
    ATT_BUILD_FLAGS (the layer's measured variant of the same source) and guards its buffers;
  * experts (prefix moe_, from muse_moe_w4a16/submission.py): the NVFP4 MoE kernel in its builds (TINY M <= 4,
    TINY2 <= 8, MID <= 16, WGS main), through its own prepare (weight packing, PRMT-scale build check) and
    kernel_fn (build choice per token count); its scratch is guarded here;
  * dense projections (prefix bfp_, from muse_decode_layer/proj_fork.py): the layer's fork of the muse_bf16_proj
    module, incl. the sk2 PDL prologue and the fused pooler + router launch (sk2f);
  * this file's glue: residual/norm boundaries, routing (top-8 of biased sigmoid scores; router weights as
    BF16 hi + lo with FP32 accumulation; pre-expert RMSNorm), Triton split-K GEMMs, and SGLang's own FA3,
    Marlin and cuBLAS paths from the judge image.

Selection: prepare() times each backend per (token count, matrix), and the expert kernels per token count,
on the judge's own GPU with synthetic activations (it reads only weights, never data), then fixes the
choices; untuned token counts take the next tuned bucket's. Attention backend by shape (q_len, window, batch).

Precision: BF16 weights stay BF16 (or bf16_proj's lossless 12-bit encoding, exact escape correction);
NVFP4 experts stay NVFP4; norms apply (1 + w) formed in FP32 at use; FP8 queries and FP8 KV cache as
SGLang; softmax P in f16, or e4m3 where SGLang's FA3 rounds it to e4m3 too; FP32 accumulation throughout;
activations BF16 wherever SGLang keeps BF16.

Launch order: at T <= 32 tokens every kernel of the layer is launched with programmatic dependent launch
(PDL). Each one waits (griddepcontrol.wait) before its first read of data produced in this call; only
weights, and inputs that no kernel in this call writes, are loaded before the wait. Above 32 tokens all
launches are plain stream-ordered.

Safety: every scratch/workspace buffer this file allocates or caches for a component is followed by
sentinel words; prepare() checks them after the tuner and an attention self-test have run every path.
"""

import math

import torch
import triton
import triton.language as tl

HIDDEN = 8192
D_EXPERT = 4096
LOCAL_EXPERTS = 32
N_EXPERTS = 256
TOP_K = 8
QKV = 2304
ROUTER_MULT = 0.1767766952966369
LOG2E = 1.4426950408889634

_TUNE_LOG = True   # prepare() prints the tuner's candidate times and picks per token count (untimed; judge .log)
FORCE_PBMAX = None
# Programmatic Dependent Launch for this file's Triton glue: each kernel launches while its producer drains and
# waits (griddepcontrol.wait) right before its first read of producer data; weights and older inputs load before.
PDL = True
PDL_MAX_T = 32   # measured: a launch-latency win at decode sizes (d1 +2%), a ~1% loss at T = 256 (big grids)


def _pdl(T):
    return PDL and T <= PDL_MAX_T
# The moe agent's kernel wins at every token count we tune (their board is 1.76x); the tuner may only
# depart from it when another backend is clearly faster, so a noisy measurement cannot flip a pick.
DEFAULT_MOE = "agent"
TUNE_MARGIN = 0.95
# Dense picks the isolated tuner gets wrong part of the time (it cannot see PDL overlap in the chain), pinned to the
# winner measured in the whole layer (pickab.py, v132, 31 reps): o at T=32 bfp 166.7 vs cuBLAS 176.1 us (32x1 sliding);
# qkv at T=128 cuBLAS 249.4 vs split-K 251.0 (16x8 sliding); qkv at T=256 cuBLAS 351.1 vs split-K 353.5 (32x8 sliding).
# pr at T=32: separate bfp 166.8 / 318.7 vs fused bfpf 167.9 / 320.2 us (32x1 sliding / full). T=256 pr/un: 4-way tie.
# un at T=1/8 (the 1024-row slice): bfp 80.7 / 80.1 / 116.7 / 138.2 vs split-K 81.9 / 81.2 / 118.8 / 140.1 us (1x1 s/f, 1x8 s/f).
PIN_PICK = {(1, "un"): "bfp", (8, "un"): "bfp", (1, "o"): "bfp", (8, "o"): "bfp", (32, "o"): "bfp",
            (1, "pr"): ("bfpf", "bfp"), (8, "pr"): ("bfpf", "bfp"), (16, "pr"): ("bfpf", "bfp"), (32, "pr"): ("bfpf", "bfp"),
            (128, "qkv"): (True, "mm16"), (256, "qkv"): (True, "mm16")}   # (8, "o"): in-chain sk2 112.5/133.4 vs cuBLAS 114.7/136.3 us (Oct 26)
# pr at T <= 32: the bf16_proj kernels (fused launch incl. the fused routing tail, or separate + _route_split), by the
# tuner's times; never the Triton split-K / cuBLAS cat GEMM (check 19ea8e60: Triton pr picked by a biased timing cost
# 6-8 us per decode shape)
# a tuple pins the choice to the faster of those picks by the tuner's own times (here: cuBLAS fp32 out vs bf16 out,
# never the Triton split-K the isolated tuner prefers)
FORCE_MM = None
FORCE_GLUEV2 = None  # None: tuner's choice; True/False: force the parallel boundary/router kernels      # None: the tuner's per-(T, matrix) dense choice; else force one backend everywhere   # set to 0/1/2 to override the tuner's megakernel threshold (pod experiments)

# Fused decode attention (QK-norm, RoPE, FP8 cache write, split-KV FP8 attention, merge, rmsnorm * sigmoid(gate)):
# copied from our muse_attention_decode entry (same author), compiled as its own extension.

# {{COMPONENT attn}}

# {{COMPONENT moe}}

# {{COMPONENT proj}}


if "bfp_CUTLASS_LAYERS" in globals():   # standalone proj composition: no CUTLASS build inside this layer's 300 s prepare
    bfp_CUTLASS_LAYERS = set()


def _bfp_prepare(weights, st):
    """Their prepare() on the five dense matrices, guarded: any failure just disables the backend."""
    st["bfp"] = None
    try:
        w5 = {k: weights[k].contiguous() for k in ("qkv_gate", "o", "pooler", "unpooler", "router")}
        st["bfp"] = bfp_prepare(w5)
    except Exception as ex:  # noqa: BLE001
        print("bfp disabled:", repr(ex)[:160], flush=True)


_BFP_LAYER = {"qkv": "qkv_gate", "o": "o", "un": "unpooler", "pool": "pooler", "router": "router"}


BFP_PDL_N = 2   # bf16_proj's sk2 / fused kernels: weight stages issued before griddepcontrol.wait (0 = off)


def _bfp_pdl(T):
    """Their launchers read SK2_PDL (bfp_SK2_PDL once inlined) at every launch: same T rule as this file's glue."""
    global bfp_SK2_PDL
    bfp_SK2_PDL = BFP_PDL_N if _pdl(T) else 0


def _bfp_gemm(st, name, x, m, n, k):
    _bfp_pdl(m)
    return bfp_kernel_fn({"x": x, "config": {"layer": _BFP_LAYER[name], "m": m, "n": n, "k": k}}, st["bfp"])


FUSE_ROUTE = True   # the fused pooler + router launch (sk2f) also does the routing and the pre-expert norm in its last
                    # CTA, replacing the _route_split launch (T <= 32, agent/Marlin experts: plain xm layout)


def _fused_route(state, data, ids, lids, wts, xm, selftest=False):
    """The route tuple for bfp_fused_pooler_router, or None when the separate _route_split kernel must run (also
    when the composed proj module's fused launch has no routing tail: the standalone muse_bf16_proj, or when the
    prepare-time self-test of the tail failed)."""
    ls = data["local_start"]
    if (not FUSE_ROUTE or ls.dtype != torch.int32 or state["g_preexp"].dtype != torch.bfloat16
            or "route" not in bfp_fused_pooler_router.__code__.co_varnames
            or not (selftest or state.get("fuse_route_ok"))):
        return None
    return (state["bias"], state["g_preexp"], ids, lids, wts, xm, ls)


FUSE_BOUNDARY = True    # T <= 8: the o-projection / unpooler-slice sk2 launches' last CTA computes _boundary1 / _boundary2
                        # (proj_fork's sk2b); enabled only when the prepare-time self-test matches the Triton kernels


def _route_selftest(st):
    """The fused routing tail against _route_split on synthetic activations (T = 1, 8, 16, 32): ids must match, the
    weights and xm within rounding. Any mismatch or exception disables the tail (st["fuse_route_ok"]); printed."""
    st["fuse_route_ok"] = False
    if st.get("bfp") is None or not FUSE_ROUTE or "route" not in bfp_fused_pooler_router.__code__.co_varnames:
        print("route tail: not available", flush=True)
        return
    dev = st["qkv"].device
    g = torch.Generator(device=dev).manual_seed(777)
    ls = torch.zeros(1, dtype=torch.int32, device=dev)
    worst = []
    for T in (1, 8, 16, 32):
        h = (torch.randn(T, HIDDEN, generator=g, device=dev) * 0.1).to(torch.bfloat16)
        a = [torch.empty((T, 8), dtype=torch.int32, device=dev), torch.empty((T, 8), dtype=torch.int32, device=dev),
             torch.empty((T, 8), dtype=torch.float32, device=dev), torch.empty((T, D_EXPERT), dtype=torch.bfloat16, device=dev)]
        b = [torch.empty_like(t) for t in a]
        rt = _fused_route(st, {"local_start": ls}, *a, selftest=True)
        try:
            _bfp_pdl(T)
            pooled, logits = bfp_fused_pooler_router(h, st["bfp"], rt)
            torch.cuda.synchronize()
        except Exception as ex:  # noqa: BLE001
            print(f"route tail T{T}: FAILED {repr(ex)[:300]}", flush=True)
            return
        _route_split[(T,)](pooled, logits, st["bias"], st["g_preexp"], b[0], b[2], b[3], ls, b[1], PERMUTED=False, num_warps=8)
        torch.cuda.synchronize()
        d_ids = int((a[0] != b[0]).sum()) + int((a[1] != b[1]).sum())
        d_w = float((a[2] - b[2]).abs().max())
        d_x = float((a[3].float() - b[3].float()).abs().max()) / max(1e-6, float(b[3].float().abs().max()))
        worst.append((T, d_ids, d_w, d_x))
        if d_ids or not d_w < 1e-4 or not d_x < 2e-2:
            print(f"route tail T{T}: MISMATCH ids {d_ids} wts {d_w:.2e} xm {d_x:.2e}", flush=True)
            return
    print("route tail ok: " + " ".join(f"T{T} ids0 w{w:.1e} x{x:.1e}" for T, _, w, x in worst), flush=True)
    st["fuse_route_ok"] = True


def _boundary_selftest(st):
    """The sk2b boundary tails (o projection -> _boundary1, unpooler slice -> _boundary2) against the Triton kernels
    at T = 1, 8 on synthetic inputs; any mismatch or exception disables them (st["fuse_bnd_ok"]); printed."""
    st["fuse_bnd_ok"] = False
    if (st.get("bfp") is None or not FUSE_BOUNDARY or "bfp_run_sk2_tail" not in globals()
            or "ab1" not in st):
        print("boundary tails: not available", flush=True)
        return
    dev = st["qkv"].device
    g = torch.Generator(device=dev).manual_seed(778)
    f32 = dict(device=dev, dtype=torch.float32)
    report = []
    for T in (1, 8):
        stream = torch.randn(T, HIDDEN, generator=g, device=dev)
        o_in = (torch.randn(T, 1024, generator=g, device=dev) * 0.1).to(torch.bfloat16)
        m = (torch.randn(T, D_EXPERT, generator=g, device=dev) * 0.1).to(torch.bfloat16)
        try:
            _bfp_pdl(T)
            s1a, ha = torch.empty((T, HIDDEN), **f32), torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
            P = globals()["bfp_run_sk2_tail"](o_in, "o", T, HIDDEN, 1024, st["bfp"],
                                              (st["ab1"][0], st["ab1"][1], stream, None, st["g_preffn"], s1a, ha))
            s2a, outa = torch.empty((T, HIDDEN), **f32), torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
            Pu = globals()["bfp_run_sk2_tail"](m, "unpooler", T, 1024, D_EXPERT, st["bfp"],
                                               (st["ab2"][0], st["ab2"][1], s1a, st["g_postffn"], st["g_next"], s2a, outa))
            torch.cuda.synchronize()
        except Exception as ex:  # noqa: BLE001
            print(f"boundary tails T{T}: FAILED {repr(ex)[:300]}", flush=True)
            return
        # the Triton kernels on the SAME GEMM outputs (P, Pu are the bf16 products the tails saw)
        s1b, hb = torch.empty((T, HIDDEN), **f32), torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
        _boundary1[(T,)](P.view(1, T, HIDDEN), 1, T, stream, st["gate1"], st["gate1"], st["g_preffn"], s1b, hb, num_warps=16)
        s2b, outb = torch.empty((T, HIDDEN), **f32), torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
        _boundary2[(T,)](Pu.view(1, T, 1024), 1, T, 1024, s1a, st["gate2"], st["gate2"], st["g_postffn"], st["g_next"],
                         s2b, outb, num_warps=16)
        torch.cuda.synchronize()
        errs = []
        for x, y in ((s1a, s1b), (ha, hb), (s2a, s2b), (outa, outb)):
            errs.append(float((x.float() - y.float()).norm() / max(1e-6, float(y.float().norm()))))
        report.append((T, errs))
        if not all(e < 2e-3 for e in errs):
            print(f"boundary tails T{T}: MISMATCH rel err s1 {errs[0]:.2e} h {errs[1]:.2e} s2 {errs[2]:.2e} out {errs[3]:.2e}", flush=True)
            return
    print("boundary tails ok: " + " ".join(f"T{T} " + "/".join(f"{e:.1e}" for e in errs) for T, errs in report), flush=True)
    st["fuse_bnd_ok"] = True


def _gate_coeffs(g):
    """alpha, beta of a residual gate from its BF16 weight, exactly as _gate_ab forms them (prepare-time, fp32)."""
    w = g.float() / 0.3
    beta = torch.sigmoid(w)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=1e-3))
    return alpha.contiguous(), beta.contiguous()


def _bfp_tail_ok(state, T):
    return (FUSE_BOUNDARY and T <= 8 and state.get("bfp") is not None and "bfp_run_sk2_tail" in globals()
            and state.get("fuse_bnd_ok") and "ab1" in state and state["g_preffn"].dtype == torch.bfloat16
            and state["g_postffn"].dtype == torch.bfloat16 and state["g_next"].dtype == torch.bfloat16)


SCRATCH_GUARD = 1 << 16   # sentinel words after each buffer my glue allocates: wide enough for far overruns (v154m-style state ~4K words out)
SCRATCH_SENTINEL = 0x5EADBEEF


def _moe_weights(weights):
    """The expert weights in the moe module's own Weights layout (uint8 codes, e4m3 block scales, fp32 global
    scales, post_norm_w = this layer's post_expert_norm)."""
    return {"w13": weights["w13"].view(torch.uint8), "w13_sf": weights["w13_sf"], "w13_gs": weights["w13_gs"].float(),
            "w2": weights["w2"].view(torch.uint8), "w2_sf": weights["w2_sf"], "w2_gs": weights["w2_gs"].float(),
            "post_norm_w": weights["post_expert_norm"]}


def _moe_sfx(weights):
    """The moe module's build suffix (its prepare() computes the same): the PRMT scale decode only when every local
    block-scale byte is a positive normal e4m3."""
    return ("-DOKC_PRMTSC=1",) if (moe_PRMT_SCALES and moe__scales_prmt_exact(weights["w13_sf"], weights["w2_sf"])) else ()


def _moe_builds(sfx):
    """The builds the moe module's prepare() will ask for (its rule), so prepare() here can run them in parallel."""
    mods = [tuple(moe__main_defines(sfx)) + sfx]
    for d in (moe_TINY_DEFINES, moe_TINY2_DEFINES, moe_MID_DEFINES):
        if d and (sfx or "-DOKC_WGS=1" not in tuple(d)):   # the WGS build needs the PRMT scale decode
            mods.append(tuple(d) + sfx)
    return mods


def _ma_prepare(weights, st):
    """The moe module's own prepare() (weight packing in its layouts, build check, scratch), its scratch guarded:
    the scratch size depends on every build its kernel_fn may pick, so only its allocator is trusted to size it."""
    st["ma"] = moe_prepare(_moe_weights(weights))
    st["ma"]["scratch"] = [_guarded(t) for t in st["ma"]["scratch"]]


def _ma_run(state, x, lids, wts, T):
    """Their kernel_fn: the build for this token count, at most moe_MMAX tokens per launch. PDL as this call's rule
    (their extensions keep one flag each, set at every call; capture-time Python only)."""
    pdl = 1 if _pdl(T) else 0
    for ext in moe__EXTS.values():
        ext.okc_set_pdl(pdl)
    return moe_kernel_fn({"hidden_states": x, "topk_ids": lids, "topk_weights": wts,
                          "config": {"local_start": 0, "local_count": LOCAL_EXPERTS}}, state["ma"])






def _to_i32(x64):
    x64 = x64 & 0xFFFFFFFF
    return (x64 - (x64 >= 2**31).to(torch.int64) * 2**32).to(torch.int32)


def _repack_q(wp, interleave):
    """wp uint8 [E, R, 2048] (low nibble = even k) -> int32 [E, R/16, 64, 32, 4] mma-fragment words."""
    E, R, _ = wp.shape
    if interleave:
        wp = wp.view(E, 2, R // 32, 16, 2048).transpose(1, 2).reshape(E, R, 2048)
    x = wp.view(E, R // 16, 16, 256, 8).permute(0, 1, 3, 2, 4).to(torch.int64)  # [E, RB, kb, row, j]
    B3 = x[..., :8, :4]
    B2 = x[..., 8:, :4]
    B1 = x[..., :8, 4:]
    B0 = x[..., 8:, 4:]
    w = ((B0 & 15) | ((B1 & 15) << 4) | ((B2 & 15) << 8) | ((B3 & 15) << 12)
         | ((B0 >> 4) << 16) | ((B1 >> 4) << 20) | ((B2 >> 4) << 24) | ((B3 >> 4) << 28))
    w = _to_i32(w).reshape(E, R // 16, 64, 4, 32).permute(0, 1, 2, 4, 3).contiguous()
    return w


def _prepack_frag(w):
    """bf16 [N, K] -> int32 [N/16, K/16, 32 lanes, 4]: mma.m16n8k16 A-fragment order per (rowblock, k16)."""
    N, K = w.shape
    dev = w.device
    g = torch.arange(8, device=dev).view(8, 1, 1)
    c = torch.arange(4, device=dev).view(1, 4, 1)
    j = torch.arange(8, device=dev).view(1, 1, 8)
    rows = g + 8 * ((j // 2) % 2)
    cols = 2 * c + (j % 2) + 8 * (j // 4)
    x = w.view(N // 16, 16, K // 16, 16).permute(0, 2, 1, 3)  # [rb, kb, row, col]
    f = x[:, :, rows, cols]  # [rb, kb, 8, 4, 8]
    return f.reshape(N // 16, K // 16, 32, 8).contiguous().view(torch.int32)


def _repack_s(sf, interleave):
    """sf uint8 [E, R, 256] -> uint8 [E, R/16, 64, 8, 8]: (g, [rows g kb0..3, rows g+8 kb0..3])."""
    E, R, _ = sf.shape
    if interleave:
        sf = sf.view(E, 2, R // 32, 16, 256).transpose(1, 2).reshape(E, R, 256)
    x = sf.view(E, R // 16, 16, 64, 4).permute(0, 1, 3, 2, 4)  # [E, RB, ch, row, kk]
    return torch.cat([x[:, :, :, :8, :], x[:, :, :, 8:, :]], dim=-1).contiguous()


# ----------------------------------------------------------------------------------------------
# dense skinny GEMM: P[s, t, n] = sum_{k in split s} X[t, k] * W[n, k]
# ----------------------------------------------------------------------------------------------
@triton.jit
def _gemm_splitk(X, W, P, T, N, K, KSPLIT: tl.constexpr, BN: tl.constexpr, BT: tl.constexpr,
                 BK: tl.constexpr, USE_PDL: tl.constexpr = False):
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    pt = tl.program_id(2)
    rn = pn * BN + tl.arange(0, BN)
    rt = pt * BT + tl.arange(0, BT)
    rk = tl.arange(0, BK)
    kbase = ps * KSPLIT
    w_ptr = W + rn[:, None].to(tl.int64) * K + (kbase + rk)[None, :]
    x_ptr = X + rt[None, :] * K + (kbase + rk)[:, None]
    tmask = rt[None, :] < T
    acc = tl.zeros([BN, BT], dtype=tl.float32)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()   # X is the producer's output
    for _k in range(0, KSPLIT, BK):
        w = tl.load(w_ptr)
        x = tl.load(x_ptr, mask=tmask, other=0.0)
        acc = tl.dot(w, x, acc)
        w_ptr += BK
        x_ptr += BK
    tl.store(P + ps * T * N + rt[None, :] * N + rn[:, None], acc, mask=tmask)


@triton.jit
def _gemm_splitk_ng(X, G, W, P, T, N, K, KSPLIT: tl.constexpr, BN: tl.constexpr, BT: tl.constexpr,
                    BK: tl.constexpr, USE_PDL: tl.constexpr = False):
    """Like _gemm_splitk with x = rmsnorm_per_64(X) * sigmoid(G) applied on the fly (X bf16, G fp32)."""
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    pt = tl.program_id(2)
    rn = pn * BN + tl.arange(0, BN)
    rt = pt * BT + tl.arange(0, BT)
    rk = tl.arange(0, BK)
    kbase = ps * KSPLIT
    w_ptr = W + rn[:, None].to(tl.int64) * K + (kbase + rk)[None, :]
    x_ptr = X + rt[None, :] * K + (kbase + rk)[:, None]
    g_ptr = G + rt[None, :] * K + (kbase + rk)[:, None]
    tmask = rt[None, :] < T
    acc = tl.zeros([BN, BT], dtype=tl.float32)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()
    for _k in range(0, KSPLIT, BK):
        w = tl.load(w_ptr)
        x = tl.load(x_ptr, mask=tmask, other=0.0).to(tl.float32)
        gt = tl.load(g_ptr, mask=tmask, other=0.0)
        x3 = tl.reshape(x, [BK // 64, 64, BT])
        r = tl.rsqrt(tl.sum(x3 * x3, 1) / 64.0 + 1e-5)
        xn = tl.reshape(x3 * r[:, None, :], [BK, BT]) * tl.sigmoid(gt)
        acc = tl.dot(w, xn.to(tl.bfloat16), acc)
        w_ptr += BK
        x_ptr += BK
        g_ptr += BK
    tl.store(P + ps * T * N + rt[None, :] * N + rn[:, None], acc, mask=tmask)


@triton.jit
def _normgate(X, G, OUT, USE_PDL: tl.constexpr = False):
    t = tl.program_id(0)
    h = tl.arange(0, 16)[:, None]
    d = tl.arange(0, 64)[None, :]
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()
    x = tl.load(X + t * 1024 + h * 64 + d).to(tl.float32)
    x = x * tl.rsqrt(tl.sum(x * x, 1) / 64.0 + 1e-5)[:, None]
    x = x * tl.sigmoid(tl.load(G + t * 1024 + h * 64 + d))
    tl.store(OUT + t * 1024 + h * 64 + d, x.to(tl.bfloat16))


def dense_ng(x, gate, w, T, state=None):
    N, K = w.shape
    if T > 32 or (state is not None and state.get("mm_pick", {}).get((T, "o"), False)):
        xn = torch.empty((T, K), device=x.device, dtype=torch.bfloat16)
        _normgate[(T,)](x, gate, xn, num_warps=4, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
        return dense(xn, w, T, state, "o")
    BN, BK = 64, 128
    BT = 16 if T <= 16 else 32
    S = 1
    P = torch.empty((S, T, N), device=x.device, dtype=torch.float32)
    grid = (N // BN, S, triton.cdiv(T, BT))
    _gemm_splitk_ng[grid](x, gate, w, P, T, N, K, KSPLIT=K // S, BN=BN, BT=BT, BK=BK, num_warps=4, num_stages=4, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    return P, S


TUNED_T = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def _bucket(T):
    """The tuned token count whose choices apply to T (the next one at or above it)."""
    for t in TUNED_T:
        if T <= t:
            return t
    return TUNED_T[-1]


BFP_MAX_T = 32   # bf16_proj's qkv / o / unpooler kernels where they win inside the chain (T=1..32: +4-9% on those
                 # loads); at T=256 their routes measured 10 us slower than cuBLAS on 32x8 sliding, at T=128 a wash


def dense(x, w, T, state=None, name=None):
    """Returns (partials [S, T, N] fp32, S). The tuner picks cuBLAS or the split-K Triton kernel per
    (token count, weight matrix): cuBLAS wins on the o projection, the Triton one on the pooler+router."""
    N, K = w.shape
    pick = state.get("mm_pick", {}).get((T, name), False) if state is not None else False
    if pick == "bfp" and T <= BFP_MAX_T and name in _BFP_LAYER and state.get("bfp") is not None:
        # bf16_proj's kernel (BF16 out, as SGLang's GEMM): the consumers read P as one [1, T, N] partial
        return _bfp_gemm(state, name, x, T, N, K).view(1, T, N), 1
    if pick == "mm16":
        # cuBLAS with BF16 output (SGLang's own bf16 GEMM call): a different kernel from the fp32-output call, which at
        # T = 128/256 lands on a 19-24 us split-K kernel for the o projection (bf16 out: 8-10 us); the consumers read
        # P in any dtype (the bf16_proj paths at T <= 32 already hand them bf16)
        return torch.nn.functional.linear(x, w).view(1, T, N), 1
    if pick is True or pick == "bfp":       # above BFP_MAX_T a bfp pick runs as cuBLAS (measured in the chain)
        out = torch.mm(x, w.t(), out_dtype=torch.float32)
        return out.view(1, T, N), 1
    if T > 32:
        out = torch.mm(x, w.t(), out_dtype=torch.float32)
        return out.view(1, T, N), 1
    BN, BK = 64, 128
    BT = 16 if T <= 16 else 32
    tiles = (N // BN) * triton.cdiv(T, BT)
    S = 1
    while tiles * S * 2 <= 600 and K // (S * 2) >= 2 * BK:
        S *= 2
    P = torch.empty((S, T, N), device=x.device, dtype=torch.float32)
    grid = (N // BN, S, triton.cdiv(T, BT))
    _gemm_splitk[grid](x, w, P, T, N, K, KSPLIT=K // S, BN=BN, BT=BT, BK=BK, num_warps=4, num_stages=4, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    return P, S


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
@triton.jit
def _e4m3_to_f32(b):
    """uint8/int32 e4m3fn bits -> float32 (exact)."""
    b = b.to(tl.int32)
    h = (((b & 0x7F) << 7) | ((b & 0x80) << 8)).to(tl.int16).to(tl.float16, bitcast=True)
    return h.to(tl.float32) * 256.0


@triton.jit
def _f32_to_e4m3(x):
    """float32 -> e4m3fn bits (int32), round to nearest even, saturating to +-448."""
    x = tl.minimum(tl.maximum(x, -448.0), 448.0)
    bits = x.to(tl.int32, bitcast=True)
    sign = (bits >> 24) & 0x80
    a = tl.abs(x)
    # subnormal: multiples of 2^-9
    sub = (a * 512.0 + 0.5).to(tl.int32)  # round half up (ties are measure-zero)
    ab = a.to(tl.int32, bitcast=True)
    r = ab + 0x7FFFF + ((ab >> 20) & 1)
    r = r & 0x7FF00000
    e = (r >> 23) - 120
    m = (r >> 20) & 7
    nrm = (e << 3) | m
    nrm = tl.minimum(nrm, 0x7E)
    code = tl.where(a < 0.015625, sub, nrm)
    return code | sign


# ----------------------------------------------------------------------------------------------
# QK norm + RoPE + cache write
# ----------------------------------------------------------------------------------------------
@triton.jit
def _qk_prep(P, S: tl.constexpr, T, POS, LOC, INVF, KC, VC, KS, VS, QOUT, Q8OUT, GOUT, qk_factor,
             ROPE: tl.constexpr, USE_PDL: tl.constexpr = False):
    t = tl.program_id(0)
    base = P + t * 2304
    hq = tl.arange(0, 16)[:, None]
    hk = tl.arange(0, 2)[:, None]
    i = tl.arange(0, 32)[None, :]
    d = tl.arange(0, 64)[None, :]
    qe = tl.zeros([16, 32], dtype=tl.float32)
    qo = tl.zeros([16, 32], dtype=tl.float32)
    ke = tl.zeros([2, 32], dtype=tl.float32)
    ko = tl.zeros([2, 32], dtype=tl.float32)
    v = tl.zeros([2, 64], dtype=tl.float32)
    g = tl.zeros([1024], dtype=tl.float32)
    gi = tl.arange(0, 1024)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()
    for s in range(S):
        b = base + s * T * 2304
        qe += tl.load(b + hq * 64 + 2 * i)
        qo += tl.load(b + hq * 64 + 2 * i + 1)
        ke += tl.load(b + 1024 + hk * 64 + 2 * i)
        ko += tl.load(b + 1024 + hk * 64 + 2 * i + 1)
        v += tl.load(b + 1152 + hk * 64 + d)
        g += tl.load(b + 1280 + gi)
    rq = tl.rsqrt(tl.sum(qe * qe + qo * qo, 1) / 64.0 + 1e-5)[:, None]
    rk = tl.rsqrt(tl.sum(ke * ke + ko * ko, 1) / 64.0 + 1e-5)[:, None]
    qe = qe * rq
    qo = qo * rq
    ke = ke * rk
    ko = ko * rk
    if ROPE:
        pos = tl.load(POS + t).to(tl.float32)
        ang = pos * tl.load(INVF + i)
        c = tl.cos(ang)
        sn = tl.sin(ang)
        qe, qo = qe * c - qo * sn, qe * sn + qo * c
        ke, ko = ke * c - ko * sn, ke * sn + ko * c
    qb = QOUT + t * 1024 + hq * 64 + 2 * i
    tl.store(qb, (qe * qk_factor).to(tl.bfloat16))
    tl.store(qb + 1, (qo * qk_factor).to(tl.bfloat16))
    q8b = Q8OUT + t * 1024 + hq * 64 + 2 * i
    tl.store(q8b, _f32_to_e4m3(qe).to(tl.uint8))
    tl.store(q8b + 1, _f32_to_e4m3(qo).to(tl.uint8))
    tl.store(GOUT + t * 1024 + gi, g)
    loc = tl.load(LOC + t).to(tl.int64)
    kinv = 1.0 / tl.load(KS)
    vinv = 1.0 / tl.load(VS)
    kb = KC + loc * 128 + hk * 64 + 2 * i
    tl.store(kb, _f32_to_e4m3(ke * kinv).to(tl.uint8))
    tl.store(kb + 1, _f32_to_e4m3(ko * kinv).to(tl.uint8))
    tl.store(VC + loc * 128 + hk * 64 + d, _f32_to_e4m3(v * vinv).to(tl.uint8))


# ----------------------------------------------------------------------------------------------
# attention (split-KV flash decoding over a paged FP8 cache)
# ----------------------------------------------------------------------------------------------
@triton.jit
def _attn_split(Q, KC, VC, PT, SEQ, KS, ACC, ML, pt_stride, page_size, window, sm_scale,
                L: tl.constexpr, NSPLIT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                SLIDING: tl.constexpr):
    """One program = one request, both KV heads (128-byte K/V rows), one key split."""
    b = tl.program_id(0)
    s = tl.program_id(1)
    seq = tl.load(SEQ + b)
    m_off = tl.arange(0, BM)
    l_idx = m_off // 8
    hh = m_off % 8
    row_ok = m_off < L * 8
    qpos = seq + l_idx
    d = tl.arange(0, 64)
    qrow = (b * L + l_idx) * 16 + hh
    q0 = tl.load(Q + qrow[:, None] * 64 + d[None, :], mask=row_ok[:, None], other=0.0).to(tl.float16)
    q1 = tl.load(Q + (qrow + 8)[:, None] * 64 + d[None, :], mask=row_ok[:, None], other=0.0).to(tl.float16)
    total = seq + L
    if SLIDING:
        lo = tl.maximum(seq - window + 1, 0)
    else:
        lo = seq * 0
    n = total - lo
    per = tl.cdiv(tl.cdiv(n, NSPLIT), BN) * BN
    kst = lo + s * per
    ken = tl.minimum(total, kst + per)
    scale = sm_scale * 256.0 * tl.load(KS) * 1.4426950408889634
    m0 = tl.full([BM], float("-inf"), dtype=tl.float32)
    m1 = tl.full([BM], float("-inf"), dtype=tl.float32)
    l0 = tl.zeros([BM], dtype=tl.float32)
    l1 = tl.zeros([BM], dtype=tl.float32)
    a0 = tl.zeros([BM, 64], dtype=tl.float32)
    a1 = tl.zeros([BM, 64], dtype=tl.float32)
    for kb in range(kst, ken, BN):
        offs = kb + tl.arange(0, BN)
        kv_ok = offs < ken
        page = tl.load(PT + b * pt_stride + offs // page_size, mask=kv_ok, other=0)
        row = page.to(tl.int64) * page_size + offs % page_size
        kptr = row[:, None] * 128 + d[None, :]
        ok = kv_ok[None, :] & (offs[None, :] <= qpos[:, None])
        if SLIDING:
            ok = ok & (offs[None, :] > qpos[:, None] - window)
        k8 = tl.load(KC + kptr, mask=kv_ok[:, None], other=0).to(tl.int32)
        k16 = (((k8 & 0x7F) << 7) | ((k8 & 0x80) << 8)).to(tl.int16).to(tl.float16, bitcast=True)
        k8b = tl.load(KC + kptr + 64, mask=kv_ok[:, None], other=0).to(tl.int32)
        k16b = (((k8b & 0x7F) << 7) | ((k8b & 0x80) << 8)).to(tl.int16).to(tl.float16, bitcast=True)
        v8 = tl.load(VC + kptr, mask=kv_ok[:, None], other=0).to(tl.int32)
        v16 = (((v8 & 0x7F) << 7) | ((v8 & 0x80) << 8)).to(tl.int16).to(tl.float16, bitcast=True)
        v8b = tl.load(VC + kptr + 64, mask=kv_ok[:, None], other=0).to(tl.int32)
        v16b = (((v8b & 0x7F) << 7) | ((v8b & 0x80) << 8)).to(tl.int16).to(tl.float16, bitcast=True)
        # head 0
        sc = tl.where(ok, tl.dot(q0, tl.trans(k16)) * scale, float("-inf"))
        mn = tl.maximum(m0, tl.max(sc, 1))
        mu = tl.where(mn == float("-inf"), 0.0, mn)
        p = tl.exp2(sc - mu[:, None])
        al = tl.exp2(m0 - mu)
        l0 = l0 * al + tl.sum(p, 1)
        a0 = a0 * al[:, None] + tl.dot(p.to(tl.float16), v16)
        m0 = mn
        # head 1
        sc = tl.where(ok, tl.dot(q1, tl.trans(k16b)) * scale, float("-inf"))
        mn = tl.maximum(m1, tl.max(sc, 1))
        mu = tl.where(mn == float("-inf"), 0.0, mn)
        p = tl.exp2(sc - mu[:, None])
        al = tl.exp2(m1 - mu)
        l1 = l1 * al + tl.sum(p, 1)
        a1 = a1 * al[:, None] + tl.dot(p.to(tl.float16), v16b)
        m1 = mn
    ob0 = ((b * 2 + 0) * NSPLIT + s) * BM + m_off
    ob1 = ((b * 2 + 1) * NSPLIT + s) * BM + m_off
    tl.store(ACC + ob0[:, None] * 64 + d[None, :], a0)
    tl.store(ACC + ob1[:, None] * 64 + d[None, :], a1)
    tl.store(ML + ob0 * 2, m0)
    tl.store(ML + ob0 * 2 + 1, l0)
    tl.store(ML + ob1 * 2, m1)
    tl.store(ML + ob1 * 2 + 1, l1)


@triton.jit
def _attn_combine(ACC, ML, GATE, VS, OUT, L: tl.constexpr, NSPLIT: tl.constexpr, BM: tl.constexpr,
                  SB: tl.constexpr, NORMGATE: tl.constexpr = True):
    """One program per (token, q head): merge the splits, per-head RMSNorm, * sigmoid(gate)."""
    t = tl.program_id(0)
    h = tl.program_id(1)
    b = t // L
    l = t % L
    g = h // 8
    m_row = l * 8 + h % 8
    d = tl.arange(0, 64)
    base = (b * 2 + g) * NSPLIT
    M = tl.full([SB], float("-inf"), dtype=tl.float32)
    for s0 in range(0, NSPLIT, SB):
        sidx = s0 + tl.arange(0, SB)
        sm = sidx < NSPLIT
        M = tl.maximum(M, tl.load(ML + ((base + sidx) * BM + m_row) * 2, mask=sm, other=float("-inf")))
    Mx = tl.max(M, 0)
    Mx = tl.where(Mx == float("-inf"), 0.0, Mx)
    den = tl.zeros([SB], dtype=tl.float32)
    num = tl.zeros([64], dtype=tl.float32)
    for s0 in range(0, NSPLIT, SB):
        sidx = s0 + tl.arange(0, SB)
        sm = sidx < NSPLIT
        rows = (base + sidx) * BM + m_row
        m_s = tl.load(ML + rows * 2, mask=sm, other=float("-inf"))
        l_s = tl.load(ML + rows * 2 + 1, mask=sm, other=0.0)
        w = tl.exp2(m_s - Mx)
        den += w * l_s
        a = tl.load(ACC + rows[:, None] * 64 + d[None, :], mask=sm[:, None], other=0.0)
        num += tl.sum(w[:, None] * a, 0)
    o = num / tl.sum(den, 0) * (256.0 * tl.load(VS))
    if NORMGATE:
        o = o * tl.rsqrt(tl.sum(o * o, 0) / 64.0 + 1e-5)
        gt = tl.load(GATE + t * 1024 + h * 64 + d)
        o = o * tl.sigmoid(gt)
    tl.store(OUT + t * 1024 + h * 64 + d, o.to(tl.bfloat16))


# ----------------------------------------------------------------------------------------------
# boundaries, routing
# ----------------------------------------------------------------------------------------------
@triton.jit
def _gemm_o(X, G, W, AOUT, SUMS, STREAM, A1, B1, T, N, K: tl.constexpr, BN: tl.constexpr, BT: tl.constexpr,
            BK: tl.constexpr, NORMGATE: tl.constexpr):
    """o projection without a K split (K = 1024), so each program owns complete rows: it writes a and
    accumulates the four per-token sums the post-attention boundary needs (a^2, (a1 s)^2, (a1 s)(b1 a), (b1 a)^2)."""
    pn = tl.program_id(0)
    pt = tl.program_id(1)
    rn = pn * BN + tl.arange(0, BN)
    rt = pt * BT + tl.arange(0, BT)
    rk = tl.arange(0, BK)
    tmask = rt[None, :] < T
    acc = tl.zeros([BN, BT], dtype=tl.float32)
    w_ptr = W + rn[:, None].to(tl.int64) * K + rk[None, :]
    x_ptr = X + rt[None, :] * K + rk[:, None]
    for _k in range(0, K, BK):
        w = tl.load(w_ptr)
        if NORMGATE:
            xv = tl.load(x_ptr, mask=tmask, other=0.0).to(tl.float32)
            gv = tl.load(G + rt[None, :] * K + (rk[:, None] + _k))
            # per-64 rmsnorm * sigmoid(gate) on the fly (BK is a multiple of 64)
            xr = tl.reshape(xv, [BK // 64, 64, BT])
            rs = tl.rsqrt(tl.sum(xr * xr, 1) / 64.0 + 1e-5)
            xv = tl.reshape(xr * rs[:, None, :], [BK, BT]) * tl.sigmoid(gv)
            x = xv.to(tl.bfloat16)
        else:
            x = tl.load(x_ptr, mask=tmask, other=0.0)
        acc = tl.dot(w, x, acc)
        w_ptr += BK
        x_ptr += BK
    tl.store(AOUT + rt[None, :] * N + rn[:, None], acc, mask=tmask)
    st = tl.load(STREAM + rt[None, :] * N + rn[:, None], mask=tmask, other=0.0)
    a1v, b1v = _gate_ab(A1, rn)
    as_ = a1v[:, None] * st
    ba = b1v[:, None] * acc
    ok = (rt[None, :] < T) + tl.zeros([BN, BT], tl.int1)
    q0 = tl.sum(tl.where(ok, acc * acc, 0.0), 0)
    q1 = tl.sum(tl.where(ok, as_ * as_, 0.0), 0)
    q2 = tl.sum(tl.where(ok, as_ * ba, 0.0), 0)
    q3 = tl.sum(tl.where(ok, ba * ba, 0.0), 0)
    m = rt < T
    tl.atomic_add(SUMS + rt * 4 + 0, q0, mask=m)
    tl.atomic_add(SUMS + rt * 4 + 1, q1, mask=m)
    tl.atomic_add(SUMS + rt * 4 + 2, q2, mask=m)
    tl.atomic_add(SUMS + rt * 4 + 3, q3, mask=m)


@triton.jit
def _bnd1_apply(AOUT, SUMS, STREAM, A1, B1, G, S1OUT, HOUT, N: tl.constexpr, BC: tl.constexpr):
    """s1 = a1 * stream + b1 * a * rstd(a), h = s1 * rstd(s1) * g, from the four sums (one block per column slice)."""
    t = tl.program_id(0)
    c0 = tl.program_id(1) * BC
    offs = c0 + tl.arange(0, BC)
    s0 = tl.load(SUMS + t * 4 + 0)
    s1q = tl.load(SUMS + t * 4 + 1)
    s2q = tl.load(SUMS + t * 4 + 2)
    s3q = tl.load(SUMS + t * 4 + 3)
    r = tl.rsqrt(s0 / N + 1e-8)
    r1 = tl.rsqrt((s1q + 2.0 * r * s2q + r * r * s3q) / N + 1e-5)
    a = tl.load(AOUT + t * N + offs)
    a1v, b1v = _gate_ab(A1, offs)
    s1 = a1v * tl.load(STREAM + t * N + offs) + b1v * a * r
    tl.store(S1OUT + t * N + offs, s1)
    tl.store(HOUT + t * N + offs, (s1 * r1 * (1.0 + tl.load(G + offs).to(tl.float32))).to(tl.bfloat16))


@triton.jit
def _route_a(P, S: tl.constexpr, T, BIAS, IDS, WTS, LS, LIDS, POOL, PSUM, BC: tl.constexpr):
    """Reduce the pooler/router partials: one block per (token, pooled slice); slice 0 also does the top-8."""
    t = tl.program_id(0)
    j = tl.program_id(1)
    c = j * BC + tl.arange(0, BC)
    pooled = tl.zeros([BC], dtype=tl.float32)
    for s in range(S):
        pooled += tl.load(P + (s * T + t) * 4608 + c)
    tl.store(POOL + t * 4096 + c, pooled)
    tl.atomic_add(PSUM + t, tl.sum(pooled * pooled, 0))
    if j == 0:
        e = tl.arange(0, 256)
        logit = tl.zeros([256], dtype=tl.float32)
        for s in range(S):
            b = P + (s * T + t) * 4608
            logit += tl.load(b + 4096 + e) + tl.load(b + 4352 + e)
        sc = tl.sigmoid(logit * 0.1767766952966369)
        sel = sc + tl.load(BIAS + e)
        k8 = tl.arange(0, 8)
        ids = tl.zeros([8], dtype=tl.int32)
        wv = tl.zeros([8], dtype=tl.float32)
        for k in tl.static_range(8):
            idx = tl.argmax(sel, 0)
            val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
            ids = tl.where(k8 == k, idx, ids)
            wv = tl.where(k8 == k, val, wv)
            sel = tl.where(e == idx, float("-inf"), sel)
        wv = wv / (tl.sum(wv, 0) + 1e-15)
        tl.store(IDS + t * 8 + k8, ids)
        lid = (ids - tl.load(LS) + 256) % 256
        tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
        tl.store(WTS + t * 8 + k8, wv)


@triton.jit
def _route_b(POOL, PSUM, GPRE, XOUT, BC: tl.constexpr, PERMUTED: tl.constexpr):
    """x = rmsnorm(pooled) * (1 + pre_expert_norm), in the expert kernel's layout."""
    t = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    r = tl.rsqrt(tl.load(PSUM + t) / 4096.0 + 1e-5)
    x = tl.load(POOL + t * 4096 + c) * r * (1.0 + tl.load(GPRE + c).to(tl.float32))
    if PERMUTED:
        ch = c // 64
        kk = c % 64
        kb = kk // 16
        rr = kk % 16
        po = ((t * 64 + ch) * 4 + (rr % 8) // 2) * 16 + kb * 4 + (rr // 8) * 2 + (rr % 2)
        tl.store(XOUT + po, x.to(tl.bfloat16).to(tl.float16))
    else:
        tl.store(XOUT + t * 4096 + c, x.to(tl.bfloat16))


@triton.jit
def _bnd2_sums(P, S: tl.constexpr, T, NU: tl.constexpr, S1, A2, B2, GPOST, FOUT, SUMS, BC: tl.constexpr):
    """f = (unpooler partials summed) * g_postffn, and the four sums for the FFN boundary."""
    t = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    m = c < NU
    f = tl.zeros([BC], dtype=tl.float32)
    for s in range(S):
        f += tl.load(P + (s * T + t) * NU + c, mask=m, other=0.0)
    f = f * (1.0 + tl.load(GPOST + c).to(tl.float32))
    tl.store(FOUT + t * 8192 + c, f)
    st = tl.load(S1 + t * 8192 + c)
    a2v, b2v = _gate_ab(A2, c)
    as_ = a2v * st
    bf = b2v * f
    tl.atomic_add(SUMS + t * 4 + 0, tl.sum(f * f, 0))
    tl.atomic_add(SUMS + t * 4 + 1, tl.sum(as_ * as_, 0))
    tl.atomic_add(SUMS + t * 4 + 2, tl.sum(as_ * bf, 0))
    tl.atomic_add(SUMS + t * 4 + 3, tl.sum(bf * bf, 0))


@triton.jit
def _bnd2_apply(FOUT, SUMS, S1, A2, B2, GNEXT, SOUT, HOUT, N: tl.constexpr, BC: tl.constexpr):
    t = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    s0 = tl.load(SUMS + t * 4 + 0)
    s1q = tl.load(SUMS + t * 4 + 1)
    s2q = tl.load(SUMS + t * 4 + 2)
    s3q = tl.load(SUMS + t * 4 + 3)
    r = tl.rsqrt(s0 / N + 1e-8)
    r2 = tl.rsqrt((s1q + 2.0 * r * s2q + r * r * s3q) / N + 1e-5)
    a2v, b2v = _gate_ab(A2, c)
    s2 = a2v * tl.load(S1 + t * N + c) + b2v * tl.load(FOUT + t * N + c) * r
    tl.store(SOUT + t * N + c, s2)
    tl.store(HOUT + t * N + c, (s2 * r2 * (1.0 + tl.load(GNEXT + c).to(tl.float32))).to(tl.bfloat16))


@triton.jit
def _gate_ab(G, offs):
    """alpha, beta of a residual gate, from the checkpoint's BF16 weight (as SGLang's fused module does)."""
    w = tl.load(G + offs).to(tl.float32) / 0.3
    beta = tl.sigmoid(w)
    alpha = tl.sqrt(tl.maximum(tl.sigmoid(-w) * (1.0 + beta), 1e-3))
    return alpha, beta


@triton.jit
def _boundary1(P, S: tl.constexpr, T, STREAM, A1, B1, G, S1OUT, HOUT, USE_PDL: tl.constexpr = False):
    t = tl.program_id(0)
    offs = tl.arange(0, 8192)
    a1v, b1v = _gate_ab(A1, offs)              # weights and the layer's input stream: no producer dependency
    stv = tl.load(STREAM + t * 8192 + offs)
    gw = 1.0 + tl.load(G + offs).to(tl.float32)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()   # P (the o-projection) is the producer's output
    a = tl.zeros([8192], dtype=tl.float32)
    for s in range(S):
        a += tl.load(P + (s * T + t) * 8192 + offs)
    n = a * tl.rsqrt(tl.sum(a * a, 0) / 8192.0 + 1e-8)
    s1 = a1v * stv + b1v * n
    tl.store(S1OUT + t * 8192 + offs, s1)
    hh = s1 * tl.rsqrt(tl.sum(s1 * s1, 0) / 8192.0 + 1e-5) * gw
    tl.store(HOUT + t * 8192 + offs, hh.to(tl.bfloat16))


@triton.jit
def _route(P, S: tl.constexpr, T, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, PERMUTED: tl.constexpr, USE_PDL: tl.constexpr = False):
    t = tl.program_id(0)
    e = tl.arange(0, 256)
    logit = tl.zeros([256], dtype=tl.float32)
    pooled = tl.zeros([4096], dtype=tl.float32)
    c = tl.arange(0, 4096)
    biasv = tl.load(BIAS + e)                      # weights and local_start: no producer dependency
    gpre = 1.0 + tl.load(GPRE + c).to(tl.float32)
    lsv = tl.load(LS)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()   # P (pooler + router) is the producer's output
    for s in range(S):
        b = P + (s * T + t) * 4608
        logit += tl.load(b + 4096 + e) + tl.load(b + 4352 + e)
        pooled += tl.load(b + c)
    sc = tl.sigmoid(logit * 0.1767766952966369)
    sel = sc + biasv
    k8 = tl.arange(0, 8)
    ids = tl.zeros([8], dtype=tl.int32)
    wv = tl.zeros([8], dtype=tl.float32)
    for k in tl.static_range(8):
        idx = tl.argmax(sel, 0)
        val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
        ids = tl.where(k8 == k, idx, ids)
        wv = tl.where(k8 == k, val, wv)
        sel = tl.where(e == idx, float("-inf"), sel)
    wv = wv / (tl.sum(wv, 0) + 1e-15)
    tl.store(IDS + t * 8 + k8, ids)
    lid = (ids - lsv + 256) % 256
    tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
    tl.store(WTS + t * 8 + k8, wv)
    x = pooled * tl.rsqrt(tl.sum(pooled * pooled, 0) / 4096.0 + 1e-5) * gpre
    # BF16 activation, stored as exact fp16 in the MoE kernel's fragment-permuted layout
    ch = c // 64
    kk = c % 64
    kb = kk // 16
    r = kk % 16
    if PERMUTED:
        po = ((t * 64 + ch) * 4 + (r % 8) // 2) * 16 + kb * 4 + (r // 8) * 2 + (r % 2)
        tl.store(XOUT + po, x.to(tl.bfloat16).to(tl.float16))
    else:
        tl.store(XOUT + t * 4096 + c, x.to(tl.bfloat16))


@triton.jit
def _route_split(POOLED, LOGITS, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, PERMUTED: tl.constexpr, USE_PDL: tl.constexpr = False):
    """Same as _route but the pooler output (bf16) and the router logits (fp32) arrive as separate tensors."""
    t = tl.program_id(0)
    e = tl.arange(0, 256)
    c = tl.arange(0, 4096)
    biasv = tl.load(BIAS + e)                      # weights and local_start: no producer dependency
    gpre = 1.0 + tl.load(GPRE + c).to(tl.float32)
    lsv = tl.load(LS)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()   # pooled / logits are the producer's output
    logit = tl.load(LOGITS + t * 256 + e)
    pooled = tl.load(POOLED + t * 4096 + c).to(tl.float32)
    sc = tl.sigmoid(logit * 0.1767766952966369)
    sel = sc + biasv
    k8 = tl.arange(0, 8)
    ids = tl.zeros([8], dtype=tl.int32)
    wv = tl.zeros([8], dtype=tl.float32)
    for k in tl.static_range(8):
        idx = tl.argmax(sel, 0)
        val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
        ids = tl.where(k8 == k, idx, ids)
        wv = tl.where(k8 == k, val, wv)
        sel = tl.where(e == idx, float("-inf"), sel)
    wv = wv / (tl.sum(wv, 0) + 1e-15)
    tl.store(IDS + t * 8 + k8, ids)
    lid = (ids - lsv + 256) % 256
    tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
    tl.store(WTS + t * 8 + k8, wv)
    x = pooled * tl.rsqrt(tl.sum(pooled * pooled, 0) / 4096.0 + 1e-5) * gpre
    ch = c // 64
    kk = c % 64
    kb = kk // 16
    r = kk % 16
    if PERMUTED:
        po = ((t * 64 + ch) * 4 + (r % 8) // 2) * 16 + kb * 4 + (r // 8) * 2 + (r % 2)
        tl.store(XOUT + po, x.to(tl.bfloat16).to(tl.float16))
    else:
        tl.store(XOUT + t * 4096 + c, x.to(tl.bfloat16))


@triton.jit
def _moe_align(IDS, LS, R, CNT, OFF, SORTED, RPOS, ACTIVE, NACT, CH: tl.constexpr):
    start = tl.load(LS)
    ex = tl.arange(0, 32)
    cnt = tl.zeros([32], dtype=tl.int32)
    for c0 in range(0, R, CH):
        r = c0 + tl.arange(0, CH)
        rv = r < R
        ids = tl.load(IDS + r, mask=rv, other=0)
        local = (ids - start + 256) % 256
        isl = rv & (local < 32)
        oh = ((local[:, None] == ex[None, :]) & isl[:, None]).to(tl.int32)
        cnt += tl.sum(oh, 0)
    off = tl.cumsum(cnt, 0) - cnt
    run = off
    for c0 in range(0, R, CH):
        r = c0 + tl.arange(0, CH)
        rv = r < R
        ids = tl.load(IDS + r, mask=rv, other=0)
        local = (ids - start + 256) % 256
        isl = rv & (local < 32)
        oh = ((local[:, None] == ex[None, :]) & isl[:, None]).to(tl.int32)
        rank = tl.sum((tl.cumsum(oh, 0) - oh) * oh, 1)
        pos = tl.sum(run[None, :] * oh, 1) + rank
        tl.store(RPOS + r, tl.where(isl, pos, -1), mask=rv)
        tl.store(SORTED + pos, r, mask=isl)
        run += tl.sum(oh, 0)
    tl.store(CNT + ex, cnt)
    tl.store(OFF + ex, off)
    act = (cnt > 0).to(tl.int32)
    arank = tl.cumsum(act, 0) - act
    tl.store(ACTIVE + arank, ex, mask=act > 0)
    tl.store(NACT, tl.sum(act, 0))


# ----------------------------------------------------------------------------------------------
# MoE experts (NVFP4 W4A16)
# ----------------------------------------------------------------------------------------------
@triton.jit
def _dequant(w8, s8, BN: tl.constexpr, BK: tl.constexpr):
    """w8 uint8 [BN, BK/2] packed e2m1 (low nibble first), s8 uint8 [BN, BK/16] e4m3 -> bf16 [BN, BK]."""
    w = w8.to(tl.int32)
    codes = tl.reshape(tl.join(w & 0xF, w >> 4), [BN, BK])
    v = (((codes & 7) << 9) | ((codes & 8) << 12)).to(tl.int16).to(tl.float16, bitcast=True)
    sc = _e4m3_to_f32(s8) * 16384.0
    scf = tl.reshape(tl.broadcast_to(sc[:, :, None], [BN, BK // 16, 16]), [BN, BK])
    return (v.to(tl.float32) * scf).to(tl.bfloat16)


@triton.jit
def _moe_w13(X, W13, SF13, GS13, SORTED, CNT, OFF, ACTIVE, NACT, INTER,
             BN: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr):
    a = tl.program_id(0)
    j = tl.program_id(1)
    if a >= tl.load(NACT):
        return
    e = tl.load(ACTIVE + a)
    n_e = tl.load(CNT + e)
    off = tl.load(OFF + e)
    rows = j * BN + tl.arange(0, BN)
    wbase = W13 + e.to(tl.int64) * (8192 * 2048)
    sbase = SF13 + e.to(tl.int64) * (8192 * 256)
    gs = tl.load(GS13 + e)
    tt = tl.arange(0, BT)
    for t0 in range(0, n_e, BT):
        rv = (t0 + tt) < n_e
        route = tl.load(SORTED + off + t0 + tt, mask=rv, other=0)
        tok = route // 8
        accg = tl.zeros([BN, BT], dtype=tl.float32)
        accu = tl.zeros([BN, BT], dtype=tl.float32)
        for k in range(0, 4096, BK):
            x = tl.load(X + tok[None, :] * 4096 + (k + tl.arange(0, BK))[:, None], mask=rv[None, :], other=0.0)
            cb = k // 2 + tl.arange(0, BK // 2)
            cs = k // 16 + tl.arange(0, BK // 16)
            wg = tl.load(wbase + rows[:, None] * 2048 + cb[None, :])
            wu = tl.load(wbase + (rows + 4096)[:, None] * 2048 + cb[None, :])
            sg = tl.load(sbase + rows[:, None] * 256 + cs[None, :])
            su = tl.load(sbase + (rows + 4096)[:, None] * 256 + cs[None, :])
            accg = tl.dot(_dequant(wg, sg, BN, BK), x, accg)
            accu = tl.dot(_dequant(wu, su, BN, BK), x, accu)
        gg = accg * gs
        uu = accu * gs
        inter = gg / (1.0 + tl.exp(-gg)) * uu
        tl.store(INTER + (off + t0 + tt)[None, :] * 4096 + rows[:, None], inter.to(tl.bfloat16), mask=rv[None, :])


@triton.jit
def _moe_w2(INTER, W2, SF2, GS2, POSTW, CNT, OFF, ACTIVE, NACT, Z, SSP,
            BN: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, NT: tl.constexpr):
    a = tl.program_id(0)
    j = tl.program_id(1)
    if a >= tl.load(NACT):
        return
    e = tl.load(ACTIVE + a)
    n_e = tl.load(CNT + e)
    off = tl.load(OFF + e)
    rows = j * BN + tl.arange(0, BN)
    wbase = W2 + e.to(tl.int64) * (4096 * 2048)
    sbase = SF2 + e.to(tl.int64) * (4096 * 256)
    gs = tl.load(GS2 + e)
    pw = tl.load(POSTW + rows).to(tl.float32)
    tt = tl.arange(0, BT)
    for t0 in range(0, n_e, BT):
        rv = (t0 + tt) < n_e
        rr = off + t0 + tt
        acc = tl.zeros([BN, BT], dtype=tl.float32)
        for k in range(0, 4096, BK):
            x = tl.load(INTER + rr[None, :] * 4096 + (k + tl.arange(0, BK))[:, None], mask=rv[None, :], other=0.0)
            cb = k // 2 + tl.arange(0, BK // 2)
            cs = k // 16 + tl.arange(0, BK // 16)
            w = tl.load(wbase + rows[:, None] * 2048 + cb[None, :])
            s = tl.load(sbase + rows[:, None] * 256 + cs[None, :])
            acc = tl.dot(_dequant(w, s, BN, BK), x, acc)
        z = acc * gs * pw[:, None]
        tl.store(Z + rr[None, :] * 4096 + rows[:, None], z.to(tl.bfloat16), mask=rv[None, :])
        tl.store(SSP + rr * NT + j, tl.sum(z * z, 0), mask=rv)


@triton.jit
def _pexp_norm_sum(x_ptr, w_ptr, s_ptr, out_ptr, ids_ptr, TOPK: tl.constexpr, H: tl.constexpr, eps,
                   BLOCK_H: tl.constexpr):
    """post-expert norm (weight first) * routing weight, summed over slots. Same kernel as SGLang's
    fused_post_expert_norm_weighted_sum (Apache 2.0), called directly: its wrapper falls back to a chain of
    ~8 torch ops below 32 tokens, which costs 39-43 us on the judge instead of ~6."""
    tok = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    w = 1.0 + tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)   # the checkpoint's BF16 w; 1 + w formed here
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for k in tl.static_range(TOPK):
        if tl.load(ids_ptr + tok * TOPK + k) >= 0:
            x = tl.load(x_ptr + (tok * TOPK + k) * H + offs, mask=mask, other=0.0).to(tl.float32)
            y = x * w
            y = y * tl.rsqrt(tl.sum(y * y, axis=0) / H + eps)
            acc += y * tl.load(s_ptr + tok * TOPK + k).to(tl.float32)
    tl.store(out_ptr + tok * H + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _moe_combine(Z, SSP, RPOS, WTS, M, NT: tl.constexpr):
    t = tl.program_id(0)
    c = tl.arange(0, 4096)
    jj = tl.arange(0, NT)
    acc = tl.zeros([4096], dtype=tl.float32)
    for k in range(8):
        p = tl.load(RPOS + t * 8 + k)
        if p >= 0:
            ss = tl.sum(tl.load(SSP + p * NT + jj), 0)
            w = tl.load(WTS + t * 8 + k) * tl.rsqrt(ss / 4096.0 + 1e-8)
            acc += w * tl.load(Z + p * 4096 + c).to(tl.float32)
    tl.store(M + t * 4096 + c, acc.to(tl.bfloat16))


@triton.jit
def _boundary2(P, S: tl.constexpr, T, NU: tl.constexpr, S1, A2, B2, GPOST, GNEXT, SOUT, HOUT,
               USE_PDL: tl.constexpr = False):
    t = tl.program_id(0)
    offs = tl.arange(0, 8192)
    m = offs < NU
    gp = 1.0 + tl.load(GPOST + offs).to(tl.float32)      # weights, and S1 from _boundary1: every kernel between them (pooler/router, route,
    gn = 1.0 + tl.load(GNEXT + offs).to(tl.float32)      # moe, unpooler) triggers only after its own wait, so _boundary1 completed before this kernel began
    a2v, b2v = _gate_ab(A2, offs)
    s1v = tl.load(S1 + t * 8192 + offs)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()   # P (the unpooler) is the producer's output
    f = tl.zeros([8192], dtype=tl.float32)
    for s in range(S):
        f += tl.load(P + (s * T + t) * NU + offs, mask=m, other=0.0)
    f = f * gp
    n = f * tl.rsqrt(tl.sum(f * f, 0) / 8192.0 + 1e-8)
    s2 = a2v * s1v + b2v * n
    tl.store(SOUT + t * 8192 + offs, s2)
    hh = s2 * tl.rsqrt(tl.sum(s2 * s2, 0) / 8192.0 + 1e-5) * gn
    tl.store(HOUT + t * 8192 + offs, hh.to(tl.bfloat16))


# ----------------------------------------------------------------------------------------------

def _marlin_bank(weights):
    """The local experts in Marlin's layout (SGLang's own loader step), as baseline.py builds them."""
    import types
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
    from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
    device = weights["w13"].device
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous().clone(),
        w2_weight=weights["w2"].contiguous().clone(),
        w13_weight_scale=weights["w13_sf"].contiguous().clone(),
        w2_weight_scale=weights["w2_sf"].contiguous().clone(),
        w13_weight_scale_2=weights["w13_gs"].contiguous().clone(),
        w2_weight_scale_2=weights["w2_gs"].contiguous().clone(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
    expert_map[:LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "expert_map": expert_map, "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
    }


def _marlin_experts(x, topk_weights, topk_ids, state):
    """MuseSpark1xFp4Experts._marlin_expert_outputs + post-expert-norm finalize (as baseline.py).
    moe_block_size and use_atomic_add come from the prepare()-time sweep in _tune."""
    from sgl_kernel.scalar_type import scalar_types
    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    local_ids = topk_ids  # already (expert - local_start) % 256 mapped to 0..31 / -1 by the router kernel
    mc = state.get("marlin_cfg", {})
    cfgm = mc.get(num_tokens, mc.get(_bucket(num_tokens)))
    if cfgm is None:
        for block_size_m in (8, 16, 32, 48, 64):
            if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
                break
        atomic = False
    else:
        block_size_m, atomic = cfgm
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=atomic,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    m = torch.empty((num_tokens, size_k), device=x.device, dtype=x.dtype)
    _pexp_norm_sum[(num_tokens,)](y, state["g_postexp"], topk_weights, m, local_ids, TOPK=TOP_K,
                                  H=size_k, eps=1e-8, BLOCK_H=4096, num_warps=8, num_stages=2)
    return m


def _graph_time(fn, flush, reps=9):
    """Median graph-replay time with a cold L2. The flush is a READ sweep: zeroing a buffer leaves ~120 MB of
    dirty lines to write back, which inflates the next kernel by 15-25% (measured on the judge's H200) and not
    evenly across kernels, so it would skew every choice below."""
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(reps):
        _FLUSH_SINK.copy_(flush.sum())
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]


_FLUSH_SINK = None


def _calib_bias(st, dev, g, log=False):
    """Give the 32 local experts a skewed bias so that synthetic routing matches the judge's busiest-range load:
    ~1.6 local routes per token and ~15.7 distinct local experts at 32 tokens (reference statistics). Two
    parameters (skew width, bias size) found by a torch simulation of the real router; no input data involved."""
    router = st["pr"][D_EXPERT:].float()  # [256 hi; 256 lo] rows of width HIDDEN
    router = router[:N_EXPERTS] + router[N_EXPERTS:]
    gc = torch.Generator(device=dev).manual_seed(987654321)
    h = torch.randn(512, HIDDEN, generator=gc, device=dev)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-5)
    sc = torch.sigmoid(h.double() @ router.double().T * 0.1767766952966369).float()
    idx = torch.arange(LOCAL_EXPERTS, device=dev).float()

    def stats(shape, delta):
        sel = sc + st["bias"]
        sel[:, :LOCAL_EXPERTS] += delta * shape
        ids = sel.topk(TOP_K, dim=-1).indices
        loc = ids < LOCAL_EXPERTS
        rpt = float(loc.float().sum(-1).mean())
        counts = [torch.unique(ids[i:i + 32][loc[i:i + 32]]).numel() for i in range(0, 512, 32)]
        d32 = sum(counts) / len(counts)
        return rpt, d32

    best = None
    for width in (2.0, 3.0, 4.5, 6.0, 9.0):
        shape = torch.exp(-idx / width)
        lo, hi = 0.0, 3.0
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            if stats(shape, mid)[0] < 1.6:
                lo = mid
            else:
                hi = mid
        rpt, d32 = stats(shape, hi)
        if best is None or abs(d32 - 15.7) < abs(best[3] - 15.7):
            best = (shape, hi, rpt, d32)
    st["bias"][:LOCAL_EXPERTS] += best[1] * best[0]
    if log:
        print(f"calib delta {best[1]:.3f} routes/token {best[2]:.2f} experts@32 {best[3]:.1f}", flush=True)


def _tune(st, log=False):
    """prepare()-time choice of the post-attention path per token count: the persistent megakernel or the
    multi-kernel pipeline, and which routed-expert kernel. Uses synthetic activations only (never input data)."""
    dev = st["qkv"].device
    global _FLUSH_SINK
    flush = torch.zeros(int(60e6), dtype=torch.float32, device=dev)  # 240 MB, read-swept between replays
    _FLUSH_SINK = torch.zeros((), dtype=torch.float32, device=dev)
    g = torch.Generator(device=dev).manual_seed(12345)
    # Synthetic routing is uniform over all 256 experts, but the judge picks each input's local range as the
    # busiest one: ~1.6 local routes per token (2.6 at T=1) over few distinct experts. Calibrate a skewed bias
    # on the local experts (torch-side simulation with the real router weights) so the tuned load is realistic.
    bias0 = st["bias"].clone()
    _calib_bias(st, dev, g, log)
    st["moe_pick"] = {}
    st["gluev2"] = {}
    st["marlin_cfg"] = {}
    st["mm_pick"] = {}
    pb_ok = {}
    for T in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        data = {"stream": torch.randn(T, HIDDEN, generator=g, device=dev),
                "local_start": torch.zeros(1, dtype=torch.int32, device=dev)}
        o_in = (torch.randn(T, 1024, generator=g, device=dev) * 0.1).to(torch.bfloat16)
        if T <= 4:  # redraw until the token(s) route to about as many local experts as the judge's inputs do
            want = max(2, int(round(2.57 * T / 1.0)) if T == 1 else 2 * T)
            for _ in range(40):
                _o, _s, _ids = _after_attention(data, st, o_in, None, T)
                if int(((_ids >= 0) & (_ids < LOCAL_EXPERTS)).sum()) >= min(want, 8 * T):
                    break
                data["stream"] = torch.randn(T, HIDDEN, generator=g, device=dev)
                o_in = (torch.randn(T, 1024, generator=g, device=dev) * 0.1).to(torch.bfloat16)
        best, bestt, bestv2 = None, 1e30, True
        times, dfltv2 = {}, False
        msg = []
        nloc = ndis = -1
        for name, w, K in (("qkv", st["qkv"], HIDDEN), ("o", st["o"], 1024), ("pr", st["pr"], HIDDEN),
                           ("un", st["unpool_slice"] if T <= 8 else st["unpool"], D_EXPERT)):
            xd = (torch.randn(T, K, generator=g, device=dev) * 0.1).to(torch.bfloat16)
            # the pooler + router candidates are timed WITH the routing kernel / the fused routing tail they imply
            r_ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
            r_lids = torch.empty((T, 8), device=dev, dtype=torch.int32)
            r_wts = torch.empty((T, 8), device=dev, dtype=torch.float32)
            r_xm = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)

            def _pr_split(pooled, logits):
                _route_split[(T,)](pooled, logits, st["bias"], st["g_preexp"], r_ids, r_wts, r_xm, data["local_start"],
                                   r_lids, PERMUTED=False, num_warps=8, USE_PDL=_pdl(T), launch_pdl=_pdl(T))

            def _pr_route(PS):   # the combined pooler+router GEMM output -> the routing kernel that consumes it
                P_, S_ = PS
                _route[(T,)](P_, S_, T, st["bias"], st["g_preexp"], r_ids, r_wts, r_xm, data["local_start"], r_lids,
                             PERMUTED=False, num_warps=8, USE_PDL=_pdl(T), launch_pdl=_pdl(T))

            def _dense_t(pick):   # time what dense() runs for this pick (+ the routing kernel for "pr")
                st["mm_pick"][(T, name)] = pick
                if name == "pr":
                    return _graph_time(lambda: _pr_route(dense(xd, w, T, st, name)), flush, reps=5)
                return _graph_time(lambda: dense(xd, w, T, st, name), flush, reps=5)

            if st.get("bfp") is not None and T <= BFP_MAX_T:   # above BFP_MAX_T a "bfp" pick runs as cuBLAS: not a candidate
                st["mm_pick"][(T, name)] = "bfp"
                try:
                    if name == "pr":
                        t_bfp = _graph_time(lambda: _pr_split(_bfp_gemm(st, "pool", xd, T, D_EXPERT, HIDDEN),
                                                              _bfp_gemm(st, "router", xd, T, N_EXPERTS, HIDDEN)),
                                            flush, reps=5)
                    else:
                        nn = w.shape[0]
                        t_bfp = _graph_time(lambda: _bfp_gemm(st, name, xd, T, nn, K), flush, reps=5)
                except Exception:  # noqa: BLE001
                    t_bfp = 1e30
            else:
                t_bfp = 1e30
            try:
                t_tri = _dense_t(False)
            except Exception:  # noqa: BLE001
                t_tri = 1e30
            try:
                t_mm = _dense_t(True)
            except Exception:  # noqa: BLE001
                t_mm = 1e30
            t_bfpf = 1e30
            if name == "pr" and T <= 32 and st.get("bfp") is not None and "bfp_fused_pooler_router" in globals():
                try:
                    _bfp_pdl(T)
                    rt = _fused_route(st, data, r_ids, r_lids, r_wts, r_xm)
                    if rt is not None:
                        t_bfpf = _graph_time(lambda: bfp_fused_pooler_router(xd, st["bfp"], rt), flush, reps=5)
                    else:
                        t_bfpf = _graph_time(lambda: _pr_split(*bfp_fused_pooler_router(xd, st["bfp"])), flush, reps=5)
                except Exception as ex:  # noqa: BLE001  (sm_80 or an older bf16_proj file: not offered)
                    t_bfpf = 1e30
                    if log:
                        msg.append(f"bfpfX:{repr(ex)[:120]}")
            if log and name == "pr":
                msg.append(f"pr[T{t_tri:.0f} M{t_mm:.0f} B{t_bfp:.0f} F{t_bfpf:.0f}]")
            t_mm16 = 1e30
            if name != "pr":   # the router logits stay fp32 (hi + lo rows); the other three may come out bf16 as SGLang's
                st["mm_pick"][(T, name)] = "mm16"
                try:
                    t_mm16 = _graph_time(lambda: dense(xd, w, T, st, name), flush, reps=5)
                except Exception:  # noqa: BLE001
                    t_mm16 = 1e30
            opts = {False: t_tri, True: t_mm, "bfp": t_bfp, "bfpf": t_bfpf, "mm16": t_mm16}
            st.setdefault("mm_times", {})[(T, name)] = opts
            bestp = min(opts, key=lambda p: opts[p])
            st["mm_pick"][(T, name)] = bestp
            if log:
                msg.append(f"{name}{ {False: 'T', True: 'M', 'bfp': 'B', 'bfpf': 'F', 'mm16': 'm'}[bestp] }:{opts[bestp]:.0f}")
        if st["bank"] is not None and T <= 384:
            xm = (torch.randn(T, D_EXPERT, generator=g, device=dev) * 0.1).to(torch.bfloat16)
            raw = torch.randint(0, N_EXPERTS, (T, 8), generator=g, device=dev, dtype=torch.int32)
            # the convention the router kernel writes: local index 0..31, -1 for another GPU's expert
            mids = torch.where(raw < LOCAL_EXPERTS, raw, torch.full_like(raw, -1)).contiguous()
            mwt = torch.full((T, 8), 0.125, device=dev)
            bestm, bestmt = None, 1e30
            for bs in (8, 16, 32, 48, 64):
                for atomic in (False, True):
                    st["marlin_cfg"][T] = (bs, atomic)
                    try:
                        tm = _graph_time(lambda: _marlin_experts(xm, mwt, mids, st), flush, reps=5)
                    except Exception:  # noqa: BLE001
                        continue
                    if tm < bestmt:
                        bestm, bestmt = (bs, atomic), tm
            if bestm is not None:
                st["marlin_cfg"][T] = bestm
                if log:
                    msg.append(f"mcfg{bestm[0]}{'a' if bestm[1] else ''}:{bestmt:.0f}")
            else:
                st["marlin_cfg"].pop(T, None)
        # my own MoE kernel (FP16 expert input) is offered only where neither SGLang's Marlin nor the moe agent's
        # kernel is available (never on the judge image): both beat it everywhere, and SGLang keeps BF16 there
        cands = [c for c in ("marlin", "agent") if (c != "marlin" or st["bank"] is not None)
                 and (c != "agent" or st["ma_ok"])]
        for moe in cands:
            if moe == "marlin" and st["bank"] is None:
                continue
            if moe == "agent" and not st["ma_ok"]:
                continue
            st["moe_pick"][T] = moe
            for v2 in (True, False):
                st["gluev2"][T] = v2
                try:
                    t = _graph_time(lambda: _after_attention(data, st, o_in, None, T), flush)
                except Exception as ex:  # noqa: BLE001
                    if log:
                        msg.append(f"{moe[:3]}{2 if v2 else 1}:X{repr(ex)[:40]}")
                    continue
                if log:
                    msg.append(f"{moe[:3]}{2 if v2 else 1}:{t:.0f}")
                    if nloc < 0:
                        _o, _s, _ids = _after_attention(data, st, o_in, None, T)
                        loc = _ids[(_ids >= 0) & (_ids < LOCAL_EXPERTS)]
                        nloc, ndis = int(loc.numel()), int(torch.unique(loc).numel())
                if t < times.get(moe, 1e30):
                    times[moe] = t
                    if moe == DEFAULT_MOE:
                        dfltv2 = v2
                if t < bestt:
                    best, bestt, bestv2 = moe, t, v2
        if (DEFAULT_MOE in times and best is not None and best != DEFAULT_MOE
                and times[best] > TUNE_MARGIN * times[DEFAULT_MOE]):
            best, bestv2 = DEFAULT_MOE, dfltv2
        st["moe_pick"][T] = best if best is not None else DEFAULT_MOE
        st["gluev2"][T] = bestv2
        tb = 1e30
        if log:
            print(f"T{T}/{nloc}r{ndis}e " + " ".join(msg) + (f" pb:{tb:.0f}" if tb < 1e9 else ""), flush=True)
        pb_ok[T] = tb < bestt
    pbmax = 0
    for T in (1, 2, 4, 8, 16, 32):
        if pb_ok.get(T):
            pbmax = T
        else:
            break
    st["bias"].copy_(bias0)
    st["pbmax"] = pbmax if FORCE_PBMAX is None else FORCE_PBMAX
    if log:
        print("tune: pbmax", pbmax, "moe", st["moe_pick"], "gluev2", st["gluev2"], flush=True)
    del flush


def prepare(weights):
    dev = weights["qkv_gate"].device
    router = weights["router"].float()
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    st = {}
    st["qkv"] = weights["qkv_gate"].contiguous()
    st["o"] = weights["o"].contiguous()
    st["pr"] = torch.cat([weights["pooler"], hi, lo], 0).contiguous()
    st["unpool"] = weights["unpooler"].contiguous()
    st["unpool_slice"] = weights["unpooler"][:1024].contiguous()
    # Weight storage, to the letter of TASK.md: every BF16 checkpoint weight stays BF16. The norms are
    # applied as (1 + w), which my kernels now form in FP32 at use; the one tensor SGLang itself
    # precomputes (the post-expert effective weight) is stored exactly as SGLang stores it, in BF16.
    st["gate1"] = weights["attn_residual_gate"].contiguous()
    st["gate2"] = weights["ffn_residual_gate"].contiguous()
    st["ab1"] = _gate_coeffs(st["gate1"])   # (alpha, beta) fp32 for the fused boundary tails (FUSE_BOUNDARY)
    st["ab2"] = _gate_coeffs(st["gate2"])
    st["g_preffn"] = weights["pre_ffn_norm"].contiguous()
    st["g_preexp"] = weights["pre_expert_norm"].contiguous()
    st["g_postexp"] = weights["post_expert_norm"].contiguous()   # BF16 as given; _pexp_norm_sum forms 1 + w
    st["g_postffn"] = weights["post_ffn_norm"].contiguous()
    st["g_next"] = weights["next_pre_attn_norm"].contiguous()
    st["bias"] = weights["router_bias"].float().contiguous()
    st["invf"] = (1.0 / (500000.0 ** (torch.arange(0, 64, 2, device=dev).float() / 64))).contiguous()
    import threading
    errs = []

    def _build(fn):
        try:
            fn()
        except Exception as ex:  # noqa: BLE001
            errs.append(ex)

    sfx = _moe_sfx(weights)
    builds = [_att7] + [(lambda d=d: moe__ext(d)) for d in _moe_builds(sfx)]
    ths = [threading.Thread(target=_build, args=(f,)) for f in builds]  # ~85 s of nvcc, in parallel
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    if errs:
        raise errs[0]
    _ma_prepare(weights, st)
    _bfp_prepare(weights, st)
    for selftest in (_route_selftest, _boundary_selftest):   # the fused tails, each enabled only when it matches
        try:
            selftest(st)
        except Exception as ex:  # noqa: BLE001
            print(f"{selftest.__name__} failed: {repr(ex)[:300]}", flush=True)
    st["nsm"] = torch.cuda.get_device_properties(dev).multi_processor_count
    st["fa3"] = None
    st["bank"] = None
    st["att_ok"] = True
    st["ma_ok"] = True
    st["pbmax"] = 2
    st["moe_pick"] = {}
    try:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache
        st["fa3"] = flash_attn_with_kvcache
        st["bank"] = _marlin_bank(weights)
    except Exception as ex:  # noqa: BLE001  (local GPUs without sgl_kernel: fall back to our own kernels)
        print("fallback (no sgl_kernel/sglang):", repr(ex)[:200])
    try:
        _tune(st, log=_TUNE_LOG)
    except Exception as ex:  # noqa: BLE001
        print("tune failed:", repr(ex)[:200], flush=True)
        st["pbmax"] = 2
        st["moe_pick"] = {}
        st["gluev2"] = {}
        st["marlin_cfg"] = {}
        st["mm_pick"] = {}
    for k, v in PIN_PICK.items():      # in-chain winners the isolated tuner sometimes misses (no PDL in its timing)
        if isinstance(v, tuple):
            times = st.get("mm_times", {}).get(k, {})
            v = min(v, key=lambda p: times.get(p, 1e30))
        st.setdefault("mm_pick", {})[k] = v
    torch.cuda.synchronize()
    if st.get("att_ok"):
        _attn_selftest(st)     # every attention plan family once, on synthetic inputs
    _check_guards()            # after the tuner (every moe build) and the attention self-test
    return st


def _attn_cfg(B, L, sliding):
    BM = 16 if L == 1 else 64
    if sliding:
        nsplit = max(1, min(32, 264 // B))
    else:
        nsplit = max(1, min(132, 528 // B))
    return BM, nsplit


# The attention module's extension, built once for this layer: its source with the layer's measured flags (the
# standalone entry's defaults are QDIRECT=1, PK2=1, W0ENABLE=0; this build is their "v298w1").
ATT_BUILD_FLAGS = ()   # A/B 7d23a37c vs c8b5aab9: the standalone defaults (QDIRECT=1 PK2=1 W0ENABLE=0) measured 0.3-3 us
                       # faster per attention launch on 10 of 12 shapes than the layer's old build (0/0/1)


def _att7():
    global att__MOD
    if att__MOD is None:
        att__MOD = att__build("w1", ATT_BUILD_FLAGS)
    return att__MOD


ATT_HAS_PDL = True   # does the embedded attention launcher take pdl= (their v63+)?


def _decode_attention7(data, state, qkv):
    """The attention module's own launcher (att__launch): plan, workspace sizes and the coop / w0 bits always match
    the kernel version composed in. This layer's config already carries their keys (heads, head_dim, page_size,
    window, rope_theta, scales)."""
    cfg = data["config"]
    pdl = {"pdl": _pdl(cfg["batch"] * cfg["q_len"])} if ATT_HAS_PDL else {}   # their launcher's PDL switch
    out = att__launch({"config": cfg, "qkv": qkv, "seq_lens": data["seq_lens"], "cache_loc": data["cache_loc"],
                    "k_cache": data["k_cache"], "v_cache": data["v_cache"], "page_table": data["page_table"],
                    "k_scale": data["k_scale"], "v_scale": data["v_scale"]}, **pdl)
    return out.view(cfg["batch"] * cfg["q_len"], 1024)


# Guards around every component buffer this glue allocates or caches (POD.md item 6 follow-up): the moe
# scratch, the attention workspace and its cluster buffers. Each is a view into a larger allocation whose
# tail is SCRATCH_GUARD sentinel words; prepare() checks every tail after the tuner and an attention self-test
# have run each path, so an out-of-bounds write fails loudly there instead of corrupting a neighbour.
_GUARDS = []
_GUARDED_IDS = set()


def _guarded(t):
    nbytes = t.numel() * t.element_size()
    words = (nbytes + 3) // 4
    full = torch.full((words + SCRATCH_GUARD,), SCRATCH_SENTINEL, dtype=torch.int32, device=t.device)
    full.view(torch.uint8)[:nbytes].copy_(t.reshape(-1).view(torch.uint8))
    v = full.view(torch.uint8)[:nbytes].view(t.dtype).view(t.shape)
    _GUARDS.append((full, words))
    _GUARDED_IDS.add(id(v))
    return v


def _check_guards():
    torch.cuda.synchronize()
    for full, words in _GUARDS:
        bad = int((full[words:] != SCRATCH_SENTINEL).sum())
        if bad:
            raise RuntimeError(f"a component kernel wrote {bad} words past a {words * 4}-byte buffer")


# The attention module's allocators are rebound (its launcher looks them up by name at every call) so that whatever
# they newly allocate is guarded and cached back into their own dictionaries.
_att_workspace_raw, _att_cluster_bufs_raw = att__workspace, att__cluster_bufs


def att__workspace(n_rows, device):          # their allocator, then guard whatever it newly allocated
    ws = _att_workspace_raw(n_rows, device)
    if id(ws[0]) not in _GUARDED_IDS:
        ws = tuple(_guarded(t) for t in ws)
        att__WS[str(device)] = ws
    return ws


def att__cluster_bufs(n_ctas, L, device):
    cb = _att_cluster_bufs_raw(n_ctas, L, device)
    if id(cb[0]) not in _GUARDED_IDS:
        cb = tuple(_guarded(t) for t in cb)
        att__CL[str(device)] = cb
    return cb


def _attn_selftest(st):
    """Run the attention kernel once per plan family on synthetic inputs (problem constants, no data), so its
    lazily allocated buffers exist and are guarded before any timed call, and the guards see every path."""
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(4242)
    nq = st["qkv"].shape[0]
    for window, L, B, ctx, P in ((2048, 1, 1, 3000, 1), (2048, 1, 16, 2500, 1), (2048, 1, 32, 2500, 1),
                                 (0, 1, 1, 3000, 1), (0, 1, 16, 1500, 1), (0, 1, 33, 1000, 1),
                                 (0, 8, 1, 3000, 1), (0, 8, 16, 1500, 1), (0, 8, 32, 800, 1),
                                 (2048, 1, 4, 3000, 64), (0, 8, 4, 3000, 64)):
        cfg = dict(batch=B, q_len=L, num_q_heads=16, num_kv_heads=2, head_dim=64, page_size=P, window=window,
                   rope_theta=500000.0 if window else 0.0, qk_scale_factor=5.4730064863875, sm_scale=0.125)
        if not att__supported(cfg):
            continue
        seq = torch.full((B,), ctx, dtype=torch.int32, device=dev)
        per = (ctx + L + P - 1) // P
        pt = torch.arange(B * per, dtype=torch.int32, device=dev).view(B, per)
        pos = (seq[:, None].long() + torch.arange(L, device=dev)).view(B, L)
        loc = (torch.gather(pt.long(), 1, pos // P) * P + pos % P).reshape(-1)
        kc = torch.zeros((B * per, P, 2, 64), dtype=torch.float8_e4m3fn, device=dev)
        vc = torch.zeros_like(kc)
        qkv = (torch.randn(B * L, nq, generator=g, device=dev) * 0.1).to(torch.bfloat16)
        one = torch.ones((), dtype=torch.float32, device=dev)
        _decode_attention7({"config": cfg, "seq_lens": seq, "cache_loc": loc, "k_cache": kc, "v_cache": vc,
                            "page_table": pt, "k_scale": one, "v_scale": one}, st, qkv)

def _att_fits7(data, cfg):
    """The attention entry's launcher host-checks these sizes (32-bit index math, OKC_OFF32) and raises otherwise;
    mirrored here so such shapes take SGLang's FA3 instead (fuzz: batch 64 x 150k context exceeds the caches' 4 GB)."""
    kc, vc = data["k_cache"], data["v_cache"]
    rows = kc.numel() // max(1, cfg["num_kv_heads"] * cfg["head_dim"])
    return (kc.numel() < (1 << 32) and vc.numel() < (1 << 32) and rows < (1 << 31)
            and data["page_table"].numel() < (1 << 31)
            and cfg["batch"] * cfg["q_len"] * 2304 < (1 << 31)
            and cfg["batch"] * (cfg["num_kv_heads"] // 2) <= 4096)


def _kernel_fn_fa3(data, state):
    st2 = dict(state)
    st2["att_ok"] = False
    return kernel_fn(data, st2)


def kernel_fn(data, state):
    cfg = data["config"]
    B, L, T = cfg["batch"], cfg["q_len"], cfg["tokens"]
    # q_len 1 only: on the kv_scale = 1/64 test their q_len 8 path's query scaling rounds coarser than FA3
    # (conformance test 9: hidden/stream 1.41x SGLang's error), so FA3 keeps every q_len 8 shape.
    # q_len 8: their kernel on full layers, and on sliding only for a single request, where fusing QK-norm, RoPE,
    # the cache write and the norm-gate into one launch beats FA3 + my two glue kernels by 5.6% in-layer;
    # at B >= 16 sliding FA3 still matches or wins (16x8 -1.2%, 32x8 +0.8%), below my 4% bar for q_len 8.
    agent_attn = (state["att_ok"] and att__supported(cfg) and _att_fits7(data, cfg)
                 )
    if agent_attn:
        # fused decode attention (QK-norm, RoPE, cache write, split-KV, merge, rmsnorm * sigmoid(gate))
        qpick = state.get("mm_pick", {}).get((T, "qkv"), True)
        if qpick == "bfp" and T <= BFP_MAX_T and state.get("bfp") is not None:
            qkv = _bfp_gemm(state, "qkv", data["attn_input"], T, state["qkv"].shape[0], HIDDEN)
        elif qpick is True or qpick == "bfp":
            qkv = torch.mm(data["attn_input"], state["qkv"].t())
        else:
            Pq, Sq = dense(data["attn_input"], state["qkv"], T)
            qkv = Pq[0] if Sq == 1 else Pq.sum(0)
            qkv = qkv.to(torch.bfloat16)
        try:
            o_in = _decode_attention7(data, state, qkv)
        except RuntimeError:
            # their launcher's host-side size checks run before any launch: nothing half-captured, take FA3
            return _kernel_fn_fa3(data, state)
        return _after_attention(data, state, o_in, None, T)
    dev = data["attn_input"].device
    x = data["attn_input"]
    sliding = cfg["window"] > 0
    use_fa3 = state["fa3"] is not None
    # 1. qkv projection, QK-norm, RoPE, FP8 cache write
    P, S = dense(x, state["qkv"], T)
    q = torch.empty((T, 1024), device=dev, dtype=torch.bfloat16)
    q8 = torch.empty((T, 1024), device=dev, dtype=torch.uint8)
    gate = torch.empty((T, 1024), device=dev, dtype=torch.float32)
    kc = data["k_cache"].view(torch.uint8)
    vc = data["v_cache"].view(torch.uint8)
    _qk_prep[(T,)](P, S, T, data["positions"], data["cache_loc"], state["invf"], kc, vc,
                   data["k_scale"], data["v_scale"], q, q8, gate, cfg["qk_scale_factor"],
                   ROPE=bool(cfg["rope_theta"]), num_warps=4, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    # 2. attention
    if use_fa3:
        descale = {}
        if cfg["kv_scale"] != 1.0:
            descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(B, 2),
                           v_descale=data["v_scale"].reshape(1, 1).expand(B, 2))
        window = cfg["window"]
        out = state["fa3"](q=q8.view(torch.float8_e4m3fn).view(T, 16, 64), k_cache=data["k_cache"],
                           v_cache=data["v_cache"], page_table=data["page_table"],
                           cache_seqlens=data["cache_seqlens"], cu_seqlens_q=data["cu_seqlens_q"],
                           max_seqlen_q=L, softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
                           window_size=(window - 1, 0) if window else (-1, -1),
                           scheduler_metadata=data["scheduler_metadata"], **descale)
        if isinstance(out, tuple):
            out = out[0]
        o_in = out.reshape(T, 1024)
    else:
        BM, NS = _attn_cfg(B, L, sliding)
        acc = torch.empty((B * 2 * NS * BM, 64), device=dev, dtype=torch.float32)
        ml = torch.empty((B * 2 * NS * BM, 2), device=dev, dtype=torch.float32)
        pt = data["page_table"]
        _attn_split[(B, NS)](q, kc, vc, pt, data["seq_lens"], data["k_scale"], acc, ml,
                             pt.stride(0), cfg["page_size"], cfg["window"], cfg["sm_scale"],
                             L=L, NSPLIT=NS, BM=BM, BN=64, SLIDING=sliding, num_warps=4, num_stages=3)
        o_in = torch.empty((T, 1024), device=dev, dtype=torch.bfloat16)
        _attn_combine[(T, 16)](acc, ml, gate, data["v_scale"], o_in, L=L, NSPLIT=NS, BM=BM,
                               SB=min(64, triton.next_power_of_2(NS)), NORMGATE=False, num_warps=4)
    return _after_attention(data, state, o_in, gate, T)


def _after_attention(data, state, o_in, gate, T):
    """o projection onwards: residual boundary, router, experts, unpooler, final boundary.
    gate is None when o_in already carries rmsnorm(o) * sigmoid(gate). state["gluev2"] picks between the
    one-program-per-token boundary kernels and the parallel (atomic-sum) ones; the tuner chooses per T."""
    dev = o_in.device
    f32 = dict(device=dev, dtype=torch.float32)
    gv = state.get("gluev2", {})
    v2 = gv.get(T, gv.get(_bucket(T), False)) if FORCE_GLUEV2 is None else FORCE_GLUEV2
    s1 = torch.empty((T, HIDDEN), **f32)
    h = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    if v2:
        # o projection with no K split (each program owns whole rows) + the boundary's four per-token sums
        a_out = torch.empty((T, HIDDEN), **f32)
        sums = torch.zeros((T, 4), **f32)
        BN, BT, BK = 64, (16 if T <= 16 else 32), 128
        _gemm_o[(HIDDEN // BN, triton.cdiv(T, BT))](
            o_in, gate, state["o"], a_out, sums, data["stream"], state["gate1"], state["gate1"], T, HIDDEN,
            K=1024, BN=BN, BT=BT, BK=BK, NORMGATE=gate is not None, num_warps=4, num_stages=3)
        BC1 = 2048
        _bnd1_apply[(T, HIDDEN // BC1)](a_out, sums, data["stream"], state["gate1"], state["gate1"], state["g_preffn"],
                                        s1, h, N=HIDDEN, BC=BC1, num_warps=8)
    elif (gate is None and _bfp_tail_ok(state, T) and state.get("mm_pick", {}).get((T, "o")) == "bfp"
          and data["stream"].dtype == torch.float32 and data["stream"].is_contiguous()):
        # o projection whose last CTA also computes the post-attention boundary (s1, h): no _boundary1 launch
        _bfp_pdl(T)
        globals()["bfp_run_sk2_tail"](o_in, "o", T, HIDDEN, 1024, state["bfp"],
                         (state["ab1"][0], state["ab1"][1], data["stream"], None, state["g_preffn"], s1, h))
    else:
        if gate is None:
            P, S = dense(o_in, state["o"], T, state, "o")
        else:
            P, S = dense_ng(o_in, gate, state["o"], T, state)
        _boundary1[(T,)](P, S, T, data["stream"], state["gate1"], state["gate1"], state["g_preffn"], s1, h,
                         num_warps=16, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    # 4. pooler + router (BF16 hi + lo rows, FP32 accumulation), routing, pre-expert norm
    ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    lids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    wts = torch.empty((T, 8), **f32)
    moe = state["moe_pick"].get(T, state["moe_pick"].get(_bucket(T), "agent"))
    if moe == "marlin" and state["bank"] is None:
        moe = "agent"
    if moe != "marlin" and not state["ma_ok"]:
        moe = "marlin"
    if moe == "marlin" and state["bank"] is None:
        moe = "agent"
    use_marlin = moe == "marlin"
    use_ma = moe == "agent"
    if not (use_marlin or use_ma):
        raise RuntimeError("no expert backend: neither the moe agent's kernel nor SGLang's Marlin is available")
    xm = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
    mmp = state.get("mm_pick", {})
    ppick = mmp.get((T, "pr"), mmp.get((_bucket(T), "pr"), False))
    if ppick == "bfpf" and T <= 32 and state.get("bfp") is not None:
        # bf16_proj's fused pooler + router: one launch, both read h once (bf16 [T, 4096], fp32 [T, 256]); with
        # FUSE_ROUTE its last CTA also does the routing and the pre-expert norm (no _route_split launch)
        _bfp_pdl(T)
        route = _fused_route(state, data, ids, lids, wts, xm) if (use_marlin or use_ma) else None
        if route is None:   # (the standalone proj module's launch takes no route argument)
            P, S = bfp_fused_pooler_router(h, state["bfp"]), -1
        else:
            P, S = bfp_fused_pooler_router(h, state["bfp"], route), -2
    elif ppick == "bfp" and state.get("bfp") is not None:
        pooled = _bfp_gemm(state, "pool", h, T, D_EXPERT, HIDDEN)          # bf16 [T, 4096]
        logits = _bfp_gemm(state, "router", h, T, N_EXPERTS, HIDDEN)       # fp32 [T, 256], hi+lo already summed
        P, S = (pooled, logits), -1
    else:
        P, S = dense(h, state["pr"], T, state, "pr")
    if S == -2:  # the routing and xm were produced inside the fused pooler + router launch
        pass
    elif S == -1:  # pooler and router came from the bf16_proj kernels as two separate tensors
        pooled, logits = P
        _route_split[(T,)](pooled, logits, state["bias"], state["g_preexp"], ids, wts, xm,
                           data["local_start"], lids, PERMUTED=not (use_marlin or use_ma), num_warps=8, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    elif v2:
        pool = torch.empty((T, D_EXPERT), **f32)
        psum = torch.zeros((T,), **f32)
        BCR = 1024
        _route_a[(T, D_EXPERT // BCR)](P, S, T, state["bias"], ids, wts, data["local_start"], lids, pool, psum,
                                       BC=BCR, num_warps=8)
        _route_b[(T, D_EXPERT // BCR)](pool, psum, state["g_preexp"], xm, BC=BCR,
                                       PERMUTED=not (use_marlin or use_ma), num_warps=8)
    else:
        _route[(T,)](P, S, T, state["bias"], state["g_preexp"], ids, wts, xm, data["local_start"], lids,
                     PERMUTED=not (use_marlin or use_ma), num_warps=8, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    # 5. experts
    if use_ma:
        m = _ma_run(state, xm, lids, wts, T)   # lids: local expert index, -1 = another GPU's expert
    elif use_marlin:
        m = _marlin_experts(xm, wts, lids, state)
    # 6. unpooler + post-FFN boundary + next layer's norm
    wu = state["unpool_slice"] if T <= 8 else state["unpool"]
    nu = wu.shape[0]
    s2 = torch.empty((T, HIDDEN), **f32)
    out = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    if not v2 and _bfp_tail_ok(state, T) and state.get("mm_pick", {}).get((T, "un")) == "bfp":
        # unpooler slice whose last CTA also computes the post-FFN boundary (s2, out): no _boundary2 launch
        _bfp_pdl(T)
        globals()["bfp_run_sk2_tail"](m, "unpooler", T, nu, D_EXPERT, state["bfp"],
                         (state["ab2"][0], state["ab2"][1], s1, state["g_postffn"], state["g_next"], s2, out))
        return out, s2, ids
    P, S = dense(m, wu, T, state, "un")
    if v2:
        fbuf = torch.zeros((T, HIDDEN), **f32)
        sums2 = torch.zeros((T, 4), **f32)
        BC2 = 1024
        _bnd2_sums[(T, HIDDEN // BC2)](P, S, T, nu, s1, state["gate2"], state["gate2"], state["g_postffn"], fbuf, sums2,
                                       BC=BC2, num_warps=8)
        _bnd2_apply[(T, HIDDEN // BC2)](fbuf, sums2, s1, state["gate2"], state["gate2"], state["g_next"], s2, out,
                                        N=HIDDEN, BC=BC2, num_warps=8)
    else:
        _boundary2[(T,)](P, S, T, nu, s1, state["gate2"], state["gate2"], state["g_postffn"], state["g_next"],
                         s2, out, num_warps=16, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    return out, s2, ids
