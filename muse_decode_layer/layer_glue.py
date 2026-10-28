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
import os
import time

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
PDL_MAX_T = 128  # measured: a launch-latency win at decode sizes (d1 +2%); with the CUDA glue the T = 128 shapes gain
                 # 1.2-2.3 us (H200 paired, seeds 0-2, vm_ab/ab10b) while T = 256 is a wash (+0.6/-0.6: the cuBLAS
                 # nodes stay plain launches there anyway) - off at 256 as before


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
            # H200 paired replays (vm_ab/ab5, both seeds): o sk2 beat cuBLAS bf16 out by 2.0-2.6 us at T = 16 (the
            # isolated tuner flipped to mm16 one prepare in three) and by 2-3.4 at T = 32, cuBLAS fp32 out by 1.1-2.1 at
            # T = 1 and 1.9-3.3 at T = 8; the unpooler slice's cuBLAS lost 2.7-4.1 at T = 1/8 (its Triton split-K ties at
            # T = 1, loses 1.3-1.9 at T = 8); qkv cuBLAS lost 3-5 us and bf16 out 8-13 us at every T <= 32
            (16, "o"): "bfp", (1, "qkv"): "bfp", (8, "qkv"): "bfp", (16, "qkv"): "bfp", (32, "qkv"): "bfp",
            # T = 128 / 256 (H200, vm_ab/ab4: 20 variants x 4 shapes x 2 seeds, 25 replays each; the isolated tuner
            # flipped these between prepares): qkv cuBLAS fp32 out beat its bf16-out / K-major / proj-module forms by
            # 1.2-3.5 us per shape; un on the K-major copy by 0-2 us; o and pr are ties within +-1 us (one fixed pick
            # each so a prepare cannot flip them); the Triton split-K and the proj module's pr route lose 2-14 us
            (128, "qkv"): True, (256, "qkv"): True, (128, "o"): "mm16", (256, "o"): "torchT",
            (128, "pr"): True, (256, "pr"): True, (128, "un"): "torchT", (256, "un"): "torchT"}
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


M1_ROUTE_PIN = {"qkv_gate": "sk2", "o": "sk2", "unpooler": "sk2"}
# the fork's m = 1 route per layer, pinned: its isolated autotuner flips between sk2 and the GEMVs (cgemv / sgemv)
# from one prepare to the next; on the chain the qkv / o GEMVs cost +6.5 us at T = 1 (H200 paired, seeds 0-2,
# vm_ab/ab16: sk2 streams its weights under the previous kernel's tail through PDL, the plain-launched GEMV cannot)
# and the unpooler slice as sk2 (slice8_r192) is 0.9-1.4 us faster than the GEMVs (ab18: rows -1.6/-1.3/-0.8 d1s,
# -0.7/-1.6/-0.6 d1f; the GEMVs as PDL launches recover at most 1.0 of it)


def _pin_m1_routes(bst):
    """Apply M1_ROUTE_PIN to the fork's autotuned choices where the pinned kind is built and validated."""
    pinned = []
    for layer, kind in M1_ROUTE_PIN.items():
        ok = False
        if kind == "sk2":
            ok = (bfp__sk2_name(layer, 1) in bst.get("sk2", {})
                  and bfp__route_key(layer, 1, "sk2") not in bst.get("disabled", ()))
        elif kind == "cgemv":
            ok = (layer, bfp_CG_CONFIGS[layer][0]) in bst.get("cg", {}) and bfp__route_key(layer, 1, "cgemv") not in bst.get("disabled", ())
        elif kind == "sgemv":
            ok = layer == "unpooler" and layer in bst.get("raw", {}) and bfp__route_key(layer, 1, "sgemv") not in bst.get("disabled", ())
        if ok:
            bst.setdefault("choice", {})[(layer, 1)] = (kind,)
            pinned.append(f"{layer}:{kind}")
    print("m=1 routes pinned: " + (" ".join(pinned) or "none") + "; autotuned: "
          + " ".join(f"{l_}:{bst.get('choice', {}).get((l_, 1), ('?',))[0]}" for l_ in ("qkv_gate", "o", "unpooler")), flush=True)


def _bfp_prepare(weights, st):
    """Their prepare() on the five dense matrices, guarded: any failure just disables the backend."""
    st["bfp"] = None
    try:
        w5 = {k: weights[k].contiguous() for k in ("qkv_gate", "o", "pooler", "unpooler", "router")}
        st["bfp"] = bfp_prepare(w5)
        try:
            _pin_m1_routes(st["bfp"])
        except Exception as ex:  # noqa: BLE001
            print("m=1 route pin skipped:", repr(ex)[:200], flush=True)
    except Exception as ex:  # noqa: BLE001
        s = repr(ex)
        print("bfp disabled:", s[:200] + (" ... " + s[-1500:] if len(s) > 1700 else s[200:]), flush=True)


_BFP_LAYER = {"qkv": "qkv_gate", "o": "o", "un": "unpooler", "pool": "pooler", "router": "router"}
_T_COPY = {"qkv": "qkvT", "o": "oT", "pr": "prT", "un": "unpoolT"}   # dense name -> state key of the K-major weight copy


BFP_PDL_N = 2   # bf16_proj's sk2 / fused kernels: weight stages issued before griddepcontrol.wait (0 = off)


PDL_N_SWEEP = (2, 3, 4, 6)   # prepare(): per T, the sk2 pre-wait weight-stage count timed on the whole post-attention chain
_PDL_N_BY_T = {}             # T -> the winner (the ring has 4-6 stages at tokr 8; 6 = the whole ring streams before the wait)
# Chain-timed picks decided with statistics on an H200 (vm_ab/ab1 + ab5: 25 paired graph replays x 2 seeds per shape,
# 8e96589 build, Oct 28) instead of the tuner's single-sample coin flips; a T in a lock skips its chain timing in _tune.
PDL_LOCK = {1: 2, 2: 2, 4: 2, 8: 2, 16: 2, 32: 2}   # 3/4/6 lost 1.5-3.5 us at T = 16/32, 0.4-3 us at T = 8; no effect at T = 1
ROUTE_MODE_LOCK = {1: "split", 2: "split", 4: "split", 8: "split", 16: "split", 32: "split"}   # sk2f / moe / moe_nb lost 3-10 us everywhere
PR_FORM_LOCK = {1: ("bfpf", False), 2: ("bfpf", False), 4: ("bfpf", False), 8: ("bfpf", False), 16: ("bfpf", False),
                32: ("bfp", False)}   # T -> (form, bnd1 prologue): the fused launch + _route_split won by 1-2 us at T = 1,
                                      # 3-4 us at T = 8, 0.5-1.5 us at T = 16; the separate GEMMs by 2-3 us at T = 32; the
                                      # boundary-1 prologue cost 0.8-1.3 us at T = 1 (ab1 / ab5, both seeds)


SK2_PDL_FLAGS = 0   # OR-ed into the proj fork's PDL word: 0x100 = the sk2 kernels trigger launch_dependents at entry
                    # (not after their main loop), 0x200 = the m = 1 cgemv is a PDL launch (weights before its wait,
                    # trigger at entry). Measured on the H200 (vm_ab/ab14).


def _bfp_pdl(T):
    """Their launchers read SK2_PDL (bfp_SK2_PDL once inlined) at every launch: same T rule as this file's glue."""
    global bfp_SK2_PDL
    bfp_SK2_PDL = (_PDL_N_BY_T.get(T, BFP_PDL_N) | SK2_PDL_FLAGS) if _pdl(T) else 0


def _bfp_gemm(st, name, x, m, n, k):
    _bfp_pdl(m)
    return bfp_kernel_fn({"x": x, "config": {"layer": _BFP_LAYER[name], "m": m, "n": n, "k": k}}, st["bfp"])


FUSE_ROUTE = False  # [off: lost 3-7 us at every T on the H200, vm_ab/ab1; ROUTE_MODE_LOCK = split] the fused pooler + router launch (sk2f) also does the pre-expert norm after a grid barrier (every CTA its
                    # own columns) and the routing of token t by CTA t (parallel, no serial tail; the d467a5c5 "last CTA does
                    # everything" tail was latency-bound: 29/35/43/64 us at T 1/8/16/32). Per T the tuner picks among
                    # _route_split / routing in the MoE kernel / this, on the chain time. The fused launch's last
                    # CTA, replacing the _route_split launch (T <= 32, agent/Marlin experts: plain xm layout)


def _fused_route(state, data, ids, lids, wts, xm, selftest=False):
    """The route tuple for bfp_fused_pooler_router, or None when the separate _route_split kernel must run (also
    when the composed proj module's fused launch has no routing tail: the standalone muse_bf16_proj, or when the
    prepare-time self-test of the tail failed)."""
    ls = data["local_start"]
    if (not FUSE_ROUTE or ls.dtype != torch.int32 or state["g_preexp"].dtype != torch.bfloat16
            or "route" not in bfp_fused_pooler_router.__code__.co_varnames
            or not (selftest or (state.get("fuse_route_ok") and state.get("route_mode", {}).get(T_of(ids)) == "sk2f"))):
        return None
    return (state["bias"], state["g_preexp"], ids, lids, wts, xm, ls)


def T_of(t):
    return int(t.shape[0])


FUSE_BOUNDARY = os.environ.get("OKC_FUSE_BOUNDARY", "0") == "1"   # OFF (vm_ab/ab20, tail v3 with the v4 math: still a
                                                                   # one-CTA memory pipeline of ~130 KB per token that cannot
                                                                   # start before the producer's last row - isolated o launch
                                                                   # 8.0 -> 18.8 us, unpooler slice 6.9 -> 12.9 at T=1; chain
                                                                   # +2.6..+11.8 us at T=1, +26..+76 at T=8; the separate v4
                                                                   # glue kernel hides 100 KB of those loads under PDL)
                        # OFF: the v2 tails (every load in flight, one reduction, alpha/beta from the gate in-kernel) are
                        # exact but compute-bound on one CTA: o launch 8.8 -> 46.0 us, unpooler slice 7.2 -> 21.5
                        # (a15e0698, a3816fbb; the ticket flavor made no difference). T = 1: the o-projection / unpooler-slice sk2 launches' last CTA computes _boundary1 / _boundary2
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
            or st["gate1"].dtype != torch.bfloat16):
        print("boundary tails: not available", flush=True)
        return
    dev = st["qkv"].device
    g = torch.Generator(device=dev).manual_seed(778)
    f32 = dict(device=dev, dtype=torch.float32)
    report = []
    for T in (1, 8):   # the v3 tails loop over the rows of one CTA
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
        # the tails' own cost, isolated (graph replays, cold L2): each launch with and without its tail
        flush_ = torch.zeros(int(60e6), dtype=torch.float32, device=dev)
        global _FLUSH_SINK
        if _FLUSH_SINK is None:
            _FLUSH_SINK = torch.zeros((), dtype=torch.float32, device=dev)
        t_o = _graph_time(lambda: _bfp_gemm(st, "o", o_in, T, HIDDEN, 1024), flush_, reps=9)
        t_ot = _graph_time(lambda: globals()["bfp_run_sk2_tail"](o_in, "o", T, HIDDEN, 1024, st["bfp"],
                                                                 (st["ab1"][0], st["ab1"][1], stream, None, st["g_preffn"], s1a, ha)), flush_, reps=9)
        t_u = _graph_time(lambda: _bfp_gemm(st, "un", m, T, 1024, D_EXPERT), flush_, reps=9)
        t_ut = _graph_time(lambda: globals()["bfp_run_sk2_tail"](m, "unpooler", T, 1024, D_EXPERT, st["bfp"],
                                                                 (st["ab2"][0], st["ab2"][1], s1a, st["g_postffn"], st["g_next"], s2a, outa)), flush_, reps=9)
        del flush_
        print(f"boundary tails T{T}: o launch {t_o:.1f} -> with the tail {t_ot:.1f} us; unpooler slice {t_u:.1f} -> {t_ut:.1f} us", flush=True)
    print("boundary tails ok: " + " ".join(f"T{T} " + "/".join(f"{e:.1e}" for e in errs) for T, errs in report), flush=True)
    st["fuse_bnd_ok"] = True


FUSE_BND1 = False  # [off: the prologue cost 0.8-1.3 us at T = 1 on the H200, vm_ab/ab1] the fused pooler + router launch computes h (its own x) and s1 in its PROLOGUE from the o-projection
                   # output (every CTA redundantly, before its x boxes; the weight ring is already streaming): no
                   # _boundary1 launch at T <= BND1_MAX_T. Enabled only when the prepare-time self-test matches.
PRFORM_MAX_T = 32  # the pooler + router form (separate / fused [/ fused + prologue]) is chosen on the chain per T up to here
BND1_MAX_T = 8     # cooperative prologue: CTA c owns its 64 columns of every row (<= 3 KB per CTA), two grid tickets
                   # at the kernel start, the weight ring streams meanwhile (the per-CTA whole-row versions were
                   # load-issue bound: +10..13 us isolated)


def _bnd1_fuse_ok(state, T):
    return (FUSE_BND1 and T <= BND1_MAX_T and T in state.get("fuse_bnd1_ok_T", ()) and state.get("bfp") is not None
            and "bnd1" in bfp_fused_pooler_router.__code__.co_varnames and "ab1" in state
            and state["g_preffn"].dtype == torch.bfloat16 and state.get("bnd1_on", {}).get(T, True))


def _bnd1_selftest(st):
    """The fused launch's boundary-1 prologue against _boundary1 (+ the plain fused launch on _boundary1's h) at
    T = 1 and 8 on synthetic inputs; any mismatch or exception disables it (st["fuse_bnd1_ok"]); printed."""
    st["fuse_bnd1_ok"] = False
    if (not FUSE_BND1 or st.get("bfp") is None or "bnd1" not in bfp_fused_pooler_router.__code__.co_varnames
            or "ab1" not in st):
        print("boundary-1 prologue: not available", flush=True)
        return
    dev = st["qkv"].device
    g = torch.Generator(device=dev).manual_seed(779)
    f32 = dict(device=dev, dtype=torch.float32)
    report = []
    for T in (1, 8):   # the cooperative prologue handles M <= 8 rows
        a = (torch.randn(T, HIDDEN, generator=g, device=dev) * 0.1).to(torch.bfloat16)
        s = torch.randn(T, HIDDEN, generator=g, device=dev)
        h = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
        s1 = torch.empty((T, HIDDEN), **f32)
        try:
            _bfp_pdl(T)
            pooled, logits = bfp_fused_pooler_router(h, st["bfp"], None, (a, s, st["ab1"][0], st["ab1"][1], st["g_preffn"], s1))
            torch.cuda.synchronize()
        except Exception as ex:  # noqa: BLE001
            print(f"boundary-1 prologue T{T}: FAILED {repr(ex)[:300]}", flush=True)
            return
        s1r = torch.empty((T, HIDDEN), **f32)
        hr = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
        _boundary1[(T,)](a.view(1, T, HIDDEN), 1, T, s, st["gate1"], st["gate1"], st["g_preffn"], s1r, hr, num_warps=16)
        pooled_r, logits_r = bfp_fused_pooler_router(hr, st["bfp"])
        torch.cuda.synchronize()
        # h itself lives only in the launch's shared memory (the global x buffer is not written): judged through the
        # pooler / router outputs computed from it, against the plain launch on _boundary1's h
        errs = [float((x.float() - y.float()).norm() / max(1e-6, float(y.float().norm())))
                for x, y in ((s1, s1r), (pooled, pooled_r), (logits, logits_r))]
        report.append((T, errs))
        if not all(e < 2e-3 for e in errs):
            print(f"boundary-1 prologue T{T}: MISMATCH rel err s1 {errs[0]:.2e} pooled {errs[1]:.2e} "
                  f"logits {errs[2]:.2e}", flush=True)
            return
        # the prologue's own cost, isolated (graph replays, cold L2): it replaces a ~4.4 us launch, so it is enabled
        # only when it adds less than BND1_MAX_COST_US to the fused launch (check 0f57b859: +12 us in the chain)
        flush_ = torch.zeros(int(60e6), dtype=torch.float32, device=dev)
        global _FLUSH_SINK
        if _FLUSH_SINK is None:
            _FLUSH_SINK = torch.zeros((), dtype=torch.float32, device=dev)
        t_plain = _graph_time(lambda: bfp_fused_pooler_router(hr, st["bfp"]), flush_, reps=9)
        t_bnd1 = _graph_time(lambda: bfp_fused_pooler_router(h, st["bfp"], None, (a, s, st["ab1"][0], st["ab1"][1], st["g_preffn"], s1)),
                             flush_, reps=9)
        del flush_
        print(f"boundary-1 prologue T{T}: fused launch {t_plain:.1f} us -> with the prologue {t_bnd1:.1f} us", flush=True)
        if not (t_bnd1 - t_plain < BND1_MAX_COST_US):   # (the chain timing in _tune makes the final on/off choice)
            print(f"boundary-1 prologue T{T}: too costly (+{t_bnd1 - t_plain:.1f} us), off at T{T}", flush=True)
            continue   # per T: a costly row count does not switch the other off
        st.setdefault("fuse_bnd1_ok_T", set()).add(T)
    print("boundary-1 prologue ok: " + " ".join(f"T{T} " + "/".join(f"{e:.1e}" for e in errs) for T, errs in report)
          + f"; enabled at T {sorted(st.get('fuse_bnd1_ok_T', ()))}", flush=True)
    st["fuse_bnd1_ok"] = bool(st.get("fuse_bnd1_ok_T"))


BND1_MAX_COST_US = 6.0   # isolated gate (a guard against a broken prologue); the chain timing per T decides below it


def _norm_selftest(st):
    """Routing in the MoE kernel + the fused launch's grid-barrier pre-expert norm, against _route_split + the MoE's
    plain entry on synthetic activations (T = 1, 8, 16, 32): xm must match within rounding, the chosen ids exactly,
    the expert output within rounding. Any mismatch or exception disables the path (st["route_moe_ok"]); printed."""
    st["route_moe_ok"] = False
    if (not ROUTE_IN_MOE or st.get("bfp") is None or "moe_kernel_fn_logits" not in globals()
            or "norm" not in bfp_fused_pooler_router.__code__.co_varnames or st.get("ma") is None
            or "wts" not in st["ma"]):
        print("routing in the MoE kernel: not available", flush=True)
        return
    dev = st["qkv"].device
    g = torch.Generator(device=dev).manual_seed(780)
    ls = torch.zeros(1, dtype=torch.int32, device=dev)
    report = []
    for T in (1, 8, 16, 32):
        h = (torch.randn(T, HIDDEN, generator=g, device=dev) * 0.1).to(torch.bfloat16)
        xm = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
        ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
        try:
            _bfp_pdl(T)
            pooled, logits = bfp_fused_pooler_router(h, st["bfp"], None, None, (st["g_preexp"], xm))
            # a synthetic skew so that some routes land on the 32 local experts (as the judge's inputs do)
            logits = logits + 0.25 * torch.randn(T, N_EXPERTS, generator=g, device=dev)
            out = _ma_run_logits(st, xm, logits, st["bias"], ls, ids, T)
            torch.cuda.synchronize()
        except Exception as ex:  # noqa: BLE001
            print(f"routing in the MoE kernel T{T}: FAILED {repr(ex)[:300]}", flush=True)
            return
        ids2 = torch.empty_like(ids)
        lids2 = torch.empty_like(ids)
        wts2 = torch.empty((T, 8), device=dev, dtype=torch.float32)
        xm2 = torch.empty_like(xm)
        _route_split[(T,)](pooled, logits, st["bias"], st["g_preexp"], ids2, wts2, xm2, ls, lids2, PERMUTED=False, num_warps=8)
        out2 = _ma_run(st, xm2, lids2, wts2, T)
        torch.cuda.synchronize()
        d_ids = int((ids != ids2).sum())
        d_x = float((xm.float() - xm2.float()).norm() / max(1e-6, float(xm2.float().norm())))
        d_o = float((out.float() - out2.float()).norm() / max(1e-6, float(out2.float().norm())))
        report.append((T, d_ids, d_x, d_o))
        if d_ids or not d_x < 2e-3 or not d_o < 2e-2:
            print(f"routing in the MoE kernel T{T}: MISMATCH ids {d_ids} xm {d_x:.2e} out {d_o:.2e}", flush=True)
            return
    print("routing in the MoE kernel ok: " + " ".join(f"T{T} ids0 x{x:.1e} o{o:.1e}" for T, _, x, o in report), flush=True)
    st["route_moe_ok"] = True
    # mode 3 (no barrier): the MoE kernel also forms the per-token RMS scale from the fused launch's partial sums; the
    # experts' input is rounded to bf16 before that scale, so the output differs from the _route_split path at the
    # bf16-rounding level (ids stay exact)
    st["route_moe_nb_ok"] = False
    if not (_moe_partials_ok() and "gain" in bfp_fused_pooler_router.__code__.co_varnames):
        print("routing in the MoE kernel (no barrier): not available", flush=True)
        return
    report = []
    for T in (1, 8, 16, 32):
        h = (torch.randn(T, HIDDEN, generator=g, device=dev) * 0.1).to(torch.bfloat16)
        ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
        try:
            _bfp_pdl(T)
            pooled, logits = bfp_fused_pooler_router(h, st["bfp"], None, None, None, st["g_preexp"])
            logits = logits + 0.25 * torch.randn(T, N_EXPERTS, generator=g, device=dev)
            out = _ma_run_logits(st, pooled, logits, st["bias"], ls, ids, T,
                                 partials=st["bfp"]["fused_part2"][:T * 128].view(T, 128))
            torch.cuda.synchronize()
        except Exception as ex:  # noqa: BLE001
            print(f"routing in the MoE kernel (no barrier) T{T}: FAILED {repr(ex)[:300]}", flush=True)
            return
        pooled2, logits2 = bfp_fused_pooler_router(h, st["bfp"])
        logits2 = logits   # the same perturbed logits
        ids2 = torch.empty_like(ids)
        lids2 = torch.empty_like(ids)
        wts2 = torch.empty((T, 8), device=dev, dtype=torch.float32)
        xm2 = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
        _route_split[(T,)](pooled2, logits2, st["bias"], st["g_preexp"], ids2, wts2, xm2, ls, lids2, PERMUTED=False, num_warps=8)
        out2 = _ma_run(st, xm2, lids2, wts2, T)
        torch.cuda.synchronize()
        d_ids = int((ids != ids2).sum())
        d_o = float((out.float() - out2.float()).norm() / max(1e-6, float(out2.float().norm())))
        report.append((T, d_ids, d_o))
        if d_ids or not d_o < 2e-2:
            print(f"routing in the MoE kernel (no barrier) T{T}: MISMATCH ids {d_ids} out {d_o:.2e}", flush=True)
            return
    print("routing in the MoE kernel (no barrier) ok: " + " ".join(f"T{T} ids0 o{o:.1e}" for T, _, o in report), flush=True)
    st["route_moe_nb_ok"] = True


def _gate_coeffs(g):
    """alpha, beta of a residual gate from its BF16 weight, exactly as _gate_ab forms them (prepare-time, fp32)."""
    w = g.float() / 0.3
    beta = torch.sigmoid(w)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=1e-3))
    return alpha.contiguous(), beta.contiguous()


BTAIL_MAX_T = 8   # the v3 tails loop over the rows (one CTA): T <= 8
BTAIL_LOCK = {}   # T -> layers ("o" = boundary 1 in the o-projection sk2, "unpooler" = boundary 2 in the unpooler slice
                  # sk2) whose boundary runs in the producer's last CTA; decided on the H200 (vm_ab/ab20)


def _bfp_tail_ok(state, T, layer):
    """The boundary tail of this launch (o -> boundary 1, unpooler slice -> boundary 2) at this T: self-tested,
    and chosen on the chain in _tune ("btail")."""
    return (FUSE_BOUNDARY and T <= BTAIL_MAX_T and state.get("bfp") is not None and "bfp_run_sk2_tail" in globals()
            and state.get("fuse_bnd_ok") and state.get("btail_on", {}).get((T, layer), False)
            and state["gate1"].dtype == torch.bfloat16 and state["gate2"].dtype == torch.bfloat16
            and state["g_preffn"].dtype == torch.bfloat16 and state["g_postffn"].dtype == torch.bfloat16
            and state["g_next"].dtype == torch.bfloat16)


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


if "moe_ROUTE_LOGITS_BUILDS" in globals():   # the MoE module's in-kernel routing from the router logits (its prepare()
    moe_ROUTE_LOGITS_BUILDS = True             # then also builds the OKC_ROUTE_LOGITS variants; see _ma_run_logits)
ROUTE_IN_MOE = False  # [off: moe / moe_nb lost 4-10 us at every T on the H200, vm_ab/ab1] offer "routing in the MoE kernel" (no _route_split launch; the fused pooler+router launch
                      # normalizes xm after a grid barrier); the tuner picks per T on the chain time, self-test gated


MOE_TINY_H0_BUILD = True   # also build the TINY (M <= 8) variant with OKC_WSH0 (helper warp = hardware warp 0) for the
                           # chain A/B (vm_ab/ab15; the standalone measured it neutral); selected by moe_TINY_DEFINES


def _moe_builds(sfx):
    """The builds the moe module's prepare() will ask for (its rule), so prepare() here can run them in parallel."""
    mods = [tuple(moe__main_defines(sfx)) + sfx]
    for d in (moe_TINY_DEFINES, moe_TINY2_DEFINES, moe_MID_DEFINES):
        if d and (sfx or "-DOKC_WGS=1" not in tuple(d)):   # the WGS build needs the PRMT scale decode
            mods.append(tuple(d) + sfx)
    if MOE_TINY_H0_BUILD and moe_TINY_DEFINES and "-DOKC_WSH0=1" not in tuple(moe_TINY_DEFINES):
        mods.append(tuple(moe_TINY_DEFINES) + ("-DOKC_WSH0=1",) + sfx)
    if globals().get("moe_ROUTE_LOGITS_BUILDS") and "moe__logits_defines" in globals():
        mods += [moe__logits_defines(d) for d in list(mods)]
    return mods


def _route_mode(state, T):
    """Where this call's routing runs: "split" (the _route_split kernel), "sk2f" (CTA t of the fused pooler+router
    launch after its grid barrier), "moe" (every MoE CTA, xm normalized by the fused launch after a grid barrier) or
    "moe_nb" (every MoE CTA, which also forms the RMS scale from the fused launch's partial sums: no barrier)."""
    mode = state.get("route_mode", {}).get(T, "split")
    if mode == "moe" and not (ROUTE_IN_MOE and state.get("route_moe_ok") and "moe_kernel_fn_logits" in globals()):
        return "split"
    if mode == "moe_nb" and not (ROUTE_IN_MOE and state.get("route_moe_nb_ok") and _moe_partials_ok()):
        return "split"
    return mode


def _route_in_moe_ok(state, T):
    return _route_mode(state, T) == "moe"


def _moe_partials_ok():
    """Does the MoE module's logits entry take pre_norm_partials (the per-token RMS scale formed in its GEMM1 epilogue)?"""
    return ("moe_kernel_fn_logits" in globals()
            and "pre_norm_partials" in (moe_kernel_fn_logits.__code__.co_consts or ()))


def _ma_run_logits(state, x, logits, bias, ls, ids, T, partials=None):
    """The MoE module's in-kernel routing entry: top-8 of sigmoid(logit * 0.1768) + bias per token (ties to the lower
    id), renormalized weights, local ids, computed by every CTA from the router logits; CTA 0 writes the chosen global
    ids into `ids`. With partials (fp32 [T, 128], token-major partial sums of pooled^2), x is pooled * (1 + w) in bf16
    and the kernel applies rs_t = rsqrt(mean + 1e-5) in its GEMM1 epilogue. Returns the expert output bf16 [T, 4096]."""
    pdl = 1 if _pdl(T) else 0
    for ext in moe__EXTS.values():
        ext.okc_set_pdl(pdl)
    data = {"hidden_states": x, "router_logits": logits, "router_bias": bias, "local_start": ls, "topk_out": ids,
            "config": {"local_count": LOCAL_EXPERTS}}
    if partials is not None:
        data["pre_norm_partials"] = partials
    out, _ = moe_kernel_fn_logits(data, state["ma"])
    return out


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


TUNED_T = (1, 8, 16, 32, 128, 256)   # the scored token counts; T = 2 / 4 / 64 take the next row's picks (the 300 s
                                     # prepare budget: tuning them cost ~60 s of the VM's 307 s)
TRY_MARLIN = False   # SGLang's Marlin experts as a candidate when the moe agent's kernel is available: it lost by
                     # 30-50% at every T in every prepare (10 config + 2 chain timings per T); still the fallback


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
        # cuBLAS with BF16 output (SGLang's own bf16 GEMM call), a different kernel choice from the fp32-output call;
        # the consumers read P in any dtype (the bf16_proj paths at T <= 32 already hand them bf16). (At T = 128 the
        # judge profile's split-K kernel is the qkv GEMM: N = 2304 is 24 tiles of 96 x 64; o and unpooler share the
        # 128 x 64 kernel at ~10.6 / ~19 us, the same as the proj module's isolated timings.)
        return torch.nn.functional.linear(x, w).view(1, T, N), 1
    if pick == "torchT" and name in _T_COPY and _T_COPY[name] in state and not (name == "un" and T <= 8):
        # cuBLAS on a K-major copy of W (x [T, K] @ W^T stored [K, N]): a different kernel choice again (the proj
        # module measured it faster for the unpooler at m = 128); fp32 out only for the router logits ("pr")
        wT = state[_T_COPY[name]]
        if name == "pr":
            return torch.mm(x, wT, out_dtype=torch.float32).view(1, T, N), 1
        return torch.mm(x, wT).view(1, T, N), 1
    if pick == "bfq" and name in _BFP_LAYER and state.get("bfp") is not None:
        # the proj module's own autotuned route at this token count (gq / torch / ... at m >= 33), any T
        return _bfp_gemm(state, name, x, T, N, K).view(1, T, N), 1
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
# The three glue kernels as CUDA. ncu on the H200 (glue.ncu-rep, T = 1 and 8): each Triton program is ONE CTA of 16
# warps that runs ~790 instructions per warp with 4 warps per scheduler (No Eligible 61%, 10.6 cycles per issued
# instruction, 14 GB/s of memory traffic) = 6.2-6.6 us per boundary kernel and 5.6-5.8 us for the routing, the same
# per program at T = 8 (one program per token): a latency / issue chain, not bandwidth and not the S loop (every
# producer emits one reduced row, S = 1). A first CUDA version (8 CTAs per token + a ticket barrier, gate math in
# the kernel, 8 serial argmax rounds) measured no faster (glue2.ncu-rep: 5.6 / 7.5 / 5.3 us): the barrier and the
# dependency chains cost what the parallelism bought. This version: boundaries = one 512-thread CTA per token, 16
# coalesced columns per thread, alpha / beta of the residual gate precomputed at prepare (st["ab1"] / ["ab2"]: ~12
# instructions per element), the four row sums in ONE block reduction (sum s^2 = A + 2rB + r^2 D gives both norms), no
# inter-CTA traffic; routing = one 1024-thread CTA per token, top-8 by rank with 64-bit (ordered score, 255 - e) keys
# in shared memory (4 threads per expert; ties to the lower index as tl.argmax), the pre-expert norm fused: a 16-bit
# SIMD pre-pass (__vsetgtu2 on the keys' top halves, 2 keys per instruction) gives the rank up to 16-bit ties, and
# only the candidates (16-bit rank < 8) resolve the exact rank among themselves (an expert tied with a candidate at
# 16 bits is itself a candidate). Weights and inputs load before griddepcontrol.wait, the producer's data after it.
GLUE_CUDA = True
GLUE_MAX_T = 256
GLUE_MODE = 0        # PDL placement bits: 1 = launch_dependents after the stores (not at the start), 2 = wait before the prologue
glue__MOD = None

GLUE_CPP_SRC = r"""
void glue_bnd(int64_t kind, torch::Tensor a, int64_t nu, torch::Tensor st, torch::Tensor alpha, torch::Tensor beta,
              torch::Tensor g1, torch::Tensor g2, torch::Tensor sout, torch::Tensor hout, int64_t T, int64_t pdl,
              int64_t mode);
void glue_route(torch::Tensor pooled, std::optional<torch::Tensor> logits, torch::Tensor bias, torch::Tensor gpre,
                torch::Tensor ids, torch::Tensor wts, torch::Tensor xm, torch::Tensor ls, torch::Tensor lids,
                int64_t T, int64_t pdl, int64_t mode);
"""

GLUE_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <optional>

namespace okc_glue {

constexpr int HID = 8192, DEX = 4096, NEXP = 256, PRW = 4608;
constexpr int BNT = 512;            // boundary: one CTA per token, 512 threads x 16 columns (4 coalesced chunks of 4);
                                    // 1024 threads x 8 columns under mode bit 8 (A/B)
constexpr int RNT = 1024;           // routing: one CTA per token, 4 threads per expert, 4 pooled columns per thread

__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}
__device__ __forceinline__ float bf_lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf_hi(uint32_t v) { return __uint_as_float(v & 0xFFFF0000u); }
__device__ __forceinline__ uint32_t pack_bf2(float a, float b) {
  __nv_bfloat162 p = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&p);
}
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
// four floats / four bf16 at p (16 B / 8 B aligned); weights and inputs: plain loads (may be ld.global.nc); the
// producer's data (written by the grid this one may overlap under PDL): __ldcg, never ld.global.nc (ptxas hoists
// those above griddepcontrol.wait)
__device__ __forceinline__ void ld4f(const float* p, float* v) {
  const float4 q = *reinterpret_cast<const float4*>(p);
  v[0] = q.x; v[1] = q.y; v[2] = q.z; v[3] = q.w;
}
__device__ __forceinline__ void ld4bf(const __nv_bfloat16* p, float* v) {
  const uint2 q = *reinterpret_cast<const uint2*>(p);
  v[0] = bf_lo(q.x); v[1] = bf_hi(q.x); v[2] = bf_lo(q.y); v[3] = bf_hi(q.y);
}
template <typename T> __device__ __forceinline__ void ld4cg(const T* p, float* v);
template <> __device__ __forceinline__ void ld4cg<float>(const float* p, float* v) {
  const float4 q = __ldcg(reinterpret_cast<const float4*>(p));
  v[0] = q.x; v[1] = q.y; v[2] = q.z; v[3] = q.w;
}
template <> __device__ __forceinline__ void ld4cg<__nv_bfloat16>(const __nv_bfloat16* p, float* v) {
  const uint2 q = __ldcg(reinterpret_cast<const uint2*>(p));
  v[0] = bf_lo(q.x); v[1] = bf_hi(q.x); v[2] = bf_lo(q.y); v[3] = bf_hi(q.y);
}
__device__ __forceinline__ void st4f(float* p, const float* v) {
  *reinterpret_cast<float4*>(p) = make_float4(v[0], v[1], v[2], v[3]);
}
__device__ __forceinline__ void st4bf(__nv_bfloat16* p, const float* v) {
  *reinterpret_cast<uint2*>(p) = make_uint2(pack_bf2(v[0], v[1]), pack_bf2(v[2], v[3]));
}

// KIND 1 (post-attention): a = o-projection output [T, lda], st = the layer's input stream fp32 [T, 8192], alpha /
//   beta fp32 [8192] of attn_residual_gate (prepare-time _gate_coeffs), g1 = pre_ffn_norm:
//   s = alpha st + beta a rstd(a); h = s rstd(s) (1 + g1).
// KIND 2 (post-FFN): a = unpooler output [T, lda] (columns >= nu are zero: the 1024-row slice at T <= 8), st = s1,
//   alpha / beta of ffn_residual_gate, g1 = post_ffn_norm (scales a before its norm), g2 = next_pre_attn_norm:
//   f = a (1 + g1); s = alpha st + beta f rstd(f); h = s rstd(s) (1 + g2).
// One 512-thread CTA per token, thread tid owns chunks 4 (tid + k 512), k < 4 (coalesced). The four row sums
// (sum x^2, A = sum (alpha st)^2, B = sum (alpha st)(beta x), D = sum (beta x)^2) in ONE block reduction;
// sum s^2 = A + 2 r B + r^2 D gives the second norm without a second pass. No inter-CTA traffic.
template <int KIND, typename AT, int NT>
__global__ void __launch_bounds__(NT) bnd_kernel(const AT* a, int lda, int nu, const float* __restrict__ st,
    const float* __restrict__ alpha, const float* __restrict__ beta, const __nv_bfloat16* __restrict__ g1,
    const __nv_bfloat16* __restrict__ g2, float* sout, __nv_bfloat16* hout, int pdl, int mode) {
  constexpr int BCH = HID / (4 * NT);   // 16-B chunks per thread (4 at 512 threads, 2 at 1024)
  const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  __shared__ float red[NT / 32][4];
  __shared__ float scl[2];
  if (pdl && !(mode & 1)) pdl_trigger();
  if (pdl && (mode & 2)) pdl_wait();
  float sv[BCH][4], al[BCH][4], be[BCH][4], gw[BCH][4], gn[BCH][4];
#pragma unroll
  for (int k = 0; k < BCH; ++k) {
    const int col = 4 * (tid + k * NT);
    ld4f(st + (size_t)t * HID + col, sv[k]);
    ld4f(alpha + col, al[k]);
    ld4f(beta + col, be[k]);
    ld4bf(g1 + col, gw[k]);
    if (KIND == 2) ld4bf(g2 + col, gn[k]);
  }
  if (pdl) pdl_wait();
  float x[BCH][4];
#pragma unroll
  for (int k = 0; k < BCH; ++k) {
    const int col = 4 * (tid + k * NT);
    x[k][0] = x[k][1] = x[k][2] = x[k][3] = 0.f;
    if (KIND == 1 || col < nu) ld4cg<AT>(a + (size_t)t * lda + col, x[k]);
    if (KIND == 2) {
#pragma unroll
      for (int i = 0; i < 4; ++i) x[k][i] *= 1.f + gw[k][i];
    }
  }
  float q0 = 0.f, q1 = 0.f, q2 = 0.f, q3 = 0.f;
#pragma unroll
  for (int k = 0; k < BCH; ++k)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float as = al[k][i] * sv[k][i], bx = be[k][i] * x[k][i];
      q0 = fmaf(x[k][i], x[k][i], q0); q1 = fmaf(as, as, q1); q2 = fmaf(as, bx, q2); q3 = fmaf(bx, bx, q3);
    }
  q0 = warp_sum(q0); q1 = warp_sum(q1); q2 = warp_sum(q2); q3 = warp_sum(q3);
  if (lane == 0) { red[warp][0] = q0; red[warp][1] = q1; red[warp][2] = q2; red[warp][3] = q3; }
  __syncthreads();
  if (warp == 0) {
    float4 p = make_float4(0.f, 0.f, 0.f, 0.f);
    if (lane < NT / 32) p = make_float4(red[lane][0], red[lane][1], red[lane][2], red[lane][3]);
#pragma unroll
    for (int o = (NT / 32) / 2; o > 0; o >>= 1) {
      p.x += __shfl_xor_sync(0xffffffffu, p.x, o); p.y += __shfl_xor_sync(0xffffffffu, p.y, o);
      p.z += __shfl_xor_sync(0xffffffffu, p.z, o); p.w += __shfl_xor_sync(0xffffffffu, p.w, o);
    }
    if (lane == 0) {
      const float r = rsqrtf(p.x / (float)HID + 1e-8f);
      scl[0] = r;
      scl[1] = rsqrtf((p.y + 2.f * r * p.z + r * r * p.w) / (float)HID + 1e-5f);
    }
  }
  __syncthreads();
  const float r = scl[0], r1 = scl[1];
#pragma unroll
  for (int k = 0; k < BCH; ++k) {
    const int col = 4 * (tid + k * NT);
    float s[4], h[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float as = al[k][i] * sv[k][i], bx = be[k][i] * x[k][i];
      s[i] = fmaf(bx, r, as);
      h[i] = s[i] * r1 * (1.f + (KIND == 1 ? gw[k][i] : gn[k][i]));
    }
    st4f(sout + (size_t)t * HID + col, s);
    st4bf(hout + (size_t)t * HID + col, h);
  }
}

// order-preserving map of a float to a uint32 (NaN-free inputs)
__device__ __forceinline__ uint32_t ordered(float f) {
  const uint32_t u = __float_as_uint(f);
  return u ^ ((u >> 31) ? 0xFFFFFFFFu : 0x80000000u);
}

// One 1024-thread CTA per token. Routing: thread tid owns expert e = tid & 255 and compares its key against quarter
// (tid >> 8) of the 256 keys; key = (ordered score, 255 - e) so equal scores rank the lower index first (tl.argmax);
// rank r < 8 means slot r of the top-8 (no serial rounds). Pre-expert norm: 4 pooled columns per thread, one block
// reduction. SPLIT: pooled bf16 [T, 4096] + logits fp32 [T, 256]; else the combined fp32 pooler + router GEMM output
// P [T, 4608] (pooled = P[:, :4096], logits = P[:, 4096:4352] + P[:, 4352:4608]).
template <bool SPLIT>
__global__ void __launch_bounds__(RNT) route_kernel(const void* pooled_, const float* logits,
    const float* __restrict__ bias, const __nv_bfloat16* __restrict__ gpre, int* ids, float* wts,
    __nv_bfloat16* xm, const int* __restrict__ ls, int* lids, int pdl, int mode) {
  const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int e = tid & (NEXP - 1), quarter = tid >> 8;
  const int col = 4 * tid;   // pooled columns of this thread
  __shared__ unsigned long long key_s[NEXP];   // exact keys: (ordered score << 32) | (255 - e)
  __shared__ uint32_t k16_s[NEXP / 2];           // the keys' top 16 bits, pair j = experts 2j (low half), 2j + 1
  __shared__ int rk_s[4][NEXP];
  __shared__ int cand_s[NEXP];
  __shared__ int ncand_s;
  __shared__ int sel_idx[8];
  __shared__ float sel_val[8];
  __shared__ float qred[RNT / 32];
  __shared__ float scl;
  if (pdl && !(mode & 1)) pdl_trigger();
  if (pdl && (mode & 2)) pdl_wait();
  const float bv = bias[e];
  const int lsv = *ls;
  float gp[4];
  ld4bf(gpre + col, gp);
  if (pdl) pdl_wait();
  float lg, pv[4];
  if (SPLIT) {
    lg = __ldcg(logits + (size_t)t * NEXP + e);
    ld4cg<__nv_bfloat16>(reinterpret_cast<const __nv_bfloat16*>(pooled_) + (size_t)t * DEX + col, pv);
  } else {
    const float* P = reinterpret_cast<const float*>(pooled_) + (size_t)t * PRW;
    lg = __ldcg(P + DEX + e) + __ldcg(P + DEX + NEXP + e);
    ld4cg<float>(P + col, pv);
  }
  float q = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) q = fmaf(pv[i], pv[i], q);
  q = warp_sum(q);
  if (lane == 0) qred[warp] = q;
  const float sc = sigm(lg * 0.1767766952966369f);
  const float sel = sc + bv;
  const uint32_t ok = ordered(sel);
  const uint32_t k16 = ok >> 16;
  const uint32_t nb16 = __shfl_down_sync(0xffffffffu, k16, 1);   // expert e + 1 is the next lane
  if (quarter == 0) {
    key_s[e] = ((unsigned long long)ok << 32) | (unsigned)(NEXP - 1 - e);
    if (!(e & 1)) k16_s[e >> 1] = k16 | (nb16 << 16);
  }
  if (tid == 0) ncand_s = 0;
  __syncthreads();
  int rk;
  if (mode & 4) {   // the exact 64-bit rank over this thread's quarter of the keys (the previous form, for A/B)
    const unsigned long long mykey = key_s[e];
    const unsigned long long* kq = key_s + quarter * (NEXP / 4);
    rk = 0;
#pragma unroll 16
    for (int j = 0; j < NEXP / 4; ++j) rk += kq[j] > mykey;
  } else {
    // 16-bit SIMD pre-pass: how many keys beat mine in their top 16 bits (2 keys per compare, 8 x 16 B of smem)
    const uint32_t me2 = k16 * 0x00010001u;
    uint32_t acc = 0;
    const uint4* kp = reinterpret_cast<const uint4*>(k16_s) + quarter * (NEXP / 2 / 4 / 4);
#pragma unroll
    for (int i = 0; i < NEXP / 2 / 4 / 4; ++i) {
      const uint4 v = kp[i];
      acc = __vadd2(acc, __vsetgtu2(v.x, me2)); acc = __vadd2(acc, __vsetgtu2(v.y, me2));
      acc = __vadd2(acc, __vsetgtu2(v.z, me2)); acc = __vadd2(acc, __vsetgtu2(v.w, me2));
    }
    rk = (int)((acc & 0xFFFFu) + (acc >> 16));
  }
  rk_s[quarter][e] = rk;
  __syncthreads();
  bool cand = false;
  int rk0 = 0;
  if (quarter == 0) {
    rk0 = rk_s[0][e] + rk_s[1][e] + rk_s[2][e] + rk_s[3][e];
    if (mode & 4) {
      if (rk0 < 8) { sel_idx[rk0] = e; sel_val[rk0] = sc; }
    } else if (rk0 < 8) {   // a top-8 candidate by 16-bit rank: exact rank among the candidates below
      cand = true;
      cand_s[atomicAdd(&ncand_s, 1)] = e;
    }
  }
  if (tid == 0) {
    float qs = 0.f;
#pragma unroll
    for (int w = 0; w < RNT / 32; ++w) qs += qred[w];
    scl = rsqrtf(qs / (float)DEX + 1e-5f);
  }
  __syncthreads();
  if (cand) {
    // every expert tied with a candidate at 16 bits has the same 16-bit rank, so it is a candidate too: the exact
    // rank is the 16-bit rank plus the tied candidates whose full key beats mine (ties to the lower index)
    const int n = ncand_s;
    const unsigned long long mk = key_s[e];
    int extra = 0;
    for (int i = 0; i < n; ++i) {
      const unsigned long long kd = key_s[cand_s[i]];
      extra += ((uint32_t)(kd >> 48) == k16) && (kd > mk);
    }
    const int r = rk0 + extra;
    if (r < 8) { sel_idx[r] = e; sel_val[r] = sc; }
  }
  __syncthreads();
  if (tid < 8) {
    float tot = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) tot += sel_val[k];
    const int id = sel_idx[tid];
    ids[t * 8 + tid] = id;
    wts[t * 8 + tid] = sel_val[tid] / (tot + 1e-15f);
    const int lid = (id - lsv + 256) % 256;
    lids[t * 8 + tid] = lid < 32 ? lid : -1;
  }
  const float r = scl;
  float xo[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) xo[i] = pv[i] * r * (1.f + gp[i]);   // the pre-expert norm's weight is (1 + g)
  st4bf(xm + (size_t)t * DEX + col, xo);
}

static void set_pdl(cudaLaunchConfig_t& lc, cudaLaunchAttribute* at_, int pdl) {
  lc.stream = at::cuda::getCurrentCUDAStream();
  lc.dynamicSmemBytes = 0;
  lc.numAttrs = 0;
  lc.attrs = at_;
  if (pdl) {   // programmatic dependent launch: may start while the previous kernel in the stream finishes
    at_[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at_[0].val.programmaticStreamSerializationAllowed = 1;
    lc.numAttrs = 1;
  }
}

template <int KIND, typename AT, int NT>
static void launch_bnd_nt(const torch::Tensor& a, int lda, int nu, const torch::Tensor& st, const torch::Tensor& alpha,
                          const torch::Tensor& beta, const torch::Tensor& g1, const torch::Tensor& g2, torch::Tensor& sout,
                          torch::Tensor& hout, int T, int pdl, int mode) {
  cudaLaunchConfig_t lc = {};
  cudaLaunchAttribute at_[1];
  lc.gridDim = dim3(T);
  lc.blockDim = dim3(NT);
  set_pdl(lc, at_, pdl);
  C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, bnd_kernel<KIND, AT, NT>, reinterpret_cast<const AT*>(a.data_ptr()), lda, nu,
      st.data_ptr<float>(), alpha.data_ptr<float>(), beta.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(g1.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(g2.data_ptr()),
      sout.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(hout.data_ptr()), pdl, mode));
}
template <int KIND, typename AT>
static void launch_bnd(const torch::Tensor& a, int lda, int nu, const torch::Tensor& st, const torch::Tensor& alpha,
                       const torch::Tensor& beta, const torch::Tensor& g1, const torch::Tensor& g2, torch::Tensor& sout,
                       torch::Tensor& hout, int T, int pdl, int mode) {
  if (mode & 8) launch_bnd_nt<KIND, AT, 1024>(a, lda, nu, st, alpha, beta, g1, g2, sout, hout, T, pdl, mode);
  else launch_bnd_nt<KIND, AT, BNT>(a, lda, nu, st, alpha, beta, g1, g2, sout, hout, T, pdl, mode);
}

template <bool SPLIT>
static void launch_route(const void* pooled, const float* logits, const torch::Tensor& bias, const torch::Tensor& gpre,
                         torch::Tensor& ids, torch::Tensor& wts, torch::Tensor& xm, const torch::Tensor& ls,
                         torch::Tensor& lids, int T, int pdl, int mode) {
  cudaLaunchConfig_t lc = {};
  cudaLaunchAttribute at_[1];
  lc.gridDim = dim3(T);
  lc.blockDim = dim3(RNT);
  set_pdl(lc, at_, pdl);
  C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, route_kernel<SPLIT>, pooled, logits, bias.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(gpre.data_ptr()), ids.data_ptr<int>(), wts.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(xm.data_ptr()), ls.data_ptr<int>(), lids.data_ptr<int>(), pdl, mode));
}

}  // namespace okc_glue

void glue_bnd(int64_t kind, torch::Tensor a, int64_t nu, torch::Tensor st, torch::Tensor alpha, torch::Tensor beta,
              torch::Tensor g1, torch::Tensor g2, torch::Tensor sout, torch::Tensor hout, int64_t T, int64_t pdl,
              int64_t mode) {
  using namespace okc_glue;
  TORCH_CHECK(kind == 1 || kind == 2, "glue_bnd: kind");
  TORCH_CHECK(a.is_contiguous() && st.is_contiguous() && sout.is_contiguous() && hout.is_contiguous(), "glue_bnd: contiguous");
  TORCH_CHECK(alpha.is_contiguous() && beta.is_contiguous() && alpha.numel() == HID && beta.numel() == HID, "glue_bnd: alpha/beta");
  TORCH_CHECK(alpha.scalar_type() == at::kFloat && beta.scalar_type() == at::kFloat, "glue_bnd: alpha/beta fp32");
  TORCH_CHECK(a.dim() == 2 && a.size(0) == T && st.size(0) == T && sout.size(0) == T && hout.size(0) == T, "glue_bnd: rows");
  TORCH_CHECK(st.scalar_type() == at::kFloat && sout.scalar_type() == at::kFloat && hout.scalar_type() == at::kBFloat16, "glue_bnd: dtypes");
  TORCH_CHECK(g1.scalar_type() == at::kBFloat16 && g2.scalar_type() == at::kBFloat16 && g1.numel() == HID && g2.numel() == HID, "glue_bnd: norm weights");
  TORCH_CHECK(st.size(1) == HID && sout.size(1) == HID && hout.size(1) == HID, "glue_bnd: width");
  TORCH_CHECK(nu > 0 && nu <= HID && nu % 4 == 0 && a.size(1) >= nu, "glue_bnd: nu");
  const int lda = (int)a.size(1);
  const bool bf = a.scalar_type() == at::kBFloat16;
  TORCH_CHECK(bf || a.scalar_type() == at::kFloat, "glue_bnd: a dtype");
  if (kind == 1) {
    if (bf) launch_bnd<1, __nv_bfloat16>(a, lda, (int)nu, st, alpha, beta, g1, g2, sout, hout, (int)T, (int)pdl, (int)mode);
    else launch_bnd<1, float>(a, lda, (int)nu, st, alpha, beta, g1, g2, sout, hout, (int)T, (int)pdl, (int)mode);
  } else {
    if (bf) launch_bnd<2, __nv_bfloat16>(a, lda, (int)nu, st, alpha, beta, g1, g2, sout, hout, (int)T, (int)pdl, (int)mode);
    else launch_bnd<2, float>(a, lda, (int)nu, st, alpha, beta, g1, g2, sout, hout, (int)T, (int)pdl, (int)mode);
  }
}

void glue_route(torch::Tensor pooled, std::optional<torch::Tensor> logits, torch::Tensor bias, torch::Tensor gpre,
                torch::Tensor ids, torch::Tensor wts, torch::Tensor xm, torch::Tensor ls, torch::Tensor lids,
                int64_t T, int64_t pdl, int64_t mode) {
  using namespace okc_glue;
  TORCH_CHECK(pooled.is_contiguous() && xm.is_contiguous() && ids.is_contiguous() && wts.is_contiguous() && lids.is_contiguous(), "glue_route: contiguous");
  TORCH_CHECK(pooled.dim() == 2 && pooled.size(0) == T && xm.size(0) == T && xm.size(1) == DEX, "glue_route: shapes");
  TORCH_CHECK(bias.scalar_type() == at::kFloat && bias.numel() == NEXP && gpre.scalar_type() == at::kBFloat16 && gpre.numel() == DEX, "glue_route: weights");
  TORCH_CHECK(ids.scalar_type() == at::kInt && lids.scalar_type() == at::kInt && wts.scalar_type() == at::kFloat && ls.scalar_type() == at::kInt, "glue_route: outputs");
  TORCH_CHECK(xm.scalar_type() == at::kBFloat16, "glue_route: xm bf16");
  if (logits.has_value()) {
    const torch::Tensor lg = logits.value();
    TORCH_CHECK(lg.is_contiguous() && lg.scalar_type() == at::kFloat && lg.size(0) == T && lg.size(1) == NEXP, "glue_route: logits");
    TORCH_CHECK(pooled.scalar_type() == at::kBFloat16 && pooled.size(1) == DEX, "glue_route: pooled bf16 [T, 4096]");
    launch_route<true>(pooled.data_ptr(), lg.data_ptr<float>(), bias, gpre, ids, wts, xm, ls, lids, (int)T, (int)pdl, (int)mode);
  } else {
    TORCH_CHECK(pooled.scalar_type() == at::kFloat && pooled.size(1) == PRW, "glue_route: P fp32 [T, 4608]");
    launch_route<false>(pooled.data_ptr(), nullptr, bias, gpre, ids, wts, xm, ls, lids, (int)T, (int)pdl, (int)mode);
  }
}
"""


def _glue_build():
    from torch.utils.cpp_extension import load_inline
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}" + ("a" if major >= 9 else "")
    return load_inline(
        name=f"okc_glue_v7_{arch}",
        cpp_sources=GLUE_CPP_SRC,
        cuda_sources=GLUE_CUDA_SRC,
        functions=["glue_bnd", "glue_route"],
        extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"],
        verbose=False,
    )


def _glue7():
    """Build the glue extension (in prepare's build thread pool); a failure only disables it."""
    global glue__MOD
    if GLUE_CUDA and glue__MOD is None:
        try:
            glue__MOD = _glue_build()
        except Exception as ex:  # noqa: BLE001
            print("glue kernels: build failed:", repr(ex)[:300], flush=True)


def _glue_ok(state, T, kind="r"):
    """kind: 1 (boundary 1), 2 (boundary 2), "r" (routing); state["glue_kinds"] narrows the set (diagnosis)."""
    return (GLUE_CUDA and glue__MOD is not None and state.get("glue_ok") and T <= GLUE_MAX_T
            and state.get("glue_on", {}).get(T, True) and kind in state.get("glue_kinds", (1, 2, "r")))


def _glue_bnd(kind, a, nu, st_in, ab, g1, g2, sout, hout, state, T):
    """ab = (alpha, beta) fp32 [8192] of the boundary's residual gate (state["ab1"] / ["ab2"], prepare-time)."""
    glue__MOD.glue_bnd(kind, a, nu, st_in, ab[0], ab[1], g1, g2, sout, hout, T, int(_pdl(T)),
                       GLUE_MODE if state.get("glue_mode") is None else state["glue_mode"])


def _glue_route(pooled, logits, state, ls, ids, wts, xm, lids, T):
    glue__MOD.glue_route(pooled, logits, state["bias"], state["g_preexp"], ids, wts, xm, ls, lids, T, int(_pdl(T)),
                         GLUE_MODE if state.get("glue_mode") is None else state["glue_mode"])


def _glue_selftest(st):
    """The CUDA glue kernels against the Triton ones at T = 1, 8, 32, 128 on synthetic inputs (bf16 and fp32
    producers, the 1024-wide unpooler slice, the split and the combined routing inputs); any mismatch or exception
    disables them (st["glue_ok"]); printed."""
    st["glue_ok"] = False
    if not GLUE_CUDA or glue__MOD is None:
        print("glue kernels: not available", flush=True)
        return
    dev = st["qkv"].device
    f32 = dict(device=dev, dtype=torch.float32)
    g = torch.Generator(device=dev).manual_seed(781)
    ls = torch.zeros(1, device=dev, dtype=torch.int32)
    rel = lambda x, y: float(((x.float() - y.float()).norm() / (y.float().norm() + 1e-30)).item())  # noqa: E731
    report, worst = [], 0.0
    ids_ok = True
    for T in (1, 8, 32, 128):
        for a_f32 in ((False,) if T < 128 else (False, True)):
            a = (torch.randn(T, HIDDEN, generator=g, device=dev) * 0.1).to(torch.bfloat16)
            if a_f32:
                a = a.float()
            stream = torch.randn(T, HIDDEN, generator=g, device=dev)
            s1r, hr = torch.empty((T, HIDDEN), **f32), torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
            s1, h = torch.empty((T, HIDDEN), **f32), torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
            _boundary1[(T,)](a.view(1, T, HIDDEN), 1, T, stream, st["gate1"], st["gate1"], st["g_preffn"], s1r, hr, num_warps=16)
            _glue_bnd(1, a, HIDDEN, stream, st["ab1"], st["g_preffn"], st["g_preffn"], s1, h, st, T)
            e1, e2 = rel(s1, s1r), rel(h, hr)
            nu = 1024 if T <= 8 else HIDDEN
            u = (torch.randn(T, nu, generator=g, device=dev) * 0.1).to(torch.bfloat16)
            if a_f32:
                u = u.float()
            s1x = torch.randn(T, HIDDEN, generator=g, device=dev)
            s2r, outr = torch.empty((T, HIDDEN), **f32), torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
            s2, out = torch.empty((T, HIDDEN), **f32), torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
            _boundary2[(T,)](u.view(1, T, nu), 1, T, nu, s1x, st["gate2"], st["gate2"], st["g_postffn"], st["g_next"],
                             s2r, outr, num_warps=16)
            _glue_bnd(2, u, nu, s1x, st["ab2"], st["g_postffn"], st["g_next"], s2, out, st, T)
            e3, e4 = rel(s2, s2r), rel(out, outr)
            worst = max(worst, e1, e3, e2 / 8, e4 / 8)   # bf16 outputs: a few 1-ulp flips are 1e-3..1e-2 of rel L2
            report.append(f"T{T}{'f' if a_f32 else ''} b1 {e1:.1e}/{e2:.1e} b2 {e3:.1e}/{e4:.1e}")
        pooled = (torch.randn(T, D_EXPERT, generator=g, device=dev) * 0.1).to(torch.bfloat16)
        logits = torch.randn(T, N_EXPERTS, generator=g, device=dev) * 3.0
        if T in (8, 128):   # coarse logits: many scores equal at 16 bits and exactly (the rank's tie paths)
            logits = torch.round(logits * 2.0) / 2.0
        outs = [(torch.empty((T, 8), device=dev, dtype=torch.int32), torch.empty((T, 8), **f32),
                 torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16), torch.empty((T, 8), device=dev, dtype=torch.int32))
                for _ in range(4)]
        _route_split[(T,)](pooled, logits, st["bias"], st["g_preexp"], outs[0][0], outs[0][1], outs[0][2], ls, outs[0][3],
                           PERMUTED=False, num_warps=8)
        _glue_route(pooled, logits, st, ls, outs[1][0], outs[1][1], outs[1][2], outs[1][3], T)
        P = torch.randn(T, 4608, generator=g, device=dev)
        P[:, :D_EXPERT] *= 0.1
        _route[(T,)](P.view(1, T, 4608), 1, T, st["bias"], st["g_preexp"], outs[2][0], outs[2][1], outs[2][2], ls, outs[2][3],
                     PERMUTED=False, num_warps=8)
        _glue_route(P, None, st, ls, outs[3][0], outs[3][1], outs[3][2], outs[3][3], T)
        for ref, mine in ((outs[0], outs[1]), (outs[2], outs[3])):
            ids_ok &= bool(torch.equal(ref[0], mine[0]) and torch.equal(ref[3], mine[3]))
            e5, e6 = rel(mine[1], ref[1]), rel(mine[2], ref[2])
            worst = max(worst, e5, e6 / 8)
            report.append(f"T{T} route w {e5:.1e} x {e6:.1e}")
    torch.cuda.synchronize()
    st["glue_ok"] = ids_ok and worst < 1e-3
    print(f"glue kernels {'ok' if st['glue_ok'] else 'MISMATCH'}: ids {'equal' if ids_ok else 'DIFFER'}; " + " ".join(report), flush=True)


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
    for T in TUNED_T:
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
            pin = PIN_PICK.get((T, name))
            if (pin is not None and not isinstance(pin, tuple)) or (name == "pr" and T in PR_FORM_LOCK):
                continue   # decided (PIN_PICK, applied after the tuner; PR_FORM_LOCK below): no isolated timing
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
                try:
                    t_mm16 = _dense_t("mm16")
                except Exception:  # noqa: BLE001
                    t_mm16 = 1e30
            t_torchT = t_bfq = 1e30
            if T > 32:   # the T > 32 candidates: cuBLAS on the K-major copy, and the proj module's own tuned route
                try:
                    t_torchT = _dense_t("torchT")
                except Exception:  # noqa: BLE001
                    t_torchT = 1e30
                if st.get("bfp") is not None:
                    try:
                        if name == "pr":
                            t_bfq = _graph_time(lambda: _pr_split(_bfp_gemm(st, "pool", xd, T, D_EXPERT, HIDDEN),
                                                                  _bfp_gemm(st, "router", xd, T, N_EXPERTS, HIDDEN)),
                                                flush, reps=5)
                        else:
                            t_bfq = _dense_t("bfq")
                    except Exception as ex:  # noqa: BLE001
                        t_bfq = 1e30
                        if log:
                            msg.append(f"bfqX:{repr(ex)[:100]}")
            opts = {False: t_tri, True: t_mm, "bfp": t_bfp, "bfpf": t_bfpf, "mm16": t_mm16, "torchT": t_torchT, "bfq": t_bfq}
            st.setdefault("mm_times", {})[(T, name)] = opts
            bestp = min(opts, key=lambda p: opts[p])
            st["mm_pick"][(T, name)] = bestp
            if log:
                letters = {False: 'T', True: 'M', 'bfp': 'B', 'bfpf': 'F', 'mm16': 'm', 'torchT': 't', 'bfq': 'q'}
                msg.append(f"{name}{letters[bestp]}:{opts[bestp]:.0f}")
                if T >= 128:   # DIAG: every candidate's graph time (us) for the T > 32 dense GEMMs
                    msg.append(f"{name}[" + " ".join(f"{letters[p]}{v:.1f}" for p, v in opts.items() if v < 1e29) + "]")
        try_marlin = st["bank"] is not None and (TRY_MARLIN or not st["ma_ok"])
        if try_marlin and T <= 384:
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
        cands = [c for c in ("marlin", "agent") if (c != "marlin" or try_marlin)
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
        if T in PDL_LOCK:
            _PDL_N_BY_T[T] = PDL_LOCK[T]
        elif _pdl(T) and st.get("bfp") is not None:   # sk2 PDL depth: weight stages issued before griddepcontrol.wait,
            ptimes = {}                               # timed on the whole post-attention chain (where the overlap is)
            for n in PDL_N_SWEEP:
                _PDL_N_BY_T[T] = n
                try:
                    ptimes[n] = _graph_time(lambda: _after_attention(data, st, o_in, None, T), flush)
                except Exception:  # noqa: BLE001
                    ptimes[n] = 1e30
            nbest = min(ptimes, key=lambda n: ptimes[n])
            _PDL_N_BY_T[T] = nbest
            if log:
                msg.append("pdl[" + " ".join(f"{n}:{t:.0f}" for n, t in ptimes.items()) + f"]->{nbest}")
        modes = (["split"] + (["sk2f"] if st.get("fuse_route_ok") else []) + (["moe"] if st.get("route_moe_ok") else [])
                 + (["moe_nb"] if st.get("route_moe_nb_ok") else []))
        if T in ROUTE_MODE_LOCK:
            st.setdefault("route_mode", {})[T] = ROUTE_MODE_LOCK[T] if ROUTE_MODE_LOCK[T] in modes else "split"
        elif len(modes) > 1 and T <= 32:   # where the routing runs: the _route_split kernel, CTA t of the fused pooler +
            rtimes = {}                   # router launch after its grid barrier, or every CTA of the MoE kernel
            for mode in modes:
                st.setdefault("route_mode", {})[T] = mode
                try:
                    rtimes[mode] = _graph_time(lambda: _after_attention(data, st, o_in, None, T), flush)
                except Exception as ex:  # noqa: BLE001
                    rtimes[mode] = 1e30
                    if log:
                        msg.append(f"rmodeX:{mode}:{repr(ex)[:80]}")
            st["route_mode"][T] = min(rtimes, key=lambda m_: rtimes[m_])
            if log:
                msg.append("rmode[" + " ".join(f"{m_}:{t:.0f}" for m_, t in rtimes.items()) + f"]->{st['route_mode'][T]}")
        if T in PR_FORM_LOCK and st.get("bfp") is not None:
            form, flag = PR_FORM_LOCK[T]
            st["mm_pick"][(T, "pr")] = form
            st.setdefault("bnd1_on", {})[T] = bool(flag) and T in st.get("fuse_bnd1_ok_T", ())
        elif T <= PRFORM_MAX_T and st.get("bfp") is not None:
            # the pooler + router form at this T, on the chain: separate launches (pooler cgemv / sk2 + router rgemv /
            # sk2) + _boundary1, the fused launch + _boundary1, or (T <= BND1_MAX_T) the fused launch computing the
            # boundary in its prologue (no _boundary1 launch); the prologue and the separate m = 1 GEMVs exclude each
            # other (1d2eac51: separate won by 3 us at T = 1)
            cands = [("bfp", False), ("bfpf", False)] + ([("bfpf", True)] if T in st.get("fuse_bnd1_ok_T", ()) else [])
            btimes = {}
            for pr_pick, flag in cands:
                st["mm_pick"][(T, "pr")] = pr_pick
                st.setdefault("bnd1_on", {})[T] = flag
                try:
                    btimes[(pr_pick, flag)] = _graph_time(lambda: _after_attention(data, st, o_in, None, T), flush)
                except Exception as ex:  # noqa: BLE001
                    btimes[(pr_pick, flag)] = 1e30
                    if log:
                        msg.append(f"prX:{pr_pick}{int(flag)}:{repr(ex)[:80]}")
            best_pr = min(btimes, key=lambda k_: btimes[k_])
            st["mm_pick"][(T, "pr")], st["bnd1_on"][T] = best_pr
            if log:
                msg.append("prform[" + " ".join(f"{p_}{'+bnd1' if f_ else ''}:{t:.0f}" for (p_, f_), t in btimes.items())
                           + f"]->{best_pr[0]}{'+bnd1' if best_pr[1] else ''}")
        if T <= BTAIL_MAX_T and st.get("fuse_bnd_ok"):   # the boundary tails per (T, layer): locked, not timed
            for l_ in ("unpooler", "o"):
                st.setdefault("btail_on", {})[(T, l_)] = l_ in BTAIL_LOCK.get(T, ())
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
    _t0 = time.perf_counter()   # wall time per phase printed (the judge's .log): the budget is 300 s

    def _ph(what):
        torch.cuda.synchronize()
        print(f"prepare: {what} at {time.perf_counter() - _t0:.1f} s", flush=True)

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
    for name, key in _T_COPY.items():   # K-major copies for the "torchT" cuBLAS candidate at T > 32 (~200 MB)
        st[key] = st[{"qkv": "qkv", "o": "o", "pr": "pr", "un": "unpool"}[name]].t().contiguous()
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
    builds = [_att7, _glue7] + [(lambda d=d: moe__ext(d)) for d in _moe_builds(sfx)]
    ths = [threading.Thread(target=_build, args=(f,)) for f in builds]  # ~85 s of nvcc, in parallel
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    if errs:
        raise errs[0]
    _ph("builds")
    _ma_prepare(weights, st)
    _ph("moe prepare")
    _bfp_prepare(weights, st)
    _ph("proj prepare")
    for selftest in (_route_selftest, _boundary_selftest, _bnd1_selftest, _norm_selftest, _glue_selftest):   # the fused paths, each enabled only when it matches
        try:
            selftest(st)
        except Exception as ex:  # noqa: BLE001
            print(f"{selftest.__name__} failed: {repr(ex)[:300]}", flush=True)
    _ph("self-tests")
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
    _ph("marlin bank")
    try:
        _tune(st, log=_TUNE_LOG)
    except Exception as ex:  # noqa: BLE001
        print("tune failed:", repr(ex)[:200], flush=True)
        st["pbmax"] = 2
        st["moe_pick"] = {}
        st["gluev2"] = {}
        st["marlin_cfg"] = {}
        st["mm_pick"] = {}
    _ph("tuner")
    for k, v in PIN_PICK.items():      # in-chain winners the isolated tuner sometimes misses (no PDL in its timing)
        if k[1] == "pr" and k[0] <= PRFORM_MAX_T:
            continue   # the pooler + router form at T <= PRFORM_MAX_T is chosen on the chain in _tune ("prform")
        if isinstance(v, tuple):
            times = st.get("mm_times", {}).get(k, {})
            v = min(v, key=lambda p: times.get(p, 1e30))
        st.setdefault("mm_pick", {})[k] = v
    torch.cuda.synchronize()
    if st.get("att_ok"):
        _attn_selftest(st)     # every attention plan family once, on synthetic inputs
    _check_guards()            # after the tuner (every moe build) and the attention self-test
    _ph("done")
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
ATT_BUILD_FLAGS = tuple(f for f in os.environ.get("OKC_ATT_FLAGS", "").split(",") if f)   # (GPU A/B hook; default ())
                       # A/B 7d23a37c vs c8b5aab9: the standalone defaults (QDIRECT=1 PK2=1 W0ENABLE=0) measured 0.3-3 us
                       # faster per attention launch on 10 of 12 shapes than the layer's old build (0/0/1)


def _att7():
    global att__MOD
    if att__MOD is None:
        att__MOD = att__build("w1", ATT_BUILD_FLAGS)
    return att__MOD


ATT_HAS_PDL = True   # does the embedded attention launcher take pdl= (their v63+)?


ATT_PLAN_OVERRIDE = {}   # (B, q_len, sliding) -> plan kwargs for the att module's launcher (n_splits / flat / cs / ncl /
                         # nc / coop / w0): the chain-timed alternatives to its own plan (vm_ab/ab17); None = all shapes


def _decode_attention7(data, state, qkv):
    """The attention module's own launcher (att__launch): plan, workspace sizes and the coop / w0 bits always match
    the kernel version composed in. This layer's config already carries their keys (heads, head_dim, page_size,
    window, rope_theta, scales)."""
    cfg = data["config"]
    pdl = {"pdl": _pdl(cfg["batch"] * cfg["q_len"])} if ATT_HAS_PDL else {}   # their launcher's PDL switch
    plan = ATT_PLAN_OVERRIDE.get((cfg["batch"], cfg["q_len"], bool(cfg["window"])), ATT_PLAN_OVERRIDE.get(None, {}))
    out = att__launch({"config": cfg, "qkv": qkv, "seq_lens": data["seq_lens"], "cache_loc": data["cache_loc"],
                    "k_cache": data["k_cache"], "v_cache": data["v_cache"], "page_table": data["page_table"],
                    "k_scale": data["k_scale"], "v_scale": data["v_scale"]}, **pdl, **plan)
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
    bnd1 = None   # set when _boundary1 is deferred into the fused pooler + router launch's prologue
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
    elif (gate is None and _bfp_tail_ok(state, T, "o") and state.get("mm_pick", {}).get((T, "o")) == "bfp"
          and data["stream"].dtype == torch.float32 and data["stream"].is_contiguous()):
        # o projection whose last CTA (atomic ticket) also computes the post-attention boundary (s1, h): no
        # _boundary1 launch
        _bfp_pdl(T)
        globals()["bfp_run_sk2_tail"](o_in, "o", T, HIDDEN, 1024, state["bfp"],
                         (state["ab1"][0], state["ab1"][1], data["stream"], None, state["g_preffn"], s1, h))
    else:
        if gate is None:
            P, S = dense(o_in, state["o"], T, state, "o")
        else:
            P, S = dense_ng(o_in, gate, state["o"], T, state)
        if (gate is None and S == 1 and P.dtype == torch.bfloat16 and _bnd1_fuse_ok(state, T)
                and data["stream"].dtype == torch.float32 and data["stream"].is_contiguous()):
            # deferred: the fused pooler + router launch computes h (its own x) and s1 in its prologue (no launch)
            bnd1 = (P.view(T, HIDDEN), data["stream"], state["ab1"][0], state["ab1"][1], state["g_preffn"], s1)
        elif (S == 1 and P.dtype in (torch.bfloat16, torch.float32) and _glue_ok(state, T, 1)
              and data["stream"].dtype == torch.float32 and data["stream"].is_contiguous()):
            _glue_bnd(1, P.view(T, HIDDEN), HIDDEN, data["stream"], state["ab1"], state["g_preffn"], state["g_preffn"],
                      s1, h, state, T)
        else:
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
    use_fused_pr = ppick == "bfpf" and T <= 32 and state.get("bfp") is not None
    if bnd1 is not None and not use_fused_pr:   # the deferred boundary has no fused launch to live in: run it
        _boundary1[(T,)](bnd1[0].view(1, T, HIDDEN), 1, T, bnd1[1], state["gate1"], state["gate1"], state["g_preffn"],
                         s1, h, num_warps=16, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
        bnd1 = None
    if use_fused_pr:
        # bf16_proj's fused pooler + router: one launch, both read h once (bf16 [T, 4096], fp32 [T, 256]); with
        # FUSE_ROUTE its last CTA also does the routing and the pre-expert norm (no _route_split launch); with
        # bnd1 its prologue computes h (= its x) and s1 from the o-projection output (no _boundary1 launch)
        _bfp_pdl(T)
        route = _fused_route(state, data, ids, lids, wts, xm) if (use_marlin or use_ma) else None
        if route is None and use_ma and _route_in_moe_ok(state, T):
            # the launch also normalizes xm (grid barrier, every CTA its own columns); the routing is the MoE
            # kernel's, from the router logits: no _route_split launch
            P, S = bfp_fused_pooler_router(h, state["bfp"], None, bnd1, (state["g_preexp"], xm)), -3
        elif route is None and use_ma and _route_mode(state, T) == "moe_nb":
            # no barrier: the launch writes pooled * (1 + w) and the per-token partial sums; the MoE kernel forms the
            # RMS scale in its GEMM1 epilogue and routes from the logits (pooled IS the experts' input)
            P, S = bfp_fused_pooler_router(h, state["bfp"], None, bnd1, None, state["g_preexp"]), -4
        elif route is None:   # (the standalone proj module's launch takes no route argument)
            if bnd1 is not None:
                P, S = bfp_fused_pooler_router(h, state["bfp"], None, bnd1), -1
            else:
                P, S = bfp_fused_pooler_router(h, state["bfp"]), -1
        else:
            P, S = bfp_fused_pooler_router(h, state["bfp"], route, bnd1), -2
    elif ppick in ("bfp", "bfq") and state.get("bfp") is not None:   # ("bfq": the proj module's m >= 33 routes)
        pooled = _bfp_gemm(state, "pool", h, T, D_EXPERT, HIDDEN)          # bf16 [T, 4096]
        logits = _bfp_gemm(state, "router", h, T, N_EXPERTS, HIDDEN)       # fp32 [T, 256], hi+lo already summed
        P, S = (pooled, logits), -1
    else:
        P, S = dense(h, state["pr"], T, state, "pr")
    if S == -2:  # the routing and xm were produced inside the fused pooler + router launch
        pass
    elif S == -3:  # xm was normalized inside the fused launch; the MoE kernel routes from the logits (below)
        pooled, logits = P
    elif S == -4:  # pooled * (1 + w) + partial sums from the fused launch; the MoE kernel scales and routes (below)
        pooled, logits = P
    elif S == -1:  # pooler and router came from the bf16_proj kernels as two separate tensors
        pooled, logits = P
        if ((use_marlin or use_ma) and _glue_ok(state, T) and pooled.dtype == torch.bfloat16
                and logits.dtype == torch.float32 and data["local_start"].dtype == torch.int32):
            _glue_route(pooled, logits, state, data["local_start"], ids, wts, xm, lids, T)
        else:
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
    elif (S == 1 and (use_marlin or use_ma) and _glue_ok(state, T) and P.dtype == torch.float32
          and P.shape[-1] == 4608 and data["local_start"].dtype == torch.int32):
        _glue_route(P.view(T, 4608), None, state, data["local_start"], ids, wts, xm, lids, T)
    else:
        _route[(T,)](P, S, T, state["bias"], state["g_preexp"], ids, wts, xm, data["local_start"], lids,
                     PERMUTED=not (use_marlin or use_ma), num_warps=8, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    # 5. experts
    if use_ma and S == -3:
        # routing in the MoE kernel from the router logits (every CTA; CTA 0 writes the chosen global ids into ids)
        m = _ma_run_logits(state, xm, logits, state["bias"], data["local_start"], ids, T)
    elif use_ma and S == -4:
        # as above, plus the per-token RMS scale from the fused launch's partial sums (token-major [T, 128] fp32)
        m = _ma_run_logits(state, pooled, logits, state["bias"], data["local_start"], ids, T,
                           partials=state["bfp"]["fused_part2"][:T * 128].view(T, 128))
    elif use_ma:
        m = _ma_run(state, xm, lids, wts, T)   # lids: local expert index, -1 = another GPU's expert
    elif use_marlin:
        m = _marlin_experts(xm, wts, lids, state)
    # 6. unpooler + post-FFN boundary + next layer's norm
    wu = state["unpool_slice"] if T <= 8 else state["unpool"]
    nu = wu.shape[0]
    s2 = torch.empty((T, HIDDEN), **f32)
    out = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    if not v2 and _bfp_tail_ok(state, T, "unpooler") and state.get("mm_pick", {}).get((T, "un")) == "bfp":
        # unpooler slice whose last CTA (atomic ticket) also computes the post-FFN boundary (s2, out): no
        # _boundary2 launch
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
    elif S == 1 and P.dtype in (torch.bfloat16, torch.float32) and P.shape[-1] == nu and _glue_ok(state, T, 2):
        _glue_bnd(2, P.view(T, nu), nu, s1, state["ab2"], state["g_postffn"], state["g_next"], s2, out, state, T)
    else:
        _boundary2[(T,)](P, S, T, nu, s1, state["gate2"], state["gate2"], state["g_postffn"], state["g_next"],
                         s2, out, num_warps=16, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    return out, s2, ids
