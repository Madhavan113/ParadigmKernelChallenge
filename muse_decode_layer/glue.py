# okc: problem=muse_decode_layer
"""One Muse Spark Open decoder layer on one GPU (TP8/EP8), composed from this repo's kernels:

  qkv_gate GEMM (Triton, packed tiles)      -> fused attention kernel (QK-norm, RoPE, FP8 cache
  write, split-KV flash decoding, gated norm) -> o GEMM -> post-attention boundary (one row kernel:
  post-norm, residual gate, pre-FFN norm) -> router GEMM (BF16 hi+lo, fp32 accumulate) + biased
  top-8 kernel -> pooler GEMM -> pre-expert norm -> NVFP4 experts (routing sort, two dequantizing
  GEMMs, post-expert norm weighted sum) -> unpooler GEMM -> post-FFN boundary.

Built by tools/build_decode_layer.py from the shared regions of muse_bf16_proj,
muse_attention_decode and muse_moe_w4a16 (do not edit the generated submission.py by hand).
"""

import math
from typing import TypedDict

import torch
import triton
import triton.language as tl

HIDDEN = 8192
HEAD_DIM = 64
HQ, HKV = 16, 2
Q_SIZE, KV_SIZE = HQ * HEAD_DIM, HKV * HEAD_DIM
QKV_SIZE = 2 * Q_SIZE + 2 * KV_SIZE
D_EXPERT = 4096
N_EXPERTS = 256
TOP_K = 8
UNPOOL_SLICE = HIDDEN // 8
SLICED_MAX_TOKENS = 8
EPS_RMS = 1e-5
EPS_POST = 1e-8
ROUTER_MULTIPLIER = 0.1767766952966369
ROUTE_EPS = 1e-15
RESIDUAL_TEMPERATURE = 0.3
RESIDUAL_GATE_EPS = 1e-3
ACT_TORCH = torch.bfloat16   # activations between the kernels (BF16, as SGLang keeps them)
MAX_TOKENS = 256             # 32 verify requests x 8 queries
ROW_SPLIT = False            # split the row kernels over several programs with a per-row ticket: measured on the
                             # judge (G=8/4): boundary 5.5 -> 4.7 us per call but rms_gain 2.3 -> 4.0 us; the two
                             # atomic round trips cost what the split saves, so rows stay single-program

BENCH = False                # the shared regions' prepare-time microbenchmarks stay off in the layer

# {{SHARED proj}}

# {{SHARED attn}}
ATTN_BENCH = False

# {{SHARED moe}}

# ----------------------------------------------------------------------------------------------
# the layer's own small kernels
# ----------------------------------------------------------------------------------------------


@triton.jit
def _boundary_kernel(stream_ptr, br_ptr, gain_ptr, alpha_ptr, beta_ptr, next_ptr, s_out_ptr, h_out_ptr,
                     part_ptr, cnt_ptr, stride_br, eps_post, eps_pre,
                     H: tl.constexpr, BR_N: tl.constexpr, HAS_GAIN: tl.constexpr, G: tl.constexpr):
    """Per row: s' = alpha * s + beta * rms(branch * gain, eps_post); h = rms(s', eps_pre) * next.
    The branch may be narrower than H (the sliced unpooler): the missing columns are zero.
    A row is split over G programs (adjacent program ids): each reads its H/G slice once, publishes
    four partial sums (sum br^2, sum (alpha s)^2, sum (alpha s)(beta br), sum (beta br)^2), waits on
    the row's ticket until all G have published, then finishes its slice. The second mean square is
    sum (alpha s + r beta br)^2 = A + 2 r B + r^2 C with r = rsqrt(mean br^2 + eps_post), so one
    pass over the row suffices. The counters only grow; a program's target is the next multiple of G."""
    pid = tl.program_id(0)
    t = pid // G
    g = pid % G
    C: tl.constexpr = H // G
    d = g * C + tl.arange(0, C)
    s = tl.load(stream_ptr + t * H + d)
    brm = d < BR_N
    br = tl.load(br_ptr + t * stride_br + d, mask=brm, other=0.0).to(tl.float32)
    if HAS_GAIN:
        br = br * tl.load(gain_ptr + d)
    sa = tl.load(alpha_ptr + d) * s
    bb = tl.load(beta_ptr + d) * br
    pd_ = tl.sum(br * br, 0)
    pa = tl.sum(sa * sa, 0)
    pb = tl.sum(sa * bb, 0)
    pc = tl.sum(bb * bb, 0)
    if G == 1:
        td = pd_
        ta = pa
        tb = pb
        tc = pc
    else:
        base = part_ptr + (t * G + g) * 4
        tl.store(base + 0, pd_)
        tl.store(base + 1, pa)
        tl.store(base + 2, pb)
        tl.store(base + 3, pc)
        tl.debug_barrier()
        old = tl.atomic_add(cnt_ptr + t, 1, sem="release")
        target = (old // G + 1) * G
        cur = tl.atomic_add(cnt_ptr + t, 0, sem="acquire")
        while cur < target:
            cur = tl.atomic_add(cnt_ptr + t, 0, sem="acquire")
        gi = tl.arange(0, G)
        pbase = part_ptr + (t * G + gi) * 4
        td = tl.sum(tl.load(pbase + 0, cache_modifier=".cg"), 0)
        ta = tl.sum(tl.load(pbase + 1, cache_modifier=".cg"), 0)
        tb = tl.sum(tl.load(pbase + 2, cache_modifier=".cg"), 0)
        tc = tl.sum(tl.load(pbase + 3, cache_modifier=".cg"), 0)
    r = tl.rsqrt(td / H + eps_post)
    s2 = sa + r * bb
    tl.store(s_out_ptr + t * H + d, s2)
    ms2 = (ta + 2.0 * r * tb + r * r * tc) / H
    h = s2 * tl.rsqrt(ms2 + eps_pre) * tl.load(next_ptr + d)
    tl.store(h_out_ptr + t * H + d, h.to(h_out_ptr.dtype.element_ty))


@triton.jit
def _rms_gain_kernel(x_ptr, gain_ptr, out_ptr, part_ptr, cnt_ptr, eps, N: tl.constexpr, G: tl.constexpr):
    """y = rms(x, eps) * gain; each row split over G adjacent programs (ticket as in _boundary_kernel)."""
    pid = tl.program_id(0)
    t = pid // G
    g = pid % G
    C: tl.constexpr = N // G
    d = g * C + tl.arange(0, C)
    x = tl.load(x_ptr + t * N + d).to(tl.float32)
    ps = tl.sum(x * x, 0)
    if G == 1:
        ms = ps / N
    else:
        tl.store(part_ptr + t * G + g, ps)
        tl.debug_barrier()
        old = tl.atomic_add(cnt_ptr + t, 1, sem="release")
        target = (old // G + 1) * G
        cur = tl.atomic_add(cnt_ptr + t, 0, sem="acquire")
        while cur < target:
            cur = tl.atomic_add(cnt_ptr + t, 0, sem="acquire")
        gi = tl.arange(0, G)
        ms = tl.sum(tl.load(part_ptr + t * G + gi, cache_modifier=".cg"), 0) / N
    y = x * tl.rsqrt(ms + eps) * tl.load(gain_ptr + d)
    tl.store(out_ptr + t * N + d, y.to(out_ptr.dtype.element_ty))


@triton.jit
def _route_rms_kernel(logit_ptr, bias_ptr, ls_ptr, ids_ptr, lids_ptr, w_ptr, x_ptr, gain_ptr, out_ptr,
                      mult, eps, eps_n, E: tl.constexpr, TOPK: tl.constexpr, N: tl.constexpr):
    """One launch for two independent row jobs (both only need the router and pooler GEMM outputs):
    (a) routing: sigmoid scores; the bias only chooses the top 8 (ties to the lower id); the chosen
    unbiased scores are renormalized; each choice also written as this GPU's local id;
    (b) pre-expert norm: out = rms(x, eps_n) * gain."""
    t = tl.program_id(0)
    # (b) first: its loads are the larger ones, issue them before the routing's dependent chain
    d = tl.arange(0, N)
    x = tl.load(x_ptr + t * N + d).to(tl.float32)
    e = tl.arange(0, E)
    sc = tl.sigmoid(tl.load(logit_ptr + t * E + e) * mult)
    biased = sc + tl.load(bias_ptr + e)
    ls = tl.load(ls_ptr)
    ms = tl.sum(x * x, 0) / N
    y = x * tl.rsqrt(ms + eps_n) * tl.load(gain_ptr + d)
    tl.store(out_ptr + t * N + d, y.to(out_ptr.dtype.element_ty))
    rank = tl.full((E,), -1, tl.int32)
    for s in tl.static_range(TOPK):
        idx = tl.argmax(biased, 0)
        rank = tl.where(e == idx, s, rank)
        biased = tl.where(e == idx, -1.0e30, biased)
        tl.store(ids_ptr + t * TOPK + s, idx.to(tl.int32))
        tl.store(lids_ptr + t * TOPK + s, (((idx - ls) % E + E) % E).to(tl.int32))
    total = tl.sum(tl.where(rank >= 0, sc, 0.0), 0)
    for s in tl.static_range(TOPK):
        val = tl.sum(tl.where(rank == s, sc, 0.0), 0)
        tl.store(w_ptr + t * TOPK + s, val / (total + eps))


# ----------------------------------------------------------------------------------------------
# prepare / kernel_fn
# ----------------------------------------------------------------------------------------------


def _residual_coefficients(gate):
    w = gate.float() / RESIDUAL_TEMPERATURE
    beta = torch.sigmoid(w)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=RESIDUAL_GATE_EPS))
    return alpha.contiguous(), beta.contiguous()


def prepare(weights):
    state = _proj_state({k: weights[k] for k in ("qkv_gate", "o", "pooler", "unpooler", "router")})
    state.update(_moe_state({"w13": weights["w13"], "w13_sf": weights["w13_sf"], "w13_gs": weights["w13_gs"],
                             "w2": weights["w2"], "w2_sf": weights["w2_sf"], "w2_gs": weights["w2_gs"],
                             "post_norm_w": weights["post_expert_norm"]}))
    state["attn_alpha"], state["attn_beta"] = _residual_coefficients(weights["attn_residual_gate"])
    state["ffn_alpha"], state["ffn_beta"] = _residual_coefficients(weights["ffn_residual_gate"])
    for name, key in (("pre_ffn_norm", "pre_ffn_gain"), ("post_ffn_norm", "post_ffn_gain"),
                      ("next_pre_attn_norm", "next_gain"), ("pre_expert_norm", "pre_expert_gain")):
        state[key] = (1.0 + weights[name].float()).contiguous()
    state["router_bias"] = weights["router_bias"].float().contiguous()
    dev = weights["router_bias"].device
    # row-kernel tickets (one slot per launch site): partial sums + counters that only grow
    state["row_part"] = torch.zeros((3, MAX_TOKENS * 8 * 4), dtype=torch.float32, device=dev)
    state["row_cnt"] = torch.zeros((3, MAX_TOKENS), dtype=torch.int32, device=dev)
    return state


def _row_split(T, n):
    """Programs per row of an n-wide row kernel: more at small token counts (one program per row
    is latency-bound: ~5.5 us for a 8192-wide boundary row), fewer when rows already fill the GPU."""
    if not ROW_SPLIT:
        return 1
    if n >= 8192:
        return 8 if T <= 64 else 4
    return 4 if T <= 64 else 2


def kernel_fn(data, state):
    cfg = data["config"]
    B, L, T = cfg["batch"], cfg["q_len"], cfg["tokens"]
    x = data["attn_input"]
    dev = x.device
    # attention
    qkv = _proj_forward(x, "qkv_gate", QKV_SIZE, state)
    attn = _attention_forward({
        "qkv": qkv.view(B, L, QKV_SIZE), "cache_loc": data["cache_loc"], "seq_lens": data["seq_lens"],
        "page_table": data["page_table"], "k_cache": data["k_cache"], "v_cache": data["v_cache"],
        "k_scale": data["k_scale"], "v_scale": data["v_scale"], "config": cfg})
    a = _proj_forward(attn.view(T, Q_SIZE), "o", HIDDEN, state)
    stream1 = torch.empty((T, HIDDEN), dtype=torch.float32, device=dev)
    h = torch.empty((T, HIDDEN), dtype=ACT_TORCH, device=dev)
    g = _row_split(T, HIDDEN)
    _boundary_kernel[(T * g,)](data["stream"], a, a, state["attn_alpha"], state["attn_beta"], state["pre_ffn_gain"],
                               stream1, h, state["row_part"][0], state["row_cnt"][0], a.stride(0), EPS_POST,
                               EPS_RMS, H=HIDDEN, BR_N=HIDDEN, HAS_GAIN=False, G=g, num_warps=4 if g > 1 else 8)
    # routing + pre-expert norm (one launch; both depend only on h)
    logits = _proj_forward(h, "router", N_EXPERTS, state)
    pp = _proj_forward(h, "pooler", D_EXPERT, state)
    ids = torch.empty((T, TOP_K), dtype=torch.int32, device=dev)
    lids = torch.empty((T, TOP_K), dtype=torch.int32, device=dev)
    tw = torch.empty((T, TOP_K), dtype=torch.float32, device=dev)
    p = torch.empty((T, D_EXPERT), dtype=ACT_TORCH, device=dev)
    _route_rms_kernel[(T,)](logits, state["router_bias"], data["local_start"], ids, lids, tw,
                            pp, state["pre_expert_gain"], p, ROUTER_MULTIPLIER, ROUTE_EPS, EPS_RMS,
                            E=N_EXPERTS, TOPK=TOP_K, N=D_EXPERT, num_warps=4)
    moe = _moe_forward(p, lids, tw, 0, D_EXPERT, state)   # Marlin GEMMs with this file's routing (shared moe region)
    n_un = UNPOOL_SLICE if T <= SLICED_MAX_TOKENS else HIDDEN
    f = _proj_forward(moe, "unpooler", n_un, state)
    stream2 = torch.empty((T, HIDDEN), dtype=torch.float32, device=dev)
    hidden = torch.empty((T, HIDDEN), dtype=ACT_TORCH, device=dev)
    g = _row_split(T, HIDDEN)
    _boundary_kernel[(T * g,)](stream1, f, state["post_ffn_gain"], state["ffn_alpha"], state["ffn_beta"],
                               state["next_gain"], stream2, hidden, state["row_part"][2], state["row_cnt"][2],
                               f.stride(0), EPS_POST, EPS_RMS, H=HIDDEN, BR_N=n_un, HAS_GAIN=True, G=g,
                               num_warps=4 if g > 1 else 8)
    return hidden, stream2, ids
