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

# {{SHARED proj}}

# {{SHARED attn}}
ATTN_BENCH = False

# {{SHARED moe}}

# ---- SGLang's own Marlin W4A16 expert path (as the stock layer runs it), selectable ----
MOE_BACKEND = "marlin"   # "marlin": SGLang's moe_wna16_marlin_gemm; "triton": this repo's kernels
try:
    import types as _types
    from sgl_kernel.scalar_type import scalar_types as _scalar_types
    from sglang.kernels.ops.activation.activation import silu_and_mul as _silu_and_mul
    from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm as _marlin_gemm
    from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
        fused_post_expert_norm_weighted_sum as _post_norm_sum)
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size as _moe_align
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace as _marlin_ws
    from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin as _marlin_prep
    _HAVE_MARLIN = True
except Exception:   # not in the image (or no CUDA): fall back to the Triton experts
    _HAVE_MARLIN = False

_MARLIN_BLOCK_M = (8, 16, 32, 48, 64)


def _marlin_bank(weights):
    device = weights["w13"].device
    layer = _types.SimpleNamespace(
        quant_config=_types.SimpleNamespace(group_size=16),
        moe_runner_config=_types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous(), w2_weight=weights["w2"].contiguous(),
        w13_weight_scale=weights["w13_sf"].contiguous(), w2_weight_scale=weights["w2_sf"].contiguous(),
        w13_weight_scale_2=weights["w13_gs"].contiguous(), w2_weight_scale_2=weights["w2_gs"].contiguous())
    _marlin_prep(layer)
    expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
    expert_map[:LOCAL] = torch.arange(LOCAL, dtype=torch.int32, device=device)
    return {"w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
            "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
            "expert_map": expert_map, "size_n": layer.intermediate_size_per_partition,
            "workspace": _marlin_ws(device, max_blocks_per_sm=4),
            "post_w": (weights["post_expert_norm"].float() + 1.0).to(torch.bfloat16).contiguous()}


def _experts_marlin(x, topk_weights, local_ids, bank):
    """x bf16 [T, 4096], local_ids int32 [T, 8] (0-31 local, else another GPU's): the weighted,
    post-normed local expert outputs, bf16 [T, 4096]."""
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    ids = bank["expert_map"][local_ids].contiguous()
    tw = topk_weights.to(torch.float32).contiguous()
    for block_size_m in _MARLIN_BLOCK_M:
        if num_tokens * TOP_K / LOCAL / block_size_m < 0.9:
            break
    sorted_ids, block_experts, n_padded = _moe_align(ids, block_size_m, N_EXPERTS)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=_scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    cache1 = _marlin_gemm(x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
                          sorted_ids, block_experts, n_padded, tw, top_k=TOP_K,
                          size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    _silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=ids.view(-1), expert_step=1)
    y = _marlin_gemm(cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
                     sorted_ids, block_experts, n_padded, tw, top_k=1,
                     size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    return _post_norm_sum(y, tw, norm_weight=bank["post_w"], eps=EPS_POST, weight_first=True, slot_expert_ids=ids)


# ----------------------------------------------------------------------------------------------
# the layer's own small kernels
# ----------------------------------------------------------------------------------------------


@triton.jit
def _boundary_kernel(stream_ptr, br_ptr, gain_ptr, alpha_ptr, beta_ptr, next_ptr, s_out_ptr, h_out_ptr,
                     stride_br, eps_post, eps_pre,
                     H: tl.constexpr, BR_N: tl.constexpr, HAS_GAIN: tl.constexpr):
    """Per row: s' = alpha * s + beta * rms(branch * gain, eps_post); h = rms(s', eps_pre) * next.
    The branch may be narrower than H (the sliced unpooler): the missing columns are zero."""
    t = tl.program_id(0)
    d = tl.arange(0, H)
    s = tl.load(stream_ptr + t * H + d)
    brm = d < BR_N
    br = tl.load(br_ptr + t * stride_br + d, mask=brm, other=0.0).to(tl.float32)
    if HAS_GAIN:
        br = br * tl.load(gain_ptr + d)
    ms = tl.sum(br * br, 0) / H
    brn = br * tl.rsqrt(ms + eps_post)
    s2 = tl.load(alpha_ptr + d) * s + tl.load(beta_ptr + d) * brn
    tl.store(s_out_ptr + t * H + d, s2)
    ms2 = tl.sum(s2 * s2, 0) / H
    h = s2 * tl.rsqrt(ms2 + eps_pre) * tl.load(next_ptr + d)
    tl.store(h_out_ptr + t * H + d, h.to(h_out_ptr.dtype.element_ty))


@triton.jit
def _rms_gain_kernel(x_ptr, gain_ptr, out_ptr, eps, N: tl.constexpr):
    t = tl.program_id(0)
    d = tl.arange(0, N)
    x = tl.load(x_ptr + t * N + d).to(tl.float32)
    ms = tl.sum(x * x, 0) / N
    y = x * tl.rsqrt(ms + eps) * tl.load(gain_ptr + d)
    tl.store(out_ptr + t * N + d, y.to(out_ptr.dtype.element_ty))


@triton.jit
def _route_topk_kernel(logit_ptr, bias_ptr, ls_ptr, ids_ptr, lids_ptr, w_ptr, mult, eps,
                       E: tl.constexpr, TOPK: tl.constexpr):
    """Sigmoid scores; the bias only chooses the top 8 (ties to the lower id); the chosen
    unbiased scores are renormalized. Also writes each choice as this GPU's local id."""
    t = tl.program_id(0)
    e = tl.arange(0, E)
    sc = tl.sigmoid(tl.load(logit_ptr + t * E + e) * mult)
    biased = sc + tl.load(bias_ptr + e)
    ls = tl.load(ls_ptr)
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
    if MOE_BACKEND == "marlin" and _HAVE_MARLIN:
        state["bank"] = _marlin_bank(weights)
        torch.cuda.synchronize()
    return state


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
    _boundary_kernel[(T,)](data["stream"], a, a, state["attn_alpha"], state["attn_beta"], state["pre_ffn_gain"],
                           stream1, h, a.stride(0), EPS_POST, EPS_RMS, H=HIDDEN, BR_N=HIDDEN, HAS_GAIN=False,
                           num_warps=8)
    # routing
    logits = _proj_forward(h, "router", N_EXPERTS, state)
    ids = torch.empty((T, TOP_K), dtype=torch.int32, device=dev)
    lids = torch.empty((T, TOP_K), dtype=torch.int32, device=dev)
    tw = torch.empty((T, TOP_K), dtype=torch.float32, device=dev)
    _route_topk_kernel[(T,)](logits, state["router_bias"], data["local_start"], ids, lids, tw,
                             ROUTER_MULTIPLIER, ROUTE_EPS, E=N_EXPERTS, TOPK=TOP_K, num_warps=1)
    # experts
    pp = _proj_forward(h, "pooler", D_EXPERT, state)
    p = torch.empty((T, D_EXPERT), dtype=ACT_TORCH, device=dev)
    _rms_gain_kernel[(T,)](pp, state["pre_expert_gain"], p, EPS_RMS, N=D_EXPERT, num_warps=4)
    if "bank" in state:
        moe = _experts_marlin(p, tw, lids, state["bank"])
    else:
        moe = _moe_forward(p, lids, tw, 0, D_EXPERT, state)
    n_un = UNPOOL_SLICE if T <= SLICED_MAX_TOKENS else HIDDEN
    f = _proj_forward(moe, "unpooler", n_un, state)
    stream2 = torch.empty((T, HIDDEN), dtype=torch.float32, device=dev)
    hidden = torch.empty((T, HIDDEN), dtype=ACT_TORCH, device=dev)
    _boundary_kernel[(T,)](stream1, f, state["post_ffn_gain"], state["ffn_alpha"], state["ffn_beta"],
                           state["next_gain"], stream2, hidden, f.stride(0), EPS_POST, EPS_RMS,
                           H=HIDDEN, BR_N=n_un, HAS_GAIN=True, num_warps=8)
    return hidden, stream2, ids
