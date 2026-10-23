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

# ---- shared: proj (muse_bf16_proj/submission.py) ----
# ----------------------------------------------------------------------------------------------
# Triton kernel
# ----------------------------------------------------------------------------------------------


@triton.jit
def _skinny_gemm_kernel(
    x_ptr, w_ptr, w2_ptr, out_ptr, part_ptr, cnt_ptr,
    M, N, K,
    stride_xm, stride_wn, stride_om,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr, TWO_W: tl.constexpr,
):
    """out[M, N] = x[M, K] @ W[N, K].T (+ x @ W2.T when TWO_W), fp32 accumulate."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    n_tiles = tl.num_programs(0)
    m_tiles = tl.num_programs(2)

    # Each split's K range is a multiple of BLOCK_K (the host checks it): say so, or the
    # pointer alignment is unknown and the loads degrade to scalar 2-byte copies (no cp.async).
    k_chunk = tl.multiple_of(K // SPLIT_K, BLOCK_K)
    k0 = tl.multiple_of(pid_k * k_chunk, BLOCK_K)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M

    # x tile [BLOCK_M, BLOCK_K] (rows past M read as zero); W tile as [BLOCK_K, BLOCK_N] so the
    # dot needs no transpose. W is [N, K] row-major: K is the contiguous axis.
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :]
    w_ptrs = w_ptr + (k0 + offs_k)[:, None] + offs_n[None, :] * stride_wn
    if TWO_W:
        w2_ptrs = w2_ptr + (k0 + offs_k)[:, None] + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, k_chunk, BLOCK_K):
        x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptrs)
        acc = tl.dot(x, w, acc)
        if TWO_W:
            w2 = tl.load(w2_ptrs)
            acc = tl.dot(x, w2, acc)
            w2_ptrs += BLOCK_K
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    if SPLIT_K == 1:
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None])
    else:
        # Partials: [SPLIT_K, m_tiles * BLOCK_M, N] fp32, whole tiles (no masking needed).
        m_pad = m_tiles * BLOCK_M
        part_ptrs = part_ptr + (pid_k * m_pad + offs_m)[:, None] * N + offs_n[None, :]
        tl.store(part_ptrs, acc)
        # All of this CTA's partial stores are issued before one thread takes the ticket
        # (release, gpu scope: the stores become visible before the increment does).
        tl.debug_barrier()
        tile = pid_m * n_tiles + pid_n
        ticket = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel", scope="gpu")
        tl.debug_barrier()
        if ticket == SPLIT_K - 1:
            # Last CTA of this tile: every other split's partial is visible (acquire). Read
            # them from L2 (.cg bypasses L1, so no stale line, and the loads still vectorize).
            total = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for s in tl.static_range(SPLIT_K):
                p = part_ptr + (s * m_pad + offs_m)[:, None] * N + offs_n[None, :]
                total += tl.load(p, cache_modifier=".cg")
            tl.store(out_ptrs, total.to(out_ptr.dtype.element_ty), mask=m_mask[:, None])
            # Reset the ticket for the next launch (graph replays reuse the counters).
            tl.atomic_xchg(cnt_ptr + tile, 0)


@triton.jit
def _skinny_gemm_tma_kernel(
    x_desc, w_desc, w2_desc, out_ptr, part_ptr, cnt_ptr,
    M, N, K, stride_om,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr, TWO_W: tl.constexpr,
):
    """Same computation, with the tiles fetched by TMA (bulk tensor copies; rows past M are
    zero-filled by the hardware)."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    n_tiles = tl.num_programs(0)
    m_tiles = tl.num_programs(2)
    k_chunk = tl.multiple_of(K // SPLIT_K, BLOCK_K)
    k0 = tl.multiple_of(pid_k * k_chunk, BLOCK_K)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(k0, k0 + k_chunk, BLOCK_K):
        x = x_desc.load([pid_m * BLOCK_M, k])
        w = w_desc.load([pid_n * BLOCK_N, k])
        acc = tl.dot(x, w.T, acc)
        if TWO_W:
            w2 = w2_desc.load([pid_n * BLOCK_N, k])
            acc = tl.dot(x, w2.T, acc)
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    if SPLIT_K == 1:
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None])
    else:
        m_pad = m_tiles * BLOCK_M
        part_ptrs = part_ptr + (pid_k * m_pad + offs_m)[:, None] * N + offs_n[None, :]
        tl.store(part_ptrs, acc)
        tl.debug_barrier()
        tile = pid_m * n_tiles + pid_n
        ticket = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel", scope="gpu")
        tl.debug_barrier()
        if ticket == SPLIT_K - 1:
            total = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for s in tl.static_range(SPLIT_K):
                p = part_ptr + (s * m_pad + offs_m)[:, None] * N + offs_n[None, :]
                total += tl.load(p, cache_modifier=".cg")
            tl.store(out_ptrs, total.to(out_ptr.dtype.element_ty), mask=m_mask[:, None])
            tl.atomic_xchg(cnt_ptr + tile, 0)


@triton.jit
def _tiled_gemm_kernel(
    x_ptr, w_ptr, w2_ptr, out_ptr, part_ptr, cnt_ptr,
    M, N, K, stride_xm, stride_om,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr, TWO_W: tl.constexpr,
):
    """W repacked in prepare() as [N/BLOCK_N, K/BLOCK_K, BLOCK_N, BLOCK_K]: every stage reads one
    contiguous tile (DRAM-friendly) instead of BLOCK_N rows that sit 2K bytes apart."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    n_tiles = tl.num_programs(0)
    m_tiles = tl.num_programs(2)
    KT = K // BLOCK_K
    kt_chunk = KT // SPLIT_K
    TILE: tl.constexpr = BLOCK_N * BLOCK_K
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    k0 = tl.multiple_of(pid_k * kt_chunk * BLOCK_K, BLOCK_K)
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :]
    tile0 = (pid_n * KT + pid_k * kt_chunk) * TILE
    w_ptrs = w_ptr + tile0 + tl.arange(0, BLOCK_N)[:, None] * BLOCK_K + offs_k[None, :]
    if TWO_W:
        w2_ptrs = w2_ptr + tile0 + tl.arange(0, BLOCK_N)[:, None] * BLOCK_K + offs_k[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, kt_chunk):
        x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptrs)
        acc = tl.dot(x, tl.trans(w), acc)
        if TWO_W:
            w2 = tl.load(w2_ptrs)
            acc = tl.dot(x, tl.trans(w2), acc)
            w2_ptrs += TILE
        x_ptrs += BLOCK_K
        w_ptrs += TILE
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    if SPLIT_K == 1:
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None])
    else:
        m_pad = m_tiles * BLOCK_M
        part_ptrs = part_ptr + (pid_k * m_pad + offs_m)[:, None] * N + offs_n[None, :]
        tl.store(part_ptrs, acc)
        tl.debug_barrier()
        tile = pid_m * n_tiles + pid_n
        ticket = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel", scope="gpu")
        tl.debug_barrier()
        if ticket == SPLIT_K - 1:
            total = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for s in tl.static_range(SPLIT_K):
                p = part_ptr + (s * m_pad + offs_m)[:, None] * N + offs_n[None, :]
                total += tl.load(p, cache_modifier=".cg")
            tl.store(out_ptrs, total.to(out_ptr.dtype.element_ty), mask=m_mask[:, None])
            tl.atomic_xchg(cnt_ptr + tile, 0)


@triton.jit
def _tiled_gemm_tma_kernel(
    x_desc, w_desc, w2_desc, out_ptr, part_ptr, cnt_ptr,
    M, N, K, stride_om,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr, TWO_W: tl.constexpr,
):
    """The tiled layout seen as a 2-D [N/BLOCK_N * K/BLOCK_K * BLOCK_N, BLOCK_K] matrix: tile
    (n, kt) is rows (n * KT + kt) * BLOCK_N onward, fetched by TMA."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    n_tiles = tl.num_programs(0)
    m_tiles = tl.num_programs(2)
    KT = K // BLOCK_K
    kt_chunk = KT // SPLIT_K
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    kt0 = pid_k * kt_chunk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kt in range(kt0, kt0 + kt_chunk):
        x = x_desc.load([pid_m * BLOCK_M, kt * BLOCK_K])
        w = w_desc.load([(pid_n * KT + kt) * BLOCK_N, 0])
        acc = tl.dot(x, w.T, acc)
        if TWO_W:
            w2 = w2_desc.load([(pid_n * KT + kt) * BLOCK_N, 0])
            acc = tl.dot(x, w2.T, acc)
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    if SPLIT_K == 1:
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None])
    else:
        m_pad = m_tiles * BLOCK_M
        part_ptrs = part_ptr + (pid_k * m_pad + offs_m)[:, None] * N + offs_n[None, :]
        tl.store(part_ptrs, acc)
        tl.debug_barrier()
        tile = pid_m * n_tiles + pid_n
        ticket = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel", scope="gpu")
        tl.debug_barrier()
        if ticket == SPLIT_K - 1:
            total = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for s in tl.static_range(SPLIT_K):
                p = part_ptr + (s * m_pad + offs_m)[:, None] * N + offs_n[None, :]
                total += tl.load(p, cache_modifier=".cg")
            tl.store(out_ptrs, total.to(out_ptr.dtype.element_ty), mask=m_mask[:, None])
            tl.atomic_xchg(cnt_ptr + tile, 0)


# ----------------------------------------------------------------------------------------------
# Configs
# ----------------------------------------------------------------------------------------------

# (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages) per (layer, N, K, m-bucket).
# Aims: >= ~256 CTAs so every SM has 2+ resident, each with ~48 KB of weight in flight, and
# as little split-K partial traffic as that allows.
MAX_TRITON_M = 64        # BF16 layers above this go to cuBLAS
COUNTERS = 8192
PARTIALS_FLOATS = 8 << 20  # 32 MB fp32 workspace


def _block_m(m: int) -> int:
    if m <= 16:
        return 16
    if m <= 32:
        return 32
    if m <= 64:
        return 64
    return 128


# Sweep table: (BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages) per (layer, n, m). The m <= 32
# shapes move the same bytes, so giving each m a different configuration measures four
# pipeline shapes per layer in one run.
SWEEP = {
    # (BLOCK_N, BLOCK_K, SPLIT_K, warps, stages); with USE_TILED the tile shape is TILE_SHAPE[layer]
    ("pooler", 4096, 1):   (32, 256, 1, 4, 4),
    ("pooler", 4096, 8):   (32, 256, 1, 4, 6),
    ("pooler", 4096, 16):  (32, 256, 1, 4, 6),
    ("pooler", 4096, 32):  (32, 256, 1, 4, 5),
    ("unpooler", 8192, 16): (32, 256, 1, 4, 6),
    ("unpooler", 8192, 32): (32, 256, 1, 4, 5),
    ("unpooler", 1024, 1): (32, 256, 4, 4, 4),
    ("unpooler", 1024, 8): (32, 256, 4, 4, 6),
    ("o", 8192, 1):   (64, 256, 1, 4, 4),
    ("o", 8192, 8):   (64, 256, 1, 4, 4),
    ("o", 8192, 16):  (64, 256, 1, 4, 4),
    ("o", 8192, 32):  (64, 256, 1, 4, 4),
    ("qkv_gate", 2304, 1):  (32, 256, 4, 4, 4),
    ("qkv_gate", 2304, 8):  (32, 256, 4, 4, 6),
    ("qkv_gate", 2304, 16): (32, 256, 2, 4, 6),
    ("qkv_gate", 2304, 32): (32, 256, 4, 4, 5),
    ("router", 256, 1):   (32, 256, 16, 4, 3),
    ("router", 256, 8):   (32, 256, 16, 4, 3),
    ("router", 256, 16):  (32, 256, 16, 4, 4),
    ("router", 256, 32):  (32, 256, 8, 4, 4),
}


# Packed tile shape per weight: (BLOCK_N, BLOCK_K). Configs for the tiled kernels use these.
TILE_SHAPE = {"qkv_gate": (32, 256), "o": (64, 256), "pooler": (32, 256), "unpooler": (32, 256), "router": (32, 256)}
USE_TILED = True
USE_TMA = False  # fetch tiles with TMA descriptors (box inner dim <= 256 elements)


def _config(layer: str, m: int, n: int, k: int):
    cfg = _config_table(layer, m, n, k)
    if cfg is None:
        return None
    if USE_TILED:
        cfg["tiled"] = True
        cfg["BLOCK_N"], cfg["BLOCK_K"] = TILE_SHAPE[layer]
        cfg["tma"] = bool(TMA_BY_M.get(m, USE_TMA))
    elif USE_TMA:
        cfg["tma"] = True
        cfg["BLOCK_K"] = min(cfg["BLOCK_K"], 256)
    return cfg


# sweep: which m-slots fetch with TMA (the m <= 32 shapes cost the same bytes)
TMA_BY_M = {1: False, 8: True, 16: False, 32: True}


def _config_table(layer: str, m: int, n: int, k: int):
    bm = _block_m(m)
    if layer == "router":
        if m > 64:
            return None   # cuBLAS, one GEMM over [hi | lo]
        bn, bk, sk, nw, ns = SWEEP.get((layer, n, m), (16, 256, 8, 4, 4))
        return dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=sk, num_warps=nw, num_stages=ns)
    if m > MAX_TRITON_M:
        return None
    default = {
        "qkv_gate": (16, 256, 1, 4, 4),
        "o": (32, 256, 1, 4, 4),
        "pooler": (32, 256, 1, 4, 4),
        "unpooler": (32, 256, 1, 4, 4) if n > 1024 else (32, 256, 4, 4, 4),
    }[layer]
    bn, bk, sk, nw, ns = SWEEP.get((layer, n, m), default)
    if bm > 32 and bk > 256:   # keep shared memory in budget for the 64-row tiles
        bk = 256
    return dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=sk, num_warps=nw, num_stages=ns)


def _usable(cfg, m: int, n: int, k: int) -> bool:
    if cfg is None:
        return False
    if n % cfg["BLOCK_N"] or k % cfg["SPLIT_K"]:
        return False
    chunk = k // cfg["SPLIT_K"]
    if chunk % cfg["BLOCK_K"]:
        return False
    if cfg.get("tiled") and (n % cfg["BLOCK_N"] or k % cfg["BLOCK_K"]):
        return False
    m_tiles = triton.cdiv(m, cfg["BLOCK_M"])
    n_tiles = n // cfg["BLOCK_N"]
    if m_tiles * n_tiles > COUNTERS:
        return False
    if cfg["SPLIT_K"] > 1 and cfg["SPLIT_K"] * m_tiles * cfg["BLOCK_M"] * n > PARTIALS_FLOATS:
        return False
    return True


# ----------------------------------------------------------------------------------------------
# prepare / kernel_fn
# ----------------------------------------------------------------------------------------------


def _pack_tiles(w: torch.Tensor, bn: int, bk: int) -> torch.Tensor:
    """[N, K] -> [N/bn * K/bk * bn, bk]: tile (n, kt) contiguous, n-major. Same values, bf16."""
    n, k = w.shape
    return w.view(n // bn, bn, k // bk, bk).permute(0, 2, 1, 3).reshape(-1, bk).contiguous()


def _proj_state(weights) -> dict:
    """Contiguous copies, the router's hi/lo split, the packed tiles and the split-K workspace."""
    state = {name: w.contiguous() for name, w in weights.items()}
    router = state["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi"] = hi.contiguous()
    state["router_lo"] = lo.contiguous()
    state["router_hilo_t"] = torch.cat((hi, lo), dim=1).t().contiguous()   # [2K, N]
    device = router.device
    for name in ("qkv_gate", "o", "pooler", "unpooler"):
        state[name + "_tiled"] = _pack_tiles(state[name], *TILE_SHAPE[name])
    state["router_hi_tiled"] = _pack_tiles(state["router_hi"], *TILE_SHAPE["router"])
    state["router_lo_tiled"] = _pack_tiles(state["router_lo"], *TILE_SHAPE["router"])
    state["counters"] = torch.zeros((COUNTERS,), dtype=torch.int32, device=device)
    state["partials"] = torch.empty((PARTIALS_FLOATS,), dtype=torch.float32, device=device)
    return state


def _proj_forward(x, layer: str, n: int, state) -> torch.Tensor:
    """x @ W[:n].T for one of the dense weights: bf16 out (fp32 for the router)."""
    if x.stride(1) != 1:
        x = x.contiguous()
    m, k = x.shape
    if layer == "router":
        cfg = _config(layer, m, n, k)
        if _usable(cfg, m, n, k):
            out = torch.empty((m, n), dtype=torch.float32, device=x.device)
            return _launch(x, state["router_hi"], state["router_lo"], out, state, cfg, m, n, k, layer)
        # one GEMM: [x | x] @ [hi | lo]^T == x @ hi^T + x @ lo^T, fp32 accumulation
        x2 = torch.cat((x, x), dim=1)
        return torch.mm(x2, state["router_hilo_t"], out_dtype=torch.float32)
    w = state[layer]
    if w.shape[0] != n:
        w = w[:n]
    cfg = _config(layer, m, n, k)
    if _usable(cfg, m, n, k):
        out = torch.empty((m, n), dtype=x.dtype, device=x.device)
        return _launch(x, w, None, out, state, cfg, m, n, k, layer)
    return torch.matmul(x, w.t())
def _launch(x, w, w2, out, state, cfg, m, n, k, layer=None):
    m_tiles = triton.cdiv(m, cfg["BLOCK_M"])
    n_tiles = n // cfg["BLOCK_N"]
    grid = (n_tiles, cfg["SPLIT_K"], m_tiles)
    if cfg.get("tiled"):
        # the packed weight: n-major tiles, so the unpooler's first 1024 rows are its first tiles
        if layer == "router":
            wt, w2t = state["router_hi_tiled"], state["router_lo_tiled"]
        else:
            wt, w2t = state[layer + "_tiled"], None
        if cfg.get("tma"):
            from triton.tools.tensor_descriptor import TensorDescriptor
            x_desc = TensorDescriptor.from_tensor(x, [cfg["BLOCK_M"], cfg["BLOCK_K"]])
            w_desc = TensorDescriptor.from_tensor(wt, [cfg["BLOCK_N"], cfg["BLOCK_K"]])
            w2_desc = TensorDescriptor.from_tensor(w2t if w2t is not None else wt, [cfg["BLOCK_N"], cfg["BLOCK_K"]])
            _tiled_gemm_tma_kernel[grid](
                x_desc, w_desc, w2_desc, out, state["partials"], state["counters"],
                m, n, k, out.stride(0),
                BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
                SPLIT_K=cfg["SPLIT_K"], TWO_W=w2t is not None,
                num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
        else:
            _tiled_gemm_kernel[grid](
                x, wt, w2t if w2t is not None else wt, out, state["partials"], state["counters"],
                m, n, k, x.stride(0), out.stride(0),
                BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
                SPLIT_K=cfg["SPLIT_K"], TWO_W=w2t is not None,
                num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
        return out
    if cfg.get("tma"):
        from triton.tools.tensor_descriptor import TensorDescriptor
        x_desc = TensorDescriptor.from_tensor(x, [cfg["BLOCK_M"], cfg["BLOCK_K"]])
        w_desc = TensorDescriptor.from_tensor(w, [cfg["BLOCK_N"], cfg["BLOCK_K"]])
        w2_desc = TensorDescriptor.from_tensor(w2 if w2 is not None else w, [cfg["BLOCK_N"], cfg["BLOCK_K"]])
        _skinny_gemm_tma_kernel[grid](
            x_desc, w_desc, w2_desc, out, state["partials"], state["counters"],
            m, n, k, out.stride(0),
            BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
            SPLIT_K=cfg["SPLIT_K"], TWO_W=w2 is not None,
            num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
        )
        return out
    _skinny_gemm_kernel[grid](
        x, w, w2 if w2 is not None else w, out, state["partials"], state["counters"],
        m, n, k,
        x.stride(0), w.stride(0), out.stride(0),
        BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
        SPLIT_K=cfg["SPLIT_K"], TWO_W=w2 is not None,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )
    return out


# ---- shared: attn (muse_attention_decode/submission.py) ----
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
    CH: tl.constexpr,
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
                             cache_modifier=".cg")                                 # [CH, RB, D]
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
        splits = max(1, min(16, 512 // ctas))
        block_t = 64 if cfg["window"] <= 2048 else 128
    else:
        splits = max(1, min(cap, 1024 // ctas))
        block_t = 128
    lb = 16
    return splits, block_t, rb, lb


def _attention_forward(data) -> torch.Tensor:
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
    ch = min(splits, 8 if rb <= 16 else 4)        # splits merged per load group
    nw = 4 if rb <= 16 else 8
    _attn_decode_kernel[grid](
        qkv, data["cache_loc"], data["seq_lens"], page_table,
        data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"],
        out, part, ml, cnt,
        page_table.shape[1], cfg["qk_scale_factor"], cfg["sm_scale"] * LOG2E,
        math.log2(theta) if theta else 0.0,
        L=L, HQ=HQ, HKV=HKV, D=D, PS=cfg["page_size"], WINDOW=cfg["window"],
        ROPE=bool(theta), SPLITS=splits, RB=rb, LB=lb, BLOCK_T=block_t, ACT=ACT_DTYPE, FP8_ASM=FP8_ASM,
        CH=ch, num_warps=nw, num_stages=3,
    )
    return out


# ---- shared: moe (muse_moe_w4a16/submission.py) ----
TOPK = 8
LOCAL = 32
POST_NORM_EPS = 1e-8
MAX_M = 1024
NUM_SMS = 132
MOE_OUT_DTYPE = torch.bfloat16
DEQUANT_ASM = True          # inline PTX for the nibble -> fp16 placement (False: plain Triton ops)
SCALE_FIX = 2.0 ** 14       # folded into the global scales: the placement trick yields value * 2^-14


# ----------------------------------------------------------------------------------------------
# 1. routing
# ----------------------------------------------------------------------------------------------


@triton.jit
def _route_kernel(
    ids_ptr, x_ptr, x16_ptr,
    sorted_tok_ptr, row_of_pair_ptr, blk_e_ptr, blk_row0_ptr, blk_end_ptr, nblk_ptr,
    M, local_start, n_elems, n_route_ctas,
    PB: tl.constexpr, CH: tl.constexpr, BM: tl.constexpr, MAXB: tl.constexpr, E: tl.constexpr,
    TOPK: tl.constexpr, XBLOCK: tl.constexpr,
):
    """Sort-free routing. CTA c < n_route_ctas owns pairs [c*PB, (c+1)*PB): each pair's rank among
    the pairs of its expert (how many earlier pairs route there) gives its padded row; every CTA
    recounts the per-expert totals (cheap), CTA 0 also writes the block list. The remaining
    CTAs convert x to fp16 (exact for BF16 values)."""
    pid = tl.program_id(0)
    if pid >= n_route_ctas:
        offs = (pid - n_route_ctas) * XBLOCK + tl.arange(0, XBLOCK)
        m = offs < n_elems
        v = tl.load(x_ptr + offs, mask=m, other=0.0)
        tl.store(x16_ptr + offs, v.to(tl.float16), mask=m)
        return
    npairs = M * TOPK
    ex = tl.arange(0, E)
    i = pid * PB + tl.arange(0, PB)
    valid = i < npairs
    e = tl.load(ids_ptr + i, mask=valid, other=0) - local_start
    is_local = valid & (e >= 0) & (e < E)
    e_key = tl.where(is_local, e, E)                       # E: not this GPU's
    counts = tl.zeros((E,), dtype=tl.int32)
    rank = tl.zeros((PB,), dtype=tl.int32)
    for c0 in range(0, npairs, CH):
        ci = c0 + tl.arange(0, CH)
        cv = ci < npairs
        ce = tl.load(ids_ptr + ci, mask=cv, other=0) - local_start
        cl = cv & (ce >= 0) & (ce < E)
        ce_key = tl.where(cl, ce, E)
        counts += tl.sum((ce_key[None, :] == ex[:, None]).to(tl.int32), 1)
        rank += tl.sum(((ce_key[None, :] == e_key[:, None]) & (ci[None, :] < i[:, None])).to(tl.int32), 1)
    padded = (counts + BM - 1) // BM * BM
    pad_off = tl.cumsum(padded, 0) - padded                 # padded row offset per expert
    onehot = (e_key[:, None] == ex[None, :]).to(tl.int32)   # [PB, E]
    row = tl.sum(onehot * pad_off[None, :], 1) + rank
    tl.store(sorted_tok_ptr + row, i // TOPK, mask=is_local)
    tl.store(row_of_pair_ptr + i, tl.where(is_local, row, -1), mask=valid)
    if pid == 0:
        nb = padded // BM
        blk_off = tl.cumsum(nb, 0) - nb
        total = tl.sum(nb, 0)
        j = tl.arange(0, MAXB)
        ej = tl.sum(((blk_off + nb)[None, :] <= j[:, None]).to(tl.int32), 1)     # expert of block j
        jm = j < total
        oh = (ej[:, None] == ex[None, :]).to(tl.int32)
        bo = tl.sum(oh * blk_off[None, :], 1)
        po = tl.sum(oh * pad_off[None, :], 1)
        ce2 = tl.sum(oh * counts[None, :], 1)
        tl.store(blk_e_ptr + j, ej + local_start, mask=jm)
        tl.store(blk_row0_ptr + j, po + (j - bo) * BM, mask=jm)
        tl.store(blk_end_ptr + j, po + ce2, mask=jm)
        tl.store(nblk_ptr, total)


# ----------------------------------------------------------------------------------------------
# 2./3. dequantizing GEMM
# ----------------------------------------------------------------------------------------------

# 4 packed bytes (8 nibbles) -> 8 fp16 (x 2^-14). Byte j holds elements 2j (low nibble) and
# 2j+1 (high nibble). Outputs, as Triton packs 4 fp16 into 2 registers in element order:
#   $0 = (e0, e2)  $1 = (e4, e6)   [low nibbles]     $2 = (e1, e3)  $3 = (e5, e7)   [high nibbles]
# prmt first spreads bytes (0, 1) and (2, 3) into the two 16-bit halves of a word, so one shift
# serves both halves: magnitude bits (e1 e0 m) land at fp16 bits [11:9], the sign at bit 15.
_NIBBLE_ASM = tl.constexpr("""
{
    .reg .b32 a, b, t, u, z;
    mov.b32 z, 0;
    prmt.b32 a, $4, z, 0x4140;
    prmt.b32 b, $4, z, 0x4342;
    shl.b32 t, a, 9;    and.b32 t, t, 0x0E000E00;
    shl.b32 u, a, 12;   and.b32 u, u, 0x80008000;  or.b32 $0, t, u;
    shl.b32 t, b, 9;    and.b32 t, t, 0x0E000E00;
    shl.b32 u, b, 12;   and.b32 u, u, 0x80008000;  or.b32 $1, t, u;
    shl.b32 t, a, 5;    and.b32 t, t, 0x0E000E00;
    shl.b32 u, a, 8;    and.b32 u, u, 0x80008000;  or.b32 $2, t, u;
    shl.b32 t, b, 5;    and.b32 t, t, 0x0E000E00;
    shl.b32 u, b, 8;    and.b32 u, u, 0x80008000;  or.b32 $3, t, u;
}
""")


@triton.jit
def _dequant(wq, sf, gs, ASM: tl.constexpr):
    """wq uint8 [BN, BK/2] packed E2M1, sf fp8 [BN, BK/16], gs fp32 (x 2^14) -> fp16 [BN, BK]."""
    if ASM:
        lo, hi = tl.inline_asm_elementwise(
            asm=_NIBBLE_ASM, constraints="=r,=r,=r,=r,r", args=[wq],
            dtype=(tl.float16, tl.float16), is_pure=True, pack=4)
    else:
        lo = (wq & 0x0F).to(tl.int16)
        hi = (wq >> 4).to(tl.int16)
        lo = (((lo & 7) << 9) | ((lo & 8) << 12)).to(tl.float16, bitcast=True)
        hi = (((hi & 7) << 9) | ((hi & 8) << 12)).to(tl.float16, bitcast=True)
    w = tl.interleave(lo, hi)                                   # [BN, BK]
    s = (sf.to(tl.float32) * gs).to(tl.float16)                 # [BN, BK/16]
    s = tl.interleave(s, s)
    s = tl.interleave(s, s)
    s = tl.interleave(s, s)
    s = tl.interleave(s, s)                                     # [BN, BK]: each scale x16
    return w * s


@triton.jit
def _moe_gemm_kernel(
    a_ptr, w_ptr, sf_ptr, gs_ptr, out_ptr,
    sorted_tok_ptr, blk_e_ptr, blk_row0_ptr, blk_end_ptr, nblk_ptr,
    stride_am, stride_om, K, N, stride_we, stride_wn, stride_se, stride_sn, up_rows,
    NUM_CTAS, n_tiles,
    GATHER: tl.constexpr, GATED: tl.constexpr, ASM: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0)
    nblk = tl.load(nblk_ptr)
    total = nblk * n_tiles
    offs_m = tl.arange(0, BM)
    offs_k = tl.arange(0, BK)
    offs_kb = tl.arange(0, BK // 2)
    offs_ks = tl.arange(0, BK // 16)
    for item in range(pid, total, NUM_CTAS):
        j = item // n_tiles
        nt = item % n_tiles
        e = tl.load(blk_e_ptr + j)
        row0 = tl.load(blk_row0_ptr + j)
        end = tl.load(blk_end_ptr + j)
        rows = row0 + offs_m
        rmask = rows < end
        if GATHER:
            arow = tl.load(sorted_tok_ptr + rows, mask=rmask, other=0)
        else:
            arow = rows
        offs_n = nt * BN + tl.arange(0, BN)
        gs = tl.load(gs_ptr + e)
        a_ptrs = a_ptr + arow[:, None] * stride_am + offs_k[None, :]
        w_ptrs = w_ptr + e * stride_we + offs_n[:, None] * stride_wn + offs_kb[None, :]
        s_ptrs = sf_ptr + e * stride_se + offs_n[:, None] * stride_sn + offs_ks[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        if GATED:
            w2_ptrs = w_ptrs + up_rows * stride_wn
            s2_ptrs = s_ptrs + up_rows * stride_sn
            acc2 = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, K, BK):
            a = tl.load(a_ptrs, mask=rmask[:, None], other=0.0)
            wq = tl.load(w_ptrs)
            sf = tl.load(s_ptrs)
            w = _dequant(wq, sf, gs, ASM)
            acc = tl.dot(a, tl.trans(w), acc)
            if GATED:
                wq2 = tl.load(w2_ptrs)
                sf2 = tl.load(s2_ptrs)
                w2 = _dequant(wq2, sf2, gs, ASM)
                acc2 = tl.dot(a, tl.trans(w2), acc2)
                w2_ptrs += BK // 2
                s2_ptrs += BK // 16
            a_ptrs += BK
            w_ptrs += BK // 2
            s_ptrs += BK // 16
        o_ptrs = out_ptr + rows[:, None] * stride_om + offs_n[None, :]
        if GATED:
            h = acc * tl.sigmoid(acc) * acc2
            # SiLU(gate) * up kept at BF16 precision (as SGLang's path does), stored in fp16
            tl.store(o_ptrs, h.to(tl.bfloat16).to(tl.float16), mask=rmask[:, None])
        else:
            tl.store(o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=rmask[:, None])


# ----------------------------------------------------------------------------------------------
# 4. post-expert norm and weighted sum
# ----------------------------------------------------------------------------------------------


@triton.jit
def _finalize_kernel(y_ptr, row_of_pair_ptr, tw_ptr, nw_ptr, out_ptr, eps,
                     H: tl.constexpr, TOPK: tl.constexpr):
    t = tl.program_id(0)
    d = tl.arange(0, H)
    nw = tl.load(nw_ptr + d)
    acc = tl.zeros((H,), dtype=tl.float32)
    for s in tl.static_range(TOPK):
        r = tl.load(row_of_pair_ptr + t * TOPK + s)
        ok = r >= 0
        y = tl.load(y_ptr + r * H + d, mask=ok & (d < H), other=0.0).to(tl.float32) * nw
        ms = tl.sum(y * y, 0) / H
        y = y * tl.rsqrt(ms + eps)
        g = tl.load(tw_ptr + t * TOPK + s)
        acc += tl.where(ok, g, 0.0) * y
    tl.store(out_ptr + t * H + d, acc.to(out_ptr.dtype.element_ty))


# ----------------------------------------------------------------------------------------------
# host side
# ----------------------------------------------------------------------------------------------


def _gemm_config(bs: int):
    """(BM, BN, BK, num_warps, num_stages) by tokens per step."""
    if bs <= 8:
        return 16, 32, 256, 4, 4
    if bs <= 64:
        return 16, 64, 128, 4, 4
    return 64, 128, 64, 4, 3


def _moe_state(weights) -> dict:
    dev = weights["w13"].device
    state = {
        "w13": weights["w13"].contiguous(), "w13_sf": weights["w13_sf"].contiguous(),
        "w2": weights["w2"].contiguous(), "w2_sf": weights["w2_sf"].contiguous(),
        # the 2^14 of the fp16 placement trick, folded in (an exact power-of-two scaling)
        "w13_gs": (weights["w13_gs"].float() * SCALE_FIX).contiguous(),
        "w2_gs": (weights["w2_gs"].float() * SCALE_FIX).contiguous(),
        "norm_w": (weights["post_norm_w"].float() + 1.0).contiguous(),
    }
    H = weights["w2"].shape[1]
    I = weights["w2"].shape[2] * 2
    NP = MAX_M * TOPK
    rows_max = NP + LOCAL * 64
    maxb = NP // 16 + LOCAL
    state.update({
        "x16": torch.empty((MAX_M, H), dtype=torch.float16, device=dev),
        "sorted_tok": torch.zeros((rows_max,), dtype=torch.int32, device=dev),
        "row_of_pair": torch.full((NP,), -1, dtype=torch.int32, device=dev),
        "blk_e": torch.zeros((maxb,), dtype=torch.int32, device=dev),
        "blk_row0": torch.zeros((maxb,), dtype=torch.int32, device=dev),
        "blk_end": torch.zeros((maxb,), dtype=torch.int32, device=dev),
        "nblk": torch.zeros((1,), dtype=torch.int32, device=dev),
        "inter": torch.empty((rows_max, I), dtype=torch.float16, device=dev),
        "y": torch.empty((rows_max, H), dtype=torch.bfloat16, device=dev),
    })
    return state


def _moe_forward(x, ids, tw, local_start: int, I: int, state) -> torch.Tensor:
    """x [M, H] (bf16), ids [M, 8] global expert ids, tw [M, 8] fp32: the local experts'
    weighted post-normed outputs summed per token, bf16 [M, H]. Experts local_start .. +31 use
    w13[0..31]."""
    M, H = x.shape
    if ids.dtype != torch.int32:
        ids = ids.to(torch.int32)
    if not ids.is_contiguous():
        ids = ids.contiguous()
    tw = tw.contiguous()
    dev = x.device
    out = torch.empty((M, H), dtype=MOE_OUT_DTYPE, device=dev)

    BM, BN, BK, nw, ns = _gemm_config(M)
    NP = max(16, triton.next_power_of_2(M * TOPK))
    MAXB = triton.next_power_of_2(NP // BM + LOCAL)
    x16 = state["x16"][:M]
    XBLOCK = 4096
    n_elems = M * H
    PB = min(NP, 64)                      # pairs per routing CTA
    CH = min(NP, 256)                     # pairs scanned per step
    n_route = triton.cdiv(M * TOPK, PB)
    grid0 = (n_route + triton.cdiv(n_elems, XBLOCK),)
    _route_kernel[grid0](
        ids, x, x16,
        state["sorted_tok"], state["row_of_pair"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
        M, local_start, n_elems, n_route,
        PB=PB, CH=CH, BM=BM, MAXB=MAXB, E=LOCAL, TOPK=TOPK, XBLOCK=XBLOCK, num_warps=4,
    )
    w13, w13_sf, w2, w2_sf = state["w13"], state["w13_sf"], state["w2"], state["w2_sf"]
    max_items = (NP // BM + LOCAL) * (max(I, H) // BN)
    num_ctas = min(2 * NUM_SMS, max_items)
    common = dict(GATHER=True, GATED=True, ASM=DEQUANT_ASM, BM=BM, BN=BN, BK=BK, num_warps=nw, num_stages=ns)
    # gate/up GEMM + SiLU: A = x16 [M, H], W13[e] rows [0, I) gate and [I, 2I) up, K = H, N = I
    _moe_gemm_kernel[(num_ctas,)](
        x16, w13, w13_sf, state["w13_gs"], state["inter"],
        state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
        x16.stride(0), state["inter"].stride(0), H, I,
        w13.stride(0), w13.stride(1), w13_sf.stride(0), w13_sf.stride(1), I,
        num_ctas, I // BN, **common,
    )
    # down GEMM: A = inter [rows, I], W2[e] [H, I/2], K = I, N = H
    common.update(GATHER=False, GATED=False)
    _moe_gemm_kernel[(num_ctas,)](
        state["inter"], w2, w2_sf, state["w2_gs"], state["y"],
        state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
        state["inter"].stride(0), state["y"].stride(0), I, H,
        w2.stride(0), w2.stride(1), w2_sf.stride(0), w2_sf.stride(1), 0,
        num_ctas, H // BN, **common,
    )
    _finalize_kernel[(M,)](
        state["y"], state["row_of_pair"], tw, state["norm_w"], out, POST_NORM_EPS,
        H=H, TOPK=TOPK, num_warps=4,
    )
    return out



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
    moe = _moe_forward(p, lids, tw, 0, D_EXPERT, state)
    n_un = UNPOOL_SLICE if T <= SLICED_MAX_TOKENS else HIDDEN
    f = _proj_forward(moe, "unpooler", n_un, state)
    stream2 = torch.empty((T, HIDDEN), dtype=torch.float32, device=dev)
    hidden = torch.empty((T, HIDDEN), dtype=ACT_TORCH, device=dev)
    _boundary_kernel[(T,)](stream1, f, state["post_ffn_gain"], state["ffn_alpha"], state["ffn_beta"],
                           state["next_gain"], stream2, hidden, f.stride(0), EPS_POST, EPS_RMS,
                           H=HIDDEN, BR_N=n_un, HAS_GAIN=True, num_warps=8)
    return hidden, stream2, ids
