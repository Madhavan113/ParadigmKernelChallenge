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
    SPLIT_K: tl.constexpr, TWO_W: tl.constexpr, SWAP_MN: tl.constexpr,
):
    """out[M, N] = x[M, K] @ W[N, K].T (+ x @ W2.T when TWO_W), fp32 accumulate."""
    pid_k = tl.program_id(1)
    if SWAP_MN:  # grid (m_tiles, SPLIT_K, n_tiles): the m-tiles of one weight slab run back to back
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(2)
        m_tiles = tl.num_programs(0)
        n_tiles = tl.num_programs(2)
    else:
        pid_n = tl.program_id(0)
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
    SPLIT_K: tl.constexpr, TWO_W: tl.constexpr, SWAP_MN: tl.constexpr,
):
    """Same computation, with the tiles fetched by TMA (bulk tensor copies; rows past M are
    zero-filled by the hardware)."""
    pid_k = tl.program_id(1)
    if SWAP_MN:  # grid (m_tiles, SPLIT_K, n_tiles): the m-tiles of one weight slab run back to back
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(2)
        m_tiles = tl.num_programs(0)
        n_tiles = tl.num_programs(2)
    else:
        pid_n = tl.program_id(0)
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
    SPLIT_K: tl.constexpr, TWO_W: tl.constexpr, SWAP_MN: tl.constexpr,
):
    """W repacked in prepare() as [N/BLOCK_N, K/BLOCK_K, BLOCK_N, BLOCK_K]: every stage reads one
    contiguous tile (DRAM-friendly) instead of BLOCK_N rows that sit 2K bytes apart."""
    pid_k = tl.program_id(1)
    if SWAP_MN:  # grid (m_tiles, SPLIT_K, n_tiles): the m-tiles of one weight slab run back to back
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(2)
        m_tiles = tl.num_programs(0)
        n_tiles = tl.num_programs(2)
    else:
        pid_n = tl.program_id(0)
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
    SPLIT_K: tl.constexpr, TWO_W: tl.constexpr, SWAP_MN: tl.constexpr,
):
    """The tiled layout seen as a 2-D [N/BLOCK_N * K/BLOCK_K * BLOCK_N, BLOCK_K] matrix: tile
    (n, kt) is rows (n * KT + kt) * BLOCK_N onward, fetched by TMA."""
    pid_k = tl.program_id(1)
    if SWAP_MN:  # grid (m_tiles, SPLIT_K, n_tiles): the m-tiles of one weight slab run back to back
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(2)
        m_tiles = tl.num_programs(0)
        n_tiles = tl.num_programs(2)
    else:
        pid_n = tl.program_id(0)
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


@triton.jit
def _tiled_gemv_kernel(
    x_ptr, w_ptr, w2_ptr, out_ptr, part_ptr, cnt_ptr,
    N, K,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr, TWO_W: tl.constexpr,
):
    """M == 1: out[n] = sum_k W[n, k] x[k] on the FMA pipe (no tensor cores, no operand layout
    conversions), over the packed [N/BLOCK_N, K/BLOCK_K, BLOCK_N, BLOCK_K] tiles."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    n_tiles = tl.num_programs(0)
    KT = K // BLOCK_K
    kt_chunk = KT // SPLIT_K
    TILE: tl.constexpr = BLOCK_N * BLOCK_K
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    k0 = tl.multiple_of(pid_k * kt_chunk * BLOCK_K, BLOCK_K)
    x_ptrs = x_ptr + k0 + offs_k
    tile0 = (pid_n * KT + pid_k * kt_chunk) * TILE
    w_ptrs = w_ptr + tile0 + tl.arange(0, BLOCK_N)[:, None] * BLOCK_K + offs_k[None, :]
    if TWO_W:
        w2_ptrs = w2_ptr + tile0 + tl.arange(0, BLOCK_N)[:, None] * BLOCK_K + offs_k[None, :]
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for _ in range(0, kt_chunk):
        x = tl.load(x_ptrs).to(tl.float32)
        w = tl.load(w_ptrs).to(tl.float32)
        acc += tl.sum(w * x[None, :], 1)
        if TWO_W:
            w2 = tl.load(w2_ptrs).to(tl.float32)
            acc += tl.sum(w2 * x[None, :], 1)
            w2_ptrs += TILE
        x_ptrs += BLOCK_K
        w_ptrs += TILE
    out_ptrs = out_ptr + offs_n
    if SPLIT_K == 1:
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty))
    else:
        tl.store(part_ptr + pid_k * N + offs_n, acc)
        tl.debug_barrier()
        ticket = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
        tl.debug_barrier()
        if ticket == SPLIT_K - 1:
            total = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for s in tl.static_range(SPLIT_K):
                total += tl.load(part_ptr + s * N + offs_n, cache_modifier=".cg")
            tl.store(out_ptrs, total.to(out_ptr.dtype.element_ty))
            tl.atomic_xchg(cnt_ptr + pid_n, 0)


@triton.jit
def _sum_partials_kernel(part_ptr, out_ptr, numel, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    """out[i] = sum_s part[s, i] (fp32 partials of a batched split-K GEMM), cast to the output dtype."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in tl.static_range(SPLIT_K):
        acc += tl.load(part_ptr + s * numel + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _dup_cols_kernel(x_ptr, out_ptr, K, BLOCK: tl.constexpr):
    """out[i, :] = [x[i, :] | x[i, :]] (the router's one-GEMM input over [hi | lo]); K % BLOCK == 0."""
    pid_m = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + pid_m * K + offs)
    tl.store(out_ptr + pid_m * 2 * K + offs, v)
    tl.store(out_ptr + pid_m * 2 * K + K + offs, v)


# == BEGIN CUDA ==
PROJ_CUDA_SRC = r"""
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ uint4 ldg_stream(const uint4* p) {
    uint4 v;
    asm("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ float bf_lo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bf_hi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ void dot8(float& acc, const uint4& w, const uint4& x) {
    acc = fmaf(bf_lo(w.x), bf_lo(x.x), acc); acc = fmaf(bf_hi(w.x), bf_hi(x.x), acc);
    acc = fmaf(bf_lo(w.y), bf_lo(x.y), acc); acc = fmaf(bf_hi(w.y), bf_hi(x.y), acc);
    acc = fmaf(bf_lo(w.z), bf_lo(x.z), acc); acc = fmaf(bf_hi(w.z), bf_hi(x.z), acc);
    acc = fmaf(bf_lo(w.w), bf_lo(x.w), acc); acc = fmaf(bf_hi(w.w), bf_hi(x.w), acc);
}
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// ---- pure streaming read: each CTA reads a contiguous chunk, UNROLL 16-byte loads in flight per thread
template <int UNROLL, int THREADS>
__global__ void __launch_bounds__(THREADS) stream_kernel(const uint4* __restrict__ p, long n_vec, uint32_t* __restrict__ out)
{
    const long chunk = n_vec / gridDim.x;
    const uint4* base = p + (long)blockIdx.x * chunk;
    constexpr long BATCH = (long)UNROLL * THREADS;
    const long n_iter = (chunk + BATCH - 1) / BATCH;   // the tail re-reads the chunk's last vector
    uint32_t acc = 0;
    // register double-buffering: the next batch's UNROLL loads are issued before this batch is
    // consumed, so UNROLL loads per thread stay in flight (ptxas cannot sink them past the loop)
    uint4 cur[UNROLL];
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) cur[u] = ldg_stream(base + min((long)threadIdx.x + u * THREADS, chunk - 1));
    for (long it = 0; it < n_iter; ++it) {
        const long nxt_it = it + 1 < n_iter ? it + 1 : it;
        const long i0 = nxt_it * BATCH + threadIdx.x;
        uint4 nxt[UNROLL];
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) nxt[u] = ldg_stream(base + min(i0 + u * THREADS, chunk - 1));
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) acc ^= cur[u].x ^ cur[u].y ^ cur[u].z ^ cur[u].w;
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) cur[u] = nxt[u];
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc ^= __shfl_xor_sync(0xffffffffu, acc, o);
    if ((threadIdx.x & 31) == 0) out[blockIdx.x * (THREADS / 32) + (threadIdx.x >> 5)] = acc;
}

// ---- register GEMV. 8 warps per CTA; KSPLIT warps share RPW rows (each takes a K part);
// a warp step covers 32 lanes x 8 elements = 256 of K; UNROLL steps are in flight per lane.
template <int M, int RPW, int UNROLL, int KSPLIT, bool TWO_W>
__global__ void __launch_bounds__(256) gemv_kernel(const __nv_bfloat16* __restrict__ X, int ldx, int m,
                                                   const __nv_bfloat16* __restrict__ W, const __nv_bfloat16* __restrict__ W2, int ldw,
                                                   __nv_bfloat16* __restrict__ out_bf, float* __restrict__ out_f, int ldo, int K)
{
    constexpr int WARPS = 8, GROUPS = WARPS / KSPLIT, ROWS = GROUPS * RPW;
    constexpr int NW = TWO_W ? 2 : 1;
    __shared__ float red[GROUPS][RPW][M];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int grp = warp / KSPLIT, kp = warp % KSPLIT;
    const int n0 = blockIdx.x * ROWS + grp * RPW;
    const int kc = K / KSPLIT;
    const int steps = kc / 256;
    const int kbeg = kp * kc;

    float acc[RPW][M];
    #pragma unroll
    for (int r = 0; r < RPW; ++r)
        #pragma unroll
        for (int i = 0; i < M; ++i) acc[r][i] = 0.f;

    const uint4* wp[NW][RPW];
    #pragma unroll
    for (int r = 0; r < RPW; ++r) {
        wp[0][r] = reinterpret_cast<const uint4*>(W + (size_t)(n0 + r) * ldw + kbeg) + lane;
        if (TWO_W) wp[NW - 1][r] = reinterpret_cast<const uint4*>(W2 + (size_t)(n0 + r) * ldw + kbeg) + lane;
    }
    const uint4* xp[M];
    #pragma unroll
    for (int i = 0; i < M; ++i) {
        const int row = i < m ? i : m - 1;
        xp[i] = reinterpret_cast<const uint4*>(X + (size_t)row * ldx + kbeg) + lane;
    }

    // register double-buffering of the weight loads (see stream_kernel)
    uint4 cur[NW][UNROLL][RPW];
    #pragma unroll
    for (int w = 0; w < NW; ++w)
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u)
            #pragma unroll
            for (int r = 0; r < RPW; ++r) cur[w][u][r] = ldg_stream(wp[w][r] + u * 32);
    for (int s = 0; s < steps; s += UNROLL) {
        const int sn = s + UNROLL < steps ? s + UNROLL : s;
        uint4 nxt[NW][UNROLL][RPW];
        #pragma unroll
        for (int w = 0; w < NW; ++w)
            #pragma unroll
            for (int u = 0; u < UNROLL; ++u)
                #pragma unroll
                for (int r = 0; r < RPW; ++r) nxt[w][u][r] = ldg_stream(wp[w][r] + (sn + u) * 32);
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            uint4 xv[M];
            #pragma unroll
            for (int i = 0; i < M; ++i) xv[i] = xp[i][(s + u) * 32];
            #pragma unroll
            for (int w = 0; w < NW; ++w)
                #pragma unroll
                for (int r = 0; r < RPW; ++r)
                    #pragma unroll
                    for (int i = 0; i < M; ++i) dot8(acc[r][i], cur[w][u][r], xv[i]);
        }
        #pragma unroll
        for (int w = 0; w < NW; ++w)
            #pragma unroll
            for (int u = 0; u < UNROLL; ++u)
                #pragma unroll
                for (int r = 0; r < RPW; ++r) cur[w][u][r] = nxt[w][u][r];
    }

    #pragma unroll
    for (int r = 0; r < RPW; ++r)
        #pragma unroll
        for (int i = 0; i < M; ++i) acc[r][i] = warp_sum(acc[r][i]);

    if (KSPLIT == 1) {
        if (lane == 0) {
            #pragma unroll
            for (int r = 0; r < RPW; ++r)
                #pragma unroll
                for (int i = 0; i < M; ++i)
                    if (i < m) {
                        if (out_f) out_f[(size_t)i * ldo + n0 + r] = acc[r][i];
                        else out_bf[(size_t)i * ldo + n0 + r] = __float2bfloat16(acc[r][i]);
                    }
        }
        return;
    }
    if (threadIdx.x < GROUPS * RPW * M) (&red[0][0][0])[threadIdx.x] = 0.f;
    __syncthreads();
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < RPW; ++r)
            #pragma unroll
            for (int i = 0; i < M; ++i) atomicAdd(&red[grp][r][i], acc[r][i]);
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < GROUPS * RPW * M; idx += 256) {
        const int i = idx % M, r = (idx / M) % RPW, g = idx / (M * RPW);
        if (i >= m) continue;
        const float v = red[g][r][i];
        const int n = blockIdx.x * ROWS + g * RPW + r;
        if (out_f) out_f[(size_t)i * ldo + n] = v;
        else out_bf[(size_t)i * ldo + n] = __float2bfloat16(v);
    }
}

template <int M, int RPW, int UNROLL, int KSPLIT, bool TWO_W>
void launch_gemv(const __nv_bfloat16* X, int ldx, int m, const __nv_bfloat16* W, const __nv_bfloat16* W2, int ldw,
                 __nv_bfloat16* ob, float* of, int ldo, int K, int N, cudaStream_t stream)
{
    constexpr int ROWS = (8 / KSPLIT) * RPW;
    gemv_kernel<M, RPW, UNROLL, KSPLIT, TWO_W><<<N / ROWS, 256, 0, stream>>>(X, ldx, m, W, W2, ldw, ob, of, ldo, K);
}

__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ---- tensor-core skinny GEMM, M <= 16 * MT, no shared-memory staging of the weight: a lane's
// 16-byte load of W (row n0 + 8r + g, k = 8t .. 8t+7 of a 32-wide k chunk) is the B fragment of two
// mma.m16n8k16 steps under a k permutation, and x's A fragments are loaded with the same
// permutation (two 16-byte loads per m-tile from L1/L2). KSPLIT warps share RPW 8-row tiles and
// reduce through shared memory; UNROLL chunks of W per lane stay in flight (register double-buffer).
template <int MT, int RPW, int UNROLL, int KSPLIT>
__global__ void __launch_bounds__(256) skinny_mma_kernel(const __nv_bfloat16* __restrict__ X, int ldx, int m,
                                                         const __nv_bfloat16* __restrict__ W, int ldw,
                                                         __nv_bfloat16* __restrict__ out, int ldo, int K)
{
    constexpr int WARPS = 8, GROUPS = WARPS / KSPLIT, TROWS = RPW * 8, ROWS = GROUPS * TROWS;
    __shared__ float red[KSPLIT > 1 ? GROUPS : 1][MT * 16][TROWS];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int grp = warp / KSPLIT, kp = warp % KSPLIT;
    const int n0 = blockIdx.x * ROWS + grp * TROWS;
    const int kc = K / KSPLIT, steps = kc / 32, kbeg = kp * kc;

    float acc[MT][RPW][4];
    #pragma unroll
    for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int r = 0; r < RPW; ++r)
            #pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][r][q] = 0.f;

    const uint4* wp[RPW];
    #pragma unroll
    for (int r = 0; r < RPW; ++r) wp[r] = reinterpret_cast<const uint4*>(W + (size_t)(n0 + 8 * r + g) * ldw + kbeg + 8 * t);
    const uint4* xp[MT][2];
    bool xok[MT][2];
    #pragma unroll
    for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = i * 16 + g + 8 * h;
            xok[i][h] = row < m;
            xp[i][h] = reinterpret_cast<const uint4*>(X + (size_t)(row < m ? row : 0) * ldx + kbeg + 8 * t);
        }

    uint4 cur[UNROLL][RPW];
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u)
        #pragma unroll
        for (int r = 0; r < RPW; ++r) cur[u][r] = ldg_stream(wp[r] + 4 * u);
    for (int s = 0; s < steps; s += UNROLL) {
        const int sn = s + UNROLL < steps ? s + UNROLL : s;
        uint4 nxt[UNROLL][RPW];
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u)
            #pragma unroll
            for (int r = 0; r < RPW; ++r) nxt[u][r] = ldg_stream(wp[r] + 4 * (sn + u));
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            uint4 xa[MT], xb[MT];
            #pragma unroll
            for (int i = 0; i < MT; ++i) {
                xa[i] = xok[i][0] ? xp[i][0][4 * (s + u)] : make_uint4(0u, 0u, 0u, 0u);
                xb[i] = xok[i][1] ? xp[i][1][4 * (s + u)] : make_uint4(0u, 0u, 0u, 0u);
            }
            #pragma unroll
            for (int r = 0; r < RPW; ++r)
                #pragma unroll
                for (int i = 0; i < MT; ++i) {
                    mma16816(acc[i][r], xa[i].x, xb[i].x, xa[i].y, xb[i].y, cur[u][r].x, cur[u][r].y);
                    mma16816(acc[i][r], xa[i].z, xb[i].z, xa[i].w, xb[i].w, cur[u][r].z, cur[u][r].w);
                }
        }
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u)
            #pragma unroll
            for (int r = 0; r < RPW; ++r) cur[u][r] = nxt[u][r];
    }

    if (KSPLIT == 1) {
        #pragma unroll
        for (int i = 0; i < MT; ++i)
            #pragma unroll
            for (int r = 0; r < RPW; ++r) {
                const int col = n0 + 8 * r + 2 * t;
                const int ra = i * 16 + g, rb = ra + 8;
                if (ra < m) *reinterpret_cast<__nv_bfloat162*>(out + (size_t)ra * ldo + col) = __floats2bfloat162_rn(acc[i][r][0], acc[i][r][1]);
                if (rb < m) *reinterpret_cast<__nv_bfloat162*>(out + (size_t)rb * ldo + col) = __floats2bfloat162_rn(acc[i][r][2], acc[i][r][3]);
            }
        return;
    }
    constexpr int RED = GROUPS * MT * 16 * TROWS;
    for (int idx = threadIdx.x; idx < RED; idx += 256) (&red[0][0][0])[idx] = 0.f;
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const int col = 8 * r + 2 * t;
            atomicAdd(&red[grp][i * 16 + g][col], acc[i][r][0]);
            atomicAdd(&red[grp][i * 16 + g][col + 1], acc[i][r][1]);
            atomicAdd(&red[grp][i * 16 + g + 8][col], acc[i][r][2]);
            atomicAdd(&red[grp][i * 16 + g + 8][col + 1], acc[i][r][3]);
        }
    __syncthreads();
    for (int idx = threadIdx.x; idx < RED / 2; idx += 256) {
        const int c2 = idx % (TROWS / 2), row = (idx / (TROWS / 2)) % (MT * 16), gq = idx / (TROWS / 2 * MT * 16);
        if (row >= m) continue;
        const float* v = &red[gq][row][2 * c2];
        *reinterpret_cast<__nv_bfloat162*>(out + (size_t)row * ldo + blockIdx.x * ROWS + gq * TROWS + 2 * c2) = __floats2bfloat162_rn(v[0], v[1]);
    }
}

template <int MT, int RPW, int UNROLL, int KSPLIT>
void launch_mma(const __nv_bfloat16* X, int ldx, int m, const __nv_bfloat16* W, int ldw, __nv_bfloat16* out, int ldo, int K, int N, cudaStream_t stream)
{
    constexpr int ROWS = (8 / KSPLIT) * RPW * 8;
    skinny_mma_kernel<MT, RPW, UNROLL, KSPLIT><<<N / ROWS, 256, 0, stream>>>(X, ldx, m, W, ldw, out, ldo, K);
}

} // namespace

void stream_bytes(torch::Tensor buf, torch::Tensor out, int64_t grid, int64_t threads, int64_t unroll)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    const uint4* p = reinterpret_cast<const uint4*>(buf.data_ptr());
    const long n_vec = (long)(buf.numel() * buf.element_size() / 16);
    uint32_t* o = reinterpret_cast<uint32_t*>(out.data_ptr());
    const int key = (int)threads * 100 + (int)unroll;
    switch (key) {
        case 25604: stream_kernel<4, 256><<<grid, 256, 0, stream>>>(p, n_vec, o); break;
        case 25608: stream_kernel<8, 256><<<grid, 256, 0, stream>>>(p, n_vec, o); break;
        case 25616: stream_kernel<16, 256><<<grid, 256, 0, stream>>>(p, n_vec, o); break;
        case 51204: stream_kernel<4, 512><<<grid, 512, 0, stream>>>(p, n_vec, o); break;
        case 51208: stream_kernel<8, 512><<<grid, 512, 0, stream>>>(p, n_vec, o); break;
        case 12808: stream_kernel<8, 128><<<grid, 128, 0, stream>>>(p, n_vec, o); break;
        case 12816: stream_kernel<16, 128><<<grid, 128, 0, stream>>>(p, n_vec, o); break;
        default: TORCH_CHECK(false, "unsupported stream config");
    }
}

#define GEMV_CASE(M_, R_, U_, S_, T_) \
    case (M_ * 100000 + R_ * 10000 + U_ * 100 + S_ * 10 + (T_ ? 1 : 0)): \
        launch_gemv<M_, R_, U_, S_, T_>(Xp, ldx, m, Wp, W2p, ldw, ob, of, ldo, K, N, stream); break;

void gemv(torch::Tensor X, torch::Tensor W, torch::Tensor W2, torch::Tensor out, int64_t mpad, int64_t rpw, int64_t unroll, int64_t ksplit, bool two_w)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    const int m = (int)X.size(0), K = (int)X.size(1), N = (int)W.size(0);
    const __nv_bfloat16* Xp = reinterpret_cast<const __nv_bfloat16*>(X.data_ptr());
    const __nv_bfloat16* Wp = reinterpret_cast<const __nv_bfloat16*>(W.data_ptr());
    const __nv_bfloat16* W2p = two_w ? reinterpret_cast<const __nv_bfloat16*>(W2.data_ptr()) : nullptr;
    __nv_bfloat16* ob = out.scalar_type() == at::kBFloat16 ? reinterpret_cast<__nv_bfloat16*>(out.data_ptr()) : nullptr;
    float* of = out.scalar_type() == at::kFloat ? out.data_ptr<float>() : nullptr;
    const int ldx = (int)X.stride(0), ldw = (int)W.stride(0), ldo = (int)out.stride(0);
    const int key = (int)mpad * 100000 + (int)rpw * 10000 + (int)unroll * 100 + (int)ksplit * 10 + (two_w ? 1 : 0);
    switch (key) {
        GEMV_CASE(1, 2, 4, 1, false)
        GEMV_CASE(1, 1, 4, 2, false)
        GEMV_CASE(1, 2, 2, 2, false)
        GEMV_CASE(1, 2, 4, 2, false)
        GEMV_CASE(1, 4, 2, 2, false)
        GEMV_CASE(1, 1, 4, 4, false)
        GEMV_CASE(1, 2, 2, 4, false)
        GEMV_CASE(1, 2, 4, 4, false)
        GEMV_CASE(1, 1, 2, 4, false)
        GEMV_CASE(1, 1, 2, 8, false)
        GEMV_CASE(1, 1, 4, 8, true)
        GEMV_CASE(1, 1, 2, 8, true)
        GEMV_CASE(1, 2, 2, 8, true)
        default: TORCH_CHECK(false, "unsupported gemv config");
    }
}
#define MMA_CASE(T_, R_, U_, S_) \
    case (T_ * 10000 + R_ * 1000 + U_ * 10 + S_): launch_mma<T_, R_, U_, S_>(Xp, ldx, m, Wp, ldw, ob, ldo, K, N, stream); break;

void skinny_mma(torch::Tensor X, torch::Tensor W, torch::Tensor out, int64_t mt, int64_t rpw, int64_t unroll, int64_t ksplit)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    const int m = (int)X.size(0), K = (int)X.size(1), N = (int)W.size(0);
    const __nv_bfloat16* Xp = reinterpret_cast<const __nv_bfloat16*>(X.data_ptr());
    const __nv_bfloat16* Wp = reinterpret_cast<const __nv_bfloat16*>(W.data_ptr());
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    const int ldx = (int)X.stride(0), ldw = (int)W.stride(0), ldo = (int)out.stride(0);
    const int key = (int)mt * 10000 + (int)rpw * 1000 + (int)unroll * 10 + (int)ksplit;
    switch (key) {
        MMA_CASE(1, 4, 4, 8)
        MMA_CASE(1, 2, 8, 8)
        MMA_CASE(1, 2, 4, 8)
        MMA_CASE(1, 1, 8, 8)
        MMA_CASE(1, 2, 4, 4)
        MMA_CASE(1, 4, 2, 8)
        MMA_CASE(1, 1, 4, 4)
        MMA_CASE(1, 4, 2, 4)
        MMA_CASE(2, 4, 4, 8)
        MMA_CASE(2, 2, 8, 8)
        MMA_CASE(2, 2, 4, 8)
        MMA_CASE(2, 1, 8, 8)
        MMA_CASE(2, 2, 4, 4)
        MMA_CASE(2, 4, 2, 8)
        MMA_CASE(2, 1, 4, 4)
        MMA_CASE(2, 4, 2, 4)
        default: TORCH_CHECK(false, "unsupported mma config");
    }
}
"""

PROJ_CPP_SRC = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
void stream_bytes(torch::Tensor buf, torch::Tensor out, int64_t grid, int64_t threads, int64_t unroll);
void gemv(torch::Tensor X, torch::Tensor W, torch::Tensor W2, torch::Tensor out, int64_t mpad, int64_t rpw, int64_t unroll, int64_t ksplit, bool two_w);
void skinny_mma(torch::Tensor X, torch::Tensor W, torch::Tensor out, int64_t mt, int64_t rpw, int64_t unroll, int64_t ksplit);
"""

_PROJ_EXT = None


def _proj_build():
    global _PROJ_EXT
    if _PROJ_EXT is None:
        from torch.utils.cpp_extension import load_inline
        _PROJ_EXT = load_inline(
            name="okc_proj_cuda2", cpp_sources=[PROJ_CPP_SRC],
            cuda_sources=["#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n" + PROJ_CUDA_SRC],
            functions=["stream_bytes", "gemv", "skinny_mma"], extra_cuda_cflags=["-O3", "-arch=sm_90a"], verbose=False)
    return _PROJ_EXT


def gemv_usable(m: int, n: int, k: int, mpad: int, rpw: int, unroll: int, ksplit: int) -> bool:
    rows = (8 // ksplit) * rpw
    if m > mpad or n % rows or k % (ksplit * 256):
        return False
    steps = k // ksplit // 256
    return steps % unroll == 0


def mma_usable(m: int, n: int, k: int, mt: int, rpw: int, unroll: int, ksplit: int) -> bool:
    rows = (8 // ksplit) * rpw * 8
    if m > 16 * mt or n % rows or k % (ksplit * 32):
        return False
    steps = k // ksplit // 32
    return steps % unroll == 0


# == END CUDA ==


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
    # (BLOCK_N, BLOCK_K, SPLIT_K, warps, stages, tma): best of the on-device sweep (run 486e0009,
    # graph replays, warm L2); with USE_TILED the tile shape is TILE_SHAPE[layer]
    ("pooler", 4096, 1):   (32, 256, 1, 4, 6, False),     # 18.2
    ("pooler", 4096, 8):   (32, 256, 1, 2, 6, True),      # 18.5
    ("pooler", 4096, 16):  (32, 256, 1, 2, 6, True),      # 18.3
    ("pooler", 4096, 32):  (32, 256, 1, 4, 6, True),      # 20.6
    ("unpooler", 8192, 16): (32, 256, 1, 4, 4, True),     # 18.4
    ("unpooler", 8192, 32): (32, 256, 1, 4, 4, True),     # 19.2
    ("unpooler", 1024, 1): (32, 256, 8, 4, 4, True),      # 5.6
    ("unpooler", 1024, 8): (32, 256, 8, 4, 4, True),      # 5.4
    ("o", 8192, 1):   (64, 256, 1, 4, 4, True),           # 5.0
    ("o", 8192, 8):   (64, 256, 1, 4, 4, True),           # 4.9
    ("o", 8192, 16):  (64, 256, 1, 4, 4, True),           # 4.2
    ("o", 8192, 32):  (64, 256, 1, 4, 4, True),           # 5.3
    ("qkv_gate", 2304, 1):  (32, 256, 4, 4, 4, False),    # 12.2
    ("qkv_gate", 2304, 8):  (64, 128, 8, 4, 4, True),     # 13.7 vs 14.6 for (32, 256, 4, 4, 4) (run 662944e4, cold)
    ("qkv_gate", 2304, 16): (32, 256, 4, 4, 4, True),     # judge: 13.8 here vs 14.5 on (64, 128, 8, 4, 4) (run df4bcce7)
    ("qkv_gate", 2304, 32): (32, 256, 4, 4, 4, True),     # 15.1
    ("router", 256, 1):   (32, 256, 16, 4, 3, True),      # 5.2
    ("router", 256, 8):   (32, 256, 16, 4, 3, True),      # 5.1
    ("router", 256, 16):  (32, 256, 16, 4, 3, True),      # 5.1
    ("router", 256, 32):  (32, 256, 16, 4, 4, True),      # 6.2
}


# Packed tile shape per weight: (BLOCK_N, BLOCK_K). Configs for the tiled kernels use these.
TILE_SHAPE = {"qkv_gate": (32, 256), "o": (64, 256), "pooler": (32, 256), "unpooler": (32, 256), "router": (32, 256)}
# Further packings (BLOCK_N, BLOCK_K) kept in state as "<layer>_tiled_<bn>x<bk>" for the other tile
# shapes a config may ask for (cfg["BLOCK_N"], cfg["BLOCK_K"]); the default packing is one of them.
EXTRA_PACKS = {
    "qkv_gate": [(32, 128), (64, 128), (128, 64), (64, 64), (64, 256)],
    "o":        [(32, 128), (32, 256), (128, 64), (64, 64)],
    "pooler":   [(32, 128), (64, 128), (128, 64), (64, 64), (64, 256)],
    "unpooler": [(32, 128), (64, 128), (128, 64), (64, 64), (64, 256)],
    "router":   [(32, 128), (32, 64)],
}
USE_TILED = True
USE_TMA = False  # fetch tiles with TMA descriptors (box inner dim <= 256 elements)


_OVERRIDE = None   # a config dict, during in-prepare benchmarking


# (layer, n, m) -> (mpad, rows per warp, unroll, ksplit): the CUDA register GEMV (m = 1). Check
# 2a945e84 (cold L2): pooler 22.1 vs 24.2 Triton, qkv 14.9 vs 17.9, unpooler slice 6.2 vs 7.4,
# router 5.6 vs 7.2; o (K = 1024) stays Triton.
GEMV_SHAPES = {
    # pooler m=1 GEMV won the bench (18.5 vs 21.3) but lost on the judge (20.3 vs 18.6, run d39344c2): Triton
    ("qkv_gate", 2304, 1): (1, 2, 2, 4),   # 10.6 vs 11.4 for (1, 1, 4, 2) (run 662944e4)
    ("unpooler", 1024, 1): (1, 1, 4, 2),
    ("router", 256, 1): (1, 1, 2, 8),
}
# (layer, n, m) -> (mt, rows per warp / 8, unroll, ksplit): the CUDA mma.sync register-streaming GEMM (m <= 32)
MMA_SHAPES = {}
# (layer, n, m) -> SPLIT_K: cuBLAS batched split-K (bmm over K chunks) + fp32 partial sum. Check
# 2a945e84: qkv 18.6 vs 20.6 (m=128), 21.6 vs 24.5 (m=256); pooler m=256 32.4 vs 34.0.
BMM_SHAPES = {("qkv_gate", 2304, 128): 4, ("qkv_gate", 2304, 256): 4, ("pooler", 4096, 256): 2,
              ("pooler", 4096, 128): 2}   # pooler m=128: 26.7 vs 28.3 cuBLAS (runs d39344c2, df4bcce7)
# shapes that go to cuBLAS although m <= MAX_TRITON_M: the judge measured our Triton configs behind
# SGLang's cuBLAS call there (qkv m=32: 18.2-19.6 vs 14.0-14.5; o m=32: 7.0 vs 6.8)
CUBLAS_SHAPES = {("qkv_gate", 2304, 32), ("o", 8192, 32)}
ROUTER_DUP = True   # _proj_build [x | x] with a Triton kernel (torch.cat's CatArrayBatchedCopy took 2.6-3.9 us)
# m -> config for the router above 64 tokens (Triton two-weight GEMM); missing: cuBLAS over [hi | lo]
ROUTER_BIG = {256: dict(BLOCK_M=128, BLOCK_N=32, BLOCK_K=128, SPLIT_K=8, num_warps=8, num_stages=4, tma=True)}   # 12.8 vs 14.2


def _config(layer: str, m: int, n: int, k: int):
    if _OVERRIDE is not None:
        if _OVERRIDE.get("none"):
            return None
        if _OVERRIDE.get("table"):
            return _triton_config(layer, m, n, k)
        return dict(_OVERRIDE)
    if (layer, n, m) in GEMV_SHAPES:
        mpad, rpw, unroll, ksplit = GEMV_SHAPES[(layer, n, m)]
        return dict(gemv=True, mpad=mpad, rpw=rpw, unroll=unroll, ksplit=ksplit)
    if (layer, n, m) in MMA_SHAPES:
        mt, rpw, unroll, ksplit = MMA_SHAPES[(layer, n, m)]
        return dict(mma=True, mt=mt, rpw=rpw, unroll=unroll, ksplit=ksplit)
    if (layer, n, m) in BMM_SHAPES:
        return dict(bmm=True, SPLIT_K=BMM_SHAPES[(layer, n, m)])
    if (layer, n, m) in CUBLAS_SHAPES:
        return None
    if layer == "router" and m > MAX_TRITON_M:
        cfg = ROUTER_BIG.get(m)
        return dict(cfg) if cfg else None
    return _triton_config(layer, m, n, k)


def _triton_config(layer: str, m: int, n: int, k: int):
    """The Triton per-shape table (SWEEP / defaults), tiled packing and shared-memory clamp."""
    cfg = _config_table(layer, m, n, k)
    if cfg is None:
        return None
    if cfg.get("big"):
        return cfg
    if USE_TILED:
        cfg["tiled"] = True
        if (cfg["BLOCK_N"], cfg["BLOCK_K"]) not in [TILE_SHAPE[layer]] + EXTRA_PACKS[layer]:
            cfg["BLOCK_N"], cfg["BLOCK_K"] = TILE_SHAPE[layer]
        if cfg.get("tma") is None:
            cfg["tma"] = bool(TMA_BY_M.get(m, USE_TMA))
    elif USE_TMA:
        cfg["tma"] = True
        cfg["BLOCK_K"] = min(cfg["BLOCK_K"], 256)
    # shared memory: stages x (x tile + weight tiles), within the H200's 227 KB
    streams = 2 if layer == "router" else 1
    per_stage = (cfg["BLOCK_M"] + streams * cfg["BLOCK_N"]) * cfg["BLOCK_K"] * 2
    cfg["num_stages"] = max(1, min(cfg["num_stages"], SMEM_BUDGET // per_stage))
    return cfg


SMEM_BUDGET = 200 * 1024


# sweep: which m-slots fetch with TMA (the m <= 32 shapes cost the same bytes)
TMA_BY_M = {1: False, 8: True, 16: True, 32: True}


def _config_table(layer: str, m: int, n: int, k: int):
    cfg = _config_table_(layer, m, n, k)
    if cfg is not None:
        cfg["tma"] = SWEEP.get((layer, n, m), (None,) * 6)[5]
    return cfg


# m >= 128: (BLOCK_N, BLOCK_K, SPLIT_K, warps, stages) for the TMA wgmma kernel over the row-major
# weight (None: cuBLAS). The judge's m=128/256 shapes are where cuBLAS sits at parity.
BIG_M = {
    ("qkv_gate", 128): (128, 64, 4, 8, 4), ("qkv_gate", 256): (128, 64, 4, 8, 4),
    ("o", 128): (128, 64, 1, 8, 4), ("o", 256): (128, 64, 1, 8, 4),
    ("pooler", 128): (128, 64, 4, 8, 4), ("pooler", 256): (128, 64, 2, 8, 4),
    ("unpooler", 128): (128, 64, 2, 8, 4), ("unpooler", 256): (128, 64, 2, 8, 4),
}
USE_BIG_M = False   # measured 1.5-2x slower than cuBLAS at m=128/256 (run d007f8c6)


def _config_table_(layer: str, m: int, n: int, k: int):
    bm = _block_m(m)
    if layer != "router" and m > MAX_TRITON_M and USE_BIG_M and (layer, m) in BIG_M:
        bn, bk, sk, nw, ns = BIG_M[(layer, m)]
        return dict(BLOCK_M=128, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=sk, num_warps=nw, num_stages=ns, tma=True, big=True)
    if layer == "router":
        if m > MAX_TRITON_M:
            return None   # cuBLAS, one GEMM over [hi | lo]
        bn, bk, sk, nw, ns = SWEEP.get((layer, n, m), (16, 256, 8, 4, 4, None))[:5]
        return dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=sk, num_warps=nw, num_stages=ns)
    if m > MAX_TRITON_M:
        return None
    default = {
        "qkv_gate": (16, 256, 1, 4, 4),
        "o": (32, 256, 1, 4, 4),
        "pooler": (32, 256, 1, 4, 4),
        "unpooler": (32, 256, 1, 4, 4) if n > 1024 else (32, 256, 4, 4, 4),
    }[layer]
    bn, bk, sk, nw, ns = SWEEP.get((layer, n, m), default + (None,))[:5]
    if bm > 32 and bk > 256:   # keep shared memory in budget for the 64-row tiles
        bk = 256
    return dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=sk, num_warps=nw, num_stages=ns)


def _usable(cfg, m: int, n: int, k: int) -> bool:
    if cfg is None:
        return False
    if cfg.get("gemv"):
        return gemv_usable(m, n, k, cfg["mpad"], cfg["rpw"], cfg["unroll"], cfg["ksplit"])
    if cfg.get("mma"):
        return mma_usable(m, n, k, cfg["mt"], cfg["rpw"], cfg["unroll"], cfg["ksplit"])
    if cfg.get("bmm"):
        return k % (cfg["SPLIT_K"] * 64) == 0
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
    state["router_hilo_b"] = torch.stack((hi, lo)).contiguous()             # [2, N, K]
    device = router.device
    for name in ("qkv_gate", "o", "pooler", "unpooler"):
        for bn, bk in [TILE_SHAPE[name]] + EXTRA_PACKS[name]:
            state[f"{name}_tiled_{bn}x{bk}"] = _pack_tiles(state[name], bn, bk)
        state[name + "_tiled"] = state[f"{name}_tiled_{TILE_SHAPE[name][0]}x{TILE_SHAPE[name][1]}"]
    for bn, bk in [TILE_SHAPE["router"]] + EXTRA_PACKS["router"]:
        state[f"router_hi_tiled_{bn}x{bk}"] = _pack_tiles(state["router_hi"], bn, bk)
        state[f"router_lo_tiled_{bn}x{bk}"] = _pack_tiles(state["router_lo"], bn, bk)
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
        if ROUTER_DUP and k % 2048 == 0:
            x2 = torch.empty((m, 2 * k), dtype=x.dtype, device=x.device)
            _dup_cols_kernel[(m, k // 2048)](x, x2, k, BLOCK=2048, num_warps=4)
        else:
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


def _sum_partials(part, out):
    numel = out.numel()
    _sum_partials_kernel[(triton.cdiv(numel, 2048),)](part, out, numel, SPLIT_K=part.shape[0], BLOCK=2048, num_warps=4)
    return out


def _launch(x, w, w2, out, state, cfg, m, n, k, layer=None):
    if cfg.get("gemv"):
        ext = _proj_build()
        two_w = layer == "router"
        ext.gemv(x, w, w2 if two_w else w, out, cfg["mpad"], cfg["rpw"], cfg["unroll"], cfg["ksplit"], two_w)
        return out
    if cfg.get("mma"):
        _proj_build().skinny_mma(x, w, out, cfg["mt"], cfg["rpw"], cfg["unroll"], cfg["ksplit"])
        return out
    if cfg.get("bmm"):
        sk = cfg["SPLIT_K"]
        if layer == "router":
            # batch = (hi, lo): partials [2, m, n] fp32, summed
            xb = x.unsqueeze(0).expand(2, m, k)
            part = torch.bmm(xb, state["router_hilo_b"].transpose(1, 2), out_dtype=torch.float32)
            return _sum_partials(part, out)
        xb = x.view(m, sk, k // sk).transpose(0, 1)                 # [sk, m, k/sk]
        wb = w.view(n, sk, k // sk).permute(1, 2, 0)                 # [sk, k/sk, n]
        part = torch.bmm(xb, wb, out_dtype=torch.float32)            # [sk, m, n]
        return _sum_partials(part, out)
    m_tiles = triton.cdiv(m, cfg["BLOCK_M"])
    n_tiles = n // cfg["BLOCK_N"]
    swap = bool(cfg.get("swap_mn"))
    grid = (m_tiles, cfg["SPLIT_K"], n_tiles) if swap else (n_tiles, cfg["SPLIT_K"], m_tiles)
    if cfg.get("tiled"):
        # the packed weight: n-major tiles, so the unpooler's first 1024 rows are its first tiles
        bn, bk = cfg["BLOCK_N"], cfg["BLOCK_K"]
        if layer == "router":
            wt, w2t = state[f"router_hi_tiled_{bn}x{bk}"], state[f"router_lo_tiled_{bn}x{bk}"]
        else:
            wt, w2t = state[f"{layer}_tiled_{bn}x{bk}"], None
        if cfg.get("tma"):
            from triton.tools.tensor_descriptor import TensorDescriptor
            x_desc = TensorDescriptor.from_tensor(x, [cfg["BLOCK_M"], cfg["BLOCK_K"]])
            w_desc = TensorDescriptor.from_tensor(wt, [cfg["BLOCK_N"], cfg["BLOCK_K"]])
            w2_desc = TensorDescriptor.from_tensor(w2t if w2t is not None else wt, [cfg["BLOCK_N"], cfg["BLOCK_K"]])
            _tiled_gemm_tma_kernel[grid](
                x_desc, w_desc, w2_desc, out, state["partials"], state["counters"],
                m, n, k, out.stride(0),
                BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
                SPLIT_K=cfg["SPLIT_K"], TWO_W=w2t is not None, SWAP_MN=swap,
                num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
        else:
            _tiled_gemm_kernel[grid](
                x, wt, w2t if w2t is not None else wt, out, state["partials"], state["counters"],
                m, n, k, x.stride(0), out.stride(0),
                BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
                SPLIT_K=cfg["SPLIT_K"], TWO_W=w2t is not None, SWAP_MN=swap,
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
            SPLIT_K=cfg["SPLIT_K"], TWO_W=w2 is not None, SWAP_MN=swap,
            num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
        )
        return out
    _skinny_gemm_kernel[grid](
        x, w, w2 if w2 is not None else w, out, state["partials"], state["counters"],
        m, n, k,
        x.stride(0), w.stride(0), out.stride(0),
        BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
        SPLIT_K=cfg["SPLIT_K"], TWO_W=w2 is not None, SWAP_MN=swap,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )
    return out


# ---- shared: attn (muse_attention_decode/submission.py) ----
RMS_EPS = tl.constexpr(1e-5)
OUT_DTYPE = torch.bfloat16
ACT_DTYPE = tl.float16    # the dots' operand dtype: exact for the E4M3 cache values and finer than BF16
FP8_ASM = True            # hardware e4m3x2 -> f16x2 conversion (False: generic cast, for the interpreter)
QK_FP8_ROWS = True        # 64-row (verify) tiles: E4M3 queries and keys straight into the QK^T MMA
EX2_F16 = True            # exp2 of the (<= 0) logits in fp16 pairs: one MUFU op per two keys (needs inline asm)
PV_FP8 = False            # P.V with E4M3 P and the raw E4M3 values (what FA3's FP8 path does)
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
def _ex2_f16(x):
    """2^x for x <= 0 (fp32) as fp16, two per MUFU op. Below the fp16 range the input rounds to
    -inf and the result to 0; the result feeds the P.V dot as fp16 anyway."""
    return tl.inline_asm_elementwise(asm="ex2.approx.f16x2 $0, $1;", constraints="=r,r", args=[x.to(tl.float16)],
                                     dtype=tl.float16, is_pure=True, pack=2)


@triton.jit
def _prefetch_l2(ptrs):
    """Warm the L2 with the lines at ptrs (a hint: no data, no fault), so a later dependent load
    or atomic on them does not pay a cold DRAM round trip."""
    tl.inline_asm_elementwise(asm="prefetch.global.L2 [$1]; mov.b32 $0, 0;", constraints="=r,l",
                              args=[ptrs.to(tl.int64, bitcast=True)], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _rms_rope(xe, xo, pos, inv_freq, scale, ROPE: tl.constexpr, D: tl.constexpr):
    """Per-row RMSNorm of a 64-wide head row split into even/odd halves [R, DW/2] (a row's
    other-head columns, if any, are zero), scaled, then interleaved RoPE (pairs (2i, 2i+1)
    rotate by pos * theta^(-2i/64)) when ROPE."""
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
def _prep_q(qkv_ptr, kc_ptr, tok, g, hh, rmask, hbase, pos, inv_freq, qk_scale, k_scale,
            QKV_W: tl.constexpr, GROUP: tl.constexpr, D: tl.constexpr, DW: tl.constexpr, ROPE: tl.constexpr,
            QK_FP8: tl.constexpr, ACT: tl.constexpr):
    """The set's query rows [RB, DW]: RMSNorm x qk_scale, RoPE; columns of another head (DW > D)
    are zero, so one dot over the whole DW-wide cache row scores each row against its own head."""
    dh = tl.arange(0, DW // 2)
    hc = dh // (D // 2)
    i = dh % (D // 2)
    qp = qkv_ptr + tok[:, None] * QKV_W + ((hbase + hc[None, :]) * GROUP + g[:, None]) * D + 2 * i[None, :]
    m = rmask[:, None] & (hc[None, :] == hh[:, None])
    qe = tl.load(qp, mask=m, other=0.0).to(tl.float32)
    qo = tl.load(qp + 1, mask=m, other=0.0).to(tl.float32)
    if QK_FP8:
        # FA3's FP8 path: the normalized query is rounded to E4M3; the 5.473 factor and the
        # cache scale move into the softmax scale
        qe, qo = _rms_rope(qe, qo, pos, inv_freq, 1.0, ROPE, D)
        return tl.interleave(qe, qo).to(kc_ptr.dtype.element_ty)
    else:
        qe, qo = _rms_rope(qe, qo, pos, inv_freq, qk_scale, ROPE, D)
        # keys are used as their raw E4M3 values: the cache scale multiplies q instead
        return (tl.interleave(qe, qo) * k_scale).to(ACT)


@triton.jit
def _tile(q, k8, v8, ok, m_i, l_i, acc, mult, MASKED: tl.constexpr, QK_FP8: tl.constexpr, ACT: tl.constexpr,
          FP8_ASM: tl.constexpr, EX2H: tl.constexpr, PV8: tl.constexpr, ABL: tl.constexpr):
    """One online-softmax step over a key tile: k8/v8 [T, DW] raw E4M3, q [RB, DW].
    mult turns raw scores into log2-domain logits; m_i/l_i/acc are the running statistics.
    ABL (benchmark ablations, wrong results): 1 = skip the V path, 2 = skip the QK^T dot."""
    if ABL == 2:
        kmax = tl.max(k8.to(tl.uint8, bitcast=True), 1).to(tl.float32)        # a cheap use of the K tile
        sc = tl.zeros((q.shape[0], k8.shape[0]), tl.float32) + kmax[None, :] * 1e-30
    elif QK_FP8:
        sc = tl.dot(q, tl.trans(k8))                              # e4m3 x e4m3, fp32 accumulate
    else:
        sc = tl.dot(q, tl.trans(_cache_to_act(k8, ACT, FP8_ASM)))  # [RB, T]
    if MASKED:
        sc = tl.where(ok, sc, NEG)
    m_new = tl.maximum(m_i, tl.max(sc, 1) * mult)
    alpha = tl.exp2(m_i - m_new)
    x = sc * mult - m_new[:, None]                                # one FMA per logit
    if EX2H:
        p16 = _ex2_f16(x)
        if MASKED:
            p16 = tl.where(ok, p16, 0.0)
        l_i = l_i * alpha + tl.sum(p16, 1).to(tl.float32)
    else:
        p = tl.exp2(x)
        if MASKED:
            p = tl.where(ok, p, 0.0)
        l_i = l_i * alpha + tl.sum(p, 1)
        p16 = p.to(ACT)
    if ABL == 1:
        vmax = tl.max(v8.to(tl.uint8, bitcast=True), 0).to(tl.float32)        # a cheap use of the V tile
        pv = tl.zeros_like(acc) + vmax[None, :] * 1e-30
    elif PV8:
        pv = tl.dot(p16.to(tl.float8e4nv), v8)
    else:
        pv = tl.dot(p16, _cache_to_act(v8, ACT, FP8_ASM))
    acc = acc * alpha[:, None] + pv
    return m_new, l_i, acc


@triton.jit
def _finish(qkv_ptr, out_ptr, o, tok, g, hh, rmask, hbase,
            QKV_W: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, GROUP: tl.constexpr, D: tl.constexpr, DW: tl.constexpr):
    """o [RB, DW]: the normalized attention rows -> RMSNorm x sigmoid(gate) -> output."""
    c = tl.arange(0, DW)
    hcc = c // D
    dd = c % D
    qh = (hbase + hh) * GROUP + g
    m = rmask[:, None] & (hcc[None, :] == hh[:, None])
    if DW > D:
        o = tl.where(hcc[None, :] == hh[:, None], o, 0.0)   # the other head's half of P.V
    ss = tl.sum(o * o, 1) / D
    o = o * tl.rsqrt(ss + RMS_EPS)[:, None]
    gp = qkv_ptr + tok[:, None] * QKV_W + (HQ + 2 * HKV + qh)[:, None] * D + dd[None, :]
    gate = tl.load(gp, mask=m, other=0.0).to(tl.float32)
    o = o * tl.sigmoid(gate)
    op = out_ptr + tok[:, None] * (HQ * D) + qh[:, None] * D + dd[None, :]
    tl.store(op, o.to(out_ptr.dtype.element_ty), mask=m)


@triton.jit
def _merge(part_ptr, ml_ptr, base0, stride, rows, N: tl.constexpr, RB: tl.constexpr, DW: tl.constexpr, CH: tl.constexpr):
    """Combine N partial row sets (normalized rows a_i = acc_i / l_i in part, (m_i, l_i) in ml) at
    base0 + i * stride: the overall max m, weight sum l = sum_i 2^(m_i - m) l_i and
    o = sum_i 2^(m_i - m) l_i a_i (so o / l is the combined normalized row), CH sets per step."""
    CHN: tl.constexpr = N if CH > N else CH   # sets per step (never past the N that exist)
    c = tl.arange(0, DW)
    m_all = tl.full((RB,), NEG, tl.float32)
    for s0 in tl.static_range(0, N, CHN):
        sidx = s0 + tl.arange(0, CHN)
        b2 = base0 + sidx * stride                                                 # [CHN]
        m2 = tl.load(ml_ptr + (b2[:, None] + rows[None, :]) * 2, cache_modifier=".cg")   # [CHN, RB]
        m_all = tl.maximum(m_all, tl.max(m2, 0))
    l_all = tl.zeros((RB,), tl.float32)
    o = tl.zeros((RB, DW), tl.float32)
    for s0 in tl.static_range(0, N, CHN):
        sidx = s0 + tl.arange(0, CHN)
        b2 = base0 + sidx * stride
        m2 = tl.load(ml_ptr + (b2[:, None] + rows[None, :]) * 2, cache_modifier=".cg")
        l2 = tl.load(ml_ptr + (b2[:, None] + rows[None, :]) * 2 + 1, cache_modifier=".cg")
        a2 = tl.load(part_ptr + (b2[:, None, None] + rows[None, :, None]) * DW + c[None, None, :],
                     cache_modifier=".cg").to(tl.float32)                            # [CH, RB, DW]
        w = tl.exp2(m2 - m_all[None, :]) * l2                                         # [CH, RB]
        l_all += tl.sum(w, 0)
        o += tl.sum(a2 * w[:, :, None], 0)
    return m_all, l_all, o


@triton.jit
def _put_partial(part_ptr, ml_ptr, base, rows, m, l, acc, DW: tl.constexpr):
    """Store one row set's (m, l, acc / l) at row index base."""
    c = tl.arange(0, DW)
    inv = tl.where(l > 0, 1.0 / l, 0.0)
    tl.store(part_ptr + (base + rows)[:, None] * DW + c[None, :], (acc * inv[:, None]).to(part_ptr.dtype.element_ty))
    tl.store(ml_ptr + (base + rows) * 2, m)
    tl.store(ml_ptr + (base + rows) * 2 + 1, l)


@triton.jit
def _attn_decode_kernel(
    qkv_ptr, loc_ptr, seq_ptr, pt_ptr, kc_ptr, vc_ptr, kscale_ptr, vscale_ptr,
    out_ptr, part_ptr, ml_ptr, cnt_ptr,
    max_pages, qk_scale, sm_scale_log2, log2_theta,
    L: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
    PS: tl.constexpr, WINDOW: tl.constexpr, ROPE: tl.constexpr, SPLITS: tl.constexpr,
    RB: tl.constexpr, LB: tl.constexpr, BLOCK_T: tl.constexpr, ACT: tl.constexpr, FP8_ASM: tl.constexpr,
    CH: tl.constexpr, QK_FP8: tl.constexpr, HP: tl.constexpr, BD: tl.constexpr, EX2H: tl.constexpr, PV8: tl.constexpr,
    G1: tl.constexpr, PF: tl.constexpr, ABL: tl.constexpr, PFX: tl.constexpr,
):
    GROUP: tl.constexpr = HQ // HKV
    HPS: tl.constexpr = HP if BD else 1        # heads sharing one row set
    NSET: tl.constexpr = HP // HPS             # row sets (each with its own dots and statistics)
    DW: tl.constexpr = HPS * D                 # width of a row set's slice of a cache row
    R: tl.constexpr = L * GROUP * HPS          # live rows of a set
    LBX: tl.constexpr = LB * HPS               # the block's new tokens, per head of the set
    QKV_W: tl.constexpr = (2 * HQ + 2 * HKV) * D
    ROW_W: tl.constexpr = HKV * D   # one cache row (all KV heads) in elements
    NG: tl.constexpr = HKV // HP    # head groups per request
    G2: tl.constexpr = SPLITS // G1   # the merge: G2 groups of G1 splits, then the groups

    bp = tl.program_id(0)
    s = tl.program_id(1)
    b = bp // NG
    h0 = (bp % NG) * HP
    seq = tl.load(seq_ptr + b)   # cached tokens; the block sits at positions seq .. seq+L-1
    k_scale = tl.load(kscale_ptr)
    v_scale = tl.load(vscale_ptr)

    rows = tl.arange(0, RB)
    hh = rows // (L * GROUP)     # head of the row within its set (BD only; else 0)
    j = (rows // GROUP) % L      # block token of the row
    g = rows % GROUP             # query head within the group
    rmask = rows < R
    tok = b * L + j
    c = tl.arange(0, DW)
    dh = tl.arange(0, DW // 2)   # pair index: dims 2i (even) and 2i+1 (odd) of head dh // 32
    inv_freq = tl.exp2(-(2.0 * (dh % (D // 2)).to(tl.float32) / D) * log2_theta)

    if PF:
        # warm the lines the merge chain ends on: this CTA's ticket and the gate rows
        qh_c = tl.minimum((h0 + hh) * GROUP + g, HQ - 1)
        _prefetch_l2(qkv_ptr + tok * QKV_W + (HQ + 2 * HKV + qh_c) * D)
        if NSET == 2:
            _prefetch_l2(qkv_ptr + tok * QKV_W + (HQ + 2 * HKV + tl.minimum(qh_c + GROUP, HQ - 1)) * D)
        if SPLITS > 1:
            _prefetch_l2(cnt_ptr + bp * (G2 + 1) + s // G1 + rows * 0)

    # ---- queries ----
    q0 = _prep_q(qkv_ptr, kc_ptr, tok, g, hh, rmask, h0, seq + j, inv_freq, qk_scale, k_scale,
                 QKV_W, GROUP, D, DW, ROPE, QK_FP8, ACT)
    if NSET == 2:
        q1 = _prep_q(qkv_ptr, kc_ptr, tok, g, hh, rmask, h0 + 1, seq + j, inv_freq, qk_scale, k_scale,
                     QKV_W, GROUP, D, DW, ROPE, QK_FP8, ACT)
    if QK_FP8:
        qk_mult = sm_scale_log2 * qk_scale * k_scale
    else:
        qk_mult = sm_scale_log2

    # ---- the block's new keys and values: [LBX, DW] per set, block-diagonal over heads ----
    kr = tl.arange(0, LBX)
    khh = kr // LB
    kj = kr % LB
    kmask = kj < L
    ktok = b * L + kj
    hc = dh // (D // 2)
    i2 = 2 * (dh % (D // 2))
    hcc = c // D
    dd = c % D
    kp = qkv_ptr + ktok[:, None] * QKV_W + (HQ + h0 + hc[None, :]) * D + i2[None, :]
    km = kmask[:, None] & (hc[None, :] == khh[:, None])
    ke = tl.load(kp, mask=km, other=0.0).to(tl.float32)
    ko = tl.load(kp + 1, mask=km, other=0.0).to(tl.float32)
    ke, ko = _rms_rope(ke, ko, seq + kj, inv_freq, 1.0, ROPE, D)
    k_new0 = (tl.interleave(ke, ko) / k_scale).to(kc_ptr.dtype.element_ty)
    vp = qkv_ptr + ktok[:, None] * QKV_W + (HQ + HKV + h0 + hcc[None, :]) * D + dd[None, :]
    vm = kmask[:, None] & (hcc[None, :] == khh[:, None])
    v_new0 = (tl.load(vp, mask=vm, other=0.0).to(tl.float32) / v_scale).to(vc_ptr.dtype.element_ty)
    if NSET == 2:
        ke = tl.load(kp + D, mask=km, other=0.0).to(tl.float32)
        ko = tl.load(kp + D + 1, mask=km, other=0.0).to(tl.float32)
        ke, ko = _rms_rope(ke, ko, seq + kj, inv_freq, 1.0, ROPE, D)
        k_new1 = (tl.interleave(ke, ko) / k_scale).to(kc_ptr.dtype.element_ty)
        v_new1 = (tl.load(vp + D, mask=vm, other=0.0).to(tl.float32) / v_scale).to(vc_ptr.dtype.element_ty)
    if s == 0:
        loc = tl.load(loc_ptr + ktok, mask=kmask, other=0)
        cpn = loc[:, None] * ROW_W + h0 * D + c[None, :]
        tl.store(kc_ptr + cpn, k_new0, mask=vm)
        tl.store(vc_ptr + cpn, v_new0, mask=vm)
        if NSET == 2:
            tl.store(kc_ptr + cpn + D, k_new1, mask=vm)
            tl.store(vc_ptr + cpn + D, v_new1, mask=vm)

    # ---- online softmax over this split's slice of the past keys ----
    m0 = tl.full((RB,), NEG, tl.float32)
    l0 = tl.zeros((RB,), tl.float32)
    acc0 = tl.zeros((RB, DW), tl.float32)
    if NSET == 2:
        m1 = tl.full((RB,), NEG, tl.float32)
        l1 = tl.zeros((RB,), tl.float32)
        acc1 = tl.zeros((RB, DW), tl.float32)
    if WINDOW > 0:
        lo = tl.maximum(seq - (WINDOW - 1), 0)
    else:
        lo = seq * 0
    n_keys = seq - lo
    chunk = (n_keys + SPLITS - 1) // SPLITS
    start = lo + s * chunk
    end = tl.minimum(seq, start + chunk)
    first = (seq + j) - (WINDOW - 1)   # each row's oldest attended position (sliding)
    ta = tl.arange(0, BLOCK_T)
    t0 = start
    if WINDOW > 0:
        if s == 0:
            # the window's edge: rows differ in their oldest key -> masked tile
            t = t0 + ta
            valid = t < end
            page = tl.load(pt_ptr + b * max_pages + t // PS, mask=valid, other=0)
            row = page.to(tl.int64) * PS + (t % PS).to(tl.int64)
            cp = row[:, None] * ROW_W + h0 * D + c[None, :]
            ok = valid[None, :] & (t[None, :] >= first[:, None])
            k8 = tl.load(kc_ptr + cp, mask=valid[:, None], other=0.0)
            v8 = tl.load(vc_ptr + cp, mask=valid[:, None], other=0.0)
            m0, l0, acc0 = _tile(q0, k8, v8, ok, m0, l0, acc0, qk_mult, True, QK_FP8, ACT, FP8_ASM, EX2H, PV8, ABL)
            if NSET == 2:
                k8 = tl.load(kc_ptr + cp + D, mask=valid[:, None], other=0.0)
                v8 = tl.load(vc_ptr + cp + D, mask=valid[:, None], other=0.0)
                m1, l1, acc1 = _tile(q1, k8, v8, ok, m1, l1, acc1, qk_mult, True, QK_FP8, ACT, FP8_ASM, EX2H, PV8, ABL)
            t0 = t0 + BLOCK_T
    # interior tiles: every key valid, no masks at all
    for tt in range(t0, end - (BLOCK_T - 1), BLOCK_T):
        t = tt + ta
        page = tl.load(pt_ptr + b * max_pages + t // PS)
        row = page.to(tl.int64) * PS + (t % PS).to(tl.int64)
        cp = row[:, None] * ROW_W + h0 * D + c[None, :]
        if PFX:
            # the other KV head's two 32-byte sectors of the same 128-byte line: fetched with ours, so
            # its CTA (launched next to this one) hits L2 and the DRAM sees one 128-byte access per row
            sec = tl.arange(0, 2) * 32
            cpx = row[:, None] * ROW_W + (h0 ^ 1) * D + sec[None, :]
            _prefetch_l2(kc_ptr + cpx)
            _prefetch_l2(vc_ptr + cpx)
        k8 = tl.load(kc_ptr + cp)
        v8 = tl.load(vc_ptr + cp)
        m0, l0, acc0 = _tile(q0, k8, v8, rmask, m0, l0, acc0, qk_mult, False, QK_FP8, ACT, FP8_ASM, EX2H, PV8, ABL)
        if NSET == 2:
            k8 = tl.load(kc_ptr + cp + D)
            v8 = tl.load(vc_ptr + cp + D)
            m1, l1, acc1 = _tile(q1, k8, v8, rmask, m1, l1, acc1, qk_mult, False, QK_FP8, ACT, FP8_ASM, EX2H, PV8, ABL)
    # the slice's tail (fewer than BLOCK_T keys): masked tile
    tt = t0 + (tl.maximum(end - t0, 0) // BLOCK_T) * BLOCK_T
    if tt < end:
        t = tt + ta
        valid = t < end
        page = tl.load(pt_ptr + b * max_pages + t // PS, mask=valid, other=0)
        row = page.to(tl.int64) * PS + (t % PS).to(tl.int64)
        cp = row[:, None] * ROW_W + h0 * D + c[None, :]
        if WINDOW > 0:
            ok = valid[None, :] & (t[None, :] >= first[:, None])
        else:
            ok = valid[None, :] & (rows[:, None] >= 0)
        k8 = tl.load(kc_ptr + cp, mask=valid[:, None], other=0.0)
        v8 = tl.load(vc_ptr + cp, mask=valid[:, None], other=0.0)
        m0, l0, acc0 = _tile(q0, k8, v8, ok, m0, l0, acc0, qk_mult, True, QK_FP8, ACT, FP8_ASM, EX2H, PV8, ABL)
        if NSET == 2:
            k8 = tl.load(kc_ptr + cp + D, mask=valid[:, None], other=0.0)
            v8 = tl.load(vc_ptr + cp + D, mask=valid[:, None], other=0.0)
            m1, l1, acc1 = _tile(q1, k8, v8, ok, m1, l1, acc1, qk_mult, True, QK_FP8, ACT, FP8_ASM, EX2H, PV8, ABL)

    # ---- split 0: the block's own tokens (causal inside the block, own head only) ----
    if s == 0:
        ok = kmask[None, :] & (kj[None, :] <= j[:, None]) & (khh[None, :] == hh[:, None])
        m0, l0, acc0 = _tile(q0, k_new0, v_new0, ok, m0, l0, acc0, qk_mult, True, QK_FP8, ACT, FP8_ASM, EX2H, PV8, ABL)
        if NSET == 2:
            m1, l1, acc1 = _tile(q1, k_new1, v_new1, ok, m1, l1, acc1, qk_mult, True, QK_FP8, ACT, FP8_ASM, EX2H, PV8, ABL)

    acc0 = acc0 * v_scale   # values were accumulated as raw E4M3
    if NSET == 2:
        acc1 = acc1 * v_scale

    # ---- merge the splits, then the gated norm: the last CTA of each group of G1 splits merges
    # its group; the last group to finish merges the G2 group results ----
    if SPLITS > 1:
        ncta = tl.num_programs(0)
        grp = s // G1
        base = ((bp * SPLITS + s) * NSET) * RB
        _put_partial(part_ptr, ml_ptr, base, rows, m0, l0, acc0, DW)
        if NSET == 2:
            _put_partial(part_ptr, ml_ptr, base + RB, rows, m1, l1, acc1, DW)
        tl.debug_barrier()
        ticket = tl.atomic_add(cnt_ptr + bp * (G2 + 1) + grp, 1, sem="acq_rel", scope="gpu")
        tl.debug_barrier()
        if ticket == G1 - 1:
            gfirst = ((bp * SPLITS + grp * G1) * NSET) * RB         # the group's first split
            m0, l0, acc0 = _merge(part_ptr, ml_ptr, gfirst, NSET * RB, rows, G1, RB, DW, CH)
            if NSET == 2:
                m1, l1, acc1 = _merge(part_ptr, ml_ptr, gfirst + RB, NSET * RB, rows, G1, RB, DW, CH)
            if G2 == 1:
                _finish(qkv_ptr, out_ptr, acc0 / l0[:, None], tok, g, hh, rmask, h0, QKV_W, HQ, HKV, GROUP, D, DW)
                if NSET == 2:
                    _finish(qkv_ptr, out_ptr, acc1 / l1[:, None], tok, g, hh, rmask, h0 + 1, QKV_W, HQ, HKV, GROUP, D, DW)
                tl.atomic_xchg(cnt_ptr + bp * (G2 + 1) + grp, 0)
            else:
                # the group results live after the split partials
                gbase = (ncta * SPLITS + bp * G2 + grp) * NSET * RB
                _put_partial(part_ptr, ml_ptr, gbase, rows, m0, l0, acc0, DW)
                if NSET == 2:
                    _put_partial(part_ptr, ml_ptr, gbase + RB, rows, m1, l1, acc1, DW)
                tl.atomic_xchg(cnt_ptr + bp * (G2 + 1) + grp, 0)
                tl.debug_barrier()
                ticket2 = tl.atomic_add(cnt_ptr + bp * (G2 + 1) + G2, 1, sem="acq_rel", scope="gpu")
                tl.debug_barrier()
                if ticket2 == G2 - 1:
                    gfirst2 = (ncta * SPLITS + bp * G2) * NSET * RB
                    m0, l0, acc0 = _merge(part_ptr, ml_ptr, gfirst2, NSET * RB, rows, G2, RB, DW, CH)
                    _finish(qkv_ptr, out_ptr, acc0 / l0[:, None], tok, g, hh, rmask, h0, QKV_W, HQ, HKV, GROUP, D, DW)
                    if NSET == 2:
                        m1, l1, acc1 = _merge(part_ptr, ml_ptr, gfirst2 + RB, NSET * RB, rows, G2, RB, DW, CH)
                        _finish(qkv_ptr, out_ptr, acc1 / l1[:, None], tok, g, hh, rmask, h0 + 1, QKV_W, HQ, HKV, GROUP, D, DW)
                    tl.atomic_xchg(cnt_ptr + bp * (G2 + 1) + G2, 0)
    else:
        _finish(qkv_ptr, out_ptr, acc0 / l0[:, None], tok, g, hh, rmask, h0, QKV_W, HQ, HKV, GROUP, D, DW)
        if NSET == 2:
            _finish(qkv_ptr, out_ptr, acc1 / l1[:, None], tok, g, hh, rmask, h0 + 1, QKV_W, HQ, HKV, GROUP, D, DW)


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


def _pow2(n: int) -> int:
    return 1 << (max(1, n).bit_length() - 1)


def _plan(cfg):
    """Kernel configuration from the shape only: dict(splits, block_t, hp, bd, warps, stages, ex2h, pv8[, g1, qk8, pf]).
    Measured on the H200 (checks ef648771, 1994f6b7, db00ede3: CUDA-graph replays after an L2 flush)."""
    b, hkv, L = cfg["batch"], cfg["num_kv_heads"], cfg["q_len"]
    rows = L * (cfg["num_q_heads"] // hkv)   # query rows per KV head
    sliding = cfg["window"] > 0
    ctas = b * hkv
    base = dict(block_t=128, hp=1, bd=False, warps=4, stages=5, ex2h=EX2_F16, pv8=PV_FP8)
    if rows <= 16 and not sliding:
        if ctas <= 4 or hkv % 2:
            # tiny batches: many CTAs, one head each
            return dict(base, splits=_pow2(min(128, 1024 // ctas)))
        # full-context decode: both heads per CTA over whole 128-byte cache rows, 64-key tiles
        return dict(base, splits=_pow2(min(16, 512 // ctas)), block_t=64, hp=2, bd=True)
    if rows <= 16:
        # sliding decode: one 128-key tile per CTA for a tiny batch, else 64-key tiles
        if ctas <= 2:
            return dict(base, splits=16)
        return dict(base, splits=_pow2(min(8, 512 // ctas)), block_t=64)
    if sliding:
        # sliding verify: 64-key tiles, 8 warps (shorter per-tile latency), 1-8 tiles per CTA
        return dict(base, splits=_pow2(min(32, 256 // ctas)), block_t=64, warps=8)
    if ctas <= 4:
        # tiny-batch verify over the whole context: 64 splits x 8 warps judged faster than 128 x 4
        return dict(base, splits=_pow2(min(64, 1024 // ctas)), warps=8)
    return dict(base, splits=_pow2(min(8, 256 // ctas)))


_PLAN_OVERRIDE = None   # dict of plan keys to force, during benchmarking
ATTN_BENCH = False      # True: one on-device sweep in the first eager call (never under graph capture); prints to the log
_BENCH_DONE = False


def _attn_bench(data):
    """Times kernel configurations on synthetic inputs shaped like the scored cases, the way the
    judge does: each as a CUDA graph, replayed after an L2 flush; prints the median."""
    import time
    global _PLAN_OVERRIDE
    t0 = time.time()
    dev = data["qkv"].device
    cfg0 = dict(data["config"])
    HQ, HKV, D = cfg0["num_q_heads"], cfg0["num_kv_heads"], cfg0["head_dim"]
    g = torch.Generator(device=dev).manual_seed(0)
    flush = torch.zeros((160 << 20,), dtype=torch.uint8, device=dev)
    sink = torch.empty(((160 << 20) >> 16,), dtype=torch.int32, device=dev)

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

    def timeit(fn, reps=11):
        fn(); fn()                       # compile, allocate the workspace
        torch.cuda.synchronize()
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            gph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gph, stream=st):
                fn()
        torch.cuda.current_stream().wait_stream(st)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(reps):
            sink.copy_(flush.view(-1, 1 << 16).sum(1, dtype=torch.int32))   # cold L2 by READING, as the judge does (a write flush leaves dirty lines)
            e0.record()
            gph.replay()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1000.0)
        ts.sort()
        return ts[len(ts) // 2]

    print("BENCH attention: us per graph replay after an L2 flush (median), plan overrides")
    W = 2048
    cases = {"B1 L1 sliding": (1, 1, W), "B1 L1 full": (1, 1, 0), "B16 L1 sliding": (16, 1, W), "B16 L1 full": (16, 1, 0),
             "B32 L1 sliding": (32, 1, W), "B32 L1 full": (32, 1, 0), "B1 L8 sliding": (1, 8, W), "B1 L8 full": (1, 8, 0),
             "B16 L8 sliding": (16, 8, W), "B16 L8 full": (16, 8, 0), "B32 L8 sliding": (32, 8, W), "B32 L8 full": (32, 8, 0)}
    bd = dict(hp=2, bd=True, stages=5, block_t=64)                   # both heads per CTA, whole 128B rows
    nw = dict(hp=1, bd=False, stages=5, ex2h=True, block_t=128)      # one head per CTA
    w8 = dict(nw, block_t=64, warps=8)
    plans = {
        "B1 L1 sliding": [dict(nw, splits=16), dict(nw, splits=16, pfx=True)],
        "B1 L1 full": [dict(nw, splits=128), dict(nw, splits=128, pfx=True)],
        "B16 L1 sliding": [dict(nw, splits=8, block_t=64), dict(nw, splits=8, block_t=64, pfx=True)],
        "B16 L1 full": [dict(bd, splits=16)],
        "B32 L1 sliding": [dict(nw, splits=8, block_t=64), dict(nw, splits=8, block_t=64, pfx=True)],
        "B32 L1 full": [dict(bd, splits=8)],
        "B1 L8 sliding": [dict(w8, splits=32), dict(w8, splits=32, pfx=True), dict(w8, splits=32, g1=8, pfx=True)],
        "B1 L8 full": [dict(nw, splits=64, warps=8), dict(nw, splits=64, warps=8, pfx=True), dict(nw, splits=64, warps=8, pfx=True, stages=8)],
        "B16 L8 sliding": [dict(w8, splits=8), dict(w8, splits=8, pfx=True), dict(w8, splits=4, pfx=True)],
        "B16 L8 full": [dict(nw, splits=8), dict(nw, splits=8, pfx=True), dict(nw, splits=8, pfx=True, stages=8), dict(nw, splits=16, pfx=True),
                        dict(nw, splits=4, pfx=True), dict(nw, splits=8, pfx=True, pf=False)],
        "B32 L8 sliding": [dict(w8, splits=4), dict(w8, splits=4, pfx=True)],
        "B32 L8 full": [dict(nw, splits=4), dict(nw, splits=4, pfx=True), dict(nw, splits=4, pfx=True, stages=6), dict(nw, splits=8, pfx=True)],
    }
    for name, (B, L, window) in cases.items():
        d = make(B, L, window)
        for plan in plans[name]:
            if time.time() - t0 > 300:
                print("BENCH budget reached"); break
            _PLAN_OVERRIDE = plan
            try:
                us = timeit(lambda: _attention_forward(d))
                print(f"  {name:15s} {str(plan):75s} {us:9.1f} us")
            except Exception as e:
                print(f"  {name:15s} {str(plan):75s} failed: {str(e)[:120]}")
        _PLAN_OVERRIDE = None
        del d
    _PLAN_OVERRIDE = None
    print(f"BENCH attention done in {time.time() - t0:.0f}s")


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
    group = HQ // HKV
    qkv = data["qkv"]
    if qkv.stride(-1) != 1:
        qkv = qkv.contiguous()
    device = qkv.device
    out = torch.empty((B, L, HQ * D), dtype=OUT_DTYPE, device=device)

    plan = _plan(cfg)
    if _PLAN_OVERRIDE is not None:
        plan.update(_PLAN_OVERRIDE)
    hp = plan["hp"] if HKV % plan["hp"] == 0 else 1
    bd = plan["bd"] and hp == 2
    hps = hp if bd else 1
    rb = max(int(plan.get("rb", 16)), triton.next_power_of_2(L * group * hps))   # dots need >= 16 rows
    ctas = B * HKV // hp
    splits = max(1, min(plan["splits"], 1 << (max(1, 1024 // ctas).bit_length() - 1)))
    splits = 1 << (splits.bit_length() - 1)
    qk_fp8 = QK_FP8_ROWS and (rb > 16 or bool(plan.get("qk8", False)))   # E4M3 queries x raw E4M3 keys
    ex2h = bool(plan["ex2h"]) and FP8_ASM
    pv8 = bool(plan["pv8"]) and FP8_ASM
    lb = 32 if pv8 else 16
    g1 = plan.get("g1") or {1: 1, 2: 2, 4: 4, 8: 8, 16: 16, 32: 8, 64: 8, 128: 16, 256: 16, 512: 32}[splits]   # merge groups (one level up to 16)
    g2 = splits // g1
    nset = hp // hps
    n_rows = ctas * nset * rb * (splits + g2)          # split partials, then the group partials
    part, ml, cnt = _workspace(device, n_rows * hps * D, n_rows * 2, max(ctas * (g2 + 1), 1024))
    page_table = data["page_table"]
    theta = cfg["rope_theta"]
    grid = (ctas, splits)
    # partial sets merged per load group: wide for tiny batches (one CTA per SM, registers are free)
    ch_default = (16 if rb <= 16 else 8) if ctas <= 2 else (8 if rb <= 16 else (4 if rb <= 64 else 1))
    ch = min(max(g1, g2), int(plan.get("ch") or ch_default))
    extra = {}
    if plan.get("maxnreg"):
        extra["maxnreg"] = plan["maxnreg"]
    _attn_decode_kernel[grid](
        qkv, data["cache_loc"], data["seq_lens"], page_table,
        data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"],
        out, part, ml, cnt,
        page_table.shape[1], cfg["qk_scale_factor"], cfg["sm_scale"] * LOG2E,
        math.log2(theta) if theta else 0.0,
        L=L, HQ=HQ, HKV=HKV, D=D, PS=cfg["page_size"], WINDOW=cfg["window"],
        ROPE=bool(theta), SPLITS=splits, RB=rb, LB=lb, BLOCK_T=plan["block_t"], ACT=ACT_DTYPE, FP8_ASM=FP8_ASM,
        CH=ch, QK_FP8=qk_fp8, HP=hp, BD=bd, EX2H=ex2h, PV8=pv8, G1=g1, PF=FP8_ASM and bool(plan.get("pf", True)),
        ABL=int(plan.get("abl", 0)), PFX=FP8_ASM and hp == 1 and bool(plan.get("pfx", False)),
        num_warps=plan["warps"], num_stages=plan["stages"],
        **extra,
    )
    return out

ATTN_BENCH = False

# ---- shared: moe (muse_moe_w4a16/submission.py) ----
TOPK = 8
LOCAL = 32
POST_NORM_EPS = 1e-8
MAX_M = 1024
NUM_SMS = 132
MOE_OUT_DTYPE = torch.bfloat16
MOE_BACKEND = "cuda"     # "cuda": this file's mma.sync W4A16 GEMM; "marlin": SGLang's W4A16 GEMMs with this file's routing/finalize; "triton": this file's GEMMs
DEQUANT_ASM = True          # inline PTX for the nibble -> fp16 placement (False: plain Triton ops)
SCALE_3D = tl.constexpr(False)   # scale expansion: 3-D broadcast instead of 4 interleaves
SCALE_VARIANT = tl.constexpr(4)  # scaled-asm dequant: 2 = interleave-expanded scales, 3 = broadcast-reshape
SCALE_FIX = 2.0 ** 14       # folded into the global scales: the placement trick yields value * 2^-14


# ----------------------------------------------------------------------------------------------
# 1. routing
# ----------------------------------------------------------------------------------------------


@triton.jit
def _route_kernel(
    ids_ptr, x_ptr, x16_ptr,
    sorted_tok_ptr, row_of_pair_ptr, blk_e_ptr, blk_row0_ptr, blk_end_ptr, nblk_ptr,
    sorted_pair_ptr, lids_ptr, npad_ptr,
    M, local_start, n_elems, n_route_ctas, xscale,
    PB: tl.constexpr, CH: tl.constexpr, BM: tl.constexpr, MAXB: tl.constexpr, E: tl.constexpr,
    TOPK: tl.constexpr, XBLOCK: tl.constexpr, MARLIN: tl.constexpr, MULTI: tl.constexpr,
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
        tl.store(x16_ptr + offs, (v.to(tl.float32) * xscale).to(tl.float16), mask=m)   # xscale: a power of two
        return
    npairs = M * TOPK
    ex = tl.arange(0, E)
    i = pid * PB + tl.arange(0, PB)
    valid = i < npairs
    e = tl.load(ids_ptr + i, mask=valid, other=0) - local_start
    is_local = valid & (e >= 0) & (e < E)
    e_key = tl.where(is_local, e, E)                       # E: not this GPU's
    onehot = (e_key[:, None] == ex[None, :]).to(tl.int32)   # [PB, E]
    my0 = pid * PB
    counts = tl.zeros((E,), dtype=tl.int32)
    rank = tl.zeros((PB,), dtype=tl.int32)
    for c0 in range(0, npairs, CH):
        ci = c0 + tl.arange(0, CH)
        cv = ci < npairs
        ce = tl.load(ids_ptr + ci, mask=cv, other=0) - local_start
        cl = cv & (ce >= 0) & (ce < E)
        ce_key = tl.where(cl, ce, E)
        cnt = tl.sum((ce_key[None, :] == ex[:, None]).to(tl.int32), 1)     # per-expert pairs in this chunk
        counts += cnt
        if MULTI:
            if c0 + CH <= my0:                              # chunk entirely before this CTA's pairs
                rank += tl.sum(onehot * cnt[None, :], 1)
            else:
                if c0 <= my0:                               # the chunk holding this CTA's pairs: its earlier pairs
                    rank += tl.sum(((ce_key[None, :] == e_key[:, None]) & (ci[None, :] < my0)).to(tl.int32), 1)
        else:                                               # one chunk: direct earlier-pair compare
            rank += tl.sum(((ce_key[None, :] == e_key[:, None]) & (ci[None, :] < i[:, None])).to(tl.int32), 1)
    if MULTI:   # earlier pairs of the same expert within this CTA's own range
        rank += tl.sum(((e_key[None, :] == e_key[:, None]) & (i[None, :] < i[:, None])).to(tl.int32), 1)
    padded = (counts + BM - 1) // BM * BM
    pad_off = tl.cumsum(padded, 0) - padded                 # padded row offset per expert
    row = tl.sum(onehot * pad_off[None, :], 1) + rank
    tl.store(sorted_tok_ptr + row, i // TOPK, mask=is_local)
    tl.store(row_of_pair_ptr + i, tl.where(is_local, row, -1), mask=valid)
    if MARLIN:
        # Marlin's moe_align_block_size format: pair indices grouped by expert, padding = npairs
        tl.store(sorted_pair_ptr + row, i, mask=is_local)
        tl.store(lids_ptr + i, tl.where(is_local, e, -1), mask=valid)
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
        if MARLIN:
            tl.store(npad_ptr, total * BM)
            pj = tl.arange(0, BM)
            prow = (pad_off + counts)[:, None] + pj[None, :]                       # [E, BM]
            pm = pj[None, :] < (padded - counts)[:, None]
            tl.store(sorted_pair_ptr + prow, tl.full((E, BM), 0, tl.int32) + npairs, mask=pm)


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


_NIBBLE_SCALED_ASM = tl.constexpr("""
{
    .reg .b32 a, b, t, u, z;
    mov.b32 z, 0;
    prmt.b32 a, $4, z, 0x4140;
    prmt.b32 b, $4, z, 0x4342;
    shl.b32 t, a, 9;    and.b32 t, t, 0x0E000E00;
    shl.b32 u, a, 12;   and.b32 u, u, 0x80008000;  or.b32 t, t, u;   mul.f16x2 $0, t, $5;
    shl.b32 t, b, 9;    and.b32 t, t, 0x0E000E00;
    shl.b32 u, b, 12;   and.b32 u, u, 0x80008000;  or.b32 t, t, u;   mul.f16x2 $1, t, $5;
    shl.b32 t, a, 5;    and.b32 t, t, 0x0E000E00;
    shl.b32 u, a, 8;    and.b32 u, u, 0x80008000;  or.b32 t, t, u;   mul.f16x2 $2, t, $5;
    shl.b32 t, b, 5;    and.b32 t, t, 0x0E000E00;
    shl.b32 u, b, 8;    and.b32 u, u, 0x80008000;  or.b32 t, t, u;   mul.f16x2 $3, t, $5;
}
""")


@triton.jit
def _dequant_scaled(wq, s8, gs16, VARIANT: tl.constexpr):
    """Scale applied inside the nibble asm (packed f16x2), so the result keeps the asm output
    layout and feeds the MMA without a shared-memory round trip.
    wq uint8 [BN, BK/2]; s8: VARIANT 1 -> fp8 [BN, BK/2] (each group's scale gathered per byte),
    VARIANT 2 -> fp8 [BN, BK/16] expanded here by interleaves. gs16: fp16 global scale x 2^14."""
    if VARIANT == 1:
        s = _cache_scale_to_f16(s8) * gs16                       # [BN, BK/2] fp16
    elif VARIANT == 3:
        s = (s8.to(tl.float32).to(tl.float16)) * gs16            # [BN, BK/16]
        s = tl.reshape(tl.broadcast_to(s[:, :, None], (s.shape[0], s.shape[1], 8)), (s.shape[0], s.shape[1] * 8))
    else:   # 2 and 4 (4: the scales were loaded pair-blocked, see SF_BLOCKED)
        s = (s8.to(tl.float32).to(tl.float16)) * gs16            # [BN, BK/16]
        s = tl.interleave(s, s)
        s = tl.interleave(s, s)
        s = tl.interleave(s, s)                                  # [BN, BK/2]
    lo, hi = tl.inline_asm_elementwise(
        asm=_NIBBLE_SCALED_ASM, constraints="=r,=r,=r,=r,r,r,r", args=[wq, s],
        dtype=(tl.float16, tl.float16), is_pure=True, pack=4)
    return tl.interleave(lo, hi)


@triton.jit
def _cache_scale_to_f16(x8):
    """fp8 e4m3 -> fp16, one cvt per pair, keeping the input's layout."""
    b = x8.to(tl.uint8, bitcast=True)
    return tl.inline_asm_elementwise(
        asm="{ .reg .b16 lo, hi; mov.b32 {lo, hi}, $2; cvt.rn.f16x2.e4m3x2 $0, lo; cvt.rn.f16x2.e4m3x2 $1, hi; }",
        constraints="=r,=r,r", args=[b], dtype=tl.float16, is_pure=True, pack=4)


@triton.jit
def _dequant(wq, sf, gs, ASM: tl.constexpr, MODE: tl.constexpr = 0):
    """wq uint8 [BN, BK/2] packed E2M1, sf fp8 [BN, BK/16], gs fp32 (x 2^14) -> fp16 [BN, BK]."""
    if MODE == 5:
        lo = wq.to(tl.float16)
        hi = lo
    elif ASM:
        lo, hi = tl.inline_asm_elementwise(
            asm=_NIBBLE_ASM, constraints="=r,=r,=r,=r,r", args=[wq],
            dtype=(tl.float16, tl.float16), is_pure=True, pack=4)
    else:
        lo = (wq & 0x0F).to(tl.int16)
        hi = (wq >> 4).to(tl.int16)
        lo = (((lo & 7) << 9) | ((lo & 8) << 12)).to(tl.float16, bitcast=True)
        hi = (((hi & 7) << 9) | ((hi & 8) << 12)).to(tl.float16, bitcast=True)
    w = tl.interleave(lo, hi)                                   # [BN, BK]
    if MODE == 4:
        return w
    s = (sf.to(tl.float32) * gs).to(tl.float16)                 # [BN, BK/16]
    if MODE == 6 or SCALE_3D:
        # one scale per 16 consecutive K elements: broadcast over a [BN, BK/16, 16] view
        w3 = tl.reshape(w, (w.shape[0], w.shape[1] // 16, 16))
        return tl.reshape(w3 * s[:, :, None], (w.shape[0], w.shape[1]))
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
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, MODE: tl.constexpr, SF_BLOCKED: tl.constexpr,
):
    # MODE (benchmark ablations; outputs are wrong unless 0): 1 = fp16 weights read from w_ptr
    # (no dequant), 2 = dequant but no dot, 3 = loads only
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
        if MODE == 7:   # scale gathered per packed byte: [BN, BK/2]
            s_ptrs = sf_ptr + e * stride_se + offs_n[:, None] * stride_sn + (offs_kb // 8)[None, :]
        elif SF_BLOCKED:
            # scales stored as [N/8, G/2, 8, 2] (16-byte blocks): pairs are contiguous, so the
            # load vectorizes by 2 and, after three interleaves, lands in the packed tile's layout
            G = K // 16
            s_ptrs = (sf_ptr + e * stride_se + (offs_n // 8)[:, None] * (G * 8) + (offs_n % 8)[:, None] * 2
                      + (offs_ks // 2)[None, :] * 16 + (offs_ks % 2)[None, :])
        else:
            s_ptrs = sf_ptr + e * stride_se + offs_n[:, None] * stride_sn + offs_ks[None, :]
        gs16 = gs.to(tl.float16)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        if GATED:
            w2_ptrs = w_ptrs + up_rows * stride_wn
            s2_ptrs = s_ptrs + up_rows * stride_sn
            acc2 = tl.zeros((BM, BN), dtype=tl.float32)
        if MODE == 1:
            wf_ptrs = w_ptr.to(tl.pointer_type(tl.float16)) + e * (stride_we // 2) + offs_n[:, None] * (stride_wn // 2) + offs_k[None, :]
            if GATED:
                wf2_ptrs = wf_ptrs + up_rows * (stride_wn // 2)
        for _ in range(0, K, BK):
            a = tl.load(a_ptrs, mask=rmask[:, None], other=0.0)
            if MODE == 1:
                w = tl.load(wf_ptrs)
                acc = tl.dot(a, tl.trans(w), acc)
                wf_ptrs += BK
                if GATED:
                    w2 = tl.load(wf2_ptrs)
                    acc2 = tl.dot(a, tl.trans(w2), acc2)
                    wf2_ptrs += BK
            else:
                wq = tl.load(w_ptrs)
                sf = tl.load(s_ptrs)
                if MODE == 3:
                    acc += tl.sum(wq.to(tl.float32), 1)[None, :] + tl.sum(sf.to(tl.float32), 1)[None, :]
                elif MODE == 11:
                    # per-16-group loads and dots; the group's scale multiplies the fp32 dot output
                    for g in tl.static_range(BK // 16):
                        wq_g = tl.load(w_ptrs + g * 8)                                      # [BN, 8] bytes
                        a_g = tl.load(a_ptrs + g * 16, mask=rmask[:, None], other=0.0)      # [BM, 16]
                        s_g = tl.sum(tl.where(offs_ks[None, :] == g, sf.to(tl.float32), 0.0), 1) * gs   # [BN]
                        w_g = _dequant(wq_g, sf, gs, ASM, 4)                                # [BN, 16] unscaled
                        acc += tl.dot(a_g, tl.trans(w_g)) * s_g[None, :]
                elif MODE == 9:
                    # per-16-group batched dot; scales applied to the dot output (fp32)
                    w = _dequant(wq, sf, gs, ASM, 4)                                  # [BN, BK] unscaled
                    w3 = tl.permute(tl.reshape(w, (BN, BK // 16, 16)), (1, 2, 0))     # [G, 16, BN]
                    a3 = tl.permute(tl.reshape(a, (BM, BK // 16, 16)), (1, 0, 2))     # [G, BM, 16]
                    o3 = tl.dot(a3, w3)                                               # [G, BM, BN]
                    s3 = (sf.to(tl.float32) * gs)                                     # [BN, G]
                    acc += tl.sum(o3 * tl.permute(s3, (1, 0))[:, None, :], 0)
                elif MODE == 7 or MODE == 8 or MODE == 10:
                    w = _dequant_scaled(wq, sf, gs16, 1 if MODE == 7 else (3 if MODE == 10 else 2))
                    acc = tl.dot(a, tl.trans(w), acc)
                elif MODE == 0 and ASM:
                    w = _dequant_scaled(wq, sf, gs16, SCALE_VARIANT)
                    acc = tl.dot(a, tl.trans(w), acc)
                else:
                    w = _dequant(wq, sf, gs, ASM, MODE)
                    if MODE == 2:
                        acc += tl.sum(w.to(tl.float32), 1)[None, :]
                    else:
                        acc = tl.dot(a, tl.trans(w), acc)
                if GATED:
                    wq2 = tl.load(w2_ptrs)
                    sf2 = tl.load(s2_ptrs)
                    if MODE == 3:
                        acc2 += tl.sum(wq2.to(tl.float32), 1)[None, :] + tl.sum(sf2.to(tl.float32), 1)[None, :]
                    elif MODE == 11:
                        for g in tl.static_range(BK // 16):
                            wq_g = tl.load(w2_ptrs + g * 8)
                            a_g = tl.load(a_ptrs + g * 16, mask=rmask[:, None], other=0.0)
                            s_g = tl.sum(tl.where(offs_ks[None, :] == g, sf2.to(tl.float32), 0.0), 1) * gs
                            w_g = _dequant(wq_g, sf2, gs, ASM, 4)
                            acc2 += tl.dot(a_g, tl.trans(w_g)) * s_g[None, :]
                    elif MODE == 9:
                        w2 = _dequant(wq2, sf2, gs, ASM, 4)
                        w3 = tl.permute(tl.reshape(w2, (BN, BK // 16, 16)), (1, 2, 0))
                        a3 = tl.permute(tl.reshape(a, (BM, BK // 16, 16)), (1, 0, 2))
                        o3 = tl.dot(a3, w3)
                        s3 = (sf2.to(tl.float32) * gs)
                        acc2 += tl.sum(o3 * tl.permute(s3, (1, 0))[:, None, :], 0)
                    elif MODE == 7 or MODE == 8 or MODE == 10:
                        w2 = _dequant_scaled(wq2, sf2, gs16, 1 if MODE == 7 else (3 if MODE == 10 else 2))
                        acc2 = tl.dot(a, tl.trans(w2), acc2)
                    elif MODE == 0 and ASM:
                        w2 = _dequant_scaled(wq2, sf2, gs16, SCALE_VARIANT)
                        acc2 = tl.dot(a, tl.trans(w2), acc2)
                    else:
                        w2 = _dequant(wq2, sf2, gs, ASM, MODE)
                        if MODE == 2:
                            acc2 += tl.sum(w2.to(tl.float32), 1)[None, :]
                        else:
                            acc2 = tl.dot(a, tl.trans(w2), acc2)
                    w2_ptrs += BK // 2
                    s2_ptrs += (BK // 2) // 8 if MODE == 7 else (BK // 2 if SF_BLOCKED else BK // 16)
                w_ptrs += BK // 2
                s_ptrs += (BK // 2) // 8 if MODE == 7 else (BK // 2 if SF_BLOCKED else BK // 16)
            a_ptrs += BK
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


@triton.jit
def _finalize_pairs_kernel(y_ptr, lids_ptr, tw_ptr, nw_ptr, out_ptr, eps, H: tl.constexpr, TOPK: tl.constexpr):
    t = tl.program_id(0)
    d = tl.arange(0, H)
    nw = tl.load(nw_ptr + d)
    acc = tl.zeros((H,), dtype=tl.float32)
    for s in tl.static_range(TOPK):
        ok = tl.load(lids_ptr + t * TOPK + s) >= 0
        y = tl.load(y_ptr + (t * TOPK + s) * H + d, mask=ok & (d < H), other=0.0).to(tl.float32) * nw
        ms = tl.sum(y * y, 0) / H
        y = y * tl.rsqrt(ms + eps)
        g = tl.load(tw_ptr + t * TOPK + s)
        acc += tl.where(ok, g, 0.0) * y
    tl.store(out_ptr + t * H + d, acc.to(out_ptr.dtype.element_ty))


# ----------------------------------------------------------------------------------------------
# CUDA W4A16 GEMM (mma.sync, weights repacked into the B-fragment nibble order; see cuda_moe.py)
# ----------------------------------------------------------------------------------------------

# == BEGIN CUDA ==
CUDA_SRC = r"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(addr));
}

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// two E4M3 bytes (packed in the low 16 bits) -> f16x2
__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint32_t packed16) {
    uint32_t r;
    asm("{ .reg .b16 h; cvt.u16.u32 h, %1; cvt.rn.f16x2.e4m3x2 %0, h; }\n" : "=r"(r) : "r"(packed16));
    return r;
}

__device__ __forceinline__ uint32_t f16x2_mul(uint32_t a, uint32_t b) {
    uint32_t r;
    asm("mul.f16x2 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

// 8 nibbles (one 32-bit word: two k16 steps) -> the two steps' B fragments (b0, b1 each), scaled.
// Word layout (per step: 16 bits): byte0 = e(2t) | e(2t+8) << 4, byte1 = e(2t+1) | e(2t+9) << 4.
__device__ __forceinline__ void dequant_word(uint32_t w, uint32_t s_lo, uint32_t s_hi,
                                             uint32_t& b0_s0, uint32_t& b1_s0, uint32_t& b0_s1, uint32_t& b1_s1) {
    uint32_t lo = w & 0x0F0F0F0Fu;                     // e(2t), e(2t+1) per step
    uint32_t hi = (w >> 4) & 0x0F0F0F0Fu;              // e(2t+8), e(2t+9) per step
    uint32_t elo = ((lo & 0x08080808u) << 4) | ((lo & 0x07070707u) << 2);
    uint32_t ehi = ((hi & 0x08080808u) << 4) | ((hi & 0x07070707u) << 2);
    b0_s0 = f16x2_mul(e4m3x2_to_f16x2(elo & 0xFFFFu), s_lo);
    b1_s0 = f16x2_mul(e4m3x2_to_f16x2(ehi & 0xFFFFu), s_lo);
    b0_s1 = f16x2_mul(e4m3x2_to_f16x2(elo >> 16), s_hi);
    b1_s1 = f16x2_mul(e4m3x2_to_f16x2(ehi >> 16), s_hi);
}

constexpr int KC = 128;          // K chunk
constexpr int A_STRIDE = KC + 8;  // padded smem row stride (halves)
constexpr int NWARPS = 4;
constexpr int NW_MAX = 4;
constexpr int NSTAGE = 2;         // A tile cp.async stages; B registers prefetched one chunk ahead

// MT: m16 tiles per block (rows = 16*MT); NPER: n8 blocks per warp (1 or 2).
// GATED: warp w's n8 blocks are gate blocks nb (NPER of them, contiguous) and the matching up
// blocks nb + NB/2; the CTA covers 8*NPER*4 output columns. Else: 8*NPER*4 consecutive columns.
template <bool GATED, int MT, int NPER, int KS>
__global__ void __launch_bounds__(128 * KS)
moe_w4a16_gemm_kernel(const __half* __restrict__ A, int lda,
                      const int* __restrict__ sorted_tok,
                      const int* __restrict__ blk_e, const int* __restrict__ blk_row0,
                      const int* __restrict__ blk_end, const int* __restrict__ nblk,
                      const uint4* __restrict__ Wr, const uint2* __restrict__ Sr, const float* __restrict__ gs,
                      __half* __restrict__ out16, __nv_bfloat16* __restrict__ out_bf, float* __restrict__ out_f32, int ldo,
                      int K, int N)
{
    // split-K across CTAs: blockIdx.z owns chunks [z*NC/KSPLIT, (z+1)*NC/KSPLIT); out_f32 != null
    const int KSPLIT = gridDim.z;
    constexpr int BM = 16 * MT;
    constexpr int NW = GATED ? 2 * NPER : NPER;     // weight n8 blocks per warp
    const int j = blockIdx.y;
    if (j >= *nblk) return;
    const int e = blk_e[j];
    const int row0 = blk_row0[j];
    const int end = blk_end[j];
    const int tile = blockIdx.x;
    const int gidx = threadIdx.x >> 7;                       // K-split group
    const int tid = threadIdx.x & 127;
    const int warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;

    __shared__ __align__(16) __half As_all[KS][NSTAGE][BM * A_STRIDE];
    __shared__ float red[KS > 1 ? NW_MAX * MT * 4 * 128 : 1];
    auto As = As_all[gidx];

    const int NB = N / 8;
    const int NC_ALL = K / KC;
    const int NC = NC_ALL / KSPLIT;                      // this CTA's chunks
    const int c_base = blockIdx.z * NC;
    int nbs[NW];
    #pragma unroll
    for (int i = 0; i < NPER; ++i) {
        if (GATED) { nbs[i] = (tile * NWARPS + warp) * NPER + i; nbs[NPER + i] = nbs[i] + NB / 2; }
        else       { nbs[i] = (tile * NWARPS + warp) * NPER + i; }
    }
    const uint4* Wp[NW]; const uint2* Sp[NW];
    #pragma unroll
    for (int i = 0; i < NW; ++i) {
        Wp[i] = Wr + ((size_t)e * NB + nbs[i]) * NC * 32 + lane;
        Sp[i] = Sr + ((size_t)e * NB + nbs[i]) * NC * 8 + g;
    }
    const __half gsh = __float2half(gs[e]);
    const uint32_t gs2 = (uint32_t)__half_as_ushort(gsh) * 0x00010001u;

    // A rows: 128 threads copy BM rows x 256B per chunk: thread handles rows r = tid/8 + 16*i
    const int aseg = tid & 7;
    int atok[MT]; bool avalid[MT];
    #pragma unroll
    for (int i = 0; i < MT; ++i) {
        const int r = row0 + (tid >> 3) + 16 * i;
        avalid[i] = r < end;
        atok[i] = avalid[i] ? (GATED ? sorted_tok[r] : r) : 0;
    }
    auto issue_A = [&](int c, int buf) {
        #pragma unroll
        for (int i = 0; i < MT; ++i) {
            const int rloc = (tid >> 3) + 16 * i;
            const __half* Ag = A + (size_t)atok[i] * lda + (c_base + c) * KC;
            #pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int col = (aseg + 8 * q) * 8;
                cp_async16(smem_u32(&As[buf][rloc * A_STRIDE + col]), Ag + col, avalid[i] ? 16 : 0);
            }
        }
    };

    float acc[NW][MT][4];
    #pragma unroll
    for (int i = 0; i < NW; ++i)
        #pragma unroll
        for (int m = 0; m < MT; ++m)
            #pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][m][q] = 0.f;

    uint4 wq[NW]; uint2 sq[NW];
    #pragma unroll
    for (int i = 0; i < NW; ++i) { wq[i] = Wp[i][(c_base + gidx) * 32]; sq[i] = Sp[i][(c_base + gidx) * 8]; }
    #pragma unroll
    for (int st = 0; st < NSTAGE - 1; ++st) {
        if (gidx + st * KS < NC) issue_A(gidx + st * KS, st);
        cp_async_commit();
    }

    // named barrier per K-split group (128 threads each)
    auto group_sync = [&]() { asm volatile("bar.sync %0, 128;\n" :: "r"(gidx + 1)); };

    int it = 0;
    for (int c = gidx; c < NC; c += KS, ++it) {
        const int buf = it % NSTAGE;
        const int cn = c + (NSTAGE - 1) * KS;
        if (cn < NC) issue_A(cn, (it + NSTAGE - 1) % NSTAGE);
        cp_async_commit();
        uint4 wn[NW]; uint2 sn[NW];
        if (c + KS < NC) {
            #pragma unroll
            for (int i = 0; i < NW; ++i) { wn[i] = Wp[i][(c_base + c + KS) * 32]; sn[i] = Sp[i][(c_base + c + KS) * 8]; }
        }
        cp_async_wait<NSTAGE - 1>();
        group_sync();
        uint32_t sc[NW][4];
        #pragma unroll
        for (int i = 0; i < NW; ++i) {
            sc[i][0] = f16x2_mul(e4m3x2_to_f16x2(sq[i].x & 0xFFFFu), gs2);
            sc[i][1] = f16x2_mul(e4m3x2_to_f16x2(sq[i].x >> 16), gs2);
            sc[i][2] = f16x2_mul(e4m3x2_to_f16x2(sq[i].y & 0xFFFFu), gs2);
            sc[i][3] = f16x2_mul(e4m3x2_to_f16x2(sq[i].y >> 16), gs2);
        }
        #pragma unroll
        for (int p = 0; p < 4; ++p) {           // word p = k16 steps 2p, 2p+1
            uint32_t a[2][MT][4];
            #pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int kk = (2 * p + q) * 16 + (lane >> 4) * 8;
                #pragma unroll
                for (int m = 0; m < MT; ++m) {
                    const int r = 16 * m + (lane & 15);
                    ldmatrix_x4(a[q][m][0], a[q][m][1], a[q][m][2], a[q][m][3], smem_u32(&As[buf][r * A_STRIDE + kk]));
                }
            }
            #pragma unroll
            for (int i = 0; i < NW; ++i) {
                const uint32_t w = p == 0 ? wq[i].x : p == 1 ? wq[i].y : p == 2 ? wq[i].z : wq[i].w;
                const uint32_t slo = (sc[i][p] & 0xFFFFu) * 0x00010001u, shi = (sc[i][p] >> 16) * 0x00010001u;
                uint32_t b0a, b1a, b0b, b1b;
                dequant_word(w, slo, shi, b0a, b1a, b0b, b1b);
                #pragma unroll
                for (int m = 0; m < MT; ++m) {
                    mma16816(acc[i][m], a[0][m], b0a, b1a);
                    mma16816(acc[i][m], a[1][m], b0b, b1b);
                }
            }
        }
        group_sync();
        if (c + KS < NC) {
            #pragma unroll
            for (int i = 0; i < NW; ++i) { wq[i] = wn[i]; sq[i] = sn[i]; }
        }
    }
    if (KS > 1) {
        // sum the groups' accumulators: group 1 writes, group 0 adds
        __syncthreads();
        if (gidx == 1) {
            #pragma unroll
            for (int i = 0; i < NW; ++i)
                #pragma unroll
                for (int m = 0; m < MT; ++m)
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) red[((i * MT + m) * 4 + q) * 128 + tid] = acc[i][m][q];
        }
        __syncthreads();
        if (gidx == 1) return;
        #pragma unroll
        for (int i = 0; i < NW; ++i)
            #pragma unroll
            for (int m = 0; m < MT; ++m)
                #pragma unroll
                for (int q = 0; q < 4; ++q) acc[i][m][q] += red[((i * MT + m) * 4 + q) * 128 + tid];
    }

    if (out_f32 != nullptr) {
        #pragma unroll
        for (int m = 0; m < MT; ++m)
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = row0 + 16 * m + g + 8 * h;
                if (r >= end) continue;
                #pragma unroll
                for (int i = 0; i < NW; ++i) {
                    const int col = nbs[i] * 8 + 2 * t;        // GATED: gate cols, then up cols at +N/2
                    atomicAdd(out_f32 + (size_t)r * ldo + col, acc[i][m][2 * h]);
                    atomicAdd(out_f32 + (size_t)r * ldo + col + 1, acc[i][m][2 * h + 1]);
                }
            }
        return;
    }
    #pragma unroll
    for (int m = 0; m < MT; ++m) {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = row0 + 16 * m + g + 8 * h;
            if (r >= end) continue;
            if (GATED) {
                #pragma unroll
                for (int i = 0; i < NPER; ++i) {
                    const int col = nbs[i] * 8 + 2 * t;
                    float g0 = acc[i][m][2 * h], g1 = acc[i][m][2 * h + 1];
                    float u0 = acc[NPER + i][m][2 * h], u1 = acc[NPER + i][m][2 * h + 1];
                    float h0 = g0 / (1.f + __expf(-g0)) * u0, h1 = g1 / (1.f + __expf(-g1)) * u1;
                    __half2 v = __halves2half2(__float2half(__bfloat162float(__float2bfloat16(h0))),
                                               __float2half(__bfloat162float(__float2bfloat16(h1))));
                    *reinterpret_cast<__half2*>(out16 + (size_t)r * ldo + col) = v;
                }
            } else {
                #pragma unroll
                for (int i = 0; i < NW; ++i) {
                    const int col = nbs[i] * 8 + 2 * t;
                    *reinterpret_cast<__nv_bfloat162*>(out_bf + (size_t)r * ldo + col) =
                        __floats2bfloat162_rn(acc[i][m][2 * h], acc[i][m][2 * h + 1]);
                }
            }
        }
    }
}

__global__ void silu_mul_f32_kernel(const float* __restrict__ gu, int ldgu, __half* __restrict__ out, int ldo,
                                    const int* __restrict__ blk_row0, const int* __restrict__ blk_end, const int* __restrict__ nblk,
                                    int BM, int I)
{
    const int j = blockIdx.y;
    if (j >= *nblk) return;
    const int r = blk_row0[j] + blockIdx.x;
    if (blockIdx.x >= BM || r >= blk_end[j]) return;
    for (int c = threadIdx.x * 2; c < I; c += blockDim.x * 2) {
        const float2 gg = *reinterpret_cast<const float2*>(gu + (size_t)r * ldgu + c);
        const float2 uu = *reinterpret_cast<const float2*>(gu + (size_t)r * ldgu + I + c);
        const float h0 = gg.x / (1.f + __expf(-gg.x)) * uu.x, h1 = gg.y / (1.f + __expf(-gg.y)) * uu.y;
        *reinterpret_cast<__half2*>(out + (size_t)r * ldo + c) = __halves2half2(
            __float2half(__bfloat162float(__float2bfloat16(h0))), __float2half(__bfloat162float(__float2bfloat16(h1))));
    }
}

__global__ void f32_to_bf16_rows_kernel(const float* __restrict__ src, int lds, __nv_bfloat16* __restrict__ dst, int ldd,
                                        const int* __restrict__ blk_row0, const int* __restrict__ blk_end, const int* __restrict__ nblk,
                                        int BM, int H)
{
    const int j = blockIdx.y;
    if (j >= *nblk) return;
    const int r = blk_row0[j] + blockIdx.x;
    if (blockIdx.x >= BM || r >= blk_end[j]) return;
    for (int c = threadIdx.x * 2; c < H; c += blockDim.x * 2) {
        const float2 v = *reinterpret_cast<const float2*>(src + (size_t)r * lds + c);
        *reinterpret_cast<__nv_bfloat162*>(dst + (size_t)r * ldd + c) = __floats2bfloat162_rn(v.x, v.y);
    }
}

template <bool GATED, int MT, int NPER, int KS>
void launch(const __half* A, int lda, const int* st, const int* be, const int* br, const int* bend, const int* nb,
            const uint4* Wr, const uint2* Sr, const float* gs, __half* o16, __nv_bfloat16* obf, float* of32, int ldo, int K, int N,
            int max_blocks, int ksplit, cudaStream_t stream)
{
    const int cols = 8 * NPER * NWARPS;                 // output columns per CTA
    const int n_tiles = GATED ? (N / 2) / cols : N / cols;
    dim3 grid(n_tiles, max_blocks, ksplit);
    moe_w4a16_gemm_kernel<GATED, MT, NPER, KS><<<grid, 128 * KS, 0, stream>>>(A, lda, st, be, br, bend, nb, Wr, Sr, gs, o16, obf, of32, ldo, K, N);
}


// ---------------------------------------------------------------------------------------------
// Weights-as-A GEMM for small expert blocks (<= 8*NT tokens): out[tok][n] = sum_k W[n][k] x[tok][k]
// with the weight rows as the mma M dimension and the tokens as N, so a block of <= 8 tokens
// costs one m16n8k16 per 16x16 weight tile (no 16-row token padding). One CTA = 4 warps x MW
// m16 weight tiles (GATED: MW gate tiles + the matching MW up tiles) x 8*NT tokens of one
// expert block; K streamed in 128-wide chunks through an NSTAGE cp.async ring (weights, scales
// and the token tile); KS = 2 splits each chunk's 8 k16 steps across two 4-warp groups.
//
// Weight words (repack_weights_wt): per lane and k16 step one 32-bit word holds the 8 nibbles of its
// A fragment; byte b carries nibble A_b pre-spread into E4M3 position (s<<7 | e1<<4 | e0<<3 | m<<2,
// exactly e2m1 * 2^-6) and nibble B_b at (e1<<6 | s<<5 | e0<<1 | m), so the dequant is
//   elo = w & 0x9C9C9C9C                                   bytes A_0..A_3
//   ehi = (w << 2) & 0x8C8C8C8C | (w >> 2) & 0x10101010    bytes B_0..B_3 (now E4M3-placed)
// and four cvt.rn.f16x2.e4m3x2: (A_0,A_1) = a0 rows g k 2t,2t+1; (A_2,A_3) = a1 rows g+8; (B_0,B_1) = a2
// rows g k 2t+8,2t+9; (B_2,B_3) = a3 rows g+8. Each pair is multiplied by its row's E4M3 block scale (as fp16:
// the product has <= 6 significant bits, so the fp16 weights are exact) and gs * 2^6 scales the fp32 sums.
// ---------------------------------------------------------------------------------------------

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
}
__device__ __forceinline__ uint32_t e4m3x2_lo16_to_f16x2(uint32_t w) {
    uint32_t r;
    asm("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %1; cvt.rn.f16x2.e4m3x2 %0, lo; }\n" : "=r"(r) : "r"(w));
    return r;
}
__device__ __forceinline__ uint32_t e4m3x2_hi16_to_f16x2(uint32_t w) {
    uint32_t r;
    asm("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %1; cvt.rn.f16x2.e4m3x2 %0, hi; }\n" : "=r"(r) : "r"(w));
    return r;
}

constexpr int WT_NWARPS = 4;
__host__ __device__ constexpr int wt_stage_bytes(int tpw, int nt) {
    return tpw * WT_NWARPS * 1024 + tpw * WT_NWARPS * 128 + 8 * nt * A_STRIDE * 2;
}

template <bool GATED, int MW, int NT, int KS, int NSTAGE, bool SPLITK>
__global__ void __launch_bounds__(128 * KS)
moe_w4a16_wt_kernel(const __half* __restrict__ X, int ldx,
                    const int* __restrict__ sorted_tok,
                    const int* __restrict__ blk_e, const int* __restrict__ blk_row0,
                    const int* __restrict__ blk_end, const int* __restrict__ nblk,
                    const uint8_t* __restrict__ Wr, const uint8_t* __restrict__ Sr, const float* __restrict__ gs,
                    __half* __restrict__ out16, __nv_bfloat16* __restrict__ out_bf, int ldo,
                    float* __restrict__ part, int ldp, int part_rows, int* __restrict__ counters,
                    int K, int NTILES, float gs_mul)
{
    // gridDim.z > 1: split-K; CTA z owns chunks [z*NC/KSPLIT, (z+1)*NC/KSPLIT) and writes its fp32 partial
    // sums (times gs) to part[z][row][col] (GATED: gate at col, up at col + N/2); the last CTA to arrive at
    // counters[(y, x)] sums the slices and writes the output (the counter resets itself).
    constexpr int TPW = (GATED ? 2 : 1) * MW;        // m16 weight tiles per warp
    constexpr int TPC = TPW * WT_NWARPS;              // per CTA
    constexpr int W_STAGE = TPC * 1024;
    constexpr int S_STAGE = TPC * 128;
    constexpr int STAGE_BYTES = wt_stage_bytes(TPW, NT);
    constexpr int NTHR = 128 * KS;
    constexpr int CH = NT == 1 ? 2 : 1;               // accumulator chains per tile (latency)
    extern __shared__ __align__(16) uint8_t wt_smem[];

    const int j = blockIdx.y;
    if (j >= *nblk) return;
    const int e = blk_e[j];
    const int row0 = blk_row0[j];
    const int nrows = blk_end[j] - row0;              // valid tokens in this block (<= 8*NT)
    const int ntb = (nrows + 7) >> 3;                 // token n8 blocks in use
    const int tid = threadIdx.x;
    const int warp = (tid >> 5) & 3, lane = tid & 31, gidx = tid >> 7;
    const int g = lane >> 2, t = lane & 3;
    const int KSPLIT = SPLITK ? (int)gridDim.z : 1;
    const int NC_ALL = K / KC;
    const int NC = NC_ALL / KSPLIT;                   // chunks of this CTA
    const int c0 = SPLITK ? (int)blockIdx.z * NC : 0;

    // CTA tile ti (0..TPC-1) -> global m16 tile index
    auto tile_of = [&](int ti) {
        const int w = ti / TPW, i = ti % TPW;
        int tile = (blockIdx.x * WT_NWARPS + w) * MW + (i % MW);
        if (GATED && i >= MW) tile += NTILES / 2;
        return tile;
    };
    const size_t ebase = (size_t)e * NTILES;

    // cp.async segments of this thread, per chunk: WSEG weight segments (16B), one scale segment
    // (threads < TPC*8) and XSEG token-row segments; sources advance by a constant per chunk.
    constexpr int WSEG = TPC * 64 / NTHR;
    constexpr int XSEG = (NT * 128 + NTHR - 1) / NTHR;
    const uint8_t* wsrc[WSEG];
    #pragma unroll
    for (int i = 0; i < WSEG; ++i) {
        const int sidx = tid + i * NTHR;
        wsrc[i] = Wr + ((ebase + tile_of(sidx >> 6)) * NC_ALL + c0) * 1024 + (sidx & 63) * 16;
    }
    const uint8_t* ssrc = Sr + ((ebase + tile_of((tid & (TPC * 8 - 1)) >> 3)) * NC_ALL + c0) * 128 + (tid & 7) * 16;
    const __half* xsrc[XSEG]; int xbytes[XSEG];
    #pragma unroll
    for (int i = 0; i < XSEG; ++i) {
        const int sidx = tid + i * NTHR;
        const int r = sidx >> 4;
        const bool valid = (sidx < NT * 128) && (r < nrows);
        const int tok = valid ? (GATED ? sorted_tok[row0 + r] : row0 + r) : 0;
        xsrc[i] = X + (size_t)tok * ldx + c0 * KC + (sidx & 15) * 8;
        xbytes[i] = valid ? 16 : 0;
    }
    const uint32_t smem_base = smem_u32(wt_smem);

    auto issue = [&](int c, int stage) {
        const uint32_t ws = smem_base + stage * STAGE_BYTES;
        const uint32_t ss = ws + W_STAGE;
        const uint32_t xs = ss + S_STAGE;
        #pragma unroll
        for (int i = 0; i < WSEG; ++i)
            cp_async16(ws + (tid + i * NTHR) * 16, wsrc[i] + c * 1024, 16);
        if (tid < TPC * 8) cp_async16(ss + tid * 16, ssrc + c * 128, 16);
        #pragma unroll
        for (int i = 0; i < XSEG; ++i) {
            const int sidx = tid + i * NTHR;
            if (sidx < NT * 128)
                cp_async16(xs + ((sidx >> 4) * A_STRIDE + (sidx & 15) * 8) * 2, xsrc[i] + c * KC, xbytes[i]);
        }
    };

    float acc[TPW][NT * CH][4];
    #pragma unroll
    for (int i = 0; i < TPW; ++i)
        #pragma unroll
        for (int n = 0; n < NT * CH; ++n)
            #pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][n][q] = 0.f;

    const float gsf = gs[e] * gs_mul;                                   // gs * 64: applied to the fp32 sums

    #pragma unroll
    for (int st = 0; st < NSTAGE - 1; ++st) {
        if (st < NC) issue(st, st);
        cp_async_commit();
    }

    for (int c = 0; c < NC; ++c) {
        cp_async_wait<NSTAGE - 2>();
        __syncthreads();
        {
            const int cn = c + NSTAGE - 1;
            if (cn < NC) issue(cn, cn % NSTAGE);
            cp_async_commit();
        }
        const uint8_t* ws = wt_smem + (c % NSTAGE) * STAGE_BYTES;
        const uint8_t* ss = ws + W_STAGE;
        const __half* xs = reinterpret_cast<const __half*>(ss + S_STAGE);
        #pragma unroll
        for (int hh = 0; hh < 2 / KS; ++hh) {
            const int h = KS == 1 ? hh : gidx;                       // half chunk: k16 steps 4h..4h+3
            uint4 wq[TPW]; uint2 sq[TPW];
            #pragma unroll
            for (int i = 0; i < TPW; ++i) {
                wq[i] = *reinterpret_cast<const uint4*>(ws + ((warp * TPW + i) * 2 + h) * 512 + lane * 16);
                sq[i] = *reinterpret_cast<const uint2*>(ss + (warp * TPW + i) * 128 + g * 16 + h * 8);
            }
            #pragma unroll
            for (int ps = 0; ps < 2; ++ps) {                         // steps 2ps, 2ps+1 of the half
                uint32_t b[NT][4];
                #pragma unroll
                for (int nb = 0; nb < NT; ++nb) {
                    if (nb < ntb)
                        ldmatrix_x4(b[nb][0], b[nb][1], b[nb][2], b[nb][3],
                                    smem_u32(xs + (nb * 8 + (lane & 7)) * A_STRIDE + h * 64 + ps * 32 + (lane >> 3) * 8));
                }
                #pragma unroll
                for (int q = 0; q < 2; ++q) {
                    const int s = 2 * ps + q;
                    #pragma unroll
                    for (int i = 0; i < TPW; ++i) {
                        const uint32_t w = s == 0 ? wq[i].x : s == 1 ? wq[i].y : s == 2 ? wq[i].z : wq[i].w;
                        const uint32_t sw = ps == 0 ? sq[i].x : sq[i].y;
                        const uint32_t sp = q == 0 ? e4m3x2_lo16_to_f16x2(sw) : e4m3x2_hi16_to_f16x2(sw);   // (s_g, s_g8)
                        const uint32_t S0 = prmt(sp, sp, 0x1010u);          // row g   block scale, both halves
                        const uint32_t S1 = prmt(sp, sp, 0x3232u);          // row g+8
                        // fp16 weights = e2m1 * 2^-6 * blockscale: <= 6 significant bits, exact
                        const uint32_t elo = w & 0x9C9C9C9Cu;
                        const uint32_t ehi = ((w << 2) & 0x8C8C8C8Cu) | ((w >> 2) & 0x10101010u);
                        uint32_t a[4];
                        a[0] = f16x2_mul(e4m3x2_lo16_to_f16x2(elo), S0);
                        a[1] = f16x2_mul(e4m3x2_hi16_to_f16x2(elo), S1);
                        a[2] = f16x2_mul(e4m3x2_lo16_to_f16x2(ehi), S0);
                        a[3] = f16x2_mul(e4m3x2_hi16_to_f16x2(ehi), S1);
                        #pragma unroll
                        for (int nb = 0; nb < NT; ++nb) {
                            if (nb < ntb) mma16816(acc[i][nb * CH + (CH == 2 ? q : 0)], a, b[nb][2 * q], b[nb][2 * q + 1]);
                        }
                    }
                }
            }
        }
    }
    cp_async_wait<0>();

    if (KS == 2) {
        __syncthreads();                                   // the ring is free: reuse it for the reduction
        float* red = reinterpret_cast<float*>(wt_smem);
        if (gidx == 1) {
            #pragma unroll
            for (int i = 0; i < TPW; ++i)
                #pragma unroll
                for (int n = 0; n < NT * CH; ++n)
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) red[((i * NT * CH + n) * 4 + q) * 128 + (tid & 127)] = acc[i][n][q];
        }
        __syncthreads();
        if (gidx == 1) return;
        #pragma unroll
        for (int i = 0; i < TPW; ++i)
            #pragma unroll
            for (int n = 0; n < NT * CH; ++n)
                #pragma unroll
                for (int q = 0; q < 4; ++q) acc[i][n][q] += red[((i * NT * CH + n) * 4 + q) * 128 + (tid & 127)];
    }
    if (CH == 2) {
        #pragma unroll
        for (int i = 0; i < TPW; ++i)
            #pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][0][q] += acc[i][1][q];
    }

    // C fragment: rows (weight rows) g, g+8 of the tile -> output column; cols (tokens) 2t, 2t+1
    if constexpr (SPLITK) {
        __shared__ int s_last;
        #pragma unroll
        for (int i = 0; i < MW; ++i) {
            const int tile = (blockIdx.x * WT_NWARPS + warp) * MW + i;
            #pragma unroll
            for (int nb = 0; nb < NT; ++nb)
                #pragma unroll
                for (int hr = 0; hr < 2; ++hr) {
                    const int col = tile * 16 + g + 8 * hr;
                    #pragma unroll
                    for (int cc = 0; cc < 2; ++cc) {
                        const int tok = nb * 8 + 2 * t + cc;
                        if (tok >= nrows) continue;
                        float* pp = part + ((size_t)blockIdx.z * part_rows + row0 + tok) * ldp + col;
                        pp[0] = acc[i][nb * CH][2 * hr + cc] * gsf;
                        if (GATED) pp[(size_t)NTILES * 8] = acc[MW + i][nb * CH][2 * hr + cc] * gsf;
                    }
                }
        }
        __threadfence();
        __syncthreads();
        if (tid == 0) {
            const int old = atomicAdd(counters + blockIdx.y * gridDim.x + blockIdx.x, 1);
            s_last = (old == KSPLIT - 1);
            if (old == KSPLIT - 1) counters[blockIdx.y * gridDim.x + blockIdx.x] = 0;
        }
        __syncthreads();
        if (!s_last) return;
        __threadfence();
    }
    #pragma unroll
    for (int i = 0; i < MW; ++i) {
        const int tile = (blockIdx.x * WT_NWARPS + warp) * MW + i;
        #pragma unroll
        for (int nb = 0; nb < NT; ++nb) {
            #pragma unroll
            for (int hr = 0; hr < 2; ++hr) {
                const int col = tile * 16 + g + 8 * hr;
                #pragma unroll
                for (int cc = 0; cc < 2; ++cc) {
                    const int tok = nb * 8 + 2 * t + cc;
                    if (tok >= nrows) continue;
                    float gv = acc[i][nb * CH][2 * hr + cc] * gsf;
                    float uv = GATED ? acc[MW + i][nb * CH][2 * hr + cc] * gsf : 0.f;
                    if constexpr (SPLITK) {
                        for (int z = 0; z < KSPLIT; ++z) {
                            if (z == (int)blockIdx.z) continue;
                            const float* pp = part + ((size_t)z * part_rows + row0 + tok) * ldp + col;
                            gv += __ldcg(pp);
                            if (GATED) uv += __ldcg(pp + (size_t)NTILES * 8);
                        }
                    }
                    const size_t o = (size_t)(row0 + tok) * ldo + col;
                    if (GATED) {
                        const float hv = gv / (1.f + __expf(-gv)) * uv;
                        out16[o] = __float2half(__bfloat162float(__float2bfloat16(hv)));   // bf16 value in an fp16 container
                    } else {
                        out_bf[o] = __float2bfloat16(gv);
                    }
                }
            }
        }
    }
}

template <bool GATED, int MW, int NT, int KS, int NSTAGE, bool SPLITK>
void launch_wt(const __half* X, int ldx, const int* st, const int* be, const int* br, const int* bend, const int* nb,
               const uint8_t* Wr, const uint8_t* Sr, const float* gs, __half* o16, __nv_bfloat16* obf, int ldo,
               float* part, int ldp, int part_rows, int* counters, int ksplit,
               int K, int ntiles, int max_blocks, float gs_mul, cudaStream_t stream)
{
    constexpr int TPW = (GATED ? 2 : 1) * MW;
    constexpr int smem = NSTAGE * wt_stage_bytes(TPW, NT);
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(moe_w4a16_wt_kernel<GATED, MW, NT, KS, NSTAGE, SPLITK>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        attr_set = true;
    }
    const int n_x = (GATED ? ntiles / 2 : ntiles) / (WT_NWARPS * MW);
    dim3 grid(n_x, max_blocks, ksplit);
    moe_w4a16_wt_kernel<GATED, MW, NT, KS, NSTAGE, SPLITK><<<grid, 128 * KS, smem, stream>>>(
        X, ldx, st, be, br, bend, nb, Wr, Sr, gs, o16, obf, ldo, part, ldp, part_rows, counters, K, ntiles, gs_mul);
}

// post-expert RMSNorm x routing weight, summed per token: grid = tokens, 8 warps (one per route). Phase 1: each
// warp loads its route's row (all H/256 16B pieces per lane at once) and reduces the sum of squares; phase 2:
// every thread re-reads 8 elements of all 8 rows (L2-hot) and writes the weighted sum. No shared-memory tile,
// so the SM keeps its carveout.
template <int H>
__global__ void __launch_bounds__(256)
moe_finalize_kernel(const __nv_bfloat16* __restrict__ y, int ldy, const int* __restrict__ row_of_pair,
                    const float* __restrict__ tw, const float* __restrict__ nw, __nv_bfloat16* __restrict__ out, float eps)
{
    constexpr int NSEG = H / 8 / 32;                     // 16B pieces per lane
    __shared__ float sc_sh[8];
    __shared__ int row_sh[8];
    const int tok = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = row_of_pair[tok * 8 + warp];
    float ss = 0.f;
    if (r >= 0) {
        const uint4* src = reinterpret_cast<const uint4*>(y + (size_t)r * ldy);
        uint4 u[NSEG];
        #pragma unroll
        for (int q = 0; q < NSEG; ++q) u[q] = src[lane + 32 * q];
        #pragma unroll
        for (int q = 0; q < NSEG; ++q) {
            const int seg = lane + 32 * q;
            const float4 n0 = *reinterpret_cast<const float4*>(nw + seg * 8), n1 = *reinterpret_cast<const float4*>(nw + seg * 8 + 4);
            const __nv_bfloat162* pb = reinterpret_cast<const __nv_bfloat162*>(&u[q]);
            float2 f0 = __bfloat1622float2(pb[0]), f1 = __bfloat1622float2(pb[1]);
            float2 f2 = __bfloat1622float2(pb[2]), f3 = __bfloat1622float2(pb[3]);
            f0.x *= n0.x; f0.y *= n0.y; f1.x *= n0.z; f1.y *= n0.w;
            f2.x *= n1.x; f2.y *= n1.y; f3.x *= n1.z; f3.y *= n1.w;
            ss += f0.x * f0.x + f0.y * f0.y + f1.x * f1.x + f1.y * f1.y + f2.x * f2.x + f2.y * f2.y + f3.x * f3.x + f3.y * f3.y;
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    }
    if (lane == 0) {
        sc_sh[warp] = r >= 0 ? rsqrtf(ss / (float)H + eps) * tw[tok * 8 + warp] : 0.f;
        row_sh[warp] = r;
    }
    __syncthreads();
    #pragma unroll
    for (int base = threadIdx.x * 8; base < H; base += 256 * 8) {
        uint4 v[8];
        #pragma unroll
        for (int s2 = 0; s2 < 8; ++s2) {
            const int rs = row_sh[s2];
            v[s2] = rs >= 0 ? *reinterpret_cast<const uint4*>(y + (size_t)rs * ldy + base) : make_uint4(0u, 0u, 0u, 0u);
        }
        float acc[8];
        #pragma unroll
        for (int q = 0; q < 8; ++q) acc[q] = 0.f;
        #pragma unroll
        for (int s2 = 0; s2 < 8; ++s2) {
            const float scs = sc_sh[s2];
            const __nv_bfloat162* pb = reinterpret_cast<const __nv_bfloat162*>(&v[s2]);
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float2 f = __bfloat1622float2(pb[q]);
                acc[2 * q] += scs * f.x;
                acc[2 * q + 1] += scs * f.y;
            }
        }
        const float4 n0 = *reinterpret_cast<const float4*>(nw + base), n1 = *reinterpret_cast<const float4*>(nw + base + 4);
        __nv_bfloat162 o[4];
        o[0] = __floats2bfloat162_rn(acc[0] * n0.x, acc[1] * n0.y); o[1] = __floats2bfloat162_rn(acc[2] * n0.z, acc[3] * n0.w);
        o[2] = __floats2bfloat162_rn(acc[4] * n1.x, acc[5] * n1.y); o[3] = __floats2bfloat162_rn(acc[6] * n1.z, acc[7] * n1.w);
        *reinterpret_cast<uint4*>(out + (size_t)tok * H + base) = *reinterpret_cast<const uint4*>(o);
    }
}

} // namespace

void moe_w4a16_gemm(torch::Tensor A, torch::Tensor sorted_tok, torch::Tensor blk_e, torch::Tensor blk_row0,
                    torch::Tensor blk_end, torch::Tensor nblk, torch::Tensor Wr, torch::Tensor Sr, torch::Tensor gs,
                    torch::Tensor out, torch::Tensor scratch, int64_t K, int64_t N, int64_t max_blocks, bool gated,
                    int64_t mt, int64_t nper, int64_t ks, int64_t ksplit)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    const __half* Ap = reinterpret_cast<const __half*>(A.data_ptr());
    const uint4* Wp = reinterpret_cast<const uint4*>(Wr.data_ptr());
    const uint2* Sp = reinterpret_cast<const uint2*>(Sr.data_ptr());
    __half* o16 = gated ? reinterpret_cast<__half*>(out.data_ptr()) : nullptr;
    __nv_bfloat16* obf = gated ? nullptr : reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    float* of32 = ksplit > 1 ? scratch.data_ptr<float>() : nullptr;
    const int ldo = ksplit > 1 ? (int)scratch.stride(0) : (int)out.stride(0);
    const int rows = (int)(max_blocks * 16 * mt);
    if (ksplit > 1) cudaMemsetAsync(of32, 0, (size_t)rows * scratch.stride(0) * sizeof(float), stream);
#define OKC_L(G, M, P, S) launch<G, M, P, S>(Ap, (int)A.stride(0), sorted_tok.data_ptr<int>(), blk_e.data_ptr<int>(), \
        blk_row0.data_ptr<int>(), blk_end.data_ptr<int>(), nblk.data_ptr<int>(), Wp, Sp, gs.data_ptr<float>(), o16, obf, of32, \
        ldo, (int)K, (int)N, (int)max_blocks, (int)ksplit, stream)
    const int key = (gated ? 1000 : 0) + (int)mt * 100 + (int)nper * 10 + (int)ks;
    switch (key) {
        case 1221: OKC_L(true, 2, 2, 1); break;   case 221:  OKC_L(false, 2, 2, 1); break;
        default: TORCH_CHECK(false, "unsupported (mt, nper, ks)");
    }
#undef OKC_L
    if (ksplit > 1) {
        dim3 grid2((unsigned)(16 * mt), (unsigned)max_blocks);
        if (gated)
            silu_mul_f32_kernel<<<grid2, 256, 0, stream>>>(of32, ldo, reinterpret_cast<__half*>(out.data_ptr()), (int)out.stride(0),
                blk_row0.data_ptr<int>(), blk_end.data_ptr<int>(), nblk.data_ptr<int>(), (int)(16 * mt), (int)(N / 2));
        else
            f32_to_bf16_rows_kernel<<<grid2, 256, 0, stream>>>(of32, ldo, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (int)out.stride(0),
                blk_row0.data_ptr<int>(), blk_end.data_ptr<int>(), nblk.data_ptr<int>(), (int)(16 * mt), (int)N);
    }
}

void moe_w4a16_wt(torch::Tensor X, torch::Tensor sorted_tok, torch::Tensor blk_e, torch::Tensor blk_row0,
                  torch::Tensor blk_end, torch::Tensor nblk, torch::Tensor Wr, torch::Tensor Sr, torch::Tensor gs,
                  torch::Tensor out, torch::Tensor part, torch::Tensor counters, int64_t ksplit, int64_t K, int64_t ntiles,
                  int64_t max_blocks, bool gated, int64_t mw, int64_t nt, int64_t ks, int64_t nstage, double gs_mul)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    const __half* Xp = reinterpret_cast<const __half*>(X.data_ptr());
    const uint8_t* Wp = reinterpret_cast<const uint8_t*>(Wr.data_ptr());
    const uint8_t* Sp = reinterpret_cast<const uint8_t*>(Sr.data_ptr());
    __half* o16 = gated ? reinterpret_cast<__half*>(out.data_ptr()) : nullptr;
    __nv_bfloat16* obf = gated ? nullptr : reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    const int ldo = (int)out.stride(0);
    TORCH_CHECK(ksplit == 1 || part.dim() == 3, "split-K needs a [ksplit, rows, cols] fp32 buffer");
    float* pp = ksplit > 1 ? part.data_ptr<float>() : nullptr;
    const int ldp = ksplit > 1 ? (int)part.stride(1) : 0;
    const int prow = ksplit > 1 ? (int)part.size(1) : 0;
    int* cnt = ksplit > 1 ? counters.data_ptr<int>() : nullptr;
    if (ksplit > 1) {
        const int n_x = (gated ? ntiles / 2 : ntiles) / (WT_NWARPS * mw);
        TORCH_CHECK(counters.numel() >= max_blocks * n_x, "split-K counters too small");
    }
#define OKC_W(G, M, T, S, NS, SK) launch_wt<G, M, T, S, NS, SK>(Xp, (int)X.stride(0), sorted_tok.data_ptr<int>(), blk_e.data_ptr<int>(), \
        blk_row0.data_ptr<int>(), blk_end.data_ptr<int>(), nblk.data_ptr<int>(), Wp, Sp, gs.data_ptr<float>(), o16, obf, ldo, \
        pp, ldp, prow, cnt, (int)ksplit, (int)K, (int)ntiles, (int)max_blocks, (float)gs_mul, stream)
    const int key = (ksplit > 1 ? 1000000 : 0) + (gated ? 100000 : 0) + (int)mw * 10000 + (int)nt * 1000 + (int)ks * 100 + (int)nstage;
    switch (key) {
        case 111102: OKC_W(true, 1, 1, 1, 2, false); break;   case 111104: OKC_W(true, 1, 1, 1, 4, false); break;
        case 111108: OKC_W(true, 1, 1, 1, 8, false); break;
        case 112102: OKC_W(true, 1, 2, 1, 2, false); break;   case 112104: OKC_W(true, 1, 2, 1, 4, false); break;
        case 114102: OKC_W(true, 1, 4, 1, 2, false); break;   case 114104: OKC_W(true, 1, 4, 1, 4, false); break;
        case 11102: OKC_W(false, 1, 1, 1, 2, false); break;   case 11104: OKC_W(false, 1, 1, 1, 4, false); break;
        case 11108: OKC_W(false, 1, 1, 1, 8, false); break;
        case 21102: OKC_W(false, 2, 1, 1, 2, false); break;   case 21104: OKC_W(false, 2, 1, 1, 4, false); break;
        case 12104: OKC_W(false, 1, 2, 1, 4, false); break;   case 22102: OKC_W(false, 2, 2, 1, 2, false); break;
        case 22104: OKC_W(false, 2, 2, 1, 4, false); break;   case 14104: OKC_W(false, 1, 4, 1, 4, false); break;
        case 24102: OKC_W(false, 2, 4, 1, 2, false); break;   case 24104: OKC_W(false, 2, 4, 1, 4, false); break;
        // split-K (SPLITK = true) instantiations removed: measured slower than the plain variants on every shape
        default: TORCH_CHECK(false, "unsupported wt variant (mw, nt, ks, nstage, ksplit)");
    }
#undef OKC_W
}

void moe_finalize(torch::Tensor y, torch::Tensor row_of_pair, torch::Tensor tw, torch::Tensor nw, torch::Tensor out,
                  int64_t M, double eps)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(out.size(1) == 4096, "finalize: H must be 4096");
    moe_finalize_kernel<4096><<<(unsigned)M, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(y.data_ptr()), (int)y.stride(0), row_of_pair.data_ptr<int>(),
        tw.data_ptr<float>(), nw.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (float)eps);
}
"""

CPP_SRC = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
void moe_w4a16_gemm(torch::Tensor A, torch::Tensor sorted_tok, torch::Tensor blk_e, torch::Tensor blk_row0,
                    torch::Tensor blk_end, torch::Tensor nblk, torch::Tensor Wr, torch::Tensor Sr, torch::Tensor gs,
                    torch::Tensor out, torch::Tensor scratch, int64_t K, int64_t N, int64_t max_blocks, bool gated,
                    int64_t mt, int64_t nper, int64_t ks, int64_t ksplit);
void moe_w4a16_wt(torch::Tensor X, torch::Tensor sorted_tok, torch::Tensor blk_e, torch::Tensor blk_row0,
                  torch::Tensor blk_end, torch::Tensor nblk, torch::Tensor Wr, torch::Tensor Sr, torch::Tensor gs,
                  torch::Tensor out, torch::Tensor part, torch::Tensor counters, int64_t ksplit, int64_t K, int64_t ntiles,
                  int64_t max_blocks, bool gated, int64_t mw, int64_t nt, int64_t ks, int64_t nstage, double gs_mul);
void moe_finalize(torch::Tensor y, torch::Tensor row_of_pair, torch::Tensor tw, torch::Tensor nw, torch::Tensor out,
                  int64_t M, double eps);
"""

_EXT = None


def build():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline
        _EXT = load_inline(
            name="okc_moe_w4a16", cpp_sources=[CPP_SRC],
            cuda_sources=["#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n" + CUDA_SRC],
            functions=["moe_w4a16_gemm", "moe_w4a16_wt", "moe_finalize"], extra_cuda_cflags=["-O3", "-arch=sm_90a", "--use_fast_math"], verbose=False)
    return _EXT


def repack_weights(w: torch.Tensor, sf: torch.Tensor):
    """w uint8 [E, N, K/2] packed E2M1 (low nibble = even k), sf fp8 [E, N, K/16] -> (Wr uint8 [E, N/8, K/128,
    32, 16], Sr uint8 [E, N/8, K/128, 8, 8]). A permutation of the stored bytes/nibbles: lossless."""
    E, N, K2 = w.shape
    K = 2 * K2
    dev = w.device
    NB, NC = N // 8, K // 128
    # lane (g, t), step j, position q in {e(2t), e(2t+8), e(2t+1), e(2t+9)}: k index
    lane = torch.arange(32, device=dev)
    g, t = lane // 4, lane % 4
    step = torch.arange(8, device=dev)
    # byte0 = e(2t) | e(2t+8)<<4 ; byte1 = e(2t+1) | e(2t+9)<<4  (nibble q: 0 lo of byte0, 1 hi of byte0, 2 lo of byte1, 3 hi of byte1)
    koff = torch.stack([2 * t, 2 * t + 8, 2 * t + 1, 2 * t + 9], -1)                 # [32, 4]
    k_idx = (step[None, :, None] * 16 + koff[:, None, :])                               # [32, 8, 4]
    Wr = torch.empty((E, NB, NC, 32, 16), dtype=torch.uint8, device=dev)
    for e in range(E):
        we = w[e]                                                                      # [N, K/2]
        lo = we & 0x0F
        hi = we >> 4
        nib = torch.stack([lo, hi], -1).view(N, K)                                     # [N, K] nibble per k
        blk = nib.view(NB, 8, NC, 128)                                                 # [nb, g, c, k]
        # gather: for lane (g,t), chunk c, step j, q -> nib[nb, g, c, 16 j + koff]
        sel = blk[:, g, :, :]                                                          # [NB, 32, NC, 128]
        sel = sel.gather(3, k_idx.view(1, 32, 1, 32).expand(NB, 32, NC, 32))           # [NB, 32, NC, 32] nibbles (step-major, q)
        sel = sel.view(NB, 32, NC, 8, 4)                                               # [.., step, q]
        bytes_ = (sel[..., 0] | (sel[..., 1] << 4)), (sel[..., 2] | (sel[..., 3] << 4)) # byte0, byte1 per step
        packed = torch.stack(bytes_, -1).view(NB, 32, NC, 16)                          # 8 steps x 2 bytes
        Wr[e] = packed.permute(0, 2, 1, 3)                                             # [NB, NC, 32, 16]
    Sr = sf.view(torch.uint8).view(E, NB, 8, NC, 8).permute(0, 1, 3, 2, 4).contiguous()  # [E, NB, NC, g, step]
    return Wr.contiguous(), Sr


def repack_weights_wt(w: torch.Tensor, sf: torch.Tensor):
    """w uint8 [E, N, K/2] packed E2M1 (low nibble = even k), sf fp8 [E, N, K/16] ->
    Wr uint8 [E, N/16, K/128, 2, 32, 16]: per (expert, m16 tile, chunk, half, lane) 4 words = the lane's
    A-fragment nibbles for k16 steps 4*half .. 4*half+3 (byte b: nibble A_b at bits {7,4,3,2} = E4M3
    position, nibble B_b at (e1:6, s:5, e0:1, m:0); A_0,A_1 = row g k 2t,2t+1; A_2,A_3 = row g+8 k 2t,2t+1;
    B_0,B_1 = row g k 2t+8,2t+9; B_2,B_3 = row g+8 k 2t+8,2t+9);
    Sr uint8 [E, N/16, K/128, 8, 16]: per g the 8 steps' (scale row g, scale row g+8) byte pairs.
    A permutation of the stored bits: lossless."""
    E, N, K2 = w.shape
    K = 2 * K2
    dev = w.device
    NTL, NC = N // 16, K // 128
    t = torch.arange(4, device=dev)
    kidx = torch.stack([2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9], -1)          # [t, q]
    Wr = torch.empty((E, NTL, NC, 2, 32, 16), dtype=torch.uint8, device=dev)
    for e in range(E):
        we = w[e]
        nib = torch.stack([we & 0x0F, we >> 4], -1).view(N, K).to(torch.int32)     # [N, K]
        n6 = nib.view(NTL, 2, 8, NC, 8, 16)                                       # tile, rh, g, chunk, step, k16
        sel = n6[..., kidx]                                                        # tile, rh, g, chunk, step, t, q
        # byte b of the word: (rh, q) -> A nibble of byte (rh*2 + q) for q in {0,1}; B nibble for q in {2,3}
        word = torch.zeros((NTL, 8, NC, 8, 4), dtype=torch.int32, device=dev)      # tile, g, chunk, step, t
        for rh in (0, 1):
            for q in range(4):
                v = sel[:, rh, :, :, :, :, q]
                b = rh * 2 + (q & 1)
                sgn, e1, e0, m = (v >> 3) & 1, (v >> 2) & 1, (v >> 1) & 1, v & 1
                if q < 2:   # A: E4M3 placement
                    byte = (sgn << 7) | (e1 << 4) | (e0 << 3) | (m << 2)
                else:       # B
                    byte = (e1 << 6) | (sgn << 5) | (e0 << 1) | m
                word |= byte << (8 * b)
        word = word.view(NTL, 8, NC, 2, 4, 4).permute(0, 2, 3, 1, 5, 4).contiguous()   # tile, chunk, half, g, t, step4
        Wr[e] = word.view(NTL, NC, 2, 32, 4).view(torch.uint8).view(NTL, NC, 2, 32, 16)
        del nib, n6, sel, word
    sfu = sf.view(torch.uint8).view(E, NTL, 2, 8, NC, 8)                            # e, tile, rh, g, chunk, step
    Sr = sfu.permute(0, 1, 4, 3, 5, 2).contiguous().view(E, NTL, NC, 8, 16)         # e, tile, chunk, g, (step, rh)
    return Wr.contiguous(), Sr

# == END CUDA ==


# (rows per block / 16, n8 blocks per warp) by tokens per step; overridable for benchmarking
_CUDA_OVERRIDE = None


def _cuda_variant(M: int):
    """(rows per block / 16, n8 blocks per warp, K-split warp groups, K-split CTAs). On-device
    sweeps (runs 30863d92, cc8d876d): (1,2,1) is best up to 64 tokens, 32-row blocks above; the
    warp-group and inter-CTA K splits both cost more than they save."""
    if _CUDA_OVERRIDE is not None:
        return _CUDA_OVERRIDE
    return 2, 2, 1, 1   # the only instantiated variant now (this path serves tokens > CUDA_MIN_TOKENS as a fallback)


def _moe_forward_cuda(x, ids, tw, local_start: int, I: int, state) -> torch.Tensor:
    M, H = x.shape
    if ids.dtype != torch.int32:
        ids = ids.to(torch.int32)
    ids = ids.contiguous()
    tw = tw.contiguous()
    mt, nper, ks, ksplit = _cuda_variant(M)
    BM = 16 * mt
    NP, MAXB = _route(x, ids, local_start, BM, state, marlin=False)
    x16 = state["x16"][:M]
    out = torch.empty((M, H), dtype=MOE_OUT_DTYPE, device=x.device)
    ext = build()
    inter = state["inter"]
    y = state["y"]
    scratch = state["scratch32"]
    ext.moe_w4a16_gemm(x16, state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                       state["Wr13"], state["Sr13"], state["gs13_64"], inter, scratch, H, 2 * I, MAXB, True, mt, nper, ks, ksplit)
    ext.moe_w4a16_gemm(inter, state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                       state["Wr2"], state["Sr2"], state["gs2_64"], y, scratch, I, H, MAXB, False, mt, nper, ks, ksplit)
    _finalize_kernel[(M,)](y, state["row_of_pair"], tw, state["norm_w"], out, POST_NORM_EPS, H=H, TOPK=TOPK, num_warps=4)
    return out


# (token n8 blocks per expert block, GEMM2 m16 tiles per warp, K-split warp groups, cp.async stages, K-split CTAs)
# Locked from the in-prepare cold-L2 sweeps (checks f2f756d9, a3d79f15): 8 cp.async stages up to 8 tokens, 4 above;
# 8-token blocks up to 64 tokens, 16-token (GEMM2: 2 tiles per warp) up to 128, 32-token with 2 stages beyond.
WT_DEFAULT = {8: (1, 1, 1, 8, 1), 64: (1, 1, 1, 4, 1), 128: (2, 2, 1, 4, 1), 100000: (4, 2, 1, 2, 1)}
WT_SAFE = (1, 1, 1, 4, 1)
PART_SPLIT_MAX = 4      # fp32 partial slices for split-K
PART_ROWS = 1024        # padded rows the partial buffer holds (split-K is a small-batch tool)
_WT_OVERRIDE = None


def _wt_variant(M: int, state):
    if _WT_OVERRIDE is not None:
        return _WT_OVERRIDE
    pol = state.get("wt_policy", WT_DEFAULT)
    keys = sorted(pol)
    v = pol[keys[-1]]
    for k in keys:
        if M <= k:
            v = pol[k]
            break
    if v in state.get("wt_bad", ()):
        v = WT_SAFE
    return v


def _wt_max_blocks(M: int, BM: int) -> int:
    """Blocks are per expert, padded to BM rows: at most one partial block per routed expert plus npairs // BM."""
    npairs = M * TOPK
    return min(npairs, LOCAL) + npairs // BM


def _moe_forward_wt(x, ids, tw, local_start: int, I: int, state) -> torch.Tensor:
    """Experts through the weights-as-A CUDA GEMMs (8*NT-token blocks per expert) and the CUDA finalize."""
    M, H = x.shape
    if ids.dtype != torch.int32:
        ids = ids.to(torch.int32)
    ids = ids.contiguous()
    tw = tw.contiguous()
    nt, mw2, ks, nstage, ksplit = _wt_variant(M, state)
    BM = 8 * nt
    if M * TOPK + LOCAL * (BM - 1) > PART_ROWS:
        ksplit = 1
    NP, MAXB = _route(x, ids, local_start, BM, state, marlin=False)
    maxb = min(MAXB, _wt_max_blocks(M, BM))
    x16 = state["x16"][:M]
    out = torch.empty((M, H), dtype=MOE_OUT_DTYPE, device=x.device)
    ext = build()
    inter = state["inter"]
    y = state["y"]
    part = state["part"]
    counters = state["counters"]
    ext.moe_w4a16_wt(x16, state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                     state["Wt13"], state["St13"], state["gs13"], inter, part, counters, ksplit, H, 2 * I // 16, maxb, True, 1, nt, ks,
                     nstage, state["gs_mul"])
    ext.moe_w4a16_wt(inter, state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                     state["Wt2"], state["St2"], state["gs2"], y, part, counters, ksplit, I, H // 16, maxb, False, mw2, nt, ks, nstage,
                     state["gs_mul"])
    ext.moe_finalize(y, state["row_of_pair"], tw, state["norm_w"], out, M, POST_NORM_EPS)
    return out


# ----------------------------------------------------------------------------------------------
# host side
# ----------------------------------------------------------------------------------------------


# Sweep (one hypothesis per scored shape): (BM, BN, BK, num_warps, num_stages)
GEMM_SWEEP = {
    1: (16, 32, 256, 4, 4),
    8: (16, 64, 128, 4, 4),
    16: (16, 64, 128, 4, 4),
    32: (16, 64, 128, 4, 4),
    128: (64, 128, 128, 8, 3),
    256: (64, 128, 128, 8, 3),
}


_GEMM_OVERRIDE = None   # (BM, BN, BK, num_warps, num_stages) during in-prepare benchmarking
_MODE = 0               # benchmark ablation mode (see _moe_gemm_kernel)
_CTAS_PER_SM = 3


def _gemm_config(bs: int):
    """(BM, BN, BK, num_warps, num_stages) by tokens per step."""
    if _GEMM_OVERRIDE is not None:
        return _GEMM_OVERRIDE
    if bs in GEMM_SWEEP:
        return GEMM_SWEEP[bs]
    if bs <= 8:
        return 16, 32, 256, 4, 4
    if bs <= 64:
        return 16, 64, 128, 4, 4
    return 64, 128, 128, 8, 3   # (64, 128, 64, 4, 3) crashes Triton 3.8's pass pipeline with the scaled asm


def _moe_state(weights) -> dict:
    dev = weights["w13"].device
    def block_scales(sf):   # [E, N, G] fp8 -> [E, N/8, G/2, 8, 2] contiguous (same bytes, pair-blocked)
        E_, N_, G_ = sf.shape
        return sf.view(E_, N_ // 8, 8, G_ // 2, 2).permute(0, 1, 3, 2, 4).contiguous().view(E_, N_, G_)

    state = {
        "w13_sfb": block_scales(weights["w13_sf"]), "w2_sfb": block_scales(weights["w2_sf"]),
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
    maxb = NP // 8 + LOCAL   # blocks of >= 8 rows
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
        "sorted_pair": torch.zeros((rows_max,), dtype=torch.int32, device=dev),
        "lids": torch.full((NP,), -1, dtype=torch.int32, device=dev),
        "npad": torch.zeros((1,), dtype=torch.int32, device=dev),
    })
    if MOE_BACKEND == "cuda":
        try:
            import time as _time
            t0 = _time.time()
            build()   # nvcc, once (prepare is untimed)
            print(f"cuda build in {_time.time() - t0:.0f}s")
            t0 = _time.time()
            # this GPU's experts only (ep8: 0-31); blk_e indexes them from local_start = 0
            state["Wr13"], state["Sr13"] = repack_weights(weights["w13"][:LOCAL], weights["w13_sf"][:LOCAL])
            state["Wr2"], state["Sr2"] = repack_weights(weights["w2"][:LOCAL], weights["w2_sf"][:LOCAL])
            state["gs13_64"] = (weights["w13_gs"][:LOCAL].float() * 64.0).contiguous()   # the E4M3 placement is value x 2^-6
            state["gs2_64"] = (weights["w2_gs"][:LOCAL].float() * 64.0).contiguous()
            # fp32 split-K scratch: rows up to the padded row count, 2I columns (gate | up)
            state["scratch32"] = torch.zeros((rows_max, 2 * I), dtype=torch.float32, device=dev)
            torch.cuda.synchronize()
            print(f"cuda backend ready in {_time.time() - t0:.0f}s")
            t0 = _time.time()
            # weights-as-A layout (small blocks): fragment-ordered words; scales kept as E4M3 bytes
            state["Wt13"], state["St13"] = repack_weights_wt(weights["w13"][:LOCAL], weights["w13_sf"][:LOCAL])
            state["Wt2"], state["St2"] = repack_weights_wt(weights["w2"][:LOCAL], weights["w2_sf"][:LOCAL])
            state["gs13"] = weights["w13_gs"][:LOCAL].float().contiguous()
            state["gs2"] = weights["w2_gs"][:LOCAL].float().contiguous()
            state["gs_mul"] = 64.0   # the E4M3 placement is e2m1 * 2^-6; applied with gs to the fp32 sums
            state["xscale"] = 1.0
            state["part"] = torch.empty((PART_SPLIT_MAX, PART_ROWS, 2 * I), dtype=torch.float32, device=dev)   # split-K partials
            state["counters"] = torch.zeros((maxb * 64,), dtype=torch.int32, device=dev)   # split-K arrival counters (self-resetting)
            torch.cuda.synchronize()
            print(f"wt layout ready in {_time.time() - t0:.0f}s")
        except Exception as e:
            print("cuda backend unavailable:", repr(e)[:200])
    if MOE_BACKEND in ("marlin", "cuda"):
        try:
            state["bank"] = _marlin_bank(weights)
            state["cache13"] = torch.empty((MAX_M * TOPK * max(2 * I, H),), dtype=torch.bfloat16, device=dev)
            state["cache2"] = torch.empty((MAX_M * TOPK * I,), dtype=torch.bfloat16, device=dev)
        except Exception as e:   # not in the image: the Triton experts
            print("marlin unavailable:", repr(e)[:120])
    return state


def _marlin_bank(weights):
    """The local experts in Marlin's layout, through SGLang's own loader step (the stock path)."""
    import types
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
    from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
    w13 = weights["w13"][:LOCAL]
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=w13.shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=w13.contiguous(), w2_weight=weights["w2"][:LOCAL].contiguous(),
        w13_weight_scale=weights["w13_sf"][:LOCAL].contiguous(), w2_weight_scale=weights["w2_sf"][:LOCAL].contiguous(),
        w13_weight_scale_2=weights["w13_gs"][:LOCAL].contiguous(), w2_weight_scale_2=weights["w2_gs"][:LOCAL].contiguous())
    prepare_moe_nvfp4_layer_for_marlin(layer)
    device = w13.device
    torch.cuda.synchronize()
    return {"w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
            "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
            "size_n": layer.intermediate_size_per_partition,
            "workspace": marlin_make_workspace(device, max_blocks_per_sm=4)}


def _route(x, ids, local_start: int, BM: int, state, marlin: bool):
    """Runs the routing kernel (plus the fp16 copy of x for the Triton GEMMs); returns (NP, MAXB)."""
    M, H = x.shape
    NP = max(16, triton.next_power_of_2(M * TOPK))
    MAXB = triton.next_power_of_2(NP // BM + LOCAL)
    x16 = state["x16"][:M]
    XBLOCK = 4096
    n_elems = M * H
    PB = min(NP, 64)                      # pairs per routing CTA
    CH = min(NP, 256)                     # pairs scanned per step
    n_route = triton.cdiv(M * TOPK, PB)
    grid0 = (n_route + (0 if marlin else triton.cdiv(n_elems, XBLOCK)),)
    _route_kernel[grid0](
        ids, x, x16,
        state["sorted_tok"], state["row_of_pair"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
        state["sorted_pair"], state["lids"], state["npad"],
        M, local_start, n_elems, n_route, float(state.get("xscale", 1.0)),
        PB=PB, CH=CH, BM=BM, MAXB=MAXB, E=LOCAL, TOPK=TOPK, XBLOCK=XBLOCK, MARLIN=marlin, MULTI=NP > CH, num_warps=4,
    )
    return NP, MAXB


MARLIN_BLOCK_M = (8, 16, 32, 48, 64)   # MuseSpark1xFp4Experts' candidates, smallest first


_MARLIN_BM_OVERRIDE = None   # in-prepare benchmarking


def _marlin_block_m(num_tokens: int) -> int:
    if _MARLIN_BM_OVERRIDE is not None:
        return _MARLIN_BM_OVERRIDE
    # the stock density rule (a per-batch sweep on synthetic heavy routing preferred 16 at
    # 16-64 tokens, but on the judge's lighter routing it measured no better: run 7d2d04b3)
    bm = MARLIN_BLOCK_M[0]
    for b in MARLIN_BLOCK_M:
        if num_tokens * TOPK / LOCAL / b < 0.9:
            break
        bm = b
    return bm


def _moe_forward_marlin(x, ids, tw, local_start: int, I: int, state) -> torch.Tensor:
    """Experts through SGLang's Marlin W4A16 GEMMs, with this file's routing (one launch, in
    Marlin's sorted-ids format) and finalize."""
    from sgl_kernel.scalar_type import scalar_types
    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
    M, H = x.shape
    bank = state["bank"]
    x = x.contiguous()
    if ids.dtype != torch.int32:
        ids = ids.to(torch.int32)
    ids = ids.contiguous()
    tw = tw.contiguous()
    BM = _marlin_block_m(M)
    _route(x, ids, local_start, BM, state, marlin=True)
    out = torch.empty((M, H), dtype=MOE_OUT_DTYPE, device=x.device)
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    lids = state["lids"][: M * TOPK].view(M, TOPK)
    sorted_ids = state["sorted_pair"]
    cache1 = state["cache13"][: M * TOPK * gemm1_n].view(-1, gemm1_n)
    cache2 = state["cache2"][: M * TOPK * size_n].view(-1, size_n)
    cache3 = state["cache13"][: M * TOPK * H].view(-1, H)
    common = dict(moe_block_size=BM, mul_topk_weights=False, is_ep=True, b_q_type=scalar_types.float4_e2m1f,
                  is_k_full=True, use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, state["blk_e"], state["npad"], tw, top_k=TOPK,
        size_m=M, size_n=gemm1_n, size_k=H, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=lids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, state["blk_e"], state["npad"], tw, top_k=1,
        size_m=M * TOPK, size_n=H, size_k=size_n, **common).view(M * TOPK, H)
    _finalize_pairs_kernel[(M,)](y, lids, tw, state["norm_w"], out, POST_NORM_EPS, H=H, TOPK=TOPK, num_warps=4)
    return out


CUDA_MIN_TOKENS = 65   # Marlin (stock GEMM + this file's routing) below, this file's CUDA GEMM from here
WT_MAX_TOKENS = 100000   # the weights-as-A path up to here (when it passed its self-test), else the policy above


def _moe_forward(x, ids, tw, local_start: int, I: int, state) -> torch.Tensor:
    M = x.shape[0]
    if "Wt13" in state and state.get("wt_ok", True) and M <= WT_MAX_TOKENS:
        return _moe_forward_wt(x, ids, tw, local_start, I, state)
    if "Wr13" in state and (M >= CUDA_MIN_TOKENS or "bank" not in state):
        return _moe_forward_cuda(x, ids, tw, local_start, I, state)
    if "bank" in state:
        return _moe_forward_marlin(x, ids, tw, local_start, I, state)
    return _moe_forward_triton(x, ids, tw, local_start, I, state)


def _moe_forward_triton(x, ids, tw, local_start: int, I: int, state) -> torch.Tensor:
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
    NP, MAXB = _route(x, ids, local_start, BM, state, marlin=False)
    x16 = state["x16"][:M]
    w13, w13_sf, w2, w2_sf = state["w13"], state["w13_sf"], state["w2"], state["w2_sf"]
    max_items = (NP // BM + LOCAL) * (max(I, H) // BN)
    num_ctas = min(_CTAS_PER_SM * NUM_SMS, max_items)
    blocked = DEQUANT_ASM and _MODE == 0 and SCALE_VARIANT == 4
    sf13 = state["w13_sfb"] if blocked else w13_sf
    sf2 = state["w2_sfb"] if blocked else w2_sf
    common = dict(GATHER=True, GATED=True, ASM=DEQUANT_ASM, BM=BM, BN=BN, BK=BK, MODE=_MODE, SF_BLOCKED=blocked,
                  num_warps=nw, num_stages=ns)
    # gate/up GEMM + SiLU: A = x16 [M, H], W13[e] rows [0, I) gate and [I, 2I) up, K = H, N = I
    _moe_gemm_kernel[(num_ctas,)](
        x16, w13, sf13, state["w13_gs"], state["inter"],
        state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
        x16.stride(0), state["inter"].stride(0), H, I,
        w13.stride(0), w13.stride(1), w13_sf.stride(0), w13_sf.stride(1), I,
        num_ctas, I // BN, **common,
    )
    # down GEMM: A = inter [rows, I], W2[e] [H, I/2], K = I, N = H
    common.update(GATHER=False, GATED=False)
    _moe_gemm_kernel[(num_ctas,)](
        state["inter"], w2, sf2, state["w2_gs"], state["y"],
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
