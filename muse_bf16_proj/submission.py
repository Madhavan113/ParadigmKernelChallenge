# okc: problem=muse_bf16_proj
"""Dense projections for Muse Spark Open: x @ W.T at skinny M.

Small M (the decode / verify shapes) is a bandwidth problem: the whole weight is streamed
once and x stays in L2. Each shape runs as ONE Triton kernel:

  * the grid is (N tiles) x (split-K); each CTA streams a [BLOCK_N, K/SPLIT_K] slab of W
    through a multi-stage cp.async pipeline and accumulates in fp32 with tensor cores;
  * with split-K, the partials go to an fp32 workspace and the LAST CTA of a tile (an atomic
    ticket, acq_rel at gpu scope, as CUTLASS's split-K semaphore does) sums them and writes
    the output, so there is no second launch and no memset; it also resets its ticket.

The router's FP32 weight runs as a BF16 hi + lo pair with FP32 accumulation (both weights are
dotted in the same kernel), which TASK.md allows at any M. Large-M BF16 shapes go to cuBLAS.
"""

from typing import TypedDict

import torch
import triton
import triton.language as tl


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int      # output width; the unpooler's is 1024 up to 8 tokens (this GPU's slice), else 8192
    k: int
    layer: str  # which weight: "qkv_gate", "o", "pooler", "unpooler" or "router"


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


# == BEGIN SHARED proj ==
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


# == BEGIN CUDA ==
CUDA_SRC = r"""
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
        GEMV_CASE(1, 1, 8, 1, false)
        GEMV_CASE(1, 2, 8, 1, false)
        GEMV_CASE(1, 4, 4, 1, false)
        GEMV_CASE(1, 4, 8, 1, false)
        GEMV_CASE(1, 1, 4, 1, false)
        GEMV_CASE(1, 2, 4, 1, false)
        GEMV_CASE(1, 1, 4, 2, false)
        GEMV_CASE(1, 2, 2, 2, false)
        GEMV_CASE(1, 1, 8, 2, false)
        GEMV_CASE(1, 1, 4, 4, false)
        GEMV_CASE(1, 1, 4, 8, true)
        GEMV_CASE(1, 1, 2, 8, true)
        GEMV_CASE(1, 1, 8, 4, true)
        GEMV_CASE(8, 1, 4, 1, false)
        GEMV_CASE(8, 1, 8, 1, false)
        GEMV_CASE(8, 2, 4, 1, false)
        GEMV_CASE(8, 1, 2, 2, false)
        GEMV_CASE(8, 1, 4, 2, false)
        GEMV_CASE(8, 1, 4, 8, true)
        GEMV_CASE(8, 1, 2, 8, true)
        default: TORCH_CHECK(false, "unsupported gemv config");
    }
}
"""

CPP_SRC = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
void stream_bytes(torch::Tensor buf, torch::Tensor out, int64_t grid, int64_t threads, int64_t unroll);
void gemv(torch::Tensor X, torch::Tensor W, torch::Tensor W2, torch::Tensor out, int64_t mpad, int64_t rpw, int64_t unroll, int64_t ksplit, bool two_w);
"""

_EXT = None


def build():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline
        _EXT = load_inline(
            name="okc_proj_gemv", cpp_sources=[CPP_SRC],
            cuda_sources=["#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n" + CUDA_SRC],
            functions=["stream_bytes", "gemv"], extra_cuda_cflags=["-O3", "-arch=sm_90a"], verbose=False)
    return _EXT


def gemv_usable(m: int, n: int, k: int, mpad: int, rpw: int, unroll: int, ksplit: int) -> bool:
    rows = (8 // ksplit) * rpw
    if m > mpad or n % rows or k % (ksplit * 256):
        return False
    steps = k // ksplit // 256
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
    ("qkv_gate", 2304, 8):  (32, 256, 4, 4, 4, True),     # 11.7
    ("qkv_gate", 2304, 16): (32, 256, 4, 4, 4, True),     # 11.0
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
    "qkv_gate": [(32, 128), (64, 128), (128, 64), (64, 64)],
    "o":        [(32, 128), (32, 256), (128, 64), (64, 64)],
    "pooler":   [(32, 128), (64, 128), (128, 64), (64, 64)],
    "unpooler": [(32, 128), (64, 128), (128, 64), (64, 64)],
    "router":   [(32, 128), (32, 64)],
}
USE_TILED = True
USE_TMA = False  # fetch tiles with TMA descriptors (box inner dim <= 256 elements)


_OVERRIDE = None   # a config dict, during in-prepare benchmarking


# (layer, n, m) -> (mpad, rows per warp, unroll, ksplit): the CUDA register GEMV (m <= 8)
GEMV_SHAPES = {}
# (layer, n, m) -> SPLIT_K: cuBLAS batched split-K (bmm over K chunks) + fp32 partial sum
BMM_SHAPES = {}
# m -> config for the router above 64 tokens (Triton two-weight GEMM); missing: cuBLAS over [hi | lo]
ROUTER_BIG = {}


def _config(layer: str, m: int, n: int, k: int):
    if _OVERRIDE is not None:
        return None if _OVERRIDE.get("none") else dict(_OVERRIDE)
    if (layer, n, m) in GEMV_SHAPES:
        mpad, rpw, unroll, ksplit = GEMV_SHAPES[(layer, n, m)]
        return dict(gemv=True, mpad=mpad, rpw=rpw, unroll=unroll, ksplit=ksplit)
    if (layer, n, m) in BMM_SHAPES:
        return dict(bmm=True, SPLIT_K=BMM_SHAPES[(layer, n, m)])
    if layer == "router" and m > MAX_TRITON_M:
        cfg = ROUTER_BIG.get(m)
        return dict(cfg) if cfg else None
    cfg = _config_table(layer, m, n, k)
    if cfg is None:
        return None
    if cfg.get("big"):
        return cfg
    if USE_TILED:
        cfg["tiled"] = True
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
        ext = build()
        two_w = layer == "router"
        ext.gemv(x, w, w2 if two_w else w, out, cfg["mpad"], cfg["rpw"], cfg["unroll"], cfg["ksplit"], two_w)
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
# == END SHARED proj ==


BENCH = True        # microbenchmarks in prepare (untimed; results go to the run log)
BENCH_BUDGET_S = 170


def _bench(state):
    import time
    global _OVERRIDE
    t0 = time.time()
    dev = state["router"].device
    layers = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096),
              "router": (256, 8192)}
    xs = {(m, k): torch.randn((m, k), device=dev).to(torch.bfloat16) for m in (1, 8, 16, 32, 128, 256) for k in (8192, 1024, 4096)}
    flush = torch.empty((128 << 20,), dtype=torch.uint8, device=dev)

    def _graph_time(fn, reps):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(reps):
                fn()
        g.replay()
        torch.cuda.synchronize()
        best = 1e9
        for _ in range(4):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); g.replay(); e1.record()
            torch.cuda.synchronize()
            best = min(best, e0.elapsed_time(e1) * 1000.0 / reps)
        return best

    def timeit(fn, reps=10):
        # as the judge: CUDA-graph replays, each preceded by an L2 flush (its cost measured and removed)
        fn(); fn()
        torch.cuda.synchronize()
        t_flush = _graph_time(lambda: flush.fill_(1), reps)
        t_both = _graph_time(lambda: (flush.fill_(1), fn()), reps)
        return t_both - t_flush

    def budget():
        return time.time() - t0 < BENCH_BUDGET_S

    refs = {}

    def ref_of(layer, m, n):
        key = (layer, m, n)
        if key not in refs:
            x = xs[(m, layers[layer][1])]
            if layer == "router":
                refs[key] = x.float() @ (state["router_hi"].float() + state["router_lo"].float()).T
            else:
                refs[key] = x.float() @ state[layer][:n].float().T
        return refs[key]

    def check(out, ref, layer):
        err = float((out.float() - ref).norm() / ref.norm())
        return "" if err < (2e-4 if layer == "router" else 5e-3) else f"!{err:.0e}"

    def row(label, layer, m, n, cfg):
        global _OVERRIDE
        x = xs[(m, layers[layer][1])]
        _OVERRIDE = cfg
        try:
            if cfg is not None and not cfg.get("none") and not _usable(cfg, m, n, layers[layer][1]):
                return f"{label} n/a"
            out = _proj_forward(x, layer, n, state)
            t = timeit(lambda: _proj_forward(x, layer, n, state))
            return f"{label} {t:5.1f}{check(out, ref_of(layer, m, n), layer)}"
        except Exception as e:
            print("   fail", layer, m, label, str(e).splitlines()[0][:90])
            return f"{label} fail"
        finally:
            _OVERRIDE = None

    print("BENCH proj (us per call, cold L2; '!' = rel err above tol)")
    try:
        ext = build()
        print(f"  cuda built {time.time() - t0:.0f}s")
        # ---- A. streaming ceiling: dot-free reads of the pooler weight (67 MB)
        wbuf = state["pooler"]; nbytes = wbuf.numel() * 2
        sink = torch.empty((4096 * 16,), dtype=torch.int32, device=dev)
        for threads, unroll in ((256, 4), (256, 8), (256, 16), (512, 8), (128, 16)):
            parts = []
            for grid in (132, 264, 528, 1056, 2112):
                t = timeit(lambda: ext.stream_bytes(wbuf, sink, grid, threads, unroll))
                parts.append(f"g{grid} {t:5.1f} ({nbytes / t / 1e6:.2f})")
            print(f"  stream t{threads} u{unroll}: " + " | ".join(parts))
        obuf = state["o"]
        parts = []
        for grid in (264, 528, 1056):
            t = timeit(lambda: ext.stream_bytes(obuf, sink, grid, 256, 8))
            parts.append(f"g{grid} {t:5.1f} ({obuf.numel() * 2 / t / 1e6:.2f})")
        print("  stream o-weight 16MB t256 u8: " + " | ".join(parts))
        # ---- B. register GEMV at m=1 / m=8 vs the Triton config in use
        gemv1 = [(1, 1, 8, 1), (1, 2, 8, 1), (1, 4, 4, 1), (1, 4, 8, 1), (1, 2, 4, 1), (1, 1, 4, 2), (1, 1, 8, 2), (1, 1, 4, 4), (1, 2, 2, 2)]
        gemv1_router = [(1, 1, 4, 8), (1, 1, 8, 4), (1, 1, 2, 8)]
        gemv8 = [(8, 1, 4, 1), (8, 1, 8, 1), (8, 2, 4, 1), (8, 1, 4, 2), (8, 1, 2, 2)]
        gemv8_router = [(8, 1, 4, 8), (8, 1, 2, 8)]
        for m, cands, cands_r in ((1, gemv1, gemv1_router), (8, gemv8, gemv8_router)):
            for layer in ("pooler", "qkv_gate", "o", "unpooler", "router"):
                if not budget():
                    break
                N, K = layers[layer]
                n = 1024 if (layer == "unpooler" and m <= 8) else N
                parts = [row("triton", layer, m, n, None)]
                for mpad, rpw, unroll, ksplit in (cands_r if layer == "router" else cands):
                    parts.append(row(f"g{rpw}/{unroll}/{ksplit}", layer, m, n, dict(gemv=True, mpad=mpad, rpw=rpw, unroll=unroll, ksplit=ksplit)))
                print(f"  m={m} {layer:9s}: " + " | ".join(parts))
    except Exception as e:
        print("  cuda part failed:", repr(e)[:200])
    _OVERRIDE = None
    # ---- D. m=128/256: cuBLAS vs batched split-K vs Triton wgmma (row-major TMA and (128,64) tiles)
    print("BENCH m=128/256 (BLOCK_N, BLOCK_K, SPLIT_K, warps, stages, tiled, swap)")
    big = [(128, 64, 4, 8, 4, True, False), (128, 64, 4, 8, 6, True, False), (128, 64, 2, 8, 6, True, False),
           (128, 64, 4, 8, 4, False, False), (128, 64, 4, 8, 6, False, False), (128, 64, 4, 8, 6, True, True),
           (64, 64, 4, 4, 8, True, True), (256, 64, 8, 8, 4, False, False)]
    for layer in ("pooler", "unpooler", "qkv_gate", "o"):
        N, K = layers[layer]
        for m in (128, 256):
            if not budget():
                break
            parts = [row("cuBLAS", layer, m, N, {"none": True})]
            for sk in (2, 4, 8):
                parts.append(row(f"bmm{sk}", layer, m, N, dict(bmm=True, SPLIT_K=sk)))
            for bn, bk, sk, nw, ns, tiled, swap in big:
                cfg = dict(BLOCK_M=128, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=sk, num_warps=nw, num_stages=ns, tma=True, big=True,
                           tiled=tiled, swap_mn=swap)
                parts.append(row(f"{(bn, bk, sk, nw, ns, int(tiled), int(swap))}", layer, m, N, cfg))
            print(f"  {layer:9s} m={m}: " + " | ".join(parts))
    # ---- E. router m=128/256: cat + cuBLAS vs bmm(hi, lo) vs Triton two-weight GEMM
    print("BENCH router m=128/256 (BLOCK_N, BLOCK_K, SPLIT_K, warps, stages, tiled)")
    rbig = [(32, 64, 16, 8, 4, False), (32, 64, 16, 8, 6, False), (64, 64, 8, 8, 4, False), (32, 128, 8, 8, 4, False),
            (32, 64, 16, 8, 6, True), (32, 128, 16, 8, 4, True), (64, 64, 16, 8, 4, False)]
    for m in (128, 256):
        if not budget():
            break
        parts = [row("cat+mm", "router", m, 256, {"none": True}), row("bmm2", "router", m, 256, dict(bmm=True, SPLIT_K=2))]
        for bn, bk, sk, nw, ns, tiled in rbig:
            cfg = dict(BLOCK_M=128, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=sk, num_warps=nw, num_stages=ns, tma=True, tiled=tiled)
            parts.append(row(f"{(bn, bk, sk, nw, ns, int(tiled))}", "router", m, 256, cfg))
        print(f"  router m={m}: " + " | ".join(parts))
    # ---- C. Triton m=16/32: deeper / narrower pipelines (tiled packings), 2 CTAs per SM, m-tile pairs
    print("BENCH m=16/32 Triton (BLOCK_N, BLOCK_K, SPLIT_K, warps, stages, tma); m32/bm16 = two m-tiles, m-fastest grid")
    plan = {
        "pooler":   [(32, 128, 1, 4, 8, True), (32, 128, 1, 4, 12, True), (32, 128, 2, 4, 8, True), (64, 128, 1, 4, 6, True),
                     (32, 128, 1, 4, 8, False)],
        "qkv_gate": [(32, 128, 4, 4, 8, True), (32, 128, 2, 4, 8, True), (64, 128, 2, 4, 6, True), (32, 128, 4, 4, 8, False),
                     (32, 256, 2, 4, 6, True)],
        "unpooler": [(32, 128, 1, 4, 8, True), (32, 128, 2, 4, 8, True), (64, 128, 1, 4, 6, True)],
        "o":        [(32, 128, 1, 4, 8, True), (64, 128, 1, 4, 6, True), (32, 256, 1, 4, 6, True)],
        "router":   [(32, 128, 16, 4, 6, True), (32, 64, 16, 4, 8, True), (32, 128, 8, 4, 8, True)],
    }
    full = ("pooler", "qkv_gate")
    for layer, cfgs in plan.items():
        N, K = layers[layer]
        cur = f"cur m16 {row('', layer, 16, N, None)} m32 {row('', layer, 32, N, None)}"
        for bn, bk, sk, nw, ns, tma in cfgs:
            if not budget():
                break
            parts = []
            for m, bm, swap in ((16, 16, False), (32, 32, False), (32, 16, True)):
                if bm == 32 and layer not in full:
                    continue
                cfg = dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=sk, num_warps=nw, num_stages=ns, tiled=True, tma=tma, swap_mn=swap)
                parts.append(row(f"m{m}/bm{bm}", layer, m, N, cfg))
            print(f"  {layer:9s} {str((bn, bk, sk, nw, ns, tma)):30s} {cur} | " + " | ".join(parts))
    _OVERRIDE = None
    print(f"BENCH proj done in {time.time() - t0:.0f}s")


def prepare(weights: Weights, warm: bool = True):
    """Called once, untimed; then every scored shape is run once so the judge never compiles
    under capture."""
    state = _proj_state(weights)
    device = state["router"].device
    if not warm:
        return state
    if BENCH:
        try:
            _bench(state)
        except Exception as e:
            print("BENCH failed:", repr(e))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    # Warm-up / compile for the shapes the judge will run (tests and scored), with throwaway x.
    layers = {"qkv_gate": 8192, "o": 1024, "pooler": 8192, "unpooler": 4096, "router": 8192}
    try:
        for m in (1, 5, 7, 8, 9, 16, 32, 33, 128, 130, 256):
            for layer, k in layers.items():
                n = state[layer].shape[0]
                if layer == "unpooler" and m <= 8:
                    n = 1024
                x = torch.randn((m, k), device=device).to(torch.bfloat16)
                kernel_fn({"x": x, "config": {"m": m, "n": n, "k": k, "layer": layer}}, state)
        torch.cuda.synchronize()
    except Exception as e:  # never let warm-up sink the run
        print("warm-up failed:", repr(e))
    return state


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    cfg_in = data["config"]
    return _proj_forward(data["x"], cfg_in["layer"], cfg_in["n"], state)
