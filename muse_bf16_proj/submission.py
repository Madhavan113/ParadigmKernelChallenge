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


# == BEGIN CUDA ==
CUDA_SRC = r"""
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
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t& b0, uint32_t& b1, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(b0), "=r"(b1) : "r"(addr));
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

constexpr int KC = 64;             // K chunk per warp step
constexpr int STRIDE = KC + 8;     // padded smem row stride (bf16 elements)
constexpr int NWARPS = 4;
constexpr int NSTAGE = 4;

// BM in {16, 32}: rows (M <= BM, zero-filled); BN in {16, 32}: output columns per CTA.
template <int BM, int BN, bool TWO_W>
__global__ void __launch_bounds__(128)
skinny_gemm_kernel(const __nv_bfloat16* __restrict__ X, int ldx, int M,
                   const __nv_bfloat16* __restrict__ W, const __nv_bfloat16* __restrict__ W2, int ldw,
                   __nv_bfloat16* __restrict__ out_bf, float* __restrict__ out_f32, int ldo, int K)
{
    constexpr int NW = TWO_W ? 2 : 1;
    constexpr int MT = BM / 16, NT = BN / 8;
    const int n0 = blockIdx.x * BN;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;

    // per-warp pipeline buffers
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16* smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    constexpr int XS = BM * STRIDE, WS = BN * STRIDE;
    constexpr int STAGE = XS + NW * WS;                // elements per stage
    __nv_bfloat16* my = smem + warp * (NSTAGE * STAGE);

    const int NC = K / KC;                              // chunks in K
    // this warp's chunks: c = warp, warp + 4, ...
    auto issue = [&](int c, int st) {
        __nv_bfloat16* xs = my + st * STAGE;
        __nv_bfloat16* ws = xs + XS;
        const int k0 = c * KC;
        // X: BM rows x 128B (8 segments of 16B) -> BM*8 copies over 32 lanes
        #pragma unroll
        for (int i = 0; i < (BM * 8) / 32; ++i) {
            const int idx = lane + 32 * i, r = idx >> 3, seg = idx & 7;
            const bool ok = r < M;
            const __nv_bfloat16* src = X + (size_t)(ok ? r : 0) * ldx + k0 + seg * 8;
            cp_async16(smem_u32(xs + r * STRIDE + seg * 8), src, ok ? 16 : 0);
        }
        // W: BN rows x 128B
        #pragma unroll
        for (int i = 0; i < (BN * 8) / 32; ++i) {
            const int idx = lane + 32 * i, r = idx >> 3, seg = idx & 7;
            cp_async16(smem_u32(ws + r * STRIDE + seg * 8), W + (size_t)(n0 + r) * ldw + k0 + seg * 8, 16);
            if (TWO_W)
                cp_async16(smem_u32(ws + WS + r * STRIDE + seg * 8), W2 + (size_t)(n0 + r) * ldw + k0 + seg * 8, 16);
        }
    };

    float acc[MT][NT][4];
    #pragma unroll
    for (int m = 0; m < MT; ++m)
        #pragma unroll
        for (int n = 0; n < NT; ++n)
            #pragma unroll
            for (int q = 0; q < 4; ++q) acc[m][n][q] = 0.f;

    int c = warp;
    #pragma unroll
    for (int st = 0; st < NSTAGE - 1; ++st) {
        if (c + st * NWARPS < NC) issue(c + st * NWARPS, st);
        cp_async_commit();
    }
    int st = 0;
    for (; c < NC; c += NWARPS, st = (st + 1) % NSTAGE) {
        const int cn = c + (NSTAGE - 1) * NWARPS;
        if (cn < NC) issue(cn, (st + NSTAGE - 1) % NSTAGE);
        cp_async_commit();
        cp_async_wait<NSTAGE - 1>();
        __syncwarp();
        const __nv_bfloat16* xs = my + st * STAGE;
        const __nv_bfloat16* ws = xs + XS;
        #pragma unroll
        for (int k = 0; k < KC / 16; ++k) {
            uint32_t a[MT][4];
            #pragma unroll
            for (int m = 0; m < MT; ++m)
                ldmatrix_x4(a[m][0], a[m][1], a[m][2], a[m][3],
                            smem_u32(xs + (16 * m + (lane & 15)) * STRIDE + 16 * k + (lane >> 4) * 8));
            #pragma unroll
            for (int w2 = 0; w2 < NW; ++w2) {
                #pragma unroll
                for (int n = 0; n < NT; ++n) {
                    uint32_t b0, b1;
                    // 8 rows n, k 0-7 (lanes 0-7) and k 8-15 (lanes 8-15)
                    ldmatrix_x2(b0, b1, smem_u32(ws + w2 * WS + (8 * n + (lane & 7)) * STRIDE + 16 * k + ((lane >> 3) & 1) * 8));
                    #pragma unroll
                    for (int m = 0; m < MT; ++m) mma16816(acc[m][n], a[m], b0, b1);
                }
            }
        }
        __syncwarp();
    }
    cp_async_wait<0>();

    // cross-warp reduction through shared memory (reuse the pipeline buffers)
    __syncthreads();
    float* red = reinterpret_cast<float*>(smem_raw);     // [NWARPS][BM][BN]
    #pragma unroll
    for (int m = 0; m < MT; ++m)
        #pragma unroll
        for (int n = 0; n < NT; ++n) {
            const int r0 = 16 * m + g, cc = 8 * n + 2 * t;
            red[(warp * BM + r0) * BN + cc] = acc[m][n][0];
            red[(warp * BM + r0) * BN + cc + 1] = acc[m][n][1];
            red[(warp * BM + r0 + 8) * BN + cc] = acc[m][n][2];
            red[(warp * BM + r0 + 8) * BN + cc + 1] = acc[m][n][3];
        }
    __syncthreads();
    for (int idx = threadIdx.x; idx < BM * BN; idx += 128) {
        const int r = idx / BN, cc = idx % BN;
        if (r >= M) continue;
        float v = 0.f;
        #pragma unroll
        for (int w = 0; w < NWARPS; ++w) v += red[(w * BM + r) * BN + cc];
        if (out_f32) out_f32[(size_t)r * ldo + n0 + cc] = v;
        else out_bf[(size_t)r * ldo + n0 + cc] = __float2bfloat16(v);
    }
}

template <int BM, int BN, bool TWO_W>
void launch(const __nv_bfloat16* X, int ldx, int M, const __nv_bfloat16* W, const __nv_bfloat16* W2, int ldw,
            __nv_bfloat16* ob, float* of, int ldo, int K, int N, cudaStream_t stream)
{
    constexpr int smem = NWARPS * NSTAGE * (BM * STRIDE + (TWO_W ? 2 : 1) * BN * STRIDE) * 2;
    static bool attr = false;
    if (!attr) { cudaFuncSetAttribute(skinny_gemm_kernel<BM, BN, TWO_W>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
    skinny_gemm_kernel<BM, BN, TWO_W><<<N / BN, 128, smem, stream>>>(X, ldx, M, W, W2, ldw, ob, of, ldo, K);
}

} // namespace

void skinny_gemm(torch::Tensor X, torch::Tensor W, torch::Tensor W2, torch::Tensor out, int64_t bm, int64_t bn, bool two_w)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    const int M = (int)X.size(0), K = (int)X.size(1), N = (int)W.size(0);
    const __nv_bfloat16* Xp = reinterpret_cast<const __nv_bfloat16*>(X.data_ptr());
    const __nv_bfloat16* Wp = reinterpret_cast<const __nv_bfloat16*>(W.data_ptr());
    const __nv_bfloat16* W2p = two_w ? reinterpret_cast<const __nv_bfloat16*>(W2.data_ptr()) : nullptr;
    __nv_bfloat16* ob = out.scalar_type() == at::kBFloat16 ? reinterpret_cast<__nv_bfloat16*>(out.data_ptr()) : nullptr;
    float* of = out.scalar_type() == at::kFloat ? out.data_ptr<float>() : nullptr;
    const int ldx = (int)X.stride(0), ldw = (int)W.stride(0), ldo = (int)out.stride(0);
    const int key = (int)bm * 100 + (int)bn * 10 + (two_w ? 1 : 0);
    switch (key) {
        case 1616 + 0: launch<16, 16, false>(Xp, ldx, M, Wp, W2p, ldw, ob, of, ldo, K, N, stream); break;
        case 1616 + 1: launch<16, 16, true>(Xp, ldx, M, Wp, W2p, ldw, ob, of, ldo, K, N, stream); break;
        case 1616 + 160: launch<16, 32, false>(Xp, ldx, M, Wp, W2p, ldw, ob, of, ldo, K, N, stream); break;
        case 1616 + 161: launch<16, 32, true>(Xp, ldx, M, Wp, W2p, ldw, ob, of, ldo, K, N, stream); break;
        case 3216 + 0: launch<32, 16, false>(Xp, ldx, M, Wp, W2p, ldw, ob, of, ldo, K, N, stream); break;
        case 3216 + 1: launch<32, 16, true>(Xp, ldx, M, Wp, W2p, ldw, ob, of, ldo, K, N, stream); break;
        case 3216 + 160: launch<32, 32, false>(Xp, ldx, M, Wp, W2p, ldw, ob, of, ldo, K, N, stream); break;
        case 3216 + 161: launch<32, 32, true>(Xp, ldx, M, Wp, W2p, ldw, ob, of, ldo, K, N, stream); break;
        default: TORCH_CHECK(false, "unsupported (bm, bn)");
    }
}
"""

CPP_SRC = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
void skinny_gemm(torch::Tensor X, torch::Tensor W, torch::Tensor W2, torch::Tensor out, int64_t bm, int64_t bn, bool two_w);
"""

_EXT = None


def build():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline
        _EXT = load_inline(
            name="okc_skinny_gemm", cpp_sources=[CPP_SRC],
            cuda_sources=["#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n" + CUDA_SRC],
            functions=["skinny_gemm"], extra_cuda_cflags=["-O3", "-arch=sm_90a"], verbose=False)
    return _EXT
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
USE_TILED = True
USE_TMA = False  # fetch tiles with TMA descriptors (box inner dim <= 256 elements)


_OVERRIDE = None   # a config dict, during in-prepare benchmarking


CUDA_SHAPES = {}   # (layer, n, m) -> (BM, BN): use the CUDA skinny GEMM for this shape


def _config(layer: str, m: int, n: int, k: int):
    if _OVERRIDE is not None:
        return None if _OVERRIDE.get("none") else dict(_OVERRIDE)
    if (layer, n, m) in CUDA_SHAPES and m <= 32:
        bm, bn = CUDA_SHAPES[(layer, n, m)]
        return dict(cuda=True, BM=bm, BN=bn, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=64, SPLIT_K=1, num_warps=4, num_stages=4)
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
        cfg["gemv"] = GEMV_M1 and m == 1
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
GEMV_M1 = False  # m == 1: plain-FMA GEMV instead of the tensor-core path


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
        if m > 64:
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
    if cfg.get("cuda"):
        return m <= cfg["BM"] and n % cfg["BN"] == 0 and k % 64 == 0
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
    if cfg.get("cuda"):
        ext = build()
        if layer == "router":
            ext.skinny_gemm(x, state["router_hi"], state["router_lo"], out, cfg["BM"], cfg["BN"], True)
        else:
            ext.skinny_gemm(x, w, w, out, cfg["BM"], cfg["BN"], False)
        return out
    m_tiles = triton.cdiv(m, cfg["BLOCK_M"])
    n_tiles = n // cfg["BLOCK_N"]
    grid = (n_tiles, cfg["SPLIT_K"], m_tiles)
    if cfg.get("tiled"):
        # the packed weight: n-major tiles, so the unpooler's first 1024 rows are its first tiles
        if layer == "router":
            wt, w2t = state["router_hi_tiled"], state["router_lo_tiled"]
        else:
            wt, w2t = state[layer + "_tiled"], None
        if cfg.get("gemv") and m == 1:
            _tiled_gemv_kernel[(n_tiles, cfg["SPLIT_K"])](
                x, wt, w2t if w2t is not None else wt, out, state["partials"], state["counters"],
                n, k,
                BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"], SPLIT_K=cfg["SPLIT_K"], TWO_W=w2t is not None,
                num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
            return out
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
# == END SHARED proj ==


BENCH = True        # microbenchmarks in prepare (untimed; results go to the run log)
BENCH_BUDGET_S = 200


def _bench(state):
    import time
    global _OVERRIDE
    t0 = time.time()
    dev = state["router"].device
    layers = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096),
              "router": (256, 8192)}
    xs = {(m, k): torch.randn((m, k), device=dev).to(torch.bfloat16) for m in (1, 8, 16, 32) for k in (8192, 1024, 4096)}

    def timeit(fn, reps=20):
        # as the judge: CUDA-graph replays (host launch cost, e.g. TMA descriptor encoding, excluded)
        fn(); fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(reps):
                fn()
        g.replay()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(3):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) * 1000.0 / (3 * reps)

    print("BENCH proj: us per call (warm L2), configs = (BLOCK_N, BLOCK_K, SPLIT_K, warps, stages, tma)")
    plan = {} if True else {
        "pooler":   [(32, 256, 1, 4, 4, True), (32, 256, 1, 4, 6, True), (32, 256, 1, 4, 8, True), (32, 256, 2, 4, 4, True),
                     (32, 256, 1, 4, 4, False), (32, 256, 1, 4, 6, False), (32, 256, 1, 2, 6, True), (32, 256, 1, 8, 6, True)],
        "unpooler": [(32, 256, 1, 4, 4, True), (32, 256, 1, 4, 6, True), (32, 256, 1, 4, 8, True), (32, 256, 2, 4, 4, True),
                     (32, 256, 1, 4, 6, False), (32, 256, 4, 4, 4, True), (32, 256, 4, 4, 6, True), (32, 256, 8, 4, 4, True)],
        "o":        [(64, 256, 1, 4, 4, True), (64, 256, 1, 4, 5, True), (64, 256, 1, 4, 4, False), (64, 256, 1, 2, 4, True),
                     (64, 256, 1, 8, 4, True), (64, 256, 2, 4, 4, True)],
        "qkv_gate": [(32, 256, 4, 4, 4, True), (32, 256, 4, 4, 6, True), (32, 256, 2, 4, 6, True), (32, 256, 8, 4, 4, True),
                     (32, 256, 4, 4, 4, False), (32, 256, 2, 4, 4, False), (32, 256, 1, 4, 6, True)],
        "router":   [(32, 256, 16, 4, 3, True), (32, 256, 16, 4, 3, False), (32, 256, 8, 4, 4, True), (32, 256, 8, 4, 6, True),
                     (32, 256, 16, 4, 4, True), (32, 256, 32, 4, 3, True)],
    }
    print("BENCH proj small m: current Triton config vs CUDA skinny GEMM (BM, BN)")
    try:
        build()
        for layer in ("pooler", "unpooler", "o", "qkv_gate", "router"):
            N, K = layers[layer]
            for m in (1, 8, 16, 32):
                n = 1024 if (layer == "unpooler" and m <= 8) else N
                x = xs[(m, K)]
                _OVERRIDE = None
                row = [f"triton {timeit(lambda: _proj_forward(x, layer, n, state)):6.1f}"]
                for bm, bn in ((16, 16), (16, 32), (32, 16), (32, 32)):
                    if m > bm or n % bn:
                        continue
                    _OVERRIDE = dict(cuda=True, BM=bm, BN=bn, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=64, SPLIT_K=1, num_warps=4, num_stages=4)
                    try:
                        row.append(f"cuda{(bm, bn)} {timeit(lambda: _proj_forward(x, layer, n, state)):6.1f}")
                    except Exception as e:
                        row.append(f"cuda{(bm, bn)} fail:{str(e)[:40]}")
                _OVERRIDE = None
                print(f"  {layer:9s} m={m:3d} n={n}: " + " | ".join(row))
    except Exception as e:
        print("  cuda skinny gemm unavailable:", repr(e)[:160])
    _OVERRIDE = None
    print("BENCH proj large m: cuBLAS vs TMA wgmma split-K (BLOCK_N, BLOCK_K, SPLIT_K, warps, stages)")
    xbig = {(m, k): torch.randn((m, k), device=dev).to(torch.bfloat16) for m in (128, 256) for k in (8192, 1024, 4096)}
    for layer in ():
        N, K = layers[layer]
        row = []
        for m in (128, 256):
            _OVERRIDE = {"none": True}
            row.append(f"cuBLAS {timeit(lambda: torch.matmul(xbig[(m, K)], state[layer].t())):6.1f}")
            for bn, bk, sk, nw, ns in ((128, 64, 1, 8, 4), (128, 64, 2, 8, 4), (128, 64, 4, 8, 4), (128, 64, 8, 8, 4),
                                       (64, 64, 2, 4, 4), (64, 64, 4, 4, 4), (128, 128, 2, 8, 3), (256, 64, 2, 8, 3)):
                _OVERRIDE = dict(BLOCK_M=128, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=sk, num_warps=nw, num_stages=ns, tma=True, big=True)
                try:
                    if not _usable(_OVERRIDE, m, N, K):
                        row.append("n/a"); continue
                    row.append(f"{(bn, bk, sk, nw, ns)} {timeit(lambda: _proj_forward(xbig[(m, K)], layer, N, state)):6.1f}")
                except Exception as e:
                    row.append(f"{(bn, bk, sk, nw, ns)} fail")
            print(f"  {layer:9s} m={m}: " + " | ".join(row)); row = []
        _OVERRIDE = None
    for layer, cfgs in plan.items():
        N, K = layers[layer]
        for bn, bk, sk, nw, ns, tma in cfgs:
            if time.time() - t0 > BENCH_BUDGET_S:
                print("BENCH budget reached"); break
            row = []
            for m in (1, 8, 16, 32):
                n = 1024 if (layer == "unpooler" and m <= 8) else N
                _OVERRIDE = dict(BLOCK_M=_block_m(m), BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=sk, num_warps=nw, num_stages=ns,
                                 tiled=True, tma=tma)
                try:
                    if not _usable(_OVERRIDE, m, n, K):
                        row.append("   n/a"); continue
                    x = xs[(m, K)]
                    row.append(f"{timeit(lambda: _proj_forward(x, layer, n, state)):6.1f}")
                except Exception as e:
                    row.append(" fail")
                    print("   ", layer, (bn, bk, sk, nw, ns, tma), m, str(e)[:100])
            _OVERRIDE = None
            print(f"  {layer:9s} {str((bn, bk, sk, nw, ns, tma)):32s} m1 {row[0]} m8 {row[1]} m16 {row[2]} m32 {row[3]}")
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
