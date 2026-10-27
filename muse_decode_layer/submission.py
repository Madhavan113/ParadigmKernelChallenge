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
def _decode12(L, Nb, lut0, lut1, ASM: tl.constexpr):
    """L: [R, C] uint32 of 4 raw low bytes, Nb: [R, C] uint32 of 4 nibbles (sign << 3 | off >> 1):
    returns the 4 bf16 bit patterns as uint16 e0..e3 (elements 4j .. 4j + 3)."""
    if ASM:
        w01, w23 = tl.inline_asm_elementwise(
            "{ .reg .b32 h, t, e, u; and.b32 t, $3, 0x7777; prmt.b32 h, $4, $5, t; prmt.b32 e, $3, 0, 0x1100; "
            "and.b32 u, e, 0x00080008; shl.b32 u, u, 4; and.b32 e, e, 0x80008000; or.b32 u, u, e; or.b32 h, h, u; "
            "prmt.b32 $0, $2, h, 0x5140; prmt.b32 $1, $2, h, 0x7362; }",
            "=r,=r,r,r,r,r", [L, Nb, lut0, lut1], dtype=(tl.uint32, tl.uint32), is_pure=True, pack=1)
    else:
        hb = lut0 & 0xFF
        k0 = Nb & 7
        k1 = (Nb >> 4) & 7
        k2 = (Nb >> 8) & 7
        k3 = (Nb >> 12) & 7
        s0 = (Nb >> 3) & 1
        s1 = (Nb >> 7) & 1
        s2 = (Nb >> 11) & 1
        s3 = (Nb >> 15) & 1
        h0 = (hb + k0) | (s0 << 7)
        h1 = (hb + k1) | (s1 << 7)
        h2 = (hb + k2) | (s2 << 7)
        h3 = (hb + k3) | (s3 << 7)
        w01 = (L & 0xFF) | (h0 << 8) | (((L >> 8) & 0xFF) << 16) | (h1 << 24)
        w23 = ((L >> 16) & 0xFF) | (h2 << 8) | ((L >> 24) << 16) | (h3 << 24)
    e0 = (w01 & 0xFFFF).to(tl.uint16)
    e1 = (w01 >> 16).to(tl.uint16)
    e2 = (w23 & 0xFFFF).to(tl.uint16)
    e3 = (w23 >> 16).to(tl.uint16)
    return e0, e1, e2, e3


@triton.jit
def _tiled12_gemm_kernel(
    x_ptr, lo_ptr, nb_ptr, out_ptr, part_ptr, cnt_ptr,
    esc_off_ptr, esc_n_ptr, esc_k_ptr, esc_d_ptr,
    M, N, K, stride_xm, stride_om, EB,
    BLOCK_M: tl.constexpr, SPLIT_K: tl.constexpr, ASM: tl.constexpr, INTERP: tl.constexpr,
):
    """x @ W.T over the 12-bit packed weight (tiles of 32 rows x 256 k: lo_ptr = uint32 view of the
    low bytes [tiles * 32, 64], nb_ptr = uint16 view of the nibbles [tiles * 32, 64]); the tile is
    decoded to bf16 in registers before tl.dot. Split-K as _tiled_gemm_kernel (ticket); the
    finishing CTA adds the tile's escapes (n_local, k, delta)."""
    BLOCK_N: tl.constexpr = 32
    BLOCK_K: tl.constexpr = 256
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    n_tiles = tl.num_programs(0)
    m_tiles = tl.num_programs(2)
    KT = K // BLOCK_K
    kt_chunk = KT // SPLIT_K
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    k0 = tl.multiple_of(pid_k * kt_chunk * BLOCK_K, BLOCK_K)
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :]
    rows = tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_K // 4)
    tile0 = (pid_n * KT + pid_k * kt_chunk) * (BLOCK_N * BLOCK_K // 4)   # uint32 / uint16 units per tile
    l_ptrs = lo_ptr + tile0 + rows[:, None] * (BLOCK_K // 4) + cols[None, :]
    n_ptrs = nb_ptr + tile0 + rows[:, None] * (BLOCK_K // 4) + cols[None, :]
    hb = (EB >> 1).to(tl.uint32)
    lut0 = tl.zeros((BLOCK_N, BLOCK_K // 4), dtype=tl.uint32) + (hb | ((hb + 1) << 8) | ((hb + 2) << 16) | ((hb + 3) << 24))
    lut1 = tl.zeros((BLOCK_N, BLOCK_K // 4), dtype=tl.uint32) + ((hb + 4) | ((hb + 5) << 8) | ((hb + 6) << 16) | ((hb + 7) << 24))
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, kt_chunk):
        x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        L = tl.load(l_ptrs).to(tl.uint32, bitcast=True)
        Nb = tl.load(n_ptrs).to(tl.uint16, bitcast=True).to(tl.uint32)
        e0, e1, e2, e3 = _decode12(L, Nb, lut0, lut1, ASM)
        # join(a, b) adds a NEW last axis; reshape flattens the outer join fastest: (e0 e1 e2 e3) needs join(join(e0, e2), join(e1, e3))
        w16 = tl.join(tl.join(e0, e2), tl.join(e1, e3)).reshape(BLOCK_N, BLOCK_K)
        if INTERP:   # CPU interpreter: no bf16 there, so widen the bit patterns to fp32 and dot in fp32
            w32 = (w16.to(tl.uint32) << 16).to(tl.float32, bitcast=True)
            acc = tl.dot(x.to(tl.float32), tl.trans(w32), acc)
        else:
            acc = tl.dot(x, tl.trans(w16.to(tl.bfloat16, bitcast=True)), acc)
        x_ptrs += BLOCK_K
        l_ptrs += BLOCK_N * BLOCK_K // 4
        n_ptrs += BLOCK_N * BLOCK_K // 4
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    if SPLIT_K == 1:
        total = acc
        finish = True
    else:
        m_pad = m_tiles * BLOCK_M
        part_ptrs = part_ptr + (pid_k * m_pad + offs_m)[:, None] * N + offs_n[None, :]
        tl.store(part_ptrs, acc)
        tl.debug_barrier()
        tile = pid_m * n_tiles + pid_n
        ticket = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel", scope="gpu")
        tl.debug_barrier()
        finish = ticket == SPLIT_K - 1
        total = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if finish:
            for s_ in tl.static_range(SPLIT_K):
                p = part_ptr + (s_ * m_pad + offs_m)[:, None] * N + offs_n[None, :]
                total += tl.load(p, cache_modifier=".cg")
    if finish:
        e_beg = tl.load(esc_off_ptr + pid_n)
        e_end = tl.load(esc_off_ptr + pid_n + 1)
        ECH: tl.constexpr = 16
        for e in range(e_beg, e_end, ECH):
            idx = e + tl.arange(0, ECH)
            emask = idx < e_end
            nl = tl.load(esc_n_ptr + idx, mask=emask, other=-1)
            kk = tl.load(esc_k_ptr + idx, mask=emask, other=0)
            d = tl.load(esc_d_ptr + idx, mask=emask, other=0.0)
            xe = tl.load(x_ptr + offs_m[:, None] * stride_xm + kk[None, :], mask=m_mask[:, None] & emask[None, :], other=0.0).to(tl.float32)
            contrib = xe * d[None, :]                                           # [BM, ECH]
            onehot = (nl[None, :] == rows[:, None]).to(tl.float32)              # [BN, ECH]
            total += tl.sum(contrib[:, None, :] * onehot[None, :, :], axis=2)   # [BM, BN]
        tl.store(out_ptrs, total.to(out_ptr.dtype.element_ty), mask=m_mask[:, None])
        if SPLIT_K > 1:
            tl.atomic_xchg(cnt_ptr + pid_m * n_tiles + pid_n, 0)


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


# ----------------------------------------------------------------------------------------------
# Lossless 12-bit weight packing
# ----------------------------------------------------------------------------------------------
# A BF16 weight is sign(1) | exponent(8) | mantissa(7). The exponents of a layer's nonzero weights
# span ~27 values, and 16 consecutive exponents [eb, eb + 15] (eb even, reaching the maximum) hold
# all but ~1e-4 of them. Per element we keep the raw low byte (bit 7 = exponent LSB, bits 6..0 =
# mantissa) and a nibble (sign << 3) | (off >> 1) with off = exponent - eb; the decoder rebuilds
# the high byte as (sign << 7) | ((eb >> 1) + (off >> 1)). Elements outside the window (including
# exact zeros) are stored with off = 0 (true low byte, so a wrong exponent) and listed as escapes
# (n, k, true - stored) that the kernels add in their epilogue. Tiles are 32 rows x 256 k, n-major:
# low bytes [tiles, 32, 256] uint8 (8 KB per tile), nibbles [tiles, 32, 128] uint8 (4 KB per tile),
# two per byte (element 2j in the low nibble). Storage is exact: 12 bits + the escape list.
PACK_BN, PACK_BK = 32, 256
PACK_ESCAPE_CAP = 4e-3   # fraction of elements; above it the tensor stays in plain BF16 tiles


def _pack12(w: torch.Tensor):
    """w [N, K] bf16 -> dict(lo, nib, eb, esc_off, esc_n, esc_k, esc_delta, n_esc) or None if the
    escape count exceeds PACK_ESCAPE_CAP."""
    n, k = w.shape
    assert n % PACK_BN == 0 and k % PACK_BK == 0
    bits = w.view(torch.int16).to(torch.int32) & 0xFFFF
    exp = (bits >> 7) & 0xFF
    sign = bits >> 15
    nzexp = exp[exp > 0]
    emax = int(nzexp.max()) if nzexp.numel() else 1
    eb = emax - 15
    if eb < 1:
        eb = 1
    if eb & 1:
        eb += 1
    off = exp - eb
    ok = (exp > 0) & (off >= 0) & (off <= 15)
    n_esc = int((~ok).sum())
    if n_esc > PACK_ESCAPE_CAP * w.numel():
        return None
    off = torch.where(ok, off, torch.zeros_like(off))
    nib = (sign << 3) | (off >> 1)                                   # [N, K] 0..15
    lo = (bits & 0xFF).to(torch.uint8)
    # n-major tiles: [N/32, K/256, 32, 256]
    lo_t = lo.view(n // PACK_BN, PACK_BN, k // PACK_BK, PACK_BK).permute(0, 2, 1, 3).contiguous()
    nib_t = nib.view(n // PACK_BN, PACK_BN, k // PACK_BK, PACK_BK // 2, 2).permute(0, 2, 1, 3, 4)
    nib_t = (nib_t[..., 0] | (nib_t[..., 1] << 4)).to(torch.uint8).contiguous()   # [N/32, K/256, 32, 128]
    # escapes, sorted by n-tile
    idx = (~ok).nonzero()
    en, ek = idx[:, 0], idx[:, 1]
    order = torch.argsort(en, stable=True)          # by row: tile-major, rows ascending within a tile
    en, ek = en[order], ek[order]
    tile = en // PACK_BN
    stored_bits = (sign[en, ek] << 15) | ((eb + (exp[en, ek] & 1)) << 7) | (bits[en, ek] & 0x7F)
    stored = stored_bits.to(torch.int16).view(torch.bfloat16).float()
    delta = w[en, ek].float() - stored
    esc_off = torch.zeros((n // PACK_BN + 1,), dtype=torch.int32, device=w.device)
    esc_off[1:] = torch.cumsum(torch.bincount(tile, minlength=n // PACK_BN), 0).to(torch.int32)
    esc_roff = torch.zeros((n + 1,), dtype=torch.int32, device=w.device)   # per-row ranges (rows sorted)
    esc_roff[1:] = torch.cumsum(torch.bincount(en, minlength=n), 0).to(torch.int32)
    if n_esc == 0:   # keep the arrays non-empty for the kernels' pointers
        en = ek = torch.zeros((1,), dtype=torch.int64, device=w.device); delta = torch.zeros((1,), dtype=torch.float32, device=w.device)
    return dict(lo=lo_t.view(-1, PACK_BK), nib=nib_t.view(-1, PACK_BK // 2), eb=eb, n_esc=n_esc, esc_off=esc_off, esc_roff=esc_roff,
                esc_n=(en % PACK_BN).to(torch.int32).contiguous(), esc_k=ek.to(torch.int32).contiguous(), esc_delta=delta.contiguous())


def _unpack12(pk, n: int, k: int) -> torch.Tensor:
    """Reference decoder (the kernels' arithmetic in torch): stored values, before the escapes."""
    lo = pk["lo"].view(n // PACK_BN, k // PACK_BK, PACK_BN, PACK_BK).permute(0, 2, 1, 3).reshape(n, k).to(torch.int32)
    nb = pk["nib"].view(n // PACK_BN, k // PACK_BK, PACK_BN, PACK_BK // 2).permute(0, 2, 1, 3).reshape(n, k // 2).to(torch.int32)
    nib = torch.stack((nb & 0xF, nb >> 4), -1).reshape(n, k)
    hi = ((nib >> 3) << 7) | ((pk["eb"] >> 1) + (nib & 7))
    return ((hi << 8) | lo).to(torch.int16).view(torch.bfloat16)


# == BEGIN CUDA ==
PROJ_CUDA_SRC = r"""
#include <cuda.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ uint4 ldg_stream(const uint4* p) {
    uint4 v;
    asm("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
// volatile variants: ptxas keeps volatile asm statements in program order, so loads written before
// the (volatile) mma asms are issued before them instead of being sunk to the end of the loop body
__device__ __forceinline__ uint4 ldg_stream_v(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ uint2 ldg_v2_v(const uint2* p) {
    uint2 v;
    asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}
__device__ __forceinline__ uint4 ldg_l1_v(const uint4* p) {   // L1-cached (x, re-read by many warps)
    uint4 v;
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
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

// ---- stream variant: every WARP reads its own contiguous chunk (THREADS/32 x gridDim.x streams)
template <int UNROLL, int THREADS>
__global__ void __launch_bounds__(THREADS) stream_warp_kernel(const uint4* __restrict__ p, long n_vec, uint32_t* __restrict__ out)
{
    constexpr int WARPS = THREADS / 32;
    const long chunk = n_vec / ((long)gridDim.x * WARPS);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const uint4* base = p + ((long)blockIdx.x * WARPS + warp) * chunk;
    constexpr long BATCH = (long)UNROLL * 32;
    const long n_iter = (chunk + BATCH - 1) / BATCH;
    uint32_t acc = 0;
    uint4 cur[UNROLL];
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) cur[u] = ldg_stream(base + min((long)lane + u * 32, chunk - 1));
    for (long it = 0; it < n_iter; ++it) {
        const long nxt_it = it + 1 < n_iter ? it + 1 : it;
        const long i0 = nxt_it * BATCH + lane;
        uint4 nxt[UNROLL];
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) nxt[u] = ldg_stream(base + min(i0 + u * 32, chunk - 1));
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) acc ^= cur[u].x ^ cur[u].y ^ cur[u].z ^ cur[u].w;
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) cur[u] = nxt[u];
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc ^= __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) out[blockIdx.x * WARPS + warp] = acc;
}

// ---- tiled GEMV (M = 1) over the packed [N/BN][K/256][BN][256] weight: a CTA walks one n-tile's
// k-tiles, every step reading one contiguous BN x 512 B tile (WARPS warps x RPW rows); SK CTAs
// split the k-tiles and the last one (atomic ticket) sums the fp32 partials.
template <int BN, int WARPS, int UNROLL, int SK>
__global__ void __launch_bounds__(WARPS * 32) tgemv_kernel(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ Wt,
                                                           __nv_bfloat16* __restrict__ out, float* __restrict__ part, int* __restrict__ cnt, int N, int K)
{
    constexpr int RPW = BN / WARPS, TILE = BN * 256, TILE4 = TILE / 8;
    __shared__ int s_last;
    const int pn = blockIdx.x, sk = blockIdx.y;
    const int KT = K / 256, ktc = KT / SK, kt0 = sk * ktc;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const uint4* wp = reinterpret_cast<const uint4*>(Wt + (size_t)(pn * KT + kt0) * TILE + warp * RPW * 256) + lane;
    const uint4* xp = reinterpret_cast<const uint4*>(X + kt0 * 256) + lane;
    float acc[RPW];
    #pragma unroll
    for (int r = 0; r < RPW; ++r) acc[r] = 0.f;
    uint4 cur[UNROLL][RPW];
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u)
        #pragma unroll
        for (int r = 0; r < RPW; ++r) cur[u][r] = ldg_stream(wp + u * TILE4 + r * 32);
    for (int kt = 0; kt < ktc; kt += UNROLL) {
        const int kn = kt + UNROLL < ktc ? kt + UNROLL : kt;
        uint4 nxt[UNROLL][RPW];
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u)
            #pragma unroll
            for (int r = 0; r < RPW; ++r) nxt[u][r] = ldg_stream(wp + (kn + u) * TILE4 + r * 32);
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const uint4 xv = xp[(kt + u) * 32];
            #pragma unroll
            for (int r = 0; r < RPW; ++r) dot8(acc[r], cur[u][r], xv);
        }
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u)
            #pragma unroll
            for (int r = 0; r < RPW; ++r) cur[u][r] = nxt[u][r];
    }
    #pragma unroll
    for (int r = 0; r < RPW; ++r) acc[r] = warp_sum(acc[r]);
    const int n0 = pn * BN + warp * RPW;
    if (SK == 1) {
        if (lane == 0) {
            #pragma unroll
            for (int r = 0; r < RPW; ++r) out[n0 + r] = __float2bfloat16(acc[r]);
        }
        return;
    }
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < RPW; ++r) part[(size_t)sk * N + n0 + r] = acc[r];
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) s_last = (atomicAdd(cnt + pn, 1) == SK - 1);
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    for (int idx = threadIdx.x; idx < BN; idx += WARPS * 32) {
        float v = 0.f;
        #pragma unroll
        for (int s2 = 0; s2 < SK; ++s2) v += __ldcg(part + (size_t)s2 * N + pn * BN + idx);
        out[pn * BN + idx] = __float2bfloat16(v);
    }
    if (threadIdx.x == 0) cnt[pn] = 0;
}

template <int BN, int WARPS, int UNROLL, int SK>
void launch_tgemv(const __nv_bfloat16* X, const __nv_bfloat16* Wt, __nv_bfloat16* out, float* part, int* cnt, int N, int K, cudaStream_t stream)
{
    tgemv_kernel<BN, WARPS, UNROLL, SK><<<dim3(N / BN, SK), WARPS * 32, 0, stream>>>(X, Wt, out, part, cnt, N, K);
}

// ---- 12-bit packed GEMV (M = 1). Tiles of 32 rows x 256 k: low bytes [32][256] (raw BF16 low byte),
// nibbles [32][128] ((sign << 3) | (off >> 1), two per byte). The high byte is rebuilt with one PRMT
// from an 8-entry LUT of (eb >> 1) + k, the sign OR'd in, then two PRMTs interleave low/high bytes
// into bf16x2 words. A half-warp covers one tile row per step (16 lanes x 16 elements); WARPS warps
// cover 2 * WARPS rows per step; SK CTAs split the k-tiles (fp32 partials + ticket). The last CTA
// adds the tile's escapes (n_local, k, delta) and writes bf16.
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel)); return d;
}
__device__ __forceinline__ void decode4(uint32_t L, uint32_t n16, uint32_t lut0, uint32_t lut1, uint32_t& w01, uint32_t& w23) {
    // nibble i of n16 = (sign << 3) | (off >> 1) for element i; E = [b0, b0, b1, b1] puts nibble i in
    // byte i (low nibble for even i, high for odd), so its sign bit is byte bit 3 (even) / 7 (odd)
    const uint32_t E = prmt(n16, 0u, 0x1100u);
    const uint32_t H = prmt(lut0, lut1, n16 & 0x7777u) | ((E & 0x00080008u) << 4) | (E & 0x80008000u);
    w01 = prmt(L, H, 0x5140u);
    w23 = prmt(L, H, 0x7362u);
}
__device__ __forceinline__ float dot2(uint32_t w, uint32_t x, float acc) {
    acc = fmaf(bf_lo(w), bf_lo(x), acc);
    return fmaf(bf_hi(w), bf_hi(x), acc);
}

template <int WARPS, int UNROLL, int SK, bool TWO>
__global__ void __launch_bounds__(WARPS * 32) gemv12_kernel(const __nv_bfloat16* __restrict__ X,
        const uint8_t* __restrict__ LO, const uint8_t* __restrict__ NB, int eb,
        const int* __restrict__ esc_off, const int* __restrict__ esc_n, const int* __restrict__ esc_k, const float* __restrict__ esc_d,
        const uint8_t* __restrict__ LO2, const uint8_t* __restrict__ NB2, int eb2,
        const int* __restrict__ esc_off2, const int* __restrict__ esc_n2, const int* __restrict__ esc_k2, const float* __restrict__ esc_d2,
        __nv_bfloat16* __restrict__ out, float* __restrict__ outf, float* __restrict__ part, int* __restrict__ cnt, int N, int K)
{
    // TWO: a second packed plane (the router's BF16 lo) with its own exponent base and escapes is
    // decoded and accumulated into the same fp32 sum (hi . x + lo . x); outf != nullptr: fp32 output.
    constexpr int RPT = 32 / (2 * WARPS);          // steps (row pairs) per warp per k-tile
    constexpr int NP = TWO ? 2 : 1;
    __shared__ float red[32];
    __shared__ int s_last;
    const int pn = blockIdx.x, sk = blockIdx.y;
    const int KT = K / 256, ktc = KT / SK, kt0 = sk * ktc;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, half = lane >> 4, l16 = lane & 15;
    uint32_t lut0[NP], lut1[NP];
    {
        const uint32_t hb = (uint32_t)(eb >> 1);
        lut0[0] = hb | (hb + 1) << 8 | (hb + 2) << 16 | (hb + 3) << 24;
        lut1[0] = (hb + 4) | (hb + 5) << 8 | (hb + 6) << 16 | (hb + 7) << 24;
        if (TWO) {
            const uint32_t hb2 = (uint32_t)(eb2 >> 1);
            lut0[NP - 1] = hb2 | (hb2 + 1) << 8 | (hb2 + 2) << 16 | (hb2 + 3) << 24;
            lut1[NP - 1] = (hb2 + 4) | (hb2 + 5) << 8 | (hb2 + 6) << 16 | (hb2 + 7) << 24;
        }
    }
    // this lane's rows: r = 2 * (warp + WARPS * s) + half, s = 0..RPT-1
    const size_t row0 = ((size_t)(pn * KT + kt0) * 32 + 2 * warp + half);
    const uint8_t* lo_base[NP]; const uint8_t* nb_base[NP];
    lo_base[0] = LO + row0 * 256 + l16 * 16;
    nb_base[0] = NB + row0 * 128 + l16 * 8;
    if (TWO) { lo_base[NP - 1] = LO2 + row0 * 256 + l16 * 16; nb_base[NP - 1] = NB2 + row0 * 128 + l16 * 8; }
    const __nv_bfloat16* x_base = X + kt0 * 256 + l16 * 16;
    float acc[RPT];
    #pragma unroll
    for (int s2 = 0; s2 < RPT; ++s2) acc[s2] = 0.f;

    uint4 cl[NP][UNROLL][RPT]; uint2 cn[NP][UNROLL][RPT];
    #pragma unroll
    for (int q = 0; q < NP; ++q)
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u)
            #pragma unroll
            for (int s2 = 0; s2 < RPT; ++s2) {
                cn[q][u][s2] = *reinterpret_cast<const uint2*>(nb_base[q] + ((size_t)u * 32 + 2 * WARPS * s2) * 128);
                cl[q][u][s2] = ldg_stream(reinterpret_cast<const uint4*>(lo_base[q] + ((size_t)u * 32 + 2 * WARPS * s2) * 256));
            }
    for (int kt = 0; kt < ktc; kt += UNROLL) {
        const int kn = kt + UNROLL < ktc ? kt + UNROLL : kt;
        uint4 nl[NP][UNROLL][RPT]; uint2 nn[NP][UNROLL][RPT];
        #pragma unroll
        for (int q = 0; q < NP; ++q)
            #pragma unroll
            for (int u = 0; u < UNROLL; ++u)
                #pragma unroll
                for (int s2 = 0; s2 < RPT; ++s2) {
                    nn[q][u][s2] = *reinterpret_cast<const uint2*>(nb_base[q] + ((size_t)(kn + u) * 32 + 2 * WARPS * s2) * 128);
                    nl[q][u][s2] = ldg_stream(reinterpret_cast<const uint4*>(lo_base[q] + ((size_t)(kn + u) * 32 + 2 * WARPS * s2) * 256));
                }
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const uint4* xp = reinterpret_cast<const uint4*>(x_base + (kt + u) * 256);
            const uint4 x0 = xp[0], x1 = xp[1];
            #pragma unroll
            for (int s2 = 0; s2 < RPT; ++s2) {
                float a = acc[s2];
                #pragma unroll
                for (int q = 0; q < NP; ++q) {
                    uint32_t w01, w23;
                    decode4(cl[q][u][s2].x, cn[q][u][s2].x & 0xFFFFu, lut0[q], lut1[q], w01, w23); a = dot2(w01, x0.x, a); a = dot2(w23, x0.y, a);
                    decode4(cl[q][u][s2].y, cn[q][u][s2].x >> 16, lut0[q], lut1[q], w01, w23);     a = dot2(w01, x0.z, a); a = dot2(w23, x0.w, a);
                    decode4(cl[q][u][s2].z, cn[q][u][s2].y & 0xFFFFu, lut0[q], lut1[q], w01, w23); a = dot2(w01, x1.x, a); a = dot2(w23, x1.y, a);
                    decode4(cl[q][u][s2].w, cn[q][u][s2].y >> 16, lut0[q], lut1[q], w01, w23);     a = dot2(w01, x1.z, a); a = dot2(w23, x1.w, a);
                }
                acc[s2] = a;
            }
        }
        #pragma unroll
        for (int q = 0; q < NP; ++q)
            #pragma unroll
            for (int u = 0; u < UNROLL; ++u)
                #pragma unroll
                for (int s2 = 0; s2 < RPT; ++s2) { cl[q][u][s2] = nl[q][u][s2]; cn[q][u][s2] = nn[q][u][s2]; }
    }
    // reduce each row over its 16 lanes
    #pragma unroll
    for (int s2 = 0; s2 < RPT; ++s2) {
        float v = acc[s2];
        v += __shfl_xor_sync(0xffffffffu, v, 8); v += __shfl_xor_sync(0xffffffffu, v, 4);
        v += __shfl_xor_sync(0xffffffffu, v, 2); v += __shfl_xor_sync(0xffffffffu, v, 1);
        if (l16 == 0) red[2 * (warp + WARPS * s2) + half] = v;
    }
    __syncthreads();
    if (SK > 1) {
        if (threadIdx.x < 32) part[(size_t)sk * N + pn * 32 + threadIdx.x] = red[threadIdx.x];
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) s_last = (atomicAdd(cnt + pn, 1) == SK - 1);
        __syncthreads();
        if (!s_last) return;
        __threadfence();
    }
    // escapes of this n-tile (both planes): staged in parallel (one entry per thread per pass), then row threads scan
    __shared__ float s_ev[WARPS * 32];
    float v = 0.f;
    if (threadIdx.x < 32) {
        if (SK > 1) {
            #pragma unroll
            for (int s2 = 0; s2 < SK; ++s2) v += __ldcg(part + (size_t)s2 * N + pn * 32 + threadIdx.x);
        } else {
            v = red[threadIdx.x];
        }
    }
    // esc_off / esc_off2 are PER-ROW offsets (entries sorted by row): the tile's entries are
    // [roff[pn * 32], roff[pn * 32 + 32]); row thread r sums its own range within each staged pass
    #pragma unroll
    for (int q = 0; q < NP; ++q) {
        const int* ro = q == 0 ? esc_off : esc_off2;
        const int* ek = q == 0 ? esc_k : esc_k2; const float* ed = q == 0 ? esc_d : esc_d2;
        const int e0 = ro[pn * 32], e1 = ro[pn * 32 + 32];
        int r0 = 0, r1 = 0;
        if (threadIdx.x < 32) { r0 = ro[pn * 32 + threadIdx.x]; r1 = ro[pn * 32 + threadIdx.x + 1]; }
        for (int base = e0; base < e1; base += WARPS * 32) {
            const int e = base + (int)threadIdx.x;
            if (e < e1) s_ev[threadIdx.x] = ed[e] * __bfloat162float(X[ek[e]]);
            __syncthreads();
            if (threadIdx.x < 32) {
                const int lo_ = max(r0, base), hi_ = min(r1, base + WARPS * 32);
                for (int j = lo_; j < hi_; ++j) v += s_ev[j - base];
            }
            __syncthreads();
        }
    }
    if (threadIdx.x < 32) {
        if (outf) outf[pn * 32 + threadIdx.x] = v;
        else out[pn * 32 + threadIdx.x] = __float2bfloat16(v);
        if (SK > 1 && threadIdx.x == 0) cnt[pn] = 0;
    }
}

template <int WARPS, int UNROLL, int SK, bool TWO>
void launch_gemv12(const __nv_bfloat16* X, const uint8_t* LO, const uint8_t* NB, int eb, const int* eo, const int* en, const int* ek, const float* ed,
                   const uint8_t* LO2, const uint8_t* NB2, int eb2, const int* eo2, const int* en2, const int* ek2, const float* ed2,
                   __nv_bfloat16* out, float* outf, float* part, int* cnt, int N, int K, cudaStream_t stream)
{
    gemv12_kernel<WARPS, UNROLL, SK, TWO><<<dim3(N / 32, SK), WARPS * 32, 0, stream>>>(X, LO, NB, eb, eo, en, ek, ed, LO2, NB2, eb2, eo2, en2, ek2, ed2, out, outf, part, cnt, N, K);
}

// ---- 12-bit packed tensor-core GEMM, M <= 16 * MT. One CTA = one 32-row n-tile (x SK k-splits);
// 8 warps = (4 / RPW) row groups of RPW x 8 rows, each row group split KS = 2 * RPW ways over the
// k-tiles. Per 64-k chunk a lane (g = lane / 4, t = lane % 4) holds row g's real k = 16t .. 16t+15
// (16 low bytes + 8 nibble bytes); decode4 of its j-th 4-element group gives the B fragment of
// mma k-step j under the k permutation, and x's A fragments are the same 16-element groups of
// rows g / g+8 (two 16-byte loads each from L1/L2). Warps reduce through shared memory; the
// finishing CTA adds the escapes and writes bf16.
template <int MT, int RPW, int UNROLL, int SK>
__global__ void __launch_bounds__(256) mma12_kernel(const __nv_bfloat16* __restrict__ X, int ldx, int m,
        const uint8_t* __restrict__ LO, const uint8_t* __restrict__ NB, int eb,
        const int* __restrict__ esc_off, const int* __restrict__ esc_n, const int* __restrict__ esc_k, const float* __restrict__ esc_d,
        __nv_bfloat16* __restrict__ out, int ldo, float* __restrict__ part, int* __restrict__ cnt, int N, int K)
{
    constexpr int GROUPS = 4 / RPW, KS = 8 / GROUPS, MR = 16 * MT;
    __shared__ float red[32][MR + 1];
    __shared__ float redw[8][32][MR];
    __shared__ int s_en[64];
    __shared__ float s_ev[64][MR];
    __shared__ int s_last;
    const int pn = blockIdx.x, sk = blockIdx.y;
    const int KT = K / 256, ktc = KT / SK / KS;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int grp = warp / KS, kp = warp % KS;
    const int kt0 = sk * (KT / SK) + kp * ktc;                 // this warp's first k-tile
    const int nchunk = ktc * 4;
    const uint32_t hb = (uint32_t)(eb >> 1);
    const uint32_t lut0 = hb | (hb + 1) << 8 | (hb + 2) << 16 | (hb + 3) << 24;
    const uint32_t lut1 = (hb + 4) | (hb + 5) << 8 | (hb + 6) << 16 | (hb + 7) << 24;

    // weight pointers: row grp * RPW * 8 + 8 r + g of tile (pn, kt0 + c / 4), chunk q = c % 4
    const uint8_t* lo_base[RPW]; const uint8_t* nb_base[RPW];
    #pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const int row = grp * RPW * 8 + 8 * r + g;
        lo_base[r] = LO + ((size_t)(pn * KT + kt0) * 32 + row) * 256 + 16 * t;
        nb_base[r] = NB + ((size_t)(pn * KT + kt0) * 32 + row) * 128 + 8 * t;
    }
    auto lo_at = [&](int r, int c) { return reinterpret_cast<const uint4*>(lo_base[r] + (size_t)(c >> 2) * 32 * 256 + (c & 3) * 64); };
    auto nb_at = [&](int r, int c) { return reinterpret_cast<const uint2*>(nb_base[r] + (size_t)(c >> 2) * 32 * 128 + (c & 3) * 32); };
    // x rows g / g+8 of each m-tile
    const uint4* xp[MT][2]; bool xok[MT][2];
    #pragma unroll
    for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = 16 * i + g + 8 * h;
            xok[i][h] = row < m;
            xp[i][h] = reinterpret_cast<const uint4*>(X + (size_t)(row < m ? row : 0) * ldx + kt0 * 256 + 16 * t);
        }
    float acc[MT][RPW][4];
    #pragma unroll
    for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int r = 0; r < RPW; ++r)
            #pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][r][q] = 0.f;

    // x fragments of a chunk: [m-tile][row g / g+8][elements 0-7 / 8-15], prefetched one chunk ahead
    auto load_x = [&](int cc, uint4 (&xa)[MT][2][2]) {
        #pragma unroll
        for (int i = 0; i < MT; ++i)
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint4* xq = xp[i][h] + (size_t)(cc >> 2) * 32 + (cc & 3) * 8;   // k-tile = 32 uint4, 64-k chunk = 8 uint4
                if (xok[i][h]) { xa[i][h][0] = ldg_l1_v(xq); xa[i][h][1] = ldg_l1_v(xq + 1); }
                else { xa[i][h][0] = make_uint4(0u, 0u, 0u, 0u); xa[i][h][1] = make_uint4(0u, 0u, 0u, 0u); }
            }
    };
    uint4 cl[UNROLL][RPW]; uint2 cn[UNROLL][RPW];
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u)
        #pragma unroll
        for (int r = 0; r < RPW; ++r) { cn[u][r] = ldg_v2_v(nb_at(r, u)); cl[u][r] = ldg_stream_v(lo_at(r, u)); }
    uint4 xcur[MT][2][2];
    load_x(0, xcur);
    for (int c = 0; c < nchunk; c += UNROLL) {
        const int cnx = c + UNROLL < nchunk ? c + UNROLL : c;
        uint4 nl[UNROLL][RPW]; uint2 nn[UNROLL][RPW];
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u)
            #pragma unroll
            for (int r = 0; r < RPW; ++r) { nn[u][r] = ldg_v2_v(nb_at(r, cnx + u)); nl[u][r] = ldg_stream_v(lo_at(r, cnx + u)); }
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int cc = c + u;
            uint4 xnxt[MT][2][2];
            load_x(cc + 1 < nchunk ? cc + 1 : cc, xnxt);
            uint4 (&xa)[MT][2][2] = xcur;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                #pragma unroll
                for (int r = 0; r < RPW; ++r) {
                    const uint32_t Lw = j == 0 ? cl[u][r].x : j == 1 ? cl[u][r].y : j == 2 ? cl[u][r].z : cl[u][r].w;
                    const uint32_t n16 = j == 0 ? (cn[u][r].x & 0xFFFFu) : j == 1 ? (cn[u][r].x >> 16) : j == 2 ? (cn[u][r].y & 0xFFFFu) : (cn[u][r].y >> 16);
                    uint32_t b0, b1;
                    decode4(Lw, n16, lut0, lut1, b0, b1);
                    #pragma unroll
                    for (int i = 0; i < MT; ++i) {
                        const uint4 ga = xa[i][0][j >> 1], gb = xa[i][1][j >> 1];
                        const uint32_t a0 = (j & 1) ? ga.z : ga.x, a2 = (j & 1) ? ga.w : ga.y;
                        const uint32_t a1 = (j & 1) ? gb.z : gb.x, a3 = (j & 1) ? gb.w : gb.y;
                        mma16816(acc[i][r], a0, a1, a2, a3, b0, b1);
                    }
                }
            }
            #pragma unroll
            for (int i = 0; i < MT; ++i)
                #pragma unroll
                for (int h = 0; h < 2; ++h) { xcur[i][h][0] = xnxt[i][h][0]; xcur[i][h][1] = xnxt[i][h][1]; }
        }
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u)
            #pragma unroll
            for (int r = 0; r < RPW; ++r) { cl[u][r] = nl[u][r]; cn[u][r] = nn[u][r]; }
    }

    // reduce the KS warps of each row group: red[n_local][m]
    // cross-warp reduction without atomics: each warp writes its fragments to its own slab
    // (distinct addresses per lane), then every thread sums the KS warps of its row group
    #pragma unroll
    for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const int n0 = grp * RPW * 8 + 8 * r + 2 * t;
            redw[warp][n0][16 * i + g] = acc[i][r][0];
            redw[warp][n0 + 1][16 * i + g] = acc[i][r][1];
            redw[warp][n0][16 * i + g + 8] = acc[i][r][2];
            redw[warp][n0 + 1][16 * i + g + 8] = acc[i][r][3];
        }
    __syncthreads();
    for (int idx = threadIdx.x; idx < 32 * MR; idx += 256) {
        const int n = idx & 31, i = idx >> 5, gq = n / (RPW * 8);
        float v = 0.f;
        #pragma unroll
        for (int kq = 0; kq < KS; ++kq) v += redw[gq * KS + kq][n][i];
        red[n][i] = v;
    }
    __syncthreads();
    if (SK > 1) {
        // partials [SK][MR][N] fp32 of this n-tile, ticket, last CTA sums
        for (int idx = threadIdx.x; idx < 32 * MR; idx += 256) {
            const int n = idx & 31, i = idx >> 5;
            part[((size_t)sk * MR + i) * N + pn * 32 + n] = red[n][i];
        }
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) s_last = (atomicAdd(cnt + pn, 1) == SK - 1);
        __syncthreads();
        if (!s_last) return;
        __threadfence();
        for (int idx = threadIdx.x; idx < 32 * MR; idx += 256) {
            const int n = idx & 31, i = idx >> 5;
            float v = 0.f;
            #pragma unroll
            for (int s2 = 0; s2 < SK; ++s2) v += __ldcg(part + ((size_t)s2 * MR + i) * N + pn * 32 + n);
            red[n][i] = v;
        }
        __syncthreads();
    }
    // escapes: staged 64 at a time (one entry per thread incl. its x column), then row threads scan
    const int e0 = esc_off[pn], e1 = esc_off[pn + 1];
    for (int base = e0; base < e1; base += 64) {
        const int e = base + (int)threadIdx.x;
        if (threadIdx.x < 64) {
            if (e < e1) {
                s_en[threadIdx.x] = esc_n[e];
                const float d = esc_d[e]; const int kk = esc_k[e];
                for (int i = 0; i < MR; ++i) s_ev[threadIdx.x][i] = i < m ? d * __bfloat162float(X[(size_t)i * ldx + kk]) : 0.f;
            } else {
                s_en[threadIdx.x] = -1;
            }
        }
        __syncthreads();
        if (threadIdx.x < 32) {
            const int cnt_ = min(e1 - base, 64);
            for (int j = 0; j < cnt_; ++j)
                if (s_en[j] == (int)threadIdx.x)
                    for (int i = 0; i < MR; ++i) red[threadIdx.x][i] += s_ev[j][i];
        }
        __syncthreads();
    }
    for (int idx = threadIdx.x; idx < 32 * MR; idx += 256) {
        const int n = idx & 31, i = idx >> 5;
        if (i < m) out[(size_t)i * ldo + pn * 32 + n] = __float2bfloat16(red[n][i]);
    }
    if (SK > 1 && threadIdx.x == 0) cnt[pn] = 0;
}

template <int MT, int RPW, int UNROLL, int SK>
void launch_mma12(const __nv_bfloat16* X, int ldx, int m, const uint8_t* LO, const uint8_t* NB, int eb, const int* eo, const int* en, const int* ek, const float* ed,
                  __nv_bfloat16* out, int ldo, float* part, int* cnt, int N, int K, cudaStream_t stream)
{
    mma12_kernel<MT, RPW, UNROLL, SK><<<dim3(N / 32, SK), 256, 0, stream>>>(X, ldx, m, LO, NB, eb, eo, en, ek, ed, out, ldo, part, cnt, N, K);
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async8(uint32_t dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

// ---- 12-bit mma GEMM with cp.async staging: same math / mapping as mma12_kernel, but every lane's
// chunk data (RPW x 16 B low bytes, RPW x 8 B nibbles, MT x 2 x 32 B of x) is copied asynchronously
// into its own shared-memory slot NSTAGE chunks ahead (commit / wait groups), so the loads are in
// flight regardless of how ptxas schedules the compute.
template <int MT, int RPW, int NSTAGE, int SK>
__global__ void __launch_bounds__(256) mma12cp_kernel(const __nv_bfloat16* __restrict__ X, int ldx, int m,
        const uint8_t* __restrict__ LO, const uint8_t* __restrict__ NB, int eb,
        const int* __restrict__ esc_off, const int* __restrict__ esc_n, const int* __restrict__ esc_k, const float* __restrict__ esc_d,
        __nv_bfloat16* __restrict__ out, int ldo, float* __restrict__ part, int* __restrict__ cnt, int N, int K)
{
    constexpr int GROUPS = 4 / RPW, KS = 8 / GROUPS, MR = 16 * MT;
    constexpr int LOB = RPW * 16, NBB = RPW * 8, XB = MT * 2 * 32, LANEB = ((LOB + NBB + XB + 15) / 16) * 16, WSTAGE = 32 * LANEB;
    extern __shared__ __align__(16) uint8_t dsmem[];
    __shared__ float red[32][MR + 1];
    __shared__ float redw[8][32][MR];
    __shared__ int s_en[64];
    __shared__ float s_ev[64][MR];
    __shared__ int s_last;
    const int pn = blockIdx.x, sk = blockIdx.y;
    const int KT = K / 256, ktc = KT / SK / KS;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int grp = warp / KS, kp = warp % KS;
    const int kt0 = sk * (KT / SK) + kp * ktc;
    const int nchunk = ktc * 4;
    const uint32_t hb = (uint32_t)(eb >> 1);
    const uint32_t lut0 = hb | (hb + 1) << 8 | (hb + 2) << 16 | (hb + 3) << 24;
    const uint32_t lut1 = (hb + 4) | (hb + 5) << 8 | (hb + 6) << 16 | (hb + 7) << 24;
    uint8_t* my = dsmem + warp * (NSTAGE * WSTAGE) + lane * LANEB;

    const uint8_t* lo_base[RPW]; const uint8_t* nb_base[RPW];
    #pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const int row = grp * RPW * 8 + 8 * r + g;
        lo_base[r] = LO + ((size_t)(pn * KT + kt0) * 32 + row) * 256 + 16 * t;
        nb_base[r] = NB + ((size_t)(pn * KT + kt0) * 32 + row) * 128 + 8 * t;
    }
    const __nv_bfloat16* xp[MT][2]; bool xok[MT][2];
    #pragma unroll
    for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = 16 * i + g + 8 * h;
            xok[i][h] = row < m;
            xp[i][h] = X + (size_t)(row < m ? row : 0) * ldx + kt0 * 256 + 16 * t;
        }
    auto issue = [&](int c, int st) {
        uint8_t* slot = my + st * WSTAGE;
        const size_t lo_off = (size_t)(c >> 2) * 32 * 256 + (c & 3) * 64, nb_off = (size_t)(c >> 2) * 32 * 128 + (c & 3) * 32;
        #pragma unroll
        for (int r = 0; r < RPW; ++r) {
            cp_async16(smem_u32(slot + 16 * r), lo_base[r] + lo_off, 16);
            cp_async8(smem_u32(slot + LOB + 8 * r), nb_base[r] + nb_off);
        }
        const size_t x_off = (size_t)(c >> 2) * 256 + (c & 3) * 64;   // elements
        #pragma unroll
        for (int i = 0; i < MT; ++i)
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const __nv_bfloat16* src = xp[i][h] + x_off;
                const int ok = xok[i][h] ? 16 : 0;
                cp_async16(smem_u32(slot + LOB + NBB + (2 * i + h) * 32), src, ok);
                cp_async16(smem_u32(slot + LOB + NBB + (2 * i + h) * 32 + 16), src + 8, ok);
            }
    };

    float acc[MT][RPW][4];
    #pragma unroll
    for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int r = 0; r < RPW; ++r)
            #pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][r][q] = 0.f;

    #pragma unroll
    for (int st = 0; st < NSTAGE - 1; ++st) {
        if (st < nchunk) issue(st, st);
        cp_async_commit();
    }
    int st = 0;
    for (int c = 0; c < nchunk; ++c, st = (st + 1 == NSTAGE ? 0 : st + 1)) {
        const int cn = c + NSTAGE - 1;
        if (cn < nchunk) issue(cn, (st + NSTAGE - 1) % NSTAGE);
        cp_async_commit();
        cp_async_wait<NSTAGE - 1>();
        __syncwarp();
        const uint8_t* slot = my + st * WSTAGE;
        uint4 cl[RPW]; uint2 cnb[RPW];
        #pragma unroll
        for (int r = 0; r < RPW; ++r) {
            cl[r] = *reinterpret_cast<const uint4*>(slot + 16 * r);
            cnb[r] = *reinterpret_cast<const uint2*>(slot + LOB + 8 * r);
        }
        uint4 xa[MT][2][2];
        #pragma unroll
        for (int i = 0; i < MT; ++i)
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                xa[i][h][0] = *reinterpret_cast<const uint4*>(slot + LOB + NBB + (2 * i + h) * 32);
                xa[i][h][1] = *reinterpret_cast<const uint4*>(slot + LOB + NBB + (2 * i + h) * 32 + 16);
            }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            #pragma unroll
            for (int r = 0; r < RPW; ++r) {
                const uint32_t Lw = j == 0 ? cl[r].x : j == 1 ? cl[r].y : j == 2 ? cl[r].z : cl[r].w;
                const uint32_t n16 = j == 0 ? (cnb[r].x & 0xFFFFu) : j == 1 ? (cnb[r].x >> 16) : j == 2 ? (cnb[r].y & 0xFFFFu) : (cnb[r].y >> 16);
                uint32_t b0, b1;
                decode4(Lw, n16, lut0, lut1, b0, b1);
                #pragma unroll
                for (int i = 0; i < MT; ++i) {
                    const uint4 ga = xa[i][0][j >> 1], gb = xa[i][1][j >> 1];
                    const uint32_t a0 = (j & 1) ? ga.z : ga.x, a2 = (j & 1) ? ga.w : ga.y;
                    const uint32_t a1 = (j & 1) ? gb.z : gb.x, a3 = (j & 1) ? gb.w : gb.y;
                    mma16816(acc[i][r], a0, a1, a2, a3, b0, b1);
                }
            }
        }
        __syncwarp();
    }
    cp_async_wait<0>();

    // cross-warp reduction without atomics: each warp writes its fragments to its own slab
    // (distinct addresses per lane), then every thread sums the KS warps of its row group
    #pragma unroll
    for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const int n0 = grp * RPW * 8 + 8 * r + 2 * t;
            redw[warp][n0][16 * i + g] = acc[i][r][0];
            redw[warp][n0 + 1][16 * i + g] = acc[i][r][1];
            redw[warp][n0][16 * i + g + 8] = acc[i][r][2];
            redw[warp][n0 + 1][16 * i + g + 8] = acc[i][r][3];
        }
    __syncthreads();
    for (int idx = threadIdx.x; idx < 32 * MR; idx += 256) {
        const int n = idx & 31, i = idx >> 5, gq = n / (RPW * 8);
        float v = 0.f;
        #pragma unroll
        for (int kq = 0; kq < KS; ++kq) v += redw[gq * KS + kq][n][i];
        red[n][i] = v;
    }
    __syncthreads();
    if (SK > 1) {
        for (int idx = threadIdx.x; idx < 32 * MR; idx += 256) {
            const int n = idx & 31, i = idx >> 5;
            part[((size_t)sk * MR + i) * N + pn * 32 + n] = red[n][i];
        }
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) s_last = (atomicAdd(cnt + pn, 1) == SK - 1);
        __syncthreads();
        if (!s_last) return;
        __threadfence();
        for (int idx = threadIdx.x; idx < 32 * MR; idx += 256) {
            const int n = idx & 31, i = idx >> 5;
            float v = 0.f;
            #pragma unroll
            for (int s2 = 0; s2 < SK; ++s2) v += __ldcg(part + ((size_t)s2 * MR + i) * N + pn * 32 + n);
            red[n][i] = v;
        }
        __syncthreads();
    }
    const int e0 = esc_off[pn], e1 = esc_off[pn + 1];
    for (int base = e0; base < e1; base += 64) {
        const int e = base + (int)threadIdx.x;
        if (threadIdx.x < 64) {
            if (e < e1) {
                s_en[threadIdx.x] = esc_n[e];
                const float d = esc_d[e]; const int kk = esc_k[e];
                for (int i = 0; i < MR; ++i) s_ev[threadIdx.x][i] = i < m ? d * __bfloat162float(X[(size_t)i * ldx + kk]) : 0.f;
            } else {
                s_en[threadIdx.x] = -1;
            }
        }
        __syncthreads();
        if (threadIdx.x < 32) {
            const int cnt_ = min(e1 - base, 64);
            for (int j = 0; j < cnt_; ++j)
                if (s_en[j] == (int)threadIdx.x)
                    for (int i = 0; i < MR; ++i) red[threadIdx.x][i] += s_ev[j][i];
        }
        __syncthreads();
    }
    for (int idx = threadIdx.x; idx < 32 * MR; idx += 256) {
        const int n = idx & 31, i = idx >> 5;
        if (i < m) out[(size_t)i * ldo + pn * 32 + n] = __float2bfloat16(red[n][i]);
    }
    if (SK > 1 && threadIdx.x == 0) cnt[pn] = 0;
}

template <int MT, int RPW, int NSTAGE, int SK>
void launch_mma12cp(const __nv_bfloat16* X, int ldx, int m, const uint8_t* LO, const uint8_t* NB, int eb, const int* eo, const int* en, const int* ek, const float* ed,
                    __nv_bfloat16* out, int ldo, float* part, int* cnt, int N, int K, cudaStream_t stream)
{
    constexpr int LANEB = ((RPW * 16 + RPW * 8 + MT * 2 * 32 + 15) / 16) * 16;
    constexpr int smem = NSTAGE * 8 * 32 * LANEB;
    static bool attr = false;
    if (!attr) { cudaFuncSetAttribute(mma12cp_kernel<MT, RPW, NSTAGE, SK>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
    mma12cp_kernel<MT, RPW, NSTAGE, SK><<<dim3(N / 32, SK), 256, smem, stream>>>(X, ldx, m, LO, NB, eb, eo, en, ek, ed, out, ldo, part, cnt, N, K);
}

// ---- w12: dense 12-bit W x BF16 x^T GEMM, weights as the mma M side (one m16 tile of W per warp, 4 per CTA),
// tokens as N (NT n8 blocks, m <= 8*NT), K streamed in 128-wide chunks through an NSTAGE cp.async ring shared by
// the CTA. Weight layout W12 (see _repack_w12): per (m16 tile, chunk) 3 KB = low bytes [4 quads][32 lanes][16 B]
// (quad q: k16 steps 2q, 2q+1; per step 8 bytes = L0 (rows g, g+8 at k 2t, 2t+1) then L1 (k 2t+8, 2t+9)) and
// nibbles [2 halves][32 lanes][16 B] (half h: steps 4h..4h+3; per step one word, nibble i = element i of the
// L0 then L1 order), grouped by 4 tiles: [N/64][K/128][4][3 KB]. Decode per step: decode4 (PRMT LUT of the
// high byte, sign OR'd, L/H bytes interleaved) twice -> a0..a3 bf16x2, mma bf16 -> fp32. The CTA writes its
// fp32 partial slice part[z][i][n]; w12_reduce sums the slices, adds the escapes and writes bf16.
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }
__device__ __forceinline__ void ldsm_x4(uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(addr));
}
constexpr int W12_KC = 128, W12_XSTRIDE = W12_KC + 8;   // token tile row stride (halves)
constexpr int W12_TILE_BYTES = 3072;                   // per (m16 tile, chunk)
__host__ __device__ constexpr int w12_stage_bytes(int nt) { return 4 * W12_TILE_BYTES + 8 * nt * W12_XSTRIDE * 2; }


// w12 epilogue, out of line (__noinline__) so the GEMM loop's register allocation and code do not depend on it.
// SPLITK: the CTA stores its fp32 partial slice (plain stores), takes a ticket on cnt[tile group]; the last arrival
// sums the slices from L2 (ld.global.cg), adds the group's escapes (two 32-row pack tiles) in parallel over
// (escape, token) with shared-memory atomics, writes bf16 and resets the ticket. KSPLIT == 1: the CTA's own sums.
__device__ __forceinline__ void fence_acq_rel_gpu() { asm volatile("fence.acq_rel.gpu;\n" ::: "memory"); }

template <int NT, bool SPLITK>
__device__ __noinline__ void w12_epilogue(const float* __restrict__ accv, int m, int N, int tid, int warp, int g, int t,
        float* __restrict__ part, int* __restrict__ cnt,
        const int* __restrict__ esc_off, const int* __restrict__ esc_n, const int* __restrict__ esc_k, const float* __restrict__ esc_d,
        const __nv_bfloat16* __restrict__ X, int ldx, __nv_bfloat16* __restrict__ out, int ldo, int emode)
{
    // Latency-aware: every loop below issues all of its global loads before using any. emode (bench diagnostics):
    // 4 = return at entry, 5 = partial stores + ticket only, 6 = skip the escapes; 0 = full.
    __shared__ __align__(16) float red[64][33];
    __shared__ int s_last;
    if (emode == 4) return;
    const int col0 = (blockIdx.x * 4 + warp) * 16 + g;                  // output column (row of W) of fragment row g
    if constexpr (SPLITK) {
        const int KSPLIT = (int)gridDim.y;
        float* pz = part + (size_t)blockIdx.y * m * N;
        #pragma unroll
        for (int nbk = 0; nbk < NT; ++nbk)
            #pragma unroll
            for (int hr = 0; hr < 2; ++hr)
                #pragma unroll
                for (int cc = 0; cc < 2; ++cc) {
                    const int tok = nbk * 8 + 2 * t + cc;
                    if (tok < m) pz[(size_t)tok * N + col0 + 8 * hr] = accv[nbk * 4 + 2 * hr + cc];
                }
        __syncthreads();
        if (tid == 0) {
            fence_acq_rel_gpu();                                        // release this CTA's slice (cumulative over the barrier)
            const int old = atomicAdd(cnt + blockIdx.x, 1);
            s_last = (old == KSPLIT - 1);
            if (old == KSPLIT - 1) cnt[blockIdx.x] = 0;
            fence_acq_rel_gpu();                                        // acquire the other slices
        }
        __syncthreads();
        if (!s_last || emode == 5) return;
        // slice sum, 16 tokens per pass: thread -> token il = tid >> 3, columns 8*oct .. 8*oct+7 (two float4)
        const int il = tid >> 3, oct = tid & 7;
        for (int p0 = 0; p0 < m; p0 += 16) {
            const int i = p0 + il;
            float4 acc0 = make_float4(0.f, 0.f, 0.f, 0.f), acc1 = acc0;
            if (i < m) {
                for (int zb = 0; zb < KSPLIT; zb += 8) {
                    float4 v0[8], v1[8];
                    #pragma unroll
                    for (int z = 0; z < 8; ++z) {
                        if (zb + z < KSPLIT) {
                            const float* src = part + ((size_t)(zb + z) * m + i) * N + blockIdx.x * 64 + oct * 8;
                            v0[z] = __ldcg(reinterpret_cast<const float4*>(src));
                            v1[z] = __ldcg(reinterpret_cast<const float4*>(src + 4));
                        } else {
                            v0[z] = make_float4(0.f, 0.f, 0.f, 0.f); v1[z] = v0[z];
                        }
                    }
                    #pragma unroll
                    for (int z = 0; z < 8; ++z) {
                        acc0.x += v0[z].x; acc0.y += v0[z].y; acc0.z += v0[z].z; acc0.w += v0[z].w;
                        acc1.x += v1[z].x; acc1.y += v1[z].y; acc1.z += v1[z].z; acc1.w += v1[z].w;
                    }
                }
                red[oct * 8 + 0][i] = acc0.x; red[oct * 8 + 1][i] = acc0.y; red[oct * 8 + 2][i] = acc0.z; red[oct * 8 + 3][i] = acc0.w;
                red[oct * 8 + 4][i] = acc1.x; red[oct * 8 + 5][i] = acc1.y; red[oct * 8 + 6][i] = acc1.z; red[oct * 8 + 7][i] = acc1.w;
            }
        }
    } else {
        #pragma unroll
        for (int nbk = 0; nbk < NT; ++nbk)
            #pragma unroll
            for (int hr = 0; hr < 2; ++hr)
                #pragma unroll
                for (int cc = 0; cc < 2; ++cc) {
                    const int tok = nbk * 8 + 2 * t + cc;
                    if (tok < m) red[warp * 16 + g + 8 * hr][tok] = accv[nbk * 4 + 2 * hr + cc];
                }
    }
    __syncthreads();
    // escapes of pack tiles 2*bx, 2*bx + 1: one escape per thread per batch, its m token values loaded together
    const int e0 = esc_off[2 * blockIdx.x], e1 = emode == 6 ? esc_off[2 * blockIdx.x] : esc_off[2 * blockIdx.x + 2];   // contiguous lists
    for (int base = e0; base < e1; base += 128) {
        const int e = base + tid;
        if (e < e1) {
            const int nl = esc_n[e] + (e >= esc_off[2 * blockIdx.x + 1] ? 32 : 0);
            const int kk = esc_k[e];
            const float d = esc_d[e];
            float xv[8 * NT];
            #pragma unroll
            for (int i = 0; i < 8 * NT; ++i) xv[i] = i < m ? __bfloat162float(X[(size_t)i * ldx + kk]) : 0.f;
            #pragma unroll
            for (int i = 0; i < 8 * NT; ++i)
                if (i < m) atomicAdd(&red[nl][i], d * xv[i]);
        }
    }
    __syncthreads();
    // bf16x2 stores: thread -> columns 2*(tid & 31), tokens (tid >> 5) + 4r
    for (int i = tid >> 5; i < m; i += 4) {
        const int n = 2 * (tid & 31);
        *reinterpret_cast<__nv_bfloat162*>(out + (size_t)i * ldo + blockIdx.x * 64 + n) = __floats2bfloat162_rn(red[n][i], red[n + 1][i]);
    }
}

typedef void (*w12_epi_fn)(const float*, int, int, int, int, int, int, float*, int*, const int*, const int*, const int*, const float*,
                           const __nv_bfloat16*, int, __nv_bfloat16*, int, int);

template <int NT, int NSTAGE, bool SPLITK, int MODE = 0>
__global__ void __launch_bounds__(128)
w12_kernel(const __nv_bfloat16* __restrict__ X, int ldx, int m, const uint8_t* __restrict__ W, int eb,
           float* __restrict__ part, int N, int K,
           const int* __restrict__ esc_off, const int* __restrict__ esc_n, const int* __restrict__ esc_k, const float* __restrict__ esc_d,
           __nv_bfloat16* __restrict__ out, int ldo, int* __restrict__ cnt, int emode)
{
    constexpr int STAGE_BYTES = w12_stage_bytes(NT);
    constexpr int W_STAGE = 4 * W12_TILE_BYTES;
    constexpr int CH = NT == 1 ? 2 : 1;                                   // accumulator chains (mma latency)
    extern __shared__ __align__(128) uint8_t w12_smem[];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KSPLIT = SPLITK ? (int)gridDim.y : 1;
    const int NC_ALL = K / W12_KC;
    const int NC = NC_ALL / KSPLIT;
    const int c0 = SPLITK ? (int)blockIdx.y * NC : 0;
    const int ntb = (m + 7) >> 3;
    const uint32_t hb = (uint32_t)(eb >> 1);
    const uint32_t lut0 = hb | (hb + 1) << 8 | (hb + 2) << 16 | (hb + 3) << 24;
    const uint32_t lut1 = (hb + 4) | (hb + 5) << 8 | (hb + 6) << 16 | (hb + 7) << 24;

    // cp.async segments per chunk: 768 weight segments (6 per thread) + NT x 128 token segments (NT per thread)
    const uint8_t* wsrc = W + ((size_t)blockIdx.x * NC_ALL + c0) * W_STAGE + tid * 16;      // + i * 2048 + c * W_STAGE
    const __nv_bfloat16* xsrc[NT]; int xbytes[NT];
    #pragma unroll
    for (int i = 0; i < NT; ++i) {
        const int sidx = tid + i * 128, r = sidx >> 4;
        xbytes[i] = r < m ? 16 : 0;
        xsrc[i] = X + (size_t)(r < m ? r : 0) * ldx + c0 * W12_KC + (sidx & 15) * 8;
    }
    const uint32_t sbase = smem_u32(w12_smem);
    auto issue = [&](int c, int stage) {
        const uint32_t ws = sbase + stage * STAGE_BYTES;
        #pragma unroll
        for (int i = 0; i < 6; ++i) cp_async16(ws + i * 2048 + tid * 16, wsrc + (size_t)c * W_STAGE + i * 2048, 16);
        const uint32_t xs = ws + W_STAGE;
        #pragma unroll
        for (int i = 0; i < NT; ++i) {
            const int sidx = tid + i * 128;
            cp_async16(xs + ((sidx >> 4) * W12_XSTRIDE + (sidx & 15) * 8) * 2, xsrc[i] + c * W12_KC, xbytes[i]);
        }
    };

    float acc[NT * CH][4];
    #pragma unroll
    for (int n = 0; n < NT * CH; ++n)
        #pragma unroll
        for (int q = 0; q < 4; ++q) acc[n][q] = 0.f;

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
        const uint8_t* ws = w12_smem + (c % NSTAGE) * STAGE_BYTES + warp * W12_TILE_BYTES;
        const __half* xs = reinterpret_cast<const __half*>(w12_smem + (c % NSTAGE) * STAGE_BYTES + W_STAGE);
        if constexpr (MODE == 1) {                                        // loads only
            if (lane == 0) acc[0][0] += __uint_as_float(*reinterpret_cast<const uint32_t*>(ws + warp * 16) & 0x007FFFFFu);
            continue;
        }
        #pragma unroll
        for (int h = 0; h < 2; ++h) {                                     // steps 4h..4h+3
            uint4 lo[2]; uint4 nb;
            lo[0] = *reinterpret_cast<const uint4*>(ws + (2 * h) * 512 + lane * 16);          // steps 4h, 4h+1
            lo[1] = *reinterpret_cast<const uint4*>(ws + (2 * h + 1) * 512 + lane * 16);      // steps 4h+2, 4h+3
            nb = *reinterpret_cast<const uint4*>(ws + 2048 + h * 512 + lane * 16);          // steps 4h..4h+3
            #pragma unroll
            for (int ps = 0; ps < 2; ++ps) {                                 // steps 4h+2ps, +1
                uint32_t b[NT][4];
                #pragma unroll
                for (int nbk = 0; nbk < NT; ++nbk) {
                    if (nbk < ntb)
                        ldsm_x4(b[nbk][0], b[nbk][1], b[nbk][2], b[nbk][3],
                                smem_u32(xs + (nbk * 8 + (lane & 7)) * W12_XSTRIDE + h * 64 + ps * 32 + (lane >> 3) * 8));
                }
                #pragma unroll
                for (int q = 0; q < 2; ++q) {
                    const uint32_t L0 = q == 0 ? lo[ps].x : lo[ps].z, L1 = q == 0 ? lo[ps].y : lo[ps].w;
                    const int si = 2 * ps + q;
                    const uint32_t Nw = si == 0 ? nb.x : si == 1 ? nb.y : si == 2 ? nb.z : nb.w;
                    uint32_t a0, a1, a2, a3;
                    decode4(L0, Nw & 0xFFFFu, lut0, lut1, a0, a1);
                    decode4(L1, Nw >> 16, lut0, lut1, a2, a3);
                    #pragma unroll
                    for (int nbk = 0; nbk < NT; ++nbk) {
                        if (nbk < ntb) mma16816(acc[nbk * CH + (CH == 2 ? q : 0)], a0, a1, a2, a3, b[nbk][2 * q], b[nbk][2 * q + 1]);
                    }
                }
            }
        }
    }
    cp_async_wait<0>();
    float accv[NT * 4];
    #pragma unroll
    for (int nbk = 0; nbk < NT; ++nbk)
        #pragma unroll
        for (int q = 0; q < 4; ++q) accv[nbk * 4 + q] = acc[nbk * CH][q] + (CH == 2 ? acc[nbk * CH + 1][q] : 0.f);
    if constexpr (MODE == 2) {                                            // bench: GEMM only (keep the sums live)
        if (lane == 0 && accv[0] == 12345.678f) part[tid] = accv[NT * 4 - 1];
        return;
    }
    // the epilogue through an opaque pointer: a real call, so the loop's register allocation does not see it
    w12_epi_fn fp = w12_epilogue<NT, SPLITK>;
    asm volatile("" : "+l"(fp));
    fp(accv, m, N, tid, warp, g, t, part, cnt, esc_off, esc_n, esc_k, esc_d, X, ldx, out, ldo, emode);
}

struct W12Args { const int* eo; const int* en; const int* ek; const float* ed; __nv_bfloat16* out; int ldo; int* cnt; int emode; };
template <int NT, int NSTAGE, bool SPLITK, int MODE = 0>
void launch_w12(const __nv_bfloat16* X, int ldx, int m, const uint8_t* W, int eb, float* part, int N, int K, int sk, const W12Args& a, cudaStream_t stream)
{
    constexpr int smem = NSTAGE * w12_stage_bytes(NT);
    static bool attr = false;
    if (!attr) { cudaFuncSetAttribute(w12_kernel<NT, NSTAGE, SPLITK, MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
    w12_kernel<NT, NSTAGE, SPLITK, MODE><<<dim3(N / 64, sk), 128, smem, stream>>>(X, ldx, m, W, eb, part, N, K, a.eo, a.en, a.ek, a.ed, a.out, a.ldo, a.cnt, a.emode);
}

// ---- w12b: single launch. CTA = one 16-row tile of W over the full K; W warps, warp w streams chunks
// [w*NCW, (w+1)*NCW) of the tile through a private NST-stage cp.async ring (3 KB per chunk: the W12 layout),
// loads its B fragments (x[tok][k]) straight from L2 with a one-chunk register prefetch, and accumulates fp32.
// Epilogue: the W partial accumulators are summed through smem, the tile's escapes (from the 32-row pack tile's
// list) are added, bf16 is written. No K-split across CTAs, no second launch.
template <int NT, int W, int NST>
__global__ void __launch_bounds__(W * 32, (W <= 8 ? 2 : 1))
w12b_kernel(const __nv_bfloat16* __restrict__ X, int ldx, int m, const uint8_t* __restrict__ Wt, int eb,
            const int* __restrict__ esc_off, const int* __restrict__ esc_n, const int* __restrict__ esc_k, const float* __restrict__ esc_d,
            __nv_bfloat16* __restrict__ out, int ldo, int N, int K)
{
    constexpr int CH = NT == 1 ? 2 : 1;
    extern __shared__ __align__(128) uint8_t w12b_smem[];              // W x NST x 3 KB
    __shared__ float red[16][33];
    const int tile = blockIdx.x, grp = tile >> 2, slot = tile & 3;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int NC_ALL = K / W12_KC, NCW = NC_ALL / W, c0 = warp * NCW;
    const uint32_t hb = (uint32_t)(eb >> 1);
    const uint32_t lut0 = hb | (hb + 1) << 8 | (hb + 2) << 16 | (hb + 3) << 24;
    const uint32_t lut1 = (hb + 4) | (hb + 5) << 8 | (hb + 6) << 16 | (hb + 7) << 24;
    // weights: [grp][chunk][slot][3 KB]; this lane copies 6 x 16 B per chunk (offsets i*512 + lane*16)
    const uint8_t* wsrc = Wt + ((size_t)grp * NC_ALL + c0) * (4 * W12_TILE_BYTES) + slot * W12_TILE_BYTES + lane * 16;
    const uint32_t ring = smem_u32(w12b_smem) + warp * (NST * W12_TILE_BYTES);
    auto issue = [&](int c, int st) {
        #pragma unroll
        for (int i = 0; i < 6; ++i)
            cp_async16(ring + st * W12_TILE_BYTES + i * 512 + lane * 16, wsrc + (size_t)c * (4 * W12_TILE_BYTES) + i * 512, 16);
    };
    // x B fragments of chunk c: b0 = x[tok][k0 + 2t, +1], b1 = x[tok][k0 + 2t + 8, +9], tok = nb*8 + g
    auto load_x = [&](int c, uint32_t (&xb)[NT][8][2]) {
        #pragma unroll
        for (int nb = 0; nb < NT; ++nb) {
            const int tok = nb * 8 + g;
            const __nv_bfloat16* xr = X + (size_t)(tok < m ? tok : 0) * ldx + (c0 + c) * W12_KC + 2 * t;
            #pragma unroll
            for (int st = 0; st < 8; ++st) {
                uint32_t v0 = 0u, v1 = 0u;
                if (tok < m) {
                    v0 = __ldg(reinterpret_cast<const uint32_t*>(xr + st * 16));
                    v1 = __ldg(reinterpret_cast<const uint32_t*>(xr + st * 16 + 8));
                }
                xb[nb][st][0] = v0; xb[nb][st][1] = v1;
            }
        }
    };

    float acc[NT * CH][4];
    #pragma unroll
    for (int n = 0; n < NT * CH; ++n)
        #pragma unroll
        for (int q = 0; q < 4; ++q) acc[n][q] = 0.f;

    #pragma unroll
    for (int st = 0; st < NST - 1; ++st) {
        if (st < NCW) issue(st, st);
        cp_async_commit();
    }
    uint32_t xb[2][NT][8][2];
    load_x(0, xb[0]);
    for (int c = 0; c < NCW; ++c) {
        if (c + 1 < NCW) load_x(c + 1, xb[(c + 1) & 1]);               // prefetch next chunk's B fragments
        cp_async_wait<NST - 2>();
        __syncwarp();
        {
            const int cn = c + NST - 1;
            if (cn < NCW) issue(cn, cn % NST);
            cp_async_commit();
        }
        const uint8_t* ws = w12b_smem + warp * (NST * W12_TILE_BYTES) + (c % NST) * W12_TILE_BYTES;
        uint32_t (&xc)[NT][8][2] = xb[c & 1];
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            uint4 lo[2]; uint4 nbw;
            lo[0] = *reinterpret_cast<const uint4*>(ws + (2 * h) * 512 + lane * 16);
            lo[1] = *reinterpret_cast<const uint4*>(ws + (2 * h + 1) * 512 + lane * 16);
            nbw = *reinterpret_cast<const uint4*>(ws + 2048 + h * 512 + lane * 16);
            #pragma unroll
            for (int ps = 0; ps < 2; ++ps) {
                #pragma unroll
                for (int q = 0; q < 2; ++q) {
                    const int si = 2 * ps + q, s_abs = 4 * h + si;
                    const uint32_t L0 = q == 0 ? lo[ps].x : lo[ps].z, L1 = q == 0 ? lo[ps].y : lo[ps].w;
                    const uint32_t Nw = si == 0 ? nbw.x : si == 1 ? nbw.y : si == 2 ? nbw.z : nbw.w;
                    uint32_t a0, a1, a2, a3;
                    decode4(L0, Nw & 0xFFFFu, lut0, lut1, a0, a1);
                    decode4(L1, Nw >> 16, lut0, lut1, a2, a3);
                    #pragma unroll
                    for (int nb = 0; nb < NT; ++nb)
                        mma16816(acc[nb * CH + (CH == 2 ? q : 0)], a0, a1, a2, a3, xc[nb][s_abs][0], xc[nb][s_abs][1]);
                }
            }
        }
        __syncwarp();                                                   // all lanes done with this ring slot
    }
    cp_async_wait<0>();
    if (CH == 2) {
        #pragma unroll
        for (int q = 0; q < 4; ++q) acc[0][q] += acc[1][q];
    }
    // reduce the W warps' accumulators through smem: red[row][tok]
    for (int i = tid; i < 16 * 33; i += W * 32) (&red[0][0])[i] = 0.f;
    __syncthreads();
    #pragma unroll
    for (int nb = 0; nb < NT; ++nb)
        #pragma unroll
        for (int hr = 0; hr < 2; ++hr)
            #pragma unroll
            for (int cc = 0; cc < 2; ++cc) {
                const int tok = nb * 8 + 2 * t + cc;
                if (tok < m) atomicAdd(&red[g + 8 * hr][tok], acc[nb * CH][2 * hr + cc]);
            }
    __syncthreads();
    // escapes of this tile's rows: pack tile = tile / 2, local rows [16 * (tile & 1), +16)
    const int pt = tile >> 1, rbase = 16 * (tile & 1);
    const int e0 = esc_off[pt], ecnt = esc_off[pt + 1] - e0;
    for (int idx = tid; idx < ecnt * m; idx += W * 32) {
        const int e = e0 + idx / m, i = idx - (idx / m) * m;
        const int nl = esc_n[e] - rbase;
        if (nl >= 0 && nl < 16) atomicAdd(&red[nl][i], esc_d[e] * __bfloat162float(X[(size_t)i * ldx + esc_k[e]]));
    }
    __syncthreads();
    for (int idx = tid; idx < 16 * m; idx += W * 32) {
        const int n = idx & 15, i = idx >> 4;
        out[(size_t)i * ldo + tile * 16 + n] = __float2bfloat16(red[n][i]);
    }
}

template <int NT, int W, int NST>
void launch_w12b(const __nv_bfloat16* X, int ldx, int m, const uint8_t* Wt, int eb, const int* eo, const int* en, const int* ek, const float* ed,
                 __nv_bfloat16* out, int ldo, int N, int K, cudaStream_t stream)
{
    constexpr int smem = W * NST * W12_TILE_BYTES;
    static bool attr = false;
    if (!attr) { cudaFuncSetAttribute(w12b_kernel<NT, W, NST>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
    w12b_kernel<NT, W, NST><<<N / 16, W * 32, smem, stream>>>(X, ldx, m, Wt, eb, eo, en, ek, ed, out, ldo, N, K);
}

// ---- w12m: 12-bit W x BF16 x^T GEMM for m = 8*NT tokens (128 / 256). CTA = 8 warps, warp w owns MW m16 weight
// tiles (rows (blockIdx.x*8 + w)*MW*16 ..); the W12 layout [n/64][k/128][4][3 KB] is read as 2*MW consecutive
// 4-tile groups per chunk. All warps share the chunk's x tile (8*NT tokens x 128 k, cp.async'd once per CTA, B
// fragments by ldmatrix); the fragments of a warp's MW tiles are reused over the NT n8 token blocks. gridDim.y
// CTAs split K; each writes fp32 partials part[z][tok][n]; w12m_reduce_kernel (PDL) sums the slices, adds the
// escapes of row n (esc_roff per row: delta x x[tok][k]) and writes bf16.
constexpr int W12M_WARPS = 8;
// non-volatile forms: ptxas may schedule the fragment loads ahead of the mmas (the loads keep a memory clobber so they
// stay behind the barrier that publishes the stage)
__device__ __forceinline__ void ldsm_x4_nv(uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint32_t addr) {
    asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(addr) : "memory");
}
__device__ __forceinline__ void mma16816_nv(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__host__ __device__ constexpr int w12m_stage_bytes(int mw, int nt) { return W12M_WARPS * mw * W12_TILE_BYTES + 8 * nt * W12_XSTRIDE * 2; }

template <int MW, int NT, int NSTAGE, int MODE = 0>
__global__ void __launch_bounds__(256, 1)
w12m_kernel(const __nv_bfloat16* __restrict__ X, int ldx, int m, const uint8_t* __restrict__ W, int eb,
            float* __restrict__ part, int N, int K)
{
    constexpr int STAGE_BYTES = w12m_stage_bytes(MW, NT);
    constexpr int W_STAGE = W12M_WARPS * MW * W12_TILE_BYTES;          // 24 KB (MW = 1) / 48 KB
    constexpr int GROUP_BYTES = 4 * W12_TILE_BYTES;                     // one 4-tile group per chunk: 12 KB
    constexpr int WSEG = W_STAGE / 16 / 256;                            // 16-B weight segments per thread per chunk
    constexpr int XSEG = NT * 128 / 256;                                // token-row segments per thread per chunk
    static_assert(W_STAGE % (16 * 256) == 0 && (NT * 128) % 256 == 0, "w12m: segment counts");
    extern __shared__ __align__(128) uint8_t w12m_smem[];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KSPLIT = (int)gridDim.y;
    const int NC_ALL = K / W12_KC;
    const int NC = NC_ALL / KSPLIT;
    const int c0 = (int)blockIdx.y * NC;
    const uint32_t hb = (uint32_t)(eb >> 1);
    const uint32_t lut0 = hb | (hb + 1) << 8 | (hb + 2) << 16 | (hb + 3) << 24;
    const uint32_t lut1 = (hb + 4) | (hb + 5) << 8 | (hb + 6) << 16 | (hb + 7) << 24;

    // weight segments: the stage holds the CTA's 2*MW groups back to back (tile ti = gi*4 + slot at ti * 3 KB)
    const int group0 = (int)blockIdx.x * 2 * MW;
    // segment i of this thread: sidx = tid + 256 i -> group gi = i / 3 (768 segments per group), offset (i % 3) * 4 KB
    const size_t gstride = (size_t)NC_ALL * GROUP_BYTES;
    const uint8_t* wbase = W + ((size_t)group0 * NC_ALL + c0) * GROUP_BYTES + tid * 16;
    const int xr0 = tid >> 4;                                           // token row of segment 0 (+16 per segment)
    const __nv_bfloat16* xbase = X + (size_t)xr0 * ldx + c0 * W12_KC + (tid & 15) * 8;
    const uint32_t sbase = smem_u32(w12m_smem);
    auto issue = [&](int c, int stage) {
        const uint32_t ws = sbase + stage * STAGE_BYTES;
        #pragma unroll
        for (int i = 0; i < WSEG; ++i)
            cp_async16(ws + (tid + i * 256) * 16, wbase + (size_t)(i / 3) * gstride + (i % 3) * 4096 + (size_t)c * GROUP_BYTES, 16);
        const uint32_t xs = ws + W_STAGE;
        #pragma unroll
        for (int i = 0; i < XSEG; ++i) {
            const int r = xr0 + 16 * i;
            cp_async16(xs + (r * W12_XSTRIDE + (tid & 15) * 8) * 2, xbase + (size_t)(r < m ? 16 * i : -xr0) * ldx + c * W12_KC, r < m ? 16 : 0);
        }
    };

    float acc[MW][NT][4];
    #pragma unroll
    for (int i = 0; i < MW; ++i)
        #pragma unroll
        for (int nb = 0; nb < NT; ++nb)
            #pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][nb][q] = 0.f;

    #pragma unroll
    for (int st = 0; st < NSTAGE - 1; ++st) {
        if (st < NC) issue(st, st);
        cp_async_commit();
    }
    int stage = 0, stage_n = NSTAGE - 1;
    for (int c = 0; c < NC; ++c) {
        cp_async_wait<NSTAGE - 2>();
        __syncthreads();
        {
            const int cn = c + NSTAGE - 1;
            if (cn < NC) issue(cn, stage_n);
            cp_async_commit();
        }
        const uint8_t* ws = w12m_smem + stage * STAGE_BYTES + warp * MW * W12_TILE_BYTES;
        const __half* xs = reinterpret_cast<const __half*>(w12m_smem + stage * STAGE_BYTES + W_STAGE);
        stage = stage + 1 == NSTAGE ? 0 : stage + 1;
        stage_n = stage_n + 1 == NSTAGE ? 0 : stage_n + 1;
        const uint32_t xb = smem_u32(xs) + ((lane & 7) * W12_XSTRIDE + (lane >> 3) * 8) * 2;
        if constexpr (MODE == 1) {                                        // bench ablation: the cp.async ring only
            if (lane == 0) acc[0][0][0] += __uint_as_float(*reinterpret_cast<const uint32_t*>(ws + warp * 16) & 0x007FFFFFu);
            continue;
        }
        #pragma unroll
        for (int h = 0; h < 2; ++h) {                                     // k16 steps 4h..4h+3
            uint4 lo[MW][2], nbw[MW];
            #pragma unroll
            for (int i = 0; i < MW; ++i) {
                lo[i][0] = *reinterpret_cast<const uint4*>(ws + i * W12_TILE_BYTES + (2 * h) * 512 + lane * 16);
                lo[i][1] = *reinterpret_cast<const uint4*>(ws + i * W12_TILE_BYTES + (2 * h + 1) * 512 + lane * 16);
                nbw[i] = *reinterpret_cast<const uint4*>(ws + i * W12_TILE_BYTES + 2048 + h * 512 + lane * 16);
            }
            #pragma unroll
            for (int ps = 0; ps < 2; ++ps) {                                 // steps 4h+2ps, +1
                uint32_t a[MW][2][4];
                #pragma unroll
                for (int i = 0; i < MW; ++i)
                    #pragma unroll
                    for (int q = 0; q < 2; ++q) {
                        const uint32_t L0 = q == 0 ? lo[i][ps].x : lo[i][ps].z, L1 = q == 0 ? lo[i][ps].y : lo[i][ps].w;
                        const int si = 2 * ps + q;
                        const uint32_t Nw = si == 0 ? nbw[i].x : si == 1 ? nbw[i].y : si == 2 ? nbw[i].z : nbw[i].w;
                        decode4(L0, Nw & 0xFFFFu, lut0, lut1, a[i][q][0], a[i][q][1]);
                        decode4(L1, Nw >> 16, lut0, lut1, a[i][q][2], a[i][q][3]);
                    }
                // B fragments of GB token blocks at a time, then their mmas in an order that keeps the mmas on one
                // accumulator GB*MW apart (warps issue in order: a dependent mma right behind its producer stalls)
                constexpr int GB = NT <= 16 ? NT : 8;
                #pragma unroll
                for (int nb0 = 0; nb0 < NT; nb0 += GB) {
                    uint32_t b[GB][4];
                    #pragma unroll
                    for (int j = 0; j < GB; ++j)
                        ldsm_x4_nv(b[j][0], b[j][1], b[j][2], b[j][3], xb + ((nb0 + j) * 8 * W12_XSTRIDE + h * 64 + ps * 32) * 2);
                    #pragma unroll
                    for (int q = 0; q < 2; ++q)
                        #pragma unroll
                        for (int j = 0; j < GB; ++j)
                            #pragma unroll
                            for (int i = 0; i < MW; ++i)
                                mma16816_nv(acc[i][nb0 + j], a[i][q][0], a[i][q][1], a[i][q][2], a[i][q][3], b[j][2 * q], b[j][2 * q + 1]);
                }
            }
        }
    }
    cp_async_wait<0>();
    pdl_trigger();                                                       // the reduce may start launching
    // fp32 partial slice: part[z][tok][n], n = tile*16 + g (+8), tok = nb*8 + 2t (+1)
    float* pz = part + (size_t)blockIdx.y * m * N;
    #pragma unroll
    for (int i = 0; i < MW; ++i) {
        const int col0 = (((int)blockIdx.x * W12M_WARPS + warp) * MW + i) * 16 + g;
        #pragma unroll
        for (int nb = 0; nb < NT; ++nb)
            #pragma unroll
            for (int hr = 0; hr < 2; ++hr) {
                const int tok = nb * 8 + 2 * t;
                if (tok < m) pz[(size_t)tok * N + col0 + 8 * hr] = acc[i][nb][2 * hr];
                if (tok + 1 < m) pz[(size_t)(tok + 1) * N + col0 + 8 * hr] = acc[i][nb][2 * hr + 1];
            }
    }
}

__global__ void __launch_bounds__(256)
w12m_reduce_kernel(const float* __restrict__ part, int KSPLIT, int m, int N,
                   const int* __restrict__ esc_roff, const int* __restrict__ esc_k, const float* __restrict__ esc_d,
                   const __nv_bfloat16* __restrict__ X, int ldx, __nv_bfloat16* __restrict__ out, int ldo)
{
    const int i = blockIdx.y;                                            // token
    const int n0 = ((int)blockIdx.x * 256 + (int)threadIdx.x) * 2;       // column pair
    if (n0 >= N) return;
    const int e00 = esc_roff[n0], e01 = esc_roff[n0 + 1], e11 = esc_roff[n0 + 2];
    pdl_wait();                                                          // the partials come from the GEMM launch
    float sx = 0.f, sy = 0.f;
    for (int z = 0; z < KSPLIT; ++z) {
        const float2 v = __ldcg(reinterpret_cast<const float2*>(part + ((size_t)z * m + i) * N + n0));
        sx += v.x; sy += v.y;
    }
    const __nv_bfloat16* xr = X + (size_t)i * ldx;
    for (int e = e00; e < e01; ++e) sx += esc_d[e] * __bfloat162float(xr[esc_k[e]]);
    for (int e = e01; e < e11; ++e) sy += esc_d[e] * __bfloat162float(xr[esc_k[e]]);
    *reinterpret_cast<__nv_bfloat162*>(out + (size_t)i * ldo + n0) = __floats2bfloat162_rn(sx, sy);
}

template <int MW, int NT, int NSTAGE, int MODE = 0>
void launch_w12m(const __nv_bfloat16* X, int ldx, int m, const uint8_t* W, int eb, float* part, int N, int K, int sk, cudaStream_t stream)
{
    constexpr int smem = NSTAGE * w12m_stage_bytes(MW, NT);
    static bool attr = false;
    if (!attr) { cudaFuncSetAttribute(w12m_kernel<MW, NT, NSTAGE, MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
    w12m_kernel<MW, NT, NSTAGE, MODE><<<dim3(N / (128 * MW), sk), 256, smem, stream>>>(X, ldx, m, W, eb, part, N, K);
}

// == BEGIN t12 device ==
// ---- TMA bulk-copy helpers (1-D cp.async.bulk global -> shared, completion on an mbarrier)
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
    asm volatile("{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}\n"
                 :: "r"(bar), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void fence_mbar_init() { asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}

// ---- bstream: streaming-read microbenchmark through a shared-memory ring. Each CTA reads a contiguous slice of
// the buffer in SB-byte stages; BULK: one elected thread issues cp.async.bulk copies (one mbarrier per stage);
// else every thread copies 16 B segments with cp.async (commit / wait_group). Each stage is touched (128 x 4 B) so
// the ring is really consumed. 128 threads; the host pads the dynamic smem to pin the CTAs per SM.
template <int NST, int SB, bool BULK>
__global__ void __launch_bounds__(128) bstream_kernel(const uint8_t* __restrict__ p, long nbytes, uint32_t* __restrict__ out)
{
    extern __shared__ __align__(128) uint8_t bs_smem[];
    __shared__ __align__(8) unsigned long long bars[NST];
    const long per = (nbytes / gridDim.x) / SB;                       // stages per CTA
    const uint8_t* base = p + (long)blockIdx.x * per * SB;
    const uint32_t sb = smem_u32(bs_smem);
    const int tid = threadIdx.x;
    uint32_t acc = 0;
    if constexpr (BULK) {
        if (tid == 0) {
            #pragma unroll
            for (int s = 0; s < NST; ++s) mbar_init(smem_u32(&bars[s]), 1);
            fence_mbar_init();
        }
        __syncthreads();
        if (tid == 0) {
            #pragma unroll
            for (int s = 0; s < NST; ++s)
                if (s < per) { mbar_expect_tx(smem_u32(&bars[s]), SB); bulk_g2s(sb + s * SB, base + (long)s * SB, SB, smem_u32(&bars[s])); }
        }
        for (long c = 0; c < per; ++c) {
            const int st = (int)(c % NST);
            mbar_wait(smem_u32(&bars[st]), (uint32_t)((c / NST) & 1));
            acc ^= *reinterpret_cast<const uint32_t*>(bs_smem + st * SB + tid * 16);
            __syncthreads();                                          // everyone has read stage st
            if (tid == 0) {
                const long cn = c + NST;
                if (cn < per) {
                    fence_proxy_async();
                    mbar_expect_tx(smem_u32(&bars[st]), SB);
                    bulk_g2s(sb + st * SB, base + cn * SB, SB, smem_u32(&bars[st]));
                }
            }
        }
    } else {
        constexpr int SEG = SB / 2048;                                 // 16 B segments per thread per stage
        auto issue = [&](long c, int st) {
            #pragma unroll
            for (int i = 0; i < SEG; ++i)
                cp_async16(sb + st * SB + (i * 128 + tid) * 16, base + c * SB + (i * 128 + tid) * 16, 16);
        };
        #pragma unroll
        for (int s = 0; s < NST - 1; ++s) {
            if (s < per) issue(s, s);
            cp_async_commit();
        }
        for (long c = 0; c < per; ++c) {
            const int st = (int)(c % NST);
            cp_async_wait<NST - 2>();
            __syncthreads();
            {
                const long cn = c + NST - 1;
                if (cn < per) issue(cn, (int)(cn % NST));
                cp_async_commit();
            }
            acc ^= *reinterpret_cast<const uint32_t*>(bs_smem + st * SB + tid * 16);
        }
        cp_async_wait<0>();
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc ^= __shfl_xor_sync(0xffffffffu, acc, o);
    if ((tid & 31) == 0) out[blockIdx.x * 4 + (tid >> 5)] = acc;
}

// ---- t12: single-launch 12-bit GEMM for m <= 32 (tokens = the mma N side, 8*NT), one CTA per R-row tile of W over
// the FULL K: no cross-CTA reduction, no second launch. Weights in the T12 layout (_repack_t12): per (R-row tile,
// KC = 128*CK wide chunk) one contiguous block of (R/16) x CK W12 fragment blocks (3 KB each, [r][ck]), streamed by
// TMA bulk copies: thread 0 issues one cp.async.bulk per stage into an NSW-deep ring (one mbarrier per stage). The
// x tile (m token rows x KC bf16, rows padded to KC + 8 halves) comes from L2 by one bulk copy per token row into an
// NSX-deep ring. Warp w owns the m16 row tile r_w = w / WPR and, per chunk, UPW half-blocks (64 k each): PRMT-LUT
// decode (decode4) into the A fragments, B fragments by ldmatrix.x4 from the x tile, mma.sync m16n8k16 bf16 -> fp32.
// The row's escapes (esc_roff / esc_k / esc_d: delta * x[tok][k]) are computed during the pipeline fill; the
// epilogue sums the warps' partials per (row, token) through smem, adds them and stores bf16. DIAG (bench only):
// %globaltimer at start / first stage landed / loop end / store end for CTA 0, and the min start / max end over CTAs.
template <int R, int CK, int W, int NSW, int NSX, int NT, bool DIAG>
__global__ void __launch_bounds__(W * 32)
t12_kernel(const __nv_bfloat16* __restrict__ X, int ldx, int m, const uint8_t* __restrict__ Wt, int eb,
           const int* __restrict__ esc_roff, const int* __restrict__ esc_k, const float* __restrict__ esc_d,
           __nv_bfloat16* __restrict__ out, int ldo, int K, unsigned long long* __restrict__ diag)
{
    constexpr int KC = 128 * CK, XS = KC + 8;                          // x stage row stride (halves): 16 B multiple, conflict-free
    constexpr int WST = (R / 16) * CK * 3072, XST = 8 * NT * XS * 2;
    constexpr int U = (R / 8) * CK, UPW = U / W, WPR = W / (R / 16);
    static_assert(U % W == 0 && (2 * CK) % UPW == 0 && W % (R / 16) == 0, "t12: warp / unit split");
    constexpr int CH = NT == 1 ? 2 : 1;
    constexpr int NTHR = W * 32, TPR = NTHR / R;                       // tokens per epilogue pass
    static_assert(NTHR % R == 0, "t12: threads per row");
    constexpr int TPT = (8 * NT + TPR - 1) / TPR;                      // tokens per thread
    static_assert(NSX <= NSW, "t12: x ring deeper than the weight ring");
    extern __shared__ __align__(128) uint8_t t12_smem[];
    __shared__ __align__(8) unsigned long long barw[NSW], barx[NSX];
    __shared__ float red[W][16][8 * NT + 1];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int tile = blockIdx.x, NC = K / KC, ntb = (m + 7) >> 3;
    const int r_w = warp / WPR;
    unsigned long long t0 = 0, t1 = 0;
    if constexpr (DIAG) t0 = gtimer();
    const uint32_t hb = (uint32_t)(eb >> 1);
    const uint32_t lut0 = hb | (hb + 1) << 8 | (hb + 2) << 16 | (hb + 3) << 24;
    const uint32_t lut1 = (hb + 4) | (hb + 5) << 8 | (hb + 6) << 16 | (hb + 7) << 24;
    const uint8_t* wsrc = Wt + (size_t)tile * NC * WST;
    const uint32_t sw = smem_u32(t12_smem), sx = sw + NSW * WST;
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < NSW; ++s) mbar_init(smem_u32(&barw[s]), 1);
        #pragma unroll
        for (int s = 0; s < NSX; ++s) mbar_init(smem_u32(&barx[s]), 1);
        fence_mbar_init();
    }
    __syncthreads();
    auto issue_w = [&](int c, int st) {                                 // thread 0
        const uint32_t bar = smem_u32(&barw[st]);
        mbar_expect_tx(bar, WST);
        bulk_g2s(sw + st * WST, wsrc + (size_t)c * WST, WST, bar);
    };
    auto issue_x = [&](int c, int st) {                                 // warp W-1: lane i copies token row i
        const uint32_t bar = smem_u32(&barx[st]);
        if (lane == 0) mbar_expect_tx(bar, (uint32_t)m * KC * 2);
        if (lane < m) bulk_g2s(sx + st * XST + lane * XS * 2, X + (size_t)lane * ldx + (size_t)c * KC, KC * 2, bar);
    };
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < NSW; ++s) if (s < NC) issue_w(s, s);
    }
    if (warp == W - 1) {
        #pragma unroll
        for (int s = 0; s < NSX; ++s) if (s < NC) issue_x(s, s);
    }
    // the row's escapes, during the pipeline fill: thread -> row rr, tokens i0 + j * TPR
    const int rr = tid % R, i0 = tid / R, row = tile * R + rr;
    float esc[TPT];
    #pragma unroll
    for (int j = 0; j < TPT; ++j) esc[j] = 0.f;
    {
        const int e0 = esc_roff[row], e1 = esc_roff[row + 1];
        for (int e = e0; e < e1; ++e) {
            const int kk = esc_k[e];
            const float d = esc_d[e];
            #pragma unroll
            for (int j = 0; j < TPT; ++j) {
                const int i = i0 + j * TPR;
                if (i < m) esc[j] += d * __bfloat162float(X[(size_t)i * ldx + kk]);
            }
        }
    }
    // this warp's units: stage-relative byte offsets of the low-byte quads / nibble words, x k offset (halves)
    int lo_off[UPW], nb_off[UPW], xk[UPW];
    #pragma unroll
    for (int j = 0; j < UPW; ++j) {
        const int u = warp * UPW + j, ck = (u % (2 * CK)) >> 1, h = u & 1;
        const int woff = (r_w * CK + ck) * 3072;
        lo_off[j] = woff + (2 * h) * 512 + lane * 16;
        nb_off[j] = woff + 2048 + h * 512 + lane * 16;
        xk[j] = ck * 128 + h * 64;
    }
    float acc[NT * CH][4];
    #pragma unroll
    for (int n = 0; n < NT * CH; ++n)
        #pragma unroll
        for (int q = 0; q < 4; ++q) acc[n][q] = 0.f;

    for (int c = 0; c < NC; ++c) {
        const int swi = c % NSW, sxi = c % NSX;
        if (c > 0) {
            __syncthreads();                                            // all warps are done with chunk c-1
            if (tid == 0 && c - 1 + NSW < NC) { fence_proxy_async(); issue_w(c - 1 + NSW, (c - 1) % NSW); }
            if (warp == W - 1 && c - 1 + NSX < NC) { fence_proxy_async(); issue_x(c - 1 + NSX, (c - 1) % NSX); }
        }
        mbar_wait(smem_u32(&barw[swi]), (uint32_t)((c / NSW) & 1));
        mbar_wait(smem_u32(&barx[sxi]), (uint32_t)((c / NSX) & 1));
        if constexpr (DIAG) { if (c == 0) t1 = gtimer(); }
        const uint8_t* ws = t12_smem + swi * WST;
        const uint32_t xs = sx + sxi * XST;
        #pragma unroll
        for (int j = 0; j < UPW; ++j) {
            const uint4 lo0 = *reinterpret_cast<const uint4*>(ws + lo_off[j]);
            const uint4 lo1 = *reinterpret_cast<const uint4*>(ws + lo_off[j] + 512);
            const uint4 nb = *reinterpret_cast<const uint4*>(ws + nb_off[j]);
            #pragma unroll
            for (int ps = 0; ps < 2; ++ps) {                             // k16 steps 2ps, 2ps+1 of the half-block
                uint32_t b[NT][4];
                #pragma unroll
                for (int nbk = 0; nbk < NT; ++nbk) {
                    if (nbk < ntb)
                        ldsm_x4(b[nbk][0], b[nbk][1], b[nbk][2], b[nbk][3],
                                xs + ((nbk * 8 + (lane & 7)) * XS + xk[j] + ps * 32 + (lane >> 3) * 8) * 2);
                }
                const uint4 lo = ps == 0 ? lo0 : lo1;
                #pragma unroll
                for (int q = 0; q < 2; ++q) {
                    const uint32_t L0 = q == 0 ? lo.x : lo.z, L1 = q == 0 ? lo.y : lo.w;
                    const int si = 2 * ps + q;
                    const uint32_t Nw = si == 0 ? nb.x : si == 1 ? nb.y : si == 2 ? nb.z : nb.w;
                    uint32_t a0, a1, a2, a3;
                    decode4(L0, Nw & 0xFFFFu, lut0, lut1, a0, a1);
                    decode4(L1, Nw >> 16, lut0, lut1, a2, a3);
                    #pragma unroll
                    for (int nbk = 0; nbk < NT; ++nbk) {
                        if (nbk < ntb) mma16816(acc[nbk * CH + (CH == 2 ? q : 0)], a0, a1, a2, a3, b[nbk][2 * q], b[nbk][2 * q + 1]);
                    }
                }
            }
        }
    }
    unsigned long long t2 = 0;
    if constexpr (DIAG) t2 = gtimer();
    // partials -> red[warp][row][token]
    #pragma unroll
    for (int nbk = 0; nbk < NT; ++nbk)
        #pragma unroll
        for (int hr = 0; hr < 2; ++hr)
            #pragma unroll
            for (int cc = 0; cc < 2; ++cc)
                red[warp][g + 8 * hr][nbk * 8 + 2 * t + cc] = acc[nbk * CH][2 * hr + cc] + (CH == 2 ? acc[nbk * CH + 1][2 * hr + cc] : 0.f);
    __syncthreads();
    const int rt = rr >> 4, rl = rr & 15;
    #pragma unroll
    for (int j = 0; j < TPT; ++j) {
        const int i = i0 + j * TPR;
        if (i < m) {
            float v = esc[j];
            #pragma unroll
            for (int w2 = 0; w2 < WPR; ++w2) v += red[rt * WPR + w2][rl][i];
            out[(size_t)i * ldo + tile * R + rr] = __float2bfloat16(v);
        }
    }
    if constexpr (DIAG) {
        __syncthreads();
        if (tid == 0) {
            const unsigned long long t3 = gtimer();
            if (tile == 0) { diag[0] = t0; diag[1] = t1; diag[2] = t2; diag[3] = t3; }
            atomicMin(diag + 4, t0); atomicMax(diag + 5, t3);
            if (tile == gridDim.x - 1) { diag[6] = t0; diag[7] = t3; }
        }
    }
}

template <int R, int CK, int W, int NSW, int NSX, int NT, bool DIAG>
void launch_t12(const __nv_bfloat16* X, int ldx, int m, const uint8_t* Wt, int eb, const int* ero, const int* ek, const float* ed,
                __nv_bfloat16* out, int ldo, int N, int K, unsigned long long* diag, cudaStream_t stream)
{
    constexpr int smem = NSW * (R / 16) * CK * 3072 + NSX * 8 * NT * (128 * CK + 8) * 2;
    static bool attr = false;
    if (!attr) { cudaFuncSetAttribute(t12_kernel<R, CK, W, NSW, NSX, NT, DIAG>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
    t12_kernel<R, CK, W, NSW, NSX, NT, DIAG><<<N / R, W * 32, smem, stream>>>(X, ldx, m, Wt, eb, ero, ek, ed, out, ldo, K, diag);
}
__host__ constexpr int t12_smem_bytes(int r, int ck, int nsw, int nsx, int nt) { return nsw * (r / 16) * ck * 3072 + nsx * 8 * nt * (128 * ck + 8) * 2; }
// ---- t12b: t12 with a producer / consumer split. Warp WC (the producer) issues the weight bulk copies into the
// NSW-stage ring against full[] (tx) / empty[] (count WC, one arrive per compute warp) mbarriers and, while the ring
// is full, computes the tile's escapes into esc_s[R][8*NT] (lane = row; delta * x[tok][k]). The WC compute warps
// fetch the x tile themselves with 16-byte cp.async (rows >= m zero-filled) into an NSX-stage ring ordered by a named
// barrier (bar.sync 1) once per chunk, decode + mma as t12, and sum / add esc_s / store in the epilogue.
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void named_bar_sync(int id, int count) {
    asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(count) : "memory");
}

template <int R, int CK, int WC, int NSW, int NSX, int NT, bool DIAG>
__global__ void __launch_bounds__((WC + 1) * 32)
t12b_kernel(const __nv_bfloat16* __restrict__ X, int ldx, int m, const uint8_t* __restrict__ Wt, int eb,
            const int* __restrict__ esc_roff, const int* __restrict__ esc_k, const float* __restrict__ esc_d,
            __nv_bfloat16* __restrict__ out, int ldo, int K, unsigned long long* __restrict__ diag)
{
    constexpr int KC = 128 * CK, XS = KC + 8;
    constexpr int WST = (R / 16) * CK * 3072, XST = 8 * NT * XS * 2;
    constexpr int U = (R / 8) * CK, UPW = U / WC, WPR = WC / (R / 16);
    static_assert(U % WC == 0 && (2 * CK) % UPW == 0 && WC % (R / 16) == 0, "t12b: warp / unit split");
    constexpr int CH = NT == 1 ? 2 : 1;
    constexpr int NTC = WC * 32, TPR = NTC / R;                        // compute threads, tokens per epilogue pass
    static_assert(NTC % R == 0 && R <= 32, "t12b: threads per row");
    constexpr int TPT = (8 * NT + TPR - 1) / TPR;
    constexpr int SEGS = 8 * NT * (KC / 8), XSEG = (SEGS + NTC - 1) / NTC;   // 16 B x segments per chunk, per thread
    extern __shared__ __align__(128) uint8_t t12b_smem[];
    __shared__ __align__(8) unsigned long long full[NSW], empty[NSW];
    __shared__ float red[WC][16][8 * NT + 1];
    __shared__ float esc_s[R][8 * NT];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int tile = blockIdx.x, NC = K / KC, ntb = (m + 7) >> 3;
    unsigned long long t0 = 0, t1 = 0;
    if constexpr (DIAG) t0 = gtimer();
    const uint32_t sw = smem_u32(t12b_smem), sx = sw + NSW * WST;
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < NSW; ++s) { mbar_init(smem_u32(&full[s]), 1); mbar_init(smem_u32(&empty[s]), WC); }
        fence_mbar_init();
    }
    __syncthreads();
    if (warp == WC) {
        // ---- producer warp
        const uint8_t* wsrc = Wt + (size_t)tile * NC * WST;
        auto issue_w = [&](int c, int st) {
            const uint32_t bar = smem_u32(&full[st]);
            mbar_expect_tx(bar, WST);
            bulk_g2s(sw + st * WST, wsrc + (size_t)c * WST, WST, bar);
        };
        if (lane == 0) {
            #pragma unroll
            for (int s = 0; s < NSW; ++s) if (s < NC) issue_w(s, s);
        }
        // the tile's escapes while the ring is full: lane = row
        for (int i = lane; i < R * 8 * NT; i += 32) (&esc_s[0][0])[i] = 0.f;
        __syncwarp();
        if (lane < R) {
            const int row = tile * R + lane;
            const int e0 = esc_roff[row], e1 = esc_roff[row + 1];
            for (int e = e0; e < e1; ++e) {
                const int kk = esc_k[e];
                const float d = esc_d[e];
                for (int i = 0; i < m; ++i) esc_s[lane][i] += d * __bfloat162float(X[(size_t)i * ldx + kk]);
            }
        }
        if (lane == 0) {
            for (int c = NSW; c < NC; ++c) {
                const int st = c % NSW;
                mbar_wait(smem_u32(&empty[st]), (uint32_t)((c / NSW - 1) & 1));
                issue_w(c, st);
            }
        }
    } else {
        // ---- compute warps
        const int r_w = warp / WPR;
        const uint32_t hb = (uint32_t)(eb >> 1);
        const uint32_t lut0 = hb | (hb + 1) << 8 | (hb + 2) << 16 | (hb + 3) << 24;
        const uint32_t lut1 = (hb + 4) | (hb + 5) << 8 | (hb + 6) << 16 | (hb + 7) << 24;
        const __nv_bfloat16* xsrc[XSEG]; int xbytes[XSEG]; uint32_t xdst[XSEG];
        #pragma unroll
        for (int i = 0; i < XSEG; ++i) {
            const int sidx = tid + i * NTC, row = sidx / (KC / 8), seg = sidx % (KC / 8);
            const bool ok = sidx < SEGS && row < m;
            xbytes[i] = ok ? 16 : 0;
            xsrc[i] = X + (size_t)(ok ? row : 0) * ldx + seg * 8;
            xdst[i] = (sidx < SEGS) ? (uint32_t)((row * XS + seg * 8) * 2) : 0xFFFFFFFFu;
        }
        auto issue_x = [&](int c, int st) {
            #pragma unroll
            for (int i = 0; i < XSEG; ++i)
                if (xdst[i] != 0xFFFFFFFFu) cp_async16(sx + st * XST + xdst[i], xsrc[i] + (size_t)c * KC, xbytes[i]);
        };
        int lo_off[UPW], nb_off[UPW], xk[UPW];
        #pragma unroll
        for (int j = 0; j < UPW; ++j) {
            const int u = warp * UPW + j, ck = (u % (2 * CK)) >> 1, h = u & 1;
            const int woff = (r_w * CK + ck) * 3072;
            lo_off[j] = woff + (2 * h) * 512 + lane * 16;
            nb_off[j] = woff + 2048 + h * 512 + lane * 16;
            xk[j] = ck * 128 + h * 64;
        }
        float acc[NT * CH][4];
        #pragma unroll
        for (int n = 0; n < NT * CH; ++n)
            #pragma unroll
            for (int q = 0; q < 4; ++q) acc[n][q] = 0.f;
        #pragma unroll
        for (int s = 0; s < NSX - 1; ++s) {
            if (s < NC) issue_x(s, s);
            cp_async_commit();
        }
        for (int c = 0; c < NC; ++c) {
            const int swi = c % NSW, sxi = c % NSX;
            cp_async_wait<NSX - 2>();                                   // this thread's x segments of chunk c
            named_bar_sync(1, NTC);                                     // all compute warps: x of chunk c landed, chunk c-1 done
            if (c > 0 && lane == 0) mbar_arrive(smem_u32(&empty[(c - 1) % NSW]));
            {
                const int cn = c + NSX - 1;
                if (cn < NC) issue_x(cn, cn % NSX);
                cp_async_commit();
            }
            mbar_wait(smem_u32(&full[swi]), (uint32_t)((c / NSW) & 1));
            if constexpr (DIAG) { if (c == 0) t1 = gtimer(); }
            const uint8_t* ws = t12b_smem + swi * WST;
            const uint32_t xs = sx + sxi * XST;
            #pragma unroll
            for (int j = 0; j < UPW; ++j) {
                const uint4 lo0 = *reinterpret_cast<const uint4*>(ws + lo_off[j]);
                const uint4 lo1 = *reinterpret_cast<const uint4*>(ws + lo_off[j] + 512);
                const uint4 nb = *reinterpret_cast<const uint4*>(ws + nb_off[j]);
                #pragma unroll
                for (int ps = 0; ps < 2; ++ps) {
                    uint32_t b[NT][4];
                    #pragma unroll
                    for (int nbk = 0; nbk < NT; ++nbk) {
                        if (nbk < ntb)
                            ldsm_x4(b[nbk][0], b[nbk][1], b[nbk][2], b[nbk][3],
                                    xs + ((nbk * 8 + (lane & 7)) * XS + xk[j] + ps * 32 + (lane >> 3) * 8) * 2);
                    }
                    const uint4 lo = ps == 0 ? lo0 : lo1;
                    #pragma unroll
                    for (int q = 0; q < 2; ++q) {
                        const uint32_t L0 = q == 0 ? lo.x : lo.z, L1 = q == 0 ? lo.y : lo.w;
                        const int si = 2 * ps + q;
                        const uint32_t Nw = si == 0 ? nb.x : si == 1 ? nb.y : si == 2 ? nb.z : nb.w;
                        uint32_t a0, a1, a2, a3;
                        decode4(L0, Nw & 0xFFFFu, lut0, lut1, a0, a1);
                        decode4(L1, Nw >> 16, lut0, lut1, a2, a3);
                        #pragma unroll
                        for (int nbk = 0; nbk < NT; ++nbk) {
                            if (nbk < ntb) mma16816(acc[nbk * CH + (CH == 2 ? q : 0)], a0, a1, a2, a3, b[nbk][2 * q], b[nbk][2 * q + 1]);
                        }
                    }
                }
            }
        }
        cp_async_wait<0>();
        #pragma unroll
        for (int nbk = 0; nbk < NT; ++nbk)
            #pragma unroll
            for (int hr = 0; hr < 2; ++hr)
                #pragma unroll
                for (int cc = 0; cc < 2; ++cc)
                    red[warp][g + 8 * hr][nbk * 8 + 2 * t + cc] = acc[nbk * CH][2 * hr + cc] + (CH == 2 ? acc[nbk * CH + 1][2 * hr + cc] : 0.f);
    }
    unsigned long long t2 = 0;
    if constexpr (DIAG) t2 = gtimer();
    __syncthreads();                                                    // partials and esc_s complete
    if (warp < WC) {
        const int rr = tid % R, i0 = tid / R, rt = rr >> 4, rl = rr & 15;
        #pragma unroll
        for (int j = 0; j < TPT; ++j) {
            const int i = i0 + j * TPR;
            if (i < m) {
                float v = esc_s[rr][i];
                #pragma unroll
                for (int w2 = 0; w2 < WPR; ++w2) v += red[rt * WPR + w2][rl][i];
                out[(size_t)i * ldo + tile * R + rr] = __float2bfloat16(v);
            }
        }
    }
    if constexpr (DIAG) {
        __syncthreads();
        if (tid == 0) {
            const unsigned long long t3 = gtimer();
            if (tile == 0) { diag[0] = t0; diag[1] = t1; diag[2] = t2; diag[3] = t3; }
            atomicMin(diag + 4, t0); atomicMax(diag + 5, t3);
            if (tile == gridDim.x - 1) { diag[6] = t0; diag[7] = t3; }
        }
    }
}

template <int R, int CK, int WC, int NSW, int NSX, int NT, bool DIAG>
void launch_t12b(const __nv_bfloat16* X, int ldx, int m, const uint8_t* Wt, int eb, const int* ero, const int* ek, const float* ed,
                 __nv_bfloat16* out, int ldo, int N, int K, unsigned long long* diag, cudaStream_t stream)
{
    constexpr int smem = NSW * (R / 16) * CK * 3072 + NSX * 8 * NT * (128 * CK + 8) * 2;
    static bool attr = false;
    if (!attr) { cudaFuncSetAttribute(t12b_kernel<R, CK, WC, NSW, NSX, NT, DIAG>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
    t12b_kernel<R, CK, WC, NSW, NSX, NT, DIAG><<<N / R, (WC + 1) * 32, smem, stream>>>(X, ldx, m, Wt, eb, ero, ek, ed, out, ldo, K, diag);
}
// ---- t12c: t12b with (1) the x prologue issued before the weight flood, (2) escapes applied per chunk by the
// producer warp from the x stage already in shared memory: a fixed-size per-tile table esc_tab[tile][64] of
// {k | row << 16, delta bits} sorted by k (sentinel k = 0xFFFF; tiles with more entries do not use this kernel), so
// no global x gathers and no dependent-load chain; (3) one __syncthreads per chunk (replaces the empty mbarriers:
// the producer refills the slot of chunk c-1 after the barrier of chunk c); (4) non-volatile ldsm / mma asm.
template <int R, int CK, int WC, int NSW, int NSX, int NT, bool DIAG>
__global__ void __launch_bounds__((WC + 1) * 32)
t12c_kernel(const __nv_bfloat16* __restrict__ X, int ldx, int m, const uint8_t* __restrict__ Wt, int eb,
            const int2* __restrict__ esc_tab, __nv_bfloat16* __restrict__ out, int ldo, int K, unsigned long long* __restrict__ diag)
{
    constexpr int KC = 128 * CK, XS = KC + 8;
    constexpr int WST = (R / 16) * CK * 3072, XST = 8 * NT * XS * 2;
    constexpr int U = (R / 8) * CK, UPW = U / WC, WPR = WC / (R / 16);
    static_assert(U % WC == 0 && (2 * CK) % UPW == 0 && WC % (R / 16) == 0, "t12c: warp / unit split");
    constexpr int CH = NT == 1 ? 2 : 1;
    constexpr int NTC = WC * 32, TPR = NTC / R;
    static_assert(NTC % R == 0 && R <= 32 && NSX >= 2, "t12c: threads per row / x ring");
    constexpr int TPT = (8 * NT + TPR - 1) / TPR;
    constexpr int SEGS = 8 * NT * (KC / 8), XSEG = (SEGS + NTC - 1) / NTC;
    extern __shared__ __align__(128) uint8_t t12c_smem[];
    __shared__ __align__(8) unsigned long long full[NSW];
    __shared__ float red[WC][16][8 * NT + 1];
    __shared__ float esc_s[R][8 * NT];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int tile = blockIdx.x, NC = K / KC, ntb = (m + 7) >> 3;
    unsigned long long t0 = 0, t1 = 0;
    if constexpr (DIAG) t0 = gtimer();
    const uint32_t sw = smem_u32(t12c_smem), sx = sw + NSW * WST;
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < NSW; ++s) mbar_init(smem_u32(&full[s]), 1);
        fence_mbar_init();
    }
    __syncthreads();
    if (warp == WC) {
        // ---- producer warp: escape table (one load per lane), weights after the x prologue, escapes per chunk
        int2 e0 = esc_tab[tile * 64 + lane], e1 = esc_tab[tile * 64 + 32 + lane];
        for (int i = lane; i < R * 8 * NT; i += 32) (&esc_s[0][0])[i] = 0.f;
        __syncthreads();                                                // (A) x prologue issued by the compute warps
        const uint8_t* wsrc = Wt + (size_t)tile * NC * WST;
        auto issue_w = [&](int c, int st) {
            const uint32_t bar = smem_u32(&full[st]);
            mbar_expect_tx(bar, WST);
            bulk_g2s(sw + st * WST, wsrc + (size_t)c * WST, WST, bar);
        };
        if (lane == 0) {
            #pragma unroll
            for (int s = 0; s < NSW; ++s) if (s < NC) issue_w(s, s);
        }
        int ep = 0;
        for (int c = 0; c < NC; ++c) {
            __syncthreads();                                            // (B_c) x of chunk c landed, chunk c-1 consumed
            if (c > 0 && lane == 0 && c - 1 + NSW < NC) issue_w(c - 1 + NSW, (c - 1) % NSW);
            const __nv_bfloat16* xs = reinterpret_cast<const __nv_bfloat16*>(t12c_smem + NSW * WST + (c % NSX) * XST);
            const int kend = (c + 1) * KC;
            while (ep < 64) {
                const int w0 = __shfl_sync(0xffffffffu, ep < 32 ? e0.x : e1.x, ep & 31);
                const int kk = w0 & 0xFFFF;
                if (kk >= kend) break;
                const int w1 = __shfl_sync(0xffffffffu, ep < 32 ? e0.y : e1.y, ep & 31);
                const int row = w0 >> 16;
                if (lane < m) esc_s[row][lane] += __int_as_float(w1) * __bfloat162float(xs[lane * XS + (kk - c * KC)]);
                ++ep;
            }
        }
    } else {
        // ---- compute warps
        const int r_w = warp / WPR;
        const uint32_t hb = (uint32_t)(eb >> 1);
        const uint32_t lut0 = hb | (hb + 1) << 8 | (hb + 2) << 16 | (hb + 3) << 24;
        const uint32_t lut1 = (hb + 4) | (hb + 5) << 8 | (hb + 6) << 16 | (hb + 7) << 24;
        const __nv_bfloat16* xsrc[XSEG]; int xbytes[XSEG]; uint32_t xdst[XSEG];
        #pragma unroll
        for (int i = 0; i < XSEG; ++i) {
            const int sidx = tid + i * NTC, row = sidx / (KC / 8), seg = sidx % (KC / 8);
            const bool ok = sidx < SEGS && row < m;
            xbytes[i] = ok ? 16 : 0;
            xsrc[i] = X + (size_t)(ok ? row : 0) * ldx + seg * 8;
            xdst[i] = (sidx < SEGS) ? (uint32_t)((row * XS + seg * 8) * 2) : 0xFFFFFFFFu;
        }
        auto issue_x = [&](int c, int st) {
            #pragma unroll
            for (int i = 0; i < XSEG; ++i)
                if (xdst[i] != 0xFFFFFFFFu) cp_async16(sx + st * XST + xdst[i], xsrc[i] + (size_t)c * KC, xbytes[i]);
        };
        #pragma unroll
        for (int s = 0; s < NSX - 1; ++s) {
            if (s < NC) issue_x(s, s);
            cp_async_commit();
        }
        __syncthreads();                                                // (A)
        int lo_off[UPW], nb_off[UPW], xk[UPW];
        #pragma unroll
        for (int j = 0; j < UPW; ++j) {
            const int u = warp * UPW + j, ck = (u % (2 * CK)) >> 1, h = u & 1;
            const int woff = (r_w * CK + ck) * 3072;
            lo_off[j] = woff + (2 * h) * 512 + lane * 16;
            nb_off[j] = woff + 2048 + h * 512 + lane * 16;
            xk[j] = ck * 128 + h * 64;
        }
        float acc[NT * CH][4];
        #pragma unroll
        for (int n = 0; n < NT * CH; ++n)
            #pragma unroll
            for (int q = 0; q < 4; ++q) acc[n][q] = 0.f;
        for (int c = 0; c < NC; ++c) {
            const int swi = c % NSW, sxi = c % NSX;
            cp_async_wait<NSX - 2>();                                   // this thread's x segments of chunk c
            __syncthreads();                                            // (B_c)
            {
                const int cn = c + NSX - 1;
                if (cn < NC) issue_x(cn, cn % NSX);
                cp_async_commit();
            }
            mbar_wait(smem_u32(&full[swi]), (uint32_t)((c / NSW) & 1));
            if constexpr (DIAG) { if (c == 0) t1 = gtimer(); }
            const uint8_t* ws = t12c_smem + swi * WST;
            const uint32_t xs = sx + sxi * XST;
            #pragma unroll
            for (int j = 0; j < UPW; ++j) {
                const uint4 lo0 = *reinterpret_cast<const uint4*>(ws + lo_off[j]);
                const uint4 lo1 = *reinterpret_cast<const uint4*>(ws + lo_off[j] + 512);
                const uint4 nb = *reinterpret_cast<const uint4*>(ws + nb_off[j]);
                #pragma unroll
                for (int ps = 0; ps < 2; ++ps) {
                    uint32_t b[NT][4];
                    #pragma unroll
                    for (int nbk = 0; nbk < NT; ++nbk) {
                        if (nbk < ntb)
                            ldsm_x4_nv(b[nbk][0], b[nbk][1], b[nbk][2], b[nbk][3],
                                       xs + ((nbk * 8 + (lane & 7)) * XS + xk[j] + ps * 32 + (lane >> 3) * 8) * 2);
                    }
                    const uint4 lo = ps == 0 ? lo0 : lo1;
                    #pragma unroll
                    for (int q = 0; q < 2; ++q) {
                        const uint32_t L0 = q == 0 ? lo.x : lo.z, L1 = q == 0 ? lo.y : lo.w;
                        const int si = 2 * ps + q;
                        const uint32_t Nw = si == 0 ? nb.x : si == 1 ? nb.y : si == 2 ? nb.z : nb.w;
                        uint32_t a0, a1, a2, a3;
                        decode4(L0, Nw & 0xFFFFu, lut0, lut1, a0, a1);
                        decode4(L1, Nw >> 16, lut0, lut1, a2, a3);
                        #pragma unroll
                        for (int nbk = 0; nbk < NT; ++nbk) {
                            if (nbk < ntb) mma16816_nv(acc[nbk * CH + (CH == 2 ? q : 0)], a0, a1, a2, a3, b[nbk][2 * q], b[nbk][2 * q + 1]);
                        }
                    }
                }
            }
        }
        cp_async_wait<0>();
        #pragma unroll
        for (int nbk = 0; nbk < NT; ++nbk)
            #pragma unroll
            for (int hr = 0; hr < 2; ++hr)
                #pragma unroll
                for (int cc = 0; cc < 2; ++cc)
                    red[warp][g + 8 * hr][nbk * 8 + 2 * t + cc] = acc[nbk * CH][2 * hr + cc] + (CH == 2 ? acc[nbk * CH + 1][2 * hr + cc] : 0.f);
    }
    unsigned long long t2 = 0;
    if constexpr (DIAG) t2 = gtimer();
    __syncthreads();                                                    // partials and esc_s complete
    if (warp < WC) {
        const int rr = tid % R, i0 = tid / R, rt = rr >> 4, rl = rr & 15;
        #pragma unroll
        for (int j = 0; j < TPT; ++j) {
            const int i = i0 + j * TPR;
            if (i < m) {
                float v = esc_s[rr][i];
                #pragma unroll
                for (int w2 = 0; w2 < WPR; ++w2) v += red[rt * WPR + w2][rl][i];
                out[(size_t)i * ldo + tile * R + rr] = __float2bfloat16(v);
            }
        }
    }
    if constexpr (DIAG) {
        __syncthreads();
        if (tid == 0) {
            const unsigned long long t3 = gtimer();
            if (tile == 0) { diag[0] = t0; diag[1] = t1; diag[2] = t2; diag[3] = t3; }
            atomicMin(diag + 4, t0); atomicMax(diag + 5, t3);
            if (tile == gridDim.x - 1) { diag[6] = t0; diag[7] = t3; }
        }
    }
}

template <int R, int CK, int WC, int NSW, int NSX, int NT, bool DIAG>
void launch_t12c(const __nv_bfloat16* X, int ldx, int m, const uint8_t* Wt, int eb, const int2* etab,
                 __nv_bfloat16* out, int ldo, int N, int K, unsigned long long* diag, cudaStream_t stream)
{
    constexpr int smem = NSW * (R / 16) * CK * 3072 + NSX * 8 * NT * (128 * CK + 8) * 2;
    static bool attr = false;
    if (!attr) { cudaFuncSetAttribute(t12c_kernel<R, CK, WC, NSW, NSX, NT, DIAG>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
    t12c_kernel<R, CK, WC, NSW, NSX, NT, DIAG><<<N / R, (WC + 1) * 32, smem, stream>>>(X, ldx, m, Wt, eb, etab, out, ldo, K, diag);
}
// ---- t12d: t12c with the x tile fetched by 2D TMA: one cp.async.bulk.tensor.2d per 64-wide k box (box = [8*NT rows]
// [64 k] bf16 = 128 B rows, 128B swizzle, rows >= m zero-filled by the TMA unit) over a tensor map of x encoded per
// call on the host (__grid_constant__ param), KC/64 boxes per chunk into an NSX-stage ring with its own mbarriers,
// all issued by the producer lane together with the weight bulk copies (x boxes first). ldmatrix / escape reads
// apply the 128B swizzle (16-byte chunk index XOR row & 7). The epilogue's red[] aliases the drained weight ring.
__device__ __forceinline__ void tma_2d_g2s(uint32_t dst, const void* tmap, int c0, int c1, uint32_t bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
                 :: "r"(dst), "l"(tmap), "r"(c0), "r"(c1), "r"(bar) : "memory");
}

template <int R, int CK, int WC, int NSW, int NSX, int NT, bool DIAG>
__global__ void __launch_bounds__((WC + 1) * 32)
t12d_kernel(const __grid_constant__ CUtensorMap xmap, int m, const uint8_t* __restrict__ Wt, int eb,
            const int2* __restrict__ esc_tab, __nv_bfloat16* __restrict__ out, int ldo, int K, unsigned long long* __restrict__ diag)
{
    constexpr int KC = 128 * CK, NBOX = KC / 64, BOXB = 8 * NT * 128;   // x: boxes of [8*NT rows][128 B]
    constexpr int WST = (R / 16) * CK * 3072, XST = NBOX * BOXB;
    constexpr int U = (R / 8) * CK, UPW = U / WC, WPR = WC / (R / 16);
    static_assert(U % WC == 0 && (2 * CK) % UPW == 0 && WC % (R / 16) == 0, "t12d: warp / unit split");
    constexpr int CH = NT == 1 ? 2 : 1;
    constexpr int NTC = WC * 32, TPR = NTC / R;
    static_assert(NTC % R == 0 && R <= 32 && NSX >= 2, "t12d: threads per row / x ring");
    constexpr int TPT = (8 * NT + TPR - 1) / TPR;
    static_assert(NSW * WST >= WC * 16 * (8 * NT + 1) * 4, "t12d: red must fit in the weight ring");
    extern __shared__ __align__(1024) uint8_t t12d_smem[];
    __shared__ __align__(8) unsigned long long full[NSW], xfull[NSX];
    __shared__ float esc_s[R][8 * NT];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int tile = blockIdx.x, NC = K / KC, ntb = (m + 7) >> 3;
    unsigned long long t0 = 0, t1 = 0;
    if constexpr (DIAG) t0 = gtimer();
    const uint32_t sw0 = smem_u32(t12d_smem);
    const uint32_t sw = (sw0 + 1023u) & ~1023u;                          // weight ring (3 KB blocks), then the x ring
    const uint32_t sx = sw + NSW * WST;                                  // WST is a multiple of 1024: x boxes 1024-aligned
    uint8_t* smem = t12d_smem + (sw - sw0);
    float (*red)[16][8 * NT + 1] = reinterpret_cast<float (*)[16][8 * NT + 1]>(smem);   // after the loop
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < NSW; ++s) mbar_init(smem_u32(&full[s]), 1);
        #pragma unroll
        for (int s = 0; s < NSX; ++s) mbar_init(smem_u32(&xfull[s]), 1);
        fence_mbar_init();
    }
    __syncthreads();
    if (warp == WC) {
        // ---- producer warp: x boxes (first), weights, escapes per chunk from the x stage
        int2 e0 = esc_tab[tile * 64 + lane], e1 = esc_tab[tile * 64 + 32 + lane];
        for (int i = lane; i < R * 8 * NT; i += 32) (&esc_s[0][0])[i] = 0.f;
        const uint8_t* wsrc = Wt + (size_t)tile * NC * WST;
        auto issue_x = [&](int c, int st) {
            const uint32_t bar = smem_u32(&xfull[st]);
            mbar_expect_tx(bar, XST);
            #pragma unroll
            for (int b = 0; b < NBOX; ++b) tma_2d_g2s(sx + st * XST + b * BOXB, &xmap, c * KC + b * 64, 0, bar);
        };
        auto issue_w = [&](int c, int st) {
            const uint32_t bar = smem_u32(&full[st]);
            mbar_expect_tx(bar, WST);
            bulk_g2s(sw + st * WST, wsrc + (size_t)c * WST, WST, bar);
        };
        if (lane == 0) {
            #pragma unroll
            for (int s = 0; s < NSX; ++s) if (s < NC) issue_x(s, s);
            #pragma unroll
            for (int s = 0; s < NSW; ++s) if (s < NC) issue_w(s, s);
        }
        int ep = 0;
        for (int c = 0; c < NC; ++c) {
            __syncthreads();                                            // (B_c): chunk c-1 consumed by every warp
            if (c > 0 && lane == 0) {
                if (c - 1 + NSX < NC) issue_x(c - 1 + NSX, (c - 1) % NSX);
                if (c - 1 + NSW < NC) issue_w(c - 1 + NSW, (c - 1) % NSW);
            }
            const int kend = (c + 1) * KC;
            bool waited = false;
            while (ep < 64) {
                const int w0 = __shfl_sync(0xffffffffu, ep < 32 ? e0.x : e1.x, ep & 31);
                const int kk = w0 & 0xFFFF;
                if (kk >= kend) break;
                if (!waited) { mbar_wait(smem_u32(&xfull[c % NSX]), (uint32_t)((c / NSX) & 1)); waited = true; }
                const int w1 = __shfl_sync(0xffffffffu, ep < 32 ? e0.y : e1.y, ep & 31);
                const int row = w0 >> 16, kc = kk - c * KC;
                if (lane < m) {
                    const uint8_t* xs = smem + NSW * WST + (c % NSX) * XST + (kc >> 6) * BOXB + lane * 128 + ((((kc & 63) >> 3) ^ (lane & 7)) << 4) + (kc & 7) * 2;
                    esc_s[row][lane] += __int_as_float(w1) * __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(xs));
                }
                ++ep;
            }
        }
    } else {
        // ---- compute warps
        const int r_w = warp / WPR;
        const uint32_t hb = (uint32_t)(eb >> 1);
        const uint32_t lut0 = hb | (hb + 1) << 8 | (hb + 2) << 16 | (hb + 3) << 24;
        const uint32_t lut1 = (hb + 4) | (hb + 5) << 8 | (hb + 6) << 16 | (hb + 7) << 24;
        int lo_off[UPW], nb_off[UPW], xb[UPW];
        #pragma unroll
        for (int j = 0; j < UPW; ++j) {
            const int u = warp * UPW + j, ck = (u % (2 * CK)) >> 1, h = u & 1;
            const int woff = (r_w * CK + ck) * 3072;
            lo_off[j] = woff + (2 * h) * 512 + lane * 16;
            nb_off[j] = woff + 2048 + h * 512 + lane * 16;
            xb[j] = (ck * 2 + h) * BOXB;                                // the unit's 64-k box
        }
        // ldmatrix row address within a box (row = nbk*8 + (lane & 7)), swizzled 16-byte chunk per ps
        uint32_t rowoff[NT], swz[2];
        #pragma unroll
        for (int nbk = 0; nbk < NT; ++nbk) rowoff[nbk] = (uint32_t)((nbk * 8 + (lane & 7)) * 128);
        #pragma unroll
        for (int ps = 0; ps < 2; ++ps) swz[ps] = (uint32_t)((((ps * 4) + (lane >> 3)) ^ (lane & 7)) << 4);
        float acc[NT * CH][4];
        #pragma unroll
        for (int n = 0; n < NT * CH; ++n)
            #pragma unroll
            for (int q = 0; q < 4; ++q) acc[n][q] = 0.f;
        for (int c = 0; c < NC; ++c) {
            const int swi = c % NSW, sxi = c % NSX;
            __syncthreads();                                            // (B_c)
            mbar_wait(smem_u32(&xfull[sxi]), (uint32_t)((c / NSX) & 1));
            mbar_wait(smem_u32(&full[swi]), (uint32_t)((c / NSW) & 1));
            if constexpr (DIAG) { if (c == 0) t1 = gtimer(); }
            const uint8_t* ws = smem + swi * WST;
            const uint32_t xs = sx + sxi * XST;
            #pragma unroll
            for (int j = 0; j < UPW; ++j) {
                const uint4 lo0 = *reinterpret_cast<const uint4*>(ws + lo_off[j]);
                const uint4 lo1 = *reinterpret_cast<const uint4*>(ws + lo_off[j] + 512);
                const uint4 nb = *reinterpret_cast<const uint4*>(ws + nb_off[j]);
                #pragma unroll
                for (int ps = 0; ps < 2; ++ps) {
                    uint32_t b[NT][4];
                    #pragma unroll
                    for (int nbk = 0; nbk < NT; ++nbk) {
                        if (nbk < ntb)
                            ldsm_x4_nv(b[nbk][0], b[nbk][1], b[nbk][2], b[nbk][3], xs + xb[j] + rowoff[nbk] + swz[ps]);
                    }
                    const uint4 lo = ps == 0 ? lo0 : lo1;
                    #pragma unroll
                    for (int q = 0; q < 2; ++q) {
                        const uint32_t L0 = q == 0 ? lo.x : lo.z, L1 = q == 0 ? lo.y : lo.w;
                        const int si = 2 * ps + q;
                        const uint32_t Nw = si == 0 ? nb.x : si == 1 ? nb.y : si == 2 ? nb.z : nb.w;
                        uint32_t a0, a1, a2, a3;
                        decode4(L0, Nw & 0xFFFFu, lut0, lut1, a0, a1);
                        decode4(L1, Nw >> 16, lut0, lut1, a2, a3);
                        #pragma unroll
                        for (int nbk = 0; nbk < NT; ++nbk) {
                            if (nbk < ntb) mma16816_nv(acc[nbk * CH + (CH == 2 ? q : 0)], a0, a1, a2, a3, b[nbk][2 * q], b[nbk][2 * q + 1]);
                        }
                    }
                }
            }
        }
        __syncthreads();                                                // (C): every warp is done with the weight ring
        #pragma unroll
        for (int nbk = 0; nbk < NT; ++nbk)
            #pragma unroll
            for (int hr = 0; hr < 2; ++hr)
                #pragma unroll
                for (int cc = 0; cc < 2; ++cc)
                    red[warp][g + 8 * hr][nbk * 8 + 2 * t + cc] = acc[nbk * CH][2 * hr + cc] + (CH == 2 ? acc[nbk * CH + 1][2 * hr + cc] : 0.f);
    }
    if (warp == WC) __syncthreads();                                    // (C) for the producer
    unsigned long long t2 = 0;
    if constexpr (DIAG) t2 = gtimer();
    __syncthreads();                                                    // partials and esc_s complete
    if (warp < WC) {
        const int rr = tid % R, i0 = tid / R, rt = rr >> 4, rl = rr & 15;
        #pragma unroll
        for (int j = 0; j < TPT; ++j) {
            const int i = i0 + j * TPR;
            if (i < m) {
                float v = esc_s[rr][i];
                #pragma unroll
                for (int w2 = 0; w2 < WPR; ++w2) v += red[rt * WPR + w2][rl][i];
                out[(size_t)i * ldo + tile * R + rr] = __float2bfloat16(v);
            }
        }
    }
    if constexpr (DIAG) {
        __syncthreads();
        if (tid == 0) {
            const unsigned long long t3 = gtimer();
            if (tile == 0) { diag[0] = t0; diag[1] = t1; diag[2] = t2; diag[3] = t3; }
            atomicMin(diag + 4, t0); atomicMax(diag + 5, t3);
            if (tile == gridDim.x - 1) { diag[6] = t0; diag[7] = t3; }
        }
    }
}

__host__ constexpr int t12d_smem_bytes(int r, int ck, int nsw, int nsx, int nt) { return 1024 + nsw * (r / 16) * ck * 3072 + nsx * (128 * ck / 64) * 8 * nt * 128; }

template <int R, int CK, int WC, int NSW, int NSX, int NT, bool DIAG>
void launch_t12d(const CUtensorMap& xmap, int m, const uint8_t* Wt, int eb, const int2* etab,
                 __nv_bfloat16* out, int ldo, int N, int K, unsigned long long* diag, cudaStream_t stream)
{
    constexpr int smem = t12d_smem_bytes(R, CK, NSW, NSX, NT);
    static bool attr = false;
    if (!attr) { cudaFuncSetAttribute(t12d_kernel<R, CK, WC, NSW, NSX, NT, DIAG>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
    t12d_kernel<R, CK, WC, NSW, NSX, NT, DIAG><<<N / R, (WC + 1) * 32, smem, stream>>>(xmap, m, Wt, eb, etab, out, ldo, K, diag);
}
// ---- t12e: t12d with one 3D TMA op per x chunk (tensor map {64 k, m rows, K/64 groups} with strides {ldx*2, 128 B},
// box {64, 8*NT, KC/64}: the smem box order [group][row][128 B, 128B-swizzled] is t12d's layout). Two TMA ops per
// chunk in all (check 1d05128d: the TMA unit costs ~33 ns per op, serialized per SM).
__device__ __forceinline__ void tma_3d_g2s(uint32_t dst, const void* tmap, int c0, int c1, int c2, uint32_t bar) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];\n"
                 :: "r"(dst), "l"(tmap), "r"(c0), "r"(c1), "r"(c2), "r"(bar) : "memory");
}

template <int R, int CK, int WC, int NSW, int NSX, int NT, bool DIAG>
__global__ void __launch_bounds__((WC + 1) * 32)
t12e_kernel(const __grid_constant__ CUtensorMap xmap, int m, const uint8_t* __restrict__ Wt, int eb,
            const int2* __restrict__ esc_tab, __nv_bfloat16* __restrict__ out, int ldo, int K, unsigned long long* __restrict__ diag)
{
    constexpr int KC = 128 * CK, NBOX = KC / 64, BOXB = 8 * NT * 128;
    constexpr int WST = (R / 16) * CK * 3072, XST = NBOX * BOXB;
    constexpr int U = (R / 8) * CK, UPW = U / WC, WPR = WC / (R / 16);
    static_assert(U % WC == 0 && (2 * CK) % UPW == 0 && WC % (R / 16) == 0, "t12e: warp / unit split");
    constexpr int CH = NT == 1 ? 2 : 1;
    constexpr int NTC = WC * 32, TPR = NTC / R;
    static_assert(NTC % R == 0 && R <= 32 && NSX >= 2, "t12e: threads per row / x ring");
    constexpr int TPT = (8 * NT + TPR - 1) / TPR;
    static_assert(NSW * WST >= WC * 16 * (8 * NT + 1) * 4, "t12e: red must fit in the weight ring");
    extern __shared__ __align__(1024) uint8_t t12e_smem[];
    __shared__ __align__(8) unsigned long long full[NSW], xfull[NSX];
    __shared__ float esc_s[R][8 * NT];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int tile = blockIdx.x, NC = K / KC, ntb = (m + 7) >> 3;
    unsigned long long t0 = 0, t1 = 0;
    if constexpr (DIAG) t0 = gtimer();
    const uint32_t sw0 = smem_u32(t12e_smem);
    const uint32_t sw = (sw0 + 1023u) & ~1023u;
    const uint32_t sx = sw + NSW * WST;
    uint8_t* smem = t12e_smem + (sw - sw0);
    float (*red)[16][8 * NT + 1] = reinterpret_cast<float (*)[16][8 * NT + 1]>(smem);
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < NSW; ++s) mbar_init(smem_u32(&full[s]), 1);
        #pragma unroll
        for (int s = 0; s < NSX; ++s) mbar_init(smem_u32(&xfull[s]), 1);
        fence_mbar_init();
    }
    __syncthreads();
    if (warp == WC) {
        int2 e0 = esc_tab[tile * 64 + lane], e1 = esc_tab[tile * 64 + 32 + lane];
        for (int i = lane; i < R * 8 * NT; i += 32) (&esc_s[0][0])[i] = 0.f;
        const uint8_t* wsrc = Wt + (size_t)tile * NC * WST;
        auto issue_x = [&](int c, int st) {
            const uint32_t bar = smem_u32(&xfull[st]);
            mbar_expect_tx(bar, XST);
            tma_3d_g2s(sx + st * XST, &xmap, 0, 0, c * NBOX, bar);
        };
        auto issue_w = [&](int c, int st) {
            const uint32_t bar = smem_u32(&full[st]);
            mbar_expect_tx(bar, WST);
            bulk_g2s(sw + st * WST, wsrc + (size_t)c * WST, WST, bar);
        };
        if (lane == 0) {
            #pragma unroll
            for (int s = 0; s < NSX; ++s) if (s < NC) issue_x(s, s);
            #pragma unroll
            for (int s = 0; s < NSW; ++s) if (s < NC) issue_w(s, s);
        }
        int ep = 0;
        for (int c = 0; c < NC; ++c) {
            __syncthreads();                                            // (B_c)
            if (c > 0 && lane == 0) {
                if (c - 1 + NSX < NC) issue_x(c - 1 + NSX, (c - 1) % NSX);
                if (c - 1 + NSW < NC) issue_w(c - 1 + NSW, (c - 1) % NSW);
            }
            const int kend = (c + 1) * KC;
            bool waited = false;
            while (ep < 64) {
                const int w0 = __shfl_sync(0xffffffffu, ep < 32 ? e0.x : e1.x, ep & 31);
                const int kk = w0 & 0xFFFF;
                if (kk >= kend) break;
                if (!waited) { mbar_wait(smem_u32(&xfull[c % NSX]), (uint32_t)((c / NSX) & 1)); waited = true; }
                const int w1 = __shfl_sync(0xffffffffu, ep < 32 ? e0.y : e1.y, ep & 31);
                const int row = w0 >> 16, kc = kk - c * KC;
                if (lane < m) {
                    const uint8_t* xs = smem + NSW * WST + (c % NSX) * XST + (kc >> 6) * BOXB + lane * 128 + ((((kc & 63) >> 3) ^ (lane & 7)) << 4) + (kc & 7) * 2;
                    esc_s[row][lane] += __int_as_float(w1) * __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(xs));
                }
                ++ep;
            }
        }
    } else {
        const int r_w = warp / WPR;
        const uint32_t hb = (uint32_t)(eb >> 1);
        const uint32_t lut0 = hb | (hb + 1) << 8 | (hb + 2) << 16 | (hb + 3) << 24;
        const uint32_t lut1 = (hb + 4) | (hb + 5) << 8 | (hb + 6) << 16 | (hb + 7) << 24;
        int lo_off[UPW], nb_off[UPW], xb[UPW];
        #pragma unroll
        for (int j = 0; j < UPW; ++j) {
            const int u = warp * UPW + j, ck = (u % (2 * CK)) >> 1, h = u & 1;
            const int woff = (r_w * CK + ck) * 3072;
            lo_off[j] = woff + (2 * h) * 512 + lane * 16;
            nb_off[j] = woff + 2048 + h * 512 + lane * 16;
            xb[j] = (ck * 2 + h) * BOXB;
        }
        uint32_t rowoff[NT], swz[2];
        #pragma unroll
        for (int nbk = 0; nbk < NT; ++nbk) rowoff[nbk] = (uint32_t)((nbk * 8 + (lane & 7)) * 128);
        #pragma unroll
        for (int ps = 0; ps < 2; ++ps) swz[ps] = (uint32_t)((((ps * 4) + (lane >> 3)) ^ (lane & 7)) << 4);
        float acc[NT * CH][4];
        #pragma unroll
        for (int n = 0; n < NT * CH; ++n)
            #pragma unroll
            for (int q = 0; q < 4; ++q) acc[n][q] = 0.f;
        for (int c = 0; c < NC; ++c) {
            const int swi = c % NSW, sxi = c % NSX;
            __syncthreads();                                            // (B_c)
            mbar_wait(smem_u32(&xfull[sxi]), (uint32_t)((c / NSX) & 1));
            mbar_wait(smem_u32(&full[swi]), (uint32_t)((c / NSW) & 1));
            if constexpr (DIAG) { if (c == 0) t1 = gtimer(); }
            const uint8_t* ws = smem + swi * WST;
            const uint32_t xs = sx + sxi * XST;
            #pragma unroll
            for (int j = 0; j < UPW; ++j) {
                const uint4 lo0 = *reinterpret_cast<const uint4*>(ws + lo_off[j]);
                const uint4 lo1 = *reinterpret_cast<const uint4*>(ws + lo_off[j] + 512);
                const uint4 nb = *reinterpret_cast<const uint4*>(ws + nb_off[j]);
                #pragma unroll
                for (int ps = 0; ps < 2; ++ps) {
                    uint32_t b[NT][4];
                    #pragma unroll
                    for (int nbk = 0; nbk < NT; ++nbk) {
                        if (nbk < ntb)
                            ldsm_x4_nv(b[nbk][0], b[nbk][1], b[nbk][2], b[nbk][3], xs + xb[j] + rowoff[nbk] + swz[ps]);
                    }
                    const uint4 lo = ps == 0 ? lo0 : lo1;
                    #pragma unroll
                    for (int q = 0; q < 2; ++q) {
                        const uint32_t L0 = q == 0 ? lo.x : lo.z, L1 = q == 0 ? lo.y : lo.w;
                        const int si = 2 * ps + q;
                        const uint32_t Nw = si == 0 ? nb.x : si == 1 ? nb.y : si == 2 ? nb.z : nb.w;
                        uint32_t a0, a1, a2, a3;
                        decode4(L0, Nw & 0xFFFFu, lut0, lut1, a0, a1);
                        decode4(L1, Nw >> 16, lut0, lut1, a2, a3);
                        #pragma unroll
                        for (int nbk = 0; nbk < NT; ++nbk) {
                            if (nbk < ntb) mma16816_nv(acc[nbk * CH + (CH == 2 ? q : 0)], a0, a1, a2, a3, b[nbk][2 * q], b[nbk][2 * q + 1]);
                        }
                    }
                }
            }
        }
        __syncthreads();                                                // (C)
        #pragma unroll
        for (int nbk = 0; nbk < NT; ++nbk)
            #pragma unroll
            for (int hr = 0; hr < 2; ++hr)
                #pragma unroll
                for (int cc = 0; cc < 2; ++cc)
                    red[warp][g + 8 * hr][nbk * 8 + 2 * t + cc] = acc[nbk * CH][2 * hr + cc] + (CH == 2 ? acc[nbk * CH + 1][2 * hr + cc] : 0.f);
    }
    if (warp == WC) __syncthreads();                                    // (C)
    unsigned long long t2 = 0;
    if constexpr (DIAG) t2 = gtimer();
    __syncthreads();
    if (warp < WC) {
        const int rr = tid % R, i0 = tid / R, rt = rr >> 4, rl = rr & 15;
        #pragma unroll
        for (int j = 0; j < TPT; ++j) {
            const int i = i0 + j * TPR;
            if (i < m) {
                float v = esc_s[rr][i];
                #pragma unroll
                for (int w2 = 0; w2 < WPR; ++w2) v += red[rt * WPR + w2][rl][i];
                out[(size_t)i * ldo + tile * R + rr] = __float2bfloat16(v);
            }
        }
    }
    if constexpr (DIAG) {
        __syncthreads();
        if (tid == 0) {
            const unsigned long long t3 = gtimer();
            if (tile == 0) { diag[0] = t0; diag[1] = t1; diag[2] = t2; diag[3] = t3; }
            atomicMin(diag + 4, t0); atomicMax(diag + 5, t3);
            if (tile == gridDim.x - 1) { diag[6] = t0; diag[7] = t3; }
        }
    }
}

template <int R, int CK, int WC, int NSW, int NSX, int NT, bool DIAG>
void launch_t12e(const CUtensorMap& xmap, int m, const uint8_t* Wt, int eb, const int2* etab,
                 __nv_bfloat16* out, int ldo, int N, int K, unsigned long long* diag, cudaStream_t stream)
{
    constexpr int smem = t12d_smem_bytes(R, CK, NSW, NSX, NT);
    static bool attr = false;
    if (!attr) { cudaFuncSetAttribute(t12e_kernel<R, CK, WC, NSW, NSX, NT, DIAG>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
    t12e_kernel<R, CK, WC, NSW, NSX, NT, DIAG><<<N / R, (WC + 1) * 32, smem, stream>>>(xmap, m, Wt, eb, etab, out, ldo, K, diag);
}
// == END t12 device ==

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
        case 51208: stream_kernel<8, 512><<<grid, 512, 0, stream>>>(p, n_vec, o); break;
        case 25704: stream_warp_kernel<4, 256><<<grid, 256, 0, stream>>>(p, n_vec, o); break;   // warp-private chunks: unroll + 100
        case 25708: stream_warp_kernel<8, 256><<<grid, 256, 0, stream>>>(p, n_vec, o); break;
        case 51308: stream_warp_kernel<8, 512><<<grid, 512, 0, stream>>>(p, n_vec, o); break;
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
        GEMV_CASE(1, 1, 2, 1, false)
        GEMV_CASE(1, 1, 4, 1, false)
        GEMV_CASE(1, 1, 8, 1, false)
        GEMV_CASE(1, 1, 16, 1, false)
        GEMV_CASE(1, 2, 4, 1, false)
        GEMV_CASE(1, 2, 8, 1, false)
        GEMV_CASE(1, 4, 4, 1, false)
        GEMV_CASE(1, 4, 8, 1, false)
        GEMV_CASE(1, 1, 4, 2, false)
        GEMV_CASE(1, 1, 8, 2, false)
        GEMV_CASE(1, 2, 2, 2, false)
        GEMV_CASE(1, 2, 4, 2, false)
        GEMV_CASE(1, 2, 8, 2, false)
        GEMV_CASE(1, 4, 2, 2, false)
        GEMV_CASE(1, 1, 4, 4, false)
        GEMV_CASE(1, 2, 2, 4, false)
        GEMV_CASE(1, 2, 4, 4, false)
        GEMV_CASE(1, 1, 2, 8, false)
        GEMV_CASE(1, 1, 4, 8, true)
        GEMV_CASE(1, 1, 2, 8, true)
        GEMV_CASE(1, 2, 2, 8, true)
        GEMV_CASE(1, 1, 8, 4, true)
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
        MMA_CASE(1, 1, 4, 4)
        MMA_CASE(2, 2, 4, 8)
        MMA_CASE(2, 2, 4, 4)
        default: TORCH_CHECK(false, "unsupported mma config");
    }
}
#define TGEMV_CASE(B_, W_, U_, S_) \
    case (B_ * 10000 + W_ * 100 + U_ * 10 + S_): launch_tgemv<B_, W_, U_, S_>(Xp, Wp, ob, part.data_ptr<float>(), cnt.data_ptr<int>(), N, K, stream); break;

void tgemv(torch::Tensor X, torch::Tensor Wt, torch::Tensor out, torch::Tensor part, torch::Tensor cnt, int64_t bn, int64_t warps, int64_t unroll, int64_t sk)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    const int K = (int)X.size(1), N = (int)out.size(1);
    const __nv_bfloat16* Xp = reinterpret_cast<const __nv_bfloat16*>(X.data_ptr());
    const __nv_bfloat16* Wp = reinterpret_cast<const __nv_bfloat16*>(Wt.data_ptr());
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    const int key = (int)bn * 10000 + (int)warps * 100 + (int)unroll * 10 + (int)sk;
    switch (key) {
        TGEMV_CASE(32, 8, 2, 1)
        TGEMV_CASE(32, 8, 4, 1)
        TGEMV_CASE(32, 8, 2, 2)
        TGEMV_CASE(32, 8, 4, 2)
        TGEMV_CASE(32, 8, 2, 4)
        TGEMV_CASE(32, 8, 4, 4)
        TGEMV_CASE(32, 16, 4, 1)
        TGEMV_CASE(32, 16, 8, 1)
        TGEMV_CASE(32, 16, 4, 2)
        TGEMV_CASE(32, 16, 8, 2)
        TGEMV_CASE(32, 4, 2, 1)
        TGEMV_CASE(32, 4, 2, 2)
        TGEMV_CASE(32, 4, 2, 4)
        default: TORCH_CHECK(false, "unsupported tgemv config");
    }
}
#define GEMV12_CASE(W_, U_, S_) \
    case (W_ * 100 + U_ * 10 + S_): launch_gemv12<W_, U_, S_, false>(Xp, LOp, NBp, (int)eb, eo, en, ek, ed, nullptr, nullptr, 0, nullptr, nullptr, nullptr, nullptr, ob, of, part.data_ptr<float>(), cnt.data_ptr<int>(), N, K, stream); break;

void gemv12(torch::Tensor X, torch::Tensor LO, torch::Tensor NB, int64_t eb, torch::Tensor esc_off, torch::Tensor esc_n, torch::Tensor esc_k, torch::Tensor esc_d,
            torch::Tensor out, torch::Tensor part, torch::Tensor cnt, int64_t warps, int64_t unroll, int64_t sk)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    const int K = (int)X.size(1), N = (int)out.size(1);
    const __nv_bfloat16* Xp = reinterpret_cast<const __nv_bfloat16*>(X.data_ptr());
    const uint8_t* LOp = LO.data_ptr<uint8_t>(); const uint8_t* NBp = NB.data_ptr<uint8_t>();
    const int* eo = esc_off.data_ptr<int>(); const int* en = esc_n.data_ptr<int>(); const int* ek = esc_k.data_ptr<int>(); const float* ed = esc_d.data_ptr<float>();
    __nv_bfloat16* ob = out.scalar_type() == at::kBFloat16 ? reinterpret_cast<__nv_bfloat16*>(out.data_ptr()) : nullptr;
    float* of = out.scalar_type() == at::kFloat ? out.data_ptr<float>() : nullptr;
    const int key = (int)warps * 100 + (int)unroll * 10 + (int)sk;
    switch (key) {
        GEMV12_CASE(8, 2, 1)
        GEMV12_CASE(8, 4, 1)
        GEMV12_CASE(8, 2, 2)
        GEMV12_CASE(8, 4, 2)
        GEMV12_CASE(8, 2, 4)
        GEMV12_CASE(8, 1, 4)
        GEMV12_CASE(16, 2, 1)
        GEMV12_CASE(16, 4, 1)
        GEMV12_CASE(16, 2, 2)
        GEMV12_CASE(16, 4, 2)
        GEMV12_CASE(4, 2, 1)
        GEMV12_CASE(4, 2, 2)
        GEMV12_CASE(4, 2, 4)
        GEMV12_CASE(4, 4, 2)
        GEMV12_CASE(8, 1, 8)
        GEMV12_CASE(8, 2, 8)
        GEMV12_CASE(4, 2, 8)
        GEMV12_CASE(16, 1, 8)
        GEMV12_CASE(8, 1, 16)
        GEMV12_CASE(4, 1, 16)
        GEMV12_CASE(8, 1, 1)
        GEMV12_CASE(16, 1, 1)
        GEMV12_CASE(4, 1, 1)
        GEMV12_CASE(8, 1, 2)
        GEMV12_CASE(16, 1, 2)
        GEMV12_CASE(4, 1, 2)
        GEMV12_CASE(16, 1, 4)
        GEMV12_CASE(4, 1, 4)
        GEMV12_CASE(4, 1, 8)
        GEMV12_CASE(16, 1, 16)
        default: TORCH_CHECK(false, "unsupported gemv12 config");
    }
}

#define GEMV12R_CASE(W_, U_, S_) \
    case (W_ * 100 + U_ * 10 + S_): launch_gemv12<W_, U_, S_, true>(Xp, LOp, NBp, (int)eb, eo, en, ek, ed, LOp2, NBp2, (int)eb2, eo2, en2, ek2, ed2, nullptr, of, part.data_ptr<float>(), cnt.data_ptr<int>(), N, K, stream); break;

// the router: two packed planes (hi, lo), fp32 accumulation and output
void gemv12r(torch::Tensor X, torch::Tensor LO, torch::Tensor NB, int64_t eb, torch::Tensor esc_off, torch::Tensor esc_n, torch::Tensor esc_k, torch::Tensor esc_d,
             torch::Tensor LO2, torch::Tensor NB2, int64_t eb2, torch::Tensor esc_off2, torch::Tensor esc_n2, torch::Tensor esc_k2, torch::Tensor esc_d2,
             torch::Tensor out, torch::Tensor part, torch::Tensor cnt, int64_t warps, int64_t unroll, int64_t sk)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    const int K = (int)X.size(1), N = (int)out.size(1);
    TORCH_CHECK(out.scalar_type() == at::kFloat, "gemv12r: fp32 output");
    const __nv_bfloat16* Xp = reinterpret_cast<const __nv_bfloat16*>(X.data_ptr());
    const uint8_t* LOp = LO.data_ptr<uint8_t>(); const uint8_t* NBp = NB.data_ptr<uint8_t>();
    const int* eo = esc_off.data_ptr<int>(); const int* en = esc_n.data_ptr<int>(); const int* ek = esc_k.data_ptr<int>(); const float* ed = esc_d.data_ptr<float>();
    const uint8_t* LOp2 = LO2.data_ptr<uint8_t>(); const uint8_t* NBp2 = NB2.data_ptr<uint8_t>();
    const int* eo2 = esc_off2.data_ptr<int>(); const int* en2 = esc_n2.data_ptr<int>(); const int* ek2 = esc_k2.data_ptr<int>(); const float* ed2 = esc_d2.data_ptr<float>();
    float* of = out.data_ptr<float>();
    const int key = (int)warps * 100 + (int)unroll * 10 + (int)sk;
    switch (key) {
        GEMV12R_CASE(8, 1, 16)
        GEMV12R_CASE(8, 2, 16)
        GEMV12R_CASE(16, 1, 16)
        GEMV12R_CASE(16, 2, 16)
        GEMV12R_CASE(4, 1, 16)
        GEMV12R_CASE(4, 2, 16)
        GEMV12R_CASE(8, 1, 32)
        GEMV12R_CASE(16, 1, 32)
        GEMV12R_CASE(4, 1, 32)
        GEMV12R_CASE(8, 1, 8)
        GEMV12R_CASE(8, 2, 8)
        default: TORCH_CHECK(false, "unsupported gemv12r config");
    }
}
#define MMA12_CASE(T_, R_, U_, S_) \
    case (T_ * 1000 + R_ * 100 + U_ * 10 + S_): launch_mma12<T_, R_, U_, S_>(Xp, ldx, m, LOp, NBp, (int)eb, eo, en, ek, ed, ob, ldo, part.data_ptr<float>(), cnt.data_ptr<int>(), N, K, stream); break;

void mma12(torch::Tensor X, torch::Tensor LO, torch::Tensor NB, int64_t eb, torch::Tensor esc_off, torch::Tensor esc_n, torch::Tensor esc_k, torch::Tensor esc_d,
           torch::Tensor out, torch::Tensor part, torch::Tensor cnt, int64_t mt, int64_t rpw, int64_t unroll, int64_t sk)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    const int m = (int)X.size(0), K = (int)X.size(1), N = (int)out.size(1);
    const __nv_bfloat16* Xp = reinterpret_cast<const __nv_bfloat16*>(X.data_ptr());
    const uint8_t* LOp = LO.data_ptr<uint8_t>(); const uint8_t* NBp = NB.data_ptr<uint8_t>();
    const int* eo = esc_off.data_ptr<int>(); const int* en = esc_n.data_ptr<int>(); const int* ek = esc_k.data_ptr<int>(); const float* ed = esc_d.data_ptr<float>();
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    const int ldx = (int)X.stride(0), ldo = (int)out.stride(0);
    const int key = (int)mt * 1000 + (int)rpw * 100 + (int)unroll * 10 + (int)sk;
    switch (key) {
        MMA12_CASE(1, 4, 2, 1)
        MMA12_CASE(1, 4, 1, 2)
        MMA12_CASE(1, 4, 2, 2)
        MMA12_CASE(1, 2, 2, 1)
        MMA12_CASE(1, 2, 4, 1)
        MMA12_CASE(1, 2, 2, 2)
        MMA12_CASE(1, 2, 2, 4)
        MMA12_CASE(1, 1, 4, 1)
        MMA12_CASE(1, 1, 4, 2)
        MMA12_CASE(1, 1, 8, 1)
        MMA12_CASE(2, 4, 1, 1)
        MMA12_CASE(2, 4, 1, 2)
        MMA12_CASE(2, 2, 2, 1)
        MMA12_CASE(2, 2, 2, 2)
        MMA12_CASE(2, 2, 1, 4)
        MMA12_CASE(2, 1, 4, 1)
        MMA12_CASE(2, 1, 2, 2)
        default: TORCH_CHECK(false, "unsupported mma12 config");
    }
}
#define MMA12CP_CASE(T_, R_, U_, S_) \
    case (T_ * 1000 + R_ * 100 + U_ * 10 + S_): launch_mma12cp<T_, R_, U_, S_>(Xp, ldx, m, LOp, NBp, (int)eb, eo, en, ek, ed, ob, ldo, part.data_ptr<float>(), cnt.data_ptr<int>(), N, K, stream); break;

void mma12cp(torch::Tensor X, torch::Tensor LO, torch::Tensor NB, int64_t eb, torch::Tensor esc_off, torch::Tensor esc_n, torch::Tensor esc_k, torch::Tensor esc_d,
             torch::Tensor out, torch::Tensor part, torch::Tensor cnt, int64_t mt, int64_t rpw, int64_t nstage, int64_t sk)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    const int m = (int)X.size(0), K = (int)X.size(1), N = (int)out.size(1);
    const __nv_bfloat16* Xp = reinterpret_cast<const __nv_bfloat16*>(X.data_ptr());
    const uint8_t* LOp = LO.data_ptr<uint8_t>(); const uint8_t* NBp = NB.data_ptr<uint8_t>();
    const int* eo = esc_off.data_ptr<int>(); const int* en = esc_n.data_ptr<int>(); const int* ek = esc_k.data_ptr<int>(); const float* ed = esc_d.data_ptr<float>();
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    const int ldx = (int)X.stride(0), ldo = (int)out.stride(0);
    const int key = (int)mt * 1000 + (int)rpw * 100 + (int)nstage * 10 + (int)sk;
    switch (key) {
        MMA12CP_CASE(1, 4, 4, 1)
        MMA12CP_CASE(1, 4, 3, 1)
        MMA12CP_CASE(1, 4, 3, 2)
        MMA12CP_CASE(1, 2, 4, 1)
        MMA12CP_CASE(1, 2, 3, 1)
        MMA12CP_CASE(1, 2, 3, 2)
        MMA12CP_CASE(1, 2, 4, 2)
        MMA12CP_CASE(1, 1, 4, 1)
        MMA12CP_CASE(1, 1, 4, 2)
        MMA12CP_CASE(2, 4, 2, 1)
        MMA12CP_CASE(2, 2, 3, 1)
        MMA12CP_CASE(2, 2, 2, 2)
        MMA12CP_CASE(2, 1, 3, 1)
        MMA12CP_CASE(2, 1, 3, 2)
        default: TORCH_CHECK(false, "unsupported mma12cp config");
    }
}

void w12(torch::Tensor X, torch::Tensor W, int64_t eb, torch::Tensor esc_off, torch::Tensor esc_n, torch::Tensor esc_k, torch::Tensor esc_d,
         torch::Tensor out, torch::Tensor part, torch::Tensor cnt, int64_t m, int64_t n, int64_t k, int64_t nt, int64_t nstage, int64_t sk, int64_t mode)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(n % 64 == 0 && k % 128 == 0 && (k / 128) % sk == 0 && m <= 8 * nt && m <= 32, "w12: shape/config mismatch");
    TORCH_CHECK(X.stride(1) == 1 && (X.stride(0) * 2) % 16 == 0 && (reinterpret_cast<uintptr_t>(X.data_ptr()) & 15) == 0, "w12: x must be 16 B aligned rows");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(W.data_ptr()) & 127) == 0, "w12: weight layout must be 128 B aligned");
    TORCH_CHECK(sk * m * n <= part.numel() && n / 64 <= cnt.numel(), "w12: workspace too small");
    const __nv_bfloat16* Xp = reinterpret_cast<const __nv_bfloat16*>(X.data_ptr());
    const uint8_t* Wp = reinterpret_cast<const uint8_t*>(W.data_ptr());
    float* pp = part.data_ptr<float>();
    const int ldx = (int)X.stride(0);
    W12Args a{esc_off.data_ptr<int>(), esc_n.data_ptr<int>(), esc_k.data_ptr<int>(), esc_d.data_ptr<float>(),
              reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (int)out.stride(0), cnt.data_ptr<int>(), (int)(mode >= 4 ? mode : 0)};
    if (mode == 3) {   // loads-only ablation of the GEMM (4 stages, K split)
        if (nt == 1) launch_w12<1, 4, true, 1>(Xp, ldx, (int)m, Wp, (int)eb, pp, (int)n, (int)k, (int)sk, a, stream);
        else if (nt == 2) launch_w12<2, 4, true, 1>(Xp, ldx, (int)m, Wp, (int)eb, pp, (int)n, (int)k, (int)sk, a, stream);
        else launch_w12<4, 4, true, 1>(Xp, ldx, (int)m, Wp, (int)eb, pp, (int)n, (int)k, (int)sk, a, stream);
        return;
    }
    if (mode == 2) {   // GEMM only (no epilogue; output not written), 4 stages, K split
        if (nt == 1) launch_w12<1, 4, true, 2>(Xp, ldx, (int)m, Wp, (int)eb, pp, (int)n, (int)k, (int)sk, a, stream);
        else if (nt == 2) launch_w12<2, 4, true, 2>(Xp, ldx, (int)m, Wp, (int)eb, pp, (int)n, (int)k, (int)sk, a, stream);
        else launch_w12<4, 4, true, 2>(Xp, ldx, (int)m, Wp, (int)eb, pp, (int)n, (int)k, (int)sk, a, stream);
        return;
    }
#define OKC_W12(T, NS, SK) launch_w12<T, NS, SK>(Xp, ldx, (int)m, Wp, (int)eb, pp, (int)n, (int)k, SK ? (int)sk : 1, a, stream)
    const int key = (int)nt * 100 + (int)nstage * 10 + (sk > 1 ? 1 : 0);
    switch (key) {
        case 140: OKC_W12(1, 4, false); break;  case 141: OKC_W12(1, 4, true); break;
        case 180: OKC_W12(1, 8, false); break;  case 181: OKC_W12(1, 8, true); break;
        case 240: OKC_W12(2, 4, false); break;  case 241: OKC_W12(2, 4, true); break;
        case 220: OKC_W12(2, 2, false); break;  case 221: OKC_W12(2, 2, true); break;
        case 440: OKC_W12(4, 4, false); break;  case 441: OKC_W12(4, 4, true); break;
        case 420: OKC_W12(4, 2, false); break;  case 421: OKC_W12(4, 2, true); break;
        default: TORCH_CHECK(false, "unsupported w12 config (nt, nstage, sk)");
    }
#undef OKC_W12
}

void w12b(torch::Tensor X, torch::Tensor W, int64_t eb, torch::Tensor esc_off, torch::Tensor esc_n, torch::Tensor esc_k, torch::Tensor esc_d,
          torch::Tensor out, int64_t m, int64_t n, int64_t k, int64_t nt, int64_t warps, int64_t nst)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(n % 64 == 0 && k % 128 == 0 && (k / 128) % warps == 0 && (k / 128) / warps >= 1 && m <= 8 * nt && m <= 32, "w12b: shape/config mismatch");
    TORCH_CHECK(X.stride(1) == 1 && (X.stride(0) * 2) % 16 == 0 && (reinterpret_cast<uintptr_t>(X.data_ptr()) & 15) == 0, "w12b: x must be 16 B aligned rows");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(W.data_ptr()) & 127) == 0, "w12b: weight layout must be 128 B aligned");
    const __nv_bfloat16* Xp = reinterpret_cast<const __nv_bfloat16*>(X.data_ptr());
    const uint8_t* Wp = reinterpret_cast<const uint8_t*>(W.data_ptr());
    __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    const int* eo = esc_off.data_ptr<int>(); const int* en = esc_n.data_ptr<int>(); const int* ek = esc_k.data_ptr<int>(); const float* ed = esc_d.data_ptr<float>();
    const int ldx = (int)X.stride(0), ldo = (int)out.stride(0);
#define OKC_B(T, WW, NS) launch_w12b<T, WW, NS>(Xp, ldx, (int)m, Wp, (int)eb, eo, en, ek, ed, op, ldo, (int)n, (int)k, stream)
    const int key = (int)nt * 1000 + (int)warps * 10 + (int)nst;
    switch (key) {
        case 1042: OKC_B(1, 4, 2); break;   case 1043: OKC_B(1, 4, 3); break;
        case 1082: OKC_B(1, 8, 2); break;   case 1083: OKC_B(1, 8, 3); break;
        case 1162: OKC_B(1, 16, 2); break;  case 1163: OKC_B(1, 16, 3); break;
        case 2042: OKC_B(2, 4, 2); break;   case 2082: OKC_B(2, 8, 2); break;  case 2083: OKC_B(2, 8, 3); break;
        case 2162: OKC_B(2, 16, 2); break;
        case 4082: OKC_B(4, 8, 2); break;   case 4042: OKC_B(4, 4, 2); break;
        default: TORCH_CHECK(false, "unsupported w12b config (nt, warps, nst)");
    }
#undef OKC_B
}

void w12m(torch::Tensor X, torch::Tensor W, int64_t eb, torch::Tensor esc_roff, torch::Tensor esc_k, torch::Tensor esc_d,
          torch::Tensor out, torch::Tensor part, int64_t m, int64_t n, int64_t k, int64_t mw, int64_t nt, int64_t nstage, int64_t sk, bool pdl, int64_t mode)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(n % (128 * mw) == 0 && k % 128 == 0 && (k / 128) % sk == 0 && (k / 128) / sk >= 1 && m <= 8 * nt && m >= 1, "w12m: shape/config mismatch");
    TORCH_CHECK(X.stride(1) == 1 && (X.stride(0) * 2) % 16 == 0 && (reinterpret_cast<uintptr_t>(X.data_ptr()) & 15) == 0, "w12m: x must be 16 B aligned rows");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(W.data_ptr()) & 127) == 0 && W.is_contiguous(), "w12m: weight layout must be 128 B aligned");
    TORCH_CHECK(sk * m * n <= part.numel() && (reinterpret_cast<uintptr_t>(part.data_ptr()) & 15) == 0, "w12m: workspace");
    TORCH_CHECK(out.stride(1) == 1 && out.stride(0) % 2 == 0 && n % 2 == 0 && esc_roff.numel() >= n + 1, "w12m: output / escapes");
    const __nv_bfloat16* Xp = reinterpret_cast<const __nv_bfloat16*>(X.data_ptr());
    const uint8_t* Wp = reinterpret_cast<const uint8_t*>(W.data_ptr());
    float* pp = part.data_ptr<float>();
    const int ldx = (int)X.stride(0), ldo = (int)out.stride(0);
#define OKC_W12M(MWv, NTv, NS) launch_w12m<MWv, NTv, NS>(Xp, ldx, (int)m, Wp, (int)eb, pp, (int)n, (int)k, (int)sk, stream)
    const int key = (int)mw * 1000 + (int)nt * 10 + (int)nstage;
    if (mode == 1) {   // bench ablation: loads only (1/16/3)
        launch_w12m<1, 16, 3, 1>(Xp, ldx, (int)m, Wp, (int)eb, pp, (int)n, (int)k, (int)sk, stream);
    } else {
    switch (key) {
        case 1163: OKC_W12M(1, 16, 3); break;   case 1162: OKC_W12M(1, 16, 2); break;
        case 2162: OKC_W12M(2, 16, 2); break;   case 1322: OKC_W12M(1, 32, 2); break;
        default: TORCH_CHECK(false, "unsupported w12m config (mw, nt, nstage)");
    }
    }
#undef OKC_W12M
    if (mode == 2) return;   // bench ablation: GEMM only (no reduce launch; output not written)
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)((n / 2 + 255) / 256), (unsigned)m);
    cfg.blockDim = dim3(256);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = pdl ? 1 : 0;
    const cudaError_t err = cudaLaunchKernelEx(&cfg, w12m_reduce_kernel, (const float*)pp, (int)sk, (int)m, (int)n,
        esc_roff.data_ptr<int>(), esc_k.data_ptr<int>(), esc_d.data_ptr<float>(), Xp, ldx,
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), ldo);
    TORCH_CHECK(err == cudaSuccess, "w12m reduce launch failed: ", cudaGetErrorString(err));
}

// == BEGIN t12 host ==
// x tensor map for the t12d kernel: 2D bf16 [m rows][K], box [8*nt rows][64 k], 128B swizzle, OOB rows zero-filled
typedef CUresult (*okc_encode_fn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                  const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                  CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static okc_encode_fn okc_get_encode()
{
    static okc_encode_fn fn = nullptr;
    if (!fn) {
        void* ptr = nullptr;
        cudaDriverEntryPointQueryResult q;
        const cudaError_t e = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &ptr, 12000, cudaEnableDefault, &q);
        TORCH_CHECK(e == cudaSuccess && ptr != nullptr && q == cudaDriverEntryPointSuccess, "t12d: cuTensorMapEncodeTiled unavailable");
        fn = reinterpret_cast<okc_encode_fn>(ptr);
    }
    return fn;
}
static CUtensorMap okc_x_map(const void* x, int m, int K, int ldx, int box_rows)
{
    CUtensorMap map;
    cuuint64_t dims[2] = {(cuuint64_t)K, (cuuint64_t)m};
    cuuint64_t strides[1] = {(cuuint64_t)ldx * 2};
    cuuint32_t box[2] = {64u, (cuuint32_t)box_rows};
    cuuint32_t estr[2] = {1u, 1u};
    const CUresult r = okc_get_encode()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(x), dims, strides, box, estr,
                                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "t12d: cuTensorMapEncodeTiled failed: ", (int)r);
    return map;
}
// 3D x map for t12e: {64 k, m rows, K/64 groups}, strides {ldx*2, 128 B}, box {64, box_rows, groups}: one op per chunk
static CUtensorMap okc_x_map3(const void* x, int m, int K, int ldx, int box_rows, int groups)
{
    CUtensorMap map;
    cuuint64_t dims[3] = {64ull, (cuuint64_t)m, (cuuint64_t)(K / 64)};
    cuuint64_t strides[2] = {(cuuint64_t)ldx * 2, 128ull};
    cuuint32_t box[3] = {64u, (cuuint32_t)box_rows, (cuuint32_t)groups};
    cuuint32_t estr[3] = {1u, 1u, 1u};
    const CUresult r = okc_get_encode()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(x), dims, strides, box, estr,
                                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "t12e: cuTensorMapEncodeTiled (3D) failed: ", (int)r);
    return map;
}

void bstream(torch::Tensor buf, torch::Tensor out, int64_t grid, int64_t nst, int64_t sb, bool bulk, int64_t smem_alloc)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    const uint8_t* p = reinterpret_cast<const uint8_t*>(buf.data_ptr());
    TORCH_CHECK((reinterpret_cast<uintptr_t>(p) & 127) == 0 && buf.is_contiguous(), "bstream: buffer must be 128 B aligned");
    const long nbytes = (long)(buf.numel() * buf.element_size());
    uint32_t* o = reinterpret_cast<uint32_t*>(out.data_ptr());
    TORCH_CHECK(out.numel() >= grid * 4, "bstream: out too small");
    const int smem = (int)std::max<int64_t>(smem_alloc, nst * sb);
#define BS_CASE(NS_, SB_) \
    if (nst == NS_ && sb == SB_) { \
        if (bulk) { static bool a1 = false; if (!a1) { cudaFuncSetAttribute(bstream_kernel<NS_, SB_, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, 232448 - 4096); a1 = true; } \
                    bstream_kernel<NS_, SB_, true><<<(unsigned)grid, 128, smem, stream>>>(p, nbytes, o); } \
        else { static bool a0 = false; if (!a0) { cudaFuncSetAttribute(bstream_kernel<NS_, SB_, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, 232448 - 4096); a0 = true; } \
               bstream_kernel<NS_, SB_, false><<<(unsigned)grid, 128, smem, stream>>>(p, nbytes, o); } \
        return; }
    BS_CASE(2, 4096) BS_CASE(4, 4096) BS_CASE(6, 4096) BS_CASE(8, 4096)
    BS_CASE(2, 8192) BS_CASE(4, 8192) BS_CASE(6, 8192) BS_CASE(8, 8192)
    BS_CASE(2, 16384) BS_CASE(4, 16384) BS_CASE(6, 16384) BS_CASE(8, 16384)
    BS_CASE(2, 32768) BS_CASE(4, 32768) BS_CASE(6, 32768)
    BS_CASE(12, 8192) BS_CASE(16, 4096)
#undef BS_CASE
    TORCH_CHECK(false, "bstream: unsupported (nst, sb)");
}

void t12(torch::Tensor X, torch::Tensor Wt, int64_t eb, torch::Tensor esc_roff, torch::Tensor esc_k, torch::Tensor esc_d,
         torch::Tensor out, torch::Tensor diag, int64_t n, int64_t k, int64_t r, int64_t ck, int64_t w, int64_t nsw, int64_t nsx, int64_t nt, bool diag_on, int64_t variant, torch::Tensor etab)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    const int m = (int)X.size(0);
    TORCH_CHECK(1 <= m && m <= 32 && m <= 8 * nt && n % r == 0 && k % (128 * ck) == 0 && nsx <= nsw, "t12: shape/config mismatch");
    TORCH_CHECK(X.size(1) == k && X.stride(1) == 1 && (X.stride(0) * 2) % 16 == 0 && (reinterpret_cast<uintptr_t>(X.data_ptr()) & 15) == 0, "t12: x must be 16 B aligned rows");
    TORCH_CHECK(Wt.is_contiguous() && (reinterpret_cast<uintptr_t>(Wt.data_ptr()) & 127) == 0 && Wt.numel() * Wt.element_size() >= n * k * 3 / 2, "t12: weight layout");
    TORCH_CHECK(out.size(0) == m && out.size(1) == n && out.stride(1) == 1, "t12: output");
    TORCH_CHECK(esc_roff.numel() >= n + 1 && esc_k.numel() >= 1 && esc_d.numel() >= 1, "t12: escapes");
    TORCH_CHECK(diag.numel() >= 8, "t12: diag buffer");
    TORCH_CHECK(variant != 3 || (etab.is_contiguous() && etab.dtype() == torch::kInt32 && etab.numel() >= (n / r) * 128 && (reinterpret_cast<uintptr_t>(etab.data_ptr()) & 7) == 0), "t12: escape table");
    const int2* etp = reinterpret_cast<const int2*>(etab.data_ptr());
    const __nv_bfloat16* Xp = reinterpret_cast<const __nv_bfloat16*>(X.data_ptr());
    const uint8_t* Wp = reinterpret_cast<const uint8_t*>(Wt.data_ptr());
    __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    const int* ero = esc_roff.data_ptr<int>(); const int* ek = esc_k.data_ptr<int>(); const float* ed = esc_d.data_ptr<float>();
    unsigned long long* dg = reinterpret_cast<unsigned long long*>(diag.data_ptr());
    const int ldx = (int)X.stride(0), ldo = (int)out.stride(0);
#define T12C_CASE(R_, C_, W_, SW_, SX_, NT_) \
    if (variant == 3 && r == R_ && ck == C_ && w == W_ && nsw == SW_ && nsx == SX_ && nt == NT_ && !diag_on) { \
        launch_t12c<R_, C_, W_, SW_, SX_, NT_, false>(Xp, ldx, m, Wp, (int)eb, etp, op, ldo, (int)n, (int)k, dg, stream); return; }
#define T12E_CASE(R_, C_, W_, SW_, SX_, NT_) \
    if (variant == 5 && r == R_ && ck == C_ && w == W_ && nsw == SW_ && nsx == SX_ && nt == NT_ && !diag_on) { \
        launch_t12e<R_, C_, W_, SW_, SX_, NT_, false>(xmap, m, Wp, (int)eb, etp, op, ldo, (int)n, (int)k, dg, stream); return; }
#define T12E_CASE_D(R_, C_, W_, SW_, SX_, NT_) \
    if (variant == 5 && r == R_ && ck == C_ && w == W_ && nsw == SW_ && nsx == SX_ && nt == NT_ && diag_on) { \
        launch_t12e<R_, C_, W_, SW_, SX_, NT_, true>(xmap, m, Wp, (int)eb, etp, op, ldo, (int)n, (int)k, dg, stream); return; }
#define T12C_NT(R_, C_, W_, SW_, SX_) T12C_CASE(R_, C_, W_, SW_, SX_, 1) T12C_CASE(R_, C_, W_, SW_, SX_, 2) T12C_CASE(R_, C_, W_, SW_, SX_, 4)
#define T12E_NT(R_, C_, W_, SW_, SX_) T12E_CASE(R_, C_, W_, SW_, SX_, 1) T12E_CASE(R_, C_, W_, SW_, SX_, 2) T12E_CASE(R_, C_, W_, SW_, SX_, 4)
#define T12E_NTD(R_, C_, W_, SW_, SX_) T12E_CASE_D(R_, C_, W_, SW_, SX_, 1) T12E_CASE_D(R_, C_, W_, SW_, SX_, 2) T12E_CASE_D(R_, C_, W_, SW_, SX_, 4)
    CUtensorMap xmap;
    if (variant == 5) xmap = okc_x_map3(X.data_ptr(), m, (int)k, ldx, 8 * (int)nt, 2 * (int)ck);
    T12E_CASE(16, 4, 4, 3, 3, 1)
    T12E_CASE(16, 4, 4, 3, 3, 2)
#undef T12C_NT
#undef T12E_NT
#undef T12E_NTD
#undef T12C_CASE
#undef T12E_CASE
#undef T12E_CASE_D
    TORCH_CHECK(false, "t12: unsupported config (r, ck, w, nsw, nsx, nt, diag)");
}
// == END t12 host ==


"""

PROJ_CPP_SRC = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
void stream_bytes(torch::Tensor buf, torch::Tensor out, int64_t grid, int64_t threads, int64_t unroll);
void gemv(torch::Tensor X, torch::Tensor W, torch::Tensor W2, torch::Tensor out, int64_t mpad, int64_t rpw, int64_t unroll, int64_t ksplit, bool two_w);
void skinny_mma(torch::Tensor X, torch::Tensor W, torch::Tensor out, int64_t mt, int64_t rpw, int64_t unroll, int64_t ksplit);
void tgemv(torch::Tensor X, torch::Tensor Wt, torch::Tensor out, torch::Tensor part, torch::Tensor cnt, int64_t bn, int64_t warps, int64_t unroll, int64_t sk);
void gemv12(torch::Tensor X, torch::Tensor LO, torch::Tensor NB, int64_t eb, torch::Tensor esc_off, torch::Tensor esc_n, torch::Tensor esc_k, torch::Tensor esc_d, torch::Tensor out, torch::Tensor part, torch::Tensor cnt, int64_t warps, int64_t unroll, int64_t sk);
void gemv12r(torch::Tensor X, torch::Tensor LO, torch::Tensor NB, int64_t eb, torch::Tensor esc_off, torch::Tensor esc_n, torch::Tensor esc_k, torch::Tensor esc_d, torch::Tensor LO2, torch::Tensor NB2, int64_t eb2, torch::Tensor esc_off2, torch::Tensor esc_n2, torch::Tensor esc_k2, torch::Tensor esc_d2, torch::Tensor out, torch::Tensor part, torch::Tensor cnt, int64_t warps, int64_t unroll, int64_t sk);
void mma12(torch::Tensor X, torch::Tensor LO, torch::Tensor NB, int64_t eb, torch::Tensor esc_off, torch::Tensor esc_n, torch::Tensor esc_k, torch::Tensor esc_d, torch::Tensor out, torch::Tensor part, torch::Tensor cnt, int64_t mt, int64_t rpw, int64_t unroll, int64_t sk);
void mma12cp(torch::Tensor X, torch::Tensor LO, torch::Tensor NB, int64_t eb, torch::Tensor esc_off, torch::Tensor esc_n, torch::Tensor esc_k, torch::Tensor esc_d, torch::Tensor out, torch::Tensor part, torch::Tensor cnt, int64_t mt, int64_t rpw, int64_t nstage, int64_t sk);
void w12(torch::Tensor X, torch::Tensor W, int64_t eb, torch::Tensor esc_off, torch::Tensor esc_n, torch::Tensor esc_k, torch::Tensor esc_d,
         torch::Tensor out, torch::Tensor part, torch::Tensor cnt, int64_t m, int64_t n, int64_t k, int64_t nt, int64_t nstage, int64_t sk, int64_t mode);
void w12b(torch::Tensor X, torch::Tensor W, int64_t eb, torch::Tensor esc_off, torch::Tensor esc_n, torch::Tensor esc_k, torch::Tensor esc_d,
          torch::Tensor out, int64_t m, int64_t n, int64_t k, int64_t nt, int64_t warps, int64_t nst);
void w12m(torch::Tensor X, torch::Tensor W, int64_t eb, torch::Tensor esc_roff, torch::Tensor esc_k, torch::Tensor esc_d,
          torch::Tensor out, torch::Tensor part, int64_t m, int64_t n, int64_t k, int64_t mw, int64_t nt, int64_t nstage, int64_t sk, bool pdl, int64_t mode);
void bstream(torch::Tensor buf, torch::Tensor out, int64_t grid, int64_t nst, int64_t sb, bool bulk, int64_t smem_alloc);
void t12(torch::Tensor X, torch::Tensor Wt, int64_t eb, torch::Tensor esc_roff, torch::Tensor esc_k, torch::Tensor esc_d,
         torch::Tensor out, torch::Tensor diag, int64_t n, int64_t k, int64_t r, int64_t ck, int64_t w, int64_t nsw, int64_t nsx, int64_t nt, bool diag_on, int64_t variant, torch::Tensor etab);
"""

_PROJ_EXT = None


def _proj_build():
    global _PROJ_EXT
    if _PROJ_EXT is None:
        from torch.utils.cpp_extension import load_inline
        _PROJ_EXT = load_inline(
            name="okc_proj_cuda14", cpp_sources=[PROJ_CPP_SRC],
            cuda_sources=["#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n" + PROJ_CUDA_SRC],
            functions=["stream_bytes", "gemv", "skinny_mma", "tgemv", "gemv12", "gemv12r", "mma12", "mma12cp", "w12", "w12b", "w12m", "bstream", "t12"], extra_cuda_cflags=["-O3", "-arch=sm_90a"], verbose=False)
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


def tgemv_usable(m: int, n: int, k: int, bn: int, warps: int, unroll: int, sk: int) -> bool:
    if m != 1 or n % bn or k % 256 or bn % warps:
        return False
    kt = k // 256
    if kt % sk:
        return False
    return (kt // sk) % unroll == 0


def gemv12_usable(m: int, n: int, k: int, warps: int, unroll: int, sk: int) -> bool:
    if m != 1 or n % 32 or k % 256 or 32 % (2 * warps):
        return False
    kt = k // 256
    return kt % sk == 0 and (kt // sk) % unroll == 0


def mma12_usable(m: int, n: int, k: int, mt: int, rpw: int, unroll: int, sk: int) -> bool:
    if m > 16 * mt or n % 32 or k % 256:
        return False
    ks = 8 // (4 // rpw)
    kt = k // 256
    if kt % (sk * ks):
        return False
    return (kt // sk // ks * 4) % unroll == 0


def w12b_usable(m: int, n: int, k: int, nt: int, warps: int, nst: int) -> bool:
    return m <= 8 * nt and m <= 32 and n % 64 == 0 and k % 128 == 0 and (k // 128) % warps == 0


def w12m_usable(m: int, n: int, k: int, mw: int, nt: int, nstage: int, sk: int) -> bool:
    return 1 <= m <= 8 * nt and n % (128 * mw) == 0 and k % 128 == 0 and (k // 128) % sk == 0 and (k // 128) // sk >= 1


def w12_usable(m: int, n: int, k: int, nt: int, nstage: int, sk: int) -> bool:
    return m <= 8 * nt and n % 64 == 0 and k % 128 == 0 and (k // 128) % sk == 0 and (k // 128) // sk >= 1


def _repack_w12(pk, n: int, k: int) -> torch.Tensor:
    """_pack12 codes (same bits, same escapes) in the w12 kernel's fragment order: uint8 [n/64, k/128, 4, 3072]:
    per (m16 tile, chunk) low bytes [4 quads][32 lanes][16 B] (quad q = steps 2q, 2q+1; per step L0 = rows g, g+8
    at k 2t, 2t+1 then L1 at k 2t+8, 2t+9) and nibbles [2 halves][32 lanes][16 B] (per step one word, nibble i =
    element i of the L0, L1 order)."""
    dev = pk["lo"].device
    bn, bk = 32, int(pk["lo"].shape[1])                                                 # _pack12's tile shape
    lo = pk["lo"].view(n // bn, k // bk, bn, bk).permute(0, 2, 1, 3).reshape(n, k)
    nb = pk["nib"].view(n // bn, k // bk, bn, bk // 2).permute(0, 2, 1, 3).reshape(n, k // 2)
    nib = torch.stack((nb & 0xF, nb >> 4), -1).reshape(n, k)                            # [n, k] uint8 codes
    NTL, NC = n // 16, k // 128
    t = torch.arange(4, device=dev)
    kidx = torch.stack([2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9], -1)                 # [t, 4]
    out = torch.empty((NTL, NC, 3072), dtype=torch.uint8, device=dev)
    for part_lo in (True, False):
        src = lo if part_lo else nib
        s6 = src.view(NTL, 2, 8, NC, 8, 16)                                             # tile, rh, g, chunk, step, k16
        sel = s6[..., kidx]                                                             # tile, rh, g, chunk, step, t, q
        # element order: L0 = (rh0,q0),(rh0,q1),(rh1,q0),(rh1,q1); L1 = (rh0,q2),(rh0,q3),(rh1,q2),(rh1,q3)
        e = torch.stack([sel[:, 0, :, :, :, :, 0], sel[:, 0, :, :, :, :, 1], sel[:, 1, :, :, :, :, 0], sel[:, 1, :, :, :, :, 1],
                         sel[:, 0, :, :, :, :, 2], sel[:, 0, :, :, :, :, 3], sel[:, 1, :, :, :, :, 2], sel[:, 1, :, :, :, :, 3]], -1)
        # e: [tile, g, chunk, step, t, 8]
        e = e.permute(0, 2, 3, 1, 4, 5)                                                 # tile, chunk, step, g, t, 8 -> lane = g*4+t
        if part_lo:
            # [tile, chunk, step, lane, 8 B] -> quads: [tile, chunk, 4, 2 steps, lane, 8 B] -> [.., 4, lane, 16 B]
            w = e.reshape(NTL, NC, 4, 2, 32, 8).permute(0, 1, 2, 4, 3, 5).reshape(NTL, NC, 2048)
            out[:, :, :2048] = w
        else:
            v = e.to(torch.int32)
            word = torch.zeros((NTL, NC, 8, 32), dtype=torch.int32, device=dev)
            for i in range(8):
                word |= v[..., i].reshape(NTL, NC, 8, 32) << (4 * i)
            # [tile, chunk, step, lane] -> halves: [tile, chunk, 2, 4 steps, lane] -> [.., 2, lane, 4 words]
            w = word.view(NTL, NC, 2, 4, 32).permute(0, 1, 2, 4, 3).contiguous().view(torch.uint8).reshape(NTL, NC, 1024)
            out[:, :, 2048:] = w
    return out.view(NTL // 4, 4, NC, 3072).permute(0, 2, 1, 3).contiguous()           # [n/64, chunk, slot, 3 KB]


def mma12cp_usable(m: int, n: int, k: int, mt: int, rpw: int, nstage: int, sk: int) -> bool:
    if m > 16 * mt or n % 32 or k % 256:
        return False
    ks = 8 // (4 // rpw)
    kt = k // 256
    return kt % (sk * ks) == 0


# == BEGIN t12 python ==
def tb12_usable(m: int, n: int, k: int, r: int, ck: int, nt: int) -> bool:
    return 1 <= m <= 32 and m <= 8 * nt and n % r == 0 and k % (128 * ck) == 0


def tb12_nt(m: int) -> int:
    return 1 if m <= 8 else (2 if m <= 16 else 4)


def _repack_t12(w12: torch.Tensor, n: int, k: int, r: int, ck: int) -> torch.Tensor:
    """The w12 fragment layout [n/64, k/128, 4, 3072] regrouped for one bulk copy per (R-row tile, 128*ck-wide chunk):
    uint8 [n/r, k/(128 ck), r/16, ck, 3072] (same bytes; tile 16-row index = T*(r/16) + rt, 128-chunk = c*ck + j)."""
    t16 = w12.view(n // 64, k // 128, 4, 3072).permute(0, 2, 1, 3).reshape(n // 16, k // 128, 3072)
    return t16.view(n // r, r // 16, k // (128 * ck), ck, 3072).permute(0, 2, 1, 3, 4).contiguous()


T12_ESC_CAP = 64


def _esc_tab(pk, n: int, k: int, r: int):
    """Per r-row tile escape table for the t12c kernel: int32 [n/r, 64, 2] = (k | local_row << 16, delta bits), entries
    sorted by k within the tile, sentinel k = 0xFFFF; None if any tile has more than 64 entries."""
    dev = pk["esc_roff"].device
    nt = n // r
    tab = torch.full((nt, T12_ESC_CAP, 2), 0xFFFF, dtype=torch.int32, device=dev)
    tab[:, :, 1] = 0
    if pk["n_esc"] > 0:
        roff = pk["esc_roff"][: n + 1].long()
        rows = torch.repeat_interleave(torch.arange(n, device=dev), roff[1:] - roff[:-1])
        ek = pk["esc_k"][: rows.numel()].long(); ed = pk["esc_delta"][: rows.numel()]
        tile = rows // r
        counts = torch.bincount(tile, minlength=nt)
        if int(counts.max()) > T12_ESC_CAP:
            return None
        order = torch.argsort(tile * k + ek, stable=True)
        tile_s, ek_s, ed_s, rows_s = tile[order], ek[order], ed[order], rows[order]
        start = torch.cumsum(counts, 0) - counts
        pos = torch.arange(rows.numel(), device=dev) - start[tile_s]
        tab[tile_s, pos, 0] = (ek_s | ((rows_s % r) << 16)).to(torch.int32)
        tab[tile_s, pos, 1] = ed_s.view(torch.int32)
    return tab.contiguous()
# == END t12 python ==


# == END CUDA ==


# ----------------------------------------------------------------------------------------------
# Configs
# ----------------------------------------------------------------------------------------------

# (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages) per (layer, N, K, m-bucket).
# Aims: >= ~256 CTAs so every SM has 2+ resident, each with ~48 KB of weight in flight, and
# as little split-K partial traffic as that allows.
MAX_TRITON_M = 64        # BF16 layers above this go to cuBLAS
COUNTERS = 8192
PARTIALS_FLOATS = 16 << 20  # 64 MB fp32 workspace


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
USE_PACK12 = True   # also keep the 12-bit packing of the BF16 layers (the kernels that use it: *_SHAPES tables)
USE_W12 = True      # the w12 fragment-ordered copy of the 12-bit packing (W12_SHAPES)
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
# (layer, n, m) -> (warps, unroll, sk): the 12-bit packed GEMV (m = 1). Check f746ebec (judge protocol):
# pooler 12.8 vs 17.9 (BF16 Triton); qkv 10.3 vs 10.4 (row GEMV), unpooler slice / o: the row GEMV / Triton win
G12_SHAPES = {("pooler", 4096, 1): (8, 1, 4), ("qkv_gate", 2304, 1): (8, 1, 8)}   # qkv: 9.5 vs 10.6 row GEMV (run 00548e41)
# (layer, n, m) -> (mt, rpw, unroll, sk): the 12-bit packed mma.sync GEMM (m <= 32)
M12_SHAPES = {}
# (layer, n, m) -> (mt, rpw, nstage, sk): the cp.async-staged 12-bit mma GEMM (m <= 32)
M12CP_SHAPES = {}
# w12: the MoE-style weights-as-M 12-bit GEMM: (layer, n, m) -> (nt, nstage, sk); empty until the judge shows a win
W12_SHAPES = {}
# w12b: single-launch form (16-row tile per CTA, W warps over K slices): (layer, n, m) -> (nt, warps, nst)
# w12m: the 12-bit GEMM for m = 128 / 256 (weights = mma M side, 128*mw-row CTA tiles, 8*nt tokens, K split sk):
# (layer, n, m) -> (mw, nt, nstage, sk); empty until the judge table shows a win
W12M_SHAPES = {}   # dormant: tensor-bound at the mma.sync rate (~250 TFLOPS measured: pooler m=128 GEMM-only 35.8 us vs 23 cuBLAS; checks f2c19115, 3a2dd69a, 8256bca0)
W12B_SHAPES = {}   # w12 ties the table at m=8 on the judge (check a6b2bd96: pooler 18.34 vs 18.18, qkv 14.36 vs 14.28) and loses at
# m=16/32 stay on the table (the w12 GEMM alone is at the memory wall, ~13-17 us, but its K-split reduce launch costs 3-5 us)
# (layer, n, m) -> (SPLIT_K, num_warps, num_stages): the Triton 12-bit tiled GEMM (m <= 64)
T12_SHAPES = {}
# tb12: the single-launch TMA bulk-copy 12-bit GEMM (one CTA per r-row tile over the full K): (layer, n, m) -> (r, ck, w, nsw, nsx)
TB12_SHAPES = {("pooler", 4096, 8): (16, 4, 4, 3, 3), ("pooler", 4096, 16): (16, 4, 4, 3, 3), ("unpooler", 8192, 16): (16, 4, 4, 3, 3)}
# t12e: 16-row tiles, 4 compute warps + producer, 3 weight + 3 x stages. Judge (check 8d975106, 4/4 and 4/3 rings): pooler m=8 15.2 vs 18.2,
# m=16 16.5 vs 18.0; bench 3/3 rings: pooler 14.8 / 15.8, unpooler m=16 16.9 vs 18.2. qkv / o / m=1 / m=32 / the unpooler slice lose or tie.
_TB12_ON = False   # set by prepare() after the tb12 self-test (table output + probe rows) passes
T12_LAYOUTS = {(16, 4)}   # (r, ck) weight layouts kept in state (the bench sweeps them)
T12_ASM = True   # PRMT decode via inline PTX (the plain-Triton decode is for the interpreter)
T12_INTERP = False   # CPU-interpreter mode (fp32 dot over widened bit patterns); never on the GPU
# (layer, n, m) -> (bn, warps, unroll, sk): the tiled CTA-cooperative GEMV (m = 1) over the packed tiles
TGEMV_SHAPES = {}
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
    if (layer, n, m) in G12_SHAPES:
        warps, unroll, sk = G12_SHAPES[(layer, n, m)]
        return dict(g12=True, warps=warps, unroll=unroll, sk=sk)
    if _TB12_ON and (layer, n, m) in TB12_SHAPES:
        r, ck, w, nsw, nsx = TB12_SHAPES[(layer, n, m)]
        return dict(tb12=True, r=r, ck=ck, w=w, nsw=nsw, nsx=nsx, nt=tb12_nt(m), v=5)
    if (layer, n, m) in M12_SHAPES:
        mt, rpw, unroll, sk = M12_SHAPES[(layer, n, m)]
        return dict(m12=True, mt=mt, rpw=rpw, unroll=unroll, sk=sk)
    if (layer, n, m) in M12CP_SHAPES:
        mt, rpw, nstage, sk = M12CP_SHAPES[(layer, n, m)]
        return dict(m12cp=True, mt=mt, rpw=rpw, nstage=nstage, sk=sk)
    if (layer, n, m) in W12_SHAPES:
        nt, nstage, sk = W12_SHAPES[(layer, n, m)]
        return dict(w12=True, nt=nt, nstage=nstage, sk=sk)
    if (layer, n, m) in W12M_SHAPES:
        mw, nt, nstage, sk = W12M_SHAPES[(layer, n, m)]
        return dict(w12m=True, mw=mw, nt=nt, nstage=nstage, sk=sk)
    if (layer, n, m) in W12B_SHAPES:
        nt, warps, nst = W12B_SHAPES[(layer, n, m)]
        return dict(w12b=True, nt=nt, warps=warps, nst=nst)
    if (layer, n, m) in T12_SHAPES:
        sk, nw, ns = T12_SHAPES[(layer, n, m)]
        return dict(t12=True, BLOCK_M=_block_m(m), BLOCK_N=32, BLOCK_K=256, SPLIT_K=sk, num_warps=nw, num_stages=ns)
    if (layer, n, m) in TGEMV_SHAPES:
        bn, warps, unroll, sk = TGEMV_SHAPES[(layer, n, m)]
        return dict(tgemv=True, bn=bn, warps=warps, unroll=unroll, sk=sk)
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
    if cfg.get("tgemv"):
        return tgemv_usable(m, n, k, cfg["bn"], cfg["warps"], cfg["unroll"], cfg["sk"])
    if cfg.get("g12"):
        return gemv12_usable(m, n, k, cfg["warps"], cfg["unroll"], cfg["sk"])
    if cfg.get("m12cp"):
        return mma12cp_usable(m, n, k, cfg["mt"], cfg["rpw"], cfg["nstage"], cfg["sk"]) and (cfg["sk"] == 1 or cfg["sk"] * 16 * cfg["mt"] * n <= PARTIALS_FLOATS)
    if cfg.get("w12"):
        return w12_usable(m, n, k, cfg["nt"], cfg["nstage"], cfg["sk"]) and cfg["sk"] * m * n <= PARTIALS_FLOATS
    if cfg.get("w12m"):
        return w12m_usable(m, n, k, cfg["mw"], cfg["nt"], cfg["nstage"], cfg["sk"]) and cfg["sk"] * m * n <= PARTIALS_FLOATS
    if cfg.get("w12b"):
        return w12b_usable(m, n, k, cfg["nt"], cfg["warps"], cfg["nst"])
    if cfg.get("tb12"):
        return tb12_usable(m, n, k, cfg["r"], cfg["ck"], cfg["nt"])
    if cfg.get("m12"):
        return mma12_usable(m, n, k, cfg["mt"], cfg["rpw"], cfg["unroll"], cfg["sk"]) and (cfg["sk"] == 1 or cfg["sk"] * 16 * cfg["mt"] * n <= PARTIALS_FLOATS)
    if cfg.get("t12"):
        if m > cfg["BLOCK_M"] or n % 32 or k % 256 or (k // 256) % cfg["SPLIT_K"]:
            return False
        m_tiles = triton.cdiv(m, cfg["BLOCK_M"])
        return cfg["SPLIT_K"] == 1 or cfg["SPLIT_K"] * m_tiles * cfg["BLOCK_M"] * n <= PARTIALS_FLOATS
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
    state["diag"] = torch.zeros((8,), dtype=torch.int64, device=device)
    if USE_PACK12:
        for name in ("qkv_gate", "o", "pooler", "unpooler"):
            pk = _pack12(state[name])
            if pk is not None:
                pk["lo32"] = pk["lo"].view(torch.int32)      # [tiles * 32, 64]
                pk["nib16"] = pk["nib"].view(torch.int16)    # [tiles * 32, 64]
                if USE_W12:
                    try:
                        nn_, kk_ = state[name].shape
                        state[name + "_w12"] = _repack_w12(pk, nn_, kk_)
                        layouts = set(T12_LAYOUTS) if BENCH else {(c[0], c[1]) for (l, _, _), c in TB12_SHAPES.items() if l == name}
                        for r_, ck_ in sorted(layouts):
                            state[f"{name}_t12_{r_}x{ck_}"] = _repack_t12(state[name + "_w12"], nn_, kk_, r_, ck_)
                        for r_ in sorted({c[0] for c in layouts}):
                            state[f"{name}_esc_{r_}"] = _esc_tab(pk, nn_, kk_, r_)
                    except Exception as e:
                        print("w12 repack failed for", name, repr(e)[:120])
            state[name + "_p12"] = pk
        # the router's BF16 hi / lo planes, each with its own base and escapes (fp32 accumulation in the kernel)
        state["router_hi_p12"] = _pack12(state["router_hi"])
        state["router_lo_p12"] = _pack12(state["router_lo"])
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
    if cfg.get("t12"):
        pk = state.get(layer + "_p12")
        if pk is None:
            return _launch(x, w, w2, out, state, _triton_config(layer, m, n, k), m, n, k, layer)
        m_tiles = triton.cdiv(m, cfg["BLOCK_M"])
        _tiled12_gemm_kernel[(n // 32, cfg["SPLIT_K"], m_tiles)](
            x, pk["lo32"], pk["nib16"], out, state["partials"], state["counters"],
            pk["esc_off"], pk["esc_n"], pk["esc_k"], pk["esc_delta"],
            m, n, k, x.stride(0), out.stride(0), pk["eb"],
            BLOCK_M=cfg["BLOCK_M"], SPLIT_K=cfg["SPLIT_K"], ASM=T12_ASM, INTERP=T12_INTERP,
            num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
        return out
    if cfg.get("w12m"):
        pk = state.get(layer + "_p12")
        w12 = state.get(layer + "_w12")
        if pk is None or w12 is None:
            return torch.matmul(x, w.t(), out=out)
        _proj_build().w12m(x, w12, pk["eb"], pk["esc_roff"], pk["esc_k"], pk["esc_delta"], out, state["partials"],
                     m, n, k, cfg["mw"], cfg["nt"], cfg["nstage"], cfg["sk"], True, cfg.get("mode", 0))
        return out
    if cfg.get("tb12"):
        pk = state.get(layer + "_p12")
        wt = state.get(f"{layer}_t12_{cfg['r']}x{cfg['ck']}")
        et = state.get(f"{layer}_esc_{cfg['r']}")
        if pk is None or wt is None or (int(cfg.get("v", 5)) >= 3 and et is None):
            return _launch(x, w, w2, out, state, _triton_config(layer, m, n, k), m, n, k, layer)
        if et is None:
            et = state["diag"]
        _proj_build().t12(x, wt, pk["eb"], pk["esc_roff"], pk["esc_k"], pk["esc_delta"], out, state["diag"],
                    n, k, cfg["r"], cfg["ck"], cfg["w"], cfg["nsw"], cfg["nsx"], cfg["nt"], bool(cfg.get("diag")), int(cfg.get("v", 5)), et)
        return out
    if cfg.get("w12b"):
        pk = state.get(layer + "_p12")
        w12 = state.get(layer + "_w12")
        if pk is None or w12 is None:
            return _launch(x, w, w2, out, state, _triton_config(layer, m, n, k), m, n, k, layer)
        _proj_build().w12b(x, w12, pk["eb"], pk["esc_off"], pk["esc_n"], pk["esc_k"], pk["esc_delta"], out,
                     m, n, k, cfg["nt"], cfg["warps"], cfg["nst"])
        return out
    if cfg.get("w12"):
        pk = state.get(layer + "_p12")
        w12 = state.get(layer + "_w12")
        if pk is None or w12 is None:
            return _launch(x, w, w2, out, state, _triton_config(layer, m, n, k), m, n, k, layer)
        _proj_build().w12(x, w12, pk["eb"], pk["esc_off"], pk["esc_n"], pk["esc_k"], pk["esc_delta"], out, state["partials"], state["counters"],
                    m, n, k, cfg["nt"], cfg["nstage"], cfg["sk"], cfg.get("mode", 0))
        return out
    if cfg.get("m12cp"):
        pk = state.get(layer + "_p12")
        if pk is None:
            return _launch(x, w, w2, out, state, _triton_config(layer, m, n, k), m, n, k, layer)
        _proj_build().mma12cp(x, pk["lo"], pk["nib"], pk["eb"], pk["esc_off"], pk["esc_n"], pk["esc_k"], pk["esc_delta"],
                        out, state["partials"], state["counters"], cfg["mt"], cfg["rpw"], cfg["nstage"], cfg["sk"])
        return out
    if cfg.get("m12"):
        pk = state.get(layer + "_p12")
        if pk is None:
            return _launch(x, w, w2, out, state, _triton_config(layer, m, n, k), m, n, k, layer)
        _proj_build().mma12(x, pk["lo"], pk["nib"], pk["eb"], pk["esc_off"], pk["esc_n"], pk["esc_k"], pk["esc_delta"],
                      out, state["partials"], state["counters"], cfg["mt"], cfg["rpw"], cfg["unroll"], cfg["sk"])
        return out
    if cfg.get("g12"):
        if layer == "router":
            p1, p2 = state.get("router_hi_p12"), state.get("router_lo_p12")
            if p1 is None or p2 is None:
                return _launch(x, w, w2, out, state, _triton_config(layer, m, n, k), m, n, k, layer)
            _proj_build().gemv12r(x, p1["lo"], p1["nib"], p1["eb"], p1["esc_roff"], p1["esc_n"], p1["esc_k"], p1["esc_delta"],
                            p2["lo"], p2["nib"], p2["eb"], p2["esc_roff"], p2["esc_n"], p2["esc_k"], p2["esc_delta"],
                            out, state["partials"], state["counters"], cfg["warps"], cfg["unroll"], cfg["sk"])
            return out
        # the unpooler's m <= 8 slice (rows 0..1023) is the first 32 n-major tiles of the full packing
        pk = state.get(layer + "_p12")
        if pk is None:   # this tensor did not fit the packing: the Triton table config
            return _launch(x, w, w2, out, state, _triton_config(layer, m, n, k), m, n, k, layer)
        _proj_build().gemv12(x, pk["lo"], pk["nib"], pk["eb"], pk["esc_roff"], pk["esc_n"], pk["esc_k"], pk["esc_delta"],
                       out, state["partials"], state["counters"], cfg["warps"], cfg["unroll"], cfg["sk"])
        return out
    if cfg.get("tgemv"):
        wt = state[f"{layer}_tiled_{cfg['bn']}x256"]
        _proj_build().tgemv(x, wt, out, state["partials"], state["counters"], cfg["bn"], cfg["warps"], cfg["unroll"], cfg["sk"])
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


# == BEGIN CUDA SLIDING ==
CUDA_SLIDING_SRC = r"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cooperative_groups.h>
#include <cstdint>
namespace cg = cooperative_groups;

// Sliding-window gated attention for one decode / verify step: a cluster of four CTAs per
// (request, KV head), each owning 512 of the 2048 window slots with 8 warps x 64 keys.
//   trip 1: every load that depends on nothing (the request's length, the scales, the query rows,
//           the block's new key/value rows; the gate rows are prefetched into L2);
//   trip 2: each lane fetches the page indices of its rows (4 lanes per row), then every K and V
//           row of the CTA goes out at once (cp.async, 16 B per lane, into a staging tile);
//   the warps convert their own 64 K and V rows once to fp16 (swizzled 128-byte rows), run the
//   softmax over their keys on mma.sync m16n8k16 with ldmatrix / ldmatrix.trans fragments, and
//   merge their (m, l, acc) partials in shared memory (the staging tile, now free) together with
//   the block's own new tokens; rank 0 combines the four CTAs over distributed shared memory and
//   applies the gated RMSNorm. The phases around the hot loop are __noinline__ functions and the
//   row-tile loop is rolled, so the loop body stays small in the instruction cache.
// Every window key enters the softmax; nothing is skipped.

#define D 64
#define G 8                     // query heads per KV head
#define WINDOW 2048
#define NCTA 4
#define NW 8
#define NT (NW * 32)
#define KEYS (WINDOW / NCTA)    // 512 slots per CTA, 64 per warp
#define QS_STRIDE 72            // fp16 elements per Q row (padded: ldmatrix conflict free)
#define RS 65                   // padded row stride (floats) of a 16 x 64 partial
#define RED_W (32 + 16 * RS)    // floats per 16-row partial: 16 m, 16 l, 16 x RS acc
#define NEG (-1.0e30f)

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }
__device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" :: "l"(p)); }
__device__ __forceinline__ uint32_t cvt_e4m3x2_f16x2(uint32_t pair) {
    uint32_t r;
    asm("{ .reg .b16 h; mov.b16 h, %1; cvt.rn.f16x2.e4m3x2 %0, h; }" : "=r"(r) : "h"((unsigned short)pair));
    return r;
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma16816(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ float2 f16x2_to_f2(uint32_t v) {
    __half2 h = *reinterpret_cast<__half2*>(&v);
    return __half22float2(h);
}
// sin/cos of the RoPE angle x (fp32, as the reference computes it): the range reduction in double,
// the evaluation with the hardware intrinsic (|r| <= pi: ~2^-21 absolute error; no local memory)
__device__ __forceinline__ void sincos_fast(float x, float& sn, float& cs) {
    const double two_pi = 6.283185307179586;
    const double xd = (double)x;
    const float r = (float)(xd - rint(xd / two_pi) * two_pi);
    __sincosf(r, &sn, &cs);
}
// byte offset of 16-byte chunk `ch` of row `r` in the fp8 staging tile ([rows][64 B], swizzled)
__device__ __forceinline__ uint32_t swz8(int r, int ch) { return (uint32_t)r * 64u + (uint32_t)((ch ^ ((r >> 1) & 3)) << 4); }
// byte offset of 16-byte chunk `ch` (8 fp16) of row `r` in an fp16 tile ([rows][128 B], swizzled)
__device__ __forceinline__ uint32_t swz16(int r, int ch) { return (uint32_t)r * 128u + (uint32_t)((ch ^ (r & 7)) << 4); }

template <int L>
struct SlidingSmem {
    static constexpr int R = L * G, NRT = (R + 15) / 16, ROWS = NRT * 16;
    static constexpr int STG = 0;                                  // fp8 staging: K rows then V rows (KEYS x 64 B each)
    static constexpr int RED = STG;                                // after the conversion: NW x RED_W floats
    static constexpr int FIN = RED + NW * RED_W * 4;               // NRT x RED_W floats
    static constexpr int STG_BYTES = 2 * KEYS * 64;
    static_assert(FIN + NRT * RED_W * 4 <= STG_BYTES, "merge buffers must fit the staging tile");
    static constexpr int K16 = STG + STG_BYTES;                    // KEYS x 128 B fp16
    static constexpr int V16 = K16 + KEYS * 128;                   // KEYS x 128 B fp16
    static constexpr int QS = V16 + KEYS * 128;                    // ROWS x QS_STRIDE fp16
    static constexpr int SNEW = QS + ROWS * QS_STRIDE * 2;         // ROWS x L floats
    static constexpr int VNEW = SNEW + ROWS * L * 4;               // L x 64 floats (raw e4m3 values)
    static constexpr int KNEW = VNEW + L * D * 4;                  // L x 64 floats
    static constexpr int BYTES = KNEW + L * D * 4;
};

struct Geo {                                   // what every phase needs
    const __nv_bfloat16* qkv; __nv_bfloat16* out;
    int b, h, seq, HQ, HKV, QKV_W, rank, abl;
    float qk_scale, k_scale, v_scale, sm_scale_log2, log2_theta;
};

// ---- prep: queries (RMSNorm x qk_scale x k_scale, RoPE) -> fp16 Qs; the block's new keys/values ----
template <int L>
__device__ __noinline__ void prep_phase(const Geo& geo, __half* Qs, float* knew, float* vnew, uint8_t* k_cache, uint8_t* v_cache,
                                        const int64_t* cache_loc) {
    using S = SlidingSmem<L>;
    constexpr int R = S::R, ROWS = S::ROWS;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int ROW_W = geo.HKV * D;
    const float inv_freq = exp2f(-(2.0f * lane / D) * geo.log2_theta);   // this lane's RoPE pair (dims 2*lane, 2*lane+1)
    for (int r = warp; r < ROWS; r += NW) {
        float x0 = 0.f, x1 = 0.f;
        const int j = r / G, qh = geo.h * G + (r % G);
        if (r < R && !(geo.abl & 16)) {
            const __nv_bfloat16* qp = geo.qkv + ((int64_t)(geo.b * L + j)) * geo.QKV_W + qh * D + 2 * lane;
            x0 = __bfloat162float(qp[0]);
            x1 = __bfloat162float(qp[1]);
        }
        float ss = x0 * x0 + x1 * x1;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        const float inv = rsqrtf(ss / D + 1e-5f) * geo.qk_scale * geo.k_scale;
        x0 *= inv; x1 *= inv;
        float sn, cs;
        sincos_fast((float)(geo.seq + j) * inv_freq, sn, cs);
        *reinterpret_cast<__half2*>(Qs + r * QS_STRIDE + 2 * lane) = __floats2half2_rn(x0 * cs - x1 * sn, x0 * sn + x1 * cs);
    }
    if (warp < L) {
        const int j = warp;
        float x0 = 0.f, x1 = 0.f, y0v = 0.f, y1v = 0.f;
        if (!(geo.abl & 16)) {
            const __nv_bfloat16* kp = geo.qkv + ((int64_t)(geo.b * L + j)) * geo.QKV_W + (geo.HQ + geo.h) * D + 2 * lane;
            const __nv_bfloat16* vp = geo.qkv + ((int64_t)(geo.b * L + j)) * geo.QKV_W + (geo.HQ + geo.HKV + geo.h) * D + 2 * lane;
            x0 = __bfloat162float(kp[0]); x1 = __bfloat162float(kp[1]);
            y0v = __bfloat162float(vp[0]); y1v = __bfloat162float(vp[1]);
        }
        float ss = x0 * x0 + x1 * x1;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        const float inv = rsqrtf(ss / D + 1e-5f);
        x0 *= inv; x1 *= inv;
        float sn, cs;
        sincos_fast((float)(geo.seq + j) * inv_freq, sn, cs);
        const float y0 = x0 * cs - x1 * sn, y1 = x0 * sn + x1 * cs;
        __nv_fp8x2_e4m3 k8(make_float2(y0 / geo.k_scale, y1 / geo.k_scale));
        __nv_fp8x2_e4m3 v8(make_float2(y0v / geo.v_scale, y1v / geo.v_scale));
        const uint16_t kb = k8.__x, vb = v8.__x;
        // the raw E4M3 values the cache will hold (what the reference attends to)
        const float2 kk = f16x2_to_f2(cvt_e4m3x2_f16x2(kb)), vv = f16x2_to_f2(cvt_e4m3x2_f16x2(vb));
        knew[j * D + 2 * lane] = kk.x; knew[j * D + 2 * lane + 1] = kk.y;
        vnew[j * D + 2 * lane] = vv.x; vnew[j * D + 2 * lane + 1] = vv.y;
        if (geo.rank == 0) {
            const int64_t loc = cache_loc[geo.b * L + j];
            *reinterpret_cast<uint16_t*>(k_cache + loc * ROW_W + geo.h * D + 2 * lane) = kb;
            *reinterpret_cast<uint16_t*>(v_cache + loc * ROW_W + geo.h * D + 2 * lane) = vb;
        }
    }
}

// ---- scores of the block's own tokens: snew[r][jj] = q_r . knew[jj] (causal), log2 domain ----
template <int L>
__device__ __noinline__ void snew_phase(const Geo& geo, const __half* Qs, const float* knew, float* snew) {
    using S = SlidingSmem<L>;
    constexpr int R = S::R, ROWS = S::ROWS;
    for (int idx = threadIdx.x; idx < ROWS * L; idx += NT) {
        const int r = idx / L, jj = idx % L;
        float sc = NEG;
        if (r < R && jj <= r / G && !(geo.abl & 8)) {
            float a = 0.f;
            #pragma unroll 8
            for (int d = 0; d < D; d++) a += __half2float(Qs[r * QS_STRIDE + d]) * knew[jj * D + d];
            sc = a * geo.sm_scale_log2;
        }
        snew[idx] = sc;
    }
}

// ---- the warp's 64 K and V rows -> fp16 tiles (each lane 2 rows) ----
__device__ __noinline__ void convert_phase(uint32_t k8_base, uint32_t v8_base, uint32_t k16_base, uint32_t v16_base) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, wrow0 = warp * 64;
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        const int r = wrow0 + lane + i * 32;
        #pragma unroll
        for (int kv = 0; kv < 2; kv++) {
            const uint32_t src = (kv == 0 ? k8_base : v8_base);
            const uint32_t dst = (kv == 0 ? k16_base : v16_base);
            #pragma unroll
            for (int c8 = 0; c8 < 4; c8++) {         // 16 B of fp8 -> 32 B of fp16 (two 16 B chunks)
                uint4 w;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(w.x), "=r"(w.y), "=r"(w.z), "=r"(w.w) : "r"(src + swz8(r, c8)));
                uint4 o0, o1;
                o0.x = cvt_e4m3x2_f16x2(w.x & 0xffffu); o0.y = cvt_e4m3x2_f16x2(w.x >> 16);
                o0.z = cvt_e4m3x2_f16x2(w.y & 0xffffu); o0.w = cvt_e4m3x2_f16x2(w.y >> 16);
                o1.x = cvt_e4m3x2_f16x2(w.z & 0xffffu); o1.y = cvt_e4m3x2_f16x2(w.z >> 16);
                o1.z = cvt_e4m3x2_f16x2(w.w & 0xffffu); o1.w = cvt_e4m3x2_f16x2(w.w >> 16);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dst + swz16(r, 2 * c8)), "r"(o0.x), "r"(o0.y), "r"(o0.z), "r"(o0.w));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dst + swz16(r, 2 * c8 + 1)), "r"(o1.x), "r"(o1.y), "r"(o1.z), "r"(o1.w));
            }
        }
    }
}

// ---- merge the warps' partials of one row tile (+ the block's own tokens on rank 0) -> fin[rt] ----
template <int L>
__device__ __noinline__ void merge_phase(const Geo& geo, int rt, const float* red, const float* snew, const float* vnew, float* F) {
    const bool own_new = (geo.rank == 0);
    const int tid = threadIdx.x;
    #pragma unroll 1
    for (int k = 0; k < (16 * D) / NT; k++) {
        const int idx = tid + k * NT, r = idx / D, d = idx % D;
        float M = NEG;
        #pragma unroll
        for (int ww = 0; ww < NW; ww++) M = fmaxf(M, red[ww * RED_W + r]);
        if (own_new) for (int jj = 0; jj < L; jj++) M = fmaxf(M, snew[(rt * 16 + r) * L + jj]);
        float O = 0.f, Ls = 0.f;
        #pragma unroll
        for (int ww = 0; ww < NW; ww++) {
            const float w = exp2f(red[ww * RED_W + r] - M);
            Ls += w * red[ww * RED_W + 16 + r];
            O += w * red[ww * RED_W + 32 + r * RS + d];
        }
        if (own_new) {
            for (int jj = 0; jj < L; jj++) {
                const float w = exp2f(snew[(rt * 16 + r) * L + jj] - M);
                Ls += w;
                O += w * vnew[jj * D + d];
            }
        }
        F[32 + r * RS + d] = O;
        if (d == 0) { F[r] = M; F[16 + r] = Ls; }
    }
}

// ---- rank 0: combine the four CTAs' partials over distributed shared memory ----
template <int L>
__device__ __noinline__ void combine_phase(float* fin, float* cw) {
    using S = SlidingSmem<L>;
    constexpr int NRT = S::NRT;
    cg::cluster_group cluster = cg::this_cluster();
    const float* P1 = cluster.map_shared_rank(fin, 1);
    const float* P2 = cluster.map_shared_rank(fin, 2);
    const float* P3 = cluster.map_shared_rank(fin, 3);
    const int tid = threadIdx.x;
    if (tid < NRT * 16) {
        const int rt = tid / 16, r = tid % 16, off = rt * RED_W;
        const float M0 = fin[off + r], M1 = P1[off + r], M2 = P2[off + r], M3 = P3[off + r];
        const float L0 = fin[off + 16 + r], L1 = P1[off + 16 + r], L2 = P2[off + 16 + r], L3 = P3[off + 16 + r];
        const float M = fmaxf(fmaxf(M0, M1), fmaxf(M2, M3));
        const float w0 = exp2f(M0 - M), w1 = exp2f(M1 - M), w2 = exp2f(M2 - M), w3 = exp2f(M3 - M);
        cw[tid * 4 + 0] = w0; cw[tid * 4 + 1] = w1; cw[tid * 4 + 2] = w2; cw[tid * 4 + 3] = w3;
        fin[off + r] = M;
        fin[off + 16 + r] = L0 * w0 + L1 * w1 + L2 * w2 + L3 * w3;
    }
    __syncthreads();
    #pragma unroll 1
    for (int k = 0; k < (NRT * 16 * D) / NT; k++) {
        const int idx = tid + k * NT, rt = idx / (16 * D), rem = idx % (16 * D), r = rem / D, d = rem % D;
        const int off = rt * RED_W, e = off + 32 + r * RS + d, wi = (rt * 16 + r) * 4;
        fin[e] = fin[e] * cw[wi] + P1[e] * cw[wi + 1] + P2[e] * cw[wi + 2] + P3[e] * cw[wi + 3];
    }
}

// ---- epilogue: o = O / L x v_scale; per-head RMSNorm x sigmoid(gate) -> bf16 ----
template <int L>
__device__ __noinline__ void epilogue_phase(const Geo& geo, const float* fin) {
    using S = SlidingSmem<L>;
    constexpr int R = S::R;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int r = warp; r < R; r += NW) {
        const float* F = fin + (r / 16) * RED_W;
        const int rr = r % 16;
        const float inv = geo.v_scale / F[16 + rr];
        const float o0 = F[32 + rr * RS + lane] * inv, o1 = F[32 + rr * RS + lane + 32] * inv;
        float ss = o0 * o0 + o1 * o1;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        const float sc = rsqrtf(ss / D + 1e-5f);
        const int j = r / G, qh = geo.h * G + (r % G);
        const __nv_bfloat16* gp = geo.qkv + ((int64_t)(geo.b * L + j)) * geo.QKV_W + (geo.HQ + 2 * geo.HKV + qh) * D;
        const float g0 = __bfloat162float(gp[lane]), g1 = __bfloat162float(gp[lane + 32]);
        __nv_bfloat16* op = geo.out + ((int64_t)(geo.b * L + j)) * geo.HQ * D + qh * D;
        op[lane] = __float2bfloat16(o0 * sc / (1.f + expf(-g0)));
        op[lane + 32] = __float2bfloat16(o1 * sc / (1.f + expf(-g1)));
    }
}

template <int L>
__global__ void __launch_bounds__(NT, 1)
sliding_attn_kernel(const __nv_bfloat16* __restrict__ qkv, const int64_t* __restrict__ cache_loc,
                    const int* __restrict__ seq_lens, const int* __restrict__ page_table,
                    uint8_t* __restrict__ k_cache, uint8_t* __restrict__ v_cache,
                    const float* __restrict__ k_scale_p, const float* __restrict__ v_scale_p,
                    __nv_bfloat16* __restrict__ out, int HQ, int HKV, int max_pages, int PS,
                    float qk_scale, float sm_scale_log2, float log2_theta, int abl) {
    using S = SlidingSmem<L>;
    constexpr int R = S::R, NRT = S::NRT;
    const int QKV_W = (2 * HQ + 2 * HKV) * D;
    const int ROW_W = HKV * D;                   // one cache row (all KV heads) in bytes

    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* K8 = smem + S::STG;
    uint8_t* V8 = K8 + KEYS * 64;
    float* red = reinterpret_cast<float*>(smem + S::RED);
    float* fin = reinterpret_cast<float*>(smem + S::FIN);
    __half* Qs = reinterpret_cast<__half*>(smem + S::QS);
    float* snew = reinterpret_cast<float*>(smem + S::SNEW);
    float* vnew = reinterpret_cast<float*>(smem + S::VNEW);
    float* knew = reinterpret_cast<float*>(smem + S::KNEW);
    const uint32_t k8_base = smem_u32(K8), v8_base = smem_u32(V8);
    const uint32_t k16_base = smem_u32(smem + S::K16), v16_base = smem_u32(smem + S::V16);
    const uint32_t qs_base = smem_u32(Qs);
    __shared__ int s_seq;

    cg::cluster_group cluster = cg::this_cluster();
    const int rank = (int)cluster.block_rank();
    const int cid = blockIdx.x / NCTA;
    const int b = cid / HKV, h = cid % HKV;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, c = lane & 3;

    // ---- trip 1: the request's length and the scales; warm the L2 with the gate rows ----
    if (tid == 0) s_seq = seq_lens[b];
    const float k_scale = *k_scale_p, v_scale = *v_scale_p;
    for (int r = tid; r < R; r += NT) {
        const int j = r / G, qh = h * G + (r % G);
        prefetch_l2(qkv + ((int64_t)(b * L + j)) * QKV_W + (HQ + 2 * HKV + qh) * D);
    }
    __syncthreads();
    const int seq = s_seq;
    const int lo = max(seq - (WINDOW - 1), 0);
    const int n = seq - lo;                      // window keys (<= 2047): slots 0..n-1 <-> positions lo..seq-1
    const int slot0 = rank * KEYS;               // this CTA's first slot
    const int wrow0 = warp * 64;                 // this warp's first row (slot within the CTA)
    Geo geo;
    geo.qkv = qkv; geo.out = out; geo.b = b; geo.h = h; geo.seq = seq; geo.HQ = HQ; geo.HKV = HKV; geo.QKV_W = QKV_W;
    geo.rank = rank; geo.abl = abl; geo.qk_scale = qk_scale; geo.k_scale = k_scale; geo.v_scale = v_scale;
    geo.sm_scale_log2 = sm_scale_log2; geo.log2_theta = log2_theta;

    // ---- trip 2: page indices of this warp's 64 rows (4 lanes per row), then every K and V row ----
    const int ch = lane & 3;
    int pg[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int slot = slot0 + wrow0 + i * 8 + (lane >> 2);
        pg[i] = (slot < n) ? page_table[(int64_t)b * max_pages + (lo + slot) / PS] : 0;
    }
    if (!(abl & 1)) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int r = wrow0 + i * 8 + (lane >> 2);
            const int slot = slot0 + r;
            const bool ok = slot < n;
            const int64_t row = (int64_t)pg[i] * PS + ((lo + slot) % PS);
            cp_async16(k8_base + swz8(r, ch), k_cache + row * ROW_W + h * D + 16 * ch, ok ? 16 : 0);
            cp_async16(v8_base + swz8(r, ch), v_cache + row * ROW_W + h * D + 16 * ch, ok ? 16 : 0);
        }
    }
    cp_async_commit();

    // ---- meanwhile: the queries and the block's new keys/values ----
    if (!(abl & 8)) prep_phase<L>(geo, Qs, knew, vnew, k_cache, v_cache, cache_loc);
    __syncthreads();
    snew_phase<L>(geo, Qs, knew, snew);

    // ---- K and V rows -> fp16 tiles ----
    if (!(abl & 1)) cp_async_wait_all();
    __syncthreads();                             // every lane's copies have landed
    if (!(abl & 2)) convert_phase(k8_base, v8_base, k16_base, v16_base);
    __syncthreads();                             // fp16 tiles complete; the staging tile is free for the merge buffers

    // ---- the window: each warp its 64 keys, per 16-row tile (rolled: one body in the instruction cache) ----
    const int wslot0 = slot0 + wrow0;            // global slot of this warp's first key
    #pragma unroll 1
    for (int rt = 0; rt < ((abl & 32) ? 0 : NRT); rt++) {
        float m_run[2] = {NEG, NEG}, l_run[2] = {0.f, 0.f};
        float acc[8][4];
        #pragma unroll
        for (int i = 0; i < 8; i++) { acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f; }
        if (!(abl & 2)) {
            // each row's first valid slot (the window's edge): rows rt*16 + g and + g + 8
            const int first0 = n - (WINDOW - 1) + (rt * 16 + g) / G, first8 = n - (WINDOW - 1) + (rt * 16 + g + 8) / G;
            // ---- S = Q K^T over the warp's 64 keys (8 n-tiles) ----
            float s[8][4];
            #pragma unroll
            for (int i = 0; i < 8; i++) { s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f; }
            #pragma unroll
            for (int ks = 0; ks < 4; ks++) {
                uint32_t a[4];
                {   // A: rows rt*16.., cols ks*16..: matrices (rows 0-7 | 8-15) x (cols 0-7 | 8-15)
                    const int arow = rt * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    const int acol = ks * 16 + (lane >> 4) * 8;
                    ldmatrix_x4(qs_base + (uint32_t)(arow * QS_STRIDE + acol) * 2u, a[0], a[1], a[2], a[3]);
                }
                #pragma unroll
                for (int p2 = 0; p2 < 4; p2++) {          // 2 n-tiles (16 keys) per ldmatrix.x4
                    const int m = lane >> 3;             // matrices: (n-tile 2p2 | 2p2+1) x (k half 0 | 1)
                    const int key = wrow0 + p2 * 16 + (m >> 1) * 8 + (lane & 7);
                    uint32_t b00, b01, b10, b11;
                    ldmatrix_x4(k16_base + swz16(key, ks * 2 + (m & 1)), b00, b01, b10, b11);
                    mma16816(s[p2 * 2 + 0], a, b00, b01);
                    mma16816(s[p2 * 2 + 1], a, b10, b11);
                }
            }
            // ---- mask (slice end, window edge), softmax in the log2 domain ----
            float mx0 = NEG, mx8 = NEG;
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                const int slot = wslot0 + nt * 8 + 2 * c;
                #pragma unroll
                for (int e = 0; e < 2; e++) {
                    const int sl = slot + e;
                    const bool ok = sl < n;
                    s[nt][e] = (ok && sl >= first0) ? s[nt][e] * sm_scale_log2 : NEG;
                    s[nt][2 + e] = (ok && sl >= first8) ? s[nt][2 + e] * sm_scale_log2 : NEG;
                    mx0 = fmaxf(mx0, s[nt][e]);
                    mx8 = fmaxf(mx8, s[nt][2 + e]);
                }
            }
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
            mx8 = fmaxf(mx8, __shfl_xor_sync(0xffffffffu, mx8, 1));
            mx8 = fmaxf(mx8, __shfl_xor_sync(0xffffffffu, mx8, 2));
            m_run[0] = mx0; m_run[1] = mx8;
            uint32_t pa[4][4];                           // P as A fragments: 4 k16 steps over the 64 keys
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                const float p0 = exp2f(s[nt][0] - mx0), p1 = exp2f(s[nt][1] - mx0);
                const float p2 = exp2f(s[nt][2] - mx8), p3 = exp2f(s[nt][3] - mx8);
                l_run[0] += p0 + p1; l_run[1] += p2 + p3;
                pa[nt >> 1][(nt & 1) * 2 + 0] = pack_f16x2(p0, p1);
                pa[nt >> 1][(nt & 1) * 2 + 1] = pack_f16x2(p2, p3);
            }
            // ---- acc = P V: V fragments by ldmatrix.trans on the fp16 tile ----
            #pragma unroll
            for (int ks = 0; ks < 4; ks++) {
                #pragma unroll
                for (int p2 = 0; p2 < 4; p2++) {          // 2 n-tiles (16 dims) per ldmatrix.x4.trans
                    const int m = lane >> 3;             // matrices: (keys lo | hi) x (n-tile 2p2 | 2p2+1)
                    const int key = wrow0 + ks * 16 + (m & 1) * 8 + (lane & 7);
                    uint32_t b00, b01, b10, b11;
                    ldmatrix_x4_trans(v16_base + swz16(key, p2 * 2 + (m >> 1)), b00, b01, b10, b11);
                    mma16816(acc[p2 * 2 + 0], pa[ks], b00, b01);
                    mma16816(acc[p2 * 2 + 1], pa[ks], b10, b11);
                }
            }
        }
        // ---- the warp's partial for this row tile -> shared ----
        l_run[0] += __shfl_xor_sync(0xffffffffu, l_run[0], 1);
        l_run[0] += __shfl_xor_sync(0xffffffffu, l_run[0], 2);
        l_run[1] += __shfl_xor_sync(0xffffffffu, l_run[1], 1);
        l_run[1] += __shfl_xor_sync(0xffffffffu, l_run[1], 2);
        float* my = red + warp * RED_W;
        if (c == 0) { my[g] = m_run[0]; my[8 + g] = m_run[1]; my[16 + g] = l_run[0]; my[24 + g] = l_run[1]; }
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            const int d0 = nt * 8 + 2 * c;           // acc tile nt: dims nt*8 + n
            my[32 + g * RS + d0] = acc[nt][0];
            my[32 + g * RS + d0 + 1] = acc[nt][1];
            my[32 + (g + 8) * RS + d0] = acc[nt][2];
            my[32 + (g + 8) * RS + d0 + 1] = acc[nt][3];
        }
        __syncthreads();
        if (!(abl & 4)) merge_phase<L>(geo, rt, red, snew, vnew, fin + rt * RED_W);
        __syncthreads();
    }

    // ---- combine the four CTAs' partials over distributed shared memory (rank 0 finishes) ----
    if (!(abl & 128)) {
        cluster.sync();
        if (rank != 0) { cluster.sync(); return; }
        if (!(abl & 4)) combine_phase<L>(fin, red);   // red is free now: the weights go there
        cluster.sync();
    } else if (rank != 0) {
        return;
    }
    if (!(abl & 64)) epilogue_phase<L>(geo, fin);
}

template <int L>
static void launch_sliding(const torch::Tensor& qkv, const torch::Tensor& cache_loc, const torch::Tensor& seq_lens,
                           const torch::Tensor& page_table, torch::Tensor& k_cache, torch::Tensor& v_cache,
                           const torch::Tensor& k_scale, const torch::Tensor& v_scale, torch::Tensor& out,
                           int HQ, int HKV, int PS, float qk_scale, float sm_scale_log2, float log2_theta, int abl) {
    const int B = seq_lens.size(0);
    const int smem = SlidingSmem<L>::BYTES;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(sliding_attn_kernel<L>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        configured = true;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(B * HKV * NCTA);
    cfg.blockDim = dim3(NT);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = NCTA; attr[0].val.clusterDim.y = 1; attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr; cfg.numAttrs = 1;
    cudaError_t err = cudaLaunchKernelEx(&cfg, sliding_attn_kernel<L>,
        reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()), cache_loc.data_ptr<int64_t>(), seq_lens.data_ptr<int>(),
        page_table.data_ptr<int>(), reinterpret_cast<uint8_t*>(k_cache.data_ptr()), reinterpret_cast<uint8_t*>(v_cache.data_ptr()),
        k_scale.data_ptr<float>(), v_scale.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        HQ, HKV, (int)page_table.size(1), PS, qk_scale, sm_scale_log2, log2_theta, abl);
    TORCH_CHECK(err == cudaSuccess, "sliding_attn launch failed: ", cudaGetErrorString(err));
}

void sliding_attn(torch::Tensor qkv, torch::Tensor cache_loc, torch::Tensor seq_lens, torch::Tensor page_table,
                  torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor k_scale, torch::Tensor v_scale,
                  torch::Tensor out, int64_t L, int64_t HQ, int64_t HKV, int64_t PS,
                  double qk_scale, double sm_scale_log2, double log2_theta, int64_t abl) {
    TORCH_CHECK(HQ == 8 * HKV, "sliding_attn: 8 query heads per KV head");
    if (L == 1) launch_sliding<1>(qkv, cache_loc, seq_lens, page_table, k_cache, v_cache, k_scale, v_scale, out,
                                  (int)HQ, (int)HKV, (int)PS, (float)qk_scale, (float)sm_scale_log2, (float)log2_theta, (int)abl);
    else if (L == 8) launch_sliding<8>(qkv, cache_loc, seq_lens, page_table, k_cache, v_cache, k_scale, v_scale, out,
                                       (int)HQ, (int)HKV, (int)PS, (float)qk_scale, (float)sm_scale_log2, (float)log2_theta, (int)abl);
    else TORCH_CHECK(false, "sliding_attn: q_len must be 1 or 8");
}
"""

CPP_SLIDING_SRC = """
#include <torch/extension.h>
void sliding_attn(torch::Tensor qkv, torch::Tensor cache_loc, torch::Tensor seq_lens, torch::Tensor page_table,
                  torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor k_scale, torch::Tensor v_scale,
                  torch::Tensor out, int64_t L, int64_t HQ, int64_t HKV, int64_t PS,
                  double qk_scale, double sm_scale_log2, double log2_theta, int64_t abl);
"""

SLIDING_CUDA = False    # sliding-window layers: the dedicated CUDA kernel below (correct; measured slower than the
                        # Triton plan on every scored sliding shape: B1 decode 15.2 vs 15.5 us, B1 verify 43 vs 23,
                        # B16/B32 wave-bound with 212 KB smem per CTA; kept as a documented experiment, off)
_SLIDING_EXT = None
_SLIDING_FAILED = False


def _sliding_ext():
    """Build the sliding-window kernel once (nvcc, ~1 min, in the first eager call); None if it fails."""
    global _SLIDING_EXT, _SLIDING_FAILED
    if _SLIDING_EXT is None and not _SLIDING_FAILED:
        try:
            from torch.utils.cpp_extension import load_inline
            _SLIDING_EXT = load_inline(
                name="muse_attn_sliding_ext",
                cpp_sources=[CPP_SLIDING_SRC],
                cuda_sources=["#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n" + CUDA_SLIDING_SRC],
                functions=["sliding_attn"], extra_cuda_cflags=["-O3", "-arch=sm_90a"], verbose=False)
        except Exception as e:   # noqa: BLE001 - any build problem: keep serving with the Triton kernel
            _SLIDING_FAILED = True
            print("sliding CUDA kernel unavailable:", str(e)[:400])
    return _SLIDING_EXT


def _sliding_forward(data, out, abl=0):
    """The dedicated sliding-window kernel; returns False (nothing done) when it does not apply.
    abl (benchmark ablations, wrong results): 1 no K/V loads, 2 no QK/PV, 4 no merge, 8 no q/k prep."""
    cfg = data["config"]
    L, HQ, HKV, D = cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    if cfg["window"] != 2048 or L not in (1, 8) or HQ != 8 * HKV or D != 64 or not cfg["rope_theta"]:
        return False
    ext = _sliding_ext()
    if ext is None:
        return False
    qkv = data["qkv"]
    if qkv.stride(-1) != 1:
        qkv = qkv.contiguous()
    ext.sliding_attn(qkv, data["cache_loc"], data["seq_lens"], data["page_table"], data["k_cache"], data["v_cache"],
                     data["k_scale"], data["v_scale"], out, L, HQ, HKV, cfg["page_size"],
                     cfg["qk_scale_factor"], cfg["sm_scale"] * LOG2E, math.log2(cfg["rope_theta"]), int(abl))
    return True

# == END CUDA SLIDING ==


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
    nw = dict(hp=1, bd=False, stages=5, ex2h=True, block_t=128, cuda=False)   # the Triton kernel, current plans
    w8 = dict(nw, block_t=64, warps=8)
    cu = dict(cuda=True)
    abl = [dict(cuda=True, cabl=k) for k in (1, 2, 4, 8, 16, 32, 64, 128, 255)]   # no K/V loads / no convert+QK+PV / no merge / no prep / no trip-1 loads / no row tiles / no epilogue / no cluster syncs / skeleton
    plans = {
        "B1 L1 sliding": [dict(nw, splits=16), cu, cu] + abl,
        "B16 L1 sliding": [dict(nw, splits=8, block_t=64), cu, cu],
        "B32 L1 sliding": [dict(nw, splits=8, block_t=64), cu, cu],
        "B1 L8 sliding": [dict(w8, splits=32), cu, cu] + abl,
        "B16 L8 sliding": [dict(w8, splits=8), cu, cu],
        "B32 L8 sliding": [dict(w8, splits=4), cu, cu],
    }
    cases = {k: v for k, v in cases.items() if k in plans}
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
    if cfg["window"] and plan.get("cuda", SLIDING_CUDA) and _sliding_forward(data, out, plan.get("cabl", 0)):
        return out
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
// A fragment; byte b carries nibble A_b at the fp16 high-byte positions (s<<7 | e1<<3 | e0<<2 | m<<1) and
// nibble B_b in the free bits (s<<6 | e1<<5 | e0<<4 | m), so the dequant is
//   ma = w & 0x8E8E8E8E                                    bytes A_0..A_3
//   mb = (w << 1) & 0x82828282 | (w >> 2) & 0x0C0C0C0C     bytes B_0..B_3 (now fp16-placed)
// and four PRMT byte-expands (byte -> high byte of an fp16, low byte 0): (A_0,A_1) = a0 rows g k 2t,2t+1;
// (A_2,A_3) = a1 rows g+8; (B_0,B_1) = a2 rows g k 2t+8,2t+9; (B_2,B_3) = a3 rows g+8. Such an fp16 is
// exactly e2m1 * 2^-14 (exponent-aligned placement, subnormals included); times the row's E4M3 block scale
// (as fp16) the product has <= 8 significant bits and stays in range, so the fp16 weights are exact, and
// gs * 2^14 scales the fp32 sums. No cvt instruction: the 16-lane/clk conversion pipe was the bottleneck.
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

// programmatic dependent launch (no-ops when the kernel was launched without the attribute)
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }

constexpr int WT_NWARPS = 4;
__host__ __device__ constexpr int wt_stage_bytes(int tpw, int nt) {
    return tpw * WT_NWARPS * 1024 + tpw * WT_NWARPS * 128 + 8 * nt * A_STRIDE * 2;
}

template <bool GATED, int MW, int NT, int KS, int NSTAGE, bool SPLITK, int ABL>
__global__ void __launch_bounds__(128 * KS)
moe_w4a16_wt_kernel(const __half* __restrict__ X, int ldx,
                    const int* __restrict__ sorted_tok,
                    const int* __restrict__ blk_e, const int* __restrict__ blk_row0,
                    const int* __restrict__ blk_end, const int* __restrict__ nblk, const int* __restrict__ blk_bm,
                    const uint8_t* __restrict__ Wr, const uint8_t* __restrict__ Sr, const float* __restrict__ gs,
                    __half* __restrict__ out16, __nv_bfloat16* __restrict__ out_bf, int ldo,
                    float* __restrict__ part, int ldp, int part_rows, int* __restrict__ counters,
                    int K, int NTILES, float gs_mul, int l2pf)
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

    if (GATED) pdl_wait();                            // the routing lists and x16 come from the previous launch
    const int j = blockIdx.y;
    if (j >= *nblk) return;
    if (blk_bm[j] != 8 * NT) return;                  // mixed block sizes: this block belongs to the other kernel
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

    (void)l2pf;                                       // the L2-prefetch experiment was removed (illegal-instruction fault)
    auto issue_w = [&](int c, int stage) {
        const uint32_t ws = smem_base + stage * STAGE_BYTES;
        const uint32_t ss = ws + W_STAGE;
        #pragma unroll
        for (int i = 0; i < WSEG; ++i)
            cp_async16(ws + (tid + i * NTHR) * 16, wsrc[i] + c * 1024, 16);
        if (tid < TPC * 8) cp_async16(ss + tid * 16, ssrc + c * 128, 16);
    };
    auto issue_x = [&](int c, int stage) {
        const uint32_t xs = smem_base + stage * STAGE_BYTES + W_STAGE + S_STAGE;
        #pragma unroll
        for (int i = 0; i < XSEG; ++i) {
            const int sidx = tid + i * NTHR;
            if (sidx < NT * 128)
                cp_async16(xs + ((sidx >> 4) * A_STRIDE + (sidx & 15) * 8) * 2, xsrc[i] + c * KC, xbytes[i]);
        }
    };
    auto issue = [&](int c, int stage) { issue_w(c, stage); issue_x(c, stage); };

    float acc[TPW][NT * CH][4];
    #pragma unroll
    for (int i = 0; i < TPW; ++i)
        #pragma unroll
        for (int n = 0; n < NT * CH; ++n)
            #pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][n][q] = 0.f;

    const float gsf = gs[e] * gs_mul;                                   // gs * 2^14: applied to the fp32 sums

    if (GATED) {
        #pragma unroll
        for (int st = 0; st < NSTAGE - 1; ++st) {
            if (st < NC) issue(st, st);
            cp_async_commit();
        }
    } else {
        // GEMM2: the weights only need the routing lists (complete long ago), so their first stages stream before
        // waiting for GEMM1's inter; group st then holds the token rows of stage st (group 0 also all the weights)
        #pragma unroll
        for (int st = 0; st < NSTAGE - 1; ++st)
            if (st < NC) issue_w(st, st);
        pdl_wait();
        #pragma unroll
        for (int st = 0; st < NSTAGE - 1; ++st) {
            if (st < NC) issue_x(st, st);
            cp_async_commit();
        }
    }

    int stage = 0, stage_n = NSTAGE - 1;                              // ring slots of chunk c and of chunk c + NSTAGE - 1
    for (int c = 0; c < NC; ++c) {
        cp_async_wait<NSTAGE - 2>();
        __syncthreads();
        {
            const int cn = c + NSTAGE - 1;
            if (cn < NC) issue(cn, stage_n);
            cp_async_commit();
        }
        const uint8_t* ws = wt_smem + stage * STAGE_BYTES;
        stage = stage + 1 == NSTAGE ? 0 : stage + 1;
        stage_n = stage_n + 1 == NSTAGE ? 0 : stage_n + 1;
        const uint8_t* ss = ws + W_STAGE;
        const __half* xs = reinterpret_cast<const __half*>(ss + S_STAGE);
        if (ABL == 3) continue;                                      // bench ablation: the cp.async ring only
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
                        uint32_t S0, S1;                                     // row g / row g+8 block scale, both halves
                        asm("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %2; mov.b32 %0, {lo, lo}; mov.b32 %1, {hi, hi}; }\n"
                            : "=r"(S0), "=r"(S1) : "r"(sp));
                        // Conversion-free dequant: byte b of the word holds nibble A_b at bits (s:7, e1:3, e0:2, m:1)
                        // and nibble B_b at (s:6, e1:5, e0:4, m:0). Masked and byte-expanded into the high byte of
                        // each fp16 (PRMT), the fp16 is s<<15 | e1e0<<10 | m<<9 = e2m1 * 2^-14 exactly (subnormals
                        // included); x the E4M3 block scale is exact in fp16 (a multiple of 2^-24 below 0.18, for
                        // any E4M3 scale). No cvt on the 16-lane/clk conversion pipe; gs * 2^14 scales the sums.
                        const uint32_t ma = w & 0x8E8E8E8Eu;
                        const uint32_t mb = ((w << 1) & 0x82828282u) | ((w >> 2) & 0x0C0C0C0Cu);
                        uint32_t a[4];
                        if (ABL == 2) {                                      // bench ablation: raw words into the mma
                            a[0] = w; a[1] = w ^ sw; a[2] = w + sw; a[3] = w ^ 0x5A5A5A5Au;
                        } else {
                        a[0] = f16x2_mul(prmt(ma, 0u, 0x1404u), S0);        // (A_0, A_1): row g,   k 2t, 2t+1
                        a[1] = f16x2_mul(prmt(ma, 0u, 0x3424u), S1);        // (A_2, A_3): row g+8
                        a[2] = f16x2_mul(prmt(mb, 0u, 0x1404u), S0);        // (B_0, B_1): row g,   k 2t+8, 2t+9
                        a[3] = f16x2_mul(prmt(mb, 0u, 0x3424u), S1);        // (B_2, B_3): row g+8
                        }
                        #pragma unroll
                        for (int nb = 0; nb < NT; ++nb) {
                            if (ABL == 1) {                                  // bench ablation: dequant without the mma
                                asm volatile("" :: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[nb][2 * q]), "r"(b[nb][2 * q + 1]));
                                continue;
                            }
                            if (nb < ntb) mma16816(acc[i][nb * CH + (CH == 2 ? q : 0)], a, b[nb][2 * q], b[nb][2 * q + 1]);
                        }
                    }
                }
            }
        }
    }
    pdl_trigger();                                    // main loop done: the next kernel may start launching
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

template <bool GATED, int MW, int NT, int KS, int NSTAGE, bool SPLITK, int ABL>
void launch_wt(const __half* X, int ldx, const int* st, const int* be, const int* br, const int* bend, const int* nb, const int* bbm,
               const uint8_t* Wr, const uint8_t* Sr, const float* gs, __half* o16, __nv_bfloat16* obf, int ldo,
               float* part, int ldp, int part_rows, int* counters, int ksplit,
               int K, int ntiles, int max_blocks, float gs_mul, bool pdl, bool l2pf, cudaStream_t stream)
{
    constexpr int TPW = (GATED ? 2 : 1) * MW;
    constexpr int smem = NSTAGE * wt_stage_bytes(TPW, NT);
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(moe_w4a16_wt_kernel<GATED, MW, NT, KS, NSTAGE, SPLITK, ABL>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        attr_set = true;
    }
    const int n_x = (GATED ? ntiles / 2 : ntiles) / (WT_NWARPS * MW);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(n_x, max_blocks, ksplit);
    cfg.blockDim = dim3(128 * KS);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = pdl ? 1 : 0;
    const cudaError_t err = cudaLaunchKernelEx(&cfg, moe_w4a16_wt_kernel<GATED, MW, NT, KS, NSTAGE, SPLITK, ABL>,
        X, ldx, st, be, br, bend, nb, bbm, Wr, Sr, gs, o16, obf, ldo, part, ldp, part_rows, counters, K, ntiles, gs_mul, (int)l2pf);
    TORCH_CHECK(err == cudaSuccess, "wt launch failed: ", cudaGetErrorString(err));
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
    const float twv = tw[tok * 8 + warp];
    pdl_wait();                                          // y comes from the previous launch
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
        sc_sh[warp] = r >= 0 ? rsqrtf(ss / (float)H + eps) * twv : 0.f;
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

// Routing for the wt path, one launch: CTA 0 ranks the (token, slot) pairs by local expert (deterministic: pair
// order within an expert) and writes the padded rows and the block list; the other CTAs convert x to fp16 (exact
// for BF16 values in fp16 range, as the Triton route kernel did). Up to MAXCH * 1024 pairs.
template <int TOPK, int E, int MAXCH>
__global__ void __launch_bounds__(1024)
moe_route_kernel(const int* __restrict__ ids, const __nv_bfloat16* __restrict__ x, __half* __restrict__ x16,
                 int* __restrict__ sorted_tok, int* __restrict__ row_of_pair, int* __restrict__ blk_e,
                 int* __restrict__ blk_row0, int* __restrict__ blk_end, int* __restrict__ nblk, int* __restrict__ blk_bm,
                 int M, int local_start, int BM, int BM2, int n_elems, float xscale)
{
    const int tid = threadIdx.x;
    if (blockIdx.x > 0) {                                        // x -> fp16, 8 elements per thread
        const int i = ((int)(blockIdx.x - 1) * 1024 + tid) * 8;
        if (i < n_elems) {
            const uint4 v = *reinterpret_cast<const uint4*>(x + i);
            const __nv_bfloat162* pb = reinterpret_cast<const __nv_bfloat162*>(&v);
            uint4 o;
            __half2* po = reinterpret_cast<__half2*>(&o);
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                float2 f = __bfloat1622float2(pb[q]);
                f.x *= xscale; f.y *= xscale;
                po[q] = __float22half2_rn(f);
            }
            *reinterpret_cast<uint4*>(x16 + i) = o;
        }
        return;
    }
    constexpr int NW = 32;
    __shared__ int wc[NW][E];                                    // this chunk's pairs per (warp, expert)
    __shared__ int wp[NW][E];                                    // exclusive prefix over the warps
    __shared__ int base[E];                                      // pairs of the earlier chunks
    __shared__ int pad_off[E], blk_off[E], nb_s[E], cnt_s[E], bm_s[E];
    __shared__ int total_s;
    const int warp = tid >> 5, lane = tid & 31;
    const int npairs = M * TOPK;
    const int nch = (npairs + 1023) >> 10;
    if (tid < E) base[tid] = 0;
    int rank[MAXCH], ex[MAXCH];
    #pragma unroll
    for (int ch = 0; ch < MAXCH; ++ch) {
        rank[ch] = 0; ex[ch] = E;
        if (ch < nch) {
            const int i = (ch << 10) + tid;
            int e = E;
            if (i < npairs) {
                const int v = ids[i] - local_start;
                if (v >= 0 && v < E) e = v;
            }
            ex[ch] = e;
            if (lane < E) wc[warp][lane] = 0;
            __syncthreads();
            const unsigned m = __match_any_sync(0xffffffffu, e);
            rank[ch] = __popc(m & ((1u << lane) - 1u));
            if (e < E && rank[ch] == 0) wc[warp][e] = __popc(m);
            __syncthreads();
            if (tid < E) {                                       // thread e: prefix over warps
                int run = base[tid];
                #pragma unroll
                for (int w = 0; w < NW; ++w) { wp[w][tid] = run; run += wc[w][tid]; }
                base[tid] = run;
            }
            __syncthreads();
            if (e < E) rank[ch] += wp[warp][e];
        }
    }
    if (warp == 0) {                                             // lane e: counts -> padded rows, block offsets
        const int cnt = base[lane];
        const int bm = (BM2 > 0 && cnt > BM) ? BM2 : BM;          // heavier experts get the larger blocks
        const int padded = (cnt + bm - 1) / bm * bm;
        const int nb = padded / bm;
        int po = padded, bo = nb;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int a = __shfl_up_sync(0xffffffffu, po, o), b = __shfl_up_sync(0xffffffffu, bo, o);
            if (lane >= o) { po += a; bo += b; }
        }
        cnt_s[lane] = cnt; pad_off[lane] = po - padded; blk_off[lane] = bo - nb; nb_s[lane] = nb; bm_s[lane] = bm;
        if (lane == 31) total_s = bo;
    }
    __syncthreads();
    #pragma unroll
    for (int ch = 0; ch < MAXCH; ++ch) {
        if (ch < nch) {
            const int i = (ch << 10) + tid;
            if (i < npairs) {
                const int e = ex[ch];
                if (e < E) {
                    const int row = pad_off[e] + rank[ch];
                    sorted_tok[row] = i / TOPK;
                    row_of_pair[i] = row;
                } else {
                    row_of_pair[i] = -1;
                }
            }
        }
    }
    const int total = total_s;
    for (int j = tid; j < total; j += 1024) {
        int ej = 0;
        #pragma unroll
        for (int e = 1; e < E; ++e) ej += (blk_off[e] <= j);     // blk_off is non-decreasing: the expert of block j
        blk_e[j] = ej + local_start;
        blk_row0[j] = pad_off[ej] + (j - blk_off[ej]) * bm_s[ej];
        blk_end[j] = pad_off[ej] + cnt_s[ej];
        blk_bm[j] = bm_s[ej];
    }
    if (tid == 0) nblk[0] = total;
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
                  torch::Tensor blk_end, torch::Tensor nblk, torch::Tensor blk_bm, torch::Tensor Wr, torch::Tensor Sr, torch::Tensor gs,
                  torch::Tensor out, torch::Tensor part, torch::Tensor counters, int64_t ksplit, int64_t K, int64_t ntiles,
                  int64_t max_blocks, bool gated, int64_t mw, int64_t nt, int64_t ks, int64_t nstage, double gs_mul, int64_t abl,
                  bool pdl, bool l2pf)
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
#define OKC_W(G, M, T, S, NS, SK, AB) launch_wt<G, M, T, S, NS, SK, AB>(Xp, (int)X.stride(0), sorted_tok.data_ptr<int>(), blk_e.data_ptr<int>(), \
        blk_row0.data_ptr<int>(), blk_end.data_ptr<int>(), nblk.data_ptr<int>(), blk_bm.data_ptr<int>(), Wp, Sp, gs.data_ptr<float>(), o16, obf, ldo, \
        pp, ldp, prow, cnt, (int)ksplit, (int)K, (int)ntiles, (int)max_blocks, (float)gs_mul, pdl, l2pf, stream)
    const int key = (int)abl * 10000000 + (ksplit > 1 ? 1000000 : 0) + (gated ? 100000 : 0) + (int)mw * 10000 + (int)nt * 1000 + (int)ks * 100 + (int)nstage;
    switch (key) {
        case 111102: OKC_W(true, 1, 1, 1, 2, false, 0); break;   case 111104: OKC_W(true, 1, 1, 1, 4, false, 0); break;
        case 111108: OKC_W(true, 1, 1, 1, 8, false, 0); break;
        case 112102: OKC_W(true, 1, 2, 1, 2, false, 0); break;   case 112104: OKC_W(true, 1, 2, 1, 4, false, 0); break;
        case 112103: OKC_W(true, 1, 2, 1, 3, false, 0); break;   case 22103: OKC_W(false, 2, 2, 1, 3, false, 0); break;
        case 114102: OKC_W(true, 1, 4, 1, 2, false, 0); break;   case 114104: OKC_W(true, 1, 4, 1, 4, false, 0); break;
        case 11102: OKC_W(false, 1, 1, 1, 2, false, 0); break;   case 11104: OKC_W(false, 1, 1, 1, 4, false, 0); break;
        case 11108: OKC_W(false, 1, 1, 1, 8, false, 0); break;
        case 21102: OKC_W(false, 2, 1, 1, 2, false, 0); break;   case 21104: OKC_W(false, 2, 1, 1, 4, false, 0); break;
        case 12104: OKC_W(false, 1, 2, 1, 4, false, 0); break;   case 22102: OKC_W(false, 2, 2, 1, 2, false, 0); break;
        case 22104: OKC_W(false, 2, 2, 1, 4, false, 0); break;   case 14104: OKC_W(false, 1, 4, 1, 4, false, 0); break;
        case 24102: OKC_W(false, 2, 4, 1, 2, false, 0); break;   case 24104: OKC_W(false, 2, 4, 1, 4, false, 0); break;
        // split-K (SPLITK = true) instantiations removed: measured slower than the plain variants on every shape
        // bench ablations (abl 1 = no mma, 2 = no dequant, 3 = loads only) are not instantiated in the submitted state
        default: TORCH_CHECK(false, "unsupported wt variant (mw, nt, ks, nstage, ksplit)");
    }
#undef OKC_W
}

void moe_finalize(torch::Tensor y, torch::Tensor row_of_pair, torch::Tensor tw, torch::Tensor nw, torch::Tensor out,
                  int64_t M, double eps, bool pdl)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(out.size(1) == 4096, "finalize: H must be 4096");
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)M);
    cfg.blockDim = dim3(256);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = pdl ? 1 : 0;
    const cudaError_t err = cudaLaunchKernelEx(&cfg, moe_finalize_kernel<4096>,
        reinterpret_cast<const __nv_bfloat16*>(y.data_ptr()), (int)y.stride(0), row_of_pair.data_ptr<int>(),
        tw.data_ptr<float>(), nw.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (float)eps);
    TORCH_CHECK(err == cudaSuccess, "finalize launch failed: ", cudaGetErrorString(err));
}

void moe_route(torch::Tensor ids, torch::Tensor x, torch::Tensor x16, torch::Tensor sorted_tok, torch::Tensor row_of_pair,
               torch::Tensor blk_e, torch::Tensor blk_row0, torch::Tensor blk_end, torch::Tensor nblk, torch::Tensor blk_bm,
               int64_t M, int64_t local_start, int64_t BM, int64_t BM2, double xscale)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    const int H = (int)x.size(1);
    const int n_elems = (int)M * H;
    TORCH_CHECK(M * 8 <= 8 * 1024, "route: too many pairs");
    TORCH_CHECK(H % 8 == 0, "route: H must be a multiple of 8");
    const unsigned grid = 1u + (unsigned)((n_elems + 8191) / 8192);
    moe_route_kernel<8, 32, 8><<<grid, 1024, 0, stream>>>(
        ids.data_ptr<int>(), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<__half*>(x16.data_ptr()),
        sorted_tok.data_ptr<int>(), row_of_pair.data_ptr<int>(), blk_e.data_ptr<int>(), blk_row0.data_ptr<int>(),
        blk_end.data_ptr<int>(), nblk.data_ptr<int>(), blk_bm.data_ptr<int>(), (int)M, (int)local_start, (int)BM, (int)BM2, n_elems, (float)xscale);
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
                  torch::Tensor blk_end, torch::Tensor nblk, torch::Tensor blk_bm, torch::Tensor Wr, torch::Tensor Sr, torch::Tensor gs,
                  torch::Tensor out, torch::Tensor part, torch::Tensor counters, int64_t ksplit, int64_t K, int64_t ntiles,
                  int64_t max_blocks, bool gated, int64_t mw, int64_t nt, int64_t ks, int64_t nstage, double gs_mul, int64_t abl,
                  bool pdl, bool l2pf);
void moe_finalize(torch::Tensor y, torch::Tensor row_of_pair, torch::Tensor tw, torch::Tensor nw, torch::Tensor out,
                  int64_t M, double eps, bool pdl);
void moe_route(torch::Tensor ids, torch::Tensor x, torch::Tensor x16, torch::Tensor sorted_tok, torch::Tensor row_of_pair,
               torch::Tensor blk_e, torch::Tensor blk_row0, torch::Tensor blk_end, torch::Tensor nblk, torch::Tensor blk_bm,
               int64_t M, int64_t local_start, int64_t BM, int64_t BM2, double xscale);
"""

_EXT = None


def build():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline
        _EXT = load_inline(
            name="okc_moe_w4a16", cpp_sources=[CPP_SRC],
            cuda_sources=["#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n" + CUDA_SRC],
            functions=["moe_w4a16_gemm", "moe_w4a16_wt", "moe_finalize", "moe_route"], extra_cuda_cflags=["-O3", "-arch=sm_90a", "--use_fast_math"], verbose=False)
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
    A-fragment nibbles for k16 steps 4*half .. 4*half+3 (byte b: nibble A_b at bits (s:7, e1:3, e0:2, m:1)
    = the fp16 high-byte placement, nibble B_b at (s:6, e1:5, e0:4, m:0); A_0,A_1 = row g k 2t,2t+1;
    A_2,A_3 = row g+8 k 2t,2t+1; B_0,B_1 = row g k 2t+8,2t+9; B_2,B_3 = row g+8 k 2t+8,2t+9);
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
                if q < 2:   # A: fp16 high-byte placement (s:7, e1:3, e0:2, m:1)
                    byte = (sgn << 7) | (e1 << 3) | (e0 << 2) | (m << 1)
                else:       # B: the free bits (s:6, e1:5, e0:4, m:0); the kernel shifts s, m up and e1 e0 down
                    byte = (sgn << 6) | (e1 << 5) | (e0 << 4) | m
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
WT_DEFAULT = {8: (1, 1, 1, 8, 1), 16: (1, 1, 1, 4, 1), 64: (1, 2, 1, 4, 1), 128: (2, 2, 1, 4, 1), 100000: (4, 2, 1, 2, 1)}
WT_SAFE = (1, 1, 1, 4, 1)
PART_SPLIT_MAX = 4      # fp32 partial slices for split-K
PART_ROWS = 1024        # padded rows the partial buffer holds (split-K is a small-batch tool)
_WT_OVERRIDE = None
_WT_ABL = 0            # bench ablation of the wt GEMMs (1 = no mma, 2 = no dequant, 3 = loads only)
PDL = True             # programmatic dependent launch GEMM1 -> GEMM2 -> finalize
_PDL_OVERRIDE = None   # bench: False forces plain launches
# mixed block sizes above this many tokens: experts with more rows than the policy's block get the big variant's blocks
WT_BIG = {128: (4, 2, 1, 2, 1), 100000: (4, 2, 1, 2, 1)}
MIXED_MIN_TOKENS = 1000000000
_MIXED_OVERRIDE = None
L2PF_MAX_TOKENS = 0   # the W2 L2-prefetch experiment was removed from the kernel (illegal-instruction fault): inert
_L2PF_OVERRIDE = None


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


ROUTE_CUDA = True   # the one-launch CUDA routing kernel (ranks + block list + fp16 copy of x) instead of the Triton one


def _wt_big_variant(M: int, state, small):
    """The variant for the heavier experts' blocks (None: a single block size)."""
    mixed = (M >= MIXED_MIN_TOKENS) if _MIXED_OVERRIDE is None else _MIXED_OVERRIDE
    if not mixed or _WT_OVERRIDE is not None:
        return None
    keys = sorted(WT_BIG)
    v = WT_BIG[keys[-1]]
    for k in keys:
        if M <= k:
            v = WT_BIG[k]
            break
    if v[0] <= small[0] or v in state.get("wt_bad", ()):
        return None
    return v


def _route_wt(x, ids, local_start: int, BM: int, state, ext, BM2: int = 0):
    M, H = x.shape
    if not ROUTE_CUDA:
        state["blk_bm"].fill_(BM)
        return _route(x, ids, local_start, BM, state, marlin=False)
    NP = max(16, triton.next_power_of_2(M * TOPK))
    MAXB = triton.next_power_of_2(NP // BM + LOCAL)
    ext.moe_route(ids, x.contiguous(), state["x16"], state["sorted_tok"], state["row_of_pair"], state["blk_e"], state["blk_row0"],
                  state["blk_end"], state["nblk"], state["blk_bm"], M, local_start, BM, BM2, float(state.get("xscale", 1.0)))
    return NP, MAXB


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
    small = _wt_variant(M, state)
    nt, mw2, ks, nstage, ksplit = small
    big = _wt_big_variant(M, state, small)
    pdl = PDL if _PDL_OVERRIDE is None else _PDL_OVERRIDE
    l2pf = pdl and ((M <= L2PF_MAX_TOKENS) if _L2PF_OVERRIDE is None else _L2PF_OVERRIDE)
    BM = 8 * nt
    BM2 = 8 * big[0] if big is not None else 0
    if M * TOPK + LOCAL * (BM - 1) > PART_ROWS:
        ksplit = 1
    ext = build()
    NP, MAXB = _route_wt(x, ids, local_start, BM, state, ext, BM2)
    maxb = min(MAXB, _wt_max_blocks(M, BM))
    x16 = state["x16"][:M]
    out = torch.empty((M, H), dtype=MOE_OUT_DTYPE, device=x.device)
    inter = state["inter"]
    y = state["y"]
    part = state["part"]
    counters = state["counters"]
    lists = (state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"], state["blk_bm"])
    # the big variant's launch is a plain one (its blocks are independent of the small kernel's, and stream order
    # keeps the next PDL kernel's wait covering both)
    ext.moe_w4a16_wt(x16, *lists, state["Wt13"], state["St13"], state["gs13"], inter, part, counters, ksplit, H, 2 * I // 16, maxb,
                     True, 1, nt, ks, nstage, state["gs_mul"], _WT_ABL, pdl, l2pf)
    if big is not None:
        ext.moe_w4a16_wt(x16, *lists, state["Wt13"], state["St13"], state["gs13"], inter, part, counters, 1, H, 2 * I // 16, maxb,
                         True, 1, big[0], big[2], big[3], state["gs_mul"], _WT_ABL, False, False)
    ext.moe_w4a16_wt(inter, *lists, state["Wt2"], state["St2"], state["gs2"], y, part, counters, ksplit, I, H // 16, maxb,
                     False, mw2, nt, ks, nstage, state["gs_mul"], _WT_ABL, pdl, l2pf)
    if big is not None:
        ext.moe_w4a16_wt(inter, *lists, state["Wt2"], state["St2"], state["gs2"], y, part, counters, 1, I, H // 16, maxb,
                         False, big[1], big[0], big[2], big[3], state["gs_mul"], _WT_ABL, False, False)
    ext.moe_finalize(y, state["row_of_pair"], tw, state["norm_w"], out, M, POST_NORM_EPS, pdl)
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
        "blk_bm": torch.zeros((maxb,), dtype=torch.int32, device=dev),
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
            state["gs_mul"] = 16384.0   # the fp16 placement is e2m1 * 2^-14; applied with gs to the fp32 sums
            for nm in ("w13_sf", "w2_sf"):   # the fp16 products are exact for every E4M3 scale; the stats are for the log
                sfv = weights[nm][:LOCAL].float()
                print(f"{nm}: block scales min {float(sfv.min()):.4g} max {float(sfv.max()):.4g}, below 2^-6: {int((sfv < 2.0 ** -6).sum())}")
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
