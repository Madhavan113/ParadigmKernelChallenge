"""CUDA W4A16 (NVFP4 weights, fp16 activations) grouped GEMM for the routed experts.

One CTA (4 warps, 128 threads) computes a [16 rows x 64 cols] tile of one expert block
(GEMM1: 32 gate/up column pairs, SiLU fused). Weights are repacked once, losslessly, into the
mma.m16n8k16 B-fragment order: per (expert, 8-row block, 128-wide K chunk) each lane owns 16
bytes = 32 nibbles = its B fragment for the chunk's 8 k16 steps, and 8 scale bytes.

Dequant per k16 step (4 values per lane): nibble -> E4M3 byte ((c & 8) << 4 | (c & 7) << 2 is
exactly e2m1(c) * 2^-6, subnormals included) -> cvt.rn.f16x2.e4m3x2 -> x (block scale x global
scale x 2^6) as an f16x2 multiply. fp32 accumulate.
"""

import torch

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

template <bool GATED, int MW, int NT, int KS, int NSTAGE>
__global__ void __launch_bounds__(128 * KS)
moe_w4a16_wt_kernel(const __half* __restrict__ X, int ldx,
                    const int* __restrict__ sorted_tok,
                    const int* __restrict__ blk_e, const int* __restrict__ blk_row0,
                    const int* __restrict__ blk_end, const int* __restrict__ nblk,
                    const uint8_t* __restrict__ Wr, const uint8_t* __restrict__ Sr, const float* __restrict__ gs,
                    __half* __restrict__ out16, __nv_bfloat16* __restrict__ out_bf, int ldo,
                    float* __restrict__ part, int ldp, int part_rows,
                    int K, int NTILES, float gs_mul)
{
    // gridDim.z > 1: split-K; CTA z owns chunks [z*NC/KSPLIT, (z+1)*NC/KSPLIT) and writes its fp32 partial
    // sums (times gs) to part[z][row][col] (GATED: gate at col, up at col + N/2) for a reduce kernel.
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
    const int KSPLIT = gridDim.z;
    const int NC_ALL = K / KC;
    const int NC = NC_ALL / KSPLIT;                   // chunks of this CTA
    const int c0 = blockIdx.z * NC;

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
                    if (KSPLIT > 1) {
                        float* pp = part + ((size_t)blockIdx.z * part_rows + row0 + tok) * ldp + col;
                        pp[0] = acc[i][nb * CH][2 * hr + cc] * gsf;
                        if (GATED) pp[(size_t)NTILES * 8] = acc[MW + i][nb * CH][2 * hr + cc] * gsf;   // up: + N/2
                        continue;
                    }
                    const size_t o = (size_t)(row0 + tok) * ldo + col;
                    if (GATED) {
                        const float gv = acc[i][nb * CH][2 * hr + cc] * gsf, uv = acc[MW + i][nb * CH][2 * hr + cc] * gsf;
                        const float hv = gv / (1.f + __expf(-gv)) * uv;
                        out16[o] = __float2half(__bfloat162float(__float2bfloat16(hv)));   // bf16 value in an fp16 container
                    } else {
                        out_bf[o] = __float2bfloat16(acc[i][nb * CH][2 * hr + cc] * gsf);
                    }
                }
            }
        }
    }
}

template <bool GATED, int MW, int NT, int KS, int NSTAGE>
void launch_wt(const __half* X, int ldx, const int* st, const int* be, const int* br, const int* bend, const int* nb,
               const uint8_t* Wr, const uint8_t* Sr, const float* gs, __half* o16, __nv_bfloat16* obf, int ldo,
               float* part, int ldp, int part_rows, int ksplit,
               int K, int ntiles, int max_blocks, float gs_mul, cudaStream_t stream)
{
    constexpr int TPW = (GATED ? 2 : 1) * MW;
    constexpr int smem = NSTAGE * wt_stage_bytes(TPW, NT);
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(moe_w4a16_wt_kernel<GATED, MW, NT, KS, NSTAGE>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        attr_set = true;
    }
    const int n_x = (GATED ? ntiles / 2 : ntiles) / (WT_NWARPS * MW);
    dim3 grid(n_x, max_blocks, ksplit);
    moe_w4a16_wt_kernel<GATED, MW, NT, KS, NSTAGE><<<grid, 128 * KS, smem, stream>>>(
        X, ldx, st, be, br, bend, nb, Wr, Sr, gs, o16, obf, ldo, part, ldp, part_rows, K, ntiles, gs_mul);
}

// split-K reduce of GEMM1's fp32 partials + SiLU(gate) * up -> inter (bf16 values in fp16); grid (rows per block, blocks)
__global__ void __launch_bounds__(256)
wt_silu_reduce_kernel(const float* __restrict__ part, int ldp, int part_rows, int ksplit, __half* __restrict__ out, int ldo,
                      const int* __restrict__ blk_row0, const int* __restrict__ blk_end, const int* __restrict__ nblk, int I)
{
    const int j = blockIdx.y;
    if (j >= *nblk) return;
    const int r = blk_row0[j] + blockIdx.x;
    if (r >= blk_end[j]) return;
    for (int c = threadIdx.x * 4; c < I; c += 256 * 4) {
        float4 gg = make_float4(0.f, 0.f, 0.f, 0.f), uu = gg;
        for (int z = 0; z < ksplit; ++z) {
            const float* row = part + ((size_t)z * part_rows + r) * ldp;
            const float4 a = *reinterpret_cast<const float4*>(row + c), b = *reinterpret_cast<const float4*>(row + I + c);
            gg.x += a.x; gg.y += a.y; gg.z += a.z; gg.w += a.w;
            uu.x += b.x; uu.y += b.y; uu.z += b.z; uu.w += b.w;
        }
        const float h0 = gg.x / (1.f + __expf(-gg.x)) * uu.x, h1 = gg.y / (1.f + __expf(-gg.y)) * uu.y;
        const float h2 = gg.z / (1.f + __expf(-gg.z)) * uu.z, h3 = gg.w / (1.f + __expf(-gg.w)) * uu.w;
        __half2 v01 = __halves2half2(__float2half(__bfloat162float(__float2bfloat16(h0))), __float2half(__bfloat162float(__float2bfloat16(h1))));
        __half2 v23 = __halves2half2(__float2half(__bfloat162float(__float2bfloat16(h2))), __float2half(__bfloat162float(__float2bfloat16(h3))));
        uint2 pk;
        pk.x = *reinterpret_cast<uint32_t*>(&v01); pk.y = *reinterpret_cast<uint32_t*>(&v23);
        *reinterpret_cast<uint2*>(out + (size_t)r * ldo + c) = pk;
    }
}

// post-expert RMSNorm x routing weight, summed per token: grid = tokens, 8 warps (one per route). y is bf16
// (ksplit == 1) or the sum of ksplit fp32 partial slices rounded to bf16 (as SGLang keeps y in bf16).
__global__ void __launch_bounds__(256)
moe_finalize_kernel(const __nv_bfloat16* __restrict__ y, int ldy, const float* __restrict__ part, int ldp, int part_rows,
                    int ksplit, const int* __restrict__ row_of_pair, const float* __restrict__ tw,
                    const float* __restrict__ nw, __nv_bfloat16* __restrict__ out, int H, float eps)
{
    extern __shared__ __align__(16) float red[];              // [8][H]
    __shared__ float sc_sh[8];
    const int tok = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = row_of_pair[tok * 8 + warp];
    float* myrow = red + warp * H;
    // pass 1: the route's row (bf16 values), scaled by (1 + w), into smem; sum of squares
    float ss = 0.f;
    for (int seg = lane; seg < H / 8; seg += 32) {
        float v[8];
        if (r < 0) {
            #pragma unroll
            for (int q = 0; q < 8; ++q) v[q] = 0.f;
        } else if (ksplit == 1) {
            const uint4 u = *reinterpret_cast<const uint4*>(y + (size_t)r * ldy + seg * 8);
            const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&u);
            #pragma unroll
            for (int q = 0; q < 4; ++q) { const float2 f = __bfloat1622float2(p[q]); v[2 * q] = f.x; v[2 * q + 1] = f.y; }
        } else {
            #pragma unroll
            for (int q = 0; q < 8; ++q) v[q] = 0.f;
            for (int z = 0; z < ksplit; ++z) {
                const float* row = part + ((size_t)z * part_rows + r) * ldp + seg * 8;
                const float4 a = *reinterpret_cast<const float4*>(row), b = *reinterpret_cast<const float4*>(row + 4);
                v[0] += a.x; v[1] += a.y; v[2] += a.z; v[3] += a.w; v[4] += b.x; v[5] += b.y; v[6] += b.z; v[7] += b.w;
            }
            #pragma unroll
            for (int q = 0; q < 8; ++q) v[q] = __bfloat162float(__float2bfloat16(v[q]));
        }
        const float4 n0 = *reinterpret_cast<const float4*>(nw + seg * 8), n1 = *reinterpret_cast<const float4*>(nw + seg * 8 + 4);
        v[0] *= n0.x; v[1] *= n0.y; v[2] *= n0.z; v[3] *= n0.w; v[4] *= n1.x; v[5] *= n1.y; v[6] *= n1.z; v[7] *= n1.w;
        #pragma unroll
        for (int q = 0; q < 8; ++q) ss += v[q] * v[q];
        *reinterpret_cast<float4*>(myrow + seg * 8) = make_float4(v[0], v[1], v[2], v[3]);
        *reinterpret_cast<float4*>(myrow + seg * 8 + 4) = make_float4(v[4], v[5], v[6], v[7]);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if (lane == 0) sc_sh[warp] = r >= 0 ? rsqrtf(ss / (float)H + eps) * tw[tok * 8 + warp] : 0.f;
    __syncthreads();
    // pass 2: weighted sum over the 8 routes
    for (int base = threadIdx.x * 8; base < H; base += 256 * 8) {
        float acc[8];
        #pragma unroll
        for (int q = 0; q < 8; ++q) acc[q] = 0.f;
        #pragma unroll
        for (int s2 = 0; s2 < 8; ++s2) {
            const float scs = sc_sh[s2];
            const float4 a = *reinterpret_cast<const float4*>(red + s2 * H + base), b = *reinterpret_cast<const float4*>(red + s2 * H + base + 4);
            acc[0] += scs * a.x; acc[1] += scs * a.y; acc[2] += scs * a.z; acc[3] += scs * a.w;
            acc[4] += scs * b.x; acc[5] += scs * b.y; acc[6] += scs * b.z; acc[7] += scs * b.w;
        }
        __nv_bfloat162 o[4];
        o[0] = __floats2bfloat162_rn(acc[0], acc[1]); o[1] = __floats2bfloat162_rn(acc[2], acc[3]);
        o[2] = __floats2bfloat162_rn(acc[4], acc[5]); o[3] = __floats2bfloat162_rn(acc[6], acc[7]);
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
                  torch::Tensor out, torch::Tensor part, int64_t ksplit, int64_t K, int64_t ntiles, int64_t max_blocks, bool gated,
                  int64_t mw, int64_t nt, int64_t ks, int64_t nstage, double gs_mul)
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
#define OKC_W(G, M, T, S, NS) launch_wt<G, M, T, S, NS>(Xp, (int)X.stride(0), sorted_tok.data_ptr<int>(), blk_e.data_ptr<int>(), \
        blk_row0.data_ptr<int>(), blk_end.data_ptr<int>(), nblk.data_ptr<int>(), Wp, Sp, gs.data_ptr<float>(), o16, obf, ldo, \
        pp, ldp, prow, (int)ksplit, (int)K, (int)ntiles, (int)max_blocks, (float)gs_mul, stream)
    const int key = (gated ? 100000 : 0) + (int)mw * 10000 + (int)nt * 1000 + (int)ks * 100 + (int)nstage;
    switch (key) {
        case 111104: OKC_W(true, 1, 1, 1, 4); break;   case 111108: OKC_W(true, 1, 1, 1, 8); break;
        case 112104: OKC_W(true, 1, 2, 1, 4); break;   case 112108: OKC_W(true, 1, 2, 1, 8); break;
        case 114104: OKC_W(true, 1, 4, 1, 4); break;
        case 11104: OKC_W(false, 1, 1, 1, 4); break;   case 11108: OKC_W(false, 1, 1, 1, 8); break;
        case 21104: OKC_W(false, 2, 1, 1, 4); break;   case 21108: OKC_W(false, 2, 1, 1, 8); break;
        case 12104: OKC_W(false, 1, 2, 1, 4); break;   case 22104: OKC_W(false, 2, 2, 1, 4); break;
        case 22108: OKC_W(false, 2, 2, 1, 8); break;   case 14104: OKC_W(false, 1, 4, 1, 4); break;
        case 24104: OKC_W(false, 2, 4, 1, 4); break;
        default: TORCH_CHECK(false, "unsupported wt variant (mw, nt, ks, nstage)");
    }
#undef OKC_W
}

void moe_wt_silu_reduce(torch::Tensor part, int64_t ksplit, torch::Tensor out, torch::Tensor blk_row0, torch::Tensor blk_end,
                        torch::Tensor nblk, int64_t rows_per_block, int64_t max_blocks, int64_t I)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((unsigned)rows_per_block, (unsigned)max_blocks);
    wt_silu_reduce_kernel<<<grid, 256, 0, stream>>>(part.data_ptr<float>(), (int)part.stride(1), (int)part.size(1), (int)ksplit,
        reinterpret_cast<__half*>(out.data_ptr()), (int)out.stride(0), blk_row0.data_ptr<int>(), blk_end.data_ptr<int>(),
        nblk.data_ptr<int>(), (int)I);
}

void moe_finalize(torch::Tensor y, torch::Tensor part, int64_t ksplit, torch::Tensor row_of_pair, torch::Tensor tw,
                  torch::Tensor nw, torch::Tensor out, int64_t M, double eps)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    const int H = (int)out.size(1);
    TORCH_CHECK(H % 2048 == 0, "finalize: H must be a multiple of 2048");
    const int smem = 8 * H * (int)sizeof(float);
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(moe_finalize_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        attr_set = true;
    }
    const float* pp = ksplit > 1 ? part.data_ptr<float>() : nullptr;
    moe_finalize_kernel<<<(unsigned)M, 256, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(y.data_ptr()), (int)y.stride(0), pp, ksplit > 1 ? (int)part.stride(1) : 0,
        ksplit > 1 ? (int)part.size(1) : 0, (int)ksplit, row_of_pair.data_ptr<int>(),
        tw.data_ptr<float>(), nw.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), H, (float)eps);
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
                  torch::Tensor out, torch::Tensor part, int64_t ksplit, int64_t K, int64_t ntiles, int64_t max_blocks, bool gated,
                  int64_t mw, int64_t nt, int64_t ks, int64_t nstage, double gs_mul);
void moe_wt_silu_reduce(torch::Tensor part, int64_t ksplit, torch::Tensor out, torch::Tensor blk_row0, torch::Tensor blk_end,
                        torch::Tensor nblk, int64_t rows_per_block, int64_t max_blocks, int64_t I);
void moe_finalize(torch::Tensor y, torch::Tensor part, int64_t ksplit, torch::Tensor row_of_pair, torch::Tensor tw,
                  torch::Tensor nw, torch::Tensor out, int64_t M, double eps);
"""

_EXT = None


def build():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline
        _EXT = load_inline(
            name="okc_moe_w4a16", cpp_sources=[CPP_SRC],
            cuda_sources=["#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n" + CUDA_SRC],
            functions=["moe_w4a16_gemm", "moe_w4a16_wt", "moe_wt_silu_reduce", "moe_finalize"], extra_cuda_cflags=["-O3", "-arch=sm_90a", "--use_fast_math"], verbose=False)
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
