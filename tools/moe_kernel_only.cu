#ifndef TORCH_CHECK
#define TORCH_CHECK(...)
#endif

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

template void launch<true, 2, 2, 1>(const __half*, int, const int*, const int*, const int*, const int*, const int*, const uint4*, const uint2*, const float*, __half*, __nv_bfloat16*, float*, int, int, int, int, int, cudaStream_t);
template void launch<false, 2, 2, 1>(const __half*, int, const int*, const int*, const int*, const int*, const int*, const uint4*, const uint2*, const float*, __half*, __nv_bfloat16*, float*, int, int, int, int, int, cudaStream_t);
template void launch_wt<true, 1, 1, 1, 2, false, 0>(const __half*, int, const int*, const int*, const int*, const int*, const int*, const int*, const uint8_t*, const uint8_t*, const float*, __half*, __nv_bfloat16*, int, float*, int, int, int*, int, int, int, int, float, bool, bool, cudaStream_t);
template void launch_wt<true, 1, 1, 1, 4, false, 0>(const __half*, int, const int*, const int*, const int*, const int*, const int*, const int*, const uint8_t*, const uint8_t*, const float*, __half*, __nv_bfloat16*, int, float*, int, int, int*, int, int, int, int, float, bool, bool, cudaStream_t);
template void launch_wt<true, 1, 1, 1, 8, false, 0>(const __half*, int, const int*, const int*, const int*, const int*, const int*, const int*, const uint8_t*, const uint8_t*, const float*, __half*, __nv_bfloat16*, int, float*, int, int, int*, int, int, int, int, float, bool, bool, cudaStream_t);
template void launch_wt<true, 1, 2, 1, 2, false, 0>(const __half*, int, const int*, const int*, const int*, const int*, const int*, const int*, const uint8_t*, const uint8_t*, const float*, __half*, __nv_bfloat16*, int, float*, int, int, int*, int, int, int, int, float, bool, bool, cudaStream_t);
template void launch_wt<true, 1, 2, 1, 4, false, 0>(const __half*, int, const int*, const int*, const int*, const int*, const int*, const int*, const uint8_t*, const uint8_t*, const float*, __half*, __nv_bfloat16*, int, float*, int, int, int*, int, int, int, int, float, bool, bool, cudaStream_t);
template void launch_wt<true, 1, 2, 1, 3, false, 0>(const __half*, int, const int*, const int*, const int*, const int*, const int*, const int*, const uint8_t*, const uint8_t*, const float*, __half*, __nv_bfloat16*, int, float*, int, int, int*, int, int, int, int, float, bool, bool, cudaStream_t);
template void launch_wt<false, 2, 2, 1, 3, false, 0>(const __half*, int, const int*, const int*, const int*, const int*, const int*, const int*, const uint8_t*, const uint8_t*, const float*, __half*, __nv_bfloat16*, int, float*, int, int, int*, int, int, int, int, float, bool, bool, cudaStream_t);
template void launch_wt<true, 1, 4, 1, 2, false, 0>(const __half*, int, const int*, const int*, const int*, const int*, const int*, const int*, const uint8_t*, const uint8_t*, const float*, __half*, __nv_bfloat16*, int, float*, int, int, int*, int, int, int, int, float, bool, bool, cudaStream_t);
template void launch_wt<true, 1, 4, 1, 4, false, 0>(const __half*, int, const int*, const int*, const int*, const int*, const int*, const int*, const uint8_t*, const uint8_t*, const float*, __half*, __nv_bfloat16*, int, float*, int, int, int*, int, int, int, int, float, bool, bool, cudaStream_t);
template void launch_wt<false, 1, 1, 1, 2, false, 0>(const __half*, int, const int*, const int*, const int*, const int*, const int*, const int*, const uint8_t*, const uint8_t*, const float*, __half*, __nv_bfloat16*, int, float*, int, int, int*, int, int, int, int, float, bool, bool, cudaStream_t);
template void launch_wt<false, 1, 1, 1, 4, false, 0>(const __half*, int, const int*, const int*, const int*, const int*, const int*, const int*, const uint8_t*, const uint8_t*, const float*, __half*, __nv_bfloat16*, int, float*, int, int, int*, int, int, int, int, float, bool, bool, cudaStream_t);
template void launch_wt<false, 1, 1, 1, 8, false, 0>(const __half*, int, const int*, const int*, const int*, const int*, const int*, const int*, const uint8_t*, const uint8_t*, const float*, __half*, __nv_bfloat16*, int, float*, int, int, int*, int, int, int, int, float, bool, bool, cudaStream_t);
template void launch_wt<false, 2, 1, 1, 2, false, 0>(const __half*, int, const int*, const int*, const int*, const int*, const int*, const int*, const uint8_t*, const uint8_t*, const float*, __half*, __nv_bfloat16*, int, float*, int, int, int*, int, int, int, int, float, bool, bool, cudaStream_t);
template void launch_wt<false, 2, 1, 1, 4, false, 0>(const __half*, int, const int*, const int*, const int*, const int*, const int*, const int*, const uint8_t*, const uint8_t*, const float*, __half*, __nv_bfloat16*, int, float*, int, int, int*, int, int, int, int, float, bool, bool, cudaStream_t);
template void launch_wt<false, 1, 2, 1, 4, false, 0>(const __half*, int, const int*, const int*, const int*, const int*, const int*, const int*, const uint8_t*, const uint8_t*, const float*, __half*, __nv_bfloat16*, int, float*, int, int, int*, int, int, int, int, float, bool, bool, cudaStream_t);
template void launch_wt<false, 2, 2, 1, 2, false, 0>(const __half*, int, const int*, const int*, const int*, const int*, const int*, const int*, const uint8_t*, const uint8_t*, const float*, __half*, __nv_bfloat16*, int, float*, int, int, int*, int, int, int, int, float, bool, bool, cudaStream_t);
template void launch_wt<false, 2, 2, 1, 4, false, 0>(const __half*, int, const int*, const int*, const int*, const int*, const int*, const int*, const uint8_t*, const uint8_t*, const float*, __half*, __nv_bfloat16*, int, float*, int, int, int*, int, int, int, int, float, bool, bool, cudaStream_t);
template void launch_wt<false, 1, 4, 1, 4, false, 0>(const __half*, int, const int*, const int*, const int*, const int*, const int*, const int*, const uint8_t*, const uint8_t*, const float*, __half*, __nv_bfloat16*, int, float*, int, int, int*, int, int, int, int, float, bool, bool, cudaStream_t);
template void launch_wt<false, 2, 4, 1, 2, false, 0>(const __half*, int, const int*, const int*, const int*, const int*, const int*, const int*, const uint8_t*, const uint8_t*, const float*, __half*, __nv_bfloat16*, int, float*, int, int, int*, int, int, int, int, float, bool, bool, cudaStream_t);
template void launch_wt<false, 2, 4, 1, 4, false, 0>(const __half*, int, const int*, const int*, const int*, const int*, const int*, const int*, const uint8_t*, const uint8_t*, const float*, __half*, __nv_bfloat16*, int, float*, int, int, int*, int, int, int, int, float, bool, bool, cudaStream_t);
template void launch_wt<true, 1, 2, 1, 4, false, 1>(const __half*, int, const int*, const int*, const int*, const int*, const int*, const int*, const uint8_t*, const uint8_t*, const float*, __half*, __nv_bfloat16*, int, float*, int, int, int*, int, int, int, int, float, bool, bool, cudaStream_t);
template void launch_wt<false, 2, 2, 1, 4, false, 1>(const __half*, int, const int*, const int*, const int*, const int*, const int*, const int*, const uint8_t*, const uint8_t*, const float*, __half*, __nv_bfloat16*, int, float*, int, int, int*, int, int, int, int, float, bool, bool, cudaStream_t);
template void launch_wt<true, 1, 2, 1, 4, false, 2>(const __half*, int, const int*, const int*, const int*, const int*, const int*, const int*, const uint8_t*, const uint8_t*, const float*, __half*, __nv_bfloat16*, int, float*, int, int, int*, int, int, int, int, float, bool, bool, cudaStream_t);
template void launch_wt<false, 2, 2, 1, 4, false, 2>(const __half*, int, const int*, const int*, const int*, const int*, const int*, const int*, const uint8_t*, const uint8_t*, const float*, __half*, __nv_bfloat16*, int, float*, int, int, int*, int, int, int, int, float, bool, bool, cudaStream_t);
template void launch_wt<true, 1, 2, 1, 4, false, 3>(const __half*, int, const int*, const int*, const int*, const int*, const int*, const int*, const uint8_t*, const uint8_t*, const float*, __half*, __nv_bfloat16*, int, float*, int, int, int*, int, int, int, int, float, bool, bool, cudaStream_t);
template void launch_wt<false, 2, 2, 1, 4, false, 3>(const __half*, int, const int*, const int*, const int*, const int*, const int*, const int*, const uint8_t*, const uint8_t*, const float*, __half*, __nv_bfloat16*, int, float*, int, int, int*, int, int, int, int, float, bool, bool, cudaStream_t);
template void launch_wt<true, 1, 1, 1, 4, false, 1>(const __half*, int, const int*, const int*, const int*, const int*, const int*, const int*, const uint8_t*, const uint8_t*, const float*, __half*, __nv_bfloat16*, int, float*, int, int, int*, int, int, int, int, float, bool, bool, cudaStream_t);
template void launch_wt<false, 1, 1, 1, 4, false, 1>(const __half*, int, const int*, const int*, const int*, const int*, const int*, const int*, const uint8_t*, const uint8_t*, const float*, __half*, __nv_bfloat16*, int, float*, int, int, int*, int, int, int, int, float, bool, bool, cudaStream_t);
template void launch_wt<true, 1, 1, 1, 4, false, 2>(const __half*, int, const int*, const int*, const int*, const int*, const int*, const int*, const uint8_t*, const uint8_t*, const float*, __half*, __nv_bfloat16*, int, float*, int, int, int*, int, int, int, int, float, bool, bool, cudaStream_t);
template void launch_wt<false, 1, 1, 1, 4, false, 2>(const __half*, int, const int*, const int*, const int*, const int*, const int*, const int*, const uint8_t*, const uint8_t*, const float*, __half*, __nv_bfloat16*, int, float*, int, int, int*, int, int, int, int, float, bool, bool, cudaStream_t);
template void launch_wt<true, 1, 1, 1, 4, false, 3>(const __half*, int, const int*, const int*, const int*, const int*, const int*, const int*, const uint8_t*, const uint8_t*, const float*, __half*, __nv_bfloat16*, int, float*, int, int, int*, int, int, int, int, float, bool, bool, cudaStream_t);
template void launch_wt<false, 1, 1, 1, 4, false, 3>(const __half*, int, const int*, const int*, const int*, const int*, const int*, const int*, const uint8_t*, const uint8_t*, const float*, __half*, __nv_bfloat16*, int, float*, int, int, int*, int, int, int, int, float, bool, bool, cudaStream_t);
void* finalize_ref = (void*)&moe_finalize_kernel<4096>;
} // namespace
