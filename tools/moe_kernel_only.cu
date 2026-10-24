
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

constexpr int BM = 16;          // rows per block
constexpr int KC = 128;         // K chunk
constexpr int A_STRIDE = KC + 8; // padded smem row stride (halves)
constexpr int NWARPS = 4;

// GATED: warp w owns gate column block nb = tile*4 + w and up block nb + I/8 (I = N/2 output
// cols per half); output cols [8*nb, 8*nb+8). Else: warp w owns blocks tile*8 + 2w, +1.
template <bool GATED>
__global__ void __launch_bounds__(128)
moe_w4a16_gemm_kernel(const __half* __restrict__ A, int lda,            // fp16 activations
                      const int* __restrict__ sorted_tok,               // row -> A row (GATED) ; identity otherwise
                      const int* __restrict__ blk_e, const int* __restrict__ blk_row0,
                      const int* __restrict__ blk_end, const int* __restrict__ nblk,
                      const uint4* __restrict__ Wr,                     // [E][N/8][K/128][32 lanes] uint4
                      const uint2* __restrict__ Sr,                     // [E][N/8][K/128][8 g] uint2 (8 scale bytes)
                      const float* __restrict__ gs,                     // [E] global scale x 64
                      __half* __restrict__ out16, __nv_bfloat16* __restrict__ out_bf, int ldo,
                      int K, int N)
{
    const int j = blockIdx.y;
    if (j >= *nblk) return;
    const int e = blk_e[j];
    const int row0 = blk_row0[j];
    const int end = blk_end[j];
    const int tile = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;

    __shared__ __align__(16) __half As[2][BM * A_STRIDE];

    const int NB = N / 8;              // n8 blocks per expert
    const int NC = K / KC;             // chunks
    int nb0, nb1;
    if (GATED) { nb0 = tile * NWARPS + warp; nb1 = nb0 + NB / 2; }
    else       { nb0 = tile * (2 * NWARPS) + 2 * warp; nb1 = nb0 + 1; }
    const uint4* W0 = Wr + ((size_t)e * NB + nb0) * NC * 32 + lane;
    const uint4* W1 = Wr + ((size_t)e * NB + nb1) * NC * 32 + lane;
    const uint2* S0 = Sr + ((size_t)e * NB + nb0) * NC * 8 + g;
    const uint2* S1 = Sr + ((size_t)e * NB + nb1) * NC * 8 + g;
    // global scale x 64 as f16x2
    const __half gsh = __float2half(gs[e]);
    const uint32_t gs2 = (uint32_t)__half_as_ushort(gsh) * 0x00010001u;

    // A rows of this block
    const int arow = row0 + (threadIdx.x >> 3);          // 16 rows x 8 segments of 16 halves... (128 threads: row = tid/8)
    const int aseg = threadIdx.x & 7;                   // 8 x 16B = 128B = 64 halves per thread-pair: each thread copies 2 x 16B
    const bool avalid = arow < end;
    const int atok = avalid ? (GATED ? sorted_tok[arow] : arow) : 0;
    const __half* Ag = A + (size_t)atok * lda;

    auto issue_A = [&](int c, int buf) {
        // chunk c: halves [c*128, c*128+128) of each row; 16 rows x 256B; thread copies 2 x 16B
        const int rloc = threadIdx.x >> 3;
        #pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int col = c * KC + (aseg + 8 * q) * 8;     // halves
            uint32_t dst = smem_u32(&As[buf][rloc * A_STRIDE + (aseg + 8 * q) * 8]);
            cp_async16(dst, Ag + col, avalid ? 16 : 0);
        }
    };

    float acc0[4] = {0.f, 0.f, 0.f, 0.f}, acc1[4] = {0.f, 0.f, 0.f, 0.f};
    uint4 wq0 = W0[0], wq1 = W1[0];
    uint2 sq0 = S0[0], sq1 = S1[0];
    issue_A(0, 0);
    cp_async_commit();

    for (int c = 0; c < NC; ++c) {
        const int buf = c & 1;
        if (c + 1 < NC) { issue_A(c + 1, buf ^ 1); }
        cp_async_commit();
        uint4 wn0, wn1; uint2 sn0, sn1;
        if (c + 1 < NC) { wn0 = W0[(c + 1) * 32]; wn1 = W1[(c + 1) * 32]; sn0 = S0[(c + 1) * 8]; sn1 = S1[(c + 1) * 8]; }
        cp_async_wait<1>();
        __syncthreads();
        // scales for the 8 steps: 8 e4m3 bytes -> 4 f16x2 (steps 2i, 2i+1) x gs
        uint32_t s0[4], s1[4];
        s0[0] = f16x2_mul(e4m3x2_to_f16x2(sq0.x & 0xFFFFu), gs2);  s0[1] = f16x2_mul(e4m3x2_to_f16x2(sq0.x >> 16), gs2);
        s0[2] = f16x2_mul(e4m3x2_to_f16x2(sq0.y & 0xFFFFu), gs2);  s0[3] = f16x2_mul(e4m3x2_to_f16x2(sq0.y >> 16), gs2);
        s1[0] = f16x2_mul(e4m3x2_to_f16x2(sq1.x & 0xFFFFu), gs2);  s1[1] = f16x2_mul(e4m3x2_to_f16x2(sq1.x >> 16), gs2);
        s1[2] = f16x2_mul(e4m3x2_to_f16x2(sq1.y & 0xFFFFu), gs2);  s1[3] = f16x2_mul(e4m3x2_to_f16x2(sq1.y >> 16), gs2);
        const uint32_t w0[4] = {wq0.x, wq0.y, wq0.z, wq0.w};
        const uint32_t w1[4] = {wq1.x, wq1.y, wq1.z, wq1.w};
        #pragma unroll
        for (int p = 0; p < 4; ++p) {           // word p = steps 2p, 2p+1
            // A fragments for steps 2p and 2p+1
            uint32_t a[2][4];
            #pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int step = 2 * p + q;
                const int r = lane & 15, kk = step * 16 + (lane >> 4) * 8;
                ldmatrix_x4(a[q][0], a[q][1], a[q][2], a[q][3], smem_u32(&As[buf][r * A_STRIDE + kk]));
            }
            // scale f16x2 (s, s) for steps 2p (low half of s[p]) and 2p+1 (high half)
            const uint32_t s0lo = (s0[p] & 0xFFFFu) * 0x00010001u, s0hi = (s0[p] >> 16) * 0x00010001u;
            const uint32_t s1lo = (s1[p] & 0xFFFFu) * 0x00010001u, s1hi = (s1[p] >> 16) * 0x00010001u;
            uint32_t b0a, b1a, b0b, b1b;
            dequant_word(w0[p], s0lo, s0hi, b0a, b1a, b0b, b1b);
            mma16816(acc0, a[0], b0a, b1a);
            mma16816(acc0, a[1], b0b, b1b);
            dequant_word(w1[p], s1lo, s1hi, b0a, b1a, b0b, b1b);
            mma16816(acc1, a[0], b0a, b1a);
            mma16816(acc1, a[1], b0b, b1b);
        }
        __syncthreads();
        wq0 = wn0; wq1 = wn1; sq0 = sn0; sq1 = sn1;
    }

    // epilogue: lane holds rows g (acc[0], acc[1]) and g+8 (acc[2], acc[3]) at cols 2t, 2t+1
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int r = row0 + g + 8 * h;
        if (r >= end) continue;
        if (GATED) {
            const int col = nb0 * 8 + 2 * t;
            float g0 = acc0[2 * h], g1 = acc0[2 * h + 1], u0 = acc1[2 * h], u1 = acc1[2 * h + 1];
            float h0 = g0 / (1.f + __expf(-g0)) * u0, h1 = g1 / (1.f + __expf(-g1)) * u1;
            // BF16 precision (as SGLang's path keeps it), stored as fp16 for the next GEMM
            __half2 v = __halves2half2(__float2half(__bfloat162float(__float2bfloat16(h0))),
                                       __float2half(__bfloat162float(__float2bfloat16(h1))));
            *reinterpret_cast<__half2*>(out16 + (size_t)r * ldo + col) = v;
        } else {
            const int col0 = nb0 * 8 + 2 * t, col1 = nb1 * 8 + 2 * t;
            *reinterpret_cast<__nv_bfloat162*>(out_bf + (size_t)r * ldo + col0) =
                __floats2bfloat162_rn(acc0[2 * h], acc0[2 * h + 1]);
            *reinterpret_cast<__nv_bfloat162*>(out_bf + (size_t)r * ldo + col1) =
                __floats2bfloat162_rn(acc1[2 * h], acc1[2 * h + 1]);
        }
    }
}


template __global__ void moe_w4a16_gemm_kernel<true>(const __half*, int, const int*, const int*, const int*, const int*, const int*, const uint4*, const uint2*, const float*, __half*, __nv_bfloat16*, int, int, int);
template __global__ void moe_w4a16_gemm_kernel<false>(const __half*, int, const int*, const int*, const int*, const int*, const int*, const uint4*, const uint2*, const float*, __half*, __nv_bfloat16*, int, int, int);
} // namespace

