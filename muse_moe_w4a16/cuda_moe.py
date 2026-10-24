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

} // namespace

void moe_w4a16_gemm(torch::Tensor A, torch::Tensor sorted_tok, torch::Tensor blk_e, torch::Tensor blk_row0,
                    torch::Tensor blk_end, torch::Tensor nblk, torch::Tensor Wr, torch::Tensor Sr, torch::Tensor gs,
                    torch::Tensor out, int64_t K, int64_t N, int64_t max_blocks, bool gated)
{
    const int n_tiles = gated ? (int)(N / 2 / 32) : (int)(N / 64);
    dim3 grid(n_tiles, (unsigned)max_blocks);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (gated) {
        moe_w4a16_gemm_kernel<true><<<grid, 128, 0, stream>>>(
            reinterpret_cast<const __half*>(A.data_ptr()), (int)A.stride(0), sorted_tok.data_ptr<int>(),
            blk_e.data_ptr<int>(), blk_row0.data_ptr<int>(), blk_end.data_ptr<int>(), nblk.data_ptr<int>(),
            reinterpret_cast<const uint4*>(Wr.data_ptr()), reinterpret_cast<const uint2*>(Sr.data_ptr()),
            gs.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr()), nullptr, (int)out.stride(0), (int)K, (int)N);
    } else {
        moe_w4a16_gemm_kernel<false><<<grid, 128, 0, stream>>>(
            reinterpret_cast<const __half*>(A.data_ptr()), (int)A.stride(0), sorted_tok.data_ptr<int>(),
            blk_e.data_ptr<int>(), blk_row0.data_ptr<int>(), blk_end.data_ptr<int>(), nblk.data_ptr<int>(),
            reinterpret_cast<const uint4*>(Wr.data_ptr()), reinterpret_cast<const uint2*>(Sr.data_ptr()),
            gs.data_ptr<float>(), nullptr, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (int)out.stride(0), (int)K, (int)N);
    }
}
"""

CPP_SRC = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
void moe_w4a16_gemm(torch::Tensor A, torch::Tensor sorted_tok, torch::Tensor blk_e, torch::Tensor blk_row0,
                    torch::Tensor blk_end, torch::Tensor nblk, torch::Tensor Wr, torch::Tensor Sr, torch::Tensor gs,
                    torch::Tensor out, int64_t K, int64_t N, int64_t max_blocks, bool gated);
"""

_EXT = None


def build():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline
        _EXT = load_inline(
            name="okc_moe_w4a16", cpp_sources=[CPP_SRC],
            cuda_sources=["#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n" + CUDA_SRC],
            functions=["moe_w4a16_gemm"], extra_cuda_cflags=["-O3", "-arch=sm_90a", "--use_fast_math"], verbose=False)
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
