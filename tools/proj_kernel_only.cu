
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

template void launch<16, 16, false>(const __nv_bfloat16*, int, int, const __nv_bfloat16*, const __nv_bfloat16*, int, __nv_bfloat16*, float*, int, int, int, cudaStream_t);
template void launch<16, 16, true>(const __nv_bfloat16*, int, int, const __nv_bfloat16*, const __nv_bfloat16*, int, __nv_bfloat16*, float*, int, int, int, cudaStream_t);
template void launch<16, 32, false>(const __nv_bfloat16*, int, int, const __nv_bfloat16*, const __nv_bfloat16*, int, __nv_bfloat16*, float*, int, int, int, cudaStream_t);
template void launch<16, 32, true>(const __nv_bfloat16*, int, int, const __nv_bfloat16*, const __nv_bfloat16*, int, __nv_bfloat16*, float*, int, int, int, cudaStream_t);
template void launch<32, 16, false>(const __nv_bfloat16*, int, int, const __nv_bfloat16*, const __nv_bfloat16*, int, __nv_bfloat16*, float*, int, int, int, cudaStream_t);
template void launch<32, 16, true>(const __nv_bfloat16*, int, int, const __nv_bfloat16*, const __nv_bfloat16*, int, __nv_bfloat16*, float*, int, int, int, cudaStream_t);
template void launch<32, 32, false>(const __nv_bfloat16*, int, int, const __nv_bfloat16*, const __nv_bfloat16*, int, __nv_bfloat16*, float*, int, int, int, cudaStream_t);
template void launch<32, 32, true>(const __nv_bfloat16*, int, int, const __nv_bfloat16*, const __nv_bfloat16*, int, __nv_bfloat16*, float*, int, int, int, cudaStream_t);
} // namespace

