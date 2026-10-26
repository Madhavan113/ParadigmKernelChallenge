"""CUDA side of muse_bf16_proj (embedded into submission.py between the CUDA markers).

  stream_bytes   dot-free read of a buffer (grid x threads x loads-in-flight sweep): the HBM
                 streaming ceiling the GEMMs are measured against.
  gemv           x[M, K] bf16 @ W[N, K]^T for M <= 8 with no shared memory: every lane keeps
                 UNROLL 16-byte weight loads in flight, unpacks bf16 -> fp32 and FMAs against
                 x (L1-resident); KSPLIT warps share a row and reduce through shared memory;
                 TWO_W adds the router's BF16 lo weight into the same fp32 accumulator.
"""

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
