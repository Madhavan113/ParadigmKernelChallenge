#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

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

// == BEGIN g12 device ==
// ---- g12: 12-bit W x BF16 x^T for m = 128 * TB tokens on wgmma. CTA = RB x TB warpgroups: warpgroup (rb, tb) owns 64
// weight rows (M) x 128 tokens (N) with a 64 x 128 fp32 accumulator (64 regs / thread). A = the 12-bit fragments decoded
// from the weight ring (T12 layout with R = 64*RB rows, KC = 128*CK: block ((rb*4 + wq)*CK + ck) of 3 KB per warp wq,
// k16 step s = 4h + 2ps + q of the block; decode4 yields the m16n8k16 A fragment, which stacked over the 4 warps is the
// wgmma m64k16 A register fragment), register double-buffered (wgmma.wait_group 1 before reuse). B = the x tile of the
// chunk in shared memory from the 3D TMA map (boxes [64 k][128*TB rows][128 B], 128B swizzle): matrix descriptor
// K-major / SW128 (SBO = 1024 B between 8-row groups), start = box + tb * 16 KB + 32 B * (k16 step within the box).
// gridDim.y CTAs split K; each writes its fp32 partial part[z][n][tok] (float2 over token pairs); g12_reduce_kernel
// (PDL) sums the splits, adds the row's escapes (esc_roff / esc_k / esc_d) and writes bf16 out[tok][n].
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }
__device__ __forceinline__ void reg_fence4(uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3) {
    asm volatile("" : "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3) :: "memory");
}
__device__ __forceinline__ uint64_t g12_desc(uint32_t saddr) {       // K-major, 128B swizzle, SBO 1024 B, LBO unused
    return (uint64_t)((saddr >> 4) & 0x3FFFu) | (1ull << 16) | (64ull << 32) | (1ull << 62);
}
__device__ __forceinline__ void wgmma_m64n128k16_rs(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %69, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, p, 1, 1, 0;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}
__device__ __forceinline__ void acc_fence(float* d) {
    asm volatile("" : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]) :: "memory");
}
template <int RB, int TB, int CK, int NSW, int NSX, bool DIAG>
__global__ void __launch_bounds__(128 * RB * TB, 1)
g12_kernel(const __grid_constant__ CUtensorMap xmap, int m, const uint8_t* __restrict__ Wt, int eb,
           float* __restrict__ part, int N, int K, unsigned long long* __restrict__ diag)
{
    constexpr int WG = RB * TB, R = 64 * RB, MP = 128 * TB;             // warpgroups, weight rows per CTA, token rows in smem
    constexpr int KC = 128 * CK, NBOX = KC / 64, BOXB = MP * 128;
    constexpr int WST = RB * 4 * CK * 3072, XST = NBOX * BOXB;
    extern __shared__ __align__(1024) uint8_t g12_smem[];
    __shared__ __align__(8) unsigned long long full[NSW], xfull[NSX];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int wg = warp >> 2, wq = warp & 3, rb = wg / TB, tb = wg % TB;
    const int tile = blockIdx.x, z = blockIdx.y, KS = gridDim.y;
    const int NC_ALL = K / KC, NC = NC_ALL / KS, c0 = z * NC;
    unsigned long long t0 = 0, t1 = 0;
    if constexpr (DIAG) t0 = gtimer();
    const uint32_t sw0 = smem_u32(g12_smem);
    const uint32_t sw = (sw0 + 1023u) & ~1023u, sx = sw + NSW * WST;
    uint8_t* smem = g12_smem + (sw - sw0);
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < NSW; ++s) mbar_init(smem_u32(&full[s]), 1);
        #pragma unroll
        for (int s = 0; s < NSX; ++s) mbar_init(smem_u32(&xfull[s]), 1);
        fence_mbar_init();
    }
    __syncthreads();
    const uint8_t* wsrc = Wt + ((size_t)tile * NC_ALL + c0) * WST;
    auto issue_x = [&](int c, int st) {
        const uint32_t bar = smem_u32(&xfull[st]);
        mbar_expect_tx(bar, XST);
        tma_3d_g2s(sx + st * XST, &xmap, 0, 0, (c0 + c) * NBOX, bar);
    };
    auto issue_w = [&](int c, int st) {
        const uint32_t bar = smem_u32(&full[st]);
        mbar_expect_tx(bar, WST);
        bulk_g2s(sw + st * WST, wsrc + (size_t)c * WST, WST, bar);
    };
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < NSX; ++s) if (s < NC) issue_x(s, s);
        #pragma unroll
        for (int s = 0; s < NSW; ++s) if (s < NC) issue_w(s, s);
    }
    const uint32_t hb = (uint32_t)(eb >> 1);
    const uint32_t lut0 = hb | (hb + 1) << 8 | (hb + 2) << 16 | (hb + 3) << 24;
    const uint32_t lut1 = (hb + 4) | (hb + 5) << 8 | (hb + 6) << 16 | (hb + 7) << 24;
    // this warp's 3 KB blocks within a stage: block (rb*4 + wq) of each 128-k sub-chunk ck
    const int wblk = (rb * 4 + wq) * CK;
    float acc[64];
    #pragma unroll
    for (int i = 0; i < 64; ++i) acc[i] = 0.f;
    uint32_t a[2][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) { a[0][i] = 0u; a[1][i] = 0u; }
    int s = 0;                                                          // k16 step counter (register set s & 1)
    for (int c = 0; c < NC; ++c) {
        const int swi = c % NSW, sxi = c % NSX;
        __syncthreads();                                                // chunk c-1 fully consumed (wgmma waited below)
        if (c > 0 && tid == 0) {
            if (c - 1 + NSX < NC) issue_x(c - 1 + NSX, (c - 1) % NSX);
            if (c - 1 + NSW < NC) issue_w(c - 1 + NSW, (c - 1) % NSW);
        }
        mbar_wait(smem_u32(&xfull[sxi]), (uint32_t)((c / NSX) & 1));
        mbar_wait(smem_u32(&full[swi]), (uint32_t)((c / NSW) & 1));
        if constexpr (DIAG) { if (c == 0) t1 = gtimer(); }
        const uint8_t* ws = smem + swi * WST;
        const uint32_t xs = sx + sxi * XST + tb * 16384u;
        #pragma unroll
        for (int ck = 0; ck < CK; ++ck) {
            const uint8_t* blk = ws + (wblk + ck) * 3072;
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint4 lo0 = *reinterpret_cast<const uint4*>(blk + (2 * h) * 512 + lane * 16);
                const uint4 lo1 = *reinterpret_cast<const uint4*>(blk + (2 * h + 1) * 512 + lane * 16);
                const uint4 nb = *reinterpret_cast<const uint4*>(blk + 2048 + h * 512 + lane * 16);
                const uint32_t box = xs + (ck * 2 + h) * BOXB;
                #pragma unroll
                for (int ps = 0; ps < 2; ++ps) {
                    const uint4 lo = ps == 0 ? lo0 : lo1;
                    #pragma unroll
                    for (int q = 0; q < 2; ++q) {
                        const uint32_t L0 = q == 0 ? lo.x : lo.z, L1 = q == 0 ? lo.y : lo.w;
                        const int si = 2 * ps + q;
                        const uint32_t Nw = si == 0 ? nb.x : si == 1 ? nb.y : si == 2 ? nb.z : nb.w;
                        uint32_t* ar = a[s & 1];
                        wgmma_wait<1>();                                // the wgmma that read this register set (step s-2) is done
                        reg_fence4(ar[0], ar[1], ar[2], ar[3]);
                        decode4(L0, Nw & 0xFFFFu, lut0, lut1, ar[0], ar[1]);
                        decode4(L1, Nw >> 16, lut0, lut1, ar[2], ar[3]);
                        wgmma_fence();
                        wgmma_m64n128k16_rs(acc, ar[0], ar[1], ar[2], ar[3], g12_desc(box + ps * 64 + q * 32));
                        wgmma_commit();
                        ++s;
                    }
                }
            }
        }
        wgmma_wait<0>();                                                // the stage may be refilled after the next barrier
        reg_fence4(a[0][0], a[0][1], a[0][2], a[0][3]);
        reg_fence4(a[1][0], a[1][1], a[1][2], a[1][3]);
    }
    acc_fence(acc);
    unsigned long long t2 = 0;
    if constexpr (DIAG) t2 = gtimer();
    pdl_trigger();
    // partials: part[z][n][tok], n = tile*R + rb*64 + wq*16 + g (+8), tok = tb*128 + 8j + 2t (+1)
    float* pz = part + ((size_t)z * N + (size_t)tile * R + rb * 64 + wq * 16 + g) * m + tb * 128 + 2 * t;
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int tok = tb * 128 + 8 * j + 2 * t;
        if (tok < m) {
            *reinterpret_cast<float2*>(pz + 8 * j) = make_float2(acc[4 * j], acc[4 * j + 1]);
            *reinterpret_cast<float2*>(pz + (size_t)8 * m + 8 * j) = make_float2(acc[4 * j + 2], acc[4 * j + 3]);
        }
    }
    if constexpr (DIAG) {
        __syncthreads();
        if (tid == 0) {
            const unsigned long long t3 = gtimer();
            if (tile == 0 && z == 0) { diag[0] = t0; diag[1] = t1; diag[2] = t2; diag[3] = t3; }
            atomicMin(diag + 4, t0); atomicMax(diag + 5, t3);
            if (tile == gridDim.x - 1 && z == KS - 1) { diag[6] = t0; diag[7] = t3; }
        }
    }
}

// sums the K splits of part[z][n][tok] (coalesced over tok), adds row n's escapes, writes bf16 out[tok][n]
__global__ void __launch_bounds__(256)
g12_reduce_kernel(const float* __restrict__ part, int KS, int m, int N,
                  const int* __restrict__ esc_roff, const int* __restrict__ esc_k, const float* __restrict__ esc_d,
                  const __nv_bfloat16* __restrict__ X, int ldx, __nv_bfloat16* __restrict__ out, int ldo)
{
    const int n = blockIdx.x, tok = threadIdx.x;
    if (tok >= m) return;
    const int e0 = esc_roff[n], e1 = esc_roff[n + 1];
    float ev = 0.f;
    for (int e = e0; e < e1; ++e) ev += esc_d[e] * __bfloat162float(X[(size_t)tok * ldx + esc_k[e]]);
    pdl_wait();
    float sum = ev;
    for (int zz = 0; zz < KS; ++zz) sum += __ldcg(part + ((size_t)zz * N + n) * m + tok);
    out[(size_t)tok * ldo + n] = __float2bfloat16(sum);
}

__host__ constexpr int g12_smem_bytes(int rb, int tb, int ck, int nsw, int nsx) { return 1024 + nsw * rb * 4 * ck * 3072 + nsx * (2 * ck) * (128 * tb) * 128; }

template <int RB, int TB, int CK, int NSW, int NSX, bool DIAG>
void launch_g12(const CUtensorMap& xmap, int m, const uint8_t* Wt, int eb, float* part, int N, int K, int ks,
                unsigned long long* diag, cudaStream_t stream)
{
    constexpr int smem = g12_smem_bytes(RB, TB, CK, NSW, NSX);
    static bool attr = false;
    if (!attr) { cudaFuncSetAttribute(g12_kernel<RB, TB, CK, NSW, NSX, DIAG>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
    g12_kernel<RB, TB, CK, NSW, NSX, DIAG><<<dim3(N / (64 * RB), ks), 128 * RB * TB, smem, stream>>>(xmap, m, Wt, eb, part, N, K, diag);
}
// == END g12 device ==

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

// == BEGIN g12 host ==
void g12(torch::Tensor X, torch::Tensor Wt, int64_t eb, torch::Tensor esc_roff, torch::Tensor esc_k, torch::Tensor esc_d,
         torch::Tensor out, torch::Tensor part, torch::Tensor diag, int64_t n, int64_t k,
         int64_t rb, int64_t tb, int64_t ck, int64_t nsw, int64_t nsx, int64_t ks, bool diag_on, bool pdl, int64_t mode)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    const int m = (int)X.size(0);
    TORCH_CHECK(2 <= m && m <= 128 * tb && m % 2 == 0 && n % (64 * rb) == 0 && k % (128 * ck * ks) == 0 && (k / (128 * ck)) / ks >= 1, "g12: shape/config mismatch");
    TORCH_CHECK(X.size(1) == k && X.stride(1) == 1 && (X.stride(0) * 2) % 16 == 0 && (reinterpret_cast<uintptr_t>(X.data_ptr()) & 15) == 0, "g12: x must be 16 B aligned rows");
    TORCH_CHECK(Wt.is_contiguous() && (reinterpret_cast<uintptr_t>(Wt.data_ptr()) & 127) == 0 && Wt.numel() * Wt.element_size() >= n * k * 3 / 2, "g12: weight layout");
    TORCH_CHECK(out.size(0) == m && out.size(1) == n && out.stride(1) == 1, "g12: output");
    TORCH_CHECK(part.numel() >= ks * m * n && (reinterpret_cast<uintptr_t>(part.data_ptr()) & 15) == 0, "g12: partials workspace");
    TORCH_CHECK(esc_roff.numel() >= n + 1 && esc_k.numel() >= 1 && esc_d.numel() >= 1 && diag.numel() >= 8, "g12: escapes / diag");
    const uint8_t* Wp = reinterpret_cast<const uint8_t*>(Wt.data_ptr());
    float* pp = part.data_ptr<float>();
    unsigned long long* dg = reinterpret_cast<unsigned long long*>(diag.data_ptr());
    const int ldx = (int)X.stride(0), ldo = (int)out.stride(0);
    const CUtensorMap xmap = okc_x_map3(X.data_ptr(), m, (int)k, ldx, 128 * (int)tb, 2 * (int)ck);
    bool launched = false;
#define G12_CASE(RB_, TB_, CK_, SW_, SX_) \
    if (!launched && rb == RB_ && tb == TB_ && ck == CK_ && nsw == SW_ && nsx == SX_) { \
        if (diag_on) launch_g12<RB_, TB_, CK_, SW_, SX_, true>(xmap, m, Wp, (int)eb, pp, (int)n, (int)k, (int)ks, dg, stream); \
        else launch_g12<RB_, TB_, CK_, SW_, SX_, false>(xmap, m, Wp, (int)eb, pp, (int)n, (int)k, (int)ks, dg, stream); \
        launched = true; }
    G12_CASE(4, 1, 1, 3, 2)
    G12_CASE(4, 1, 1, 2, 2)
    G12_CASE(2, 1, 1, 3, 3)
    G12_CASE(2, 1, 2, 2, 2)
    G12_CASE(2, 2, 1, 3, 2)
    G12_CASE(1, 2, 1, 4, 2)
#undef G12_CASE
    TORCH_CHECK(launched, "g12: unsupported config (rb, tb, ck, nsw, nsx)");
    if (mode == 2) return;                                               // bench: GEMM only
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)n);
    cfg.blockDim = dim3((unsigned)m);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = pdl ? 1 : 0;
    const cudaError_t err = cudaLaunchKernelEx(&cfg, g12_reduce_kernel, (const float*)pp, (int)ks, m, (int)n,
        esc_roff.data_ptr<int>(), esc_k.data_ptr<int>(), esc_d.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(X.data_ptr()), ldx, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), ldo);
    TORCH_CHECK(err == cudaSuccess, "g12 reduce launch failed: ", cudaGetErrorString(err));
}
// == END g12 host ==


