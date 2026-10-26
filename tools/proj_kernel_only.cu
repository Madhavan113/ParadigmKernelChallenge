#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

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

template <int WARPS, int UNROLL, int SK>
__global__ void __launch_bounds__(WARPS * 32) gemv12_kernel(const __nv_bfloat16* __restrict__ X,
        const uint8_t* __restrict__ LO, const uint8_t* __restrict__ NB, int eb,
        const int* __restrict__ esc_off, const int* __restrict__ esc_n, const int* __restrict__ esc_k, const float* __restrict__ esc_d,
        __nv_bfloat16* __restrict__ out, float* __restrict__ part, int* __restrict__ cnt, int N, int K)
{
    constexpr int RPT = 32 / (2 * WARPS);          // steps (row pairs) per warp per k-tile
    __shared__ float red[32];
    __shared__ int s_last;
    const int pn = blockIdx.x, sk = blockIdx.y;
    const int KT = K / 256, ktc = KT / SK, kt0 = sk * ktc;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, half = lane >> 4, l16 = lane & 15;
    const uint32_t hb = (uint32_t)(eb >> 1);
    const uint32_t lut0 = hb | (hb + 1) << 8 | (hb + 2) << 16 | (hb + 3) << 24;
    const uint32_t lut1 = (hb + 4) | (hb + 5) << 8 | (hb + 6) << 16 | (hb + 7) << 24;

    // this lane's rows: r = 2 * (warp + WARPS * s) + half, s = 0..RPT-1
    const uint8_t* lo_base = LO + ((size_t)(pn * KT + kt0) * 32 + 2 * warp + half) * 256 + l16 * 16;
    const uint8_t* nb_base = NB + ((size_t)(pn * KT + kt0) * 32 + 2 * warp + half) * 128 + l16 * 8;
    const __nv_bfloat16* x_base = X + kt0 * 256 + l16 * 16;
    float acc[RPT];
    #pragma unroll
    for (int s2 = 0; s2 < RPT; ++s2) acc[s2] = 0.f;

    uint4 cl[UNROLL][RPT]; uint2 cn[UNROLL][RPT];
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u)
        #pragma unroll
        for (int s2 = 0; s2 < RPT; ++s2) {
            cn[u][s2] = *reinterpret_cast<const uint2*>(nb_base + ((size_t)u * 32 + 2 * WARPS * s2) * 128);
            cl[u][s2] = ldg_stream(reinterpret_cast<const uint4*>(lo_base + ((size_t)u * 32 + 2 * WARPS * s2) * 256));
        }
    for (int kt = 0; kt < ktc; kt += UNROLL) {
        const int kn = kt + UNROLL < ktc ? kt + UNROLL : kt;
        uint4 nl[UNROLL][RPT]; uint2 nn[UNROLL][RPT];
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u)
            #pragma unroll
            for (int s2 = 0; s2 < RPT; ++s2) {
                nn[u][s2] = *reinterpret_cast<const uint2*>(nb_base + ((size_t)(kn + u) * 32 + 2 * WARPS * s2) * 128);
                nl[u][s2] = ldg_stream(reinterpret_cast<const uint4*>(lo_base + ((size_t)(kn + u) * 32 + 2 * WARPS * s2) * 256));
            }
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const uint4* xp = reinterpret_cast<const uint4*>(x_base + (kt + u) * 256);
            const uint4 x0 = xp[0], x1 = xp[1];
            #pragma unroll
            for (int s2 = 0; s2 < RPT; ++s2) {
                uint32_t w01, w23;
                float a = acc[s2];
                decode4(cl[u][s2].x, cn[u][s2].x & 0xFFFFu, lut0, lut1, w01, w23); a = dot2(w01, x0.x, a); a = dot2(w23, x0.y, a);
                decode4(cl[u][s2].y, cn[u][s2].x >> 16, lut0, lut1, w01, w23);     a = dot2(w01, x0.z, a); a = dot2(w23, x0.w, a);
                decode4(cl[u][s2].z, cn[u][s2].y & 0xFFFFu, lut0, lut1, w01, w23); a = dot2(w01, x1.x, a); a = dot2(w23, x1.y, a);
                decode4(cl[u][s2].w, cn[u][s2].y >> 16, lut0, lut1, w01, w23);     a = dot2(w01, x1.z, a); a = dot2(w23, x1.w, a);
                acc[s2] = a;
            }
        }
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u)
            #pragma unroll
            for (int s2 = 0; s2 < RPT; ++s2) { cl[u][s2] = nl[u][s2]; cn[u][s2] = nn[u][s2]; }
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
    // escapes of this n-tile: staged in parallel (one entry per thread per pass), then row threads scan
    __shared__ int s_en[WARPS * 32];
    __shared__ float s_ev[WARPS * 32];
    const int e0 = esc_off[pn], e1 = esc_off[pn + 1];
    float v = 0.f;
    if (threadIdx.x < 32) {
        if (SK > 1) {
            #pragma unroll
            for (int s2 = 0; s2 < SK; ++s2) v += __ldcg(part + (size_t)s2 * N + pn * 32 + threadIdx.x);
        } else {
            v = red[threadIdx.x];
        }
    }
    for (int base = e0; base < e1; base += WARPS * 32) {
        const int e = base + (int)threadIdx.x;
        if (e < e1) { s_en[threadIdx.x] = esc_n[e]; s_ev[threadIdx.x] = esc_d[e] * __bfloat162float(X[esc_k[e]]); }
        __syncthreads();
        if (threadIdx.x < 32) {
            const int cnt_ = min(e1 - base, WARPS * 32);
            for (int j = 0; j < cnt_; ++j) v += (s_en[j] == (int)threadIdx.x) ? s_ev[j] : 0.f;
        }
        __syncthreads();
    }
    if (threadIdx.x < 32) {
        out[pn * 32 + threadIdx.x] = __float2bfloat16(v);
        if (SK > 1 && threadIdx.x == 0) cnt[pn] = 0;
    }
}

template <int WARPS, int UNROLL, int SK>
void launch_gemv12(const __nv_bfloat16* X, const uint8_t* LO, const uint8_t* NB, int eb, const int* eo, const int* en, const int* ek, const float* ed,
                   __nv_bfloat16* out, float* part, int* cnt, int N, int K, cudaStream_t stream)
{
    gemv12_kernel<WARPS, UNROLL, SK><<<dim3(N / 32, SK), WARPS * 32, 0, stream>>>(X, LO, NB, eb, eo, en, ek, ed, out, part, cnt, N, K);
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
    for (int idx = threadIdx.x; idx < 32 * (MR + 1); idx += 256) (&red[0][0])[idx] = 0.f;
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const int n0 = grp * RPW * 8 + 8 * r + 2 * t;
            atomicAdd(&red[n0][16 * i + g], acc[i][r][0]);
            atomicAdd(&red[n0 + 1][16 * i + g], acc[i][r][1]);
            atomicAdd(&red[n0][16 * i + g + 8], acc[i][r][2]);
            atomicAdd(&red[n0 + 1][16 * i + g + 8], acc[i][r][3]);
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
    constexpr int LOB = RPW * 16, NBB = RPW * 8, XB = MT * 2 * 32, LANEB = LOB + NBB + XB, WSTAGE = 32 * LANEB;
    extern __shared__ __align__(16) uint8_t dsmem[];
    __shared__ float red[32][MR + 1];
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

    for (int idx = threadIdx.x; idx < 32 * (MR + 1); idx += 256) (&red[0][0])[idx] = 0.f;
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const int n0 = grp * RPW * 8 + 8 * r + 2 * t;
            atomicAdd(&red[n0][16 * i + g], acc[i][r][0]);
            atomicAdd(&red[n0 + 1][16 * i + g], acc[i][r][1]);
            atomicAdd(&red[n0][16 * i + g + 8], acc[i][r][2]);
            atomicAdd(&red[n0 + 1][16 * i + g + 8], acc[i][r][3]);
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
    constexpr int LANEB = RPW * 16 + RPW * 8 + MT * 2 * 32;
    constexpr int smem = NSTAGE * 8 * 32 * LANEB;
    static bool attr = false;
    if (!attr) { cudaFuncSetAttribute(mma12cp_kernel<MT, RPW, NSTAGE, SK>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
    mma12cp_kernel<MT, RPW, NSTAGE, SK><<<dim3(N / 32, SK), 256, smem, stream>>>(X, ldx, m, LO, NB, eb, eo, en, ek, ed, out, ldo, part, cnt, N, K);
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
    case (W_ * 100 + U_ * 10 + S_): launch_gemv12<W_, U_, S_>(Xp, LOp, NBp, (int)eb, eo, en, ek, ed, ob, part.data_ptr<float>(), cnt.data_ptr<int>(), N, K, stream); break;

void gemv12(torch::Tensor X, torch::Tensor LO, torch::Tensor NB, int64_t eb, torch::Tensor esc_off, torch::Tensor esc_n, torch::Tensor esc_k, torch::Tensor esc_d,
            torch::Tensor out, torch::Tensor part, torch::Tensor cnt, int64_t warps, int64_t unroll, int64_t sk)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    const int K = (int)X.size(1), N = (int)out.size(1);
    const __nv_bfloat16* Xp = reinterpret_cast<const __nv_bfloat16*>(X.data_ptr());
    const uint8_t* LOp = LO.data_ptr<uint8_t>(); const uint8_t* NBp = NB.data_ptr<uint8_t>();
    const int* eo = esc_off.data_ptr<int>(); const int* en = esc_n.data_ptr<int>(); const int* ek = esc_k.data_ptr<int>(); const float* ed = esc_d.data_ptr<float>();
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
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
        default: TORCH_CHECK(false, "unsupported gemv12 config");
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
