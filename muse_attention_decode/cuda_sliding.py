# Development copy of the sliding-window CUDA path; tools/embed_sliding.py copies everything from
# CUDA_SLIDING_SRC to the end of _sliding_forward into submission.py between the
# "# == BEGIN CUDA SLIDING ==" / "# == END CUDA SLIDING ==" markers.
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
//           the block's new key/value rows, the gate rows) is issued together;
//   trip 2: each lane fetches the page indices of its rows (4 lanes per row), then every K and V
//           row of the CTA goes out at once (cp.async, 16 B per lane, into a staging tile);
//   the warps convert their own 64 K and V rows once to fp16 (swizzled 128-byte rows), run the
//   softmax over their keys on mma.sync m16n8k16 with ldmatrix / ldmatrix.trans fragments, and
//   merge their (m, l, acc) partials in shared memory (the staging tile, now free) together with
//   the block's own new tokens; rank 0 combines the four CTAs over distributed shared memory and
//   applies the gated RMSNorm. Every window key enters the softmax; nothing is skipped.

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
    static constexpr int WTS = QS + ROWS * QS_STRIDE * 2;          // 16 x NW floats
    static constexpr int LROW = WTS + 16 * NW * 4;                 // 16 floats
    static constexpr int MROW = LROW + 16 * 4;                     // 16 floats
    static constexpr int SNEW = MROW + 16 * 4;                     // ROWS x L floats
    static constexpr int VNEW = SNEW + ROWS * L * 4;               // L x 64 floats (raw e4m3 values)
    static constexpr int KNEW = VNEW + L * D * 4;                  // L x 64 floats
    static constexpr int BYTES = KNEW + L * D * 4;
};

template <int L>
__global__ void __launch_bounds__(NT, 1)
sliding_attn_kernel(const __nv_bfloat16* __restrict__ qkv, const int64_t* __restrict__ cache_loc,
                    const int* __restrict__ seq_lens, const int* __restrict__ page_table,
                    uint8_t* __restrict__ k_cache, uint8_t* __restrict__ v_cache,
                    const float* __restrict__ k_scale_p, const float* __restrict__ v_scale_p,
                    __nv_bfloat16* __restrict__ out, int HQ, int HKV, int max_pages, int PS,
                    float qk_scale, float sm_scale_log2, float log2_theta, int abl) {
    using S = SlidingSmem<L>;
    constexpr int R = S::R, NRT = S::NRT, ROWS = S::ROWS;
    constexpr int QPW = ROWS / NW;               // query rows per warp (2 or 8)
    constexpr int GPW = (R + NW - 1) / NW;       // live rows per warp in the epilogue (1 or 8)
    const int QKV_W = (2 * HQ + 2 * HKV) * D;
    const int ROW_W = HKV * D;                   // one cache row (all KV heads) in bytes

    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* K8 = smem + S::STG;
    uint8_t* V8 = K8 + KEYS * 64;
    float* red = reinterpret_cast<float*>(smem + S::RED);
    float* fin = reinterpret_cast<float*>(smem + S::FIN);
    __half* Qs = reinterpret_cast<__half*>(smem + S::QS);
    float* wts = reinterpret_cast<float*>(smem + S::WTS);
    float* Lrow = reinterpret_cast<float*>(smem + S::LROW);
    float* Mrow = reinterpret_cast<float*>(smem + S::MROW);
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

    // ---- trip 1: every load that depends on nothing, issued together ----
    if (tid == 0) s_seq = seq_lens[b];
    const float k_scale = *k_scale_p, v_scale = *v_scale_p;
    float qx[QPW][2], gx[GPW][2];
    #pragma unroll
    for (int i = 0; i < QPW; i++) {
        const int r = warp + i * NW;
        qx[i][0] = qx[i][1] = 0.f;
        if (r < R && !(abl & 16)) {
            const int j = r / G, qh = h * G + (r % G);
            const __nv_bfloat16* qp = qkv + ((int64_t)(b * L + j)) * QKV_W + qh * D + 2 * lane;
            qx[i][0] = __bfloat162float(qp[0]);
            qx[i][1] = __bfloat162float(qp[1]);
        }
    }
    #pragma unroll
    for (int i = 0; i < GPW; i++) {
        const int r = warp + i * NW;
        gx[i][0] = gx[i][1] = 0.f;
        if (r < R && !(abl & 16)) {
            const int j = r / G, qh = h * G + (r % G);
            const __nv_bfloat16* gp = qkv + ((int64_t)(b * L + j)) * QKV_W + (HQ + 2 * HKV + qh) * D;
            gx[i][0] = __bfloat162float(gp[lane]);
            gx[i][1] = __bfloat162float(gp[lane + 32]);
        }
    }
    float kx0 = 0.f, kx1 = 0.f, vx0 = 0.f, vx1 = 0.f;   // the new key/value row this warp prepares (warp < L)
    int64_t loc = 0;
    if (warp < L && !(abl & 16)) {
        const __nv_bfloat16* kp = qkv + ((int64_t)(b * L + warp)) * QKV_W + (HQ + h) * D + 2 * lane;
        const __nv_bfloat16* vp = qkv + ((int64_t)(b * L + warp)) * QKV_W + (HQ + HKV + h) * D + 2 * lane;
        kx0 = __bfloat162float(kp[0]); kx1 = __bfloat162float(kp[1]);
        vx0 = __bfloat162float(vp[0]); vx1 = __bfloat162float(vp[1]);
        loc = cache_loc[b * L + warp];
    }
    __syncthreads();
    const int seq = s_seq;
    const int lo = max(seq - (WINDOW - 1), 0);
    const int n = seq - lo;                      // window keys (<= 2047): slots 0..n-1 <-> positions lo..seq-1
    const int slot0 = rank * KEYS;               // this CTA's first slot
    const int wrow0 = warp * 64;                 // this warp's first row (slot within the CTA)

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

    // ---- meanwhile: queries (RMSNorm x qk_scale x k_scale, RoPE) -> fp16 Qs; new keys/values ----
    const float inv_freq = exp2f(-(2.0f * lane / D) * log2_theta);   // this lane's RoPE pair (dims 2*lane, 2*lane+1)
    if (!(abl & 8)) {
        #pragma unroll
        for (int i = 0; i < QPW; i++) {
            const int r = warp + i * NW;
            float x0 = qx[i][0], x1 = qx[i][1];
            float ss = x0 * x0 + x1 * x1;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
            const float inv = rsqrtf(ss / D + 1e-5f) * qk_scale * k_scale;
            x0 *= inv; x1 *= inv;
            float sn, cs;
            sincos_fast((float)(seq + r / G) * inv_freq, sn, cs);
            *reinterpret_cast<__half2*>(Qs + r * QS_STRIDE + 2 * lane) = __floats2half2_rn(x0 * cs - x1 * sn, x0 * sn + x1 * cs);
        }
        if (warp < L) {
            const int j = warp;
            float x0 = kx0, x1 = kx1;
            float ss = x0 * x0 + x1 * x1;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
            const float inv = rsqrtf(ss / D + 1e-5f);
            x0 *= inv; x1 *= inv;
            float sn, cs;
            sincos_fast((float)(seq + j) * inv_freq, sn, cs);
            const float y0 = x0 * cs - x1 * sn, y1 = x0 * sn + x1 * cs;
            __nv_fp8x2_e4m3 k8(make_float2(y0 / k_scale, y1 / k_scale));
            __nv_fp8x2_e4m3 v8(make_float2(vx0 / v_scale, vx1 / v_scale));
            const uint16_t kb = k8.__x, vb = v8.__x;
            // the raw E4M3 values the cache will hold (what the reference attends to)
            const float2 kk = f16x2_to_f2(cvt_e4m3x2_f16x2(kb)), vv = f16x2_to_f2(cvt_e4m3x2_f16x2(vb));
            knew[j * D + 2 * lane] = kk.x; knew[j * D + 2 * lane + 1] = kk.y;
            vnew[j * D + 2 * lane] = vv.x; vnew[j * D + 2 * lane + 1] = vv.y;
            if (rank == 0) {
                *reinterpret_cast<uint16_t*>(k_cache + loc * ROW_W + h * D + 2 * lane) = kb;
                *reinterpret_cast<uint16_t*>(v_cache + loc * ROW_W + h * D + 2 * lane) = vb;
            }
        }
    }
    __syncthreads();
    // scores of the block's own tokens: snew[r][jj] = q_r . knew[jj] (causal: jj <= token(r)), log2 domain
    for (int idx = tid; idx < ROWS * L; idx += NT) {
        const int r = idx / L, jj = idx % L;
        float sc = NEG;
        if (r < R && jj <= r / G && !(abl & 8)) {
            float a = 0.f;
            #pragma unroll 8
            for (int d = 0; d < D; d++) a += __half2float(Qs[r * QS_STRIDE + d]) * knew[jj * D + d];
            sc = a * sm_scale_log2;
        }
        snew[idx] = sc;
    }

    // ---- K and V rows -> fp16 tiles (each warp its own 64 rows, each lane 2 rows) ----
    if (!(abl & 1)) cp_async_wait_all();
    __syncthreads();                             // every lane's copies have landed
    if (!(abl & 2)) {
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
    __syncthreads();                             // fp16 tiles complete; the staging tile is free for the merge buffers

    // ---- the window: each warp its 64 keys, per 16-row tile ----
    const int wslot0 = slot0 + wrow0;            // global slot of this warp's first key
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
                    // matrices: (n-tile 2p2 | 2p2+1) x (k half 0 | 1): lane -> matrix m = lane>>3
                    const int m = lane >> 3;
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
                    // matrices: (keys lo | hi) x (n-tile 2p2 | 2p2+1): lane -> matrix m = lane>>3
                    const int m = lane >> 3;
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
        // ---- merge the warps (+ the block's own tokens, on rank 0): one pass, one barrier ----
        const bool own_new = (rank == 0);
        float* F = fin + rt * RED_W;
        if (!(abl & 4)) {
            #pragma unroll
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
        __syncthreads();
    }

    // ---- combine the four CTAs' partials over distributed shared memory (rank 0 finishes) ----
    if (!(abl & 128)) {
        cluster.sync();
        if (rank != 0) { cluster.sync(); return; }
    } else if (rank != 0) return;
    if (!(abl & 4) && !(abl & 128)) {
        const float* P1 = cluster.map_shared_rank(fin, 1);
        const float* P2 = cluster.map_shared_rank(fin, 2);
        const float* P3 = cluster.map_shared_rank(fin, 3);
        float* cw = red;                         // NRT x 16 x 4 weights, then the combined M, L (red is free now)
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
        #pragma unroll
        for (int k = 0; k < (NRT * 16 * D) / NT; k++) {
            const int idx = tid + k * NT, rt = idx / (16 * D), rem = idx % (16 * D), r = rem / D, d = rem % D;
            const int off = rt * RED_W, e = off + 32 + r * RS + d, wi = (rt * 16 + r) * 4;
            fin[e] = fin[e] * cw[wi] + P1[e] * cw[wi + 1] + P2[e] * cw[wi + 2] + P3[e] * cw[wi + 3];
        }
        cluster.sync();
    } else if (!(abl & 128)) {
        cluster.sync();
    }
    if (abl & 64) return;

    // ---- epilogue: o = O / L x v_scale; per-head RMSNorm x sigmoid(gate) -> bf16 ----
    #pragma unroll
    for (int i = 0; i < GPW; i++) {
        const int r = warp + i * NW;
        if (r >= R) break;
        const float* F = fin + (r / 16) * RED_W;
        const int rr = r % 16;
        const float inv = v_scale / F[16 + rr];
        const float o0 = F[32 + rr * RS + lane] * inv, o1 = F[32 + rr * RS + lane + 32] * inv;
        float ss = o0 * o0 + o1 * o1;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        const float sc = rsqrtf(ss / D + 1e-5f);
        const int j = r / G, qh = h * G + (r % G);
        __nv_bfloat16* op = out + ((int64_t)(b * L + j)) * HQ * D + qh * D;
        op[lane] = __float2bfloat16(o0 * sc / (1.f + expf(-gx[i][0])));
        op[lane + 32] = __float2bfloat16(o1 * sc / (1.f + expf(-gx[i][1])));
    }
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

SLIDING_CUDA = True     # sliding-window layers: the dedicated CUDA kernel (False: the Triton kernel)
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
