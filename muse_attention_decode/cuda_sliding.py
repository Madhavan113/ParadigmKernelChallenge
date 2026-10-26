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

// Sliding-window gated attention for one decode / verify step: a cluster of two CTAs per
// (request, KV head), each owning 1024 of the 2048 window slots with 8 warps x 128 keys. Each
// lane fetches the page indices of its 4 slots, then every K and V row of the CTA is issued at
// once with cp.async into shared memory (a row's four 16-byte chunks XOR-swizzled by
// (row >> 1) & 3, so ldmatrix is bank-conflict free). A warp runs the online softmax over its
// 128 keys in two 64-key chunks on mma.sync m16n8k16 (fp16 operands, fp32 accumulate): Q from
// shared memory, K fragments by ldmatrix + cvt e4m3x2->f16x2, V fragments by ldmatrix.trans on
// byte pairs + prmt + cvt (no transpose pass). The warps' (m, l, acc) partials merge in shared
// memory together with the block's own new tokens; the second CTA's partial crosses over
// distributed shared memory; then the gated RMSNorm epilogue. Every window key enters the
// softmax; nothing is skipped.

#define D 64
#define G 8                     // query heads per KV head
#define WINDOW 2048
#define NCTA 2
#define NW 8
#define NT (NW * 32)
#define KEYS (WINDOW / NCTA)
#define QS_STRIDE 72            // fp16 elements per Q row (padded: ldmatrix conflict free)
#define RED_W (32 + 16 * D)     // floats per 16-row partial: 16 m, 16 l, 16 x 64 acc
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
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t sel) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(0u), "r"(sel));
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
// byte offset of 16-byte chunk `ch` of row `r` in a swizzled [rows][64 B] tile
__device__ __forceinline__ uint32_t swz(int r, int ch) { return (uint32_t)r * 64u + (uint32_t)((ch ^ ((r >> 1) & 3)) << 4); }
// Q column permutation inside each 16-dim block: lane c's A fragment holds dims 4c..4c+3 as
// (a0: 4c, 4c+1 | a2: 4c+2, 4c+3), matching the K fragments ldmatrix yields (see the K loop).
__device__ __forceinline__ int qcol(int d) {
    int blk = d & ~15, e = d & 15, c = e >> 2, k = e & 3;
    return blk + (k < 2 ? 2 * c + k : 8 + 2 * c + (k - 2));
}

template <int L>
struct SlidingSmem {
    static constexpr int R = L * G, NRT = (R + 15) / 16, ROWS = NRT * 16;
    static constexpr int KS = 0;                                   // KEYS x 64 B
    static constexpr int VS = KS + KEYS * 64;                      // KEYS x 64 B
    static constexpr int QS = VS + KEYS * 64;                      // ROWS x QS_STRIDE fp16
    static constexpr int RED = QS + ROWS * QS_STRIDE * 2;          // NW x RED_W floats
    static constexpr int WTS = RED + NW * RED_W * 4;               // 16 x NW floats
    static constexpr int LROW = WTS + 16 * NW * 4;                 // 16 floats
    static constexpr int MROW = LROW + 16 * 4;                     // 16 floats
    static constexpr int SNEW = MROW + 16 * 4;                     // ROWS x L floats
    static constexpr int VNEW = SNEW + ROWS * L * 4;               // L x 64 floats (raw e4m3 values)
    static constexpr int KNEW = VNEW + L * D * 4;                  // L x 64 floats
    static constexpr int FIN = KNEW + L * D * 4;                   // NRT x RED_W floats
    static constexpr int NEW8 = FIN + NRT * RED_W * 4;             // L x 64 K bytes, L x 64 V bytes
    static constexpr int BYTES = NEW8 + 2 * L * D;
};

template <int L>
__global__ void __launch_bounds__(NT)
sliding_attn_kernel(const __nv_bfloat16* __restrict__ qkv, const int64_t* __restrict__ cache_loc,
                    const int* __restrict__ seq_lens, const int* __restrict__ page_table,
                    uint8_t* __restrict__ k_cache, uint8_t* __restrict__ v_cache,
                    const float* __restrict__ k_scale_p, const float* __restrict__ v_scale_p,
                    __nv_bfloat16* __restrict__ out, int HQ, int HKV, int max_pages, int PS,
                    float qk_scale, float sm_scale_log2, float log2_theta) {
    using S = SlidingSmem<L>;
    constexpr int R = S::R, NRT = S::NRT, ROWS = S::ROWS;
    const int QKV_W = (2 * HQ + 2 * HKV) * D;
    const int ROW_W = HKV * D;                   // one cache row (all KV heads) in bytes

    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* Ks = smem + S::KS;
    uint8_t* Vs = smem + S::VS;
    __half* Qs = reinterpret_cast<__half*>(smem + S::QS);
    float* red = reinterpret_cast<float*>(smem + S::RED);
    float* wts = reinterpret_cast<float*>(smem + S::WTS);
    float* Lrow = reinterpret_cast<float*>(smem + S::LROW);
    float* Mrow = reinterpret_cast<float*>(smem + S::MROW);
    float* snew = reinterpret_cast<float*>(smem + S::SNEW);
    float* vnew = reinterpret_cast<float*>(smem + S::VNEW);
    float* knew = reinterpret_cast<float*>(smem + S::KNEW);
    float* fin = reinterpret_cast<float*>(smem + S::FIN);
    uint8_t* new8 = smem + S::NEW8;
    __shared__ int s_seq;

    cg::cluster_group cluster = cg::this_cluster();
    const int rank = (int)cluster.block_rank();
    const int cid = blockIdx.x / NCTA;
    const int b = cid / HKV, h = cid % HKV;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, c = lane & 3;

    // ---- trip 1: the request's length (everything else here needs nothing) ----
    if (tid == 0) s_seq = seq_lens[b];
    const float k_scale = *k_scale_p, v_scale = *v_scale_p;
    __syncthreads();
    const int seq = s_seq;
    const int lo = max(seq - (WINDOW - 1), 0);
    const int n = seq - lo;                      // window keys (<= 2047): slots 0..n-1 <-> positions lo..seq-1
    const int slot0 = rank * KEYS;               // this CTA's first slot

    // ---- trip 2: page indices of this lane's 4 slots, then every K and V row of the CTA ----
    const uint32_t ks_base = smem_u32(Ks), vs_base = smem_u32(Vs);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int r = tid * 4 + i;               // slot within the CTA
        const int slot = slot0 + r;
        const bool ok = slot < n;
        const int t = lo + slot;
        const int page = ok ? page_table[(int64_t)b * max_pages + t / PS] : 0;
        const int64_t row = (int64_t)page * PS + (t % PS);
        const uint8_t* ksrc = k_cache + row * ROW_W + h * D;
        const uint8_t* vsrc = v_cache + row * ROW_W + h * D;
        #pragma unroll
        for (int ch = 0; ch < 4; ch++) {
            cp_async16(ks_base + swz(r, ch), ksrc + 16 * ch, ok ? 16 : 0);
            cp_async16(vs_base + swz(r, ch), vsrc + 16 * ch, ok ? 16 : 0);
        }
    }
    cp_async_commit();

    // ---- meanwhile: queries (RMSNorm x qk_scale x k_scale, RoPE) -> fp16 Qs; new keys/values ----
    const float inv_freq = exp2f(-(2.0f * lane / D) * log2_theta);   // this lane's RoPE pair (dims 2*lane, 2*lane+1)
    for (int r = warp; r < ROWS; r += NW) {
        float x0 = 0.f, x1 = 0.f;
        const int j = r / G, qh = h * G + (r % G);
        if (r < R) {
            const __nv_bfloat16* qp = qkv + ((int64_t)(b * L + j)) * QKV_W + qh * D + 2 * lane;
            x0 = __bfloat162float(qp[0]);
            x1 = __bfloat162float(qp[1]);
        }
        float ss = x0 * x0 + x1 * x1;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        const float inv = rsqrtf(ss / D + 1e-5f) * qk_scale * k_scale;
        x0 *= inv; x1 *= inv;
        float sn, cs;
        sincosf((float)(seq + j) * inv_freq, &sn, &cs);
        Qs[r * QS_STRIDE + qcol(2 * lane)] = __float2half_rn(x0 * cs - x1 * sn);
        Qs[r * QS_STRIDE + qcol(2 * lane + 1)] = __float2half_rn(x0 * sn + x1 * cs);
    }
    for (int j = warp; j < L; j += NW) {
        const __nv_bfloat16* kp = qkv + ((int64_t)(b * L + j)) * QKV_W + (HQ + h) * D + 2 * lane;
        const __nv_bfloat16* vp = qkv + ((int64_t)(b * L + j)) * QKV_W + (HQ + HKV + h) * D + 2 * lane;
        float x0 = __bfloat162float(kp[0]), x1 = __bfloat162float(kp[1]);
        float ss = x0 * x0 + x1 * x1;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        const float inv = rsqrtf(ss / D + 1e-5f);
        x0 *= inv; x1 *= inv;
        float sn, cs;
        sincosf((float)(seq + j) * inv_freq, &sn, &cs);
        const float y0 = x0 * cs - x1 * sn, y1 = x0 * sn + x1 * cs;
        __nv_fp8x2_e4m3 k8(make_float2(y0 / k_scale, y1 / k_scale));
        __nv_fp8x2_e4m3 v8(make_float2(__bfloat162float(vp[0]) / v_scale, __bfloat162float(vp[1]) / v_scale));
        const uint16_t kb = k8.__x, vb = v8.__x;
        *reinterpret_cast<uint16_t*>(new8 + j * D + 2 * lane) = kb;
        *reinterpret_cast<uint16_t*>(new8 + L * D + j * D + 2 * lane) = vb;
        // the raw E4M3 values the cache will hold (what the reference attends to)
        const float2 kk = f16x2_to_f2(cvt_e4m3x2_f16x2(kb)), vv = f16x2_to_f2(cvt_e4m3x2_f16x2(vb));
        knew[j * D + 2 * lane] = kk.x; knew[j * D + 2 * lane + 1] = kk.y;
        vnew[j * D + 2 * lane] = vv.x; vnew[j * D + 2 * lane + 1] = vv.y;
        if (rank == 0) {
            const int64_t loc = cache_loc[b * L + j];
            *reinterpret_cast<uint16_t*>(k_cache + loc * ROW_W + h * D + 2 * lane) = kb;
            *reinterpret_cast<uint16_t*>(v_cache + loc * ROW_W + h * D + 2 * lane) = vb;
        }
    }
    __syncthreads();
    // scores of the block's own tokens: snew[r][jj] = q_r . knew[jj] (causal: jj <= token(r)), log2 domain
    for (int idx = tid; idx < ROWS * L; idx += NT) {
        const int r = idx / L, jj = idx % L;
        float s = NEG;
        if (r < R && jj <= r / G) {
            float a = 0.f;
            #pragma unroll 8
            for (int d = 0; d < D; d++) a += __half2float(Qs[r * QS_STRIDE + qcol(d)]) * knew[jj * D + d];
            s = a * sm_scale_log2;
        }
        snew[idx] = s;
    }

    // ---- the window: each warp its 128 keys, per 16-row tile ----
    const uint32_t qs_base = smem_u32(Qs);
    cp_async_wait_all();
    __syncthreads();
    const int wslot0 = slot0 + warp * 128;       // global slot of this warp's first key
    for (int rt = 0; rt < NRT; rt++) {
        float m_run[2] = {NEG, NEG}, l_run[2] = {0.f, 0.f};
        float acc[8][4];
        #pragma unroll
        for (int i = 0; i < 8; i++) { acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f; }
        // each row's first valid slot (the window's edge): rows rt*16 + g and + g + 8
        const int first0 = n - (WINDOW - 1) + (rt * 16 + g) / G, first8 = n - (WINDOW - 1) + (rt * 16 + g + 8) / G;
        #pragma unroll 1
        for (int kc = 0; kc < 2; kc++) {
            // ---- S = Q K^T over 64 keys (8 n-tiles) ----
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
                for (int q4 = 0; q4 < 2; q4++) {          // 4 n-tiles (32 keys) per ldmatrix.x4
                    const int key = warp * 128 + kc * 64 + q4 * 32 + (lane >> 3) * 8 + (lane & 7);
                    uint32_t rr[4];
                    ldmatrix_x4(ks_base + swz(key, ks), rr[0], rr[1], rr[2], rr[3]);
                    #pragma unroll
                    for (int m = 0; m < 4; m++) {
                        // rr[m]: dims ks*16 + 4c .. 4c+3 of key (n-tile m, row g)
                        const uint32_t b0 = cvt_e4m3x2_f16x2(rr[m] & 0xffffu), b1 = cvt_e4m3x2_f16x2(rr[m] >> 16);
                        mma16816(s[q4 * 4 + m], a, b0, b1);
                    }
                }
            }
            // ---- mask (slice end, window edge), online softmax in the log2 domain ----
            float mx0 = NEG, mx8 = NEG;
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                const int slot = wslot0 + kc * 64 + nt * 8 + 2 * c;
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
            const float mn0 = fmaxf(m_run[0], mx0), mn8 = fmaxf(m_run[1], mx8);
            const float al0 = exp2f(m_run[0] - mn0), al8 = exp2f(m_run[1] - mn8);
            m_run[0] = mn0; m_run[1] = mn8;
            l_run[0] *= al0; l_run[1] *= al8;
            #pragma unroll
            for (int i = 0; i < 8; i++) { acc[i][0] *= al0; acc[i][1] *= al0; acc[i][2] *= al8; acc[i][3] *= al8; }
            uint32_t pa[4][4];                           // P as A fragments: 4 k16 steps over the 64 keys
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                const float p0 = exp2f(s[nt][0] - mn0), p1 = exp2f(s[nt][1] - mn0);
                const float p2 = exp2f(s[nt][2] - mn8), p3 = exp2f(s[nt][3] - mn8);
                l_run[0] += p0 + p1; l_run[1] += p2 + p3;
                pa[nt >> 1][(nt & 1) * 2 + 0] = pack_f16x2(p0, p1);
                pa[nt >> 1][(nt & 1) * 2 + 1] = pack_f16x2(p2, p3);
            }
            // ---- acc += P V: V fragments by ldmatrix.trans on byte pairs ----
            #pragma unroll
            for (int ks = 0; ks < 4; ks++) {
                const int key0 = warp * 128 + kc * 64 + ks * 16;
                uint32_t vr[8];
                #pragma unroll
                for (int dg2 = 0; dg2 < 2; dg2++) {      // dims 0-31, 32-63: matrices (keys lo | hi) x (2 chunks)
                    const int mrow = key0 + ((lane >> 3) & 1) * 8 + (lane & 7);
                    const int mch = dg2 * 2 + (lane >> 4);
                    ldmatrix_x4_trans(vs_base + swz(mrow, mch), vr[dg2 * 4 + 0], vr[dg2 * 4 + 1], vr[dg2 * 4 + 2], vr[dg2 * 4 + 3]);
                }
                #pragma unroll
                for (int dg = 0; dg < 4; dg++) {
                    // vr: lo16 = (key 2c, dims 16dg+2g, +1), hi16 = (key 2c+1, same dims); second reg = keys +8
                    const uint32_t lo = vr[dg * 2 + 0], hi = vr[dg * 2 + 1];
                    const uint32_t be0 = cvt_e4m3x2_f16x2(prmt(lo, 0x7720u)), be1 = cvt_e4m3x2_f16x2(prmt(hi, 0x7720u));   // even dims
                    const uint32_t bo0 = cvt_e4m3x2_f16x2(prmt(lo, 0x7731u)), bo1 = cvt_e4m3x2_f16x2(prmt(hi, 0x7731u));   // odd dims
                    mma16816(acc[dg * 2 + 0], pa[ks], be0, be1);
                    mma16816(acc[dg * 2 + 1], pa[ks], bo0, bo1);
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
        for (int dg = 0; dg < 4; dg++) {
            #pragma unroll
            for (int par = 0; par < 2; par++) {
                const int d0 = 16 * dg + 4 * c + par, d1 = d0 + 2;   // acc tile (dg, par) col n <-> dim 16dg + 2n + par
                my[32 + g * D + d0] = acc[dg * 2 + par][0];
                my[32 + g * D + d1] = acc[dg * 2 + par][1];
                my[32 + (g + 8) * D + d0] = acc[dg * 2 + par][2];
                my[32 + (g + 8) * D + d1] = acc[dg * 2 + par][3];
            }
        }
        __syncthreads();
        // ---- merge the warps (+ the block's own tokens, on rank 0) ----
        const bool own_new = (rank == 0);
        if (tid < 16 * NW) {
            const int r = tid / NW, w = tid % NW;
            float M = NEG;
            for (int ww = 0; ww < NW; ww++) M = fmaxf(M, red[ww * RED_W + r]);
            if (own_new) for (int jj = 0; jj < L; jj++) M = fmaxf(M, snew[(rt * 16 + r) * L + jj]);
            wts[r * NW + w] = exp2f(red[w * RED_W + r] - M);
            if (w == 0) {
                float Ls = 0.f;
                for (int ww = 0; ww < NW; ww++) Ls += exp2f(red[ww * RED_W + r] - M) * red[ww * RED_W + 16 + r];
                if (own_new) for (int jj = 0; jj < L; jj++) Ls += exp2f(snew[(rt * 16 + r) * L + jj] - M);
                Lrow[r] = Ls; Mrow[r] = M;
            }
        }
        __syncthreads();
        float* F = fin + rt * RED_W;
        for (int idx = tid; idx < 16 * D; idx += NT) {
            const int r = idx / D, d = idx % D;
            float O = 0.f;
            for (int w = 0; w < NW; w++) O += wts[r * NW + w] * red[w * RED_W + 32 + idx];
            if (own_new) {
                const float M = Mrow[r];
                for (int jj = 0; jj < L; jj++) O += exp2f(snew[(rt * 16 + r) * L + jj] - M) * vnew[jj * D + d];
            }
            F[32 + idx] = O;
            if (d == 0) { F[r] = Mrow[r]; F[16 + r] = Lrow[r]; }
        }
        __syncthreads();
    }

    // ---- combine the two CTAs' partials over distributed shared memory (rank 0 finishes) ----
    cluster.sync();
    if (rank == 1) { cluster.sync(); return; }
    {
        const float* peer = cluster.map_shared_rank(fin, 1);
        for (int idx = tid; idx < NRT * 16 * D; idx += NT) {
            const int rt = idx / (16 * D), rem = idx % (16 * D), r = rem / D;
            float* F = fin + rt * RED_W;
            const float* P = peer + rt * RED_W;
            const float M0 = F[r], M1 = P[r], M = fmaxf(M0, M1);
            const float w0 = exp2f(M0 - M), w1 = exp2f(M1 - M);
            const float O = F[32 + rem] * w0 + P[32 + rem] * w1;
            const float Lc = F[16 + r] * w0 + P[16 + r] * w1;
            __syncthreads();                     // everyone has read F[r], F[16 + r] for this pass
            F[32 + rem] = O;
            if (rem % D == 0) { F[16 + r] = Lc; F[r] = M; }
            __syncthreads();
        }
        cluster.sync();
    }

    // ---- epilogue: o = O / L x v_scale; per-head RMSNorm x sigmoid(gate) -> bf16 ----
    for (int r = warp; r < R; r += NW) {
        const float* F = fin + (r / 16) * RED_W;
        const int rr = r % 16;
        const float inv = v_scale / F[16 + rr];
        const float o0 = F[32 + rr * D + lane] * inv, o1 = F[32 + rr * D + lane + 32] * inv;
        float ss = o0 * o0 + o1 * o1;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        const float sc = rsqrtf(ss / D + 1e-5f);
        const int j = r / G, qh = h * G + (r % G);
        const __nv_bfloat16* gp = qkv + ((int64_t)(b * L + j)) * QKV_W + (HQ + 2 * HKV + qh) * D;
        const float g0 = __bfloat162float(gp[lane]), g1 = __bfloat162float(gp[lane + 32]);
        __nv_bfloat16* op = out + ((int64_t)(b * L + j)) * HQ * D + qh * D;
        op[lane] = __float2bfloat16(o0 * sc / (1.f + expf(-g0)));
        op[lane + 32] = __float2bfloat16(o1 * sc / (1.f + expf(-g1)));
    }
}

template <int L>
static void launch_sliding(const torch::Tensor& qkv, const torch::Tensor& cache_loc, const torch::Tensor& seq_lens,
                           const torch::Tensor& page_table, torch::Tensor& k_cache, torch::Tensor& v_cache,
                           const torch::Tensor& k_scale, const torch::Tensor& v_scale, torch::Tensor& out,
                           int HQ, int HKV, int PS, float qk_scale, float sm_scale_log2, float log2_theta) {
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
        HQ, HKV, (int)page_table.size(1), PS, qk_scale, sm_scale_log2, log2_theta);
    TORCH_CHECK(err == cudaSuccess, "sliding_attn launch failed: ", cudaGetErrorString(err));
}

void sliding_attn(torch::Tensor qkv, torch::Tensor cache_loc, torch::Tensor seq_lens, torch::Tensor page_table,
                  torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor k_scale, torch::Tensor v_scale,
                  torch::Tensor out, int64_t L, int64_t HQ, int64_t HKV, int64_t PS,
                  double qk_scale, double sm_scale_log2, double log2_theta) {
    TORCH_CHECK(HQ == 8 * HKV, "sliding_attn: 8 query heads per KV head");
    if (L == 1) launch_sliding<1>(qkv, cache_loc, seq_lens, page_table, k_cache, v_cache, k_scale, v_scale, out,
                                  (int)HQ, (int)HKV, (int)PS, (float)qk_scale, (float)sm_scale_log2, (float)log2_theta);
    else if (L == 8) launch_sliding<8>(qkv, cache_loc, seq_lens, page_table, k_cache, v_cache, k_scale, v_scale, out,
                                       (int)HQ, (int)HKV, (int)PS, (float)qk_scale, (float)sm_scale_log2, (float)log2_theta);
    else TORCH_CHECK(false, "sliding_attn: q_len must be 1 or 8");
}
"""

CPP_SLIDING_SRC = """
#include <torch/extension.h>
void sliding_attn(torch::Tensor qkv, torch::Tensor cache_loc, torch::Tensor seq_lens, torch::Tensor page_table,
                  torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor k_scale, torch::Tensor v_scale,
                  torch::Tensor out, int64_t L, int64_t HQ, int64_t HKV, int64_t PS,
                  double qk_scale, double sm_scale_log2, double log2_theta);
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


def _sliding_forward(data, out):
    """The dedicated sliding-window kernel; returns False (nothing done) when it does not apply."""
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
                     cfg["qk_scale_factor"], cfg["sm_scale"] * LOG2E, math.log2(cfg["rope_theta"]))
    return True
