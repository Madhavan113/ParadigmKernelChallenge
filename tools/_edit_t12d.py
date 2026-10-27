"""Round 4 of t12: t12d = t12c with the x tile fetched by 2D TMA (cp.async.bulk.tensor.2d over a tensor map encoded
per call, 128B swizzle, rows >= m zero-filled by the TMA unit), red aliased onto the drained weight ring, dispatch
variant 4; TB12_SHAPES emptied (check d777cb3c: t12c loses on the judge everywhere). Run: python3 tools/_edit_t12d.py"""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
p = root / "tools/t12_kernel.cu.part"; s = p.read_text()
NEW = r'''
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
// == END t12 device =='''
if "t12d_kernel" not in s:
    s = s.replace("// == END t12 device ==", NEW.lstrip("\n"), 1)
    p.write_text(s)

w = root / "tools/t12_wrapper.cu.part"; s = w.read_text()
if "t12d" not in s:
    HOST = r'''// == BEGIN t12 host ==
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
'''
    s = s.replace("// == BEGIN t12 host ==\n", HOST, 1)
    old_cases = s[s.index("#define T12C_CASE(R_"):s.index("    TORCH_CHECK(false, \"t12: unsupported")]
    new_cases = r'''#define T12C_CASE(R_, C_, W_, SW_, SX_, NT_) \
    if (variant == 3 && r == R_ && ck == C_ && w == W_ && nsw == SW_ && nsx == SX_ && nt == NT_ && !diag_on) { \
        launch_t12c<R_, C_, W_, SW_, SX_, NT_, false>(Xp, ldx, m, Wp, (int)eb, etp, op, ldo, (int)n, (int)k, dg, stream); return; }
#define T12D_CASE(R_, C_, W_, SW_, SX_, NT_) \
    if (variant == 4 && r == R_ && ck == C_ && w == W_ && nsw == SW_ && nsx == SX_ && nt == NT_ && !diag_on) { \
        launch_t12d<R_, C_, W_, SW_, SX_, NT_, false>(xmap, m, Wp, (int)eb, etp, op, ldo, (int)n, (int)k, dg, stream); return; }
#define T12D_CASE_D(R_, C_, W_, SW_, SX_, NT_) \
    if (variant == 4 && r == R_ && ck == C_ && w == W_ && nsw == SW_ && nsx == SX_ && nt == NT_ && diag_on) { \
        launch_t12d<R_, C_, W_, SW_, SX_, NT_, true>(xmap, m, Wp, (int)eb, etp, op, ldo, (int)n, (int)k, dg, stream); return; }
#define T12C_NT(R_, C_, W_, SW_, SX_) T12C_CASE(R_, C_, W_, SW_, SX_, 1) T12C_CASE(R_, C_, W_, SW_, SX_, 2) T12C_CASE(R_, C_, W_, SW_, SX_, 4)
#define T12D_NT(R_, C_, W_, SW_, SX_) T12D_CASE(R_, C_, W_, SW_, SX_, 1) T12D_CASE(R_, C_, W_, SW_, SX_, 2) T12D_CASE(R_, C_, W_, SW_, SX_, 4)
#define T12D_NTD(R_, C_, W_, SW_, SX_) T12D_CASE_D(R_, C_, W_, SW_, SX_, 1) T12D_CASE_D(R_, C_, W_, SW_, SX_, 2) T12D_CASE_D(R_, C_, W_, SW_, SX_, 4)
    CUtensorMap xmap;
    if (variant == 4) xmap = okc_x_map(X.data_ptr(), m, (int)k, ldx, 8 * (int)nt);
    T12C_NT(32, 4, 8, 5, 3)
    T12D_NT(32, 4, 8, 4, 3)
    T12D_NT(32, 4, 8, 5, 3)
    T12D_NT(32, 4, 8, 6, 3)
    T12D_NT(32, 4, 8, 4, 4)
    T12D_NT(32, 2, 8, 8, 4)
    T12D_NT(32, 2, 8, 8, 6)
    T12D_NT(32, 2, 8, 12, 4)
    T12D_NT(16, 4, 8, 4, 3)
    T12D_NT(16, 4, 4, 4, 3)
    T12D_NT(16, 2, 4, 8, 4)
    T12D_NT(16, 2, 8, 8, 4)
    T12D_NTD(32, 4, 8, 5, 3)
    T12D_NTD(16, 4, 8, 4, 3)
#undef T12C_NT
#undef T12D_NT
#undef T12D_NTD
#undef T12C_CASE
#undef T12D_CASE
#undef T12D_CASE_D
'''
    s = s.replace(old_cases, new_cases)
    w.write_text(s)

cp = root / "muse_bf16_proj/cuda_proj.py"; s = cp.read_text()
if "#include <cuda.h>" not in s:
    s = s.replace("#include <cuda_bf16.h>\n#include <stdint.h>\n", "#include <cuda.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n", 1)
    cp.write_text(s)

sp = root / "muse_bf16_proj/submission.py"; t = sp.read_text()
a = t.index("TB12_SHAPES = {(")
b = t.index("# judged in check 5 (self-tested in prepare)\n") + len("# judged in check 5 (self-tested in prepare)\n")
t = t[:a] + "TB12_SHAPES = {}   # check d777cb3c (t12c, judged): loses everywhere (pooler m=8/16/32 19.6/22.9/30.6 vs 18.2/18.0/20.0)\n" + t[b:]
t = t.replace('int(cfg.get("v", 3)), et)', 'int(cfg.get("v", 4)), et)')
t = t.replace('(int(cfg.get("v", 3)) == 3 and et is None)', '(int(cfg.get("v", 4)) >= 3 and et is None)')
t = t.replace("nt=tb12_nt(m), v=3)", "nt=tb12_nt(m), v=4)")
sp.write_text(t)
# bench configs
eb = root / "tools/_edit_t12_bench.py"; s = eb.read_text()
a = s.index("        cfgs = ["); b = s.index("]\n", a) + 2
s = s[:a] + ("        cfgs = [(32, 4, 8, 5, 3, 3), (32, 4, 8, 4, 3, 4), (32, 4, 8, 5, 3, 4), (32, 4, 8, 6, 3, 4), (32, 4, 8, 4, 4, 4), (32, 2, 8, 8, 4, 4),\n"
             "                (32, 2, 8, 8, 6, 4), (32, 2, 8, 12, 4, 4), (16, 4, 8, 4, 3, 4), (16, 4, 4, 4, 3, 4), (16, 2, 4, 8, 4, 4), (16, 2, 8, 8, 4, 4)]\n") + s[b:]
s = s.replace("for c in ((32, 4, 8, 5, 2, 3), (16, 4, 8, 4, 2, 3))))", "for c in ((32, 4, 8, 5, 3, 4), (16, 4, 8, 4, 3, 4))))")
s = s.replace("(32, 4, 8, 5, 2, 3)", "(32, 4, 8, 5, 3, 4)").replace("(16, 4, 8, 4, 2, 3)", "(16, 4, 8, 4, 3, 4)")
eb.write_text(s)
print("ok")
