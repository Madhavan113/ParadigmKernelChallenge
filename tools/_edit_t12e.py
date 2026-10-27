"""Round 5 of t12: t12e = t12d with ONE 3D TMA op per x chunk (tensor map {64 k, m rows, K/64 groups}, strides
{ldx*2, 128 B}, box {64, 8*NT, KC/64}: smem [group][row][128 B swizzled] = t12d's layout), variant 5; t12d left
dormant (check 1d05128d: 8 boxes per chunk cost ~33 ns each). Run: python3 tools/_edit_t12e.py"""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
p = root / "tools/t12_kernel.cu.part"; s = p.read_text()
NEW = r'''
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
// == END t12 device =='''
if "t12e_kernel" not in s:
    s = s.replace("// == END t12 device ==", NEW.lstrip("\n"), 1)
    p.write_text(s)

w = root / "tools/t12_wrapper.cu.part"; s = w.read_text()
if "t12e" not in s:
    HOST3 = r'''// 3D x map for t12e: {64 k, m rows, K/64 groups}, strides {ldx*2, 128 B}, box {64, box_rows, groups}: one op per chunk
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
'''
    s = s.replace("void bstream(torch::Tensor buf", HOST3 + "\nvoid bstream(torch::Tensor buf", 1)
    old_cases = s[s.index("#define T12C_CASE(R_"):s.index("    TORCH_CHECK(false, \"t12: unsupported")]
    new_cases = r'''#define T12C_CASE(R_, C_, W_, SW_, SX_, NT_) \
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
    T12C_NT(32, 4, 8, 5, 3)
    T12E_NT(32, 4, 8, 4, 3)
    T12E_NT(32, 4, 8, 5, 3)
    T12E_NT(32, 4, 8, 4, 4)
    T12E_NT(32, 4, 8, 6, 3)
    T12E_NT(32, 2, 8, 8, 4)
    T12E_NT(32, 2, 8, 8, 6)
    T12E_NT(32, 2, 8, 12, 4)
    T12E_NT(16, 4, 8, 4, 3)
    T12E_NT(16, 4, 4, 4, 3)
    T12E_NT(16, 4, 4, 4, 4)
    T12E_NT(16, 2, 4, 8, 4)
    T12E_NT(16, 2, 4, 6, 6)
    T12E_NTD(32, 4, 8, 5, 3)
    T12E_NTD(16, 4, 8, 4, 3)
#undef T12C_NT
#undef T12E_NT
#undef T12E_NTD
#undef T12C_CASE
#undef T12E_CASE
#undef T12E_CASE_D
'''
    s = s.replace(old_cases, new_cases)
    w.write_text(s)

sp = root / "muse_bf16_proj/submission.py"; t = sp.read_text()
t = t.replace('int(cfg.get("v", 4)), et)', 'int(cfg.get("v", 5)), et)')
t = t.replace('(int(cfg.get("v", 4)) >= 3 and et is None)', '(int(cfg.get("v", 5)) >= 3 and et is None)')
t = t.replace("nt=tb12_nt(m), v=4)", "nt=tb12_nt(m), v=5)")
sp.write_text(t)
eb = root / "tools/_edit_t12_bench.py"; s = eb.read_text()
a = s.index("        cfgs = ["); b = s.index("]\n", a) + 2
s = s[:a] + ("        cfgs = [(32, 4, 8, 5, 3, 3), (32, 4, 8, 4, 3, 5), (32, 4, 8, 5, 3, 5), (32, 4, 8, 4, 4, 5), (32, 4, 8, 6, 3, 5), (32, 2, 8, 8, 4, 5),\n"
             "                (32, 2, 8, 8, 6, 5), (32, 2, 8, 12, 4, 5), (16, 4, 8, 4, 3, 5), (16, 4, 4, 4, 3, 5), (16, 4, 4, 4, 4, 5), (16, 2, 4, 8, 4, 5), (16, 2, 4, 6, 6, 5)]\n") + s[b:]
s = s.replace("(32, 4, 8, 5, 3, 4)", "(32, 4, 8, 5, 3, 5)").replace("(16, 4, 8, 4, 3, 4)", "(16, 4, 8, 4, 3, 5)")
eb.write_text(s)
print("ok")
