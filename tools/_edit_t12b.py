"""Round 2 of t12: add the t12b producer/consumer kernel to tools/t12_kernel.cu.part, extend the host wrapper
(variant arg, bstream attribute fix) and the embed script's declaration. Run: python3 tools/_edit_t12b.py"""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
p = root / "tools/t12_kernel.cu.part"; s = p.read_text()
NEW = r'''
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
// == END t12 device =='''
if "t12b_kernel" not in s:
    s = s.replace("// == END t12 device ==", NEW.lstrip("\n"), 1)
    p.write_text(s)

w = root / "tools/t12_wrapper.cu.part"; s = w.read_text()
if "variant" not in s:
    s = s.replace("cudaFuncAttributeMaxDynamicSharedMemorySize, 232448)", "cudaFuncAttributeMaxDynamicSharedMemorySize, 232448 - 4096)")
    s = s.replace("int64_t nsw, int64_t nsx, int64_t nt, bool diag_on)\n{\n    auto stream", "int64_t nsw, int64_t nsx, int64_t nt, bool diag_on, int64_t variant)\n{\n    auto stream")
    old_cases = s[s.index("#define T12_CASE(R_"):s.index("    TORCH_CHECK(false, \"t12: unsupported")]
    new_cases = r'''#define T12_CASE(R_, C_, W_, SW_, SX_, NT_) \
    if (variant == 1 && r == R_ && ck == C_ && w == W_ && nsw == SW_ && nsx == SX_ && nt == NT_ && !diag_on) { \
        launch_t12<R_, C_, W_, SW_, SX_, NT_, false>(Xp, ldx, m, Wp, (int)eb, ero, ek, ed, op, ldo, (int)n, (int)k, dg, stream); return; }
#define T12B_CASE(R_, C_, W_, SW_, SX_, NT_) \
    if (variant == 2 && r == R_ && ck == C_ && w == W_ && nsw == SW_ && nsx == SX_ && nt == NT_ && !diag_on) { \
        launch_t12b<R_, C_, W_, SW_, SX_, NT_, false>(Xp, ldx, m, Wp, (int)eb, ero, ek, ed, op, ldo, (int)n, (int)k, dg, stream); return; }
#define T12B_CASE_D(R_, C_, W_, SW_, SX_, NT_) \
    if (variant == 2 && r == R_ && ck == C_ && w == W_ && nsw == SW_ && nsx == SX_ && nt == NT_ && diag_on) { \
        launch_t12b<R_, C_, W_, SW_, SX_, NT_, true>(Xp, ldx, m, Wp, (int)eb, ero, ek, ed, op, ldo, (int)n, (int)k, dg, stream); return; }
#define T12_NT(R_, C_, W_, SW_, SX_) T12_CASE(R_, C_, W_, SW_, SX_, 1) T12_CASE(R_, C_, W_, SW_, SX_, 2) T12_CASE(R_, C_, W_, SW_, SX_, 4)
#define T12B_NT(R_, C_, W_, SW_, SX_) T12B_CASE(R_, C_, W_, SW_, SX_, 1) T12B_CASE(R_, C_, W_, SW_, SX_, 2) T12B_CASE(R_, C_, W_, SW_, SX_, 4)
#define T12B_NTD(R_, C_, W_, SW_, SX_) T12B_CASE_D(R_, C_, W_, SW_, SX_, 1) T12B_CASE_D(R_, C_, W_, SW_, SX_, 2) T12B_CASE_D(R_, C_, W_, SW_, SX_, 4)
    T12_NT(32, 2, 8, 6, 3)
    T12_NT(32, 4, 8, 4, 2)
    T12B_NT(32, 2, 8, 6, 3)
    T12B_NT(32, 2, 8, 8, 3)
    T12B_NT(32, 2, 8, 8, 2)
    T12B_NT(32, 2, 4, 8, 3)
    T12B_NT(32, 4, 8, 4, 2)
    T12B_NT(32, 4, 8, 5, 2)
    T12B_NT(32, 4, 8, 6, 2)
    T12B_NT(16, 2, 4, 8, 3)
    T12B_NT(16, 2, 4, 6, 3)
    T12B_NT(16, 4, 4, 4, 2)
    T12B_NT(16, 4, 8, 4, 2)
    T12B_NTD(32, 2, 8, 8, 3)
    T12B_NTD(32, 4, 8, 5, 2)
    T12B_NTD(16, 2, 4, 8, 3)
#undef T12_NT
#undef T12B_NT
#undef T12B_NTD
#undef T12_CASE
#undef T12B_CASE
#undef T12B_CASE_D
'''
    s = s.replace(old_cases, new_cases)
    w.write_text(s)
e = root / "tools/_edit_t12.py"; s = e.read_text()
s = s.replace("int64_t nsw, int64_t nsx, int64_t nt, bool diag_on);\\n\")", "int64_t nsw, int64_t nsx, int64_t nt, bool diag_on, int64_t variant);\\n\")")
e.write_text(s)
# submission glue: variant in the cfg
sp = root / "muse_bf16_proj/submission.py"; t = sp.read_text()
t = t.replace('cfg["nsw"], cfg["nsx"], cfg["nt"], bool(cfg.get("diag")))', 'cfg["nsw"], cfg["nsx"], cfg["nt"], bool(cfg.get("diag")), int(cfg.get("v", 2)))')
t = t.replace("        r, ck, w, nsw, nsx = TB12_SHAPES[(layer, n, m)]\n        return dict(tb12=True, r=r, ck=ck, w=w, nsw=nsw, nsx=nsx, nt=tb12_nt(m))",
              "        r, ck, w, nsw, nsx = TB12_SHAPES[(layer, n, m)]\n        return dict(tb12=True, r=r, ck=ck, w=w, nsw=nsw, nsx=nsx, nt=tb12_nt(m), v=2)")
sp.write_text(t)
print("ok")
