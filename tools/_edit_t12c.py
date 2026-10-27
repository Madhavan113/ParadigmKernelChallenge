"""Round 3 of t12: t12c kernel (escapes per chunk from the x stage via a per-tile table, x prologue before the weight
flood, non-volatile ldsm/mma, one __syncthreads per chunk), host dispatch (variant 3), esc table builder, self-tested
dispatch (TB12_SHAPES + _TB12_ON), bench plan. Run: python3 tools/_edit_t12c.py"""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
p = root / "tools/t12_kernel.cu.part"; s = p.read_text()
NEW = r'''
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
// == END t12 device =='''
if "t12c_kernel" not in s:
    s = s.replace("// == END t12 device ==", NEW.lstrip("\n"), 1)
    p.write_text(s)

w = root / "tools/t12_wrapper.cu.part"; s = w.read_text()
if "t12c" not in s:
    s = s.replace("torch::Tensor out, torch::Tensor diag, int64_t n, int64_t k, int64_t r, int64_t ck, int64_t w, int64_t nsw, int64_t nsx, int64_t nt, bool diag_on, int64_t variant)\n{",
                  "torch::Tensor out, torch::Tensor diag, int64_t n, int64_t k, int64_t r, int64_t ck, int64_t w, int64_t nsw, int64_t nsx, int64_t nt, bool diag_on, int64_t variant, torch::Tensor etab)\n{")
    s = s.replace('    TORCH_CHECK(diag.numel() >= 8, "t12: diag buffer");\n',
                  '    TORCH_CHECK(diag.numel() >= 8, "t12: diag buffer");\n'
                  '    TORCH_CHECK(variant != 3 || (etab.is_contiguous() && etab.dtype() == torch::kInt32 && etab.numel() >= (n / r) * 128 && (reinterpret_cast<uintptr_t>(etab.data_ptr()) & 7) == 0), "t12: escape table");\n'
                  '    const int2* etp = reinterpret_cast<const int2*>(etab.data_ptr());\n')
    old_cases = s[s.index("#define T12_CASE(R_"):s.index("    TORCH_CHECK(false, \"t12: unsupported")]
    new_cases = r'''#define T12C_CASE(R_, C_, W_, SW_, SX_, NT_) \
    if (variant == 3 && r == R_ && ck == C_ && w == W_ && nsw == SW_ && nsx == SX_ && nt == NT_ && !diag_on) { \
        launch_t12c<R_, C_, W_, SW_, SX_, NT_, false>(Xp, ldx, m, Wp, (int)eb, etp, op, ldo, (int)n, (int)k, dg, stream); return; }
#define T12C_CASE_D(R_, C_, W_, SW_, SX_, NT_) \
    if (variant == 3 && r == R_ && ck == C_ && w == W_ && nsw == SW_ && nsx == SX_ && nt == NT_ && diag_on) { \
        launch_t12c<R_, C_, W_, SW_, SX_, NT_, true>(Xp, ldx, m, Wp, (int)eb, etp, op, ldo, (int)n, (int)k, dg, stream); return; }
#define T12C_NT(R_, C_, W_, SW_, SX_) T12C_CASE(R_, C_, W_, SW_, SX_, 1) T12C_CASE(R_, C_, W_, SW_, SX_, 2) T12C_CASE(R_, C_, W_, SW_, SX_, 4)
#define T12C_NTD(R_, C_, W_, SW_, SX_) T12C_CASE_D(R_, C_, W_, SW_, SX_, 1) T12C_CASE_D(R_, C_, W_, SW_, SX_, 2) T12C_CASE_D(R_, C_, W_, SW_, SX_, 4)
    T12C_NT(32, 4, 8, 4, 2)
    T12C_NT(32, 4, 8, 5, 2)
    T12C_NT(32, 4, 8, 6, 2)
    T12C_NT(32, 4, 8, 5, 3)
    T12C_NT(32, 2, 8, 8, 3)
    T12C_NT(32, 2, 8, 12, 3)
    T12C_NT(16, 4, 8, 4, 2)
    T12C_NT(16, 4, 4, 4, 2)
    T12C_NT(16, 4, 4, 6, 2)
    T12C_NT(16, 2, 4, 8, 3)
    T12C_NTD(32, 4, 8, 5, 2)
    T12C_NTD(16, 4, 8, 4, 2)
#undef T12C_NT
#undef T12C_NTD
#undef T12C_CASE
#undef T12C_CASE_D
'''
    s = s.replace(old_cases, new_cases)
    w.write_text(s)

e = root / "tools/_edit_t12.py"; s = e.read_text()
s = s.replace("int64_t nsw, int64_t nsx, int64_t nt, bool diag_on, int64_t variant);\\n\")", "int64_t nsw, int64_t nsx, int64_t nt, bool diag_on, int64_t variant, torch::Tensor etab);\\n\")")
PY_OLD = s[s.index("def _repack_t12("):s.index("# == END t12 python ==")]
PY_NEW = '''def _repack_t12(w12: torch.Tensor, n: int, k: int, r: int, ck: int) -> torch.Tensor:
    """The w12 fragment layout [n/64, k/128, 4, 3072] regrouped for one bulk copy per (R-row tile, 128*ck-wide chunk):
    uint8 [n/r, k/(128 ck), r/16, ck, 3072] (same bytes; tile 16-row index = T*(r/16) + rt, 128-chunk = c*ck + j)."""
    t16 = w12.view(n // 64, k // 128, 4, 3072).permute(0, 2, 1, 3).reshape(n // 16, k // 128, 3072)
    return t16.view(n // r, r // 16, k // (128 * ck), ck, 3072).permute(0, 2, 1, 3, 4).contiguous()


T12_ESC_CAP = 64


def _esc_tab(pk, n: int, k: int, r: int):
    """Per r-row tile escape table for the t12c kernel: int32 [n/r, 64, 2] = (k | local_row << 16, delta bits), entries
    sorted by k within the tile, sentinel k = 0xFFFF; None if any tile has more than 64 entries."""
    dev = pk["esc_roff"].device
    nt = n // r
    tab = torch.full((nt, T12_ESC_CAP, 2), 0xFFFF, dtype=torch.int32, device=dev)
    tab[:, :, 1] = 0
    if pk["n_esc"] > 0:
        roff = pk["esc_roff"][: n + 1].long()
        rows = torch.repeat_interleave(torch.arange(n, device=dev), roff[1:] - roff[:-1])
        ek = pk["esc_k"][: rows.numel()].long(); ed = pk["esc_delta"][: rows.numel()]
        tile = rows // r
        counts = torch.bincount(tile, minlength=nt)
        if int(counts.max()) > T12_ESC_CAP:
            return None
        order = torch.argsort(tile * k + ek, stable=True)
        tile_s, ek_s, ed_s, rows_s = tile[order], ek[order], ed[order], rows[order]
        start = torch.cumsum(counts, 0) - counts
        pos = torch.arange(rows.numel(), device=dev) - start[tile_s]
        tab[tile_s, pos, 0] = (ek_s | ((rows_s % r) << 16)).to(torch.int32)
        tab[tile_s, pos, 1] = ed_s.view(torch.int32)
    return tab.contiguous()
'''
s = s.replace(PY_OLD, PY_NEW)
e.write_text(s)

# ---------------- submission.py glue ----------------
sp = root / "muse_bf16_proj/submission.py"; t = sp.read_text()


def rep(old, new, count=1):
    global t
    assert t.count(old) == count, (t.count(old), old[:80])
    t = t.replace(old, new)


if "_TB12_ON" not in t:
    rep('cfg["nsw"], cfg["nsx"], cfg["nt"], bool(cfg.get("diag")), int(cfg.get("v", 2)))',
        'cfg["nsw"], cfg["nsx"], cfg["nt"], bool(cfg.get("diag")), int(cfg.get("v", 3)), et)')
    rep('        wt = state.get(f"{layer}_t12_{cfg[\'r\']}x{cfg[\'ck\']}")\n        if pk is None or wt is None:\n',
        '        wt = state.get(f"{layer}_t12_{cfg[\'r\']}x{cfg[\'ck\']}")\n        et = state.get(f"{layer}_esc_{cfg[\'r\']}")\n'
        '        if pk is None or wt is None or (int(cfg.get("v", 3)) == 3 and et is None):\n')
    rep('        if pk is None or wt is None or (int(cfg.get("v", 3)) == 3 and et is None):\n            return _launch(x, w, w2, out, state, _triton_config(layer, m, n, k), m, n, k, layer)\n',
        '        if pk is None or wt is None or (int(cfg.get("v", 3)) == 3 and et is None):\n            return _launch(x, w, w2, out, state, _triton_config(layer, m, n, k), m, n, k, layer)\n'
        '        if et is None:\n            et = state["diag"]\n')
    rep("        r, ck, w, nsw, nsx = TB12_SHAPES[(layer, n, m)]\n        return dict(tb12=True, r=r, ck=ck, w=w, nsw=nsw, nsx=nsx, nt=tb12_nt(m), v=2)",
        "        r, ck, w, nsw, nsx = TB12_SHAPES[(layer, n, m)]\n        return dict(tb12=True, r=r, ck=ck, w=w, nsw=nsw, nsx=nsx, nt=tb12_nt(m), v=3)")
    rep("    if (layer, n, m) in TB12_SHAPES:\n", "    if _TB12_ON and (layer, n, m) in TB12_SHAPES:\n")
    rep("TB12_SHAPES = {}\n",
        "TB12_SHAPES = {}\n"
        "_TB12_ON = False   # set by prepare() after the tb12 self-test (table output + probe rows) passes\n")
    rep('                        for r_, ck_ in sorted(layouts):\n                            state[f"{name}_t12_{r_}x{ck_}"] = _repack_t12(state[name + "_w12"], nn_, kk_, r_, ck_)\n',
        '                        for r_, ck_ in sorted(layouts):\n                            state[f"{name}_t12_{r_}x{ck_}"] = _repack_t12(state[name + "_w12"], nn_, kk_, r_, ck_)\n'
        '                        for r_ in sorted({c[0] for c in layouts}):\n                            state[f"{name}_esc_{r_}"] = _esc_tab(pk, nn_, kk_, r_)\n')
    # self-test + enable in prepare
    rep("    if BENCH:\n        try:\n            _exp_diag(state)\n            _bench(state)\n",
        "    if TB12_SHAPES:\n        global _TB12_ON\n        _TB12_ON = False\n        try:\n            _TB12_ON = _tb12_selftest(state)\n        except Exception as e:\n            print('tb12 self-test failed:', repr(e)[:200])\n            _TB12_ON = False\n        print('tb12 dispatch', 'ON' if _TB12_ON else 'OFF', sorted(TB12_SHAPES))\n"
        "    if BENCH:\n        try:\n            _exp_diag(state)\n            _bench(state)\n")
    rep("def prepare(weights: Weights, warm: bool = True):\n",
        '''def _tb12_selftest(state) -> bool:
    """Every TB12_SHAPES entry: the tb12 output must match the table path (rel L2 < 2e-3) on random x and reproduce
    the weight columns exactly on judge-style probe rows (one-hot x at escape positions)."""
    global _OVERRIDE
    dev = state["router"].device
    ok = True
    for (layer, n, m), c in sorted(TB12_SHAPES.items()):
        k = state[layer].shape[1]
        cfg = dict(tb12=True, r=c[0], ck=c[1], w=c[2], nsw=c[3], nsx=c[4], nt=tb12_nt(m), v=3)
        pk = state.get(layer + "_p12")
        if pk is None or not _usable(cfg, m, n, k) or state.get(f"{layer}_esc_{c[0]}") is None:
            print(f"  tb12 self-test {layer} m={m}: not usable"); ok = False; continue
        x = torch.randn((m, k), device=dev).to(torch.bfloat16)
        kk = pk["esc_k"][:m].long()
        if kk.numel() < m:
            kk = torch.cat([kk, torch.arange(m - kk.numel(), device=dev)])
        xp = torch.zeros((m, k), dtype=torch.bfloat16, device=dev)
        xp[torch.arange(m, device=dev), kk] = 2.0
        _OVERRIDE = {"table": True}
        try:
            ref = _proj_forward(x, layer, n, state).float()
        finally:
            _OVERRIDE = None
        _OVERRIDE = cfg
        try:
            out = _proj_forward(x, layer, n, state).float()
            op = _proj_forward(xp, layer, n, state).float()
        finally:
            _OVERRIDE = None
        torch.cuda.synchronize()
        err = float((out - ref).norm() / ref.norm())
        want = state[layer][:n].float()[:, kk].T * 2.0
        prel = float(((op - want).abs() / want.abs().clamp(min=1e-30)).max())
        good = err < 2e-3 and prel < 1e-3 and bool(torch.isfinite(out).all())
        print(f"  tb12 self-test {layer:9s} m={m:2d} {c}: rel {err:.1e} probe {prel:.1e} -> {'ok' if good else 'FAIL'}")
        ok = ok and good
    return ok


def prepare(weights: Weights, warm: bool = True):
''')
sp.write_text(t)
print("ok")
