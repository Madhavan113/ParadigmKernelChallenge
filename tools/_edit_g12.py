"""Phase 2: g12 = 12-bit weights x BF16 x^T for m = 128 / 256 on wgmma (RS form: A = decoded 12-bit fragments in registers,
B = the x tile in shared memory through a K-major 128B-swizzled matrix descriptor), K split across CTAs, fp32 partials
+ a PDL reduce that adds the escapes. Generates tools/g12_kernel.cu.part / g12_wrapper.cu.part (the wgmma asm has 64
accumulator operands), splices them into cuda_proj.py (markers), adds the Python glue to submission.py and the bench.
Run: python3 tools/_edit_g12.py && python3 tools/embed_proj.py"""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- wgmma m64n128k16 RS asm (64 f32 accumulators)
outs = ", ".join(f"%{i}" for i in range(64))
cons_out = ", ".join(f'"+f"(d[{i}])' for i in range(64))
WGMMA = f'''__device__ __forceinline__ void wgmma_m64n128k16_rs(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {{
    asm volatile("{{\\n .reg .pred p;\\n setp.ne.b32 p, %69, 0;\\n"
                 "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {{{outs}}}, {{%64, %65, %66, %67}}, %68, p, 1, 1, 0;\\n}}\\n"
                 : {cons_out}
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}}
__device__ __forceinline__ void acc_fence(float* d) {{
    asm volatile("" : {cons_out} :: "memory");
}}
'''

KERNEL = r'''// == BEGIN g12 device ==
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
WGMMA_PLACEHOLDER
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
'''.replace("WGMMA_PLACEHOLDER\n", WGMMA)

WRAPPER = r'''// == BEGIN g12 host ==
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
'''

(root / "tools/g12_kernel.cu.part").write_text(KERNEL)
(root / "tools/g12_wrapper.cu.part").write_text(WRAPPER)

cp = root / "muse_bf16_proj/cuda_proj.py"; s = cp.read_text()


def put_region(text, begin, end, body, anchor):
    if begin in text:
        a = text.index(begin); b = text.index(end) + len(end)
        return text[:a] + body.rstrip("\n") + text[b:]
    assert text.count(anchor) == 1, anchor
    i = text.index(anchor)
    return text[:i] + body.rstrip("\n") + "\n\n" + text[i:]


s = put_region(s, "// == BEGIN g12 device ==", "// == END g12 device ==", KERNEL, "} // namespace\n\nvoid stream_bytes(")
s = put_region(s, "// == BEGIN g12 host ==", "// == END g12 host ==", WRAPPER, '\n"""\n\nCPP_SRC = """')
decl = ("void g12(torch::Tensor X, torch::Tensor Wt, int64_t eb, torch::Tensor esc_roff, torch::Tensor esc_k, torch::Tensor esc_d,\n"
        "         torch::Tensor out, torch::Tensor part, torch::Tensor diag, int64_t n, int64_t k,\n"
        "         int64_t rb, int64_t tb, int64_t ck, int64_t nsw, int64_t nsx, int64_t ks, bool diag_on, bool pdl, int64_t mode);\n")
if "void g12(" not in s.split('CPP_SRC = """')[1]:
    s = s.replace('bool diag_on, int64_t variant, torch::Tensor etab);\n"""', 'bool diag_on, int64_t variant, torch::Tensor etab);\n' + decl + '"""')
if '"t12", "g12"' not in s:
    s = s.replace('"bstream", "t12"]', '"bstream", "t12", "g12"]')
    s = s.replace('name="okc_proj_cuda14"', 'name="okc_proj_cuda15"')
PY = '''

# == BEGIN g12 python ==
def g12_usable(m: int, n: int, k: int, rb: int, tb: int, ck: int, ks: int) -> bool:
    return 2 <= m <= 128 * tb and m % 2 == 0 and n % (64 * rb) == 0 and k % (128 * ck * ks) == 0 and ks * m * n <= 16 << 20
# == END g12 python ==
'''
if "# == BEGIN g12 python ==" not in s:
    s = s.rstrip("\n") + "\n" + PY
cp.write_text(s)
print("cuda_proj.py updated")

# ---------------------------------------------------------------- submission.py glue
sp = root / "muse_bf16_proj/submission.py"; t = sp.read_text()


def rep(old, new, count=1):
    global t
    assert t.count(old) == count, (t.count(old), old[:80])
    t = t.replace(old, new)


if "WG12_SHAPES" not in t:
    rep("_TB12_ON = False   # set by prepare() after the tb12 self-test (table output + probe rows) passes\n",
        "_TB12_ON = False   # set by prepare() after the tb12 self-test (table output + probe rows) passes\n"
        "# g12: the wgmma 12-bit GEMM for m = 128 / 256 (RB x TB warpgroups of 64 rows x 128 tokens, K split ks, PDL reduce + escapes):\n"
        "# (layer, n, m) -> (rb, tb, ck, nsw, nsx, ks); dispatched only after the prepare self-test (_WG12_ON)\n"
        "WG12_SHAPES = {}\n"
        "_WG12_ON = False\n"
        "G12_LAYOUTS = {(256, 1), (128, 1), (128, 2), (64, 1)}   # (r, ck) weight layouts for the g12 bench (BENCH)\n")
    rep("    if _TB12_ON and (layer, n, m) in TB12_SHAPES:\n",
        "    if _WG12_ON and (layer, n, m) in WG12_SHAPES:\n        rb, tb, ck, nsw, nsx, ks = WG12_SHAPES[(layer, n, m)]\n"
        "        return dict(wg12=True, rb=rb, tb=tb, ck=ck, nsw=nsw, nsx=nsx, ks=ks)\n"
        "    if _TB12_ON and (layer, n, m) in TB12_SHAPES:\n")
    rep('    if cfg.get("tb12"):\n        return tb12_usable(m, n, k, cfg["r"], cfg["ck"], cfg["nt"])\n',
        '    if cfg.get("tb12"):\n        return tb12_usable(m, n, k, cfg["r"], cfg["ck"], cfg["nt"])\n'
        '    if cfg.get("wg12"):\n        return g12_usable(m, n, k, cfg["rb"], cfg["tb"], cfg["ck"], cfg["ks"])\n')
    rep('    if cfg.get("tb12"):\n        pk = state.get(layer + "_p12")\n',
        '    if cfg.get("g12"):\n'
        '        pk = state.get(layer + "_p12")\n'
        '        wt = state.get(f"{layer}_t12_{64 * cfg[\'rb\']}x{cfg[\'ck\']}")\n'
        '        if pk is None or wt is None:\n'
        '            return torch.matmul(x, w.t(), out=out)\n'
        '        build().g12(x, wt, pk["eb"], pk["esc_roff"], pk["esc_k"], pk["esc_delta"], out, state["partials"], state["diag"], n, k,\n'
        '                    cfg["rb"], cfg["tb"], cfg["ck"], cfg["nsw"], cfg["nsx"], cfg["ks"], bool(cfg.get("diag")), True, int(cfg.get("mode", 0)))\n'
        '        return out\n'
        '    if cfg.get("tb12"):\n        pk = state.get(layer + "_p12")\n')
    rep('                        layouts = set(T12_LAYOUTS) if BENCH else {(c[0], c[1]) for (l, _, _), c in TB12_SHAPES.items() if l == name}\n',
        '                        layouts = set(T12_LAYOUTS) if BENCH else {(c[0], c[1]) for (l, _, _), c in TB12_SHAPES.items() if l == name}\n'
        '                        layouts |= set(G12_LAYOUTS) if BENCH else {(64 * c[0], c[2]) for (l, _, _), c in WG12_SHAPES.items() if l == name}\n')
    rep("    if TB12_SHAPES:\n        global _TB12_ON\n",
        '''    if WG12_SHAPES:
        global _WG12_ON
        _WG12_ON = False
        try:
            _WG12_ON = _g12_selftest(state)
        except Exception as e:
            print('g12 self-test failed:', repr(e)[:200])
            _WG12_ON = False
        print('g12 dispatch', 'ON' if _WG12_ON else 'OFF', sorted(WG12_SHAPES))
    if TB12_SHAPES:
        global _TB12_ON
''')
    rep("def _tb12_selftest(state) -> bool:\n",
        '''def _g12_selftest(state) -> bool:
    """Every WG12_SHAPES entry: the g12 output must match cuBLAS (rel L2 < 2e-3) and reproduce the weight columns on
    probe rows (one-hot x at escape positions)."""
    global _OVERRIDE
    dev = state["router"].device
    ok = True
    for (layer, n, m), c in sorted(WG12_SHAPES.items()):
        k = state[layer].shape[1]
        cfg = dict(wg12=True, rb=c[0], tb=c[1], ck=c[2], nsw=c[3], nsx=c[4], ks=c[5])
        pk = state.get(layer + "_p12")
        if pk is None or not _usable(cfg, m, n, k) or state.get(f"{layer}_t12_{64 * c[0]}x{c[2]}") is None:
            print(f"  g12 self-test {layer} m={m}: not usable"); ok = False; continue
        x = torch.randn((m, k), device=dev).to(torch.bfloat16)
        ref = torch.matmul(x, state[layer][:n].t()).float()
        kk = pk["esc_k"][:m].long()
        if kk.numel() < m:
            kk = torch.cat([kk, torch.arange(m - kk.numel(), device=dev)])
        xp = torch.zeros((m, k), dtype=torch.bfloat16, device=dev)
        xp[torch.arange(m, device=dev), kk] = 2.0
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
        print(f"  g12 self-test {layer:9s} m={m:3d} {c}: rel {err:.1e} probe {prel:.1e} -> {'ok' if good else 'FAIL'}")
        ok = ok and good
    return ok


def _tb12_selftest(state) -> bool:
''')
sp.write_text(t)
print("submission.py updated")
