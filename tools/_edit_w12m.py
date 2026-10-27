"""One-off edit: w12m, the 12-bit (lossless pack12) W x BF16 x^T GEMM for m = 128 / 256 tokens. CTA = 8 warps x MW m16
weight tiles (weights = mma M side, the W12 layout of _repack_w12 reused: a 128*MW-row tile is 2*MW consecutive 4-tile
groups), tokens = N side (8*NT), one x tile per CTA per 128-wide K chunk (cp.async, ldmatrix B fragments shared by the
warps), K split over gridDim.y CTAs into fp32 partials, then w12m_reduce (PDL) sums the slices, adds the per-row
escapes (delta x x[tok][k]) and writes bf16."""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
p = root / "muse_bf16_proj/cuda_proj.py"
s = p.read_text()


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:90])
    s = s.replace(old, new)


kernels = r'''
// ---- w12m: 12-bit W x BF16 x^T GEMM for m = 8*NT tokens (128 / 256). CTA = 8 warps, warp w owns MW m16 weight
// tiles (rows (blockIdx.x*8 + w)*MW*16 ..); the W12 layout [n/64][k/128][4][3 KB] is read as 2*MW consecutive
// 4-tile groups per chunk. All warps share the chunk's x tile (8*NT tokens x 128 k, cp.async'd once per CTA, B
// fragments by ldmatrix); the fragments of a warp's MW tiles are reused over the NT n8 token blocks. gridDim.y
// CTAs split K; each writes fp32 partials part[z][tok][n]; w12m_reduce_kernel (PDL) sums the slices, adds the
// escapes of row n (esc_roff per row: delta x x[tok][k]) and writes bf16.
constexpr int W12M_WARPS = 8;
__host__ __device__ constexpr int w12m_stage_bytes(int mw, int nt) { return W12M_WARPS * mw * W12_TILE_BYTES + 8 * nt * W12_XSTRIDE * 2; }

template <int MW, int NT, int NSTAGE>
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
    const uint8_t* wsrc[WSEG];
    #pragma unroll
    for (int i = 0; i < WSEG; ++i) {
        const int sidx = tid + i * 256;
        const int gi = sidx / (GROUP_BYTES / 16), off = sidx % (GROUP_BYTES / 16);
        wsrc[i] = W + ((size_t)(group0 + gi) * NC_ALL + c0) * GROUP_BYTES + off * 16;
    }
    const __nv_bfloat16* xsrc[XSEG]; int xbytes[XSEG];
    #pragma unroll
    for (int i = 0; i < XSEG; ++i) {
        const int sidx = tid + i * 256, r = sidx >> 4;
        xbytes[i] = r < m ? 16 : 0;
        xsrc[i] = X + (size_t)(r < m ? r : 0) * ldx + c0 * W12_KC + (sidx & 15) * 8;
    }
    const uint32_t sbase = smem_u32(w12m_smem);
    auto issue = [&](int c, int stage) {
        const uint32_t ws = sbase + stage * STAGE_BYTES;
        #pragma unroll
        for (int i = 0; i < WSEG; ++i) cp_async16(ws + (tid + i * 256) * 16, wsrc[i] + (size_t)c * GROUP_BYTES, 16);
        const uint32_t xs = ws + W_STAGE;
        #pragma unroll
        for (int i = 0; i < XSEG; ++i) {
            const int sidx = tid + i * 256;
            cp_async16(xs + ((sidx >> 4) * W12_XSTRIDE + (sidx & 15) * 8) * 2, xsrc[i] + c * W12_KC, xbytes[i]);
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
                #pragma unroll
                for (int nb = 0; nb < NT; ++nb) {
                    uint32_t b0, b1, b2, b3;
                    ldsm_x4(b0, b1, b2, b3, xb + (nb * 8 * W12_XSTRIDE + h * 64 + ps * 32) * 2);
                    #pragma unroll
                    for (int i = 0; i < MW; ++i) {
                        mma16816(acc[i][nb], a[i][0][0], a[i][0][1], a[i][0][2], a[i][0][3], b0, b1);
                        mma16816(acc[i][nb], a[i][1][0], a[i][1][1], a[i][1][2], a[i][1][3], b2, b3);
                    }
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
                if (tok + 1 < m) {
                    *reinterpret_cast<float2*>(pz + (size_t)tok * N + col0 + 8 * hr) = make_float2(acc[i][nb][2 * hr], 0.f);   // placeholder, replaced below
                }
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

template <int MW, int NT, int NSTAGE>
void launch_w12m(const __nv_bfloat16* X, int ldx, int m, const uint8_t* W, int eb, float* part, int N, int K, int sk, cudaStream_t stream)
{
    constexpr int smem = NSTAGE * w12m_stage_bytes(MW, NT);
    static bool attr = false;
    if (!attr) { cudaFuncSetAttribute(w12m_kernel<MW, NT, NSTAGE>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
    w12m_kernel<MW, NT, NSTAGE><<<dim3(N / (128 * MW), sk), 256, smem, stream>>>(X, ldx, m, W, eb, part, N, K);
}

} // namespace
'''
rep("\n} // namespace\n", kernels)
# the partial store: plain scalar stores (tokens 2t, 2t+1 are rows apart)
rep("""                const int tok = nb * 8 + 2 * t;
                if (tok + 1 < m) {
                    *reinterpret_cast<float2*>(pz + (size_t)tok * N + col0 + 8 * hr) = make_float2(acc[i][nb][2 * hr], 0.f);   // placeholder, replaced below
                }""", """                const int tok = nb * 8 + 2 * t;
                if (tok < m) pz[(size_t)tok * N + col0 + 8 * hr] = acc[i][nb][2 * hr];
                if (tok + 1 < m) pz[(size_t)(tok + 1) * N + col0 + 8 * hr] = acc[i][nb][2 * hr + 1];""")

host = r'''
void w12m(torch::Tensor X, torch::Tensor W, int64_t eb, torch::Tensor esc_roff, torch::Tensor esc_k, torch::Tensor esc_d,
          torch::Tensor out, torch::Tensor part, int64_t m, int64_t n, int64_t k, int64_t mw, int64_t nt, int64_t nstage, int64_t sk, bool pdl)
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
    switch (key) {
        case 1163: OKC_W12M(1, 16, 3); break;   case 1162: OKC_W12M(1, 16, 2); break;
        case 2162: OKC_W12M(2, 16, 2); break;   case 1322: OKC_W12M(1, 32, 2); break;
        default: TORCH_CHECK(false, "unsupported w12m config (mw, nt, nstage)");
    }
#undef OKC_W12M
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
"""
'''
rep('''        default: TORCH_CHECK(false, "unsupported w12b config (nt, warps, nst)");
    }
#undef OKC_B
}
"""
''', '''        default: TORCH_CHECK(false, "unsupported w12b config (nt, warps, nst)");
    }
#undef OKC_B
}
''' + host)
rep('''void w12b(torch::Tensor X, torch::Tensor W, int64_t eb, torch::Tensor esc_off, torch::Tensor esc_n, torch::Tensor esc_k, torch::Tensor esc_d,
          torch::Tensor out, int64_t m, int64_t n, int64_t k, int64_t nt, int64_t warps, int64_t nst);
"""''', '''void w12b(torch::Tensor X, torch::Tensor W, int64_t eb, torch::Tensor esc_off, torch::Tensor esc_n, torch::Tensor esc_k, torch::Tensor esc_d,
          torch::Tensor out, int64_t m, int64_t n, int64_t k, int64_t nt, int64_t warps, int64_t nst);
void w12m(torch::Tensor X, torch::Tensor W, int64_t eb, torch::Tensor esc_roff, torch::Tensor esc_k, torch::Tensor esc_d,
          torch::Tensor out, torch::Tensor part, int64_t m, int64_t n, int64_t k, int64_t mw, int64_t nt, int64_t nstage, int64_t sk, bool pdl);
"""''')
rep('"mma12", "mma12cp", "w12", "w12b"]', '"mma12", "mma12cp", "w12", "w12b", "w12m"]')
rep("""def w12_usable(m: int, n: int, k: int, nt: int, nstage: int, sk: int) -> bool:""",
    """def w12m_usable(m: int, n: int, k: int, mw: int, nt: int, nstage: int, sk: int) -> bool:
    return 1 <= m <= 8 * nt and n % (128 * mw) == 0 and k % 128 == 0 and (k // 128) % sk == 0 and (k // 128) // sk >= 1


def w12_usable(m: int, n: int, k: int, nt: int, nstage: int, sk: int) -> bool:""")
p.write_text(s)
print("cuda w12m written")

q = root / "muse_bf16_proj/submission.py"
t = q.read_text()


def rep2(old, new, n=1):
    global t
    assert t.count(old) == n, (t.count(old), old[:90])
    t = t.replace(old, new)


rep2("PARTIALS_FLOATS = 8 << 20  # 32 MB fp32 workspace", "PARTIALS_FLOATS = 16 << 20  # 64 MB fp32 workspace")
rep2("""W12B_SHAPES = {}   # w12 ties the table""", """# w12m: the 12-bit GEMM for m = 128 / 256 (weights = mma M side, 128*mw-row CTA tiles, 8*nt tokens, K split sk):
# (layer, n, m) -> (mw, nt, nstage, sk); empty until the judge table shows a win
W12M_SHAPES = {}
W12B_SHAPES = {}   # w12 ties the table""")
rep2("""    if (layer, n, m) in W12B_SHAPES:
        nt, warps, nst = W12B_SHAPES[(layer, n, m)]
        return dict(w12b=True, nt=nt, warps=warps, nst=nst)""", """    if (layer, n, m) in W12M_SHAPES:
        mw, nt, nstage, sk = W12M_SHAPES[(layer, n, m)]
        return dict(w12m=True, mw=mw, nt=nt, nstage=nstage, sk=sk)
    if (layer, n, m) in W12B_SHAPES:
        nt, warps, nst = W12B_SHAPES[(layer, n, m)]
        return dict(w12b=True, nt=nt, warps=warps, nst=nst)""")
rep2("""    if cfg.get("w12b"):
        return w12b_usable(m, n, k, cfg["nt"], cfg["warps"], cfg["nst"])""", """    if cfg.get("w12m"):
        return w12m_usable(m, n, k, cfg["mw"], cfg["nt"], cfg["nstage"], cfg["sk"]) and cfg["sk"] * m * n <= PARTIALS_FLOATS
    if cfg.get("w12b"):
        return w12b_usable(m, n, k, cfg["nt"], cfg["warps"], cfg["nst"])""")
rep2("""    if cfg.get("w12b"):
        pk = state.get(layer + "_p12")
        w12 = state.get(layer + "_w12")
        if pk is None or w12 is None:
            return _launch(x, w, w2, out, state, _triton_config(layer, m, n, k), m, n, k, layer)
        build().w12b(""", """    if cfg.get("w12m"):
        pk = state.get(layer + "_p12")
        w12 = state.get(layer + "_w12")
        if pk is None or w12 is None:
            return torch.matmul(x, w.t(), out=out)
        build().w12m(x, w12, pk["eb"], pk["esc_roff"], pk["esc_k"], pk["esc_delta"], out, state["partials"],
                     m, n, k, cfg["mw"], cfg["nt"], cfg["nstage"], cfg["sk"], True)
        return out
    if cfg.get("w12b"):
        pk = state.get(layer + "_p12")
        w12 = state.get(layer + "_w12")
        if pk is None or w12 is None:
            return _launch(x, w, w2, out, state, _triton_config(layer, m, n, k), m, n, k, layer)
        build().w12b(""")
# bench: the m = 128 / 256 plan replaces the old m = 1 GEMV sweep
a = t.index('        print("BENCH m=1 12-bit GEMV (warps/unroll/sk) vs the current path and the Triton table")')
b = t.index("    _OVERRIDE = None\n    print(f\"BENCH proj done in {time.time() - t0:.0f}s\")")
t = t[:a] + '''        print("BENCH m=128/256 12-bit GEMM w12m (mw/nt/nstage/sk) vs the current path (cuBLAS / bmm split-K)")
        plan = [
            (128, "pooler", 4096, [(1, 16, 3, 4), (1, 16, 3, 8), (1, 16, 2, 4), (2, 16, 2, 8), (2, 16, 2, 4)]),
            (128, "qkv_gate", 2304, [(1, 16, 3, 8), (1, 16, 3, 4), (1, 16, 2, 8), (2, 16, 2, 8)]),
            (128, "unpooler", 8192, [(1, 16, 3, 2), (1, 16, 3, 4), (1, 16, 2, 2), (2, 16, 2, 4), (2, 16, 2, 2)]),
            (128, "o", 8192, [(1, 16, 3, 1), (1, 16, 3, 2), (1, 16, 2, 2), (2, 16, 2, 2), (2, 16, 2, 1)]),
            (256, "pooler", 4096, [(1, 32, 2, 4), (1, 32, 2, 8), (1, 32, 2, 2)]),
            (256, "qkv_gate", 2304, [(1, 32, 2, 8), (1, 32, 2, 4)]),
            (256, "unpooler", 8192, [(1, 32, 2, 2), (1, 32, 2, 4)]),
            (256, "o", 8192, [(1, 32, 2, 1), (1, 32, 2, 2)]),
        ]
        for m, layer, n, cands in plan:
            if not budget():
                break
            parts = [row("current", layer, m, n, None)]
            for mw, nt, ns, sk in cands:
                parts.append(row(f"w12m {mw}/{nt}/{ns}/{sk}", layer, m, n, dict(w12m=True, mw=mw, nt=nt, nstage=ns, sk=sk)))
            print(f"  m={m} {layer:9s} n={n}: " + " | ".join(parts))
''' + t[b:]
q.write_text(t)
print("submission w12m written")
