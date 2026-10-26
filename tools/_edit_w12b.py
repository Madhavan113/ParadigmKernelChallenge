"""Proj edit: w12b = single-launch 12-bit GEMM: one 16-row tile per CTA over the full K, W warps each streaming a
private K slice through its own cp.async ring (no CTA barriers in the loop), x B-fragments straight from L2 with
one-chunk register prefetch, partial accumulators reduced through smem, the tile's escapes applied in the epilogue."""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
p = root / "muse_bf16_proj/cuda_proj.py"
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


kernel = r"""
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
"""
rep("\n} // namespace\n\nvoid stream_bytes(", kernel + "\n} // namespace\n\nvoid stream_bytes(")

wrapper = r"""
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
"""
a = s.index("void mma12cp(torch::Tensor X, torch::Tensor LO")
b = s.index('"""', a)
s = s[:b] + wrapper + s[b:]
rep("""void w12(torch::Tensor X, torch::Tensor W, int64_t eb, torch::Tensor esc_off, torch::Tensor esc_n, torch::Tensor esc_k, torch::Tensor esc_d,
         torch::Tensor out, torch::Tensor part, int64_t m, int64_t n, int64_t k, int64_t nt, int64_t nstage, int64_t sk, int64_t mode);""",
    """void w12(torch::Tensor X, torch::Tensor W, int64_t eb, torch::Tensor esc_off, torch::Tensor esc_n, torch::Tensor esc_k, torch::Tensor esc_d,
         torch::Tensor out, torch::Tensor part, int64_t m, int64_t n, int64_t k, int64_t nt, int64_t nstage, int64_t sk, int64_t mode);
void w12b(torch::Tensor X, torch::Tensor W, int64_t eb, torch::Tensor esc_off, torch::Tensor esc_n, torch::Tensor esc_k, torch::Tensor esc_d,
          torch::Tensor out, int64_t m, int64_t n, int64_t k, int64_t nt, int64_t warps, int64_t nst);""")
rep('functions=["stream_bytes", "gemv", "skinny_mma", "tgemv", "gemv12", "mma12", "mma12cp", "w12"]',
    'functions=["stream_bytes", "gemv", "skinny_mma", "tgemv", "gemv12", "mma12", "mma12cp", "w12", "w12b"]')
rep("""def w12_usable(m: int, n: int, k: int, nt: int, nstage: int, sk: int) -> bool:""",
    """def w12b_usable(m: int, n: int, k: int, nt: int, warps: int, nst: int) -> bool:
    return m <= 8 * nt and m <= 32 and n % 64 == 0 and k % 128 == 0 and (k // 128) % warps == 0


def w12_usable(m: int, n: int, k: int, nt: int, nstage: int, sk: int) -> bool:""")
p.write_text(s)

q = root / "muse_bf16_proj/submission.py"
t = q.read_text()


def rep2(old, new, count=1):
    global t
    assert t.count(old) == count, (t.count(old), old[:90])
    t = t.replace(old, new)


rep2("""W12_SHAPES = {}""", """W12_SHAPES = {}
# w12b: single-launch form (16-row tile per CTA, W warps over K slices): (layer, n, m) -> (nt, warps, nst)
W12B_SHAPES = {}""")
rep2("""    if (layer, n, m) in W12_SHAPES:
        nt, nstage, sk = W12_SHAPES[(layer, n, m)]
        return dict(w12=True, nt=nt, nstage=nstage, sk=sk)""",
     """    if (layer, n, m) in W12_SHAPES:
        nt, nstage, sk = W12_SHAPES[(layer, n, m)]
        return dict(w12=True, nt=nt, nstage=nstage, sk=sk)
    if (layer, n, m) in W12B_SHAPES:
        nt, warps, nst = W12B_SHAPES[(layer, n, m)]
        return dict(w12b=True, nt=nt, warps=warps, nst=nst)""")
rep2("""    if cfg.get("w12"):
        return w12_usable(m, n, k, cfg["nt"], cfg["nstage"], cfg["sk"]) and cfg["sk"] * m * n <= PARTIALS_FLOATS""",
     """    if cfg.get("w12"):
        return w12_usable(m, n, k, cfg["nt"], cfg["nstage"], cfg["sk"]) and cfg["sk"] * m * n <= PARTIALS_FLOATS
    if cfg.get("w12b"):
        return w12b_usable(m, n, k, cfg["nt"], cfg["warps"], cfg["nst"])""")
rep2("""    if cfg.get("w12"):
        pk = state.get(layer + "_p12")
        w12 = state.get(layer + "_w12")
        if pk is None or w12 is None:
            return _launch(x, w, w2, out, state, _triton_config(layer, m, n, k), m, n, k, layer)""",
     """    if cfg.get("w12b"):
        pk = state.get(layer + "_p12")
        w12 = state.get(layer + "_w12")
        if pk is None or w12 is None:
            return _launch(x, w, w2, out, state, _triton_config(layer, m, n, k), m, n, k, layer)
        build().w12b(x, w12, pk["eb"], pk["esc_off"], pk["esc_n"], pk["esc_k"], pk["esc_delta"], out,
                     m, n, k, cfg["nt"], cfg["warps"], cfg["nst"])
        return out
    if cfg.get("w12"):
        pk = state.get(layer + "_p12")
        w12 = state.get(layer + "_w12")
        if pk is None or w12 is None:
            return _launch(x, w, w2, out, state, _triton_config(layer, m, n, k), m, n, k, layer)""")
# bench: replace the w12 sections with w12b rows
a = t.index('        print("BENCH w12 (MoE-style weights-as-M 12-bit GEMM; nt/nstage/sk) vs the table at m=8/16/32")')
b = t.index("    _OVERRIDE = None\n    print(f\"BENCH proj done in {time.time() - t0:.0f}s\")")
t = t[:a] + '''        print("BENCH w12b (single launch: 16-row tile per CTA, W warps over K slices; nt/warps/nst) vs the table")
        for layer, n in (("pooler", 4096), ("unpooler", 8192), ("qkv_gate", 2304), ("o", 8192), ("unpooler", 1024)):
            for m in (8, 16, 32):
                if not budget():
                    break
                if layer == "unpooler" and ((n == 1024) != (m <= 8)):
                    continue
                nt = 1 if m <= 8 else 2 if m <= 16 else 4
                cands = {1: [(4, 3), (8, 2), (8, 3), (16, 2), (16, 3)], 2: [(4, 2), (8, 2), (8, 3), (16, 2)], 4: [(4, 2), (8, 2)]}[nt]
                parts = [row("table", layer, m, n, TAB), row("w12 old", layer, m, n, dict(w12=True, nt=nt, nstage=4, sk=4 if layer != "o" else 2))]
                for ww, ns in cands:
                    parts.append(row(f"b {nt}/{ww}/{ns}", layer, m, n, dict(w12b=True, nt=nt, warps=ww, nst=ns)))
                print(f"  m={m:2d} {layer:9s} n={n}: " + " | ".join(parts))
''' + t[b:]
rep2("BENCH = False       # microbenchmarks in prepare", "BENCH = True        # microbenchmarks in prepare")
q.write_text(t)
print("edited")
