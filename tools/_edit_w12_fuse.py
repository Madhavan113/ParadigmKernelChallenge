"""Proj edit: fuse the K-split reduction and the escapes into the w12 GEMM epilogue (last-arriving CTA per 64-row
tile group via a self-resetting ticket), drop the separate reduce launch; W12_SHAPES for pooler/qkv at m=8/16/32."""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
p = root / "muse_bf16_proj/cuda_proj.py"
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


rep("""w12_kernel(const __nv_bfloat16* __restrict__ X, int ldx, int m, const uint8_t* __restrict__ W, int eb,
           float* __restrict__ part, int N, int K)
{""", """w12_kernel(const __nv_bfloat16* __restrict__ X, int ldx, int m, const uint8_t* __restrict__ W, int eb,
           float* __restrict__ part, int N, int K,
           const int* __restrict__ esc_off, const int* __restrict__ esc_n, const int* __restrict__ esc_k, const float* __restrict__ esc_d,
           __nv_bfloat16* __restrict__ out, int ldo, int* __restrict__ cnt)
{
    // epilogue: SPLITK: every CTA stores its fp32 slice; the last CTA to arrive at cnt[tile group] sums the slices
    // (L2-hot), adds the group's escapes (two 32-row pack tiles) and writes bf16. KSPLIT == 1: the CTA's own sums.
    __shared__ float red[64][33];
    __shared__ int s_last;""")
a = s.index("    // partial slice: rows (weight rows) g, g+8 of this warp's tile -> columns; tokens 2t, 2t+1\n    const int col0 = (blockIdx.x * 4 + warp) * 16 + g;")
b = s.index("// sum the KSPLIT partial slices of one 32-row pack tile")
new_epi = r"""    // C fragment: rows (weight rows) g, g+8 of this warp's tile -> columns; tokens 2t, 2t+1
    const int col0 = (blockIdx.x * 4 + warp) * 16 + g;                  // global output column (row of W)
    if constexpr (SPLITK) {
        float* pz = part + (size_t)blockIdx.y * m * N;
        #pragma unroll
        for (int nbk = 0; nbk < NT; ++nbk)
            #pragma unroll
            for (int hr = 0; hr < 2; ++hr)
                #pragma unroll
                for (int cc = 0; cc < 2; ++cc) {
                    const int tok = nbk * 8 + 2 * t + cc;
                    if (tok < m) pz[(size_t)tok * N + col0 + 8 * hr] = acc[nbk * CH][2 * hr + cc];
                }
        __threadfence();
        __syncthreads();
        if (tid == 0) {
            const int old = atomicAdd(cnt + blockIdx.x, 1);
            s_last = (old == KSPLIT - 1);
            if (old == KSPLIT - 1) cnt[blockIdx.x] = 0;
        }
        __syncthreads();
        if (!s_last) return;
        __threadfence();
        for (int idx = tid; idx < 64 * m; idx += 128) {
            const int n = idx & 63, i = idx >> 6;
            float v = 0.f;
            for (int z = 0; z < KSPLIT; ++z) v += __ldcg(part + ((size_t)z * m + i) * N + blockIdx.x * 64 + n);
            red[n][i] = v;
        }
    } else {
        #pragma unroll
        for (int nbk = 0; nbk < NT; ++nbk)
            #pragma unroll
            for (int hr = 0; hr < 2; ++hr)
                #pragma unroll
                for (int cc = 0; cc < 2; ++cc) {
                    const int tok = nbk * 8 + 2 * t + cc;
                    if (tok < m) red[warp * 16 + g + 8 * hr][tok] = acc[nbk * CH][2 * hr + cc];
                }
    }
    __syncthreads();
    // escapes of pack tiles 2*bx, 2*bx + 1: (n_local, k, delta) * x[i][k], parallel over (escape, token)
    #pragma unroll
    for (int pt = 0; pt < 2; ++pt) {
        const int e0 = esc_off[2 * blockIdx.x + pt], ecnt = esc_off[2 * blockIdx.x + pt + 1] - e0;
        for (int idx = tid; idx < ecnt * m; idx += 128) {
            const int e = e0 + idx / m, i = idx - (idx / m) * m;
            atomicAdd(&red[pt * 32 + esc_n[e]][i], esc_d[e] * __bfloat162float(X[(size_t)i * ldx + esc_k[e]]));
        }
    }
    __syncthreads();
    for (int idx = tid; idx < 64 * m; idx += 128) {
        const int n = idx & 63, i = idx >> 6;
        out[(size_t)i * ldo + blockIdx.x * 64 + n] = __float2bfloat16(red[n][i]);
    }
}

"""
s = s[:a] + new_epi + s[b:]
rep("""template <int NT, int NSTAGE, bool SPLITK, int MODE = 0>
void launch_w12(const __nv_bfloat16* X, int ldx, int m, const uint8_t* W, int eb, float* part, int N, int K, int sk, cudaStream_t stream)
{
    constexpr int smem = NSTAGE * w12_stage_bytes(NT);
    static bool attr = false;
    if (!attr) { cudaFuncSetAttribute(w12_kernel<NT, NSTAGE, SPLITK, MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
    w12_kernel<NT, NSTAGE, SPLITK, MODE><<<dim3(N / 64, sk), 128, smem, stream>>>(X, ldx, m, W, eb, part, N, K);
}""", """struct W12Args { const int* eo; const int* en; const int* ek; const float* ed; __nv_bfloat16* out; int ldo; int* cnt; };
template <int NT, int NSTAGE, bool SPLITK, int MODE = 0>
void launch_w12(const __nv_bfloat16* X, int ldx, int m, const uint8_t* W, int eb, float* part, int N, int K, int sk, const W12Args& a, cudaStream_t stream)
{
    constexpr int smem = NSTAGE * w12_stage_bytes(NT);
    static bool attr = false;
    if (!attr) { cudaFuncSetAttribute(w12_kernel<NT, NSTAGE, SPLITK, MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
    w12_kernel<NT, NSTAGE, SPLITK, MODE><<<dim3(N / 64, sk), 128, smem, stream>>>(X, ldx, m, W, eb, part, N, K, a.eo, a.en, a.ek, a.ed, a.out, a.ldo, a.cnt);
}""")
a = s.index("void w12(torch::Tensor X, torch::Tensor W, int64_t eb, torch::Tensor esc_off")
b = s.index('"""', a)
s = s[:a] + r"""void w12(torch::Tensor X, torch::Tensor W, int64_t eb, torch::Tensor esc_off, torch::Tensor esc_n, torch::Tensor esc_k, torch::Tensor esc_d,
         torch::Tensor out, torch::Tensor part, torch::Tensor cnt, int64_t m, int64_t n, int64_t k, int64_t nt, int64_t nstage, int64_t sk, int64_t mode)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(n % 64 == 0 && k % 128 == 0 && (k / 128) % sk == 0 && m <= 8 * nt && m <= 32, "w12: shape/config mismatch");
    TORCH_CHECK(X.stride(1) == 1 && (X.stride(0) * 2) % 16 == 0 && (reinterpret_cast<uintptr_t>(X.data_ptr()) & 15) == 0, "w12: x must be 16 B aligned rows");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(W.data_ptr()) & 127) == 0, "w12: weight layout must be 128 B aligned");
    TORCH_CHECK(sk * m * n <= part.numel() && n / 64 <= cnt.numel(), "w12: workspace too small");
    const __nv_bfloat16* Xp = reinterpret_cast<const __nv_bfloat16*>(X.data_ptr());
    const uint8_t* Wp = reinterpret_cast<const uint8_t*>(W.data_ptr());
    float* pp = part.data_ptr<float>();
    const int ldx = (int)X.stride(0);
    W12Args a{esc_off.data_ptr<int>(), esc_n.data_ptr<int>(), esc_k.data_ptr<int>(), esc_d.data_ptr<float>(),
              reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (int)out.stride(0), cnt.data_ptr<int>()};
    const int key = (int)nt * 100 + (int)nstage * 10 + (sk > 1 ? 1 : 0);
    if (mode == 3) {   // loads-only ablation of the GEMM
        if (nt == 1) launch_w12<1, 4, true, 1>(Xp, ldx, (int)m, Wp, (int)eb, pp, (int)n, (int)k, (int)sk, a, stream);
        else if (nt == 2) launch_w12<2, 4, true, 1>(Xp, ldx, (int)m, Wp, (int)eb, pp, (int)n, (int)k, (int)sk, a, stream);
        else launch_w12<4, 4, true, 1>(Xp, ldx, (int)m, Wp, (int)eb, pp, (int)n, (int)k, (int)sk, a, stream);
        return;
    }
#define OKC_W12(T, NS, SK) launch_w12<T, NS, SK>(Xp, ldx, (int)m, Wp, (int)eb, pp, (int)n, (int)k, SK ? (int)sk : 1, a, stream)
    switch (key) {
        case 140: OKC_W12(1, 4, false); break;  case 141: OKC_W12(1, 4, true); break;
        case 180: OKC_W12(1, 8, false); break;  case 181: OKC_W12(1, 8, true); break;
        case 240: OKC_W12(2, 4, false); break;  case 241: OKC_W12(2, 4, true); break;
        case 220: OKC_W12(2, 2, false); break;  case 221: OKC_W12(2, 2, true); break;
        case 440: OKC_W12(4, 4, false); break;  case 441: OKC_W12(4, 4, true); break;
        case 420: OKC_W12(4, 2, false); break;  case 421: OKC_W12(4, 2, true); break;
        default: TORCH_CHECK(false, "unsupported w12 config (nt, nstage, sk)");
    }
#undef OKC_W12
}
""" + s[b:]
rep("""void w12(torch::Tensor X, torch::Tensor W, int64_t eb, torch::Tensor esc_off, torch::Tensor esc_n, torch::Tensor esc_k, torch::Tensor esc_d,
         torch::Tensor out, torch::Tensor part, int64_t m, int64_t n, int64_t k, int64_t nt, int64_t nstage, int64_t sk, int64_t mode);""",
    """void w12(torch::Tensor X, torch::Tensor W, int64_t eb, torch::Tensor esc_off, torch::Tensor esc_n, torch::Tensor esc_k, torch::Tensor esc_d,
         torch::Tensor out, torch::Tensor part, torch::Tensor cnt, int64_t m, int64_t n, int64_t k, int64_t nt, int64_t nstage, int64_t sk, int64_t mode);""")
# drop the standalone reduce kernel (no longer launched)
a = s.index("// sum the KSPLIT partial slices of one 32-row pack tile")
b = s.index("struct W12Args {")
s = s[:a] + s[b:]
p.write_text(s)

q = root / "muse_bf16_proj/submission.py"
t = q.read_text()


def rep2(old, new, count=1):
    global t
    assert t.count(old) == count, (t.count(old), old[:90])
    t = t.replace(old, new)


rep2("""        build().w12(x, w12, pk["eb"], pk["esc_off"], pk["esc_n"], pk["esc_k"], pk["esc_delta"], out, state["partials"],
                    m, n, k, cfg["nt"], cfg["nstage"], cfg["sk"], cfg.get("mode", 0))""",
     """        build().w12(x, w12, pk["eb"], pk["esc_off"], pk["esc_n"], pk["esc_k"], pk["esc_delta"], out, state["partials"], state["counters"],
                    m, n, k, cfg["nt"], cfg["nstage"], cfg["sk"], cfg.get("mode", 0))""")
rep2("""        print("BENCH w12 parts: full | gemm only | reduce only | gemm loads-only (nt/4 stages/sk)")
        for layer, n, m, sk in (("pooler", 4096, 8, 4), ("pooler", 4096, 32, 4), ("unpooler", 8192, 16, 4), ("qkv_gate", 2304, 8, 8), ("o", 8192, 16, 2)):
            if not budget():
                break
            nt = 1 if m <= 8 else 2 if m <= 16 else 4
            parts = [row(f"mode{md}", layer, m, n, dict(w12=True, nt=nt, nstage=4, sk=sk, mode=md)) for md in (0, 1, 2, 3)]
            print(f"  m={m:2d} {layer:9s} n={n} sk={sk}: " + " | ".join(parts))""",
     """        print("BENCH w12 parts: full (fused epilogue) | gemm loads-only (nt/4 stages/sk)")
        for layer, n, m, sk in (("pooler", 4096, 8, 4), ("pooler", 4096, 32, 4), ("unpooler", 8192, 16, 4), ("qkv_gate", 2304, 8, 8), ("o", 8192, 16, 2)):
            if not budget():
                break
            nt = 1 if m <= 8 else 2 if m <= 16 else 4
            parts = [row(f"mode{md}", layer, m, n, dict(w12=True, nt=nt, nstage=4, sk=sk, mode=md)) for md in (0, 3)]
            print(f"  m={m:2d} {layer:9s} n={n} sk={sk}: " + " | ".join(parts))""")
rep2("""W12_SHAPES = {}""", """W12_SHAPES = {("pooler", 4096, 8): (1, 4, 4), ("pooler", 4096, 16): (2, 4, 4), ("pooler", 4096, 32): (4, 4, 4),
              ("qkv_gate", 2304, 8): (1, 4, 8), ("qkv_gate", 2304, 16): (2, 2, 8), ("qkv_gate", 2304, 32): (4, 2, 8)}""")
q.write_text(t)
print("fused")
