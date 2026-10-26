"""Proj edit: runtime `emode` diagnostics in the w12 epilogue (4 = return at entry, 5 = partial stores + ticket only,
6 = skip the escapes; 0 = full) and bench rows that bracket the epilogue's cost against gemm-only."""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
p = root / "muse_bf16_proj/cuda_proj.py"
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


rep("""        const __nv_bfloat16* __restrict__ X, int ldx, __nv_bfloat16* __restrict__ out, int ldo)
{
    // Latency-aware: every loop below issues all of its global loads before using any (the previous version chained
    // one L2 round trip per loop iteration and cost 12-26 us).
    __shared__ __align__(16) float red[64][33];
    __shared__ int s_last;""", """        const __nv_bfloat16* __restrict__ X, int ldx, __nv_bfloat16* __restrict__ out, int ldo, int emode)
{
    // Latency-aware: every loop below issues all of its global loads before using any. emode (bench diagnostics):
    // 4 = return at entry, 5 = partial stores + ticket only, 6 = skip the escapes; 0 = full.
    __shared__ __align__(16) float red[64][33];
    __shared__ int s_last;
    if (emode == 4) return;""")
rep("""        __syncthreads();
        if (!s_last) return;
        // slice sum, 16 tokens per pass""", """        __syncthreads();
        if (!s_last || emode == 5) return;
        // slice sum, 16 tokens per pass""")
rep("""    const int e0 = esc_off[2 * blockIdx.x], e1 = esc_off[2 * blockIdx.x + 2];       // the two lists are contiguous
    for (int base = e0; base < e1; base += 128) {""", """    const int e0 = esc_off[2 * blockIdx.x], e1 = emode == 6 ? esc_off[2 * blockIdx.x] : esc_off[2 * blockIdx.x + 2];   // contiguous lists
    for (int base = e0; base < e1; base += 128) {""")
rep("""typedef void (*w12_epi_fn)(const float*, int, int, int, int, int, int, float*, int*, const int*, const int*, const int*, const float*,
                           const __nv_bfloat16*, int, __nv_bfloat16*, int);""",
    """typedef void (*w12_epi_fn)(const float*, int, int, int, int, int, int, float*, int*, const int*, const int*, const int*, const float*,
                           const __nv_bfloat16*, int, __nv_bfloat16*, int, int);""")
rep("""    fp(accv, m, N, tid, warp, g, t, part, cnt, esc_off, esc_n, esc_k, esc_d, X, ldx, out, ldo);
}""", """    fp(accv, m, N, tid, warp, g, t, part, cnt, esc_off, esc_n, esc_k, esc_d, X, ldx, out, ldo, emode);
}""")
rep("""           __nv_bfloat16* __restrict__ out, int ldo, int* __restrict__ cnt)
{
    constexpr int STAGE_BYTES = w12_stage_bytes(NT);""", """           __nv_bfloat16* __restrict__ out, int ldo, int* __restrict__ cnt, int emode)
{
    constexpr int STAGE_BYTES = w12_stage_bytes(NT);""")
rep("""struct W12Args { const int* eo; const int* en; const int* ek; const float* ed; __nv_bfloat16* out; int ldo; int* cnt; };""",
    """struct W12Args { const int* eo; const int* en; const int* ek; const float* ed; __nv_bfloat16* out; int ldo; int* cnt; int emode; };""")
rep("""    w12_kernel<NT, NSTAGE, SPLITK, MODE><<<dim3(N / 64, sk), 128, smem, stream>>>(X, ldx, m, W, eb, part, N, K, a.eo, a.en, a.ek, a.ed, a.out, a.ldo, a.cnt);""",
    """    w12_kernel<NT, NSTAGE, SPLITK, MODE><<<dim3(N / 64, sk), 128, smem, stream>>>(X, ldx, m, W, eb, part, N, K, a.eo, a.en, a.ek, a.ed, a.out, a.ldo, a.cnt, a.emode);""")
rep("""    W12Args a{esc_off.data_ptr<int>(), esc_n.data_ptr<int>(), esc_k.data_ptr<int>(), esc_d.data_ptr<float>(),
              reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (int)out.stride(0), cnt.data_ptr<int>()};""",
    """    W12Args a{esc_off.data_ptr<int>(), esc_n.data_ptr<int>(), esc_k.data_ptr<int>(), esc_d.data_ptr<float>(),
              reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (int)out.stride(0), cnt.data_ptr<int>(), (int)(mode >= 4 ? mode : 0)};""")
p.write_text(s)

q = root / "muse_bf16_proj/submission.py"
t = q.read_text()
old = '''                parts.append(row("gemm-only", layer, m, n, dict(w12=True, nt=nt, nstage=4, sk=sks[0], mode=2)))
                parts.append(row("loads-only", layer, m, n, dict(w12=True, nt=nt, nstage=4, sk=sks[0], mode=3)))
                print(f"  m={m:2d} {layer:9s} n={n}: " + " | ".join(parts))'''
new = '''                parts.append(row("gemm-only", layer, m, n, dict(w12=True, nt=nt, nstage=4, sk=sks[0], mode=2)))
                parts.append(row("loads-only", layer, m, n, dict(w12=True, nt=nt, nstage=4, sk=sks[0], mode=3)))
                print(f"  m={m:2d} {layer:9s} n={n}: " + " | ".join(parts))
        print("BENCH w12 epilogue diagnostics (nt/4 stages/sk): e0 full | e4 call+return | e5 stores+ticket only | e6 no escapes | e2 gemm-only")
        for layer, n, m, sk in (("pooler", 4096, 8, 4), ("pooler", 4096, 32, 4), ("qkv_gate", 2304, 8, 8), ("unpooler", 8192, 16, 4)):
            if not budget():
                break
            nt = 1 if m <= 8 else 2 if m <= 16 else 4
            parts = [row(f"e{md}", layer, m, n, dict(w12=True, nt=nt, nstage=4, sk=sk, mode=md)) for md in (0, 4, 5, 6, 2)]
            print(f"  m={m:2d} {layer:9s} n={n} sk={sk}: " + " | ".join(parts))'''
assert t.count(old) == 1
q.write_text(t.replace(old, new))
print("diag edited")
