"""One-off edit: split-K as a template parameter (SPLITK) so the non-split instantiations have no partial-sum,
counter or fence code; split-K instantiations only for the candidate variants."""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
p = root / "muse_moe_w4a16/cuda_moe.py"
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:100])
    s = s.replace(old, new)


rep("""template <bool GATED, int MW, int NT, int KS, int NSTAGE>
__global__ void __launch_bounds__(128 * KS)
moe_w4a16_wt_kernel(""", """template <bool GATED, int MW, int NT, int KS, int NSTAGE, bool SPLITK>
__global__ void __launch_bounds__(128 * KS)
moe_w4a16_wt_kernel(""")
rep("""    const int KSPLIT = gridDim.z;
    const int NC_ALL = K / KC;
    const int NC = NC_ALL / KSPLIT;                   // chunks of this CTA
    const int c0 = blockIdx.z * NC;""", """    const int KSPLIT = SPLITK ? (int)gridDim.z : 1;
    const int NC_ALL = K / KC;
    const int NC = NC_ALL / KSPLIT;                   // chunks of this CTA
    const int c0 = SPLITK ? (int)blockIdx.z * NC : 0;""")
rep("""    __shared__ int s_last;
    if (KSPLIT > 1) {
        #pragma unroll
        for (int i = 0; i < MW; ++i) {""", """    if constexpr (SPLITK) {
        __shared__ int s_last;
        #pragma unroll
        for (int i = 0; i < MW; ++i) {""")
rep("""                    if (KSPLIT > 1) {
                        for (int z = 0; z < KSPLIT; ++z) {""", """                    if constexpr (SPLITK) {
                        for (int z = 0; z < KSPLIT; ++z) {""")
# launch_wt
rep("""template <bool GATED, int MW, int NT, int KS, int NSTAGE>
void launch_wt(""", """template <bool GATED, int MW, int NT, int KS, int NSTAGE, bool SPLITK>
void launch_wt(""")
rep("""        cudaFuncSetAttribute(moe_w4a16_wt_kernel<GATED, MW, NT, KS, NSTAGE>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);""",
    """        cudaFuncSetAttribute(moe_w4a16_wt_kernel<GATED, MW, NT, KS, NSTAGE, SPLITK>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);""")
rep("""    moe_w4a16_wt_kernel<GATED, MW, NT, KS, NSTAGE><<<grid, 128 * KS, smem, stream>>>(""",
    """    moe_w4a16_wt_kernel<GATED, MW, NT, KS, NSTAGE, SPLITK><<<grid, 128 * KS, smem, stream>>>(""")
# switch
a = s.index("    const int key = (gated ? 100000 : 0) + (int)mw * 10000 + (int)nt * 1000 + (int)ks * 100 + (int)nstage;")
b = s.index("#undef OKC_W")
new_sw = """    const int key = (ksplit > 1 ? 1000000 : 0) + (gated ? 100000 : 0) + (int)mw * 10000 + (int)nt * 1000 + (int)ks * 100 + (int)nstage;
    switch (key) {
        case 111102: OKC_W(true, 1, 1, 1, 2, false); break;   case 111104: OKC_W(true, 1, 1, 1, 4, false); break;
        case 111108: OKC_W(true, 1, 1, 1, 8, false); break;
        case 112102: OKC_W(true, 1, 2, 1, 2, false); break;   case 112104: OKC_W(true, 1, 2, 1, 4, false); break;
        case 114102: OKC_W(true, 1, 4, 1, 2, false); break;   case 114104: OKC_W(true, 1, 4, 1, 4, false); break;
        case 11102: OKC_W(false, 1, 1, 1, 2, false); break;   case 11104: OKC_W(false, 1, 1, 1, 4, false); break;
        case 11108: OKC_W(false, 1, 1, 1, 8, false); break;
        case 21102: OKC_W(false, 2, 1, 1, 2, false); break;   case 21104: OKC_W(false, 2, 1, 1, 4, false); break;
        case 12104: OKC_W(false, 1, 2, 1, 4, false); break;   case 22102: OKC_W(false, 2, 2, 1, 2, false); break;
        case 22104: OKC_W(false, 2, 2, 1, 4, false); break;   case 14104: OKC_W(false, 1, 4, 1, 4, false); break;
        case 24102: OKC_W(false, 2, 4, 1, 2, false); break;   case 24104: OKC_W(false, 2, 4, 1, 4, false); break;
        // split-K instantiations (the candidate small-batch variants)
        case 1111102: OKC_W(true, 1, 1, 1, 2, true); break;   case 1111104: OKC_W(true, 1, 1, 1, 4, true); break;
        case 1111108: OKC_W(true, 1, 1, 1, 8, true); break;   case 1112104: OKC_W(true, 1, 2, 1, 4, true); break;
        case 1011102: OKC_W(false, 1, 1, 1, 2, true); break;  case 1011104: OKC_W(false, 1, 1, 1, 4, true); break;
        case 1011108: OKC_W(false, 1, 1, 1, 8, true); break;  case 1021104: OKC_W(false, 2, 1, 1, 4, true); break;
        case 1021108: OKC_W(false, 2, 1, 1, 8, true); break;  case 1022104: OKC_W(false, 2, 2, 1, 4, true); break;
        default: TORCH_CHECK(false, "unsupported wt variant (mw, nt, ks, nstage, ksplit)");
    }
"""
s = s[:a] + new_sw + s[b:]
rep("#define OKC_W(G, M, T, S, NS) launch_wt<G, M, T, S, NS>(Xp,", "#define OKC_W(G, M, T, S, NS, SK) launch_wt<G, M, T, S, NS, SK>(Xp,")
p.write_text(s)

g = root / "tools/gen_moe_kernel_only.py"
u = g.read_text()
u = u.replace('''for g, m, t, s, n in re.findall(r"OKC_W\\((true|false), (\\d), (\\d), (\\d), (\\d)\\)", cu):
    inst.append(f"template void launch_wt<{g}, {m}, {t}, {s}, {n}>{wsig};")''',
              '''for g, m, t, s, n, sk in re.findall(r"OKC_W\\((true|false), (\\d), (\\d), (\\d), (\\d), (true|false)\\)", cu):
    inst.append(f"template void launch_wt<{g}, {m}, {t}, {s}, {n}, {sk}>{wsig};")''')
g.write_text(u)

# bench/selftest lists: split-K only where instantiated
q = root / "muse_moe_w4a16/submission.py"
t = q.read_text()
old = "    small = [(1, 1, 1, 2, 1), (1, 1, 1, 4, 1), (1, 1, 1, 8, 1), (1, 1, 1, 8, 2), (1, 1, 1, 8, 4), (1, 1, 1, 4, 4), (1, 2, 1, 8, 2), (1, 2, 1, 4, 2), (1, 2, 1, 2, 1), (1, 1, 1, 2, 2)]\n    big = [(2, 2, 1, 4, 1), (2, 2, 1, 2, 1), (4, 2, 1, 4, 1), (4, 2, 1, 2, 1), (2, 1, 1, 2, 1), (4, 1, 1, 4, 1), (2, 2, 1, 4, 2)]"
new = "    small = [(1, 1, 1, 2, 1), (1, 1, 1, 4, 1), (1, 1, 1, 8, 1), (1, 1, 1, 8, 2), (1, 1, 1, 8, 4), (1, 1, 1, 4, 4), (1, 2, 1, 8, 2), (1, 2, 1, 4, 2), (1, 2, 1, 2, 1), (1, 1, 1, 2, 2)]\n    big = [(2, 2, 1, 4, 1), (2, 2, 1, 2, 1), (4, 2, 1, 4, 1), (4, 2, 1, 2, 1), (2, 1, 1, 2, 1), (4, 1, 1, 4, 1)]"
assert t.count(old) == 1
t = t.replace(old, new)
old = """WT_TEST_VARIANTS = ((1, 1, 1, 4, 1), (1, 1, 1, 2, 1), (1, 1, 1, 8, 1), (1, 1, 1, 8, 2), (1, 1, 1, 8, 4), (1, 1, 1, 4, 4), (1, 1, 1, 2, 2),
                    (1, 2, 1, 8, 2), (1, 2, 1, 4, 2), (1, 2, 1, 2, 1), (2, 2, 1, 4, 1), (2, 2, 1, 2, 1), (2, 2, 1, 4, 2),
                    (4, 2, 1, 4, 1), (4, 2, 1, 2, 1), (2, 1, 1, 2, 1), (4, 1, 1, 4, 1))"""
new = """WT_TEST_VARIANTS = ((1, 1, 1, 4, 1), (1, 1, 1, 2, 1), (1, 1, 1, 8, 1), (1, 1, 1, 8, 2), (1, 1, 1, 8, 4), (1, 1, 1, 4, 4), (1, 1, 1, 2, 2),
                    (1, 2, 1, 8, 2), (1, 2, 1, 4, 2), (1, 2, 1, 2, 1), (2, 2, 1, 4, 1), (2, 2, 1, 2, 1),
                    (4, 2, 1, 4, 1), (4, 2, 1, 2, 1), (2, 1, 1, 2, 1), (4, 1, 1, 4, 1))"""
assert t.count(old) == 1
t = t.replace(old, new)
q.write_text(t)
print("edited")
