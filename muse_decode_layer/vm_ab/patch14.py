"""Round-2 items 2 and 3 as runtime-switchable variants (one build, A/B by ab_vm overrides):
  proj fork: the PDL word gains flags - 0x100 the sk2 kernels trigger launch_dependents at entry (not after the main
  loop); 0x200 the m = 1 cgemv is a PDL launch (weights before griddepcontrol.wait, x after it through L2, trigger
  at entry); layer_glue: SK2_PDL_FLAGS OR-ed into the word, the boundary kernel's thread count templated (1024 under
  glue mode bit 8), ab_vm pdlflags=N / gmode=N, check_vm --override.  Run once from the worktree root."""
import pathlib

root = pathlib.Path(__file__).resolve().parents[2]


def patch(path, edits):
    p = root / path
    s = p.read_text()
    for old, new, cnt in edits:
        n = s.count(old)
        assert n == cnt, (path, n, cnt, old[:90])
        s = s.replace(old, new)
    p.write_text(s)


patch("muse_decode_layer/proj_fork.py", [
    ("""  const int P = (CL == 1 && pdl > 0) ? min(pdl, min(NSTAGE, nst)) : 1;
""", """  const int pdlP = pdl & 0xFF;   // pre-wait weight stages; bit 8: launch_dependents at entry (dependents prefetch)
  const int P = (CL == 1 && pdlP > 0) ? min(pdlP, min(NSTAGE, nst)) : 1;
  if (pdl & 0x100) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
""", 1),
    ('if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");',
     'if (pdlP) asm volatile("griddepcontrol.wait;" ::: "memory");', 2),
    ("""  if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");   // the next kernel may start its prologue""",
     """  if (pdlP && !(pdl & 0x100)) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");   // the next kernel may start its prologue""", 1),
    ("""    const int pdl = g_sk2_pdl;
    if (CL > 1 || pdl > 0) {""", """    const int pdl = g_sk2_pdl;
    const int pdlP = pdl & 0xFF;
    if (CL > 1 || pdlP > 0) {""", 1),
    ("""      if (pdl > 0) {   // programmatic dependent launch: may start while the previous kernel in the stream finishes""",
     """      if (pdlP > 0) {   // programmatic dependent launch: may start while the previous kernel in the stream finishes""", 1),
    ("""    __nv_bfloat16* __restrict__ out, uint32_t base4) {
  constexpr int PER = K / (16 * NTH);""", """    __nv_bfloat16* __restrict__ out, uint32_t base4, int pdl) {
  constexpr int PER = K / (16 * NTH);""", 1),
    ("""  const int r0 = blockIdx.x * ROWS;
  __shared__ float red[ROWS][NWARP + 1];
  if (warp == NWARP) {   // escape warp: lane i takes entries 1 + i, 33 + i, ... (entry 0 = count); x[k] * delta per row
    const int2* tb = etab + (size_t)blockIdx.x * pmax;
    int2 mine = make_int2(0, 0);
    if (lane < pmax) mine = __ldg(&tb[lane]);
    const int cnt = __shfl_sync(0xffffffffu, mine.x, 0);
    float ec[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) ec[r] = 0.f;
    for (int i0 = 0; i0 <= cnt; i0 += 32) {
      const int i = i0 + lane;
      int2 en = mine;
      if (i0 > 0) en = (i <= cnt) ? __ldg(&tb[i]) : make_int2(0, 0);
      if (i >= 1 && i <= cnt) {
        const float c = __bfloat162float(x[en.x & 0xFFFF]) * esc_delta((uint32_t)en.y, base4 & 0xFFu);""",
     """  const int r0 = blockIdx.x * ROWS;
  __shared__ float red[ROWS][NWARP + 1];
  // programmatic dependent launch (pdl): the weights and the escape table never depend on the previous kernel and
  // load first; x (its output) only after griddepcontrol.wait, through L2 (never ld.global.nc: ptxas may hoist those
  // above the wait); dependents may start their own prologue at once
  if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (warp == NWARP) {   // escape warp: lane i takes entries 1 + i, 33 + i, ... (entry 0 = count); x[k] * delta per row
    const int2* tb = etab + (size_t)blockIdx.x * pmax;
    int2 mine = make_int2(0, 0);
    if (lane < pmax) mine = __ldg(&tb[lane]);
    const int cnt = __shfl_sync(0xffffffffu, mine.x, 0);
    float ec[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) ec[r] = 0.f;
    if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
    for (int i0 = 0; i0 <= cnt; i0 += 32) {
      const int i = i0 + lane;
      int2 en = mine;
      if (i0 > 0) en = (i <= cnt) ? __ldg(&tb[i]) : make_int2(0, 0);
      if (i >= 1 && i <= cnt) {
        const unsigned short xb = __ldcg(reinterpret_cast<const unsigned short*>(x) + (en.x & 0xFFFF));
        const float c = __uint_as_float((uint32_t)xb << 16) * esc_delta((uint32_t)en.y, base4 & 0xFFu);""", 1),
    ("""#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const __nv_bfloat16* xp = x + j * 16 * NTH + 16 * tid;
    xv[j][0] = __ldg(reinterpret_cast<const uint4*>(xp));
    xv[j][1] = __ldg(reinterpret_cast<const uint4*>(xp + 8));
  }""", """  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const __nv_bfloat16* xp = x + j * 16 * NTH + 16 * tid;
    xv[j][0] = __ldcg(reinterpret_cast<const uint4*>(xp));
    xv[j][1] = __ldcg(reinterpret_cast<const uint4*>(xp + 8));
  }""", 1),
    ("""  const uint32_t b4 = (uint32_t)base4;
  const int pm = (int)pmax;
  @@CG_DISPATCH@@""", """  const uint32_t b4 = (uint32_t)base4;
  const int pm = (int)pmax;
  const int pdl = (g_sk2_pdl & 0x200) ? 1 : 0;   // PDL launch of the m = 1 GEMV (sk2_set_pdl bit 9)
  @@CG_DISPATCH@@""", 1),
    ("""            lines.append(f"if (K == {k} && rows == {rows} && nth == {nth}) {{ cgemv_kernel<{rows}, {nth}, {k}><<<N / {rows}, {nth} + 32, 0, st>>>"
                         f"(a0, a1, xp, tp, pm, op, b4); C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }}")""",
     """            lines.append(f"if (K == {k} && rows == {rows} && nth == {nth}) {{ auto kern = cgemv_kernel<{rows}, {nth}, {k}>; "
                         f"if (pdl) {{ cudaLaunchConfig_t lc = {{}}; lc.gridDim = dim3(N / {rows}); lc.blockDim = dim3({nth} + 32); "
                         f"lc.dynamicSmemBytes = 0; lc.stream = st; cudaLaunchAttribute at_[1]; "
                         f"at_[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at_[0].val.programmaticStreamSerializationAllowed = 1; "
                         f"lc.attrs = at_; lc.numAttrs = 1; C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, a0, a1, xp, tp, pm, op, b4, 1)); }} "
                         f"else {{ kern<<<N / {rows}, {nth} + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4, 0); C10_CUDA_KERNEL_LAUNCH_CHECK(); }} return; }}")""", 1),
    ("""        cg = state["cg"][(layer, rows)]
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        _get_ext().cgemv(cg["L"], cg["C"], x, cg["tab"], cg["pmax"], out, cg["base4"], rows, nth)""",
     """        cg = state["cg"][(layer, rows)]
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        ext = _get_ext()
        ext.sk2_set_pdl(SK2_PDL)
        ext.cgemv(cg["L"], cg["C"], x, cg["tab"], cg["pmax"], out, cg["base4"], rows, nth)""", 1),
])

patch("muse_decode_layer/layer_glue.py", [
    ("""def _bfp_pdl(T):
    \"\"\"Their launchers read SK2_PDL (bfp_SK2_PDL once inlined) at every launch: same T rule as this file's glue.\"\"\"
    global bfp_SK2_PDL
    bfp_SK2_PDL = _PDL_N_BY_T.get(T, BFP_PDL_N) if _pdl(T) else 0""",
     """SK2_PDL_FLAGS = 0   # OR-ed into the proj fork's PDL word: 0x100 = the sk2 kernels trigger launch_dependents at entry
                    # (not after their main loop), 0x200 = the m = 1 cgemv is a PDL launch (weights before its wait,
                    # trigger at entry). Measured on the H200 (vm_ab/ab14).


def _bfp_pdl(T):
    \"\"\"Their launchers read SK2_PDL (bfp_SK2_PDL once inlined) at every launch: same T rule as this file's glue.\"\"\"
    global bfp_SK2_PDL
    bfp_SK2_PDL = (_PDL_N_BY_T.get(T, BFP_PDL_N) | SK2_PDL_FLAGS) if _pdl(T) else 0""", 1),
    ("""template <int KIND, typename AT>
__global__ void __launch_bounds__(BNT) bnd_kernel(const AT* a, int lda, int nu, const float* __restrict__ st,""",
     """template <int KIND, typename AT, int NT>
__global__ void __launch_bounds__(NT) bnd_kernel(const AT* a, int lda, int nu, const float* __restrict__ st,""", 1),
    ("""  const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  __shared__ float red[BNT / 32][4];
  __shared__ float scl[2];
  if (pdl && !(mode & 1)) pdl_trigger();
  if (pdl && (mode & 2)) pdl_wait();
  float sv[BCH][4], al[BCH][4], be[BCH][4], gw[BCH][4], gn[BCH][4];
#pragma unroll
  for (int k = 0; k < BCH; ++k) {
    const int col = 4 * (tid + k * BNT);""",
     """  constexpr int BCH = HID / (4 * NT);   // 16-B chunks per thread (4 at 512 threads, 2 at 1024)
  const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  __shared__ float red[NT / 32][4];
  __shared__ float scl[2];
  if (pdl && !(mode & 1)) pdl_trigger();
  if (pdl && (mode & 2)) pdl_wait();
  float sv[BCH][4], al[BCH][4], be[BCH][4], gw[BCH][4], gn[BCH][4];
#pragma unroll
  for (int k = 0; k < BCH; ++k) {
    const int col = 4 * (tid + k * NT);""", 1),
    ("""  float x[BCH][4];
#pragma unroll
  for (int k = 0; k < BCH; ++k) {
    const int col = 4 * (tid + k * BNT);""", """  float x[BCH][4];
#pragma unroll
  for (int k = 0; k < BCH; ++k) {
    const int col = 4 * (tid + k * NT);""", 1),
    ("""    if (lane < BNT / 32) p = make_float4(red[lane][0], red[lane][1], red[lane][2], red[lane][3]);
#pragma unroll
    for (int o = (BNT / 32) / 2; o > 0; o >>= 1) {""", """    if (lane < NT / 32) p = make_float4(red[lane][0], red[lane][1], red[lane][2], red[lane][3]);
#pragma unroll
    for (int o = (NT / 32) / 2; o > 0; o >>= 1) {""", 1),
    ("""#pragma unroll
  for (int k = 0; k < BCH; ++k) {
    const int col = 4 * (tid + k * BNT);
    float s[4], h[4];""", """#pragma unroll
  for (int k = 0; k < BCH; ++k) {
    const int col = 4 * (tid + k * NT);
    float s[4], h[4];""", 1),
    ("""constexpr int BNT = 512;            // boundary: one CTA per token, 512 threads x 16 columns (4 coalesced chunks of 4)
constexpr int BCH = HID / (4 * BNT);   // = 4 chunks per thread
""", """constexpr int BNT = 512;            // boundary: one CTA per token, 512 threads x 16 columns (4 coalesced chunks of 4);
                                    // 1024 threads x 8 columns under mode bit 8 (A/B)
""", 1),
    ("""template <int KIND, typename AT>
static void launch_bnd(const torch::Tensor& a, int lda, int nu, const torch::Tensor& st, const torch::Tensor& alpha,
                       const torch::Tensor& beta, const torch::Tensor& g1, const torch::Tensor& g2, torch::Tensor& sout,
                       torch::Tensor& hout, int T, int pdl, int mode) {
  cudaLaunchConfig_t lc = {};
  cudaLaunchAttribute at_[1];
  lc.gridDim = dim3(T);
  lc.blockDim = dim3(BNT);
  set_pdl(lc, at_, pdl);
  C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, bnd_kernel<KIND, AT>, reinterpret_cast<const AT*>(a.data_ptr()), lda, nu,""",
     """template <int KIND, typename AT, int NT>
static void launch_bnd_nt(const torch::Tensor& a, int lda, int nu, const torch::Tensor& st, const torch::Tensor& alpha,
                          const torch::Tensor& beta, const torch::Tensor& g1, const torch::Tensor& g2, torch::Tensor& sout,
                          torch::Tensor& hout, int T, int pdl, int mode) {
  cudaLaunchConfig_t lc = {};
  cudaLaunchAttribute at_[1];
  lc.gridDim = dim3(T);
  lc.blockDim = dim3(NT);
  set_pdl(lc, at_, pdl);
  C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, bnd_kernel<KIND, AT, NT>, reinterpret_cast<const AT*>(a.data_ptr()), lda, nu,""", 1),
    ("""      sout.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(hout.data_ptr()), pdl, mode));
}

template <bool SPLIT>
static void launch_route(""", """      sout.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(hout.data_ptr()), pdl, mode));
}
template <int KIND, typename AT>
static void launch_bnd(const torch::Tensor& a, int lda, int nu, const torch::Tensor& st, const torch::Tensor& alpha,
                       const torch::Tensor& beta, const torch::Tensor& g1, const torch::Tensor& g2, torch::Tensor& sout,
                       torch::Tensor& hout, int T, int pdl, int mode) {
  if (mode & 8) launch_bnd_nt<KIND, AT, 1024>(a, lda, nu, st, alpha, beta, g1, g2, sout, hout, T, pdl, mode);
  else launch_bnd_nt<KIND, AT, BNT>(a, lda, nu, st, alpha, beta, g1, g2, sout, hout, T, pdl, mode);
}

template <bool SPLIT>
static void launch_route(""", 1),
    ('name=f"okc_glue_v6_{arch}"', 'name=f"okc_glue_v7_{arch}"', 1),
])

patch("muse_decode_layer/ab_vm.py", [
    ("""        elif kind == "gmode":    # the CUDA glue kernels' mode bits (1 late trigger, 2 early wait, 4 exact rank loop)
            saved.append(("gmode", None, st.get("glue_mode")))
            st["glue_mode"] = int(val)""",
     """        elif kind == "gmode":    # the CUDA glue kernels' mode bits (1 late trigger, 2 early wait, 4 exact rank loop, 8 1024 threads)
            saved.append(("gmode", None, st.get("glue_mode")))
            st["glue_mode"] = int(val)
        elif kind == "pdlflags":   # the proj fork's PDL word flags: 0x100 sk2 trigger at entry, 0x200 cgemv as a PDL launch
            saved.append(("pdlflags", None, S.SK2_PDL_FLAGS))
            S.SK2_PDL_FLAGS = int(val)""", 1),
    ("""        elif kind == "gmode":
            if old is None:
                st.pop("glue_mode", None)
            else:
                st["glue_mode"] = old
        else:""", """        elif kind == "gmode":
            if old is None:
                st.pop("glue_mode", None)
            else:
                st["glue_mode"] = old
        elif kind == "pdlflags":
            S.SK2_PDL_FLAGS = old
        else:""", 1),
])

patch("muse_decode_layer/check_vm.py", [
    ("""    ap.add_argument("--no-graph", action="store_true")
    args = ap.parse_args()""", """    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--override", action="append", default=[], help="ab_vm override spec applied after prepare")
    args = ap.parse_args()""", 1),
    ("""        st = S.prepare(weights)
        torch.cuda.synchronize()
        print(f"[prepare] seed {seed}: {time.perf_counter() - t0:.1f} s", flush=True)""",
     """        st = S.prepare(weights)
        torch.cuda.synchronize()
        print(f"[prepare] seed {seed}: {time.perf_counter() - t0:.1f} s", flush=True)
        for spec in args.override:
            apply(S, st, spec, [])
            print(f"[override] {spec}", flush=True)""", 1),
    ("from ab_vm import SHAPES, parse_shape  # noqa: E402", "from ab_vm import SHAPES, parse_shape, apply  # noqa: E402", 1),
])
print("patched")
