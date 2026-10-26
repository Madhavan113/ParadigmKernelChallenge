"""One-off edit: programmatic dependent launch across GEMM1 -> GEMM2 -> finalize in cuda_moe.py / submission.py.
GEMM1 waits up front (routing lists), GEMM2 issues its first weight stages before waiting for GEMM1's inter, the
finalize reads its routing fields then waits for y; both GEMMs trigger their dependents after the main loop."""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
p = root / "muse_moe_w4a16/cuda_moe.py"
s = p.read_text()


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:90])
    s = s.replace(old, new)


rep("""constexpr int WT_NWARPS = 4;
__host__ __device__ constexpr int wt_stage_bytes(int tpw, int nt) {""",
    """// programmatic dependent launch (no-ops when the kernel was launched without the attribute)
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\\n" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory"); }

constexpr int WT_NWARPS = 4;
__host__ __device__ constexpr int wt_stage_bytes(int tpw, int nt) {""")
rep("""    extern __shared__ __align__(16) uint8_t wt_smem[];

    const int j = blockIdx.y;
    if (j >= *nblk) return;""", """    extern __shared__ __align__(16) uint8_t wt_smem[];

    if (GATED) pdl_wait();                            // the routing lists and x16 come from the previous launch
    const int j = blockIdx.y;
    if (j >= *nblk) return;""")
rep("""    auto issue = [&](int c, int stage) {
        const uint32_t ws = smem_base + stage * STAGE_BYTES;
        const uint32_t ss = ws + W_STAGE;
        const uint32_t xs = ss + S_STAGE;
        #pragma unroll
        for (int i = 0; i < WSEG; ++i)
            cp_async16(ws + (tid + i * NTHR) * 16, wsrc[i] + c * 1024, 16);
        if (tid < TPC * 8) cp_async16(ss + tid * 16, ssrc + c * 128, 16);
        #pragma unroll
        for (int i = 0; i < XSEG; ++i) {
            const int sidx = tid + i * NTHR;
            if (sidx < NT * 128)
                cp_async16(xs + ((sidx >> 4) * A_STRIDE + (sidx & 15) * 8) * 2, xsrc[i] + c * KC, xbytes[i]);
        }
    };
""", """    auto issue_w = [&](int c, int stage) {
        const uint32_t ws = smem_base + stage * STAGE_BYTES;
        const uint32_t ss = ws + W_STAGE;
        #pragma unroll
        for (int i = 0; i < WSEG; ++i)
            cp_async16(ws + (tid + i * NTHR) * 16, wsrc[i] + c * 1024, 16);
        if (tid < TPC * 8) cp_async16(ss + tid * 16, ssrc + c * 128, 16);
    };
    auto issue_x = [&](int c, int stage) {
        const uint32_t xs = smem_base + stage * STAGE_BYTES + W_STAGE + S_STAGE;
        #pragma unroll
        for (int i = 0; i < XSEG; ++i) {
            const int sidx = tid + i * NTHR;
            if (sidx < NT * 128)
                cp_async16(xs + ((sidx >> 4) * A_STRIDE + (sidx & 15) * 8) * 2, xsrc[i] + c * KC, xbytes[i]);
        }
    };
    auto issue = [&](int c, int stage) { issue_w(c, stage); issue_x(c, stage); };
""")
rep("""    #pragma unroll
    for (int st = 0; st < NSTAGE - 1; ++st) {
        if (st < NC) issue(st, st);
        cp_async_commit();
    }

    int stage = 0, stage_n = NSTAGE - 1;""", """    if (GATED) {
        #pragma unroll
        for (int st = 0; st < NSTAGE - 1; ++st) {
            if (st < NC) issue(st, st);
            cp_async_commit();
        }
    } else {
        // GEMM2: the weights only need the routing lists (complete long ago), so their first stages stream before
        // waiting for GEMM1's inter; group st then holds the token rows of stage st (group 0 also all the weights)
        #pragma unroll
        for (int st = 0; st < NSTAGE - 1; ++st)
            if (st < NC) issue_w(st, st);
        pdl_wait();
        #pragma unroll
        for (int st = 0; st < NSTAGE - 1; ++st) {
            if (st < NC) issue_x(st, st);
            cp_async_commit();
        }
    }

    int stage = 0, stage_n = NSTAGE - 1;""")
rep("""    cp_async_wait<0>();

    if (KS == 2) {""", """    pdl_trigger();                                    // main loop done: the next kernel may start launching
    cp_async_wait<0>();

    if (KS == 2) {""")
rep("""               int K, int ntiles, int max_blocks, float gs_mul, cudaStream_t stream)""",
    """               int K, int ntiles, int max_blocks, float gs_mul, bool pdl, cudaStream_t stream)""")
rep("""    const int n_x = (GATED ? ntiles / 2 : ntiles) / (WT_NWARPS * MW);
    dim3 grid(n_x, max_blocks, ksplit);
    moe_w4a16_wt_kernel<GATED, MW, NT, KS, NSTAGE, SPLITK, ABL><<<grid, 128 * KS, smem, stream>>>(
        X, ldx, st, be, br, bend, nb, Wr, Sr, gs, o16, obf, ldo, part, ldp, part_rows, counters, K, ntiles, gs_mul);
}""", """    const int n_x = (GATED ? ntiles / 2 : ntiles) / (WT_NWARPS * MW);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(n_x, max_blocks, ksplit);
    cfg.blockDim = dim3(128 * KS);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = pdl ? 1 : 0;
    const cudaError_t err = cudaLaunchKernelEx(&cfg, moe_w4a16_wt_kernel<GATED, MW, NT, KS, NSTAGE, SPLITK, ABL>,
        X, ldx, st, be, br, bend, nb, Wr, Sr, gs, o16, obf, ldo, part, ldp, part_rows, counters, K, ntiles, gs_mul);
    TORCH_CHECK(err == cudaSuccess, "wt launch failed: ", cudaGetErrorString(err));
}""")
rep("""        pp, ldp, prow, cnt, (int)ksplit, (int)K, (int)ntiles, (int)max_blocks, (float)gs_mul, stream)""",
    """        pp, ldp, prow, cnt, (int)ksplit, (int)K, (int)ntiles, (int)max_blocks, (float)gs_mul, pdl, stream)""")
rep("""                  int64_t max_blocks, bool gated, int64_t mw, int64_t nt, int64_t ks, int64_t nstage, double gs_mul, int64_t abl)""",
    """                  int64_t max_blocks, bool gated, int64_t mw, int64_t nt, int64_t ks, int64_t nstage, double gs_mul, int64_t abl,
                  bool pdl)""", 2)
# finalize: wait after the routing fields, before y
rep("""    const int r = row_of_pair[tok * 8 + warp];
    float ss = 0.f;
    if (r >= 0) {""", """    const int r = row_of_pair[tok * 8 + warp];
    const float twv = tw[tok * 8 + warp];
    pdl_wait();                                          // y comes from the previous launch
    float ss = 0.f;
    if (r >= 0) {""")
rep("""        sc_sh[warp] = r >= 0 ? rsqrtf(ss / (float)H + eps) * tw[tok * 8 + warp] : 0.f;""",
    """        sc_sh[warp] = r >= 0 ? rsqrtf(ss / (float)H + eps) * twv : 0.f;""")
rep("""                  int64_t M, double eps)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(out.size(1) == 4096, "finalize: H must be 4096");
    moe_finalize_kernel<4096><<<(unsigned)M, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(y.data_ptr()), (int)y.stride(0), row_of_pair.data_ptr<int>(),
        tw.data_ptr<float>(), nw.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (float)eps);
}""", """                  int64_t M, double eps, bool pdl)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(out.size(1) == 4096, "finalize: H must be 4096");
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)M);
    cfg.blockDim = dim3(256);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = pdl ? 1 : 0;
    const cudaError_t err = cudaLaunchKernelEx(&cfg, moe_finalize_kernel<4096>,
        reinterpret_cast<const __nv_bfloat16*>(y.data_ptr()), (int)y.stride(0), row_of_pair.data_ptr<int>(),
        tw.data_ptr<float>(), nw.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (float)eps);
    TORCH_CHECK(err == cudaSuccess, "finalize launch failed: ", cudaGetErrorString(err));
}""")
rep("""                  int64_t M, double eps);""", """                  int64_t M, double eps, bool pdl);""")
p.write_text(s)
print("cuda pdl written")

q = root / "muse_moe_w4a16/submission.py"
t = q.read_text()


def rep2(old, new, n=1):
    global t
    assert t.count(old) == n, (t.count(old), old[:90])
    t = t.replace(old, new)


rep2("_WT_ABL = 0            # bench ablation of the wt GEMMs (1 = no mma, 2 = no dequant, 3 = loads only)\n",
     "_WT_ABL = 0            # bench ablation of the wt GEMMs (1 = no mma, 2 = no dequant, 3 = loads only)\n"
     "PDL = True             # programmatic dependent launch GEMM1 -> GEMM2 -> finalize\n_PDL_OVERRIDE = None   # bench: False forces plain launches\n")
rep2("""    nt, mw2, ks, nstage, ksplit = _wt_variant(M, state)
    BM = 8 * nt""", """    nt, mw2, ks, nstage, ksplit = _wt_variant(M, state)
    pdl = PDL if _PDL_OVERRIDE is None else _PDL_OVERRIDE
    BM = 8 * nt""")
rep2("""                     nstage, state["gs_mul"], _WT_ABL)""", """                     nstage, state["gs_mul"], _WT_ABL, pdl)""")
rep2("""                     state["gs_mul"], _WT_ABL)
    ext.moe_finalize(y, state["row_of_pair"], tw, state["norm_w"], out, M, POST_NORM_EPS)""",
     """                     state["gs_mul"], _WT_ABL, pdl)
    ext.moe_finalize(y, state["row_of_pair"], tw, state["norm_w"], out, M, POST_NORM_EPS, pdl)""")
rep2(', nt, ks, ns, state["gs_mul"], _WT_ABL)', ', nt, ks, ns, state["gs_mul"], _WT_ABL, False)', 4)
rep2("""            "finalize": lambda: ext.moe_finalize(state["y"], state["row_of_pair"], tw, state["norm_w"], out, bs, POST_NORM_EPS),""",
     """            "finalize": lambda: ext.moe_finalize(state["y"], state["row_of_pair"], tw, state["norm_w"], out, bs, POST_NORM_EPS, False),""")
rep2("""    global _WT_ABL
    for bs, v in ((32, (1, 1, 1, 4, 1)), (128, (2, 2, 1, 4, 1)), (256, (2, 2, 1, 4, 1))):""",
     """    global _WT_ABL, _PDL_OVERRIDE
    for flag in (False, True):
        _PDL_OVERRIDE = flag
        try:
            r = [_timeit_cold(lambda bs=bs: _moe_forward_wt(*data[bs], 0, 4096, state), dev) for bs in shapes]
            print(f"  wt pdl={int(flag)}      : " + "  ".join(f"bs{bs} {t:7.1f}" for bs, t in zip(shapes, r)))
        except Exception as e:
            print(f"  wt pdl={int(flag)} failed: {str(e)[:80]}")
        finally:
            _PDL_OVERRIDE = None
    for bs, v in ((32, (1, 1, 1, 4, 1)), (128, (2, 2, 1, 4, 1)), (256, (2, 2, 1, 4, 1))):""")
q.write_text(t)
print("submission pdl written")
