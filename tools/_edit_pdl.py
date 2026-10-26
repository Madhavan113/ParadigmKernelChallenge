"""Round 2 edit: chunk-major weight layout (an expert's chunk c = all tiles' 1 KB blocks contiguous), programmatic
dependent launch (griddepcontrol + cudaLaunchKernelEx attribute, GEMM2 prefetches its weights before the wait),
TMA/split-K instantiations dropped from the switch (code kept)."""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
p = root / "muse_moe_w4a16/cuda_moe.py"
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:100])
    s = s.replace(old, new)


# ---- helpers
rep("""constexpr int WT_NWARPS = 4;
__host__ __device__ constexpr int wt_stage_bytes(int tpw, int nt) {""",
"""// programmatic dependent launch (no-ops when the kernel was launched without the attribute)
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\\n" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory"); }

constexpr int WT_NWARPS = 4;
__host__ __device__ constexpr int wt_stage_bytes(int tpw, int nt) {""")

# ---- kernel: layout + PDL
rep("""    // Weights: Wr[e][tile/4][chunk][tile%4][1 KB], scales Sr[e][tile/4][chunk][tile%4][128 B]: a 4-tile group's
    // chunk is 4 KB contiguous.""",
"""    // Weights: Wr[e][chunk][tile/4][tile%4][1 KB], scales Sr[e][chunk][tile/4][tile%4][128 B]: an expert's chunk is
    // one contiguous 512 KB (+64 KB) region that all of its CTAs sweep together.
    // PDL: GATED (GEMM1) waits for the routing lists first; else (GEMM2) the weights only need the lists (already
    // complete) so their prologue is issued before waiting for GEMM1's inter.""")
rep("""    extern __shared__ __align__(128) uint8_t wt_smem[];

    const int j = blockIdx.y;
    if (j >= *nblk) return;""", """    extern __shared__ __align__(128) uint8_t wt_smem[];

    if (GATED) pdl_wait();                            // the routing lists come from the previous launch
    const int j = blockIdx.y;
    if (j >= *nblk) return;""")
rep("""    if constexpr (!TMA) cp_async_wait<0>();

    if (KS == 2) {""", """    pdl_trigger();                                    // main loop done: the next kernel may start launching
    if constexpr (!TMA) cp_async_wait<0>();

    if (KS == 2) {""")
rep("""    auto waddr = [&](int tile, int c) { return Wr + ((ebase + (tile >> 2)) * NC_ALL + c) * 4096 + (tile & 3) * 1024; };
    auto saddr = [&](int tile, int c) { return Sr + ((ebase + (tile >> 2)) * NC_ALL + c) * 512 + (tile & 3) * 128; };""",
"""    const size_t WCH = (size_t)NG * 4096, SCH = (size_t)NG * 512;    // bytes per expert chunk
    auto waddr = [&](int tile, int c) { return Wr + ((size_t)e * NC_ALL + c) * WCH + (tile >> 2) * 4096 + (tile & 3) * 1024; };
    auto saddr = [&](int tile, int c) { return Sr + ((size_t)e * NC_ALL + c) * SCH + (tile >> 2) * 512 + (tile & 3) * 128; };""")
# cp.async issue split into weights / x parts
rep("""    auto issue = [&](int c, int stage) {
        const uint32_t ws = smem_base + stage * STAGE_BYTES;
        const uint32_t ss = ws + W_STAGE;
        const uint32_t xs = ss + S_STAGE;
        #pragma unroll
        for (int i = 0; i < WSEG; ++i)
            cp_async16(ws + (tid + i * NTHR) * 16, wsrc[i] + c * 4096, 16);
        if (tid < TPC * 8) cp_async16(ss + tid * 16, ssrc + c * 512, 16);
        #pragma unroll
        for (int i = 0; i < XSEG; ++i) {
            const int sidx = tid + i * NTHR;
            if (sidx < NT * 128)
                cp_async16(xs + ((sidx >> 4) * A_STRIDE + (sidx & 15) * 8) * 2, xsrc[i] + c * KC, xbytes[i]);
        }
    };""",
"""    auto issue_w = [&](int c, int stage) {
        const uint32_t ws = smem_base + stage * STAGE_BYTES;
        const uint32_t ss = ws + W_STAGE;
        #pragma unroll
        for (int i = 0; i < WSEG; ++i)
            cp_async16(ws + (tid + i * NTHR) * 16, wsrc[i] + c * WCH, 16);
        if (tid < TPC * 8) cp_async16(ss + tid * 16, ssrc + c * SCH, 16);
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
    auto issue = [&](int c, int stage) { issue_w(c, stage); issue_x(c, stage); };""")
# the x pointers must be computed after the wait for GEMM2 (inter is GEMM1's output... pointers are fine, data not)
rep("""            wgrp_src = Wr + ((ebase + grp) * NC_ALL + c0) * 4096;
            sgrp_src = Sr + ((ebase + grp) * NC_ALL + c0) * 512;""",
"""            wgrp_src = Wr + ((size_t)e * NC_ALL + c0) * WCH + grp * 4096;
            sgrp_src = Sr + ((size_t)e * NC_ALL + c0) * SCH + grp * 512;""")
rep("""            bulk_g2s(ws + warp * 4096, wgrp_src + (size_t)c * 4096, 4096, bar);
            bulk_g2s(ss + warp * 512, sgrp_src + (size_t)c * 512, 512, bar);""",
"""            bulk_g2s(ws + warp * 4096, wgrp_src + (size_t)c * WCH, 4096, bar);
            bulk_g2s(ss + warp * 512, sgrp_src + (size_t)c * SCH, 512, bar);""")
# prologue: GEMM2 issues weights, waits, then x rows (two groups); first iteration waits for both
rep("""    if constexpr (TMA) {
        #pragma unroll
        for (int st = 0; st < NSTAGE - 1; ++st)
            if (st < NC) issue_tma(st, st);
    } else {
        #pragma unroll
        for (int st = 0; st < NSTAGE - 1; ++st) {
            if (st < NC) issue(st, st);
            cp_async_commit();
        }
    }
""", """    if constexpr (TMA) {
        if (!GATED) pdl_wait();
        #pragma unroll
        for (int st = 0; st < NSTAGE - 1; ++st)
            if (st < NC) issue_tma(st, st);
    } else if (GATED) {
        #pragma unroll
        for (int st = 0; st < NSTAGE - 1; ++st) {
            if (st < NC) issue(st, st);
            cp_async_commit();
        }
    } else {
        // weights first (one group), then wait for GEMM1's inter, then the token rows (one group)
        #pragma unroll
        for (int st = 0; st < NSTAGE - 1; ++st)
            if (st < NC) issue_w(st, st);
        cp_async_commit();
        pdl_wait();
        #pragma unroll
        for (int st = 0; st < NSTAGE - 1; ++st)
            if (st < NC) issue_x(st, st);
        cp_async_commit();
    }
""")
rep("""        } else {
            cp_async_wait<NSTAGE - 2>();
            __syncthreads();
            const int cn = c + NSTAGE - 1;
            if (cn < NC) issue(cn, cn % NSTAGE);
            cp_async_commit();
        }""", """        } else {
            if (!GATED && c == 0) cp_async_wait<0>();     // the two prologue groups
            else cp_async_wait<NSTAGE - 2>();
            __syncthreads();
            const int cn = c + NSTAGE - 1;
            if (cn < NC) issue(cn, cn % NSTAGE);
            cp_async_commit();
        }""")

# ---- launch_wt via cudaLaunchKernelEx (PDL attribute)
rep("""    const int n_x = (GATED ? ntiles / 2 : ntiles) / (WT_NWARPS * MW);
    dim3 grid(n_x, max_blocks, ksplit);
    moe_w4a16_wt_kernel<GATED, MW, NT, KS, NSTAGE, SPLITK, TMA><<<grid, 128 * KS, smem, stream>>>(
        X, ldx, st, be, br, bend, nb, Wr, Sr, gs, o16, obf, ldo, part, ldp, part_rows, K, ntiles, gs_mul);
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
    cudaError_t err = cudaLaunchKernelEx(&cfg, moe_w4a16_wt_kernel<GATED, MW, NT, KS, NSTAGE, SPLITK, TMA>,
        X, ldx, st, be, br, bend, nb, Wr, Sr, gs, o16, obf, ldo, part, ldp, part_rows, K, ntiles, gs_mul);
    TORCH_CHECK(err == cudaSuccess, "wt launch failed: ", cudaGetErrorString(err));
}""")
rep("""               float* part, int ldp, int part_rows, int ksplit,
               int K, int ntiles, int max_blocks, float gs_mul, cudaStream_t stream)
{
    constexpr int TPW = (GATED ? 2 : 1) * MW;
    constexpr int smem = NSTAGE * wt_stage_bytes(TPW, NT) + (TMA ? NSTAGE * 8 : 0);""",
"""               float* part, int ldp, int part_rows, int ksplit,
               int K, int ntiles, int max_blocks, float gs_mul, bool pdl, cudaStream_t stream)
{
    constexpr int TPW = (GATED ? 2 : 1) * MW;
    constexpr int smem = NSTAGE * wt_stage_bytes(TPW, NT) + (TMA ? NSTAGE * 8 : 0);""")
rep("        pp, ldp, prow, (int)ksplit, (int)K, (int)ntiles, (int)max_blocks, (float)gs_mul, stream)",
    "        pp, ldp, prow, (int)ksplit, (int)K, (int)ntiles, (int)max_blocks, (float)gs_mul, pdl, stream)")
rep("""                  int64_t max_blocks, bool gated, int64_t mw, int64_t nt, int64_t ks, int64_t nstage, double gs_mul, bool tma)""",
    """                  int64_t max_blocks, bool gated, int64_t mw, int64_t nt, int64_t ks, int64_t nstage, double gs_mul, bool tma, bool pdl)""", 2)
# switch: only the measured-useful plain variants
a = s.index("    switch (key) {\n        case 111102: OKC_W(true, 1, 1, 1, 2, false, false); break;")
b = s.index('        default: TORCH_CHECK(false, "unsupported wt variant (mw, nt, ks, nstage, ksplit, tma)");')
s = s[:a] + """    switch (key) {
        case 111104: OKC_W(true, 1, 1, 1, 4, false, false); break;   case 111108: OKC_W(true, 1, 1, 1, 8, false, false); break;
        case 112104: OKC_W(true, 1, 2, 1, 4, false, false); break;   case 114102: OKC_W(true, 1, 4, 1, 2, false, false); break;
        case 114104: OKC_W(true, 1, 4, 1, 4, false, false); break;
        case 11104: OKC_W(false, 1, 1, 1, 4, false, false); break;   case 11108: OKC_W(false, 1, 1, 1, 8, false, false); break;
        case 21104: OKC_W(false, 2, 1, 1, 4, false, false); break;   case 22104: OKC_W(false, 2, 2, 1, 4, false, false); break;
        case 24102: OKC_W(false, 2, 4, 1, 2, false, false); break;   case 24104: OKC_W(false, 2, 4, 1, 4, false, false); break;
        // split-K (SPLITK) and TMA bulk-copy variants are kept as code but not instantiated: both measured slower
""" + s[b:]

# ---- finalize: PDL wait + launch attribute
rep("""    const int tok = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = row_of_pair[tok * 8 + warp];
    float ss = 0.f;
    if (r >= 0) {
        uint4 u[NSEG];""", """    pdl_wait();                                          // y comes from the previous launch
    const int tok = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = row_of_pair[tok * 8 + warp];
    float ss = 0.f;
    if (r >= 0) {
        uint4 u[NSEG];""")
rep("""void moe_finalize(torch::Tensor y, torch::Tensor part, int64_t ksplit, torch::Tensor row_of_pair, torch::Tensor tw,
                  torch::Tensor nw, torch::Tensor out, int64_t M, double eps)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(out.size(1) == 4096, "finalize: H must be 4096");
    const float* pp = ksplit > 1 ? part.data_ptr<float>() : nullptr;
    const int ldp = ksplit > 1 ? (int)part.stride(1) : 0, prow = ksplit > 1 ? (int)part.size(1) : 0;
#define OKC_F(KS) moe_finalize_kernel<4096, KS><<<(unsigned)M, 256, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), \\
        (int)y.stride(0), pp, ldp, prow, row_of_pair.data_ptr<int>(), tw.data_ptr<float>(), nw.data_ptr<float>(), \\
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (float)eps)""",
"""void moe_finalize(torch::Tensor y, torch::Tensor part, int64_t ksplit, torch::Tensor row_of_pair, torch::Tensor tw,
                  torch::Tensor nw, torch::Tensor out, int64_t M, double eps, bool pdl)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(out.size(1) == 4096, "finalize: H must be 4096");
    const float* pp = ksplit > 1 ? part.data_ptr<float>() : nullptr;
    const int ldp = ksplit > 1 ? (int)part.stride(1) : 0, prow = ksplit > 1 ? (int)part.size(1) : 0;
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
    cudaError_t err = cudaSuccess;
#define OKC_F(KS) err = cudaLaunchKernelEx(&cfg, moe_finalize_kernel<4096, KS>, reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), \\
        (int)y.stride(0), pp, ldp, prow, row_of_pair.data_ptr<int>(), tw.data_ptr<float>(), nw.data_ptr<float>(), \\
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (float)eps)""")
rep("""        default: TORCH_CHECK(false, "unsupported ksplit for the finalize");
    }
#undef OKC_F
}""", """        default: TORCH_CHECK(false, "unsupported ksplit for the finalize");
    }
#undef OKC_F
    TORCH_CHECK(err == cudaSuccess, "finalize launch failed: ", cudaGetErrorString(err));
}""")
rep("""void moe_finalize(torch::Tensor y, torch::Tensor part, int64_t ksplit, torch::Tensor row_of_pair, torch::Tensor tw,
                  torch::Tensor nw, torch::Tensor out, int64_t M, double eps);""",
    """void moe_finalize(torch::Tensor y, torch::Tensor part, int64_t ksplit, torch::Tensor row_of_pair, torch::Tensor tw,
                  torch::Tensor nw, torch::Tensor out, int64_t M, double eps, bool pdl);""")

# ---- repack: chunk-major
rep("""    # 4-tile groups: a group's chunk is 4 KB (weights) / 512 B (scales) contiguous, for bulk copies
    Wr = Wr.view(E, NTL // 4, 4, NC, 1024).permute(0, 1, 3, 2, 4).contiguous()      # e, group, chunk, slot, 1 KB
    Sr = Sr.view(E, NTL // 4, 4, NC, 128).permute(0, 1, 3, 2, 4).contiguous()       # e, group, chunk, slot, 128 B
    return Wr, Sr""",
"""    # chunk-major: an expert's chunk c (all N/16 tiles' 1 KB blocks, grouped by 4) is one contiguous region, so the
    # CTAs of an expert sweep memory together
    Wr = Wr.view(E, NTL // 4, 4, NC, 1024).permute(0, 3, 1, 2, 4).contiguous()      # e, chunk, group, slot, 1 KB
    Sr = Sr.view(E, NTL // 4, 4, NC, 128).permute(0, 3, 1, 2, 4).contiguous()       # e, chunk, group, slot, 128 B
    return Wr, Sr""")
rep('''    Wr uint8 [E, N/64, K/128, 4, 1024] (per expert, 4-tile group, chunk, tile slot: 2 halves x 32 lanes x 16 B): per''',
    '''    Wr uint8 [E, K/128, N/64, 4, 1024] (per expert, chunk, 4-tile group, tile slot: 2 halves x 32 lanes x 16 B): per''')
p.write_text(s)

# ---- generator
g = root / "tools/gen_moe_kernel_only.py"
u = g.read_text()
u = u.replace('"const float*, __half*, __nv_bfloat16*, int, float*, int, int, int, int, int, int, float, cudaStream_t)")',
              '"const float*, __half*, __nv_bfloat16*, int, float*, int, int, int, int, int, int, float, bool, cudaStream_t)")')
g.write_text(u)

# ---- CPU check view
c = root / "tools/check_repack_wt.py"
v = c.read_text()
v = v.replace("words = Wr.view(E, NTL // 4, NC, 4, 2, 32, 4, 4).permute(0, 1, 3, 2, 4, 5, 6, 7).reshape(E, NTL, NC, 2, 32, 16).contiguous().view(torch.int32).view(E, NTL, NC, 2, 32, 4)\nSr = Sr.view(E, NTL // 4, NC, 4, 8, 16).permute(0, 1, 3, 2, 4, 5).reshape(E, NTL, NC, 8, 16)",
              "words = Wr.view(E, NC, NTL // 4, 4, 2, 32, 4, 4).permute(0, 2, 3, 1, 4, 5, 6, 7).reshape(E, NTL, NC, 2, 32, 16).contiguous().view(torch.int32).view(E, NTL, NC, 2, 32, 4)\nSr = Sr.view(E, NC, NTL // 4, 4, 8, 16).permute(0, 2, 3, 1, 4, 5).reshape(E, NTL, NC, 8, 16)")
c.write_text(v)

# ---- Python glue
q = root / "muse_moe_w4a16/submission.py"
t = q.read_text()


def rep2(old, new, count=1):
    global t
    assert t.count(old) == count, (t.count(old), old[:100])
    t = t.replace(old, new)


rep2("""import triton
import triton.language as tl
""", """import triton
import triton.language as tl
try:
    from triton.language.extra.cuda import gdc_launch_dependents as _gdc_launch_dependents
except Exception:   # older Triton: no programmatic dependent launch intrinsics
    _gdc_launch_dependents = None
""")
rep2("""    pid = tl.program_id(0)
    if pid >= n_route_ctas:
        offs = (pid - n_route_ctas) * XBLOCK + tl.arange(0, XBLOCK)""",
"""    pid = tl.program_id(0)
    if PDL:
        _gdc_launch_dependents()      # the GEMM may start launching (it waits for our writes before reading them)
    if pid >= n_route_ctas:
        offs = (pid - n_route_ctas) * XBLOCK + tl.arange(0, XBLOCK)""")
rep2("""    TOPK: tl.constexpr, XBLOCK: tl.constexpr, MARLIN: tl.constexpr, MULTI: tl.constexpr,
):""", """    TOPK: tl.constexpr, XBLOCK: tl.constexpr, MARLIN: tl.constexpr, MULTI: tl.constexpr, PDL: tl.constexpr,
):""")
rep2("""        PB=PB, CH=CH, BM=BM, MAXB=MAXB, E=LOCAL, TOPK=TOPK, XBLOCK=XBLOCK, MARLIN=marlin, MULTI=NP > CH, num_warps=4,""",
     """        PB=PB, CH=CH, BM=BM, MAXB=MAXB, E=LOCAL, TOPK=TOPK, XBLOCK=XBLOCK, MARLIN=marlin, MULTI=NP > CH,
        PDL=bool(state.get("pdl", False)) and _gdc_launch_dependents is not None, num_warps=4,""")
rep2("""                     state["Wt13"], state["St13"], state["gs13"], inter, part, ksplit, H, 2 * I // 16, maxb, True, 1, nt, ks,
                     nstage, state["gs_mul"], bool(tma))""", """                     state["Wt13"], state["St13"], state["gs13"], inter, part, ksplit, H, 2 * I // 16, maxb, True, 1, nt, ks,
                     nstage, state["gs_mul"], bool(tma), pdl)""")
rep2("""                     state["Wt2"], state["St2"], state["gs2"], y, part, ksplit, I, H // 16, maxb, False, mw2, nt, ks, nstage,
                     state["gs_mul"], bool(tma))
    ext.moe_finalize(y, part, ksplit, state["row_of_pair"], tw, state["norm_w"], out, M, POST_NORM_EPS)""",
     """                     state["Wt2"], state["St2"], state["gs2"], y, part, ksplit, I, H // 16, maxb, False, mw2, nt, ks, nstage,
                     state["gs_mul"], bool(tma), pdl)
    ext.moe_finalize(y, part, ksplit, state["row_of_pair"], tw, state["norm_w"], out, M, POST_NORM_EPS, pdl)""")
rep2("""    nt, mw2, ks, nstage, ksplit, tma = _wt_variant(M, state)
    BM = 8 * nt""", """    nt, mw2, ks, nstage, ksplit, tma = _wt_variant(M, state)
    pdl = bool(state.get("pdl", False)) and _PDL_OVERRIDE is not False
    BM = 8 * nt""")
rep2("_WT_OVERRIDE = None\n\n\ndef _wt_variant(M: int, state):", "_WT_OVERRIDE = None\n_PDL_OVERRIDE = None     # bench: False forces plain launches\n\n\ndef _wt_variant(M: int, state):")
# bench parts: pass pdl
rep2("""state["inter"], part, ksp, 4096, 512, maxb, True, 1, nt, ks, ns, 64.0, bool(tma)),""", """state["inter"], part, ksp, 4096, 512, maxb, True, 1, nt, ks, ns, 64.0, bool(tma), False),""")
rep2("""state["y"], part, ksp, 4096, 256, maxb, False, mw2, nt, ks, ns, 64.0, bool(tma)),""", """state["y"], part, ksp, 4096, 256, maxb, False, mw2, nt, ks, ns, 64.0, bool(tma), False),""")
rep2("""            "finalize": lambda: ext.moe_finalize(state["y"], part, ksp, state["row_of_pair"], tw, state["norm_w"], out, bs, POST_NORM_EPS),""",
     """            "finalize": lambda: ext.moe_finalize(state["y"], part, ksp, state["row_of_pair"], tw, state["norm_w"], out, bs, POST_NORM_EPS, False),""")
# bench: PDL on/off comparison of the full path per shape
rep2("""    small = [(1, 1, 1, 8, 1, 0), (1, 1, 1, 4, 1, 0), (1, 1, 1, 8, 1, 1), (1, 1, 1, 4, 1, 1), (1, 2, 1, 4, 1, 1), (1, 1, 1, 4, 4, 0)]""",
     """    global _PDL_OVERRIDE
    if state.get("pdl", False):
        for flag in (False, True):
            _PDL_OVERRIDE = flag
            try:
                r = [_timeit_cold(lambda bs=bs: _moe_forward_wt(*data[bs], 0, 4096, state), dev) for bs in shapes]
                print(f"  wt pdl={int(flag)}      : " + "  ".join(f"bs{bs} {t:7.1f}" for bs, t in zip(shapes, r)))
            except Exception as e:
                print(f"  wt pdl={int(flag)} failed: {str(e)[:80]}")
        _PDL_OVERRIDE = None
    small = [(1, 1, 1, 8, 1, 0), (1, 1, 1, 4, 1, 0), (1, 2, 1, 4, 1, 0), (1, 2, 1, 8, 1, 0)]""")
rep2("""    big = [(2, 2, 1, 4, 1, 0), (4, 2, 1, 2, 1, 0), (2, 2, 1, 4, 1, 1), (4, 2, 1, 4, 1, 1), (1, 2, 1, 4, 1, 1), (2, 1, 1, 4, 1, 1)]""",
     """    big = [(2, 2, 1, 4, 1, 0), (4, 2, 1, 2, 1, 0), (4, 2, 1, 4, 1, 0), (2, 2, 1, 8, 1, 0)]""")
rep2("""WT_TEST_VARIANTS = ((1, 1, 1, 4, 1, 0), (1, 1, 1, 8, 1, 0), (2, 2, 1, 4, 1, 0), (4, 2, 1, 2, 1, 0), (1, 1, 1, 4, 4, 0),
                    (1, 1, 1, 4, 1, 1), (1, 1, 1, 8, 1, 1), (1, 2, 1, 4, 1, 1), (2, 2, 1, 4, 1, 1), (4, 2, 1, 4, 1, 1), (2, 1, 1, 4, 1, 1))""",
     """WT_TEST_VARIANTS = ((1, 1, 1, 4, 1, 0), (1, 1, 1, 8, 1, 0), (1, 2, 1, 4, 1, 0), (2, 2, 1, 4, 1, 0), (4, 2, 1, 2, 1, 0), (4, 2, 1, 4, 1, 0))""")
# PDL capture self-test in prepare (after the selftest)
rep2("""    if "Wt13" in state and "bank" in state:
        try:
            _selftest_wt(state)
        except Exception as e:
            print("selftest wt failed:", repr(e)[:200])
            state["wt_ok"] = False
        torch.cuda.synchronize()""",
"""    if "Wt13" in state and "bank" in state:
        try:
            _selftest_wt(state)
        except Exception as e:
            print("selftest wt failed:", repr(e)[:200])
            state["wt_ok"] = False
        torch.cuda.synchronize()
    if "Wt13" in state and state.get("wt_ok", True) and PDL_ENABLED:
        _pdl_selftest(state)
        torch.cuda.synchronize()""")
rep2("""AUTOTUNE = False   # pick the wt variant per shape""", """PDL_ENABLED = True   # programmatic dependent launch across the 4 launches (verified under graph capture in prepare)


def _pdl_selftest(state):
    \"\"\"PDL launches must be correct eagerly and capturable into a CUDA graph; otherwise plain launches.\"\"\"
    dev = state["w13"].device
    try:
        x, ids, tw = _judge_inputs(4, dev, seed=11)
        state["pdl"] = False
        ref = _moe_forward_wt(x, ids, tw, 0, 4096, state).float()
        state["pdl"] = True
        out = _moe_forward_wt(x, ids, tw, 0, 4096, state).float()
        torch.cuda.synchronize()
        err = float((out - ref).norm() / ref.norm().clamp(min=1e-12))
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out2 = _moe_forward_wt(x, ids, tw, 0, 4096, state)
        g.replay()
        torch.cuda.synchronize()
        err2 = float((out2.float() - ref).norm() / ref.norm().clamp(min=1e-12))
        state["pdl"] = err < 1e-3 and err2 < 1e-3
        print(f"pdl selftest: eager {err:.2e} graph {err2:.2e} -> {'enabled' if state['pdl'] else 'DISABLED'}")
    except Exception as e:
        state["pdl"] = False
        print("pdl selftest failed -> plain launches:", str(e)[:160])


AUTOTUNE = False   # pick the wt variant per shape""")
rep2("AUTOTUNE = True    # pick the wt variant per shape", "AUTOTUNE = False   # pick the wt variant per shape")
rep2("WT_DEFAULT = {8: (1, 1, 1, 8, 1, 1), 64: (1, 1, 1, 4, 1, 1), 128: (2, 2, 1, 4, 1, 1), 100000: (4, 2, 1, 2, 1, 0)}",
     "WT_DEFAULT = {8: (1, 1, 1, 8, 1, 0), 64: (1, 1, 1, 4, 1, 0), 128: (2, 2, 1, 4, 1, 0), 100000: (4, 2, 1, 2, 1, 0)}")
q.write_text(t)

# compile_moe.py: PDL constexpr
m = root / "tools/compile_moe.py"
w = m.read_text()
w = w.replace('"MARLIN": "constexpr", "MULTI": "constexpr"},', '"MARLIN": "constexpr", "MULTI": "constexpr", "PDL": "constexpr"},')
w = w.replace('"MULTI": NP > min(NP, 256)}, 4, 1, f"route_bs{bs}")', '"MULTI": NP > min(NP, 256), "PDL": True}, 4, 1, f"route_bs{bs}")')
m.write_text(w)
print("edited")
