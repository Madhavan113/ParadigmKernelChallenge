"""One-off edit: (1) mixed block sizes: the route kernel gives experts with more than BM rows blocks of BM2 rows
(blk_bm per block) and each wt kernel instantiation skips the other size's blocks, so heavy experts stream their
weights per 32-token block while light experts keep 16-token blocks; (2) l2pf: for small token counts GEMM2's CTAs
(pre-launched under PDL) prefetch the rest of their weight slice and the next two blocks' into L2 before waiting,
and GEMM1 triggers its dependents at the start and marks its streamed weights evict-first."""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
p = root / "muse_moe_w4a16/cuda_moe.py"
s = p.read_text()


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:90])
    s = s.replace(old, new)


# ---- helpers
rep("""__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory"); }
""", """__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory"); }
__device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2::evict_last [%0];\\n" :: "l"(p)); }
__device__ __forceinline__ void cp_async16_pol(uint32_t dst, const void* src, int src_bytes, uint64_t pol) {
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;\\n" :: "r"(dst), "l"(src), "r"(src_bytes), "l"(pol));
}
""")
# ---- wt kernel: blk_bm + l2pf args, early exit on the other block size
rep("""                    const int* __restrict__ blk_end, const int* __restrict__ nblk,
                    const uint8_t* __restrict__ Wr, const uint8_t* __restrict__ Sr, const float* __restrict__ gs,
                    __half* __restrict__ out16, __nv_bfloat16* __restrict__ out_bf, int ldo,
                    float* __restrict__ part, int ldp, int part_rows, int* __restrict__ counters,
                    int K, int NTILES, float gs_mul)
{""", """                    const int* __restrict__ blk_end, const int* __restrict__ nblk, const int* __restrict__ blk_bm,
                    const uint8_t* __restrict__ Wr, const uint8_t* __restrict__ Sr, const float* __restrict__ gs,
                    __half* __restrict__ out16, __nv_bfloat16* __restrict__ out_bf, int ldo,
                    float* __restrict__ part, int ldp, int part_rows, int* __restrict__ counters,
                    int K, int NTILES, float gs_mul, int l2pf)
{""")
rep("""    if (GATED) pdl_wait();                            // the routing lists and x16 come from the previous launch
    const int j = blockIdx.y;
    if (j >= *nblk) return;
    const int e = blk_e[j];""", """    if (GATED) pdl_wait();                            // the routing lists and x16 come from the previous launch
    if (GATED && l2pf) pdl_trigger();                 // small batches: let GEMM2's CTAs pre-launch and prefetch W2
    const int j = blockIdx.y;
    if (j >= *nblk) return;
    if (blk_bm[j] != 8 * NT) return;                  // mixed block sizes: this block belongs to the other kernel
    const int e = blk_e[j];""")
# weight cp.async with an L2 policy (evict-first while l2pf protects the prefetched W2)
rep("""    auto issue_w = [&](int c, int stage) {
        const uint32_t ws = smem_base + stage * STAGE_BYTES;
        const uint32_t ss = ws + W_STAGE;
        #pragma unroll
        for (int i = 0; i < WSEG; ++i)
            cp_async16(ws + (tid + i * NTHR) * 16, wsrc[i] + c * 1024, 16);
        if (tid < TPC * 8) cp_async16(ss + tid * 16, ssrc + c * 128, 16);
    };""", """    uint64_t pol;
    if (l2pf) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\\n" : "=l"(pol));
    else asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;\\n" : "=l"(pol));
    auto issue_w = [&](int c, int stage) {
        const uint32_t ws = smem_base + stage * STAGE_BYTES;
        const uint32_t ss = ws + W_STAGE;
        #pragma unroll
        for (int i = 0; i < WSEG; ++i)
            cp_async16_pol(ws + (tid + i * NTHR) * 16, wsrc[i] + c * 1024, 16, pol);
        if (tid < TPC * 8) cp_async16_pol(ss + tid * 16, ssrc + c * 128, 16, pol);
    };""")
rep("""        #pragma unroll
        for (int st = 0; st < NSTAGE - 1; ++st)
            if (st < NC) issue_w(st, st);
        pdl_wait();
        #pragma unroll
        for (int st = 0; st < NSTAGE - 1; ++st) {
            if (st < NC) issue_x(st, st);
            cp_async_commit();
        }""", """        #pragma unroll
        for (int st = 0; st < NSTAGE - 1; ++st)
            if (st < NC) issue_w(st, st);
        if (l2pf && (tid & 7) == 0) {
            // prefetch the rest of this slice and the same tiles of the next two blocks (their experts) into L2
            for (int k = 0; k < 3; ++k) {
                const int jb = j + k;
                if (jb >= *nblk) break;
                const size_t eb = (size_t)blk_e[jb] * NTILES;
                #pragma unroll
                for (int i = 0; i < WSEG; ++i) {
                    const int sidx = tid + i * NTHR;
                    const uint8_t* wb = Wr + ((eb + tile_of(sidx >> 6)) * NC_ALL + c0) * 1024 + (sidx & 63) * 16;
                    for (int c = k == 0 ? NSTAGE - 1 : 0; c < NC; ++c) prefetch_l2(wb + c * 1024);
                }
                if (tid < TPC * 8) {
                    const uint8_t* sb = Sr + ((eb + tile_of(tid >> 3)) * NC_ALL + c0) * 128 + (tid & 7) * 16;
                    for (int c = k == 0 ? NSTAGE - 1 : 0; c < NC; ++c) prefetch_l2(sb + c * 128);
                }
            }
        }
        pdl_wait();
        #pragma unroll
        for (int st = 0; st < NSTAGE - 1; ++st) {
            if (st < NC) issue_x(st, st);
            cp_async_commit();
        }""")
rep("""    pdl_trigger();                                    // main loop done: the next kernel may start launching
    cp_async_wait<0>();""", """    if (!(GATED && l2pf)) pdl_trigger();              // main loop done: the next kernel may start launching
    cp_async_wait<0>();""")
# ---- launcher / host wrapper
rep("""void launch_wt(const __half* X, int ldx, const int* st, const int* be, const int* br, const int* bend, const int* nb,
               const uint8_t* Wr, const uint8_t* Sr, const float* gs, __half* o16, __nv_bfloat16* obf, int ldo,
               float* part, int ldp, int part_rows, int* counters, int ksplit,
               int K, int ntiles, int max_blocks, float gs_mul, bool pdl, cudaStream_t stream)""",
    """void launch_wt(const __half* X, int ldx, const int* st, const int* be, const int* br, const int* bend, const int* nb, const int* bbm,
               const uint8_t* Wr, const uint8_t* Sr, const float* gs, __half* o16, __nv_bfloat16* obf, int ldo,
               float* part, int ldp, int part_rows, int* counters, int ksplit,
               int K, int ntiles, int max_blocks, float gs_mul, bool pdl, bool l2pf, cudaStream_t stream)""")
rep("""        X, ldx, st, be, br, bend, nb, Wr, Sr, gs, o16, obf, ldo, part, ldp, part_rows, counters, K, ntiles, gs_mul);""",
    """        X, ldx, st, be, br, bend, nb, bbm, Wr, Sr, gs, o16, obf, ldo, part, ldp, part_rows, counters, K, ntiles, gs_mul, (int)l2pf);""")
rep("""void moe_w4a16_wt(torch::Tensor X, torch::Tensor sorted_tok, torch::Tensor blk_e, torch::Tensor blk_row0,
                  torch::Tensor blk_end, torch::Tensor nblk, torch::Tensor Wr, torch::Tensor Sr, torch::Tensor gs,
                  torch::Tensor out, torch::Tensor part, torch::Tensor counters, int64_t ksplit, int64_t K, int64_t ntiles,
                  int64_t max_blocks, bool gated, int64_t mw, int64_t nt, int64_t ks, int64_t nstage, double gs_mul, int64_t abl,
                  bool pdl)""", """void moe_w4a16_wt(torch::Tensor X, torch::Tensor sorted_tok, torch::Tensor blk_e, torch::Tensor blk_row0,
                  torch::Tensor blk_end, torch::Tensor nblk, torch::Tensor blk_bm, torch::Tensor Wr, torch::Tensor Sr, torch::Tensor gs,
                  torch::Tensor out, torch::Tensor part, torch::Tensor counters, int64_t ksplit, int64_t K, int64_t ntiles,
                  int64_t max_blocks, bool gated, int64_t mw, int64_t nt, int64_t ks, int64_t nstage, double gs_mul, int64_t abl,
                  bool pdl, bool l2pf)""", 2)
rep("""        blk_row0.data_ptr<int>(), blk_end.data_ptr<int>(), nblk.data_ptr<int>(), Wp, Sp, gs.data_ptr<float>(), o16, obf, ldo, \\
        pp, ldp, prow, cnt, (int)ksplit, (int)K, (int)ntiles, (int)max_blocks, (float)gs_mul, pdl, stream)""",
    """        blk_row0.data_ptr<int>(), blk_end.data_ptr<int>(), nblk.data_ptr<int>(), blk_bm.data_ptr<int>(), Wp, Sp, gs.data_ptr<float>(), o16, obf, ldo, \\
        pp, ldp, prow, cnt, (int)ksplit, (int)K, (int)ntiles, (int)max_blocks, (float)gs_mul, pdl, l2pf, stream)""")
# ---- route kernel: BM2 + blk_bm
rep("""                 int* __restrict__ blk_row0, int* __restrict__ blk_end, int* __restrict__ nblk,
                 int M, int local_start, int BM, int n_elems, float xscale)""",
    """                 int* __restrict__ blk_row0, int* __restrict__ blk_end, int* __restrict__ nblk, int* __restrict__ blk_bm,
                 int M, int local_start, int BM, int BM2, int n_elems, float xscale)""")
rep("""    __shared__ int pad_off[E], blk_off[E], nb_s[E], cnt_s[E];""",
    """    __shared__ int pad_off[E], blk_off[E], nb_s[E], cnt_s[E], bm_s[E];""")
rep("""        const int cnt = base[lane];
        const int padded = (cnt + BM - 1) / BM * BM;
        const int nb = padded / BM;""", """        const int cnt = base[lane];
        const int bm = (BM2 > 0 && cnt > BM) ? BM2 : BM;          // heavier experts get the larger blocks
        const int padded = (cnt + bm - 1) / bm * bm;
        const int nb = padded / bm;""")
rep("""        cnt_s[lane] = cnt; pad_off[lane] = po - padded; blk_off[lane] = bo - nb; nb_s[lane] = nb;""",
    """        cnt_s[lane] = cnt; pad_off[lane] = po - padded; blk_off[lane] = bo - nb; nb_s[lane] = nb; bm_s[lane] = bm;""")
rep("""        blk_row0[j] = pad_off[ej] + (j - blk_off[ej]) * BM;
        blk_end[j] = pad_off[ej] + cnt_s[ej];""", """        blk_row0[j] = pad_off[ej] + (j - blk_off[ej]) * bm_s[ej];
        blk_end[j] = pad_off[ej] + cnt_s[ej];
        blk_bm[j] = bm_s[ej];""")
rep("""               torch::Tensor blk_e, torch::Tensor blk_row0, torch::Tensor blk_end, torch::Tensor nblk,
               int64_t M, int64_t local_start, int64_t BM, double xscale)""",
    """               torch::Tensor blk_e, torch::Tensor blk_row0, torch::Tensor blk_end, torch::Tensor nblk, torch::Tensor blk_bm,
               int64_t M, int64_t local_start, int64_t BM, int64_t BM2, double xscale)""", 2)   # definition + declaration
rep("""        blk_end.data_ptr<int>(), nblk.data_ptr<int>(), (int)M, (int)local_start, (int)BM, n_elems, (float)xscale);""",
    """        blk_end.data_ptr<int>(), nblk.data_ptr<int>(), blk_bm.data_ptr<int>(), (int)M, (int)local_start, (int)BM, (int)BM2, n_elems, (float)xscale);""")
p.write_text(s)
print("cuda mixed/l2pf written")

# ---- submission
q = root / "muse_moe_w4a16/submission.py"
t = q.read_text()


def rep2(old, new, n=1):
    global t
    assert t.count(old) == n, (t.count(old), old[:90])
    t = t.replace(old, new)


rep2("""        "nblk": torch.zeros((1,), dtype=torch.int32, device=dev),""",
     """        "nblk": torch.zeros((1,), dtype=torch.int32, device=dev),
        "blk_bm": torch.zeros((maxb,), dtype=torch.int32, device=dev),""")
rep2("""PDL = True             # programmatic dependent launch GEMM1 -> GEMM2 -> finalize
_PDL_OVERRIDE = None   # bench: False forces plain launches
""", """PDL = True             # programmatic dependent launch GEMM1 -> GEMM2 -> finalize
_PDL_OVERRIDE = None   # bench: False forces plain launches
# mixed block sizes above this many tokens: experts with more rows than the policy's block get the big variant's blocks
WT_BIG = {128: (4, 2, 1, 2, 1), 100000: (4, 2, 1, 2, 1)}
MIXED_MIN_TOKENS = 65
_MIXED_OVERRIDE = None
L2PF_MAX_TOKENS = 16   # up to this many tokens GEMM2 prefetches its W2 slice into L2 while GEMM1 runs (PDL)
_L2PF_OVERRIDE = None
""")
rep2("""ROUTE_CUDA = True   # the one-launch CUDA routing kernel (ranks + block list + fp16 copy of x) instead of the Triton one


def _route_wt(x, ids, local_start: int, BM: int, state, ext):
    M, H = x.shape
    if not ROUTE_CUDA:
        return _route(x, ids, local_start, BM, state, marlin=False)
    NP = max(16, triton.next_power_of_2(M * TOPK))
    MAXB = triton.next_power_of_2(NP // BM + LOCAL)
    ext.moe_route(ids, x.contiguous(), state["x16"], state["sorted_tok"], state["row_of_pair"], state["blk_e"], state["blk_row0"],
                  state["blk_end"], state["nblk"], M, local_start, BM, float(state.get("xscale", 1.0)))
    return NP, MAXB
""", """ROUTE_CUDA = True   # the one-launch CUDA routing kernel (ranks + block list + fp16 copy of x) instead of the Triton one


def _wt_big_variant(M: int, state, small):
    \"\"\"The variant for the heavier experts' blocks (None: a single block size).\"\"\"
    mixed = (M >= MIXED_MIN_TOKENS) if _MIXED_OVERRIDE is None else _MIXED_OVERRIDE
    if not mixed or _WT_OVERRIDE is not None:
        return None
    keys = sorted(WT_BIG)
    v = WT_BIG[keys[-1]]
    for k in keys:
        if M <= k:
            v = WT_BIG[k]
            break
    if v[0] <= small[0] or v in state.get("wt_bad", ()):
        return None
    return v


def _route_wt(x, ids, local_start: int, BM: int, state, ext, BM2: int = 0):
    M, H = x.shape
    if not ROUTE_CUDA:
        state["blk_bm"].fill_(BM)
        return _route(x, ids, local_start, BM, state, marlin=False)
    NP = max(16, triton.next_power_of_2(M * TOPK))
    MAXB = triton.next_power_of_2(NP // BM + LOCAL)
    ext.moe_route(ids, x.contiguous(), state["x16"], state["sorted_tok"], state["row_of_pair"], state["blk_e"], state["blk_row0"],
                  state["blk_end"], state["nblk"], state["blk_bm"], M, local_start, BM, BM2, float(state.get("xscale", 1.0)))
    return NP, MAXB
""")
rep2("""    nt, mw2, ks, nstage, ksplit = _wt_variant(M, state)
    pdl = PDL if _PDL_OVERRIDE is None else _PDL_OVERRIDE
    BM = 8 * nt
    if M * TOPK + LOCAL * (BM - 1) > PART_ROWS:
        ksplit = 1
    ext = build()
    NP, MAXB = _route_wt(x, ids, local_start, BM, state, ext)
    maxb = min(MAXB, _wt_max_blocks(M, BM))
    x16 = state["x16"][:M]
    out = torch.empty((M, H), dtype=MOE_OUT_DTYPE, device=x.device)
    inter = state["inter"]
    y = state["y"]
    part = state["part"]
    counters = state["counters"]
    ext.moe_w4a16_wt(x16, state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                     state["Wt13"], state["St13"], state["gs13"], inter, part, counters, ksplit, H, 2 * I // 16, maxb, True, 1, nt, ks,
                     nstage, state["gs_mul"], _WT_ABL, pdl)
    ext.moe_w4a16_wt(inter, state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                     state["Wt2"], state["St2"], state["gs2"], y, part, counters, ksplit, I, H // 16, maxb, False, mw2, nt, ks, nstage,
                     state["gs_mul"], _WT_ABL, pdl)
    ext.moe_finalize(y, state["row_of_pair"], tw, state["norm_w"], out, M, POST_NORM_EPS, pdl)
    return out""", """    small = _wt_variant(M, state)
    nt, mw2, ks, nstage, ksplit = small
    big = _wt_big_variant(M, state, small)
    pdl = PDL if _PDL_OVERRIDE is None else _PDL_OVERRIDE
    l2pf = pdl and ((M <= L2PF_MAX_TOKENS) if _L2PF_OVERRIDE is None else _L2PF_OVERRIDE)
    BM = 8 * nt
    BM2 = 8 * big[0] if big is not None else 0
    if M * TOPK + LOCAL * (BM - 1) > PART_ROWS:
        ksplit = 1
    ext = build()
    NP, MAXB = _route_wt(x, ids, local_start, BM, state, ext, BM2)
    maxb = min(MAXB, _wt_max_blocks(M, BM))
    x16 = state["x16"][:M]
    out = torch.empty((M, H), dtype=MOE_OUT_DTYPE, device=x.device)
    inter = state["inter"]
    y = state["y"]
    part = state["part"]
    counters = state["counters"]
    lists = (state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"], state["blk_bm"])
    # the big variant's launch is a plain one (its blocks are independent of the small kernel's, and stream order
    # keeps the next PDL kernel's wait covering both)
    ext.moe_w4a16_wt(x16, *lists, state["Wt13"], state["St13"], state["gs13"], inter, part, counters, ksplit, H, 2 * I // 16, maxb,
                     True, 1, nt, ks, nstage, state["gs_mul"], _WT_ABL, pdl, l2pf)
    if big is not None:
        ext.moe_w4a16_wt(x16, *lists, state["Wt13"], state["St13"], state["gs13"], inter, part, counters, 1, H, 2 * I // 16, maxb,
                         True, 1, big[0], big[2], big[3], state["gs_mul"], _WT_ABL, False, False)
    ext.moe_w4a16_wt(inter, *lists, state["Wt2"], state["St2"], state["gs2"], y, part, counters, ksplit, I, H // 16, maxb,
                     False, mw2, nt, ks, nstage, state["gs_mul"], _WT_ABL, pdl, l2pf)
    if big is not None:
        ext.moe_w4a16_wt(inter, *lists, state["Wt2"], state["St2"], state["gs2"], y, part, counters, 1, I, H // 16, maxb,
                         False, big[1], big[0], big[2], big[3], state["gs_mul"], _WT_ABL, False, False)
    ext.moe_finalize(y, state["row_of_pair"], tw, state["norm_w"], out, M, POST_NORM_EPS, pdl)
    return out""")
# bench parts / ablation calls: the extra list tensor + l2pf flag
rep2("""ext.moe_w4a16_wt(x16, state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],""",
     """ext.moe_w4a16_wt(x16, state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"], state["blk_bm"],""", 2)
rep2("""ext.moe_w4a16_wt(state["inter"], state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],""",
     """ext.moe_w4a16_wt(state["inter"], state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"], state["blk_bm"],""", 2)
rep2(', nt, ks, ns, state["gs_mul"], _WT_ABL, False)', ', nt, ks, ns, state["gs_mul"], _WT_ABL, False, False)', 4)
# self-test: the mixed dispatch at the shapes that use it
rep2("""        for v in WT_TEST_VARIANTS:
            _WT_OVERRIDE = v
            try:
                out = _moe_forward_wt(x, ids, tw, 0, 4096, state).float()""",
     """        for v in WT_TEST_VARIANTS + ("mixed",):
            if v == "mixed":
                if bs < MIXED_MIN_TOKENS:
                    continue
                _MIXED_OVERRIDE = True
            else:
                _WT_OVERRIDE = v
            try:
                out = _moe_forward_wt(x, ids, tw, 0, 4096, state).float()""")
rep2("""            finally:
                _WT_OVERRIDE = None
            worst[v] = max(worst.get(v, 0.0), err)
        print(f"  selftest bs={bs:3d} {dist:6s}: " + " ".join(f"{v}:{worst[v]:.2e}" for v in WT_TEST_VARIANTS))""",
     """            finally:
                _WT_OVERRIDE = None
                _MIXED_OVERRIDE = None
            worst[v] = max(worst.get(v, 0.0), err)
        print(f"  selftest bs={bs:3d} {dist:6s}: " + " ".join(f"{v}:{worst[v]:.2e}" for v in worst))""")
rep2("""    global _WT_OVERRIDE
    t0 = time.time()
    dev = state["w13"].device
    worst = {}""", """    global _WT_OVERRIDE, _MIXED_OVERRIDE
    t0 = time.time()
    dev = state["w13"].device
    worst = {}""")
rep2("""    bad = {v for v, e in worst.items() if not e < 1.2e-2}   # a broken variant is off by O(1); Marlin itself differs by ~6e-3
    state["wt_bad"] = bad""", """    bad = {v for v, e in worst.items() if not e < 1.2e-2}   # a broken variant is off by O(1); Marlin itself differs by ~6e-3
    if "mixed" in bad:
        global MIXED_MIN_TOKENS
        MIXED_MIN_TOKENS = 10 ** 9
    state["wt_bad"] = bad""")
# bench: mixed and l2pf rows
rep2("""    global _WT_ABL, _PDL_OVERRIDE
    for flag in (False, True):
        _PDL_OVERRIDE = flag""", """    global _WT_ABL, _PDL_OVERRIDE, _MIXED_OVERRIDE, _L2PF_OVERRIDE
    for name, var in (("mixed", "_MIXED_OVERRIDE"), ("l2pf", "_L2PF_OVERRIDE")):
        for flag in (False, True):
            globals()[var] = flag
            try:
                r = [_timeit_cold(lambda bs=bs: _moe_forward_wt(*data[bs], 0, 4096, state), dev) for bs in shapes]
                print(f"  wt {name}={int(flag)}    : " + "  ".join(f"bs{bs} {t:7.1f}" for bs, t in zip(shapes, r)))
            except Exception as e:
                print(f"  wt {name}={int(flag)} failed: {str(e)[:80]}")
            finally:
                globals()[var] = None
    for flag in (False, True):
        _PDL_OVERRIDE = flag""")
q.write_text(t)
print("submission mixed/l2pf written")
