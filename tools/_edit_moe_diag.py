"""Phase 3b step 1: DIAG stamps + ablation bits on the wt kernel's bulk path, and _bench_diag. Kernel: template DIAG
(bool) + a `diag` pointer param; %globaltimer at entry / first stage landed / loop end / epilogue end for CTA (0,0,0),
the per-stage wait time accumulated by its thread 0, grid-wide min start / max end; ABL 4 = no x loads, 5 = no
epilogue stores, 6 = no weight copies (expect_tx(0) keeps the mbarrier protocol). Host: `diag` tensor arg, key digit,
instantiations for the locked variant pair. Run: python3 tools/_edit_moe_diag.py && python3 tools/embed_cuda.py"""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
cp = root / "muse_moe_w4a16/cuda_moe.py"; s = cp.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


if "bool DIAG" not in s:
    # helpers
    rep("template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile(\"cp.async.wait_group %0;\\n\" :: \"n\"(N)); }\n",
        "template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile(\"cp.async.wait_group %0;\\n\" :: \"n\"(N)); }\n"
        "__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile(\"mov.u64 %0, %%globaltimer;\" : \"=l\"(t)); return t; }\n")
    rep("template <bool GATED, int MW, int NT, int KS, int NSTAGE, bool SPLITK, int ABL, bool BULK = false, int CPC = 1>\n__global__ void __launch_bounds__(128 * KS)\nmoe_w4a16_wt_kernel(",
        "template <bool GATED, int MW, int NT, int KS, int NSTAGE, bool SPLITK, int ABL, bool BULK = false, int CPC = 1, bool DIAG = false>\n__global__ void __launch_bounds__(128 * KS)\nmoe_w4a16_wt_kernel(")
    rep("                    int K, int NTILES, float gs_mul, int l2pf)\n{",
        "                    int K, int NTILES, float gs_mul, int l2pf, unsigned long long* __restrict__ diag)\n{")
    # stamps: entry
    rep("    if (GATED) pdl_wait();                            // the routing lists and x16 come from the previous launch\n    const int j = blockIdx.y;",
        "    unsigned long long t0 = 0, t1 = 0, wacc = 0;\n    if constexpr (DIAG) t0 = gtimer();\n"
        "    if (GATED) pdl_wait();                            // the routing lists and x16 come from the previous launch\n    const int j = blockIdx.y;")
    # ablation 6: no weight copies (bulk) / ablation 4: no x loads
    rep("        if constexpr (BULK) {\n            if (tid == 0) {\n                const uint32_t bar = smem_u32(&wfull[stage]);\n                fence_proxy_async();\n                mbar_expect_tx(bar, W_STAGE + S_STAGE);\n                bulk_g2s(ws, wgrp + (size_t)c * W_STAGE, W_STAGE, bar);\n                bulk_g2s(ss, sgrp + (size_t)c * S_STAGE, S_STAGE, bar);\n            }\n            return;\n        }",
        "        if constexpr (BULK) {\n            if (tid == 0) {\n                const uint32_t bar = smem_u32(&wfull[stage]);\n                fence_proxy_async();\n"
        "                if (ABL == 6) { mbar_expect_tx(bar, 0); }                 // ablation: no weight stream (the phase still completes)\n"
        "                else {\n                    mbar_expect_tx(bar, W_STAGE + S_STAGE);\n                    bulk_g2s(ws, wgrp + (size_t)c * W_STAGE, W_STAGE, bar);\n                    bulk_g2s(ss, sgrp + (size_t)c * S_STAGE, S_STAGE, bar);\n                }\n            }\n            return;\n        }")
    rep("    auto issue_x = [&](int c, int stage) {\n        const uint32_t xs = smem_base + stage * STAGE_BYTES + W_STAGE + S_STAGE;\n",
        "    auto issue_x = [&](int c, int stage) {\n        if (ABL == 4) return;                                             // ablation: no token-tile loads\n        const uint32_t xs = smem_base + stage * STAGE_BYTES + W_STAGE + S_STAGE;\n")
    # the per-stage wait block: stamp around it
    rep("    for (int c = 0; c < NC; ++c) {\n        cp_async_wait<NSTAGE - 2>();\n        __syncthreads();\n        {\n            const int cn = c + NSTAGE - 1;\n            if (cn < NC) issue(cn, stage_n);\n            cp_async_commit();\n        }\n        const uint8_t* ws = wt_smem + stage * STAGE_BYTES;\n        if constexpr (BULK) mbar_wait(smem_u32(&wfull[stage]), (uint32_t)((c / NSTAGE) & 1));   // this chunk's weights landed\n",
        "    for (int c = 0; c < NC; ++c) {\n        unsigned long long tw = 0;\n        if constexpr (DIAG) tw = gtimer();\n"
        "        cp_async_wait<NSTAGE - 2>();\n        __syncthreads();\n        {\n            const int cn = c + NSTAGE - 1;\n            if (cn < NC) issue(cn, stage_n);\n            cp_async_commit();\n        }\n        const uint8_t* ws = wt_smem + stage * STAGE_BYTES;\n        if constexpr (BULK) mbar_wait(smem_u32(&wfull[stage]), (uint32_t)((c / NSTAGE) & 1));   // this chunk's weights landed\n"
        "        if constexpr (DIAG) { const unsigned long long tn = gtimer(); wacc += tn - tw; if (c == 0) t1 = tn; }\n")
    # loop end / epilogue stamps
    rep("    pdl_trigger();                                    // main loop done: the next kernel may start launching\n    cp_async_wait<0>();\n",
        "    pdl_trigger();                                    // main loop done: the next kernel may start launching\n    cp_async_wait<0>();\n"
        "    unsigned long long t2 = 0;\n    if constexpr (DIAG) t2 = gtimer();\n"
        "    if (ABL == 5) {                                   // ablation: no epilogue (keep the sums alive)\n"
        "        if (acc[0][0][0] == 12345.678f) out_bf[tid] = __float2bfloat16(acc[TPW - 1][NT * CH - 1][3]);\n"
        "        return;\n    }\n")
    # final stamp: after the output stores (end of kernel): find the kernel's last closing brace via the launcher anchor
    rep("template <bool GATED, int MW, int NT, int KS, int NSTAGE, bool SPLITK, int ABL, bool BULK = false, int CPC = 1>\nvoid launch_wt(",
        "template <bool GATED, int MW, int NT, int KS, int NSTAGE, bool SPLITK, int ABL, bool BULK = false, int CPC = 1, bool DIAG = false>\nvoid launch_wt(")
    s = s.replace("moe_w4a16_wt_kernel<GATED, MW, NT, KS, NSTAGE, SPLITK, ABL, BULK, CPC>", "moe_w4a16_wt_kernel<GATED, MW, NT, KS, NSTAGE, SPLITK, ABL, BULK, CPC, DIAG>")
    rep("               int K, int ntiles, int max_blocks, float gs_mul, bool pdl, bool l2pf, cudaStream_t stream)\n{",
        "               int K, int ntiles, int max_blocks, float gs_mul, bool pdl, bool l2pf, unsigned long long* diag, cudaStream_t stream)\n{")
    rep("        X, ldx, st, be, br, bend, nb, bbm, Wr, Sr, gs, o16, obf, ldo, part, ldp, part_rows, counters, K, ntiles, gs_mul, (int)l2pf);",
        "        X, ldx, st, be, br, bend, nb, bbm, Wr, Sr, gs, o16, obf, ldo, part, ldp, part_rows, counters, K, ntiles, gs_mul, (int)l2pf, diag);")
    # the DIAG stores at the very end of the kernel: insert before the kernel's closing brace, which precedes the launcher template
    kend = s.index("template <bool GATED, int MW, int NT, int KS, int NSTAGE, bool SPLITK, int ABL, bool BULK = false, int CPC = 1, bool DIAG = false>\nvoid launch_wt(")
    close = s.rindex("\n}\n", 0, kend)
    s = s[:close] + ("\n    if constexpr (DIAG) {\n        __syncthreads();\n        if (tid == 0) {\n            const unsigned long long t3 = gtimer();\n"
                     "            if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { diag[0] = t0; diag[1] = t1; diag[2] = t2; diag[3] = t3; diag[6] = wacc; diag[7] = (unsigned long long)NC; }\n"
                     "            atomicMin(diag + 4, t0); atomicMax(diag + 5, t3);\n        }\n    }") + s[close:]
    # host: diag arg, key digit, instantiations
    rep("                  bool pdl, bool l2pf, int64_t cpc)\n{", "                  bool pdl, bool l2pf, int64_t cpc, torch::Tensor diag, bool diag_on)\n{")
    rep("    const int key = (int)cpc * 100000000 + (int)abl * 10000000",
        "    TORCH_CHECK(diag.numel() >= 8, \"wt: diag buffer\");\n    unsigned long long* dg = reinterpret_cast<unsigned long long*>(diag.data_ptr());\n"
        "    const int key = (diag_on ? 1000000000 : 0) + (int)cpc * 100000000 + (int)abl * 10000000")
    s = s.replace("(float)gs_mul, pdl, l2pf, stream)\n", "(float)gs_mul, pdl, l2pf, dg, stream)\n")
    rep("        case 400022102: OKC_WC(false, 2, 2, 1, 2, 4); break;",
        "        case 400022102: OKC_WC(false, 2, 2, 1, 2, 4); break;\n"
        "#define OKC_WX(G, M, T, S, NS, C, AB, DG) launch_wt<G, M, T, S, NS, false, AB, true, C, DG>(Xp, (int)X.stride(0), sorted_tok.data_ptr<int>(), blk_e.data_ptr<int>(), \\\n"
        "        blk_row0.data_ptr<int>(), blk_end.data_ptr<int>(), nblk.data_ptr<int>(), blk_bm.data_ptr<int>(), Wp, Sp, gs.data_ptr<float>(), o16, obf, ldo, \\\n"
        "        pp, ldp, prow, cnt, (int)ksplit, (int)K, (int)ntiles, (int)max_blocks, (float)gs_mul, pdl, l2pf, dg, stream)\n"
        "        // phase 3b diagnostics on the locked variant pair (gated (1,2,1,2) x cpc 4, non-gated (2,2,1,2) x cpc 4)\n"
        "        case 1400112102: OKC_WX(true, 1, 2, 1, 2, 4, 0, true); break;   case 1400022102: OKC_WX(false, 2, 2, 1, 2, 4, 0, true); break;\n"
        "        case 430112102: OKC_WX(true, 1, 2, 1, 2, 4, 3, false); break;   case 430022102: OKC_WX(false, 2, 2, 1, 2, 4, 3, false); break;\n"
        "        case 440112102: OKC_WX(true, 1, 2, 1, 2, 4, 4, false); break;   case 440022102: OKC_WX(false, 2, 2, 1, 2, 4, 4, false); break;\n"
        "        case 450112102: OKC_WX(true, 1, 2, 1, 2, 4, 5, false); break;   case 450022102: OKC_WX(false, 2, 2, 1, 2, 4, 5, false); break;\n"
        "        case 460112102: OKC_WX(true, 1, 2, 1, 2, 4, 6, false); break;   case 460022102: OKC_WX(false, 2, 2, 1, 2, 4, 6, false); break;")
    rep("#undef OKC_W\n#undef OKC_WB\n#undef OKC_WC\n}", "#undef OKC_W\n#undef OKC_WB\n#undef OKC_WC\n#undef OKC_WX\n}")
    rep("                  bool pdl, bool l2pf, int64_t cpc);\nvoid moe_finalize(", "                  bool pdl, bool l2pf, int64_t cpc, torch::Tensor diag, bool diag_on);\nvoid moe_finalize(")
    s = s.replace('name="okc_moe_w4a16d"', 'name="okc_moe_w4a16e"')
cp.write_text(s)
print("cuda_moe.py updated")

sp = root / "muse_moe_w4a16/submission.py"; t = sp.read_text()


def tr(old, new, count=1):
    global t
    assert t.count(old) == count, (t.count(old), old[:90])
    t = t.replace(old, new)


if "_bench_diag" not in t:
    # every moe_w4a16_wt call gains (diag, diag_on)
    tr("                     True, 1, nt, ks, nstage, state[\"gs_mul\"], _WT_ABL, pdl, l2pf, cpc)\n",
       "                     True, 1, nt, ks, nstage, state[\"gs_mul\"], _WT_ABL, pdl, l2pf, cpc, state[\"diag\"], False)\n")
    tr("                     False, mw2, nt, ks, nstage, state[\"gs_mul\"], _WT_ABL, pdl, l2pf, cpc)\n",
       "                     False, mw2, nt, ks, nstage, state[\"gs_mul\"], _WT_ABL, pdl, l2pf, cpc, state[\"diag\"], False)\n")
    t = t.replace("state[\"gs_mul\"], _WT_ABL, False, False, 0)", "state[\"gs_mul\"], _WT_ABL, False, False, 0, state[\"diag\"], False)")
    t = t.replace("state[\"gs_mul\"], abl, False, False, 0)) - t_empty", "state[\"gs_mul\"], abl, False, False, 0, state[\"diag\"], False)) - t_empty")
    tr("            state[\"counters\"] = torch.zeros((maxb * 64,), dtype=torch.int32, device=dev)   # split-K arrival counters (self-resetting)\n",
       "            state[\"counters\"] = torch.zeros((maxb * 64,), dtype=torch.int32, device=dev)   # split-K arrival counters (self-resetting)\n"
       "            state[\"diag\"] = torch.zeros((8,), dtype=torch.int64, device=dev)   # DIAG stamps (bench)\n")
    BENCH = r'''

# == BEGIN diag bench ==
BENCH_DIAG = True   # phase 3b: DIAG stamps + ablations of the bulk wt path at bs 1 / 8 / 16


def _bench_diag(state):
    """Standalone GEMM1 / GEMM2 of the locked bulk variant (2,2,1,2,1,4) at bs 1 / 8 / 16 (judge protocol: read flush,
    one-call graph, median, fixed cost subtracted): full / loads-only / no-x / no-epilogue / no-weights ablations, then
    the DIAG stamps of CTA (0,0,0): fill (first stage landed), loop, epilogue, per-stage wait sum, and the grid span."""
    import time
    t0 = time.time()
    dev = state["w13"].device
    ext = build()
    sink = torch.empty((4096 * 4,), dtype=torch.int32, device=dev)
    flush = torch.empty((256 << 20,), dtype=torch.uint8, device=dev)
    fl = flush.numel() // 264
    fr = torch.tensor([[0, i * fl, fl - fl % 16] for i in range(264)], dtype=torch.int64, device=dev)

    def do_flush():
        ext.bstream_ranges(flush, flush, fr, sink, 2, 32768, 65536)

    def timeit(fn, reps=9):
        fn(); fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        g.replay()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            do_flush()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            g.replay()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1000.0)
        ts.sort()
        return ts[len(ts) // 2]

    tiny = torch.empty((1,), dtype=torch.int32, device=dev)
    t_empty = timeit(lambda: tiny.fill_(0))
    v = (2, 2, 1, 2, 1, 4)
    nt, mw2, ks, ns, ksp, cpc = v
    dg = state["diag"]
    print(f"BENCH diag bulk variant {v} (judge protocol, fixed cost {t_empty:.1f} us subtracted)")
    for bs in (1, 8, 16):
        x, ids, tw = _judge_inputs(bs, dev, seed=10 + bs)
        NP, MAXB = _route_wt(x, ids, 0, 8 * nt, state, ext)
        maxb = min(MAXB, _wt_max_blocks(bs, 8 * nt))
        torch.cuda.synchronize()
        lists = (state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"], state["blk_bm"])
        nloc = int(((ids >= 0) & (ids < LOCAL)).sum()); nexp = int(torch.unique(ids[(ids >= 0) & (ids < LOCAL)]).numel())
        try:
            full = _timeit_cold(lambda: _moe_forward_wt(x, ids, tw, 0, 4096, state), dev)
        except Exception as e:
            full = float("nan"); print("  full path failed:", str(e)[:80])
        print(f"  bs{bs}: {nloc} local routes, {nexp} experts, full path {full:.1f} us")
        for name, gated, W, S, gsk, K_, ntiles, xin, outp in (("gemm1", True, state["Wg13"], state["Sg13"], "gs13", 4096, 512, state["x16"][:bs], state["inter"]),
                                                             ("gemm2", False, state[f"Wg2_{mw2}"], state[f"Sg2_{mw2}"], "gs2", 4096, 256, state["inter"], state["y"])):
            rows = []
            for abl, label in ((0, "full"), (3, "loads-only"), (4, "no-x"), (5, "no-epilogue"), (6, "no-weights")):
                try:
                    tt = timeit(lambda: ext.moe_w4a16_wt(xin, *lists, W, S, state[gsk], outp, state["part"], state["counters"], ksp, K_, ntiles, maxb,
                                                         gated, 1 if gated else mw2, nt, ks, ns, state["gs_mul"], abl, False, False, cpc, dg, False)) - t_empty
                    rows.append(f"{label} {tt:5.1f}")
                except Exception as e:
                    rows.append(f"{label} fail {str(e)[:30]}")
            res = []
            try:
                for _ in range(3):
                    dg[4] = -1; dg[5] = 0
                    do_flush()
                    ext.moe_w4a16_wt(xin, *lists, W, S, state[gsk], outp, state["part"], state["counters"], ksp, K_, ntiles, maxb,
                                     gated, 1 if gated else mw2, nt, ks, ns, state["gs_mul"], 0, False, False, cpc, dg, True)
                    torch.cuda.synchronize()
                    d = dg.cpu().tolist()
                    res.append((d[1] - d[0], d[2] - d[1], d[3] - d[2], d[5] - d[4], d[6], d[7]))
                r_ = res[-1]
                diag = (f"DIAG CTA0 fill {r_[0] / 1e3:5.2f} loop {r_[1] / 1e3:5.2f} epi {r_[2] / 1e3:5.2f} | waits {r_[4] / 1e3:5.2f} of {r_[5]} stages"
                        f" | span {r_[3] / 1e3:5.2f} (runs {', '.join(f'{q[3] / 1e3:.1f}' for q in res)})")
            except Exception as e:
                diag = f"DIAG fail {str(e)[:60]}"
            print(f"    {name}: " + " | ".join(rows) + " || " + diag)
        if time.time() - t0 > BENCH_BUDGET_S:
            print("  (budget)"); break
    print(f"BENCH diag done in {time.time() - t0:.0f}s")
# == END diag bench ==
'''
    anchor = "\n\ndef prepare(weights: Weights):\n"
    assert t.count(anchor) == 1
    t = t.replace(anchor, BENCH + anchor)
    tr("    if BENCH and BENCH_STREAM and \"Wt13\" in state:\n",
       "    if BENCH and BENCH_DIAG and \"Wt13\" in state:\n        try:\n            _bench_diag(state)\n        except Exception as e:\n            print(\"BENCH diag failed:\", repr(e))\n        torch.cuda.synchronize()\n"
       "    if BENCH and BENCH_STREAM and \"Wt13\" in state:\n")
    t = t.replace("BENCH_STREAM = True   # phase 3 gate", "BENCH_STREAM = False  # phase 3 gate")
    t = t.replace("BENCH = False      # microbenchmarks in prepare", "BENCH = True       # microbenchmarks in prepare")
sp.write_text(t)
eb = root / "tools/_edit_moe_stream.py"; u = eb.read_text()
u = u.replace("state[\"gs_mul\"], abl, False, False, 0)) - t_empty", "state[\"gs_mul\"], abl, False, False, 0, state[\"diag\"], False)) - t_empty")
eb.write_text(u)
print("submission.py updated")
