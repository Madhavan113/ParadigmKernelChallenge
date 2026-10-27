"""Phase 3 gate: splice the bstream_ranges microbenchmark into muse_moe_w4a16/cuda_moe.py (markers), export it, switch
the build to -gencode arch=compute_90a,code=sm_90a, and add _bench_stream (+ BENCH_STREAM) to submission.py.
Run: python3 tools/_edit_moe_stream.py && python3 tools/embed_cuda.py"""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
cp = root / "muse_moe_w4a16/cuda_moe.py"; s = cp.read_text()
dev = (root / "tools/moe_bstream.cu.part").read_text()
host = (root / "tools/moe_bstream_host.cu.part").read_text()


def put_region(text, begin, end, body, anchor):
    if begin in text:
        a = text.index(begin); b = text.index(end) + len(end)
        return text[:a] + body.rstrip("\n") + text[b:]
    assert text.count(anchor) == 1, anchor
    i = text.index(anchor)
    return text[:i] + body.rstrip("\n") + "\n\n" + text[i:]


s = put_region(s, "// == BEGIN bstream device ==", "// == END bstream device ==", dev, "} // namespace")
s = put_region(s, "// == BEGIN bstream host ==", "// == END bstream host ==", host, '\n"""\n\nCPP_SRC = """')
decl = "void bstream_ranges(torch::Tensor b0, torch::Tensor b1, torch::Tensor ranges, torch::Tensor out, int64_t nst, int64_t sb, int64_t smem_alloc);\n"
if "void bstream_ranges(" not in s.split('CPP_SRC = """')[1]:
    s = s.replace('int64_t M, int64_t local_start, int64_t BM, int64_t BM2, double xscale);\n"""',
                  'int64_t M, int64_t local_start, int64_t BM, int64_t BM2, double xscale);\n' + decl + '"""')
if '"moe_route", "bstream_ranges"' not in s:
    s = s.replace('functions=["moe_w4a16_gemm", "moe_w4a16_wt", "moe_finalize", "moe_route"], extra_cuda_cflags=["-O3", "-arch=sm_90a", "--use_fast_math"]',
                  'functions=["moe_w4a16_gemm", "moe_w4a16_wt", "moe_finalize", "moe_route", "bstream_ranges"], extra_cuda_cflags=["-O3", "-gencode", "arch=compute_90a,code=sm_90a", "--use_fast_math"]')
    s = s.replace('name="okc_moe_w4a16"', 'name="okc_moe_w4a16b"')
cp.write_text(s)
print("cuda_moe.py updated")

sp = root / "muse_moe_w4a16/submission.py"; t = sp.read_text()
BENCH = r'''

# == BEGIN stream bench ==
BENCH_STREAM = True   # phase 3 gate: bulk-copy streaming ceiling over the active experts' WT layouts


def _bench_stream(state):
    """Loads-only streaming over exactly the active experts' Wt/St ranges (judge-like routing at bs 1 / 8): bulk-copy
    rings NST x SB per CTA at 1/2/4 CTAs per SM (each CTA a contiguous range of total / CTAs bytes), vs the current
    wt kernel's loads-only ablation and its full GEMM, judge protocol (read flush, single graph replay, median)."""
    import time
    global _WT_ABL
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
    print(f"BENCH stream (judge protocol; replay fixed cost {t_empty:.1f} us subtracted): us (TB/s)")
    combos = [(4, 16384), (2, 32768), (6, 32768)]
    for bs in (1, 8):
        x, ids, tw = _judge_inputs(bs, dev, seed=10 + bs)
        experts = sorted(set(int(v) for v in ids[(ids >= 0) & (ids < LOCAL)].flatten().tolist()))
        nt, mw2, ks, ns, ksp = _wt_variant(bs, state)
        NP, MAXB = _route_wt(x, ids, 0, 8 * nt, state, ext)
        maxb = min(MAXB, _wt_max_blocks(bs, 8 * nt))
        torch.cuda.synchronize()
        for name, wk, sk_, gated, K_, ntiles, xin, outp in (("gemm1", "Wt13", "St13", True, 4096, 512, state["x16"][:bs], state["inter"]),
                                                            ("gemm2", "Wt2", "St2", False, 4096, 256, state["inter"], state["y"])):
            W, S = state[wk], state[sk_]
            wb, sb_ = W[0].numel() * W.element_size(), S[0].numel() * S.element_size()
            total = len(experts) * (wb + sb_)
            lists = (state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"], state["blk_bm"])
            rows = []
            for abl, label in ((3, "wt loads-only"), (0, "wt full")):
                try:
                    tt = timeit(lambda: ext.moe_w4a16_wt(xin, *lists, W, S, state["gs13" if gated else "gs2"], outp, state["part"], state["counters"], ksp, K_, ntiles, maxb,
                                                         gated, 1 if gated else mw2, nt, ks, ns, state["gs_mul"], abl, False, False)) - t_empty
                    rows.append(f"{label} {tt:5.1f} ({total / tt / 1e6:.2f})")
                except Exception as e:
                    rows.append(f"{label} fail {str(e)[:40]}")
            print(f"  {name} bs{bs}: {len(experts)} experts, {total / 1e6:.1f} MB | " + " | ".join(rows))
            for cpsm in (1, 2, 4):
                n_cta = 132 * cpsm
                piece = (total // n_cta) // 16 * 16
                segs = []
                for e in experts:                                     # the concatenated [W_e | S_e] ranges, cut into pieces
                    for which, nbytes in ((0, wb), (1, sb_)):
                        off = e * nbytes
                        done = 0
                        while done < nbytes:
                            ln = min(piece, nbytes - done)
                            segs.append([which, off + done, ln])
                            done += ln
                ranges = torch.tensor(segs, dtype=torch.int64, device=dev)
                parts = []
                for nst, sb in combos:
                    need = nst * sb
                    cap = min(233472 // cpsm - 1024, 232448 - 4096)
                    if need > cap:
                        parts.append(f"{nst}x{sb // 1024}K n/a")
                        continue
                    alloc = min(max(need, 233472 // (cpsm + 1)), cap)
                    try:
                        tt = timeit(lambda: ext.bstream_ranges(W, S, ranges, sink, nst, sb, alloc)) - t_empty
                        parts.append(f"{nst}x{sb // 1024}K {tt:5.1f} ({total / tt / 1e6:.2f})")
                    except Exception as e:
                        parts.append(f"{nst}x{sb // 1024}K fail {str(e)[:30]}")
                print(f"    c{cpsm} ({len(segs)} CTAs x {piece // 1024} KB): " + " | ".join(parts))
                if time.time() - t0 > BENCH_BUDGET_S:
                    print("  (budget)"); return
    # bulk-copy weight path vs the current policy, full path (cold-L2 graph replays like _bench_wt)
    global _WT_OVERRIDE
    print("BENCH wt bulk (full path, _timeit_cold us; '!' = rel err vs Marlin > 1.2e-2): current policy vs (nt, mw2, ks, nstage, ksplit, bulk)")
    for bs in (1, 8, 16, 32):
        x, ids, tw = _judge_inputs(bs, dev, seed=10 + bs)
        rows = []
        try:
            rows.append(f"current {_timeit_cold(lambda: _moe_forward_wt(x, ids, tw, 0, 4096, state), dev):6.1f}")
        except Exception as e:
            rows.append(f"current fail {str(e)[:40]}")
        for v in ((1, 1, 1, 8, 1, 1), (1, 1, 1, 6, 1, 1), (1, 1, 1, 4, 1, 1), (1, 2, 1, 4, 1, 1), (1, 2, 1, 6, 1, 1), (2, 2, 1, 4, 1, 1), (1, 1, 1, 8, 1, 0)):
            if bs > 8 * v[0]:
                continue
            _WT_OVERRIDE = v
            try:
                ref = _moe_forward_marlin(x, ids, tw, 0, 4096, state).float()
                out = _moe_forward_wt(x, ids, tw, 0, 4096, state).float()
                err = float((out - ref).norm() / ref.norm().clamp(min=1e-12))
                tt = _timeit_cold(lambda: _moe_forward_wt(x, ids, tw, 0, 4096, state), dev)
                rows.append(f"{v} {tt:6.1f}{'' if err < 1.2e-2 else '!'}")
            except Exception as e:
                rows.append(f"{v} fail {str(e)[:40]}")
            finally:
                _WT_OVERRIDE = None
        print(f"  bs{bs:2d}: " + " | ".join(rows))
        if time.time() - t0 > BENCH_BUDGET_S:
            print("  (budget)"); break
    print(f"BENCH stream done in {time.time() - t0:.0f}s")
# == END stream bench ==
'''
if "# == BEGIN stream bench ==" in t:
    a = t.index("\n\n# == BEGIN stream bench =="); b = t.index("# == END stream bench ==\n") + len("# == END stream bench ==\n")
    t = t[:a] + BENCH + t[b:]
else:
    anchor = "\n\ndef prepare(weights: Weights):\n"
    assert t.count(anchor) == 1
    t = t.replace(anchor, BENCH + anchor)
if "_bench_stream(state)" not in t.split("def prepare(weights: Weights):")[1]:
    t = t.replace("    if BENCH and \"Wt13\" in state:\n        try:\n            _bench_wt(state, autotune=AUTOTUNE)",
                  "    if BENCH and BENCH_STREAM and \"Wt13\" in state:\n        try:\n            _bench_stream(state)\n        except Exception as e:\n            print(\"BENCH stream failed:\", repr(e))\n        torch.cuda.synchronize()\n"
                  "    if BENCH and not BENCH_STREAM and \"Wt13\" in state:\n        try:\n            _bench_wt(state, autotune=AUTOTUNE)")
t = t.replace("BENCH = False      # microbenchmarks in prepare", "BENCH = True       # microbenchmarks in prepare")
sp.write_text(t)
print("submission.py updated")
