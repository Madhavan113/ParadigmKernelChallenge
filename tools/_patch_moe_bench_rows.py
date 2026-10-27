"""Strip the mbarrier helpers from tools/moe_bstream.cu.part (they now live after cp_async_wait in cuda_moe.py, ahead of
the wt kernel) and add bulk-vs-current full-path rows to the stream bench text in tools/_edit_moe_stream.py.
Run once: python3 tools/_patch_moe_bench_rows.py"""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
p = root / "tools/moe_bstream.cu.part"; s = p.read_text()
if "mbar_init(" in s.split("template <int NST, int SB>")[0]:
    a = s.index("__device__ __forceinline__ void mbar_init("); b = s.index("template <int NST, int SB>")
    p.write_text(s[:a] + s[b:]); print("helpers stripped from the bstream part")
pb = root / "tools/_edit_moe_stream.py"; t = pb.read_text()
old = '''    print(f"BENCH stream done in {time.time() - t0:.0f}s")'''
new = '''    # bulk-copy weight path vs the current policy, full path (cold-L2 graph replays like _bench_wt)
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
    print(f"BENCH stream done in {time.time() - t0:.0f}s")'''
if "BENCH wt bulk" not in t:
    assert t.count(old) == 1
    t = t.replace(old, new)
    t = t.replace("combos = [(4, 16384), (6, 16384), (8, 16384), (12, 16384), (2, 32768), (3, 32768), (4, 32768), (6, 32768), (8, 8192)]",
                  "combos = [(4, 16384), (2, 32768), (6, 32768)]")
    pb.write_text(t); print("bench rows added")
