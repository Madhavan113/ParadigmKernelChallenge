"""Round 2 edit: chunk-major weight layout (an expert's chunk c = all tiles' 1 KB blocks contiguous), programmatic
dependent launch (griddepcontrol + cudaLaunchKernelEx attribute, GEMM2 prefetches its weights before the wait),
TMA/split-K instantiations dropped from the switch (code kept)."""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent



def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:100])
    s = s.replace(old, new)


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
rep2("""AUTOTUNE = True    # pick the wt variant per shape""", """PDL_ENABLED = True   # programmatic dependent launch across the 4 launches (verified under graph capture in prepare)


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
