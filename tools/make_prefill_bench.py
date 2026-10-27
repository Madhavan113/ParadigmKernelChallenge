"""Build a muse_prefill_layer check file that benchmarks the proj kernels in its prepare() (an untimed 300 s H200
window): main's prefill submission + this worktree's muse_bf16_proj/submission.py inlined with every module-level name
prefixed `bfp_` (compose_decode_layer's prefix_module), plus a prepare() wrapper that, after the prefill's own
prepare, runs the proj prepare (extension build, FP64 self-test, cold-L2 autotune: its `[okc] tune ...` lines are the
measurement) within the remaining time budget. kernel_fn is untouched, so the prefill tests still pass.
Usage: python3 tools/make_prefill_bench.py [-o muse_prefill_layer/bench_proj.py]"""
import pathlib, sys, importlib.util
root = pathlib.Path(__file__).resolve().parent.parent
main_prefill = pathlib.Path("/Users/madhavanp/kerneloptimizer/muse_prefill_layer/submission.py")
compose = pathlib.Path("/tmp/dl/tools/compose_decode_layer.py")
out = pathlib.Path(sys.argv[sys.argv.index("-o") + 1]) if "-o" in sys.argv else root / "muse_prefill_layer/bench_proj.py"
spec = importlib.util.spec_from_file_location("compose_tool", compose)
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)

pre = main_prefill.read_text()
assert pre.startswith("# okc: problem=muse_prefill_layer\n"), pre[:60]
proj = C.strip_first_line((root / "muse_bf16_proj/submission.py").read_text())
body, names = C.prefix_module(proj, "bfp_")
bench = r'''

# ======================================================================================================================
# proj bench (check runs only): after the prefill's own prepare(), the proj kernels' prepare (extension build, FP64
# self-test, cold-L2 graph-replay autotune) runs on the same H200 within the remaining untimed budget; its `[okc] ...`
# lines in the log are the measurement. kernel_fn above is untouched.
# ======================================================================================================================
import sys
import time as _bench_time
_prepare_prefill = prepare
PROJ_BENCH_MIN_BUDGET = 110   # seconds the proj prepare needs (build ~50 s, repack, self-test, autotune) with margin


def prepare(weights):
    t0 = _bench_time.time()
    st = _prepare_prefill(weights)
    torch.cuda.synchronize()
    try:
        _proj_bench(weights, 290.0 - (_bench_time.time() - t0))
    except Exception as exc:  # noqa: BLE001 - the bench never fails the run
        print("[okc] proj bench failed:", repr(exc)[:600], file=sys.stderr, flush=True)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return st


def _proj_bench(weights, budget):
    print(f"[okc] proj bench: {budget:.0f} s left after the prefill prepare", file=sys.stderr, flush=True)
    if budget < PROJ_BENCH_MIN_BUDGET:
        print("[okc] proj bench: skipped (budget)", file=sys.stderr, flush=True)
        return
    global bfp_CUTLASS_LAYERS
    bfp_CUTLASS_LAYERS = set()   # no CUTLASS build in the bench slot (170 s join); the proj check measures it
    w5 = {k: weights[k].contiguous() for k in ("qkv_gate", "o", "pooler", "unpooler")}
    w5["router"] = weights["router"].float().contiguous()
    t1 = _bench_time.time()
    state = bfp_prepare(w5)
    torch.cuda.synchronize()
    print(f"[okc] proj bench: prepare took {_bench_time.time() - t1:.1f} s; choices: "
          + " ".join(f"{l}/{m}={bfp__cand_name(c)}" for (l, m), c in sorted(state.get("choice", {}).items())),
          file=sys.stderr, flush=True)
    del state
'''
src = pre.rstrip("\n") + "\n\n# ---- component: proj (muse_bf16_proj/submission.py, names prefixed bfp_) ----\n" + body.rstrip("\n") + "\n" + bench
out.write_text(src)
compile(src, str(out), "exec")
print(f"wrote {out}: {src.count(chr(10))} lines, {len(src) // 1024} KB, {len(names)} prefixed names")
