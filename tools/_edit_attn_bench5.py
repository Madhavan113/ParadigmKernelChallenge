"""Bench: run the sweep in a child process (a variant's sticky CUDA error cannot poison the judge's context); the
parent restarts the child past a crashing variant and prints the child's result file."""
import re
P = "/Users/madhavanp/kerneloptimizer/.claude/worktrees/agent-aad2dcde86ab9f089/muse_attention_decode/submission.py"
s = open(P).read()

start = s.index("def _bench(data):")
end = s.index("def kernel_fn(data: Inputs) -> torch.Tensor:")
new = '''def _bench_timeit(fn, flush, sink, reps=9):
    """us per CUDA-graph replay after an L2 flush (median, best), as the judge times."""
    fn(); fn()
    torch.cuda.synchronize()
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        gph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gph, stream=st):
            fn()
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(reps):
        sink.copy_(flush.view(-1, 1 << 16).sum(1, dtype=torch.int32))   # cold L2 by reading, as the judge does
        e0.record()
        gph.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2], ts[0]


def _bench_child(start, cfg, out_path, prog_path, t_end):
    """The sweep itself (child process): variants [start:] over BENCH_SHAPES, then (index len) the plan sweeps.
    Appends result lines to out_path and the index of each finished stage to prog_path."""
    import time, hashlib
    from concurrent.futures import ThreadPoolExecutor
    dev = torch.device("cuda")
    out = open(out_path, "a")

    def say(line):
        out.write(line + "\\n"); out.flush()

    def done(k):
        with open(prog_path, "w") as f:
            f.write(str(k))

    data = {"qkv": torch.zeros((1, 1), device=dev), "k_scale": torch.ones((1,), device=dev),
            "v_scale": torch.ones((1,), device=dev), "config": cfg}
    flush = torch.zeros((160 << 20,), dtype=torch.uint8, device=dev)
    sink = torch.empty(((160 << 20) >> 16,), dtype=torch.int32, device=dev)
    nv = len(BENCH_VARIANTS)
    todo = [v for k, v in enumerate(BENCH_VARIANTS) if k >= start and v[1]]

    def build(item):
        name, flags = item[0], item[1]
        tag = "_b" + hashlib.md5(" ".join(flags).encode()).hexdigest()[:8]
        try:
            return name, _build(tag=tag, flags=("-DOKC_FLATONLY=1", *flags)), None
        except Exception as e:   # noqa: BLE001
            return name, None, str(e)[-500:]

    t0 = time.time()
    base_mod = _module()
    with ThreadPoolExecutor(max_workers=BENCH_WORKERS) as ex:
        built = {n: (m, err) for n, m, err in ex.map(build, todo)}
    say(f"BENCH builds ({len(todo)}) done in {time.time() - t0:.0f}s: " +
        ", ".join(f"{n}:{'ok' if m else 'FAIL ' + (err or '')[-200:]}" for n, (m, err) in built.items()))
    inputs = {(B, L): _bench_inputs(data, B, L) for (B, L) in BENCH_SHAPES}
    refs = {}
    for k, v in enumerate(BENCH_VARIANTS):
        if k < start:
            continue
        name, flags = v[0], v[1]
        ov_by_L = v[2] if len(v) > 2 else {}
        mod = base_mod if not flags else built.get(name, (None, None))[0]
        if mod is None:
            done(k); continue
        for (B, L) in BENCH_SHAPES:
            if time.time() > t_end:
                say("BENCH budget reached"); done(nv); return
            d = inputs[(B, L)]
            ov = ov_by_L.get(L, {})
            try:
                o = _launch(d, _mod=mod, **ov)
                torch.cuda.synchronize()
                if (B, L) not in refs:
                    refs[(B, L)] = _launch(d, _mod=base_mod).float()
                    torch.cuda.synchronize()
                ref = refs[(B, L)]
                err = float((o.float() - ref).norm() / (ref.norm() + 1e-20))
                us, best = _bench_timeit(lambda: _launch(d, _mod=mod, **ov), flush, sink)
                say(f"BENCH B{B} L{L} full {name:12s} {str(flags) + (str(ov) if ov else ''):58s} {us:8.1f} / {best:8.1f} us  relerr {err:.2e}")
            except Exception as e:   # noqa: BLE001
                say(f"BENCH B{B} L{L} full {name:12s} failed: {str(e)[-200:]}")
        done(k)
    if start <= nv:
        for (B, L, window), plans in BENCH_PLANS:
            if time.time() > t_end:
                say("BENCH budget reached"); break
            d = _bench_inputs(data, B, L, window=window)
            ref = None
            for ov in plans:
                try:
                    o = _launch(d, _mod=base_mod, **ov)
                    torch.cuda.synchronize()
                    if ref is None:
                        ref, err = o.float(), 0.0
                    else:
                        err = float((o.float() - ref).norm() / (ref.norm() + 1e-20))
                    us, best = _bench_timeit(lambda: _launch(d, _mod=base_mod, **ov), flush, sink)
                    say(f"BENCH B{B} L{L} {'sliding' if window else 'full':7s} plan {str(ov):50s} {us:8.1f} / {best:8.1f} us  relerr {err:.2e}")
                except Exception as e:   # noqa: BLE001
                    say(f"BENCH B{B} L{L} plan {ov} failed: {str(e)[-200:]}")
            del d
        done(nv + 1)
    say(f"BENCH child done in {time.time() - t0:.0f}s")


def _bench(data):
    """Parent side: the base build, then the sweep in child processes (restarted past a crashing variant)."""
    import time, json, subprocess, sys, os, tempfile
    t0 = time.time()
    _module()   # the base build (shared through the extension cache)
    cfg = {k: v for k, v in data["config"].items()}
    tmp = tempfile.mkdtemp(prefix="okc_attn_bench_")
    out_path, prog_path, log_path = os.path.join(tmp, "out.txt"), os.path.join(tmp, "prog.txt"), os.path.join(tmp, "log.txt")
    nv = len(BENCH_VARIANTS)
    t_end = t0 + BENCH_BUDGET_S
    print(f"BENCH attention: {nv} flat builds x {BENCH_SHAPES} + {len(BENCH_PLANS)} plan sweeps (us per graph replay after an L2 flush: median / best)")
    start = 0
    while start <= nv and time.time() < t_end:
        code = ("import importlib.util, sys\\n"
                f"spec = importlib.util.spec_from_file_location('okc_attn_sub', {__file__!r})\\n"
                "S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)\\n"
                f"S._bench_child({start}, {json.dumps(cfg)}, {out_path!r}, {prog_path!r}, {t_end!r})\\n")
        with open(log_path, "a") as lf:
            try:
                r = subprocess.run([sys.executable, "-c", code], stdout=lf, stderr=subprocess.STDOUT,
                                   timeout=max(10.0, t_end - time.time() + 30.0), env=dict(os.environ))
                rc = r.returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        prog = -1
        try:
            prog = int(open(prog_path).read().strip() or -1)
        except Exception:   # noqa: BLE001
            pass
        if rc == 0 and prog >= nv + 1:
            break
        nxt = max(prog + 1, start)
        if nxt <= nv:
            crashed = BENCH_VARIANTS[nxt][0] if nxt < nv else "plan sweeps"
            print(f"BENCH child exit {rc} at stage {nxt} ({crashed}): skipped")
        if rc == "timeout" or nxt > nv:
            break
        start = nxt + 1
    try:
        print(open(out_path).read())
    except Exception:   # noqa: BLE001
        pass
    try:
        tail = open(log_path).read()[-2500:]
        if tail.strip():
            print("BENCH child log tail:\\n" + tail)
    except Exception:   # noqa: BLE001
        pass
    print(f"BENCH done in {time.time() - t0:.0f}s")


'''
s = s[:start] + new + s[end:]
# the variant order: the stage-count / partition variants first, the suspects (cache hint, PVSPLIT 2) and the probes last
old = s[s.index("BENCH_VARIANTS = ["):s.index("]\n", s.index("BENCH_VARIANTS = [")) + 2]
new_vars = """BENCH_VARIANTS = [
    ("base", ()),
    ("st1f4", ("-DOKC_ST1F=4",)),
    ("st1f5", ("-DOKC_ST1F=5",)),
    ("st8f3", ("-DOKC_ST8F=3",)),
    ("segk_lo", ("-DOKC_SEGK1=200", "-DOKC_SEGK8=300")),
    ("segk_hi", ("-DOKC_SEGK1=800", "-DOKC_SEGK8=1000")),
    ("np256", ("-DOKC_NP=256",)),
    ("np64", ("-DOKC_NP=64",)),
    ("minb3", ("-DOKC_WSMINB=3",), {1: dict(n_splits=396)}),
    ("pvsplit2", ("-DOKC_PVSPLIT=2",)),
    ("cphint", ("-DOKC_CPHINT=1",)),
    ("nocompute", ("-DOKC_NOCOMPUTE=1",)),      # (probe: wrong results) the load-bound tile time
    ("xnowaitf", ("-DOKC_XNOWAITF=1",)),        # (probe: wrong results) the consumer-bound tile time
]
"""
s = s.replace(old, new_vars)
open(P, "w").write(s)
print("ok")
