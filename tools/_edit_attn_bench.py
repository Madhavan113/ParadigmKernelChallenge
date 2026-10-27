"""Add the in-call bench of flat-kernel builds (OKC_BENCH) to muse_attention_decode/submission.py."""
P = "/Users/madhavanp/kerneloptimizer/.claude/worktrees/agent-aad2dcde86ab9f089/muse_attention_decode/submission.py"
s = open(P).read()
old = """def kernel_fn(data: Inputs) -> torch.Tensor:
    if not _supported(data["config"]) or not _fits(data):
        return _fallback(data)
    return _launch(data)
"""
new = '''OKC_BENCH = 0      # 1: one on-device sweep of flat-kernel builds in the first eager call (never under graph capture); prints to the log
_BENCH_DONE = False
# (name, extra nvcc flags) of the builds the sweep times on synthetic inputs shaped like the scored full-context cases
BENCH_VARIANTS = [
    ("base", ()),
]
BENCH_SHAPES = [(16, 1), (32, 1), (16, 8), (32, 8)]   # (batch, q_len), full layer, ctx 80000 agentic
BENCH_BUDGET_S = 420
BENCH_WORKERS = 4


def _bench_inputs(data, B, L, ctx=80000, seed=0):
    """Synthetic full-layer inputs like the judge's generator: scattered pages, agentic lengths."""
    dev = data["qkv"].device
    cfg0 = dict(data["config"])
    HQ, HKV, D = cfg0["num_q_heads"], cfg0["num_kv_heads"], cfg0["head_dim"]
    g = torch.Generator(device=dev).manual_seed(seed)
    longest = math.ceil(ctx * 1.075)
    max_pages = longest + L
    u = torch.rand((B,), generator=g, device=dev)
    seq = (ctx * (1 + 0.075 * (2 * u - 1))).long().clamp(1, longest)
    total = B * max_pages
    perm = torch.randperm(total, generator=g, device=dev).to(torch.int32)
    page_table = torch.full((B, max_pages), -1, dtype=torch.int32, device=dev)
    pages_per = seq + L
    offs = torch.cumsum(pages_per, 0) - pages_per
    for b in range(B):
        n, o = int(pages_per[b]), int(offs[b])
        page_table[b, :n] = perm[o:o + n]
    k_cache = torch.randn((total, 1, HKV, D), generator=g, device=dev).to(torch.float8_e4m3fn)
    v_cache = torch.randn((total, 1, HKV, D), generator=g, device=dev).to(torch.float8_e4m3fn)
    pos = (seq[:, None] + torch.arange(L, device=dev)).reshape(-1)
    loc = torch.gather(page_table.long(), 1, pos.view(B, L)).reshape(-1)
    qkv = (2 * torch.randn((B * L, (2 * HQ + 2 * HKV) * D), generator=g, device=dev)).to(torch.bfloat16)
    c = dict(cfg0, batch=B, q_len=L, page_size=1, window=0, rope_theta=0.0)
    return {"qkv": qkv, "positions": pos, "cache_loc": loc, "cache_seqlens": seq.int(), "cu_seqlens_q": None,
            "scheduler_metadata": None, "k_cache": k_cache, "v_cache": v_cache, "k_scale": data["k_scale"],
            "v_scale": data["v_scale"], "page_table": page_table, "seq_lens": seq.int(), "config": c}


def _bench(data):
    import time, hashlib
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    dev = data["qkv"].device
    flush = torch.zeros((160 << 20,), dtype=torch.uint8, device=dev)
    sink = torch.empty(((160 << 20) >> 16,), dtype=torch.int32, device=dev)

    def build(item):
        name, flags = item
        if not flags:
            return name, _module(), None
        tag = "_b" + hashlib.md5(" ".join(flags).encode()).hexdigest()[:8]
        try:
            return name, _build(tag=tag, flags=("-DOKC_FLATONLY=1", *flags)), None
        except Exception as e:   # noqa: BLE001
            return name, None, str(e)[-600:]

    def timeit(fn, reps=9):
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

    print(f"BENCH attention flat builds: {len(BENCH_VARIANTS)} builds x {BENCH_SHAPES} (us per graph replay after an L2 flush: median / best)")
    with ThreadPoolExecutor(max_workers=BENCH_WORKERS) as ex:
        mods = list(ex.map(build, BENCH_VARIANTS))
    print(f"BENCH builds done in {time.time() - t0:.0f}s: " + ", ".join(f"{n}:{'ok' if m else 'FAIL'}" for n, m, _ in mods))
    for n, m, err in mods:
        if err:
            print(f"BENCH build {n} failed: {err[-300:]}")
    flags_of = dict(BENCH_VARIANTS)
    for (B, L) in BENCH_SHAPES:
        if time.time() - t0 > BENCH_BUDGET_S:
            print("BENCH budget reached"); break
        d = _bench_inputs(data, B, L)
        ref = None
        for name, mod, _ in mods:
            if mod is None:
                continue
            if time.time() - t0 > BENCH_BUDGET_S:
                print("BENCH budget reached"); break
            try:
                out = _launch(d, _mod=mod)
                torch.cuda.synchronize()
                if ref is None:
                    ref = out.float()
                    err = 0.0
                else:
                    err = float((out.float() - ref).norm() / (ref.norm() + 1e-20))
                us, best = timeit(lambda: _launch(d, _mod=mod))
                print(f"BENCH B{B} L{L} full {name:14s} {str(flags_of[name]):60s} {us:8.1f} / {best:8.1f} us  relerr {err:.2e}")
            except Exception as e:   # noqa: BLE001
                print(f"BENCH B{B} L{L} full {name:14s} failed: {str(e)[-200:]}")
        del d
    # probe builds may leave garbage in the shared workspace: back to the all-zero state
    for t in _WS.get(str(dev), ()):
        t.zero_()
    torch.cuda.synchronize()
    print(f"BENCH done in {time.time() - t0:.0f}s")


def kernel_fn(data: Inputs) -> torch.Tensor:
    global _BENCH_DONE
    if OKC_BENCH and not _BENCH_DONE and torch.cuda.is_available() and not torch.cuda.is_current_stream_capturing():
        _BENCH_DONE = True
        try:
            _bench(data)
        except Exception as e:   # noqa: BLE001
            print("BENCH failed:", repr(e)[-400:])
    if not _supported(data["config"]) or not _fits(data):
        return _fallback(data)
    return _launch(data)
'''
assert s.count(old) == 1
s = s.replace(old, new)
open(P, "w").write(s)
print("ok")
