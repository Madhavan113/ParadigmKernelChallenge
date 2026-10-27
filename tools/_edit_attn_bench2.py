"""Bench extension: sliding-window synthetic inputs and launch-plan sweeps on the base build (BENCH_PLANS)."""
P = "/Users/madhavanp/kerneloptimizer/.claude/worktrees/agent-aad2dcde86ab9f089/muse_attention_decode/submission.py"
s = open(P).read()


def sub1(old, new):
    global s
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)


sub1('''def _bench_inputs(data, B, L, ctx=80000, seed=0):
    """Synthetic full-layer inputs like the judge's generator: scattered pages, agentic lengths."""
''', '''def _bench_inputs(data, B, L, ctx=80000, seed=0, window=0):
    """Synthetic inputs like the judge's generator: scattered pages, agentic lengths (fixed at batch 1)."""
''')
sub1('''    u = torch.rand((B,), generator=g, device=dev)
    seq = (ctx * (1 + 0.075 * (2 * u - 1))).long().clamp(1, longest)
''', '''    u = torch.rand((B,), generator=g, device=dev)
    seq = (ctx * (1 + 0.075 * (2 * u - 1))).long().clamp(1, longest)
    if B == 1:
        longest = ctx
        max_pages = ctx + L
        seq = torch.full((1,), ctx, dtype=torch.long, device=dev)
''')
sub1('''    c = dict(cfg0, batch=B, q_len=L, page_size=1, window=0, rope_theta=0.0)
''', '''    c = dict(cfg0, batch=B, q_len=L, page_size=1, window=window, rope_theta=500000.0 if window else 0.0)
''')
sub1('''BENCH_SHAPES = [(16, 1), (32, 1), (16, 8), (32, 8)]   # (batch, q_len), full layer, ctx 80000 agentic
''', '''BENCH_SHAPES = [(16, 1), (32, 1), (16, 8), (32, 8)]   # (batch, q_len), full layer, ctx 80000 agentic
# launch-plan sweeps on the base build: ((batch, q_len, window), [override dicts]) ({} = today's plan)
BENCH_PLANS = []
''')
sub1('''    # probe builds may leave garbage in the shared workspace: back to the all-zero state
''', '''    base_mod = _module()
    for (B, L, window), plans in BENCH_PLANS:
        if time.time() - t0 > BENCH_BUDGET_S:
            print("BENCH budget reached"); break
        d = _bench_inputs(data, B, L, window=window)
        ref = None
        for ov in plans:
            try:
                out = _launch(d, _mod=base_mod, **ov)
                torch.cuda.synchronize()
                if ref is None:
                    ref = out.float()
                    err = 0.0
                else:
                    err = float((out.float() - ref).norm() / (ref.norm() + 1e-20))
                us, best = timeit(lambda: _launch(d, _mod=base_mod, **ov))
                print(f"BENCH B{B} L{L} {'sliding' if window else 'full':7s} plan {str(ov):50s} {us:8.1f} / {best:8.1f} us  relerr {err:.2e}")
            except Exception as e:   # noqa: BLE001
                print(f"BENCH B{B} L{L} plan {ov} failed: {str(e)[-200:]}")
        del d
    # probe builds may leave garbage in the shared workspace: back to the all-zero state
''')
open(P, "w").write(s)
print("ok")
