"""Bench: variants may carry a per-q_len launch-plan override ((name, flags) or (name, flags, {L: override}))."""
P = "/Users/madhavanp/kerneloptimizer/.claude/worktrees/agent-aad2dcde86ab9f089/muse_attention_decode/submission.py"
s = open(P).read()


def sub1(old, new):
    global s
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)


sub1("""    def build(item):
        name, flags = item
""", """    def build(item):
        name, flags = item[0], item[1]
""")
sub1("""    flags_of = dict(BENCH_VARIANTS)
""", """    flags_of = {v[0]: v[1] for v in BENCH_VARIANTS}
    ov_of = {v[0]: (v[2] if len(v) > 2 else {}) for v in BENCH_VARIANTS}
""")
sub1("""            try:
                out = _launch(d, _mod=mod)
                torch.cuda.synchronize()
                if ref is None:
                    ref = out.float()
                    err = 0.0
                else:
                    err = float((out.float() - ref).norm() / (ref.norm() + 1e-20))
                us, best = timeit(lambda: _launch(d, _mod=mod))
                print(f"BENCH B{B} L{L} full {name:14s} {str(flags_of[name]):60s} {us:8.1f} / {best:8.1f} us  relerr {err:.2e}")
""", """            ov = ov_of[name].get(L, {})
            try:
                out = _launch(d, _mod=mod, **ov)
                torch.cuda.synchronize()
                if ref is None:
                    ref = out.float()
                    err = 0.0
                else:
                    err = float((out.float() - ref).norm() / (ref.norm() + 1e-20))
                us, best = timeit(lambda: _launch(d, _mod=mod, **ov))
                print(f"BENCH B{B} L{L} full {name:14s} {str(flags_of[name]) + str(ov or ''):60s} {us:8.1f} / {best:8.1f} us  relerr {err:.2e}")
""")
open(P, "w").write(s)
print("ok")
