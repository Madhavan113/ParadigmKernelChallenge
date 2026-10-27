"""Bench: the launch-plan sweeps of check 2 (sliding and batch-1 shapes on the base build)."""
P = "/Users/madhavanp/kerneloptimizer/.claude/worktrees/agent-aad2dcde86ab9f089/muse_attention_decode/submission.py"
s = open(P).read()
old = "BENCH_PLANS = []\n"
new = """W2K = 2048
BENCH_PLANS = [
    ((16, 1, W2K), [{}, dict(n_splits=8), dict(n_splits=12), dict(n_splits=24), dict(n_splits=33)]),
    ((32, 1, W2K), [{}, dict(n_splits=4), dict(n_splits=6), dict(n_splits=12), dict(n_splits=16)]),
    ((16, 8, W2K), [{}, dict(n_splits=4, coop=1), dict(n_splits=6, coop=1), dict(n_splits=12, coop=1), dict(n_splits=8, coop=0)]),
    ((32, 8, W2K), [{}, dict(n_splits=4, coop=0), dict(n_splits=6, coop=1), dict(n_splits=8, coop=1)]),
    ((1, 1, W2K), [{}, dict(n_splits=16), dict(n_splits=17), dict(n_splits=33, coop=1)]),
    ((1, 8, W2K), [{}, dict(n_splits=17, coop=1), dict(n_splits=33, coop=0)]),
    ((1, 1, 0), [{}, dict(n_splits=264, nc=8, coop=1), dict(n_splits=252, nc=4, coop=1), dict(n_splits=252, nc=8, coop=0)]),
    ((1, 8, 0), [{}, dict(n_splits=264, nc=8, coop=1), dict(n_splits=252, nc=4, coop=1), dict(n_splits=132, nc=8, coop=1)]),
]
"""
assert s.count(old) == 1
s = s.replace(old, new)
open(P, "w").write(s)
print("ok")
