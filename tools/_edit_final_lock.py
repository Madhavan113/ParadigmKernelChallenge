"""One-off edit for the final submission state: bench off, bench-only ablation instantiations dropped (ABL stays as a
template parameter at 0), feature defaults from the arguments: python3 _edit_final_lock.py <mixed_min_tokens> <l2pf_max_tokens>."""
import pathlib, re, sys
root = pathlib.Path(__file__).resolve().parent.parent
mixed_min = int(sys.argv[1]) if len(sys.argv) > 1 else 65
l2pf_max = int(sys.argv[2]) if len(sys.argv) > 2 else 16

p = root / "muse_moe_w4a16/cuda_moe.py"
s = p.read_text()
a = s.index("        // bench ablations (abl 1 = no mma, 2 = no dequant, 3 = loads only) of the scored variants\n")
b = s.index('        default: TORCH_CHECK(false, "unsupported wt variant (mw, nt, ks, nstage, ksplit)");')
s = s[:a] + "        // bench ablations (abl 1 = no mma, 2 = no dequant, 3 = loads only) are not instantiated in the submitted state\n" + s[b:]
p.write_text(s)

q = root / "muse_moe_w4a16/submission.py"
t = q.read_text()


def rep(old, new, n=1):
    global t
    assert t.count(old) == n, (t.count(old), old[:90])
    t = t.replace(old, new)


rep("BENCH = True      # microbenchmarks in prepare", "BENCH = False      # microbenchmarks in prepare")
t = re.sub(r"^MIXED_MIN_TOKENS = \d+$", f"MIXED_MIN_TOKENS = {mixed_min}", t, count=1, flags=re.M)
t = re.sub(r"^L2PF_MAX_TOKENS = \d+(\s*#.*)?$", f"L2PF_MAX_TOKENS = {l2pf_max}   # up to this many tokens GEMM2 prefetches its W2 slice into L2 while GEMM1 runs (PDL)", t, count=1, flags=re.M)
q.write_text(t)
print(f"final lock: BENCH off, MIXED_MIN_TOKENS={mixed_min}, L2PF_MAX_TOKENS={l2pf_max}")
