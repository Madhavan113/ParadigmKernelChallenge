"""Run muse_bf16_proj/submission.py under Triton's CPU interpreter and compare to reference.py.

Usage (inside the judge image, no GPU):  TRITON_INTERPRET=1 python test_proj_cpu.py [fast]
"""
import os, sys, time, math
assert os.environ.get("TRITON_INTERPRET") == "1"
sys.path.insert(0, "/work/muse_bf16_proj")
import torch
import reference as R
import submission as S

fast = len(sys.argv) > 1 and sys.argv[1] == "fast"
dev = "cpu"
torch.manual_seed(0)
# Smaller K/N than the real problem would change the configs' divisibility, so keep real sizes
# but limit the cases.
weights = R.generate_weights(R.WEIGHTS_SEED, device=dev)
state = S.prepare(weights, warm=False)

cases = [(1, "router"), (8, "router"), (9, "router"), (1, "o"), (5, "unpooler"), (33, "o")]
if not fast:
    cases += [(1, "qkv_gate"), (7, "pooler"), (16, "unpooler"), (130, "unpooler"), (128, "router"), (32, "pooler")]
for m, layer in cases:
    for dist in (("normal", "outliers") if layer == "router" else ("normal",)):
        data = R.generate_input(m, layer, seed=123 + m, dist=dist, device=dev)
        t = time.time()
        out = S.kernel_fn(data, state)
        ok, msg = R.check_implementation(data, out, weights)
        print(f"m={m:4d} {layer:9s} {dist:8s} {time.time()-t:6.1f}s -> {ok} {msg}", flush=True)
        # run twice: the ticket counters must have been reset
        out2 = S.kernel_fn(data, state)
        assert torch.equal(out, out2), "second call differs: counters not reset?"
        assert int(state["counters"].abs().sum()) == 0, "counters not zero after the call"
print("done")
