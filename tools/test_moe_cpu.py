"""Run muse_moe_w4a16/submission.py under Triton's CPU interpreter against reference.py.
Weights are generated for the 32 local experts only (RAM). Usage: TRITON_INTERPRET=1 python test_moe_cpu.py [fast]"""
import os, sys, time
assert os.environ.get("TRITON_INTERPRET") == "1"
sys.path.insert(0, "/work/muse_moe_w4a16")
import torch
import reference as R
from nvfp4 import random_weights
import submission as S
S.BENCH = False

S.DEQUANT_ASM = False   # the interpreter can't run inline PTX: use the plain-Triton placement
fast = len(sys.argv) > 1 and sys.argv[1] == "fast"
dev = "cpu"
gen = torch.Generator(device=dev).manual_seed(R.WEIGHTS_SEED)
E = 32
w13, w13_sf, gs13 = random_weights(2 * R.D_EXPERT, R.D_HIDDEN, generator=gen, device=dev, leading=(E,))
w2, w2_sf, gs2 = random_weights(R.D_HIDDEN, R.D_EXPERT, generator=gen, device=dev, leading=(E,))
weights = {"w13": w13, "w13_sf": w13_sf, "w13_gs": gs13.expand(E).contiguous(), "w2": w2, "w2_sf": w2_sf,
           "w2_gs": gs2.expand(E).contiguous(),
           "post_norm_w": (0.1 * torch.randn((R.D_HIDDEN,), generator=gen, device=dev)).to(torch.bfloat16)}
S.MAX_M = 64
state = S.prepare(weights)
# bf16 -> fp16 containers for the interpreter (exact), fp8 scales -> fp16 (exact)
state["norm_w"] = state["norm_w"].float()
for k in ("w13_sf", "w2_sf"):
    state[k] = state[k].to(torch.float16)
state["y"] = state["y"].to(torch.float16)

cases = [(1, "normal"), (4, "normal"), (9, "skewed")]
if not fast:
    cases += [(33, "normal"), (17, "skewed")]
for bs, dist in cases:
    data = R.generate_input(bs, "ep8", seed=77 + bs, dist=dist, device=dev)
    run = dict(data, hidden_states=data["hidden_states"].to(torch.float16))
    t = time.time()
    out = S.kernel_fn(run, state).to(torch.bfloat16)
    ok, msg = R.check_implementation(data, out, weights)
    nl = int((data["topk_ids"] < 32).sum())
    print(f"bs={bs:3d} {dist:7s} local pairs={nl:4d} {time.time()-t:6.1f}s -> {ok} {msg}", flush=True)
print("done")
