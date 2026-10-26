"""Interpreter test of the override paths the in-prepare bench uses (TRITON_INTERPRET=1, no GPU):
two-m-tile swap grid, 128-row tiled wgmma configs, the router's 128-row two-weight configs, the
extra tile packings and the partial-sum kernel."""
import os, sys, time
assert os.environ.get("TRITON_INTERPRET") == "1"
sys.path.insert(0, "/work/muse_bf16_proj")
import torch
import reference as R
import submission as S
S.BENCH = False
dev = "cpu"
torch.manual_seed(0)
weights = R.generate_weights(R.WEIGHTS_SEED, device=dev)
state = S.prepare(weights, warm=False)
for k in list(state):
    if isinstance(state[k], torch.Tensor) and state[k].dtype == torch.bfloat16:
        state[k] = state[k].to(torch.float16)
ref_w = {k: state[k].float() for k in ("qkv_gate", "o", "pooler", "unpooler")}
ref_w["router"] = state["router_hi"].float() + state["router_lo"].float()


def run(layer, m, cfg, n=None):
    N, K = R.LAYERS[layer]
    n = n or N
    x = R.generate_input(m, layer, seed=7 + m, device=dev)["x"].to(torch.float16)
    S._OVERRIDE = cfg
    t = time.time()
    try:
        assert S._usable(cfg, m, n, K), "not usable"
        out = S._proj_forward(x, layer, n, state)
        out2 = S._proj_forward(x, layer, n, state)
    finally:
        S._OVERRIDE = None
    ref = x.float() @ ref_w[layer][:n].T
    err = float((out.float() - ref).norm() / ref.norm())
    ok = err < (1e-3 if layer == "router" else 1e-2)
    assert torch.equal(out, out2), "second call differs"
    assert int(state["counters"].abs().sum()) == 0, "counters not reset"
    print(f"{layer:9s} m={m:4d} {str(cfg)[:110]:110s} {time.time() - t:5.1f}s -> {ok} rel L2 {err:.2e}", flush=True)
    assert ok


T = dict(tiled=True, tma=True)
# m=32 as two 16-row tiles, m-fastest grid, (32,128) packing
run("pooler", 32, dict(BLOCK_M=16, BLOCK_N=32, BLOCK_K=128, SPLIT_K=1, num_warps=4, num_stages=8, swap_mn=True, **T))
run("qkv_gate", 32, dict(BLOCK_M=16, BLOCK_N=32, BLOCK_K=128, SPLIT_K=4, num_warps=4, num_stages=8, swap_mn=True, **T))
run("qkv_gate", 33, dict(BLOCK_M=32, BLOCK_N=64, BLOCK_K=128, SPLIT_K=2, num_warps=4, num_stages=6, swap_mn=False, **T))
run("o", 16, dict(BLOCK_M=16, BLOCK_N=32, BLOCK_K=128, SPLIT_K=1, num_warps=4, num_stages=8, tiled=True, tma=False))
# m=128/256: tiled (128,64) wgmma split-K, swap and plain; row-major TMA
run("pooler", 128, dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, SPLIT_K=4, num_warps=8, num_stages=6, big=True, **T))
run("unpooler", 256, dict(BLOCK_M=128, BLOCK_N=64, BLOCK_K=64, SPLIT_K=4, num_warps=4, num_stages=8, swap_mn=True, big=True, **T))
run("o", 130, dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, SPLIT_K=4, num_warps=8, num_stages=6, big=True, tma=True))
# router at 128/256 tokens: two-weight 128-row configs, tiled and row-major
run("router", 128, dict(BLOCK_M=128, BLOCK_N=32, BLOCK_K=64, SPLIT_K=16, num_warps=8, num_stages=4, tma=True))
run("router", 256, dict(BLOCK_M=128, BLOCK_N=32, BLOCK_K=64, SPLIT_K=16, num_warps=8, num_stages=6, **T))
run("router", 9, dict(BLOCK_M=16, BLOCK_N=32, BLOCK_K=64, SPLIT_K=16, num_warps=4, num_stages=8, **T))
# partial-sum kernel (the bmm path's reduction; bmm itself has no CPU out_dtype kernel)
part = torch.randn((4, 128, 4096), dtype=torch.float32)
out = torch.empty((128, 4096), dtype=torch.float16)
S._sum_partials(part, out)
err = float((out.float() - part.sum(0)).norm() / part.sum(0).norm())
print(f"sum_partials rel L2 {err:.2e}"); assert err < 1e-3
print("done")
