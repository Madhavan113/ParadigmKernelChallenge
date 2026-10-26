"""Interpreter test of the Triton 12-bit tiled GEMM (plain decode, escapes applied in the epilogue):
pooler / unpooler / qkv / o at several m and split-K, against the fp32 product with the ORIGINAL
weights (so the escapes must be corrected to pass at 1e-2, and probe-like exactness is checked on
escape columns). TRITON_INTERPRET=1 python3 test_t12_cpu.py"""
import os, sys, time
assert os.environ.get("TRITON_INTERPRET") == "1"
sys.path.insert(0, "/work/muse_bf16_proj")
import torch
import reference as R
import submission as S
S.BENCH = False
S.T12_ASM = False
S.T12_INTERP = True
S.GEMV_SHAPES = {}; S.MMA_SHAPES = {}; S.BMM_SHAPES = {}
dev = "cpu"
weights = R.generate_weights(R.WEIGHTS_SEED, device=dev)
state = S.prepare(weights, warm=False)
# the interpreter cannot run bf16: fp16 views for x / W / packed planes are not meaningful for the
# 12-bit decode, so this test keeps bf16 weights but feeds x as fp16 and compares in fp32
for k in ("qkv_gate", "o", "pooler", "unpooler"):
    pk = state[k + "_p12"]
    assert pk is not None


def run(layer, m, sk, nw=4, ns=3):
    N, K = R.LAYERS[layer]
    n = 1024 if (layer == "unpooler" and m <= 8) else N
    x = R.generate_input(m, layer, seed=11 + m, device=dev)["x"].to(torch.float16)
    cfg = dict(t12=True, BLOCK_M=S._block_m(m), BLOCK_N=32, BLOCK_K=256, SPLIT_K=sk, num_warps=nw, num_stages=ns)
    assert S._usable(cfg, m, n, K)
    S._OVERRIDE = cfg
    t = time.time()
    try:
        out = S._proj_forward(x, layer, n, state)
        out2 = S._proj_forward(x, layer, n, state)
    finally:
        S._OVERRIDE = None
    assert torch.equal(out, out2) and int(state["counters"].abs().sum()) == 0
    ref = x.float() @ state[layer][:n].float().T
    err = float((out.float() - ref).norm() / ref.norm())
    # escape columns: rows of x with a single nonzero (probe rows) hit weight columns exactly
    rows = R.probe_rows(x)
    perr = 0.0
    if len(rows):
        d = (out[rows].float() - ref[rows]).norm(dim=-1) / ref[rows].norm(dim=-1).clamp(min=1e-30)
        perr = float(d.max())
    print(f"{layer:9s} m={m:3d} sk={sk} {time.time() - t:5.1f}s -> rel L2 {err:.2e} probe {perr:.1e}", flush=True)
    assert err < 1e-2 and perr < 1e-3


run("pooler", 8, 1)
run("unpooler", 16, 1)
run("qkv_gate", 9, 4)
run("o", 33, 1)
run("unpooler", 5, 2)
run("pooler", 32, 2)
print("t12 ok")
