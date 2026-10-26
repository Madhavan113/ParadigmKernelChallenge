"""Run muse_decode_layer/submission.py under Triton's CPU interpreter against reference.py.
Usage: TRITON_INTERPRET=1 python test_decode_cpu.py [fast]"""
import os, sys, time
assert os.environ.get("TRITON_INTERPRET") == "1"
sys.path.insert(0, "/work/muse_decode_layer")
import torch
import triton.language as tl
import reference as R
R._fa3_scheduler_metadata = lambda *a, **k: None
import submission as S

# interpreter: no bfloat16 / fp8 math, no inline PTX
S.ACT_TORCH = torch.float16
S.OUT_DTYPE = torch.float16
S.ACT_DTYPE = tl.float16
S.FP8_ASM = False
S.DEQUANT_ASM = False
S.ROW_SPLIT = False
S.MOE_BACKEND = "triton"
S.GEMV_SHAPES = {}; S.MMA_SHAPES = {}; S.BMM_SHAPES = {}   # CUDA-only proj paths
S.MOE_OUT_DTYPE = torch.float16
S.MAX_M = 64
fast = len(sys.argv) > 1 and sys.argv[1] == "fast"
dev = "cpu"
weights = R.generate_weights(R.WEIGHTS_SEED, device=dev)
w16 = {k: (v.to(torch.float16) if v.dtype == torch.bfloat16 else v) for k, v in weights.items()}
state = S.prepare(w16)
for k, v in list(state.items()):
    if isinstance(v, torch.Tensor) and v.dtype in (torch.bfloat16, torch.float8_e4m3fn):
        state[k] = v.to(torch.float16)

cases = [  # batch, ctx, layer, q_len, page, kv_scale, dist
    (1, 300, "full", 1, 1, 1.0, "normal"),
    (2, 400, "sliding", 8, 64, 1.0, "normal"),
]
if not fast:
    cases += [(2, 2100, "sliding", 1, 1, 0.015625, "outliers"), (2, 350, "full", 8, 1, 1.0, "soft")]
for i, (batch, ctx, layer, q_len, page, kv_scale, dist) in enumerate(cases):
    case = dict(batch=batch, ctx=ctx, layer=layer, seed=4242 + i, q_len=q_len, page=page, kv_scale=kv_scale, dist=dist, device=dev)
    ls = R.case_fields(case, weights)["local_start"]
    data = R.generate_input(**case, local_start=ls)
    run = dict(data, attn_input=data["attn_input"].to(torch.float16),
               k_cache=data["k_cache"].float().half(), v_cache=data["v_cache"].float().half())
    t = time.time()
    hidden, stream, ids = S.kernel_fn(run, state)
    data["k_cache"].copy_(run["k_cache"].to(R.FP8)); data["v_cache"].copy_(run["v_cache"].to(R.FP8))
    raw = R.collect(data, (hidden.to(torch.bfloat16), stream, ids))
    ok, msg = R.check_implementation(data, raw, weights)
    print(f"b={batch} ctx={ctx} {layer:7s} L={q_len} page={page:2d} kv={kv_scale:<8g} {dist:8s} ls={ls:3d} "
          f"{time.time()-t:6.1f}s -> {ok} {msg}", flush=True)
print("done")
