"""Run muse_attention_decode/submission.py under Triton's CPU interpreter against reference.py.

Usage (inside the judge image, no GPU):  TRITON_INTERPRET=1 python test_attn_cpu.py [fast]
"""
import os, sys, time
assert os.environ.get("TRITON_INTERPRET") == "1"
sys.path.insert(0, "/work/muse_attention_decode")
import torch
import reference as R
R._fa3_scheduler_metadata = lambda *a, **k: None   # FA3 isn't available on CPU
import submission as S
S.OUT_DTYPE = torch.float16   # numpy has no bfloat16
import triton.language as tl
S.ACT_DTYPE = tl.float16
S.FP8_ASM = False

fast = len(sys.argv) > 1 and sys.argv[1] == "fast"
dev = "cpu"
cases = [  # batch, ctx, heads, layer, q_len, spread, dist, page, kv_scale
    (1, 10, "tp8", "sliding", 1, "fixed", "normal", 64, 1.0),
    (2, 300, "tp8", "full", 1, "fixed", "normal", 1, 1.0),
    (2, 2100, "tp8", "sliding", 8, "mixed", "edges", 1, 1.0),
    (2, 500, "tp8", "full", 8, "mixed", "normal", 64, 0.015625),
]
if not fast:
    cases += [
        (3, 2047, "tp8", "sliding", 1, "fixed", "normal", 64, 1.0),
        (2, 3000, "tp8", "sliding", 8, "fixed", "soft", 1, 0.015625),
        (1, 4000, "tp8", "full", 1, "fixed", "coherent", 1, 1.0),
        (2, 1500, "dp", "full", 8, "mixed", "normal", 64, 1.0),
    ]
for i, (batch, ctx, heads, layer, q_len, spread, dist, page, kv_scale) in enumerate(cases):
    data = R.generate_input(batch, ctx, heads, layer, seed=1000 + i, q_len=q_len, spread=spread,
                            dist=dist, page=page, kv_scale=kv_scale, device=dev)
    t = time.time()
    # The interpreter's FP8 cast truncates and it can't hold BF16: run the kernel on float16
    # views (exact for the existing FP8 entries), then let torch round the new entries to E4M3.
    fp8 = {k: data[k] for k in ("k_cache", "v_cache")}
    run = dict(data, qkv=data["qkv"].to(torch.float16),
               k_cache=fp8["k_cache"].float().half(), v_cache=fp8["v_cache"].float().half())
    out = S.kernel_fn(run).to(torch.bfloat16)
    fp8["k_cache"].copy_(run["k_cache"].to(R.FP8))
    fp8["v_cache"].copy_(run["v_cache"].to(R.FP8))
    raw = R.collect(data, out)
    ok, msg = R.check_implementation(data, raw)
    print(f"b={batch} ctx={ctx} {heads} {layer:7s} L={q_len} {spread:6s} {dist:8s} page={page:2d} "
          f"kv={kv_scale:<8g} {time.time()-t:6.1f}s -> {ok} {msg}", flush=True)
    cnt = [v for k, v in S._WS.items()]
    for _, _, c in cnt:
        assert int(c.abs().sum()) == 0, "tickets not reset"
print("done")
