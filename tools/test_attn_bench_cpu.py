"""CPU smoke test of the attention submission's bench input generator (no GPU needed)."""
import sys, importlib.util, os
try:
    import torch
except Exception as e:   # noqa: BLE001
    print("no torch:", e); sys.exit(0)
print("torch", torch.__version__, "cpu count", os.cpu_count())
spec = importlib.util.spec_from_file_location(
    "sub", "/Users/madhavanp/kerneloptimizer/.claude/worktrees/agent-aad2dcde86ab9f089/muse_attention_decode/submission.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
data = {"qkv": torch.zeros(1, 1, 2560, dtype=torch.bfloat16), "k_scale": torch.ones(1), "v_scale": torch.ones(1),
        "config": dict(batch=1, q_len=1, num_q_heads=16, num_kv_heads=2, head_dim=64, page_size=1, window=0, rope_theta=0.0,
                       qk_scale_factor=5.473, sm_scale=0.125, kv_scale=1.0)}
d = S._bench_inputs(data, 4, 8, ctx=1000)
print({k: (tuple(v.shape), v.dtype) for k, v in d.items() if hasattr(v, "shape")})
print(d["seq_lens"], d["page_table"].shape, d["cache_loc"][:4], d["config"])
print("variants", S.BENCH_VARIANTS, "bench", S.OKC_BENCH)
