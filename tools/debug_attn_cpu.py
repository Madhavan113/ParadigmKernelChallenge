"""Run one case under the interpreter and dump the merge workspace (to chase NaNs).

TRITON_INTERPRET=1 python3 debug_attn_cpu.py [hp]
"""
import os, sys
assert os.environ.get("TRITON_INTERPRET") == "1"
sys.path.insert(0, "/work/muse_attention_decode")
import torch
import reference as R
R._fa3_scheduler_metadata = lambda *a, **k: None
import submission as S
S.OUT_DTYPE = torch.float16
import triton.language as tl
S.ACT_DTYPE = tl.float16
S.FP8_ASM = False
S.ATTN_BENCH = False
hp = int(sys.argv[1]) if len(sys.argv) > 1 else 2
S._PLAN_OVERRIDE = dict(hp=hp)
torch.set_printoptions(linewidth=200, precision=3, sci_mode=False)

data = R.generate_input(1, 10, "tp8", "sliding", seed=1001, q_len=1, spread="fixed", dist="normal", page=64, kv_scale=1.0, device="cpu")
run = dict(data, qkv=data["qkv"].to(torch.float16), k_cache=data["k_cache"].float().half(), v_cache=data["v_cache"].float().half())
out = S.kernel_fn(run)
print("out nan rows:", torch.isnan(out.float()).view(-1, 64).any(1).nonzero().view(-1).tolist())
print("out inf:", torch.isinf(out.float()).any().item())
for key, (part, ml, cnt) in S._WS.items():
    print("ws", key, "part nan:", torch.isnan(part.float()).sum().item(), "part inf:", torch.isinf(part.float()).sum().item(),
          "ml nan:", torch.isnan(ml).sum().item(), "ml inf:", torch.isinf(ml).sum().item(), "cnt nonzero:", (cnt != 0).sum().item())
    m = ml.view(-1, 2)
    print("m values (unique, first 20):", m[:, 0].unique()[:20].tolist())
    print("l values (unique, first 20):", m[:, 1].unique()[:20].tolist())
    rb = 16
    n_sets = m.shape[0] // rb
    for i in range(n_sets):
        mm, ll = m[i * rb:(i + 1) * rb, 0], m[i * rb:(i + 1) * rb, 1]
        pp = part[i * rb * 128:(i + 1) * rb * 128].float() if part.numel() >= (i + 1) * rb * 128 else None
        flag = ""
        if pp is not None and (torch.isnan(pp).any() or torch.isinf(pp).any()):
            flag = " PART-BAD"
        if torch.isnan(mm).any() or torch.isnan(ll).any() or torch.isinf(ll).any():
            flag += " ML-BAD"
        if flag:
            print(f"  set {i}: m={mm[:4].tolist()} l={ll[:4].tolist()}{flag}")
