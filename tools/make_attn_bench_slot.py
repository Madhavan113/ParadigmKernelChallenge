"""Build a muse_prefill_layer bench slot: main's prefill submission with the attention submission's first-call bench
run at the end of its prepare() (300 s untimed window; the prefill kernel_fn is untouched so its 5 tests pass).

  python3 tools/make_attn_bench_slot.py <out.py> "<python list of (name, flags[, ov]) variants>" [budget_s] [flat_base 0/1]
"""
import subprocess, sys

ROOT = "/Users/madhavanp/kerneloptimizer/.claude/worktrees/agent-aad2dcde86ab9f089"
out_path = sys.argv[1]
variants = sys.argv[2]
budget = int(sys.argv[3]) if len(sys.argv) > 3 else 150
flat_base = int(sys.argv[4]) if len(sys.argv) > 4 else 1
plans = sys.argv[5] if len(sys.argv) > 5 else "[]"
shapes = sys.argv[6] if len(sys.argv) > 6 else "[(16, 1), (32, 1), (16, 8), (32, 8)]"

pre = subprocess.run(["git", "-C", ROOT, "show", "main:muse_prefill_layer/submission.py"], capture_output=True, text=True, check=True).stdout
attn = open(f"{ROOT}/muse_attention_decode/submission.py").read()
# the attention file's own bench trigger stays off inside the slot (its _bench is called directly)
attn = attn.replace("OKC_BENCH = 1      # 1:", "OKC_BENCH = 0      # 1:")
assert "OKC_BENCH = 0" in attn

anchor = '''                print("SELFTEST", fn.__name__, "failed:", repr(exc), flush=True)
    return st
'''
assert pre.count(anchor) == 1
pre = pre.replace(anchor, '''                print("SELFTEST", fn.__name__, "failed:", repr(exc), flush=True)
    try:
        _attn_bench_slot(dev)
    except Exception as exc:  # noqa: BLE001
        print("ATTNBENCH failed:", repr(exc)[:400], flush=True)
    return st
''')
tail = f'''

# ----------------------------------------------------------------------------------------------
# attention bench slot (check runs: the muse_attention_decode submission's build sweep in prepare's untimed window)
# ----------------------------------------------------------------------------------------------
ATTN_SRC = {attn!r}
ATTN_BENCH_VARIANTS = {variants}
ATTN_BENCH_PLANS = {plans}
ATTN_BENCH_SHAPES = {shapes}
ATTN_BENCH_BUDGET = {budget}
ATTN_FLAT_BASE = {flat_base}


def _attn_bench_slot(dev):
    import time
    t0 = time.time()
    ns = {{"__name__": "okc_attn_slot", "__file__": __file__}}
    exec(compile(ATTN_SRC, "okc_attn_slot.py", "exec"), ns)
    ns["BENCH_VARIANTS"] = ATTN_BENCH_VARIANTS
    ns["BENCH_PLANS"] = ATTN_BENCH_PLANS
    ns["BENCH_SHAPES"] = ATTN_BENCH_SHAPES
    ns["BENCH_BUDGET_S"] = ATTN_BENCH_BUDGET
    if ATTN_FLAT_BASE:   # the flat kernels only (faster build; no per-request plans in this slot)
        ns["_MOD"] = ns["_build"](tag="_flatbase", flags=("-DOKC_FLATONLY=1",))
        print(f"ATTNBENCH flat base built in {{time.time() - t0:.0f}}s", flush=True)
    data = {{"qkv": torch.zeros((1, 1), device=dev, dtype=torch.bfloat16), "k_scale": torch.ones((1,), device=dev),
            "v_scale": torch.ones((1,), device=dev),
            "config": dict(batch=1, q_len=1, num_q_heads=16, num_kv_heads=2, head_dim=64, page_size=1, window=0,
                           rope_theta=0.0, qk_scale_factor=5.473, sm_scale=0.125, kv_scale=1.0)}}
    ns["_bench"](data)
    sys.stdout.flush()
    print(f"ATTNBENCH slot done in {{time.time() - t0:.0f}}s", flush=True)
'''
src = pre + tail
if "import sys" not in src.split("ATTN_SRC")[0]:
    src = src.replace("import hashlib\n", "import hashlib\nimport sys\n", 1)
open(out_path, "w").write(src)
compile(src, out_path, "exec")
print("wrote", out_path, len(src), "bytes")
