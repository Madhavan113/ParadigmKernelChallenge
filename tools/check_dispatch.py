"""GPU-free dispatch gate: for every judged / tested shape, _config must produce a config that _usable accepts or
rejects WITHOUT raising (a config-key collision between two kernels' dicts raised KeyError in check dbd64ace), and
every per-shape table entry must be a usable config for its own shape. Also exercises the bench's config dicts.
Usage (judge image or host python with torch/triton importable): python3 check_dispatch.py"""
import sys, itertools
sys.path.insert(0, "/work/muse_bf16_proj")
import submission as S

layers = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096), "router": (256, 8192)}
S._TB12_ON = True; S._WG12_ON = True
bad = 0
for m in (1, 5, 7, 8, 9, 16, 32, 33, 128, 130, 256, 512, 1000):
    for layer, (N, K) in layers.items():
        n = 1024 if (layer == "unpooler" and m <= 8) else N
        try:
            cfg = S._config(layer, m, n, K)
            ok = S._usable(cfg, m, n, K)
        except Exception as e:
            print(f"m={m} {layer}: {type(e).__name__}: {e}"); bad += 1; continue
        key = (layer, n, m)
        tables = [t for t in ("GEMV_SHAPES", "G12_SHAPES", "TB12_SHAPES", "WG12_SHAPES", "W12_SHAPES", "W12B_SHAPES", "W12M_SHAPES", "BMM_SHAPES") if key in getattr(S, t)]   # CUBLAS_SHAPES entries map to None (cuBLAS) by design
        if tables and not ok:
            print(f"m={m} {layer}: table {tables} entry is not usable: {cfg}"); bad += 1
# the bench's own config dicts must have distinct keys from every dispatch branch
keys = ["gemv", "mma", "tgemv", "g12", "m12cp", "w12", "w12m", "w12b", "tb12", "wg12", "m12", "t12", "bmm"]
assert len(set(keys)) == len(keys)
for k_, (N, K) in layers.items():
    for m in (8, 128):
        for cfg in (dict(wg12=True, rb=2, tb=1, ck=1, nsw=3, nsx=3, ks=4), dict(tb12=True, r=16, ck=4, w=4, nsw=3, nsx=3, nt=1, v=5), dict(g12=True, warps=8, unroll=1, sk=4)):
            try:
                S._usable(cfg, m, N, K)
            except Exception as e:
                print(f"bench cfg {cfg} m={m} {k_}: {type(e).__name__}: {e}"); bad += 1
print("dispatch problems:", bad)
sys.exit(1 if bad else 0)
