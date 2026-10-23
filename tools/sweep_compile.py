import subprocess, sys, itertools
sys.path.insert(0, "/work/muse_bf16_proj")
import submission as S
layers = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096), "router": (256, 8192)}
seen = set()
for m in (1, 8, 16, 32, 128):
    for layer, (N, K) in layers.items():
        n = 1024 if layer == "unpooler" and m <= 8 else N
        c = S._config(layer, m, n, K)
        if c is None: continue
        tw = 1 if layer == "router" else 0
        key = (c["BLOCK_M"], c["BLOCK_N"], c["BLOCK_K"], c["SPLIT_K"], tw, c["num_warps"], c["num_stages"])
        if key in seen: continue
        seen.add(key)
        out = subprocess.run(["python3", "compile_proj.py"] + [str(k) for k in key], capture_output=True, text=True).stdout
        lines = [l for l in out.splitlines() if l.startswith(("ttgir", "regs"))]
        print(f"{layer:9s} m={m:3d} {key} ->", " || ".join(lines), flush=True)
