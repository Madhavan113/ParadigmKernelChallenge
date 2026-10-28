"""GPU-free gate for the skx route's Python glue in muse_bf16_proj/submission.py (run inside the judge image, torch
importable, no GPU): with a mock prepare() state, every judged / tested (layer, m) must produce candidates, keys, names
and a nearby choice without raising; skx candidates appear exactly for the bf16 layers at 33 <= m <= 256; the choice
for every untuned m never routes to skx outside its bucket; the generated CUDA dispatch covers every SKX_CONFIGS entry
once.  Usage: python3 tools/check_skx_dispatch.py"""
import importlib.util
import pathlib
import sys

root = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("sub", root / "muse_bf16_proj/submission.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

layers = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096), "router": (256, 8192)}
state = {"sm90": True, "old_paths": False, "disabled": set(), "raw": {l: None for l in layers if l != "router"},
         "rawT": {"unpooler": None, "o": None}, "sk2": {}, "cg": {}, "skx": {"cnt": {}, "tiled": {}}, "nsm": 132,
         "router_f32": None, "rk_zc": None, "cut_layers": set(), "choice": {}}
bad = 0
for m in (1, 5, 7, 8, 9, 16, 32, 33, 128, 130, 200, 256, 512, 1000):
    for layer, (N, K) in layers.items():
        n = 1024 if (layer == "unpooler" and m <= 8) else N
        try:
            cands = S._candidates(layer, m, state)
            keys = [S._cand_key(layer, m, c) for c in cands]
            names = [S._cand_name(c) for c in cands]
            skx = [c for c in cands if c[0] == "skx"]
            want = layer != "router" and 32 < m <= 256 and (layer, S._skx_bn(m)) in S.SKX_CONFIGS
            if bool(skx) != want:
                print(f"m={m} {layer}: skx candidates {skx} (expected {'some' if want else 'none'})"); bad += 1
            if len(set(keys)) != len(keys):
                print(f"m={m} {layer}: duplicate candidate keys {keys}"); bad += 1
            # pretend the autotuner picked skx[0] for the scored m of this bucket, then route an untuned m
            if want:
                state["choice"][(layer, 128 if m <= 128 else 256)] = skx[0]
            ch = S._nearby_choice(layer, m, state)
            if ch is not None and ch[0] == "skx" and not (32 < m <= 256 and (m in (128, 256) or m <= 128)):
                print(f"m={m} {layer}: untuned m routed to skx: {ch}"); bad += 1
            if ch is not None and ch[0] == "skx" and m > S._skx_bn(m):
                print(f"m={m} {layer}: skx bn {S._skx_bn(m)} < m"); bad += 1
        except Exception as e:   # noqa: BLE001
            print(f"m={m} {layer}: {type(e).__name__}: {e}"); bad += 1
src = S._skx_src(True)
for (layer, bn), cfgs in S.SKX_CONFIGS.items():
    for ns, fix, tiled in cfgs:
        tag = f"launch_skx<{bn}, {ns}, {fix}, {str(bool(tiled)).lower()}>"
        if src.count(tag) != 1:
            print(f"dispatch for {(layer, bn, ns, fix, tiled)}: {src.count(tag)} occurrences of {tag}"); bad += 1
if "skx" not in S.CPP_SRC or '"skx"' not in open(root / "muse_bf16_proj/submission.py").read().split("functions=")[1][:400]:
    print("skx not exported"); bad += 1
print("skx dispatch problems:", bad)
sys.exit(1 if bad else 0)
