"""Opcode histogram of the hottest HMMA/wgmma region of a cuobjdump -sass dump (tools/nvcc_pub_moe.py --sass output).
Usage: python3 tools/sass_pub_hist.py /tmp/okc_nvcc/MID.sass [function-substring]
Finds the function, then the densest window of mma instructions (the per-unit / per-stage inner loop) and prints the
instruction mix between the first and last mma of that window, normalized per mma."""
import re, sys, collections
path = sys.argv[1]
want = sys.argv[2] if len(sys.argv) > 2 else "moe_fusedILb0"
txt = open(path).read()
funcs = re.split(r"\n\s*Function : ", txt)
for f in funcs[1:]:
    name = f.split("\n", 1)[0]
    if want not in name:
        continue
    ops = []
    for m in re.finditer(r"/\*([0-9a-f]{4,})\*/\s+(@!?U?P\d\s+)?([A-Z][A-Z0-9_.]*)", f):
        ops.append(m.group(3))
    isMMA = [o.startswith("HMMA") or o.startswith("HGMMA") for o in ops]
    n = len(ops)
    mma_idx = [i for i, b in enumerate(isMMA) if b]
    print(f"{name[:60]}: {n} instructions, {len(mma_idx)} mma")
    if not mma_idx:
        continue
    # densest window of 32 consecutive mma (the skinny unit has 32 per warp; a WGS stage has 16 per warpgroup)
    W = 32 if len(mma_idx) >= 32 else len(mma_idx)
    best = min(range(len(mma_idx) - W + 1), key=lambda k: mma_idx[k + W - 1] - mma_idx[k])
    a, b = mma_idx[best], mma_idx[best + W - 1] + 1
    h = collections.Counter(o.split(".")[0] for o in ops[a:b])
    tot = b - a
    print(f"  densest {W}-mma window: {tot} instructions ({tot / W:.1f} per mma) at [{a}, {b})")
    for k, v in h.most_common(24):
        print(f"    {k:10s} {v:5d}  {v / W:5.2f}/mma")
