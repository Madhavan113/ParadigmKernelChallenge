#!/usr/bin/env python3
"""The kernel timeline of the LAST graph replay in an nsys cuda_gpu_trace CSV (nsys stats --report cuda_gpu_trace
--format csv): start / end offsets (us) from the replay's first kernel, duration, and the gap or overlap with the
previous kernel.  A replay = a run of kernels separated from the previous one by > `--gap` us of idle (the harness
flushes L2 and synchronizes between replays).

    python3 nsys_rows.py ~/ncu_out/chain_v6_d1s_cuda_gpu_trace.csv [--gap 20] [--n 2]
"""
import argparse, csv, re

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--gap", type=float, default=20.0, help="idle us that separates replays")
ap.add_argument("--n", type=int, default=1, help="print the last n replays")
args = ap.parse_args()
rows = []
with open(args.csv, newline="") as f:
    for r in csv.DictReader(f):
        try:
            st = int(r["Start (ns)"]); du = int(r["Duration (ns)"])
        except (KeyError, ValueError):
            continue
        name = r.get("Name", "")
        rows.append((st, st + du, name))
rows.sort()
# split into replays at idle gaps
reps, cur, last_end = [], [], None
for st, en, name in rows:
    if last_end is not None and st - last_end > args.gap * 1000:
        reps.append(cur); cur = []
    cur.append((st, en, name)); last_end = max(last_end or 0, en)
if cur:
    reps.append(cur)
# drop the flush kernels (reduce_kernel / sum) at the start of a replay
def layer_kernels(rep):
    return [k for k in rep if not re.search(r"reduce_kernel|elementwise_kernel|fill|Memset|Memcpy", k[2])]
good = [layer_kernels(r) for r in reps if len(layer_kernels(r)) >= 5]
for rep in good[-args.n:]:
    t0 = rep[0][0]
    print(f"\n=== replay with {len(rep)} kernels, span {(max(e for _, e, _ in rep) - t0) / 1000:.1f} us")
    prev_end = None
    for st, en, name in rep:
        gap = "" if prev_end is None else (f"gap {(st - prev_end) / 1000:+.1f}" if st >= prev_end else f"overlap {(prev_end - st) / 1000:.1f}")
        short = re.sub(r"\(.*", "", name)[:60]
        print(f"  {(st - t0) / 1000:7.1f} - {(en - t0) / 1000:7.1f}  ({(en - st) / 1000:5.1f})  {gap:14s} {short}")
        prev_end = max(prev_end or 0, en)
