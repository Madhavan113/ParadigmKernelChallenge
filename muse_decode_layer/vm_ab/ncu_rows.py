#!/usr/bin/env python3
"""Per-kernel rows of an ncu details CSV (ncu --page details --csv): duration, grid/block, occupancy, throughput,
instruction counts and the top warp-stall reasons, for kernels whose name matches a pattern.

    python3 ncu_rows.py ~/ncu_out/glue.csv '_boundary1|_route_split|_boundary2'
"""
import csv, re, sys
from collections import OrderedDict

path, pat = sys.argv[1], re.compile(sys.argv[2] if len(sys.argv) > 2 else ".")
WANT = ["Duration", "Elapsed Cycles", "SM Frequency", "Memory Throughput", "DRAM Throughput", "L2 Cache Throughput",
        "Compute (SM) Throughput", "Achieved Occupancy", "Achieved Active Warps Per SM", "Registers Per Thread",
        "Executed Instructions", "Issued Instructions", "Executed Ipc Active", "Issue Slots Busy",
        "Warp Cycles Per Issued Instruction", "Avg. Active Threads Per Warp", "No Eligible", "Active Warps Per Scheduler",
        "Eligible Warps Per Scheduler", "Issued Warp Per Scheduler", "L1/TEX Hit Rate", "L2 Hit Rate",
        "Mem Busy", "Max Bandwidth", "Block Limit Registers", "Block Limit Shared Mem", "Theoretical Occupancy",
        "Static Shared Memory Per Block", "Dynamic Shared Memory Per Block", "Waves Per SM", "Stall Long Scoreboard",
        "Stall Short Scoreboard", "Stall Wait", "Stall Barrier", "Stall MIO Throttle", "Stall Math Pipe Throttle",
        "Stall Not Selected", "Stall LG Throttle", "Stall Branch Resolving", "Stall Dispatch Stall", "Stall Membar",
        "Stall Selected", "Stall Drain", "Stall IMC Miss", "Stall No Instruction", "Stall Sleeping", "Stall Tex Throttle"]
rows = OrderedDict()
with open(path, newline="") as f:
    for r in csv.DictReader(f):
        name = r.get("Kernel Name", "")
        if not pat.search(name):
            continue
        key = (r["ID"], name[:70], r.get("Grid Size", ""), r.get("Block Size", ""))
        d = rows.setdefault(key, OrderedDict())
        m = r.get("Metric Name", "")
        if m in WANT or m.startswith("Stall "):
            d[m] = (r.get("Metric Value", ""), r.get("Metric Unit", ""))
for (kid, name, grid, block), d in rows.items():
    print(f"\n== #{kid} {name}  grid {grid} block {block}")
    for m in WANT:
        if m in d:
            v, u = d[m]
            print(f"  {m:40s} {v:>14s} {u}")
    extra = [m for m in d if m not in WANT]
    for m in extra:
        v, u = d[m]
        print(f"  {m:40s} {v:>14s} {u}")
