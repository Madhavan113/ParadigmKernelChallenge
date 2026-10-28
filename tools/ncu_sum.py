#!/usr/bin/env python3
"""Summarize an ncu CSV pair (<name>.csv details page, <name>.raw.csv raw page) per profiled kernel.
    python3 tools/ncu_sum.py /tmp/okc_ncu/proj_set [kernel-name-substring]"""
import csv, sys

base = sys.argv[1]
pat = sys.argv[2] if len(sys.argv) > 2 else None
rows = list(csv.DictReader(open(base + ".csv")))
ids = sorted({r["ID"] for r in rows}, key=int)
want = ["Duration", "DRAM Throughput", "Memory Throughput", "Achieved Occupancy", "Issue Slots Busy", "Registers Per Thread",
        "Block Size", "Grid Size", "Dynamic Shared Memory Per Block", "Static Shared Memory Per Block", "Waves Per SM",
        "Theoretical Occupancy", "Achieved Active Warps Per SM", "Issued Warp Per Scheduler", "Eligible Warps Per Scheduler",
        "Active Warps Per Scheduler", "L2 Hit Rate", "Mem Busy", "Max Bandwidth", "Block Limit Registers",
        "Block Limit Shared Mem", "Block Limit Warps", "L1/TEX Hit Rate", "Compute (SM) Throughput", "No Eligible",
        "One or More Eligible", "Executed Ipc Active", "SM Busy", "L2 Compression Success Rate", "Mem Pipes Busy"]
for i in ids:
    rr = [r for r in rows if r["ID"] == i]
    name = rr[0]["Kernel Name"]
    if pat and pat not in name:
        continue
    d = {}
    for r in rr:
        if r["Metric Name"] in want and r["Metric Name"] not in d:
            d[r["Metric Name"]] = r["Metric Value"] + " " + r["Metric Unit"]
    print(f"== [{i}] {name[:90]}")
    print("   " + "; ".join(f"{k}={d[k]}" for k in want if k in d))
try:
    raw = list(csv.DictReader(open(base + ".raw.csv")))
except FileNotFoundError:
    sys.exit(0)
hdr = list(raw[0].keys())
units = raw[0]
keys = ["gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum", "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "lts__t_bytes.sum", "lts__t_sectors_srcunit_tex_op_read.sum", "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
        "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum", "l1tex__t_bytes.sum", "sm__warps_active.avg.pct_of_peak_sustained_active",
        "smsp__issue_active.avg.pct_of_peak_sustained_active", "launch__registers_per_thread", "launch__occupancy_limit_registers",
        "launch__occupancy_limit_shared_mem", "launch__occupancy_limit_warps", "launch__occupancy_limit_blocks", "sm__cycles_active.avg",
        "smsp__inst_executed.sum", "smsp__warps_eligible.avg.per_cycle_active", "smsp__warps_active.avg.per_cycle_active",
        "smsp__issue_active.avg.per_cycle_active", "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "l1tex__throughput.avg.pct_of_peak_sustained_active", "lts__throughput.avg.pct_of_peak_sustained_elapsed",
        "smsp__average_warp_latency_per_inst_issued.ratio", "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active"]
for r in raw[1:]:
    name = r.get("Kernel Name", "")
    if pat and pat not in name:
        continue
    print(f"== raw [{r.get('ID')}] {name[:90]}")
    for k in hdr:
        if k in keys:
            print(f"   {k} = {r[k]} {units.get(k, '')}")
    st = []
    for k in hdr:
        if k.startswith("smsp__average_warps_issue_stalled_") and k.endswith("_per_issue_active.ratio"):
            try:
                v = float(r[k].replace(",", ""))
            except ValueError:
                continue
            if v >= 0.05:
                st.append((v, k[len("smsp__average_warps_issue_stalled_"):-len("_per_issue_active.ratio")]))
    st.sort(reverse=True)
    print("   stalls/issue: " + ", ".join(f"{n}={v:.2f}" for v, n in st[:10]))
