"""Print a proj check's per-shape table next to a reference run, plus its [okc] prepare lines.
Usage: python3 tools/okc_proj_table.py <job> [ref_job]   (profiles under muse_bf16_proj/profiles/)"""
import re, sys, pathlib, json
root = pathlib.Path(__file__).resolve().parent.parent / "muse_bf16_proj" / "profiles"
job = sys.argv[1]
ref = sys.argv[2] if len(sys.argv) > 2 else None


def table(j):
    t = (root / f"submission.{j}.txt").read_text()
    rows = {}
    for l in t.splitlines():
        m = re.match(r"layer=(\w+) load=\S+ \d+ m=(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)x\s+([\d.]+)", l)
        if m:
            rows[(m.group(1), int(m.group(2)))] = (float(m.group(3)), float(m.group(4)), float(m.group(5)), float(m.group(6)))
    sp = re.search(r"speedup\s+([\d.]+)", t)
    ci = re.search(r"speedup_ci\s+\[([\d.]+) ([\d.]+)\]", t)
    return rows, (sp.group(1) if sp else "?"), (ci.groups() if ci else ("?", "?"))


rows, sp, ci = table(job)
rrows = table(ref)[0] if ref else {}
print(f"job {job}: speedup {sp} CI {ci[0]}-{ci[1]}" + (f"   (ref {ref}: {table(ref)[1]})" if ref else ""))
print(f"{'shape':22s} {'sglang':>7s} {'ours':>7s} {'x':>6s} {'sol':>6s}" + (f" {'ref':>7s} {'delta':>7s}" if ref else ""))
for (layer, m), (sg, us, x, sol) in rows.items():
    line = f"{layer:10s} m={m:<9d} {sg:7.2f} {us:7.2f} {x:6.3f} {sol:6.2f}"
    if ref and (layer, m) in rrows:
        r = rrows[(layer, m)][1]
        line += f" {r:7.2f} {us - r:+7.2f}"
    print(line)
log = root / f"submission.{job}.log"
if log.exists():
    for l in log.read_text().splitlines():
        if "[okc]" in l:
            print(l.strip())
