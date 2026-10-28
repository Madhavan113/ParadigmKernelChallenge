#!/usr/bin/env python3
"""Per-CTA timeline of the m = 1 GEMV kernels (cgemv / cg2) on the real GPU: start / end globaltimer of every CTA
(the kernels' dbg trace), after a read flush of L2 like the judge. Prints the kernel span, the launch spread, the
per-CTA durations and the active-CTA / bytes-in-flight profile over time.

    python3 tools/cg_trace.py pooler cgemv/2x512 cg2/16r2s4x512 cg2dg/16r2s4x512 [--ncu]

--ncu: one call per kernel inside an NVTX range "okc_profile" (for ncu --nvtx-include okc_profile/), no trace."""
import importlib, os, re, sys

import torch

args = [a for a in sys.argv[1:] if not a.startswith("--")]
ncu = "--ncu" in sys.argv
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pdir = os.path.join(root, "muse_bf16_proj")
sys.path.insert(0, pdir)
os.chdir(pdir)
R = importlib.import_module("reference")
S = importlib.import_module("submission")
S.AUTOTUNE = False
layer, names = args[0], args[1:]
weights = R.generate_weights(0)
state = S.prepare(weights)
ext = S._get_ext()
dev = weights[layer].device
data = R.generate_input(seed=0, m=1, layer=layer)
x = data["x"]
n = 1024 if layer == "unpooler" else weights[layer].shape[0]
k = weights[layer].shape[1]
flush_buf = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
ref = (x.double() @ weights[layer][:n].double().T).to(torch.bfloat16).double()


def run(name, dbg):
    kind, cfg = name.split("/")
    out = torch.empty((1, n), dtype=torch.bfloat16, device=dev)
    if kind == "cgemv":
        rows, nth = map(int, cfg.split("x"))
        cg = state["cg"][(layer, rows)]
        ext.cgemv(cg["L"], cg["C"], x, cg["tab"], cg["pmax"], out, cg["base4"], rows, nth, dbg)
        return out, n // rows, 1.5 * rows * k
    m = re.match(r"(\d+)r(\d+)s(\d+)x(\d+)", cfg)
    rpc, rs, ns, nth = map(int, m.groups())
    full = [c for c in S.CG2_CONFIGS[layer] if c[:4] == (rpc, rs, ns, nth)][0]
    cg = state["cg"][(layer, rpc)]
    ext.cg2(cg["L"], cg["C"], x, cg["tab"], cg["pmax"], out, cg["base4"], *full, 1 if kind == "cg2dg" else 0, dbg)
    return out, n // rpc, 1.5 * rpc * k


for name in names:
    if ncu:
        run(name, state["nodbg"])
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("okc_profile")
        run(name, state["nodbg"])
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        print(f"[ncu] {layer} {name}: profiled", flush=True)
        continue
    out, ncta, _ = run(name, state["nodbg"])
    torch.cuda.synchronize()
    err = ((out.double() - ref).norm() / ref.norm()).item() if "dg" not in name.split("/")[0] else float("nan")
    runs = []
    for rep in range(7):
        dbg = torch.zeros(2 * ncta, dtype=torch.int64, device=dev)
        flush_buf.view(torch.int32).sum()
        torch.cuda.synchronize()
        _, _, bytes_per_cta = run(name, dbg)
        torch.cuda.synchronize()
        t = dbg.view(ncta, 2).cpu()
        t0 = int(t[:, 0].min())
        st = (t[:, 0] - t0).double() / 1000.0   # us
        en = (t[:, 1] - t0).double() / 1000.0
        runs.append((float(en.max()), st, en, bytes_per_cta))
    runs.sort(key=lambda r: r[0])
    span, st, en, bpc = runs[len(runs) // 2]
    dur = en - st
    order = torch.argsort(st)
    # active CTAs per 0.5 us bin, and the stream rate implied by CTA completions
    bins = int(span / 0.5) + 1
    prof = []
    for b in range(bins):
        t_lo, t_hi = 0.5 * b, 0.5 * (b + 1)
        active = int(((st < t_hi) & (en > t_lo)).sum())
        done = int(((en >= t_lo) & (en < t_hi)).sum())
        prof.append(f"{active}/{done}")
    total_mb = bpc * ncta / 1e6
    print(f"== {layer} {name}: rel err {err:.2e}; {ncta} CTAs x {bpc / 1024:.1f} KB = {total_mb:.1f} MB; "
          f"span {span:.2f} us ({total_mb / span:.2f} TB/s over the span); all runs' spans: "
          + " ".join(f"{r[0]:.2f}" for r in runs))
    q = lambda v, p: float(torch.quantile(v, p))
    print(f"   CTA start: first 0, median {q(st, 0.5):.2f}, 90% {q(st, 0.9):.2f}, last {float(st.max()):.2f} us; "
          f"CTA end: first {float(en.min()):.2f}, median {q(en, 0.5):.2f}, 90% {q(en, 0.9):.2f}, last {span:.2f} us")
    print(f"   CTA duration: min {float(dur.min()):.2f}, median {q(dur, 0.5):.2f}, 90% {q(dur, 0.9):.2f}, max {float(dur.max()):.2f} us "
          f"(a CTA at the median moves {bpc / 1024 / q(dur, 0.5):.1f} KB/us)")
    print(f"   active CTAs / CTAs finishing, per 0.5 us bin: " + " ".join(prof))
    # the first and last CTAs by start time (launch order) and their ends
    first = order[:4].tolist()
    last = order[-4:].tolist()
    print(f"   first launched: " + ", ".join(f"cta{i} {st[i]:.2f}->{en[i]:.2f}" for i in first)
          + "; last launched: " + ", ".join(f"cta{i} {st[i]:.2f}->{en[i]:.2f}" for i in last), flush=True)
