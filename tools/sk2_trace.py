#!/usr/bin/env python3
"""Per-CTA and per-stage timeline of the sk2 kernels (m <= 32) on the real GPU: a DIAG build (the dg bits are runtime
branches) run with dg = 32 after a read flush of L2, reading the kernel's own globaltimer trace: every CTA's SM id /
start / end, and for CTA 0 and CTA grid/2 the producer issue, consumer landed and consumer done times of every stage.

    python3 tools/sk2_trace.py pooler:8 pooler:32 qkv_gate:8 unpooler:16 [name=...]"""
import importlib, os, sys

import torch

args = [a for a in sys.argv[1:] if not a.startswith("--")]
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pdir = os.path.join(root, "muse_bf16_proj")
sys.path.insert(0, pdir)
os.chdir(pdir)
R = importlib.import_module("reference")
S = importlib.import_module("submission")
S.TRACE_SK2 = True   # trace-only build (the DIAG build's runtime dg branches cost 1-5 us)
S.AUTOTUNE = False
shapes = []
for a in args:
    layer, m = a.split(":")
    shapes.append((layer, int(m)))
weights = R.generate_weights(0)
state = S.prepare(weights)
dev = weights["o"].device
flush_buf = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
cfg_by_name = {cfg[0]: cfg for cfg in S.SK2_CONFIGS}
NDBG = 8192 + 3 * 4096

for layer, m in shapes:
    w = weights[layer]
    n = 1024 if (layer == "unpooler" and m <= 8) else w.shape[0]
    k = w.shape[1]
    name = S._sk2_name(layer, m)
    cfg = cfg_by_name[name]
    _, _, n_rows, Rr, NT, NWR, WSK, UPW, NACC, CL = cfg
    hl, nstage = S._sk2_geom(cfg, S._sk2_tokr(m))
    tokr = S._sk2_tokr(m)
    ups = WSK * UPW
    wby = NWR * WSK * UPW * NT * (2048 if hl else 768)
    xby = ups * tokr * 128
    nst = k // (ups * 64)
    ncta = n // Rr * CL
    x = R.generate_input(seed=0, m=m, layer=layer)["x"]
    out = S._run_sk2(x, layer, m, n, k, state, name=name)
    torch.cuda.synchronize()
    ref = x.double() @ w[:n].double().T
    if layer != "router":
        ref = ref.to(torch.bfloat16).double()
    err = ((out.double() - ref).norm() / ref.norm()).item()
    runs = []
    for rep in range(5):
        dbg = torch.zeros(NDBG, dtype=torch.int64, device=dev)
        flush_buf.view(torch.int32).sum()
        torch.cuda.synchronize()
        S._run_sk2(x, layer, m, n, k, state, dg=32, dbg=dbg, name=name)
        torch.cuda.synchronize()
        d = dbg.cpu()
        per = d[8192:8192 + 3 * ncta].view(ncta, 3)
        t0 = int(per[:, 1].min())
        st = (per[:, 1] - t0).double() / 1000.0
        en = (per[:, 2] - t0).double() / 1000.0
        runs.append((float(en.max()), st, en, per[:, 0].clone(), d[:8192].clone(), t0))
    runs.sort(key=lambda r: r[0])
    span, st, en, smid, tr, t0 = runs[len(runs) // 2]
    dur = en - st
    bins = int(span / 0.5) + 1
    prof = []
    for b in range(bins):
        t_lo, t_hi = 0.5 * b, 0.5 * (b + 1)
        prof.append(f"{int(((st < t_hi) & (en > t_lo)).sum())}/{int(((en >= t_lo) & (en < t_hi)).sum())}")
    total_mb = (wby * nst * n // Rr) / 1e6
    q = lambda v, p: float(torch.quantile(v, p))
    nsm = len(set(smid.tolist()))
    per_sm = torch.bincount(smid.to(torch.int64), minlength=132)
    print(f"== {layer} m={m} {name}: rel err {err:.2e}; {ncta} CTAs (R{Rr}, {nst} stages x {wby / 1024:.0f} KB weights + {xby / 1024:.0f} KB x, "
          f"ring {nstage}) on {nsm} SMs (CTAs per SM: min {int(per_sm.min())} max {int(per_sm.max())}); weights {total_mb:.1f} MB; "
          f"span {span:.2f} us ({total_mb / span:.2f} TB/s over the span); spans: " + " ".join(f"{r[0]:.2f}" for r in runs))
    print(f"   CTA start: median {q(st, 0.5):.2f}, 90% {q(st, 0.9):.2f}, last {float(st.max()):.2f}; "
          f"CTA end: first {float(en.min()):.2f}, median {q(en, 0.5):.2f}, 90% {q(en, 0.9):.2f}, last {span:.2f} us; "
          f"CTA duration: min {float(dur.min()):.2f}, median {q(dur, 0.5):.2f}, max {float(dur.max()):.2f} us "
          f"(median CTA streams {wby * nst / 1024 / q(dur, 0.5):.1f} KB/us of weights)")
    print(f"   active CTAs / finishing per 0.5 us: " + " ".join(prof))
    # who finishes late: by CTA index quarter (stream address order) and by SM occupancy (CTAs sharing an SM)
    qn = 4
    byq = [f"{float(en[i * ncta // qn:(i + 1) * ncta // qn].mean()):.2f}" for i in range(qn)]
    occ = per_sm[smid.to(torch.int64)]
    single = en[occ == 1]
    double = en[occ >= 2]
    print(f"   CTA end by index quarter: {' '.join(byq)}; on single-occupancy SMs: n={single.numel()} mean "
          f"{float(single.mean()) if single.numel() else float('nan'):.2f}; on shared SMs: n={double.numel()} mean "
          f"{float(double.mean()) if double.numel() else float('nan'):.2f}; latest 8 CTAs: "
          + " ".join(f"c{int(i)}@sm{int(smid[i])}:{float(en[i]):.2f}" for i in torch.argsort(en)[-8:]))
    for base, label in ((0, "cta0"), (4096, f"cta{ncta // 2}")):
        t = tr[base:base + 4096]
        ts = int(t[0])
        if ts == 0:
            continue
        rel = lambda v: (int(v) - ts) / 1000.0 if int(v) else float("nan")
        stages = []
        for s in range(min(nst, 64)):
            iss, land, done = rel(t[16 + 3 * s]), rel(t[16 + 3 * s + 1]), rel(t[16 + 3 * s + 2])
            stages.append(f"{s}:{iss:.2f}/{land:.2f}/{done:.2f}")
        lands = [rel(t[16 + 3 * s + 1]) for s in range(nst)]
        gaps = [lands[i + 1] - lands[i] for i in range(nst - 1) if lands[i + 1] == lands[i + 1] and lands[i] == lands[i]]
        gaps_s = sorted(gaps)
        print(f"   {label} (start {rel(t[0]) + 0:.2f}, SM {int(smid[0 if base == 0 else ncta // 2])}): stage issue/landed/done (us from CTA start): "
              + " ".join(stages))
        if gaps_s:
            print(f"   {label} stage-to-stage landing gaps: min {gaps_s[0]:.2f} median {gaps_s[len(gaps_s) // 2]:.2f} max {gaps_s[-1]:.2f} us "
                  f"-> {(wby + xby) / 1024 / gaps_s[len(gaps_s) // 2]:.1f} KB/us per CTA at the median gap; "
                  f"consumer loop done (warps): " + " ".join(f"{rel(t[1 + wv]):.2f}" for wv in range(NWR * WSK))
                  + f"; epilogue marks 9/11/10/13/14/15: {rel(t[9]):.2f} {rel(t[11]):.2f} {rel(t[10]):.2f} {rel(t[13]):.2f} {rel(t[14]):.2f} {rel(t[15]):.2f}",
                  flush=True)
