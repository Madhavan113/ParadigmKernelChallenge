#!/usr/bin/env python3
"""Per-phase timeline of the attention kernel's CTAs: the DBG() globaltimer stamps of a -DOKC_DBGON=1 build, under
judge-like conditions (CUDA-graph replay, a 256 MB read flush of L2 before each replay). For every stamp: median / p90 /
max over CTAs of (stamp - earliest CTA start) in us, and the same relative to each CTA's own start; the finalizing
CTAs' stamps separately. Runs inside the judge image with the repo at /work.

    python3 muse_attention_decode/_attn_timeline.py --shape "batch=16,ctx=80000,heads=tp8,layer=sliding,page=1,spread=agentic,dist=soft" \
        [--flags "-DOKC_X=1"] [--tag _dbg] [--override n_splits=8,coop=1] [--reps 5] [--seed 0]
"""
import argparse, os, statistics, sys

import torch

sys.path.insert(0, "/work/muse_attention_decode")
os.chdir("/work/muse_attention_decode")
import reference as R  # noqa: E402
import submission as S  # noqa: E402

# stamp slot -> phase (per-request kernels; slot 6 == 1 marks the CTA that finalized the request)
NAMES = [(0, "cta start"), (1, "page idx landed"), (8, "K/V issued"), (9, "rope table"), (10, "q pass"),
         (11, "new k/v"), (2, "consumer state"), (12, "tile0 landed"), (13, "tile0 computed"), (3, "tile loop done"),
         (7, "reds issued"), (6, "reds synced"), (4, "merge ticket"), (14, "finalize start"), (5, "finalize done")]


def parse_shape(spec):
    kw = {}
    for part in spec.split(","):
        k, v = part.split("=", 1)
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        kw[k.strip()] = v
    return kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", action="append", required=True)
    ap.add_argument("--flags", default="")
    ap.add_argument("--tag", default="_dbg")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--override", default="")
    ap.add_argument("--per-cta", action="store_true", help="dump every CTA's stamps of the last replay")
    ap.add_argument("--plain", action="store_true", help="no DBGON (timing control: the stamp buffer passed, stamps compiled out)")
    ap.add_argument("--no-flush", action="store_true", help="warm L2 (no read flush before the replays)")
    ap.add_argument("--no-dbg-ptr", action="store_true", help="launch without the stamp buffer (timing control)")
    ap.add_argument("--prof", action="store_true", help="-DOKC_PROF=1 build: per-warp phase cycles of the tile loop (dbg[65536:])")
    args = ap.parse_args()
    flags = (() if args.plain else ("-DOKC_DBGON=1",)) + (("-DOKC_PROF=1",) if args.prof else ()) + tuple(f for f in args.flags.split() if f)
    mod = S._build(args.tag, flags)
    over = {}
    for part in args.override.split(","):
        if part:
            k, v = part.split("=")
            over[k] = int(v)
    flush_buf = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    for spec in args.shape:
        data = R.generate_input(seed=args.seed, **parse_shape(spec))
        dbg = torch.zeros(1 << 20, dtype=torch.int64, device="cuda")
        dptr = None if args.no_dbg_ptr else dbg
        out = S._launch(data, dbg=dptr, _mod=mod, **over)
        torch.cuda.synchronize()
        print(f"[check] {spec}: {str(R.check_implementation(data, R.collect(data, out)))[:200]}", flush=True)
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            S._launch(data, dbg=dptr, _mod=mod, **over)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            S._launch(data, dbg=dptr, _mod=mod, **over)
        torch.cuda.synchronize()
        per = {k: [] for k, _ in NAMES}       # per stamp: list over reps of (median, p90, max) relative to global start
        own = {k: [] for k, _ in NAMES}       # relative to each CTA's own start
        fin = []
        elapsed = []
        last = None
        for rep in range(args.reps):
            dbg.zero_()
            if not args.no_flush:
                flush_buf.view(torch.int32).sum()
            torch.cuda.synchronize()
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            g.replay()
            e1.record()
            torch.cuda.synchronize()
            elapsed.append(e0.elapsed_time(e1) * 1000.0)
            d = dbg.cpu().view(-1, 16)
            d = d[d[:, 0] > 0]
            last = d
            if d.numel() == 0:
                continue
            t0 = int(d[:, 0].min())
            for k, _ in NAMES:
                col = d[:, k]
                v = col[col > 0]
                if v.numel() == 0:
                    continue
                rel = (v - t0).double() / 1000.0
                rel_own = (v - d[col > 0][:, 0]).double() / 1000.0
                srt = rel.sort().values
                per[k].append((float(srt[len(srt) // 2]), float(srt[(len(srt) * 9) // 10]), float(srt[-1]), int(v.numel())))
                srt2 = rel_own.sort().values
                own[k].append((float(srt2[len(srt2) // 2]), float(srt2[(len(srt2) * 9) // 10]), float(srt2[-1])))
            f = d[(d[:, 6] == 1) | (d[:, 14] > 0)]
            if f.numel():
                fin.append(((f[:, 14] - t0).double().median().item() / 1000.0, (f[:, 5] - t0).double().median().item() / 1000.0,
                            (f[:, 5] - t0).double().max().item() / 1000.0, (f[:, 5] - f[:, 14]).double().median().item() / 1000.0,
                            int(f.shape[0]), (f[:, 14] - f[:, 4]).double().median().item() / 1000.0,
                            (f[:, 4] - f[:, 3]).double().median().item() / 1000.0 if int(f[:, 3].min()) > 0 else -1.0))
        print(f"[timeline] {spec}  graph replay med {statistics.median(elapsed):.1f} us  CTAs {last.shape[0]}  reps {args.reps}")
        if last.numel() == 0:
            continue
        print(f"  {'stamp':18s} {'n':>4s} | rel. to first CTA start (us): med    p90    max | rel. to own start: med    p90    max")
        for k, name in NAMES:
            if not per[k]:
                continue
            med = statistics.median(x[0] for x in per[k])
            p90 = statistics.median(x[1] for x in per[k])
            mx = statistics.median(x[2] for x in per[k])
            omed = statistics.median(x[0] for x in own[k])
            op90 = statistics.median(x[1] for x in own[k])
            omx = statistics.median(x[2] for x in own[k])
            print(f"  {name:18s} {per[k][0][3]:4d} | {med:30.2f} {p90:6.2f} {mx:6.2f} | {omed:24.2f} {op90:6.2f} {omx:6.2f}")
        if fin:
            print(f"  finalizers: n {fin[0][4]}  start med {statistics.median(x[0] for x in fin):.2f}  done med "
                  f"{statistics.median(x[1] for x in fin):.2f}  done max {statistics.median(x[2] for x in fin):.2f}  "
                  f"dur med {statistics.median(x[3] for x in fin):.2f} us  (own loop->ticket "
                  f"{statistics.median(x[6] for x in fin):.2f}, ticket->start {statistics.median(x[5] for x in fin):.2f})")
        if args.prof:
            pr = dbg.cpu()[65536:].view(-1, 8)
            pr = pr[pr[:, 7] > 0]
            if pr.numel():
                tiles = pr[:, 7].double()
                print(f"  [prof] warps {pr.shape[0]}  tiles/warp med {tiles.median():.1f}  cycles per tile per warp (mean over warps): "
                      + "  ".join(f"ph{k} {(pr[:, k].double() / tiles).mean():.0f}" for k in range(7))
                      + f"  | total {(pr[:, :7].double().sum(1) / tiles).mean():.0f}")
        if args.per_cta and last is not None:
            t0 = int(last[:, 0].min())
            for i in range(last.shape[0]):
                row = last[i]
                print("  cta", i, " ".join(f"{name[:6]}={(int(row[k]) - t0) / 1000.0:.2f}" for k, name in NAMES if int(row[k]) > 0),
                      "FIN" if int(row[6]) == 1 else "")


if __name__ == "__main__":
    main()
