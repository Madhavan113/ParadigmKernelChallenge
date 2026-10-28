#!/usr/bin/env python3
"""A/B the decode layer's chain-timed picks on a real GPU with statistics: ONE prepare per seed, then variants applied as
runtime overrides of the tuner's picks, each captured as its own CUDA graph; the graphs of one shape are then replayed
round-robin (paired samples: no drift / order effect), every replay preceded by a 256 MB read flush of L2 (as the judge
and tools/gpu_run.py do). Medians per (variant, shape, seed), the paired delta vs the first variant (median of the
per-round differences), and a final table of means over seeds.

    python3 muse_decode_layer/ab_vm.py --seeds 0 1 --iters 25 --shapes d1s d1f v1s v1f \
        --variant base --variant "pr:1=bfp" --variant "pr:1=bfpf;bnd1:1=1" --variant "pdl:1=6" --variant "rmode:1=moe_nb"

Override keys (T = token count): pr:T=bfp|bfpf (the pooler+router form), rmode:T=split|sk2f|moe|moe_nb,
bnd1:T=0|1 (boundary-1 prologue in the fused launch), pdl:T=N (sk2 pre-wait weight stages), btail:T.o=0|1,
btail:T.unpooler=0|1 (boundary tails), sk2:layer.tokr=config (needs OKC_SK2_ALL=1 at build), mm:T.name=pick
(dense pick: False|True|bfp|mm16|torchT|bfq), glue:T=0|1 (CUDA vs Triton glue kernels), pdlmax=N (PDL_MAX_T). A spec "s0:<overrides>|s1:<overrides>" applies per seed (e.g. an earlier
prepare's picks as the "before" of a before/after). Run inside the judge image with the repo at /work (cwd = /work).
"""
import argparse, importlib, os, re, statistics, sys, time

import torch

SHAPES = {
    "d1s": "batch=1,ctx=80000,layer=sliding,page=1,dist=soft",
    "d1f": "batch=1,ctx=80000,layer=full,page=1,dist=soft",
    "d16s": "batch=16,ctx=80000,layer=sliding,page=1,dist=soft,spread=agentic",
    "d16f": "batch=16,ctx=80000,layer=full,page=1,dist=soft,spread=agentic",
    "d32s": "batch=32,ctx=80000,layer=sliding,page=1,dist=soft,spread=agentic",
    "d32f": "batch=32,ctx=80000,layer=full,page=1,dist=soft,spread=agentic",
    "v1s": "batch=1,ctx=80000,layer=sliding,page=1,dist=soft,q_len=8",
    "v1f": "batch=1,ctx=80000,layer=full,page=1,dist=soft,q_len=8",
    "v16s": "batch=16,ctx=80000,layer=sliding,page=1,dist=soft,spread=agentic,q_len=8",
    "v16f": "batch=16,ctx=80000,layer=full,page=1,dist=soft,spread=agentic,q_len=8",
    "v32s": "batch=32,ctx=80000,layer=sliding,page=1,dist=soft,spread=agentic,q_len=8",
    "v32f": "batch=32,ctx=80000,layer=full,page=1,dist=soft,spread=agentic,q_len=8",
}


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


def parse_pick(v):
    return {"False": False, "True": True, "0": False, "1": True}.get(v, v)


def for_seed(spec, seed):
    """A per-seed spec "s0:<overrides>|s1:<overrides>" resolves to this seed's part ("base" when the seed has none)."""
    if not re.match(r"^s\d+:", spec):
        return spec
    for part in spec.split("|"):
        if part.startswith(f"s{seed}:"):
            return part[len(f"s{seed}:"):]
    return "base"


def apply(S, st, spec, saved):
    """Apply an override spec to the module / state; record what it replaced in `saved` (for restore)."""
    for item in [s for s in spec.split(";") if s and s != "base"]:
        key, val = item.split("=", 1)
        kind, sel = key.split(":", 1) if ":" in key else (key, "")
        if kind == "pr":
            T = int(sel)
            saved.append(("mm", (T, "pr"), st["mm_pick"].get((T, "pr"))))
            st["mm_pick"][(T, "pr")] = val
        elif kind == "mm":
            T, name = sel.split(".")
            k = (int(T), name)
            saved.append(("mm", k, st["mm_pick"].get(k)))
            st["mm_pick"][k] = parse_pick(val)
        elif kind == "rmode":
            T = int(sel)
            saved.append(("route_mode", T, st.setdefault("route_mode", {}).get(T)))
            st["route_mode"][T] = val
        elif kind == "bnd1":
            T = int(sel)
            saved.append(("bnd1_on", T, st.setdefault("bnd1_on", {}).get(T)))
            st["bnd1_on"][T] = val in ("1", "True")
        elif kind == "pdl":
            T = int(sel)
            saved.append(("pdl", T, S._PDL_N_BY_T.get(T)))
            S._PDL_N_BY_T[T] = int(val)
        elif kind == "btail":
            T, layer = sel.split(".")
            k = (int(T), layer)
            saved.append(("btail_on", k, st.setdefault("btail_on", {}).get(k)))
            st["btail_on"][k] = val in ("1", "True")
        elif kind == "sk2":
            layer, tokr = sel.split(".")
            k = (layer, int(tokr))
            saved.append(("sk2", k, S.bfp_SK2_PICK.get(k)))
            S.bfp_SK2_PICK[k] = val
        elif kind == "glue":   # the CUDA glue kernels (boundaries + routing) vs the Triton ones at this T
            T = int(sel)
            saved.append(("glue_on", T, st.setdefault("glue_on", {}).get(T)))
            st["glue_on"][T] = val in ("1", "True")
        elif kind == "pdlmax":   # PDL_MAX_T: programmatic launches (attention, MoE, glue, sk2) up to this T
            saved.append(("pdlmax", None, S.PDL_MAX_T))
            S.PDL_MAX_T = int(val)
        else:
            raise ValueError(item)


def restore(S, st, saved):
    for kind, k, old in reversed(saved):
        if kind == "mm":
            if old is None:
                st["mm_pick"].pop(k, None)
            else:
                st["mm_pick"][k] = old
        elif kind == "pdl":
            if old is None:
                S._PDL_N_BY_T.pop(k, None)
            else:
                S._PDL_N_BY_T[k] = old
        elif kind == "sk2":
            S.bfp_SK2_PICK[k] = old
        elif kind == "pdlmax":
            S.PDL_MAX_T = old
        else:
            d = st[kind]
            if old is None:
                d.pop(k, None)
            else:
                d[k] = old


def capture(S, st, data):
    """Two warm-up calls, then one CUDA graph of kernel_fn on its own stream."""
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(2):
            S.kernel_fn(data, st)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        S.kernel_fn(data, st)
    torch.cuda.synchronize()
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--shapes", nargs="+", default=list(SHAPES))
    ap.add_argument("--variant", action="append", required=True)
    ap.add_argument("--flush-mb", type=int, default=256)
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdir = os.path.join(root, "muse_decode_layer")
    sys.path.insert(0, pdir)
    os.chdir(pdir)
    R = importlib.import_module("reference")
    S = importlib.import_module("submission")
    torch.cuda.set_device(0)
    flush_buf = torch.empty(args.flush_mb << 20, dtype=torch.uint8, device="cuda")
    results = {}   # (variant, shape, seed) -> median us
    nan = float("nan")
    for seed in args.seeds:
        weights = R.generate_weights(seed)
        t0 = time.perf_counter()
        st = S.prepare(weights)
        torch.cuda.synchronize()
        print(f"[prepare] seed {seed}: {time.perf_counter() - t0:.1f} s", flush=True)
        picks = {k: v for k, v in st.get("mm_pick", {}).items() if k[0] in (1, 8, 16, 32, 128, 256)}
        print(f"[picks] seed {seed}: mm {picks}", flush=True)
        print(f"[picks] seed {seed}: route_mode {st.get('route_mode')} bnd1 {st.get('bnd1_on')} btail {st.get('btail_on')} "
              f"pdl {dict(S._PDL_N_BY_T)} gluev2 {st.get('gluev2')} moe {st.get('moe_pick')}", flush=True)
        datas = {name: R.generate_input(seed=seed, **parse_shape(SHAPES[name])) for name in args.shapes}
        for name in args.shapes:
            data = datas[name]
            graphs = []   # one graph per variant, captured with its overrides applied (the overrides act at launch)
            for vi, spec in enumerate(args.variant):
                saved = []
                g = None
                try:
                    apply(S, st, for_seed(spec, seed), saved)
                    g = capture(S, st, data)
                except Exception as ex:  # noqa: BLE001  (a variant this build cannot run: keep the batch going)
                    torch.cuda.synchronize()
                    print(f"[fail] v{vi} seed {seed} {name}: {repr(ex)[:300]} ({spec})", flush=True)
                restore(S, st, saved)
                graphs.append(g)
            ts = [[] for _ in graphs]
            for _ in range(args.iters):   # round-robin: every round replays every variant once (paired samples)
                for vi, g in enumerate(graphs):
                    if g is None:
                        continue
                    flush_buf.view(torch.int32).sum()
                    torch.cuda.synchronize()
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                    g.replay()
                    e1.record()
                    torch.cuda.synchronize()
                    ts[vi].append(e0.elapsed_time(e1) * 1000.0)
            for vi, spec in enumerate(args.variant):
                if graphs[vi] is None:
                    results[(vi, name, seed)] = nan
                    continue
                med = statistics.median(ts[vi])
                pair = statistics.median([a - b for a, b in zip(ts[vi], ts[0])]) if graphs[0] is not None else nan
                results[(vi, name, seed)] = med
                print(f"[time] v{vi} seed {seed} {name}: med {med:.1f} min {min(ts[vi]):.1f} pair {pair:+.1f} us ({spec})",
                      flush=True)
            del graphs
            torch.cuda.synchronize()
    # table: per variant, per shape: mean over seeds of the median, and delta vs v0
    print("\n=== medians (us), averaged over seeds; delta vs v0 in brackets")
    print("variant".ljust(44) + " ".join(n.rjust(14) for n in args.shapes))
    for vi, spec in enumerate(args.variant):
        row = []
        for name in args.shapes:
            vals = [results[(vi, name, sd)] for sd in args.seeds]
            base = [results[(0, name, sd)] for sd in args.seeds]
            m, b = statistics.mean(vals), statistics.mean(base)
            row.append(f"{m:7.1f}[{m - b:+5.1f}]")
        print(f"v{vi} {spec[:40]}".ljust(44) + " ".join(c.rjust(14) for c in row))


if __name__ == "__main__":
    main()
