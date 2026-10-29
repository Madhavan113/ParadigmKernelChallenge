#!/usr/bin/env python3
"""Paired A/B of attention builds / launch plans on a real GPU, the way the judge times: each variant captured as its
own CUDA graph on the same inputs, the graphs of a shape replayed round-robin, every replay preceded by a 256 MB read
flush of L2. Reports per (variant, shape, seed) the median replay time and the paired delta vs the first variant
(median over rounds of the per-round difference), then a table of means over seeds. Correctness is checked once per
variant and shape with the reference's collect + check_implementation. Runs inside the judge image, repo at /work.

    python3 muse_attention_decode/_attn_ab.py --shapes d16s d32s --seeds 0 1 2 --iters 25 \
        --variant base --variant "pt2:-DOKC_FOO=1" --variant "n8::n_splits=8,coop=1"

Variant spec: name[:nvcc flags[:plan overrides]]; "base" = the shipped default build (no flags, no overrides).
"""
import argparse, concurrent.futures, os, statistics, sys, time

import torch

sys.path.insert(0, "/work/muse_attention_decode")
os.chdir("/work/muse_attention_decode")
import reference as R  # noqa: E402
import submission as S  # noqa: E402

SHAPES = {
    "d1s": "batch=1,ctx=80000,heads=tp8,layer=sliding,page=1,dist=soft",
    "d1f": "batch=1,ctx=80000,heads=tp8,layer=full,page=1,dist=coherent",
    "d16s": "batch=16,ctx=80000,heads=tp8,layer=sliding,page=1,spread=agentic,dist=soft",
    "d16f": "batch=16,ctx=80000,heads=tp8,layer=full,page=1,spread=agentic,dist=coherent",
    "d32s": "batch=32,ctx=80000,heads=tp8,layer=sliding,page=1,spread=agentic,dist=soft",
    "d32f": "batch=32,ctx=80000,heads=tp8,layer=full,page=1,spread=agentic,dist=coherent",
    "v1s": "batch=1,ctx=80000,heads=tp8,layer=sliding,page=1,q_len=8,dist=soft",
    "v1f": "batch=1,ctx=80000,heads=tp8,layer=full,page=1,q_len=8,dist=coherent",
    "v16s": "batch=16,ctx=80000,heads=tp8,layer=sliding,page=1,q_len=8,spread=agentic,dist=soft",
    "v16f": "batch=16,ctx=80000,heads=tp8,layer=full,page=1,q_len=8,spread=agentic,dist=coherent",
    "v32s": "batch=32,ctx=80000,heads=tp8,layer=sliding,page=1,q_len=8,spread=agentic,dist=soft",
    "v32f": "batch=32,ctx=80000,heads=tp8,layer=full,page=1,q_len=8,spread=agentic,dist=coherent",
    # the judge's tests that exercise the per-request paths (page 64, edges, kv_scale)
    "t_s3": "batch=3,ctx=2047,heads=tp8,layer=sliding",
    "t_s4e": "batch=4,ctx=6000,heads=tp8,layer=sliding,dist=edges,page=1",
    "t_s5m": "batch=5,ctx=5000,heads=tp8,layer=sliding,q_len=8,spread=mixed,page=1",
    "t_s4q8": "batch=4,ctx=6000,heads=tp8,layer=sliding,q_len=8,dist=soft,page=1",
    "t_s3k": "batch=3,ctx=4000,heads=tp8,layer=sliding,q_len=8,page=1,kv_scale=0.015625",
    "t_s1": "batch=1,ctx=10,heads=tp8,layer=sliding",
    "t_f4": "batch=4,ctx=3000,heads=tp8,layer=full",
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


def parse_variant(spec):
    parts = spec.split(":")
    name = parts[0]
    flags = tuple(f for f in (parts[1] if len(parts) > 1 else "").split() if f)
    over = {}
    if len(parts) > 2 and parts[2]:
        for kv in parts[2].split(","):
            k, v = kv.split("=")
            over[k] = int(v)
    return name, flags, over


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", default=["d16s", "d32s", "v16s", "v32s"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--variant", action="append", required=True)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--orig", default="", help="path of a second submission file: the variant named 'orig' runs its launcher")
    args = ap.parse_args()
    variants = [parse_variant(v) for v in args.variant]
    mods = {}
    t0 = time.time()
    S2 = None
    if args.orig:
        import importlib.util
        spec = importlib.util.spec_from_file_location("submission_orig", args.orig)
        S2 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(S2)

    def build(v):
        name, flags, _ = v
        if name == "orig":
            return name, S2._build("_orig", flags)
        if name == "base" and not flags:
            return name, S._module()
        return name, S._build(f"_ab_{name}", flags)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        for name, mod in ex.map(build, variants):
            mods[name] = mod
    print(f"[build] {len(variants)} variants in {time.time() - t0:.0f} s", flush=True)
    flush_buf = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    table = {}   # (variant, shape) -> list over seeds of (median, paired delta)
    for shape in args.shapes:
        spec = SHAPES.get(shape, shape)
        for seed in args.seeds:
            data = R.generate_input(seed=seed, **parse_shape(spec))
            graphs = []
            s = torch.cuda.Stream()
            for name, flags, over in variants:
                mod = mods[name]
                SL = S2 if name == "orig" else S   # (the launcher of the file the build came from)
                out = SL._launch(data, _mod=mod, **over)
                torch.cuda.synchronize()
                if not args.no_check:
                    ok, msg = R.check_implementation(data, R.collect(data, out))
                    if not ok:
                        print(f"[check] {name} {shape} seed {seed}: FAIL {msg[:160]}", flush=True)
                    elif seed == args.seeds[0]:
                        print(f"[check] {name} {shape}: ok {msg[:100]}", flush=True)
                with torch.cuda.stream(s):
                    SL._launch(data, _mod=mod, **over)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=s):
                    SL._launch(data, _mod=mod, **over)
                torch.cuda.synchronize()
                graphs.append(g)
            times = [[] for _ in variants]
            for r in range(args.iters):
                order = range(len(variants)) if (r & 1) == 0 else range(len(variants) - 1, -1, -1)
                for vi in order:
                    flush_buf.view(torch.int32).sum()
                    torch.cuda.synchronize()
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                    graphs[vi].replay()
                    e1.record()
                    torch.cuda.synchronize()
                    times[vi].append(e0.elapsed_time(e1) * 1000.0)
            line = []
            for vi, (name, _, _) in enumerate(variants):
                med = statistics.median(times[vi])
                delta = statistics.median(a - b for a, b in zip(times[vi], times[0]))
                table.setdefault((name, shape), []).append((med, delta))
                line.append(f"{name} {med:6.2f} ({delta:+5.2f})")
            print(f"[ab] {shape:5s} seed {seed}: " + "  ".join(line), flush=True)
    print("\n[summary] mean over seeds of the per-seed medians (paired delta vs the first variant)")
    for shape in args.shapes:
        line = []
        for name, _, _ in variants:
            rows = table.get((name, shape), [])
            if rows:
                line.append(f"{name} {statistics.mean(r[0] for r in rows):6.2f} ({statistics.mean(r[1] for r in rows):+5.2f})")
        print(f"  {shape:5s}: " + "  ".join(line))


if __name__ == "__main__":
    main()
