#!/usr/bin/env python3
"""Correctness of the layer on a GPU, the judge's way: prepare(weights) once per seed, then per shape one eager step
and one CUDA-graph replay of kernel_fn, each packed with reference.collect (the three outputs + the cache entries the
step wrote) and judged by reference.check_implementation.  Prints the prepare phases (the glue prints them) and one
[check] line per shape and path.

    python3 muse_decode_layer/check_vm.py --seeds 2 --shapes d1s d1f ... (default: all 12 scored shapes)
"""
import argparse, importlib, os, sys, time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ab_vm import SHAPES, parse_shape, apply  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[2])
    ap.add_argument("--shapes", nargs="*", default=list(SHAPES))
    ap.add_argument("--spec", action="append", default=[], help="raw shape spec k=v,... (added to --shapes)")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--override", action="append", default=[], help="ab_vm override spec applied after prepare")
    args = ap.parse_args()
    cases = [(n, SHAPES[n]) for n in args.shapes] + [(f"spec{i}", sp) for i, sp in enumerate(args.spec)]
    pdir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(pdir)
    R = importlib.import_module("reference")
    S = importlib.import_module("submission")
    torch.cuda.set_device(0)
    bad = 0
    for seed in args.seeds:
        weights = R.generate_weights(seed)
        t0 = time.perf_counter()
        st = S.prepare(weights)
        torch.cuda.synchronize()
        print(f"[prepare] seed {seed}: {time.perf_counter() - t0:.1f} s", flush=True)
        for spec in args.override:
            apply(S, st, spec, [])
            print(f"[override] {spec}", flush=True)
        for name, spec in cases:
            data = R.generate_input(seed=seed, **parse_shape(spec))
            T = data["config"]["tokens"] if "config" in data and "tokens" in data["config"] else -1
            name = f"{name}[T={T} {spec}]"
            out = S.kernel_fn(data, st)
            torch.cuda.synchronize()
            ok, msg = R.check_implementation(data, R.collect(data, out), weights)
            bad += not ok
            print(f"[check] seed {seed} {name} eager: {'ok' if ok else 'FAIL'} {msg}", flush=True)
            if args.no_graph:
                continue
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                S.kernel_fn(data, st)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                gout = S.kernel_fn(data, st)
            torch.cuda.synchronize()
            g.replay()
            torch.cuda.synchronize()
            ok, msg = R.check_implementation(data, R.collect(data, gout), weights)
            bad += not ok
            print(f"[check] seed {seed} {name} graph: {'ok' if ok else 'FAIL'} {msg}", flush=True)
            del g
    print(f"[check] {'ALL OK' if bad == 0 else f'{bad} FAILED'}", flush=True)


if __name__ == "__main__":
    main()
