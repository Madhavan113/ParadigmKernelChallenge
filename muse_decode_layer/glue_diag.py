#!/usr/bin/env python3
"""Localize a CUDA-graph-only failure of the CUDA glue kernels: one prepare, then the judge's check (eager and graph
replay) per shape under each configuration: all kinds / none / boundary 1 only / routing only / boundary 2 only /
all kinds without PDL / all kinds, graph replayed 3x.

    python3 muse_decode_layer/glue_diag.py --seed 2 --shapes d1s v1s
"""
import argparse, importlib, os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ab_vm import SHAPES, parse_shape  # noqa: E402


def check_graph(S, R, st, data, weights, replays=1):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        S.kernel_fn(data, st)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        out = S.kernel_fn(data, st)
    torch.cuda.synchronize()
    for _ in range(replays):
        g.replay()
    torch.cuda.synchronize()
    ok, msg = R.check_implementation(data, R.collect(data, out), weights)
    del g
    return ok, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--shapes", nargs="+", default=["d1s", "v1s"])
    args = ap.parse_args()
    pdir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(pdir)
    R = importlib.import_module("reference")
    S = importlib.import_module("submission")
    torch.cuda.set_device(0)
    weights = R.generate_weights(args.seed)
    st = S.prepare(weights)
    torch.cuda.synchronize()
    pdl0 = S.PDL
    configs = [("none", (), True, 0), ("route-m0", ("r",), True, 0), ("route-m1", ("r",), True, 1),
               ("route-m2", ("r",), True, 2), ("route-m3", ("r",), True, 3), ("all-m1", (1, 2, "r"), True, 1),
               ("all-m3", (1, 2, "r"), True, 3), ("all-m0", (1, 2, "r"), True, 0)]
    for name in args.shapes:
        data = R.generate_input(seed=args.seed, **parse_shape(SHAPES[name]))
        for cname, kinds, pdl, mode in configs:
            st["glue_kinds"] = kinds
            st["glue_mode"] = mode
            S.PDL = pdl
            out = S.kernel_fn(data, st)
            torch.cuda.synchronize()
            ok_e, msg_e = R.check_implementation(data, R.collect(data, out), weights)
            ok_g, msg_g = check_graph(S, R, st, data, weights)
            ok_g3, msg_g3 = check_graph(S, R, st, data, weights, replays=3)
            print(f"[diag] {name} {cname:10s} eager {'ok' if ok_e else 'FAIL'} | graph {'ok' if ok_g else 'FAIL'} | graph x3 "
                  f"{'ok' if ok_g3 else 'FAIL'}  {'' if ok_g else msg_g[:120]}", flush=True)
        S.PDL = pdl0
        st.pop("glue_kinds", None)
        st.pop("glue_mode", None)


if __name__ == "__main__":
    main()
