#!/usr/bin/env python3
"""Probe: time the dormant 12-bit RS-wgmma 'wide' route (WIDE_BF16) against the m = 128 / 256 picks through the
submission's own cold-L2 graph autotune, without changing production flags (the flag is flipped in this process only).
    python3 tools/wide_probe.py"""
import importlib, os, sys

import torch

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pdir = os.path.join(root, "muse_bf16_proj")
sys.path.insert(0, pdir)
os.chdir(pdir)
R = importlib.import_module("reference")
S = importlib.import_module("submission")
S.WIDE_BF16 = True
S.TUNE_M = (128, 256)
weights = R.generate_weights(0)
state = S.prepare(weights)
for layer, m, tl in state.get("tune_log", []):
    print(f"[probe] {layer} m={m}: " + " ".join(f"{S._cand_name(c)}={t}" for (name, t), c in zip(tl, S._candidates(layer, m, state)))
          + f" -> {S._cand_name(state['choice'][(layer, m)])}", flush=True)
print("[probe] disabled:", sorted(state["disabled"]))
