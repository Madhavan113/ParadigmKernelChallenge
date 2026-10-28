#!/bin/bash
# ab13 (item 1: the routing's 16-bit SIMD rank pre-pass): graph diagnosis of the route kernel in both rank modes, the
# judge's check (seeds 2, 3), then paired: base (SIMD pre-pass + exact pass over the 16-bit candidates) vs gmode=4
# (the previous exact 64-bit rank loop) on the T <= 32 shapes, seeds 0-2.  Logs to ~/layer_logs/glue_diag13.log,
# check13.log, ab13.log.
cd ~/kerneloptimizer
date -u
~/judge.sh python3 muse_decode_layer/glue_diag.py --seed 2 --shapes d1s v1s d32s > ~/layer_logs/glue_diag13.log 2>&1
~/judge.sh python3 muse_decode_layer/check_vm.py --seeds 2 3 > ~/layer_logs/check13.log 2>&1
date -u
~/judge.sh python3 muse_decode_layer/ab_vm.py --seeds 0 1 2 --iters 30 --shapes d1s d1f v1s v1f d16s d16f d32s d32f \
  --variant base \
  --variant "gmode=4" \
  --variant base > ~/layer_logs/ab13.log 2>&1
date -u
echo AB13 DONE
