#!/bin/bash
# ab14 (items 2 and 3): PDL trigger placement and the boundary kernel's thread count, paired, T <= 32 shapes, seeds 0-2.
#   gmode=1        the glue kernels trigger launch_dependents after their stores (not at entry)
#   pdlflags=256   the sk2 kernels trigger at entry (not after their main loop)
#   pdlflags=512   the m = 1 cgemv (qkv / o / unpooler slice at T = 1) is a PDL launch: weights before its wait,
#                  trigger at entry (its dependents - attention, boundary 1 / 2 - prefetch during it)
#   pdlflags=768   both
#   gmode=8        boundaries with 1024 threads x 8 columns per token
# First the judge's check with every flag on (the new PDL paths exercised in eager and graph mode), then the A/B.
# Logs to ~/layer_logs/check14.log, ab14.log.
cd ~/kerneloptimizer
date -u
~/judge.sh python3 muse_decode_layer/check_vm.py --seeds 2 --override "pdlflags=768" --override "gmode=9" \
  --shapes d1s d1f v1s v1f d16s d32s v16s v32s > ~/layer_logs/check14.log 2>&1
~/judge.sh python3 muse_decode_layer/check_vm.py --seeds 3 --override "pdlflags=512" --shapes d1s d1f v1s v1f d16s d32s >> ~/layer_logs/check14.log 2>&1
date -u
~/judge.sh python3 muse_decode_layer/ab_vm.py --seeds 0 1 2 --iters 30 --shapes d1s d1f v1s v1f d16s d16f d32s d32f \
  --variant base \
  --variant "gmode=1" \
  --variant "pdlflags=256" \
  --variant "pdlflags=512" \
  --variant "pdlflags=768" \
  --variant "gmode=8" \
  --variant base > ~/layer_logs/ab14.log 2>&1
date -u
echo AB14 DONE
