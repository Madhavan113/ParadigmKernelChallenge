#!/bin/bash
# ab10 (glue v4): graph-replay diagnosis per kernel kind, the judge's correctness check (seeds 2, 3), then the paired
# A/B: CUDA glue vs Triton on all 12 shapes (seeds 0-2) and, for T = 128 / 256, PDL_MAX_T 128 / 256 vs 32 with the
# CUDA glue.  Logs to ~/layer_logs/glue_diag4.log, check_glue4.log, ab10.log, ab10b.log.
cd ~/kerneloptimizer
date -u
~/judge.sh python3 muse_decode_layer/glue_diag.py --seed 2 --shapes d1s v1s d16s > ~/layer_logs/glue_diag4.log 2>&1
~/judge.sh python3 muse_decode_layer/check_vm.py --seeds 2 3 > ~/layer_logs/check_glue4.log 2>&1
date -u
OFF='glue:1=0;glue:8=0;glue:16=0;glue:32=0;glue:128=0;glue:256=0'
~/judge.sh python3 muse_decode_layer/ab_vm.py --seeds 0 1 2 --iters 30 \
  --shapes d1s d1f d16s d16f d32s d32f v1s v1f v16s v16f v32s v32f \
  --variant base \
  --variant "$OFF" \
  --variant base > ~/layer_logs/ab10.log 2>&1
date -u
~/judge.sh python3 muse_decode_layer/ab_vm.py --seeds 0 1 2 --iters 30 --shapes v16s v16f v32s v32f \
  --variant base \
  --variant "pdlmax=128" \
  --variant "pdlmax=256" \
  --variant "glue:128=0;glue:256=0;pdlmax=128" \
  --variant base > ~/layer_logs/ab10b.log 2>&1
date -u
echo AB10 DONE
