#!/bin/bash
# ab11 (locked build: CUDA glue v4 everywhere + PDL_MAX_T 128): correctness (judge's check, seeds 2 and 3; graph
# diagnosis at T = 128 where PDL is new), then the paired before/after on all 12 shapes, seeds 0-2: "before" =
# Triton glue + PDL_MAX_T 32 (the feb6313 behaviour) as overrides, base repeated as the order check.
# Logs to ~/layer_logs/check_lock.log, glue_diag_lock.log, ab11.log.
cd ~/kerneloptimizer
date -u
~/judge.sh python3 muse_decode_layer/check_vm.py --seeds 2 3 > ~/layer_logs/check_lock.log 2>&1
~/judge.sh python3 muse_decode_layer/glue_diag.py --seed 2 --shapes v16s v32s > ~/layer_logs/glue_diag_lock.log 2>&1
date -u
BEFORE='glue:1=0;glue:8=0;glue:16=0;glue:32=0;glue:128=0;glue:256=0;pdlmax=32'
~/judge.sh python3 muse_decode_layer/ab_vm.py --seeds 0 1 2 --iters 30 \
  --shapes d1s d1f d16s d16f d32s d32f v1s v1f v16s v16f v32s v32f \
  --variant base \
  --variant "$BEFORE" \
  --variant base > ~/layer_logs/ab11.log 2>&1
date -u
echo AB11 DONE
