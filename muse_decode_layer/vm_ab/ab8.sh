#!/bin/bash
# ab8: the CUDA glue kernels (boundary 1 / routing / boundary 2 as 8-CTA-per-token / 1-CTA-per-token CUDA kernels)
# vs the Triton single-program kernels, paired, all 12 shapes, seeds 0-2; base (CUDA on) repeated as the order check.
# First the judge's correctness check of the build (check_vm.py, seeds 2 and 3; the glue self-test prints in prepare).
# Logs to ~/layer_logs/check_glue.log and ~/layer_logs/ab8.log.
cd ~/kerneloptimizer
date -u
~/judge.sh python3 muse_decode_layer/check_vm.py --seeds 2 3 > ~/layer_logs/check_glue.log 2>&1
date -u
OFF='glue:1=0;glue:8=0;glue:16=0;glue:32=0;glue:128=0;glue:256=0'
~/judge.sh python3 muse_decode_layer/ab_vm.py --seeds 0 1 2 --iters 30 \
  --shapes d1s d1f d16s d16f d32s d32f v1s v1f v16s v16f v32s v32f \
  --variant base \
  --variant "$OFF" \
  --variant base
date -u
