#!/bin/bash
# ab9: T = 128 / 256 with the CUDA glue: PDL for the attention / MoE / glue kernels at T > 32 (PDL_MAX_T 256; the
# cuBLAS nodes stay plain launches) vs the current PDL_MAX_T 32, each with the CUDA and the Triton glue; paired, seeds
# 0-2, base (CUDA glue, PDL_MAX_T 32) repeated as the order check.  Logs to ~/layer_logs/ab9.log.
cd ~/kerneloptimizer
date -u
~/judge.sh python3 muse_decode_layer/ab_vm.py --seeds 0 1 2 --iters 30 --shapes v16s v16f v32s v32f \
  --variant base \
  --variant "pdlmax=256" \
  --variant "glue:128=0;glue:256=0" \
  --variant "glue:128=0;glue:256=0;pdlmax=256" \
  --variant base
date -u
