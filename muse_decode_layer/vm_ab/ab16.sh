#!/bin/bash
# ab16 (item 2, the m = 1 GEMVs at T = 1): with the m = 1 routes pinned (qkv / o -> cgemv, unpooler -> sgemv; and the
# all-sk2 alternative), the PDL launch of those GEMVs (pdlflags=512: weights before the wait, trigger at entry) and the
# sk2 entry trigger (256), paired, seeds 0-2, on the T = 1 shapes.  The judge's check first with everything on.
# Logs to ~/layer_logs/check16.log, ab16.log.
cd ~/kerneloptimizer
date -u
~/judge.sh python3 muse_decode_layer/check_vm.py --seeds 2 --override "pdlflags=768" --override "m1:qkv_gate=cgemv" \
  --override "m1:o=cgemv" --override "m1:unpooler=sgemv" --shapes d1s d1f > ~/layer_logs/check16.log 2>&1
~/judge.sh python3 muse_decode_layer/check_vm.py --seeds 3 --override "pdlflags=768" --override "m1:qkv_gate=sk2" \
  --override "m1:o=sk2" --override "m1:unpooler=cgemv" --shapes d1s d1f >> ~/layer_logs/check16.log 2>&1
date -u
G='m1:qkv_gate=cgemv;m1:o=cgemv;m1:unpooler=sgemv'
K='m1:qkv_gate=sk2;m1:o=sk2;m1:unpooler=cgemv'
~/judge.sh python3 muse_decode_layer/ab_vm.py --seeds 0 1 2 --iters 30 --shapes d1s d1f \
  --variant base \
  --variant "$G" \
  --variant "$G;pdlflags=512" \
  --variant "$G;pdlflags=768" \
  --variant "$K" \
  --variant "$K;pdlflags=256" \
  --variant "$K;pdlflags=768" \
  --variant base > ~/layer_logs/ab16.log 2>&1
date -u
echo AB16 DONE
