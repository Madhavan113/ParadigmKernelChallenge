#!/bin/bash
# ab18 (item 2, the open variant): at T = 1 the fork's m=1 autotuner flips between sk2 and cgemv for qkv / o, and the
# cgemv choice costs ~6 us (ab16).  Pin qkv / o to sk2 and try the unpooler slice as cgemv / sgemv / sk2, each plain
# and (cgemv / sgemv) as a PDL launch (pdlflags=512: weights before its wait, trigger at entry, so it streams under
# the MoE's tail).  Paired, seeds 0-2, T = 1 shapes; the judge's check first for the unpooler-sk2 pin.
# Logs to ~/layer_logs/check18.log, ab18.log.
cd ~/kerneloptimizer
date -u
~/judge.sh python3 muse_decode_layer/check_vm.py --seeds 2 --override "m1:qkv_gate=sk2" --override "m1:o=sk2" \
  --override "m1:unpooler=sk2" --shapes d1s d1f > ~/layer_logs/check18.log 2>&1
date -u
P='m1:qkv_gate=sk2;m1:o=sk2'
~/judge.sh python3 muse_decode_layer/ab_vm.py --seeds 0 1 2 --iters 30 --shapes d1s d1f \
  --variant base \
  --variant "$P;m1:unpooler=cgemv" \
  --variant "$P;m1:unpooler=cgemv;pdlflags=512" \
  --variant "$P;m1:unpooler=sgemv" \
  --variant "$P;m1:unpooler=sgemv;pdlflags=512" \
  --variant "$P;m1:unpooler=sk2" \
  --variant base > ~/layer_logs/ab18.log 2>&1
date -u
echo AB18 DONE
