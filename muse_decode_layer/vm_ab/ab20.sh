#!/bin/bash
# ab20 (boundary folds, tail v3): the judge's check with both tails on at T = 1 and 8 (scored T<=8 shapes + the T<=8
# tests), then paired, seeds 0-2, on d1s d1f v1s v1f: base (separate v4 glue kernels) vs the unpooler-slice tail
# (boundary 2), the o-projection tail (boundary 1) and both, at T = 1 and at T = 8.
# Logs to ~/layer_logs/check20.log, ab20.log.
cd ~/kerneloptimizer
date -u
ON='btail:1.unpooler=1;btail:1.o=1;btail:8.unpooler=1;btail:8.o=1'
~/judge.sh python3 muse_decode_layer/check_vm.py --seeds 2 3 --override "$ON" --shapes d1s d1f v1s v1f \
  --spec "batch=1,ctx=10,layer=sliding" --spec "batch=1,ctx=3000,layer=full" --spec "batch=8,ctx=2047,layer=sliding" \
  --spec "batch=1,ctx=1048576,layer=full,q_len=8" --spec "batch=1,ctx=80000,layer=full,dist=soft,page=1" > ~/layer_logs/check20.log 2>&1
date -u
~/judge.sh python3 muse_decode_layer/ab_vm.py --seeds 0 1 2 --iters 30 --shapes d1s d1f v1s v1f \
  --variant base \
  --variant "btail:1.unpooler=1" \
  --variant "btail:1.o=1" \
  --variant "btail:1.unpooler=1;btail:1.o=1" \
  --variant "btail:8.unpooler=1" \
  --variant "btail:8.o=1" \
  --variant "btail:8.unpooler=1;btail:8.o=1" \
  --variant base > ~/layer_logs/ab20.log 2>&1
date -u
echo AB20 DONE
