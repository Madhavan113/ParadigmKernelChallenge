#!/bin/bash
# ab19 (round-2 lock: M1_ROUTE_PIN = all sk2 at m = 1): the judge's check on the 12 scored shapes (seeds 2, 3), the 13
# TASK.md tests (seed 2), then paired on the T = 1 shapes (seeds 0-2): the pinned build vs the GEMV routes the
# autotuner can draw (all cgemv; unpooler cgemv only).  Logs to ~/layer_logs/check19.log, tests19.log, ab19.log.
cd ~/kerneloptimizer
date -u
~/judge.sh python3 muse_decode_layer/check_vm.py --seeds 2 3 > ~/layer_logs/check19.log 2>&1
~/judge.sh python3 muse_decode_layer/check_vm.py --seeds 2 --shapes \
  --spec "batch=1,ctx=10,layer=sliding" \
  --spec "batch=1,ctx=3000,layer=full" \
  --spec "batch=8,ctx=2047,layer=sliding" \
  --spec "batch=9,ctx=5000,layer=sliding,spread=mixed,page=1" \
  --spec "batch=3,ctx=6000,layer=full,q_len=8,dist=outliers" \
  --spec "batch=5,ctx=5000,layer=sliding,q_len=8,spread=mixed,page=1" \
  --spec "batch=33,ctx=20000,layer=full,spread=mixed,page=1" \
  --spec "batch=2,ctx=300000,layer=sliding,q_len=8" \
  --spec "batch=1,ctx=1048576,layer=full,q_len=8" \
  --spec "batch=4,ctx=7000,layer=sliding,q_len=8,page=1,kv_scale=0.015625" \
  --spec "batch=1,ctx=80000,layer=full,dist=soft,page=1" \
  --spec "batch=4,ctx=6000,layer=sliding,q_len=8,dist=soft" \
  --spec "batch=16,ctx=20000,layer=full,spread=mixed,dist=soft,page=1" > ~/layer_logs/tests19.log 2>&1
date -u
~/judge.sh python3 muse_decode_layer/ab_vm.py --seeds 0 1 2 --iters 30 --shapes d1s d1f \
  --variant base \
  --variant "m1:qkv_gate=cgemv;m1:o=cgemv;m1:unpooler=cgemv" \
  --variant "m1:unpooler=cgemv" \
  --variant base > ~/layer_logs/ab19.log 2>&1
date -u
echo AB19 DONE
