#!/bin/bash
# ab12: the judge's 13 correctness tests (TASK.md) on the locked build, eager and graph-replayed, seeds 2 and 3
# (T = 1, 1, 8, 9, 24, 40, 33, 16, 8, 32, 1, 32, 16: the odd token counts, spread mixed, dist outliers, kv_scale
# 1/64, ctx 1048576).  Logs to ~/layer_logs/check_tests.log.
cd ~/kerneloptimizer
date -u
~/judge.sh python3 muse_decode_layer/check_vm.py --seeds 2 3 --shapes \
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
  --spec "batch=16,ctx=20000,layer=full,spread=mixed,dist=soft,page=1" > ~/layer_logs/check_tests.log 2>&1
date -u
echo AB12 DONE
