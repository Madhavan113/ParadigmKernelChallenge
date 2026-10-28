#!/bin/bash
# ab15 (item 4): the MoE TINY build (M <= 4: the T = 1 shapes) with OKC_WSH0 vs the shipped TINY, paired, seeds 0-2;
# the judge's check first with the variant on.  Logs to ~/layer_logs/check15.log, ab15.log.
cd ~/kerneloptimizer
date -u
~/judge.sh python3 muse_decode_layer/check_vm.py --seeds 2 --override "tinyh0=1" --shapes d1s d1f > ~/layer_logs/check15.log 2>&1
date -u
~/judge.sh python3 muse_decode_layer/ab_vm.py --seeds 0 1 2 --iters 30 --shapes d1s d1f \
  --variant base \
  --variant "tinyh0=1" \
  --variant base > ~/layer_logs/ab15.log 2>&1
date -u
echo AB15 DONE
