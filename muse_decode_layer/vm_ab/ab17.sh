#!/bin/bash
# ab17 (item 5): the attention plan families at B = 1 sliding q_len 1 (d1s only: the override applies to every call of
# a variant), paired, seeds 0-2: the module's plan (plain split-KV, 32 one-tile splits, 1 workspace copy) vs clusters
# of 8 / 16 (DSMEM reduction), the cooperative finalize over 4 copies, 16 two-tile splits, 64 splits.  The judge's
# check first with each plan.  Logs to ~/layer_logs/check17.log, ab17.log.
cd ~/kerneloptimizer
date -u
for pl in "cs:8" "cs:16" "coop:1,nc:4" "n_splits:16" "n_splits:64,nc:4"; do
  ~/judge.sh python3 muse_decode_layer/check_vm.py --seeds 2 --override "attplan=$pl" --shapes d1s >> ~/layer_logs/check17.log 2>&1
done
date -u
~/judge.sh python3 muse_decode_layer/ab_vm.py --seeds 0 1 2 --iters 30 --shapes d1s \
  --variant base \
  --variant "attplan=cs:8" \
  --variant "attplan=cs:16" \
  --variant "attplan=coop:1,nc:4" \
  --variant "attplan=n_splits:16" \
  --variant "attplan=n_splits:64,nc:4" \
  --variant base > ~/layer_logs/ab17.log 2>&1
date -u
echo AB17 DONE
