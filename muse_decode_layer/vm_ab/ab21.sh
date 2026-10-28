#!/bin/bash
# ab21: the attention module's plan families on the chain at the shapes never swept there (sliding B=16/32 at q1 and
# q8, full B=1 q1), one single-shape ab_vm run each (the attplan override applies to every call of a variant), paired,
# seeds 0-2.  The module's own plan is the base.  Logs to ~/layer_logs/ab21_<shape>.log.
cd ~/kerneloptimizer
date -u
run() {  # shape, variants...
  local shape=$1; shift
  local args=(--variant base)
  for v in "$@"; do args+=(--variant "attplan=$v"); done
  args+=(--variant base)
  ~/judge.sh python3 muse_decode_layer/ab_vm.py --seeds 0 1 2 --iters 30 --shapes $shape "${args[@]}" > ~/layer_logs/ab21_$shape.log 2>&1
  date -u
}
# sliding B=16 q1: module plan = plain split-KV, 4 splits per (request, pair), 1 copy (grid 16 x 16 CTAs, 0.65 waves)
run d16s "n_splits:8" "n_splits:16" "n_splits:32" "n_splits:8,nc:4" "n_splits:16,coop:1" "cs:8,ncl:2" "cs:4,ncl:4"
# sliding B=32 q1: module plan = 2 splits
run d32s "n_splits:4" "n_splits:8" "n_splits:16" "n_splits:4,nc:4" "n_splits:8,coop:1" "cs:4,ncl:2" "cs:8,ncl:1"
# sliding B=16 q8: module plan = 4 splits, coop finalize, 1 copy
run v16s "n_splits:8,coop:1" "n_splits:16,coop:1" "n_splits:32,coop:1" "n_splits:8,coop:0,nc:4" "n_splits:8,coop:1,nc:4" "cs:8,ncl:2"
# sliding B=32 q8: tuned plan = 4 splits, coop, 1 copy
run v32s "n_splits:8,coop:1" "n_splits:16,coop:1" "n_splits:2,coop:1" "n_splits:8,coop:0,nc:4" "cs:4,ncl:2" "cs:8,ncl:1"
# full B=1 q1: tuned plan = 252 splits, coop finalize over 8 copies (0.64 waves)
run d1f "n_splits:504,coop:1,nc:8" "n_splits:1008,coop:1,nc:8" "n_splits:252,coop:0,nc:8" "n_splits:252,coop:1,nc:4" "flat:1,n_splits:252,nc:4" "cs:8,ncl:32" "cs:16,ncl:16"
echo AB21 DONE
