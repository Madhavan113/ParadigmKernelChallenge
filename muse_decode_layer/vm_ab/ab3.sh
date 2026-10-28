#!/bin/bash
# ab3: the attention build flags (one prepare per flag set and seed; the judge builds with the defaults: QDIRECT 1,
# PK2 1, W0ENABLE 0, PDLKV 0).  The locked build's picks are deterministic (PDL_LOCK / ROUTE_MODE_LOCK / PR_FORM_LOCK /
# PIN_PICK), so a pick flip between prepares cannot confound this build-level A/B.  QDIRECT / PK2 / W0ENABLE are
# q_len-8 paths: verify shapes only.  Logs to ~/layer_logs/ab3.log.
cd ~/kerneloptimizer
V='v1s v1f v16s v16f v32s v32f'
ALL='d1s d1f d16s d16f d32s d32f v1s v1f v16s v16f v32s v32f'
for fl in default -DOKC_PDLKV=1 -DOKC_QDIRECT=0 -DOKC_PK2=0 -DOKC_W0ENABLE=1; do
  date -u; echo "=== ATT flags: $fl"
  if [ "$fl" = default ]; then
    ~/judge.sh python3 muse_decode_layer/ab_vm.py --seeds 0 1 --iters 25 --shapes $ALL --variant base
  elif [ "$fl" = -DOKC_PDLKV=1 ]; then
    ~/judge.sh env OKC_ATT_FLAGS=$fl python3 muse_decode_layer/ab_vm.py --seeds 0 1 --iters 25 --shapes $ALL --variant base
  else
    ~/judge.sh env OKC_ATT_FLAGS=$fl python3 muse_decode_layer/ab_vm.py --seeds 0 1 --iters 25 --shapes $V --variant base
  fi
done
date -u
