#!/bin/bash
# ab3: the attention build flags (one prepare per flag set and seed; the judge builds with the defaults: QDIRECT 1,
# PK2 1, W0ENABLE 0, PDLKV 0).  The tuner's picks are pinned (the seed-0 picks of the 8e96589 baseline) so a pick
# flip between prepares cannot confound a build-level A/B.  QDIRECT / PK2 / W0ENABLE are q_len-8 paths: verify
# shapes only.  Logs to ~/layer_logs/ab3.log.
cd ~/kerneloptimizer
PIN='pr:1=bfp;bnd1:1=0;pr:8=bfpf;rmode:8=split;pr:16=bfp;pr:32=bfp;pdl:1=2;pdl:8=2;pdl:16=2;pdl:32=2;mm:128.qkv=bfq;mm:128.o=mm16;mm:128.pr=True;mm:128.un=torchT;mm:256.qkv=bfq;mm:256.o=mm16;mm:256.pr=False;mm:256.un=mm16'
V='v1s v1f v16s v16f v32s v32f'
ALL='d1s d1f d16s d16f d32s d32f v1s v1f v16s v16f v32s v32f'
for fl in default -DOKC_PDLKV=1 -DOKC_QDIRECT=0 -DOKC_PK2=0 -DOKC_W0ENABLE=1; do
  date -u; echo "=== ATT flags: $fl"
  if [ "$fl" = default ]; then
    ~/judge.sh python3 muse_decode_layer/ab_vm.py --seeds 0 1 --iters 25 --shapes $ALL --variant "$PIN"
  elif [ "$fl" = -DOKC_PDLKV=1 ]; then
    ~/judge.sh env OKC_ATT_FLAGS=$fl python3 muse_decode_layer/ab_vm.py --seeds 0 1 --iters 25 --shapes $ALL --variant "$PIN"
  else
    ~/judge.sh env OKC_ATT_FLAGS=$fl python3 muse_decode_layer/ab_vm.py --seeds 0 1 --iters 25 --shapes $V --variant "$PIN"
  fi
done
date -u
