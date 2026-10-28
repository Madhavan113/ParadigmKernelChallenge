#!/bin/bash
# ab6: before / after of the locked build on all 12 shapes, paired (one prepare per seed of the locked build; "before" =
# the 8e96589 tuner's picks of that seed as recorded by the ab1 (T <= 32) and ab4 (T = 128 / 256) prepares, applied as
# overrides; base repeated last as the order check).  Logs to ~/layer_logs/ab6.log.
cd ~/kerneloptimizer
B0='s0:pr:1=bfp;pdl:1=4;pr:8=bfpf;rmode:8=split;pdl:8=2;pr:16=bfp;pdl:16=2;mm:16.o=bfp;pr:32=bfp;pdl:32=2;mm:128.qkv=True;mm:128.o=mm16;mm:128.pr=False;mm:128.un=torchT;mm:256.qkv=mm16;mm:256.o=torchT;mm:256.pr=False;mm:256.un=torchT'
B1='s1:pr:1=bfp;pdl:1=2;pr:8=bfpf;rmode:8=split;pdl:8=3;pr:16=bfpf;rmode:16=split;pdl:16=2;mm:16.o=mm16;pr:32=bfp;pdl:32=2;mm:128.qkv=mm16;mm:128.o=torchT;mm:128.pr=True;mm:128.un=bfq;mm:256.qkv=mm16;mm:256.o=mm16;mm:256.pr=True;mm:256.un=mm16'
date -u
~/judge.sh python3 muse_decode_layer/ab_vm.py --seeds 0 1 --iters 25 \
  --shapes d1s d1f d16s d16f d32s d32f v1s v1f v16s v16f v32s v32f \
  --variant base \
  --variant "$B0|$B1" \
  --variant base
date -u
