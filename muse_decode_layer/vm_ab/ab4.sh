#!/bin/bash
# ab4: the T = 128 / 256 dense routes per layer: Triton (False) / cuBLAS (True) / cuBLAS bf16 out (mm16) / cuBLAS on
# the K-major copy (torchT) / the proj module's m >= 33 route (bfq).  Logs to ~/layer_logs/ab4.log.
cd ~/kerneloptimizer
date -u
~/judge.sh python3 muse_decode_layer/ab_vm.py --seeds 0 1 --iters 25 --shapes v16s v16f v32s v32f \
  --variant base \
  --variant "mm:128.qkv=False;mm:256.qkv=False" \
  --variant "mm:128.qkv=True;mm:256.qkv=True" \
  --variant "mm:128.qkv=mm16;mm:256.qkv=mm16" \
  --variant "mm:128.qkv=torchT;mm:256.qkv=torchT" \
  --variant "mm:128.qkv=bfq;mm:256.qkv=bfq" \
  --variant "mm:128.o=False;mm:256.o=False" \
  --variant "mm:128.o=True;mm:256.o=True" \
  --variant "mm:128.o=mm16;mm:256.o=mm16" \
  --variant "mm:128.o=torchT;mm:256.o=torchT" \
  --variant "mm:128.o=bfq;mm:256.o=bfq" \
  --variant "mm:128.pr=False;mm:256.pr=False" \
  --variant "mm:128.pr=True;mm:256.pr=True" \
  --variant "mm:128.pr=torchT;mm:256.pr=torchT" \
  --variant "mm:128.pr=bfq;mm:256.pr=bfq" \
  --variant "mm:128.un=False;mm:256.un=False" \
  --variant "mm:128.un=True;mm:256.un=True" \
  --variant "mm:128.un=mm16;mm:256.un=mm16" \
  --variant "mm:128.un=torchT;mm:256.un=torchT" \
  --variant "mm:128.un=bfq;mm:256.un=bfq"
date -u
