#!/bin/bash
# ab5 (round-robin harness): the order check (base three times), the lock set with T1 separate (L0) vs T1 fused (L1),
# PDL 2 vs 3 under the lock set, and the isolated-tuner dense picks the chain never saw: o at T16/32 (sk2 vs cuBLAS
# bf16 out), the T16 form again, the unpooler slice and o at T1/8, qkv at T<=32.  Logs to ~/layer_logs/ab5.log.
cd ~/kerneloptimizer
L1='pr:1=bfpf;rmode:1=split;bnd1:1=0;pr:8=bfpf;rmode:8=split;pr:16=bfpf;rmode:16=split;pr:32=bfp;pdl:1=2;pdl:8=2;pdl:16=2;pdl:32=2'
L0='pr:1=bfp;bnd1:1=0;pr:8=bfpf;rmode:8=split;pr:16=bfpf;rmode:16=split;pr:32=bfp;pdl:1=2;pdl:8=2;pdl:16=2;pdl:32=2'
date -u
~/judge.sh python3 muse_decode_layer/ab_vm.py --seeds 0 1 --iters 25 --shapes d1s d1f v1s v1f d16s d16f d32s d32f \
  --variant base \
  --variant base \
  --variant "$L0" \
  --variant "$L1" \
  --variant base \
  --variant "$L1;pdl:1=3;pdl:8=3;pdl:16=3;pdl:32=3" \
  --variant "$L1;mm:16.o=mm16;mm:32.o=mm16" \
  --variant "$L1;mm:16.o=bfp;mm:32.o=bfp" \
  --variant "$L1;pr:16=bfp" \
  --variant "$L1;mm:1.o=True;mm:8.o=True" \
  --variant "$L1;mm:1.un=True;mm:8.un=True" \
  --variant "$L1;mm:1.un=False;mm:8.un=False" \
  --variant "$L1;mm:1.qkv=True;mm:8.qkv=True;mm:16.qkv=True;mm:32.qkv=True" \
  --variant "$L1;mm:1.qkv=mm16;mm:8.qkv=mm16;mm:16.qkv=mm16;mm:32.qkv=mm16"
date -u
