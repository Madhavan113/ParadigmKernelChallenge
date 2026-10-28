#!/bin/bash
# ab1: pooler+router form x route mode, the boundary-1 prologue, PDL depth (the T = 1 / 8 / 16 / 32 shapes).
# Run on the rented H200 (judge image via ~/judge.sh, repo at ~/kerneloptimizer); logs to ~/layer_logs/ab1.log.
cd ~/kerneloptimizer
date -u
~/judge.sh python3 muse_decode_layer/ab_vm.py --seeds 0 1 --iters 25 --shapes d1s d1f v1s v1f d16s d16f d32s d32f \
  --variant base \
  --variant "pr:1=bfp;pr:8=bfp;pr:16=bfp;pr:32=bfp" \
  --variant "pr:1=bfpf;pr:8=bfpf;pr:16=bfpf;pr:32=bfpf;rmode:1=split;rmode:8=split;rmode:16=split;rmode:32=split;bnd1:1=0" \
  --variant "pr:1=bfpf;pr:8=bfpf;pr:16=bfpf;pr:32=bfpf;rmode:1=sk2f;rmode:8=sk2f;rmode:16=sk2f;rmode:32=sk2f;bnd1:1=0" \
  --variant "pr:1=bfpf;pr:8=bfpf;pr:16=bfpf;pr:32=bfpf;rmode:1=moe;rmode:8=moe;rmode:16=moe;rmode:32=moe;bnd1:1=0" \
  --variant "pr:1=bfpf;pr:8=bfpf;pr:16=bfpf;pr:32=bfpf;rmode:1=moe_nb;rmode:8=moe_nb;rmode:16=moe_nb;rmode:32=moe_nb;bnd1:1=0" \
  --variant "pr:1=bfpf;rmode:1=split;bnd1:1=1" \
  --variant "pr:1=bfpf;rmode:1=sk2f;bnd1:1=1" \
  --variant "pr:1=bfpf;rmode:1=moe_nb;bnd1:1=1" \
  --variant "pdl:1=3;pdl:8=3;pdl:16=3;pdl:32=3" \
  --variant "pdl:1=4;pdl:8=4;pdl:16=4;pdl:32=4" \
  --variant "pdl:1=6;pdl:8=6;pdl:16=6;pdl:32=6"
date -u
