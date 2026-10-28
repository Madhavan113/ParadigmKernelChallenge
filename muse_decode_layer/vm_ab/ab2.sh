#!/bin/bash
# ab2: the sk2 config candidates per (layer, tokr), every config instantiated at every tokr (OKC_SK2_ALL=1 at build).
# The pooler config only matters under the separate form (pr:*=bfp), so those variants carry it and v1 is their
# reference; the unpooler at tokr 8 is the 1024-row slice config (no alternative).  Logs to ~/layer_logs/ab2.log.
cd ~/kerneloptimizer
date -u
~/judge.sh env OKC_SK2_ALL=1 python3 muse_decode_layer/ab_vm.py --seeds 0 1 --iters 25 --shapes v1s v1f d16s d16f d32s d32f \
  --variant base \
  --variant "pr:8=bfp;pr:16=bfp;pr:32=bfp" \
  --variant "sk2:o.8=o32h_r100;sk2:o.16=o8h_r88;sk2:o.32=o8h_r88" \
  --variant "sk2:o.8=o16t_r68;sk2:o.16=o32h_r100;sk2:o.32=o16t_r68" \
  --variant "pr:8=bfp;pr:16=bfp;pr:32=bfp;sk2:pooler.8=pooler;sk2:pooler.16=pooler;sk2:pooler.32=pooler" \
  --variant "pr:8=bfp;pr:16=bfp;pr:32=bfp;sk2:pooler.8=pooler32_r144;sk2:pooler.16=pooler8h_r72;sk2:pooler.32=pooler16t_r192" \
  --variant "pr:8=bfp;pr:16=bfp;pr:32=bfp;sk2:pooler.8=pooler16t_r192;sk2:pooler.16=pooler32_r144;sk2:pooler.32=pooler8h_r72" \
  --variant "sk2:qkv_gate.8=qkv8_r176;sk2:qkv_gate.16=qkv16_r176;sk2:qkv_gate.32=qkv16_r176" \
  --variant "sk2:qkv_gate.8=qkv16_r176;sk2:qkv_gate.16=qkv8_r176;sk2:qkv_gate.32=qkv16t_r200" \
  --variant "sk2:qkv_gate.8=qkv32_r208;sk2:qkv_gate.16=qkv32_r208;sk2:qkv_gate.32=qkv8_r176" \
  --variant "sk2:qkv_gate.8=qkv16t_r200;sk2:qkv_gate.16=qkv8t_r144;sk2:qkv_gate.32=qkv8t_r144" \
  --variant "sk2:unpooler.16=unpooler16t_r72;sk2:unpooler.32=unpooler16h_r100" \
  --variant "sk2:unpooler.16=unpooler32_r144;sk2:unpooler.32=unpooler16t_r72"
date -u
