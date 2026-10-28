#!/bin/bash
# ab7: the two sub-1-us tokr-16 config leads of ab2 together (o8h_r88 for o, unpooler32_r144 for the unpooler at
# tokr 16; each 0.2-1.3 us at both seeds alone) vs the current picks, paired, on the T = 16 shapes; every config
# instantiated (OKC_SK2_ALL=1).  Logs to ~/layer_logs/ab7.log.
cd ~/kerneloptimizer
date -u
~/judge.sh env OKC_SK2_ALL=1 python3 muse_decode_layer/ab_vm.py --seeds 0 1 --iters 30 --shapes d16s d16f \
  --variant base \
  --variant "sk2:o.16=o8h_r88;sk2:unpooler.16=unpooler32_r144" \
  --variant "sk2:o.16=o8h_r88" \
  --variant "sk2:unpooler.16=unpooler32_r144" \
  --variant "sk2:o.16=o32h_r100;sk2:unpooler.16=unpooler32_r144" \
  --variant base
date -u
