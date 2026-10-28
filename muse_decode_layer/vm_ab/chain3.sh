#!/bin/bash
# the remaining batches after the locks (ab6 done): the sk2 configs, then the attention flags
# nohup bash ~/kerneloptimizer/muse_decode_layer/vm_ab/chain3.sh > ~/layer_logs/chain3.log 2>&1 < /dev/null &
D=~/kerneloptimizer/muse_decode_layer/vm_ab
mkdir -p ~/layer_logs
bash $D/ab2.sh > ~/layer_logs/ab2.log 2>&1
bash $D/ab3.sh > ~/layer_logs/ab3.log 2>&1
echo CHAIN3 DONE
