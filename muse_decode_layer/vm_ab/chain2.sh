#!/bin/bash
# the second chain (after ab1 / ab4 / ab5 and the locks): the before/after, the sk2 configs, the attention flags
# nohup bash ~/kerneloptimizer/muse_decode_layer/vm_ab/chain2.sh > ~/layer_logs/chain2.log 2>&1 &
D=~/kerneloptimizer/muse_decode_layer/vm_ab
mkdir -p ~/layer_logs
bash $D/ab6.sh > ~/layer_logs/ab6.log 2>&1
bash $D/ab2.sh > ~/layer_logs/ab2.log 2>&1
bash $D/ab3.sh > ~/layer_logs/ab3.log 2>&1
echo CHAIN2 DONE
