#!/bin/bash
# the whole A/B chain on the VM: nohup bash ~/kerneloptimizer/muse_decode_layer/vm_ab/chain.sh > ~/layer_logs/chain.log 2>&1 &
D=~/kerneloptimizer/muse_decode_layer/vm_ab
mkdir -p ~/layer_logs
bash $D/ab1.sh > ~/layer_logs/ab1.log 2>&1
bash $D/ab4.sh > ~/layer_logs/ab4.log 2>&1
bash $D/ab2.sh > ~/layer_logs/ab2.log 2>&1
bash $D/ab3.sh > ~/layer_logs/ab3.log 2>&1
echo CHAIN DONE
