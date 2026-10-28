#!/bin/bash
# usage: nsys_chain.sh <name> <shape-name> [ab_vm variant spec]  -> ~/ncu_out/chain_<name>.nsys-rep + _cuda_gpu_trace.csv
# A graph-replay timeline (per-node kernel tracing) of the layer on one scored shape through the A/B harness: one
# prepare, 10 replays of the given variant (default base).  Parse with vm_ab/nsys_rows.py.
name=$1; shape=$2; var=${3:-base}
cd ~/kerneloptimizer
~/judge.sh bash -c "nsys profile -t cuda --cuda-graph-trace=node -f true -o /ncu_out/chain_$name python3 muse_decode_layer/ab_vm.py --seeds 0 --iters 10 --shapes $shape --variant '$var' > /ncu_out/chain_$name.log 2>&1; nsys stats --report cuda_gpu_trace --format csv -o /ncu_out/chain_$name /ncu_out/chain_$name.nsys-rep > /ncu_out/chain_$name.stats.log 2>&1"
echo "done $name"
