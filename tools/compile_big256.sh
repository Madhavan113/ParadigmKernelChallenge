#!/bin/bash
# Offline sm_90 compiles of 256-row wgmma tiles (m=256) and a 4-warp 128-row tile; reports spills.
cd /work/tools
run() { echo "### $*"; python3 "$@" 2>&1 | grep -E "^(TMA config|config|ttgir|ptx|regs)|rror" ; }
run compile_proj_tma.py --tiled 256 128 64 4 0 8 4
run compile_proj_tma.py --tiled 256 64 64 4 0 8 4
run compile_proj_tma.py --tiled 256 64 64 2 0 4 4
run compile_proj_tma.py --tiled 128 128 64 4 0 4 6
for f in out_tma_256_*.ptx out_tma_128_128_64_4_0_4_6.ptx; do echo "$f: $(grep -c 'st.local' $f) st.local, $(grep -c 'ld.local' $f) ld.local"; done
