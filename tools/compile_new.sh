#!/bin/bash
# Offline sm_90 compiles of the configs the check-1 bench introduces (run inside the judge image, from /work/tools).
cd /work/tools
run() { echo "### $*"; python3 "$@" 2>&1 | grep -E "^(TMA config|config|ttgir|ptx|regs)" ; }
run compile_proj_tma.py --tiled 16 32 128 1 0 4 8
run compile_proj_tma.py --tiled 16 32 128 1 0 4 12
run compile_proj_tma.py --tiled 16 32 128 1 0 4 8 --swap
run compile_proj.py --tiled 16 32 128 1 0 4 8
run compile_proj_tma.py --tiled 32 64 128 1 0 4 6
run compile_proj_tma.py --tiled 128 128 64 4 0 8 6
run compile_proj_tma.py --tiled 128 64 64 4 0 4 8 --swap
run compile_proj_tma.py --tiled 128 256 64 8 0 8 4
run compile_proj_tma.py 128 128 64 4 0 8 6
run compile_proj_tma.py 128 32 64 16 1 8 4
run compile_proj_tma.py 128 32 64 16 1 8 6
run compile_proj_tma.py --tiled 128 32 64 16 1 8 6
run compile_proj_tma.py --tiled 128 32 128 16 1 8 4
run compile_proj_tma.py --tiled 16 32 64 16 1 4 8
