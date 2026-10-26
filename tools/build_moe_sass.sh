#!/bin/bash
# Offline nvcc of tools/moe_kernel_only.cu in the judge image: ptxas -v summary + SASS dump.
# Usage: tools/build_moe_sass.sh <repo root> <sass output name under tools/>
ROOT="$1"; OUT="${2:-sass_wt.txt}"
docker run --rm --platform linux/amd64 -v "$ROOT":/work -w /work/tools lmsysorg/sglang:dev-muse-spark-open bash -c \
  "nvcc -arch=sm_90a -O3 --use_fast_math -cubin -Xptxas -v -o /tmp/moe.cubin moe_kernel_only.cu 2>&1 | grep -E 'error|Compiling|Used|spill' | grep -v 'moe_w4a16_gemm_kernel' ; cuobjdump -sass /tmp/moe.cubin > $OUT; wc -l $OUT"
