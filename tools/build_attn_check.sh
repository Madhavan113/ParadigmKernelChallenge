#!/bin/bash
# Offline sm_90a build of the attention submission in the judge image. Usage: tools/build_attn_check.sh <repo root> [-DOKC_X=1 ...]
ROOT="$1"; shift
docker run --rm --platform linux/amd64 -v "$ROOT":/work -w /work lmsysorg/sglang:dev-muse-spark-open \
  bash -c "TORCH_CUDA_ARCH_LIST=9.0a python3 tools/build_attn_check.py $* 2>&1 | grep -E 'BUILD|SASS|Used|spill|C7515|warning #|error|Compiling entry|Function properties' "
