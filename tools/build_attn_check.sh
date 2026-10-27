#!/bin/bash
# Offline sm_90a build of the attention submission in the judge image (torch-free nvcc, ptxas -v, SASS sizes).
# Usage: tools/build_attn_check.sh <repo root> [-DOKC_X=1 ...]
# Builds across agents are serialized on /tmp/okc_nvcc.lock (Docker has 7.6 GB: concurrent nvcc runs get OOM-killed).
ROOT="$1"; shift
python3 -c 'import fcntl,subprocess,sys; f=open("/tmp/okc_nvcc.lock","w"); fcntl.flock(f, fcntl.LOCK_EX); sys.exit(subprocess.call(sys.argv[1:]))' \
  docker run --rm --platform linux/amd64 -v "$ROOT":/work -w /work lmsysorg/sglang:dev-muse-spark-open \
  python3 tools/nvcc_attn.py "$@"
