#!/usr/bin/env python3
"""Run a heavy command (docker nvcc build, the Triton-interpreter test of the layer) under the shared build lock
/tmp/okc_nvcc.lock, so the agents' docker jobs run one at a time (the host has 16 GB, Docker 7.6 GB; two large
builds at once get OOM-killed). flock(2) semantics; macOS has no flock(1).

    python3 tools/okc_locked.py docker run --rm ... lmsysorg/sglang:dev-muse-spark-open ...
"""
import fcntl
import subprocess
import sys

if __name__ == "__main__":
    with open("/tmp/okc_nvcc.lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        sys.exit(subprocess.call(sys.argv[1:]))
