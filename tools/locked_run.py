"""Run a command under the shared build lock /tmp/okc_nvcc.lock (flock(2) semantics; macOS has no flock(1)).
Usage: python3 tools/locked_run.py docker run --rm ... """
import fcntl, subprocess, sys
f = open("/tmp/okc_nvcc.lock", "w")
fcntl.flock(f, fcntl.LOCK_EX)
sys.exit(subprocess.call(sys.argv[1:]))
