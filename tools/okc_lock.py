#!/usr/bin/env python3
"""flock(1) stand-in for macOS (no flock binary): hold an exclusive BSD lock on a file while running a command.
Interoperates with flock(1) / fcntl.flock users of the same lock file (same flock(2) semantics).
Usage: python3 tools/okc_lock.py [-w SECONDS] LOCKFILE CMD [ARGS...]"""
import fcntl, os, subprocess, sys, time
args = sys.argv[1:]
wait = None
if args and args[0] == "-w":
    wait = float(args[1]); args = args[2:]
lockfile, cmd = args[0], args[1:]
fd = os.open(lockfile, os.O_RDWR | os.O_CREAT, 0o666)
t0 = time.time()
while True:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        break
    except BlockingIOError:
        if wait is not None and time.time() - t0 > wait:
            print(f"okc_lock: timed out after {wait:.0f} s waiting for {lockfile}", file=sys.stderr)
            sys.exit(1)
        time.sleep(2)
print(f"okc_lock: acquired {lockfile} after {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
try:
    rc = subprocess.call(cmd)
finally:
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)
sys.exit(rc)
