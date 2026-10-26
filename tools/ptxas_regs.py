"""Register/spill usage of PTX files via Triton's bundled ptxas (sm_90a).

Usage (in the judge image): python3 ptxas_regs.py file.ptx [more.ptx ...]
"""
import os, re, subprocess, sys
import triton

ptxas = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "ptxas")
for f in sys.argv[1:]:
    r = subprocess.run([ptxas, "-arch=sm_90a", "-v", "-o", "/tmp/x.cubin", f], capture_output=True, text=True)
    out = r.stdout + r.stderr
    m = re.search(r"Used (\d+) registers", out)
    sp = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", out)
    print(f"{os.path.basename(f):45s} regs={m.group(1) if m else '?'} spill_st={sp.group(1) if sp else '?'} spill_ld={sp.group(2) if sp else '?'}"
          + ("" if r.returncode == 0 else f"  ERROR: {out[-300:]}"))
