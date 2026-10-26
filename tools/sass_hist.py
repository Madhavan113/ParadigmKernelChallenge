"""Opcode histogram of selected kernels' SASS (whole function, and the largest backward-branch loop body)."""
import re, subprocess, sys, collections
subprocess.run(["python3", "nvcc_proj.py"], capture_output=True)
subprocess.run("cuobjdump -sass /tmp/proj.o > /tmp/proj.sass", shell=True)
s = open("/tmp/proj.sass").read()
for f in re.split(r"\n\s*Function : ", s)[1:]:
    name = f.split("\n")[0]
    if not any(w in name for w in sys.argv[1:]):
        continue
    ops = []
    for l in f.split("\n"):
        m = re.search(r"/\*([0-9a-f]{4})\*/\s*(@!?P\d\s+)?([A-Z0-9_.]+)", l)
        if m:
            ops.append((int(m.group(1), 16), m.group(3).split(".")[0], l))
    # find the largest loop: a BRA to an earlier address
    best = (0, 0, 0)
    for addr, op, l in ops:
        if op == "BRA":
            t = re.search(r"BRA\s+(?:`\()?0x([0-9a-f]+)", l) or re.search(r"\(\*\"BRANCH_TARGETS[^\"]*\"\*\)|0x([0-9a-f]+)", l)
            tm = re.search(r"0x([0-9a-f]+)\s*\)?\s*;", l)
            if tm:
                tgt = int(tm.group(1), 16)
                if tgt < addr and addr - tgt > best[0]:
                    best = (addr - tgt, tgt, addr)
    body = [op for addr, op, l in ops if best[1] <= addr <= best[2]]
    print("==", re.search(r"\w*mma12_kernel\w*|\w*gemv12_kernel\w*", name).group(0)[:60], "total", len(ops), "instrs; largest loop", best[1:], len(body), "instrs")
    print("   loop:", dict(collections.Counter(body).most_common(14)))
    print("   all :", dict(collections.Counter(op for _, op, _ in ops).most_common(10)))
