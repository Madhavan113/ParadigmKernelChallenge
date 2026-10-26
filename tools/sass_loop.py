"""Opcode order of the largest loop body of selected kernels (compressed), to see where loads sit vs compute."""
import re, subprocess, sys
subprocess.run(["python3", "nvcc_proj.py"], capture_output=True)
subprocess.run("cuobjdump -sass /tmp/proj.o > /tmp/proj.sass", shell=True)
s = open("/tmp/proj.sass").read()
keep = {"LDG", "HMMA", "FFMA", "LDS", "STS", "BAR", "BRA", "DEPBAR", "LDGSTS", "LDGDEPBAR", "SHFL"}
for f in re.split(r"\n\s*Function : ", s)[1:]:
    name = f.split("\n")[0]
    if not any(w in name for w in sys.argv[1:]):
        continue
    ops = []
    for l in f.split("\n"):
        m = re.search(r"/\*([0-9a-f]{4})\*/\s*(@!?P\d\s+)?([A-Z0-9_.]+)", l)
        if m:
            ops.append((int(m.group(1), 16), m.group(3).split(".")[0], l))
    best = (0, 0, 0)
    for addr, op, l in ops:
        if op == "BRA":
            tm = re.search(r"0x([0-9a-f]+)\s*\)?\s*;", l)
            if tm:
                tgt = int(tm.group(1), 16)
                if tgt < addr and addr - tgt > best[0]:
                    best = (addr - tgt, tgt, addr)
    body = [op for addr, op, l in ops if best[1] <= addr <= best[2]]
    out = []
    other = 0
    for op in body:
        if op in keep:
            if other:
                out.append(f"..{other}"); other = 0
            if out and out[-1].startswith(op + "x"):
                n = int(out[-1][len(op) + 1:]) + 1; out[-1] = f"{op}x{n}"
            elif out and out[-1] == op:
                out[-1] = f"{op}x2"
            else:
                out.append(op)
        else:
            other += 1
    if other:
        out.append(f"..{other}")
    print("==", re.search(r"\w*(mma12cp|mma12|gemv12|skinny_mma|gemv)_kernel\w*", name).group(0)[:70])
    print("  ", " ".join(out))
