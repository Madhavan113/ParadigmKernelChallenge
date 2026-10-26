"""nvcc the proj CUDA source (via nvcc_proj.py), then summarize the SASS opcode order of selected
kernels (loads vs FMAs: are UNROLL loads issued back to back?). Run from /work/tools in the image."""
import re, subprocess, sys
subprocess.run(["python3", "nvcc_proj.py"], capture_output=True)
subprocess.run("cuobjdump -sass /tmp/proj.o > /tmp/proj.sass", shell=True)
s = open("/tmp/proj.sass").read()
funcs = re.split(r"\n\s*Function : ", s)
want = sys.argv[1:] or ["stream_kernelILi16ELi256", "stream_kernelILi4ELi256", "gemv_kernelILi1ELi1ELi8ELi1ELb0",
                        "gemv_kernelILi1ELi4ELi4ELi1", "gemv_kernelILi8ELi1ELi4ELi1ELb0", "gemv_kernelILi1ELi1ELi4ELi8ELb1"]
keep = {"LDG", "FFMA", "LOP3", "SHF", "BRA", "BAR", "SHFL", "STG", "PRMT", "FADD", "LDS", "STS", "RED", "ATOMS", "DEPBAR", "LDGSTS"}
for f in funcs[1:]:
    name = f.split("\n")[0]
    if not any(w in name for w in want):
        continue
    ops = [re.sub(r"/\*[0-9a-f]{4}\*/\s*", "", l.strip().split(";")[0]) for l in f.split("\n") if re.search(r"/\*[0-9a-f]{4}\*/", l)]
    seq = []
    for o in ops:
        t = o.split()
        if not t:
            continue
        op = t[1] if t[0].startswith("@") else t[0]
        seq.append(op.split(".")[0])
    out = []
    for op in seq:
        if out and out[-1][0] == op:
            out[-1][1] += 1
        else:
            out.append([op, 1])
    print("==", re.search(r"(stream_kernel|gemv_kernel)\w+", name).group(0)[:48], len(seq), "instrs")
    print("  ", " ".join(f"{o}x{c}" if c > 1 else o for o, c in out if o in keep))
