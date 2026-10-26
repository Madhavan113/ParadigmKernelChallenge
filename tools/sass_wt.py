"""Classify the main-loop SASS of moe_w4a16_wt_kernel variants.

Usage: python3 sass_wt.py sass_file.txt [filter ...]   (filter: template args as "GATED,MW,NT,KS,NSTAGE,SPLITK", e.g. "1,1,2,1,4,0")
For every matching kernel: total instructions, the largest backward-branch loop (the K-chunk loop), its size and
an opcode class histogram (per chunk per warp), plus the compressed opcode order of the loop body.
"""
import re, sys, collections

CLASSES = [
    ("HMMA", ("HMMA",)), ("LDSM", ("LDSM",)), ("LDS", ("LDS",)), ("LDGSTS", ("LDGSTS",)),
    ("LDG", ("LDG",)), ("STS", ("STS",)), ("BAR", ("BAR", "DEPBAR", "LDGDEPBAR", "MEMBAR")),
    ("F2FP/conv", ("F2FP", "F2F", "I2F", "F2I", "I2FP")), ("HMUL2/HFMA2", ("HMUL2", "HFMA2", "HADD2")),
    ("LOP3", ("LOP3",)), ("SHF/SHL/SHR", ("SHF", "SHL", "SHR")), ("PRMT", ("PRMT",)),
    ("IMAD/IADD/LEA", ("IMAD", "IADD3", "LEA", "IMNMX")), ("ISETP/SEL/FSEL", ("ISETP", "SEL", "FSEL", "PLOP3", "FSETP")),
    ("MOV/uniform", ("MOV", "UMOV", "R2UR", "S2R", "S2UR", "UIADD3", "ULDC", "UISETP", "USEL", "ULOP3", "USHF", "UIMAD", "ULEA", "UPRMT")),
    ("BRA/BSSY/EXIT", ("BRA", "BSSY", "BSYNC", "EXIT", "RET", "WARPSYNC")), ("NOP", ("NOP",)),
    ("FMUL/FADD/FFMA", ("FMUL", "FADD", "FFMA", "MUFU")), ("LDC", ("LDC",)),
]


def classify(op):
    base = op.split(".")[0]
    for name, prefixes in CLASSES:
        if base in prefixes:
            return name
    return "other:" + base


def demangle_args(name):
    m = re.search(r"moe_w4a16_wt_kernelI(Lb[01]E)(Li\d+E)(Li\d+E)(Li\d+E)(Li\d+E)(Lb[01]E)", name)
    if not m:
        return name[:60], ""
    vals = [re.sub(r"L[bi](\d+)E", r"\1", g) for g in m.groups()]
    return "wt<GATED=%s,MW=%s,NT=%s,KS=%s,NSTAGE=%s,SPLITK=%s>" % tuple(vals), ",".join(vals)


def main():
    s = open(sys.argv[1]).read()
    filters = sys.argv[2:]
    for f in re.split(r"\n\s*Function : ", s)[1:]:
        name = f.split("\n")[0]
        if "moe_w4a16_wt_kernel" not in name:
            continue
        tag, short = demangle_args(name)
        if filters and short not in filters:
            continue
        ops = []
        for l in f.split("\n"):
            m = re.search(r"/\*([0-9a-f]{4,5})\*/\s*(@!?U?P\d\s+)?([A-Z0-9_.]+)", l)
            if m:
                ops.append((int(m.group(1), 16), m.group(3), l, bool(m.group(2))))
        best = (0, 0, 0)
        for addr, op, l, _ in ops:
            if op.startswith("BRA"):
                tm = re.search(r"0x([0-9a-f]+)\s*\)?\s*;", l)
                if tm:
                    tgt = int(tm.group(1), 16)
                    if tgt < addr and addr - tgt > best[0]:
                        best = (addr - tgt, tgt, addr)
        body = [(op, pred) for addr, op, l, pred in ops if best[1] <= addr <= best[2]]
        hist = collections.Counter(classify(op) for op, _ in body)
        npred = sum(1 for _, p in body if p)
        print(f"== {tag}: {len(ops)} instrs total; loop [{best[1]:#x},{best[2]:#x}] = {len(body)} instrs ({npred} predicated)")
        print("   " + "  ".join(f"{k} {v}" for k, v in sorted(hist.items(), key=lambda kv: -kv[1])))
        keep = {"HMMA", "LDSM", "LDS", "LDGSTS", "BAR", "DEPBAR", "LDGDEPBAR", "F2FP", "HMUL2", "BRA"}
        out = []
        other = 0
        for op, _ in body:
            b = op.split(".")[0]
            if b in keep:
                if other:
                    out.append(f"..{other}")
                    other = 0
                if out and re.match(rf"{b}x\d+$", out[-1]):
                    out[-1] = f"{b}x{int(out[-1][len(b) + 1:]) + 1}"
                elif out and out[-1] == b:
                    out[-1] = f"{b}x2"
                else:
                    out.append(b)
            else:
                other += 1
        if other:
            out.append(f"..{other}")
        print("   order: " + " ".join(out))


main()
