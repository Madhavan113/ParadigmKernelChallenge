"""Loop/opcode summary of SASS functions whose mangled name contains a substring, with spill (STL/LDL) positions.
Usage: python3 sass_fn.py sass.txt substring [substring ...]"""
import re, sys, collections

s = open(sys.argv[1]).read()
for f in re.split(r"\n\s*Function : ", s)[1:]:
    name = f.split("\n")[0]
    if not any(x in name for x in sys.argv[2:]):
        continue
    ops = []
    for l in f.split("\n"):
        m = re.search(r"/\*([0-9a-f]{4,5})\*/\s*(@!?U?P\d\s+)?([A-Z0-9_.]+)", l)
        if m:
            ops.append((int(m.group(1), 16), m.group(3), l))
    best = (0, 0, 0)
    for addr, op, l in ops:
        if op.startswith("BRA"):
            tm = re.search(r"0x([0-9a-f]+)\s*\)?\s*;", l)
            if tm:
                tgt = int(tm.group(1), 16)
                if tgt < addr and addr - tgt > best[0]:
                    best = (addr - tgt, tgt, addr)
    body = [op.split(".")[0] for addr, op, l in ops if best[1] <= addr <= best[2]]
    spills = [(addr, op.split(".")[0]) for addr, op, l in ops if op.split(".")[0] in ("STL", "LDL")]
    in_loop = sum(1 for a, _ in spills if best[1] <= a <= best[2])
    print(f"== {name[:90]}\n   total {len(ops)} instrs; largest loop [{best[1]:#x},{best[2]:#x}] = {len(body)} instrs; "
          f"STL/LDL total {len(spills)}, in loop {in_loop}")
    print("   loop:", dict(collections.Counter(body).most_common(16)))
