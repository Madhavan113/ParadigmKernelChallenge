"""Count PTX instructions per basic block; print the biggest blocks (the hot loop).

Usage: python3 ptxstats.py file.ptx [n_blocks]
"""
import re, sys, collections

ptx = open(sys.argv[1]).read()
blocks = re.split(r"\n(\$L__BB\w+):", ptx)
res = []
for i in range(1, len(blocks), 2):
    label, body = blocks[i], blocks[i + 1]
    ins = [l.strip() for l in body.splitlines()]
    ins = [l for l in ins if re.match(r"^(@!?%p\d+\s+)?[a-z]", l)]
    ops = collections.Counter(re.sub(r"^(@!?%p\d+\s+)?", "", l).split()[0].rstrip(";") for l in ins)
    res.append((len(ins), label, ops, body))
res.sort(reverse=True)
top = int(sys.argv[2]) if len(sys.argv) > 2 else 3
for n, label, ops, body in res[:top]:
    print(f"== {label}: {n} instructions")
    for k, v in ops.most_common(45):
        print(f"   {v:5d} {k}")
    tail = body.strip().splitlines()[-2:]
    print("   ...", " | ".join(t.strip() for t in tail))
