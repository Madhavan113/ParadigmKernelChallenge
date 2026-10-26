"""Hot-loop (the loop containing HMMA) instruction counts and CALL counts of the dumped w12 kernel SASS files.
Usage: python3 w12_loops.py [prefix=after]"""
import re, sys, glob, os
prefix = sys.argv[1] if len(sys.argv) > 1 else "after"
root = os.path.dirname(os.path.abspath(__file__))


def loop_count(fn):
    ops = []
    for ln in open(fn).read().splitlines():
        m = re.match(r'\s+/\*([0-9a-f]+)\*/\s+((?:@!?U?P\w+\s+)?[A-Z][A-Z0-9_.]*.*?);', ln)
        if m:
            ops.append((int(m.group(1), 16), m.group(2).strip()))
    backs = [(int(m.group(1), 16), a) for a, o in ops for m in [re.match(r'(?:@!?U?P\w+\s+)?BRA (0x[0-9a-f]+)', o)] if m and int(m.group(1), 16) < a]
    best = None
    for lo, hi in backs:
        body = [o for a, o in ops if lo <= a <= hi]
        if any('HMMA' in o for o in body) and (best is None or len(body) < len(best)):
            best = body
    return (len(best) if best else None), sum(1 for _, o in ops if 'CALL' in o)


for fn in sorted(glob.glob(os.path.join(root, f"sass_{prefix}_nt*_mode0.txt"))):
    print(f"{os.path.basename(fn):36s} hot loop {loop_count(fn)[0]}  CALLs {loop_count(fn)[1]}")
