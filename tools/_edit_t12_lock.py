"""Round 6 of t12: lock pooler m=8/16 on t12e (check d59a9b55: 14.9 / 15.8 us vs 17.9 / 17.8), trim the instantiations
to the tuning neighbourhood, bench m=8/16 only. Run: python3 tools/_edit_t12_lock.py [final]
  final: BENCH off, instantiate only the locked configs."""
import pathlib, sys
root = pathlib.Path(__file__).resolve().parent.parent
final = len(sys.argv) > 1 and sys.argv[1] == "final"
w = root / "tools/t12_wrapper.cu.part"; s = w.read_text()
a = s.index("    CUtensorMap xmap;\n"); b = s.index("#undef T12C_NT")
if final:
    cases = ("    CUtensorMap xmap;\n"
             "    if (variant == 5) xmap = okc_x_map3(X.data_ptr(), m, (int)k, ldx, 8 * (int)nt, 2 * (int)ck);\n"
             "    T12E_CASE(16, 4, 4, 3, 3, 1)\n    T12E_CASE(16, 4, 4, 3, 3, 2)\n")
else:
    cfgs = [(16, 4, 4, 4, 4), (16, 4, 4, 4, 3), (16, 4, 4, 5, 4), (16, 4, 4, 5, 3), (16, 4, 4, 6, 4), (16, 4, 4, 6, 3),
            (16, 4, 4, 3, 4), (16, 4, 4, 3, 3), (16, 4, 8, 4, 4), (16, 4, 8, 4, 3), (16, 4, 8, 5, 4), (16, 4, 8, 6, 3)]
    cases = ("    CUtensorMap xmap;\n"
             "    if (variant == 5) xmap = okc_x_map3(X.data_ptr(), m, (int)k, ldx, 8 * (int)nt, 2 * (int)ck);\n")
    for c in cfgs:
        cases += "    T12E_CASE(%d, %d, %d, %d, %d, 1) T12E_CASE(%d, %d, %d, %d, %d, 2)\n" % (c + c)
    cases += "    T12E_CASE_D(16, 4, 4, 4, 4, 1) T12E_CASE_D(16, 4, 4, 4, 3, 2)\n"
s = s[:a] + cases + s[b:]
w.write_text(s)

sp = root / "muse_bf16_proj/submission.py"; t = sp.read_text()
a = t.index("TB12_SHAPES = {"); b = t.index("\n", a) + 1
if final:
    t = t[:a] + ('TB12_SHAPES = {("pooler", 4096, 8): (16, 4, 4, 3, 3), ("pooler", 4096, 16): (16, 4, 4, 3, 3), ("unpooler", 8192, 16): (16, 4, 4, 3, 3)}\n'
                 '# t12e: 16-row tiles, 4 compute warps + producer, 3 weight + 3 x stages. Judge (check 8d975106, 4/4 and 4/3 rings): pooler m=8 15.2 vs 18.2,\n'
                 '# m=16 16.5 vs 18.0; bench 3/3 rings: pooler 14.8 / 15.8, unpooler m=16 16.9 vs 18.2. qkv / o / m=1 / m=32 / the unpooler slice lose or tie.\n') + t[b:]
else:
    t = t[:a] + ('TB12_SHAPES = {("pooler", 4096, 8): (16, 4, 4, 4, 4), ("pooler", 4096, 16): (16, 4, 4, 4, 3)}   # t12e (check d59a9b55 bench: 14.9 / 15.8 vs 17.9 / 17.8); the other shapes lose or tie\n') + t[b:]
t = t.replace("T12_LAYOUTS = {(32, 2), (16, 2), (32, 4), (16, 4)}", "T12_LAYOUTS = {(16, 4)}")
if final:
    t = t.replace("BENCH = True        # microbenchmarks", "BENCH = False       # microbenchmarks")
else:
    t = t.replace("BENCH = False       # microbenchmarks", "BENCH = True        # microbenchmarks")
sp.write_text(t)

eb = root / "tools/_edit_t12_bench.py"; s = eb.read_text()
a = s.index("        cfgs = ["); b = s.index("]\n", a) + 2
s = s[:a] + ("        cfgs = [(16, 4, 4, 4, 4, 5), (16, 4, 4, 4, 3, 5), (16, 4, 4, 5, 4, 5), (16, 4, 4, 5, 3, 5), (16, 4, 4, 6, 4, 5), (16, 4, 4, 6, 3, 5),\n"
             "                (16, 4, 4, 3, 4, 5), (16, 4, 4, 3, 3, 5), (16, 4, 8, 4, 4, 5), (16, 4, 8, 4, 3, 5), (16, 4, 8, 5, 4, 5), (16, 4, 8, 6, 3, 5)]\n") + s[b:]
s = s.replace("        for m in (8, 16, 32, 1):\n", "        for m in (8, 16):\n")
s = s.replace('for c in ((32, 4, 8, 5, 3, 5), (16, 4, 8, 4, 3, 5))))', 'for c in ((16, 4, 4, 4, 4, 5),)))')
a = s.index("        for layer, n, m, c in ((\"pooler\""); b = s.index("            if not budget():", a)
s = s[:a] + ('        for layer, n, m, c in (("pooler", 4096, 8, (16, 4, 4, 4, 4, 5)), ("pooler", 4096, 16, (16, 4, 4, 4, 3, 5)), ("unpooler", 8192, 16, (16, 4, 4, 4, 3, 5)),\n'
             '                               ("qkv_gate", 2304, 8, (16, 4, 4, 4, 4, 5))):\n') + s[b:]
eb.write_text(s)
print("ok", "final" if final else "tuning")
