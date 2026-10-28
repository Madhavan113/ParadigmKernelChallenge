"""patch20 follow-up: the boundary self-test's two tail calls per layer (the correctness call and the isolated timing)
take (alpha, beta) instead of the gate, and run at T = 1 and 8.  Run once from the worktree root after patch20.py."""
import pathlib

root = pathlib.Path(__file__).resolve().parents[2]
g = root / "muse_decode_layer/layer_glue.py"
s = g.read_text()
i = s.index("def _boundary_selftest(st):")
j = s.index("\n\n\n", i)
blk = s[i:j]
old_t = "    for T in (1,):   # the v2 tails hold the whole row in registers: one row"
old1 = """(st["gate1"], stream, None, st["g_preffn"], s1a, ha))"""
old2 = """(st["gate2"], s1a, st["g_postffn"], st["g_next"], s2a, outa))"""
assert blk.count(old_t) == 1 and blk.count(old1) == 2 and blk.count(old2) == 2, (blk.count(old_t), blk.count(old1), blk.count(old2))
blk2 = (blk.replace(old_t, "    for T in (1, 8):   # the v3 tails loop over the rows of one CTA")
           .replace(old1, """(st["ab1"][0], st["ab1"][1], stream, None, st["g_preffn"], s1a, ha))""")
           .replace(old2, """(st["ab2"][0], st["ab2"][1], s1a, st["g_postffn"], st["g_next"], s2a, outa))"""))
s = s[:i] + blk2 + s[j:]
g.write_text(s)
print("patched")
