"""Edit (final): retire the judged-losing alternatives (small-R and cluster-of-2 sk2 configs, cgemv geometries) so
prepare builds no dead streams; the candidate mechanisms (SK2_ALT / CG_ALT / GQX_CONFIGS lists) stay for later rounds.
Run: python3 tools/_edit_prune.py"""
import pathlib, re
p = pathlib.Path("muse_bf16_proj/submission.py")
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


# sk2: drop the measured-losing config lines (keep the fused `pooler` config), empty SK2_ALT
a = s.index("    # cluster-of-2 variants (autotuned against the picks above")
b = s.index("    # the decode layer's fused pooler + router launch (sk2f, FUSE)")
s = s[:a] + '''    # retired after judging (prefill-slot runs f3e50d8b / ba7affa0, raw graph-replay us vs the pick): cluster-of-2 K splits
    # at R 48-128 (qkv32 16.7-17.7 vs 14.4, pooler32 22.1 vs 19.1, pooler16 20.5 vs 18.0, unpooler32 21.8 vs 18.5) and
    # small-R rings at 3-4 CTAs per SM (pooler8 R8 23.2 vs 18.1, pooler16 R8/R16 24.3/19.3 vs 18.5, pooler32 R8/R16
    # 36.2/22.4 vs 19.6, qkv8/16/32 R8 14.4/15.3/21.0 vs 13.6/13.3/14.4, unpooler16/32 R16 22.8/25.4 vs 17.8/18.8,
    # o8/16/32 R16 11.2/11.5/12.8 vs 9.4/9.3/9.5, slice8 R8 9.4 vs 8.1): neither halving the x bytes per SM nor more
    # bytes in flight per SM beats the picks above; per-CTA fixed costs and x re-reads dominate at small R.
''' + s[b:]
a = s.index("SK2_ALT = {  # (layer, tokr) -> alternative config names the autotuner times against SK2_PICK's")
b = s.index("}\n", a) + 2
s = s[:a] + '''SK2_ALT = {  # (layer, tokr) -> alternative config names the autotuner times against SK2_PICK's (see the retired list)
}
''' + s[b:]
# cgemv: empty CG_ALT (judged ba7affa0: o 2x64 8.6 vs 4x64 8.8 / 8x64 9.1 / 4x32 9.4 / 2x32 9.7; unpooler 2x128 7.04 vs
# 2x256 7.07 / 4x128 7.2 / 4x256 7.2 / 8x128 8.1; qkv 2x256 12.4 vs 4x128 12.9 / 2x128 13.6 / 4x256 13.8)
rep('''CG_ALT = {"o": [(4, 64), (8, 64), (4, 32), (2, 32)], "unpooler": [(4, 128), (8, 128), (4, 256), (2, 256)],
          "qkv_gate": [(4, 256), (2, 128), (4, 128)]}
''', '''# judged (ba7affa0, raw us): o 2x64 8.6 beats 4x64 8.8, 8x64 9.1, 4x32 9.4, 2x32 9.7; unpooler 2x128 7.04 beats 2x256
# 7.07, 4x128 / 4x256 7.2, 8x128 8.1; qkv 2x256 12.4 beats 4x128 12.9, 2x128 13.6, 4x256 13.8 -> no alternatives kept
CG_ALT = {}
''')
p.write_text(s)
# sanity: the retired names must not be referenced anywhere else
for name in ("qkv32c2a", "pooler8q_r48", "unpooler32c2", "slice8q_r48"):
    assert name not in s or s.count(name) <= 1, name
print("ok")
