"""Edit: sk2 cluster (CL = 2) configs for m = 16/32 as autotuned alternatives (SK2_ALT), and the rank > 0 escape
correction sum over every escape warp's copy. Run from the worktree root: python3 tools/_edit_sk2.py"""
import pathlib
p = pathlib.Path("muse_bf16_proj/submission.py")
s = p.read_text()

# 1) ranks > 0 fold ALL escape warps' corrections (NESC copies), as rank 0's final reduce does
old = '''        if (COMP && wk == 0) {
          v[0] += corr[tok * CP + row];
          v[1] += corr[tok * CP + row + 1];
          if (tok + 8 < 32) {
            v[2] += corr[(tok + 8) * CP + row];
            v[3] += corr[(tok + 8) * CP + row + 1];
          }
        }
'''
new = '''        if (COMP && wk == 0) {
#pragma unroll
          for (int c = 0; c < NESC; ++c) {
            v[0] += corr[c * CSZ + tok * CP + row];
            v[1] += corr[c * CSZ + tok * CP + row + 1];
            if (tok + 8 < TOKS) {
              v[2] += corr[c * CSZ + (tok + 8) * CP + row];
              v[3] += corr[c * CSZ + (tok + 8) * CP + row + 1];
            }
          }
        }
'''
assert old in s
s = s.replace(old, new)

# 2) CL = 2 configs: a cluster of two CTAs shares a row block and splits K (DSMEM reduce), so R doubles at the same CTA
# count and every SM ingests half the x bytes (x is re-read per CTA: 512 KB per CTA at m = 32, more than the weights)
old = '''    ('o16t_r68', 'o', 8192, 32, 2, 2, 1, 4, 2, 1),                  # tokr16 [0.937 vs 0.957]
]
'''
new = '''    ('o16t_r68', 'o', 8192, 32, 2, 2, 1, 4, 2, 1),                  # tokr16 [0.937 vs 0.957]
    # cluster-of-2 variants (autotuned against the picks above; x bytes per SM halve, weight bytes unchanged)
    ('qkv32c2a', 'qkv_gate', 2304, 48, 3, 2, 4, 1, 1, 2),           # 48 x 2 = 96 CTAs, 8 consumer warps, 256 k/stage
    ('qkv32c2b', 'qkv_gate', 2304, 48, 3, 2, 2, 2, 1, 2),           # 96 CTAs, 4 consumer warps, 256 k/stage
    ('pooler32c2', 'pooler', 4096, 64, 4, 2, 2, 2, 1, 2),           # 64 x 2 = 128 CTAs
    ('pooler16c2', 'pooler', 4096, 64, 4, 2, 2, 4, 1, 2),           # tokr16: 128 CTAs, 512 k/stage
    ('unpooler32c2', 'unpooler', 8192, 128, 4, 4, 1, 4, 1, 2),      # 64 x 2 = 128 CTAs, 4 consumer warps (no K split inside)
]
SK2_ALT = {  # (layer, tokr) -> alternative config names the autotuner times against SK2_PICK's
    ('qkv_gate', 32): ['qkv32c2a', 'qkv32c2b'],
    ('pooler', 32): ['pooler32c2'],
    ('pooler', 16): ['pooler16c2'],
    ('unpooler', 32): ['unpooler32c2'],
}
'''
assert old in s
s = s.replace(old, new)

old = '''        for tokr in (8, 16, 32):
            if SK2_PICK.get((layer, tokr)) != name:
                continue
'''
new = '''        for tokr in (8, 16, 32):
            if SK2_PICK.get((layer, tokr)) != name and name not in SK2_ALT.get((layer, tokr), ()):
                continue
'''
assert old in s
s = s.replace(old, new)

old = '''def _run_sk2(x, layer, m, n, k, state, dg=0, dbg=None):
    name = _sk2_name(layer, m)
'''
new = '''def _run_sk2(x, layer, m, n, k, state, dg=0, dbg=None, name=None):
    name = name or _sk2_name(layer, m)
'''
assert old in s
s = s.replace(old, new)

old = '''    if kind == "sk2":
        return _run_sk2(x, layer, m, n, k, state)
'''
new = '''    if kind == "sk2":
        return _run_sk2(x, layer, m, n, k, state, name=choice[1] if len(choice) > 1 else None)
'''
assert old in s
s = s.replace(old, new)

old = '''    if (m <= 32 and c[0][0] != "sk2" and _sk2_name(layer, m) in state.get("sk2", {})
            and _route_key(layer, m, "sk2") not in state["disabled"]):
        c.insert(0, ("sk2",))
'''
new = '''    if (m <= 32 and c[0][0] != "sk2" and _sk2_name(layer, m) in state.get("sk2", {})
            and _route_key(layer, m, "sk2") not in state["disabled"]):
        c.insert(0, ("sk2",))
    if m <= 32:
        for alt in SK2_ALT.get((layer, _sk2_tokr(m)), ()):
            if alt in state.get("sk2", {}) and _cand_key(layer, m, ("sk2", alt)) not in state["disabled"]:
                c.insert(0, ("sk2", alt))
'''
assert old in s
s = s.replace(old, new)
p.write_text(s)
print("ok")
