"""Rename the wgmma GEMM's config key g12 -> wg12 (g12 is the gemv12 path's key: check dbd64ace lost to the collision)
in submission.py and the generator scripts. Run: python3 tools/_fix_g12_key.py"""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
reps = [
    ("return dict(g12=True, rb=rb,", "return dict(wg12=True, rb=rb,"),
    ('    if cfg.get("g12"):\n        return g12_usable(', '    if cfg.get("wg12"):\n        return g12_usable('),
    ('    if cfg.get("g12"):\n        pk = state.get(layer + "_p12")\n        wt = state.get(f"{layer}_t12_{64 * cfg[\'rb\']}',
     '    if cfg.get("wg12"):\n        pk = state.get(layer + "_p12")\n        wt = state.get(f"{layer}_t12_{64 * cfg[\'rb\']}'),
    ("cfg = dict(g12=True, rb=c[0], tb=c[1], ck=c[2], nsw=c[3], nsx=c[4], ks=c[5])", "cfg = dict(wg12=True, rb=c[0], tb=c[1], ck=c[2], nsw=c[3], nsx=c[4], ks=c[5])"),
    ("return dict(g12=True, rb=c[0], tb=c[1], ck=c[2], nsw=c[3], nsx=c[4], ks=c[5], mode=mode)", "return dict(wg12=True, rb=c[0], tb=c[1], ck=c[2], nsw=c[3], nsx=c[4], ks=c[5], mode=mode)"),
]
for name in ("muse_bf16_proj/submission.py", "tools/_edit_g12.py", "tools/_edit_g12_bench.py"):
    p = root / name; s = p.read_text(); n = 0
    for old, new in reps:
        c = s.count(old)
        if c:
            s = s.replace(old, new); n += c
    # the generator scripts carry the same strings inside their templates (escaped quotes differ): cover those too
    for old, new in [("dict(g12=True, rb=", "dict(wg12=True, rb="), ('cfg.get(\\"g12\\")', 'cfg.get(\\"wg12\\")'), ("if cfg.get(\"g12\"):\\n        return g12_usable", "if cfg.get(\"wg12\"):\\n        return g12_usable")]:
        c = s.count(old)
        if c:
            s = s.replace(old, new); n += c
    p.write_text(s)
    print(name, n, "replacements")
s = (root / "muse_bf16_proj/submission.py").read_text()
assert s.count('cfg.get("wg12")') == 2 and "dict(wg12=True, rb=" in s and "return dict(g12=True, rb=" not in s, "wg12 key not applied everywhere"
print("ok")
