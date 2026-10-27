"""Assemble the t12 (TMA bulk-copy 12-bit GEMM) code into muse_bf16_proj/cuda_proj.py and submission.py.
Idempotent: regions between the t12 markers are replaced from tools/t12_kernel.cu.part / t12_wrapper.cu.part.
Run: python3 tools/_edit_t12.py && python3 tools/embed_proj.py"""
import pathlib, re
root = pathlib.Path(__file__).resolve().parent.parent
cp = root / "muse_bf16_proj/cuda_proj.py"
s = cp.read_text()
dev = (root / "tools/t12_kernel.cu.part").read_text()
host = (root / "tools/t12_wrapper.cu.part").read_text()


def put_region(text, begin, end, body, anchor, before=True):
    if begin in text:
        a = text.index(begin); b = text.index(end) + len(end)
        return text[:a] + body.rstrip("\n") + text[b:]
    assert text.count(anchor) == 1, anchor
    i = text.index(anchor)
    return text[:i] + body.rstrip("\n") + "\n\n" + text[i:] if before else text[:i + len(anchor)] + "\n" + body.rstrip("\n") + text[i + len(anchor):]


s = put_region(s, "// == BEGIN t12 device ==", "// == END t12 device ==", dev, "} // namespace\n\nvoid stream_bytes(")
s = put_region(s, "// == BEGIN t12 host ==", "// == END t12 host ==", host, '\n"""\n\nCPP_SRC = """')
# CPP declarations + functions list
decl = ("void bstream(torch::Tensor buf, torch::Tensor out, int64_t grid, int64_t nst, int64_t sb, bool bulk, int64_t smem_alloc);\n"
        "void t12(torch::Tensor X, torch::Tensor Wt, int64_t eb, torch::Tensor esc_roff, torch::Tensor esc_k, torch::Tensor esc_d,\n"
        "         torch::Tensor out, torch::Tensor diag, int64_t n, int64_t k, int64_t r, int64_t ck, int64_t w, int64_t nsw, int64_t nsx, int64_t nt, bool diag_on);\n")
if "void bstream(" not in s.split('CPP_SRC = """')[1]:
    s = s.replace('int64_t mw, int64_t nt, int64_t nstage, int64_t sk, bool pdl, int64_t mode);\n"""', 'int64_t mw, int64_t nt, int64_t nstage, int64_t sk, bool pdl, int64_t mode);\n' + decl + '"""')
if '"w12m", "bstream", "t12"' not in s:
    s = s.replace('"w12b", "w12m"]', '"w12b", "w12m", "bstream", "t12"]')
    s = s.replace('name="okc_proj_cuda13"', 'name="okc_proj_cuda14"')
# python helpers after mma12cp_usable (end of file)
PY = '''

# == BEGIN t12 python ==
def tb12_usable(m: int, n: int, k: int, r: int, ck: int, nt: int) -> bool:
    return 1 <= m <= 32 and m <= 8 * nt and n % r == 0 and k % (128 * ck) == 0


def tb12_nt(m: int) -> int:
    return 1 if m <= 8 else (2 if m <= 16 else 4)


def _repack_t12(w12: torch.Tensor, n: int, k: int, r: int, ck: int) -> torch.Tensor:
    """The w12 fragment layout [n/64, k/128, 4, 3072] regrouped for one bulk copy per (R-row tile, 128*ck-wide chunk):
    uint8 [n/r, k/(128 ck), r/16, ck, 3072] (same bytes; tile 16-row index = T*(r/16) + rt, 128-chunk = c*ck + j)."""
    t16 = w12.view(n // 64, k // 128, 4, 3072).permute(0, 2, 1, 3).reshape(n // 16, k // 128, 3072)
    return t16.view(n // r, r // 16, k // (128 * ck), ck, 3072).permute(0, 2, 1, 3, 4).contiguous()
# == END t12 python ==
'''
if "# == BEGIN t12 python ==" in s:
    a = s.index("\n\n# == BEGIN t12 python =="); b = s.index("# == END t12 python ==") + len("# == END t12 python ==\n")
    s = s[:a] + PY + s[b:]
else:
    s = s.rstrip("\n") + "\n" + PY
cp.write_text(s)
print("cuda_proj.py updated")

# ---------------- submission.py glue ----------------
sp = root / "muse_bf16_proj/submission.py"
t = sp.read_text()


def rep(old, new, count=1):
    global t
    assert t.count(old) == count, (t.count(old), old[:80])
    t = t.replace(old, new)


if "TB12_SHAPES" not in t:
    rep("T12_SHAPES = {}\n",
        "T12_SHAPES = {}\n"
        "# tb12: the single-launch TMA bulk-copy 12-bit GEMM (one CTA per r-row tile over the full K): (layer, n, m) -> (r, ck, w, nsw, nsx)\n"
        "TB12_SHAPES = {}\n"
        "T12_LAYOUTS = {(32, 2), (16, 2), (32, 4)}   # (r, ck) weight layouts kept in state (the bench sweeps them)\n")
    rep("    if (layer, n, m) in M12_SHAPES:\n        mt, rpw, unroll, sk = M12_SHAPES[(layer, n, m)]\n",
        "    if (layer, n, m) in TB12_SHAPES:\n        r, ck, w, nsw, nsx = TB12_SHAPES[(layer, n, m)]\n"
        "        return dict(tb12=True, r=r, ck=ck, w=w, nsw=nsw, nsx=nsx, nt=tb12_nt(m))\n"
        "    if (layer, n, m) in M12_SHAPES:\n        mt, rpw, unroll, sk = M12_SHAPES[(layer, n, m)]\n")
    rep('    if cfg.get("w12b"):\n        return w12b_usable(m, n, k, cfg["nt"], cfg["warps"], cfg["nst"])\n',
        '    if cfg.get("w12b"):\n        return w12b_usable(m, n, k, cfg["nt"], cfg["warps"], cfg["nst"])\n'
        '    if cfg.get("tb12"):\n        return tb12_usable(m, n, k, cfg["r"], cfg["ck"], cfg["nt"])\n')
    rep('    if cfg.get("w12b"):\n        pk = state.get(layer + "_p12")\n',
        '    if cfg.get("tb12"):\n'
        '        pk = state.get(layer + "_p12")\n'
        '        wt = state.get(f"{layer}_t12_{cfg[\'r\']}x{cfg[\'ck\']}")\n'
        '        if pk is None or wt is None:\n'
        '            return _launch(x, w, w2, out, state, _triton_config(layer, m, n, k), m, n, k, layer)\n'
        '        build().t12(x, wt, pk["eb"], pk["esc_roff"], pk["esc_k"], pk["esc_delta"], out, state["diag"],\n'
        '                    n, k, cfg["r"], cfg["ck"], cfg["w"], cfg["nsw"], cfg["nsx"], cfg["nt"], bool(cfg.get("diag")))\n'
        '        return out\n'
        '    if cfg.get("w12b"):\n        pk = state.get(layer + "_p12")\n')
    rep('                        state[name + "_w12"] = _repack_w12(pk, nn_, kk_)\n',
        '                        state[name + "_w12"] = _repack_w12(pk, nn_, kk_)\n'
        '                        layouts = set(T12_LAYOUTS) if BENCH else {(c[0], c[1]) for (l, _, _), c in TB12_SHAPES.items() if l == name}\n'
        '                        for r_, ck_ in sorted(layouts):\n'
        '                            state[f"{name}_t12_{r_}x{ck_}"] = _repack_t12(state[name + "_w12"], nn_, kk_, r_, ck_)\n')
    rep('    state["partials"] = torch.empty((PARTIALS_FLOATS,), dtype=torch.float32, device=device)\n',
        '    state["partials"] = torch.empty((PARTIALS_FLOATS,), dtype=torch.float32, device=device)\n'
        '    state["diag"] = torch.zeros((8,), dtype=torch.int64, device=device)\n')
sp.write_text(t)
print("submission.py updated")
