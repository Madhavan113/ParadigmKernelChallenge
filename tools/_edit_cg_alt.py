"""Edit: cgemv (rows per CTA x threads) alternatives as autotuned m = 1 candidates (CG_ALT), with a generated dispatch
(the C if-chain becomes @@CG_DISPATCH@@) and one 12-bit plane set per (layer, rows). Run: python3 tools/_edit_cg_alt.py"""
import pathlib
p = pathlib.Path("muse_bf16_proj/submission.py")
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


rep('''  if (K == 8192 && rows == 2 && nth == 256) cgemv_kernel<2, 256, 8192><<<N / 2, 256 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);
  else if (K == 1024 && rows == 2 && nth == 64) cgemv_kernel<2, 64, 1024><<<N / 2, 64 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);
  else if (K == 4096 && rows == 2 && nth == 128) cgemv_kernel<2, 128, 4096><<<N / 2, 128 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);
  else if (K == 8192 && rows == 2 && nth == 512) cgemv_kernel<2, 512, 8192><<<N / 2, 512 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);
  else TORCH_CHECK(false, "cgemv: no instantiation for K=", K, " rows=", rows, " nth=", nth);
''', '''  @@CG_DISPATCH@@
  TORCH_CHECK(false, "cgemv: no instantiation for K=", K, " rows=", rows, " nth=", nth);
''')
rep('''                    + _gq_src(sm90) + _gqx_src(sm90) + RGEMV_SRC + _rk_src(sm90))
''', '''                    + _gq_src(sm90) + _gqx_src(sm90) + _rgemv_src() + _rk_src(sm90))
''')
rep('''CG_CONFIGS = {"qkv_gate": (2, 256), "o": (2, 64), "unpooler": (2, 128), "pooler": (2, 512)}   # layer -> (rows per CTA, threads); m = 1
# (v137: pooler m=1 cgemv 2x512 measured -0.34 us vs sk2; 2x256 ties sk2, 4/8 rows or 1 row lose)
''', '''CG_CONFIGS = {"qkv_gate": (2, 256), "o": (2, 64), "unpooler": (2, 128), "pooler": (2, 512)}   # layer -> (rows per CTA, threads); m = 1
# (v137: pooler m=1 cgemv 2x512 measured -0.34 us vs sk2; 2x256 ties sk2, 4/8 rows or 1 row lose)
# alternatives the autotuner times against CG_CONFIGS (m = 1): more rows per CTA = fewer CTAs and fewer x re-reads (o at
# 2 x 64: 4096 CTAs of 3 KB each re-reading 2 KB of x: as many x bytes as weight bytes); K >= 16 threads per CTA
CG_ALT = {"o": [(4, 64), (8, 64), (4, 32), (2, 32)], "unpooler": [(4, 128), (8, 128), (4, 256), (2, 256)],
          "qkv_gate": [(4, 256), (2, 128), (4, 128)]}
CG_K = {"qkv_gate": 8192, "o": 1024, "unpooler": 4096, "pooler": 8192}


def _cg_configs(layer):
    return [CG_CONFIGS[layer]] + list(CG_ALT.get(layer, ()))


def _rgemv_src():
    lines = []
    for layer in CG_CONFIGS:
        k = CG_K[layer]
        for rows, nth in _cg_configs(layer):
            lines.append(f"if (K == {k} && rows == {rows} && nth == {nth}) {{ cgemv_kernel<{rows}, {nth}, {k}><<<N / {rows}, {nth} + 32, 0, st>>>"
                         f"(a0, a1, xp, tp, pm, op, b4); C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }}")
    return RGEMV_SRC.replace("@@CG_DISPATCH@@", "\\n  ".join(dict.fromkeys(lines)))
''')
rep('''            if name in CG_CONFIGS:
                wc = w[:1024] if name == "unpooler" else w   # the unpooler at m = 1 is the 1024-row slice
                cg = cg_compress(wc.contiguous(), CG_CONFIGS[name][0])
                if cg is not None:
                    state.setdefault("cg", {})[name] = cg
''', '''            if name in CG_CONFIGS:
                wc = w[:1024] if name == "unpooler" else w   # the unpooler at m = 1 is the 1024-row slice
                for rows in sorted({r for r, _ in _cg_configs(name)}):   # one plane set per rows-per-CTA (its table)
                    cg = cg_compress(wc.contiguous(), rows)
                    if cg is not None:
                        state.setdefault("cg", {})[(name, rows)] = cg
''')
rep('''    if kind == "cgemv":   # m = 1: 12-bit compressed GEMV (qkv, o, unpooler slice, pooler)
        cg = state["cg"][layer]
        rows, nth = CG_CONFIGS[layer]
''', '''    if kind == "cgemv":   # m = 1: 12-bit compressed GEMV (qkv, o, unpooler slice, pooler)
        rows, nth = choice[1] if len(choice) > 1 else CG_CONFIGS[layer]
        cg = state["cg"][(layer, rows)]
''')
rep('''    if m == 1 and layer in state.get("cg", {}) and _route_key(layer, m, "cgemv") not in state["disabled"]:
        c.insert(0, ("cgemv",))
''', '''    if m == 1 and layer in CG_CONFIGS:
        for cfg in _cg_configs(layer):
            if (layer, cfg[0]) in state.get("cg", {}) and _cand_key(layer, m, ("cgemv", cfg)) not in state["disabled"]:
                c.insert(0, ("cgemv", cfg))
''')
rep('''    if ch[0] == "gqx":
        return (layer, "gqx", _gqx_bn(m), ch[1])
''', '''    if ch[0] == "gqx":
        return (layer, "gqx", _gqx_bn(m), ch[1])
    if ch[0] == "cgemv" and len(ch) > 1:
        return (layer, "cgemv", ch[1])
''')
rep('''    if c[0] == "sk2" and len(c) > 1:
        return f"sk2/{c[1]}"
    return c[0]
''', '''    if c[0] == "sk2" and len(c) > 1:
        return f"sk2/{c[1]}"
    if c[0] == "cgemv" and len(c) > 1:
        return f"cgemv/{c[1][0]}x{c[1][1]}"
    return c[0]
''')
p.write_text(s)
print("ok")
