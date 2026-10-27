"""Phase 2 bench: g12 rows (m = 128 / 256) vs the current cuBLAS / bmm path, GEMM-only ablation, DIAG phases; appended
to _bench before its final print (idempotent via markers). Run: python3 tools/_edit_g12_bench.py [off]"""
import pathlib, sys
root = pathlib.Path(__file__).resolve().parent.parent
sp = root / "muse_bf16_proj/submission.py"; t = sp.read_text()
BODY = r'''    # == BEGIN g12 bench ==
    if ext is not None and BENCH_G12:
        print("BENCH g12 rb/tb/ck/nsw/nsx/ks (wgmma 12-bit, K split + PDL reduce) vs current; '!' = fp32 rel err > 5e-3; 'g' = GEMM only (no reduce)")
        # (rb, tb, ck, nsw, nsx, ks); ks per layer: 128 CTAs (pooler 32 tiles x 4, qkv 18 x 8, unpooler/o 64 x 2); check e38ce861:
        # the wgmma path is correct; GEMM-only ties cuBLAS at pooler m=128 (23.1 vs 23.3) with the pipe drained per chunk
        g12_cfgs = {128: [(2, 1, 1, 4, 4, 4), (2, 1, 1, 4, 3, 4), (2, 1, 1, 3, 4, 4), (2, 1, 1, 3, 3, 4), (2, 1, 1, 4, 4, 8), (2, 1, 1, 4, 4, 2), (1, 1, 1, 4, 4, 2), (1, 1, 1, 4, 4, 4)],
                    256: []}
        g12_ks = {"pooler": {4: 4, 8: 8, 2: 2}, "qkv_gate": {4: 8, 8: 16, 2: 4}, "unpooler": {4: 2, 8: 4, 2: 1}, "o": {4: 2, 8: 4, 2: 1}}

        def g12cfg(c, layer, mode=0):
            return dict(wg12=True, rb=c[0], tb=c[1], ck=c[2], nsw=c[3], nsx=c[4], ks=g12_ks[layer][c[5]], mode=mode)

        for m in (128, 256):
            for layer, n in (("pooler", 4096), ("qkv_gate", 2304), ("unpooler", 8192), ("o", 8192)):
                if not budget() or not g12_cfgs[m]:
                    break
                parts = [row("current", layer, m, n, None)]
                for c in g12_cfgs[m]:
                    if layer == "qkv_gate" and n % (64 * c[0]) != 0:
                        continue
                    cc = g12cfg(c, layer)
                    parts.append(row("/".join(str(v) for v in c[:5]) + f"/ks{cc['ks']}", layer, m, n, cc))
                    if c in (g12_cfgs[m][0], g12_cfgs[m][3]):
                        parts.append(row("g", layer, m, n, g12cfg(c, layer, 2)))
                print(f"  m={m} {layer:9s} n={n}: " + " | ".join(parts))
        dg = state["diag"]
        for layer, n, m, c in (("pooler", 4096, 128, (2, 1, 1, 4, 4, 4)), ("pooler", 4096, 128, (2, 1, 1, 3, 3, 4)), ("qkv_gate", 2304, 128, (2, 1, 1, 4, 4, 4)),
                               ("unpooler", 8192, 128, (2, 1, 1, 4, 4, 4)), ("pooler", 4096, 128, (1, 1, 1, 4, 4, 4))):
            if not budget():
                break
            x = xs[(m, layers[layer][1])]
            cfg = g12cfg(c, layer); cfg["diag"] = True
            if not _usable(cfg, m, n, layers[layer][1]):
                continue
            res = []
            for _ in range(3):
                dg[4] = -1; dg[5] = 0
                do_flush()
                _OVERRIDE = cfg
                try:
                    _proj_forward(x, layer, n, state)
                finally:
                    _OVERRIDE = None
                torch.cuda.synchronize()
                d = dg.cpu().tolist()
                res.append((d[1] - d[0], d[2] - d[1], d[3] - d[2], d[5] - d[4], d[6] - d[4], d[7] - d[4]))
            r_ = res[-1]
            print(f"  DIAG g12 {layer:9s} m={m:3d} {'/'.join(str(v) for v in c)}: CTA0 fill {r_[0] / 1e3:5.2f} loop {r_[1] / 1e3:5.2f} epi {r_[2] / 1e3:5.2f} us"
                  f" | span {r_[3] / 1e3:5.2f} | last CTA start +{r_[4] / 1e3:5.2f} end +{r_[5] / 1e3:5.2f}   (runs: {', '.join(f'{v[3] / 1e3:.1f}' for v in res)})")
    # == END g12 bench ==
'''
if "# == BEGIN g12 bench ==" in t:
    a = t.index("    # == BEGIN g12 bench ==\n"); b = t.index("    # == END g12 bench ==\n") + len("    # == END g12 bench ==\n")
    t = t[:a] + BODY + t[b:]
else:
    anchor = '    _OVERRIDE = None\n    print(f"BENCH proj done'
    assert t.count(anchor) == 1
    t = t.replace(anchor, BODY + anchor)
if "BENCH_G12" not in t.split("def _bench")[0]:
    t = t.replace("BENCH_T12 = True      # STEP 2: tb12 rows\n", "BENCH_T12 = True      # STEP 2: tb12 rows\nBENCH_G12 = True      # phase 2: g12 rows (m = 128 / 256)\n", 1)
off = len(sys.argv) > 1 and sys.argv[1] == "off"
t = t.replace("BENCH = True        # microbenchmarks", "BENCH = False       # microbenchmarks") if off else t.replace("BENCH = False       # microbenchmarks", "BENCH = True        # microbenchmarks")
t = t.replace("BENCH_T12 = True      # STEP 2", "BENCH_T12 = False     # STEP 2") if not off else t
sp.write_text(t)
print("g12 bench", "off" if off else "on")
