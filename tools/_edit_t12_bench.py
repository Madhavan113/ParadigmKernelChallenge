"""Replace the in-prepare bench plan in muse_bf16_proj/submission.py with the t12 plan: bulk / cp.async stream sweep
over the 12-bit pooler payload, tb12 rows per shape vs the current path (fp32 ref, table output, probe rows), DIAG.
Run: python3 tools/_edit_t12_bench.py"""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
sp = root / "muse_bf16_proj/submission.py"
t = sp.read_text()
a = t.index('    TAB = {"table": True}\n')
b = t.index('    _OVERRIDE = None\n    print(f"BENCH proj done')
NEW = r'''    TAB = {"table": True}
    print("BENCH proj, judge protocol (us per replay minus fixed cost; '!' = rel err above tol)")
    if ext is not None and BENCH_STREAM:
        # ---- STEP 1: streaming-read ceiling through a shared-memory ring over the real 12-bit pooler payload
        buf = state.get("pooler_w12")
        nbytes = int(buf.numel() * buf.element_size())
        print(f"BENCH stream {nbytes / 1e6:.1f} MB (12-bit pooler payload): us (TB/s) per stages x stage KB, c<n> = CTAs per SM")
        combos = [(4, 8), (8, 8), (12, 8), (4, 16), (6, 16), (8, 16), (2, 32), (4, 32), (6, 32)]   # (check 11ed8ae1: c4 rows + cp.async done)
        for bulk in (True,):
            for nst, kb in combos:
                sb = kb * 1024
                parts = []
                for cpsm in (1, 2):
                    need = nst * sb
                    cap = min(233472 // cpsm - 1024, 232448 - 4096)
                    if need > cap:
                        parts.append(f"c{cpsm}   n/a      ")
                        continue
                    alloc = max(need, 233472 // (cpsm + 1) - 1024 + 1024)   # pins at most cpsm CTAs per SM
                    alloc = min(alloc, cap)
                    grid = 132 * cpsm
                    try:
                        tt = timeit(lambda: ext.bstream(buf, sink, grid, nst, sb, bulk, alloc), reps=9) - t_empty
                        read = grid * ((nbytes // grid) // sb) * sb
                        parts.append(f"c{cpsm} {tt:5.1f} ({read / tt / 1e6:.2f})")
                    except Exception as e:
                        parts.append(f"c{cpsm} fail")
                        print("   stream fail", nst, kb, cpsm, str(e).splitlines()[0][:80])
                print(f"  {'bulk    ' if bulk else 'cp.async'} {nst:2d}x{kb:2d}KB: " + " | ".join(parts))
                if not budget():
                    break
    if ext is not None and BENCH_T12:
        # ---- STEP 2: the single-launch TMA 12-bit GEMM vs the current path
        print("BENCH tb12 r/ck/w/nsw/nsx vs current; '!' = fp32 rel err > 5e-3; '~' = differs from the table output by > 2e-3")
        # (r, ck, w, nsw, nsx, variant): v1 = t12 (check 11ed8ae1: x bulk copies per row throttle the ring), v2 = t12b
        cfgs = [(32, 4, 8, 4, 2, 3), (32, 4, 8, 5, 2, 3), (32, 4, 8, 6, 2, 3), (32, 4, 8, 5, 3, 3), (32, 2, 8, 8, 3, 3), (32, 2, 8, 12, 3, 3),
                (16, 4, 8, 4, 2, 3), (16, 4, 4, 4, 2, 3), (16, 4, 4, 6, 2, 3), (16, 2, 4, 8, 3, 3)]

        def tb(c, m):
            return dict(tb12=True, r=c[0], ck=c[1], w=c[2], nsw=c[3], nsx=c[4], nt=tb12_nt(m), v=c[5])

        def probe(layer, n, c, m=8):
            # judge-style probe rows: x[i] = one-hot at the k of escape i -> out[i, :] == W[:, k] exactly (bf16)
            global _OVERRIDE
            pk = state.get(layer + "_p12")
            if pk is None or not tb12_usable(m, n, layers[layer][1], c[0], c[1], tb12_nt(m)):
                return "probe n/a"
            kk = pk["esc_k"][:m].long()
            if kk.numel() < m:
                kk = torch.cat([kk, torch.arange(m - kk.numel(), device=dev)])
            xp = torch.zeros((m, layers[layer][1]), dtype=torch.bfloat16, device=dev)
            xp[torch.arange(m, device=dev), kk] = 2.0
            _OVERRIDE = tb(c, m)
            try:
                o = _proj_forward(xp, layer, n, state)
            finally:
                _OVERRIDE = None
            want = (state[layer][:n].float()[:, kk].T * 2.0)
            rel = ((o.float() - want).abs() / want.abs().clamp(min=1e-30)).max().item()
            return f"probe {'ok' if rel < 1e-3 else 'FAIL'} ({rel:.0e})"

        def row2(label, layer, m, n, cfg, ref_tab):
            global _OVERRIDE
            x = xs[(m, layers[layer][1])]
            _OVERRIDE = cfg
            try:
                if not _usable(cfg, m, n, layers[layer][1]):
                    return f"{label} n/a"
                out = _proj_forward(x, layer, n, state)
                tt = timeit(lambda: _proj_forward(x, layer, n, state)) - t_empty
                flag = check(out, ref_of(layer, m, n), layer)
                if ref_tab is not None:
                    d = float((out.float() - ref_tab.float()).norm() / ref_tab.float().norm())
                    if d > 2e-3:
                        flag += f"~{d:.0e}"
                return f"{label} {tt:5.1f}{flag}"
            except Exception as e:
                print("   fail", layer, m, label, str(e).splitlines()[0][:90])
                return f"{label} fail"
            finally:
                _OVERRIDE = None

        for m in (8, 16, 32, 1):
            for layer, n in (("pooler", 4096), ("unpooler", 8192), ("qkv_gate", 2304), ("o", 8192)):
                if not budget():
                    break
                if layer == "unpooler" and m <= 8:
                    n = 1024
                x = xs[(m, layers[layer][1])]
                _OVERRIDE = TAB
                try:
                    ref_tab = _proj_forward(x, layer, n, state)
                finally:
                    _OVERRIDE = None
                parts = [row("current", layer, m, n, None)]
                for c in cfgs:
                    parts.append(row2("/".join(str(v) for v in c), layer, m, n, tb(c, m), ref_tab))
                print(f"  m={m:2d} {layer:9s} n={n}: " + " | ".join(parts))
            if m == 8:
                for layer, n in (("pooler", 4096), ("qkv_gate", 2304), ("unpooler", 1024), ("o", 8192)):
                    print(f"  probe {layer:9s}: " + " | ".join(f"{c[0]}/{c[1]} {probe(layer, n, c)}" for c in ((32, 4, 8, 5, 2, 3), (16, 4, 8, 4, 2, 3))))
        # ---- DIAG: %globaltimer phases of CTA 0 (fill = first stage landed, loop, epilogue) and the CTA span
        dg = state["diag"]
        for layer, n, m, c in (("pooler", 4096, 8, (32, 4, 8, 5, 2, 3)), ("pooler", 4096, 32, (32, 4, 8, 5, 2, 3)), ("pooler", 4096, 8, (16, 4, 8, 4, 2, 3)), ("pooler", 4096, 16, (32, 4, 8, 5, 2, 3)),
                               ("pooler", 4096, 32, (16, 4, 8, 4, 2, 3)), ("qkv_gate", 2304, 8, (16, 4, 8, 4, 2, 3)), ("unpooler", 8192, 16, (32, 4, 8, 5, 2, 3)),
                               ("o", 8192, 8, (32, 4, 8, 5, 2, 3)), ("pooler", 4096, 1, (32, 4, 8, 5, 2, 3)), ("qkv_gate", 2304, 32, (16, 4, 8, 4, 2, 3))):
            if not budget():
                break
            x = xs[(m, layers[layer][1])]
            cfg = tb(c, m); cfg["diag"] = True
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
                t0_, t1_, t2_, t3_ = d[0], d[1], d[2], d[3]
                res.append((t1_ - t0_, t2_ - t1_, t3_ - t2_, (d[5] & 0xFFFFFFFFFFFFFFFF) - d[4], d[6] - d[4], d[7] - d[4]))
            r_ = res[-1]
            print(f"  DIAG {layer:9s} m={m:2d} {'/'.join(str(v) for v in c)}: CTA0 fill {r_[0] / 1e3:5.2f} loop {r_[1] / 1e3:5.2f} epi {r_[2] / 1e3:5.2f} us"
                  f" | span {r_[3] / 1e3:5.2f} | last CTA start +{r_[4] / 1e3:5.2f} end +{r_[5] / 1e3:5.2f}   (runs: {', '.join(f'{v[3] / 1e3:.1f}' for v in res)})")
'''
t = t[:a] + NEW + t[b:]
t = t.replace("BENCH_BUDGET_S = 170\n", "BENCH_BUDGET_S = 170\nBENCH_STREAM = True   # STEP 1: bulk-copy stream sweep\nBENCH_T12 = True      # STEP 2: tb12 rows\n", 1) if "BENCH_STREAM" not in t else t
sp.write_text(t)
print("bench updated")
