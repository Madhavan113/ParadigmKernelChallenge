"""Edit (round 2, after scored run f3e50d8b): gqx configs that fit the judged cluster capacities (cl 2: 66, cl 3: 39,
cl 4: 30, cl 6: 17, cl 8: 15 active clusters at ~200 KB smem); the losing sk2 cluster-of-2 alternatives taken out of
SK2_ALT; an sk2 attribution bench (SK2_DIAG compiled in; dg modes) printed from prepare behind DIAG_SK2.
Run from the worktree root: python3 tools/_edit_round2.py"""
import pathlib
p = pathlib.Path("muse_bf16_proj/submission.py")
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


a = s.index("GQX_CONFIGS = {")
b = s.index("USE_GQX = True")
s = s[:a] + '''GQX_CONFIGS = {
    # judged (run f3e50d8b, raw graph replay us): qkv m128 (1,3,8) 17.9 | (2,6,6) 29.0 (18 clusters of 6 > 17 active: second
    # wave) | (3,8,5) 21.8; pooler m128 gq(1,2,8) 24.5 | (2,4,6) 37.3 (32 > 30: wave) | (4,8,4) 43.6 (16 > 15: wave);
    # unpooler m128 torchT 22.6 | (2,2,6) 24.6 | (4,4,4) 42.1 (wave); o m128 torchT 10.4 | (2,2,6) 10.5 | (1,1,8) 12.8;
    # m256: qkv (1,3,5) 24.2 vs torch 22.8; pooler (2,4,4) 49 / (2,2,4) 56 vs cutlass 31; unpooler (2,2,4) 32 vs 27.3;
    # o (2,2,4) 13.3 / (1,1,5) 16.6 vs 12.4. Max active clusters (smem ~200 KB): cl2 66, cl3 39, cl4 30, cl6 17, cl8 15.
    ("qkv_gate", 128): [(1, 3, 8),    # v141: 36 row blocks x 3 = 108 CTAs
                        (2, 5, 6)],   # 18 x 5 = 90 CTAs (cl 5 capacity unknown: printed)
    ("qkv_gate", 256): [(1, 3, 5)],
    ("pooler", 128): [(2, 3, 6)],     # 32 x 3 = 96 CTAs, 32 clusters <= 39
    ("pooler", 256): [(1, 2, 5)],     # 64 x 2 = 128 CTAs, NWG 1 (NWG 2 at BN 256 ran at 35% of peak)
    ("unpooler", 128): [(1, 1, 8),    # 128 CTAs, no cluster
                        (2, 2, 6),    # 128 CTAs (24.6 vs torchT 22.6)
                        (4, 3, 4)],   # 32 x 3 = 96 CTAs
    ("unpooler", 256): [(1, 1, 5)],   # 128 CTAs, no cluster
    ("o", 128): [(2, 2, 6)],          # tie with torchT
    ("o", 256): [(2, 2, 4)],
}
''' + s[b:]

rep('''SK2_ALT = {  # (layer, tokr) -> alternative config names the autotuner times against SK2_PICK's
    ('qkv_gate', 32): ['qkv32c2a', 'qkv32c2b'],
    ('pooler', 32): ['pooler32c2'],
    ('pooler', 16): ['pooler16c2'],
    ('unpooler', 32): ['unpooler32c2'],
}
''', '''SK2_ALT = {  # (layer, tokr) -> alternative config names the autotuner times against SK2_PICK's
    # judged (run f3e50d8b, raw us): qkv32c2a 17.7 / qkv32c2b 16.7 vs qkv32_r208 14.4; pooler32c2 22.1 vs 19.1;
    # pooler16c2 20.5 vs 18.0; unpooler32c2 21.8 vs 18.5: halving the x bytes per SM never paid for the cluster epilogue
}
''')

# ---- sk2 attribution bench: SK2_DIAG compiled in when DIAG_SK2 (the dg bits become runtime branches)
rep('''def _sk2_src(sm90):
    if not sm90:
''', '''DIAG_SK2 = True   # measurement runs: compile the sk2 diagnostics in and time the dg modes from prepare (see _diag_sk2)
DIAG_SK2_SHAPES = [("pooler", 8), ("pooler", 16), ("pooler", 32), ("qkv_gate", 8), ("qkv_gate", 32), ("unpooler", 16),
                   ("unpooler", 32), ("o", 16)]
DIAG_SK2_MODES = [(0, "full"), (1, "no-epilogue"), (2, "no-compute"), (4, "no-x"), (6, "loads-only"), (8, "alu-not-mma"),
                  (16, "no-ldmatrix")]


def _sk2_src(sm90):
    if not sm90:
''')
rep('''    return (SK2_SRC.replace("@@SK2_DISPATCH@@", "\\n  ".join(lines))
            .replace("@@SK2F_DISPATCH@@", "\\n  ".join(flines)))
''', '''    src = SK2_SRC.replace("@@SK2_DISPATCH@@", "\\n  ".join(lines)).replace("@@SK2F_DISPATCH@@", "\\n  ".join(flines))
    if DIAG_SK2:
        src = "#define SK2_DIAG 1\\n" + src
    return src


def _diag_sk2(weights, state):
    """Attribution of the sk2 kernels on this GPU: graph replays from a cold L2 (as the autotuner) of each dg mode."""
    if not (DIAG_SK2 and state.get("sm90")):
        return
    dev = weights["o"].device
    flush = torch.ones(64 * 1024 * 1024 // 1024, 1024, dtype=torch.float32, device=dev)
    sink = torch.empty(1024, dtype=torch.float32, device=dev)
    g = torch.Generator(device=dev)
    g.manual_seed(999)
    for layer, m in DIAG_SK2_SHAPES:
        w = weights[layer]
        n = 1024 if (layer == "unpooler" and m <= 8) else w.shape[0]
        k = w.shape[1]
        if _sk2_name(layer, m) not in state.get("sk2", {}):
            continue
        x = torch.randn(m, k, device=dev, generator=g).to(torch.bfloat16)
        res = []
        for dg, name in DIAG_SK2_MODES:
            try:
                _run_sk2(x, layer, m, n, k, state, dg=dg)
                torch.cuda.synchronize()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    _run_sk2(x, layer, m, n, k, state, dg=dg)
                ts = []
                for _ in range(12):
                    torch.sum(flush, dim=0, out=sink)
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                    gr.replay()
                    e1.record()
                    torch.cuda.synchronize()
                    ts.append(e0.elapsed_time(e1) * 1000)
                res.append(f"{name}={sorted(ts)[len(ts) // 2]:.1f}")
                del gr
            except RuntimeError as e:   # noqa: BLE001
                res.append(f"{name}=ERR({str(e)[:40]})")
        print(f"[okc] diag sk2 {layer} m={m} ({_sk2_name(layer, m)}): " + " ".join(res), file=sys.stderr)
    del flush
    torch.cuda.synchronize()
''')
rep('''    _selftest(weights, state)
    _autotune(weights, state)
    torch.cuda.empty_cache()
    return state
''', '''    _selftest(weights, state)
    _autotune(weights, state)
    _diag_sk2(weights, state)
    torch.cuda.empty_cache()
    return state
''')
p.write_text(s)
print("ok")
