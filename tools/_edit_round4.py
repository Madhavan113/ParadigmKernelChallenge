"""Edit (round 4: the sk2 stream at 3.5 vs 4.4 TB/s): DIAG bits 64 (producer busy-spin instead of the suspend-hint
wait) and 128 (CTA-contiguous weight layout instead of stage-major); ring-depth / copy-size / 2-CTAs-per-SM variants
as SK2_ALT candidates; the diag bench times every candidate of each m <= 32 shape in full / loads-only /
loads+spin / loads+cta-contiguous / full+cta-contiguous modes. Run: python3 tools/_edit_round4.py"""
import pathlib
p = pathlib.Path("muse_bf16_proj/submission.py")
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


# ---- kernel: layout bit (128) and producer spin bit (64), both compiled out of production builds (dgv = 0)
rep('''  // stage-major weight stream: [stage][row block][...] (at any moment all CTAs read one contiguous region)
  const uint8_t* wsrc = Ws + (size_t)cta * WBY;
  const size_t wstride = (size_t)(gridDim.x / CL) * WBY;
''', '''  // stage-major weight stream: [stage][row block][...] (at any moment all CTAs read one contiguous region); DIAG bit
  // 128: a CTA-contiguous stream [row block][stage][...] (DRAM page locality per CTA instead of across CTAs)
  const uint8_t* wsrc = (dgv & 128) ? Ws + (size_t)cta * (size_t)tst * WBY : Ws + (size_t)cta * WBY;
  const size_t wstride = (dgv & 128) ? (size_t)WBY : (size_t)(gridDim.x / CL) * WBY;
''')
rep('''      for (int st = P; st < nstt; ++st) {
        const int slot = st % NSTAGE;
        if (st >= NSTAGE) SK2_WAIT(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
''', '''      for (int st = P; st < nstt; ++st) {
        const int slot = st % NSTAGE;
        if (st >= NSTAGE) {
          if (dgv & 64) mbar_wait(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);   // DIAG: busy spin (no suspend hint)
          else SK2_WAIT(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
        }
''')

# ---- configs: deeper rings / bigger copies / 2 CTAs per SM with >= 12 KB weight copies
rep('''    # the decode layer's fused pooler + router launch (sk2f, FUSE)''', '''    # round 4 (stream rate: 3.5 TB/s loads-only vs 4.4 for a pure bulk stream): deeper rings up to the smem cap for the
    # 96-128-CTA configs, 2 CTAs per SM with >= 12 KB weight copies, bigger copies per stage
    ('pooler8h_r100', 'pooler', 4096, 16, 1, 2, 2, 4, 1, 1),        # pooler8h_r72 with a 100 KB ring (5 x 20 KB), 2 CTAs/SM
    ('pooler8t_r192', 'pooler', 4096, 32, 2, 2, 2, 4, 1, 1),        # 128 CTAs, 6 x 32 KB (24 KB weight copies)
    ('pooler16h_r100', 'pooler', 4096, 16, 2, 1, 2, 4, 1, 1),       # 256 CTAs, 2/SM, 3 x 28 KB (12 KB copies)
    ('pooler16t_r220', 'pooler', 4096, 32, 2, 2, 2, 4, 1, 1),       # pooler16t_r192 with 5 stages
    ('pooler32_r216', 'pooler', 4096, 32, 2, 2, 2, 2, 1, 1),        # pooler32_r144 with 7 stages
    ('pooler32w_r216', 'pooler', 4096, 32, 2, 2, 2, 4, 1, 1),       # 48 KB stages (24 KB copies), 4 stages
    ('pooler32h_r100', 'pooler', 4096, 16, 2, 1, 2, 4, 1, 1),       # 256 CTAs, 2/SM, 12 KB copies, 28 KB x 3
    ('qkv8t_r216', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 1),         # qkv8t_r144 with 8 stages
    ('qkv16t_r216', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 1),        # qkv16t_r200 with 6 stages
    ('qkv32_r220', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 1),         # qkv32_r208 (4 x 50 KB stays 4)
    ('qkv8w_r216', 'qkv_gate', 2304, 24, 3, 1, 4, 4, 2, 1),         # 36 KB weight copies (1024 k per stage)
    ('unpooler16t_r100', 'unpooler', 8192, 32, 2, 2, 2, 2, 1, 1),   # unpooler16t_r72 with 5 stages
    ('unpooler32_r216', 'unpooler', 8192, 64, 4, 2, 2, 2, 1, 1),    # unpooler32_r144 with 5 stages
    ('unpooler32h_r100', 'unpooler', 8192, 32, 2, 2, 2, 2, 1, 1),   # 256 CTAs, 2/SM (28 KB x 3)
    # the decode layer's fused pooler + router launch (sk2f, FUSE)''')
rep('''SK2_ALT = {  # (layer, tokr) -> alternative config names the autotuner times against SK2_PICK's (see the retired list)
}
''', '''SK2_ALT = {  # (layer, tokr) -> alternative config names the autotuner times against SK2_PICK's (see the retired list)
    ('pooler', 8): ['pooler8h_r100', 'pooler8t_r192'],
    ('pooler', 16): ['pooler16h_r100', 'pooler16t_r220'],
    ('pooler', 32): ['pooler32_r216', 'pooler32w_r216', 'pooler32h_r100'],
    ('qkv_gate', 8): ['qkv8t_r216', 'qkv8w_r216'],
    ('qkv_gate', 16): ['qkv16t_r216'],
    ('qkv_gate', 32): ['qkv32_r220'],
    ('unpooler', 16): ['unpooler16t_r100'],
    ('unpooler', 32): ['unpooler32_r216', 'unpooler32h_r100'],
}
''')

# ---- diag bench: every candidate x modes; CTA-contiguous stream built on the fly
rep('''DIAG_SK2_SHAPES = [("pooler", 8), ("pooler", 16), ("pooler", 32), ("qkv_gate", 8), ("qkv_gate", 32), ("unpooler", 16),
                   ("unpooler", 32), ("o", 16)]
DIAG_SK2_MODES = [(0, "full"), (1, "no-epilogue"), (2, "no-compute"), (4, "no-x"), (6, "loads-only"), (8, "alu-not-mma"),
                  (16, "no-ldmatrix")]
''', '''DIAG_SK2_SHAPES = [("pooler", 8), ("pooler", 16), ("pooler", 32), ("qkv_gate", 8), ("qkv_gate", 16), ("qkv_gate", 32),
                   ("unpooler", 16), ("unpooler", 32), ("o", 16), ("unpooler", 8)]
DIAG_SK2_MODES = [(0, "full"), (6, "loads"), (70, "loads+spin"), (134, "loads+ctac"), (128, "full+ctac")]
''')
rep('''def _run_sk2(x, layer, m, n, k, state, dg=0, dbg=None, name=None):
    name = name or _sk2_name(layer, m)
    ci, stream, tab, pmax, b4, n_rows = state["sk2"][name]
''', '''def _run_sk2(x, layer, m, n, k, state, dg=0, dbg=None, name=None, stream_override=None):
    name = name or _sk2_name(layer, m)
    ci, stream, tab, pmax, b4, n_rows = state["sk2"][name]
    if stream_override is not None:
        stream = stream_override
''')
a = s.index("def _diag_sk2(weights, state):")
b = s.index("\n\n\n", a) + 1
s = s[:a] + '''def _diag_sk2(weights, state):
    """Attribution of the sk2 kernels on this GPU: graph replays from a cold L2 (as the autotuner) of each candidate
    config of each shape in each dg mode (DIAG builds only; the dg bits are runtime branches there)."""
    if not (DIAG_SK2 and state.get("sm90")):
        return
    dev = weights["o"].device
    flush = torch.ones(64 * 1024 * 1024 // 1024, 1024, dtype=torch.float32, device=dev)
    sink = torch.empty(1024, dtype=torch.float32, device=dev)
    g = torch.Generator(device=dev)
    g.manual_seed(999)
    cfg_by_name = {cfg[0]: cfg for cfg in SK2_CONFIGS}
    for layer, m in DIAG_SK2_SHAPES:
        w = weights[layer]
        n = 1024 if (layer == "unpooler" and m <= 8) else w.shape[0]
        k = w.shape[1]
        names = [_sk2_name(layer, m)] + list(SK2_ALT.get((layer, _sk2_tokr(m)), ()))
        x = torch.randn(m, k, device=dev, generator=g).to(torch.bfloat16)
        for name in names:
            if name not in state.get("sk2", {}):
                continue
            ci, stream, tab, pmax, b4, n_rows = state["sk2"][name]
            _, _, _, R, NT, NWR, WSK, UPW, _, CL = cfg_by_name[name]
            wby = NWR * WSK * UPW * NT * 768
            tst = k // (WSK * UPW * 64)
            ctac = stream.view(tst, n_rows // R, wby).permute(1, 0, 2).contiguous()   # [cta][stage][...]
            res = []
            for dg, mode in DIAG_SK2_MODES:
                so = ctac if (dg & 128) else None
                try:
                    _run_sk2(x, layer, m, n, k, state, dg=dg, name=name, stream_override=so)
                    torch.cuda.synchronize()
                    gr = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(gr):
                        _run_sk2(x, layer, m, n, k, state, dg=dg, name=name, stream_override=so)
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
                    res.append(f"{mode}={sorted(ts)[len(ts) // 2]:.1f}")
                    del gr
                except RuntimeError as e:   # noqa: BLE001
                    res.append(f"{mode}=ERR({str(e)[:40]})")
            hl, nstage = _sk2_geom(cfg_by_name[name], _sk2_tokr(m))
            print(f"[okc] diag sk2 {layer} m={m} {name} (R{R} {wby // 1024}KB-copies x{nstage} stages, {n_rows // R * CL} CTAs): "
                  + " ".join(res), file=sys.stderr)
            del ctac
    del flush
    torch.cuda.synchronize()
''' + s[b:]
p.write_text(s)
print("ok")
