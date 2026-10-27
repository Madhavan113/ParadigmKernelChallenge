"""Edit (round 6: size the 11-bit lever before building it): DIAG bit 256 = the producer copies only 11/12 of each
stage's weight bytes (loads-only mode then measures the stream-level ceiling of an 11-bit format); prepare prints, per
matrix, the best 8-exponent window's coverage, the escape count and the 11-bit byte totals (incl. 4 / 6 B escape
entries) versus the 12-bit stream. Run: python3 tools/_edit_round6.py"""
import pathlib
p = pathlib.Path("muse_bf16_proj/submission.py")
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


# ---- kernel: DIAG bit 256 shortens every weight copy to 11/12 of the stage (16-B multiple); x unchanged
rep('''  // stages issued by the prologue: 1, or under PDL up to pdl (CL == 1 only; never past this CTA's own stages)
  const int P = (CL == 1 && pdl > 0) ? min(pdl, min(NSTAGE, nst)) : 1;
''', '''  // stages issued by the prologue: 1, or under PDL up to pdl (CL == 1 only; never past this CTA's own stages)
  const int P = (CL == 1 && pdl > 0) ? min(pdl, min(NSTAGE, nst)) : 1;
  // DIAG bit 256: copy 11/12 of each stage's weight bytes (what an 11-bit stream would move; loads-only ceiling)
  const uint32_t wbyc = (dgv & 256) ? (uint32_t)((WBY * 11 / 12) & ~15) : (uint32_t)WBY;
''')
rep('''    for (int st = 0; st < P; ++st) {
      mbar_expect_tx(full0 + 8 * st, ((dgv & 4) ? 0 : XBYr) + WBY);
      bulk_g2s(sbase + st * SST + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * st);
    }
''', '''    for (int st = 0; st < P; ++st) {
      mbar_expect_tx(full0 + 8 * st, ((dgv & 4) ? 0 : XBYr) + wbyc);
      bulk_g2s(sbase + st * SST + XBY, wsrc + (size_t)(gs0 + st) * wstride, wbyc, full0 + 8 * st);
    }
''')
rep('''        if (!FUSE || st < nst) {
          mbar_expect_tx(full0 + 8 * slot, ((dgv & 4) ? 0 : XBYr) + WBY);
          bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * slot);
''', '''        if (!FUSE || st < nst) {
          mbar_expect_tx(full0 + 8 * slot, ((dgv & 4) ? 0 : XBYr) + wbyc);
          bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + st) * wstride, wbyc, full0 + 8 * slot);
''')

# ---- diag modes: full / loads / loads at 11/12 of the bytes
rep('''DIAG_SK2_MODES = [(0, "full"), (6, "loads"), (70, "loads+spin"), (134, "loads+ctac"), (128, "full+ctac")]
''', '''DIAG_SK2_MODES = [(0, "full"), (6, "loads"), (262, "loads-11/12")]
''')

# ---- the 11-bit window statistics per matrix (DIAG builds only)
rep('''def sk2_prepare(weights, state):
    state["sk2"] = {}
    if not (USE_SK2 and state["sm90"]):
        return
''', '''def _window_stats(layer, w):
    """What an 11-bit format (3-bit code = sign + 2-bit e71 offset, i.e. an 8-exponent window with the low byte raw)
    would cost on this matrix: best window coverage, escapes, bytes vs the 12-bit stream."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    e = (bits >> 7) & 0xFF
    hist = torch.bincount(e.flatten(), minlength=256).to(torch.float64)
    win8 = torch.stack([hist[b:b + 8].sum() for b in range(249)])
    b8 = int(win8.argmax())
    cov8 = float(win8[b8]) / (n * k)
    win16 = torch.stack([hist[b:b + 16].sum() for b in range(241)])
    b16 = int(win16.argmax())
    cov16 = float(win16[b16]) / (n * k)
    esc8 = int(round((1 - cov8) * n * k))
    mb12 = n * k * 1.5 / 1e6
    mb11 = n * k * 11 / 8 / 1e6
    print(f"[okc] 11-bit window {layer}: best 8-exponent window e[{b8},{b8 + 8}) covers {100 * cov8:.3f}% "
          f"(escapes {esc8}, {100 * (1 - cov8):.3f}%); 16-exponent window covers {100 * cov16:.4f}%; "
          f"bytes 12-bit {mb12:.1f} MB vs 11-bit {mb11:.1f} + escapes {esc8 * 4 / 1e6:.1f} MB (4 B) / "
          f"{esc8 * 6 / 1e6:.1f} MB (6 B) = {mb11 + esc8 * 4 / 1e6:.1f} / {mb11 + esc8 * 6 / 1e6:.1f} MB "
          f"({100 * (1 - (mb11 + esc8 * 4 / 1e6) / mb12):.1f}% / {100 * (1 - (mb11 + esc8 * 6 / 1e6) / mb12):.1f}% fewer)",
          file=sys.stderr)


def sk2_prepare(weights, state):
    state["sk2"] = {}
    if not (USE_SK2 and state["sm90"]):
        return
    if DIAG_SK2:
        for layer in ("pooler", "qkv_gate", "unpooler", "o"):
            _window_stats(layer, weights[layer])
''')
p.write_text(s)
print("ok")
