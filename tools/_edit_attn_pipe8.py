"""OKC_PIPE8: flat q8 loop software-pipelined across tiles -- tile i+1's stage wait and QK^T wgmma are issued right
after tile i's PV wgmma (both in flight across the loop edge), so the QK^T latency and the PV end-of-tile wait leave
the per-tile chain. Default off: the default build stays as it was."""
P = "/Users/madhavanp/kerneloptimizer/.claude/worktrees/agent-aad2dcde86ab9f089/muse_attention_decode/submission.py"
s = open(P).read()


def sub1(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)


sub1("""#ifndef OKC_FLATONLY
""", """#ifndef OKC_PIPE8
#define OKC_PIPE8 0     // flat q8: the next tile's stage wait + QK^T wgmma issued right after this tile's PV wgmma (both in
                        // flight across the loop edge: the QK^T latency and the PV end-of-tile wait leave the chain)
#endif
#ifndef OKC_FLATONLY
""")

# A1: before the loop: the switch, the cross-iteration score array, the QK^T issue lambda, tile 0 issued ahead
sub1("""  const int nit = S2 ? ((ntiles + 1) >> 1) : (F2 ? 0x7fffffff : ntiles);
""", """  const int nit = S2 ? ((ntiles + 1) >> 1) : (F2 ? 0x7fffffff : ntiles);
  // OKC_PIPE8 (flat producer mode, FP8 PV, loop-carried addressing): tile i+1's stage wait and QK^T go out right after
  // tile i's PV; the scores live across the loop edge, so they are declared outside the loop in that build only
  constexpr bool PIPE8 = OKC_PIPE8 && WS && FLAT && ADDR && PV8 && OKC_QSS && OKC_RARE && PVSPLIT && OKC_PVSPLIT == 1 &&
                         LSYNC && !PTR && !F2 && !S2 && !OKC_PROF && !OKC_TURN && !OKC_NOCOMPUTE && !OKC_XNOWAITF && OKC_WGDEC;
  float s_outer[PIPE8 ? 32 : 1];
  auto issue_qk = [&](int st_) {   // (PIPE8) the QK^T wgmmas of the tile in stage st_ into s_outer
    if constexpr (PIPE8) {
      wgmma_fence();
#if OKC_DSHFL
      const uint32_t kl = klo0 + (uint32_t)(st_ * (STAGE_BYTES >> 4));
      const uint64_t kd = ((uint64_t)((1024 >> 4) | (1u << 30)) << 32) | kl;
      const uint64_t kd2 = ((uint64_t)((1024 >> 4) | (1u << 30)) << 32) | (kl + 2u);
      const uint64_t qd2 = ((uint64_t)((512 >> 4) | (2u << 30)) << 32) | (qlo0 + 2u);
      wgmma_m64n64k32_e4m3_ss(s_outer, qdesc0, kd, 0);
      wgmma_m64n64k32_e4m3_ss(s_outer, qd2, kd2, 1);
#else
      const uint64_t kd = kdesc0 + (uint64_t)(st_ * (STAGE_BYTES >> 4));
      wgmma_m64n64k32_e4m3_ss(s_outer, qdesc0, kd, 0);
      wgmma_m64n64k32_e4m3_ss(s_outer, qdesc0 + 2, kd + 2, 1);
#endif
      wgmma_commit();
    }
  };
  if constexpr (PIPE8) {
    if (ntiles > 0) {   // tile 0: landed (+ new rows), both warpgroups meet (the q staging), its QK^T issued ahead
      mbar_wait_full(&S.full[st_c], ph_c);
      if (((uint32_t)0 - r_lo > r_span) && (0 >= i_new)) ws_patch_ni<L, NT, PTR>(&S, 0, st_c, start, seq, end, tid);
      csync<WS, NT>();
      issue_qk(st_c);
    }
  }
""")

# A2: the per-tile stage wait (+ patch) is skipped: done when the tile's QK^T was issued
sub1("""      bool patched;
      if constexpr (ADDR) {
        if constexpr (!OKC_XNOWAITF) mbar_wait_full(&S.full[st_c], ph_c);   // (probe: no wait)
""", """      bool patched;
      if constexpr (PIPE8) {
        patched = false;   // (waited for, patched and its QK^T issued when the previous tile's PV went out)
      } else if constexpr (ADDR) {
        if constexpr (!OKC_XNOWAITF) mbar_wait_full(&S.full[st_c], ph_c);   // (probe: no wait)
""")
sub1("""      if (patched || i == 0 || !OKC_WGDEC) csync<WS, NT>();
      else if (!(PV8ON && OKC_PV8FLAT && OKC_NOSYNC0)) wgsync(hk);
""", """      if constexpr (!PIPE8) {
        if (patched || i == 0 || !OKC_WGDEC) csync<WS, NT>();
        else if (!(PV8ON && OKC_PV8FLAT && OKC_NOSYNC0)) wgsync(hk);
      }
""")

# A4: the in-loop QK^T issue (skipped under PIPE8; the scores alias the outer array there)
sub1("""    float s[32];
    wgmma_fence();
    if constexpr (PV8 && OKC_QSS && ADDR) {
#if OKC_DSHFL
""", """    float s_inner[PIPE8 ? 1 : 32];
    float* s_sel = PIPE8 ? s_outer : s_inner;
    float (&s)[32] = *reinterpret_cast<float(*)[32]>(s_sel);
    if constexpr (!PIPE8) {
    wgmma_fence();
    if constexpr (PV8 && OKC_QSS && ADDR) {
#if OKC_DSHFL
""")
sub1("""      wgmma_m64n64k32_e4m3(s, qa8[1][0], qa8[1][1], qa8[1][2], qa8[1][3], desc_sw128(kaddr + 32), 1);
    }
    wgmma_commit();
    if constexpr (OKC_QKFIRST && !WS && !Cfg<L, FLAT>::S2) {   // (OKC_QKFIRST) the copies for tile i + STAGES - 1
""", """      wgmma_m64n64k32_e4m3(s, qa8[1][0], qa8[1][1], qa8[1][2], qa8[1][3], desc_sw128(kaddr + 32), 1);
    }
    wgmma_commit();
    }   // (!PIPE8)
    if constexpr (OKC_QKFIRST && !WS && !Cfg<L, FLAT>::S2) {   // (OKC_QKFIRST) the copies for tile i + STAGES - 1
""")

# A5: no wait for the previous PV before the transpose (its V^T buffer is the other one; o / lacc are only touched
# after the wgmma wait that follows the transpose)
sub1("""    if constexpr (PV8 && FLAT && OKC_PVW1) {
      // the previous tile's PV (committed before this QK^T) is done: its P registers are free for the transpose
""", """    if constexpr (PV8 && FLAT && OKC_PVW1 && !PIPE8) {
      // the previous tile's PV (committed before this QK^T) is done: its P registers are free for the transpose
""")

# A6: loop end: instead of waiting for this tile's PV, advance to the next tile and issue its QK^T across the PV
sub1("""      if constexpr (PVSPLIT) {
        if constexpr (OKC_PVSPLIT == 1) {   // done before the next tile (no wgmma in flight across the loop edge)
          wgmma_wait<0>();
""", """      if constexpr (PVSPLIT) {
        if constexpr (PIPE8) {
          // next tile: stage advance, its wait (+ new rows: both warpgroups meet) and its QK^T, across this tile's PV
          // (the wgmma wait after the next transpose covers both; o / lacc / pa are fenced there)
          if (++st_c == STAGES) { st_c = 0; ph_c ^= 1u; }
          vtb_c = reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(vtb_c) ^ vtb_x);
          vdesc_c ^= vdesc_x;
          if (i + 1 < ntiles) {
            mbar_wait_full(&S.full[st_c], ph_c);
            const bool patched_n = ((uint32_t)(i + 1) - r_lo > r_span) && (i + 1 >= i_new) &&
                                   ws_patch_ni<L, NT, PTR>(&S, i + 1, st_c, start, seq, end, tid);
            if (patched_n) csync<WS, NT>();
            issue_qk(st_c);
          }
        } else if constexpr (OKC_PVSPLIT == 1) {   // done before the next tile (no wgmma in flight across the loop edge)
          wgmma_wait<0>();
""")

# A7: the loop-carried addressing advance (PIPE8 advanced it above)
sub1("""    if constexpr (ADDR) {
      if (++st_c == STAGES) { st_c = 0; ph_c ^= 1u; }
      vtb_c = reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(vtb_c) ^ vtb_x);
      vdesc_c ^= vdesc_x;
    }
  }
  PROF_DUMP;
""", """    if constexpr (ADDR && !PIPE8) {
      if (++st_c == STAGES) { st_c = 0; ph_c ^= 1u; }
      vtb_c = reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(vtb_c) ^ vtb_x);
      vdesc_c ^= vdesc_x;
    }
  }
  PROF_DUMP;
""")
open(P, "w").write(s)
print("ok")
