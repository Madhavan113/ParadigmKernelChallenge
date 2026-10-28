"""Edit script (round 3 of skx): attribution-only DIAG levels for the main loop (4 = no wgmma, 5 = W only no wgmma,
6 = W only with wgmma) and the fix-1 tail (7 = wait only, 8 = reduce without output stores)."""
import pathlib

p = pathlib.Path(__file__).resolve().parent.parent / "muse_bf16_proj/submission.py"
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


rep("""// DIAG (measurement builds only, never self-tested, picked or routed): 1 = main loops only (no export, no tickets),
// 2 = + the fragment export, 3 = + the release fence and tickets (no wait / reduce / epilogue).""",
    """// DIAG (measurement builds only, never self-tested, picked or routed): 1 = main loops only (no export, no tickets),
// 2 = + the fragment export, 3 = + the release fence and tickets (no wait / reduce / epilogue); 4 = main loops without
// wgmma (pure TMA streaming of W + x), 5 = W only (no x copies) without wgmma, 6 = W only with wgmma (stale x tile);
// 7 = everything but the reduce (fix 1: wait only), 8 = the reduce without the output stores.""")

rep("""  constexpr int NACC = BN / 2, NB = BN / 8, NF4 = NB * NTH;
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  const uint32_t bars = sbase + NSTAGE * ST;
  const uint32_t full0 = bars, empty0 = bars + 8 * NSTAGE;
  int* s_tk = reinterpret_cast<int*>(smem_raw + (sbase - sraw) + NSTAGE * ST + 16 * NSTAGE);   // [2] tickets""",
    """  constexpr int NACC = BN / 2, NB = BN / 8, NF4 = NB * NTH;
  constexpr bool NO_MMA = DIAG == 4 || DIAG == 5, NO_X = DIAG == 5 || DIAG == 6;
  constexpr bool LOOP_ONLY = DIAG == 1 || DIAG == 4 || DIAG == 5 || DIAG == 6;
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  const uint32_t bars = sbase + NSTAGE * ST;
  const uint32_t full0 = bars, empty0 = bars + 8 * NSTAGE;
  int* s_tk = reinterpret_cast<int*>(smem_raw + (sbase - sraw) + NSTAGE * ST + 16 * NSTAGE);   // [2] tickets""")

rep("""  auto issue = [&](int u, int slot) {   // one unit (W tile + x tile) into ring slot `slot`
    mbar_expect_tx(full0 + 8 * slot, A_BYTES + xrows * 128);
    const int rb = u / C, c = u - rb * C;
    if constexpr (TILED)
      bulk_g2s_ef(sbase + slot * ST, Wt + (size_t)u * A_BYTES, A_BYTES, full0 + 8 * slot);
    else
      tma_load_2d_ef(sbase + slot * ST, &mapW, full0 + 8 * slot, c * 64, rb * BM);
    tma_load_2d_el(sbase + slot * ST + A_BYTES, &mapX, full0 + 8 * slot, c * 64, 0);
  };""",
    """  auto issue = [&](int u, int slot) {   // one unit (W tile + x tile) into ring slot `slot`
    mbar_expect_tx(full0 + 8 * slot, A_BYTES + (NO_X ? 0 : xrows * 128));
    const int rb = u / C, c = u - rb * C;
    if constexpr (TILED)
      bulk_g2s_ef(sbase + slot * ST, Wt + (size_t)u * A_BYTES, A_BYTES, full0 + 8 * slot);
    else
      tma_load_2d_ef(sbase + slot * ST, &mapW, full0 + 8 * slot, c * 64, rb * BM);
    if constexpr (!NO_X) tma_load_2d_el(sbase + slot * ST + A_BYTES, &mapX, full0 + 8 * slot, c * 64, 0);
  };""")

rep("""      const uint32_t a_st = sbase + slot * ST + wg * (64 * 128);
      const uint32_t b_st = sbase + slot * ST + A_BYTES;
#pragma unroll
      for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
      wg_fence();
#pragma unroll
      for (int j = 0; j < 4; ++j) wgmma_ss<BN>(acc, make_desc_sw128(a_st + j * 32), make_desc_sw128(b_st + j * 32));
      wg_commit();
      wg_wait<1>();   // the previous unit's group is done: hand its slot back
#pragma unroll
      for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
      if (pending >= 0 && t == 0) mbar_arrive(empty0 + 8 * pending);
      pending = slot;
    }
    wg_wait<0>();
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    if (t == 0) mbar_arrive(empty0 + 8 * pending);
    pending = -1;
    if constexpr (DIAG == 1) {   // keep the wgmma chain alive (ptxas drops accumulators nobody reads: 24 registers)
      if (acc[0] == 12345.678f && acc[NACC - 1] == -12345.678f) out[tid] = __float2bfloat16_rn(acc[1]);
      continue;
    }""",
    """      const uint32_t a_st = sbase + slot * ST + wg * (64 * 128);
      const uint32_t b_st = sbase + slot * ST + A_BYTES;
      if constexpr (!NO_MMA) {
#pragma unroll
        for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
        wg_fence();
#pragma unroll
        for (int j = 0; j < 4; ++j) wgmma_ss<BN>(acc, make_desc_sw128(a_st + j * 32), make_desc_sw128(b_st + j * 32));
        wg_commit();
        wg_wait<1>();   // the previous unit's group is done: hand its slot back
#pragma unroll
        for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
      }
      if (pending >= 0 && t == 0) mbar_arrive(empty0 + 8 * pending);
      pending = slot;
    }
    if constexpr (!NO_MMA) {
      wg_wait<0>();
#pragma unroll
      for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    }
    if (t == 0) mbar_arrive(empty0 + 8 * pending);
    pending = -1;
    if constexpr (LOOP_ONLY) {   // keep the wgmma chain alive (ptxas drops accumulators nobody reads: 24 registers)
      if (!NO_MMA && acc[0] == 12345.678f && acc[NACC - 1] == -12345.678f) out[tid] = __float2bfloat16_rn(acc[1]);
      continue;
    }""")

rep("""  if constexpr (DIAG == 1 || DIAG == 2) return;""", """  if constexpr (LOOP_ONLY || DIAG == 2) return;""")

rep("""      named_bar(1, NTH);
      const int ua = s == 0 ? u0 : rb * C, ub = s == nseg - 1 ? u1 : (rb + 1) * C;
      const int lo = (int)(((long long)(ua - rb * C) * NF4) / C), hi = (int)(((long long)(ub - rb * C) * NF4) / C);
      for (int idx = lo + tid; idx < hi; idx += NTH) {""",
    """      named_bar(1, NTH);
      if constexpr (DIAG == 7) continue;   // wait only
      const int ua = s == 0 ? u0 : rb * C, ub = s == nseg - 1 ? u1 : (rb + 1) * C;
      const int lo = (int)(((long long)(ua - rb * C) * NF4) / C), hi = (int)(((long long)(ub - rb * C) * NF4) / C);
      float4 keep = make_float4(0.f, 0.f, 0.f, 0.f);
      for (int idx = lo + tid; idx < hi; idx += NTH) {""")

rep("""        if (ip <= lc) {
          const int sa = ((int)(((long long)ip * U) / G) / C == rb) ? 0 : G;
          const float4 a = __ldcg(slab + (size_t)(sa + ip) * NF4 + idx);
          v.x += a.x; v.y += a.y; v.z += a.z; v.w += a.w;
        }
        skx_store4(out, M, N, rb, idx / NTH, idx % NTH, v);
      }""",
    """        if (ip <= lc) {
          const int sa = ((int)(((long long)ip * U) / G) / C == rb) ? 0 : G;
          const float4 a = __ldcg(slab + (size_t)(sa + ip) * NF4 + idx);
          v.x += a.x; v.y += a.y; v.z += a.z; v.w += a.w;
        }
        if constexpr (DIAG == 8) { keep.x += v.x; keep.y += v.y; keep.z += v.z; keep.w += v.w; }
        else skx_store4(out, M, N, rb, idx / NTH, idx % NTH, v);
      }
      if constexpr (DIAG == 8) {   // reduce without output stores: keep the loads alive
        if (keep.x == 12345.678f && keep.y == -12345.678f && keep.z == 1.f && keep.w == -1.f) out[tid] = __float2bfloat16_rn(keep.x);
      }""")

# config table: attribution round (all diagnostic; the autotuner times them and picks among the others)
rep("""SKX_CONFIGS = {
    ("qkv_gate", 128): [(6, 1, 0, 0), (6, 0, 0, 0), (6, 1, 0, 1), (6, 1, 0, 2), (6, 1, 0, 3)],
    ("pooler", 128): [(6, 1, 0, 0), (6, 0, 0, 0), (6, 1, 0, 1), (6, 1, 0, 2), (6, 1, 0, 3)],
    ("unpooler", 128): [(6, 1, 0, 0), (6, 0, 0, 0), (6, 1, 0, 1), (6, 1, 0, 2), (6, 1, 0, 3)],
    ("o", 128): [(6, 1, 0, 0), (6, 0, 0, 0), (6, 1, 0, 1), (6, 1, 0, 2), (6, 1, 0, 3)],
}""",
    """# Check 2 (bcdae43f, m=128 bench us; D1 loop / D2 +export / D3 +fence+tickets / full fix1 / full fix0 vs pick): qkv
# 16.7 / 18.08 / 19.33 / 27.84 / 30.62 vs gqx 18.08; o 10.4 / 13.06 / 14.3 / 24.64 / 25.28 vs torchT 10.43; pooler
# 23.62 / 25.25 / 26.53 / 35.78 / 37.22 vs gq 25.22; unpooler 22.82 / 24.93 / 26.37 / 37.34 / 37.41 vs torchT 22.72.
# The main loop alone only ties / -1.5 us the picks (0.58 us per unit: W at ~3.4 TB/s, ~48% tensor utilization) and
# the tail (wait + reduce or last-arriver epilogue) costs ~10 us on every layer. Round 3 = attribution only.
SKX_CONFIGS = {
    ("qkv_gate", 128): [(6, 1, 0, 1), (6, 1, 1, 1), (6, 1, 0, 4), (6, 1, 0, 5), (6, 1, 0, 6), (6, 1, 0, 7), (6, 1, 0, 8)],
    ("pooler", 128): [(6, 1, 0, 1), (6, 1, 1, 1), (6, 1, 0, 4), (6, 1, 0, 5), (6, 1, 0, 6), (6, 1, 0, 7), (6, 1, 0, 8)],
    ("unpooler", 128): [(6, 1, 0, 1), (6, 1, 1, 1), (6, 1, 0, 4), (6, 1, 0, 5), (6, 1, 0, 6), (6, 1, 0, 7), (6, 1, 0, 8)],
    ("o", 128): [(6, 1, 0, 1), (6, 1, 1, 1), (6, 1, 0, 4), (6, 1, 0, 5), (6, 1, 0, 6), (6, 1, 0, 7), (6, 1, 0, 8)],
}""")
p.write_text(s)
print("round-3 edits applied")
