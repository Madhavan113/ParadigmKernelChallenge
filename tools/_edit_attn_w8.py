#!/usr/bin/env python3
"""Patch muse_attention_decode/submission.py: OKC_W8 (q_len 1 per-request kernels with 8 warps on 16-key quarters),
the fin_nc1 thread guard for 256-thread CTAs, and the OKC_XP_RELAXTICKET timing probe. Idempotent-ish: asserts on
exact matches, so run once."""
import pathlib, sys

p = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "muse_attention_decode/submission.py")
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


# 1. macros
rep("""#ifndef OKC_STN1
#define OKC_STN1 OKC_STN   // ... the q_len 1 per-request kernels (sliding B >= 2: 2-4 tiles per CTA; full B = 1: 5): deeper
                           // rings put every tile of the split in flight from the prologue (one gather round trip)
#endif""",
    """#ifndef OKC_STN1
#define OKC_STN1 OKC_STN   // ... the q_len 1 per-request kernels (measured on the H200: deeper rings lose, 4/5/6 stages
                           // +0.1/+0.5/+0.9 us at b16 sliding -- the random gathers are best kept to 2 tiles in flight per CTA)
#endif
#ifndef OKC_W8
#define OKC_W8 1        // q_len 1 per-request kernels: 8 warps per CTA, warp = (KV head, 16-key quarter of the tile) instead of
                        // 4 warps on 32-key halves: half the dependent MMA / conversion chain per tile (the tile time is one
                        // warp's chain, ~0.7 us for 4 warps, not SM issue contention: 1 CTA per SM measured the same)
#endif
#ifndef OKC_XP_RELAXTICKET
#define OKC_XP_RELAXTICKET 0   // (timing probe only, wrong results possible) last-CTA ticket as a relaxed atomic: no membar
#endif""")

# 2. Cfg NT
rep("  static constexpr int NT = (L == 1) ? 128 : ((S2 || F2) ? 512 : 256);   // consumer threads per CTA",
    "  static constexpr int NT = (L == 1) ? ((!F && OKC_W8) ? 256 : 128) : ((S2 || F2) ? 512 : 256);   // consumer threads per CTA")

# 3. QPASS: at least one pass; rows past L*16 (NT 256) compute on garbage and skip the store
rep("  constexpr int QPASS = L * 16 / RPP;            // L*16 rows, RPP rows per pass",
    "  constexpr int QPASS = (L * 16 + RPP - 1) / RPP;   // L*16 rows, RPP rows per pass (NT 256 at L 1: half the threads idle)")
rep("""  for (int ps = 0; ps < QPASS; ++ps) {
    const int rr = ps * RPP + (tidL >> 3);
    const int j = rr >> 4, hh = rr & 15;
    const uint4 qv = *reinterpret_cast<const uint4*>(&S.qst()[rr * 64 + partL * 8]);""",
    """  for (int ps = 0; ps < QPASS; ++ps) {
    const int rr = ps * RPP + (tidL >> 3);
    const bool qlive = (L * 16 % RPP == 0) || rr < L * 16;   // (rows past the block: no store; the shuffles stay warp-uniform)
    const int j = rr >> 4, hh = rr & 15;
    const uint4 qv = *reinterpret_cast<const uint4*>(&S.qst()[(qlive ? rr : 0) * 64 + partL * 8]);""")
rep("""    if constexpr (WS) {   // in place over the raw row (same thread, same 16 bytes)
      *reinterpret_cast<uint4*>(S.kvq + rr * 128 + partL * 16) = qw;
    } else {
      uint8_t* qs = (hk ? S.v : S.k) + (STAGES - 1) * STAGE_BYTES;
      *reinterpret_cast<uint4*>(qs + (j * 8 + gg) * 128 + partL * 16) = qw;
    }
  }""",
    """    if constexpr (WS) {   // in place over the raw row (same thread, same 16 bytes)
      *reinterpret_cast<uint4*>(S.kvq + rr * 128 + partL * 16) = qw;
    } else if (qlive) {
      uint8_t* qs = (hk ? S.v : S.k) + (STAGES - 1) * STAGE_BYTES;
      *reinterpret_cast<uint4*>(qs + (j * 8 + gg) * 128 + partL * 16) = qw;
    }
  }""")

# 4. consumer: NTQ n-tiles (8 keys each) per warp
rep("""  if constexpr (L == 1) {
  // Q A-fragments (f16, pre-scaled to log2 units) for 4 k-steps of 16 dims: thread holds dims
  // 16*tig + 4*ks + {0,1} (a0) and + {2,3} (a2); the K B-fragments use the same dims.
  uint32_t qa[4][2];""",
    """  if constexpr (L == 1) {
  // warp = (KV head hk, key block wsub of the tile): 8*NTQ keys per warp (NTQ n-tiles of 8 keys; 4 warps: halves of 32,
  // OKC_W8: 8 warps on quarters of 16)
  constexpr int NTQ = (NT == 256) ? 2 : 4;
  // Q A-fragments (f16, pre-scaled to log2 units) for 4 k-steps of 16 dims: thread holds dims
  // 16*tig + 4*ks + {0,1} (a0) and + {2,3} (a2); the K B-fragments use the same dims.
  uint32_t qa[4][2];""")
rep("""    // warp: kv head hk, keys [32*wsub, 32*wsub+32) of the tile
    float s[4][4];
    uint32_t kw[4][4];
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int key = 32 * wsub + 8 * nt + g;
      const uint4 kb = lds128(K + key * ROWB + (((hk * 4 + tig) ^ (key & 7)) << 4));
      kw[nt][0] = kb.x; kw[nt][1] = kb.y; kw[nt][2] = kb.z; kw[nt][3] = kb.w;
    }
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {   // (scout: unpadded quads; even k-steps -> rows g of se, odd -> rows g+8 of so)""",
    """    // warp: kv head hk, keys [8*NTQ*wsub, 8*NTQ*wsub + 8*NTQ) of the tile
    float s[NTQ][4];
    uint32_t kw[NTQ][4];
#pragma unroll
    for (int nt = 0; nt < NTQ; ++nt) {
      const int key = 8 * NTQ * wsub + 8 * nt + g;
      const uint4 kb = lds128(K + key * ROWB + (((hk * 4 + tig) ^ (key & 7)) << 4));
      kw[nt][0] = kb.x; kw[nt][1] = kb.y; kw[nt][2] = kb.z; kw[nt][3] = kb.w;
    }
#pragma unroll
    for (int nt = 0; nt < NTQ; ++nt) {   // (scout: unpadded quads; even k-steps -> rows g of se, odd -> rows g+8 of so)""")
rep("""    if (!interior) {
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int pos = t0 + 32 * wsub + 8 * nt + 2 * tig + e;
          if (pos >= end) s[nt][e] = -INFINITY;
        }
    }
    float mx = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) mx = fmaxf(mx, fmaxf(s[nt][0], s[nt][1]));""",
    """    if (!interior) {
#pragma unroll
      for (int nt = 0; nt < NTQ; ++nt)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int pos = t0 + 8 * NTQ * wsub + 8 * nt + 2 * tig + e;
          if (pos >= end) s[nt][e] = -INFINITY;
        }
    }
    float mx = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < NTQ; ++nt) mx = fmaxf(mx, fmaxf(s[nt][0], s[nt][1]));""")
rep("""    const float m_new = m_r[0];
    if (m_new != -INFINITY) {
      float ls = 0.f;
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float pv = ex2(s[nt][e] - m_new);
          s[nt][e] = pv;
          ls += pv;
        }
      l_r[0] += ls;
#else""",
    """    const float m_new = m_r[0];
    if (m_new != -INFINITY) {
      float ls = 0.f;
#pragma unroll
      for (int nt = 0; nt < NTQ; ++nt)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float pv = ex2(s[nt][e] - m_new);
          s[nt][e] = pv;
          ls += pv;
        }
      l_r[0] += ls;
#else""")
rep("""      const float alpha = ex2(m_r[0] - m_new);
      m_r[0] = m_new;
      float ls = 0.f;
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float pv = ex2(s[nt][e] - m_new);
          s[nt][e] = pv;
          ls += pv;
        }""",
    """      const float alpha = ex2(m_r[0] - m_new);
      m_r[0] = m_new;
      float ls = 0.f;
#pragma unroll
      for (int nt = 0; nt < NTQ; ++nt)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float pv = ex2(s[nt][e] - m_new);
          s[nt][e] = pv;
          ls += pv;
        }""")
rep("""#pragma unroll
      for (int ks = 0; ks < (OKC_XHALFPV ? 1 : 2); ++ks) {   // (OKC_XHALFPV: timing probe only, wrong results)
        uint32_t b0[8], b1[8];
        load_v_frags(V, 32 * wsub + 16 * ks, hk, g, tig, b0, b1);""",
    """#pragma unroll
      for (int ks = 0; ks < (OKC_XHALFPV ? 1 : NTQ / 2); ++ks) {   // (OKC_XHALFPV: timing probe only, wrong results)
        uint32_t b0[8], b1[8];
        load_v_frags(V, 8 * NTQ * wsub + 16 * ks, hk, g, tig, b0, b1);""")

# 5. the exchange: wsub >= 1 warps hand their state to the wsub == 0 warp of the same head
rep("""  if (L == 1 && !CONT && !(OKC_NOXCH1 && !M_CL)) {
    // warps 2,3 (second key half) hand their state to warps 0,1 (same heads) through shared memory
    // (OKC_NOXCH1: outside cluster mode both halves add their own rescaled state to the workspace instead)
    cp_wait<0>();
    csync<WS, NT>();
    float* xs = reinterpret_cast<float*>(S.k) + ((warp & 1) * 32 + lane) * 18;
    if (wsub == 1) {
      xs[0] = m_r[0];
      xs[1] = l_r[0];
#pragma unroll
      for (int t = 0; t < 8; ++t) { xs[2 + 2 * t] = o[0][t][0]; xs[3 + 2 * t] = o[0][t][1]; }
    }
    csync<WS, NT>();
    if (wsub == 0) {
      const float m1 = xs[0], l1 = xs[1];
      const float mm = fmaxf(m_r[0], m1);
      const float f0 = (m_r[0] == -INFINITY) ? 0.f : ex2(m_r[0] - mm);
      const float f1 = (m1 == -INFINITY) ? 0.f : ex2(m1 - mm);
      m_r[0] = mm;
      l_r[0] = l_r[0] * f0 + l1 * f1;
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        o[0][t][0] = o[0][t][0] * f0 + xs[2 + 2 * t] * f1;
        o[0][t][1] = o[0][t][1] * f0 + xs[3 + 2 * t] * f1;
      }
    } else {
      do_red = false;
    }
  }""",
    """  if (L == 1 && !CONT && !(OKC_NOXCH1 && !M_CL)) {
    // the other key blocks' warps (wsub >= 1) hand their state to the wsub == 0 warp of the same head through
    // shared memory (OKC_NOXCH1: outside cluster mode every warp adds its own rescaled state to the workspace instead)
    constexpr int NXW = NT / 64 - 1;   // partner warps per head (1 with 4 warps, 3 with 8)
    cp_wait<0>();
    csync<WS, NT>();
    float* xs = reinterpret_cast<float*>(S.k) + (((wsub > 0 ? wsub - 1 : 0) * 2 + (warp & 1)) * 32 + lane) * 18;
    if (wsub > 0) {
      xs[0] = m_r[0];
      xs[1] = l_r[0];
#pragma unroll
      for (int t = 0; t < 8; ++t) { xs[2 + 2 * t] = o[0][t][0]; xs[3 + 2 * t] = o[0][t][1]; }
    }
    csync<WS, NT>();
    if (wsub == 0) {
      float mq[NXW], lq[NXW];
      float mm = m_r[0];
#pragma unroll
      for (int q = 0; q < NXW; ++q) {
        mq[q] = xs[(q * 2 * 32) * 18];
        lq[q] = xs[(q * 2 * 32) * 18 + 1];
        mm = fmaxf(mm, mq[q]);
      }
      const float f0 = (m_r[0] == -INFINITY) ? 0.f : ex2(m_r[0] - mm);
      l_r[0] *= f0;
#pragma unroll
      for (int t = 0; t < 8; ++t) { o[0][t][0] *= f0; o[0][t][1] *= f0; }
#pragma unroll
      for (int q = 0; q < NXW; ++q) {
        const float fq = (mq[q] == -INFINITY) ? 0.f : ex2(mq[q] - mm);
        const float* xq = xs + (q * 2 * 32) * 18;
        l_r[0] += lq[q] * fq;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
          o[0][t][0] += xq[2 + 2 * t] * fq;
          o[0][t][1] += xq[3 + 2 * t] * fq;
        }
      }
      m_r[0] = mm;
    } else {
      do_red = false;
    }
  }""")

# 6. fin_nc1: at most 128 threads take part (16 rows x 8 threads); NT 256 CTAs: the rest return
rep("""template <int L, int NT>
__device__ __forceinline__ void fin_nc1(const Params& p, const int b, const int pair, const int bp, const float vscale,
                                     const int dry) {
  const int tid = threadIdx.x;""",
    """template <int L, int NT>
__device__ __forceinline__ void fin_nc1(const Params& p, const int b, const int pair, const int bp, const float vscale,
                                     const int dry) {
  const int tid = threadIdx.x;
  if (tid >= NT) return;   // (NT 256 CTAs at q_len 1: the first 128 threads finalize; no barrier below)""")
rep("    if (p.cs <= 1 && !p.coop && p.nc == 1) fin_nc1<L, NT>(p, b, pair, bp, 0.f, 1);",
    "    if (p.cs <= 1 && !p.coop && p.nc == 1) fin_nc1<L, (L == 1 && NT > 128) ? 128 : NT>(p, b, pair, bp, 0.f, 1);")
rep("    fin_nc1<L, NT>(p, b, pair, bp, vscale, 0);",
    "    fin_nc1<L, (L == 1 && NT > 128) ? 128 : NT>(p, b, pair, bp, vscale, 0);")

# 7. the relaxed-ticket probe
rep("""  if (tid == 0) {
    const int mine = end - start;
    int old;
    if constexpr (MRG2) asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\\n" : "=r"(old) : "l"(p.ws_cnt + bp), "r"(mine) : "memory");
    else old = atomicAdd(p.ws_cnt + bp, mine);
    S.flag = (old + mine == n_keys);
  }""",
    """  if (tid == 0) {
    const int mine = end - start;
    int old;
    if constexpr (OKC_XP_RELAXTICKET) old = atomicAdd(p.ws_cnt + bp, mine);   // (probe: no release / acquire)
    else if constexpr (MRG2) asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\\n" : "=r"(old) : "l"(p.ws_cnt + bp), "r"(mine) : "memory");
    else old = atomicAdd(p.ws_cnt + bp, mine);
    S.flag = (old + mine == n_keys);
  }""")
p.write_text(s)
print("patched", p)
