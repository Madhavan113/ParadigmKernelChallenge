#!/usr/bin/env python3
"""Patch: generalize OKC_BULKRED -- level 1: one bulk op for the whole partial (q_len 1 only); level 2: one bulk op per
row (q_len 1 and 8) from a padded shared-memory partial (row stride 72 floats: conflict-free float2 / float4 stores),
the row sums as one more op. OKC_BULKRED_NOTONE: keep the red.v4 path in the ONE builds (single-tile splits)."""
import pathlib, sys

p = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "muse_attention_decode/submission.py")
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


rep("""#ifndef OKC_BULKRED
#define OKC_BULKRED 1   // q_len 1 per-request merge: the rescaled partial (16 rows x 64 fp32 + 16 row sums) goes to shared memory
                        // and ONE thread adds it into the workspace with cp.reduce.async.bulk (TMA, .add.f32) instead of 272
                        // red.global.v4 per CTA; the ticket's release then waits for one bulk group, not for contended atomics
                        // (H200: the acq_rel ticket cost 1.37 us after the reds, 0.61 us relaxed)
#endif""",
    """#ifndef OKC_BULKRED
#define OKC_BULKRED 2   // per-request merge: the rescaled partial (L*16 rows x 64 fp32 + the row sums) goes to shared memory and
                        // is added into the workspace with cp.reduce.async.bulk (TMA, .add.f32) instead of 272 (q1) / 2176 (q8)
                        // red.global ops per CTA, so the ticket's release waits for a bulk group, not for contended atomics
                        // (H200: the acq_rel ticket cost 1.37 us after the reds at b16 q1, 0.61 us relaxed). 1: one op for the
                        // whole partial (q_len 1 only); 2: one 256-byte op per row from a padded smem partial (q_len 1 and 8)
#endif
#ifndef OKC_BULKRED_NOTONE
#define OKC_BULKRED_NOTONE 1   // ... not in the ONE builds (one tile per split: b1 sliding measured -0.27 us with the reds)
#endif""")

rep("""  // OKC_BULKRED (q_len 1, no clusters): the partial goes to shared memory (the free V stages), one thread adds it
  // into the workspace with a TMA bulk reduction below
  constexpr bool BULKRED = OKC_BULKRED && !FLAT && L == 1 && OKC_SM90;
  float* pbuf = reinterpret_cast<float*>(S.v);   // [16 rows][64 dims] fp32, then 16 row sums (16-byte aligned)""",
    """  // OKC_BULKRED (no clusters): the partial goes to shared memory (the free stages) and TMA bulk reductions add it into
  // the workspace below. Level 1: one op for the whole partial ([L*16][64] contiguous, q_len 1); level 2: one op per
  // row from a padded partial (row stride 72 floats: the 8 rows a warp stores per instruction land on distinct banks)
  constexpr int BRL = (OKC_SM90 && !FLAT && !Cfg<L, FLAT>::S2 && !Cfg<L, FLAT>::F2 && !(OKC_BULKRED_NOTONE && ONE))
                          ? ((L == 1) ? OKC_BULKRED : (OKC_BULKRED == 2 ? 2 : 0)) : 0;
  constexpr bool BULKRED = BRL > 0;
  constexpr int PSTR = (BRL == 2) ? 72 : 64;        // smem partial row stride (floats)
  constexpr int LROW = L * 16 * PSTR;               // the row sums follow the rows (16-byte aligned)
  float* pbuf = reinterpret_cast<float*>((L == 1) ? S.v : S.k);   // q1: the V stages (the K stages hold the exchange)""")

rep("""    if constexpr (BULKRED) {
#pragma unroll
      for (int c = 0; c < 2; ++c)
#pragma unroll
        for (int t4 = 0; t4 < 8; t4 += 4)
          *reinterpret_cast<float4*>(pbuf + hh * 64 + 16 * tig + 8 * c + t4) =
              make_float4(o[mt][t4][2 * hf + c] * f, o[mt][t4 + 1][2 * hf + c] * f,
                          o[mt][t4 + 2][2 * hf + c] * f, o[mt][t4 + 3][2 * hf + c] * f);
      if (tig == 0) pbuf[1024 + hh] = lsum * f;
      continue;
    } else if constexpr (L == 1) {""",
    """    if constexpr (BULKRED && L == 1) {   // permuted dims (see below): float4 of n-tiles t4..t4+3 at dim 16*tig + 8*c + t4
#pragma unroll
      for (int c = 0; c < 2; ++c)
#pragma unroll
        for (int t4 = 0; t4 < 8; t4 += 4)
          *reinterpret_cast<float4*>(pbuf + hh * PSTR + 16 * tig + 8 * c + t4) =
              make_float4(o[mt][t4][2 * hf + c] * f, o[mt][t4 + 1][2 * hf + c] * f,
                          o[mt][t4 + 2][2 * hf + c] * f, o[mt][t4 + 3][2 * hf + c] * f);
      if (tig == 0) pbuf[LROW + hh] = lsum * f;
      continue;
    } else if constexpr (BULKRED) {   // natural dims: n-tile t, column c <-> dim 8*t + 2*tig + c; row j*16 + hh
      const int row = j * 16 + hh;
#pragma unroll
      for (int t = 0; t < 8; ++t)
        *reinterpret_cast<float2*>(pbuf + row * PSTR + 8 * t + 2 * tig) =
            make_float2(o[mt][t][2 * hf] * f, o[mt][t][2 * hf + 1] * f);
      if (tig == 0) pbuf[LROW + row] = lsum * f;
      continue;
    } else if constexpr (L == 1) {""")

rep("""  if constexpr (BULKRED) {
    fence_proxy_async();   // the partial's st.shared, visible to the async proxy
    csync<WS, NT>();
    if (tid == 0) {
      bulk_red_add_f32(p.ws_acc + (ix_t)((bp * p.nc + copy) * L) * 16 * 64, pbuf, L * 16 * 64 * 4);
      bulk_red_add_f32(p.ws_l + (bp * p.nc + copy) * L * 16, pbuf + 1024, L * 16 * 4);
      bulk_commit();
      bulk_wait_all();
      fence_proxy_async_global();   // ... performed, and ordered before the ticket's release
    }
  }""",
    """  if constexpr (BULKRED) {
    fence_proxy_async();   // the partial's st.shared, visible to the async proxy
    csync<WS, NT>();
    constexpr int NR = L * 16;
    if constexpr (BRL == 1) {
      if (tid == 0) {
        bulk_red_add_f32(p.ws_acc + (ix_t)((bp * p.nc + copy) * L) * 16 * 64, pbuf, NR * 64 * 4);
        bulk_red_add_f32(p.ws_l + (bp * p.nc + copy) * L * 16, pbuf + LROW, NR * 4);
        bulk_commit();
        bulk_wait_all();
        fence_proxy_async_global();   // ... performed, and ordered before the ticket's release
      }
    } else {
      if (tid < NR) bulk_red_add_f32(p.ws_acc + ((ix_t)(bp * p.nc + copy) * NR + tid) * 64, pbuf + tid * PSTR, 256);
      else if (tid == NR) bulk_red_add_f32(p.ws_l + (bp * p.nc + copy) * NR, pbuf + LROW, NR * 4);
      if (tid <= NR) {
        bulk_commit();
        bulk_wait_all();
        fence_proxy_async_global();   // ... performed, and ordered (through the CTA barrier) before the ticket's release
      }
    }
  }""")
p.write_text(s)
print("patched", p)
