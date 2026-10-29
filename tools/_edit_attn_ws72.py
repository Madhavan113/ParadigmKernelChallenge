#!/usr/bin/env python3
"""Patch: OKC_WS72 -- workspace rows of 72 floats (64 dims, the row sum in column 64, 7 pad) so one TMA bulk reduction
covers a CTA's whole partial (rows and sums) and the padded shared-memory copy stores conflict-free; OKC_BULKRED
becomes a single op for q_len 1 and 8 (the per-row level is dropped)."""
import pathlib, re, sys

p = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "muse_attention_decode/submission.py")
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:100])
    s = s.replace(old, new)


# 1. macros
rep("""#ifndef OKC_BULKRED
#define OKC_BULKRED 2   // per-request merge: the rescaled partial (L*16 rows x 64 fp32 + the row sums) goes to shared memory and
                        // is added into the workspace with cp.reduce.async.bulk (TMA, .add.f32) instead of 272 (q1) / 2176 (q8)
                        // red.global ops per CTA, so the ticket's release waits for a bulk group, not for contended atomics
                        // (H200: the acq_rel ticket cost 1.37 us after the reds at b16 q1, 0.61 us relaxed). 1: one op for the
                        // whole partial (q_len 1 only); 2: one 256-byte op per row from a padded smem partial (q_len 1 and 8)
#endif""",
    """#ifndef OKC_BULKRED
#define OKC_BULKRED 1   // per-request merge: the rescaled partial (L*16 rows x 64 fp32 + the row sums) goes to shared memory and
                        // ONE thread adds it into the workspace with one cp.reduce.async.bulk (TMA, .add.f32) instead of 272 (q1)
                        // / 2176 (q8) red.global ops per CTA, so the ticket's release waits for a bulk group, not for contended
                        // atomics (H200: the acq_rel ticket cost 1.37 us after the reds at b16 q1, 0.61 us relaxed; one op per
                        // row instead measured +0.2 us at q8: the TMA unit serializes the ops). Needs OKC_WS72.
#endif
#ifndef OKC_WS72
#define OKC_WS72 1      // workspace rows of 72 floats: the 64 dims, the row sum in column 64, 7 pad -- one contiguous bulk
                        // reduction per partial, and the padded shared-memory copy's row stores land on distinct banks
#endif""")

# 2. device constants / helper after the Params struct
rep("""  CUtensorMap tpt;      // producer mode: the page table as one int32 vector, 64-entry boxes (zero past the end)
};""",
    """  CUtensorMap tpt;      // producer mode: the page table as one int32 vector, 64-entry boxes (zero past the end)
};
constexpr int WSTR = OKC_WS72 ? 72 : 64;   // workspace row stride (floats)
constexpr int WSL = 64;                    // OKC_WS72: the row sum's column""")
rep("""__device__ __forceinline__ uint32_t smem_u32(const void* p) {""",
    """// the workspace row sum of row `row` (OKC_WS72: column 64 of the row; else the separate ws_l vector)
template <class IX>
__device__ __forceinline__ float* wsl(const Params& p, IX row) {
  if constexpr (OKC_WS72) return p.ws_acc + (long long)row * WSTR + WSL;
  else return p.ws_l + row;
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) {""")

# 3. workspace accesses: row stride, row sums
for frag in ("p.ws_acc + ((ix_t)(bp * p.nc + copy) * L * 16 + row) * 64 + (e & 15) * 4",
             "p.ws_acc + ((ix_t)(bp * p.nc + c) * L * 16 + row) * 64 + (e & 15) * 4"):
    n64 = s.count(frag)
    assert n64 in (1, 2), (n64, frag)
    s = s.replace(frag, frag.replace("* 64 +", "* WSTR +"))
rep("    for (int e = tid; e < RPC; e += NT) red_f32(p.ws_l + (bp * p.nc + copy) * L * 16 + row0 + e, RL[e]);",
    "    for (int e = tid; e < RPC; e += NT) red_f32(wsl(p, (bp * p.nc + copy) * L * 16 + row0 + e), RL[e]);")
rep("          float* lw = p.ws_l + (bp * p.nc + c) * L * 16 + row0 + e;",
    "          float* lw = wsl(p, (bp * p.nc + c) * L * 16 + row0 + e);")
n = s.count(") * 64 + part * 8;")
assert n == 9, n
s = s.replace(") * 64 + part * 8;", ") * WSTR + part * 8;")
# ws_l reads / zeroes in sum_copies / fin_nc1
s, k = re.subn(r"__ldcg\(p\.ws_l \+ ([^;]*?)\);", r"__ldcg(wsl(p, \1));", s)
assert k == 5, k
s, k = re.subn(r"p\.ws_l\[([^\]]*)\] = 0\.f;", r"*wsl(p, \1) = 0.f;", s)
assert k == 5, k
rep("    float* acc = p.ws_acc + ((ix_t)((bp * p.nc + copy) * L + j) * 16 + hh) * 64;",
    "    float* acc = p.ws_acc + ((ix_t)((bp * p.nc + copy) * L + j) * 16 + hh) * WSTR;")
rep("    if (tig == 0) red_f32(p.ws_l + ((bp * p.nc + copy) * L + j) * 16 + hh, lsum * f);",
    "    if (tig == 0) red_f32(wsl(p, ((bp * p.nc + copy) * L + j) * 16 + hh), lsum * f);")

# 4. the bulk merge: one op from a padded partial (rows of WSTR floats, the sum in column 64, pad zeroed)
rep("""  // OKC_BULKRED (no clusters): the partial goes to shared memory (the free stages) and TMA bulk reductions add it into
  // the workspace below. Level 1: one op for the whole partial ([L*16][64] contiguous, q_len 1); level 2: one op per
  // row from a padded partial (row stride 72 floats: the 8 rows a warp stores per instruction land on distinct banks)
  constexpr int BRL = (OKC_SM90 && !FLAT && !Cfg<L, FLAT>::S2 && !Cfg<L, FLAT>::F2 && !(OKC_BULKRED_NOTONE && ONE))
                          ? ((L == 1) ? OKC_BULKRED : (OKC_BULKRED == 2 ? 2 : 0)) : 0;
  constexpr bool BULKRED = BRL > 0;
  constexpr int PSTR = (BRL == 2) ? 72 : 64;        // smem partial row stride (floats)
  constexpr int LROW = L * 16 * PSTR;               // the row sums follow the rows (16-byte aligned)
  float* pbuf = reinterpret_cast<float*>((L == 1) ? S.v : S.k);   // q1: the V stages (the K stages hold the exchange)""",
    """  // OKC_BULKRED (no clusters): the partial goes to shared memory (the free stages) in the workspace's own row layout
  // ([L*16][72]: dims, the row sum in column 64, pad) and one TMA bulk reduction adds it into the workspace below
  constexpr bool BULKRED = OKC_BULKRED && OKC_WS72 && OKC_SM90 && !FLAT && !Cfg<L, FLAT>::S2 && !Cfg<L, FLAT>::F2 &&
                           !(OKC_BULKRED_NOTONE && ONE);
  constexpr int PSTR = WSTR;                        // smem partial row stride (floats)
  float* pbuf = reinterpret_cast<float*>((L == 1) ? S.v : S.k);   // q1: the V stages (the K stages hold the exchange)""")
rep("""      if (tig == 0) pbuf[LROW + hh] = lsum * f;
      continue;
    } else if constexpr (BULKRED) {   // natural dims: n-tile t, column c <-> dim 8*t + 2*tig + c; row j*16 + hh""",
    """      if (tig == 0) {   // the row sum in column 64, the pad zeroed (the bulk op adds whole rows)
        *reinterpret_cast<float4*>(pbuf + hh * PSTR + WSL) = make_float4(lsum * f, 0.f, 0.f, 0.f);
        *reinterpret_cast<float4*>(pbuf + hh * PSTR + WSL + 4) = make_float4(0.f, 0.f, 0.f, 0.f);
      }
      continue;
    } else if constexpr (BULKRED) {   // natural dims: n-tile t, column c <-> dim 8*t + 2*tig + c; row j*16 + hh""")
rep("""      if (tig == 0) pbuf[LROW + row] = lsum * f;
      continue;
    } else if constexpr (L == 1) {""",
    """      if (tig == 0) {
        *reinterpret_cast<float4*>(pbuf + row * PSTR + WSL) = make_float4(lsum * f, 0.f, 0.f, 0.f);
        *reinterpret_cast<float4*>(pbuf + row * PSTR + WSL + 4) = make_float4(0.f, 0.f, 0.f, 0.f);
      }
      continue;
    } else if constexpr (L == 1) {""")
rep("""    constexpr int NR = L * 16;
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
  }""",
    """    constexpr int NR = L * 16;
    if (tid == 0) {
      bulk_red_add_f32(p.ws_acc + (ix_t)((bp * p.nc + copy) * L) * 16 * WSTR, pbuf, NR * WSTR * 4);
      bulk_commit();
      bulk_wait_all();
      fence_proxy_async_global();   // ... performed, and ordered before the ticket's release
    }
  }""")

assert "p.ws_l" not in s.replace("p.ws_l = ws_l", "").replace("return p.ws_l", ""), [m.start() for m in re.finditer(r"p\.ws_l", s)]
# 5. host: the workspace rows are WSTR floats
rep("""        ws = (torch.zeros(n * 64, dtype=torch.float32, device=device),""",
    """        ws = (torch.zeros(n * 72, dtype=torch.float32, device=device),   # rows of 72 floats (OKC_WS72: the row sum in column 64)""")
rep("""    if ws is None or ws[0].numel() < n_rows * 64:""", """    if ws is None or ws[0].numel() < n_rows * 72:""")
p.write_text(s)
print("patched", p)
