#!/usr/bin/env python3
"""Patch: OKC_BULKRED -- the q_len 1 per-request merge adds its rescaled partial into the workspace with one TMA bulk
reduction (cp.reduce.async.bulk .add.f32 from shared memory) instead of 272 red.global.v4 ops per CTA, so the ticket's
release fence has no contended atomics to wait for."""
import pathlib, sys

p = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "muse_attention_decode/submission.py")
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


# 1. macro
rep("""#ifndef OKC_XP_RELAXTICKET
#define OKC_XP_RELAXTICKET 0   // (timing probe only, wrong results possible) last-CTA ticket as a relaxed atomic: no membar
#endif""",
    """#ifndef OKC_XP_RELAXTICKET
#define OKC_XP_RELAXTICKET 0   // (timing probe only, wrong results possible) last-CTA ticket as a relaxed atomic: no membar
#endif
#ifndef OKC_BULKRED
#define OKC_BULKRED 1   // q_len 1 per-request merge: the rescaled partial (16 rows x 64 fp32 + 16 row sums) goes to shared memory
                        // and ONE thread adds it into the workspace with cp.reduce.async.bulk (TMA, .add.f32) instead of 272
                        // red.global.v4 per CTA; the ticket's release then waits for one bulk group, not for contended atomics
                        // (H200: the acq_rel ticket cost 1.37 us after the reds, 0.61 us relaxed)
#endif""")

# 2. helpers (after bulk_load)
rep("""__device__ __forceinline__ void tma_load_1d(void* dst, const CUtensorMap* map, int c0, uint64_t* bar) {""",
    """// TMA bulk reduction: global[dst .. dst + bytes) += shared[src ..] (fp32 add), tracked by the thread's bulk async-group
__device__ __forceinline__ void bulk_red_add_f32(float* dst, const void* src, int bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;\\n"
               ::"l"(dst), "r"(smem_u32(src)), "r"(bytes) : "memory");
#endif
}
__device__ __forceinline__ void bulk_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("cp.async.bulk.commit_group;\\n" ::: "memory");
#endif
}
__device__ __forceinline__ void bulk_wait_all() {   // every bulk op of this thread complete (writes performed)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("cp.async.bulk.wait_group 0;\\n" ::: "memory");
#endif
}
__device__ __forceinline__ void fence_proxy_async_global() {   // async-proxy global writes ordered before generic ones
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("fence.proxy.async.global;\\n" ::: "memory");
#endif
}
__device__ __forceinline__ void tma_load_1d(void* dst, const CUtensorMap* map, int c0, uint64_t* bar) {""")

# 3. the merge
rep("""  if (do_red) {   // (one branch, not one per row: the post-loop merge runs straight-line from a cold cache)
#pragma unroll
  for (int r = 0; r < NROW; ++r) {
    const int mt = r >> 1, hf = r & 1;
    const int j = (L == 1) ? 0 : 2 * (wsub + mt) + hf;
    const int hh = hk * 8 + g;
    const float lsum = quad_sum(l_r[r]);
    const float f = (m_r[r] == -INFINITY) ? 0.f : ex2(fminf(m_r[r], 126.f));
    float* acc = p.ws_acc + ((ix_t)((bp * p.nc + copy) * L + j) * 16 + hh) * 64;
    if constexpr (L == 1) {
#pragma unroll
      for (int c = 0; c < 2; ++c)
#pragma unroll
        for (int t4 = 0; t4 < 8; t4 += 4)
          red_v4(acc + 16 * tig + 8 * c + t4, o[mt][t4][2 * hf + c] * f, o[mt][t4 + 1][2 * hf + c] * f,
                 o[mt][t4 + 2][2 * hf + c] * f, o[mt][t4 + 3][2 * hf + c] * f);
    } else if constexpr (OKC_RED4) {""",
    """  // OKC_BULKRED (q_len 1, no clusters): the partial goes to shared memory (the free V stages), one thread adds it
  // into the workspace with a TMA bulk reduction below
  constexpr bool BULKRED = OKC_BULKRED && !FLAT && L == 1 && OKC_SM90;
  float* pbuf = reinterpret_cast<float*>(S.v);   // [16 rows][64 dims] fp32, then 16 row sums (16-byte aligned)
  if (do_red) {   // (one branch, not one per row: the post-loop merge runs straight-line from a cold cache)
#pragma unroll
  for (int r = 0; r < NROW; ++r) {
    const int mt = r >> 1, hf = r & 1;
    const int j = (L == 1) ? 0 : 2 * (wsub + mt) + hf;
    const int hh = hk * 8 + g;
    const float lsum = quad_sum(l_r[r]);
    const float f = (m_r[r] == -INFINITY) ? 0.f : ex2(fminf(m_r[r], 126.f));
    float* acc = p.ws_acc + ((ix_t)((bp * p.nc + copy) * L + j) * 16 + hh) * 64;
    if constexpr (BULKRED) {
#pragma unroll
      for (int c = 0; c < 2; ++c)
#pragma unroll
        for (int t4 = 0; t4 < 8; t4 += 4)
          *reinterpret_cast<float4*>(pbuf + hh * 64 + 16 * tig + 8 * c + t4) =
              make_float4(o[mt][t4][2 * hf + c] * f, o[mt][t4 + 1][2 * hf + c] * f,
                          o[mt][t4 + 2][2 * hf + c] * f, o[mt][t4 + 3][2 * hf + c] * f);
      if (tig == 0) pbuf[1024 + hh] = lsum * f;
      continue;
    } else if constexpr (L == 1) {
#pragma unroll
      for (int c = 0; c < 2; ++c)
#pragma unroll
        for (int t4 = 0; t4 < 8; t4 += 4)
          red_v4(acc + 16 * tig + 8 * c + t4, o[mt][t4][2 * hf + c] * f, o[mt][t4 + 1][2 * hf + c] * f,
                 o[mt][t4 + 2][2 * hf + c] * f, o[mt][t4 + 3][2 * hf + c] * f);
    } else if constexpr (OKC_RED4) {""")
rep("""  if (OKC_MRGDBG) DBG(7);   // (debug: reds issued)
  // OKC_MERGE2 (last-CTA finalizes): no per-thread fences; one acq_rel atomic by thread 0 after the CTA barrier""",
    """  if constexpr (BULKRED) {
    fence_proxy_async();   // the partial's st.shared, visible to the async proxy
    csync<WS, NT>();
    if (tid == 0) {
      bulk_red_add_f32(p.ws_acc + (ix_t)((bp * p.nc + copy) * L) * 16 * 64, pbuf, L * 16 * 64 * 4);
      bulk_red_add_f32(p.ws_l + (bp * p.nc + copy) * L * 16, pbuf + 1024, L * 16 * 4);
      bulk_commit();
      bulk_wait_all();
      fence_proxy_async_global();   // ... performed, and ordered before the ticket's release
    }
  }
  if (OKC_MRGDBG) DBG(7);   // (debug: reds issued)
  // OKC_MERGE2 (last-CTA finalizes): no per-thread fences; one acq_rel atomic by thread 0 after the CTA barrier""")
p.write_text(s)
print("patched", p)
