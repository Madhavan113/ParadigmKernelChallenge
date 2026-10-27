"""Add the flat-kernel probe switches (OKC_XNOWAITF, OKC_NOCOMPUTE for q8 flat) and OKC_FLATONLY (bench builds)."""
import re
P = "/Users/madhavanp/kerneloptimizer/.claude/worktrees/agent-aad2dcde86ab9f089/muse_attention_decode/submission.py"
s = open(P).read()
def sub1(old, new):
    global s
    assert s.count(old) == 1, old
    s = s.replace(old, new)
sub1("""#ifndef OKC_FENCEAR
""", """#ifndef OKC_XNOWAITF
#define OKC_XNOWAITF 0  // (probe) flat consumers: no wait for the stage's copies (garbage), the consumer-bound tile time
#endif
#ifndef OKC_FLATONLY
#define OKC_FLATONLY 0  // bench builds: only the flat kernels carry code (the per-request instances return at once)
#endif
#ifndef OKC_FENCEAR
""")
sub1("""  const int pair = blockIdx.z;
  DBG(0);
#if OKC_SMDBG
""", """  if constexpr (!FLAT && OKC_FLATONLY) return;   // (bench builds of the flat kernels only)
  const int pair = blockIdx.z;
  DBG(0);
#if OKC_SMDBG
""")
sub1("""  auto ws_acquire = [&](int i, int st) {
    mbar_wait_full(&S.full[st], ((tb + i) / STAGES) & 1);
""", """  auto ws_acquire = [&](int i, int st) {
    if constexpr (!OKC_XNOWAITF) mbar_wait_full(&S.full[st], ((tb + i) / STAGES) & 1);   // (probe: no wait)
""")
sub1("""      if constexpr (ADDR) {
        mbar_wait_full(&S.full[st_c], ph_c);
""", """      if constexpr (ADDR) {
        if constexpr (!OKC_XNOWAITF) mbar_wait_full(&S.full[st_c], ph_c);   // (probe: no wait)
""")
sub1("""    const int st = ADDR ? st_c : (tb + i) % STAGES;
    const uint32_t kaddr = smem_u32(S.k + st * STAGE_BYTES) + hk * 64;
""", """#if OKC_NOCOMPUTE
    if constexpr (WS && ADDR) {   // (probe) the stage is released untouched: the load-bound tile time
      __syncwarp();
      if (OKC_XP_ARRALL || lane == 0) mbar_arrive(&S.empty[st_c]);
      if (++st_c == STAGES) { st_c = 0; ph_c ^= 1u; }
      vtb_c = reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(vtb_c) ^ vtb_x);
      vdesc_c ^= vdesc_x;
      continue;
    }
#endif
    const int st = ADDR ? st_c : (tb + i) % STAGES;
    const uint32_t kaddr = smem_u32(S.k + st * STAGE_BYTES) + hk * 64;
""")
open(P, "w").write(s)
print("ok")
