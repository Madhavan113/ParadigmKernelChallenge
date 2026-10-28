"""Edit script (round 2 of skx): CUTLASS-style release/acquire (one thread, fence.acq_rel.gpu) instead of a per-thread
__threadfence, DIAG measurement levels, config tuples (nstage, fix, tiled, diag), self-test / autotune guards."""
import pathlib

p = pathlib.Path(__file__).resolve().parent.parent / "muse_bf16_proj/submission.py"
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:80])
    s = s.replace(old, new)


# 1. template: DIAG level
rep("""template <int BN, int NSTAGE, int FIX, bool TILED>
__global__ void __launch_bounds__(288, 1)
skx_kernel(""", """// DIAG (measurement builds only, never self-tested, picked or routed): 1 = main loops only (no export, no tickets),
// 2 = + the fragment export, 3 = + the release fence and tickets (no wait / reduce / epilogue).
template <int BN, int NSTAGE, int FIX, bool TILED, int DIAG>
__global__ void __launch_bounds__(288, 1)
skx_kernel(""")

# 2. export skip for DIAG 1
rep("""    if (t == 0) mbar_arrive(empty0 + 8 * pending);
    pending = -1;
    // export this segment's fragments (no fence yet: a straddler goes straight on to its second segment)
    float4* sl = slab + (size_t)(FIX == 0 ? rb : (s ? G : 0) + ci) * NF4 + tid;""",
    """    if (t == 0) mbar_arrive(empty0 + 8 * pending);
    pending = -1;
    if constexpr (DIAG == 1) continue;
    // export this segment's fragments (no fence yet: a straddler goes straight on to its second segment)
    float4* sl = slab + (size_t)(FIX == 0 ? rb : (s ? G : 0) + ci) * NF4 + tid;""")

# 3. fence + tickets: CUTLASS-style (CTA barrier, then one thread: fence.acq_rel.gpu + relaxed atomics + fence)
rep("""  __threadfence();
  named_bar(1, NTH);
  if (tid == 0) {
    for (int s = 0; s < nseg; ++s) s_tk[s] = atomicAdd(cnt + rb0 + s, 1);
  }
  named_bar(1, NTH);
  for (int s = 0; s < nseg; ++s) {""",
    """  if constexpr (DIAG == 1 || DIAG == 2) return;
  // every consumer thread's exports precede the tickets: CTA barrier, then one thread releases at gpu scope
  // (fence.acq_rel.gpu is cumulative over what the barrier made visible; a per-thread __threadfence = fence.sc.gpu
  // costs microseconds), takes the tickets and acquires
  named_bar(1, NTH);
  if (tid == 0) {
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
    for (int s = 0; s < nseg; ++s) s_tk[s] = atomicAdd(cnt + rb0 + s, 1);
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
  }
  named_bar(1, NTH);
  if constexpr (DIAG == 3) return;
  for (int s = 0; s < nseg; ++s) {""")

# 4. FIX 0 last arriver: no per-thread fence (thread 0 acquired above, the barrier orders the loads)
rep("""      if (s_tk[s] != P - 1) continue;   // not the last arriver of this row block
      __threadfence();
      const float4* src = slab + (size_t)rb * NF4 + tid;""",
    """      if (s_tk[s] != P - 1) continue;   // not the last arriver of this row block
      const float4* src = slab + (size_t)rb * NF4 + tid;""")

# 5. FIX 1 wait: thread 0 acquires, the barrier orders the other threads' loads
rep("""      named_bar(1, NTH);
      __threadfence();
      const int ua = s == 0 ? u0 : rb * C, ub = s == nseg - 1 ? u1 : (rb + 1) * C;
      const int lo =""",
    """      named_bar(1, NTH);
      const int ua = s == 0 ? u0 : rb * C, ub = s == nseg - 1 ? u1 : (rb + 1) * C;
      const int lo =""")

# 6. launcher: DIAG template parameter + dispatch + config tuples
rep("""template <int BN, int NSTAGE, int FIX, bool TILED>
static void launch_skx(""", """template <int BN, int NSTAGE, int FIX, bool TILED, int DIAG>
static void launch_skx(""")
rep("""  auto kern = skx_kernel<BN, NSTAGE, FIX, TILED>;""", """  auto kern = skx_kernel<BN, NSTAGE, FIX, TILED, DIAG>;""")
rep("""    fprintf(stderr, "[okc] skx<bn %d, ns %d, fix %d, tiled %d>: smem %d, CTAs/SM %d, grid %d\\n", BN, NSTAGE, FIX, (int)TILED,
            (int)smem, nb, grid);""",
    """    fprintf(stderr, "[okc] skx<bn %d, ns %d, fix %d, tiled %d, diag %d>: smem %d, CTAs/SM %d, grid %d\\n", BN, NSTAGE, FIX,
            (int)TILED, DIAG, (int)smem, nb, grid);""")
rep("""void skx(at::Tensor W, at::Tensor Wt, at::Tensor x, at::Tensor out, at::Tensor slab, at::Tensor cnt, int64_t M, int64_t N,
         int64_t K, int64_t bn, int64_t nstage, int64_t fix, int64_t tiled, int64_t grid) {""",
    """void skx(at::Tensor W, at::Tensor Wt, at::Tensor x, at::Tensor out, at::Tensor slab, at::Tensor cnt, int64_t M, int64_t N,
         int64_t K, int64_t bn, int64_t nstage, int64_t fix, int64_t tiled, int64_t diag, int64_t grid) {""")
rep("""void skx(torch::Tensor W, torch::Tensor Wt, torch::Tensor x, torch::Tensor out, torch::Tensor slab, torch::Tensor cnt,
         int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nstage, int64_t fix, int64_t tiled, int64_t grid);""",
    """void skx(torch::Tensor W, torch::Tensor Wt, torch::Tensor x, torch::Tensor out, torch::Tensor slab, torch::Tensor cnt,
         int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nstage, int64_t fix, int64_t tiled, int64_t diag, int64_t grid);""")
rep("""        return ('void skx(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, '
                'int64_t, int64_t, int64_t, int64_t, int64_t, int64_t) { TORCH_CHECK(false, "skx needs sm_90"); }')""",
    """        return ('void skx(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, '
                'int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t) { TORCH_CHECK(false, "skx needs sm_90"); }')""")
rep("""    for (layer, bn), cfgs in sorted(SKX_CONFIGS.items()):
        for ns, fix, tiled in cfgs:
            lines.append(f"if (bn == {bn} && nstage == {ns} && fix == {fix} && tiled == {tiled}) "
                         f"{{ launch_skx<{bn}, {ns}, {fix}, {str(bool(tiled)).lower()}>(W, Wt, x, out, slab, cnt, M, N, K, (int)grid); return; }}")""",
    """    for (layer, bn), cfgs in sorted(SKX_CONFIGS.items()):
        for ns, fix, tiled, diag in cfgs:
            lines.append(f"if (bn == {bn} && nstage == {ns} && fix == {fix} && tiled == {tiled} && diag == {diag}) "
                         f"{{ launch_skx<{bn}, {ns}, {fix}, {str(bool(tiled)).lower()}, {diag}>(W, Wt, x, out, slab, cnt, M, N, K, (int)grid); return; }}")""")

# 7. config table for check 2
rep("""# (layer, bn) -> [(nstage, fix, tiled), ...]: persistent stream-K candidates (grid = SM count, 1 CTA per SM, BM = 128);
# fix 0 = red.add slabs + last-arriver epilogue (self-cleaning), fix 1 = per-CTA slabs + co-resident distributed reduce;
# tiled 1 = W from the pre-swizzled contiguous copy (one bulk copy per unit).  Every entry is an autotune candidate.
SKX_CONFIGS = {
    ("qkv_gate", 128): [(6, 1, 0), (6, 0, 0), (6, 1, 1), (6, 0, 1)],
    ("pooler", 128): [(6, 1, 0), (6, 0, 0), (6, 1, 1), (6, 0, 1)],
    ("unpooler", 128): [(6, 1, 0), (6, 0, 0), (6, 1, 1), (6, 0, 1)],
    ("o", 128): [(6, 1, 0), (6, 0, 0), (6, 1, 1), (6, 0, 1)],
    ("qkv_gate", 256): [(4, 1, 0)],
    ("pooler", 256): [(4, 1, 0)],
    ("unpooler", 256): [(4, 1, 0)],
    ("o", 256): [(4, 1, 0)],
}
USE_SKX = True""",
    """# (layer, bn) -> [(nstage, fix, tiled, diag), ...]: persistent stream-K candidates (grid = SM count, 1 CTA per SM, BM = 128);
# fix 0 = red.add slabs + last-arriver epilogue (self-cleaning), fix 1 = per-CTA slabs + co-resident distributed reduce;
# tiled 1 = W from the pre-swizzled contiguous copy (one bulk copy per unit); diag > 0 = deliberately incomplete
# measurement build (timed by the autotuner, never self-tested, picked or routed).  Every entry is a candidate.
# Check 1 (e79f1d99, m=128 bench us): qkv fix1/fix0/tiled 28.3/31.1/28.1/30.7 vs gqx 18.0; o 25.0/25.8/24.6/25.4 vs
# torchT 10.5; pooler 36.2/37.6/35.8/37.7 vs gq 25.0; unpooler 37.8/37.8/37.1/37.4 vs torchT 22.5; m=256 (4,1,0) 38-54
# vs 23-31: ~0.5 us per unit (ingest-bound as modeled) but ~18 us of fixup overhead (per-thread fence.sc.gpu + ticket).
SKX_CONFIGS = {
    ("qkv_gate", 128): [(6, 1, 0, 0), (6, 0, 0, 0), (6, 1, 0, 1), (6, 1, 0, 2), (6, 1, 0, 3)],
    ("pooler", 128): [(6, 1, 0, 0), (6, 0, 0, 0), (6, 1, 0, 1), (6, 1, 0, 2), (6, 1, 0, 3)],
    ("unpooler", 128): [(6, 1, 0, 0), (6, 0, 0, 0), (6, 1, 0, 1), (6, 1, 0, 2), (6, 1, 0, 3)],
    ("o", 128): [(6, 1, 0, 0), (6, 0, 0, 0), (6, 1, 0, 1), (6, 1, 0, 2), (6, 1, 0, 3)],
}
USE_SKX = True


def _skx_diag(layer, m, ch):
    \"\"\"True for a diagnostic (incomplete) skx candidate: timed, never self-tested, picked or routed.\"\"\"
    return ch[0] == "skx" and SKX_CONFIGS[(layer, _skx_bn(m))][ch[1] if len(ch) > 1 else 0][3] > 0""")

rep("""    cfgs = [(layer, bn, ns, fix, tiled) for (layer, bn), lst in SKX_CONFIGS.items() for (ns, fix, tiled) in lst
            if layer in state["raw"]]""",
    """    cfgs = [(layer, bn, ns, fix, tiled) for (layer, bn), lst in SKX_CONFIGS.items() for (ns, fix, tiled, diag) in lst
            if layer in state["raw"]]""")
rep("""    ns, fix, tiled = SKX_CONFIGS[(layer, _skx_bn(m))][idx]
    sk = state["skx"]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    w = state["raw"][layer]
    wt = sk["tiled"][layer] if tiled else w
    _get_ext().skx(w, wt, x, out, sk["slab0" if fix == 0 else "slab1"], sk["cnt"][(layer, fix)], m, n, k, _skx_bn(m),
                   ns, fix, tiled, state["nsm"])
    return out""",
    """    ns, fix, tiled, diag = SKX_CONFIGS[(layer, _skx_bn(m))][idx]
    sk = state["skx"]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    w = state["raw"][layer]
    wt = sk["tiled"][layer] if tiled else w
    _get_ext().skx(w, wt, x, out, sk["slab0" if fix == 0 else "slab1"], sk["cnt"][(layer, fix)], m, n, k, _skx_bn(m),
                   ns, fix, tiled, diag, state["nsm"])
    return out""")
rep("""    state["skx"] = sk
""", """    state["skx"] = sk
    print(f"[okc] skx configs (nstage, fix, tiled, diag): {SKX_CONFIGS}", file=sys.stderr)
""")

# 8. self-test skips diag candidates; autotune never picks them
rep("""            for ch in _candidates(layer, m, state):
                if ch[0] == "torch":
                    continue
                key = _cand_key(layer, m, ch)""",
    """            for ch in _candidates(layer, m, state):
                if ch[0] == "torch" or _skx_diag(layer, m, ch):
                    continue
                key = _cand_key(layer, m, ch)""")
rep("""            med = [sorted(v)[len(v) // 2] for v in per]
            best = min(range(len(cands)), key=lambda i: med[i])
            state["choice"][(layer, m)] = cands[best]
            # CUTLASS is tuned for the scored m only (its partial token tiles are slow: m=129 +14 us, m=200 +4 us vs
            # cuBLAS): other m of the bucket take the best other candidate
            alt = [i for i in range(len(cands)) if cands[i][0] != "cutlass"]""",
    """            med = [sorted(v)[len(v) // 2] for v in per]
            real = [i for i in range(len(cands)) if not _skx_diag(layer, m, cands[i])]   # diagnostic builds: timed only
            best = min(real, key=lambda i: med[i])
            state["choice"][(layer, m)] = cands[best]
            # CUTLASS is tuned for the scored m only (its partial token tiles are slow: m=129 +14 us, m=200 +4 us vs
            # cuBLAS): other m of the bucket take the best other candidate
            alt = [i for i in real if cands[i][0] != "cutlass"]""")
p.write_text(s)
print("edits applied")
