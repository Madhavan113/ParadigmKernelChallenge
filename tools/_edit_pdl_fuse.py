"""Edit: port the decode layer's sk2 PDL prologue (sk2_set_pdl / griddepcontrol) and the fused pooler + router launch
(FUSE template flag, sk2f entry point, fused_pooler_router) from muse_decode_layer/proj_fork.py (branch
worktree-agent-ab41706bb5b7e284f) onto the standalone file, behind flags that default off (SK2_PDL = 0; sk2f only when
called). Run from the worktree root: python3 tools/_edit_pdl_fuse.py"""
import pathlib
p = pathlib.Path("muse_bf16_proj/submission.py")
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:80])
    s = s.replace(old, new)


# ---- 1) PDL switch (host-side global read at every launch)
rep('''#ifndef SK2_DIAG
#define SK2_DIAG 0
#endif
// ============================================================================ sk2 (m <= 32): swapped operands
''', '''#ifndef SK2_DIAG
#define SK2_DIAG 0
#endif
// PDL (programmatic dependent launch) for the sk2 launches: 0 = off (plain launches); N > 0 = launch with programmatic
// stream serialization and issue the first N weight stages before griddepcontrol.wait (set by sk2_set_pdl). Only the
// decode layer sets it (there the previous kernel in the stream produces x); standalone scoring runs with 0.
static int g_sk2_pdl = 0;
void sk2_set_pdl(int64_t n) { g_sk2_pdl = (int)n; }
// ============================================================================ sk2 (m <= 32): swapped operands
''')

# ---- 2) kernel template / parameters
rep('''template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT,
          bool PK>
__global__ void __launch_bounds__((NWR * WSK + 1 + (COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1)) * 32, 1)
sk2_kernel(const __grid_constant__ CUtensorMap xmap, const uint8_t* __restrict__ Ws,
           const __nv_bfloat16* __restrict__ x, OutT* __restrict__ out,
           const int2* __restrict__ ptab, int pmax, uint32_t base4, int M, int N, int K, int dg,
           unsigned long long* __restrict__ dbg) {
''', '''template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT,
          bool PK, bool FUSE>
__global__ void __launch_bounds__((NWR * WSK + 1 + (COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1)) * 32, 1)
sk2_kernel(const __grid_constant__ CUtensorMap xmap, const uint8_t* __restrict__ Ws,
           const __nv_bfloat16* __restrict__ x, OutT* __restrict__ out,
           const int2* __restrict__ ptab, int pmax, uint32_t base4, int M, int N, int K, int dg,
           unsigned long long* __restrict__ dbg, const uint8_t* __restrict__ Wsr, float* __restrict__ outr,
           float* __restrict__ part, int* __restrict__ rcnt, int pdl) {
''')

# ---- 3) FUSE constants + PDL prologue depth
rep('''  const uint8_t* wsrc = Ws + (size_t)cta * WBY;
  const size_t wstride = (size_t)(gridDim.x / CL) * WBY;

  if (tid == 0) {
''', '''  const uint8_t* wsrc = Ws + (size_t)cta * WBY;
  const size_t wstride = (size_t)(gridDim.x / CL) * WBY;
  // FUSE (pooler + router in one launch): after its pooler stages every CTA streams one router piece: 8 router rows
  // (tile rt = cta / 4) over one quarter of K (rq = cta % 4), from the router's own stage-major hi|lo stream
  // ([16 stages][32 tiles][8 warps][2048 B]); 4 more stages through the same ring
  constexpr int NSTR = FUSE ? 4 : 0;
  constexpr int WBYR = NW * 2048;
  static_assert(!FUSE || (COMP && CL == 1 && UPW == 1 && UPS == 8 && WBYR <= WBY), "sk2 fused: pooler-like config only");
  const int rt = cta >> 2, rq = cta & 3;
  const uint8_t* wsr = FUSE ? Wsr + (size_t)rt * WBYR : nullptr;
  const int nstt = nst + NSTR;
  // stages issued by the prologue: 1, or under PDL up to pdl (CL == 1 only; never past this CTA's own stages)
  const int P = (CL == 1 && pdl > 0) ? min(pdl, min(NSTAGE, nst)) : 1;

  if (tid == 0) {
''')

# ---- 4) prologue: P stages' weights before griddepcontrol.wait, their x boxes after it
rep('''    asm volatile("prefetch.tensormap [%0];" :: "l"(&xmap) : "memory");
    SK2_DLY_SEED_SET();
    mbar_init(full0, 1);
    if constexpr (CL > 1) {
''', '''    asm volatile("prefetch.tensormap [%0];" :: "l"(&xmap) : "memory");
    SK2_DLY_SEED_SET();
    mbar_init(full0, 1);
    if constexpr (CL == 1)   // PDL: the first P stages' weights go out before the wait; their barriers first
      for (int i = 1; i < P; ++i) mbar_init(full0 + 8 * i, 1);
    if constexpr (CL > 1) {
''')
rep('''    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_expect_tx(full0, ((dgv & 4) ? 0 : XBYr) + WBY);
    bulk_g2s(sbase + XBY, wsrc + (size_t)gs0 * wstride, WBY, full0);
    if (!(dgv & 4)) tma_load_3d(sbase, &xmap, full0, 0, 0, gs0 * UPS);
    if (COMP) {   // this CTA's escape table
''', '''    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    // weights never depend on the previous kernel: under PDL (pdl > 0) they stream while it finishes; x (its output)
    // only after griddepcontrol.wait (a no-op when the launch has no programmatic dependency)
    for (int st = 0; st < P; ++st) {
      mbar_expect_tx(full0 + 8 * st, ((dgv & 4) ? 0 : XBYr) + WBY);
      bulk_g2s(sbase + st * SST + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * st);
    }
    if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
    if (!(dgv & 4))
      for (int st = 0; st < P; ++st) tma_load_3d(sbase + st * SST, &xmap, full0 + 8 * st, 0, 0, (gs0 + st) * UPS);
    if (COMP) {   // this CTA's escape table
''')
rep('''    if constexpr (CL == 1) {
      mbar_init(empty0, NW + (COMP ? NESC : 0));
      for (int i = 1; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NW + (COMP ? NESC : 0)); }
      if (!COMP) mbar_init(pbar, 1);
''', '''    if constexpr (CL == 1) {
      mbar_init(empty0, NW + (COMP ? NESC : 0));
      for (int i = 1; i < NSTAGE; ++i) {
        if (i >= P) mbar_init(full0 + 8 * i, 1);
        mbar_init(empty0 + 8 * i, NW + (COMP ? NESC : 0));
      }
      if (!COMP) mbar_init(pbar, 1);
''')
rep('''  __syncthreads();
  if constexpr (CL > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");   // inits fenced above

  if (warp == NW) {
    // producer warp: lane 0 refills the ring
    if (lane == 0) {
      for (int st = 1; st < nst; ++st) {
        const int slot = st % NSTAGE;
        if (st >= NSTAGE) SK2_WAIT(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
        if (trace) tr[16 + 3 * st] = gtime();
        const uint32_t s0 = sbase + slot * SST;
        mbar_expect_tx(full0 + 8 * slot, ((dgv & 4) ? 0 : XBYr) + WBY);
        bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * slot);
        if (!(dgv & 4)) tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, (gs0 + st) * UPS);
      }
    }
''', '''  __syncthreads();
  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");   // every thread: x and the outputs are ours from here
  if constexpr (CL > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");   // inits fenced above

  if (warp == NW) {
    // producer warp: lane 0 refills the ring
    if (lane == 0) {
      for (int st = P; st < nstt; ++st) {
        const int slot = st % NSTAGE;
        if (st >= NSTAGE) SK2_WAIT(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
        if (trace) tr[16 + 3 * st] = gtime();
        const uint32_t s0 = sbase + slot * SST;
        if (!FUSE || st < nst) {
          mbar_expect_tx(full0 + 8 * slot, ((dgv & 4) ? 0 : XBYr) + WBY);
          bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * slot);
          if (!(dgv & 4)) tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, (gs0 + st) * UPS);
        } else {   // router piece: global router stage rq * 4 + j of tile rt
          const int gsr = rq * NSTR + (st - nst);
          mbar_expect_tx(full0 + 8 * slot, XBYr + WBYR);
          bulk_g2s(s0 + XBY, wsr + (size_t)gsr * (32 * WBYR), WBYR, full0 + 8 * slot);
          tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, gsr * UPS);
        }
      }
    }
''')

# ---- 5) escape warps: observe the router stages too (hand-back only)
rep('''      int pe = 1;
      for (int st = 0; st < nst; ++st) {
        const int kbeg = (gs0 + st) * KST, kend = kbeg + KST;
        const int slot = st % NSTAGE;
''', '''      int pe = 1;
      for (int st = 0; st < nstt; ++st) {
        const int slot = st % NSTAGE;
        if (FUSE && st >= nst) {   // router stages: nothing to correct, only the slot hand-back (in order, as below)
          SK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);
          __syncwarp();
          if (lane == 0) mbar_arrive(empty0 + 8 * slot);   // every escape warp hands every stage back
          continue;
        }
        const int kbeg = (gs0 + st) * KST, kend = kbeg + KST;
''')

# ---- 6) after the main loop: PDL trigger + the fused router piece
rep('''  if (trace && lane == 0) tr[1 + warp] = gtime();   // per consumer warp: main loop done

  // ---------------- epilogue: K-warps reduce through shared memory (the ring is free now)
''', '''  if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");   // the next kernel may start its prologue
  if constexpr (FUSE) {
    // router piece: 8 rows (one tile) over 4 stages of 8 units; hi and lo BF16 tiles into two FP32 accumulators
    float accr[2][MT][4];
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) accr[h][mt][0] = accr[h][mt][1] = accr[h][mt][2] = accr[h][mt][3] = 0.f;
    for (int st = nst; st < nstt; ++st) {
      const int slot = st % NSTAGE;
      mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
      const uint32_t s0 = sbase + slot * SST;
      const int ui = wk;
      const uint32_t xt = s0 + ui * XU;
      const uint32_t a0 = s0 + XBY + warp * 2048 + lane * 32;
      uint32_t bh[8], bo[8];
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bh[0]), "=r"(bh[1]), "=r"(bh[2]), "=r"(bh[3]) : "r"(a0));
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bh[4]), "=r"(bh[5]), "=r"(bh[6]), "=r"(bh[7]) : "r"(a0 + 16));
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bo[0]), "=r"(bo[1]), "=r"(bo[2]), "=r"(bo[3]) : "r"(a0 + 1024));
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bo[4]), "=r"(bo[5]), "=r"(bo[6]), "=r"(bo[7]) : "r"(a0 + 1040));
      uint32_t a[4][MT][4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
          ldsm_x4((PK && trow[mt] >= M) ? zrow
                      : xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * Mb + trow[mt]) & 7)) << 4),
                  a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          mma_bf16(accr[0][mt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], bh[2 * j], bh[2 * j + 1]);
          mma_bf16(accr[1][mt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], bo[2 * j], bo[2 * j + 1]);
        }
    }
    // k-warp reduction of the 8 x TOKS piece through the (now idle) ring: redr[wk][tok][8]
    if constexpr (COMP) mbar_wait(cbar, 0);   // the escape warps are done with the ring too
    named_bar(1, NC);
    float* redr = reinterpret_cast<float*>(gbase);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      const int row = 2 * t4, tok = mt * 16 + g;
      redr[(wk * TOKS + tok) * 8 + row] = accr[0][mt][0] + accr[1][mt][0];
      redr[(wk * TOKS + tok) * 8 + row + 1] = accr[0][mt][1] + accr[1][mt][1];
      redr[(wk * TOKS + tok + 8) * 8 + row] = accr[0][mt][2] + accr[1][mt][2];
      redr[(wk * TOKS + tok + 8) * 8 + row + 1] = accr[0][mt][3] + accr[1][mt][3];
    }
    named_bar(1, NC);
    // this CTA's quarter-K partial -> part[rt][rq][TOKS][8]; the last of the 4 quarters (release/acquire counter)
    // adds them in quarter order (deterministic) and writes the router output
    float* pp = part + ((size_t)rt * 4 + rq) * 32 * 8;
    for (int i = tid; i < TOKS * 8; i += NC) {
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < WSK; ++w) v += redr[(w * TOKS + (i >> 3)) * 8 + (i & 7)];
      __stcg(pp + i, v);
    }
    named_bar(1, NC);
    __shared__ int s_rlast;
    if (tid == 0) {
      int old;
      asm volatile("atom.add.release.gpu.global.s32 %0, [%1], 1;" : "=r"(old) : "l"(rcnt + rt) : "memory");
      s_rlast = (old & 3) == 3;
      if (s_rlast) asm volatile("fence.acq_rel.gpu;" ::: "memory");
    }
    named_bar(1, NC);
    if (s_rlast) {
      const float* pr = part + (size_t)rt * 4 * 32 * 8;
      for (int i = tid; i < M * 8; i += NC) {
        const float v = __ldcg(pr + i) + __ldcg(pr + 256 + i) + __ldcg(pr + 512 + i) + __ldcg(pr + 768 + i);
        outr[(size_t)(i >> 3) * 256 + rt * 8 + (i & 7)] = v;
      }
    }
    named_bar(1, NC);   // the ring (redr) is reused by the pooler epilogue below
  }
  if (trace && lane == 0) tr[1 + warp] = gtime();   // per consumer warp: main loop done

  // ---------------- epilogue: K-warps reduce through shared memory (the ring is free now)
''')

# ---- 7) launcher: FUSE template flag, extra arguments, PDL launch attribute
rep('''template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT>
static void launch_sk2(const at::Tensor& Ws, const at::Tensor& x, at::Tensor& out, const at::Tensor& ptab,
                       int64_t pmax, int64_t base4, int M, int N, int K, int64_t dg, unsigned long long* dbg) {
''', '''template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT,
          bool FUSE = false>
static void launch_sk2(const at::Tensor& Ws, const at::Tensor& x, at::Tensor& out, const at::Tensor& ptab,
                       int64_t pmax, int64_t base4, int M, int N, int K, int64_t dg, unsigned long long* dbg,
                       const uint8_t* wsr = nullptr, float* outr = nullptr, float* part = nullptr, int* rcnt = nullptr) {
''')
rep('''    auto kern = sk2_kernel<MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL, OutT, PK>;
''', '''    auto kern = sk2_kernel<MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL, OutT, PK, FUSE>;
''')
rep('''    const int2* tp = reinterpret_cast<const int2*>(ptab.data_ptr());
    if constexpr (CL > 1) {
      cudaLaunchConfig_t lc = {};
      lc.gridDim = dim3(N / R * CL);
      lc.blockDim = dim3((NW + 1 + NESC) * 32);
      lc.dynamicSmemBytes = smem;
      lc.stream = at::cuda::getCurrentCUDAStream();
      cudaLaunchAttribute at_[1];
      at_[0].id = cudaLaunchAttributeClusterDimension;
      at_[0].val.clusterDim.x = CL;
      at_[0].val.clusterDim.y = 1;
      at_[0].val.clusterDim.z = 1;
      lc.attrs = at_;
      lc.numAttrs = 1;
      C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, xm, wsp, xp, op, tp, (int)pmax, (uint32_t)base4, M, N, K, (int)dg, dbg));
    } else {
      kern<<<N / R, (NW + 1 + NESC) * 32, smem, at::cuda::getCurrentCUDAStream()>>>(xm, wsp, xp, op, tp, (int)pmax,
                                                                               (uint32_t)base4, M, N, K, (int)dg, dbg);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
''', '''    const int2* tp = reinterpret_cast<const int2*>(ptab.data_ptr());
    const int pdl = g_sk2_pdl;
    if (CL > 1 || pdl > 0) {
      cudaLaunchConfig_t lc = {};
      lc.gridDim = dim3(N / R * CL);
      lc.blockDim = dim3((NW + 1 + NESC) * 32);
      lc.dynamicSmemBytes = smem;
      lc.stream = at::cuda::getCurrentCUDAStream();
      cudaLaunchAttribute at_[2];
      int na = 0;
      if (CL > 1) {
        at_[na].id = cudaLaunchAttributeClusterDimension;
        at_[na].val.clusterDim.x = CL;
        at_[na].val.clusterDim.y = 1;
        at_[na].val.clusterDim.z = 1;
        ++na;
      }
      if (pdl > 0) {   // programmatic dependent launch: may start while the previous kernel in the stream finishes
        at_[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        at_[na].val.programmaticStreamSerializationAllowed = 1;
        ++na;
      }
      lc.attrs = at_;
      lc.numAttrs = na;
      C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, xm, wsp, xp, op, tp, (int)pmax, (uint32_t)base4, M, N, K, (int)dg, dbg,
                                        wsr, outr, part, rcnt, pdl));
    } else {
      kern<<<N / R, (NW + 1 + NESC) * 32, smem, at::cuda::getCurrentCUDAStream()>>>(xm, wsp, xp, op, tp, (int)pmax,
                                                                               (uint32_t)base4, M, N, K, (int)dg, dbg,
                                                                               wsr, outr, part, rcnt, 0);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
''')
rep('''void sk2(at::Tensor Ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4,
         int64_t M, int64_t N, int64_t K, int64_t cfg, int64_t tokr, int64_t dg, at::Tensor dbgt) {
  unsigned long long* dbg = dbgt.numel() >= 8192 + 3 * 1024 ? reinterpret_cast<unsigned long long*>(dbgt.data_ptr()) : nullptr;
  @@SK2_DISPATCH@@
  TORCH_CHECK(false, "sk2: no config");
}
''', '''// fused pooler + router (decode layer): the `pooler` config streams its pooler rows, then one router piece per CTA
void sk2f(at::Tensor Ws, at::Tensor Wsr, at::Tensor x, at::Tensor out, at::Tensor outr, at::Tensor ptab, int64_t pmax,
          int64_t base4, at::Tensor part, at::Tensor rcnt, int64_t M, int64_t tokr) {
  @@SK2F_DISPATCH@@
  TORCH_CHECK(false, "sk2f: no config");
}

void sk2(at::Tensor Ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4,
         int64_t M, int64_t N, int64_t K, int64_t cfg, int64_t tokr, int64_t dg, at::Tensor dbgt) {
  unsigned long long* dbg = dbgt.numel() >= 8192 + 3 * 1024 ? reinterpret_cast<unsigned long long*>(dbgt.data_ptr()) : nullptr;
  @@SK2_DISPATCH@@
  TORCH_CHECK(false, "sk2: no config");
}
''')

# ---- 8) host: declarations, exports, dispatch generation, config, flag, fused entry
rep('''void sk2(torch::Tensor Ws, torch::Tensor x, torch::Tensor out, torch::Tensor ptab, int64_t pmax, int64_t base4,
         int64_t M, int64_t N, int64_t K, int64_t cfg, int64_t tokr, int64_t dg, torch::Tensor dbgt);
''', '''void sk2(torch::Tensor Ws, torch::Tensor x, torch::Tensor out, torch::Tensor ptab, int64_t pmax, int64_t base4,
         int64_t M, int64_t N, int64_t K, int64_t cfg, int64_t tokr, int64_t dg, torch::Tensor dbgt);
void sk2f(torch::Tensor Ws, torch::Tensor Wsr, torch::Tensor x, torch::Tensor out, torch::Tensor outr, torch::Tensor ptab,
          int64_t pmax, int64_t base4, torch::Tensor part, torch::Tensor rcnt, int64_t M, int64_t tokr);
void sk2_set_pdl(int64_t n);
''')
rep('''            functions=["skinny_tma_a", "gemm90", "wide", "sk2", "gq", "gqx", "rgemv", "sgemv", "cgemv", "rk"],
''', '''            functions=["skinny_tma_a", "gemm90", "wide", "sk2", "gq", "gqx", "rgemv", "sgemv", "cgemv", "rk", "sk2f",
                       "sk2_set_pdl"],
''')
rep('''            name=f"okc_bf16proj_v141_{arch}" + (f"_dly{_SK2_DELAY}" if _SK2_DELAY else ""),
''', '''            name=f"okc_bf16proj_v142_{arch}" + (f"_dly{_SK2_DELAY}" if _SK2_DELAY else ""),
''')
rep('''    if not sm90:
        return ('void sk2(at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, '
                'int64_t, int64_t, int64_t, at::Tensor) { TORCH_CHECK(false, "sk2 needs sm_90"); }')
''', '''    if not sm90:
        return ('void sk2(at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, '
                'int64_t, int64_t, int64_t, at::Tensor) { TORCH_CHECK(false, "sk2 needs sm_90"); }\\n'
                'void sk2f(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, '
                'at::Tensor, at::Tensor, int64_t, int64_t) { TORCH_CHECK(false, "sk2f needs sm_90"); }\\n'
                'void sk2_set_pdl(int64_t) {}')
''')
rep('''    return SK2_SRC.replace("@@SK2_DISPATCH@@", "\\n  ".join(lines))
''', '''    flines = []   # fused pooler + router: the `pooler` config (NT 4, WSK 8, UPW 1) with FUSE, every tokr
    for ci, cfg in enumerate(SK2_CONFIGS):
        name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
        if name != SK2_FUSED_CONFIG:
            continue
        for tokr in (8, 16, 32):
            hl, nstage = _sk2_geom(cfg, tokr)
            mt = 2 if tokr == 32 else 1
            nacc = 1 if (mt == 2 and nstage >= 4 and NT == 4) else NACC
            flines.append(f"if (tokr == {tokr}) {{ launch_sk2<{mt}, {tokr}, {NT}, {NWR}, {WSK}, {UPW}, {nstage}, true, false, "
                          f"{nacc}, 1, __nv_bfloat16, true>(Ws, x, out, ptab, pmax, base4, M, 4096, 8192, 0, nullptr, "
                          f"reinterpret_cast<const uint8_t*>(Wsr.data_ptr()), outr.data_ptr<float>(), part.data_ptr<float>(), "
                          f"rcnt.data_ptr<int>()); return; }}")
    return (SK2_SRC.replace("@@SK2_DISPATCH@@", "\\n  ".join(lines))
            .replace("@@SK2F_DISPATCH@@", "\\n  ".join(flines)))
''')
rep('''    ('unpooler32c2', 'unpooler', 8192, 128, 4, 4, 1, 4, 1, 2),      # 64 x 2 = 128 CTAs, 4 consumer warps (no K split inside)
]
''', '''    ('unpooler32c2', 'unpooler', 8192, 128, 4, 4, 1, 4, 1, 2),      # 64 x 2 = 128 CTAs, 4 consumer warps (no K split inside)
    # the decode layer's fused pooler + router launch (sk2f, FUSE): pooler rows then one router piece per CTA through the
    # same ring; its stream is built like any other config, the kernel is instantiated only with FUSE
    ('pooler', 'pooler', 4096, 32, 4, 1, 8, 1, 2, 1),
]
SK2_FUSED_CONFIG = 'pooler'   # the config sk2f runs (NT 4, WSK 8, UPW 1: 8 consumer warps = the router piece's 8 K-units)
''')
rep('''def _run_sk2(x, layer, m, n, k, state, dg=0, dbg=None, name=None):
    name = name or _sk2_name(layer, m)
    ci, stream, tab, pmax, b4, n_rows = state["sk2"][name]
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    _get_ext().sk2(stream, x, out, tab, pmax, b4, m, n, k, ci, _sk2_tokr(m), dg, state["dummy"] if dbg is None else dbg)
    return out
''', '''def _run_sk2(x, layer, m, n, k, state, dg=0, dbg=None, name=None):
    name = name or _sk2_name(layer, m)
    ci, stream, tab, pmax, b4, n_rows = state["sk2"][name]
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    ext = _get_ext()
    ext.sk2_set_pdl(SK2_PDL)
    ext.sk2(stream, x, out, tab, pmax, b4, m, n, k, ci, _sk2_tokr(m), dg, state["dummy"] if dbg is None else dbg)
    return out


def fused_pooler_router(x, state):
    """Pooler and router of one decode step in ONE launch (m <= 32, sm_90): x [m, 8192] bf16 ->
    (pooler [m, 4096] bf16, router [m, 256] fp32). Same numerics as the separate sk2 kernels: 12-bit lossless pooler
    weights with exact escape corrections, router as BF16 hi + lo with FP32 accumulation (quarter-K partials added in
    a fixed order). Used by the decode layer (pooler and router read the same h back to back); standalone scoring
    never calls it."""
    m = x.shape[0]
    assert m <= 32 and state.get("sm90") and SK2_FUSED_CONFIG in state["sk2"] and "router" in state["sk2"]
    ci, stream, tab, pmax, b4, n_rows = state["sk2"][SK2_FUSED_CONFIG]
    rstream = state["sk2"]["router"][1]
    if "fused_part" not in state:
        state["fused_part"] = torch.empty(32 * 4 * 32 * 8, dtype=torch.float32, device=x.device)
        state["fused_cnt"] = torch.zeros(32, dtype=torch.int32, device=x.device)
    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
    outr = torch.empty((m, 256), dtype=torch.float32, device=x.device)
    ext = _get_ext()
    ext.sk2_set_pdl(SK2_PDL)
    ext.sk2f(stream, rstream, x, out, outr, tab, pmax, b4, state["fused_part"], state["fused_cnt"], m, _sk2_tokr(m))
    return out, outr


SK2_PDL = 0   # decode layer: set to N > 0 for programmatic dependent launches (N weight stages before the wait); 0 here
''')
p.write_text(s)
print("ok")
