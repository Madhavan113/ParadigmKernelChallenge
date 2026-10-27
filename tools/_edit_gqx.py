"""Edit: generalize the gqx kernel (NWG 1..4, CL 1..8, fragments aliased into the drained ring, runtime owned-block
reduce from shared memory, per-config candidate lists). Run from the worktree root: python3 tools/_edit_gqx.py"""
import pathlib
p = pathlib.Path("muse_bf16_proj/submission.py")
s = p.read_text()
a = s.index('GQX_SRC = r"""')
b = s.index('RGEMV_SRC = r"""')
new = r'''GQX_SRC = r"""
// ============================================================================ gqx (m 33..256): BF16 weights, big tiles
// CTA = BM = 64 NWG weight rows x BN tokens over 1/CL of K; a cluster of CL CTAs covers all of K for one row block.
// Ring of TMA'd W [BM x 64] and x [BN x 64] tiles (128B swizzle) fed by a producer warp; warpgroup w runs SS-wgmma
// m64nBN on rows [64w, 64w + 64). Reduction (CL > 1): after its main loop every rank parks its accumulator fragments
// in its own, drained ring ([token block][thread] float4) and arrives on a cluster barrier; rank r owns token blocks
// [r NB / CL, (r+1) NB / CL): it sums the other ranks' copies over DSMEM, stages BF16 [token][BM rows] behind the
// fragments and writes each token's BM rows as contiguous 16-byte chunks; a last cluster barrier keeps every rank's
// fragments alive until all remote reads are done. Bigger BM (NWG 2..4) with deeper K splits (CL up to 8) cuts the x
// bytes every SM has to ingest (x is re-read per row block: at BM = 64 it is twice the weight traffic at m = 128).
template <int BN, int NWG, int NSTAGE, int CL>
__global__ void __launch_bounds__(NWG * 128 + 32, 1)
gqx_kernel(const __grid_constant__ CUtensorMap mapW, const __grid_constant__ CUtensorMap mapX,
           __nv_bfloat16* __restrict__ out, int M, int N, int K) {
#if __CUDA_ARCH__ >= 900
  constexpr int BM = NWG * 64;
  constexpr int A_BYTES = BM * 128;
  constexpr int B_BYTES = BN * 128;
  constexpr int ST = A_BYTES + B_BYTES;
  constexpr int NACC = BN / 2;
  constexpr int NTH = NWG * 128;
  constexpr int NB = BN / 8;                 // 8-token blocks of the tile; rank r owns blocks [r NB / CL, (r+1) NB / CL)
  constexpr int JM = (NB + CL - 1) / CL;     // most blocks a rank owns
  constexpr int FRAG_BYTES = CL > 1 ? NB * NTH * 16 : 0;
  static_assert(CL <= NB, "gqx: every rank owns at least one token block");
  static_assert(FRAG_BYTES + JM * 8 * BM * 2 <= NSTAGE * ST, "gqx: fragments + BF16 staging tile must fit in the ring");
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t bars = sbase + NSTAGE * ST;
  const uint32_t full0 = bars, empty0 = bars + 8 * NSTAGE;
  float4* frag = reinterpret_cast<float4*>(gbase);                              // [NB][NTH] (the ring, once drained)
  __nv_bfloat16* stg = reinterpret_cast<__nv_bfloat16*>(gbase + FRAG_BYTES);   // [owned tokens][BM rows]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  uint32_t crank = 0;
  if constexpr (CL > 1) asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(crank));
  const int rb = blockIdx.x / CL;
  const int U = K >> 6;
  const int u0 = ((int)crank * U) / CL, u1 = (((int)crank + 1) * U) / CL;
  const int nkb = u1 - u0;
  const int xrows = min(M, BN);

  if (tid == 0) {
    for (int i = 0; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NWG); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapW) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapX) : "memory");
    for (int i = 0; i < NSTAGE && i < nkb; ++i) {
      mbar_expect_tx(full0 + 8 * i, A_BYTES + xrows * 128);
      const int kg = (u0 + i) * 64;
      tma_load_2d_ef(sbase + i * ST, &mapW, full0 + 8 * i, kg, rb * BM);
      tma_load_2d_el(sbase + i * ST + A_BYTES, &mapX, full0 + 8 * i, kg, 0);
    }
  }
  __syncthreads();

  if (warp == NWG * 4) {   // producer: refills; then joins the cluster barriers
    if (lane == 0) {
      for (int i = NSTAGE; i < nkb; ++i) {
        const int slot = i % NSTAGE;
        mbar_wait(empty0 + 8 * slot, ((i / NSTAGE) - 1) & 1);   // busy wait: a sleeping producer refills late
        mbar_expect_tx(full0 + 8 * slot, A_BYTES + xrows * 128);
        const int kg = (u0 + i) * 64;
        tma_load_2d_ef(sbase + slot * ST, &mapW, full0 + 8 * slot, kg, rb * BM);
        tma_load_2d_el(sbase + slot * ST + A_BYTES, &mapX, full0 + 8 * slot, kg, 0);
      }
    }
    __syncwarp();
    if constexpr (CL > 1) {   // the consumers' two cluster barriers (fragments written, reads done)
      asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
      asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    }
    return;
  }

  const int wg = warp >> 2, t = tid & 127;
  float acc[NACC];
#pragma unroll
  for (int q = 0; q < NACC; ++q) acc[q] = 0.f;
  for (int i = 0; i < nkb; ++i) {
    const int slot = i % NSTAGE;
    mbar_wait(full0 + 8 * slot, (i / NSTAGE) & 1);
    const uint32_t a_st = sbase + slot * ST + wg * (64 * 128);
    const uint32_t b_st = sbase + slot * ST + A_BYTES;
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j) wgmma_ss<BN>(acc, make_desc_sw128(a_st + j * 32), make_desc_sw128(b_st + j * 32));
    wg_commit();
    wg_wait<1>();
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    if (i > 0 && t == 0) mbar_arrive(empty0 + 8 * ((i - 1) % NSTAGE));
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");

  const int g = lane >> 2, t4 = lane & 3;
  const int row0 = rb * BM + wg * 64 + (warp & 3) * 16 + g;
  if constexpr (CL > 1) {
    // 1) every warpgroup is done reading the ring; fragments go in (own area: the other ranks' main loops need not be
    // over), 2) cluster barrier, 3) the owner of each token block sums the other ranks' copies over DSMEM (its own
    // copy comes back from shared memory: a runtime block index cannot address the accumulator registers), stages
    // BF16 [token][BM rows] behind the fragments and 4) writes each token's BM rows as contiguous 16-byte chunks;
    // 5) exit barrier
    if constexpr (NWG > 1) named_bar(1, NTH);
#pragma unroll
    for (int j = 0; j < NB; ++j) frag[j * NTH + tid] = make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]);
    asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    const int jlo = ((int)crank * NB) / CL, jhi = (((int)crank + 1) * NB) / CL;
    const int rl = wg * 64 + (warp & 3) * 16 + g;   // this thread's first row within the tile
    const uint32_t fl = (uint32_t)__cvta_generic_to_shared(frag + tid);
    uint32_t ra[CL];
#pragma unroll
    for (int r2 = 0; r2 < CL; ++r2)
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra[r2]) : "r"(fl), "r"(r2));
    for (int j = jlo; j < jhi; ++j) {
      float4 v = frag[j * NTH + tid];
      float4 pv[CL];
#pragma unroll
      for (int r2 = 0; r2 < CL; ++r2) {
        if (r2 == (int)crank) continue;
        asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];"
                     : "=f"(pv[r2].x), "=f"(pv[r2].y), "=f"(pv[r2].z), "=f"(pv[r2].w)
                     : "r"(ra[r2] + (uint32_t)(j * NTH * 16)) : "memory");
      }
#pragma unroll
      for (int r2 = 0; r2 < CL; ++r2) {
        if (r2 == (int)crank) continue;
        v.x += pv[r2].x; v.y += pv[r2].y; v.z += pv[r2].z; v.w += pv[r2].w;
      }
      const int tl = 8 * (j - jlo) + 2 * t4;   // token within the owned slice
      stg[tl * BM + rl] = __float2bfloat16_rn(v.x);
      stg[tl * BM + rl + 8] = __float2bfloat16_rn(v.z);
      stg[(tl + 1) * BM + rl] = __float2bfloat16_rn(v.y);
      stg[(tl + 1) * BM + rl + 8] = __float2bfloat16_rn(v.w);
    }
    named_bar(2, NTH);
    {
      constexpr int CH = BM / 8;                 // 16-byte chunks per token row
      const int nt = 8 * (jhi - jlo);
      for (int idx = tid; idx < nt * CH; idx += NTH) {
        const int tt = idx / CH, c = idx - tt * CH;
        const int tok = 8 * jlo + tt;
        if (tok < M)
          *reinterpret_cast<uint4*>(out + (size_t)tok * N + rb * BM + c * 8) =
              *reinterpret_cast<const uint4*>(stg + tt * BM + c * 8);
      }
    }
    // nobody leaves while another rank may still read its fragments
    asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
  } else {
#pragma unroll
    for (int j = 0; j < BN / 8; ++j) {
      const int t0 = 8 * j + 2 * t4;
      if (t0 < M) {
        out[(size_t)t0 * N + row0] = __float2bfloat16_rn(acc[4 * j]);
        out[(size_t)t0 * N + row0 + 8] = __float2bfloat16_rn(acc[4 * j + 2]);
      }
      if (t0 + 1 < M) {
        out[(size_t)(t0 + 1) * N + row0] = __float2bfloat16_rn(acc[4 * j + 1]);
        out[(size_t)(t0 + 1) * N + row0 + 8] = __float2bfloat16_rn(acc[4 * j + 3]);
      }
    }
  }
#endif
}

template <int BN, int NWG, int NSTAGE, int CL>
static void launch_gqx(const at::Tensor& W, const at::Tensor& x, at::Tensor& out, int M, int N, int K) {
  constexpr int BM = NWG * 64;
  constexpr int ST = BM * 128 + BN * 128;
  const size_t smem = 1024 + (size_t)NSTAGE * ST + 16 * NSTAGE + 16;
  auto kern = gqx_kernel<BN, NWG, NSTAGE, CL>;
  TORCH_CHECK(N % BM == 0, "gqx: N % BM");
  cudaLaunchConfig_t lc = {};
  lc.gridDim = dim3(N / BM * CL);
  lc.blockDim = dim3(NWG * 128 + 32);
  lc.dynamicSmemBytes = smem;
  lc.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at_[1];
  at_[0].id = cudaLaunchAttributeClusterDimension;
  at_[0].val.clusterDim.x = CL;
  at_[0].val.clusterDim.y = 1;
  at_[0].val.clusterDim.z = 1;
  lc.attrs = at_;
  lc.numAttrs = 1;
  static bool attr_done = false;
  if (!attr_done) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    if (CL > 8) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    // how many clusters the GPU holds at once (GPC packing): a grid beyond it runs in two waves
    int ncl = -1;
    cudaOccupancyMaxActiveClusters(&ncl, kern, &lc);
    cudaGetLastError();
    fprintf(stderr, "[okc] gqx<bn %d, nwg %d, ns %d, cl %d>: smem %d, max active clusters %d, grid clusters %d\n",
            BN, NWG, NSTAGE, CL, (int)smem, ncl, N / BM);
    attr_done = true;
  }
  CUtensorMap mW = make_map_2d(W.data_ptr(), (uint64_t)W.size(0), (uint64_t)W.size(1), BM);
  CUtensorMap mX = make_map_2d(x.data_ptr(), (uint64_t)M, (uint64_t)K, (uint32_t)(M < BN ? M : BN));
  __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, mW, mX, op, M, N, K));
}

void gqx(at::Tensor W, at::Tensor x, at::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
         int64_t cl, int64_t nstage) {
  @@GQX_DISPATCH@@
  TORCH_CHECK(false, "gqx: no config");
}
"""

# (layer, bn) -> [(nwg, cl, nstage), ...]: BM = 64 nwg rows per CTA, grid = N / BM * cl CTAs (cl-CTA clusters split K);
# every entry is a candidate the autotuner times against cuBLAS (and gq / torchT / CUTLASS where present).
# Bytes each SM ingests = W (BM x K/cl x 2) + x (BN x K/cl x 2): at BM = 64 the x re-reads are 2x the weights (m = 128),
# so wider row blocks with deeper K splits (DSMEM reduce) cut the per-SM traffic; cl <= 8 (portable cluster size).
GQX_CONFIGS = {
    ("qkv_gate", 128): [(1, 3, 8),    # v141: 36 row blocks x 3 K-thirds = 108 CTAs (cuBLAS 16.4 -> 14.0 us at m=128)
                        (2, 6, 6),    # 18 x 6 = 108 CTAs, 0.70 MB per SM
                        (3, 8, 5)],   # 12 x 8 = 96 CTAs, 0.64 MB per SM
    ("qkv_gate", 256): [(1, 3, 5),    # 108 CTAs, 1.6 MB per SM
                        (2, 6, 4),    # 108 CTAs, 0.9 MB per SM
                        (2, 4, 4)],   # 72 CTAs
    ("pooler", 128): [(2, 4, 6),      # 32 x 4 = 128 CTAs, 1 MB per SM (gq: 1.5 MB)
                      (4, 8, 4)],     # 16 x 8 = 128 CTAs, 0.75 MB per SM
    ("pooler", 256): [(2, 4, 4),      # 128 CTAs, 1.5 MB per SM
                      (2, 2, 4)],     # 64 CTAs
    ("unpooler", 128): [(2, 2, 6),    # 64 x 2 = 128 CTAs, 1 MB per SM
                        (4, 4, 4)],   # 32 x 4 = 128 CTAs, 0.75 MB per SM
    ("unpooler", 256): [(2, 2, 4),    # 128 CTAs, 1.5 MB per SM
                        (4, 2, 3)],   # 64 CTAs (acc 128 regs x 4 warpgroups: only if ptxas keeps it spill-free)
    ("o", 128): [(2, 2, 6),           # 128 CTAs, 8 stages of K each
                 (1, 1, 8)],          # 128 CTAs, no cluster
    ("o", 256): [(2, 2, 4),
                 (1, 1, 5)],
}
USE_GQX = True
TORCHT_LAYERS = ("unpooler", "o")   # layers that also get a K-major BF16 copy for the "torchT" (mm(x, W^T)) route


def _gqx_src(sm90):
    if not sm90:
        return ('void gqx(at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, '
                'int64_t) { TORCH_CHECK(false, "gqx needs sm_90"); }')
    lines = []
    for (layer, bn), cfgs in sorted(GQX_CONFIGS.items()):
        for nwg, cl, ns in cfgs:
            lines.append(f"if (bn == {bn} && nwg == {nwg} && cl == {cl} && nstage == {ns}) "
                         f"{{ launch_gqx<{bn}, {nwg}, {ns}, {cl}>(W, x, out, M, N, K); return; }}")
    return GQX_SRC.replace("@@GQX_DISPATCH@@", "\n  ".join(dict.fromkeys(lines)))


def _gqx_bn(m):
    return 128 if m <= 128 else 256


def _run_gqx(x, layer, m, n, k, state, idx=0):
    nwg, cl, ns = GQX_CONFIGS[(layer, _gqx_bn(m))][idx]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    _get_ext().gqx(state["raw"][layer], x, out, m, n, k, _gqx_bn(m), nwg, cl, ns)
    return out



'''
s = s[:a] + new + s[b:]

# ---- glue: candidates carry the config index; keys for the selftest's disable set; run; nearby choice; tune log
old = '''    if (USE_GQX and state["sm90"] and 32 < m <= 256 and layer != "router" and (layer, _gqx_bn(m)) in GQX_CONFIGS
            and _route_key(layer, m, "gqx") not in state["disabled"]):
        c.insert(0, ("gqx",))
'''
new = '''    if USE_GQX and state["sm90"] and 32 < m <= 256 and layer != "router":
        for i in range(len(GQX_CONFIGS.get((layer, _gqx_bn(m)), ()))):
            if _cand_key(layer, m, ("gqx", i)) not in state["disabled"]:
                c.insert(0, ("gqx", i))
'''
assert old in s
s = s.replace(old, new)

old = '''                key = (("skinny_alt", layer, m) if (ch[0] == "skinny" and ch[2][0] == "alt")
                       else _route_key(layer, m, ch[0]))
'''
new = '''                key = _cand_key(layer, m, ch)
'''
assert old in s
s = s.replace(old, new)

old = '''SELFTEST_M = (1, 8, 16, 32, 33, 128, 200, 256)
'''
new = '''SELFTEST_M = (1, 8, 16, 32, 33, 128, 200, 256)


def _cand_key(layer, m, ch):
    """The disable-set key of a candidate tuple (per config where a route has several)."""
    if ch[0] == "skinny" and ch[2][0] == "alt":
        return ("skinny_alt", layer, m)
    if ch[0] == "gqx":
        return (layer, "gqx", _gqx_bn(m), ch[1])
    if ch[0] == "sk2" and len(ch) > 1:
        return (layer, "sk2", ch[1], _sk2_tokr(m))
    return _route_key(layer, m, ch[0])
'''
assert old in s
s = s.replace(old, new)

old = '''    if kind == "gqx":
        return _run_gqx(x, layer, m, n, k, state)
'''
new = '''    if kind == "gqx":
        return _run_gqx(x, layer, m, n, k, state, choice[1] if len(choice) > 1 else 0)
'''
assert old in s
s = s.replace(old, new)

old = '''    if ch[0] == "gq" and (layer, _gq_bn(m)) not in GQ_CONFIGS:
        return None
    return ch
'''
new = '''    if ch[0] == "gq" and (layer, _gq_bn(m)) not in GQ_CONFIGS:
        return None
    if ch[0] == "gqx" and (layer, _gqx_bn(m)) not in GQX_CONFIGS:
        return None
    return ch
'''
assert old in s
s = s.replace(old, new)

old = '''        desc = " ".join(f"{name}{'/' + 'x'.join(map(str, c[1])) if c[0] == 'skinny' else ''}={t}"
                        for (name, t), c in zip(tl, _candidates(layer, m, state)))
        pick = ch[0] + ("/" + "x".join(map(str, ch[1])) if ch[0] == "skinny" else "")
'''
new = '''        desc = " ".join(f"{_cand_name(c)}={t}" for (name, t), c in zip(tl, _candidates(layer, m, state)))
        pick = _cand_name(ch)
'''
assert old in s
s = s.replace(old, new)

old = '''def _nearby_choice(layer, m, state):
'''
new = '''def _cand_name(c):
    if c[0] == "skinny":
        return c[0] + "/" + "x".join(map(str, c[1]))
    if c[0] == "gqx" and len(c) > 1:
        return f"gqx[{c[1]}]"
    if c[0] == "sk2" and len(c) > 1:
        return f"sk2/{c[1]}"
    return c[0]


def _nearby_choice(layer, m, state):
'''
assert old in s
s = s.replace(old, new)
p.write_text(s)
print("ok")
