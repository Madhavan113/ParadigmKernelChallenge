#!/usr/bin/env python3
"""cgemv/cg2 diagnostics: a per-CTA globaltimer trace (dbg buffer, nullptr in production), a loads-only cg2 variant
(DG template parameter, instantiated for the configs in CG2_DIAG), and cgemv register-prefetch alternatives
(CG_ALT: ROWS 3/4, PER 2) for the autotuner."""
import pathlib, sys

p = pathlib.Path(__file__).resolve().parent.parent / "muse_bf16_proj" / "submission.py"
s = p.read_text()


def rep(old, new, count=1):
    global s
    n = s.count(old)
    if n != count:
        print(f"anchor x{n} (want {count}): {old[:90]!r}", file=sys.stderr)
        sys.exit(1)
    s = s.replace(old, new)


# ---- cgemv: dbg trace
rep("""template <int ROWS, int NTH, int K>
__global__ void __launch_bounds__(NTH + 32) cgemv_kernel(const uint8_t* __restrict__ L, const uint8_t* __restrict__ C,
    const __nv_bfloat16* __restrict__ x, const int2* __restrict__ etab, int pmax,
    __nv_bfloat16* __restrict__ out, uint32_t base4) {
  constexpr int PER = K / (16 * NTH);
  constexpr int NWARP = NTH / 32;
  static_assert(PER >= 1, "cgemv: K >= 16 NTH");
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
""", """__device__ __forceinline__ unsigned long long cg_gtime() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
template <int ROWS, int NTH, int K>
__global__ void __launch_bounds__(NTH + 32) cgemv_kernel(const uint8_t* __restrict__ L, const uint8_t* __restrict__ C,
    const __nv_bfloat16* __restrict__ x, const int2* __restrict__ etab, int pmax,
    __nv_bfloat16* __restrict__ out, uint32_t base4, unsigned long long* __restrict__ dbg) {
  constexpr int PER = K / (16 * NTH);
  constexpr int NWARP = NTH / 32;
  static_assert(PER >= 1, "cgemv: K >= 16 NTH");
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const unsigned long long t_start = dbg ? cg_gtime() : 0ull;   // diagnostics: per-CTA start / end times
""")
rep("""  if (tid < ROWS) {
    float v = 0.f;
#pragma unroll
    for (int q = 0; q <= NWARP; ++q) v += red[tid][q];
    out[r0 + tid] = __float2bfloat16_rn(v);
  }
}

void cgemv(at::Tensor L, at::Tensor C, at::Tensor x, at::Tensor etab, int64_t pmax, at::Tensor out, int64_t base4,
           int64_t rows, int64_t nth) {
  const int N = out.size(1), K = L.size(1);
  TORCH_CHECK(x.size(0) == 1 && N % rows == 0 && N <= L.size(0), "cgemv: m = 1");
  auto st = at::cuda::getCurrentCUDAStream();
  auto a0 = L.data_ptr<uint8_t>(); auto a1 = C.data_ptr<uint8_t>();
  auto xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto tp = reinterpret_cast<const int2*>(etab.data_ptr<int>());
  auto op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  const uint32_t b4 = (uint32_t)base4;
  const int pm = (int)pmax;
  @@CG_DISPATCH@@
""", """  if (tid < ROWS) {
    float v = 0.f;
#pragma unroll
    for (int q = 0; q <= NWARP; ++q) v += red[tid][q];
    out[r0 + tid] = __float2bfloat16_rn(v);
  }
  if (dbg && tid == 0) { dbg[2 * blockIdx.x] = t_start; dbg[2 * blockIdx.x + 1] = cg_gtime(); }
}

void cgemv(at::Tensor L, at::Tensor C, at::Tensor x, at::Tensor etab, int64_t pmax, at::Tensor out, int64_t base4,
           int64_t rows, int64_t nth, at::Tensor dbg) {
  const int N = out.size(1), K = L.size(1);
  TORCH_CHECK(x.size(0) == 1 && N % rows == 0 && N <= L.size(0), "cgemv: m = 1");
  auto st = at::cuda::getCurrentCUDAStream();
  auto a0 = L.data_ptr<uint8_t>(); auto a1 = C.data_ptr<uint8_t>();
  auto xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto tp = reinterpret_cast<const int2*>(etab.data_ptr<int>());
  auto op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  const uint32_t b4 = (uint32_t)base4;
  const int pm = (int)pmax;
  unsigned long long* dbp = dbg.numel() >= 2 * (N / rows) ? reinterpret_cast<unsigned long long*>(dbg.data_ptr()) : nullptr;
  @@CG_DISPATCH@@
""")
rep("""(a0, a1, xp, tp, pm, op, b4); C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }}")
    return RGEMV_SRC.replace("@@CG_DISPATCH@@", "\\n  ".join(dict.fromkeys(lines))) + _cg2_src()""",
    """(a0, a1, xp, tp, pm, op, b4, dbp); C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }}")
    return RGEMV_SRC.replace("@@CG_DISPATCH@@", "\\n  ".join(dict.fromkeys(lines))) + _cg2_src()""")
rep("void cgemv(torch::Tensor L, torch::Tensor C, torch::Tensor x, torch::Tensor etab, int64_t pmax, torch::Tensor out, int64_t base4, int64_t rows, int64_t nth);\n",
    "void cgemv(torch::Tensor L, torch::Tensor C, torch::Tensor x, torch::Tensor etab, int64_t pmax, torch::Tensor out, int64_t base4, int64_t rows, int64_t nth, torch::Tensor dbg);\n")
rep("""        _get_ext().cgemv(cg["L"], cg["C"], x, cg["tab"], cg["pmax"], out, cg["base4"], rows, nth)
""", """        _get_ext().cgemv(cg["L"], cg["C"], x, cg["tab"], cg["pmax"], out, cg["base4"], rows, nth, state["nodbg"])
""")

# ---- cg2: DG (loads-only) template parameter + dbg trace
rep("""template <int RPC, int RS, int NSTAGE, int NTH, int PER, int MINB>
__global__ void __launch_bounds__(NTH + 32, MINB) cg2_kernel(const uint8_t* __restrict__ L, const uint8_t* __restrict__ C,
    const __nv_bfloat16* __restrict__ x, const int2* __restrict__ etab, int pmax,
    __nv_bfloat16* __restrict__ out, uint32_t base4) {
  constexpr int K = 16 * NTH * PER;
  constexpr int NWARP = NTH / 32;
  constexpr int NST = RPC / RS;
  static_assert(RPC % RS == 0 && RPC <= NTH, "cg2: RS divides RPC <= NTH");
  constexpr int LB = K, CB = K / 2, SROW = LB + CB, SST = RS * SROW;
  extern __shared__ __align__(16) unsigned char cg2_smem[];
  __shared__ float red[RPC][NWARP + 1];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int r0 = blockIdx.x * RPC;
""", """template <int RPC, int RS, int NSTAGE, int NTH, int PER, int MINB, int DG>
__global__ void __launch_bounds__(NTH + 32, MINB) cg2_kernel(const uint8_t* __restrict__ L, const uint8_t* __restrict__ C,
    const __nv_bfloat16* __restrict__ x, const int2* __restrict__ etab, int pmax,
    __nv_bfloat16* __restrict__ out, uint32_t base4, unsigned long long* __restrict__ dbg) {
  constexpr int K = 16 * NTH * PER;
  constexpr int NWARP = NTH / 32;
  constexpr int NST = RPC / RS;
  static_assert(RPC % RS == 0 && RPC <= NTH, "cg2: RS divides RPC <= NTH");
  constexpr int LB = K, CB = K / 2, SROW = LB + CB, SST = RS * SROW;
  extern __shared__ __align__(16) unsigned char cg2_smem[];
  __shared__ float red[RPC][NWARP + 1];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int r0 = blockIdx.x * RPC;
  const unsigned long long t_start = dbg ? cg_gtime() : 0ull;   // diagnostics: per-CTA start / end times
""")
rep("""        const uint32_t lws[4] = {l0, l1, l2, l3};
        const uint32_t cws[2] = {c0, c1};
#pragma unroll
        for (int g = 0; g < 2; ++g) {           // 8-element groups: low bytes lws[2g], lws[2g+1]; codes cws[g]
          const uint32_t ha = cg_hib(cws[g] & 0x0F0F0F0Fu, base4);
          const uint32_t hb = cg_hib((cws[g] >> 4) & 0x0F0F0F0Fu, base4);
          const uint32_t la = lws[2 * g], lb = lws[2 * g + 1];
          const uint32_t p01 = __byte_perm(la, ha, 0x5140), p23 = __byte_perm(la, ha, 0x7362);
          const uint32_t p45 = __byte_perm(lb, hb, 0x5140), p67 = __byte_perm(lb, hb, 0x7362);
          const float* xg = xf[j] + 8 * g;
          v = fmaf(__uint_as_float(p01 << 16), xg[0], v); v = fmaf(__uint_as_float(p01 & 0xFFFF0000u), xg[1], v);
          v = fmaf(__uint_as_float(p23 << 16), xg[2], v); v = fmaf(__uint_as_float(p23 & 0xFFFF0000u), xg[3], v);
          v = fmaf(__uint_as_float(p45 << 16), xg[4], v); v = fmaf(__uint_as_float(p45 & 0xFFFF0000u), xg[5], v);
          v = fmaf(__uint_as_float(p67 << 16), xg[6], v); v = fmaf(__uint_as_float(p67 & 0xFFFF0000u), xg[7], v);
        }
      }
      acc[st * RS + r] = v;
""", """        if constexpr (DG == 1) {   // loads-only diagnostic: the bytes are consumed, nothing is multiplied
          v += __uint_as_float((l0 ^ l1 ^ l2 ^ l3 ^ c0 ^ c1) & 0x007fffffu);
          continue;
        }
        const uint32_t lws[4] = {l0, l1, l2, l3};
        const uint32_t cws[2] = {c0, c1};
#pragma unroll
        for (int g = 0; g < 2; ++g) {           // 8-element groups: low bytes lws[2g], lws[2g+1]; codes cws[g]
          const uint32_t ha = cg_hib(cws[g] & 0x0F0F0F0Fu, base4);
          const uint32_t hb = cg_hib((cws[g] >> 4) & 0x0F0F0F0Fu, base4);
          const uint32_t la = lws[2 * g], lb = lws[2 * g + 1];
          const uint32_t p01 = __byte_perm(la, ha, 0x5140), p23 = __byte_perm(la, ha, 0x7362);
          const uint32_t p45 = __byte_perm(lb, hb, 0x5140), p67 = __byte_perm(lb, hb, 0x7362);
          const float* xg = xf[j] + 8 * g;
          v = fmaf(__uint_as_float(p01 << 16), xg[0], v); v = fmaf(__uint_as_float(p01 & 0xFFFF0000u), xg[1], v);
          v = fmaf(__uint_as_float(p23 << 16), xg[2], v); v = fmaf(__uint_as_float(p23 & 0xFFFF0000u), xg[3], v);
          v = fmaf(__uint_as_float(p45 << 16), xg[4], v); v = fmaf(__uint_as_float(p45 & 0xFFFF0000u), xg[5], v);
          v = fmaf(__uint_as_float(p67 << 16), xg[6], v); v = fmaf(__uint_as_float(p67 & 0xFFFF0000u), xg[7], v);
        }
      }
      acc[st * RS + r] = v;
""")
rep("""  if (tid < RPC) {
    float v = 0.f;
#pragma unroll
    for (int q = 0; q <= NWARP; ++q) v += red[tid][q];
    out[r0 + tid] = __float2bfloat16_rn(v);
  }
}

template <int RPC, int RS, int NSTAGE, int NTH, int PER, int MINB>
static void launch_cg2(const uint8_t* a0, const uint8_t* a1, const __nv_bfloat16* xp, const int2* tp, int pm,
                       __nv_bfloat16* op, uint32_t b4, int N, cudaStream_t st) {
  constexpr int K = 16 * NTH * PER;
  constexpr int smem = NSTAGE * RS * (K + K / 2);
  auto kern = cg2_kernel<RPC, RS, NSTAGE, NTH, PER, MINB>;
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    attr_done = true;
  }
  kern<<<N / RPC, NTH + 32, smem, st>>>(a0, a1, xp, tp, pm, op, b4);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void cg2(at::Tensor L, at::Tensor C, at::Tensor x, at::Tensor etab, int64_t pmax, at::Tensor out, int64_t base4,
         int64_t rpc, int64_t rs, int64_t nstage, int64_t nth, int64_t per, int64_t minb) {
  const int N = out.size(1), K = L.size(1);
  TORCH_CHECK(x.size(0) == 1 && N % rpc == 0 && N <= L.size(0), "cg2: m = 1");
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  auto a0 = L.data_ptr<uint8_t>(); auto a1 = C.data_ptr<uint8_t>();
  auto xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto tp = reinterpret_cast<const int2*>(etab.data_ptr<int>());
  auto op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  const uint32_t b4 = (uint32_t)base4;
  const int pm = (int)pmax;
  @@CG2_DISPATCH@@
""", """  if (tid < RPC) {
    float v = 0.f;
#pragma unroll
    for (int q = 0; q <= NWARP; ++q) v += red[tid][q];
    out[r0 + tid] = __float2bfloat16_rn(v);
  }
  if (dbg && tid == 0) { dbg[2 * blockIdx.x] = t_start; dbg[2 * blockIdx.x + 1] = cg_gtime(); }
}

template <int RPC, int RS, int NSTAGE, int NTH, int PER, int MINB, int DG>
static void launch_cg2(const uint8_t* a0, const uint8_t* a1, const __nv_bfloat16* xp, const int2* tp, int pm,
                       __nv_bfloat16* op, uint32_t b4, int N, cudaStream_t st, unsigned long long* dbp) {
  constexpr int K = 16 * NTH * PER;
  constexpr int smem = NSTAGE * RS * (K + K / 2);
  auto kern = cg2_kernel<RPC, RS, NSTAGE, NTH, PER, MINB, DG>;
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    attr_done = true;
  }
  kern<<<N / RPC, NTH + 32, smem, st>>>(a0, a1, xp, tp, pm, op, b4, dbp);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void cg2(at::Tensor L, at::Tensor C, at::Tensor x, at::Tensor etab, int64_t pmax, at::Tensor out, int64_t base4,
         int64_t rpc, int64_t rs, int64_t nstage, int64_t nth, int64_t per, int64_t minb, int64_t dg, at::Tensor dbg) {
  const int N = out.size(1), K = L.size(1);
  TORCH_CHECK(x.size(0) == 1 && N % rpc == 0 && N <= L.size(0), "cg2: m = 1");
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  auto a0 = L.data_ptr<uint8_t>(); auto a1 = C.data_ptr<uint8_t>();
  auto xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto tp = reinterpret_cast<const int2*>(etab.data_ptr<int>());
  auto op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  const uint32_t b4 = (uint32_t)base4;
  const int pm = (int)pmax;
  unsigned long long* dbp = dbg.numel() >= 2 * (N / rpc) ? reinterpret_cast<unsigned long long*>(dbg.data_ptr()) : nullptr;
  @@CG2_DISPATCH@@
""")
rep('''def _cg2_src():
    lines = []
    for layer, cfgs in CG2_CONFIGS.items():
        k = CG_K[layer]
        for rpc, rs, ns, nth, per, minb in cfgs:
            assert k == 16 * nth * per and w_rows[layer] % rpc == 0, (layer, rpc, nth, per)
            lines.append(f"if (K == {k} && rpc == {rpc} && rs == {rs} && nstage == {ns} && nth == {nth} && per == {per} && minb == {minb}) "
                         f"{{ launch_cg2<{rpc}, {rs}, {ns}, {nth}, {per}, {minb}>(a0, a1, xp, tp, pm, op, b4, N, st); return; }}")
    return CG2_SRC.replace("@@CG2_DISPATCH@@", "\\n  ".join(dict.fromkeys(lines)))
''', '''CG2_DIAG = [("pooler", (16, 2, 4, 512, 1, 2)), ("qkv_gate", (9, 1, 8, 512, 1, 2))]   # loads-only (dg 1) instantiations


def _cg2_src():
    lines = []
    for layer, cfgs in CG2_CONFIGS.items():
        k = CG_K[layer]
        for rpc, rs, ns, nth, per, minb in cfgs:
            assert k == 16 * nth * per and w_rows[layer] % rpc == 0, (layer, rpc, nth, per)
            for dg in (0, 1) if (layer, (rpc, rs, ns, nth, per, minb)) in CG2_DIAG else (0,):
                lines.append(f"if (K == {k} && rpc == {rpc} && rs == {rs} && nstage == {ns} && nth == {nth} && per == {per} && minb == {minb} && dg == {dg}) "
                             f"{{ launch_cg2<{rpc}, {rs}, {ns}, {nth}, {per}, {minb}, {dg}>(a0, a1, xp, tp, pm, op, b4, N, st, dbp); return; }}")
    return CG2_SRC.replace("@@CG2_DISPATCH@@", "\\n  ".join(dict.fromkeys(lines)))
''')
rep("""void cg2(torch::Tensor L, torch::Tensor C, torch::Tensor x, torch::Tensor etab, int64_t pmax, torch::Tensor out, int64_t base4,
         int64_t rpc, int64_t rs, int64_t nstage, int64_t nth, int64_t per, int64_t minb);
""", """void cg2(torch::Tensor L, torch::Tensor C, torch::Tensor x, torch::Tensor etab, int64_t pmax, torch::Tensor out, int64_t base4,
         int64_t rpc, int64_t rs, int64_t nstage, int64_t nth, int64_t per, int64_t minb, int64_t dg, torch::Tensor dbg);
""")
rep("""        _get_ext().cg2(cg["L"], cg["C"], x, cg["tab"], cg["pmax"], out, cg["base4"], rpc, rs, ns, nth, per, minb)
""", """        _get_ext().cg2(cg["L"], cg["C"], x, cg["tab"], cg["pmax"], out, cg["base4"], rpc, rs, ns, nth, per, minb, 0, state["nodbg"])
""")
# the empty debug tensor (no trace) lives in state
rep("""    state["dummy"] = torch.zeros(2, dtype=torch.int32, device=dev)
""", """    state["dummy"] = torch.zeros(2, dtype=torch.int32, device=dev)
    state["nodbg"] = torch.empty(0, dtype=torch.int64, device=dev)   # cgemv / cg2: no per-CTA trace
""")

# ---- cgemv register-prefetch alternatives (the autotuner times them against CG_CONFIGS)
rep("""CG_ALT = {}
""", """CG_ALT = {   # deeper register prefetch per thread (uint4 of L in flight = rows x chunks): 4 / 3 rows, 2 chunks (PER 2)
    "pooler": [(4, 512), (2, 256)],
    "qkv_gate": [(4, 256), (3, 256), (2, 512), (4, 512)],
    "o": [(4, 64), (2, 32)],
    "unpooler": [(4, 128), (2, 64)],
}
""")

p.write_text(s)
print("ok", p)
