#!/usr/bin/env python3
"""cg2: a streamed 12-bit GEMV for m = 1 (per-thread cp.async ring, x once per CTA, one reduction per CTA) as an
autotune candidate next to cgemv. Patches muse_bf16_proj/submission.py in place (every anchor must occur once)."""
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


# A. host declaration
rep("void cgemv(torch::Tensor L, torch::Tensor C, torch::Tensor x, torch::Tensor etab, int64_t pmax, torch::Tensor out, int64_t base4, int64_t rows, int64_t nth);\n",
    "void cgemv(torch::Tensor L, torch::Tensor C, torch::Tensor x, torch::Tensor etab, int64_t pmax, torch::Tensor out, int64_t base4, int64_t rows, int64_t nth);\n"
    "void cg2(torch::Tensor L, torch::Tensor C, torch::Tensor x, torch::Tensor etab, int64_t pmax, torch::Tensor out, int64_t base4,\n"
    "         int64_t rpc, int64_t rs, int64_t nstage, int64_t nth, int64_t per, int64_t minb);\n")

# B. extension name / exported functions
rep('name=f"okc_bf16proj_v143_{arch}"', 'name=f"okc_bf16proj_v144_{arch}"')
rep('functions=["skinny_tma_a", "gemm90", "wide", "sk2", "gq", "gqx", "rgemv", "sgemv", "cgemv", "rk", "sk2f",\n                       "sk2_set_pdl", "skx"],',
    'functions=["skinny_tma_a", "gemm90", "wide", "sk2", "gq", "gqx", "rgemv", "sgemv", "cgemv", "rk", "sk2f",\n                       "sk2_set_pdl", "skx", "cg2"],')

# C. the kernel, its configs and dispatch source
CG2 = r'''
CG2_SRC = r"""
// ============================================================================ cg2 (m = 1): streamed 12-bit GEMV
// The cgemv planes (L [N][K] low bytes, C [N][K/2] 4-bit codes), streamed instead of loaded whole: a CTA owns RPC
// rows; its NTH threads split K (PER chunks of 16 k, chunk j of thread t at k = 16 (j NTH + t)); the rows flow
// through a per-thread cp.async ring of NSTAGE stages x RS rows in shared memory. A thread only ever reads the bytes
// it copied itself, so the loop has no block-level synchronization (cp.async.wait_group is per thread) and the ring
// holds NSTAGE x RS rows in flight per CTA at any time; x is read once per CTA and kept in registers; the RPC
// partial sums are reduced once at the end. The escape warp corrects the escapes of all RPC rows from the per-CTA
// table (built for RPC rows per CTA, as cgemv's).
__device__ __forceinline__ void cg2_cp16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cg2_cp8(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cg2_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cg2_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }

template <int RPC, int RS, int NSTAGE, int NTH, int PER, int MINB>
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
  if (warp == NWARP) {   // escape warp (as cgemv_kernel's, over the CTA's RPC rows)
    const int2* tb = etab + (size_t)blockIdx.x * pmax;
    int2 mine = make_int2(0, 0);
    if (lane < pmax) mine = __ldg(&tb[lane]);
    const int cnt = __shfl_sync(0xffffffffu, mine.x, 0);
    float ec[RPC];
#pragma unroll
    for (int r = 0; r < RPC; ++r) ec[r] = 0.f;
    for (int i0 = 0; i0 <= cnt; i0 += 32) {
      const int i = i0 + lane;
      int2 en = mine;
      if (i0 > 0) en = (i <= cnt) ? __ldg(&tb[i]) : make_int2(0, 0);
      if (i >= 1 && i <= cnt) {
        const float c = __bfloat162float(x[en.x & 0xFFFF]) * esc_delta((uint32_t)en.y, base4 & 0xFFu);
        const int rl = en.x >> 16;
#pragma unroll
        for (int r = 0; r < RPC; ++r) ec[r] += (rl == r) ? c : 0.f;
      }
    }
#pragma unroll
    for (int r = 0; r < RPC; ++r) {
      float v = ec[r];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0) red[r][NWARP] = v;
    }
    __syncthreads();
    return;
  }
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(cg2_smem);
  // x once: this thread's PER chunks of 16 k as floats
  float xf[PER][16];
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const __nv_bfloat16* xp = x + 16 * (j * NTH + tid);
    const uint4 x0 = __ldg(reinterpret_cast<const uint4*>(xp)), x1 = __ldg(reinterpret_cast<const uint4*>(xp + 8));
    const uint32_t xw[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
    for (int e = 0; e < 8; ++e) { xf[j][2 * e] = __uint_as_float(xw[e] << 16); xf[j][2 * e + 1] = __uint_as_float(xw[e] & 0xFFFF0000u); }
  }
  // ring: stage s (rows r0 + s RS .. + RS) in slot s % NSTAGE; inside a stage row, this thread's bytes are at
  // [L: 16 (j NTH + t)] and [LB + 8 (j NTH + t)] (the global layout, so every copy instruction is warp-contiguous)
  const uint8_t* lp = L + (size_t)r0 * K + 16 * tid;
  const uint8_t* cp = C + (size_t)r0 * CB + 8 * tid;
  const uint32_t so = 16 * tid, co = LB + 8 * tid;
  auto issue = [&](int st) {
    const uint32_t s0 = sb + (st % NSTAGE) * SST;
#pragma unroll
    for (int r = 0; r < RS; ++r) {
      const int row = st * RS + r;
#pragma unroll
      for (int j = 0; j < PER; ++j) {
        cg2_cp16(s0 + r * SROW + so + j * 16 * NTH, lp + (size_t)row * K + j * 16 * NTH);
        cg2_cp8(s0 + r * SROW + co + j * 8 * NTH, cp + (size_t)row * CB + j * 8 * NTH);
      }
    }
    cg2_commit();
  };
#pragma unroll
  for (int st = 0; st < NSTAGE; ++st) {
    if (st < NST) issue(st); else cg2_commit();   // (empty groups keep the wait_group count uniform)
  }
  float acc[RPC];
#pragma unroll
  for (int r = 0; r < RPC; ++r) acc[r] = 0.f;
#pragma unroll
  for (int st = 0; st < NST; ++st) {
    cg2_wait<NSTAGE - 1>();   // stage st has landed (this thread's copies)
    const uint32_t s0 = sb + (st % NSTAGE) * SST;
#pragma unroll
    for (int r = 0; r < RS; ++r) {
      float v = acc[st * RS + r];
#pragma unroll
      for (int j = 0; j < PER; ++j) {
        uint32_t l0, l1, l2, l3, c0, c1;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(l0), "=r"(l1), "=r"(l2), "=r"(l3) : "r"(s0 + r * SROW + so + j * 16 * NTH));
        asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(c0), "=r"(c1) : "r"(s0 + r * SROW + co + j * 8 * NTH));
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
    }
    // the stage's bytes are consumed (multiplied: v depends on every shared-memory load of the stage), so its slot
    // can take stage st + NSTAGE now
    asm volatile("" :: "f"(acc[st * RS + RS - 1]));
    if (st + NSTAGE < NST) issue(st + NSTAGE); else cg2_commit();
  }
  cg2_wait<0>();
#pragma unroll
  for (int r = 0; r < RPC; ++r) {
    float v = acc[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) red[r][warp] = v;
  }
  __syncthreads();
  if (tid < RPC) {
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
  TORCH_CHECK(false, "cg2: no instantiation for K=", K, " rpc=", rpc, " rs=", rs, " nstage=", nstage, " nth=", nth);
}
"""

# m = 1 streamed-GEMV candidates: layer -> [(rows per CTA, rows per stage, ring stages, threads, chunks per thread,
# min CTAs per SM)]; K = 16 threads chunks; grid = N / rows per CTA; ring = stages x rows per stage x 1.5 K bytes
CG2_CONFIGS = {
    "pooler": [(16, 2, 4, 512, 1, 2), (16, 1, 8, 512, 1, 2), (32, 2, 8, 512, 1, 1), (8, 2, 4, 512, 1, 2)],
    "qkv_gate": [(16, 2, 4, 512, 1, 2), (9, 1, 8, 512, 1, 2), (18, 2, 8, 512, 1, 1)],
    "unpooler": [(4, 1, 4, 256, 1, 3), (8, 2, 4, 256, 1, 3)],   # the 1024-row slice, K = 4096
    "o": [(32, 4, 8, 64, 1, 8), (16, 2, 8, 64, 1, 8)],          # K = 1024
}


def _cg2_src():
    lines = []
    for layer, cfgs in CG2_CONFIGS.items():
        k = CG_K[layer]
        for rpc, rs, ns, nth, per, minb in cfgs:
            assert k == 16 * nth * per and w_rows[layer] % rpc == 0, (layer, rpc, nth, per)
            lines.append(f"if (K == {k} && rpc == {rpc} && rs == {rs} && nstage == {ns} && nth == {nth} && per == {per} && minb == {minb}) "
                         f"{{ launch_cg2<{rpc}, {rs}, {ns}, {nth}, {per}, {minb}>(a0, a1, xp, tp, pm, op, b4, N, st); return; }}")
    return CG2_SRC.replace("@@CG2_DISPATCH@@", "\n  ".join(dict.fromkeys(lines)))


def _gq_src(sm90):'''
rep("\n\n\ndef _gq_src(sm90):", "\n\n" + CG2)

# D. the rgemv translation unit carries cg2
rep('    return RGEMV_SRC.replace("@@CG_DISPATCH@@", "\\n  ".join(dict.fromkeys(lines)))',
    '    return RGEMV_SRC.replace("@@CG_DISPATCH@@", "\\n  ".join(dict.fromkeys(lines))) + _cg2_src()')

# E. prepare: planes once per layer, one escape table per rows-per-CTA
rep("""                for rows in sorted({r for r, _ in _cg_configs(name)}):   # one plane set per rows-per-CTA (its table)
                    cg = cg_compress(wc.contiguous(), rows)
                    if cg is not None:
                        state.setdefault("cg", {})[(name, rows)] = cg
""", """                rows_all = {r for r, _ in _cg_configs(name)} | {c[0] for c in CG2_CONFIGS.get(name, ())}
                planes = {}   # the planes are shared across rows-per-CTA values; only the escape table differs
                for rows in sorted(rows_all):
                    cg = cg_compress(wc.contiguous(), rows, planes)
                    if cg is not None:
                        state.setdefault("cg", {})[(name, rows)] = cg
""")

# F. cg_compress with a planes cache
old_cc = s[s.index("def cg_compress(w, rows_per_cta):"):]
old_cc = old_cc[:old_cc.index('    return {"L": L, "C": C, "tab": tab.contiguous(), "pmax": pmax, "base4": base * 0x01010101}\n')
                + len('    return {"L": L, "C": C, "tab": tab.contiguous(), "pmax": pmax, "base4": base * 0x01010101}\n')]
new_cc = '''def cg_compress(w, rows_per_cta, planes=None):
    """Row-major 12-bit planes for cgemv / cg2 (lossless; escapes corrected exactly from a per-CTA table). `planes`
    (a dict) caches the planes and the escape list across rows-per-CTA values of the same matrix."""
    n, k = w.shape
    if planes is None:
        planes = {}
    if "bad" in planes:
        return None
    if "L" not in planes:
        bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
        hi, low = bits >> 8, bits & 0xFF
        e71, sign = hi & 0x7F, hi >> 7
        hist = torch.bincount(e71.flatten(), minlength=128)
        win = torch.stack([hist[b:b + 8].sum() for b in range(121)])
        base = int(win.argmax())
        o = (e71 - base).clamp(0, 7)
        code = ((sign << 3) | o).to(torch.uint8)
        esc = (e71 < base) | (e71 > base + 7)
        if int(esc.sum()) > n * k // 200:
            planes["bad"] = True
            return None
        g = code.view(n, k // 8, 8)
        C = (g[:, :, 0:4] | (g[:, :, 4:8] << 4)).reshape(n, k // 2).contiguous()
        L = low.to(torch.uint8).contiguous()
        rows, cols = esc.nonzero(as_tuple=True)          # row-major order
        dec = ((((sign[rows, cols] << 7) | (base + o[rows, cols])) << 8) | low[rows, cols]).to(torch.int16).view(torch.bfloat16).float()
        wtrue = w[rows, cols].float()
        delta = (wtrue - dec).contiguous()
        if not bool(((dec + delta) == wtrue).all()):
            planes["bad"] = True
            return None
        planes.update({"L": L, "C": C, "base4": base * 0x01010101, "rows": rows, "cols": cols,
                       "wbits": bits[rows, cols].to(torch.int32)})   # the original BF16 weights of the escapes
    rows, cols = planes["rows"], planes["cols"]
    ncta = n // rows_per_cta
    cta = rows // rows_per_cta
    counts = torch.bincount(cta, minlength=ncta)
    pmax = int(counts.max()) + 1 if rows.numel() else 1
    start = torch.cumsum(counts, 0) - counts
    pos = torch.arange(rows.numel(), device=w.device) - start[cta] + 1
    tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=w.device)
    tab[:, 0, 0] = counts.to(torch.int32)
    tab[cta, pos, 0] = (((rows - cta * rows_per_cta) << 16) | cols).to(torch.int32)
    tab[cta, pos, 1] = planes["wbits"]   # the kernel forms the correction from the original weight
    return {"L": planes["L"], "C": planes["C"], "tab": tab.contiguous(), "pmax": pmax, "base4": planes["base4"]}
'''
rep(old_cc, new_cc)

# G. candidate key
rep('''    if ch[0] == "cgemv" and len(ch) > 1:
        return (layer, "cgemv", ch[1])
''', '''    if ch[0] == "cgemv" and len(ch) > 1:
        return (layer, "cgemv", ch[1])
    if ch[0] == "cg2":
        return (layer, "cg2", ch[1])
''')

# H. candidates
rep('''            if (layer, cfg[0]) in state.get("cg", {}) and _cand_key(layer, m, ("cgemv", cfg)) not in state["disabled"]:
                c.insert(0, ("cgemv", cfg))
''', '''            if (layer, cfg[0]) in state.get("cg", {}) and _cand_key(layer, m, ("cgemv", cfg)) not in state["disabled"]:
                c.insert(0, ("cgemv", cfg))
    if m == 1 and layer in CG2_CONFIGS:
        for cfg in CG2_CONFIGS[layer]:
            if (layer, cfg[0]) in state.get("cg", {}) and _cand_key(layer, m, ("cg2", cfg)) not in state["disabled"]:
                c.insert(0, ("cg2", cfg))
''')

# I. run
rep('''    if kind == "sgemv":   # unpooler slice, m = 1: BF16 weights straight from memory (4 rows per CTA)
''', '''    if kind == "cg2":   # m = 1: the cgemv planes streamed through a per-thread cp.async ring (see CG2_SRC)
        rpc, rs, ns, nth, per, minb = choice[1]
        cg = state["cg"][(layer, rpc)]
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        _get_ext().cg2(cg["L"], cg["C"], x, cg["tab"], cg["pmax"], out, cg["base4"], rpc, rs, ns, nth, per, minb)
        return out
    if kind == "sgemv":   # unpooler slice, m = 1: BF16 weights straight from memory (4 rows per CTA)
''')

# J. names
rep('''    if c[0] == "cgemv" and len(c) > 1:
        return f"cgemv/{c[1][0]}x{c[1][1]}"
''', '''    if c[0] == "cgemv" and len(c) > 1:
        return f"cgemv/{c[1][0]}x{c[1][1]}"
    if c[0] == "cg2":
        return f"cg2/{c[1][0]}r{c[1][1]}s{c[1][2]}x{c[1][3]}"
''')

p.write_text(s)
print("ok", p)
