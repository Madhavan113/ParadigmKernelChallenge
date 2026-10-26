"""Proj edit: w12 = dense 12-bit W x BF16 x^T GEMM ported from the MoE weights-as-A kernel (weights M side, tokens N
side, cp.async ring, ldmatrix token tile, PRMT-LUT decode of the 12-bit codes into bf16x2 A fragments, fp32 partial
slices + a reduce kernel that applies the escapes), with its fragment-ordered repack, dispatch table, bench rows."""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
p = root / "muse_bf16_proj/cuda_proj.py"
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:100])
    s = s.replace(old, new)


kernel = r"""
// ---- w12: dense 12-bit W x BF16 x^T GEMM, weights as the mma M side (one m16 tile of W per warp, 4 per CTA),
// tokens as N (NT n8 blocks, m <= 8*NT), K streamed in 128-wide chunks through an NSTAGE cp.async ring shared by
// the CTA. Weight layout W12 (see _repack_w12): per (m16 tile, chunk) 3 KB = low bytes [4 quads][32 lanes][16 B]
// (quad q: k16 steps 2q, 2q+1; per step 8 bytes = L0 (rows g, g+8 at k 2t, 2t+1) then L1 (k 2t+8, 2t+9)) and
// nibbles [2 halves][32 lanes][16 B] (half h: steps 4h..4h+3; per step one word, nibble i = element i of the
// L0 then L1 order), grouped by 4 tiles: [N/64][K/128][4][3 KB]. Decode per step: decode4 (PRMT LUT of the
// high byte, sign OR'd, L/H bytes interleaved) twice -> a0..a3 bf16x2, mma bf16 -> fp32. The CTA writes its
// fp32 partial slice part[z][i][n]; w12_reduce sums the slices, adds the escapes and writes bf16.
__device__ __forceinline__ void ldsm_x4(uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(addr));
}
constexpr int W12_KC = 128, W12_XSTRIDE = W12_KC + 8;   // token tile row stride (halves)
constexpr int W12_TILE_BYTES = 3072;                   // per (m16 tile, chunk)
__host__ __device__ constexpr int w12_stage_bytes(int nt) { return 4 * W12_TILE_BYTES + 8 * nt * W12_XSTRIDE * 2; }

template <int NT, int NSTAGE, bool SPLITK>
__global__ void __launch_bounds__(128)
w12_kernel(const __nv_bfloat16* __restrict__ X, int ldx, int m, const uint8_t* __restrict__ W, int eb,
           float* __restrict__ part, int N, int K)
{
    constexpr int STAGE_BYTES = w12_stage_bytes(NT);
    constexpr int W_STAGE = 4 * W12_TILE_BYTES;
    constexpr int CH = NT == 1 ? 2 : 1;                                   // accumulator chains (mma latency)
    extern __shared__ __align__(128) uint8_t w12_smem[];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KSPLIT = SPLITK ? (int)gridDim.y : 1;
    const int NC_ALL = K / W12_KC;
    const int NC = NC_ALL / KSPLIT;
    const int c0 = SPLITK ? (int)blockIdx.y * NC : 0;
    const int ntb = (m + 7) >> 3;
    const uint32_t hb = (uint32_t)(eb >> 1);
    const uint32_t lut0 = hb | (hb + 1) << 8 | (hb + 2) << 16 | (hb + 3) << 24;
    const uint32_t lut1 = (hb + 4) | (hb + 5) << 8 | (hb + 6) << 16 | (hb + 7) << 24;

    // cp.async segments per chunk: 768 weight segments (6 per thread) + NT x 128 token segments (NT per thread)
    const uint8_t* wsrc = W + ((size_t)blockIdx.x * NC_ALL + c0) * W_STAGE + tid * 16;      // + i * 2048 + c * W_STAGE
    const __nv_bfloat16* xsrc[NT]; int xbytes[NT];
    #pragma unroll
    for (int i = 0; i < NT; ++i) {
        const int sidx = tid + i * 128, r = sidx >> 4;
        xbytes[i] = r < m ? 16 : 0;
        xsrc[i] = X + (size_t)(r < m ? r : 0) * ldx + c0 * W12_KC + (sidx & 15) * 8;
    }
    const uint32_t sbase = smem_u32(w12_smem);
    auto issue = [&](int c, int stage) {
        const uint32_t ws = sbase + stage * STAGE_BYTES;
        #pragma unroll
        for (int i = 0; i < 6; ++i) cp_async16(ws + i * 2048 + tid * 16, wsrc + (size_t)c * W_STAGE + i * 2048, 16);
        const uint32_t xs = ws + W_STAGE;
        #pragma unroll
        for (int i = 0; i < NT; ++i) {
            const int sidx = tid + i * 128;
            cp_async16(xs + ((sidx >> 4) * W12_XSTRIDE + (sidx & 15) * 8) * 2, xsrc[i] + c * W12_KC, xbytes[i]);
        }
    };

    float acc[NT * CH][4];
    #pragma unroll
    for (int n = 0; n < NT * CH; ++n)
        #pragma unroll
        for (int q = 0; q < 4; ++q) acc[n][q] = 0.f;

    #pragma unroll
    for (int st = 0; st < NSTAGE - 1; ++st) {
        if (st < NC) issue(st, st);
        cp_async_commit();
    }
    for (int c = 0; c < NC; ++c) {
        cp_async_wait<NSTAGE - 2>();
        __syncthreads();
        {
            const int cn = c + NSTAGE - 1;
            if (cn < NC) issue(cn, cn % NSTAGE);
            cp_async_commit();
        }
        const uint8_t* ws = w12_smem + (c % NSTAGE) * STAGE_BYTES + warp * W12_TILE_BYTES;
        const __half* xs = reinterpret_cast<const __half*>(w12_smem + (c % NSTAGE) * STAGE_BYTES + W_STAGE);
        #pragma unroll
        for (int h = 0; h < 2; ++h) {                                     // steps 4h..4h+3
            uint4 lo[2]; uint4 nb;
            lo[0] = *reinterpret_cast<const uint4*>(ws + (2 * h) * 512 + lane * 16);          // steps 4h, 4h+1
            lo[1] = *reinterpret_cast<const uint4*>(ws + (2 * h + 1) * 512 + lane * 16);      // steps 4h+2, 4h+3
            nb = *reinterpret_cast<const uint4*>(ws + 2048 + h * 512 + lane * 16);          // steps 4h..4h+3
            #pragma unroll
            for (int ps = 0; ps < 2; ++ps) {                                 // steps 4h+2ps, +1
                uint32_t b[NT][4];
                #pragma unroll
                for (int nbk = 0; nbk < NT; ++nbk) {
                    if (nbk < ntb)
                        ldsm_x4(b[nbk][0], b[nbk][1], b[nbk][2], b[nbk][3],
                                smem_u32(xs + (nbk * 8 + (lane & 7)) * W12_XSTRIDE + h * 64 + ps * 32 + (lane >> 3) * 8));
                }
                #pragma unroll
                for (int q = 0; q < 2; ++q) {
                    const uint32_t L0 = q == 0 ? lo[ps].x : lo[ps].z, L1 = q == 0 ? lo[ps].y : lo[ps].w;
                    const int si = 2 * ps + q;
                    const uint32_t Nw = si == 0 ? nb.x : si == 1 ? nb.y : si == 2 ? nb.z : nb.w;
                    uint32_t a0, a1, a2, a3;
                    decode4(L0, Nw & 0xFFFFu, lut0, lut1, a0, a1);
                    decode4(L1, Nw >> 16, lut0, lut1, a2, a3);
                    #pragma unroll
                    for (int nbk = 0; nbk < NT; ++nbk) {
                        if (nbk < ntb) mma16816(acc[nbk * CH + (CH == 2 ? q : 0)], a0, a1, a2, a3, b[nbk][2 * q], b[nbk][2 * q + 1]);
                    }
                }
            }
        }
    }
    cp_async_wait<0>();
    if (CH == 2) {
        #pragma unroll
        for (int q = 0; q < 4; ++q) acc[0][q] += acc[1][q];
    }
    // partial slice: rows (weight rows) g, g+8 of this warp's tile -> columns; tokens 2t, 2t+1
    const int col0 = (blockIdx.x * 4 + warp) * 16 + g;
    float* pz = part + (size_t)(SPLITK ? blockIdx.y : 0) * m * N;
    #pragma unroll
    for (int nbk = 0; nbk < NT; ++nbk)
        #pragma unroll
        for (int hr = 0; hr < 2; ++hr)
            #pragma unroll
            for (int cc = 0; cc < 2; ++cc) {
                const int tok = nbk * 8 + 2 * t + cc;
                if (tok < m) pz[(size_t)tok * N + col0 + 8 * hr] = acc[nbk * CH][2 * hr + cc];
            }
}

// sum the KSPLIT partial slices of one 32-row pack tile, add its escapes (n_local, k, delta) * x[i][k], write bf16
__global__ void __launch_bounds__(256)
w12_reduce_kernel(const float* __restrict__ part, int ksplit, int m, int N, const __nv_bfloat16* __restrict__ X, int ldx,
                  const int* __restrict__ esc_off, const int* __restrict__ esc_n, const int* __restrict__ esc_k,
                  const float* __restrict__ esc_d, __nv_bfloat16* __restrict__ out, int ldo)
{
    __shared__ float red[32][33];
    __shared__ int s_en[64];
    __shared__ float s_ev[64][32];
    const int pn = blockIdx.x;
    for (int idx = threadIdx.x; idx < 32 * m; idx += 256) {
        const int n = idx & 31, i = idx >> 5;
        float v = 0.f;
        for (int z = 0; z < ksplit; ++z) v += __ldcg(part + ((size_t)z * m + i) * N + pn * 32 + n);
        red[n][i] = v;
    }
    __syncthreads();
    const int e0 = esc_off[pn], e1 = esc_off[pn + 1];
    for (int base = e0; base < e1; base += 64) {
        const int e = base + (int)threadIdx.x;
        if (threadIdx.x < 64) {
            if (e < e1) {
                s_en[threadIdx.x] = esc_n[e];
                const float d = esc_d[e]; const int kk = esc_k[e];
                for (int i = 0; i < m; ++i) s_ev[threadIdx.x][i] = d * __bfloat162float(X[(size_t)i * ldx + kk]);
            } else {
                s_en[threadIdx.x] = -1;
            }
        }
        __syncthreads();
        if (threadIdx.x < 32) {
            const int cnt_ = min(e1 - base, 64);
            for (int jj = 0; jj < cnt_; ++jj)
                if (s_en[jj] == (int)threadIdx.x)
                    for (int i = 0; i < m; ++i) red[threadIdx.x][i] += s_ev[jj][i];
        }
        __syncthreads();
    }
    for (int idx = threadIdx.x; idx < 32 * m; idx += 256) {
        const int n = idx & 31, i = idx >> 5;
        out[(size_t)i * ldo + pn * 32 + n] = __float2bfloat16(red[n][i]);
    }
}

template <int NT, int NSTAGE, bool SPLITK>
void launch_w12(const __nv_bfloat16* X, int ldx, int m, const uint8_t* W, int eb, float* part, int N, int K, int sk, cudaStream_t stream)
{
    constexpr int smem = NSTAGE * w12_stage_bytes(NT);
    static bool attr = false;
    if (!attr) { cudaFuncSetAttribute(w12_kernel<NT, NSTAGE, SPLITK>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
    w12_kernel<NT, NSTAGE, SPLITK><<<dim3(N / 64, sk), 128, smem, stream>>>(X, ldx, m, W, eb, part, N, K);
}
"""
rep("\n} // namespace\n\nvoid stream_bytes(", kernel + "\n} // namespace\n\nvoid stream_bytes(")

wrapper = r"""
void w12(torch::Tensor X, torch::Tensor W, int64_t eb, torch::Tensor esc_off, torch::Tensor esc_n, torch::Tensor esc_k, torch::Tensor esc_d,
         torch::Tensor out, torch::Tensor part, int64_t m, int64_t n, int64_t k, int64_t nt, int64_t nstage, int64_t sk)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(n % 64 == 0 && k % 128 == 0 && (k / 128) % sk == 0 && m <= 8 * nt, "w12: shape/config mismatch");
    TORCH_CHECK(X.stride(1) == 1 && (X.stride(0) * 2) % 16 == 0 && (reinterpret_cast<uintptr_t>(X.data_ptr()) & 15) == 0, "w12: x must be 16 B aligned rows");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(W.data_ptr()) & 127) == 0, "w12: weight layout must be 128 B aligned");
    TORCH_CHECK(sk * m * n <= part.numel(), "w12: partials too small");
    const __nv_bfloat16* Xp = reinterpret_cast<const __nv_bfloat16*>(X.data_ptr());
    const uint8_t* Wp = reinterpret_cast<const uint8_t*>(W.data_ptr());
    float* pp = part.data_ptr<float>();
    const int ldx = (int)X.stride(0);
    const int key = (int)nt * 100 + (int)nstage * 10 + (sk > 1 ? 1 : 0);
    switch (key) {
        case 140: launch_w12<1, 4, false>(Xp, ldx, (int)m, Wp, (int)eb, pp, (int)n, (int)k, 1, stream); break;
        case 141: launch_w12<1, 4, true>(Xp, ldx, (int)m, Wp, (int)eb, pp, (int)n, (int)k, (int)sk, stream); break;
        case 180: launch_w12<1, 8, false>(Xp, ldx, (int)m, Wp, (int)eb, pp, (int)n, (int)k, 1, stream); break;
        case 181: launch_w12<1, 8, true>(Xp, ldx, (int)m, Wp, (int)eb, pp, (int)n, (int)k, (int)sk, stream); break;
        case 240: launch_w12<2, 4, false>(Xp, ldx, (int)m, Wp, (int)eb, pp, (int)n, (int)k, 1, stream); break;
        case 241: launch_w12<2, 4, true>(Xp, ldx, (int)m, Wp, (int)eb, pp, (int)n, (int)k, (int)sk, stream); break;
        case 220: launch_w12<2, 2, false>(Xp, ldx, (int)m, Wp, (int)eb, pp, (int)n, (int)k, 1, stream); break;
        case 221: launch_w12<2, 2, true>(Xp, ldx, (int)m, Wp, (int)eb, pp, (int)n, (int)k, (int)sk, stream); break;
        case 440: launch_w12<4, 4, false>(Xp, ldx, (int)m, Wp, (int)eb, pp, (int)n, (int)k, 1, stream); break;
        case 441: launch_w12<4, 4, true>(Xp, ldx, (int)m, Wp, (int)eb, pp, (int)n, (int)k, (int)sk, stream); break;
        case 420: launch_w12<4, 2, false>(Xp, ldx, (int)m, Wp, (int)eb, pp, (int)n, (int)k, 1, stream); break;
        case 421: launch_w12<4, 2, true>(Xp, ldx, (int)m, Wp, (int)eb, pp, (int)n, (int)k, (int)sk, stream); break;
        default: TORCH_CHECK(false, "unsupported w12 config (nt, nstage, sk)");
    }
    w12_reduce_kernel<<<(unsigned)(n / 32), 256, 0, stream>>>(pp, (int)sk, (int)m, (int)n, Xp, ldx, esc_off.data_ptr<int>(),
        esc_n.data_ptr<int>(), esc_k.data_ptr<int>(), esc_d.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (int)out.stride(0));
}
"""
a = s.index("void mma12cp(torch::Tensor X, torch::Tensor LO")
b = s.index('"""', a)
s = s[:b] + wrapper + s[b:]
rep("""void mma12cp(torch::Tensor X, torch::Tensor LO, torch::Tensor NB, int64_t eb, torch::Tensor esc_off, torch::Tensor esc_n, torch::Tensor esc_k, torch::Tensor esc_d, torch::Tensor out, torch::Tensor part, torch::Tensor cnt, int64_t mt, int64_t rpw, int64_t nstage, int64_t sk);
\"\"\"""", """void mma12cp(torch::Tensor X, torch::Tensor LO, torch::Tensor NB, int64_t eb, torch::Tensor esc_off, torch::Tensor esc_n, torch::Tensor esc_k, torch::Tensor esc_d, torch::Tensor out, torch::Tensor part, torch::Tensor cnt, int64_t mt, int64_t rpw, int64_t nstage, int64_t sk);
void w12(torch::Tensor X, torch::Tensor W, int64_t eb, torch::Tensor esc_off, torch::Tensor esc_n, torch::Tensor esc_k, torch::Tensor esc_d,
         torch::Tensor out, torch::Tensor part, int64_t m, int64_t n, int64_t k, int64_t nt, int64_t nstage, int64_t sk);
\"\"\"""")
rep('functions=["stream_bytes", "gemv", "skinny_mma", "tgemv", "gemv12", "mma12", "mma12cp"]',
    'functions=["stream_bytes", "gemv", "skinny_mma", "tgemv", "gemv12", "mma12", "mma12cp", "w12"]')
# usable + repack appended after mma12cp_usable
rep("""def mma12cp_usable(m: int, n: int, k: int, mt: int, rpw: int, nstage: int, sk: int) -> bool:""",
    '''def w12_usable(m: int, n: int, k: int, nt: int, nstage: int, sk: int) -> bool:
    return m <= 8 * nt and n % 64 == 0 and k % 128 == 0 and (k // 128) % sk == 0 and (k // 128) // sk >= 1


def _repack_w12(pk, n: int, k: int) -> torch.Tensor:
    """_pack12 codes (same bits, same escapes) in the w12 kernel's fragment order: uint8 [n/64, k/128, 4, 3072]:
    per (m16 tile, chunk) low bytes [4 quads][32 lanes][16 B] (quad q = steps 2q, 2q+1; per step L0 = rows g, g+8
    at k 2t, 2t+1 then L1 at k 2t+8, 2t+9) and nibbles [2 halves][32 lanes][16 B] (per step one word, nibble i =
    element i of the L0, L1 order)."""
    dev = pk["lo"].device
    bn, bk = 32, int(pk["lo"].shape[1])                                                 # _pack12's tile shape
    lo = pk["lo"].view(n // bn, k // bk, bn, bk).permute(0, 2, 1, 3).reshape(n, k)
    nb = pk["nib"].view(n // bn, k // bk, bn, bk // 2).permute(0, 2, 1, 3).reshape(n, k // 2)
    nib = torch.stack((nb & 0xF, nb >> 4), -1).reshape(n, k)                            # [n, k] uint8 codes
    NTL, NC = n // 16, k // 128
    t = torch.arange(4, device=dev)
    kidx = torch.stack([2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9], -1)                 # [t, 4]
    out = torch.empty((NTL, NC, 3072), dtype=torch.uint8, device=dev)
    for part_lo in (True, False):
        src = lo if part_lo else nib
        s6 = src.view(NTL, 2, 8, NC, 8, 16)                                             # tile, rh, g, chunk, step, k16
        sel = s6[..., kidx]                                                             # tile, rh, g, chunk, step, t, q
        # element order: L0 = (rh0,q0),(rh0,q1),(rh1,q0),(rh1,q1); L1 = (rh0,q2),(rh0,q3),(rh1,q2),(rh1,q3)
        e = torch.stack([sel[:, 0, :, :, :, :, 0], sel[:, 0, :, :, :, :, 1], sel[:, 1, :, :, :, :, 0], sel[:, 1, :, :, :, :, 1],
                         sel[:, 0, :, :, :, :, 2], sel[:, 0, :, :, :, :, 3], sel[:, 1, :, :, :, :, 2], sel[:, 1, :, :, :, :, 3]], -1)
        # e: [tile, g, chunk, step, t, 8]
        e = e.permute(0, 2, 3, 1, 4, 5)                                                 # tile, chunk, step, g, t, 8 -> lane = g*4+t
        if part_lo:
            # [tile, chunk, step, lane, 8 B] -> quads: [tile, chunk, 4, 2 steps, lane, 8 B] -> [.., 4, lane, 16 B]
            w = e.reshape(NTL, NC, 4, 2, 32, 8).permute(0, 1, 2, 4, 3, 5).reshape(NTL, NC, 2048)
            out[:, :, :2048] = w
        else:
            v = e.to(torch.int32)
            word = torch.zeros((NTL, NC, 8, 32), dtype=torch.int32, device=dev)
            for i in range(8):
                word |= v[..., i].reshape(NTL, NC, 8, 32) << (4 * i)
            # [tile, chunk, step, lane] -> halves: [tile, chunk, 2, 4 steps, lane] -> [.., 2, lane, 4 words]
            w = word.view(NTL, NC, 2, 4, 32).permute(0, 1, 2, 4, 3).contiguous().view(torch.uint8).reshape(NTL, NC, 1024)
            out[:, :, 2048:] = w
    return out.view(NTL // 4, 4, NC, 3072).permute(0, 2, 1, 3).contiguous()           # [n/64, chunk, slot, 3 KB]


def mma12cp_usable(m: int, n: int, k: int, mt: int, rpw: int, nstage: int, sk: int) -> bool:''')
p.write_text(s)

# ---- submission.py glue
q = root / "muse_bf16_proj/submission.py"
t = q.read_text()


def rep2(old, new, count=1):
    global t
    assert t.count(old) == count, (t.count(old), old[:100])
    t = t.replace(old, new)


rep2("""M12CP_SHAPES = {}""", """M12CP_SHAPES = {}
# w12: the MoE-style weights-as-M 12-bit GEMM: (layer, n, m) -> (nt, nstage, sk); empty until the judge shows a win
W12_SHAPES = {}""")
rep2("""    if (layer, n, m) in M12CP_SHAPES:
        mt, rpw, nstage, sk = M12CP_SHAPES[(layer, n, m)]
        return dict(m12cp=True, mt=mt, rpw=rpw, nstage=nstage, sk=sk)""",
     """    if (layer, n, m) in M12CP_SHAPES:
        mt, rpw, nstage, sk = M12CP_SHAPES[(layer, n, m)]
        return dict(m12cp=True, mt=mt, rpw=rpw, nstage=nstage, sk=sk)
    if (layer, n, m) in W12_SHAPES:
        nt, nstage, sk = W12_SHAPES[(layer, n, m)]
        return dict(w12=True, nt=nt, nstage=nstage, sk=sk)""")
rep2("""    if cfg.get("m12cp"):
        return mma12cp_usable(m, n, k, cfg["mt"], cfg["rpw"], cfg["nstage"], cfg["sk"]) and (cfg["sk"] == 1 or cfg["sk"] * 16 * cfg["mt"] * n <= PARTIALS_FLOATS)""",
     """    if cfg.get("m12cp"):
        return mma12cp_usable(m, n, k, cfg["mt"], cfg["rpw"], cfg["nstage"], cfg["sk"]) and (cfg["sk"] == 1 or cfg["sk"] * 16 * cfg["mt"] * n <= PARTIALS_FLOATS)
    if cfg.get("w12"):
        return w12_usable(m, n, k, cfg["nt"], cfg["nstage"], cfg["sk"]) and cfg["sk"] * m * n <= PARTIALS_FLOATS""")
rep2("""    if cfg.get("m12cp"):
        pk = state.get(layer + "_p12")
        if pk is None:
            return _launch(x, w, w2, out, state, _triton_config(layer, m, n, k), m, n, k, layer)
        build().mma12cp(""", """    if cfg.get("w12"):
        pk = state.get(layer + "_p12")
        w12 = state.get(layer + "_w12")
        if pk is None or w12 is None:
            return _launch(x, w, w2, out, state, _triton_config(layer, m, n, k), m, n, k, layer)
        build().w12(x, w12, pk["eb"], pk["esc_off"], pk["esc_n"], pk["esc_k"], pk["esc_delta"], out, state["partials"],
                    m, n, k, cfg["nt"], cfg["nstage"], cfg["sk"])
        return out
    if cfg.get("m12cp"):
        pk = state.get(layer + "_p12")
        if pk is None:
            return _launch(x, w, w2, out, state, _triton_config(layer, m, n, k), m, n, k, layer)
        build().mma12cp(""")
rep2("""            if pk is not None:
                pk["lo32"] = pk["lo"].view(torch.int32)      # [tiles * 32, 64]
                pk["nib16"] = pk["nib"].view(torch.int16)    # [tiles * 32, 64]
            state[name + "_p12"] = pk""", """            if pk is not None:
                pk["lo32"] = pk["lo"].view(torch.int32)      # [tiles * 32, 64]
                pk["nib16"] = pk["nib"].view(torch.int16)    # [tiles * 32, 64]
                if USE_W12:
                    try:
                        nn_, kk_ = state[name].shape
                        state[name + "_w12"] = _repack_w12(pk, nn_, kk_)
                    except Exception as e:
                        print("w12 repack failed for", name, repr(e)[:120])
            state[name + "_p12"] = pk""")
rep2("""USE_PACK12 = True   # also keep the 12-bit packing of the BF16 layers (the kernels that use it: *_SHAPES tables)""",
     """USE_PACK12 = True   # also keep the 12-bit packing of the BF16 layers (the kernels that use it: *_SHAPES tables)
USE_W12 = True      # the w12 fragment-ordered copy of the 12-bit packing (W12_SHAPES)""")
# bench: w12 rows
rep2("""        print("BENCH m=1 12-bit GEMV, more CTAs for qkv / the unpooler slice")""",
     """        print("BENCH w12 (MoE-style weights-as-M 12-bit GEMM; nt/nstage/sk) vs the table at m=8/16/32")
        w12c = {"pooler": [4, 8], "unpooler": [2, 4, 8], "qkv_gate": [8, 16], "o": [2, 4]}
        for layer, n in (("pooler", 4096), ("unpooler", 8192), ("qkv_gate", 2304), ("o", 8192), ("unpooler", 1024)):
            for m in (8, 16, 32):
                if not budget():
                    break
                if layer == "unpooler" and ((n == 1024) != (m <= 8)):
                    continue
                nt = 1 if m <= 8 else 2 if m <= 16 else 4
                parts = [row("table", layer, m, n, TAB)]
                sks = w12c[layer] if n != 1024 else [4, 8, 16]
                for sk in sks:
                    for ns in ((4, 8) if nt == 1 else (2, 4)):
                        parts.append(row(f"w12 {nt}/{ns}/{sk}", layer, m, n, dict(w12=True, nt=nt, nstage=ns, sk=sk)))
                print(f"  m={m:2d} {layer:9s} n={n}: " + " | ".join(parts))
        print("BENCH m=1 12-bit GEMV, more CTAs for qkv / the unpooler slice")""")
rep2("BENCH = False       # microbenchmarks in prepare", "BENCH = True        # microbenchmarks in prepare")
q.write_text(t)

# ---- CPU check of the repack
(root / "tools/check_w12.py").write_text('''"""CPU check of _repack_w12: rebuild every element's 12-bit code from the fragment-ordered layout (the kernel's
addressing: tile, chunk, lane (g, t), step, L0/L1 order, nibble i) and compare with _pack12's lo/nibble codes; then
emulate decode4 on a sample of words and compare the bf16 bits with _unpack12. Run in the judge image."""
import sys
sys.path.insert(0, "/work/muse_bf16_proj")
import torch
import submission as S
import cuda_proj as C

torch.manual_seed(0)
n, k = 128, 512
w = (torch.randn((n, k)) * 3).to(torch.bfloat16)
pk = S._pack12(w)
assert pk is not None
W = C._repack_w12(pk, n, k)                                   # [n/64, k/128, 4, 3072]
lo = pk["lo"].view(n // 32, k // 256, 32, 256).permute(0, 2, 1, 3).reshape(n, k)
nb = pk["nib"].view(n // 32, k // 256, 32, 128).permute(0, 2, 1, 3).reshape(n, k // 2)
nib = torch.stack((nb & 0xF, nb >> 4), -1).reshape(n, k)
bad_lo = bad_nb = 0
for tile in range(n // 16):
    for c in range(k // 128):
        blk = W[tile // 4, c, tile % 4]
        for lane in range(32):
            g, t = lane // 4, lane % 4
            for s in range(8):
                q, half = s // 2, s // 4
                lob = blk[q * 512 + lane * 16 + (s % 2) * 8: q * 512 + lane * 16 + (s % 2) * 8 + 8].tolist()
                word = int(blk[2048 + half * 512 + lane * 16 + (s % 4) * 4: 2048 + half * 512 + lane * 16 + (s % 4) * 4 + 4].view(torch.int32)[0]) & 0xFFFFFFFF
                rows = [tile * 16 + g, tile * 16 + g, tile * 16 + g + 8, tile * 16 + g + 8] * 2
                ks = [c * 128 + s * 16 + 2 * t, c * 128 + s * 16 + 2 * t + 1] * 2 + [c * 128 + s * 16 + 2 * t + 8, c * 128 + s * 16 + 2 * t + 9] * 2
                for i in range(8):
                    bad_lo += lob[i] != int(lo[rows[i], ks[i]])
                    bad_nb += ((word >> (4 * i)) & 0xF) != int(nib[rows[i], ks[i]])
print("lo byte mismatches", bad_lo, "nibble mismatches", bad_nb)
assert bad_lo == 0 and bad_nb == 0


def prmt(a, b, sel):
    src = [(a >> (8 * i)) & 0xFF for i in range(4)] + [(b >> (8 * i)) & 0xFF for i in range(4)]
    d = 0
    for i in range(4):
        sl = (sel >> (4 * i)) & 0xF
        byte = src[sl & 7]
        if sl & 8:
            byte = 0xFF if byte & 0x80 else 0
        d |= byte << (8 * i)
    return d


hb = pk["eb"] >> 1
lut0 = hb | (hb + 1) << 8 | (hb + 2) << 16 | (hb + 3) << 24
lut1 = (hb + 4) | (hb + 5) << 8 | (hb + 6) << 16 | (hb + 7) << 24
dec = S._unpack12(pk, n, k).view(torch.int16).numpy().astype("uint16")


def decode4(L, n16):
    E = prmt(n16, 0, 0x1100)
    H = prmt(lut0, lut1, n16 & 0x7777) | ((E & 0x00080008) << 4) | (E & 0x80008000)
    return prmt(L, H, 0x5140), prmt(L, H, 0x7362)


bad = 0
for tile in range(n // 16):
    for c in range(k // 128):
        blk = W[tile // 4, c, tile % 4]
        for lane in range(32):
            g, t = lane // 4, lane % 4
            for s in range(8):
                q, half = s // 2, s // 4
                lw = blk[q * 512 + lane * 16 + (s % 2) * 8: q * 512 + lane * 16 + (s % 2) * 8 + 8].view(torch.int32).tolist()
                L0, L1 = lw[0] & 0xFFFFFFFF, lw[1] & 0xFFFFFFFF
                Nw = int(blk[2048 + half * 512 + lane * 16 + (s % 4) * 4: 2048 + half * 512 + lane * 16 + (s % 4) * 4 + 4].view(torch.int32)[0]) & 0xFFFFFFFF
                a0, a1 = decode4(L0, Nw & 0xFFFF)
                a2, a3 = decode4(L1, Nw >> 16)
                r0, r1 = tile * 16 + g, tile * 16 + g + 8
                kb = c * 128 + s * 16 + 2 * t
                exp = [(r0, kb), (r0, kb + 1), (r1, kb), (r1, kb + 1), (r0, kb + 8), (r0, kb + 9), (r1, kb + 8), (r1, kb + 9)]
                got = [a0 & 0xFFFF, a0 >> 16, a1 & 0xFFFF, a1 >> 16, a2 & 0xFFFF, a2 >> 16, a3 & 0xFFFF, a3 >> 16]
                for (r, kk), v in zip(exp, got):
                    bad += int(dec[r, kk]) != v
print("decoded bf16 mismatches vs _unpack12:", bad)
assert bad == 0
print("w12 ok")
''')
print("edited")
