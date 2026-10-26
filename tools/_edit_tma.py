"""Round 2 edit: TMA bulk-copy path (cp.async.bulk + mbarrier ring) for the wt kernel, 4-tile-grouped weight
layout (4 KB contiguous per group and chunk), variant tuple gains a 6th element (tma)."""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
p = root / "muse_moe_w4a16/cuda_moe.py"
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:100])
    s = s.replace(old, new)


# ---- helpers (TMA / mbarrier)
rep("""constexpr int WT_NWARPS = 4;
__host__ __device__ constexpr int wt_stage_bytes(int tpw, int nt) {""",
"""__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\\n" :: "r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
    asm volatile("{\\n .reg .pred p;\\n WAIT_%=:\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\\n @!p bra WAIT_%=;\\n}\\n"
                 :: "r"(bar), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\\n"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory"); }
__device__ __forceinline__ void fence_mbar_init() { asm volatile("fence.mbarrier_init.release.cluster;\\n" ::: "memory"); }

constexpr int WT_NWARPS = 4;
__host__ __device__ constexpr int wt_stage_bytes(int tpw, int nt) {""")

# ---- kernel rewrite
a = s.index("template <bool GATED, int MW, int NT, int KS, int NSTAGE, bool SPLITK>\n__global__ void __launch_bounds__(128 * KS)\nmoe_w4a16_wt_kernel(")
b = s.index("template <bool GATED, int MW, int NT, int KS, int NSTAGE, bool SPLITK>\nvoid launch_wt(")
new_kernel = r"""template <bool GATED, int MW, int NT, int KS, int NSTAGE, bool SPLITK, bool TMA>
__global__ void __launch_bounds__(128 * KS)
moe_w4a16_wt_kernel(const __half* __restrict__ X, int ldx,
                    const int* __restrict__ sorted_tok,
                    const int* __restrict__ blk_e, const int* __restrict__ blk_row0,
                    const int* __restrict__ blk_end, const int* __restrict__ nblk,
                    const uint8_t* __restrict__ Wr, const uint8_t* __restrict__ Sr, const float* __restrict__ gs,
                    __half* __restrict__ out16, __nv_bfloat16* __restrict__ out_bf, int ldo,
                    float* __restrict__ part, int ldp, int part_rows,
                    int K, int NTILES, float gs_mul)
{
    // Weights: Wr[e][tile/4][chunk][tile%4][1 KB], scales Sr[e][tile/4][chunk][tile%4][128 B]: a 4-tile group's
    // chunk is 4 KB contiguous. TMA: thread 0 issues cp.async.bulk copies (per chunk: TPC/4 weight groups, TPC/4
    // scale groups, the valid token rows) tracked by one mbarrier per stage; else 16 B cp.async by all threads.
    // SPLITK: gridDim.z CTAs split K; CTA z owns chunks [z*NC/KSPLIT, (z+1)*NC/KSPLIT) and stores its fp32
    // partial sums (times gs) to part[z][row][col] (GATED: gate at col, up at col + N/2) for the reduce kernels.
    constexpr int TPW = (GATED ? 2 : 1) * MW;        // m16 weight tiles per warp
    constexpr int TPC = TPW * WT_NWARPS;              // per CTA
    constexpr int GT = WT_NWARPS * MW;                // gate (or all) tiles per CTA = 4-tile groups * 4
    constexpr int NGRP = TPC / 4;                     // 4-tile groups per CTA
    constexpr int W_STAGE = TPC * 1024;
    constexpr int S_STAGE = TPC * 128;
    constexpr int STAGE_BYTES = wt_stage_bytes(TPW, NT);
    constexpr int NTHR = 128 * KS;
    constexpr int CH = NT == 1 ? 2 : 1;               // accumulator chains per tile (latency)
    extern __shared__ __align__(128) uint8_t wt_smem[];

    const int j = blockIdx.y;
    if (j >= *nblk) return;
    const int e = blk_e[j];
    const int row0 = blk_row0[j];
    const int nrows = blk_end[j] - row0;              // valid tokens in this block (<= 8*NT)
    const int ntb = (nrows + 7) >> 3;                 // token n8 blocks in use
    const int tid = threadIdx.x;
    const int warp = (tid >> 5) & 3, lane = tid & 31, gidx = tid >> 7;
    const int g = lane >> 2, t = lane & 3;
    const int KSPLIT = SPLITK ? (int)gridDim.z : 1;
    const int NC_ALL = K / KC;
    const int NC = NC_ALL / KSPLIT;                   // chunks of this CTA
    const int c0 = SPLITK ? (int)blockIdx.z * NC : 0;
    const int NG = NTILES / 4;                        // 4-tile groups per expert

    // CTA tile ti = warp * TPW + i -> global m16 tile and its smem slot (TMA lands whole groups: gate groups first)
    auto tile_of = [&](int ti) {
        const int w = ti / TPW, i = ti % TPW;
        int tile = (blockIdx.x * WT_NWARPS + w) * MW + (i % MW);
        if (GATED && i >= MW) tile += NTILES / 2;
        return tile;
    };
    auto slot_of = [&](int ti) {
        if (!TMA) return ti;
        const int w = ti / TPW, i = ti % TPW;
        return (GATED && i >= MW ? GT : 0) + w * MW + (i % MW);
    };
    const size_t ebase = (size_t)e * NG;
    // global address of (tile, chunk c) weights / scales
    auto waddr = [&](int tile, int c) { return Wr + ((ebase + (tile >> 2)) * NC_ALL + c) * 4096 + (tile & 3) * 1024; };
    auto saddr = [&](int tile, int c) { return Sr + ((ebase + (tile >> 2)) * NC_ALL + c) * 512 + (tile & 3) * 128; };

    const uint32_t smem_base = smem_u32(wt_smem);
    const uint32_t bars = smem_base + NSTAGE * STAGE_BYTES;      // TMA: one mbarrier per stage

    // ---- cp.async issue (all threads) ----
    constexpr int WSEG = TPC * 64 / NTHR;
    constexpr int XSEG = (NT * 128 + NTHR - 1) / NTHR;
    const uint8_t* wsrc[TMA ? 1 : WSEG];
    const uint8_t* ssrc = nullptr;
    const __half* xsrc[TMA ? 1 : XSEG]; int xbytes[TMA ? 1 : XSEG];
    if constexpr (!TMA) {
        #pragma unroll
        for (int i = 0; i < WSEG; ++i) {
            const int sidx = tid + i * NTHR;
            wsrc[i] = waddr(tile_of(sidx >> 6), c0) + (sidx & 63) * 16;
        }
        ssrc = saddr(tile_of((tid & (TPC * 8 - 1)) >> 3), c0) + (tid & 7) * 16;
        #pragma unroll
        for (int i = 0; i < XSEG; ++i) {
            const int sidx = tid + i * NTHR;
            const int r = sidx >> 4;
            const bool valid = (sidx < NT * 128) && (r < nrows);
            const int tok = valid ? (GATED ? sorted_tok[row0 + r] : row0 + r) : 0;
            xsrc[i] = X + (size_t)tok * ldx + c0 * KC + (sidx & 15) * 8;
            xbytes[i] = valid ? 16 : 0;
        }
    }
    auto issue = [&](int c, int stage) {
        const uint32_t ws = smem_base + stage * STAGE_BYTES;
        const uint32_t ss = ws + W_STAGE;
        const uint32_t xs = ss + S_STAGE;
        #pragma unroll
        for (int i = 0; i < WSEG; ++i)
            cp_async16(ws + (tid + i * NTHR) * 16, wsrc[i] + c * 4096, 16);
        if (tid < TPC * 8) cp_async16(ss + tid * 16, ssrc + c * 512, 16);
        #pragma unroll
        for (int i = 0; i < XSEG; ++i) {
            const int sidx = tid + i * NTHR;
            if (sidx < NT * 128)
                cp_async16(xs + ((sidx >> 4) * A_STRIDE + (sidx & 15) * 8) * 2, xsrc[i] + c * KC, xbytes[i]);
        }
    };

    // ---- TMA issue (thread 0) ----
    const int nvalid = nrows < 8 * NT ? nrows : 8 * NT;
    const __half* xrow[TMA ? 8 * NT : 1];
    if constexpr (TMA) {
        if (tid == 0) {
            #pragma unroll
            for (int r = 0; r < 8 * NT; ++r) {
                const int tok = r < nrows ? (GATED ? sorted_tok[row0 + r] : row0 + r) : 0;
                xrow[r] = X + (size_t)tok * ldx + c0 * KC;
            }
            #pragma unroll
            for (int st = 0; st < NSTAGE; ++st) mbar_init(bars + 8 * st, 1);
            fence_mbar_init();
        }
        __syncthreads();
    }
    auto issue_tma = [&](int c, int stage) {
        const uint32_t ws = smem_base + stage * STAGE_BYTES;
        const uint32_t ss = ws + W_STAGE;
        const uint32_t xs = ss + S_STAGE;
        const uint32_t bar = bars + 8 * stage;
        mbar_expect_tx(bar, (uint32_t)(W_STAGE + S_STAGE + nvalid * KC * 2));
        #pragma unroll
        for (int gi = 0; gi < NGRP; ++gi) {
            // group gi: gate groups blockIdx.x*MW + gi (gi < MW), up groups beyond (GATED)
            int grp = blockIdx.x * MW + (gi % MW);
            if (GATED && gi >= MW) grp += NG / 2;
            const uint8_t* wsrc_g = Wr + ((ebase + grp) * NC_ALL + c0 + c) * 4096;
            const uint8_t* ssrc_g = Sr + ((ebase + grp) * NC_ALL + c0 + c) * 512;
            bulk_g2s(ws + gi * 4096, wsrc_g, 4096, bar);
            bulk_g2s(ss + gi * 512, ssrc_g, 512, bar);
        }
        for (int r = 0; r < nvalid; ++r)
            bulk_g2s(xs + r * A_STRIDE * 2, xrow[r] + c * KC, KC * 2, bar);
    };

    float acc[TPW][NT * CH][4];
    #pragma unroll
    for (int i = 0; i < TPW; ++i)
        #pragma unroll
        for (int n = 0; n < NT * CH; ++n)
            #pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][n][q] = 0.f;

    const float gsf = gs[e] * gs_mul;                                   // gs * 64: applied to the fp32 sums

    if constexpr (TMA) {
        if (tid == 0) {
            #pragma unroll
            for (int st = 0; st < NSTAGE - 1; ++st)
                if (st < NC) issue_tma(st, st);
        }
    } else {
        #pragma unroll
        for (int st = 0; st < NSTAGE - 1; ++st) {
            if (st < NC) issue(st, st);
            cp_async_commit();
        }
    }

    for (int c = 0; c < NC; ++c) {
        const int stage = c % NSTAGE;
        if constexpr (TMA) {
            __syncthreads();                                         // everyone is done with stage (c-1) % NSTAGE
            if (tid == 0) {
                const int cn = c + NSTAGE - 1;
                if (cn < NC) { fence_proxy_async(); issue_tma(cn, cn % NSTAGE); }
            }
            mbar_wait(bars + 8 * stage, (uint32_t)((c / NSTAGE) & 1));
        } else {
            cp_async_wait<NSTAGE - 2>();
            __syncthreads();
            const int cn = c + NSTAGE - 1;
            if (cn < NC) issue(cn, cn % NSTAGE);
            cp_async_commit();
        }
        const uint8_t* ws = wt_smem + stage * STAGE_BYTES;
        const uint8_t* ss = ws + W_STAGE;
        const __half* xs = reinterpret_cast<const __half*>(ss + S_STAGE);
        #pragma unroll
        for (int hh = 0; hh < 2 / KS; ++hh) {
            const int h = KS == 1 ? hh : gidx;                       // half chunk: k16 steps 4h..4h+3
            uint4 wq[TPW]; uint2 sq[TPW];
            #pragma unroll
            for (int i = 0; i < TPW; ++i) {
                const int sl = slot_of(warp * TPW + i);
                wq[i] = *reinterpret_cast<const uint4*>(ws + (sl * 2 + h) * 512 + lane * 16);
                sq[i] = *reinterpret_cast<const uint2*>(ss + sl * 128 + g * 16 + h * 8);
            }
            #pragma unroll
            for (int ps = 0; ps < 2; ++ps) {                         // steps 2ps, 2ps+1 of the half
                uint32_t b[NT][4];
                #pragma unroll
                for (int nb = 0; nb < NT; ++nb) {
                    if (nb < ntb)
                        ldmatrix_x4(b[nb][0], b[nb][1], b[nb][2], b[nb][3],
                                    smem_u32(xs + (nb * 8 + (lane & 7)) * A_STRIDE + h * 64 + ps * 32 + (lane >> 3) * 8));
                }
                #pragma unroll
                for (int q = 0; q < 2; ++q) {
                    const int s = 2 * ps + q;
                    #pragma unroll
                    for (int i = 0; i < TPW; ++i) {
                        const uint32_t w = s == 0 ? wq[i].x : s == 1 ? wq[i].y : s == 2 ? wq[i].z : wq[i].w;
                        const uint32_t sw = ps == 0 ? sq[i].x : sq[i].y;
                        const uint32_t sp = q == 0 ? e4m3x2_lo16_to_f16x2(sw) : e4m3x2_hi16_to_f16x2(sw);   // (s_g, s_g8)
                        const uint32_t S0 = prmt(sp, sp, 0x1010u);          // row g   block scale, both halves
                        const uint32_t S1 = prmt(sp, sp, 0x3232u);          // row g+8
                        // fp16 weights = e2m1 * 2^-6 * blockscale: <= 6 significant bits, exact
                        const uint32_t elo = w & 0x9C9C9C9Cu;
                        const uint32_t ehi = ((w << 2) & 0x8C8C8C8Cu) | ((w >> 2) & 0x10101010u);
                        uint32_t a[4];
                        a[0] = f16x2_mul(e4m3x2_lo16_to_f16x2(elo), S0);
                        a[1] = f16x2_mul(e4m3x2_hi16_to_f16x2(elo), S1);
                        a[2] = f16x2_mul(e4m3x2_lo16_to_f16x2(ehi), S0);
                        a[3] = f16x2_mul(e4m3x2_hi16_to_f16x2(ehi), S1);
                        #pragma unroll
                        for (int nb = 0; nb < NT; ++nb) {
                            if (nb < ntb) mma16816(acc[i][nb * CH + (CH == 2 ? q : 0)], a, b[nb][2 * q], b[nb][2 * q + 1]);
                        }
                    }
                }
            }
        }
    }
    if constexpr (!TMA) cp_async_wait<0>();

    if (KS == 2) {
        __syncthreads();                                   // the ring is free: reuse it for the reduction
        float* red = reinterpret_cast<float*>(wt_smem);
        if (gidx == 1) {
            #pragma unroll
            for (int i = 0; i < TPW; ++i)
                #pragma unroll
                for (int n = 0; n < NT * CH; ++n)
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) red[((i * NT * CH + n) * 4 + q) * 128 + (tid & 127)] = acc[i][n][q];
        }
        __syncthreads();
        if (gidx == 1) return;
        #pragma unroll
        for (int i = 0; i < TPW; ++i)
            #pragma unroll
            for (int n = 0; n < NT * CH; ++n)
                #pragma unroll
                for (int q = 0; q < 4; ++q) acc[i][n][q] += red[((i * NT * CH + n) * 4 + q) * 128 + (tid & 127)];
    }
    if (CH == 2) {
        #pragma unroll
        for (int i = 0; i < TPW; ++i)
            #pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][0][q] += acc[i][1][q];
    }

    // C fragment: rows (weight rows) g, g+8 of the tile -> output column; cols (tokens) 2t, 2t+1
    if constexpr (SPLITK) {
        #pragma unroll
        for (int i = 0; i < MW; ++i) {
            const int tile = (blockIdx.x * WT_NWARPS + warp) * MW + i;
            #pragma unroll
            for (int nb = 0; nb < NT; ++nb)
                #pragma unroll
                for (int hr = 0; hr < 2; ++hr) {
                    const int col = tile * 16 + g + 8 * hr;
                    #pragma unroll
                    for (int cc = 0; cc < 2; ++cc) {
                        const int tok = nb * 8 + 2 * t + cc;
                        if (tok >= nrows) continue;
                        float* pp = part + ((size_t)blockIdx.z * part_rows + row0 + tok) * ldp + col;
                        pp[0] = acc[i][nb * CH][2 * hr + cc] * gsf;
                        if (GATED) pp[(size_t)NTILES * 8] = acc[MW + i][nb * CH][2 * hr + cc] * gsf;
                    }
                }
        }
        return;
    }
    #pragma unroll
    for (int i = 0; i < MW; ++i) {
        const int tile = (blockIdx.x * WT_NWARPS + warp) * MW + i;
        #pragma unroll
        for (int nb = 0; nb < NT; ++nb) {
            #pragma unroll
            for (int hr = 0; hr < 2; ++hr) {
                const int col = tile * 16 + g + 8 * hr;
                #pragma unroll
                for (int cc = 0; cc < 2; ++cc) {
                    const int tok = nb * 8 + 2 * t + cc;
                    if (tok >= nrows) continue;
                    const float gv = acc[i][nb * CH][2 * hr + cc] * gsf;
                    const float uv = GATED ? acc[MW + i][nb * CH][2 * hr + cc] * gsf : 0.f;
                    const size_t o = (size_t)(row0 + tok) * ldo + col;
                    if (GATED) {
                        const float hv = gv / (1.f + __expf(-gv)) * uv;
                        out16[o] = __float2half(__bfloat162float(__float2bfloat16(hv)));   // bf16 value in an fp16 container
                    } else {
                        out_bf[o] = __float2bfloat16(gv);
                    }
                }
            }
        }
    }
}

"""
s = s[:a] + new_kernel + s[b:]

# ---- launch_wt
a = s.index("template <bool GATED, int MW, int NT, int KS, int NSTAGE, bool SPLITK>\nvoid launch_wt(")
b = s.index("// split-K reduce of GEMM1's fp32 partial slices")
new_launch = r"""template <bool GATED, int MW, int NT, int KS, int NSTAGE, bool SPLITK, bool TMA>
void launch_wt(const __half* X, int ldx, const int* st, const int* be, const int* br, const int* bend, const int* nb,
               const uint8_t* Wr, const uint8_t* Sr, const float* gs, __half* o16, __nv_bfloat16* obf, int ldo,
               float* part, int ldp, int part_rows, int ksplit,
               int K, int ntiles, int max_blocks, float gs_mul, cudaStream_t stream)
{
    constexpr int TPW = (GATED ? 2 : 1) * MW;
    constexpr int smem = NSTAGE * wt_stage_bytes(TPW, NT) + (TMA ? NSTAGE * 8 : 0);
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(moe_w4a16_wt_kernel<GATED, MW, NT, KS, NSTAGE, SPLITK, TMA>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        attr_set = true;
    }
    const int n_x = (GATED ? ntiles / 2 : ntiles) / (WT_NWARPS * MW);
    dim3 grid(n_x, max_blocks, ksplit);
    moe_w4a16_wt_kernel<GATED, MW, NT, KS, NSTAGE, SPLITK, TMA><<<grid, 128 * KS, smem, stream>>>(
        X, ldx, st, be, br, bend, nb, Wr, Sr, gs, o16, obf, ldo, part, ldp, part_rows, K, ntiles, gs_mul);
}

"""
s = s[:a] + new_launch + s[b:]

# ---- wrapper: tma flag
rep("""                  torch::Tensor out, torch::Tensor part, int64_t ksplit, int64_t K, int64_t ntiles,
                  int64_t max_blocks, bool gated, int64_t mw, int64_t nt, int64_t ks, int64_t nstage, double gs_mul)""",
    """                  torch::Tensor out, torch::Tensor part, int64_t ksplit, int64_t K, int64_t ntiles,
                  int64_t max_blocks, bool gated, int64_t mw, int64_t nt, int64_t ks, int64_t nstage, double gs_mul, bool tma)""", 2)
rep("#define OKC_W(G, M, T, S, NS, SK) launch_wt<G, M, T, S, NS, SK>(Xp,", "#define OKC_W(G, M, T, S, NS, SK, TM) launch_wt<G, M, T, S, NS, SK, TM>(Xp,")
a = s.index("    const int key = (ksplit > 1 ? 1000000 : 0) + (gated ? 100000 : 0)")
b = s.index("#undef OKC_W")
new_sw = """    const int key = (tma ? 10000000 : 0) + (ksplit > 1 ? 1000000 : 0) + (gated ? 100000 : 0) + (int)mw * 10000 + (int)nt * 1000 + (int)ks * 100 + (int)nstage;
    switch (key) {
        case 111102: OKC_W(true, 1, 1, 1, 2, false, false); break;   case 111104: OKC_W(true, 1, 1, 1, 4, false, false); break;
        case 111108: OKC_W(true, 1, 1, 1, 8, false, false); break;
        case 112104: OKC_W(true, 1, 2, 1, 4, false, false); break;   case 114104: OKC_W(true, 1, 4, 1, 4, false, false); break;
        case 114102: OKC_W(true, 1, 4, 1, 2, false, false); break;
        case 11104: OKC_W(false, 1, 1, 1, 4, false, false); break;   case 11108: OKC_W(false, 1, 1, 1, 8, false, false); break;
        case 21104: OKC_W(false, 2, 1, 1, 4, false, false); break;   case 22104: OKC_W(false, 2, 2, 1, 4, false, false); break;
        case 22102: OKC_W(false, 2, 2, 1, 2, false, false); break;   case 24102: OKC_W(false, 2, 4, 1, 2, false, false); break;
        case 24104: OKC_W(false, 2, 4, 1, 4, false, false); break;   case 14104: OKC_W(false, 1, 4, 1, 4, false, false); break;
        // split-K (SPLITK = true): partial-slice stores, for the small batches
        case 1111104: OKC_W(true, 1, 1, 1, 4, true, false); break;   case 1011104: OKC_W(false, 1, 1, 1, 4, true, false); break;
        // TMA bulk-copy ring
        case 10111104: OKC_W(true, 1, 1, 1, 4, false, true); break;  case 10111108: OKC_W(true, 1, 1, 1, 8, false, true); break;
        case 10112104: OKC_W(true, 1, 2, 1, 4, false, true); break;  case 10114104: OKC_W(true, 1, 4, 1, 4, false, true); break;
        case 10011104: OKC_W(false, 1, 1, 1, 4, false, true); break; case 10011108: OKC_W(false, 1, 1, 1, 8, false, true); break;
        case 10021104: OKC_W(false, 2, 1, 1, 4, false, true); break; case 10022104: OKC_W(false, 2, 2, 1, 4, false, true); break;
        case 10024104: OKC_W(false, 2, 4, 1, 4, false, true); break;
        default: TORCH_CHECK(false, "unsupported wt variant (mw, nt, ks, nstage, ksplit, tma)");
    }
"""
s = s[:a] + new_sw + s[b:]

# ---- repack: grouped layout
rep("""    sfu = sf.view(torch.uint8).view(E, NTL, 2, 8, NC, 8)                            # e, tile, rh, g, chunk, step
    Sr = sfu.permute(0, 1, 4, 3, 5, 2).contiguous().view(E, NTL, NC, 8, 16)         # e, tile, chunk, g, (step, rh)
    return Wr.contiguous(), Sr""",
"""    sfu = sf.view(torch.uint8).view(E, NTL, 2, 8, NC, 8)                            # e, tile, rh, g, chunk, step
    Sr = sfu.permute(0, 1, 4, 3, 5, 2).contiguous().view(E, NTL, NC, 8, 16)         # e, tile, chunk, g, (step, rh)
    # 4-tile groups: a group's chunk is 4 KB (weights) / 512 B (scales) contiguous, for bulk copies
    Wr = Wr.view(E, NTL // 4, 4, NC, 1024).permute(0, 1, 3, 2, 4).contiguous()      # e, group, chunk, slot, 1 KB
    Sr = Sr.view(E, NTL // 4, 4, NC, 128).permute(0, 1, 3, 2, 4).contiguous()       # e, group, chunk, slot, 128 B
    return Wr, Sr""")
rep('''    Wr uint8 [E, N/16, K/128, 2, 32, 16]: per (expert, m16 tile, chunk, half, lane) 4 words = the lane's''',
    '''    Wr uint8 [E, N/64, K/128, 4, 1024] (per expert, 4-tile group, chunk, tile slot: 2 halves x 32 lanes x 16 B): per
    (tile, chunk, half, lane) 4 words = the lane's''')
p.write_text(s)

# ---- generator
g = root / "tools/gen_moe_kernel_only.py"
u = g.read_text()
u = u.replace('''for g, m, t, s, n, sk in re.findall(r"OKC_W\\((true|false), (\\d), (\\d), (\\d), (\\d), (true|false)\\)", cu):
    inst.append(f"template void launch_wt<{g}, {m}, {t}, {s}, {n}, {sk}>{wsig};")''',
              '''for g, m, t, s, n, sk, tm in re.findall(r"OKC_W\\((true|false), (\\d), (\\d), (\\d), (\\d), (true|false), (true|false)\\)", cu):
    inst.append(f"template void launch_wt<{g}, {m}, {t}, {s}, {n}, {sk}, {tm}>{wsig};")''')
g.write_text(u)

# ---- Python glue: 6-tuple variants
q = root / "muse_moe_w4a16/submission.py"
t = q.read_text()


def rep2(old, new, count=1):
    global t
    assert t.count(old) == count, (t.count(old), old[:100])
    t = t.replace(old, new)


rep2("WT_DEFAULT = {8: (1, 1, 1, 8, 1), 64: (1, 1, 1, 4, 1), 128: (2, 2, 1, 4, 1), 100000: (4, 2, 1, 2, 1)}",
     "WT_DEFAULT = {8: (1, 1, 1, 8, 1, 1), 64: (1, 1, 1, 4, 1, 1), 128: (2, 2, 1, 4, 1, 1), 100000: (4, 2, 1, 2, 1, 0)}")
rep2("WT_SAFE = (1, 1, 1, 4, 1)", "WT_SAFE = (1, 1, 1, 4, 1, 0)")
rep2("# (token n8 blocks per expert block, GEMM2 m16 tiles per warp, K-split warp groups, cp.async stages, K-split CTAs)",
     "# (token n8 blocks per expert block, GEMM2 m16 tiles per warp, K-split warp groups, stages, K-split CTAs, TMA bulk copies)")
rep2("    nt, mw2, ks, nstage, ksplit = _wt_variant(M, state)\n    BM = 8 * nt", "    nt, mw2, ks, nstage, ksplit, tma = _wt_variant(M, state)\n    BM = 8 * nt")
rep2("""                     state["Wt13"], state["St13"], state["gs13"], inter, part, ksplit, H, 2 * I // 16, maxb, True, 1, nt, ks,
                     nstage, state["gs_mul"])""", """                     state["Wt13"], state["St13"], state["gs13"], inter, part, ksplit, H, 2 * I // 16, maxb, True, 1, nt, ks,
                     nstage, state["gs_mul"], bool(tma))""")
rep2("""                     state["Wt2"], state["St2"], state["gs2"], y, part, ksplit, I, H // 16, maxb, False, mw2, nt, ks, nstage,
                     state["gs_mul"])""", """                     state["Wt2"], state["St2"], state["gs2"], y, part, ksplit, I, H // 16, maxb, False, mw2, nt, ks, nstage,
                     state["gs_mul"], bool(tma))""")
rep2("    for bs in (1, 8):\n        x, ids, tw = data[bs]\n        nt, mw2, ks, ns, ksp = _wt_variant(bs, state)",
     "    for bs in (1, 8):\n        x, ids, tw = data[bs]\n        nt, mw2, ks, ns, ksp, tma = _wt_variant(bs, state)")
rep2("""state["inter"], part, ksp, 4096, 512, maxb, True, 1, nt, ks, ns, 64.0),""", """state["inter"], part, ksp, 4096, 512, maxb, True, 1, nt, ks, ns, 64.0, bool(tma)),""")
rep2("""state["y"], part, ksp, 4096, 256, maxb, False, mw2, nt, ks, ns, 64.0),""", """state["y"], part, ksp, 4096, 256, maxb, False, mw2, nt, ks, ns, 64.0, bool(tma)),""")
rep2("""            print(f"  wt parts bs{bs} {(nt, mw2, ks, ns, ksp)}: \"""", """            print(f"  wt parts bs{bs} {(nt, mw2, ks, ns, ksp, tma)}: \"""")
rep2("    small = [(1, 1, 1, 8, 1), (1, 1, 1, 4, 2), (1, 1, 1, 4, 4), (1, 1, 1, 4, 8), (1, 1, 1, 2, 4), (1, 1, 1, 2, 8), (1, 2, 1, 4, 4), (1, 2, 1, 4, 8)]",
     "    small = [(1, 1, 1, 8, 1, 0), (1, 1, 1, 4, 1, 0), (1, 1, 1, 8, 1, 1), (1, 1, 1, 4, 1, 1), (1, 2, 1, 4, 1, 1), (1, 1, 1, 4, 4, 0)]")
rep2("    big = [(2, 2, 1, 4, 1), (2, 2, 1, 2, 1), (4, 2, 1, 4, 1), (4, 2, 1, 2, 1), (2, 1, 1, 2, 1), (4, 1, 1, 4, 1)]",
     "    big = [(2, 2, 1, 4, 1, 0), (4, 2, 1, 2, 1, 0), (2, 2, 1, 4, 1, 1), (4, 2, 1, 4, 1, 1), (1, 2, 1, 4, 1, 1), (2, 1, 1, 4, 1, 1)]")
rep2("WT_TEST_VARIANTS = ((1, 1, 1, 4, 1), (1, 1, 1, 8, 1), (2, 2, 1, 4, 1), (4, 2, 1, 2, 1), (1, 1, 1, 4, 2), (1, 1, 1, 4, 4), (1, 1, 1, 2, 8), (1, 2, 1, 4, 4))",
     "WT_TEST_VARIANTS = ((1, 1, 1, 4, 1, 0), (1, 1, 1, 8, 1, 0), (2, 2, 1, 4, 1, 0), (4, 2, 1, 2, 1, 0), (1, 1, 1, 4, 4, 0),\n                    (1, 1, 1, 4, 1, 1), (1, 1, 1, 8, 1, 1), (1, 2, 1, 4, 1, 1), (2, 2, 1, 4, 1, 1), (4, 2, 1, 4, 1, 1), (2, 1, 1, 4, 1, 1))")
q.write_text(t)

# ---- CPU check: grouped view
c = root / "tools/check_repack_wt.py"
v = c.read_text()
v = v.replace("words = Wr.view(E, NTL, NC, 2, 32, 4, 4).view(torch.int32).view(E, NTL, NC, 2, 32, 4)   # ..., half, lane, step4",
              "words = Wr.view(E, NTL // 4, NC, 4, 2, 32, 4, 4).permute(0, 1, 3, 2, 4, 5, 6, 7).reshape(E, NTL, NC, 2, 32, 16).contiguous().view(torch.int32).view(E, NTL, NC, 2, 32, 4)\nSr = Sr.view(E, NTL // 4, NC, 4, 8, 16).permute(0, 1, 3, 2, 4, 5).reshape(E, NTL, NC, 8, 16)")
c.write_text(v)
print("edited")
