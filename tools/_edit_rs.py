"""Round 3 edit: register-streaming WT variant (no smem staging of weights): each warp owns one m16 weight tile and
streams its K range with 16 B ld.global.nc loads into an unrolled register ring of U chunks; token tile via a small
cp.async ring; GATED pairs gate/up warps through smem for the fused SiLU; split-K via partial slices (existing reduce)."""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
p = root / "muse_moe_w4a16/cuda_moe.py"
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:100])
    s = s.replace(old, new)


kernel = r"""
// ---------------------------------------------------------------------------------------------
// Register-streaming variant for small batches: no shared-memory staging of weights. CTA = 4 warps, one m16 weight
// tile per warp (GATED: warps 0,1 gate tiles t0,t0+1 and warps 2,3 the matching up tiles; the up accumulators cross
// to the gate warps through smem for the fused SiLU). Each lane streams its tile's K range: per chunk 2 x 16 B of
// fragment-ordered nibbles + 16 B of scales, in an unrolled ring of U chunks in flight (ld.global.nc, L1 bypass).
// The 8*NT-token tile goes through an XST-stage cp.async ring (2 KB per chunk, L2-resident). SPLITK: gridDim.z CTAs
// split K and store fp32 partial slices for the reduce kernels. MODE 1: loads only (bench ablation).
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ uint4 ldg_stream(const uint8_t* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

template <bool GATED, int NT, int U, bool SPLITK, int MODE>
__global__ void __launch_bounds__(128)
moe_w4a16_rs_kernel(const __half* __restrict__ X, int ldx,
                    const int* __restrict__ sorted_tok,
                    const int* __restrict__ blk_e, const int* __restrict__ blk_row0,
                    const int* __restrict__ blk_end, const int* __restrict__ nblk,
                    const uint8_t* __restrict__ Wr, const uint8_t* __restrict__ Sr, const float* __restrict__ gs,
                    __half* __restrict__ out16, __nv_bfloat16* __restrict__ out_bf, int ldo,
                    float* __restrict__ part, int ldp, int part_rows,
                    int K, int NTILES, float gs_mul)
{
    constexpr int XST = 4;                            // token-tile stages
    constexpr int X_STAGE = 8 * NT * A_STRIDE * 2;
    constexpr int CH = NT == 1 ? 2 : 1;
    __shared__ __align__(16) uint8_t xs_all[XST * X_STAGE];
    __shared__ float xch[GATED ? 2 * 128 * NT * CH * 4 : 1];     // up accumulators of warps 2,3

    if (GATED) pdl_wait();
    const int j = blockIdx.y;
    if (j >= *nblk) return;
    const int e = blk_e[j];
    const int row0 = blk_row0[j];
    const int nrows = blk_end[j] - row0;
    const int ntb = (nrows + 7) >> 3;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KSPLIT = SPLITK ? (int)gridDim.z : 1;
    const int NC_ALL = K / KC;
    const int NC = NC_ALL / KSPLIT;
    const int c0 = SPLITK ? (int)blockIdx.z * NC : 0;
    const int NG = NTILES / 4;
    // this warp's tile: GATED: gate tiles 2*bx + (warp & 1), up tiles + NTILES/2 for warps 2,3; else 4*bx + warp
    int tile = GATED ? blockIdx.x * 2 + (warp & 1) : blockIdx.x * 4 + warp;
    if (GATED && warp >= 2) tile += NTILES / 2;
    const size_t ebase = (size_t)e * NG;
    const uint8_t* wbase = Wr + ((ebase + (tile >> 2)) * NC_ALL + c0) * 4096 + (tile & 3) * 1024 + lane * 16;
    const uint8_t* sbase = Sr + ((ebase + (tile >> 2)) * NC_ALL + c0) * 512 + (tile & 3) * 128 + g * 16;

    // token-tile ring (cp.async, all threads): segment sidx = tid (+128) -> row sidx>>4, 16 B piece sidx&15
    constexpr int XSEG = NT;
    const __half* xsrc[XSEG]; int xbytes[XSEG];
    #pragma unroll
    for (int i = 0; i < XSEG; ++i) {
        const int sidx = tid + i * 128;
        const int r = sidx >> 4;
        const bool valid = r < nrows;
        const int tok = valid ? (GATED ? sorted_tok[row0 + r] : row0 + r) : 0;
        xsrc[i] = X + (size_t)tok * ldx + c0 * KC + (sidx & 15) * 8;
        xbytes[i] = (valid && MODE != 4) ? 16 : 0;
    }
    const uint32_t xs_base = smem_u32(xs_all);
    auto issue_x = [&](int c, int stage) {
        #pragma unroll
        for (int i = 0; i < XSEG; ++i) {
            const int sidx = tid + i * 128;
            cp_async16(xs_base + stage * X_STAGE + ((sidx >> 4) * A_STRIDE + (sidx & 15) * 8) * 2, xsrc[i] + c * KC, xbytes[i]);
        }
    };
    if (!GATED) pdl_wait();                           // inter comes from GEMM1 (the weights above did not)
    #pragma unroll
    for (int st = 0; st < XST - 1; ++st) {
        if (st < NC) issue_x(st, st);
        cp_async_commit();
    }

    // weight register ring: U chunks in flight per lane
    uint4 wr[U][2]; uint4 sr[U];
    #pragma unroll
    for (int u = 0; u < U; ++u) {
        if (u < NC) {
            wr[u][0] = ldg_stream(wbase + (size_t)u * 4096);
            wr[u][1] = ldg_stream(wbase + (size_t)u * 4096 + 512);
            sr[u] = __ldg(reinterpret_cast<const uint4*>(sbase + (size_t)u * 512));
        }
    }

    float acc[NT * CH][4];
    #pragma unroll
    for (int n = 0; n < NT * CH; ++n)
        #pragma unroll
        for (int q = 0; q < 4; ++q) acc[n][q] = 0.f;
    const float gsf = gs[e] * gs_mul;

    for (int cb = 0; cb < NC; cb += U) {
        #pragma unroll
        for (int u = 0; u < U; ++u) {
            const int c = cb + u;
            if (c < NC) {
                const int stage = c % XST;
                cp_async_wait<XST - 2>();
                __syncthreads();
                {
                    const int cn = c + XST - 1;
                    if (cn < NC) issue_x(cn, cn % XST);
                    cp_async_commit();
                }
                const __half* xs = reinterpret_cast<const __half*>(xs_all + stage * X_STAGE);
                const uint4 w0 = wr[u][0], w1 = wr[u][1], sq = sr[u];
                // refill this ring slot with chunk c + U
                if (c + U < NC) {
                    wr[u][0] = ldg_stream(wbase + (size_t)(c + U) * 4096);
                    wr[u][1] = ldg_stream(wbase + (size_t)(c + U) * 4096 + 512);
                    sr[u] = __ldg(reinterpret_cast<const uint4*>(sbase + (size_t)(c + U) * 512));
                }
                if constexpr (MODE == 1) {
                    if (lane == 0) acc[0][0] += __uint_as_float((w0.x ^ w1.x ^ sq.x) & 0x007FFFFFu) * 1e-30f;
                    continue;
                }
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const uint4 wq = h == 0 ? w0 : w1;
                    const uint2 sq2 = h == 0 ? make_uint2(sq.x, sq.y) : make_uint2(sq.z, sq.w);
                    #pragma unroll
                    for (int ps = 0; ps < 2; ++ps) {
                        uint32_t b[NT][4];
                        #pragma unroll
                        for (int nb = 0; nb < NT; ++nb) {
                            if (nb < ntb)
                                ldmatrix_x4(b[nb][0], b[nb][1], b[nb][2], b[nb][3],
                                            smem_u32(xs + (nb * 8 + (lane & 7)) * A_STRIDE + h * 64 + ps * 32 + (lane >> 3) * 8));
                        }
                        #pragma unroll
                        for (int q = 0; q < 2; ++q) {
                            const int sidx = 2 * ps + q;
                            const uint32_t w = sidx == 0 ? wq.x : sidx == 1 ? wq.y : sidx == 2 ? wq.z : wq.w;
                            const uint32_t sw = ps == 0 ? sq2.x : sq2.y;
                            const uint32_t sp = q == 0 ? e4m3x2_lo16_to_f16x2(sw) : e4m3x2_hi16_to_f16x2(sw);
                            const uint32_t S0 = prmt(sp, sp, 0x1010u);
                            const uint32_t S1 = prmt(sp, sp, 0x3232u);
                            const uint32_t elo = w & 0x9C9C9C9Cu;
                            const uint32_t ehi = ((w << 2) & 0x8C8C8C8Cu) | ((w >> 2) & 0x10101010u);
                            uint32_t a[4];
                            a[0] = f16x2_mul(e4m3x2_lo16_to_f16x2(elo), S0);
                            a[1] = f16x2_mul(e4m3x2_hi16_to_f16x2(elo), S1);
                            a[2] = f16x2_mul(e4m3x2_lo16_to_f16x2(ehi), S0);
                            a[3] = f16x2_mul(e4m3x2_hi16_to_f16x2(ehi), S1);
                            #pragma unroll
                            for (int nb = 0; nb < NT; ++nb) {
                                if (nb < ntb) mma16816(acc[nb * CH + (CH == 2 ? q : 0)], a, b[nb][2 * q], b[nb][2 * q + 1]);
                            }
                        }
                    }
                }
            }
        }
    }
    pdl_trigger();
    cp_async_wait<0>();
    if (CH == 2) {
        #pragma unroll
        for (int q = 0; q < 4; ++q) acc[0][q] += acc[1][q];
    }

    // epilogue: C fragment rows g, g+8 of the tile -> output column; cols (tokens) 2t, 2t+1
    const int gtile = GATED ? blockIdx.x * 2 + (warp & 1) : tile;      // output column tile (gate tile for GATED)
    if constexpr (SPLITK) {
        #pragma unroll
        for (int nb = 0; nb < NT; ++nb)
            #pragma unroll
            for (int hr = 0; hr < 2; ++hr) {
                const int col = gtile * 16 + g + 8 * hr + (GATED && warp >= 2 ? NTILES * 8 : 0);   // up: + N/2
                #pragma unroll
                for (int cc = 0; cc < 2; ++cc) {
                    const int tok = nb * 8 + 2 * t + cc;
                    if (tok >= nrows) continue;
                    part[((size_t)blockIdx.z * part_rows + row0 + tok) * ldp + col] = acc[nb * CH][2 * hr + cc] * gsf;
                }
            }
        return;
    }
    if constexpr (GATED) {
        if (warp >= 2) {
            #pragma unroll
            for (int n = 0; n < NT; ++n)
                #pragma unroll
                for (int q = 0; q < 4; ++q) xch[(((warp - 2) * NT + n) * 4 + q) * 32 + lane] = acc[n * CH][q];
        }
        __syncthreads();
        if (warp >= 2) return;
    }
    #pragma unroll
    for (int nb = 0; nb < NT; ++nb) {
        #pragma unroll
        for (int hr = 0; hr < 2; ++hr) {
            const int col = gtile * 16 + g + 8 * hr;
            #pragma unroll
            for (int cc = 0; cc < 2; ++cc) {
                const int tok = nb * 8 + 2 * t + cc;
                if (tok >= nrows) continue;
                const float gv = acc[nb * CH][2 * hr + cc] * gsf;
                const size_t o = (size_t)(row0 + tok) * ldo + col;
                if constexpr (GATED) {
                    const float uv = xch[((warp * NT + nb) * 4 + 2 * hr + cc) * 32 + lane] * gsf;
                    const float hv = gv / (1.f + __expf(-gv)) * uv;
                    out16[o] = __float2half(__bfloat162float(__float2bfloat16(hv)));
                } else {
                    out_bf[o] = __float2bfloat16(gv);
                }
            }
        }
    }
}

template <bool GATED, int NT, int U, bool SPLITK, int MODE>
void launch_rs(const __half* X, int ldx, const int* st, const int* be, const int* br, const int* bend, const int* nb,
               const uint8_t* Wr, const uint8_t* Sr, const float* gs, __half* o16, __nv_bfloat16* obf, int ldo,
               float* part, int ldp, int part_rows, int ksplit, int K, int ntiles, int max_blocks, float gs_mul, bool pdl,
               cudaStream_t stream)
{
    const int n_x = GATED ? ntiles / 4 : ntiles / 4;   // GATED: 2 gate tiles (+2 up) per CTA over N/2 tiles; else 4 tiles
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(n_x, max_blocks, ksplit);
    cfg.blockDim = dim3(128);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = pdl ? 1 : 0;
    cudaError_t err = cudaLaunchKernelEx(&cfg, moe_w4a16_rs_kernel<GATED, NT, U, SPLITK, MODE>,
        X, ldx, st, be, br, bend, nb, Wr, Sr, gs, o16, obf, ldo, part, ldp, part_rows, K, ntiles, gs_mul);
    TORCH_CHECK(err == cudaSuccess, "rs launch failed: ", cudaGetErrorString(err));
}
"""
rep("// split-K reduce of GEMM1's fp32 partial slices", kernel + "\n// split-K reduce of GEMM1's fp32 partial slices")

wrapper = r"""
void moe_w4a16_rs(torch::Tensor X, torch::Tensor sorted_tok, torch::Tensor blk_e, torch::Tensor blk_row0,
                  torch::Tensor blk_end, torch::Tensor nblk, torch::Tensor Wr, torch::Tensor Sr, torch::Tensor gs,
                  torch::Tensor out, torch::Tensor part, int64_t ksplit, int64_t K, int64_t ntiles, int64_t max_blocks,
                  bool gated, int64_t nt, int64_t u, int64_t mode, double gs_mul, bool pdl)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    const __half* Xp = reinterpret_cast<const __half*>(X.data_ptr());
    const uint8_t* Wp = reinterpret_cast<const uint8_t*>(Wr.data_ptr());
    const uint8_t* Sp = reinterpret_cast<const uint8_t*>(Sr.data_ptr());
    __half* o16 = gated ? reinterpret_cast<__half*>(out.data_ptr()) : nullptr;
    __nv_bfloat16* obf = gated ? nullptr : reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    const int ldo = (int)out.stride(0);
    float* pp = ksplit > 1 ? part.data_ptr<float>() : nullptr;
    const int ldp = ksplit > 1 ? (int)part.stride(1) : 0;
    const int prow = ksplit > 1 ? (int)part.size(1) : 0;
#define OKC_R(G, T, UU, SK, MD) launch_rs<G, T, UU, SK, MD>(Xp, (int)X.stride(0), sorted_tok.data_ptr<int>(), blk_e.data_ptr<int>(), \
        blk_row0.data_ptr<int>(), blk_end.data_ptr<int>(), nblk.data_ptr<int>(), Wp, Sp, gs.data_ptr<float>(), o16, obf, ldo, \
        pp, ldp, prow, (int)ksplit, (int)K, (int)ntiles, (int)max_blocks, (float)gs_mul, pdl, stream)
    const int key = (gated ? 1000 : 0) + (ksplit > 1 ? 100 : 0) + (int)nt * 10 + (int)u / 4 + (int)mode * 10000;
    switch (key) {
        case 1011: OKC_R(true, 1, 4, false, 0); break;   case 1012: OKC_R(true, 1, 8, false, 0); break;
        case 1111: OKC_R(true, 1, 4, true, 0); break;    case 1112: OKC_R(true, 1, 8, true, 0); break;
        case 1021: OKC_R(true, 2, 4, false, 0); break;   case 1022: OKC_R(true, 2, 8, false, 0); break;
        case 11: OKC_R(false, 1, 4, false, 0); break;    case 12: OKC_R(false, 1, 8, false, 0); break;
        case 111: OKC_R(false, 1, 4, true, 0); break;    case 112: OKC_R(false, 1, 8, true, 0); break;
        case 21: OKC_R(false, 2, 4, false, 0); break;    case 22: OKC_R(false, 2, 8, false, 0); break;
        // loads-only ablation (U = 8)
        case 11012: OKC_R(true, 1, 8, false, 1); break;  case 10012: OKC_R(false, 1, 8, false, 1); break;
        case 11112: OKC_R(true, 1, 8, true, 1); break;   case 10112: OKC_R(false, 1, 8, true, 1); break;
        default: TORCH_CHECK(false, "unsupported rs variant (gated, ksplit, nt, u, mode)");
    }
#undef OKC_R
}
"""
rep("\nvoid moe_wt_silu_reduce(torch::Tensor part, int64_t ksplit, torch::Tensor out, torch::Tensor blk_row0, torch::Tensor blk_end,\n                        torch::Tensor nblk, int64_t rows_per_block, int64_t max_blocks, int64_t I)\n{",
    wrapper + "\nvoid moe_wt_silu_reduce(torch::Tensor part, int64_t ksplit, torch::Tensor out, torch::Tensor blk_row0, torch::Tensor blk_end,\n                        torch::Tensor nblk, int64_t rows_per_block, int64_t max_blocks, int64_t I)\n{")
rep("""void moe_wt_silu_reduce(torch::Tensor part, int64_t ksplit, torch::Tensor out, torch::Tensor blk_row0, torch::Tensor blk_end,
                        torch::Tensor nblk, int64_t rows_per_block, int64_t max_blocks, int64_t I);""",
    """void moe_w4a16_rs(torch::Tensor X, torch::Tensor sorted_tok, torch::Tensor blk_e, torch::Tensor blk_row0,
                  torch::Tensor blk_end, torch::Tensor nblk, torch::Tensor Wr, torch::Tensor Sr, torch::Tensor gs,
                  torch::Tensor out, torch::Tensor part, int64_t ksplit, int64_t K, int64_t ntiles, int64_t max_blocks,
                  bool gated, int64_t nt, int64_t u, int64_t mode, double gs_mul, bool pdl);
void moe_wt_silu_reduce(torch::Tensor part, int64_t ksplit, torch::Tensor out, torch::Tensor blk_row0, torch::Tensor blk_end,
                        torch::Tensor nblk, int64_t rows_per_block, int64_t max_blocks, int64_t I);""")
rep('functions=["moe_w4a16_gemm", "moe_w4a16_wt", "moe_wt_silu_reduce", "moe_finalize"]',
    'functions=["moe_w4a16_gemm", "moe_w4a16_wt", "moe_w4a16_rs", "moe_wt_silu_reduce", "moe_finalize"]')
p.write_text(s)

g = root / "tools/gen_moe_kernel_only.py"
u = g.read_text()
u = u.replace('inst.append("void* finalize_ref1', '''rsig = ("(const __half*, int, const int*, const int*, const int*, const int*, const int*, const uint8_t*, const uint8_t*, "
        "const float*, __half*, __nv_bfloat16*, int, float*, int, int, int, int, int, int, float, bool, cudaStream_t)")
for g, t, uu, sk, md in re.findall(r"OKC_R\\((true|false), (\\d), (\\d), (true|false), (\\d)\\)", cu):
    inst.append(f"template void launch_rs<{g}, {t}, {uu}, {sk}, {md}>{rsig};")
inst.append("void* finalize_ref1''')
g.write_text(u)

# ---- Python glue: rs variants encoded with ks >= 20 (ks = 20 + U); ablation harness in the bench
q = root / "muse_moe_w4a16/submission.py"
t = q.read_text()


def rep2(old, new, count=1):
    global t
    assert t.count(old) == count, (t.count(old), old[:100])
    t = t.replace(old, new)


rep2("""    part = state["part"]
    ext.moe_w4a16_wt(x16, state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                     state["Wt13"], state["St13"], state["gs13"], inter, part, ksplit, H, 2 * I // 16, maxb, True, 1, nt, ks,
                     nstage, state["gs_mul"], bool(tma), pdl)
    if ksplit > 1:
        ext.moe_wt_silu_reduce(part, ksplit, inter, state["blk_row0"], state["blk_end"], state["nblk"], BM, maxb, I)
    ext.moe_w4a16_wt(inter, state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                     state["Wt2"], state["St2"], state["gs2"], y, part, ksplit, I, H // 16, maxb, False, mw2, nt, ks, nstage,
                     state["gs_mul"], bool(tma), pdl)""",
"""    part = state["part"]
    if ks >= 20:   # register-streaming variant, U = ks - 20 chunks in flight per lane
        ext.moe_w4a16_rs(x16, state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                         state["Wt13"], state["St13"], state["gs13"], inter, part, ksplit, H, 2 * I // 16, maxb, True, nt, ks - 20, 0,
                         state["gs_mul"], pdl)
        if ksplit > 1:
            ext.moe_wt_silu_reduce(part, ksplit, inter, state["blk_row0"], state["blk_end"], state["nblk"], BM, maxb, I)
        ext.moe_w4a16_rs(inter, state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                         state["Wt2"], state["St2"], state["gs2"], y, part, ksplit, I, H // 16, maxb, False, nt, ks - 20, 0,
                         state["gs_mul"], pdl)
        ext.moe_finalize(y, part, ksplit, state["row_of_pair"], tw, state["norm_w"], out, M, POST_NORM_EPS, pdl)
        return out
    ext.moe_w4a16_wt(x16, state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                     state["Wt13"], state["St13"], state["gs13"], inter, part, ksplit, H, 2 * I // 16, maxb, True, 1, nt, ks,
                     nstage, state["gs_mul"], bool(tma), pdl)
    if ksplit > 1:
        ext.moe_wt_silu_reduce(part, ksplit, inter, state["blk_row0"], state["blk_end"], state["nblk"], BM, maxb, I)
    ext.moe_w4a16_wt(inter, state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                     state["Wt2"], state["St2"], state["gs2"], y, part, ksplit, I, H // 16, maxb, False, mw2, nt, ks, nstage,
                     state["gs_mul"], bool(tma), pdl)""")
# bench: rs ablation (loads-only) + rs full parts at bs1/bs8, candidates
rep2("""            # ablations of the (1,1,1,8) kernels: 1 loads only, 2 no dequant, 3 no MMA, 4 no token tile
            for mode in (1, 2, 3, 4):
                t1 = _timeit_cold(lambda: ext.moe_w4a16_wt(x16, state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                                  state["Wt13"], state["St13"], state["gs13"], state["inter"], part, 1, 4096, 512, maxb, True, 1, 1, 10 + mode, 8, 64.0, False, False), dev)
                t2 = _timeit_cold(lambda: ext.moe_w4a16_wt(state["inter"], state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                                  state["Wt2"], state["St2"], state["gs2"], state["y"], part, 1, 4096, 256, maxb, False, 1, 1, 10 + mode, 8, 64.0, False, False), dev)
                print(f"    ablation mode {mode} ({'loads only' if mode == 1 else 'no dequant' if mode == 2 else 'no mma' if mode == 3 else 'no x tile'}): gemm1 {t1:6.1f}  gemm2 {t2:6.1f}")""",
"""            # register-streaming kernels: loads-only (mode 1, U=8) and full (U = 4, 8) at K-splits 1, 2, 4
            for ksp2 in (1, 2, 4):
                for mode, uu in ((1, 8), (0, 4), (0, 8)):
                    t1 = _timeit_cold(lambda: ext.moe_w4a16_rs(x16, state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                                      state["Wt13"], state["St13"], state["gs13"], state["inter"], part, ksp2, 4096, 512, maxb, True, 1, uu, mode, 64.0, False), dev)
                    t2 = _timeit_cold(lambda: ext.moe_w4a16_rs(state["inter"], state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                                      state["Wt2"], state["St2"], state["gs2"], state["y"], part, ksp2, 4096, 256, maxb, False, 1, uu, mode, 64.0, False), dev)
                    print(f"    rs ksplit {ksp2} U {uu} {'loads only' if mode == 1 else 'full      '}: gemm1 {t1:6.1f}  gemm2 {t2:6.1f}")""")
rep2("""    small = [(1, 1, 1, 8, 1, 0), (1, 1, 1, 4, 1, 0), (1, 2, 1, 4, 1, 0), (1, 2, 1, 8, 1, 0)]""",
     """    small = [(1, 1, 1, 8, 1, 0), (1, 1, 24, 0, 1, 0), (1, 1, 28, 0, 1, 0), (1, 1, 24, 0, 2, 0), (1, 1, 28, 0, 2, 0), (1, 1, 24, 0, 4, 0), (1, 1, 28, 0, 4, 0),
             (2, 1, 24, 0, 1, 0), (2, 1, 28, 0, 1, 0)]""")
rep2("""    big = [(2, 2, 1, 4, 1, 0), (4, 2, 1, 2, 1, 0), (4, 2, 1, 4, 1, 0), (2, 2, 1, 8, 1, 0)]""",
     """    big = [(2, 2, 1, 4, 1, 0), (4, 2, 1, 2, 1, 0)]""")
rep2("""WT_TEST_VARIANTS = ((1, 1, 1, 4, 1, 0), (1, 1, 1, 8, 1, 0), (1, 2, 1, 4, 1, 0), (2, 2, 1, 4, 1, 0), (4, 2, 1, 2, 1, 0), (4, 2, 1, 4, 1, 0))""",
     """WT_TEST_VARIANTS = ((1, 1, 1, 4, 1, 0), (1, 1, 1, 8, 1, 0), (1, 2, 1, 4, 1, 0), (2, 2, 1, 4, 1, 0), (4, 2, 1, 2, 1, 0),
                    (1, 1, 24, 0, 1, 0), (1, 1, 28, 0, 1, 0), (1, 1, 28, 0, 2, 0), (1, 1, 24, 0, 4, 0), (2, 1, 28, 0, 1, 0), (2, 1, 24, 0, 1, 0))""")
rep2("BENCH = False      # microbenchmarks in prepare", "BENCH = True       # microbenchmarks in prepare")
q.write_text(t)
print("edited")
