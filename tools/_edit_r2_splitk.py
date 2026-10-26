"""Round 2 edit: clean inter-CTA split-K for small batches: SPLITK instantiations store fp32 partial slices only
(no fences/atomics), a one-row-per-CTA SiLU-reduce with all loads in flight, and a finalize that sums GEMM2's
slices (writing the bf16 y row) before the usual weighted sum."""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
p = root / "muse_moe_w4a16/cuda_moe.py"
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:100])
    s = s.replace(old, new)


# ---- kernel: drop counters, split-K = plain partial stores
rep("""                    float* __restrict__ part, int ldp, int part_rows, int* __restrict__ counters,
                    int K, int NTILES, float gs_mul)
{
    // gridDim.z > 1: split-K; CTA z owns chunks [z*NC/KSPLIT, (z+1)*NC/KSPLIT) and writes its fp32 partial
    // sums (times gs) to part[z][row][col] (GATED: gate at col, up at col + N/2); the last CTA to arrive at
    // counters[(y, x)] sums the slices and writes the output (the counter resets itself).""",
"""                    float* __restrict__ part, int ldp, int part_rows,
                    int K, int NTILES, float gs_mul)
{
    // SPLITK: gridDim.z CTAs split K; CTA z owns chunks [z*NC/KSPLIT, (z+1)*NC/KSPLIT) and stores its fp32
    // partial sums (times gs) to part[z][row][col] (GATED: gate at col, up at col + N/2) for the reduce kernels.""")
a = s.index("    if constexpr (SPLITK) {\n        __shared__ int s_last;")
b = s.index("    #pragma unroll\n    for (int i = 0; i < MW; ++i) {\n        const int tile = (blockIdx.x * WT_NWARPS + warp) * MW + i;\n        #pragma unroll\n        for (int nb = 0; nb < NT; ++nb) {\n            #pragma unroll\n            for (int hr = 0; hr < 2; ++hr) {\n                const int col = tile * 16 + g + 8 * hr;\n                #pragma unroll\n                for (int cc = 0; cc < 2; ++cc) {\n                    const int tok = nb * 8 + 2 * t + cc;\n                    if (tok >= nrows) continue;\n                    float gv")
new_split = """    if constexpr (SPLITK) {
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
"""
s = s[:a] + new_split + s[b:]
rep("""                    float gv = acc[i][nb * CH][2 * hr + cc] * gsf;
                    float uv = GATED ? acc[MW + i][nb * CH][2 * hr + cc] * gsf : 0.f;
                    if constexpr (SPLITK) {
                        for (int z = 0; z < KSPLIT; ++z) {
                            if (z == (int)blockIdx.z) continue;
                            const float* pp = part + ((size_t)z * part_rows + row0 + tok) * ldp + col;
                            gv += __ldcg(pp);
                            if (GATED) uv += __ldcg(pp + (size_t)NTILES * 8);
                        }
                    }
                    const size_t o""", """                    const float gv = acc[i][nb * CH][2 * hr + cc] * gsf;
                    const float uv = GATED ? acc[MW + i][nb * CH][2 * hr + cc] * gsf : 0.f;
                    const size_t o""")
rep("""               float* part, int ldp, int part_rows, int* counters, int ksplit,
               int K, int ntiles, int max_blocks, float gs_mul, cudaStream_t stream)""",
    """               float* part, int ldp, int part_rows, int ksplit,
               int K, int ntiles, int max_blocks, float gs_mul, cudaStream_t stream)""")
rep("        X, ldx, st, be, br, bend, nb, Wr, Sr, gs, o16, obf, ldo, part, ldp, part_rows, counters, K, ntiles, gs_mul);",
    "        X, ldx, st, be, br, bend, nb, Wr, Sr, gs, o16, obf, ldo, part, ldp, part_rows, K, ntiles, gs_mul);")

# ---- reduce kernels + finalize
a = s.index("// post-expert RMSNorm x routing weight, summed per token")
b = s.index("} // namespace")
new_tail = r"""// split-K reduce of GEMM1's fp32 partial slices + SiLU(gate) * up -> inter (bf16 values in an fp16 container).
// grid (rows per block, blocks); one row per CTA, 16 columns per thread, all slice loads in flight.
template <int KSPLIT>
__global__ void __launch_bounds__(256)
wt_silu_reduce_kernel(const float* __restrict__ part, int ldp, int part_rows, __half* __restrict__ out, int ldo,
                      const int* __restrict__ blk_row0, const int* __restrict__ blk_end, const int* __restrict__ nblk, int I)
{
    const int j = blockIdx.y;
    if (j >= *nblk) return;
    const int r = blk_row0[j] + blockIdx.x;
    if (r >= blk_end[j]) return;
    for (int c = threadIdx.x * 16; c < I; c += 256 * 16) {
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int cc = c + half * 8;
            float4 ga[KSPLIT][2], ua[KSPLIT][2];
            #pragma unroll
            for (int z = 0; z < KSPLIT; ++z) {
                const float* row = part + ((size_t)z * part_rows + r) * ldp;
                ga[z][0] = __ldcg(reinterpret_cast<const float4*>(row + cc));
                ga[z][1] = __ldcg(reinterpret_cast<const float4*>(row + cc + 4));
                ua[z][0] = __ldcg(reinterpret_cast<const float4*>(row + I + cc));
                ua[z][1] = __ldcg(reinterpret_cast<const float4*>(row + I + cc + 4));
            }
            float gsum[8], usum[8];
            #pragma unroll
            for (int q = 0; q < 8; ++q) { gsum[q] = 0.f; usum[q] = 0.f; }
            #pragma unroll
            for (int z = 0; z < KSPLIT; ++z) {
                gsum[0] += ga[z][0].x; gsum[1] += ga[z][0].y; gsum[2] += ga[z][0].z; gsum[3] += ga[z][0].w;
                gsum[4] += ga[z][1].x; gsum[5] += ga[z][1].y; gsum[6] += ga[z][1].z; gsum[7] += ga[z][1].w;
                usum[0] += ua[z][0].x; usum[1] += ua[z][0].y; usum[2] += ua[z][0].z; usum[3] += ua[z][0].w;
                usum[4] += ua[z][1].x; usum[5] += ua[z][1].y; usum[6] += ua[z][1].z; usum[7] += ua[z][1].w;
            }
            __half hv[8];
            #pragma unroll
            for (int q = 0; q < 8; ++q) {
                const float h = gsum[q] / (1.f + __expf(-gsum[q])) * usum[q];
                hv[q] = __float2half(__bfloat162float(__float2bfloat16(h)));
            }
            *reinterpret_cast<uint4*>(out + (size_t)r * ldo + cc) = *reinterpret_cast<const uint4*>(hv);
        }
    }
}

// post-expert RMSNorm x routing weight, summed per token: grid = tokens, 8 warps (one per route). Phase 1: each
// warp loads its route's row (all H/256 16B pieces per lane at once) and reduces the sum of squares; KSPLIT > 1:
// the row is the sum of GEMM2's fp32 partial slices, rounded to bf16 and written back to y. Phase 2: every
// thread re-reads 8 elements of all 8 rows (L2-hot) and writes the weighted sum. No smem tile.
template <int H, int KSPLIT>
__global__ void __launch_bounds__(256)
moe_finalize_kernel(__nv_bfloat16* __restrict__ y, int ldy, const float* __restrict__ part, int ldp, int part_rows,
                    const int* __restrict__ row_of_pair, const float* __restrict__ tw, const float* __restrict__ nw,
                    __nv_bfloat16* __restrict__ out, float eps)
{
    constexpr int NSEG = H / 8 / 32;                     // 16B pieces per lane
    constexpr int PP = KSPLIT <= 4 ? 4 : 2;              // pieces per pass in the split-K reduction
    __shared__ float sc_sh[8];
    __shared__ int row_sh[8];
    const int tok = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = row_of_pair[tok * 8 + warp];
    float ss = 0.f;
    if (r >= 0) {
        uint4 u[NSEG];
        if constexpr (KSPLIT == 1) {
            const uint4* src = reinterpret_cast<const uint4*>(y + (size_t)r * ldy);
            #pragma unroll
            for (int q = 0; q < NSEG; ++q) u[q] = src[lane + 32 * q];
        } else {
            #pragma unroll
            for (int p0 = 0; p0 < NSEG; p0 += PP) {
                float4 a[PP][KSPLIT][2];
                #pragma unroll
                for (int q = 0; q < PP; ++q)
                    #pragma unroll
                    for (int z = 0; z < KSPLIT; ++z) {
                        const float* row = part + ((size_t)z * part_rows + r) * ldp + (lane + 32 * (p0 + q)) * 8;
                        a[q][z][0] = __ldcg(reinterpret_cast<const float4*>(row));
                        a[q][z][1] = __ldcg(reinterpret_cast<const float4*>(row + 4));
                    }
                #pragma unroll
                for (int q = 0; q < PP; ++q) {
                    float v[8];
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) v[k] = 0.f;
                    #pragma unroll
                    for (int z = 0; z < KSPLIT; ++z) {
                        v[0] += a[q][z][0].x; v[1] += a[q][z][0].y; v[2] += a[q][z][0].z; v[3] += a[q][z][0].w;
                        v[4] += a[q][z][1].x; v[5] += a[q][z][1].y; v[6] += a[q][z][1].z; v[7] += a[q][z][1].w;
                    }
                    __nv_bfloat162 o[4];
                    o[0] = __floats2bfloat162_rn(v[0], v[1]); o[1] = __floats2bfloat162_rn(v[2], v[3]);
                    o[2] = __floats2bfloat162_rn(v[4], v[5]); o[3] = __floats2bfloat162_rn(v[6], v[7]);
                    u[p0 + q] = *reinterpret_cast<const uint4*>(o);
                    *reinterpret_cast<uint4*>(y + (size_t)r * ldy + (lane + 32 * (p0 + q)) * 8) = u[p0 + q];
                }
            }
        }
        #pragma unroll
        for (int q = 0; q < NSEG; ++q) {
            const int seg = lane + 32 * q;
            const float4 n0 = *reinterpret_cast<const float4*>(nw + seg * 8), n1 = *reinterpret_cast<const float4*>(nw + seg * 8 + 4);
            const __nv_bfloat162* pb = reinterpret_cast<const __nv_bfloat162*>(&u[q]);
            float2 f0 = __bfloat1622float2(pb[0]), f1 = __bfloat1622float2(pb[1]);
            float2 f2 = __bfloat1622float2(pb[2]), f3 = __bfloat1622float2(pb[3]);
            f0.x *= n0.x; f0.y *= n0.y; f1.x *= n0.z; f1.y *= n0.w;
            f2.x *= n1.x; f2.y *= n1.y; f3.x *= n1.z; f3.y *= n1.w;
            ss += f0.x * f0.x + f0.y * f0.y + f1.x * f1.x + f1.y * f1.y + f2.x * f2.x + f2.y * f2.y + f3.x * f3.x + f3.y * f3.y;
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    }
    if (lane == 0) {
        sc_sh[warp] = r >= 0 ? rsqrtf(ss / (float)H + eps) * tw[tok * 8 + warp] : 0.f;
        row_sh[warp] = r;
    }
    __syncthreads();                                     // (KSPLIT > 1: the y rows written above are this CTA's own)
    #pragma unroll
    for (int base = threadIdx.x * 8; base < H; base += 256 * 8) {
        uint4 v[8];
        #pragma unroll
        for (int s2 = 0; s2 < 8; ++s2) {
            const int rs = row_sh[s2];
            v[s2] = rs >= 0 ? *reinterpret_cast<const uint4*>(y + (size_t)rs * ldy + base) : make_uint4(0u, 0u, 0u, 0u);
        }
        float acc[8];
        #pragma unroll
        for (int q = 0; q < 8; ++q) acc[q] = 0.f;
        #pragma unroll
        for (int s2 = 0; s2 < 8; ++s2) {
            const float scs = sc_sh[s2];
            const __nv_bfloat162* pb = reinterpret_cast<const __nv_bfloat162*>(&v[s2]);
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float2 f = __bfloat1622float2(pb[q]);
                acc[2 * q] += scs * f.x;
                acc[2 * q + 1] += scs * f.y;
            }
        }
        const float4 n0 = *reinterpret_cast<const float4*>(nw + base), n1 = *reinterpret_cast<const float4*>(nw + base + 4);
        __nv_bfloat162 o[4];
        o[0] = __floats2bfloat162_rn(acc[0] * n0.x, acc[1] * n0.y); o[1] = __floats2bfloat162_rn(acc[2] * n0.z, acc[3] * n0.w);
        o[2] = __floats2bfloat162_rn(acc[4] * n1.x, acc[5] * n1.y); o[3] = __floats2bfloat162_rn(acc[6] * n1.z, acc[7] * n1.w);
        *reinterpret_cast<uint4*>(out + (size_t)tok * H + base) = *reinterpret_cast<const uint4*>(o);
    }
}

"""
s = s[:a] + new_tail + s[b:]

# ---- wrappers
rep("""    int* cnt = ksplit > 1 ? counters.data_ptr<int>() : nullptr;
    if (ksplit > 1) {
        const int n_x = (gated ? ntiles / 2 : ntiles) / (WT_NWARPS * mw);
        TORCH_CHECK(counters.numel() >= max_blocks * n_x, "split-K counters too small");
    }
""", "")
rep("        pp, ldp, prow, cnt, (int)ksplit, (int)K, (int)ntiles, (int)max_blocks, (float)gs_mul, stream)",
    "        pp, ldp, prow, (int)ksplit, (int)K, (int)ntiles, (int)max_blocks, (float)gs_mul, stream)")
rep("""                  torch::Tensor out, torch::Tensor part, torch::Tensor counters, int64_t ksplit, int64_t K, int64_t ntiles,
                  int64_t max_blocks, bool gated, int64_t mw, int64_t nt, int64_t ks, int64_t nstage, double gs_mul)""",
    """                  torch::Tensor out, torch::Tensor part, int64_t ksplit, int64_t K, int64_t ntiles,
                  int64_t max_blocks, bool gated, int64_t mw, int64_t nt, int64_t ks, int64_t nstage, double gs_mul)""", 2)
rep("""        // split-K (SPLITK = true) instantiations removed: measured slower than the plain variants on every shape
""", """        // split-K (SPLITK = true): partial-slice stores, for the small batches
        case 1111102: OKC_W(true, 1, 1, 1, 2, true); break;   case 1111104: OKC_W(true, 1, 1, 1, 4, true); break;
        case 1011102: OKC_W(false, 1, 1, 1, 2, true); break;  case 1011104: OKC_W(false, 1, 1, 1, 4, true); break;
        case 1021104: OKC_W(false, 2, 1, 1, 4, true); break;
""")
a = s.index("void moe_finalize(torch::Tensor y")
b = s.index('"""\n\nCPP_SRC = """')
s = s[:a] + r"""void moe_wt_silu_reduce(torch::Tensor part, int64_t ksplit, torch::Tensor out, torch::Tensor blk_row0, torch::Tensor blk_end,
                        torch::Tensor nblk, int64_t rows_per_block, int64_t max_blocks, int64_t I)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((unsigned)rows_per_block, (unsigned)max_blocks);
#define OKC_R(KS) wt_silu_reduce_kernel<KS><<<grid, 256, 0, stream>>>(part.data_ptr<float>(), (int)part.stride(1), (int)part.size(1), \
        reinterpret_cast<__half*>(out.data_ptr()), (int)out.stride(0), blk_row0.data_ptr<int>(), blk_end.data_ptr<int>(), \
        nblk.data_ptr<int>(), (int)I)
    switch (ksplit) {
        case 2: OKC_R(2); break;  case 4: OKC_R(4); break;  case 8: OKC_R(8); break;
        default: TORCH_CHECK(false, "unsupported ksplit for the SiLU reduce");
    }
#undef OKC_R
}

void moe_finalize(torch::Tensor y, torch::Tensor part, int64_t ksplit, torch::Tensor row_of_pair, torch::Tensor tw,
                  torch::Tensor nw, torch::Tensor out, int64_t M, double eps)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(out.size(1) == 4096, "finalize: H must be 4096");
    const float* pp = ksplit > 1 ? part.data_ptr<float>() : nullptr;
    const int ldp = ksplit > 1 ? (int)part.stride(1) : 0, prow = ksplit > 1 ? (int)part.size(1) : 0;
#define OKC_F(KS) moe_finalize_kernel<4096, KS><<<(unsigned)M, 256, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), \
        (int)y.stride(0), pp, ldp, prow, row_of_pair.data_ptr<int>(), tw.data_ptr<float>(), nw.data_ptr<float>(), \
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (float)eps)
    switch (ksplit) {
        case 1: OKC_F(1); break;  case 2: OKC_F(2); break;  case 4: OKC_F(4); break;  case 8: OKC_F(8); break;
        default: TORCH_CHECK(false, "unsupported ksplit for the finalize");
    }
#undef OKC_F
}
""" + s[b:]
# CPP declarations
a = s.index("void moe_finalize(torch::Tensor y", s.index('CPP_SRC = """'))
b = s.index('"""\n\n_EXT = None')
s = s[:a] + """void moe_wt_silu_reduce(torch::Tensor part, int64_t ksplit, torch::Tensor out, torch::Tensor blk_row0, torch::Tensor blk_end,
                        torch::Tensor nblk, int64_t rows_per_block, int64_t max_blocks, int64_t I);
void moe_finalize(torch::Tensor y, torch::Tensor part, int64_t ksplit, torch::Tensor row_of_pair, torch::Tensor tw,
                  torch::Tensor nw, torch::Tensor out, int64_t M, double eps);
""" + s[b:]
rep('functions=["moe_w4a16_gemm", "moe_w4a16_wt", "moe_finalize"]', 'functions=["moe_w4a16_gemm", "moe_w4a16_wt", "moe_wt_silu_reduce", "moe_finalize"]')
p.write_text(s)

g = root / "tools/gen_moe_kernel_only.py"
u = g.read_text()
u = u.replace('"const float*, __half*, __nv_bfloat16*, int, float*, int, int, int*, int, int, int, int, float, cudaStream_t)")',
              '"const float*, __half*, __nv_bfloat16*, int, float*, int, int, int, int, int, int, float, cudaStream_t)")')
u = u.replace('inst.append("void* finalize_ref = (void*)&moe_finalize_kernel<4096>;")',
              'inst.append("void* finalize_ref1 = (void*)&moe_finalize_kernel<4096, 1>;")\ninst.append("void* finalize_ref4 = (void*)&moe_finalize_kernel<4096, 4>;")\ninst.append("void* finalize_ref8 = (void*)&moe_finalize_kernel<4096, 8>;")\ninst.append("void* silu_ref = (void*)&wt_silu_reduce_kernel<4>;")')
g.write_text(u)

# ---- Python glue
q = root / "muse_moe_w4a16/submission.py"
t = q.read_text()


def rep2(old, new, count=1):
    global t
    assert t.count(old) == count, (t.count(old), old[:100])
    t = t.replace(old, new)


rep2("""    part = state["part"]
    counters = state["counters"]
    ext.moe_w4a16_wt(x16, state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                     state["Wt13"], state["St13"], state["gs13"], inter, part, counters, ksplit, H, 2 * I // 16, maxb, True, 1, nt, ks,
                     nstage, state["gs_mul"])
    ext.moe_w4a16_wt(inter, state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                     state["Wt2"], state["St2"], state["gs2"], y, part, counters, ksplit, I, H // 16, maxb, False, mw2, nt, ks, nstage,
                     state["gs_mul"])
    ext.moe_finalize(y, state["row_of_pair"], tw, state["norm_w"], out, M, POST_NORM_EPS)""",
"""    part = state["part"]
    ext.moe_w4a16_wt(x16, state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                     state["Wt13"], state["St13"], state["gs13"], inter, part, ksplit, H, 2 * I // 16, maxb, True, 1, nt, ks,
                     nstage, state["gs_mul"])
    if ksplit > 1:
        ext.moe_wt_silu_reduce(part, ksplit, inter, state["blk_row0"], state["blk_end"], state["nblk"], BM, maxb, I)
    ext.moe_w4a16_wt(inter, state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                     state["Wt2"], state["St2"], state["gs2"], y, part, ksplit, I, H // 16, maxb, False, mw2, nt, ks, nstage,
                     state["gs_mul"])
    ext.moe_finalize(y, part, ksplit, state["row_of_pair"], tw, state["norm_w"], out, M, POST_NORM_EPS)""")
rep2("""            state["counters"] = torch.zeros((maxb * 64,), dtype=torch.int32, device=dev)   # split-K arrival counters (self-resetting)
""", "")
rep2("PART_SPLIT_MAX = 4      # fp32 partial slices for split-K", "PART_SPLIT_MAX = 8      # fp32 partial slices for split-K")
# bench parts
rep2("""        counters = state["counters"]
        parts = {
            "route": lambda: _route(x, ids, 0, BM, state, marlin=False),
            "gemm1": lambda: ext.moe_w4a16_wt(x16, state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                                              state["Wt13"], state["St13"], state["gs13"], state["inter"], part, counters, ksp, 4096, 512, maxb, True, 1, nt, ks, ns, 64.0),
            "gemm2": lambda: ext.moe_w4a16_wt(state["inter"], state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                                              state["Wt2"], state["St2"], state["gs2"], state["y"], part, counters, ksp, 4096, 256, maxb, False, mw2, nt, ks, ns, 64.0),
            "finalize": lambda: ext.moe_finalize(state["y"], state["row_of_pair"], tw, state["norm_w"], out, bs, POST_NORM_EPS),
        }""",
"""        parts = {
            "route": lambda: _route(x, ids, 0, BM, state, marlin=False),
            "gemm1": lambda: ext.moe_w4a16_wt(x16, state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                                              state["Wt13"], state["St13"], state["gs13"], state["inter"], part, ksp, 4096, 512, maxb, True, 1, nt, ks, ns, 64.0),
            "silu_red": lambda: ext.moe_wt_silu_reduce(part, max(ksp, 2), state["inter"], state["blk_row0"], state["blk_end"], state["nblk"], BM, maxb, 4096),
            "gemm2": lambda: ext.moe_w4a16_wt(state["inter"], state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                                              state["Wt2"], state["St2"], state["gs2"], state["y"], part, ksp, 4096, 256, maxb, False, mw2, nt, ks, ns, 64.0),
            "finalize": lambda: ext.moe_finalize(state["y"], part, ksp, state["row_of_pair"], tw, state["norm_w"], out, bs, POST_NORM_EPS),
        }""")
rep2("    for bs in (1, 32):\n        x, ids, tw = data[bs]\n        nt, mw2, ks, ns, ksp = _wt_variant(bs, state)",
     "    for bs in (1, 8):\n        x, ids, tw = data[bs]\n        nt, mw2, ks, ns, ksp = _wt_variant(bs, state)")
rep2("    small = [(1, 1, 1, 2, 1), (1, 1, 1, 4, 1), (1, 1, 1, 8, 1), (1, 2, 1, 4, 1), (1, 2, 1, 8, 1)]",
     "    small = [(1, 1, 1, 8, 1), (1, 1, 1, 4, 2), (1, 1, 1, 4, 4), (1, 1, 1, 4, 8), (1, 1, 1, 2, 4), (1, 1, 1, 2, 8), (1, 2, 1, 4, 4), (1, 2, 1, 4, 8)]")
rep2("WT_TEST_VARIANTS = ((1, 1, 1, 4, 1), (1, 1, 1, 8, 1), (2, 2, 1, 4, 1), (4, 2, 1, 2, 1))",
     "WT_TEST_VARIANTS = ((1, 1, 1, 4, 1), (1, 1, 1, 8, 1), (2, 2, 1, 4, 1), (4, 2, 1, 2, 1), (1, 1, 1, 4, 2), (1, 1, 1, 4, 4), (1, 1, 1, 2, 8), (1, 2, 1, 4, 4))")
rep2("BENCH = False      # microbenchmarks in prepare", "BENCH = True       # microbenchmarks in prepare")
rep2("AUTOTUNE = False   # pick the wt variant per shape", "AUTOTUNE = True    # pick the wt variant per shape")
q.write_text(t)
print("edited")
