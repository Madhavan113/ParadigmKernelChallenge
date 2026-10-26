"""One-off edit: in-kernel split-K reduction by the last CTA (self-resetting counters), finalize with grouped loads,
drop the separate SiLU-reduce kernel and the finalize's fp32 partial path."""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
p = root / "muse_moe_w4a16/cuda_moe.py"
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:100])
    s = s.replace(old, new)


# ---- kernel signature: counters
rep("""                    float* __restrict__ part, int ldp, int part_rows,
                    int K, int NTILES, float gs_mul)
{
    // gridDim.z > 1: split-K; CTA z owns chunks [z*NC/KSPLIT, (z+1)*NC/KSPLIT) and writes its fp32 partial
    // sums (times gs) to part[z][row][col] (GATED: gate at col, up at col + N/2) for a reduce kernel.""",
"""                    float* __restrict__ part, int ldp, int part_rows, int* __restrict__ counters,
                    int K, int NTILES, float gs_mul)
{
    // gridDim.z > 1: split-K; CTA z owns chunks [z*NC/KSPLIT, (z+1)*NC/KSPLIT) and writes its fp32 partial
    // sums (times gs) to part[z][row][col] (GATED: gate at col, up at col + N/2); the last CTA to arrive at
    // counters[(y, x)] sums the slices and writes the output (the counter resets itself).""")

# ---- epilogue rewrite
a = s.index("    // C fragment: rows (weight rows) g, g+8 of the tile -> output column; cols (tokens) 2t, 2t+1")
b = s.index("template <bool GATED, int MW, int NT, int KS, int NSTAGE>\nvoid launch_wt(")
new_epi = r"""    // C fragment: rows (weight rows) g, g+8 of the tile -> output column; cols (tokens) 2t, 2t+1
    __shared__ int s_last;
    if (KSPLIT > 1) {
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
        __threadfence();
        __syncthreads();
        if (tid == 0) {
            const int old = atomicAdd(counters + blockIdx.y * gridDim.x + blockIdx.x, 1);
            s_last = (old == KSPLIT - 1);
            if (old == KSPLIT - 1) counters[blockIdx.y * gridDim.x + blockIdx.x] = 0;
        }
        __syncthreads();
        if (!s_last) return;
        __threadfence();
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
                    float gv = acc[i][nb * CH][2 * hr + cc] * gsf;
                    float uv = GATED ? acc[MW + i][nb * CH][2 * hr + cc] * gsf : 0.f;
                    if (KSPLIT > 1) {
                        for (int z = 0; z < KSPLIT; ++z) {
                            if (z == (int)blockIdx.z) continue;
                            const float* pp = part + ((size_t)z * part_rows + row0 + tok) * ldp + col;
                            gv += __ldcg(pp);
                            if (GATED) uv += __ldcg(pp + (size_t)NTILES * 8);
                        }
                    }
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
s = s[:a] + new_epi + s[b:]

# ---- drop the SiLU-reduce kernel
a = s.index("// split-K reduce of GEMM1's fp32 partials + SiLU(gate) * up -> inter")
b = s.index("// post-expert RMSNorm x routing weight, summed per token")
s = s[:a] + s[b:]

# ---- finalize: bf16 only, grouped loads
a = s.index("// post-expert RMSNorm x routing weight, summed per token")
b = s.index("} // namespace")
new_fin = r"""// post-expert RMSNorm x routing weight, summed per token: grid = tokens, 8 warps (one per route); dynamic smem
// 8 x H floats. Rows are loaded in groups of 8 x 16B per lane so the loads overlap.
__global__ void __launch_bounds__(256)
moe_finalize_kernel(const __nv_bfloat16* __restrict__ y, int ldy, const int* __restrict__ row_of_pair,
                    const float* __restrict__ tw, const float* __restrict__ nw, __nv_bfloat16* __restrict__ out,
                    int H, float eps)
{
    extern __shared__ __align__(16) float red[];              // [8][H]
    __shared__ float sc_sh[8];
    constexpr int GS = 8;
    const int tok = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = row_of_pair[tok * 8 + warp];
    const uint4* src = r >= 0 ? reinterpret_cast<const uint4*>(y + (size_t)r * ldy) : nullptr;
    float* myrow = red + warp * H;
    float ss = 0.f;
    for (int seg0 = 0; seg0 < H / 8; seg0 += 32 * GS) {
        uint4 u[GS];
        #pragma unroll
        for (int q = 0; q < GS; ++q) {
            const int seg = seg0 + lane + 32 * q;
            u[q] = (src != nullptr && seg < H / 8) ? src[seg] : make_uint4(0u, 0u, 0u, 0u);
        }
        #pragma unroll
        for (int q = 0; q < GS; ++q) {
            const int seg = seg0 + lane + 32 * q;
            if (seg < H / 8) {
                const __nv_bfloat162* pb = reinterpret_cast<const __nv_bfloat162*>(&u[q]);
                const float4 n0 = *reinterpret_cast<const float4*>(nw + seg * 8), n1 = *reinterpret_cast<const float4*>(nw + seg * 8 + 4);
                float v[8];
                #pragma unroll
                for (int k = 0; k < 4; ++k) { const float2 f = __bfloat1622float2(pb[k]); v[2 * k] = f.x; v[2 * k + 1] = f.y; }
                v[0] *= n0.x; v[1] *= n0.y; v[2] *= n0.z; v[3] *= n0.w; v[4] *= n1.x; v[5] *= n1.y; v[6] *= n1.z; v[7] *= n1.w;
                #pragma unroll
                for (int k = 0; k < 8; ++k) ss += v[k] * v[k];
                *reinterpret_cast<float4*>(myrow + seg * 8) = make_float4(v[0], v[1], v[2], v[3]);
                *reinterpret_cast<float4*>(myrow + seg * 8 + 4) = make_float4(v[4], v[5], v[6], v[7]);
            }
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if (lane == 0) sc_sh[warp] = r >= 0 ? rsqrtf(ss / (float)H + eps) * tw[tok * 8 + warp] : 0.f;
    __syncthreads();
    for (int base = threadIdx.x * 8; base < H; base += 256 * 8) {
        float acc[8];
        #pragma unroll
        for (int q = 0; q < 8; ++q) acc[q] = 0.f;
        #pragma unroll
        for (int s2 = 0; s2 < 8; ++s2) {
            const float scs = sc_sh[s2];
            const float4 a = *reinterpret_cast<const float4*>(red + s2 * H + base), b = *reinterpret_cast<const float4*>(red + s2 * H + base + 4);
            acc[0] += scs * a.x; acc[1] += scs * a.y; acc[2] += scs * a.z; acc[3] += scs * a.w;
            acc[4] += scs * b.x; acc[5] += scs * b.y; acc[6] += scs * b.z; acc[7] += scs * b.w;
        }
        __nv_bfloat162 o[4];
        o[0] = __floats2bfloat162_rn(acc[0], acc[1]); o[1] = __floats2bfloat162_rn(acc[2], acc[3]);
        o[2] = __floats2bfloat162_rn(acc[4], acc[5]); o[3] = __floats2bfloat162_rn(acc[6], acc[7]);
        *reinterpret_cast<uint4*>(out + (size_t)tok * H + base) = *reinterpret_cast<const uint4*>(o);
    }
}

"""
s = s[:a] + new_fin + s[b:]

# ---- launch_wt: counters
rep("""               float* part, int ldp, int part_rows, int ksplit,
               int K, int ntiles, int max_blocks, float gs_mul, cudaStream_t stream)""",
"""               float* part, int ldp, int part_rows, int* counters, int ksplit,
               int K, int ntiles, int max_blocks, float gs_mul, cudaStream_t stream)""")
rep("        X, ldx, st, be, br, bend, nb, Wr, Sr, gs, o16, obf, ldo, part, ldp, part_rows, K, ntiles, gs_mul);",
    "        X, ldx, st, be, br, bend, nb, Wr, Sr, gs, o16, obf, ldo, part, ldp, part_rows, counters, K, ntiles, gs_mul);")

# ---- wrappers
rep("""                  torch::Tensor out, torch::Tensor part, int64_t ksplit, int64_t K, int64_t ntiles, int64_t max_blocks, bool gated,
                  int64_t mw, int64_t nt, int64_t ks, int64_t nstage, double gs_mul)
{
    auto stream = at::cuda::getCurrentCUDAStream();""",
"""                  torch::Tensor out, torch::Tensor part, torch::Tensor counters, int64_t ksplit, int64_t K, int64_t ntiles,
                  int64_t max_blocks, bool gated, int64_t mw, int64_t nt, int64_t ks, int64_t nstage, double gs_mul)
{
    auto stream = at::cuda::getCurrentCUDAStream();""")
rep("""    const int prow = ksplit > 1 ? (int)part.size(1) : 0;
#define OKC_W(G, M, T, S, NS)""",
"""    const int prow = ksplit > 1 ? (int)part.size(1) : 0;
    int* cnt = ksplit > 1 ? counters.data_ptr<int>() : nullptr;
    if (ksplit > 1) {
        const int n_x = (gated ? ntiles / 2 : ntiles) / (WT_NWARPS * mw);
        TORCH_CHECK(counters.numel() >= max_blocks * n_x, "split-K counters too small");
    }
#define OKC_W(G, M, T, S, NS)""")
rep("        pp, ldp, prow, (int)ksplit, (int)K, (int)ntiles, (int)max_blocks, (float)gs_mul, stream)",
    "        pp, ldp, prow, cnt, (int)ksplit, (int)K, (int)ntiles, (int)max_blocks, (float)gs_mul, stream)")
a = s.index("void moe_wt_silu_reduce(torch::Tensor part")
b = s.index("void moe_finalize(torch::Tensor y")
s = s[:a] + s[b:]
a = s.index("void moe_finalize(torch::Tensor y")
b = s.index('"""\n\nCPP_SRC = """')
s = s[:a] + r"""void moe_finalize(torch::Tensor y, torch::Tensor row_of_pair, torch::Tensor tw, torch::Tensor nw, torch::Tensor out,
                  int64_t M, double eps)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    const int H = (int)out.size(1);
    TORCH_CHECK(H % 2048 == 0, "finalize: H must be a multiple of 2048");
    const int smem = 8 * H * (int)sizeof(float);
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(moe_finalize_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        attr_set = true;
    }
    moe_finalize_kernel<<<(unsigned)M, 256, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(y.data_ptr()), (int)y.stride(0), row_of_pair.data_ptr<int>(),
        tw.data_ptr<float>(), nw.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), H, (float)eps);
}
""" + s[b:]
# CPP declarations
a = s.index("void moe_w4a16_wt(torch::Tensor X", s.index('CPP_SRC = """'))
b = s.index('"""\n\n_EXT = None')
s = s[:a] + """void moe_w4a16_wt(torch::Tensor X, torch::Tensor sorted_tok, torch::Tensor blk_e, torch::Tensor blk_row0,
                  torch::Tensor blk_end, torch::Tensor nblk, torch::Tensor Wr, torch::Tensor Sr, torch::Tensor gs,
                  torch::Tensor out, torch::Tensor part, torch::Tensor counters, int64_t ksplit, int64_t K, int64_t ntiles,
                  int64_t max_blocks, bool gated, int64_t mw, int64_t nt, int64_t ks, int64_t nstage, double gs_mul);
void moe_finalize(torch::Tensor y, torch::Tensor row_of_pair, torch::Tensor tw, torch::Tensor nw, torch::Tensor out,
                  int64_t M, double eps);
""" + s[b:]
rep('functions=["moe_w4a16_gemm", "moe_w4a16_wt", "moe_wt_silu_reduce", "moe_finalize"]',
    'functions=["moe_w4a16_gemm", "moe_w4a16_wt", "moe_finalize"]')
p.write_text(s)

# ---- Python glue
q = root / "muse_moe_w4a16/submission.py"
t = q.read_text()


def rep2(old, new, count=1):
    global t
    assert t.count(old) == count, (t.count(old), old[:100])
    t = t.replace(old, new)


rep2("""    part = state["part"]
    ext.moe_w4a16_wt(x16, state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                     state["Wt13"], state["St13"], state["gs13"], inter, part, ksplit, H, 2 * I // 16, maxb, True, 1, nt, ks, nstage,
                     state["gs_mul"])
    if ksplit > 1:
        ext.moe_wt_silu_reduce(part, ksplit, inter, state["blk_row0"], state["blk_end"], state["nblk"], BM, maxb, I)
    ext.moe_w4a16_wt(inter, state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                     state["Wt2"], state["St2"], state["gs2"], y, part, ksplit, I, H // 16, maxb, False, mw2, nt, ks, nstage,
                     state["gs_mul"])
    ext.moe_finalize(y, part, ksplit, state["row_of_pair"], tw, state["norm_w"], out, M, POST_NORM_EPS)""",
"""    part = state["part"]
    counters = state["counters"]
    ext.moe_w4a16_wt(x16, state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                     state["Wt13"], state["St13"], state["gs13"], inter, part, counters, ksplit, H, 2 * I // 16, maxb, True, 1, nt, ks,
                     nstage, state["gs_mul"])
    ext.moe_w4a16_wt(inter, state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                     state["Wt2"], state["St2"], state["gs2"], y, part, counters, ksplit, I, H // 16, maxb, False, mw2, nt, ks, nstage,
                     state["gs_mul"])
    ext.moe_finalize(y, state["row_of_pair"], tw, state["norm_w"], out, M, POST_NORM_EPS)""")
rep2("""            state["part"] = torch.empty((PART_SPLIT_MAX, PART_ROWS, 2 * I), dtype=torch.float32, device=dev)   # split-K partials
""", """            state["part"] = torch.empty((PART_SPLIT_MAX, PART_ROWS, 2 * I), dtype=torch.float32, device=dev)   # split-K partials
            state["counters"] = torch.zeros((maxb * 64,), dtype=torch.int32, device=dev)   # split-K arrival counters (self-resetting)
""")
# bench parts
rep2("""        parts = {
            "route": lambda: _route(x, ids, 0, BM, state, marlin=False),
            "gemm1": lambda: ext.moe_w4a16_wt(x16, state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                                              state["Wt13"], state["St13"], state["gs13"], state["inter"], part, ksp, 4096, 512, maxb, True, 1, nt, ks, ns, 64.0),
            "silu_red": lambda: ext.moe_wt_silu_reduce(part, max(ksp, 2), state["inter"], state["blk_row0"], state["blk_end"], state["nblk"], BM, maxb, 4096),
            "gemm2": lambda: ext.moe_w4a16_wt(state["inter"], state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                                              state["Wt2"], state["St2"], state["gs2"], state["y"], part, ksp, 4096, 256, maxb, False, mw2, nt, ks, ns, 64.0),
            "finalize": lambda: ext.moe_finalize(state["y"], part, ksp, state["row_of_pair"], tw, state["norm_w"], out, bs, POST_NORM_EPS),
        }""",
"""        counters = state["counters"]
        parts = {
            "route": lambda: _route(x, ids, 0, BM, state, marlin=False),
            "gemm1": lambda: ext.moe_w4a16_wt(x16, state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                                              state["Wt13"], state["St13"], state["gs13"], state["inter"], part, counters, ksp, 4096, 512, maxb, True, 1, nt, ks, ns, 64.0),
            "gemm2": lambda: ext.moe_w4a16_wt(state["inter"], state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                                              state["Wt2"], state["St2"], state["gs2"], state["y"], part, counters, ksp, 4096, 256, maxb, False, mw2, nt, ks, ns, 64.0),
            "finalize": lambda: ext.moe_finalize(state["y"], state["row_of_pair"], tw, state["norm_w"], out, bs, POST_NORM_EPS),
        }""")
rep2("    small = [(1, 1, 1, 4, 1), (1, 1, 1, 8, 1), (1, 2, 1, 8, 1), (1, 1, 1, 8, 2), (1, 2, 1, 8, 2), (1, 1, 1, 4, 4), (1, 2, 1, 4, 4), (1, 2, 1, 4, 2)]\n    big = [(2, 2, 1, 4, 1), (2, 2, 1, 8, 1), (2, 1, 1, 8, 1), (4, 2, 1, 4, 1), (1, 2, 1, 8, 1)]",
     "    small = [(1, 1, 1, 4, 1), (1, 1, 1, 8, 1), (1, 1, 1, 8, 2), (1, 1, 1, 8, 4), (1, 1, 1, 4, 4), (1, 2, 1, 8, 2), (1, 2, 1, 4, 2), (1, 2, 1, 4, 4)]\n    big = [(2, 2, 1, 4, 1), (4, 2, 1, 4, 1), (2, 2, 1, 4, 2), (2, 1, 1, 4, 1), (4, 1, 1, 4, 1)]")
rep2("""WT_TEST_VARIANTS = ((1, 1, 1, 4, 1), (1, 1, 1, 8, 1), (1, 2, 1, 8, 1), (1, 1, 1, 8, 2), (1, 1, 1, 4, 4), (1, 2, 1, 4, 2),
                    (2, 2, 1, 8, 1), (2, 1, 1, 8, 1), (4, 2, 1, 4, 1), (1, 2, 1, 8, 1))""",
     """WT_TEST_VARIANTS = ((1, 1, 1, 4, 1), (1, 1, 1, 8, 1), (1, 1, 1, 8, 2), (1, 1, 1, 8, 4), (1, 1, 1, 4, 4), (1, 2, 1, 8, 2),
                    (1, 2, 1, 4, 2), (1, 2, 1, 4, 4), (2, 2, 1, 4, 1), (2, 2, 1, 4, 2), (4, 2, 1, 4, 1), (2, 1, 1, 4, 1), (4, 1, 1, 4, 1))""")
rep2("WT_DEFAULT = {1: (1, 1, 1, 8, 2), 8: (1, 2, 1, 8, 1), 16: (1, 1, 1, 8, 1), 32: (1, 2, 1, 8, 1), 128: (2, 2, 1, 8, 1), 256: (2, 2, 1, 8, 1)}",
     "WT_DEFAULT = {1: (1, 1, 1, 8, 2), 8: (1, 1, 1, 8, 1), 16: (1, 1, 1, 4, 1), 32: (1, 1, 1, 4, 1), 128: (2, 2, 1, 4, 1), 256: (4, 2, 1, 4, 1)}")
q.write_text(t)

g = root / "tools/gen_moe_kernel_only.py"
u = g.read_text()
u = u.replace('"const float*, __half*, __nv_bfloat16*, int, float*, int, int, int, int, int, int, float, cudaStream_t)")',
              '"const float*, __half*, __nv_bfloat16*, int, float*, int, int, int*, int, int, int, int, float, cudaStream_t)")')
u = u.replace('inst.append("void* silu_ref = (void*)&wt_silu_reduce_kernel;")\n', '')
g.write_text(u)
print("edited")
