"""One-off edit: split-K partials, SiLU-reduce kernel, new finalize, NSTAGE 8 variants in cuda_moe.py."""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
p = root / "muse_moe_w4a16/cuda_moe.py"
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:80])
    s = s.replace(old, new)


# ---- kernel: split-K over gridDim.z with fp32 partial slices
rep("""                    __half* __restrict__ out16, __nv_bfloat16* __restrict__ out_bf, int ldo,
                    int K, int NTILES, float gs_mul)
{
    constexpr int TPW = (GATED ? 2 : 1) * MW;        // m16 weight tiles per warp""",
"""                    __half* __restrict__ out16, __nv_bfloat16* __restrict__ out_bf, int ldo,
                    float* __restrict__ part, int ldp, int part_rows,
                    int K, int NTILES, float gs_mul)
{
    // gridDim.z > 1: split-K; CTA z owns chunks [z*NC/KSPLIT, (z+1)*NC/KSPLIT) and writes its fp32 partial
    // sums (times gs) to part[z][row][col] (GATED: gate at col, up at col + N/2) for a reduce kernel.
    constexpr int TPW = (GATED ? 2 : 1) * MW;        // m16 weight tiles per warp""")
rep("""    const int g = lane >> 2, t = lane & 3;
    const int NC = K / KC;

    // CTA tile ti (0..TPC-1) -> global m16 tile index""",
"""    const int g = lane >> 2, t = lane & 3;
    const int KSPLIT = gridDim.z;
    const int NC_ALL = K / KC;
    const int NC = NC_ALL / KSPLIT;                   // chunks of this CTA
    const int c0 = blockIdx.z * NC;

    // CTA tile ti (0..TPC-1) -> global m16 tile index""")
rep("        wsrc[i] = Wr + (ebase + tile_of(sidx >> 6)) * NC * 1024 + (sidx & 63) * 16;",
    "        wsrc[i] = Wr + ((ebase + tile_of(sidx >> 6)) * NC_ALL + c0) * 1024 + (sidx & 63) * 16;")
rep("    const uint8_t* ssrc = Sr + (ebase + tile_of((tid & (TPC * 8 - 1)) >> 3)) * NC * 128 + (tid & 7) * 16;",
    "    const uint8_t* ssrc = Sr + ((ebase + tile_of((tid & (TPC * 8 - 1)) >> 3)) * NC_ALL + c0) * 128 + (tid & 7) * 16;")
rep("        xsrc[i] = X + (size_t)tok * ldx + (sidx & 15) * 8;", "        xsrc[i] = X + (size_t)tok * ldx + c0 * KC + (sidx & 15) * 8;")
rep("""                    const size_t o = (size_t)(row0 + tok) * ldo + col;
                    if (GATED) {""",
"""                    if (KSPLIT > 1) {
                        float* pp = part + ((size_t)blockIdx.z * part_rows + row0 + tok) * ldp + col;
                        pp[0] = acc[i][nb * CH][2 * hr + cc] * gsf;
                        if (GATED) pp[(size_t)NTILES * 8] = acc[MW + i][nb * CH][2 * hr + cc] * gsf;   // up: + N/2
                        continue;
                    }
                    const size_t o = (size_t)(row0 + tok) * ldo + col;
                    if (GATED) {""")

# ---- launch_wt: ksplit
rep("""               const uint8_t* Wr, const uint8_t* Sr, const float* gs, __half* o16, __nv_bfloat16* obf, int ldo,
               int K, int ntiles, int max_blocks, float gs_mul, cudaStream_t stream)
{""",
"""               const uint8_t* Wr, const uint8_t* Sr, const float* gs, __half* o16, __nv_bfloat16* obf, int ldo,
               float* part, int ldp, int part_rows, int ksplit,
               int K, int ntiles, int max_blocks, float gs_mul, cudaStream_t stream)
{""")
rep("""    const int n_x = (GATED ? ntiles / 2 : ntiles) / (WT_NWARPS * MW);
    dim3 grid(n_x, max_blocks);
    moe_w4a16_wt_kernel<GATED, MW, NT, KS, NSTAGE><<<grid, 128 * KS, smem, stream>>>(
        X, ldx, st, be, br, bend, nb, Wr, Sr, gs, o16, obf, ldo, K, ntiles, gs_mul);""",
"""    const int n_x = (GATED ? ntiles / 2 : ntiles) / (WT_NWARPS * MW);
    dim3 grid(n_x, max_blocks, ksplit);
    moe_w4a16_wt_kernel<GATED, MW, NT, KS, NSTAGE><<<grid, 128 * KS, smem, stream>>>(
        X, ldx, st, be, br, bend, nb, Wr, Sr, gs, o16, obf, ldo, part, ldp, part_rows, K, ntiles, gs_mul);""")

# ---- split-K reduce + SiLU (GEMM1) and the new finalize
rep("// post-expert RMSNorm x routing weight, summed per token: grid = tokens, 8 warps (one per route).",
"""// split-K reduce of GEMM1's fp32 partials + SiLU(gate) * up -> inter (bf16 values in fp16); grid (rows per block, blocks)
__global__ void __launch_bounds__(256)
wt_silu_reduce_kernel(const float* __restrict__ part, int ldp, int part_rows, int ksplit, __half* __restrict__ out, int ldo,
                      const int* __restrict__ blk_row0, const int* __restrict__ blk_end, const int* __restrict__ nblk, int I)
{
    const int j = blockIdx.y;
    if (j >= *nblk) return;
    const int r = blk_row0[j] + blockIdx.x;
    if (r >= blk_end[j]) return;
    for (int c = threadIdx.x * 4; c < I; c += 256 * 4) {
        float4 gg = make_float4(0.f, 0.f, 0.f, 0.f), uu = gg;
        for (int z = 0; z < ksplit; ++z) {
            const float* row = part + ((size_t)z * part_rows + r) * ldp;
            const float4 a = *reinterpret_cast<const float4*>(row + c), b = *reinterpret_cast<const float4*>(row + I + c);
            gg.x += a.x; gg.y += a.y; gg.z += a.z; gg.w += a.w;
            uu.x += b.x; uu.y += b.y; uu.z += b.z; uu.w += b.w;
        }
        const float h0 = gg.x / (1.f + __expf(-gg.x)) * uu.x, h1 = gg.y / (1.f + __expf(-gg.y)) * uu.y;
        const float h2 = gg.z / (1.f + __expf(-gg.z)) * uu.z, h3 = gg.w / (1.f + __expf(-gg.w)) * uu.w;
        __half2 v01 = __halves2half2(__float2half(__bfloat162float(__float2bfloat16(h0))), __float2half(__bfloat162float(__float2bfloat16(h1))));
        __half2 v23 = __halves2half2(__float2half(__bfloat162float(__float2bfloat16(h2))), __float2half(__bfloat162float(__float2bfloat16(h3))));
        uint2 pk;
        pk.x = *reinterpret_cast<uint32_t*>(&v01); pk.y = *reinterpret_cast<uint32_t*>(&v23);
        *reinterpret_cast<uint2*>(out + (size_t)r * ldo + c) = pk;
    }
}

// post-expert RMSNorm x routing weight, summed per token: grid = tokens, 8 warps (one per route). y is bf16
// (ksplit == 1) or the sum of ksplit fp32 partial slices rounded to bf16 (as SGLang keeps y in bf16).""")
a = s.index("__global__ void __launch_bounds__(256)\nmoe_finalize_kernel(")
b = s.index("template <bool GATED, int MW, int NT, int KS, int NSTAGE>\nvoid launch_wt(")
new_fin = r"""__global__ void __launch_bounds__(256)
moe_finalize_kernel(const __nv_bfloat16* __restrict__ y, int ldy, const float* __restrict__ part, int ldp, int part_rows,
                    int ksplit, const int* __restrict__ row_of_pair, const float* __restrict__ tw,
                    const float* __restrict__ nw, __nv_bfloat16* __restrict__ out, int H, float eps)
{
    extern __shared__ __align__(16) float red[];              // [8][H]
    __shared__ float sc_sh[8];
    const int tok = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = row_of_pair[tok * 8 + warp];
    float* myrow = red + warp * H;
    // pass 1: the route's row (bf16 values), scaled by (1 + w), into smem; sum of squares
    float ss = 0.f;
    for (int seg = lane; seg < H / 8; seg += 32) {
        float v[8];
        if (r < 0) {
            #pragma unroll
            for (int q = 0; q < 8; ++q) v[q] = 0.f;
        } else if (ksplit == 1) {
            const uint4 u = *reinterpret_cast<const uint4*>(y + (size_t)r * ldy + seg * 8);
            const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&u);
            #pragma unroll
            for (int q = 0; q < 4; ++q) { const float2 f = __bfloat1622float2(p[q]); v[2 * q] = f.x; v[2 * q + 1] = f.y; }
        } else {
            #pragma unroll
            for (int q = 0; q < 8; ++q) v[q] = 0.f;
            for (int z = 0; z < ksplit; ++z) {
                const float* row = part + ((size_t)z * part_rows + r) * ldp + seg * 8;
                const float4 a = *reinterpret_cast<const float4*>(row), b = *reinterpret_cast<const float4*>(row + 4);
                v[0] += a.x; v[1] += a.y; v[2] += a.z; v[3] += a.w; v[4] += b.x; v[5] += b.y; v[6] += b.z; v[7] += b.w;
            }
            #pragma unroll
            for (int q = 0; q < 8; ++q) v[q] = __bfloat162float(__float2bfloat16(v[q]));
        }
        const float4 n0 = *reinterpret_cast<const float4*>(nw + seg * 8), n1 = *reinterpret_cast<const float4*>(nw + seg * 8 + 4);
        v[0] *= n0.x; v[1] *= n0.y; v[2] *= n0.z; v[3] *= n0.w; v[4] *= n1.x; v[5] *= n1.y; v[6] *= n1.z; v[7] *= n1.w;
        #pragma unroll
        for (int q = 0; q < 8; ++q) ss += v[q] * v[q];
        *reinterpret_cast<float4*>(myrow + seg * 8) = make_float4(v[0], v[1], v[2], v[3]);
        *reinterpret_cast<float4*>(myrow + seg * 8 + 4) = make_float4(v[4], v[5], v[6], v[7]);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if (lane == 0) sc_sh[warp] = r >= 0 ? rsqrtf(ss / (float)H + eps) * tw[tok * 8 + warp] : 0.f;
    __syncthreads();
    // pass 2: weighted sum over the 8 routes
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

# ---- torch wrappers
a = s.index("void moe_w4a16_wt(torch::Tensor X")
b = s.index('"""\n\nCPP_SRC = """')
new_wrap = r"""void moe_w4a16_wt(torch::Tensor X, torch::Tensor sorted_tok, torch::Tensor blk_e, torch::Tensor blk_row0,
                  torch::Tensor blk_end, torch::Tensor nblk, torch::Tensor Wr, torch::Tensor Sr, torch::Tensor gs,
                  torch::Tensor out, torch::Tensor part, int64_t ksplit, int64_t K, int64_t ntiles, int64_t max_blocks, bool gated,
                  int64_t mw, int64_t nt, int64_t ks, int64_t nstage, double gs_mul)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    const __half* Xp = reinterpret_cast<const __half*>(X.data_ptr());
    const uint8_t* Wp = reinterpret_cast<const uint8_t*>(Wr.data_ptr());
    const uint8_t* Sp = reinterpret_cast<const uint8_t*>(Sr.data_ptr());
    __half* o16 = gated ? reinterpret_cast<__half*>(out.data_ptr()) : nullptr;
    __nv_bfloat16* obf = gated ? nullptr : reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    const int ldo = (int)out.stride(0);
    TORCH_CHECK(ksplit == 1 || part.dim() == 3, "split-K needs a [ksplit, rows, cols] fp32 buffer");
    float* pp = ksplit > 1 ? part.data_ptr<float>() : nullptr;
    const int ldp = ksplit > 1 ? (int)part.stride(1) : 0;
    const int prow = ksplit > 1 ? (int)part.size(1) : 0;
#define OKC_W(G, M, T, S, NS) launch_wt<G, M, T, S, NS>(Xp, (int)X.stride(0), sorted_tok.data_ptr<int>(), blk_e.data_ptr<int>(), \
        blk_row0.data_ptr<int>(), blk_end.data_ptr<int>(), nblk.data_ptr<int>(), Wp, Sp, gs.data_ptr<float>(), o16, obf, ldo, \
        pp, ldp, prow, (int)ksplit, (int)K, (int)ntiles, (int)max_blocks, (float)gs_mul, stream)
    const int key = (gated ? 100000 : 0) + (int)mw * 10000 + (int)nt * 1000 + (int)ks * 100 + (int)nstage;
    switch (key) {
        case 111104: OKC_W(true, 1, 1, 1, 4); break;   case 111108: OKC_W(true, 1, 1, 1, 8); break;
        case 112104: OKC_W(true, 1, 2, 1, 4); break;   case 112108: OKC_W(true, 1, 2, 1, 8); break;
        case 114104: OKC_W(true, 1, 4, 1, 4); break;
        case 11104: OKC_W(false, 1, 1, 1, 4); break;   case 11108: OKC_W(false, 1, 1, 1, 8); break;
        case 21104: OKC_W(false, 2, 1, 1, 4); break;   case 21108: OKC_W(false, 2, 1, 1, 8); break;
        case 12104: OKC_W(false, 1, 2, 1, 4); break;   case 22104: OKC_W(false, 2, 2, 1, 4); break;
        case 22108: OKC_W(false, 2, 2, 1, 8); break;   case 14104: OKC_W(false, 1, 4, 1, 4); break;
        case 24104: OKC_W(false, 2, 4, 1, 4); break;
        default: TORCH_CHECK(false, "unsupported wt variant (mw, nt, ks, nstage)");
    }
#undef OKC_W
}

void moe_wt_silu_reduce(torch::Tensor part, int64_t ksplit, torch::Tensor out, torch::Tensor blk_row0, torch::Tensor blk_end,
                        torch::Tensor nblk, int64_t rows_per_block, int64_t max_blocks, int64_t I)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((unsigned)rows_per_block, (unsigned)max_blocks);
    wt_silu_reduce_kernel<<<grid, 256, 0, stream>>>(part.data_ptr<float>(), (int)part.stride(1), (int)part.size(1), (int)ksplit,
        reinterpret_cast<__half*>(out.data_ptr()), (int)out.stride(0), blk_row0.data_ptr<int>(), blk_end.data_ptr<int>(),
        nblk.data_ptr<int>(), (int)I);
}

void moe_finalize(torch::Tensor y, torch::Tensor part, int64_t ksplit, torch::Tensor row_of_pair, torch::Tensor tw,
                  torch::Tensor nw, torch::Tensor out, int64_t M, double eps)
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
    const float* pp = ksplit > 1 ? part.data_ptr<float>() : nullptr;
    moe_finalize_kernel<<<(unsigned)M, 256, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(y.data_ptr()), (int)y.stride(0), pp, ksplit > 1 ? (int)part.stride(1) : 0,
        ksplit > 1 ? (int)part.size(1) : 0, (int)ksplit, row_of_pair.data_ptr<int>(),
        tw.data_ptr<float>(), nw.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), H, (float)eps);
}
"""
s = s[:a] + new_wrap + s[b:]
a = s.index("void moe_w4a16_wt(torch::Tensor X", s.index('CPP_SRC = """'))
b = s.index('"""\n\n_EXT = None')
new_decl = """void moe_w4a16_wt(torch::Tensor X, torch::Tensor sorted_tok, torch::Tensor blk_e, torch::Tensor blk_row0,
                  torch::Tensor blk_end, torch::Tensor nblk, torch::Tensor Wr, torch::Tensor Sr, torch::Tensor gs,
                  torch::Tensor out, torch::Tensor part, int64_t ksplit, int64_t K, int64_t ntiles, int64_t max_blocks, bool gated,
                  int64_t mw, int64_t nt, int64_t ks, int64_t nstage, double gs_mul);
void moe_wt_silu_reduce(torch::Tensor part, int64_t ksplit, torch::Tensor out, torch::Tensor blk_row0, torch::Tensor blk_end,
                        torch::Tensor nblk, int64_t rows_per_block, int64_t max_blocks, int64_t I);
void moe_finalize(torch::Tensor y, torch::Tensor part, int64_t ksplit, torch::Tensor row_of_pair, torch::Tensor tw,
                  torch::Tensor nw, torch::Tensor out, int64_t M, double eps);
"""
s = s[:a] + new_decl + s[b:]
rep('functions=["moe_w4a16_gemm", "moe_w4a16_wt", "moe_finalize"]',
    'functions=["moe_w4a16_gemm", "moe_w4a16_wt", "moe_wt_silu_reduce", "moe_finalize"]')
p.write_text(s)
print("cuda_moe.py edited")
