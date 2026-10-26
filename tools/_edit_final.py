"""One-off edit: finalize v3 (no dynamic smem, 16 loads in flight), route MULTI constexpr, locked policy, bench off,
split-K instantiations pruned (kept as code; measured slower on every shape)."""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
p = root / "muse_moe_w4a16/cuda_moe.py"
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:100])
    s = s.replace(old, new)


# ---- finalize v3
a = s.index("// post-expert RMSNorm x routing weight, summed per token")
b = s.index("} // namespace")
new_fin = r"""// post-expert RMSNorm x routing weight, summed per token: grid = tokens, 8 warps (one per route). Phase 1: each
// warp loads its route's row (all H/256 16B pieces per lane at once) and reduces the sum of squares; phase 2:
// every thread re-reads 8 elements of all 8 rows (L2-hot) and writes the weighted sum. No shared-memory tile,
// so the SM keeps its carveout.
template <int H>
__global__ void __launch_bounds__(256)
moe_finalize_kernel(const __nv_bfloat16* __restrict__ y, int ldy, const int* __restrict__ row_of_pair,
                    const float* __restrict__ tw, const float* __restrict__ nw, __nv_bfloat16* __restrict__ out, float eps)
{
    constexpr int NSEG = H / 8 / 32;                     // 16B pieces per lane
    __shared__ float sc_sh[8];
    __shared__ int row_sh[8];
    const int tok = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = row_of_pair[tok * 8 + warp];
    float ss = 0.f;
    if (r >= 0) {
        const uint4* src = reinterpret_cast<const uint4*>(y + (size_t)r * ldy);
        uint4 u[NSEG];
        #pragma unroll
        for (int q = 0; q < NSEG; ++q) u[q] = src[lane + 32 * q];
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
    __syncthreads();
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
s = s[:a] + new_fin + s[b:]
a = s.index("void moe_finalize(torch::Tensor y")
b = s.index('"""\n\nCPP_SRC = """')
s = s[:a] + r"""void moe_finalize(torch::Tensor y, torch::Tensor row_of_pair, torch::Tensor tw, torch::Tensor nw, torch::Tensor out,
                  int64_t M, double eps)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(out.size(1) == 4096, "finalize: H must be 4096");
    moe_finalize_kernel<4096><<<(unsigned)M, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(y.data_ptr()), (int)y.stride(0), row_of_pair.data_ptr<int>(),
        tw.data_ptr<float>(), nw.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (float)eps);
}
""" + s[b:]

# ---- prune split-K instantiations (code kept behind the SPLITK template)
a = s.index("        // split-K instantiations (the candidate small-batch variants)")
b = s.index('        default: TORCH_CHECK(false, "unsupported wt variant (mw, nt, ks, nstage, ksplit)");')
s = s[:a] + """        // split-K (SPLITK = true) instantiations removed: measured slower than the plain variants on every shape
""" + s[b:]
p.write_text(s)

g = root / "tools/gen_moe_kernel_only.py"
u = g.read_text()
u = u.replace('inst.append("void* finalize_ref = (void*)&moe_finalize_kernel;")', 'inst.append("void* finalize_ref = (void*)&moe_finalize_kernel<4096>;")')
g.write_text(u)

# ---- submission.py
q = root / "muse_moe_w4a16/submission.py"
t = q.read_text()


def rep2(old, new, count=1):
    global t
    assert t.count(old) == count, (t.count(old), old[:100])
    t = t.replace(old, new)


rep2("""    TOPK: tl.constexpr, XBLOCK: tl.constexpr, MARLIN: tl.constexpr,
):""", """    TOPK: tl.constexpr, XBLOCK: tl.constexpr, MARLIN: tl.constexpr, MULTI: tl.constexpr,
):""")
rep2("""        cnt = tl.sum((ce_key[None, :] == ex[:, None]).to(tl.int32), 1)     # per-expert pairs in this chunk
        counts += cnt
        if c0 + CH <= my0:                                  # chunk entirely before this CTA's pairs
            rank += tl.sum(onehot * cnt[None, :], 1)
        else:
            if c0 <= my0:                                   # the chunk holding this CTA's pairs: its earlier pairs
                rank += tl.sum(((ce_key[None, :] == e_key[:, None]) & (ci[None, :] < my0)).to(tl.int32), 1)
    # earlier pairs of the same expert within this CTA's own range
    rank += tl.sum(((e_key[None, :] == e_key[:, None]) & (i[None, :] < i[:, None])).to(tl.int32), 1)""",
"""        cnt = tl.sum((ce_key[None, :] == ex[:, None]).to(tl.int32), 1)     # per-expert pairs in this chunk
        counts += cnt
        if MULTI:
            if c0 + CH <= my0:                              # chunk entirely before this CTA's pairs
                rank += tl.sum(onehot * cnt[None, :], 1)
            else:
                if c0 <= my0:                               # the chunk holding this CTA's pairs: its earlier pairs
                    rank += tl.sum(((ce_key[None, :] == e_key[:, None]) & (ci[None, :] < my0)).to(tl.int32), 1)
        else:                                               # one chunk: direct earlier-pair compare
            rank += tl.sum(((ce_key[None, :] == e_key[:, None]) & (ci[None, :] < i[:, None])).to(tl.int32), 1)
    if MULTI:   # earlier pairs of the same expert within this CTA's own range
        rank += tl.sum(((e_key[None, :] == e_key[:, None]) & (i[None, :] < i[:, None])).to(tl.int32), 1)""")
rep2("""        PB=PB, CH=CH, BM=BM, MAXB=MAXB, E=LOCAL, TOPK=TOPK, XBLOCK=XBLOCK, MARLIN=marlin, num_warps=4,""",
     """        PB=PB, CH=CH, BM=BM, MAXB=MAXB, E=LOCAL, TOPK=TOPK, XBLOCK=XBLOCK, MARLIN=marlin, MULTI=NP > CH, num_warps=4,""")
# policy
rep2("WT_DEFAULT = {1: (1, 1, 1, 8, 2), 8: (1, 1, 1, 8, 1), 16: (1, 1, 1, 4, 1), 32: (1, 1, 1, 4, 1), 128: (2, 2, 1, 4, 1), 256: (4, 2, 1, 4, 1)}",
     """# Locked from the in-prepare cold-L2 sweeps (checks f2f756d9, a3d79f15): 8 cp.async stages up to 8 tokens, 4 above;
# 8-token blocks up to 64 tokens, 16-token (GEMM2: 2 tiles per warp) up to 128, 32-token with 2 stages beyond.
WT_DEFAULT = {8: (1, 1, 1, 8, 1), 64: (1, 1, 1, 4, 1), 128: (2, 2, 1, 4, 1), 100000: (4, 2, 1, 2, 1)}""")
rep2("""WT_TEST_VARIANTS = ((1, 1, 1, 4, 1), (1, 1, 1, 2, 1), (1, 1, 1, 8, 1), (1, 1, 1, 8, 2), (1, 1, 1, 8, 4), (1, 1, 1, 4, 4), (1, 1, 1, 2, 2),
                    (1, 2, 1, 8, 2), (1, 2, 1, 4, 2), (1, 2, 1, 2, 1), (2, 2, 1, 4, 1), (2, 2, 1, 2, 1),
                    (4, 2, 1, 4, 1), (4, 2, 1, 2, 1), (2, 1, 1, 2, 1), (4, 1, 1, 4, 1))""",
     """WT_TEST_VARIANTS = ((1, 1, 1, 4, 1), (1, 1, 1, 8, 1), (2, 2, 1, 4, 1), (4, 2, 1, 2, 1))""")
rep2("    small = [(1, 1, 1, 2, 1), (1, 1, 1, 4, 1), (1, 1, 1, 8, 1), (1, 1, 1, 8, 2), (1, 1, 1, 8, 4), (1, 1, 1, 4, 4), (1, 2, 1, 8, 2), (1, 2, 1, 4, 2), (1, 2, 1, 2, 1), (1, 1, 1, 2, 2)]",
     "    small = [(1, 1, 1, 2, 1), (1, 1, 1, 4, 1), (1, 1, 1, 8, 1), (1, 2, 1, 4, 1), (1, 2, 1, 8, 1)]")
rep2("BENCH = True       # microbenchmarks in prepare (untimed; results go to the run log)",
     "BENCH = False      # microbenchmarks in prepare (untimed; results go to the run log)")
rep2("AUTOTUNE = True    # pick the wt variant per shape", "AUTOTUNE = False   # pick the wt variant per shape")
q.write_text(t)
print("edited")
