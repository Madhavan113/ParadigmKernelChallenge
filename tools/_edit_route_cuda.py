"""One-off edit: add the CUDA routing kernel (moe_route) to cuda_moe.py and switch the wt path to it in submission.py."""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
p = root / "muse_moe_w4a16/cuda_moe.py"
s = p.read_text()


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:80])
    s = s.replace(old, new)


kernel = r'''
// Routing for the wt path, one launch: CTA 0 ranks the (token, slot) pairs by local expert (deterministic: pair
// order within an expert) and writes the padded rows and the block list; the other CTAs convert x to fp16 (exact
// for BF16 values in fp16 range, as the Triton route kernel did). Up to MAXCH * 1024 pairs.
template <int TOPK, int E, int MAXCH>
__global__ void __launch_bounds__(1024)
moe_route_kernel(const int* __restrict__ ids, const __nv_bfloat16* __restrict__ x, __half* __restrict__ x16,
                 int* __restrict__ sorted_tok, int* __restrict__ row_of_pair, int* __restrict__ blk_e,
                 int* __restrict__ blk_row0, int* __restrict__ blk_end, int* __restrict__ nblk,
                 int M, int local_start, int BM, int n_elems, float xscale)
{
    const int tid = threadIdx.x;
    if (blockIdx.x > 0) {                                        // x -> fp16, 8 elements per thread
        const int i = ((int)(blockIdx.x - 1) * 1024 + tid) * 8;
        if (i < n_elems) {
            const uint4 v = *reinterpret_cast<const uint4*>(x + i);
            const __nv_bfloat162* pb = reinterpret_cast<const __nv_bfloat162*>(&v);
            uint4 o;
            __half2* po = reinterpret_cast<__half2*>(&o);
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                float2 f = __bfloat1622float2(pb[q]);
                f.x *= xscale; f.y *= xscale;
                po[q] = __float22half2_rn(f);
            }
            *reinterpret_cast<uint4*>(x16 + i) = o;
        }
        return;
    }
    constexpr int NW = 32;
    __shared__ int wc[NW][E];                                    // this chunk's pairs per (warp, expert)
    __shared__ int wp[NW][E];                                    // exclusive prefix over the warps
    __shared__ int base[E];                                      // pairs of the earlier chunks
    __shared__ int pad_off[E], blk_off[E], nb_s[E], cnt_s[E];
    __shared__ int total_s;
    const int warp = tid >> 5, lane = tid & 31;
    const int npairs = M * TOPK;
    const int nch = (npairs + 1023) >> 10;
    if (tid < E) base[tid] = 0;
    int rank[MAXCH], ex[MAXCH];
    #pragma unroll
    for (int ch = 0; ch < MAXCH; ++ch) {
        rank[ch] = 0; ex[ch] = E;
        if (ch < nch) {
            const int i = (ch << 10) + tid;
            int e = E;
            if (i < npairs) {
                const int v = ids[i] - local_start;
                if (v >= 0 && v < E) e = v;
            }
            ex[ch] = e;
            if (lane < E) wc[warp][lane] = 0;
            __syncthreads();
            const unsigned m = __match_any_sync(0xffffffffu, e);
            rank[ch] = __popc(m & ((1u << lane) - 1u));
            if (e < E && rank[ch] == 0) wc[warp][e] = __popc(m);
            __syncthreads();
            if (tid < E) {                                       // thread e: prefix over warps
                int run = base[tid];
                #pragma unroll
                for (int w = 0; w < NW; ++w) { wp[w][tid] = run; run += wc[w][tid]; }
                base[tid] = run;
            }
            __syncthreads();
            if (e < E) rank[ch] += wp[warp][e];
        }
    }
    if (warp == 0) {                                             // lane e: counts -> padded rows, block offsets
        const int cnt = base[lane];
        const int padded = (cnt + BM - 1) / BM * BM;
        const int nb = padded / BM;
        int po = padded, bo = nb;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int a = __shfl_up_sync(0xffffffffu, po, o), b = __shfl_up_sync(0xffffffffu, bo, o);
            if (lane >= o) { po += a; bo += b; }
        }
        cnt_s[lane] = cnt; pad_off[lane] = po - padded; blk_off[lane] = bo - nb; nb_s[lane] = nb;
        if (lane == 31) total_s = bo;
    }
    __syncthreads();
    #pragma unroll
    for (int ch = 0; ch < MAXCH; ++ch) {
        if (ch < nch) {
            const int i = (ch << 10) + tid;
            if (i < npairs) {
                const int e = ex[ch];
                if (e < E) {
                    const int row = pad_off[e] + rank[ch];
                    sorted_tok[row] = i / TOPK;
                    row_of_pair[i] = row;
                } else {
                    row_of_pair[i] = -1;
                }
            }
        }
    }
    const int total = total_s;
    for (int j = tid; j < total; j += 1024) {
        int ej = 0;
        #pragma unroll
        for (int e = 1; e < E; ++e) ej += (blk_off[e] <= j);     // blk_off is non-decreasing: the expert of block j
        blk_e[j] = ej + local_start;
        blk_row0[j] = pad_off[ej] + (j - blk_off[ej]) * BM;
        blk_end[j] = pad_off[ej] + cnt_s[ej];
    }
    if (tid == 0) nblk[0] = total;
}

} // namespace
'''
rep("\n} // namespace\n", kernel)
host = r'''
void moe_route(torch::Tensor ids, torch::Tensor x, torch::Tensor x16, torch::Tensor sorted_tok, torch::Tensor row_of_pair,
               torch::Tensor blk_e, torch::Tensor blk_row0, torch::Tensor blk_end, torch::Tensor nblk,
               int64_t M, int64_t local_start, int64_t BM, double xscale)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    const int H = (int)x.size(1);
    const int n_elems = (int)M * H;
    TORCH_CHECK(M * 8 <= 8 * 1024, "route: too many pairs");
    TORCH_CHECK(H % 8 == 0, "route: H must be a multiple of 8");
    const unsigned grid = 1u + (unsigned)((n_elems + 8191) / 8192);
    moe_route_kernel<8, 32, 8><<<grid, 1024, 0, stream>>>(
        ids.data_ptr<int>(), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<__half*>(x16.data_ptr()),
        sorted_tok.data_ptr<int>(), row_of_pair.data_ptr<int>(), blk_e.data_ptr<int>(), blk_row0.data_ptr<int>(),
        blk_end.data_ptr<int>(), nblk.data_ptr<int>(), (int)M, (int)local_start, (int)BM, n_elems, (float)xscale);
}
"""
'''
rep('''        tw.data_ptr<float>(), nw.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (float)eps);
}
"""
''', '''        tw.data_ptr<float>(), nw.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (float)eps);
}
''' + host)
rep('''void moe_finalize(torch::Tensor y, torch::Tensor row_of_pair, torch::Tensor tw, torch::Tensor nw, torch::Tensor out,
                  int64_t M, double eps);
"""''', '''void moe_finalize(torch::Tensor y, torch::Tensor row_of_pair, torch::Tensor tw, torch::Tensor nw, torch::Tensor out,
                  int64_t M, double eps);
void moe_route(torch::Tensor ids, torch::Tensor x, torch::Tensor x16, torch::Tensor sorted_tok, torch::Tensor row_of_pair,
               torch::Tensor blk_e, torch::Tensor blk_row0, torch::Tensor blk_end, torch::Tensor nblk,
               int64_t M, int64_t local_start, int64_t BM, double xscale);
"""''')
rep('functions=["moe_w4a16_gemm", "moe_w4a16_wt", "moe_finalize"]', 'functions=["moe_w4a16_gemm", "moe_w4a16_wt", "moe_finalize", "moe_route"]')
p.write_text(s)
print("cuda route written")

p = root / "muse_moe_w4a16/submission.py"
s = p.read_text()
rep("""    NP, MAXB = _route(x, ids, local_start, BM, state, marlin=False)
    maxb = min(MAXB, _wt_max_blocks(M, BM))
    x16 = state["x16"][:M]
    out = torch.empty((M, H), dtype=MOE_OUT_DTYPE, device=x.device)
    ext = build()
""", """    ext = build()
    NP, MAXB = _route_wt(x, ids, local_start, BM, state, ext)
    maxb = min(MAXB, _wt_max_blocks(M, BM))
    x16 = state["x16"][:M]
    out = torch.empty((M, H), dtype=MOE_OUT_DTYPE, device=x.device)
""")
rep("""def _wt_max_blocks(M: int, BM: int) -> int:""", """ROUTE_CUDA = True   # the one-launch CUDA routing kernel (ranks + block list + fp16 copy of x) instead of the Triton one


def _route_wt(x, ids, local_start: int, BM: int, state, ext):
    M, H = x.shape
    if not ROUTE_CUDA:
        return _route(x, ids, local_start, BM, state, marlin=False)
    NP = max(16, triton.next_power_of_2(M * TOPK))
    MAXB = triton.next_power_of_2(NP // BM + LOCAL)
    ext.moe_route(ids, x.contiguous(), state["x16"], state["sorted_tok"], state["row_of_pair"], state["blk_e"], state["blk_row0"],
                  state["blk_end"], state["nblk"], M, local_start, BM, float(state.get("xscale", 1.0)))
    return NP, MAXB


def _wt_max_blocks(M: int, BM: int) -> int:""")
rep("""        NP, MAXB = _route(x, ids, 0, BM, state, marlin=False)
        maxb = min(MAXB, _wt_max_blocks(bs, BM))
        ext = build()
        x16 = state["x16"][:bs]
        out = torch.empty((bs, 4096), dtype=MOE_OUT_DTYPE, device=dev)
        part = state["part"]
        counters = state["counters"]
        parts = {
            "route": lambda: _route(x, ids, 0, BM, state, marlin=False),""",
    """        ext = build()
        NP, MAXB = _route_wt(x, ids, 0, BM, state, ext)
        maxb = min(MAXB, _wt_max_blocks(bs, BM))
        x16 = state["x16"][:bs]
        out = torch.empty((bs, 4096), dtype=MOE_OUT_DTYPE, device=dev)
        part = state["part"]
        counters = state["counters"]
        parts = {
            "route": lambda: _route_wt(x, ids, 0, BM, state, ext),
            "route_triton": lambda: _route(x, ids, 0, BM, state, marlin=False),""")
rep("""        NP, MAXB = _route(x, ids, 0, BM, state, marlin=False)
        maxb = min(MAXB, _wt_max_blocks(bs, BM))
        x16 = state["x16"][:bs]
        row = []""", """        NP, MAXB = _route_wt(x, ids, 0, BM, state, ext)
        maxb = min(MAXB, _wt_max_blocks(bs, BM))
        x16 = state["x16"][:bs]
        row = []""")
p.write_text(s)
print("submission patched")
