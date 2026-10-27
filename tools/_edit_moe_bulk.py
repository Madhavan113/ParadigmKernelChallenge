"""Phase 3 step 2: BULK weight path in the wt kernel (one thread issues two cp.async.bulk copies per chunk: the CTA's
T weight tiles (T KB) and their scales (T x 128 B), from a layout regrouped in the CTA's tile order; mbarrier per
ring slot; x stays cp.async), host `bulk` arg + instantiations, grouped repack, 6-element policy tuple (bulk flag),
self-test variants and bench rows. Run: python3 tools/_edit_moe_bulk.py && python3 tools/embed_cuda.py"""
import pathlib, re
root = pathlib.Path(__file__).resolve().parent.parent
cp = root / "muse_moe_w4a16/cuda_moe.py"; s = cp.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


if "bool BULK" not in s:
    # 1. the bulk / mbarrier helpers must precede the wt kernel: move them out of the bstream part to just after cp_async_wait
    helpers_a = s.index("__device__ __forceinline__ void mbar_init(")
    helpers_b = s.index("template <int NST, int SB>\n__global__ void __launch_bounds__(128) bstream_ranges_kernel")
    helpers = s[helpers_a:helpers_b]
    s = s[:helpers_a] + s[helpers_b:]
    anchor = "template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile(\"cp.async.wait_group %0;\\n\" :: \"n\"(N)); }\n"
    assert s.count(anchor) == 1
    s = s.replace(anchor, anchor + helpers)
    # 2. kernel template + barrier ring + bulk issue
    rep("template <bool GATED, int MW, int NT, int KS, int NSTAGE, bool SPLITK, int ABL>\n__global__ void __launch_bounds__(128 * KS)\nmoe_w4a16_wt_kernel(",
        "template <bool GATED, int MW, int NT, int KS, int NSTAGE, bool SPLITK, int ABL, bool BULK = false>\n__global__ void __launch_bounds__(128 * KS)\nmoe_w4a16_wt_kernel(")
    rep("    extern __shared__ __align__(16) uint8_t wt_smem[];\n\n    if (GATED) pdl_wait();",
        "    extern __shared__ __align__(16) uint8_t wt_smem[];\n    __shared__ __align__(8) unsigned long long wfull[NSTAGE];   // BULK: one mbarrier per ring slot (weights + scales)\n\n    if (GATED) pdl_wait();")
    rep("    const uint32_t smem_base = smem_u32(wt_smem);\n\n    (void)l2pf;",
        "    const uint32_t smem_base = smem_u32(wt_smem);\n"
        "    // BULK: the CTA's TPC tiles are one contiguous T KB block per chunk in the grouped layout [e][n_x][chunk][TPC][1 KB]\n"
        "    // (scales [e][n_x][chunk][TPC][128 B]), in the kernel's own tile order tile_of(ti): one bulk copy each per chunk\n"
        "    const uint8_t* wgrp = Wr + (((size_t)e * gridDim.x + blockIdx.x) * NC_ALL + c0) * W_STAGE;\n"
        "    const uint8_t* sgrp = Sr + (((size_t)e * gridDim.x + blockIdx.x) * NC_ALL + c0) * S_STAGE;\n"
        "    if constexpr (BULK) {\n"
        "        if (tid == 0) {\n"
        "            #pragma unroll\n"
        "            for (int st = 0; st < NSTAGE; ++st) mbar_init(smem_u32(&wfull[st]), 1);\n"
        "            fence_mbar_init();\n"
        "        }\n"
        "        __syncthreads();\n"
        "    }\n\n    (void)l2pf;")
    rep("    auto issue_w = [&](int c, int stage) {\n        const uint32_t ws = smem_base + stage * STAGE_BYTES;\n        const uint32_t ss = ws + W_STAGE;\n        #pragma unroll\n        for (int i = 0; i < WSEG; ++i)\n            cp_async16(ws + (tid + i * NTHR) * 16, wsrc[i] + c * 1024, 16);\n        if (tid < TPC * 8) cp_async16(ss + tid * 16, ssrc + c * 128, 16);\n    };",
        "    auto issue_w = [&](int c, int stage) {\n        const uint32_t ws = smem_base + stage * STAGE_BYTES;\n        const uint32_t ss = ws + W_STAGE;\n"
        "        if constexpr (BULK) {\n"
        "            if (tid == 0) {\n"
        "                const uint32_t bar = smem_u32(&wfull[stage]);\n"
        "                fence_proxy_async();\n"
        "                mbar_expect_tx(bar, W_STAGE + S_STAGE);\n"
        "                bulk_g2s(ws, wgrp + (size_t)c * W_STAGE, W_STAGE, bar);\n"
        "                bulk_g2s(ss, sgrp + (size_t)c * S_STAGE, S_STAGE, bar);\n"
        "            }\n"
        "            return;\n"
        "        }\n"
        "        #pragma unroll\n        for (int i = 0; i < WSEG; ++i)\n            cp_async16(ws + (tid + i * NTHR) * 16, wsrc[i] + c * 1024, 16);\n        if (tid < TPC * 8) cp_async16(ss + tid * 16, ssrc + c * 128, 16);\n    };")
    rep("        const uint8_t* ws = wt_smem + stage * STAGE_BYTES;\n        stage = stage + 1 == NSTAGE ? 0 : stage + 1;",
        "        const uint8_t* ws = wt_smem + stage * STAGE_BYTES;\n"
        "        if constexpr (BULK) mbar_wait(smem_u32(&wfull[stage]), (uint32_t)((c / NSTAGE) & 1));   // this chunk's weights landed\n"
        "        stage = stage + 1 == NSTAGE ? 0 : stage + 1;")
    # 3. launcher + host
    rep("template <bool GATED, int MW, int NT, int KS, int NSTAGE, bool SPLITK, int ABL>\nvoid launch_wt(",
        "template <bool GATED, int MW, int NT, int KS, int NSTAGE, bool SPLITK, int ABL, bool BULK = false>\nvoid launch_wt(")
    s = s.replace("moe_w4a16_wt_kernel<GATED, MW, NT, KS, NSTAGE, SPLITK, ABL>", "moe_w4a16_wt_kernel<GATED, MW, NT, KS, NSTAGE, SPLITK, ABL, BULK>")
    rep("                  int64_t max_blocks, bool gated, int64_t mw, int64_t nt, int64_t ks, int64_t nstage, double gs_mul, int64_t abl,\n                  bool pdl, bool l2pf)\n{",
        "                  int64_t max_blocks, bool gated, int64_t mw, int64_t nt, int64_t ks, int64_t nstage, double gs_mul, int64_t abl,\n                  bool pdl, bool l2pf, bool bulk)\n{")
    rep("    const int key = (int)abl * 10000000 + (ksplit > 1 ? 1000000 : 0) + (gated ? 100000 : 0) + (int)mw * 10000 + (int)nt * 1000 + (int)ks * 100 + (int)nstage;\n    switch (key) {",
        "    if (bulk) {\n"
        "        const int T = (gated ? 2 : 1) * (int)mw * WT_NWARPS;\n"
        "        TORCH_CHECK((reinterpret_cast<uintptr_t>(Wp) & 127) == 0 && (reinterpret_cast<uintptr_t>(Sp) & 127) == 0 && Wr.is_contiguous() && Sr.is_contiguous(), \"wt bulk: layout alignment\");\n"
        "        TORCH_CHECK(Wr.numel() % (T * 1024) == 0 && Sr.numel() % (T * 128) == 0, \"wt bulk: grouped layout size\");\n"
        "    }\n"
        "    const int key = (bulk ? 100000000 : 0) + (int)abl * 10000000 + (ksplit > 1 ? 1000000 : 0) + (gated ? 100000 : 0) + (int)mw * 10000 + (int)nt * 1000 + (int)ks * 100 + (int)nstage;\n"
        "#define OKC_WB(G, M, T, S, NS) launch_wt<G, M, T, S, NS, false, 0, true>(Xp, (int)X.stride(0), sorted_tok.data_ptr<int>(), blk_e.data_ptr<int>(), \\\n"
        "        blk_row0.data_ptr<int>(), blk_end.data_ptr<int>(), nblk.data_ptr<int>(), blk_bm.data_ptr<int>(), Wp, Sp, gs.data_ptr<float>(), o16, obf, ldo, \\\n"
        "        pp, ldp, prow, cnt, (int)ksplit, (int)K, (int)ntiles, (int)max_blocks, (float)gs_mul, pdl, l2pf, stream)\n"
        "    switch (key) {\n"
        "        case 100111108: OKC_WB(true, 1, 1, 1, 8); break;   case 100111106: OKC_WB(true, 1, 1, 1, 6); break;   case 100111104: OKC_WB(true, 1, 1, 1, 4); break;\n"
        "        case 100112104: OKC_WB(true, 1, 2, 1, 4); break;   case 100112106: OKC_WB(true, 1, 2, 1, 6); break;   case 100112103: OKC_WB(true, 1, 2, 1, 3); break;\n"
        "        case 100011108: OKC_WB(false, 1, 1, 1, 8); break;  case 100011106: OKC_WB(false, 1, 1, 1, 6); break;  case 100011104: OKC_WB(false, 1, 1, 1, 4); break;\n"
        "        case 100021104: OKC_WB(false, 2, 1, 1, 4); break;  case 100021106: OKC_WB(false, 2, 1, 1, 6); break;  case 100021108: OKC_WB(false, 2, 1, 1, 8); break;\n"
        "        case 100012104: OKC_WB(false, 1, 2, 1, 4); break;  case 100022104: OKC_WB(false, 2, 2, 1, 4); break;  case 100022103: OKC_WB(false, 2, 2, 1, 3); break;\n"
        "        case 100012106: OKC_WB(false, 1, 2, 1, 6); break;")
    rep("#undef OKC_W\n}", "#undef OKC_W\n#undef OKC_WB\n}")
    rep("                  bool pdl, bool l2pf);\nvoid moe_finalize(", "                  bool pdl, bool l2pf, bool bulk);\nvoid moe_finalize(")
    s = s.replace('name="okc_moe_w4a16b"', 'name="okc_moe_w4a16c"')
    # 4. grouped repack
    s = s.rstrip("\n") + '''


def repack_wt_grouped(Wt: torch.Tensor, St: torch.Tensor, ntiles: int, gated: bool, mw: int):
    """The wt layouts regrouped in the wt kernel's CTA tile order for the BULK path: Wg uint8 [E, n_x, NC, T, 1024] /
    Sg [E, n_x, NC, T, 128] with T = (2 if gated else 1) * mw * 4 tiles per CTA, slot ti = w*TPW + i -> tile
    (x*4 + w)*mw + i % mw (+ ntiles/2 for the up tiles, i >= mw). Same bytes as Wt / St."""
    E, NTL, NC = Wt.shape[0], Wt.shape[1], Wt.shape[2]
    assert NTL == ntiles
    tpw = (2 if gated else 1) * mw
    T = tpw * 4
    n_x = (ntiles // 2 if gated else ntiles) // (4 * mw)
    x = torch.arange(n_x, device=Wt.device)
    ti = torch.arange(T, device=Wt.device)
    w, i = ti // tpw, ti % tpw
    tile = (x[:, None] * 4 + w[None, :]) * mw + (i % mw)[None, :]
    if gated:
        tile = tile + torch.where(i >= mw, ntiles // 2, 0)[None, :]
    tile = tile.reshape(-1)                                                   # [n_x * T] in (x, ti) order
    Wg = Wt.reshape(E, NTL, NC, 1024)[:, tile].reshape(E, n_x, T, NC, 1024).permute(0, 1, 3, 2, 4).contiguous()
    Sg = St.reshape(E, NTL, NC, 128)[:, tile].reshape(E, n_x, T, NC, 128).permute(0, 1, 3, 2, 4).contiguous()
    return Wg, Sg
'''
cp.write_text(s)
print("cuda_moe.py updated")

# ---------------------------------------------------------------- submission.py glue
sp = root / "muse_moe_w4a16/submission.py"; t = sp.read_text()


def tr(old, new, count=1):
    global t
    assert t.count(old) == count, (t.count(old), old[:90])
    t = t.replace(old, new)


if "WT_BULK" not in t:
    tr("WT_SAFE = (1, 1, 1, 4, 1)\n",
       "WT_SAFE = (1, 1, 1, 4, 1)\n"
       "# bulk-copy weight path (6th tuple element = 1): policy by token count, used when WT_USE_BULK and the self-test passed\n"
       "WT_BULK = {8: (1, 1, 1, 8, 1, 1), 16: (1, 1, 1, 6, 1, 1), 64: (1, 2, 1, 4, 1, 1)}\n"
       "WT_USE_BULK = False\n")
    tr("def _wt_variant(M: int, state):\n    if _WT_OVERRIDE is not None:\n        return _WT_OVERRIDE\n    pol = state.get(\"wt_policy\", WT_DEFAULT)\n",
       "def _wt_variant(M: int, state):\n    if _WT_OVERRIDE is not None:\n        return _WT_OVERRIDE\n    pol = state.get(\"wt_policy\", WT_DEFAULT)\n"
       "    if WT_USE_BULK and state.get(\"wt_bulk_ok\") and M <= max(WT_BULK):\n        pol = WT_BULK\n")
    tr("    small = _wt_variant(M, state)\n    nt, mw2, ks, nstage, ksplit = small\n",
       "    small = _wt_variant(M, state)\n    nt, mw2, ks, nstage, ksplit = small[:5]\n    bulk = bool(small[5]) if len(small) > 5 else False\n"
       "    if bulk and (f\"Wg2_{mw2}\" not in state or \"Wg13\" not in state):\n        bulk = False\n"
       "    W13, S13 = (state[\"Wg13\"], state[\"Sg13\"]) if bulk else (state[\"Wt13\"], state[\"St13\"])\n"
       "    W2, S2 = (state[f\"Wg2_{mw2}\"], state[f\"Sg2_{mw2}\"]) if bulk else (state[\"Wt2\"], state[\"St2\"])\n")
    tr("    ext.moe_w4a16_wt(x16, *lists, state[\"Wt13\"], state[\"St13\"], state[\"gs13\"], inter, part, counters, ksplit, H, 2 * I // 16, maxb,\n                     True, 1, nt, ks, nstage, state[\"gs_mul\"], _WT_ABL, pdl, l2pf)\n",
       "    ext.moe_w4a16_wt(x16, *lists, W13, S13, state[\"gs13\"], inter, part, counters, ksplit, H, 2 * I // 16, maxb,\n                     True, 1, nt, ks, nstage, state[\"gs_mul\"], _WT_ABL, pdl, l2pf, bulk)\n")
    tr("                         True, 1, big[0], big[2], big[3], state[\"gs_mul\"], _WT_ABL, False, False)\n",
       "                         True, 1, big[0], big[2], big[3], state[\"gs_mul\"], _WT_ABL, False, False, False)\n")
    tr("    ext.moe_w4a16_wt(inter, *lists, state[\"Wt2\"], state[\"St2\"], state[\"gs2\"], y, part, counters, ksplit, I, H // 16, maxb,\n                     False, mw2, nt, ks, nstage, state[\"gs_mul\"], _WT_ABL, pdl, l2pf)\n",
       "    ext.moe_w4a16_wt(inter, *lists, W2, S2, state[\"gs2\"], y, part, counters, ksplit, I, H // 16, maxb,\n                     False, mw2, nt, ks, nstage, state[\"gs_mul\"], _WT_ABL, pdl, l2pf, bulk)\n")
    tr("                         False, big[1], big[0], big[2], big[3], state[\"gs_mul\"], _WT_ABL, False, False)\n",
       "                         False, big[1], big[0], big[2], big[3], state[\"gs_mul\"], _WT_ABL, False, False, False)\n")
    # the bench's standalone GEMM calls
    t = t.replace("state[\"gs_mul\"], _WT_ABL, False, False),", "state[\"gs_mul\"], _WT_ABL, False, False, False),")
    t = t.replace("gated, 1 if gated else mw2, nt, ks, ns, state[\"gs_mul\"], abl, False, False)) - t_empty",
                  "gated, 1 if gated else mw2, nt, ks, ns, state[\"gs_mul\"], abl, False, False, False)) - t_empty")
    # grouped layouts in state
    tr("            state[\"gs_mul\"] = 16384.0   # the fp16 placement is e2m1 * 2^-14; applied with gs to the fp32 sums\n",
       "            state[\"gs_mul\"] = 16384.0   # the fp16 placement is e2m1 * 2^-14; applied with gs to the fp32 sums\n"
       "            state[\"Wg13\"], state[\"Sg13\"] = repack_wt_grouped(state[\"Wt13\"], state[\"St13\"], 2 * I // 16, True, 1)\n"
       "            for mw_ in (1, 2):\n"
       "                state[f\"Wg2_{mw_}\"], state[f\"Sg2_{mw_}\"] = repack_wt_grouped(state[\"Wt2\"], state[\"St2\"], H // 16, False, mw_)\n")
    # self-test: bulk variants too; wt_bulk_ok
    tr("WT_TEST_VARIANTS = ((1, 1, 1, 4, 1), (1, 1, 1, 8, 1), (1, 2, 1, 4, 1), (2, 2, 1, 4, 1), (2, 2, 1, 3, 1), (4, 2, 1, 2, 1))\n",
       "WT_TEST_VARIANTS = ((1, 1, 1, 4, 1), (1, 1, 1, 8, 1), (1, 2, 1, 4, 1), (2, 2, 1, 4, 1), (2, 2, 1, 3, 1), (4, 2, 1, 2, 1),\n"
       "                    (1, 1, 1, 8, 1, 1), (1, 1, 1, 6, 1, 1), (1, 2, 1, 4, 1, 1))\n")
    tr("    state[\"wt_bad\"] = bad\n    state[\"wt_ok\"] = WT_SAFE not in bad\n",
       "    state[\"wt_bad\"] = bad\n    state[\"wt_ok\"] = WT_SAFE not in bad\n"
       "    state[\"wt_bulk_ok\"] = all(v not in bad for v in WT_BULK.values())\n")
sp.write_text(t)
print("submission.py updated")
