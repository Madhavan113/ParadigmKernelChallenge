"""Phase 3 step 3: CPC chunks per bulk copy in the wt kernel's BULK path (super-stages of CPC consecutive K chunks:
one 16-32 KB bulk copy for the weights, one for the scales; x by cp.async per super-stage; the compute loop walks the
CPC chunks within a stage). Host: the bool `bulk` becomes `cpc` (0 = cp.async path, >= 1 = bulk with CPC chunks per
copy); policy 6th element = cpc. Run: python3 tools/_edit_moe_cpc.py && python3 tools/embed_cuda.py"""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
cp = root / "muse_moe_w4a16/cuda_moe.py"; s = cp.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


if "int CPC" not in s:
    rep("template <bool GATED, int MW, int NT, int KS, int NSTAGE, bool SPLITK, int ABL, bool BULK = false>\n__global__ void __launch_bounds__(128 * KS)\nmoe_w4a16_wt_kernel(",
        "template <bool GATED, int MW, int NT, int KS, int NSTAGE, bool SPLITK, int ABL, bool BULK = false, int CPC = 1>\n__global__ void __launch_bounds__(128 * KS)\nmoe_w4a16_wt_kernel(")
    rep("    constexpr int W_STAGE = TPC * 1024;\n    constexpr int S_STAGE = TPC * 128;\n    constexpr int STAGE_BYTES = wt_stage_bytes(TPW, NT);\n",
        "    static_assert(BULK || CPC == 1, \"CPC > 1 needs the bulk path\");\n"
        "    constexpr int W_CH = TPC * 1024, S_CH = TPC * 128, X_CH = 8 * NT * A_STRIDE * 2;   // per K chunk\n"
        "    constexpr int W_STAGE = CPC * W_CH;                 // a stage holds CPC consecutive chunks: [W x CPC][S x CPC][X x CPC]\n"
        "    constexpr int S_STAGE = CPC * S_CH;\n"
        "    constexpr int STAGE_BYTES = CPC * wt_stage_bytes(TPW, NT);\n")
    rep("    const int NC_ALL = K / KC;\n    const int NC = NC_ALL / KSPLIT;                   // chunks of this CTA\n    const int c0 = SPLITK ? (int)blockIdx.z * NC : 0;\n",
        "    const int NC_ALL = K / KC;\n    const int NC = NC_ALL / KSPLIT / CPC;             // stages (CPC chunks each) of this CTA\n    const int c0 = SPLITK ? (int)blockIdx.z * NC * CPC : 0;\n")
    # bulk sources: per chunk W_CH / S_CH, a stage copy = CPC chunks
    rep("    const uint8_t* wgrp = Wr + (((size_t)e * gridDim.x + blockIdx.x) * NC_ALL + c0) * W_STAGE;\n    const uint8_t* sgrp = Sr + (((size_t)e * gridDim.x + blockIdx.x) * NC_ALL + c0) * S_STAGE;\n",
        "    const uint8_t* wgrp = Wr + (((size_t)e * gridDim.x + blockIdx.x) * NC_ALL + c0) * W_CH;\n    const uint8_t* sgrp = Sr + (((size_t)e * gridDim.x + blockIdx.x) * NC_ALL + c0) * S_CH;\n")
    # x: per stage CPC chunks
    rep("    auto issue_x = [&](int c, int stage) {\n        const uint32_t xs = smem_base + stage * STAGE_BYTES + W_STAGE + S_STAGE;\n        #pragma unroll\n        for (int i = 0; i < XSEG; ++i) {\n            const int sidx = tid + i * NTHR;\n            if (sidx < NT * 128)\n                cp_async16(xs + ((sidx >> 4) * A_STRIDE + (sidx & 15) * 8) * 2, xsrc[i] + c * KC, xbytes[i]);\n        }\n    };",
        "    auto issue_x = [&](int c, int stage) {\n        const uint32_t xs = smem_base + stage * STAGE_BYTES + W_STAGE + S_STAGE;\n"
        "        #pragma unroll\n        for (int k = 0; k < CPC; ++k) {\n"
        "            #pragma unroll\n            for (int i = 0; i < XSEG; ++i) {\n                const int sidx = tid + i * NTHR;\n                if (sidx < NT * 128)\n"
        "                    cp_async16(xs + k * X_CH + ((sidx >> 4) * A_STRIDE + (sidx & 15) * 8) * 2, xsrc[i] + (c * CPC + k) * KC, xbytes[i]);\n            }\n        }\n    };")
    # the cp.async weight path advances by W_CH per stage (CPC == 1 there), the bulk copy by W_STAGE
    rep("            cp_async16(ws + (tid + i * NTHR) * 16, wsrc[i] + c * 1024, 16);\n        if (tid < TPC * 8) cp_async16(ss + tid * 16, ssrc + c * 128, 16);\n    };",
        "            cp_async16(ws + (tid + i * NTHR) * 16, wsrc[i] + c * 1024, 16);\n        if (tid < TPC * 8) cp_async16(ss + tid * 16, ssrc + c * 128, 16);\n    };\n"
        "    // (bulk: wgrp + c * W_STAGE covers the CPC consecutive chunks of stage c in the grouped layout)")
    # the compute body: walk the CPC chunks of the stage
    rep("        const uint8_t* ss = ws + W_STAGE;\n        const __half* xs = reinterpret_cast<const __half*>(ss + S_STAGE);\n        if (ABL == 3) continue;                                      // bench ablation: the cp.async ring only\n        #pragma unroll\n        for (int hh = 0; hh < 2 / KS; ++hh) {",
        "        if (ABL == 3) continue;                                      // bench ablation: the cp.async ring only\n"
        "        #pragma unroll\n        for (int kc = 0; kc < CPC; ++kc) {\n"
        "        const uint8_t* wsk = ws + kc * W_CH;\n        const uint8_t* ss = ws + W_STAGE + kc * S_CH;\n        const __half* xs = reinterpret_cast<const __half*>(ws + W_STAGE + S_STAGE + kc * X_CH);\n"
        "        #pragma unroll\n        for (int hh = 0; hh < 2 / KS; ++hh) {")
    rep("                wq[i] = *reinterpret_cast<const uint4*>(ws + ((warp * TPW + i) * 2 + h) * 512 + lane * 16);",
        "                wq[i] = *reinterpret_cast<const uint4*>(wsk + ((warp * TPW + i) * 2 + h) * 512 + lane * 16);")
    # close the kc loop: the hh loop's closing brace is followed by the chunk loop's closing brace and pdl_trigger
    rep("            }\n        }\n    }\n    pdl_trigger();                                    // main loop done: the next kernel may start launching",
        "            }\n        }\n        }   // kc\n    }\n    pdl_trigger();                                    // main loop done: the next kernel may start launching")
    # launcher + host
    rep("template <bool GATED, int MW, int NT, int KS, int NSTAGE, bool SPLITK, int ABL, bool BULK = false>\nvoid launch_wt(",
        "template <bool GATED, int MW, int NT, int KS, int NSTAGE, bool SPLITK, int ABL, bool BULK = false, int CPC = 1>\nvoid launch_wt(")
    s = s.replace("moe_w4a16_wt_kernel<GATED, MW, NT, KS, NSTAGE, SPLITK, ABL, BULK>", "moe_w4a16_wt_kernel<GATED, MW, NT, KS, NSTAGE, SPLITK, ABL, BULK, CPC>")
    rep("    constexpr int smem = NSTAGE * wt_stage_bytes(TPW, NT);\n    static bool attr_set = false;",
        "    constexpr int smem = NSTAGE * CPC * wt_stage_bytes(TPW, NT);\n    static bool attr_set = false;")
    rep("                  bool pdl, bool l2pf, bool bulk)\n{", "                  bool pdl, bool l2pf, int64_t cpc)\n{")
    rep("    if (bulk) {\n", "    const bool bulk = cpc > 0;\n    if (bulk) {\n")
    rep("    const int key = (bulk ? 100000000 : 0) + (int)abl * 10000000",
        "    TORCH_CHECK(cpc <= 1 || (K / 128) % cpc == 0, \"wt bulk: chunks per copy must divide K / 128\");\n"
        "    const int key = (int)cpc * 100000000 + (int)abl * 10000000")
    rep("#define OKC_WB(G, M, T, S, NS) launch_wt<G, M, T, S, NS, false, 0, true>(",
        "#define OKC_WB(G, M, T, S, NS) launch_wt<G, M, T, S, NS, false, 0, true, 1>(")
    rep("        case 100012106: OKC_WB(false, 1, 2, 1, 6); break;",
        "        case 100012106: OKC_WB(false, 1, 2, 1, 6); break;\n"
        "#define OKC_WC(G, M, T, S, NS, C) launch_wt<G, M, T, S, NS, false, 0, true, C>(Xp, (int)X.stride(0), sorted_tok.data_ptr<int>(), blk_e.data_ptr<int>(), \\\n"
        "        blk_row0.data_ptr<int>(), blk_end.data_ptr<int>(), nblk.data_ptr<int>(), blk_bm.data_ptr<int>(), Wp, Sp, gs.data_ptr<float>(), o16, obf, ldo, \\\n"
        "        pp, ldp, prow, cnt, (int)ksplit, (int)K, (int)ntiles, (int)max_blocks, (float)gs_mul, pdl, l2pf, stream)\n"
        "        // super-stages: CPC chunks per bulk copy (gated TPC = 8 -> 16 / 32 KB; non-gated mw 1 -> 8 / 16 / 32 KB)\n"
        "        case 400111102: OKC_WC(true, 1, 1, 1, 2, 4); break;   case 400111103: OKC_WC(true, 1, 1, 1, 3, 4); break;\n"
        "        case 200111104: OKC_WC(true, 1, 1, 1, 4, 2); break;   case 200111103: OKC_WC(true, 1, 1, 1, 3, 2); break;\n"
        "        case 400112102: OKC_WC(true, 1, 2, 1, 2, 4); break;   case 200112103: OKC_WC(true, 1, 2, 1, 3, 2); break;\n"
        "        case 400011102: OKC_WC(false, 1, 1, 1, 2, 4); break;  case 400011103: OKC_WC(false, 1, 1, 1, 3, 4); break;\n"
        "        case 800011102: OKC_WC(false, 1, 1, 1, 2, 8); break;  case 200011104: OKC_WC(false, 1, 1, 1, 4, 2); break;\n"
        "        case 400021102: OKC_WC(false, 2, 1, 1, 2, 4); break;  case 400021103: OKC_WC(false, 2, 1, 1, 3, 4); break;\n"
        "        case 400012102: OKC_WC(false, 1, 2, 1, 2, 4); break;  case 200012103: OKC_WC(false, 1, 2, 1, 3, 2); break;\n"
        "        case 400022102: OKC_WC(false, 2, 2, 1, 2, 4); break;")
    rep("#undef OKC_W\n#undef OKC_WB\n}", "#undef OKC_W\n#undef OKC_WB\n#undef OKC_WC\n}")
    rep("                  bool pdl, bool l2pf, bool bulk);\nvoid moe_finalize(", "                  bool pdl, bool l2pf, int64_t cpc);\nvoid moe_finalize(")
    s = s.replace('name="okc_moe_w4a16c"', 'name="okc_moe_w4a16d"')
cp.write_text(s)
print("cuda_moe.py updated")

sp = root / "muse_moe_w4a16/submission.py"; t = sp.read_text()


def tr(old, new, count=1):
    global t
    assert t.count(old) == count, (t.count(old), old[:90])
    t = t.replace(old, new)


if "cpc = int(small[5])" not in t:
    tr("    nt, mw2, ks, nstage, ksplit = small[:5]\n    bulk = bool(small[5]) if len(small) > 5 else False\n",
       "    nt, mw2, ks, nstage, ksplit = small[:5]\n    cpc = int(small[5]) if len(small) > 5 else 0\n    bulk = cpc > 0\n")
    tr("                     True, 1, nt, ks, nstage, state[\"gs_mul\"], _WT_ABL, pdl, l2pf, bulk)\n",
       "                     True, 1, nt, ks, nstage, state[\"gs_mul\"], _WT_ABL, pdl, l2pf, cpc)\n")
    tr("                     False, mw2, nt, ks, nstage, state[\"gs_mul\"], _WT_ABL, pdl, l2pf, bulk)\n",
       "                     False, mw2, nt, ks, nstage, state[\"gs_mul\"], _WT_ABL, pdl, l2pf, cpc)\n")
    t = t.replace("state[\"gs_mul\"], _WT_ABL, False, False, False)", "state[\"gs_mul\"], _WT_ABL, False, False, 0)")
    t = t.replace("state[\"gs_mul\"], abl, False, False, False)) - t_empty", "state[\"gs_mul\"], abl, False, False, 0)) - t_empty")
    tr("WT_BULK = {8: (1, 1, 1, 8, 1, 1), 16: (1, 1, 1, 6, 1, 1), 64: (1, 2, 1, 4, 1, 1)}\n",
       "WT_BULK = {8: (1, 1, 1, 2, 1, 4), 16: (1, 1, 1, 2, 1, 4), 64: (1, 2, 1, 2, 1, 4)}   # 6th element = chunks per bulk copy (32 KB stages)\n")
    tr("                    (1, 1, 1, 8, 1, 1), (1, 1, 1, 6, 1, 1), (1, 2, 1, 4, 1, 1))\n",
       "                    (1, 1, 1, 8, 1, 1), (1, 1, 1, 2, 1, 4), (1, 1, 1, 4, 1, 2), (1, 2, 1, 2, 1, 4), (1, 1, 1, 3, 1, 4))\n")
sp.write_text(t)
eb = root / "tools/_edit_moe_stream.py"; u = eb.read_text()
u = u.replace("state[\"gs_mul\"], abl, False, False, False)) - t_empty", "state[\"gs_mul\"], abl, False, False, 0)) - t_empty")
u = u.replace("        for v in ((1, 1, 1, 8, 1, 1), (1, 1, 1, 6, 1, 1), (1, 1, 1, 4, 1, 1), (1, 2, 1, 4, 1, 1), (1, 2, 1, 6, 1, 1), (2, 2, 1, 4, 1, 1), (1, 1, 1, 8, 1, 0)):",
              "        for v in ((1, 1, 1, 2, 1, 4), (1, 1, 1, 3, 1, 4), (1, 1, 1, 4, 1, 2), (1, 1, 1, 3, 1, 2), (1, 2, 1, 2, 1, 4), (1, 2, 1, 3, 1, 2), (2, 2, 1, 2, 1, 4), (1, 1, 1, 8, 1, 1), (1, 1, 1, 8, 1, 0)):")
u = u.replace("    for bs in (1, 8):\n        x, ids, tw = _judge_inputs(bs, dev, seed=10 + bs)\n        experts = sorted", "    for bs in ():\n        x, ids, tw = _judge_inputs(bs, dev, seed=10 + bs)\n        experts = sorted")
eb.write_text(u)
print("submission.py + bench updated")
