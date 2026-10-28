"""Boundary tails v3: the sk2 kernel's last-arriving CTA computes the decode layer's boundary with the v4 glue's math
(alpha / beta precomputed, one fused four-sum reduction, two passes: sums then stores from registers), for M <= 8 rows.
proj fork: the BT epilogue rewritten, sk2b takes (alpha, beta) instead of the gate; layer_glue: FUSE_BOUNDARY on by
default (the per-(T, layer) switch st["btail_on"] stays off unless BTAIL_LOCK / ab_vm btail:T.layer=1 sets it),
BTAIL_MAX_T 8, the tuner's btail timing replaced by BTAIL_LOCK, call sites and self-test pass (alpha, beta).
Run once from the worktree root."""
import pathlib

root = pathlib.Path(__file__).resolve().parents[2]


def patch(path, edits):
    p = root / path
    s = p.read_text()
    for old, new, cnt in edits:
        n = s.count(old)
        assert n == cnt, (path, n, cnt, old[:90])
        s = s.replace(old, new)
    p.write_text(s)


# ---------------------------------------------------------------- proj_fork.py: the tail body
p = root / "muse_decode_layer/proj_fork.py"
s = p.read_text()
i0 = s.index("      if (M == 1) {\n        float4 sv[CH][2];")
i1 = s.index("  if constexpr (FUSE) {\n    if (dcnt == nullptr && part2 != nullptr) {")
# the block to replace ends with the closing of `if (M == 1) {`, of `if (b_cnt != nullptr) {` and of `if constexpr (BT) {`
tail_end = s.rindex("      }\n    }\n  }\n", i0, i1) + len("      }\n    }\n  }\n")
new_tail = r'''      // ---- tail v3 (M <= 8): the v4 glue's math per token row. alpha / beta of the residual gate are precomputed
      // (prepare-time _gate_coeffs), the four row sums are reduced ONCE (sum s^2 = A + 2rB + r^2 D), as = alpha sin
      // and bx = beta a g stay in registers between the two passes; pass 2 loads gn and stores s and h.
      for (int row = 0; row < M; ++row) {
        const float* sin_r = b_sin + (size_t)row * 8192;
        const OutT* a_r = out + (size_t)row * N;
        const bool hg = b_gp != nullptr;
        float as_[CH][8], bx_[CH][8];
        float pA = 0.f, pB = 0.f, pC = 0.f, pD = 0.f;
#pragma unroll
        for (int j = 0; j < CH; ++j) {
          const int c8 = (tid + j * NC) * 8;
          const float4 s0 = __ldg(reinterpret_cast<const float4*>(sin_r + c8));
          const float4 s1 = __ldg(reinterpret_cast<const float4*>(sin_r + c8 + 4));
          const float4 al0 = __ldg(reinterpret_cast<const float4*>(b_alpha + c8));
          const float4 al1 = __ldg(reinterpret_cast<const float4*>(b_alpha + c8 + 4));
          const float4 be0 = __ldg(reinterpret_cast<const float4*>(b_beta + c8));
          const float4 be1 = __ldg(reinterpret_cast<const float4*>(b_beta + c8 + 4));
          const uint4 av = (c8 < N) ? __ldcg(reinterpret_cast<const uint4*>(a_r + c8)) : make_uint4(0u, 0u, 0u, 0u);
          const uint4 gpv = (hg && c8 < N) ? __ldg(reinterpret_cast<const uint4*>(b_gp + c8)) : make_uint4(0u, 0u, 0u, 0u);
          const float sl[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
          const float alv[8] = {al0.x, al0.y, al0.z, al0.w, al1.x, al1.y, al1.z, al1.w};
          const float bev[8] = {be0.x, be0.y, be0.z, be0.w, be1.x, be1.y, be1.z, be1.w};
          const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&av);
          const __nv_bfloat162* p2 = reinterpret_cast<const __nv_bfloat162*>(&gpv);
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const float2 af = __bfloat1622float2(a2[q]), pf = __bfloat1622float2(p2[q]);
            const float aa[2] = {af.x, af.y}, pp[2] = {pf.x, pf.y};
#pragma unroll
            for (int e = 0; e < 2; ++e) {
              const int i = 2 * q + e;
              float ag = (c8 < N) ? aa[e] : 0.f;
              if (hg) ag *= 1.f + pp[e];
              const float as1 = alv[i] * sl[i], bx1 = bev[i] * ag;
              as_[j][i] = as1; bx_[j][i] = bx1;
              pC = fmaf(ag, ag, pC); pA = fmaf(as1, as1, pA); pB = fmaf(as1, bx1, pB); pD = fmaf(bx1, bx1, pD);
            }
          }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
          pA += __shfl_xor_sync(0xffffffffu, pA, off); pB += __shfl_xor_sync(0xffffffffu, pB, off);
          pC += __shfl_xor_sync(0xffffffffu, pC, off); pD += __shfl_xor_sync(0xffffffffu, pD, off);
        }
        if (lane == 0) { s_bred4[warp][0] = pA; s_bred4[warp][1] = pB; s_bred4[warp][2] = pC; s_bred4[warp][3] = pD; }
        named_bar(1, NC);
        float A = 0.f, B = 0.f, C = 0.f, D = 0.f;
#pragma unroll
        for (int w = 0; w < NW; ++w) { A += s_bred4[w][0]; B += s_bred4[w][1]; C += s_bred4[w][2]; D += s_bred4[w][3]; }
        named_bar(1, NC);   // s_bred4 is reused by the next row
        const float r = rsqrtf(C * (1.f / 8192.f) + 1e-8f);
        const float c = rsqrtf((A + 2.f * r * B + r * r * D) * (1.f / 8192.f) + 1e-5f);
        float* sout_r = b_sout + (size_t)row * 8192;
        __nv_bfloat16* hout_r = b_hout + (size_t)row * 8192;
#pragma unroll
        for (int j = 0; j < CH; ++j) {
          const int c8 = (tid + j * NC) * 8;
          const uint4 gnv = __ldg(reinterpret_cast<const uint4*>(b_gn + c8));
          float s8[8];
#pragma unroll
          for (int i = 0; i < 8; ++i) s8[i] = fmaf(bx_[j][i], r, as_[j][i]);
          *reinterpret_cast<float4*>(sout_r + c8) = make_float4(s8[0], s8[1], s8[2], s8[3]);
          *reinterpret_cast<float4*>(sout_r + c8 + 4) = make_float4(s8[4], s8[5], s8[6], s8[7]);
          const __nv_bfloat162* n2 = reinterpret_cast<const __nv_bfloat162*>(&gnv);
          uint4 hv;
          __nv_bfloat162* h2 = reinterpret_cast<__nv_bfloat162*>(&hv);
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const float2 nf = __bfloat1622float2(n2[q]);
            h2[q] = __floats2bfloat162_rn(s8[2 * q] * c * (1.f + nf.x), s8[2 * q + 1] * c * (1.f + nf.y));
          }
          *reinterpret_cast<uint4*>(hout_r + c8) = hv;
        }
      }
    }
  }
'''
s = s[:i0] + new_tail + s[tail_end:]
p.write_text(s)

patch("muse_decode_layer/proj_fork.py", [
    # the v2 comment block above the tail: mention v3
    ("""      // ---- boundary tail v2 (decode layer, M = 1): the LAST CTA by atomic ticket (no grid barrier) computes the""",
     """      // ---- boundary tail (decode layer, M <= 8; v3 math below): the LAST CTA by atomic ticket (no grid barrier) computes the""", 1),
    # host entry: (alpha, beta) instead of the gate
    ("""void sk2b(at::Tensor Ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4, int64_t M, int64_t N,
          int64_t K, int64_t cfg, int64_t tokr, at::Tensor gate, at::Tensor sin_, at::Tensor gp,
          at::Tensor gn, at::Tensor sout, at::Tensor hout, at::Tensor part, at::Tensor cnt, int64_t has_gp) {
  TORCH_CHECK(M >= 1 && M <= 8, "sk2b: M <= 8");""",
     """void sk2b(at::Tensor Ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4, int64_t M, int64_t N,
          int64_t K, int64_t cfg, int64_t tokr, at::Tensor alpha, at::Tensor beta, at::Tensor sin_, at::Tensor gp,
          at::Tensor gn, at::Tensor sout, at::Tensor hout, at::Tensor part, at::Tensor cnt, int64_t has_gp) {
  TORCH_CHECK(M >= 1 && M <= 8, "sk2b: M <= 8");
  TORCH_CHECK(alpha.scalar_type() == at::kFloat && beta.scalar_type() == at::kFloat && alpha.numel() == 8192 && beta.numel() == 8192, "sk2b: alpha / beta fp32 [8192]");
  TORCH_CHECK(sin_.scalar_type() == at::kFloat && sout.scalar_type() == at::kFloat && sin_.size(0) >= M && sout.size(0) >= M && hout.size(0) >= M, "sk2b: rows");""", 1),
    ("""void sk2b(torch::Tensor Ws, torch::Tensor x, torch::Tensor out, torch::Tensor ptab, int64_t pmax, int64_t base4, int64_t M,
          int64_t N, int64_t K, int64_t cfg, int64_t tokr, torch::Tensor gate, torch::Tensor sin_,
          torch::Tensor gp, torch::Tensor gn, torch::Tensor sout, torch::Tensor hout, torch::Tensor part, torch::Tensor cnt,
          int64_t has_gp);""",
     """void sk2b(torch::Tensor Ws, torch::Tensor x, torch::Tensor out, torch::Tensor ptab, int64_t pmax, int64_t base4, int64_t M,
          int64_t N, int64_t K, int64_t cfg, int64_t tokr, torch::Tensor alpha, torch::Tensor beta, torch::Tensor sin_,
          torch::Tensor gp, torch::Tensor gn, torch::Tensor sout, torch::Tensor hout, torch::Tensor part, torch::Tensor cnt,
          int64_t has_gp);""", 1),
    # the dispatch lines: alpha / beta in the b_alpha / b_beta slots, no gate
    ("""                      f"nullptr, nullptr, nullptr, nullptr, sin_.data_ptr<float>(), \"""",
     """                      f"nullptr, nullptr, alpha.data_ptr<float>(), beta.data_ptr<float>(), sin_.data_ptr<float>(), \"""", 1),
    ("""                      f"reinterpret_cast<__nv_bfloat16*>(hout.data_ptr()), nullptr, cnt.data_ptr<int>(), reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr())); return; }}")""",
     """                      f"reinterpret_cast<__nv_bfloat16*>(hout.data_ptr()), nullptr, cnt.data_ptr<int>(), nullptr); return; }}")""", 1),
    # the python entry
    ("""    assert m == 1 and layer in ("o", "unpooler")
    name = _sk2_name(layer, m)
    ci, stream, tab, pmax, b4, n_rows = state["sk2"][name]
    gate, sin_, gp, gn, sout, hout = tail   # gate: the residual gate's bf16 weight [8192] (alpha / beta formed in-kernel)
    assert gate.dtype == torch.bfloat16 and sin_.dtype == torch.float32 and gn.dtype == torch.bfloat16""",
     """    assert 1 <= m <= 8 and layer in ("o", "unpooler")
    name = _sk2_name(layer, m)
    ci, stream, tab, pmax, b4, n_rows = state["sk2"][name]
    alpha, beta, sin_, gp, gn, sout, hout = tail   # alpha / beta: the residual gate's coefficients, fp32 [8192] (prepare-time)
    assert alpha.dtype == torch.float32 and beta.dtype == torch.float32 and sin_.dtype == torch.float32 and gn.dtype == torch.bfloat16""", 1),
    ("""    ext.sk2b(stream, x, out, tab, pmax, b4, m, n, k, ci, _sk2_tokr(m), gate, sin_, gn if gp is None else gp, gn,
             sout, hout, state["dummy"], cnt, 0 if gp is None else 1)""",
     """    ext.sk2b(stream, x, out, tab, pmax, b4, m, n, k, ci, _sk2_tokr(m), alpha, beta, sin_, gn if gp is None else gp, gn,
             sout, hout, state["dummy"], cnt, 0 if gp is None else 1)""", 1),
])
# the docstring of run_sk2_tail
patch("muse_decode_layer/proj_fork.py", [
    ("""    layer's residual/norm boundary: tail = (alpha fp32 [8192], beta fp32 [8192], sin fp32 [m, 8192] (the stream the""",
     """    layer's residual/norm boundary (M <= 8 rows): tail = (alpha fp32 [8192], beta fp32 [8192], sin fp32 [m, 8192] (the stream the""", 1),
])
# the sm_80 stub of sk2b (arity must match the declaration: alpha and beta replace the gate)
patch("muse_decode_layer/proj_fork.py", [
    ("""                'void sk2b(at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, '
                'int64_t, int64_t, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, '
                'at::Tensor, at::Tensor, int64_t) { TORCH_CHECK(false, "sk2b needs sm_90"); }\\n'""",
     """                'void sk2b(at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, '
                'int64_t, int64_t, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, '
                'at::Tensor, at::Tensor, int64_t) { TORCH_CHECK(false, "sk2b needs sm_90"); }\\n'""", 1),
])

# ---------------------------------------------------------------- layer_glue.py
patch("muse_decode_layer/layer_glue.py", [
    ("""FUSE_BOUNDARY = os.environ.get("OKC_FUSE_BOUNDARY", "0") == "1"   # (GPU A/B hook; the judge sees the default)""",
     """FUSE_BOUNDARY = os.environ.get("OKC_FUSE_BOUNDARY", "1") == "1"   # the tails are built and self-tested; each (T, layer) runs
                                                                   # only when BTAIL_LOCK (or an ab_vm override) turns it on""", 1),
    ("""BTAIL_MAX_T = 1   # the v2 tails load the whole row into registers in one round: one row only""",
     """BTAIL_MAX_T = 8   # the v3 tails loop over the rows (one CTA): T <= 8
BTAIL_LOCK = {}   # T -> layers ("o" = boundary 1 in the o-projection sk2, "unpooler" = boundary 2 in the unpooler slice
                  # sk2) whose boundary runs in the producer's last CTA; decided on the H200 (vm_ab/ab20)""", 1),
    ("""        if T <= BTAIL_MAX_T and st.get("fuse_bnd_ok"):
            # the boundary tails (unpooler slice -> boundary 2, o projection -> boundary 1), each vs its Triton launch,
            # on the chain: none / unpooler / o / both
            ttimes = {}
            for combo in ((), ("unpooler",), ("o",), ("unpooler", "o")):
                for l_ in ("unpooler", "o"):
                    st.setdefault("btail_on", {})[(T, l_)] = l_ in combo
                try:
                    ttimes[combo] = _graph_time(lambda: _after_attention(data, st, o_in, None, T), flush)
                except Exception as ex:  # noqa: BLE001
                    ttimes[combo] = 1e30
                    if log:
                        msg.append(f"btailX:{'+'.join(combo) or 'none'}:{repr(ex)[:80]}")
            best_c = min(ttimes, key=lambda k_: ttimes[k_])
            for l_ in ("unpooler", "o"):
                st["btail_on"][(T, l_)] = l_ in best_c
            if log:
                msg.append("btail[" + " ".join(f"{'+'.join(c_) or 'none'}:{t:.0f}" for c_, t in ttimes.items())
                           + f"]->{'+'.join(best_c) or 'none'}")""",
     """        if T <= BTAIL_MAX_T and st.get("fuse_bnd_ok"):   # the boundary tails per (T, layer): locked, not timed
            for l_ in ("unpooler", "o"):
                st.setdefault("btail_on", {})[(T, l_)] = l_ in BTAIL_LOCK.get(T, ())""", 1),
    # call sites: (alpha, beta) tuples
    ("""        globals()["bfp_run_sk2_tail"](o_in, "o", T, HIDDEN, 1024, state["bfp"],
                         (state["gate1"], data["stream"], None, state["g_preffn"], s1, h))""",
     """        globals()["bfp_run_sk2_tail"](o_in, "o", T, HIDDEN, 1024, state["bfp"],
                         (state["ab1"][0], state["ab1"][1], data["stream"], None, state["g_preffn"], s1, h))""", 1),
    ("""        globals()["bfp_run_sk2_tail"](m, "unpooler", T, nu, D_EXPERT, state["bfp"],
                         (state["gate2"], s1, state["g_postffn"], state["g_next"], s2, out))""",
     """        globals()["bfp_run_sk2_tail"](m, "unpooler", T, nu, D_EXPERT, state["bfp"],
                         (state["ab2"][0], state["ab2"][1], s1, state["g_postffn"], state["g_next"], s2, out))""", 1),
])
# the self-test: T in (1, 8), (alpha, beta) tuples
g = root / "muse_decode_layer/layer_glue.py"
s = g.read_text()
i = s.index("def _boundary_selftest(st):")
j = s.index("\n\n\n", i)
blk = s[i:j]
blk2 = blk.replace("    for T in (1,):   # the v2 tails hold the whole row in registers: one row", "    for T in (1, 8):   # the v3 tails loop over the rows of one CTA")
old1 = """(st["gate1"], stream, None, st["g_preffn"], s1a, ha))"""
old2 = """(st["gate2"], s1a, st["g_postffn"], st["g_next"], s2a, outa))"""
assert blk2.count(old1) == 1 and blk2.count(old2) == 1, (blk2.count(old1), blk2.count(old2))
blk2 = blk2.replace(old1, """(st["ab1"][0], st["ab1"][1], stream, None, st["g_preffn"], s1a, ha))""")
blk2 = blk2.replace(old2, """(st["ab2"][0], st["ab2"][1], s1a, st["g_postffn"], st["g_next"], s2a, outa))""")
assert blk2 != blk
s = s[:i] + blk2 + s[j:]
g.write_text(s)
print("patched")
