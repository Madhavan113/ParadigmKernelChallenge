"""Round 2 edit: ablation MODEs of the wt kernel for the bench (bs1/bs8 GEMM1/GEMM2 breakdown):
MODE 1 loads only (no compute), 2 no dequant (raw words as fragments), 3 no MMA (dequant kept), 4 no token tile."""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
p = root / "muse_moe_w4a16/cuda_moe.py"
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:100])
    s = s.replace(old, new)


rep("""template <bool GATED, int MW, int NT, int KS, int NSTAGE, bool SPLITK, bool TMA>
__global__ void __launch_bounds__(128 * KS)
moe_w4a16_wt_kernel(""", """template <bool GATED, int MW, int NT, int KS, int NSTAGE, bool SPLITK, bool TMA, int MODE = 0>
__global__ void __launch_bounds__(128 * KS)
moe_w4a16_wt_kernel(""")
# x tile: MODE 4 skips the token-row copies (zero bytes) and the ldmatrix
rep("""            xsrc[i] = X + (size_t)tok * ldx + c0 * KC + (sidx & 15) * 8;
            xbytes[i] = valid ? 16 : 0;""", """            xsrc[i] = X + (size_t)tok * ldx + c0 * KC + (sidx & 15) * 8;
            xbytes[i] = (valid && MODE != 4) ? 16 : 0;""")
# compute block
rep("""        const uint8_t* ws = wt_smem + stage * STAGE_BYTES;
        const uint8_t* ss = ws + W_STAGE;
        const __half* xs = reinterpret_cast<const __half*>(ss + S_STAGE);
        #pragma unroll
        for (int hh = 0; hh < 2 / KS; ++hh) {""", """        const uint8_t* ws = wt_smem + stage * STAGE_BYTES;
        const uint8_t* ss = ws + W_STAGE;
        const __half* xs = reinterpret_cast<const __half*>(ss + S_STAGE);
        if constexpr (MODE == 1) {                                   // loads only: touch one word per warp
            if (lane == 0) acc[0][0][0] += __uint_as_float(*reinterpret_cast<const uint32_t*>(ws + warp * 16)) * 1e-30f;
            continue;
        }
        #pragma unroll
        for (int hh = 0; hh < 2 / KS; ++hh) {""")
rep("""                #pragma unroll
                for (int nb = 0; nb < NT; ++nb) {
                    if (nb < ntb)
                        ldmatrix_x4(b[nb][0], b[nb][1], b[nb][2], b[nb][3],
                                    smem_u32(xs + (nb * 8 + (lane & 7)) * A_STRIDE + h * 64 + ps * 32 + (lane >> 3) * 8));
                }""", """                #pragma unroll
                for (int nb = 0; nb < NT; ++nb) {
                    if (MODE == 4) { b[nb][0] = b[nb][1] = b[nb][2] = b[nb][3] = 0u; }
                    else if (nb < ntb)
                        ldmatrix_x4(b[nb][0], b[nb][1], b[nb][2], b[nb][3],
                                    smem_u32(xs + (nb * 8 + (lane & 7)) * A_STRIDE + h * 64 + ps * 32 + (lane >> 3) * 8));
                }""")
rep("""                        uint32_t a[4];
                        a[0] = f16x2_mul(e4m3x2_lo16_to_f16x2(elo), S0);
                        a[1] = f16x2_mul(e4m3x2_hi16_to_f16x2(elo), S1);
                        a[2] = f16x2_mul(e4m3x2_lo16_to_f16x2(ehi), S0);
                        a[3] = f16x2_mul(e4m3x2_hi16_to_f16x2(ehi), S1);
                        #pragma unroll
                        for (int nb = 0; nb < NT; ++nb) {
                            if (nb < ntb) mma16816(acc[i][nb * CH + (CH == 2 ? q : 0)], a, b[nb][2 * q], b[nb][2 * q + 1]);
                        }""", """                        uint32_t a[4];
                        if constexpr (MODE == 2) {                   // no dequant: raw words as fragments
                            a[0] = w; a[1] = sw; a[2] = elo; a[3] = ehi;
                        } else {
                            a[0] = f16x2_mul(e4m3x2_lo16_to_f16x2(elo), S0);
                            a[1] = f16x2_mul(e4m3x2_hi16_to_f16x2(elo), S1);
                            a[2] = f16x2_mul(e4m3x2_lo16_to_f16x2(ehi), S0);
                            a[3] = f16x2_mul(e4m3x2_hi16_to_f16x2(ehi), S1);
                        }
                        if constexpr (MODE == 3) {                   // no MMA: keep the fragments live
                            acc[i][0][0] += __uint_as_float((a[0] ^ a[1] ^ a[2] ^ a[3]) & 0x007FFFFFu);
                        } else {
                            #pragma unroll
                            for (int nb = 0; nb < NT; ++nb) {
                                if (nb < ntb) mma16816(acc[i][nb * CH + (CH == 2 ? q : 0)], a, b[nb][2 * q], b[nb][2 * q + 1]);
                            }
                        }""")
# launch_wt: MODE
rep("""template <bool GATED, int MW, int NT, int KS, int NSTAGE, bool SPLITK, bool TMA>
void launch_wt(""", """template <bool GATED, int MW, int NT, int KS, int NSTAGE, bool SPLITK, bool TMA, int MODE = 0>
void launch_wt(""")
rep("""        cudaFuncSetAttribute(moe_w4a16_wt_kernel<GATED, MW, NT, KS, NSTAGE, SPLITK, TMA>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);""",
    """        cudaFuncSetAttribute(moe_w4a16_wt_kernel<GATED, MW, NT, KS, NSTAGE, SPLITK, TMA, MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);""")
rep("""    cudaError_t err = cudaLaunchKernelEx(&cfg, moe_w4a16_wt_kernel<GATED, MW, NT, KS, NSTAGE, SPLITK, TMA>,""",
    """    cudaError_t err = cudaLaunchKernelEx(&cfg, moe_w4a16_wt_kernel<GATED, MW, NT, KS, NSTAGE, SPLITK, TMA, MODE>,""")
# wrapper: mode via ks digit >= 5 (ks is otherwise 1): ks = 10 + mode
rep("""    const int key = (tma ? 10000000 : 0) + (ksplit > 1 ? 1000000 : 0) + (gated ? 100000 : 0) + (int)mw * 10000 + (int)nt * 1000 + (int)ks * 100 + (int)nstage;
    switch (key) {""", """    const int mode = ks >= 10 ? (int)ks - 10 : 0;          // bench ablations: ks = 10 + MODE on the (1,1,1,8) variant
    if (mode > 0) {
        TORCH_CHECK(mw == 1 && nt == 1 && nstage == 8 && !tma && ksplit == 1, "ablation modes: (1,1,1,8) only");
#define OKC_WM(G, MD) launch_wt<G, 1, 1, 1, 8, false, false, MD>(Xp, (int)X.stride(0), sorted_tok.data_ptr<int>(), blk_e.data_ptr<int>(), \
        blk_row0.data_ptr<int>(), blk_end.data_ptr<int>(), nblk.data_ptr<int>(), Wp, Sp, gs.data_ptr<float>(), o16, obf, ldo, \
        pp, ldp, prow, 1, (int)K, (int)ntiles, (int)max_blocks, (float)gs_mul, pdl, stream)
        if (gated) { switch (mode) { case 1: OKC_WM(true, 1); break; case 2: OKC_WM(true, 2); break; case 3: OKC_WM(true, 3); break; default: OKC_WM(true, 4); break; } }
        else { switch (mode) { case 1: OKC_WM(false, 1); break; case 2: OKC_WM(false, 2); break; case 3: OKC_WM(false, 3); break; default: OKC_WM(false, 4); break; } }
#undef OKC_WM
        return;
    }
    const int key = (tma ? 10000000 : 0) + (ksplit > 1 ? 1000000 : 0) + (gated ? 100000 : 0) + (int)mw * 10000 + (int)nt * 1000 + (int)ks * 100 + (int)nstage;
    switch (key) {""")
p.write_text(s)

g = root / "tools/gen_moe_kernel_only.py"
u = g.read_text()
u = u.replace('''for g, m, t, s, n, sk, tm in re.findall(r"OKC_W\\((true|false), (\\d), (\\d), (\\d), (\\d), (true|false), (true|false)\\)", cu):
    inst.append(f"template void launch_wt<{g}, {m}, {t}, {s}, {n}, {sk}, {tm}>{wsig};")''',
              '''for g, m, t, s, n, sk, tm in re.findall(r"OKC_W\\((true|false), (\\d), (\\d), (\\d), (\\d), (true|false), (true|false)\\)", cu):
    inst.append(f"template void launch_wt<{g}, {m}, {t}, {s}, {n}, {sk}, {tm}, 0>{wsig};")
for g, md in re.findall(r"OKC_WM\\((true|false), (\\d)\\)", cu):
    inst.append(f"template void launch_wt<{g}, 1, 1, 1, 8, false, false, {md}>{wsig};")''')
g.write_text(u)

q = root / "muse_moe_w4a16/submission.py"
t = q.read_text()
old = """            print(f"  wt parts bs{bs} {(nt, mw2, ks, ns, ksp, tma)}: " + "  ".join(f"{k} {_timeit_cold(f, dev):6.1f}" for k, f in parts.items())
                  + f"  | local routes {int(((ids >= 0) & (ids < 32)).sum())} distinct {int(torch.unique(ids[(ids < 32)]).numel())}")
        except Exception as e:
            print(f"  wt parts bs{bs} failed: {str(e)[:100]}")"""
new = """            print(f"  wt parts bs{bs} {(nt, mw2, ks, ns, ksp, tma)}: " + "  ".join(f"{k} {_timeit_cold(f, dev):6.1f}" for k, f in parts.items())
                  + f"  | local routes {int(((ids >= 0) & (ids < 32)).sum())} distinct {int(torch.unique(ids[(ids < 32)]).numel())}")
            # ablations of the (1,1,1,8) kernels: 1 loads only, 2 no dequant, 3 no MMA, 4 no token tile
            for mode in (1, 2, 3, 4):
                t1 = _timeit_cold(lambda: ext.moe_w4a16_wt(x16, state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                                  state["Wt13"], state["St13"], state["gs13"], state["inter"], part, 1, 4096, 512, maxb, True, 1, 1, 10 + mode, 8, 64.0, False, False), dev)
                t2 = _timeit_cold(lambda: ext.moe_w4a16_wt(state["inter"], state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                                  state["Wt2"], state["St2"], state["gs2"], state["y"], part, 1, 4096, 256, maxb, False, 1, 1, 10 + mode, 8, 64.0, False, False), dev)
                print(f"    ablation mode {mode} ({'loads only' if mode == 1 else 'no dequant' if mode == 2 else 'no mma' if mode == 3 else 'no x tile'}): gemm1 {t1:6.1f}  gemm2 {t2:6.1f}")
        except Exception as e:
            print(f"  wt parts bs{bs} failed: {str(e)[:100]}")"""
assert t.count(old) == 1
t = t.replace(old, new)
# routing of the first forward (eager) for the ablation path may leave inter garbage: the bench only times
q.write_text(t)
print("edited")
