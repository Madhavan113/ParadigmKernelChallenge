"""Offline-compile muse_moe_w4a16's kernels for sm_90. Usage: python compile_moe.py [bs]"""
import sys, re
sys.path.insert(0, "/work/muse_moe_w4a16")
import triton, triton.language as tl
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
import submission as S

bs = int(sys.argv[1]) if len(sys.argv) > 1 else 16
BM, BN, BK, NW, NS = S._gemm_config(bs)
def count(pat, s): return len(re.findall(pat, s))
def build(fn, sig, consts, nw, ns, tag):
    names = list(sig)
    attrs = {(i,): [["tt.divisibility", 16]] for i, n in enumerate(names) if sig[n].startswith("*") or n.startswith("stride") or n in ("K", "N", "n_elems", "up_rows")}
    k = triton.compile(ASTSource(fn=fn, signature=sig, constexprs=consts, attrs=attrs), target=GPUTarget("cuda", 90, 32),
                       options={"num_warps": nw, "num_stages": ns})
    ttgir, ptx = k.asm["ttgir"], k.asm["ptx"]
    print(f"{tag}: async_copy {count(r'async_copy_global_to_local', ttgir)} | wgmma {count(r'wgmma.mma_async', ptx)} | mma.sync {count(r'mma.sync', ptx)}"
          f" | convert_layout {count(r'convert_layout', ttgir)} | cvt {count(r'cvt[.]', ptx)} | shared {k.metadata.shared} B"
          f" | local {bool(re.search(r'[.]local', ptx))} st.local {count(r'st[.]local', ptx)} | ptx lines {ptx.count(chr(10))}")
    open(f"/work/tools/out_moe_{tag}.ptx", "w").write(ptx)
    open(f"/work/tools/out_moe_{tag}.ttgir", "w").write(ttgir)

NP = max(16, triton.next_power_of_2(bs * 8)); MAXB = triton.next_power_of_2(NP // BM + 32)
build(S._route_kernel,
      {"ids_ptr": "*i32", "x_ptr": "*bf16", "x16_ptr": "*fp16", "sorted_tok_ptr": "*i32", "row_of_pair_ptr": "*i32",
       "blk_e_ptr": "*i32", "blk_row0_ptr": "*i32", "blk_end_ptr": "*i32", "nblk_ptr": "*i32", "sorted_pair_ptr": "*i32", "lids_ptr": "*i32", "npad_ptr": "*i32",
       "M": "i32", "local_start": "i32",
       "n_elems": "i32", "n_route_ctas": "i32", "xscale": "fp32", "PB": "constexpr", "CH": "constexpr", "BM": "constexpr", "MAXB": "constexpr", "E": "constexpr", "TOPK": "constexpr", "XBLOCK": "constexpr", "MARLIN": "constexpr", "MULTI": "constexpr", "PDL": "constexpr"},
      {"PB": min(NP, 64), "CH": min(NP, 256), "BM": BM, "MAXB": MAXB, "E": 32, "TOPK": 8, "XBLOCK": 4096, "MARLIN": True, "MULTI": NP > min(NP, 256), "PDL": True}, 4, 1, f"route_bs{bs}")
gsig = {"a_ptr": "*fp16", "w_ptr": "*u8", "sf_ptr": "*fp8e4nv", "gs_ptr": "*fp32", "out_ptr": "*fp16", "sorted_tok_ptr": "*i32",
        "blk_e_ptr": "*i32", "blk_row0_ptr": "*i32", "blk_end_ptr": "*i32", "nblk_ptr": "*i32", "stride_am": "i32", "stride_om": "i32",
        "K": "i32", "N": "i32", "stride_we": "i64", "stride_wn": "i32", "stride_se": "i64", "stride_sn": "i32", "up_rows": "i32",
        "NUM_CTAS": "i32", "n_tiles": "i32", "GATHER": "constexpr", "GATED": "constexpr", "ASM": "constexpr", "BM": "constexpr", "BN": "constexpr", "BK": "constexpr", "MODE": "constexpr", "SF_BLOCKED": "constexpr"}
for asm in (True,):
    build(S._moe_gemm_kernel, gsig, {"GATHER": True, "GATED": True, "ASM": asm, "BM": BM, "BN": BN, "BK": BK, "MODE": 0, "SF_BLOCKED": True}, NW, NS, f"gemm1_bs{bs}_asm{int(asm)}")
    build(S._moe_gemm_kernel, gsig, {"GATHER": True, "GATED": True, "ASM": asm, "BM": BM, "BN": BN, "BK": BK, "MODE": 6, "SF_BLOCKED": False}, NW, NS, f"gemm1_bs{bs}_MODE6")
    build(S._moe_gemm_kernel, gsig, {"GATHER": True, "GATED": True, "ASM": asm, "BM": BM, "BN": BN, "BK": BK, "MODE": 4, "SF_BLOCKED": False}, NW, NS, f"gemm1_bs{bs}_MODE4")
    build(S._moe_gemm_kernel, gsig, {"GATHER": True, "GATED": True, "ASM": asm, "BM": BM, "BN": BN, "BK": BK, "MODE": 7, "SF_BLOCKED": False}, NW, NS, f"gemm1_bs{bs}_MODE7")
    build(S._moe_gemm_kernel, gsig, {"GATHER": True, "GATED": True, "ASM": asm, "BM": BM, "BN": BN, "BK": BK, "MODE": 8, "SF_BLOCKED": False}, NW, NS, f"gemm1_bs{bs}_MODE8")
    build(S._moe_gemm_kernel, gsig, {"GATHER": True, "GATED": True, "ASM": asm, "BM": BM, "BN": BN, "BK": BK, "MODE": 11, "SF_BLOCKED": False}, NW, NS, f"gemm1_bs{bs}_MODE11")
gsig2 = dict(gsig, out_ptr="*bf16")
build(S._moe_gemm_kernel, gsig2, {"GATHER": False, "GATED": False, "ASM": True, "BM": BM, "BN": BN, "BK": BK, "MODE": 0, "SF_BLOCKED": True}, NW, NS, f"gemm2_bs{bs}_asm1")
build(S._finalize_kernel, {"y_ptr": "*bf16", "row_of_pair_ptr": "*i32", "tw_ptr": "*fp32", "nw_ptr": "*fp32", "out_ptr": "*bf16", "eps": "fp32",
                           "H": "constexpr", "TOPK": "constexpr"}, {"H": 4096, "TOPK": 8}, 4, 1, "finalize")
