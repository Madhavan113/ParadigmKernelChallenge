"""Offline-compile the decode layer's own kernels for sm_90."""
import sys, re
sys.path.insert(0, "/work/muse_decode_layer")
import triton, triton.language as tl
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
import submission as S
def count(pat, s): return len(re.findall(pat, s))
def build(fn, sig, consts, nw, tag):
    names = list(sig)
    attrs = {(i,): [["tt.divisibility", 16]] for i, n in enumerate(names) if sig[n].startswith("*") or n.startswith("stride")}
    k = triton.compile(ASTSource(fn=fn, signature=sig, constexprs=consts, attrs=attrs), target=GPUTarget("cuda", 90, 32),
                       options={"num_warps": nw, "num_stages": 1})
    ptx = k.asm["ptx"]
    print(f"{tag}: ld.global {count(r'ld[.]global', ptx)} | st.global {count(r'st[.]global', ptx)} | shared {k.metadata.shared} B"
          f" | local {bool(re.search(r'[.]local', ptx))} | ptx lines {ptx.count(chr(10))}")
build(S._boundary_kernel, {"stream_ptr": "*fp32", "br_ptr": "*bf16", "gain_ptr": "*fp32", "alpha_ptr": "*fp32", "beta_ptr": "*fp32",
      "next_ptr": "*fp32", "s_out_ptr": "*fp32", "h_out_ptr": "*bf16", "part_ptr": "*fp32", "cnt_ptr": "*i32", "stride_br": "i32",
      "eps_post": "fp32", "eps_pre": "fp32", "H": "constexpr", "BR_N": "constexpr", "HAS_GAIN": "constexpr", "G": "constexpr"}, {"H": 8192, "BR_N": 1024, "HAS_GAIN": True, "G": 8}, 4, "boundary_sliced_g8")
build(S._boundary_kernel, {"stream_ptr": "*fp32", "br_ptr": "*bf16", "gain_ptr": "*fp32", "alpha_ptr": "*fp32", "beta_ptr": "*fp32",
      "next_ptr": "*fp32", "s_out_ptr": "*fp32", "h_out_ptr": "*bf16", "part_ptr": "*fp32", "cnt_ptr": "*i32", "stride_br": "i32",
      "eps_post": "fp32", "eps_pre": "fp32", "H": "constexpr", "BR_N": "constexpr", "HAS_GAIN": "constexpr", "G": "constexpr"}, {"H": 8192, "BR_N": 8192, "HAS_GAIN": False, "G": 4}, 4, "boundary_attn_g4")
build(S._route_rms_kernel, {"logit_ptr": "*fp32", "bias_ptr": "*fp32", "ls_ptr": "*i32", "ids_ptr": "*i32", "lids_ptr": "*i32", "w_ptr": "*fp32",
      "x_ptr": "*bf16", "gain_ptr": "*fp32", "out_ptr": "*bf16", "mult": "fp32", "eps": "fp32", "eps_n": "fp32",
      "E": "constexpr", "TOPK": "constexpr", "N": "constexpr"}, {"E": 256, "TOPK": 8, "N": 4096}, 4, "route_rms")
build(S._rms_gain_kernel, {"x_ptr": "*bf16", "gain_ptr": "*fp32", "out_ptr": "*bf16", "part_ptr": "*fp32", "cnt_ptr": "*i32", "eps": "fp32",
      "N": "constexpr", "G": "constexpr"}, {"N": 4096, "G": 1}, 4, "rms_gain_g1")
