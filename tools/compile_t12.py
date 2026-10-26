"""Offline sm_90 compile of _tiled12_gemm_kernel (12-bit packed weight, decode before tl.dot): reports
whether the plane loads are pipelined (cp.async), the prmt count and shared memory.
Usage: python3 compile_t12.py BLOCK_M SPLIT_K warps stages [--noasm]"""
import sys, re
sys.path.insert(0, "/work/muse_bf16_proj")
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
import submission as S
a = [x for x in sys.argv[1:] if not x.startswith("--")]
BM, SK, NW, NS = [int(x) for x in (a + ["16", "1", "4", "4"][len(a):])]
sig = {"x_ptr": "*bf16", "lo_ptr": "*i32", "nb_ptr": "*i16", "out_ptr": "*bf16", "part_ptr": "*fp32", "cnt_ptr": "*i32",
       "esc_off_ptr": "*i32", "esc_n_ptr": "*i32", "esc_k_ptr": "*i32", "esc_d_ptr": "*fp32",
       "M": "i32", "N": "i32", "K": "i32", "stride_xm": "i32", "stride_om": "i32", "EB": "i32",
       "BLOCK_M": "constexpr", "SPLIT_K": "constexpr", "ASM": "constexpr", "INTERP": "constexpr"}
consts = {"BLOCK_M": BM, "SPLIT_K": SK, "ASM": "--noasm" not in sys.argv, "INTERP": False}
names = list(sig)
attrs = {(i,): [["tt.divisibility", 16]] for i, n in enumerate(names) if sig[n].startswith("*") or n.startswith("stride") or n in ("N", "K")}
k = triton.compile(ASTSource(fn=S._tiled12_gemm_kernel, signature=sig, constexprs=consts, attrs=attrs),
                   target=GPUTarget("cuda", 90, 32), options={"num_warps": NW, "num_stages": NS})
ttgir, ptx = k.asm["ttgir"], k.asm["ptx"]
def count(pat, s): return len(re.findall(pat, s))
print(f"t12 config BM={BM} SK={SK} warps={NW} stages={NS} asm={consts['ASM']}")
print("ttgir: async_copy_global_to_local", count(r"async_copy_global_to_local", ttgir), "| async_wait", count(r"async_wait", ttgir),
      "| convert_layout", count(r"convert_layout", ttgir), "| local_alloc", count(r"local_alloc", ttgir), "| local_store", count(r"local_store", ttgir))
print("ptx: cp.async", count(r"cp\.async\.cg|cp\.async\.ca", ptx), "| prmt", count(r"prmt\.b32", ptx), "| mma.sync", count(r"mma\.sync", ptx),
      "| ldmatrix", count(r"ldmatrix", ptx), "| st.shared", count(r"st\.shared", ptx), "| ld.shared", count(r"ld\.shared", ptx),
      "| ld.global", count(r"ld\.global", ptx), "| shared", k.metadata.shared, "| local", bool(re.search(r"\.local", ptx)), "| ptx lines", ptx.count("\n"))
open(f"/work/tools/out_t12_{BM}_{SK}_{NW}_{NS}.ptx", "w").write(ptx)
open(f"/work/tools/out_t12_{BM}_{SK}_{NW}_{NS}.ttgir", "w").write(ttgir)
