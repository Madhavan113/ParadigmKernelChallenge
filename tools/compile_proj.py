"""Offline-compile muse_bf16_proj's Triton kernel for sm_90 (no GPU needed) and report what
the pipeline lowered to. Usage: python compile_proj.py [BLOCK_M BLOCK_N BLOCK_K SPLIT_K TWO_W warps stages]"""
import sys, re
sys.path.insert(0, "/work/muse_bf16_proj")
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
import submission as S

a = sys.argv[1:]
BM, BN, BK, SK, TW, NW, NS = [int(x) for x in (a + ["16", "32", "256", "1", "0", "4", "4"][len(a):])]
sig = {"x_ptr": "*bf16", "w_ptr": "*bf16", "w2_ptr": "*bf16", "out_ptr": "*bf16", "part_ptr": "*fp32",
       "cnt_ptr": "*i32", "M": "i32", "N": "i32", "K": "i32", "stride_xm": "i32", "stride_wn": "i32",
       "stride_om": "i32", "BLOCK_M": "constexpr", "BLOCK_N": "constexpr", "BLOCK_K": "constexpr",
       "SPLIT_K": "constexpr", "TWO_W": "constexpr"}
consts = {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK, "SPLIT_K": SK, "TWO_W": bool(TW)}
# the JIT's specialization: every pointer and stride is 16-byte aligned / divisible by 16
names = list(sig)
attrs = {(i,): [["tt.divisibility", 16]] for i, n in enumerate(names)
         if sig[n].startswith("*") or n.startswith("stride") or n in ("N", "K")}
src = ASTSource(fn=S._skinny_gemm_kernel, signature=sig, constexprs=consts, attrs=attrs)
k = triton.compile(src, target=GPUTarget("cuda", 90, 32), options={"num_warps": NW, "num_stages": NS})
ttgir = k.asm["ttgir"]; ptx = k.asm["ptx"]
def count(pat, s): return len(re.findall(pat, s))
print(f"config BM={BM} BN={BN} BK={BK} SK={SK} TWO_W={TW} warps={NW} stages={NS}")
print("ttgir: async_copy_global_to_local", count(r"async_copy_global_to_local", ttgir),
      "| async_wait", count(r"async_wait", ttgir), "| async_tma", count(r"async_tma", ttgir),
      "| warp_group_dot", count(r"warp_group_dot", ttgir), "| local_alloc", count(r"local_alloc", ttgir))
print("ptx: cp.async", count(r"cp\.async\.cg|cp\.async\.ca", ptx), "| cp.async.commit", count(r"cp\.async\.commit", ptx),
      "| wgmma", count(r"wgmma\.mma_async", ptx), "| mma.sync", count(r"mma\.sync", ptx),
      "| ldmatrix", count(r"ldmatrix", ptx), "| ld.global", count(r"ld\.global", ptx),
      "| ld.volatile", count(r"ld\.volatile", ptx), "| ld.global.cg", count(r"ld\.global\.cg", ptx))
m = re.search(r"\.reqntid (\d+)", ptx); r = re.search(r"// (\d+) bytes stack|\.local", ptx)
print("regs/spills hint:", k.metadata.num_warps, "warps; shared", k.metadata.shared, "bytes; spill-local:", bool(re.search(r"\.local", ptx)))
open(f"/work/tools/out_proj_{BM}_{BN}_{BK}_{SK}_{TW}_{NW}_{NS}.ttgir", "w").write(ttgir)
open(f"/work/tools/out_proj_{BM}_{BN}_{BK}_{SK}_{TW}_{NW}_{NS}.ptx", "w").write(ptx)
