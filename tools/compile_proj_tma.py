import sys, re
sys.path.insert(0, "/work/muse_bf16_proj")
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
import submission as S
KERNEL = S._tiled_gemm_tma_kernel if "--tiled" in sys.argv else S._skinny_gemm_tma_kernel
a = [x for x in sys.argv[1:] if x != "--tiled"]
BM, BN, BK, SK, TW, NW, NS = [int(x) for x in (a + ["16", "32", "256", "1", "0", "4", "4"][len(a):])]
sig = {"x_desc": f"tensordesc<bf16[{BM}, {BK}]>", "w_desc": f"tensordesc<bf16[{BN}, {BK}]>", "w2_desc": f"tensordesc<bf16[{BN}, {BK}]>",
       "out_ptr": "*bf16", "part_ptr": "*fp32", "cnt_ptr": "*i32", "M": "i32", "N": "i32", "K": "i32", "stride_om": "i32",
       "BLOCK_M": "constexpr", "BLOCK_N": "constexpr", "BLOCK_K": "constexpr", "SPLIT_K": "constexpr", "TWO_W": "constexpr"}
consts = {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK, "SPLIT_K": SK, "TWO_W": bool(TW)}
names = list(sig)
attrs = {(i,): [["tt.divisibility", 16]] for i, n in enumerate(names) if sig[n].startswith("*") or n in ("N", "K", "stride_om")}
k = triton.compile(ASTSource(fn=KERNEL, signature=sig, constexprs=consts, attrs=attrs),
                   target=GPUTarget("cuda", 90, 32), options={"num_warps": NW, "num_stages": NS})
ttgir, ptx = k.asm["ttgir"], k.asm["ptx"]
def count(pat, s): return len(re.findall(pat, s))
print(f"TMA config BM={BM} BN={BN} BK={BK} SK={SK} TWO_W={TW} warps={NW} stages={NS}")
print("ttgir: async_tma_copy", count(r"async_tma_copy_global_to_local", ttgir), "| wait_barrier", count(r"wait_barrier", ttgir),
      "| warp_group_dot", count(r"warp_group_dot", ttgir), "| local_alloc", count(r"local_alloc", ttgir))
print("ptx: cp.async.bulk.tensor", count(r"cp\.async\.bulk\.tensor", ptx), "| mbarrier", count(r"mbarrier", ptx), "| wgmma", count(r"wgmma\.mma_async", ptx),
      "| mma.sync", count(r"mma\.sync", ptx), "| ldmatrix", count(r"ldmatrix", ptx), "| shared", k.metadata.shared, "| local", bool(re.search(r"\.local", ptx)))
open(f"/work/tools/out_tma_{BM}_{BN}_{BK}_{SK}_{TW}_{NW}_{NS}.ptx", "w").write(ptx)
