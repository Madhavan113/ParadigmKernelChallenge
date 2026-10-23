"""Offline-compile muse_attention_decode's kernel for sm_90. Usage: python compile_attn.py L WINDOW SPLITS BLOCK_T [HQ HKV PS warps stages]"""
import sys, re
sys.path.insert(0, "/work/muse_attention_decode")
import triton, triton.language as tl
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
import submission as S

a = sys.argv[1:]
L, WINDOW, SPLITS, BLOCK_T, HQ, HKV, PS, NW, NS = [int(x) for x in (a + ["1", "2048", "16", "64", "16", "2", "1", "4", "3"][len(a):])]
D = 64
RB = max(16, triton.next_power_of_2(L * (HQ // HKV)))
sig = {"qkv_ptr": "*bf16", "loc_ptr": "*i64", "seq_ptr": "*i32", "pt_ptr": "*i32", "kc_ptr": "*fp8e4nv", "vc_ptr": "*fp8e4nv",
       "kscale_ptr": "*fp32", "vscale_ptr": "*fp32", "out_ptr": "*bf16", "part_ptr": "*fp32", "ml_ptr": "*fp32", "cnt_ptr": "*i32",
       "max_pages": "i32", "qk_scale": "fp32", "sm_scale_log2": "fp32", "log2_theta": "fp32",
       "L": "constexpr", "HQ": "constexpr", "HKV": "constexpr", "D": "constexpr", "PS": "constexpr", "WINDOW": "constexpr",
       "ROPE": "constexpr", "SPLITS": "constexpr", "RB": "constexpr", "LB": "constexpr", "BLOCK_T": "constexpr", "ACT": "constexpr", "FP8_ASM": "constexpr"}
consts = {"L": L, "HQ": HQ, "HKV": HKV, "D": D, "PS": PS, "WINDOW": WINDOW, "ROPE": WINDOW > 0, "SPLITS": SPLITS,
          "RB": RB, "LB": 16, "BLOCK_T": BLOCK_T, "ACT": tl.float16, "FP8_ASM": True}
names = list(sig)
attrs = {(i,): [["tt.divisibility", 16]] for i, n in enumerate(names) if sig[n].startswith("*")}
src = ASTSource(fn=S._attn_decode_kernel, signature=sig, constexprs=consts, attrs=attrs)
k = triton.compile(src, target=GPUTarget("cuda", 90, 32), options={"num_warps": NW, "num_stages": NS})
ttgir = k.asm["ttgir"]; ptx = k.asm["ptx"]
def count(pat, s): return len(re.findall(pat, s))
print(f"config L={L} WINDOW={WINDOW} SPLITS={SPLITS} BLOCK_T={BLOCK_T} HQ={HQ} HKV={HKV} PS={PS} RB={RB} warps={NW} stages={NS}")
print("ttgir: async_copy", count(r"async_copy_global_to_local", ttgir), "| warp_group_dot", count(r"warp_group_dot", ttgir),
      "| convert_layout", count(r"convert_layout", ttgir))
print("ptx: cp.async", count(r"cp\.async\.c", ptx), "| wgmma", count(r"wgmma\.mma_async", ptx), "| mma.sync", count(r"mma\.sync", ptx),
      "| ld.global", count(r"ld\.global", ptx), "| cvt e4m3", count(r"e4m3", ptx), "| local(spill)", bool(re.search(r"\.local", ptx)),
      "| shared", k.metadata.shared, "B")
open(f"/work/tools/out_attn_{L}_{WINDOW}_{SPLITS}_{BLOCK_T}_{HQ}_{PS}.ptx", "w").write(ptx)
