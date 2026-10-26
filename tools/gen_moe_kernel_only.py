"""Regenerate tools/moe_kernel_only.cu from muse_moe_w4a16/cuda_moe.py (device code only, explicit template
instantiations of every OKC_L / OKC_W case) for an offline `nvcc -cubin -Xptxas -v` in the judge image:

  docker run --rm --platform linux/amd64 -v "$PWD":/work -w /work/tools lmsysorg/sglang:dev-muse-spark-open \
    bash -c "nvcc -arch=sm_90a -O3 --use_fast_math -cubin -Xptxas -v -o /tmp/moe.cubin moe_kernel_only.cu && cuobjdump -sass /tmp/moe.cubin > sass.txt"
"""
import pathlib, re
root = pathlib.Path(__file__).resolve().parent.parent
src = (root / "muse_moe_w4a16/cuda_moe.py").read_text()
a = src.index('CUDA_SRC = r"""') + len('CUDA_SRC = r"""')
b = src.index('"""', a)
cu = src[a:b]
ns = cu.index("} // namespace")
body = cu[:ns]
inst = []
sig = ("(const __half*, int, const int*, const int*, const int*, const int*, const int*, const uint4*, const uint2*, "
       "const float*, __half*, __nv_bfloat16*, float*, int, int, int, int, int, cudaStream_t)")
for g, m, p, s in re.findall(r"OKC_L\((true|false), (\d), (\d), (\d)\)", cu):
    inst.append(f"template void launch<{g}, {m}, {p}, {s}>{sig};")
# launch_wt's parameter list, taken from the source so the instantiation tracks signature changes
m = re.search(r"void launch_wt\((.*?)\)\n\{", cu, re.S)
params = [p.strip() for p in m.group(1).replace("\n", " ").split(",")]
types = []
for p in params:
    p = re.sub(r"\b\w+$", "", p).strip()          # drop the parameter name
    types.append(p)
wsig = "(" + ", ".join(types) + ")"
for args in re.findall(r"OKC_W\(([^)]*)\)", cu):
    args = [x.strip() for x in args.split(",")]
    if args[0] not in ("true", "false"): continue      # the macro definition itself
    inst.append(f"template void launch_wt<{', '.join(args)}>{wsig};")
inst.append("void* finalize_ref = (void*)&moe_finalize_kernel<4096>;")
out = "#ifndef TORCH_CHECK\n#define TORCH_CHECK(...)\n#endif\n" + body + "\n".join(inst) + "\n} // namespace\n"
(root / "tools/moe_kernel_only.cu").write_text(out)
print("wrote", len(out.splitlines()), "lines,", len(inst), "instantiations")
