"""Regenerate tools/moe_kernel_only.cu from muse_moe_w4a16/cuda_moe.py (device code only, explicit template
instantiations) for an offline `nvcc -cubin -Xptxas -v` in the judge image."""
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
wsig = ("(const __half*, int, const int*, const int*, const int*, const int*, const int*, const uint8_t*, const uint8_t*, "
        "const float*, __half*, __nv_bfloat16*, int, float*, int, int, int*, int, int, int, int, float, cudaStream_t)")
for g, m, t, s, n in re.findall(r"OKC_W\((true|false), (\d), (\d), (\d), (\d)\)", cu):
    inst.append(f"template void launch_wt<{g}, {m}, {t}, {s}, {n}>{wsig};")
# keep the finalize kernel referenced
inst.append("void* finalize_ref = (void*)&moe_finalize_kernel;")
out = body + "\n".join(inst) + "\n} // namespace\n"
(root / "tools/moe_kernel_only.cu").write_text(out)
print("wrote", len(out.splitlines()), "lines,", len(inst), "instantiations")
