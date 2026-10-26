"""Offline nvcc (sm_90a) of muse_bf16_proj/cuda_proj.py's CUDA_SRC inside the judge image: syntax, registers, spills.
Usage: python3 nvcc_proj.py  (run from /work/tools in the image)"""
import subprocess, sys, pathlib
sys.path.insert(0, "/work/muse_bf16_proj")
import cuda_proj as C
import torch.utils.cpp_extension as E
src = "#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n" + C.CUDA_SRC
p = pathlib.Path("/work/tools/proj_kernel_only.cu"); p.write_text(src)
import sysconfig
inc = sum([["-I", d] for d in E.include_paths(device_type="cuda") + [sysconfig.get_paths()["include"]]], [])
cmd = ["nvcc", "-arch=sm_90a", "-O3", "-std=c++20", "-c", "-o", "/tmp/proj.o", "-Xptxas", "-v", "-DTORCH_API_INCLUDE_EXTENSION_H",
       "-D_GLIBCXX_USE_CXX11_ABI=" + ("1" if getattr(E, "_GLIBCXX_USE_CXX11_ABI", True) else "0")] + inc + [str(p)]
r = subprocess.run(cmd, capture_output=True, text=True)
out = r.stdout + r.stderr
lines = [l for l in out.splitlines() if "error" in l.lower() or "Used" in l or "Function properties" in l or "spill" in l or "Compiling entry" in l]
print("\n".join(lines[-200:]))
print("rc", r.returncode)
