"""Offline nvcc (sm_90a) of muse_bf16_proj/cuda_proj.py's CUDA_SRC inside the judge image: syntax, registers, spills.
Usage: python3 nvcc_proj.py  (run from /work/tools in the image)"""
import subprocess, sys, pathlib
sys.path.insert(0, "/work/muse_bf16_proj")
import importlib.util
spec = importlib.util.spec_from_file_location("cp_mod", sys.argv[1]); C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
import torch.utils.cpp_extension as E
src = "#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n" + C.CUDA_SRC
p = pathlib.Path("/tmp/proj_kernel_only_" + sys.argv[2] + ".cu"); p.write_text(src)
import sysconfig
inc = sum([["-I", d] for d in E.include_paths(device_type="cuda") + [sysconfig.get_paths()["include"]]], [])
cmd = ["nvcc", "-arch=sm_90a", "-O3", "-std=c++20", "-c", "-o", "/tmp/proj_" + sys.argv[2] + ".o", "-Xptxas", "-v", "-DTORCH_API_INCLUDE_EXTENSION_H",
       "-D_GLIBCXX_USE_CXX11_ABI=" + ("1" if getattr(E, "_GLIBCXX_USE_CXX11_ABI", True) else "0")] + inc + [str(p)]
r = subprocess.run(cmd, capture_output=True, text=True)
out = r.stdout + r.stderr
lines = [l for l in out.splitlines() if "error" in l.lower() or "Used" in l or "Function properties" in l or "spill" in l or "Compiling entry" in l]
print("\n".join(lines))
print("rc", r.returncode)
import re
sass = subprocess.run(["cuobjdump", "-sass", "/tmp/proj_" + sys.argv[2] + ".o"], capture_output=True, text=True).stdout
for name in re.findall(r"Function : (\S*w12_kernel\S*)", sass):
    i = sass.index("Function : " + name); j = sass.find("Function : ", i + 10)
    mm = re.search(r"w12_kernelILi(\d)ELi(\d)ELb([01])ELi(\d)E", name)
    if mm:
        pathlib.Path("/work/tools/sass_" + sys.argv[2] + "_nt" + mm.group(1) + "_ns" + mm.group(2) + "_sk" + mm.group(3) + "_mode" + mm.group(4) + ".txt").write_text(sass[i:j if j > 0 else None])
