"""Offline nvcc (sm_90a) of muse_moe_w4a16/cuda_moe.py's FULL CUDA_SRC (host wrappers included, torch headers) inside
the judge image: host + device compile errors, registers, spills. The kernel-only tool (gen_moe_kernel_only.py) skips the
host code, which is where check 6ab9edd6 failed (a data_ptr<long long> instantiation libtorch does not export).
Usage: python3 nvcc_moe.py  (run from /work/tools in the image)"""
import subprocess, sys, pathlib, re
sys.path.insert(0, "/work/muse_moe_w4a16")
import cuda_moe as C
import torch.utils.cpp_extension as E
src = "#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n" + C.CUDA_SRC
p = pathlib.Path("/work/tools/moe_full_only.cu"); p.write_text(src)
import sysconfig
inc = sum([["-I", d] for d in E.include_paths(device_type="cuda") + [sysconfig.get_paths()["include"]]], [])
cmd = ["nvcc", "-gencode", "arch=compute_90a,code=sm_90a", "-O3", "--use_fast_math", "-std=c++20", "-c", "-o", "/tmp/moe_full.o", "-Xptxas", "-v",
       "-DTORCH_API_INCLUDE_EXTENSION_H", "-D_GLIBCXX_USE_CXX11_ABI=1"] + inc + [str(p)]
r = subprocess.run(cmd, capture_output=True, text=True)
out = r.stdout + r.stderr
lines = [l for l in out.splitlines() if "error" in l.lower() or "spill" in l]
print("\n".join(lines[:40]))
# libtorch exports data_ptr<T>() only for its dtypes: flag unexported instantiations
bad = re.findall(r"data_ptr<(long long|unsigned long long|unsigned int|unsigned long)>", C.CUDA_SRC)
print("unexported data_ptr instantiations:", bad)
print("rc", r.returncode, "entries", out.count("Compiling entry"))
sys.exit(1 if (r.returncode or bad) else 0)
