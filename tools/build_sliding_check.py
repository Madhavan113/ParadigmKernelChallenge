"""Compile the sliding-window CUDA kernel inside the judge image without a GPU (nvcc + ptxas -v).

  TORCH_CUDA_ARCH_LIST=9.0a python3 build_sliding_check.py
"""
import os, sys, re
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
sys.path.insert(0, "/work/muse_attention_decode")
import torch
from torch.utils.cpp_extension import load_inline
import submission as S

try:
    ext = load_inline(
        name="muse_attn_sliding_check",
        cpp_sources=[S.CPP_SLIDING_SRC],
        cuda_sources=["#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n" + S.CUDA_SLIDING_SRC],
        functions=["sliding_attn"],
        extra_cuda_cflags=["-O3", "-arch=sm_90a", "-Xptxas", "-v"], verbose=True)
    print("BUILD OK", ext)
    # SASS size per kernel: instructions x 16 B (H100 L1.5 instruction cache ~128 KB per SM)
    import subprocess, collections
    sass = subprocess.run(["cuobjdump", "-sass", ext.__file__], capture_output=True, text=True).stdout
    func, counts = None, collections.OrderedDict()
    for line in sass.splitlines():
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            func = m.group(1); counts[func] = 0
        elif func and re.match(r"\s*/\*[0-9a-f]{4,}\*/", line):
            counts[func] += 1
    for f, n in counts.items():
        print(f"SASS {n:6d} instructions = {n * 16 / 1024:7.1f} KB  {f[:90]}")
except Exception as e:
    msg = str(e)
    print("BUILD FAILED")
    # the interesting lines: errors and ptxas resource usage
    for line in msg.splitlines():
        if re.search(r"error|warning #|Used \d+ registers|spill|ptxas info", line):
            print(line[:400])
    print(msg[-3000:])
