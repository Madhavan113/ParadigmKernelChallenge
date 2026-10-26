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
except Exception as e:
    msg = str(e)
    print("BUILD FAILED")
    # the interesting lines: errors and ptxas resource usage
    for line in msg.splitlines():
        if re.search(r"error|warning #|Used \d+ registers|spill|ptxas info", line):
            print(line[:400])
    print(msg[-3000:])
