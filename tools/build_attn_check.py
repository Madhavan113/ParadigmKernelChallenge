"""Offline nvcc of muse_attention_decode/submission.py's CUDA_SRC inside the judge image (no GPU): ptxas -v summary
(registers, spills, C7515 warnings) per kernel. Extra -D flags from argv. Run from /work (repo root in the container):

  docker run --rm --platform linux/amd64 -v "$ROOT":/work -w /work lmsysorg/sglang:dev-muse-spark-open \
      bash -c "TORCH_CUDA_ARCH_LIST=9.0a python3 tools/build_attn_check.py -DOKC_ST1F=4"
"""
import os, sys, re, time, subprocess, collections
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
sys.path.insert(0, "/work/muse_attention_decode")
import torch
from torch.utils.cpp_extension import load_inline
import submission as S

flags = [a for a in sys.argv[1:] if a.startswith("-D")]
tag = re.sub(r"[^A-Za-z0-9]", "_", "".join(flags))[:60]
t0 = time.time()
try:
    ext = load_inline(
        name=f"okc_attn_chk{tag}",
        cpp_sources=S.CPP_SRC,
        cuda_sources=S.CUDA_SRC,
        functions=["run"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-DOKC_SM90=1", "--expt-relaxed-constexpr",
                           "-Xptxas", "-v", *flags],
        verbose=True)
    print(f"BUILD OK in {time.time() - t0:.0f}s flags={flags}")
    sass = subprocess.run(["cuobjdump", "-sass", ext.__file__], capture_output=True, text=True).stdout
    func, counts = None, collections.OrderedDict()
    for line in sass.splitlines():
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            func = m.group(1); counts[func] = 0
        elif func and re.match(r"\s*/\*[0-9a-f]{4,}\*/", line):
            counts[func] += 1
    for f, n in counts.items():
        print(f"SASS {n:6d} instr = {n * 16 / 1024:7.1f} KB  {f[:100]}")
except Exception as e:
    msg = str(e)
    print(f"BUILD FAILED in {time.time() - t0:.0f}s flags={flags}")
    for line in msg.splitlines():
        if re.search(r"error|warning #|Used \d+ registers|spill|ptxas info|C7515", line):
            print(line[:400])
    print(msg[-3000:])
