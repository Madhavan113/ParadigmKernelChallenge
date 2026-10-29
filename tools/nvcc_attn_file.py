#!/usr/bin/env python3
"""Torch-free offline nvcc (sm_90a) of a given attention submission file's CUDA_SRC, SASS dumped per kernel to a
directory, for comparing two file states instruction by instruction (register numbers included). Runs inside the judge
image from /work:  python3 tools/nvcc_attn_file.py <submission.py> <outdir> [-DOKC_X=1 ...]
"""
import glob, os, re, subprocess, sys

src, outdir = sys.argv[1], sys.argv[2]
flags = [a for a in sys.argv[3:] if a.startswith("-D")]
text = open(src).read()
m = re.search(r'CUDA_SRC = r"""(.*?)"""\n', text, re.S)
assert m, "CUDA_SRC not found"
os.makedirs(outdir, exist_ok=True)
cu = os.path.join(outdir, "attn_tu.cu")
open(cu, "w").write(m.group(1))
site = glob.glob("/opt/sglang/lib/python3.*/site-packages")[0]
inc = [f"-I{site}/torch/include", f"-I{site}/torch/include/torch/csrc/api/include"]
pyinc = glob.glob("/usr/include/python3.*") + glob.glob("/opt/sglang/include/python3.*")
inc += [f"-I{p}" for p in pyinc[:1]]
cubin = os.path.join(outdir, "attn_tu.cubin")
cmd = ["nvcc", "-O3", "-gencode=arch=compute_90a,code=sm_90a", "-DOKC_SM90=1", "--expt-relaxed-constexpr",
       "-DTORCH_EXTENSION_NAME=okc_attn_chk", "-DTORCH_API_INCLUDE_EXTENSION_H", "-D_GLIBCXX_USE_CXX11_ABI=1",
       "-std=c++20", "-cubin", "-o", cubin, cu, *inc, *flags]
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode != 0:
    print("BUILD FAILED", r.stderr[-2000:])
    sys.exit(1)
sass = subprocess.run(["cuobjdump", "-sass", cubin], capture_output=True, text=True).stdout
func, lines = None, {}
for line in sass.splitlines():
    mm = re.match(r"\s*Function : (\S+)", line)
    if mm:
        func = mm.group(1)
        lines[func] = []
    elif func is not None:
        mm = re.match(r"\s*/\*[0-9a-f]{4,}\*/\s*(.*?)\s*;", line)
        if mm:
            lines[func].append(mm.group(1))
for f, ls in lines.items():
    short = re.sub(r"[^A-Za-z0-9_]", "_", f)[-60:]
    open(os.path.join(outdir, short + ".sass"), "w").write("\n".join(ls) + "\n")
    print(f"{len(ls):6d} instr  {f[-50:]}")
