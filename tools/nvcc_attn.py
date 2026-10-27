"""Torch-free offline nvcc of muse_attention_decode/submission.py's CUDA_SRC (sm_90a, ptxas -v, SASS sizes).
Runs inside the judge image from /work:  python3 tools/nvcc_attn.py -DOKC_ST1F=4 ...
"""
import re, sys, subprocess, time, collections, glob

SRC = "/work/muse_attention_decode/submission.py"
text = open(SRC).read()
m = re.search(r'CUDA_SRC = r"""(.*?)"""\n', text, re.S)
assert m, "CUDA_SRC not found"
cu = m.group(1)
flags = [a for a in sys.argv[1:] if a.startswith("-D")]
open("/tmp/attn_tu.cu", "w").write(cu)
site = glob.glob("/opt/sglang/lib/python3.*/site-packages")[0]
inc = [f"-I{site}/torch/include", f"-I{site}/torch/include/torch/csrc/api/include"]
pyinc = glob.glob("/usr/include/python3.*") + glob.glob("/opt/sglang/include/python3.*")
inc += [f"-I{p}" for p in pyinc[:1]]
cmd = ["nvcc", "-O3", "-gencode=arch=compute_90a,code=sm_90a", "-DOKC_SM90=1", "--expt-relaxed-constexpr",
       "-DTORCH_EXTENSION_NAME=okc_attn_chk", "-DTORCH_API_INCLUDE_EXTENSION_H", "-D_GLIBCXX_USE_CXX11_ABI=1",
       "-std=c++20", "-Xptxas", "-v", "-cubin", "-o", "/tmp/attn_tu.cubin", "/tmp/attn_tu.cu", *inc, *flags]
t0 = time.time()
r = subprocess.run(cmd, capture_output=True, text=True)
out = r.stdout + r.stderr
if r.returncode != 0:
    print(f"BUILD FAILED in {time.time() - t0:.0f}s flags={flags}")
    for line in out.splitlines():
        if re.search(r"error|warning #|C7515", line):
            print(line[:400])
    print(out[-2500:])
    sys.exit(1)
print(f"BUILD OK in {time.time() - t0:.0f}s flags={flags}")
func = None
for line in out.splitlines():
    mm = re.search(r"Compiling entry function '(\S+)'", line)
    if mm:
        func = mm.group(1)
    elif "Used" in line and func:
        print(f"REGS {func[-40:]}: {line.strip()[:120]}")
    elif re.search(r"spill stores [1-9]|spill loads [1-9]|C7515|warning #", line):
        print("WARN", line.strip()[:300])
sass = subprocess.run(["cuobjdump", "-sass", "/tmp/attn_tu.cubin"], capture_output=True, text=True).stdout
counts = collections.OrderedDict()
for line in sass.splitlines():
    mm = re.match(r"\s*Function : (\S+)", line)
    if mm:
        func = mm.group(1); counts[func] = 0
    elif func and re.match(r"\s*/\*[0-9a-f]{4,}\*/", line):
        counts[func] += 1
for f, n in counts.items():
    print(f"SASS {n:6d} instr = {n * 16 / 1024:7.1f} KB  {f[-40:]}")
