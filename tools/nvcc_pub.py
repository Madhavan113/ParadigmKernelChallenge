"""Offline nvcc (sm_90a) of muse_bf16_proj/submission.py's FULL CUDA translation unit (the same string _get_ext builds),
inside the judge image: host + device errors, registers, spills per kernel. Usage (from /work/tools in the image):
    python3 nvcc_pub.py [grep-pattern-for-kernels]"""
import subprocess, sys, pathlib, re, importlib.util
if "--build" in sys.argv:   # run the emitted command (no torch in this process), summarize registers / spills
    sys.argv.remove("--build")
    pat = sys.argv[1] if len(sys.argv) > 1 else None
    cmd = pathlib.Path("/work/tools/pub_nvcc_cmd.txt").read_text().split("\n")
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = r.stdout + r.stderr
    lines = out.splitlines()
    print("\n".join([l for l in lines if "error" in l.lower()][:40]))
    cur = None
    for l in lines:
        m = re.search(r"Compiling entry function '(\S+)'", l)
        if m:
            cur = m.group(1); continue
        if cur and ("Used" in l or "spill" in l):
            if pat is None or re.search(pat, cur):
                if "spill" in l and "0 bytes spill stores, 0 bytes spill loads" in l:
                    continue
                print(cur[:150], "|", l.strip())
    print("rc", r.returncode, "entries", out.count("Compiling entry"))
    sys.exit(1 if r.returncode else 0)
spec = importlib.util.spec_from_file_location("sub", "/work/muse_bf16_proj/submission.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
sm90 = True
common = S.CUDA_SRC.replace("@@MBAR@@", S.MBAR_HELPERS).replace("@@BODY@@", S.WG_HELPERS)
defs = S.wgmma_defs([8, 16, 32, 64, 128])
only = None
if "--only" in sys.argv:   # a subset TU (keeps the local nvcc under the Docker memory limit): --only sk2,gq,gqx
    i = sys.argv.index("--only"); only = sys.argv[i + 1].split(","); del sys.argv[i:i + 2]
if "--diag" in sys.argv:   # assemble the sk2 source as a DIAG build (the dg bits become runtime branches)
    sys.argv.remove("--diag"); S.DIAG_SK2 = True
if "--sk2-alt-only" in sys.argv:   # instantiate only the SK2_ALT configs (the picks are already validated on the judge)
    sys.argv.remove("--sk2-alt-only"); S.SK2_PICK = {}
if only is None:
    cuda_src = (common + defs + S._tma_srcs(sm90) + S.MAP_SRC + S._gemm_src(sm90) + S._wide_src(sm90) + S._sk2_src(sm90)
                + S._gq_src(sm90) + S._gqx_src(sm90) + S._rgemv_src() + S._rk_src(sm90))
else:
    parts = {"sk2": S._sk2_src, "gq": S._gq_src, "gqx": S._gqx_src, "rk": S._rk_src, "wide": S._wide_src,
             "gemm": S._gemm_src, "tma": S._tma_srcs, "rgemv": lambda sm: S._rgemv_src()}
    cuda_src = common + defs + S.wgmma_ss_defs([64, 128, 256]) + S.MAP_SRC + "".join(parts[k](sm90) for k in only)
import torch.utils.cpp_extension as E
src = "#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n" + cuda_src
p = pathlib.Path("/work/tools/pub_full.cu"); p.write_text(src)
import sysconfig
inc = sum([["-I", d] for d in E.include_paths(device_type="cuda") + [sysconfig.get_paths()["include"]]], [])
cmd = ["nvcc", "-gencode", "arch=compute_90a,code=sm_90a", "-O3", "-std=c++20", "-c", "-o", "/tmp/pub_full.o", "-Xptxas", "-v",
       "-DTORCH_API_INCLUDE_EXTENSION_H", "-D_GLIBCXX_USE_CXX11_ABI=1"] + inc + [str(p)]
if "--emit" in sys.argv:   # write the TU and the nvcc command only (a torch-loaded python next to nvcc exceeds the Docker RAM)
    pathlib.Path("/work/tools/pub_nvcc_cmd.txt").write_text("\n".join(cmd))
    print("emitted", p, len(src), "bytes"); sys.exit(0)
r = subprocess.run(cmd, capture_output=True, text=True)
out = r.stdout + r.stderr
pat = sys.argv[1] if len(sys.argv) > 1 else None
lines = out.splitlines()
errs = [l for l in lines if "error" in l.lower()]
print("\n".join(errs[:40]))
# per-kernel summary: entry name (demangled roughly) + registers + spills
cur = None
for l in lines:
    m = re.search(r"Compiling entry function '(\S+)'", l)
    if m:
        cur = m.group(1)
        continue
    if cur and ("Used" in l or "spill" in l):
        if pat is None or re.search(pat, cur):
            if "spill" in l and "0 bytes spill stores, 0 bytes spill loads" in l:
                continue
            print(cur[:150], "|", l.strip())
print("rc", r.returncode, "entries", out.count("Compiling entry"))
sys.exit(1 if r.returncode else 0)
