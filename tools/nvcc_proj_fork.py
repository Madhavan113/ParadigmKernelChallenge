"""Offline nvcc (sm_90a) of the decode layer's proj fork (muse_decode_layer/proj_fork.py) as its _get_ext() builds it,
inside the judge image, no GPU: syntax, registers, spills of the sk2 kernels. Run under the shared build lock:

    python3 tools/okc_locked.py docker run --rm -v "$PWD:/work:ro" -v /tmp/okc_nvcc:/out lmsysorg/sglang:dev-muse-spark-open \
        python3 /work/tools/nvcc_proj_fork.py [--all]

Default: only the sk2 sources (the fused pooler + router kernel lives there; ~2 min); --all: the whole TU as _get_ext().
"""
import importlib.util
import pathlib
import subprocess
import sys
import sysconfig

import torch.utils.cpp_extension as E

spec = importlib.util.spec_from_file_location("proj_fork", "/work/muse_decode_layer/proj_fork.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
sm90 = True
common = P.CUDA_SRC.replace("@@MBAR@@", P.MBAR_HELPERS).replace("@@BODY@@", P.WG_HELPERS)
defs = P.wgmma_defs([8, 16, 32, 64, 128])
if "--all" in sys.argv:
    body = P._tma_srcs(sm90) + P.MAP_SRC + P._gemm_src(sm90) + P._wide_src(sm90) + P._sk2_src(sm90) + P._gq_src(sm90)
else:
    body = P.MAP_SRC + P._sk2_src(sm90)
src = common + defs + body
out = pathlib.Path("/out")
out.mkdir(parents=True, exist_ok=True)
cu = out / "proj_fork.cu"
cu.write_text(src)
inc = sum([["-I", d] for d in E.include_paths(device_type="cuda") + [sysconfig.get_paths()["include"]]], [])
cmd = ["nvcc", "-gencode", "arch=compute_90a,code=sm_90a", "-O3", "-std=c++20", "-c", "-o", str(out / "proj_fork.o"),
       "-Xptxas", "-v", "-DTORCH_API_INCLUDE_EXTENSION_H", "-D_GLIBCXX_USE_CXX11_ABI=1"] + inc + [str(cu)]
r = subprocess.run(cmd, capture_output=True, text=True)
text = r.stdout + r.stderr
keep = [l for l in text.splitlines() if ("error" in l.lower() or "warning" in l.lower() or "Used" in l or "spill" in l
                                         or "Compiling entry" in l)]
print("\n".join(keep[-120:]))
print("rc", r.returncode)
sys.exit(r.returncode)
