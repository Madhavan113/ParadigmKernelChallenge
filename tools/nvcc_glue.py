"""Offline nvcc (sm_90a) of the decode layer's CUDA glue kernels (GLUE_CUDA_SRC in muse_decode_layer/layer_glue.py) as
load_inline builds them, inside the judge image, no GPU: syntax, registers, spills. Run under the shared build lock:

    python3 tools/okc_locked.py docker run --rm -v "$PWD:/work:ro" -v /tmp/okc_nvcc:/out lmsysorg/sglang:dev-muse-spark-open \
        python3 /work/tools/nvcc_glue.py [/work/muse_decode_layer/layer_glue.py]
"""
import pathlib
import re
import subprocess
import sys
import sysconfig

import torch.utils.cpp_extension as E

path = sys.argv[1] if len(sys.argv) > 1 else "/work/muse_decode_layer/layer_glue.py"
text = pathlib.Path(path).read_text()
m = re.search(r'^GLUE_CUDA_SRC = r"""(.*?)^"""', text, re.S | re.M)
if not m:
    print("GLUE_CUDA_SRC not found")
    sys.exit(2)
out = pathlib.Path("/out")
out.mkdir(parents=True, exist_ok=True)
cu = out / "glue.cu"
cu.write_text(m.group(1))
inc = sum([["-I", d] for d in E.include_paths(device_type="cuda") + [sysconfig.get_paths()["include"]]], [])
cmd = ["nvcc", "-gencode", "arch=compute_90a,code=sm_90a", "-O3", "-std=c++20", "-c", "-o", str(out / "glue.o"),
       "-Xptxas", "-v", "-DTORCH_API_INCLUDE_EXTENSION_H", "-D_GLIBCXX_USE_CXX11_ABI=1"] + inc + [str(cu)]
r = subprocess.run(cmd, capture_output=True, text=True)
lines = (r.stdout + r.stderr).splitlines()
keep = [l for l in lines if ("error" in l.lower() or "warning" in l.lower() or "Used" in l or "spill" in l
                             or "Compiling entry" in l)]
print("\n".join(keep[-80:]))
print("rc", r.returncode)
sys.exit(r.returncode)
