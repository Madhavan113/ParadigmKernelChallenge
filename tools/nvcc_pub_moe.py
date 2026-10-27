"""Offline nvcc (sm_90a) of muse_moe_w4a16/submission.py's CUDA_SRC inside the judge image, one process per build,
reporting registers / spills of moe_fused<false|true> (and optionally a SASS dump). Run in the image from /work:
  python3 tools/nvcc_pub_moe.py [build ...] [--sass] [--extra=-DOKC_X=1,...]
builds: TINY TINY2 MID MAIN (from submission.py), any name in submission.EXTRA_BUILDS, or NAME=-DOKC_A=1,-DOKC_B=2."""
import subprocess, sys, pathlib, re, concurrent.futures, importlib.util
sys.path.insert(0, "/work/muse_moe_w4a16")
spec = importlib.util.spec_from_file_location("sub", "/work/muse_moe_w4a16/submission.py")
sub = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(sub)
except Exception as ex:  # torch import works on CPU; anything else is a real error
    print("import failed:", ex); sys.exit(2)
import torch.utils.cpp_extension as E, sysconfig
src = "#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n" + sub.CUDA_SRC
out_dir = pathlib.Path("/tmp/okc_nvcc"); out_dir.mkdir(exist_ok=True)
cu = out_dir / "pub_moe.cu"; cu.write_text(src)
inc = sum([["-I", d] for d in E.include_paths(device_type="cuda") + [sysconfig.get_paths()["include"]]], [])
base = ["-DOKC_NS=%d" % sub.MAIN_NS, "-DOKC_FNST=4", "-DOKC_NSF=3", "-DOKC_DQ7=1", "-DOKC_PRMTSC=1"]
builds = {"TINY": sub.TINY_DEFINES, "TINY2": sub.TINY2_DEFINES, "MID": sub.MID_DEFINES, "MAIN": sub.MAIN_DEFINES}
for k, v in list(getattr(sub, "EXTRA_BUILDS", {}).items()):
    builds[k] = v
args = [a for a in sys.argv[1:] if not a.startswith("--")]
sass = "--sass" in sys.argv
extra = [x for a in sys.argv[1:] if a.startswith("--extra=") for x in a[8:].split(",") if x]
sel = {}
for a in args or list(builds):
    if "=" in a and not a.startswith("-D"):
        n, d = a.split("=", 1); sel[n] = tuple(x for x in d.split(",") if x)
    else:
        sel[a] = tuple(builds[a])


def run(name, defs):
    o = out_dir / f"{name}.o"
    defs = list(defs) + [b for b in base if not any(d.split("=")[0] == b.split("=")[0] for d in defs)]
    cmd = ["nvcc", "-gencode", "arch=compute_90a,code=sm_90a", "-O3", "-std=c++20", "-c", "-o", str(o), "-Xptxas", "-v",
           "-DTORCH_API_INCLUDE_EXTENSION_H", "-D_GLIBCXX_USE_CXX11_ABI=1"] + inc + defs + extra + [str(cu)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    txt = r.stdout + r.stderr
    lines = []
    ent = None
    for l in txt.splitlines():
        if "error" in l.lower():
            lines.append(l)
        m = re.search(r"Compiling entry function '(\S+)'", l)
        if m:
            ent = m.group(1)
            ent = "fused<true>" if "moe_fusedILb1" in ent else "fused<false>" if "moe_fusedILb0" in ent else ent[:40]
        if "Used" in l and "registers" in l:
            lines.append(f"  {ent}: {l.strip()}")
        if "bytes spill" in l and "0 bytes spill stores, 0 bytes spill loads" not in l:
            lines.append(f"  {ent}: {l.strip()}")
    if sass and r.returncode == 0:
        s = subprocess.run(["cuobjdump", "-sass", str(o)], capture_output=True, text=True).stdout
        (out_dir / f"{name}.sass").write_text(s)
        lines.append(f"  sass -> /tmp/okc_nvcc/{name}.sass ({s.count(chr(10))} lines)")
    if r.returncode != 0 and not lines:
        lines.append(txt[-3000:])
    return name, r.returncode, "\n".join(lines)


jobs = int(next((a[7:] for a in sys.argv[1:] if a.startswith("--jobs=")), "2"))
with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
    for name, rc, rep in ex.map(lambda kv: run(*kv), sel.items()):
        print(f"[{name}] rc={rc} defs={' '.join(sel[name])}\n{rep}", flush=True)
