"""Offline nvcc (sm_90a) of muse_moe_w4a16/submission.py's CUDA_SRC inside the judge image, torch-free (the file is
parsed with ast, include paths are the image's), one build at a time, reporting registers / spills of
moe_fused<false|true> and optionally a SASS dump. Run from the host under the shared build lock:
  python3 tools/locked_run.py docker run --rm --platform linux/amd64 -v <repo>:/work -w /work \
      lmsysorg/sglang:dev-muse-spark-open python3 tools/nvcc_pub_moe.py [build ...] [--sass] [--extra=-DOKC_X=1,...]
builds: TINY TINY2 MID MAIN (from submission.py) or NAME=-DOKC_A=1,-DOKC_B=2 (custom)."""
import subprocess, sys, pathlib, re, ast
src_path = pathlib.Path(next((a[6:] for a in sys.argv[1:] if a.startswith("--src=")), "/work/muse_moe_w4a16/submission.py"))  # --src=: another copy (e.g. a committed version)
tree = ast.parse(src_path.read_text())
consts = {}
for node in tree.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        try:
            consts[node.targets[0].id] = ast.literal_eval(node.value)
        except Exception:
            pass
src = "#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n" + consts["CUDA_SRC"]
out_dir = pathlib.Path("/work/tools/nvcc_out"); out_dir.mkdir(exist_ok=True)  # mounted: survives the container
cu = out_dir / "pub_moe.cu"; cu.write_text(src)
INC = ["/opt/sglang/lib/python3.12/site-packages/torch/include",
       "/opt/sglang/lib/python3.12/site-packages/torch/include/torch/csrc/api/include",
       "/usr/local/cuda/include", "/usr/include/python3.12"]
inc = sum([["-I", d] for d in INC], [])
base = ["-DOKC_NS=%d" % consts.get("MAIN_NS", 3), "-DOKC_FNST=4", "-DOKC_NSF=3", "-DOKC_DQ7=1", "-DOKC_PRMTSC=1"]
builds = {"TINY": consts["TINY_DEFINES"], "TINY2": consts["TINY2_DEFINES"], "MID": consts["MID_DEFINES"], "MAIN": consts["MAIN_DEFINES"]}
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
        if "error" in l.lower() or "C75" in l or "serializ" in l.lower():
            lines.append(l[:300])
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
        lines.append(f"  sass -> tools/nvcc_out/{name}.sass ({s.count(chr(10))} lines)")
    if r.returncode != 0 and not lines:
        lines.append(txt[-3000:])
    if o.exists():
        o.unlink()
    return name, r.returncode, "\n".join(lines)


for name, defs in sel.items():
    n, rc, rep = run(name, defs)
    print(f"[{n}] rc={rc} defs={' '.join(defs)}\n{rep}", flush=True)
