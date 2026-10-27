"""Gate: every function exported by muse_bf16_proj/cuda_proj.py's build() must have a CPP_SRC declaration whose
parameter list matches its CUDA_SRC definition (an undefined symbol at load time cost check a101cd38).
Usage: python3 tools/check_cpp_decls.py [path/to/cuda_proj.py or submission.py]"""
import re, sys, pathlib
root = pathlib.Path(__file__).resolve().parent.parent
path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else root / "muse_bf16_proj/cuda_proj.py"
s = path.read_text()
cuda = s[s.index('CUDA_SRC = r"""') + len('CUDA_SRC = r"""'):]
cuda = cuda[:cuda.index('\n"""')]
cpp = s[s.index('CPP_SRC = """') + len('CPP_SRC = """'):]
cpp = cpp[:cpp.index('\n"""')]
funcs = re.search(r"functions=\[([^\]]*)\]", s).group(1)
names = re.findall(r'"(\w+)"', funcs)


def norm(sig):
    sig = re.sub(r"\s+", " ", sig).strip()
    return re.sub(r"\s*,\s*", ", ", sig)


bad = 0
for n in names:
    d = re.search(r"\bvoid\s+" + n + r"\s*\(([^)]*)\)\s*;", cpp)
    f = re.search(r"\bvoid\s+" + n + r"\s*\(([^)]*)\)\s*\{", cuda)
    if d is None or f is None:
        print(f"{n}: declaration {'ok' if d else 'MISSING'}, definition {'ok' if f else 'MISSING'}"); bad += 1; continue
    if norm(d.group(1)) != norm(f.group(1)):
        print(f"{n}: SIGNATURE MISMATCH\n  decl: {norm(d.group(1))}\n  def:  {norm(f.group(1))}"); bad += 1
print(f"{len(names)} exported functions checked, {bad} problems")
sys.exit(1 if bad else 0)
