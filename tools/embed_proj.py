"""Re-embed muse_bf16_proj/cuda_proj.py (CUDA_SRC .. gemv_usable) into submission.py between the CUDA markers."""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
sub = root / "muse_bf16_proj/submission.py"; t = sub.read_text()
s = (root / "muse_bf16_proj/cuda_proj.py").read_text()
embed = s[s.index('CUDA_SRC = r"""'):].rstrip() + "\n"
start = t.index("# == BEGIN CUDA ==\n") + len("# == BEGIN CUDA ==\n"); end = t.index("# == END CUDA ==")
sub.write_text(t[:start] + embed + "\n\n" + t[end:])
print("embedded", len(embed.splitlines()), "lines")
