"""Re-embed muse_moe_w4a16/cuda_moe.py's CUDA_SRC/CPP_SRC/build/repack_weights into submission.py between markers."""
import re, pathlib
root = pathlib.Path(__file__).resolve().parent.parent
sub = (root / "muse_moe_w4a16/submission.py").read_text()
src = (root / "muse_moe_w4a16/cuda_moe.py").read_text()
a = src.index('CUDA_SRC = r"""'); b = src.index('def _moe_forward_cuda') if 'def _moe_forward_cuda' in src else len(src)
embed = src[a:b].rstrip() + "\n"
start = sub.index('# == BEGIN CUDA ==\n') + len('# == BEGIN CUDA ==\n')
end = sub.index('# == END CUDA ==')
sub = sub[:start] + embed + "\n" + sub[end:]
(root / "muse_moe_w4a16/submission.py").write_text(sub)
print("embedded", len(embed.splitlines()), "lines")
