"""Embed muse_attention_decode/cuda_sliding.py (CUDA_SLIDING_SRC .. _sliding_forward) into submission.py
between the '# == BEGIN CUDA SLIDING ==' / '# == END CUDA SLIDING ==' markers (created before the host side
if missing), and add the sliding dispatch to _attention_forward if missing."""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
sub_p = root / "muse_attention_decode/submission.py"
sub = sub_p.read_text()
src = (root / "muse_attention_decode/cuda_sliding.py").read_text()
a = src.index('CUDA_SLIDING_SRC = r"""')
embed = src[a:].rstrip() + "\n"
B, E = "# == BEGIN CUDA SLIDING ==\n", "# == END CUDA SLIDING ==\n"
if B not in sub:
    anchor = "# ----------------------------------------------------------------------------------------------\n# host side"
    assert anchor in sub
    sub = sub.replace(anchor, B + E + "\n\n" + anchor, 1)
start = sub.index(B) + len(B)
end = sub.index(E)
sub = sub[:start] + embed + "\n" + sub[end:]
old = '''    plan = _plan(cfg)
    if _PLAN_OVERRIDE is not None:
        plan.update(_PLAN_OVERRIDE)
'''
new = '''    plan = _plan(cfg)
    if _PLAN_OVERRIDE is not None:
        plan.update(_PLAN_OVERRIDE)
    if cfg["window"] and plan.get("cuda", SLIDING_CUDA) and _sliding_forward(data, out):
        return out
'''
if 'plan.get("cuda", SLIDING_CUDA)' not in sub:
    assert old in sub
    sub = sub.replace(old, new, 1)
sub_p.write_text(sub)
print("embedded", len(embed.splitlines()), "lines")
