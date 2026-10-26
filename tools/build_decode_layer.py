"""Assemble muse_decode_layer/submission.py from glue.py and the shared regions of the three
kernel modules (single-file submissions)."""
import re, pathlib
root = pathlib.Path(__file__).resolve().parent.parent
def region(path, tag):
    s = (root / path).read_text()
    m = re.search(rf"# == BEGIN SHARED {tag} ==\n(.*?)# == END SHARED {tag} ==", s, re.S)
    assert m, (path, tag)
    return m.group(1)
glue = (root / "muse_decode_layer/glue.py").read_text()
for tag, path in (("proj", "muse_bf16_proj/submission.py"), ("attn", "muse_attention_decode/submission.py"),
                  ("moe", "muse_moe_w4a16/submission.py")):
    body = region(path, tag)
    if tag == "proj":   # the proj and moe regions both carry a CUDA extension: keep both namespaces apart
        for a, b in (("CUDA_SRC", "PROJ_CUDA_SRC"), ("CPP_SRC", "PROJ_CPP_SRC"), ("_EXT", "_PROJ_EXT"), ("build", "_proj_build")):
            body = re.sub(rf"\b{a}\b", b, body)
        body = body.replace('name="okc_proj_cuda2"', 'name="okc_proj_cuda2"')
    glue = glue.replace("# {{SHARED %s}}" % tag, f"# ---- shared: {tag} ({path}) ----\n" + body)
(root / "muse_decode_layer/submission.py").write_text(glue)
print("built", len(glue.splitlines()), "lines")
