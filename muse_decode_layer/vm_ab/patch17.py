"""Round-2 item 5: attention plan overrides from the layer (the att module's launcher takes plan kwargs: n_splits, flat,
cs, ncl, nc, coop, w0, tma).  layer_glue: ATT_PLAN_OVERRIDE[(B, L, sliding)] merged into the launch; ab_vm
`attplan=k:v,k:v` sets the override for every call of the variant (use single-shape batches).  Run once from the root."""
import pathlib

root = pathlib.Path(__file__).resolve().parents[2]


def patch(path, edits):
    p = root / path
    s = p.read_text()
    for old, new, cnt in edits:
        n = s.count(old)
        assert n == cnt, (path, n, cnt, old[:90])
        s = s.replace(old, new)
    p.write_text(s)


patch("muse_decode_layer/layer_glue.py", [
    ("""def _decode_attention7(data, state, qkv):
    \"\"\"The attention module's own launcher (att__launch): plan, workspace sizes and the coop / w0 bits always match
    the kernel version composed in. This layer's config already carries their keys (heads, head_dim, page_size,
    window, rope_theta, scales).\"\"\"
    cfg = data["config"]
    pdl = {"pdl": _pdl(cfg["batch"] * cfg["q_len"])} if ATT_HAS_PDL else {}   # their launcher's PDL switch
    out = att__launch({"config": cfg, "qkv": qkv, "seq_lens": data["seq_lens"], "cache_loc": data["cache_loc"],
                    "k_cache": data["k_cache"], "v_cache": data["v_cache"], "page_table": data["page_table"],
                    "k_scale": data["k_scale"], "v_scale": data["v_scale"]}, **pdl)""",
     """ATT_PLAN_OVERRIDE = {}   # (B, q_len, sliding) -> plan kwargs for the att module's launcher (n_splits / flat / cs / ncl /
                         # nc / coop / w0): the chain-timed alternatives to its own plan (vm_ab/ab17); None = all shapes


def _decode_attention7(data, state, qkv):
    \"\"\"The attention module's own launcher (att__launch): plan, workspace sizes and the coop / w0 bits always match
    the kernel version composed in. This layer's config already carries their keys (heads, head_dim, page_size,
    window, rope_theta, scales).\"\"\"
    cfg = data["config"]
    pdl = {"pdl": _pdl(cfg["batch"] * cfg["q_len"])} if ATT_HAS_PDL else {}   # their launcher's PDL switch
    plan = ATT_PLAN_OVERRIDE.get((cfg["batch"], cfg["q_len"], bool(cfg["window"])), ATT_PLAN_OVERRIDE.get(None, {}))
    out = att__launch({"config": cfg, "qkv": qkv, "seq_lens": data["seq_lens"], "cache_loc": data["cache_loc"],
                    "k_cache": data["k_cache"], "v_cache": data["v_cache"], "page_table": data["page_table"],
                    "k_scale": data["k_scale"], "v_scale": data["v_scale"]}, **pdl, **plan)""", 1),
])

patch("muse_decode_layer/ab_vm.py", [
    ("""        elif kind == "m1":   # pin a layer's m = 1 route in the proj fork: m1:unpooler=sgemv|cgemv|sk2 (qkv_gate / o / unpooler)""",
     """        elif kind == "attplan":   # attention plan kwargs for every call of this variant: attplan=cs:8,nc:4 (single-shape batches)
            saved.append(("attplan", None, dict(S.ATT_PLAN_OVERRIDE)))
            S.ATT_PLAN_OVERRIDE[None] = {kv.split(":")[0]: int(kv.split(":")[1]) for kv in val.split(",") if kv}
        elif kind == "m1":   # pin a layer's m = 1 route in the proj fork: m1:unpooler=sgemv|cgemv|sk2 (qkv_gate / o / unpooler)""", 1),
    ("""        elif kind == "m1":
            if old is None:
                st["bfp"]["choice"].pop(k, None)
            else:
                st["bfp"]["choice"][k] = old
        else:""", """        elif kind == "m1":
            if old is None:
                st["bfp"]["choice"].pop(k, None)
            else:
                st["bfp"]["choice"][k] = old
        elif kind == "attplan":
            S.ATT_PLAN_OVERRIDE.clear()
            S.ATT_PLAN_OVERRIDE.update(old)
        else:""", 1),
])
print("patched")
