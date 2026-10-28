"""Round-2 item 4: the MoE TINY build with OKC_WSH0 as a run-time switchable alternative at T <= TINY_MAX_M.
layer_glue: MOE_TINY_H0_BUILD adds `moe_TINY_DEFINES + ("-DOKC_WSH0=1",)` to prepare's parallel builds; ab_vm
`tinyh0=1` swaps moe_TINY_DEFINES (the MoE module's kernel_fn reads it per call).  Run once from the worktree root."""
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
    ("""def _moe_builds(sfx):
    \"\"\"The builds the moe module's prepare() will ask for (its rule), so prepare() here can run them in parallel.\"\"\"
    mods = [tuple(moe__main_defines(sfx)) + sfx]
    for d in (moe_TINY_DEFINES, moe_TINY2_DEFINES, moe_MID_DEFINES):
        if d and (sfx or "-DOKC_WGS=1" not in tuple(d)):   # the WGS build needs the PRMT scale decode
            mods.append(tuple(d) + sfx)""",
     """MOE_TINY_H0_BUILD = True   # also build the TINY (M <= 8) variant with OKC_WSH0 (helper warp = hardware warp 0) for the
                           # chain A/B (vm_ab/ab15; the standalone measured it neutral); selected by moe_TINY_DEFINES


def _moe_builds(sfx):
    \"\"\"The builds the moe module's prepare() will ask for (its rule), so prepare() here can run them in parallel.\"\"\"
    mods = [tuple(moe__main_defines(sfx)) + sfx]
    for d in (moe_TINY_DEFINES, moe_TINY2_DEFINES, moe_MID_DEFINES):
        if d and (sfx or "-DOKC_WGS=1" not in tuple(d)):   # the WGS build needs the PRMT scale decode
            mods.append(tuple(d) + sfx)
    if MOE_TINY_H0_BUILD and moe_TINY_DEFINES and "-DOKC_WSH0=1" not in tuple(moe_TINY_DEFINES):
        mods.append(tuple(moe_TINY_DEFINES) + ("-DOKC_WSH0=1",) + sfx)""", 1),
])

patch("muse_decode_layer/ab_vm.py", [
    ("""        elif kind == "pdlflags":   # the proj fork's PDL word flags: 0x100 sk2 trigger at entry, 0x200 cgemv as a PDL launch
            saved.append(("pdlflags", None, S.SK2_PDL_FLAGS))
            S.SK2_PDL_FLAGS = int(val)""",
     """        elif kind == "pdlflags":   # the proj fork's PDL word flags: 0x100 sk2 trigger at entry, 0x200 cgemv as a PDL launch
            saved.append(("pdlflags", None, S.SK2_PDL_FLAGS))
            S.SK2_PDL_FLAGS = int(val)
        elif kind == "tinyh0":   # the MoE TINY build with OKC_WSH0 (pre-built when MOE_TINY_H0_BUILD)
            saved.append(("tinyh0", None, S.moe_TINY_DEFINES))
            base = tuple(d for d in S.moe_TINY_DEFINES if d != "-DOKC_WSH0=1")
            S.moe_TINY_DEFINES = base + (("-DOKC_WSH0=1",) if val in ("1", "True") else ())""", 1),
    ("""        elif kind == "pdlflags":
            S.SK2_PDL_FLAGS = old
        else:""", """        elif kind == "pdlflags":
            S.SK2_PDL_FLAGS = old
        elif kind == "tinyh0":
            S.moe_TINY_DEFINES = old
        else:""", 1),
])
print("patched")
