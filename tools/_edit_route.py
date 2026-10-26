"""One-off edit: cheaper ranks in _route_kernel (count gathers for earlier chunks, one [PB x PB] compare in range)."""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
p = root / "muse_moe_w4a16/submission.py"
s = p.read_text()
old = """    e_key = tl.where(is_local, e, E)                       # E: not this GPU's
    counts = tl.zeros((E,), dtype=tl.int32)
    rank = tl.zeros((PB,), dtype=tl.int32)
    for c0 in range(0, npairs, CH):
        ci = c0 + tl.arange(0, CH)
        cv = ci < npairs
        ce = tl.load(ids_ptr + ci, mask=cv, other=0) - local_start
        cl = cv & (ce >= 0) & (ce < E)
        ce_key = tl.where(cl, ce, E)
        counts += tl.sum((ce_key[None, :] == ex[:, None]).to(tl.int32), 1)
        rank += tl.sum(((ce_key[None, :] == e_key[:, None]) & (ci[None, :] < i[:, None])).to(tl.int32), 1)
    padded = (counts + BM - 1) // BM * BM
    pad_off = tl.cumsum(padded, 0) - padded                 # padded row offset per expert
    onehot = (e_key[:, None] == ex[None, :]).to(tl.int32)   # [PB, E]
    row = tl.sum(onehot * pad_off[None, :], 1) + rank"""
new = """    e_key = tl.where(is_local, e, E)                       # E: not this GPU's
    onehot = (e_key[:, None] == ex[None, :]).to(tl.int32)   # [PB, E]
    my0 = pid * PB
    counts = tl.zeros((E,), dtype=tl.int32)
    rank = tl.zeros((PB,), dtype=tl.int32)
    for c0 in range(0, npairs, CH):
        ci = c0 + tl.arange(0, CH)
        cv = ci < npairs
        ce = tl.load(ids_ptr + ci, mask=cv, other=0) - local_start
        cl = cv & (ce >= 0) & (ce < E)
        ce_key = tl.where(cl, ce, E)
        cnt = tl.sum((ce_key[None, :] == ex[:, None]).to(tl.int32), 1)     # per-expert pairs in this chunk
        counts += cnt
        if c0 + CH <= my0:                                  # chunk entirely before this CTA's pairs
            rank += tl.sum(onehot * cnt[None, :], 1)
        else:
            if c0 <= my0:                                   # the chunk holding this CTA's pairs: its earlier pairs
                rank += tl.sum(((ce_key[None, :] == e_key[:, None]) & (ci[None, :] < my0)).to(tl.int32), 1)
    # earlier pairs of the same expert within this CTA's own range
    rank += tl.sum(((e_key[None, :] == e_key[:, None]) & (i[None, :] < i[:, None])).to(tl.int32), 1)
    padded = (counts + BM - 1) // BM * BM
    pad_off = tl.cumsum(padded, 0) - padded                 # padded row offset per expert
    row = tl.sum(onehot * pad_off[None, :], 1) + rank"""
assert s.count(old) == 1
s = s.replace(old, new)
p.write_text(s)
print("route kernel edited")
