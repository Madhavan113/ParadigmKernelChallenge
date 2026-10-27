"""CPU round trip of the 12-bit weight packing: decode(lo, nib) must equal W bit-exactly on every
non-escape element, and stored + delta must equal W (to fp32 rounding) on the escapes.
Usage (judge image, no GPU): python3 check_pack12.py [seed]"""
import sys, time
sys.path.insert(0, "/work/muse_bf16_proj")
import torch
import reference as R
import submission as S

seed = int(sys.argv[1]) if len(sys.argv) > 1 else R.WEIGHTS_SEED
weights = R.generate_weights(seed, device="cpu")
_r = weights["router"]
weights["router_hi"] = _r.to(torch.bfloat16)
weights["router_lo"] = (_r - weights["router_hi"].float()).to(torch.bfloat16)
for name in ("qkv_gate", "o", "pooler", "unpooler", "router_hi", "router_lo"):
    w = weights[name]
    t = time.time()
    pk = S._pack12(w)
    assert pk is not None, name
    n, k = w.shape
    dec = S._unpack12(pk, n, k)
    same = torch.equal(dec.view(torch.int16), w.view(torch.int16))
    diff = (dec.view(torch.int16) != w.view(torch.int16))
    n_diff = int(diff.sum())
    # escapes cover exactly the differing elements
    en, ek, d = pk["esc_n"], pk["esc_k"], pk["esc_delta"]
    off = pk["esc_off"]
    rows = torch.repeat_interleave(torch.arange(n // S.PACK_BN), (off[1:] - off[:-1]).long()) * S.PACK_BN + en.long()
    if pk["n_esc"]:
        fixed = dec.float().clone()
        fixed[rows, ek.long()] += d
        err = (fixed - w.float()).abs()
        rel = float((err[rows, ek.long()] / w.float()[rows, ek.long()].abs().clamp(min=1e-30)).max())
        covered = bool(diff[rows, ek.long()].all()) and int(diff.sum()) == pk["n_esc"]
        maxerr = float(err.max())
    else:
        covered, rel, maxerr = n_diff == 0, 0.0, float((dec.float() - w.float()).abs().max())
    bytes_ = pk["lo"].numel() + pk["nib"].numel() + 12 * max(pk["n_esc"], 1) + 4 * off.numel()
    print(f"{name:9s} eb {pk['eb']} escapes {pk['n_esc']} ({pk['n_esc'] / w.numel():.1e}) differing {n_diff} covered {covered}"
          f" | max abs err after escapes {maxerr:.2e} (rel on escapes {rel:.1e}) | bytes {bytes_ / (w.numel() * 2):.4f} of bf16 | {time.time() - t:.1f}s")
    assert covered and maxerr < 1e-9
    assert torch.equal(pk["esc_roff"][::S.PACK_BN], pk["esc_off"]), "per-row offsets disagree with tile offsets"
print("pack12 ok")
