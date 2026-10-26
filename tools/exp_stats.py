"""BF16 exponent statistics of the muse_bf16_proj weights (reference generator, CPU): per-layer exponent
histogram, per-row / per-64 / per-16-block exponent range distributions, zeros / denormals / NaN, and
escape rates for sign + 7-bit mantissa + an exponent offset of B bits below a per-tensor / per-row /
per-block max. Usage: python3 exp_stats.py [seed]"""
import sys, math
sys.path.insert(0, "/work/muse_bf16_proj")
import torch
import reference as R

seed = int(sys.argv[1]) if len(sys.argv) > 1 else R.WEIGHTS_SEED
weights = R.generate_weights(seed, device="cpu")


def stats(name, w):
    if w.dtype == torch.float32:
        hi = w.to(torch.bfloat16); lo = (w - hi.float()).to(torch.bfloat16)
        stats(name + "_hi", hi); stats(name + "_lo", lo); return
    bits = w.view(torch.int16).to(torch.int32) & 0xFFFF
    exp = (bits >> 7) & 0xFF
    mant = bits & 0x7F
    n = w.numel()
    zeros = int(((exp == 0) & (mant == 0)).sum()); den = int(((exp == 0) & (mant != 0)).sum()); nan = int((exp == 255).sum())
    h = torch.bincount(exp.flatten(), minlength=256)
    nz = h.nonzero().flatten().tolist()
    print(f"{name}: shape {tuple(w.shape)} zeros {zeros} denormals {den} nan/inf {nan}; exponent range {nz[0]}..{nz[-1]} ({len(nz)} values)")
    print("   hist (exp: count):", ", ".join(f"{e}:{int(h[e])}" for e in nz))
    emax = nz[-1]
    for B in (3, 4, 5):
        esc = int((exp < emax - (2 ** B - 1)).sum())
        print(f"   per-tensor base {emax}, {B}-bit offset: escapes {esc} ({esc / n:.2e})")
    for blk, label in ((w.shape[1], "row"), (64, "blk64"), (16, "blk16")):
        e2 = exp.view(w.shape[0], w.shape[1] // blk, blk)
        bmax = e2.amax(-1); bmin = e2.amin(-1)
        rng = (bmax - bmin).flatten()
        hist = torch.bincount(rng, minlength=40)
        cum = hist.cumsum(0) / rng.numel()
        print(f"   {label} ranges: max {int(rng.max())}; frac of blocks with range <=7: {float(cum[7]):.4f} <=15: {float(cum[15]):.5f} <=31: {float(cum[31]):.6f}")
        for B in (3, 4, 5):
            esc = int((e2 < (bmax - (2 ** B - 1)).unsqueeze(-1)).sum())
            print(f"      {label} base, {B}-bit offset: escapes {esc} ({esc / n:.2e}); base-field cost {w.shape[0] * (w.shape[1] // blk) * 8 / (n * 16):.4f} of bf16 bits (8-bit bases)")


for name in ("qkv_gate", "o", "pooler", "unpooler", "router"):
    stats(name, weights[name])
