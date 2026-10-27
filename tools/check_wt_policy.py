"""GPU-free check of the wt policy selection: _wt_variant must return the bulk 6-tuple for M <= 8 when WT_USE_BULK and
wt_bulk_ok, the default 5-tuples otherwise, and fall back to WT_SAFE / the default policy when the self-test marks the
bulk variant bad; every chosen variant must be an instantiated key. Usage: python3 check_wt_policy.py (judge image)"""
import sys, re
sys.path.insert(0, "/work/muse_moe_w4a16")
import submission as S
src = open("/work/muse_moe_w4a16/submission.py").read()
keys = set(int(k) for k in re.findall(r"case (\d+): OKC_W", src))


def key(gated, v):
    nt, mw2, ks, ns, ksp = v[:5]
    cpc = v[5] if len(v) > 5 else 0
    return cpc * 100000000 + (100000 if gated else 0) + (1 if gated else mw2) * 10000 + nt * 1000 + ks * 100 + ns


bad = 0
for ok in (True, False):
    st = {"wt_bulk_ok": ok}
    for M in (1, 8, 9, 16, 32, 64, 128, 256):
        v = S._wt_variant(M, st)
        want_bulk = S.WT_USE_BULK and ok and M <= max(S.WT_BULK)
        is_bulk = len(v) > 5 and v[5] > 0
        if want_bulk != is_bulk:
            print(f"wt_bulk_ok={ok} M={M}: got {v}, bulk expected {want_bulk}"); bad += 1
        for gated in (True, False):
            if key(gated, v) not in keys:
                print(f"M={M} {v} gated={gated}: key {key(gated, v)} not instantiated"); bad += 1
print("policy problems:", bad)
sys.exit(1 if bad else 0)
