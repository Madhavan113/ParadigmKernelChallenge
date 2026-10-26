"""Emulate gemv12_kernel's per-lane addressing and PRMT decode in Python on the packed planes of one
layer and compare against the BF16 weights, to locate the GEMV12 mismatch."""
import sys
sys.path.insert(0, "/work/muse_bf16_proj")
import torch
import reference as R
import submission as S

w = R.generate_weights(R.WEIGHTS_SEED, device="cpu")["pooler"][:64]   # 2 n-tiles
pk = S._pack12(w)
N, K = w.shape
KT = K // 256
lo = pk["lo"].reshape(-1).numpy(); nb = pk["nib"].reshape(-1).numpy()
eb = pk["eb"]; hb = eb >> 1


def prmt(a, b, sel):
    src = [(a >> (8 * i)) & 0xFF for i in range(4)] + [(b >> (8 * i)) & 0xFF for i in range(4)]
    d = 0
    for i in range(4):
        s = (sel >> (4 * i)) & 0xF
        byte = src[s & 7]
        if s & 8:
            byte = 0xFF if byte & 0x80 else 0
        d |= byte << (8 * i)
    return d


lut0 = hb | (hb + 1) << 8 | (hb + 2) << 16 | (hb + 3) << 24
lut1 = (hb + 4) | (hb + 5) << 8 | (hb + 6) << 16 | (hb + 7) << 24


def decode4(L, n16):
    E = prmt(n16, 0, 0x1100)
    H = prmt(lut0, lut1, n16 & 0x7777) | ((E & 0x00080008) << 4) | (E & 0x80008000)
    return prmt(L, H, 0x5140), prmt(L, H, 0x7362)


wb = w.view(torch.int16).numpy().astype("uint16")
bad = 0
WARPS = 8
for pn in range(N // 32):
    for kt in range(KT):
        for warp in range(WARPS):
            for half in range(2):
                for s2 in range(32 // (2 * WARPS)):
                    row = 2 * warp + half + 2 * WARPS * s2
                    for l16 in range(16):
                        lo_off = ((pn * KT + kt) * 32 + row) * 256 + l16 * 16
                        nb_off = ((pn * KT + kt) * 32 + row) * 128 + l16 * 8
                        Lw = [int.from_bytes(lo[lo_off + 4 * i: lo_off + 4 * i + 4].tobytes(), "little") for i in range(4)]
                        Nw = [int.from_bytes(nb[nb_off + 4 * i: nb_off + 4 * i + 4].tobytes(), "little") for i in range(2)]
                        n16s = [Nw[0] & 0xFFFF, Nw[0] >> 16, Nw[1] & 0xFFFF, Nw[1] >> 16]
                        vals = []
                        for i in range(4):
                            w01, w23 = decode4(Lw[i], n16s[i])
                            vals += [w01 & 0xFFFF, w01 >> 16, w23 & 0xFFFF, w23 >> 16]
                        n = pn * 32 + row
                        for i in range(16):
                            kk = kt * 256 + l16 * 16 + i
                            if vals[i] != int(wb[n, kk]):
                                bad += 1
                                if bad < 8:
                                    print(f"mismatch n={n} k={kk}: got {vals[i]:04x} want {int(wb[n, kk]):04x} (L={Lw[i // 4]:08x} n16={n16s[i // 4]:04x})")
print("mismatches", bad, "of", N * K, "(escapes expected:", pk["n_esc"], ")")
