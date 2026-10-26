"""CPU check of _repack_w12: rebuild every element's 12-bit code from the fragment-ordered layout (the kernel's
addressing: tile, chunk, lane (g, t), step, L0/L1 order, nibble i) and compare with _pack12's lo/nibble codes; then
emulate decode4 on a sample of words and compare the bf16 bits with _unpack12. Run in the judge image."""
import sys
sys.path.insert(0, "/work/muse_bf16_proj")
import torch
import submission as S
import cuda_proj as C

torch.manual_seed(0)
n, k = 128, 512
w = (torch.randn((n, k)) * 3).to(torch.bfloat16)
pk = S._pack12(w)
assert pk is not None
W = C._repack_w12(pk, n, k)                                   # [n/64, k/128, 4, 3072]
lo = pk["lo"].view(n // 32, k // 256, 32, 256).permute(0, 2, 1, 3).reshape(n, k)
nb = pk["nib"].view(n // 32, k // 256, 32, 128).permute(0, 2, 1, 3).reshape(n, k // 2)
nib = torch.stack((nb & 0xF, nb >> 4), -1).reshape(n, k)
bad_lo = bad_nb = 0
for tile in range(n // 16):
    for c in range(k // 128):
        blk = W[tile // 4, c, tile % 4]
        for lane in range(32):
            g, t = lane // 4, lane % 4
            for s in range(8):
                q, half = s // 2, s // 4
                lob = blk[q * 512 + lane * 16 + (s % 2) * 8: q * 512 + lane * 16 + (s % 2) * 8 + 8].tolist()
                word = int(blk[2048 + half * 512 + lane * 16 + (s % 4) * 4: 2048 + half * 512 + lane * 16 + (s % 4) * 4 + 4].view(torch.int32)[0]) & 0xFFFFFFFF
                rows = [tile * 16 + g, tile * 16 + g, tile * 16 + g + 8, tile * 16 + g + 8] * 2
                ks = [c * 128 + s * 16 + 2 * t, c * 128 + s * 16 + 2 * t + 1] * 2 + [c * 128 + s * 16 + 2 * t + 8, c * 128 + s * 16 + 2 * t + 9] * 2
                for i in range(8):
                    bad_lo += lob[i] != int(lo[rows[i], ks[i]])
                    bad_nb += ((word >> (4 * i)) & 0xF) != int(nib[rows[i], ks[i]])
print("lo byte mismatches", bad_lo, "nibble mismatches", bad_nb)
assert bad_lo == 0 and bad_nb == 0


def prmt(a, b, sel):
    src = [(a >> (8 * i)) & 0xFF for i in range(4)] + [(b >> (8 * i)) & 0xFF for i in range(4)]
    d = 0
    for i in range(4):
        sl = (sel >> (4 * i)) & 0xF
        byte = src[sl & 7]
        if sl & 8:
            byte = 0xFF if byte & 0x80 else 0
        d |= byte << (8 * i)
    return d


hb = pk["eb"] >> 1
lut0 = hb | (hb + 1) << 8 | (hb + 2) << 16 | (hb + 3) << 24
lut1 = (hb + 4) | (hb + 5) << 8 | (hb + 6) << 16 | (hb + 7) << 24
dec = S._unpack12(pk, n, k).view(torch.int16).numpy().astype("uint16")


def decode4(L, n16):
    E = prmt(n16, 0, 0x1100)
    H = prmt(lut0, lut1, n16 & 0x7777) | ((E & 0x00080008) << 4) | (E & 0x80008000)
    return prmt(L, H, 0x5140), prmt(L, H, 0x7362)


bad = 0
for tile in range(n // 16):
    for c in range(k // 128):
        blk = W[tile // 4, c, tile % 4]
        for lane in range(32):
            g, t = lane // 4, lane % 4
            for s in range(8):
                q, half = s // 2, s // 4
                lw = blk[q * 512 + lane * 16 + (s % 2) * 8: q * 512 + lane * 16 + (s % 2) * 8 + 8].view(torch.int32).tolist()
                L0, L1 = lw[0] & 0xFFFFFFFF, lw[1] & 0xFFFFFFFF
                Nw = int(blk[2048 + half * 512 + lane * 16 + (s % 4) * 4: 2048 + half * 512 + lane * 16 + (s % 4) * 4 + 4].view(torch.int32)[0]) & 0xFFFFFFFF
                a0, a1 = decode4(L0, Nw & 0xFFFF)
                a2, a3 = decode4(L1, Nw >> 16)
                r0, r1 = tile * 16 + g, tile * 16 + g + 8
                kb = c * 128 + s * 16 + 2 * t
                exp = [(r0, kb), (r0, kb + 1), (r1, kb), (r1, kb + 1), (r0, kb + 8), (r0, kb + 9), (r1, kb + 8), (r1, kb + 9)]
                got = [a0 & 0xFFFF, a0 >> 16, a1 & 0xFFFF, a1 >> 16, a2 & 0xFFFF, a2 >> 16, a3 & 0xFFFF, a3 >> 16]
                for (r, kk), v in zip(exp, got):
                    bad += int(dec[r, kk]) != v
print("decoded bf16 mismatches vs _unpack12:", bad)
assert bad == 0
print("w12 ok")
