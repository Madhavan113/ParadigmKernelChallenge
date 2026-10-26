"""Emulate w12b_kernel's math in numpy (judge image, no GPU): W warps each own a K slice of a 16-row tile; per lane
and k16 step the A fragments come from the W12 layout via decode4, the B fragments straight from x
(b0 = x[tok][k0+2t, +1], b1 = x[tok][k0+2t+8, +9], tok = nb*8 + g), mma.m16n8k16 semantics, the warps' partials
summed, the tile's escapes added; compare with x @ W.T (fp32)."""
import sys
sys.path.insert(0, "/work/muse_bf16_proj")
import numpy as np
import torch
import submission as S
import cuda_proj as C

torch.manual_seed(0)
n, k, m, W = 64, 1024, 8, 8                     # 4 tiles, 8 chunks -> 1 chunk per warp
w = (torch.randn((n, k)) * 3).to(torch.bfloat16)
w.view(torch.int16)[3, 5] = 0                   # an exact zero: becomes an escape
pk = S._pack12(w)
Wr = C._repack_w12(pk, n, k)                    # [n/64, k/128, 4, 3072]
x = torch.randn((m, k)).to(torch.bfloat16)
ref = (x.float() @ w.float().T).numpy()
xb = x.view(torch.int16).numpy().astype(np.uint16)


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


def decode4(L, n16):
    E = prmt(n16, 0, 0x1100)
    H = prmt(lut0, lut1, n16 & 0x7777) | ((E & 0x00080008) << 4) | (E & 0x80008000)
    return prmt(L, H, 0x5140), prmt(L, H, 0x7362)


def bf(u16):
    return np.frombuffer(np.array([int(u16) << 16], dtype=np.uint32).tobytes(), dtype=np.float32)[0]


NC = k // 128
NCW = NC // W
out = np.zeros((m, n), dtype=np.float32)
for tile in range(n // 16):
    grp, slot = tile >> 2, tile & 3
    acc = np.zeros((16, 8), dtype=np.float32)        # rows x tokens (NT = 1)
    for warp in range(W):
        for c in range(warp * NCW, (warp + 1) * NCW):
            blk = Wr[grp, c, slot]
            for s in range(8):
                q, half = s // 2, s // 4
                k0 = c * 128 + s * 16
                Afull = np.zeros((16, 16), dtype=np.float32); Bfull = np.zeros((16, 8), dtype=np.float32)
                for lane in range(32):
                    g, t = lane // 4, lane % 4
                    lw = blk[q * 512 + lane * 16 + (s % 2) * 8: q * 512 + lane * 16 + (s % 2) * 8 + 8].view(torch.int32).tolist()
                    L0, L1 = lw[0] & 0xFFFFFFFF, lw[1] & 0xFFFFFFFF
                    Nw = int(blk[2048 + half * 512 + lane * 16 + (s % 4) * 4: 2048 + half * 512 + lane * 16 + (s % 4) * 4 + 4].view(torch.int32)[0]) & 0xFFFFFFFF
                    a0, a1 = decode4(L0, Nw & 0xFFFF)
                    a2, a3 = decode4(L1, Nw >> 16)
                    for (r, kk), v in {(g, 2 * t): bf(a0 & 0xFFFF), (g, 2 * t + 1): bf(a0 >> 16), (g + 8, 2 * t): bf(a1 & 0xFFFF), (g + 8, 2 * t + 1): bf(a1 >> 16),
                                       (g, 2 * t + 8): bf(a2 & 0xFFFF), (g, 2 * t + 9): bf(a2 >> 16), (g + 8, 2 * t + 8): bf(a3 & 0xFFFF), (g + 8, 2 * t + 9): bf(a3 >> 16)}.items():
                        Afull[r, kk] = v
                    tok = g                                          # b0 = x[tok][k0+2t,+1], b1 = x[tok][k0+2t+8,+9]
                    for kk in (2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9):
                        Bfull[kk, tok] = bf(xb[tok, k0 + kk])
                acc += Afull @ Bfull
    # escapes for this tile
    pt, rbase = tile >> 1, 16 * (tile & 1)
    e0, e1 = int(pk["esc_off"][pt]), int(pk["esc_off"][pt + 1])
    for e in range(e0, e1):
        nl = int(pk["esc_n"][e]) - rbase
        if 0 <= nl < 16:
            kk = int(pk["esc_k"][e]); d = float(pk["esc_delta"][e])
            for i in range(m):
                acc[nl, i] += d * bf(xb[i, kk])
    out[:, tile * 16: tile * 16 + 16] = acc.T
err = np.abs(out - ref).max() / np.abs(ref).max()
print(f"w12b emulation: max rel err {err:.2e} vs fp32 reference (escapes {pk['n_esc']})")
assert err < 1e-5
print("w12b emulation ok")
