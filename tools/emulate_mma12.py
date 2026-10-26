"""Emulate one warp-step (8 weight rows x 64 k, MT m-tiles) of mma12_kernel: build the A fragments from
x's 16-element groups and the B fragments from decode4 of the packed planes exactly as the kernel
does, run mma.m16n8k16 semantics per k-step, and compare the C tile with the true partial product."""
import sys
sys.path.insert(0, "/work/muse_bf16_proj")
import numpy as np
import torch
import reference as R
import submission as S

torch.manual_seed(0)
w = R.generate_weights(R.WEIGHTS_SEED, device="cpu")["pooler"][:32]        # one n-tile
pk = S._pack12(w)
N, K = w.shape; KT = K // 256
lo = pk["lo"].reshape(-1).numpy(); nb = pk["nib"].reshape(-1).numpy(); hb = pk["eb"] >> 1
m = 32; MT = 2
x = torch.randn((m, K)).to(torch.bfloat16)
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


lut0 = hb | (hb + 1) << 8 | (hb + 2) << 16 | (hb + 3) << 24
lut1 = (hb + 4) | (hb + 5) << 8 | (hb + 6) << 16 | (hb + 7) << 24


def decode4(L, n16):
    E = prmt(n16, 0, 0x1100)
    H = prmt(lut0, lut1, n16 & 0x7777) | ((E & 0x00080008) << 4) | (E & 0x80008000)
    return prmt(L, H, 0x5140), prmt(L, H, 0x7362)


def bf(u16):
    return np.array([u16], dtype=np.uint16).view(np.float16) if False else float(torch.tensor([int(u16)], dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float())


def mma(C, A, B):
    """C[16x8] += A[16x16] . B[16x8] (virtual k), A/B given as per-lane fragment registers."""
    for g in range(8):
        for t in range(4):
            a0, a1, a2, a3 = A[(g, t)]
            b0, b1 = B[(g, t)]
            # a0: A[g][2t,2t+1]; a1: A[g+8][2t,2t+1]; a2: A[g][2t+8,2t+9]; a3: A[g+8][2t+8,2t+9]
            # b0: B[2t,2t+1][g];  b1: B[2t+8,2t+9][g]
            for (row, reg) in ((g, a0), (g + 8, a1)):
                for e in range(2):
                    Amat[row][2 * t + e] = bf((reg >> (16 * e)) & 0xFFFF)
            for (row, reg) in ((g, a2), (g + 8, a3)):
                for e in range(2):
                    Amat[row][2 * t + 8 + e] = bf((reg >> (16 * e)) & 0xFFFF)
            for e in range(2):
                Bmat[2 * t + e][g] = bf((b0 >> (16 * e)) & 0xFFFF)
                Bmat[2 * t + 8 + e][g] = bf((b1 >> (16 * e)) & 0xFFFF)
    C += Amat @ Bmat


rows0 = 8     # row group 1 of the tile (rows 8..15), chunk c = 5 (k-tile 1, q = 1)
c = 5; kt, q = c >> 2, c & 3
Cfull = np.zeros((16 * MT, 8))
for i in range(MT):
    Amat = np.zeros((16, 16)); Bmat = np.zeros((16, 8)); C = np.zeros((16, 8))
    for j in range(4):
        A = {}; B = {}
        for g in range(8):
            for t in range(4):
                row = rows0 + g
                lo_off = ((0 * KT + kt) * 32 + row) * 256 + 64 * q + 16 * t
                nb_off = ((0 * KT + kt) * 32 + row) * 128 + 32 * q + 8 * t
                Lw = [int.from_bytes(lo[lo_off + 4 * e: lo_off + 4 * e + 4].tobytes(), "little") for e in range(4)]
                Nw = [int.from_bytes(nb[nb_off + 4 * e: nb_off + 4 * e + 4].tobytes(), "little") for e in range(2)]
                n16 = [Nw[0] & 0xFFFF, Nw[0] >> 16, Nw[1] & 0xFFFF, Nw[1] >> 16][j]
                B[(g, t)] = decode4(Lw[j], n16)
                k0 = kt * 256 + 64 * q + 16 * t                  # the lane's 16 real k
                def xw(r, e):   # x row r, word e (elements 2e, 2e+1 of the 16-group)
                    return int(xb[r, k0 + 2 * e]) | (int(xb[r, k0 + 2 * e + 1]) << 16)
                ra, rb = 16 * i + g, 16 * i + g + 8
                A[(g, t)] = (xw(ra, 2 * j), xw(rb, 2 * j), xw(ra, 2 * j + 1), xw(rb, 2 * j + 1))
        mma(C, A, B)
    Cfull[16 * i: 16 * i + 16] = C
# truth: x[:, chunk k] @ W[rows, chunk k]^T with the STORED (pre-escape) weights
dec = S._unpack12(pk, N, K).float()
k0, k1 = kt * 256 + 64 * q, kt * 256 + 64 * q + 64
truth = (x.float()[:, k0:k1] @ dec[rows0: rows0 + 8, k0:k1].T).numpy()
err = np.abs(Cfull - truth).max() / (np.abs(truth).max() + 1e-30)
print("max rel err of the emulated warp-step vs truth:", err)
assert err < 1e-5, "fragment mapping wrong"
print("mma12 fragment mapping ok")
