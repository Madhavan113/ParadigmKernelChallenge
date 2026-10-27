"""CPU check of _repack_t12 (the t12 kernel's bulk-copy weight layout): for every (R-row tile, chunk, warp unit, lane,
k16 step) rebuild the 12-bit codes from the layout with the kernel's addressing (stage-relative offsets lo_off / nb_off,
L0/L1 order, nibble i) and compare with _pack12's lo / nibble codes; then emulate decode4 and compare the bf16 bits
with _unpack12; finally emulate the whole kernel's math (escapes included) against x @ W^T for one (R, CK).
Run in the judge image: python3 check_tma12.py"""
import sys, itertools
sys.path.insert(0, "/work/muse_bf16_proj")
import torch
import submission as S
import cuda_proj as C

torch.manual_seed(0)
n, k = 128, 1024
w = (torch.randn((n, k)) * 3).to(torch.bfloat16)
w[3, 5] = 0.0; w[70, 999] = 1e-30   # forced escapes
pk = S._pack12(w)
assert pk is not None and pk["n_esc"] >= 2
w12 = C._repack_w12(pk, n, k)
lo = pk["lo"].view(n // 32, k // 256, 32, 256).permute(0, 2, 1, 3).reshape(n, k)
nb = pk["nib"].view(n // 32, k // 256, 32, 128).permute(0, 2, 1, 3).reshape(n, k // 2)
nib = torch.stack((nb & 0xF, nb >> 4), -1).reshape(n, k)
dec = S._unpack12(pk, n, k).view(torch.int16).numpy().astype("uint16")


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


def bf16(u):
    return torch.tensor([u], dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float().item() if u < 0x8000 else \
        torch.tensor([u - 0x10000], dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float().item()


for R, CK, W in [(32, 2, 8), (16, 2, 4), (32, 4, 8), (32, 2, 4), (16, 2, 8)]:
    KC = 128 * CK
    Wt = C._repack_t12(w12, n, k, R, CK)
    WST = (R // 16) * CK * 3072
    assert tuple(Wt.shape) == (n // R, k // KC, R // 16, CK, 3072) and Wt.is_contiguous()
    flat = Wt.reshape(n // R, k // KC, WST)
    U, UPW, WPR = (R // 8) * CK, (R // 8) * CK // W, W // (R // 16)
    bad_lo = bad_nb = bad_dec = 0
    for T, c in itertools.product(range(n // R), range(k // KC)):
        blk = flat[T, c]
        for warp in range(W):
            r_w = warp // WPR
            for j in range(UPW):
                u = warp * UPW + j; ck = (u % (2 * CK)) >> 1; h = u & 1
                assert r_w == u // (2 * CK)
                woff = (r_w * CK + ck) * 3072
                for lane in range(32):
                    g, t = lane // 4, lane % 4
                    lo_off = woff + 2 * h * 512 + lane * 16
                    nb_off = woff + 2048 + h * 512 + lane * 16
                    nbw = blk[nb_off:nb_off + 16].view(torch.int32).tolist()
                    for ps in range(2):
                        low = blk[lo_off + ps * 512: lo_off + ps * 512 + 16].view(torch.int32).tolist()
                        for q in range(2):
                            L0, L1 = low[2 * q] & 0xFFFFFFFF, low[2 * q + 1] & 0xFFFFFFFF
                            Nw = nbw[2 * ps + q] & 0xFFFFFFFF
                            step = 4 * h + 2 * ps + q                     # k16 step within the 128-block
                            r0, r1 = T * R + r_w * 16 + g, T * R + r_w * 16 + g + 8
                            kb = c * KC + ck * 128 + step * 16 + 2 * t
                            exp = [(r0, kb), (r0, kb + 1), (r1, kb), (r1, kb + 1), (r0, kb + 8), (r0, kb + 9), (r1, kb + 8), (r1, kb + 9)]
                            lob = [(L0 >> (8 * i)) & 0xFF for i in range(4)] + [(L1 >> (8 * i)) & 0xFF for i in range(4)]
                            for i, (rr, kk) in enumerate(exp):
                                bad_lo += lob[i] != int(lo[rr, kk])
                                bad_nb += ((Nw >> (4 * i)) & 0xF) != int(nib[rr, kk])
                            a0, a1 = decode4(L0, Nw & 0xFFFF); a2, a3 = decode4(L1, Nw >> 16)
                            got = [a0 & 0xFFFF, a0 >> 16, a1 & 0xFFFF, a1 >> 16, a2 & 0xFFFF, a2 >> 16, a3 & 0xFFFF, a3 >> 16]
                            for (rr, kk), v in zip(exp, got):
                                bad_dec += int(dec[rr, kk]) != v
    print(f"R={R} CK={CK} W={W}: lo mismatches {bad_lo} nibble {bad_nb} decoded {bad_dec}")
    assert bad_lo == 0 and bad_nb == 0 and bad_dec == 0

# whole-kernel math (R=32, CK=2): stored codes via the layout + escapes == x @ W^T
R, CK = 32, 2
Wt = C._repack_t12(w12, n, k, R, CK)
m = 13
x = torch.randn((m, k)).to(torch.bfloat16)
stored = S._unpack12(pk, n, k).float()
out = x.float() @ stored.T
ero, ek, ed = pk["esc_roff"], pk["esc_k"], pk["esc_delta"]
for row in range(n):
    for e in range(int(ero[row]), int(ero[row + 1])):
        out[:, row] += float(ed[e]) * x[:, int(ek[e])].float()
ref = x.float() @ w.float().T
err = float((out - ref).norm() / ref.norm())
print(f"emulated kernel (stored + per-row escapes) vs x @ W^T: rel err {err:.2e}")
assert err < 1e-5
print("tma12 ok")
