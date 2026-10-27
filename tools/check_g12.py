"""CPU checks for the g12 wgmma kernel (no GPU): (1) the weight layout + decode under the g12 addressing: for every
(tile, chunk, rb, wq, ck, lane, k16 step) the decoded fragment {a0,a1,a2,a3} must be the bf16 bits of W at rows
tile*R + rb*64 + wq*16 + g (+8) and k = chunk*KC + ck*128 + step*16 + 2t (+1, +8, +9) -- which, stacked over the 4 warps
of a warpgroup, is the wgmma m64k16 A register fragment (PTX ISA: warp w of the warpgroup holds rows 16w..16w+15 in the
m16n8k16 A layout). (2) the x tile: simulate the 3D TMA box placement ([group][row][128 B], 128B swizzle: 16-byte chunk
index XOR row & 7) and check that the matrix descriptor's canonical K-major SW128 addressing (start + (row/8)*1024 +
(row%8)*128 + ((chunk ^ (row%8)) << 4), chunk = k16-step*2 + (k%16)/8) lands on x[tok][k] for every step / tb.
(3) the partial-store index math (part[z][n][tok] float2 pairs) against the accumulator layout (warp w rows 16w+g (+8),
columns 8j + 2t (+1)). (4) the escape emulation: partials + per-row escapes == x @ W^T.
Run in the judge image: python3 check_g12.py"""
import sys, itertools
sys.path.insert(0, "/work/muse_bf16_proj")
import torch
import submission as S
import cuda_proj as C

torch.manual_seed(2)
n, k = 512, 512
w = (torch.randn((n, k)) * 3).to(torch.bfloat16)
w[3, 5] = 0.0; w[70, 499] = 1e-30; w[300, 7] = 0.0
pk = S._pack12(w)
assert pk is not None and pk["n_esc"] >= 3
w12 = C._repack_w12(pk, n, k)
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


# (1) weight layout under the g12 addressing
for RB, CK in [(4, 1), (2, 1), (2, 2), (1, 1)]:
    R, KC = 64 * RB, 128 * CK
    Wt = C._repack_t12(w12, n, k, R, CK)
    WST = RB * 4 * CK * 3072
    flat = Wt.reshape(n // R, k // KC, WST)
    bad = 0
    for T, c in itertools.product(range(n // R), range(k // KC)):
        stage = flat[T, c]
        for rb, wq, ck in itertools.product(range(RB), range(4), range(CK)):
            blk = stage[((rb * 4 + wq) * CK + ck) * 3072:][:3072]
            for lane in range(32):
                g, t = lane // 4, lane % 4
                for h in range(2):
                    lo0 = blk[(2 * h) * 512 + lane * 16:][:16].view(torch.int32).tolist()
                    lo1 = blk[(2 * h + 1) * 512 + lane * 16:][:16].view(torch.int32).tolist()
                    nbw = blk[2048 + h * 512 + lane * 16:][:16].view(torch.int32).tolist()
                    for ps in range(2):
                        lo = lo0 if ps == 0 else lo1
                        for q in range(2):
                            L0, L1 = lo[2 * q] & 0xFFFFFFFF, lo[2 * q + 1] & 0xFFFFFFFF
                            Nw = nbw[2 * ps + q] & 0xFFFFFFFF
                            a0, a1 = decode4(L0, Nw & 0xFFFF); a2, a3 = decode4(L1, Nw >> 16)
                            step = 4 * h + 2 * ps + q
                            r0 = T * R + rb * 64 + wq * 16 + g; r1 = r0 + 8
                            kb = c * KC + ck * 128 + step * 16 + 2 * t
                            exp = [(r0, kb), (r0, kb + 1), (r1, kb), (r1, kb + 1), (r0, kb + 8), (r0, kb + 9), (r1, kb + 8), (r1, kb + 9)]
                            got = [a0 & 0xFFFF, a0 >> 16, a1 & 0xFFFF, a1 >> 16, a2 & 0xFFFF, a2 >> 16, a3 & 0xFFFF, a3 >> 16]
                            for (rr, kk), v in zip(exp, got):
                                bad += int(dec[rr, kk]) != v
    print(f"RB={RB} CK={CK}: A-fragment mismatches {bad}")
    assert bad == 0

# (2) x tile boxes + descriptor addressing (one 64-k group per box, rows = 128*TB, 128B swizzle)
for TB, CK in [(1, 1), (2, 1), (1, 2)]:
    m, KC = 128 * TB, 128 * CK
    MP, NBOX, BOXB = 128 * TB, KC // 64, 128 * TB * 128
    x = torch.randn((m, k)).to(torch.bfloat16)
    xb = x.view(torch.int16).numpy().astype("uint16")
    bad = 0
    for c in range(k // KC):
        # simulate the TMA placement of chunk c: box b = k group c*NBOX + b; row r; 16-byte chunk s holds k [8s, 8s+8) of the group
        smem = {}
        for b in range(NBOX):
            for r in range(MP):
                for s_ in range(8):
                    phys = b * BOXB + r * 128 + ((s_ ^ (r & 7)) << 4)
                    for e in range(8):
                        smem[phys + 2 * e] = xb[r, (c * NBOX + b) * 64 + 8 * s_ + e] if r < m else 0
        for tb in range(TB):
            for ck in range(CK):
                for h in range(2):
                    box = (ck * 2 + h) * BOXB
                    for ps in range(2):
                        for q in range(2):
                            start = box + tb * 16384 + ps * 64 + q * 32      # the descriptor's start address (stage-relative)
                            kstep = ck * 128 + h * 64 + ps * 32 + q * 16    # k offset of the k16 step within the chunk
                            for tok in range(128):                          # the n128 rows of the B operand
                                row = tok                                   # tb's rows start at the +16 KB offset
                                for kk in range(16):
                                    chunk = ((start % 128) + 2 * kk) // 16  # logical 16-byte chunk within the 128 B row
                                    addr = (start - start % 128) + (row // 8) * 1024 + (row % 8) * 128 + ((chunk ^ (row % 8)) << 4) + (2 * kk) % 16
                                    want = xb[tb * 128 + tok, c * KC + kstep + kk]
                                    bad += smem[addr] != want
    print(f"TB={TB} CK={CK}: descriptor addressing mismatches {bad}")
    assert bad == 0

# (3) + (4): accumulator layout -> partial stores -> reduce with escapes == x @ W^T (fp32)
RB, TB, CK, KS = 2, 1, 1, 2
R, KC, m = 64 * RB, 128 * CK, 128
x = torch.randn((m, k)).to(torch.bfloat16)
stored = S._unpack12(pk, n, k).float()
part = torch.zeros((KS, n, m))
NC = k // KC // KS
for T, z in itertools.product(range(n // R), range(KS)):
    for rb, wq in itertools.product(range(RB), range(4)):
        rows = T * R + rb * 64 + wq * 16
        kslice = slice(z * NC * KC, (z + 1) * NC * KC)
        D = stored[rows:rows + 16, kslice] @ x.float()[:, kslice].T          # [16 rows][128 tok] = this warp's slab of the 64x128 accumulator
        for lane in range(32):
            g, t = lane // 4, lane % 4
            for j in range(16):
                acc = [D[g, 8 * j + 2 * t], D[g, 8 * j + 2 * t + 1], D[g + 8, 8 * j + 2 * t], D[g + 8, 8 * j + 2 * t + 1]]
                nrow = rows + g
                tok = 8 * j + 2 * t
                part[z, nrow, tok] = acc[0]; part[z, nrow, tok + 1] = acc[1]
                part[z, nrow + 8, tok] = acc[2]; part[z, nrow + 8, tok + 1] = acc[3]
out = part.sum(0).T.clone()                                               # [tok][n]
ero, ek, ed = pk["esc_roff"], pk["esc_k"], pk["esc_delta"]
for row in range(n):
    for e in range(int(ero[row]), int(ero[row + 1])):
        out[:, row] += float(ed[e]) * x[:, int(ek[e])].float()
ref = x.float() @ w.float().T
err = float((out - ref).norm() / ref.norm())
print(f"partials + reduce emulation vs x @ W^T: rel err {err:.2e}")
assert err < 1e-5
print("g12 ok")
