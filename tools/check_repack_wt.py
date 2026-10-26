"""CPU check of repack_weights_wt + the kernel's dequant arithmetic (mask/shift spread, E4M3 -> fp16,
fp16 scale multiply) against nvfp4.dequantize. Run in the judge image: python3 check_repack_wt.py"""
import sys, math
sys.path.insert(0, '/work/muse_moe_w4a16')
import torch, numpy as np
from cuda_moe import repack_weights_wt
from nvfp4 import dequantize, random_weights

E, N, K = 2, 64, 512
gen = torch.Generator().manual_seed(1)
w, sf, gs = random_weights(N, K, generator=gen, device="cpu", leading=(E,))
ref = dequantize(w, sf, gs)                       # [E, N, K] fp32
Wr, Sr = repack_weights_wt(w, sf)
NTL, NC = N // 16, K // 128
words = Wr.view(E, NC, NTL // 4, 4, 2, 32, 4, 4).permute(0, 2, 3, 1, 4, 5, 6, 7).reshape(E, NTL, NC, 2, 32, 16).contiguous().view(torch.int32).view(E, NTL, NC, 2, 32, 4)
Sr = Sr.view(E, NC, NTL // 4, 4, 8, 16).permute(0, 2, 3, 1, 4, 5).reshape(E, NTL, NC, 8, 16)
gsf = float(gs) * 64.0            # applied to the fp32 sums in the kernel; here to each exact fp16 weight


def e4m3x2_to_f16x2(v16):
    b = np.array([v16 & 0xFF, (v16 >> 8) & 0xFF], dtype=np.uint8)
    return torch.from_numpy(b).view(torch.float8_e4m3fn).to(torch.float16)


max_err = 0.0
n = 0
cnt = {}
for e in range(E):
    for tile in range(NTL):
        for c in range(NC):
            for lane in range(32):
                g, t = lane // 4, lane % 4
                for s in range(8):
                    wv = int(words[e, tile, c, s // 4, lane, s % 4]) & 0xFFFFFFFF
                    sb = Sr[e, tile, c, g]
                    sp = e4m3x2_to_f16x2(int(sb[2 * s]) | (int(sb[2 * s + 1]) << 8))
                    S0, S1 = sp[0], sp[1]
                    elo = wv & 0x9C9C9C9C
                    ehi = ((wv << 2) & 0x8C8C8C8C) | ((wv >> 2) & 0x10101010)
                    pairs = [(e4m3x2_to_f16x2(elo & 0xFFFF), S0, 0, 0), (e4m3x2_to_f16x2(elo >> 16), S1, 8, 0),
                             (e4m3x2_to_f16x2(ehi & 0xFFFF), S0, 0, 8), (e4m3x2_to_f16x2(ehi >> 16), S1, 8, 8)]
                    for p, (half, S, row_off, k_off) in enumerate(pairs):
                        got = (half * S).to(torch.float16).float() * gsf
                        row = tile * 16 + g + row_off
                        k0 = c * 128 + s * 16 + 2 * t + k_off
                        exp = ref[e, row, k0:k0 + 2]
                        err = float((got - exp).abs().max() / (exp.abs().max() + 1e-12))
                        ok = err < 1e-5
                        cnt[(p, ok)] = cnt.get((p, ok), 0) + 1
                        if not ok and cnt[(p, ok)] <= 3:
                            print(f"p={p} lane={lane} s={s} word={wv:08x} got={got.tolist()} exp={exp.tolist()}")
                        max_err = max(max_err, err)
                        n += 1
print(cnt)
print(f"checked {n} pairs: max rel err {max_err:.3e} (should be ~1e-7: exact fp16 weights)")
assert max_err < 1e-5
print("OK")
