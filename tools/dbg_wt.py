import sys, math
sys.path.insert(0, '/work/muse_moe_w4a16')
import torch, numpy as np
from cuda_moe import repack_weights_wt
from nvfp4 import dequantize, random_weights
E, N, K = 1, 16, 128
gen = torch.Generator().manual_seed(1)
w, sf, gs = random_weights(N, K, generator=gen, device="cpu", leading=(E,))
ref = dequantize(w, sf, gs)
Wr, Sr = repack_weights_wt(w, sf)
words = Wr.view(E, 1, 1, 2, 32, 4, 4).view(torch.int32).view(E, 1, 1, 2, 32, 4)
S3max = float(sf.float().max()) * float(gs) * 2.0 ** 21
m = max(0, math.ceil(math.log2(S3max / 32768.0)))
gsf = float(gs) * 2.0 ** (14 - m)
masks = {0: 0x8E008E00, 1: 0x80708070, 2: 0x8E008E00, 3: 0x801C801C}
shifts = {0: 0, 1: 1, 2: 3, 3: 2}
lo, hi = w & 0x0F, w >> 4
nib = torch.stack([lo, hi], -1).view(E, N, K)
cnt = {}
for lane in range(32):
    g, t = lane // 4, lane % 4
    for s in range(8):
        wv = int(words[0, 0, 0, s // 4, lane, s % 4]) & 0xFFFFFFFF
        sb = Sr[0, 0, 0, g]
        s_g = float(sb[2 * s].view(torch.float8_e4m3fn).float()); s_g8 = float(sb[2 * s + 1].view(torch.float8_e4m3fn).float())
        for p, (row_off, k_off, sc, jj) in enumerate([(0, 0, s_g, 0), (8, 0, s_g8, 5), (0, 8, s_g, 0), (8, 8, s_g8, 7)]):
            val = ((wv << shifts[p]) & 0xFFFFFFFF) & masks[p]
            half = np.array([val & 0xFFFF, val >> 16], dtype=np.uint16).view(np.float16).astype(np.float64)
            row = g + row_off; k0 = s * 16 + 2 * t + k_off
            exp_unscaled = ref[0, row, k0:k0+2].numpy() / (sc * float(gs))      # e2m1 values
            got_unscaled = half * 2.0 ** (14 + jj)
            ok = np.allclose(exp_unscaled, got_unscaled, atol=1e-6)
            cnt[(p, ok)] = cnt.get((p, ok), 0) + 1
            if not ok and cnt[(p, ok)] <= 3:
                print(f"p={p} lane={lane} s={s} row={row} k0={k0} word={wv:08x} val={val:08x} got={got_unscaled} exp={exp_unscaled} nibs={[int(nib[0,row,k0]), int(nib[0,row,k0+1])]} s_g={s_g} s_g8={s_g8} sf={float(sf[0,row,s].float())}")
print(cnt)
