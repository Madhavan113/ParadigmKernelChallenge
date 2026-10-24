import torch, sys
sys.path.insert(0, '/work/muse_moe_w4a16')
from cuda_moe import repack_weights
torch.manual_seed(0)
E, N, K = 2, 16, 256
w = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8)
sf = torch.randint(0, 256, (E, N, K // 16), dtype=torch.uint8).view(torch.float8_e4m3fn)
Wr, Sr = repack_weights(w, sf)
lo, hi = w & 0x0F, w >> 4
nib = torch.stack([lo, hi], -1).view(E, N, K)
bad = 0
for e in range(E):
    for nb in range(N // 8):
        for c in range(K // 128):
            for lane in range(32):
                g, t = lane // 4, lane % 4
                bytes_ = Wr[e, nb, c, lane]
                for step in range(8):
                    b0, b1 = int(bytes_[2 * step]), int(bytes_[2 * step + 1])
                    k0 = 128 * c + 16 * step; n = 8 * nb + g
                    exp = [int(nib[e, n, k0 + 2 * t]), int(nib[e, n, k0 + 2 * t + 8]), int(nib[e, n, k0 + 2 * t + 1]), int(nib[e, n, k0 + 2 * t + 9])]
                    got = [b0 & 0xF, b0 >> 4, b1 & 0xF, b1 >> 4]
                    bad += exp != got
sfu = sf.view(torch.uint8); sbad = 0
for e in range(E):
    for nb in range(N // 8):
        for c in range(K // 128):
            for g in range(8):
                for step in range(8):
                    sbad += int(Sr[e, nb, c, g, step]) != int(sfu[e, 8 * nb + g, 8 * c + step])
print("repack nibble mismatches:", bad, "| scale mismatches:", sbad, "| Wr", tuple(Wr.shape), "Sr", tuple(Sr.shape))
