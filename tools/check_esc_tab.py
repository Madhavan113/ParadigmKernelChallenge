"""CPU check of _esc_tab (the t12c kernel's per-tile escape table): rebuild every escape from the table (tile, slot ->
k, local row, delta), compare the set with _pack12's per-row lists, check the k ordering within each tile and the
sentinel fill; then emulate the kernel's per-chunk application (entries with k in the chunk, x from the chunk) and
compare with x @ W^T. Run in the judge image: python3 check_esc_tab.py"""
import sys
sys.path.insert(0, "/work/muse_bf16_proj")
import torch
import submission as S
import cuda_proj as C

torch.manual_seed(1)
n, k = 256, 1024
w = (torch.randn((n, k)) * 3).to(torch.bfloat16)
w[3, 5] = 0.0; w[70, 999] = 1e-30; w[70, 7] = 0.0; w[255, 0] = 0.0
pk = S._pack12(w)
assert pk is not None and pk["n_esc"] >= 4
roff = pk["esc_roff"]
truth = set()
for row in range(n):
    for e in range(int(roff[row]), int(roff[row + 1])):
        truth.add((row, int(pk["esc_k"][e]), float(pk["esc_delta"][e])))
for r in (16, 32):
    tab = C._esc_tab(pk, n, k, r)
    assert tab is not None and tuple(tab.shape) == (n // r, 64, 2)
    got = set()
    for T in range(n // r):
        last = -1
        for j in range(64):
            w0, w1 = int(tab[T, j, 0]), int(tab[T, j, 1])
            kk = w0 & 0xFFFF
            if kk == 0xFFFF:
                assert all(int(tab[T, jj, 0]) & 0xFFFF == 0xFFFF for jj in range(j, 64)), "sentinel must be a suffix"
                break
            assert kk >= last, "entries must be sorted by k"
            last = kk
            row = T * r + (w0 >> 16)
            got.add((row, kk, torch.tensor([w1], dtype=torch.int32).view(torch.float32).item()))
    assert got == truth, (r, len(got), len(truth))
    # emulate the kernel: per chunk, apply entries with k in the chunk from x
    m, KC = 13, 512
    x = torch.randn((m, k)).to(torch.bfloat16)
    out = x.float() @ S._unpack12(pk, n, k).float().T
    for T in range(n // r):
        ep = 0
        for c in range(k // KC):
            kend = (c + 1) * KC
            while ep < 64:
                w0 = int(tab[T, ep, 0]); kk = w0 & 0xFFFF
                if kk >= kend:
                    break
                row = T * r + (w0 >> 16)
                d = torch.tensor([int(tab[T, ep, 1])], dtype=torch.int32).view(torch.float32).item()
                out[:, row] += d * x[:, kk].float()
                ep += 1
    ref = x.float() @ w.float().T
    err = float((out - ref).norm() / ref.norm())
    print(f"r={r}: {len(got)} escapes match, per-chunk emulation rel err {err:.2e}")
    assert err < 1e-5
# the cap: a tile with 65 escapes -> None
w2 = w.clone(); w2[0, :65] = 0.0
pk2 = S._pack12(w2)
assert C._esc_tab(pk2, n, k, 32) is None and C._esc_tab(pk2, n, k, 16) is None
print("esc_tab ok")
