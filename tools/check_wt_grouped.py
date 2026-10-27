"""CPU check of repack_wt_grouped: for every (expert, CTA x, slot ti, chunk) the grouped 1 KB weight block and 128 B
scale block must equal Wt / St at tile_of(x, ti) = (x*4 + w)*mw + i % mw (+ ntiles/2 for i >= mw when gated), ti = w*tpw + i,
i.e. the wt kernel's own CTA tile order, so one bulk copy per chunk lands in the stage layout [ti][1 KB] / [ti][128 B].
Run in the judge image: python3 check_wt_grouped.py"""
import sys, itertools
sys.path.insert(0, "/work/muse_moe_w4a16")
import torch
import cuda_moe as C

torch.manual_seed(3)
E, N, K = 2, 512, 512
w = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8)
sf = torch.randint(0, 256, (E, N, K // 16), dtype=torch.uint8).view(torch.float8_e4m3fn)
Wt, St = C.repack_weights_wt(w, sf)
NTL, NC = N // 16, K // 128
for gated, mw in ((True, 1), (False, 1), (False, 2)):
    ntiles = NTL
    Wg, Sg = C.repack_wt_grouped(Wt, St, ntiles, gated, mw)
    tpw = (2 if gated else 1) * mw
    T = tpw * 4
    n_x = (ntiles // 2 if gated else ntiles) // (4 * mw)
    assert tuple(Wg.shape) == (E, n_x, NC, T, 1024) and tuple(Sg.shape) == (E, n_x, NC, T, 128), (Wg.shape, Sg.shape)
    assert Wg.numel() == Wt.numel() and Sg.numel() == St.numel()
    bad = 0
    Wf = Wt.reshape(E, NTL, NC, 1024); Sf = St.reshape(E, NTL, NC, 128)
    for e, x, ti, c in itertools.product(range(E), range(n_x), range(T), range(NC)):
        wq, i = ti // tpw, ti % tpw
        tile = (x * 4 + wq) * mw + (i % mw) + (ntiles // 2 if (gated and i >= mw) else 0)
        bad += int(not torch.equal(Wg[e, x, c, ti], Wf[e, tile, c])) + int(not torch.equal(Sg[e, x, c, ti], Sf[e, tile, c]))
    print(f"gated={gated} mw={mw}: T={T} n_x={n_x} block mismatches {bad}")
    assert bad == 0
print("wt grouped ok")
