"""CPU emulation of gqx_kernel's CL > 1 epilogue (muse_bf16_proj/submission.py): every rank parks its wgmma accumulator
fragments in frag[j][tid] (float4), the owner of token block j sums the CL copies, stages BF16 [token][BM] and writes
16-byte chunks. Checks, for each (BN, NWG, CL) config: every (token < M, row) of the tile is written exactly once with the
sum over ranks of the right fragment element, for M in {BN, BN-3, 1}. Usage: python3 tools/emulate_gqx_reduce.py"""
import importlib.util, pathlib, sys
import numpy as np
root = pathlib.Path(__file__).resolve().parent.parent
src = (root / "muse_bf16_proj/submission.py").read_text()
a = src.index("GQX_CONFIGS = {"); b = src.index("USE_GQX = True")
ns = {}; exec(src[a:b], ns)
cfgs = sorted({(bn, nwg, cl, nst) for (layer, bn), lst in ns["GQX_CONFIGS"].items() for (nwg, cl, nst) in lst})


def frag_elem(acc_of, t, q):
    """wgmma m64nBN fp32 accumulator layout: thread t of a warpgroup (warp w = t / 32, lane l), element q:
    j = q / 4, pair = q % 4 -> row = 16 w + (l / 4) + 8 * (pair >= 2), col (token) = 8 j + 2 (l % 4) + (pair & 1)."""
    w, l = t // 32, t % 32
    j, pr = q // 4, q % 4
    return 16 * w + (l >> 2) + (8 if pr >= 2 else 0), 8 * j + 2 * (l & 3) + (pr & 1)


bad = 0
for bn, nwg, cl, nst in cfgs:
    if cl == 1:
        continue
    BM, NTH, NB, NACC = 64 * nwg, 128 * nwg, bn // 8, bn // 2
    ST = BM * 128 + bn * 128
    JM = (NB + cl - 1) // cl
    assert NB * NTH * 16 + JM * 8 * BM * 2 <= nst * ST, (bn, nwg, cl, nst)
    for M in (bn, bn - 3, 1):
        rng = np.random.default_rng(1)
        # per rank, per thread, acc[NACC]: the true partial for (row, token) of that rank
        part = rng.standard_normal((cl, BM, bn)).astype(np.float32)
        out = np.full((M, BM), np.nan, dtype=np.float32)
        writes = np.zeros((M, BM), dtype=np.int32)
        for r in range(cl):
            jlo, jhi = (r * NB) // cl, ((r + 1) * NB) // cl
            stg = np.full((JM * 8, BM), np.nan, dtype=np.float32)
            for tid in range(NTH):
                wg, warp, lane = tid // 128, tid // 32, tid % 32
                g, t4 = lane >> 2, lane & 3
                rl = wg * 64 + (warp & 3) * 16 + g
                for j in range(jlo, jhi):
                    # v = own frag[j][tid] + other ranks' frag[j][tid]: element q = 4 j + e of thread t = tid & 127 of wg
                    t = tid & 127
                    vs = []
                    for e in range(4):
                        row, col = frag_elem(None, t, 4 * j + e)
                        row += wg * 64
                        vs.append(sum(part[r2, row, col] for r2 in range(cl)))
                    tl = 8 * (j - jlo) + 2 * t4
                    stg[tl, rl] = vs[0]; stg[tl, rl + 8] = vs[2]; stg[tl + 1, rl] = vs[1]; stg[tl + 1, rl + 8] = vs[3]
            CH = BM // 8
            nt = 8 * (jhi - jlo)
            for idx in range(nt * CH):
                tt, c = idx // CH, idx % CH
                tok = 8 * jlo + tt
                if tok < M:
                    out[tok, c * 8:(c + 1) * 8] = stg[tt, c * 8:(c + 1) * 8]
                    writes[tok, c * 8:(c + 1) * 8] += 1
        ref = part.sum(0).T[:M]   # [token][row]
        ok = np.all(writes == 1) and np.allclose(out, ref)
        if not ok:
            bad += 1
        print(f"bn={bn} nwg={nwg} cl={cl} nst={nst} M={M}: {'ok' if ok else 'MISMATCH'} (writes min/max {writes.min()}/{writes.max()})")
print("bad", bad)
sys.exit(1 if bad else 0)
