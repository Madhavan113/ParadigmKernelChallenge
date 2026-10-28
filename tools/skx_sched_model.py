"""CPU model of the skx (persistent stream-K wgmma GEMM) schedule in muse_bf16_proj/submission.py.

Mirrors the kernel's integer formulas exactly (all operands non-negative, so C and Python division agree):
  G CTAs, C = K/64 chunks per row block, RB = N/128 row blocks, U = RB*C work units in row-block-major order
  u0(i) = i*U/G, u1(i) = (i+1)*U/G                      (CTA i owns units [u0, u1): 1 or 2 segments)
  cta_of(u) = ((u+1)*G - 1)/U                            (the CTA whose range holds unit u)
  first(rb) = cta_of(rb*C), last(rb) = cta_of(rb*C+C-1)  (P = last-first+1 CTAs touch row block rb)
  segidx(i', rb) = (u0(i')/C == rb) ? 0 : 1              (which of CTA i's segments is row block rb)
  slab(i', seg) = (seg ? G : 0) + i'                     (FIX=1: per-CTA-segment fp32 partial slabs)
  slice(i, rb) = [cum0*NF4/C, cum1*NF4/C)                (FIX=1: the float4 indices CTA i reduces, cum = unit offsets)
Checks: every unit covered exactly once; <= 2 segments per CTA; P[rb] consistent; slices partition the tile; and a
numpy emulation of both fixup modes (FIX=0 atomics + last arriver, FIX=1 dist reduce) in wgmma fragment order
reproduces x @ W^T for every (layer, m) shape.  Usage: python3 tools/skx_sched_model.py
"""
import sys
import numpy as np

LAYERS = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096)}
G = 132
BM = 128
NTH = 256


def frag_pos(tid, q, BN):
    """wgmma m64nBN fp32 accumulator element q of consumer thread tid (two warpgroups): (row in tile, token)."""
    wg, t = tid >> 7, tid & 127
    w, l = t // 32, t % 32
    j, pr = q // 4, q % 4
    return 64 * wg + 16 * w + (l >> 2) + (8 if pr >= 2 else 0), 8 * j + 2 * (l & 3) + (pr & 1)


def schedule(N, K, G):
    C, RB = K // 64, N // BM
    U = RB * C
    assert U >= G, "every CTA needs a unit"
    u0 = [(i * U) // G for i in range(G)]
    u1 = [((i + 1) * U) // G for i in range(G)]
    cta_of = lambda u: ((u + 1) * G - 1) // U
    segs = {}   # i -> [(rb, ua, ub)]
    for i in range(G):
        assert u1[i] - u0[i] <= C, "range longer than a row block"
        assert u1[i] > u0[i]
        rb0, rbl = u0[i] // C, (u1[i] - 1) // C
        assert rbl - rb0 <= 1
        lst = []
        for rb in range(rb0, rbl + 1):
            ua, ub = max(u0[i], rb * C), min(u1[i], (rb + 1) * C)
            lst.append((rb, ua, ub))
        segs[i] = lst
    # coverage
    cov = np.zeros(U, dtype=np.int32)
    for i, lst in segs.items():
        for rb, ua, ub in lst:
            cov[ua:ub] += 1
    assert (cov == 1).all(), "units not covered exactly once"
    # cta_of consistency
    for i in range(G):
        for u in range(u0[i], u1[i]):
            assert cta_of(u) == i, (i, u, cta_of(u))
    P = {}
    for rb in range(RB):
        f, l = cta_of(rb * C), cta_of(rb * C + C - 1)
        P[rb] = l - f + 1
        touch = [i for i in range(G) if any(s[0] == rb for s in segs[i])]
        assert touch == list(range(f, l + 1)), (rb, touch, f, l)
        for i in touch:
            si = 0 if u0[i] // C == rb else 1
            assert segs[i][si][0] == rb, (i, rb, si, segs[i])
    return C, RB, U, u0, u1, segs, P, cta_of


def emulate(N, K, M, BN, FIX, rng):
    C, RB, U, u0, u1, segs, P, cta_of = schedule(N, K, G)
    NB = BN // 8
    NF4 = NB * NTH
    x = rng.standard_normal((M, K)).astype(np.float32)
    W = (rng.standard_normal((N, K)) / np.sqrt(K)).astype(np.float32)
    ref = x @ W.T
    xt = np.zeros((BN, K), dtype=np.float32)   # the x tile: rows >= M are garbage (stale smem)
    xt[:M] = x
    xt[M:] = rng.standard_normal((BN - M, K)) * 1e3
    out = np.full((M, N), np.nan, dtype=np.float32)
    written = np.zeros((M, N), dtype=np.int32)
    # per-(CTA, segment) partial tiles in fragment order: frag[j*NTH + tid] = float4 -> we keep [BM, BN] tiles and map
    def partial(i, rb, ua, ub):
        k0, k1 = (ua - rb * C) * 64, (ub - rb * C) * 64
        return W[rb * BM:(rb + 1) * BM, k0:k1] @ xt[:, k0:k1].T   # [BM rows, BN tokens]

    NACC = BN // 2
    fr = np.zeros((NF4, 4), dtype=np.int64)
    ft = np.zeros((NF4, 4), dtype=np.int64)
    for tid in range(NTH):
        for q in range(NACC):
            r, t = frag_pos(tid, q, BN)
            fr[(q // 4) * NTH + tid, q % 4] = r
            ft[(q // 4) * NTH + tid, q % 4] = t

    def tile_to_frag(T):
        """[BM, BN] fp32 -> fragment-ordered float4 array [NF4, 4] (wgmma accumulator layout, 2 warpgroups)."""
        return T[fr, ft]

    def store_frag(rb, idx, v):
        """float4 index idx of row block rb's tile -> output positions."""
        j, tp = idx // NTH, idx % NTH
        wg, w4, lane = tp >> 7, (tp >> 5) & 3, tp & 31
        g, t4 = lane >> 2, lane & 3
        row = rb * BM + wg * 64 + w4 * 16 + g
        t0 = 8 * j + 2 * t4
        for (dt, dr, val) in ((0, 0, v[0]), (1, 0, v[1]), (0, 8, v[2]), (1, 8, v[3])):
            if t0 + dt < M:
                out[t0 + dt, row + dr] = val
                written[t0 + dt, row + dr] += 1

    if FIX == 0:
        acc = np.zeros((RB, NF4, 4), dtype=np.float32)
        cnt = np.zeros(RB, dtype=np.int32)
        order = list(range(G))
        rng.shuffle(order)
        for i in order:
            for (rb, ua, ub) in segs[i]:
                acc[rb] += tile_to_frag(partial(i, rb, ua, ub))
                cnt[rb] += 1
                if cnt[rb] == P[rb]:   # last arriver
                    for idx in range(NF4):
                        store_frag(rb, idx, acc[rb, idx])
                    acc[rb] = 0
                    cnt[rb] = 0
        assert (cnt == 0).all() and (acc == 0).all()
    else:
        slab = {}
        for i in range(G):
            for si, (rb, ua, ub) in enumerate(segs[i]):
                slab[(si * G if si else 0) + i] = tile_to_frag(partial(i, rb, ua, ub))
        for i in range(G):
            for (rb, ua, ub) in segs[i]:
                f, l = cta_of(rb * C), cta_of(rb * C + C - 1)
                cum0, cum1 = ua - rb * C, ub - rb * C
                lo, hi = cum0 * NF4 // C, cum1 * NF4 // C
                for idx in range(lo, hi):
                    v = np.zeros(4, dtype=np.float32)
                    for ip in range(f, l + 1):
                        si = 0 if u0[ip] // C == rb else 1
                        v += slab[(G if si else 0) + ip][idx]
                    store_frag(rb, idx, v)
    assert (written == 1).all(), "output positions not written exactly once"
    err = np.linalg.norm(out - ref) / np.linalg.norm(ref)
    return err, P


def main():
    rng = np.random.default_rng(0)
    for layer, (N, K) in LAYERS.items():
        C, RB, U, u0, u1, segs, P, cta_of = schedule(N, K, G)
        units = sorted({u1[i] - u0[i] for i in range(G)})
        nseg = sorted({len(segs[i]) for i in range(G)})
        print(f"{layer:9s} N={N} K={K}: C={C} RB={RB} U={U} units/CTA={units} segments/CTA={nseg} "
              f"P[rb] in {sorted(set(P.values()))} straddlers={sum(len(segs[i]) == 2 for i in range(G))}")
    bad = 0
    for layer, (N, K) in LAYERS.items():
        for BN, M in ((128, 128), (128, 33), (256, 256), (256, 200)):
            for FIX in (0, 1):
                err, P = emulate(N, K, M, BN, FIX, rng)
                ok = err < 1e-4
                bad += not ok
                print(f"  emulate {layer} BN={BN} M={M} FIX={FIX}: rel err {err:.2e} {'ok' if ok else 'FAIL'}")
    print("schedule model:", "ok" if not bad else f"{bad} FAILURES")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
