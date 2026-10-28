"""CPU (numpy) reference of the in-kernel routing of muse_moe_w4a16 (OKC_ROUTE_LOGITS): per token, scores =
sigmoid(logit * ROUTER_MULTIPLIER) in fp32, biased = scores + bias, top-8 by biased with ties to the LOWER expert id,
weights = chosen scores / (sum + 1e-15), local id = (id - local_start) mod 256 if < 32 else -1.

Checks the kernel's warp-cooperative procedure (lane = 8 experts, 8 rounds of a lane-local strict max followed by a
butterfly max with (value desc, index asc) tiebreak, -inf masking) against a direct argsort reference on random
logits WITH deliberate ties, and prints the layer-formula weights. Usage: python3 tools/check_route_logits.py [seed]"""
import sys
import numpy as np

MULT = np.float32(0.1767766952966369)
TOPK, NE = 8, 256


def ref_select(logits, bias):
    """Direct reference: stable sort by (-biased, index)."""
    sc = (np.float32(1) / (np.float32(1) + np.exp(-(logits * MULT)))).astype(np.float32)
    sel = (sc + bias).astype(np.float32)
    T = logits.shape[0]
    ids = np.zeros((T, TOPK), np.int64)
    w = np.zeros((T, TOPK), np.float32)
    for t in range(T):
        order = np.lexsort((np.arange(NE), -sel[t]))  # primary: -sel ascending (sel desc), secondary: index asc
        ids[t] = order[:TOPK]
        picked = sc[t, ids[t]]
        w[t] = picked / (np.float32(picked.sum()) + np.float32(1e-15))
    return ids, w


def kernel_select(logits, bias):
    """The kernel's procedure: lane l holds experts 8l..8l+7; per round a lane-local strict max (lowest k on ties),
    then a butterfly over the 32 lanes keeping (value desc, index asc); the winner is masked to -inf."""
    sc = (np.float32(1) / (np.float32(1) + np.exp(-(logits * MULT)))).astype(np.float32)
    sel = (sc + bias).astype(np.float32).copy()
    T = logits.shape[0]
    ids = np.zeros((T, TOPK), np.int64)
    w = np.zeros((T, TOPK), np.float32)
    for t in range(T):
        s = sel[t].copy()
        chosen, ws = [], []
        for r in range(TOPK):
            m = np.full(32, -np.inf, np.float32)
            gi = np.zeros(32, np.int64)
            for lane in range(32):
                mv, mi = s[lane * 8], 0
                for k in range(1, 8):
                    if s[lane * 8 + k] > mv:
                        mv, mi = s[lane * 8 + k], k
                m[lane], gi[lane] = mv, lane * 8 + mi
            off = 16
            while off:
                om, oi = m[np.arange(32) ^ off], gi[np.arange(32) ^ off]
                take = (om > m) | ((om == m) & (oi < gi))
                m = np.where(take, om, m)
                gi = np.where(take, oi, gi)
                off >>= 1
            assert (gi == gi[0]).all()
            g = int(gi[0])
            chosen.append(g)
            ws.append(sc[t, g])
            s[g] = -np.inf
        ids[t] = chosen
        ws = np.array(ws, np.float32)
        ssum = ((ws[0] + ws[1]) + (ws[2] + ws[3])) + ((ws[4] + ws[5]) + (ws[6] + ws[7]))
        w[t] = ws / (np.float32(ssum) + np.float32(1e-15))
    return ids, w


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = np.random.default_rng(seed)
    T = 64
    logits = rng.standard_normal((T, NE)).astype(np.float32) * np.float32(3)
    bias = (rng.standard_normal(NE).astype(np.float32) * np.float32(0.05))
    # deliberate ties: tokens 0..15 get duplicated logits and zero bias on the duplicates (equal biased scores)
    for t in range(16):
        e = rng.choice(NE, 6, replace=False)
        logits[t, e] = np.float32(12.0)
        bias[e[:3]] = np.float32(0.0)
    logits[16, :] = np.float32(0.0)  # all equal: must pick experts 0..7 (lowest ids)
    ids_ref, w_ref = ref_select(logits, bias)
    ids_k, w_k = kernel_select(logits, bias)
    same = (ids_ref == ids_k).all()
    print(f"ids identical to the sorted reference (incl. ties): {same}")
    if not same:
        bad = np.where((ids_ref != ids_k).any(1))[0]
        print("mismatching tokens:", bad[:10], ids_ref[bad[0]], ids_k[bad[0]])
    print(f"token 16 (all-equal logits) -> {ids_k[16].tolist()}")
    print(f"max |w_kernel - w_ref| = {np.abs(w_k - w_ref).max():.2e} (summation-order ulps)")
    for ls in (0, 32, 224):
        lid = (ids_k - ls + 256) % 256
        loc = np.where(lid < 32, lid, -1)
        print(f"local_start {ls}: local routes {(loc >= 0).sum()} of {T * TOPK}")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
