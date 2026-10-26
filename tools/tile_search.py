"""Nibble placement search for the weights-as-A dequant: pair_p = (w << k_p) & (0x80008000 | (0x0E00 >> j_p) * 0x10001)
must extract 8 disjoint (sign, e1 e0 m) bit-sets tiling the 32-bit word. Recursive with pruning."""
sols = []


def bits_of(k, j):
    u = j + k
    return {31 - k, 15 - k, 27 - u, 26 - u, 25 - u, 11 - u, 10 - u, 9 - u}


def rec(used, chosen):
    if len(chosen) == 4:
        if len(used) == 32:
            sols.append(tuple(chosen))
        return
    for k in range(0, 16):
        if any(c[0] == k for c in chosen):
            continue
        for j in range(0, 10):
            s = bits_of(k, j)
            if min(s) < 0 or len(s) < 8 or used & s:
                continue
            rec(used | s, chosen + [(k, j)])


rec(set(), [])
print(len(sols))
good = [s for s in sols if len({j for _, j in s}) <= 2 and all([j for _, j in s].count(j) == 2 for _, j in s)]
print("two-class:", good[:10])
print("sample:", sols[:12])
shared = [s for s in sols if len({j for _, j in s}) == 3]
shared.sort(key=lambda s: max(j for _, j in s))
print("one shared class, by max j:", shared[:12])
