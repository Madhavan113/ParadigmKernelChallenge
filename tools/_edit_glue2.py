"""One-off edit of submission.py: 5-tuple wt variants (nt, mw2, ks, nstage, ksplit), split-K partial buffer, tight
grid.y, per-variant self-test, new bench candidates."""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
p = root / "muse_moe_w4a16/submission.py"
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:100])
    s = s.replace(old, new)


# ---- forward path
a = s.index("# (token n8 blocks per expert block, GEMM2 m16 tiles per warp, K-split warp groups, cp.async stages)")
b = s.index("# ----------------------------------------------------------------------------------------------\n# host side")
new_fwd = '''# (token n8 blocks per expert block, GEMM2 m16 tiles per warp, K-split warp groups, cp.async stages, K-split CTAs)
WT_DEFAULT = {1: (1, 1, 1, 8, 2), 8: (1, 2, 1, 8, 1), 16: (1, 1, 1, 8, 1), 32: (1, 2, 1, 8, 1), 128: (2, 2, 1, 8, 1), 256: (2, 2, 1, 8, 1)}
WT_SAFE = (1, 1, 1, 4, 1)
PART_SPLIT_MAX = 4      # fp32 partial slices for split-K
PART_ROWS = 1024        # padded rows the partial buffer holds (split-K is a small-batch tool)
_WT_OVERRIDE = None


def _wt_variant(M: int, state):
    if _WT_OVERRIDE is not None:
        return _WT_OVERRIDE
    pol = state.get("wt_policy", WT_DEFAULT)
    keys = sorted(pol)
    v = pol[keys[-1]]
    for k in keys:
        if M <= k:
            v = pol[k]
            break
    if v in state.get("wt_bad", ()):
        v = WT_SAFE
    return v


def _wt_max_blocks(M: int, BM: int) -> int:
    """Blocks are per expert, padded to BM rows: at most one partial block per routed expert plus npairs // BM."""
    npairs = M * TOPK
    return min(npairs, LOCAL) + npairs // BM


def _moe_forward_wt(x, ids, tw, local_start: int, I: int, state) -> torch.Tensor:
    """Experts through the weights-as-A CUDA GEMMs (8*NT-token blocks per expert) and the CUDA finalize."""
    M, H = x.shape
    if ids.dtype != torch.int32:
        ids = ids.to(torch.int32)
    ids = ids.contiguous()
    tw = tw.contiguous()
    nt, mw2, ks, nstage, ksplit = _wt_variant(M, state)
    BM = 8 * nt
    if M * TOPK + LOCAL * (BM - 1) > PART_ROWS:
        ksplit = 1
    NP, MAXB = _route(x, ids, local_start, BM, state, marlin=False)
    maxb = min(MAXB, _wt_max_blocks(M, BM))
    x16 = state["x16"][:M]
    out = torch.empty((M, H), dtype=MOE_OUT_DTYPE, device=x.device)
    ext = build()
    inter = state["inter"]
    y = state["y"]
    part = state["part"]
    ext.moe_w4a16_wt(x16, state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                     state["Wt13"], state["St13"], state["gs13"], inter, part, ksplit, H, 2 * I // 16, maxb, True, 1, nt, ks, nstage,
                     state["gs_mul"])
    if ksplit > 1:
        ext.moe_wt_silu_reduce(part, ksplit, inter, state["blk_row0"], state["blk_end"], state["nblk"], BM, maxb, I)
    ext.moe_w4a16_wt(inter, state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                     state["Wt2"], state["St2"], state["gs2"], y, part, ksplit, I, H // 16, maxb, False, mw2, nt, ks, nstage,
                     state["gs_mul"])
    ext.moe_finalize(y, part, ksplit, state["row_of_pair"], tw, state["norm_w"], out, M, POST_NORM_EPS)
    return out


'''
s = s[:a] + new_fwd + s[b:]

# ---- state: partial buffer
rep('''            state["gs_mul"] = 64.0   # the E4M3 placement is e2m1 * 2^-6; applied with gs to the fp32 sums
            state["xscale"] = 1.0
''', '''            state["gs_mul"] = 64.0   # the E4M3 placement is e2m1 * 2^-6; applied with gs to the fp32 sums
            state["xscale"] = 1.0
            state["part"] = torch.empty((PART_SPLIT_MAX, PART_ROWS, 2 * I), dtype=torch.float32, device=dev)   # split-K partials
''')

# ---- self-test: per variant
a = s.index("def _selftest_wt(state):")
b = s.index("def _bench_wt(state, autotune=False):")
new_self = '''WT_TEST_VARIANTS = ((1, 1, 1, 4, 1), (1, 1, 1, 8, 1), (1, 2, 1, 8, 1), (1, 1, 1, 8, 2), (1, 1, 1, 4, 4), (1, 2, 1, 4, 2),
                    (2, 2, 1, 8, 1), (2, 1, 1, 8, 1), (4, 2, 1, 4, 1), (1, 2, 1, 8, 1))


def _selftest_wt(state):
    """Every wt variant against the Marlin path on judge-like routings; variants that are off are excluded."""
    import time
    global _WT_OVERRIDE
    t0 = time.time()
    dev = state["w13"].device
    worst = {}
    for bs, dist, seed in ((1, "normal", 1), (4, "normal", 2), (33, "normal", 3), (128, "skewed", 4), (257, "normal", 5)):
        x, ids, tw = _judge_inputs(bs, dev, seed=seed, dist=dist)
        ref = _moe_forward_marlin(x, ids, tw, 0, 4096, state).float()
        for v in WT_TEST_VARIANTS:
            _WT_OVERRIDE = v
            try:
                out = _moe_forward_wt(x, ids, tw, 0, 4096, state).float()
                torch.cuda.synchronize()
                err = float((out - ref).norm() / ref.norm().clamp(min=1e-12))
                if not torch.isfinite(out).all():
                    err = float("inf")
            except Exception as e:
                err = float("inf")
                print(f"  selftest bs={bs} {v} failed: {str(e)[:100]}")
            finally:
                _WT_OVERRIDE = None
            worst[v] = max(worst.get(v, 0.0), err)
        print(f"  selftest bs={bs:3d} {dist:6s}: " + " ".join(f"{v}:{worst[v]:.2e}" for v in WT_TEST_VARIANTS))
    bad = {v for v, e in worst.items() if not e < 6e-3}
    state["wt_bad"] = bad
    state["wt_ok"] = WT_SAFE not in bad
    print(f"selftest wt: bad variants {sorted(bad)} -> {'enabled' if state['wt_ok'] else 'DISABLED'} ({time.time() - t0:.0f}s)")


'''
s = s[:a] + new_self + s[b:]

# ---- bench: parts + candidates
a = s.index("    # per-launch breakdown of the wt path at bs 1 and 32")
b = s.index("    print(\"  best per shape:\", best)")
new_bench = '''    # per-launch breakdown of the wt path at bs 1 and 32
    for bs in (1, 32):
        x, ids, tw = data[bs]
        nt, mw2, ks, ns, ksp = _wt_variant(bs, state)
        BM = 8 * nt
        NP, MAXB = _route(x, ids, 0, BM, state, marlin=False)
        maxb = min(MAXB, _wt_max_blocks(bs, BM))
        ext = build()
        x16 = state["x16"][:bs]
        out = torch.empty((bs, 4096), dtype=MOE_OUT_DTYPE, device=dev)
        part = state["part"]
        parts = {
            "route": lambda: _route(x, ids, 0, BM, state, marlin=False),
            "gemm1": lambda: ext.moe_w4a16_wt(x16, state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                                              state["Wt13"], state["St13"], state["gs13"], state["inter"], part, ksp, 4096, 512, maxb, True, 1, nt, ks, ns, 64.0),
            "silu_red": lambda: ext.moe_wt_silu_reduce(part, max(ksp, 2), state["inter"], state["blk_row0"], state["blk_end"], state["nblk"], BM, maxb, 4096),
            "gemm2": lambda: ext.moe_w4a16_wt(state["inter"], state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
                                              state["Wt2"], state["St2"], state["gs2"], state["y"], part, ksp, 4096, 256, maxb, False, mw2, nt, ks, ns, 64.0),
            "finalize": lambda: ext.moe_finalize(state["y"], part, ksp, state["row_of_pair"], tw, state["norm_w"], out, bs, POST_NORM_EPS),
        }
        try:
            print(f"  wt parts bs{bs} {(nt, mw2, ks, ns, ksp)}: " + "  ".join(f"{k} {_timeit_cold(f, dev):6.1f}" for k, f in parts.items())
                  + f"  | local routes {int(((ids >= 0) & (ids < 32)).sum())} distinct {int(torch.unique(ids[(ids < 32)]).numel())}")
        except Exception as e:
            print(f"  wt parts bs{bs} failed: {str(e)[:100]}")
    small = [(1, 1, 1, 4, 1), (1, 1, 1, 8, 1), (1, 2, 1, 8, 1), (1, 1, 1, 8, 2), (1, 2, 1, 8, 2), (1, 1, 1, 4, 4), (1, 2, 1, 4, 4), (1, 2, 1, 4, 2)]
    big = [(2, 2, 1, 4, 1), (2, 2, 1, 8, 1), (2, 1, 1, 8, 1), (4, 2, 1, 4, 1), (1, 2, 1, 8, 1)]
    bad = state.get("wt_bad", set())
    best = {}
    for bs in shapes:
        cands = [v for v in (small if bs <= 32 else big) if v not in bad]
        row = []
        for v in cands:
            _WT_OVERRIDE = v
            try:
                t = _timeit_cold(lambda: _moe_forward_wt(*data[bs], 0, 4096, state), dev)
            except Exception as e:
                t = float("inf")
                print(f"  bs{bs} {v} failed: {str(e)[:80]}")
            finally:
                _WT_OVERRIDE = None
            row.append((t, v))
        print(f"  wt bs{bs:3d}: " + "  ".join(f"{v} {t:7.1f}" for t, v in row))
        best[bs] = min(row)[1]
'''
s = s[:a] + new_bench + s[b:]
p.write_text(s)

g = root / "tools/gen_moe_kernel_only.py"
t = g.read_text()
t = t.replace('"const float*, __half*, __nv_bfloat16*, int, int, int, int, float, cudaStream_t)")',
              '"const float*, __half*, __nv_bfloat16*, int, float*, int, int, int, int, int, int, float, cudaStream_t)")')
t = t.replace('inst.append("void* finalize_ref = (void*)&moe_finalize_kernel;")',
              'inst.append("void* finalize_ref = (void*)&moe_finalize_kernel;")\ninst.append("void* silu_ref = (void*)&wt_silu_reduce_kernel;")')
g.write_text(t)
print("submission.py + gen edited")
