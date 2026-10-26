"""One-off edit: remove the W2 L2-prefetch feature (createpolicy / cp.async cache_hint / prefetch.evict_last / early
trigger): check 16d5a4bc died with a sticky illegal-instruction error on the first wt launch with l2pf on. The l2pf
argument stays in the plumbing but is inert."""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
p = root / "muse_moe_w4a16/cuda_moe.py"
s = p.read_text()


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:90])
    s = s.replace(old, new)


rep("""__device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2::evict_last [%0];\\n" :: "l"(p)); }
__device__ __forceinline__ void cp_async16_pol(uint32_t dst, const void* src, int src_bytes, uint64_t pol) {
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;\\n" :: "r"(dst), "l"(src), "r"(src_bytes), "l"(pol));
}
""", "")
rep("""    if (GATED && l2pf) pdl_trigger();                 // small batches: let GEMM2's CTAs pre-launch and prefetch W2
""", "")
rep("""    uint64_t pol;
    if (l2pf) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\\n" : "=l"(pol));
    else asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;\\n" : "=l"(pol));
    auto issue_w = [&](int c, int stage) {
        const uint32_t ws = smem_base + stage * STAGE_BYTES;
        const uint32_t ss = ws + W_STAGE;
        #pragma unroll
        for (int i = 0; i < WSEG; ++i)
            cp_async16_pol(ws + (tid + i * NTHR) * 16, wsrc[i] + c * 1024, 16, pol);
        if (tid < TPC * 8) cp_async16_pol(ss + tid * 16, ssrc + c * 128, 16, pol);
    };""", """    (void)l2pf;                                       // the L2-prefetch experiment was removed (illegal-instruction fault)
    auto issue_w = [&](int c, int stage) {
        const uint32_t ws = smem_base + stage * STAGE_BYTES;
        const uint32_t ss = ws + W_STAGE;
        #pragma unroll
        for (int i = 0; i < WSEG; ++i)
            cp_async16(ws + (tid + i * NTHR) * 16, wsrc[i] + c * 1024, 16);
        if (tid < TPC * 8) cp_async16(ss + tid * 16, ssrc + c * 128, 16);
    };""")
a = s.index("        if (l2pf && (tid & 7) == 0) {\n")
b = s.index("        pdl_wait();\n", a)
s = s[:a] + s[b:]
rep("""    if (!(GATED && l2pf)) pdl_trigger();              // main loop done: the next kernel may start launching""",
    """    pdl_trigger();                                    // main loop done: the next kernel may start launching""")
assert "l2pf &&" not in s and "createpolicy" not in s and "cache_hint" not in s and "prefetch.global" not in s
p.write_text(s)
print("cuda l2pf removed")

q = root / "muse_moe_w4a16/submission.py"
t = q.read_text()


def rep2(old, new, n=1):
    global t
    assert t.count(old) == n, (t.count(old), old[:90])
    t = t.replace(old, new)


rep2("""L2PF_MAX_TOKENS = 16   # up to this many tokens GEMM2 prefetches its W2 slice into L2 while GEMM1 runs (PDL)""",
     """L2PF_MAX_TOKENS = 0    # W2 L2-prefetch experiment removed from the kernel (illegal-instruction fault on the judge); inert""")
rep2("""    for name, var in (("mixed", "_MIXED_OVERRIDE"), ("l2pf", "_L2PF_OVERRIDE")):""",
     """    for name, var in (("mixed", "_MIXED_OVERRIDE"),):""")
q.write_text(t)
print("submission l2pf disabled")
