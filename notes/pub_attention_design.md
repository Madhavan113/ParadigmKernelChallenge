# Design memo: published `muse_attention_decode.py` (vjeux/okc-2026, Apache-2.0; check 2.58x vs SGLang FA3, H200)

Source: /tmp/okc_pub/muse_attention_decode.py (extension okc_attn_v298). Line references are to that file.
Reference design only: re-implement. [code] = read from source; [derived] = arithmetic; [note] = author's measurement.

## 0. Summary of what makes it fast

1. ONE kernel launch per call, no memset, no second pass: the fp32 workspace is persistent and starts zeroed; the
   finalizing CTAs zero what they read; counters reset themselves via parity/epoch counters.
2. Splits never exchange a running max: each split adds o*2^m and l*2^m (m = its own offset in log2 units) into the
   workspace with red.global.add.v4.f32, i.e. rescaled to a fixed reference 2^0. QK-norm bounds the logits (|s| <=
   ~63 in log2 units at d=64, so 2^s in [2^-64, 2^64]; fminf(m,126) is only a guard), so fp32 never overflows. The
   last CTA (ticket counting KEYS, not CTAs) or every CTA (cooperative grid barrier) normalizes.
3. Two kernel families: per-request split-KV, grid (n_splits, B, Hkv/2), for sliding windows and single requests;
   "flat" kernel, grid (264,1,Hkv/2) = 2 CTAs/SM, for full context with B > 1: all requests' keys split into equal
   shares (with a per-request-boundary penalty SEGK) and a dedicated producer warpgroup.
4. One CTA = one request x one KV pair (16 query heads). A 128-byte cache row holds both KV heads, fetched once as
   8 x 16-byte cp.async.cg for K and again for V.
5. 64-key tiles in 128B-swizzled smem stages (row r at r*128, chunk c at ((c ^ (r&7)) << 4)): exactly the wgmma B
   descriptor swizzle (desc_sw128, SBO 1024), so q8 wgmma reads K in place with no conversion pass.
6. Math by q_len: q1 = FP16 mma.sync with K and V converted e4m3->f16 in registers (NOT FP8 MMAs); q8 = FP8 wgmma for
   QK^T (m64n64k32) and PV (m64n72k32): 8 extra rows of e4m3 ones in V^T give the row sums for free; lazy softmax
   offset checked by a bit test on the packed P bytes.
7. Sliding windows (weight 46) are latency-bound (1-9 tiles per CTA), so the work went into once-per-launch code:
   speculate seq_len from the page table width (B=1), issue K/V as soon as page indices land, straight-line
   predicated copies built per launch mode, magic-multiplier division, constants via cp.async, cooperative finalize.
8. Per-shape launch plans tuned on the judge with CUDA-graph timings (`_TUNED` overrides L4969-4976).
9. Flat pipeline: producer warpgroup with setmaxnreg (56/200 regs at q1, 32/104 at q8), mbarrier full/empty ring of 3
   (q1) or 4 (q8) stages, q1 page indices by cp.async.bulk 3 tiles ahead, SEGK penalty per request boundary.
10. SASS-level work: uniform registers via lane-0 shuffles, avoiding ptxas C7515 (serializes every wgmma), 32-bit
    offsets, fences replaced by one acq_rel atomic, fewer taken branches on cold code ("60-100 cycles each", L1045).

## 1. Variants, dispatch, resources

`attn_kernel<L, FLAT, MODE>` (L4076-4649), L in {1,8}. MODE bits: MODE&3: 1 = plain split-KV + last-CTA finalize,
2 = clusters (DSMEM; unused for scored shapes), 3 = cooperative finalize; |4 = ONE (every split holds at most one
tile); |8 = SPEC (speculative seq_len). 10 kernels per L; each build carries only its own epilogue ("once-per-launch
code runs from a cold instruction cache", L1699).
Cfg (L511-538): per-request q1: 128 threads (4 warps), MINB 2, 3 stages, no setmaxnreg; per-request q8: 256 (2
warpgroups), 3 stages, <= 128 regs; flat q1: 128 consumers + 128 producers, 3 stages, producers setmaxnreg.dec 56 /
consumers .inc 200; flat q8: 256 + 128, 4 stages, 32/104. Named barriers: 1 consumers, 2 producers, 3/4 per
warpgroup. Warp roles q1: hk = warp&1 = KV head (8 query heads), wsub = warp>>1 = key half of the tile; q8:
warpgroup = KV head, wsub = warp&3 = m-tile (rows 16*wsub..+15 = tokens 2wsub, 2wsub+1 x 8 heads; r = j*8 + head).
Smem (Smem<L,F>, L1193-1245; +1024 for alignment): per-request q1 58 KB (K 24 KB + V 24 KB + idx/pidx/kvq);
per-request q8 94 KB (+ kvq 18 KB for V^T 4 x 4608, q8s 8 KB, kvst/gst); flat q1 62 KB; flat q8 115 KB (2 CTAs/SM =
226 KB). TILE = 64 keys, ROWB = 128 B, STAGE_BYTES = 8 KB per K and per V.
Dispatch (`_plan` L4990-5059, pairs = Hkv/2 = 1 at TP8, target = 264): full layer 1 < B <= 512 -> flat, n_splits =
264, nc = 1 (coop only for q8); full B=1 q8: n = min(tiles, 252), nc = 8, coop; full B=1 q1: _TUNED {nc 8, coop 1,
252 splits}; sliding q1 reqs < 24 -> plain, n = min(tiles, 264/reqs), nc 1; sliding q8 reqs < 16 -> plain, coop;
sliding q8 reqs >= 16 -> n = min(tiles, (264 if reqs >= 24 else 132)/reqs), coop; _TUNED (32,8,sliding) -> 4 splits;
q1 sliding reqs >= 24 -> plain 264/reqs. ONE chosen when n_splits*64 >= kmax (window+L-1 on sliding); SPEC when
B==1 && P==1 && grid.y==1; coop downgraded to plain if the launched instance is not fully resident (occupancy per
instance). Chunk = umulhi(tiles+n-1, ceil(2^32/n)) << 6 keys (whole tiles).
Scored shapes [derived]: decode 1 sliding: <1,false,13> plain|ONE|SPEC, grid (32,1,1), 128 thr, 1 tile per CTA,
last-CTA finalize; decode 16 sliding: <1,false,1> grid (16,16,1), 2 tiles/CTA; decode 32 sliding: (8,32,1), 4
tiles/CTA; verify 1 sliding: <8,false,15> coop|ONE|SPEC, (33,1,1), 256 thr, cooperative finalize 4 rows per CTA;
verify 16 sliding: <8,false,3> coop, (8,16,1) = 128 CTAs one per SM; verify 32 sliding: (4,32,1) = 128 CTAs, 9 tiles
each; decode 1 full: <1,false,11> coop|SPEC, (252,1,1), nc 8 copies; decode 16/32 full: <1,true> flat (264,1,1), 256
thr, ~76/~152 tiles per CTA, last CTA per request; verify 1 full: <8,false,11> (252,1,1); verify 16/32 full: <8,true>
flat, 384 thr, grid-wide finalize. In the generator, batch-1 shapes have max_pages = 80000 + q_len so SPEC always hits.

## 2. Pipeline

Tile fetch: a K/V row at page size 1 = Hkv*64 = 128 B = both heads; 8 threads copy 16 B each with
`cp.async.cg.shared.global [dst],[src],16,src_bytes` (src_bytes 0 zero-fills rows past end); address = (idx << lrb)
+ pair*128 + ch*16 (shift-add). Copies per thread per tile: per-request q1 8 (K+V), q8 4, flat producers 8. K/V by
TMA exists but is compiled out (OKC_TMA 0).
Per-request prologue (`segment()` L1945-2531): (1) seq_ld = __ldg(seq_lens+b); SPEC builds use seq = max_pages - L
immediately; compute lo, n_keys, chunk, start, end; empty splits return without touching the merge. (2) page indices
of tiles 0..STAGES-2 by cp.async.ca 4 B per entry (64 threads), commit group 0. (3) griddepcontrol.wait (no-op). (4)
group 1: q rows (L*16 rows x 128 B) into S.qst, new k/v rows into S.kvst (only if end > seq), constants (inv_freq,
k_scale, v_scale, coop epoch counter, cache_loc); split 0 prefetch.global.L2 of the gate rows. (5) cp_wait<1> +
csync (IDXFIRST: wait only for the 256 B index group); SPEC builds verify the guess here (drain + return false on
mismatch; the host re-runs with the real length). (6) K/V for tiles 0..STAGES-2 (ONE: one predicated straight-line
pass; else unrolled loop issue_kv + issue_idx(k+STAGES-1)). (7) cp_wait<STAGES-1> (q group). (8) RoPE table
(cos/sin L positions x 32 freqs), csync. (9) q pass, new-k/v pass, patch new rows into issued stages. (10) loop:
cp_wait<STAGES-2>, __syncthreads, issue_kv(i+STAGES-1), issue_idx(i+2*STAGES-2), commit (page indices STAGES-1
tiles ahead of K/V, K/V 2 tiles ahead of compute; q8 QKFIRST: the tile's QK^T wgmma issued before the refill).
Flat partition (L4160-4527): thread 32 inits the first segment's stage mbarriers (full[st] count NP, empty[st] count
NT) while warp 0 loads seq_lens (one request per lane, shuffle prefix sum); T = sum n_keys + (B-1)*SEGK (400 q1, 600
q8: crossing a boundary costs a segment, "~7 us at q_len 8"); CTA c gets [cT/G, (c+1)T/G) mapped to real keys (q1
ballots FPART; q8 binary search + serial segment list); PTPF prefetch.global.L2 of the first STAGES tiles of page
indices; run_segments calls segment() per (request, range), each re-initializing barriers, draining, merging.
Flat producer per tile (L1744-1943): q1 page indices by bulk copy (TMAPG): thread 0 copies the 16-B-aligned window
of the tile's 64 indices (<= 272 B) into S.pidx[slot][80], PS=4 slots filled 3 tiles ahead:
`mbar_expect_tx; cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [pidx],[pt+w0],bytes,[pbar]`;
producers wait pbar[slot], read 4 indices; bar.sync 2 frees the slot (q8 uses per-thread __ldg, "TMAPG8 ~+0.7%"
loss). Wait empty[st] (try_wait.parity); issue 8 cp.async per thread (PFAST branch-free shift-add on fully cached
tiles; rows past end zero-filled; rows in [seq,end) left for the consumer patch); then
`cp.async.mbarrier.arrive.noinc.shared::cta.b64 [full[st]]`. Consumers: q1 ws_acquire waits full[st], patches new
rows if any; q8 carries stage/phase counters across iterations (ADDR), RARE finds mask-free tiles once (one compare
per tile), patch is a __noinline__ call, the two warpgroups meet only at tile 0 and new-row tiles (WGDEC, NOSYNC0);
release: every consumer lane mbarrier.arrive on empty[st] (XP_ARRALL, "vf32 -0.8%").
Shortening the chain: SPEC (B=1, P=1): SGLang's page_table width = seq_len + q_len is a SHAPE, so seq = width - L is
guessed and the page indices are issued in the same round trip as the seq_lens load; verified after the index group
lands; chain becomes max(seq_lens, page-idx) -> K/V. IDXFIRST: K/V copies go out once the index group lands while the
q/new-kv/constants group (up to 16 KB) is still in flight. No cold loads mid-prologue (constants in the q group;
epoch read early). Fewer cold instructions: unrolled tfor with predicated tails, PREUNR ("s1 -2%"), ONE straight-line
predicated copies, MAGICDIV + fdiv_i, __fdividef, fast sincos (Cody-Waite + hardware sin/cos), 32-bit offsets, lane-0
shuffles for uniform values, LATETID (tid re-read with volatile asm after the index copies so ptxas cannot hoist q
math above them: "vs1 -3%, vf1 -1.6%, s32/vs32 +1.5%"), EARLYBAR, FPART ("1.18 us between prefix sums and segment
list" saved), PTPF.

## 3. Math

q1 (per-request and flat): query RMS-normalized in fp32 (8 threads per row, 3 xor shuffles, rsqrtf(ss/64+1e-5)), RoPE
on sliding, then f16 pre-scaled to log2 units by sl2 = sm_scale*qk_scale_factor*log2e*k_scale (no e4m3 rounding);
staged in the free last stage / S.kvq, words in order 0,2,1,3. QK^T: mma.sync.m16n8k16.row.col.f32.f16.f16.f32 with an
"unpadded quads" layout: each thread one LDS.128 per key (16 contiguous e4m3 dims, chunk hk*4+tig), cvt.rn.f16x2.e4m3x2;
the 16 MMA rows hold the 8 heads twice (rows g: dims 16tig+8h+{0..3} vs B = kw[2h]; rows g+8: +{4..7} vs kw[2h+1]); two
MMAs per h, useful halves summed: 4 MMAs per 8 keys, no zero A registers, no ldmatrix. Softmax: thread max, quad_max
(2 shuffles), alpha = ex2(m_old - m_new), P = ex2.approx.ftz.f32(s - m), l from fp32 P; ONE builds skip alpha. PV:
mma_f16(o, a0 = pack_h2(P), 0, a2 = pack_h2(P), 0, b0, b1) (rows g+8 zero); V fragments by 4 x LDS.64 per 16 keys
(keys kb+2tig,+1,+8,+9, dims 8g..8g+7), prmt 0x5140/0x7362 to pair the two keys' bytes, cvt f16x2.e4m3x2; V exact in
f16, never written back to smem. Masking only pos >= end on non-interior tiles.
q8 (sm_90): query quantized to E4M3 from the unit-RMS post-norm post-RoPE value, NOT pre-scaled ("pre-scaling by the
kv scale would push it into e4m3's subnormals"); scores scaled in the exponent FMA ex2(fmaf(s, sl2, -offset)).
Per-request: fp32 -> cvt.rn.satfinite.e4m3x2.f32 straight into the A tile S.q8s (per head 64 rows x 64 B, 64B swizzle,
4 threads per row). QK^T: two SS wgmmas per tile per warpgroup: `wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3
{32 f32}, qdesc, kdesc, p,1,1` (A = q8s desc_sw64 SBO 512; B = the K stage as filled by cp.async at head offset hk*64,
desc_sw128 SBO 1024; second k-step = desc+2). V^T transpose per tile per warpgroup: 2 x ldmatrix.x4.trans (e4m3 byte
pairs as b16) + byte prmt (0x6420/0x7531), stores by 2 x stmatrix.x4 (per-request; "vs16 -2.5%") or 8 x st.shared.b32
(flat); layout per head per buffer 72 rows x 64 keys K-major 64B swizzle = 4608 B, double-buffered; KEYS PERMUTED INTO
P's REGISTER ORDER (k = 4q+2h+b for key 8h+2q+b) so P needs no shuffle; rows 64..71 filled once with e4m3 ones (0x38);
fence.proxy.async.shared::cta placed late just before PV. P and PV: pa = e4m3x4f(ex2(fmaf(sv,ssc,-dlo))...);
`wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 {32 o + 4 lacc}, {pa[0..3]}, vdesc+2kk, 1,1,1`: 64 output dims
+ 8 copies of the row sum in lacc (over the same e4m3-rounded P that multiplies V: consistent numerator/denominator);
l = 0.25*lacc[0] quad-summed later. PVSPLIT=1: each 32-key half issued once packed, waited at tile end (nothing in
flight across the loop edge: avoids C7515); flat PK2: both halves packed in one straight-line block of 32 independent
chains. OZF/reg_fence asm barriers keep non-wgmma accumulator definitions out of the wgmma pipeline. Lazy softmax
offset (LAZYMAX): first tile sets per-row offsets cr = quad-reduced row max; afterwards P may reach 2^8 = 256 < 448;
per-request: per-thread partial max then __any_sync(m > cr+8); flat (PCHK): test the packed P bytes
(((pa+0x07070707)|...) & 0x80808080: byte > 0x78 sets bit 7); rare offset move rescales o and lacc and repacks (at
most one retry). Masks only on edge tiles.
New tokens (L2456-2512): one pass over L*4 rows (k head 0/1, v head 0/1 per token); k rows RMSNorm + RoPE, v raw; both
x __fdividef(1, scale) -> e4m3 SATFINITE into S.kn/S.vn; the attention reads the in-smem e4m3 copy.

## 4. Merge

Absolute-frame accumulation (L3642-3755): q1 warps with wsub=1 hand (m,l,o) through smem to the wsub=0 warps (standard
max-rescale), then per row: f = (m == -inf) ? 0 : ex2(fminf(m,126)); red.global.add.v4.f32 [ws_acc+...], {o*f};
tig==0: red.global.add.f32 [ws_l], quad_sum(l)*f (q8 RED4: lane-pair swap so each lane holds 4 consecutive dims = one
16-B red). Workspace: ws_acc[((bp*nc + copy)*L + j)*16 + hh][64] and ws_l, bp = b*pairs + pair, copy = split % nc;
nc copies spread same-address contention ("coop with 1/2/4 copies +26/+16/+4%: same-address red serialization of 250
splits"); buffers >= 32768 rows x 64 floats (8 MB) + ws_l + int ws_cnt, zeroed once. ws_cnt map: [bp] last-CTA key
ticket; [8192+bp]/[12288+bp] coop parity counters; [24576+bp] per-request epoch; [30000+] flat counters;
[30020/30021] flat parity; [30022] flat epoch.
Last CTA (mode 1, L3896-3945): reds; csync; tid 0: `atom.acq_rel.gpu.global.add.s32 old, [ws_cnt+bp], (end-start)`;
S.flag = (old + mine == n_keys); csync; if flag: finalize (fin_nc1 or nc loop), ws_cnt[bp] = 0. The ticket counts
KEYS (empty splits never arrive; flat segments of a request add their own counts). No per-thread __threadfence: the
single acq_rel atomic after bar.sync releases every thread's reds through cumulativity ("as CUTLASS's semaphore
release") and acquires the others' for the reader (MERGE2/MERGE3). Finalizer reads with __ldcg.
Cooperative finalize (MODE 3/7/11/15, flat q8; L3769-3895): all splits resident (host occupancy check; device trap
after 2^24 polls); PARITY: thread 0 reads the epoch ep (cp.async in the prologue), atom.acq_rel add of its key count on
counter ep&1; the last arriver zeroes the other counter and writes epoch = ep+1; others spin ld.acquire.gpu +
__nanosleep(64) until v >= n_keys; slices: each CTA finalizes rows [split*R, ...) with R = ceil(L*16/nact), divisions
pinned above the spin (PINDIV); sum_copies reads and zeroes its slice (SC1 skips the copy loop at nc==1; SC8 loads
all 8 copies in one round trip). Flat q8: grid-wide version, each CTA ceil(B*pairs*128/G) rows. Why cooperative: the
last-CTA finalize "reads and zeroes tens of KB from one SM". Zero state: finalizer stores zeros into exactly what it
read; counters reset by the last CTA or the parity scheme; nothing memset on the host.

## 5. Epilogue and cache write

Epilogue (fin_nc1 L1600-1658): per (token j, head hh) row, 8 threads x 8 dims: inv = lsum > 0 ? __fdividef(vscale,
lsum) : 0; y = acc*inv; rn = rsqrtf(oct_sum(sum y^2)/64 + 1e-5); out = bf16(y*rn*sigm(gate)), sigm(x) = __fdividef(1,
1+__expf(-x)); gate = 16-B bf16 load from qkv columns (Hq+2Hkv)*64 + head*64 issued with the workspace loads (one L2
round trip; split 0 prefetched it); 16-B output store. Cache write: only CTAs with end > seq compute the new rows; the
CTA whose [start,end) contains seq+jj writes the 8-byte e4m3 chunk to cache_row(cache, cache_loc[b*L+jj], (2*pair+hh)*64
+ part*8) (cache_loc staged by an 8-B cp.async); each token written by exactly one CTA; new rows patched into smem
stages with st.shared (+ fence.proxy.async for q8).

## 6. Switches (shipped values and author notes)

On: NP 128 ("128 since v111: f16 -0.7%, f32 -0.4% vs 256"); WS/WS8 producer warpgroups; NP8 128; WSMINB 2;
SETMAXNREG; TMAPG 1 ("q1 flat only: f16 -2.4%, f32 -1.7%; q8 flat lost ~0.7%"); MERGE2/MERGE3 (one acq_rel atomic /
barrier arrive instead of per-thread fences); IDXFIRST; ONETILE; MAGICDIV; SC1; ONE; QDIRECT; SPEC; QP16; ONENEW; SC8;
PINDIV; ONEPRED; LATETID (SPEC only); PFASTPR ("s1 +1.5%" with the prologue copy, so in-loop only); PFAST; COOP;
WGDEC; PV8; PV8FLAT (needs QSS: 104-register consumers); QSS; Q8RAW; PCHK; RARE; PREUNR ("s1 -2%"); PVW1; PVSPLIT 1;
INV2; OZF; XP_ARRALL ("vf32 -0.8%"); PATCHNI ("-1.5%"); DSHFL; PARITY; UNRN ("b32 q8 sliding -0.5 us"); ADDRN; OFF32;
RED4; LATESYNC/LATESYNCN; QKFIRST; FPART ("1.18 us"); EARLYBAR; PTPF; COOPGUARD ("a %globaltimer read per poll slowed
the coop barriers ~1%"); PK2; UARGS/UWARP; ADDR ("-1% since UWARP/UARGS"); FASTSC; STSMPR; TCHUNK; NOSYNC0; LAZYMAX.
Numbers: ST1F 3 ("3 measured best: b16 full 93 -> 91.5 us vs 6"); STN 3; ST8F 4; SEGK8 600 ("b16 107.4 -> 103, b32
187.4 -> 184"); SEGK1 400 ("b16 90.4 -> 88.7"); UNR8 2 ("-4.5%").
Off (measured losses): TMAPG8 (+0.7%); RAWTHR (slower: +10 static fp32 instr); TURN (+1.3%); PDLKV (standalone ~0.3%
worse); BRFREE neutral; UNEV no gain; W0L1 (+3..5 us); W0SPREAD (8 sequential polls); NOXCH1 slower; QKZ (QKZ=0 vf32
-0.4%); IRANGE neutral; SENSE/SENSE2 slower; INEW slower; STSM in flat (~1.5% slower); EXPPOLY (slower: issue-bound);
LAZY1 neutral; F2 (one producer warpgroup per SM limits); S2 slower; PTR (producers build V^T: much slower).
Python: OKC_FLATCOOP 1; OKC_PDL 0 ("for a fused layer: overlaps the qkv GEMM's tail"); OKC_FLAT1B 0 ("per-request
coop plan is now faster: f1 13.2 -> 12.6 us"); OKC_PLAIN8 1.
Pattern of what won: removing barriers/fences; removing branches and instructions on once-per-launch code;
straight-line ILP in the exponential block; uniform registers; FP8 everywhere on q8; one CTA per SM for latency-bound
q8 sliding; 2 CTAs/SM with a producer warpgroup for long runs. What lost: extra producer duties, two streams per CTA,
FMA-pipe exponentials, uneven per-SM shares, dynamic chunks, sense-reversing barriers.

## 7. Sliding-window shapes (weight 46)

Author notes: decode 1 sliding "b1 33 splits (one tile each) 6.6 -> 6.5 us vs 32"; decode 16 sliding "16 splits 9.4 ->
9.2 us vs 12"; verify 1 sliding "coop beats clusters 13.0 -> 10.2 us", "one copy since v101: 9.8 -> 9.6"; verify 16
sliding "19.0 -> 17.2 us vs clusters of 8", "coop helps at 8 splits per request (16.2 -> 15.85)"; verify 32 sliding
_TUNED "4 splits per request, 128 CTAs one per SM, 16.10 -> 15.50; 3 splits +13%, 5 +10.6%, 6 +3.1%"; q1 b32 "plain
split-KV spreads evenly (b32 q8 27.7 -> 23.5, q1 14.4 -> 13.1)".
Design: B=1 one 64-key tile per CTA (32/33 CTAs); B=16/32 128-256 CTAs with 2-9 tiles; whole-tile chunks via the magic
multiplier; the decode-1 build <1,false,13> folds in ntiles=1 compile-time, SPEC, IDXFIRST, a branch-free predicated
copy of the single tile, no alpha rescale, last-CTA fin_nc1 with one 16-row pass of 128 threads; verify 1 <8,false,15>
same + cooperative finalize (33 CTAs each finalize 4 of 128 rows); one CTA per SM for q8 sliding B >= 16; masks only on
edge tiles; cold-cache awareness via a "scout census" of instruction and taken-branch counts on the once-path ("~168
instr, 11 taken branches here on s1, cold").

## 8. Host side

Build on first kernel_fn call: load_inline(-O3, -gencode arch=compute_90a,code=sm_90a, -DOKC_SM90=1,
--expt-relaxed-constexpr), all 20 kernels (the judge's 16 tests warm it before timing). First launch per L:
cudaFuncSetAttribute(MaxDynamicSharedMemorySize) per instance (+ NonPortableClusterSizeAllowed when relevant),
cudaOccupancyMaxActiveBlocksPerMultiprocessor for coop/ONE/SPEC/flat instances, SM count; cached in statics. Each call:
_plan is a pure function of shapes (B, L, window, page_table.shape[1], P); torch.empty output; cached inv_freq,
workspace; one run(). Launch: plain <<<grid, NTX, smem, stream>>> unless clusters/PDL (cudaLaunchKernelEx);
griddepcontrol.wait before the first qkv read and launch_dependents after the tile loop (no-ops without PDL). Graph
safety: no host syncs, no input-value reads, single kernel node, no memsets; SPEC reads only a shape and is verified on
the device; cooperative grids depend on co-residency (host check + device trap). Fallbacks for unsupported layouts.

## 9. Implications for our kernel

1. Full context B=16/32: the flat kernel (264 equal-work CTAs with SEGK, warp-specialized cp.async producer feeding
   an mbarrier ring) vs our (B x groups) x pow2 splits.
2. Merge: absolute-frame merge (o*2^m, l*2^m by fp32 atomic add, key-count ticket, finalizer zeroing) removes
   partial-max buffers and any second merge level; nc copies quantify same-address contention.
3. Sliding B=1: SPEC is doable (page_table.shape[1] is a shape); 1-tile-per-CTA plans (32/33 CTAs) and cooperative
   finalize for q8 vs our 16-split 128-key-tile plan.
4. q1 math: plain f16 mma.sync with in-register e4m3->f16 conversion (not FP8 QK^T); speed comes from layout (both
   heads per 128-B row, one 16-B load per key) and pipeline structure.
5. q8 math: FP8 PV with the ones-row row sum and P-ordered V^T avoids the row-sum reduction and P shuffles.
Docstring vs code: 3/3/4 stages (not 4); q1 uses f16 queries and f16 MMAs; q8 uses an FP8 PV wgmma.
