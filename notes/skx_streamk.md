# skx: persistent stream-K wgmma GEMM for the m = 128 / 256 dense projections (muse_bf16_proj)

Branch worktree-agent-ae728c9dfa8f50305. Goal: beat cuBLAS / gqx / gq / torchT on the m=128 (and, as a stretch, m=256)
rows with one persistent kernel whose per-SM ingest is W once + x once per 128-row block (the gqx/gq forms re-read x
twice per 64-row block; uniform split-K cannot fill 132 SMs for these N).

## Design (SKX_SRC in muse_bf16_proj/submission.py)

- Grid = SM count (132 on H200), 1 CTA per SM, 288 threads: 2 consumer warpgroups (rows [0,64) / [64,128) of the W
  tile) + 1 producer warp. Ring of NSTAGE stages of W [128 x 64 k] (16 KB) + x [BN x 64 k] (16 / 32 KB) TMA tiles
  (128B swizzle, SS-wgmma m64nBNk16, W evict_first / x evict_last). Delayed stage release (wait_group 1, release the
  previous unit's slot). 6 stages at BN=128 (197.7 KB smem), 4 at BN=256.
- Work units = (row block, 64-k chunk) in row-block-major order, U = RB*C, CTA i owns [i*U/G, (i+1)*U/G): a range
  spans at most two row blocks (host assert: units per CTA <= C). cta_of(u) = ((u+1)G-1)/U; row block rb is covered by
  P = cta_of(rb*C+C-1) - cta_of(rb*C) + 1 CTAs.
  qkv: 17-18 units/CTA, P=8, 12 straddlers; pooler 31-32, P=5, 28; unpooler 31-32, P=3, 60; o 7-8, P=2-3, 52.
- Fixup, fp32 partials in wgmma fragment order ([BN/8 token blocks][256 threads] float4 = 64 KB per 128x128 tile):
  - FIX 0: red.add.v4.f32 into the row block's slab + ticket; the last arriver converts to BF16, stores the tile,
    re-zeroes the slab and resets the ticket (slabs/tickets zero between launches; torch.zeros once in prepare).
  - FIX 1: per-(CTA, segment) slabs [(seg ? G : 0) + cta], monotonic counters (target = (old/P+1)*P), every CTA of the
    row block waits for the P arrivals (co-resident, 2 s trap) and reduces the slice proportional to its units.
  - A straddler exports segment 1 without fencing, runs segment 2, then fences once and takes both tickets.
- TILED: W from a pre-swizzled contiguous copy ([rb][chunk][128 x 128 B], `swizzle_tiles(w, 128)`, one bulk copy per
  unit with the evict_first hint) instead of the 2-D TMA box from the row-major matrix.
- Route: `("skx", i)` candidates for 33 <= m <= 256 (bf16 layers) through the existing FP64 self-test + cold-L2 graph
  autotune; `_nearby_choice` keeps 129..255 on cuBLAS as before.

## Offline gates (before any check)

- tools/skx_sched_model.py: schedule + numpy emulation of both fixup modes in fragment order (every unit covered once,
  <= 2 segments per CTA, P consistent, slices partition the tile, output mapping exact for M < BN with garbage x
  rows): all shapes ok.
- tools/nvcc_pub.py --only skx: 5 instantiations, no spills, no ptxas advisories; 92-96 regs at BN=128 (fix 0 / 1),
  157 at BN=256 (fix 1).
- tools/check_skx_dispatch.py (judge image, no GPU): candidates / keys / names / nearby choice over every judged and
  tested m.

## Model (per-SM ingest ~70 GB/s from the judge rows; DRAM 4.8 TB/s; wgmma ~55% at 2 WGs)

| shape m=128 | now (judge) | L2->SM bytes per SM | ingest est. | DRAM floor | compute at 55% |
|---|---|---|---|---|---|
| qkv  | gqx 14.9 | 558 KB | 8.0 us | 8.3 | 8.8 |
| pooler | gq 22.3 | 1.0 MB | 14 | 14.4 | 15.7 |
| unpooler | torchT 18.7-22 | 1.0 MB | 14 | 14.4 | 15.7 |
| o | cuBLAS 7.2 | 250 KB | 3.5 | 3.5 | 3.9 |
plus the fixup tail (~1-2 us: 64 KB exported per segment, ~64-74 KB read per CTA).

## Judge results

### Check 1: e79f1d997676444f (commit 25c08d1), 1.4756x (CI 1.453-1.498), 9/9, nothing disabled

Candidates (nstage, fix, tiled): [0]=(6,1,0) [1]=(6,0,0) [2]=(6,1,1) [3]=(6,0,1) at BN=128, (4,1,0) at BN=256.
In-prepare bench (cold-L2 graph replay, us incl. ~3.5 us replay cost; the judge table carries the unchanged picks):

| m=128 | skx[0] fix1 | skx[1] fix0 | skx[2] fix1 tiled | skx[3] fix0 tiled | current pick | judged pick |
|---|---|---|---|---|---|---|
| qkv | 28.32 | 31.07 | 28.06 | 30.72 | gqx 17.98 | 14.59 |
| o | 24.96 | 25.82 | 24.61 | 25.44 | torchT 10.46 | 7.59 |
| pooler | 36.19 | 37.57 | 35.84 | 37.66 | gq 24.96 | 21.44 |
| unpooler | 37.82 | 37.76 | 37.12 | 37.38 | torchT 22.53 | 18.86 |
| m=256 | skx (4,1,0) | | | | | |
| qkv | 38.08 | | | | torch 23.49 | 19.47 |
| o | 36.74 | | | | torch 12.54 | 9.37 |
| pooler | 49.73 | | | | cutlass 30.59 | 27.29 |
| unpooler | 54.08 | | | | torchT 27.3 | 23.84 |

Reading: fix 0 == fix 1 and tiled == untiled (+-1 us), so the shared machinery is at fault. Fit over the four layers
(units/CTA 8 / 17.5 / 31 / 31, segments/CTA 1.39 / 1.09 / 1.21 / 1.45): ~0.52 us per unit (= 63 GB/s per SM: the
main loop is ingest-bound as modeled, qkv ~9 us) + ~18 us for the first segment's export/fixup + ~7 us per extra
segment. The same +6-11 us "partial stores & ticket" cost was seen in the w12/g12 rounds; all three used a per-thread
__threadfence() (fence.sc.gpu) before the ticket. Round 2: CUTLASS-style release (CTA barrier, one thread
fence.acq_rel.gpu + atomics) and DIAG levels 1/2/3 (main loop only / + export / + fence + tickets) to attribute.

### Check 2: bcdae43fe30442e7 (commit 38fd64d), 1.4620x (CI 1.440-1.484), 9/9, nothing disabled

CUTLASS-style release/acquire (CTA barrier, one thread fence.acq_rel.gpu + atomics) and DIAG levels (timed, never
self-tested / picked / routed; accumulators kept live). Bench us at m=128 (judge table carries the unchanged picks:
qkv 14.70, o 7.33, pooler 21.74, unpooler 19.22):

| layer | D1 main loop only | D2 + export | D3 + fence + tickets | full fix1 | full fix0 | current pick |
|---|---|---|---|---|---|---|
| qkv | 16.70 | 18.08 | 19.33 | 27.84 | 30.62 | gqx 18.08 |
| o | 10.40 | 13.06 | 14.30 | 24.64 | 25.28 | torchT 10.43 |
| pooler | 23.62 | 25.25 | 26.53 | 35.78 | 37.22 | gq 25.22 |
| unpooler | 22.82 | 24.93 | 26.37 | 37.34 | 37.41 | torchT 22.72 |

Reading:
- The fence pattern was not it: the tail (fix 1: wait + distributed reduce; fix 0: red.add export + last-arriver
  epilogue) still costs ~9-11 us on every layer, nearly independent of N, K, P and units; the export 1.4-2.7 us; the
  fence + tickets ~1.3 us.
- Decisive: the main loop ALONE (D1) beats the picks by only 1.4-1.6 us (qkv, pooler) and ties (o, unpooler). After
  ~2 us of fixed cost it runs 0.58 us per 32 KB unit = W streamed at ~3.3-3.5 TB/s = ~48% tensor utilization (2.1
  MFLOP per unit). The 2-warpgroup BM=128 form halves the x traffic as designed but the per-unit time is set by
  compute / DRAM rather than the 70 GB/s per-SM ingest the gqx / gq rows suggested (those are x-traffic-bound).
- D2 (main loop + plain export, no reduce at all) already only ties the picks, so no fixup can make this form win at
  m=128 unless the main loop itself gets ~30% faster. Round 3 attributes the main loop (no wgmma / W only / W only +
  wgmma / tiled W) and the fix-1 tail (wait only / reduce without stores).

### Check 3: 0aad51440ee940f2 (commit 292bcf1), 1.4781x (CI 1.456-1.500), 9/9 (attribution only; picks unchanged:
qkv 14.89, o 7.41, pooler 21.92, unpooler 19.12 judged)

Bench us at m=128, every skx entry a diagnostic (timed, never picked):

| layer | loop (D1) | loop, tiled W | W+x stream, no wgmma | W stream only | W only + wgmma | full minus reduce | reduce, no stores | pick |
|---|---|---|---|---|---|---|---|---|
| qkv | 16.90 | 16.51 | 15.81 | 14.69 | 15.62 | 21.12 | 27.87 | gqx 18.14 |
| o | 10.50 | 10.43 | 9.86 | 9.66 | 10.50 | 15.97 | 24.03 | torchT 10.72 |
| pooler | 23.90 | 23.36 | 22.78 | 21.31 | 22.27 | 28.10 | 35.65 | gq 25.38 |
| unpooler | 23.30 | 22.82 | 22.08 | 21.57 | 22.56 | 27.87 | 36.74 | torchT 23.01 |

Attribution (kernel = bench - ~3.5 us replay):
- W streaming alone (TMA 2-D boxes, 6 x 16 KB in flight per SM, 132 persistent CTAs): qkv 11.2 us for 37.7 MB,
  pooler 17.8 us for 67 MB, unpooler 18.1 us: ~4.1-4.2 TB/s steady state plus ~2 us of ramp / drain per launch. The
  contiguous pre-swizzled copy (tiled) gains only ~0.4 us: the box pattern is not the limiter.
- x ingest through L2 adds ~0.2-1.5 us (multicast could save at most ~0.7 us); wgmma adds ~1 us (mostly overlapped;
  the main loop is DRAM/ramp-bound, not tensor-bound).
- Tail (fix 1): wait ~1.6 us; the reduce's LOADS 7-9 us with the stores free (full fix1 == reduce-without-stores):
  a latency chain (P peers two at a time, 2-3 slices per thread = ~12 dependent L2 round trips), fixable by issuing
  all P loads per slice at once (~1-2 us). Export 1.4-2.7 us, fence + tickets ~1.3 us (check 2).

Conclusion: even with a perfect reduce, export + tickets + wait cost ~4-5 us on top of a main loop that beats the
current picks by only 1.2-1.6 us (and the loop cannot get much faster: W streaming is at the practical DRAM rate and
the ramp is inherent to a short launch). No stream-K / global-fixup form wins at m=128 on this GPU; the gqx / gq
cluster-DSMEM forms win precisely because they have no global fixup, and BM=128 cluster split-K cannot fill one wave
(GPC packing). At m=256 the compute per unit doubles (~1.0 us per unit at the observed ~48% tensor utilization), so
the main loop alone ties cuBLAS. Route left dormant (SKX_CONFIGS = {}): kernel, schedule model and gates kept.

What would have to change for a win at m=128: a reduction that costs < 1 us end to end (DSMEM within one wave, i.e.
a cluster geometry the H200 can hold for 18-64 row blocks of 128 rows) or a main loop that beats the picks by >= 5 us
(impossible at ~4.2 TB/s: qkv's W alone needs 9 us + ramp).
