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

(filled in per check below)
