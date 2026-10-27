# Design memo: published `muse_bf16_proj.py` (vjeux/okc-2026, Apache-2.0; 1.50x vs SGLang on H200)

Source: /tmp/okc_pub/muse_bf16_proj.py (internal version tag v141). Line references are to that file.
Reference design only: re-implement, do not copy.

## 0. The docstring does not describe what runs on H200

- The swap-AB cross-CTA split-K kernel (`skinny_tma_kernel`, L248-664) is OFF on H200: `old_paths = not (sm90_ and USE_SK2)` (L3652).
- The straight-to-register weight load (`ldg_stream`, L61-66) is dead code ("LDG skinny kernel unused in v6 (TMA path)", L1900).
- m=8/16/32 runs `sk2`: x is the mma A operand (16 tokens), weights are the B operand (8 rows). NO cross-CTA reduction:
  each CTA owns R rows over the whole K; K is split only across warps inside the CTA. Escapes are applied by dedicated
  escape warps while the weights stream. The router uses a 3-CTA cluster with a DSMEM reduction.
- m=1 is autotuned among `cgemv` (12-bit CUDA-core GEMV, `__ldg` to registers), `sgemv`/`rgemv`, `sk2`, cuBLAS.
- m=128/256 NEVER uses the 12-bit format: `WIDE_BF16 = False` ("the wide (RS-wgmma) kernel never beats cuBLAS for BF16
  m 33..256", L3359). Candidates are raw BF16: cuBLAS, `torchT` (cuBLAS on a K-major copy), `gq`, `gqx`, CUTLASS; plus
  `rk`/`wide`/`gemm90` for the router.
- "Self-resetting semaphore" = monotonic counter read modulo `splits`, never reset (L607-618); only in the legacy kernels.

## 1. Per-shape dispatch

Choice made at prepare time: `_candidates` (L3967-4000) per (layer, m); `_selftest` (L3815-3844) vs an FP64 reference at
m in {1,8,16,32,33,128,200,256} (tol 5e-3 BF16 layers, 2e-5 router) disables failing candidates; `_autotune` (L4007-4070)
times candidates for TUNE_M=(1,16,32,8,128,256): CUDA-graph capture, replay from cold L2 (256 MB torch.sum sweep before
each replay), 24 reps interleaved with alternating order, median wins. Picks are printed as `[okc] tune ...`.

Candidates (sm_90):
| m | qkv_gate | o | pooler | unpooler | router |
|---|---|---|---|---|---|
| 1 | cgemv, sk2 qkv8t_r144, cuBLAS | cgemv, sk2 o8h_r88, cuBLAS | cgemv (v137: -0.34 us vs sk2), sk2 pooler8h_r72, cuBLAS | slice n=1024: cgemv, sgemv, sk2 slice8_r192, cuBLAS | rgemv (fp32 weights), sk2 router, cuBLAS hi+lo |
| 8 | sk2 qkv8t_r144 / cuBLAS | sk2 o8h_r88 | sk2 pooler8h_r72 | slice: sk2 slice8_r192 | sk2 router |
| 16 | sk2 qkv16t_r200 | sk2 o16t_r68 | sk2 pooler16t_r192 | sk2 unpooler16t_r72 | sk2 router |
| 32 | sk2 qkv32_r208 | sk2 o32h_r100 | sk2 pooler32_r144 | sk2 unpooler32_r144 | sk2 router |
| 128 | gqx (cuBLAS 16.4 -> 14.0 us) / cuBLAS | cuBLAS / torchT | gq / cuBLAS | cuBLAS / torchT | rk, wide, gemm90, cuBLAS |
| 256 | cuBLAS | cuBLAS / torchT | cuBLAS / CUTLASS (31.3 vs 34.5 raw) | cuBLAS / torchT | rk, wide, gemm90, cuBLAS |

sk2 geometry (`SK2_CONFIGS` L3322-3338, `_sk2_geom` L3370-3393, `launch_sk2` L1410-1465): R = NWR*NT*8 rows per CTA,
grid = N/R*CL; threads = (NWR*WSK consumers + 1 producer + NESC escape warps)*32, NESC = min(NSTAGE,4) (1 for router);
each consumer warp owns NT*8 rows and takes UPW consecutive 64-k units per stage out of UPS = WSK*UPW units.
| shape | config | R | NT x NWR | WSK x UPW | k/stage | warps | grid | NSTAGE | smem | CTAs/SM |
|---|---|---|---|---|---|---|---|---|---|---|
| o m8 | o8h_r88 | 32 | 4x1 | 4x1 | 256 | 4+1+4 | 256 | 4 (all K) | 73 KB | 2 |
| o m16 | o16t_r68 | 32 | 2x2 | 1x4 | 256 | 2+1+3 | 256 | 3 | 67 KB | 2 |
| o m32 | o32h_r100 | 32 | 2x2 | 2x2 | 256 | 4+1+3 | 256 | 3 | 97 KB | 2 |
| pooler m8 | pooler8h_r72 | 16 | 1x2 | 2x4 | 512 | 4+1+3 | 256 | 3 | 64 KB | 2 |
| pooler m16 | pooler16t_r192 | 32 | 2x2 | 2x4 | 512 | 4+1+4 | 128 | 4 | 169 KB | 1 |
| pooler m32 | pooler32_r144 | 32 | 2x2 | 2x2 | 256 | 4+1+4 | 128 | 5 | 157 KB | 1 |
| qkv m8 | qkv8t_r144 | 24 | 3x1 | 4x2 | 512 | 4+1+4 | 96 | 5 | 137 KB | 1 |
| qkv m16 | qkv16t_r200 | 24 | 3x1 | 4x2 | 512 | 4+1+4 | 96 | 5 | 177 KB | 1 |
| qkv m32 | qkv32_r208 | 24 | 3x1 | 4x2 | 512 | 4+1+4 | 96 | 4 | 213 KB | 1 |
| unpooler m<=8 (n=1024) | slice8_r192 | 8 | 1x1 | 8x2 | 1024 | 8+1+4 | 128 | 4 (all K) | 115 KB | 1 |
| unpooler m16 | unpooler16t_r72 | 32 | 2x2 | 2x2 | 256 | 4+1+3 | 256 | 3 | 67 KB | 2 |
| unpooler m32 | unpooler32_r144 | 64 | 4x2 | 2x2 | 256 | 4+1+3 | 128 | 3 | 145 KB | 1 |
| router m8-32 | router (CL=3) | 8 | 1x1 | 8x1 | 512 | 8+1+1 | 96 (32 clusters x 3) | 4-6 | 153-210 KB | 1 |
(`_rNNN` suffix caps the ring at NNN KB so two CTAs fit per SM for the 256-CTA configs.)

m=1 kernels (L2579-2774; configs L3460): cgemv 2 rows/CTA x (256|64|128|512 threads + 32-thread escape warp), grids
1152/4096/512/2048; sgemv (slice, bf16) 4 rows x 512 thr, 256 CTAs; rgemv (router fp32) 1 row x 256 thr, 8 float4/thread.
m>=128: gq (pooler m128: BM=64, BN=128, CL=2, 8 stages, 128 CTAs x 160 thr), gqx (qkv m128: CL=3, 108 CTAs),
rk (router: S=12/8 splits), wide, gemm90, CUTLASS Pingpong 128x64x64 persistent swap-AB (pooler m256).

## 2. The 12-bit format

BF16 = s | e[7:0] | m[6:0]. High byte = s | e[7:1] ("e71"); low byte = e[0] | m[6:0] stored raw.
code = s<<3 | o, o = clamp(e71 - base, 0, 7) (L3206, L3435, L3473): an 8-value window of e71 = 16 consecutive
exponents, even-aligned. Window base per matrix = argmax over the 8-wide histogram window of e71 (densest window,
L3203-3205), base4 = base*0x01010101; matrix rejected (stays BF16) if escapes > n*k/200. (Ours anchors at emax-15.)
Code -> high byte: hi = (code>>3)<<7 | (base + (code&7)).
Decode (`decode8`, L93-103), 8 weights -> 4 bf16x2 regs in fragment order:
```
ha = lop3_sel(nw & 0x07070707, nw << 4, 0x80808080) + base4;   // low nibbles -> hi bytes w0..3
hb = lop3_sel((nw >> 4) & 0x07070707, nw, 0x80808080) + base4; // high nibbles -> hi bytes w4..7
r0 = prmt(lw0, ha, 0x5140); r1 = prmt(lw0, ha, 0x7362); r2 = prmt(lw1, hb, 0x5140); r3 = prmt(lw1, hb, 0x7362);
```
(~12 int ops per 8 weights; no LUT.) m=1 variant `cg_hib` (L2584): ((c&0x07070707)+base4) | ((c<<4)&0x80808080),
byte_perm, shift/mask to fp32 for FFMA.

Escapes: weight with e71 outside [base, base+7]; decodes with exponent clamped to base or base+7. sk2 table
(`sk2_table`, L3556-3577) per CTA int2[ncta][pmax]: entry0 = (count, row-map offset); entries = ((row_local<<16)|k,
original bf16 bits) sorted by k; pmax = max count+1 rounded even (bulk-copyable). The kernel recomputes the delta
(L139-145): dhi = (hi&0x80) | (e71 < bq ? bq : bq+7); delta = f32(t<<16) - f32(((dhi<<8)|(t&0xFF))<<16).
Who applies (sk2, L1130-1181): NESC escape warps take turns on stages (warp ew handles stages st % NESC == ew);
lane = token; x read from the stage's swizzled x tile in the ring; per-warp private corr[TOKS][R+1] (odd stride, no
atomics); every escape warp waits on every stage and ARRIVES on its empty barrier (empty barrier count = NW + NESC)
to avoid a ring deadlock; table arrives by one bulk copy on pbar. Cost per escape: 1 LDS x + ~10 ops + LDS/STS RMW
over M lanes, all overlapped with the stream; consumers wait on cbar only at the epilogue; epilogue adds NESC*4
scalar LDS per output float4. qkv escape balancing (L3516-3553): rows sorted by escape count in groups of 4 and
snake-dealt across R-row blocks (row map appended to the table; epilogue stores through it); v136: qkv m8/16 -0.5 us.

## 3. Load path (sk2)

Weights: one 1-D bulk copy (cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes, L176-179) of WBY =
12-24 KB per stage per CTA. x: a 3-D TMA (cp.async.bulk.tensor.3d, L729-732) per stage; map dims {64, M, K/64},
strides {K*2, 128}, box {64, Mb, UPS}, 128B swizzle (L747-758). Both complete the same full mbarrier
(expect_tx = XBYr + WBY). Stage-major global stream [stage][cta][warp][uu][nt][768 B] (L3500-3513): "at any moment all
CTAs read one contiguous region". Ramp: thread 0 inits only full0, fences, prefetches the tensor map, issues stage-0
weights + x box + the escape table, THEN inits the rest (L1070-1106; a cluster-scope release after a copy in flight
costs ~0.7 us, L1073). Producer = warp NW lane 0 refills slot st % NSTAGE after the empty barrier (L1113-1129).
NSTAGE 3-6 -> 36-96 KB of weights in flight per CTA. Consumer per unit (L1206-1249): all smem reads first (NT x
(ld.shared.v4 16 B low bytes + ld.shared.v2 8 B codes), then 4*MT ldmatrix.x4), release the slot (UPW==1) and ONLY
THEN decode and MMA. Consumers busy-poll try_wait; producer/escape warps use try_wait with a suspend hint 0x989680
(L147-157) so they don't steal issue slots. No cache hints on sk2 copies. x is re-read by every CTA from L2
(pooler m32: 512 KB x 128 CTAs = 64 MB L2->SM vs 48 MB weights; accepted).
m=1 GEMVs: plain __ldg to registers, all loads issued before the first FMA (L2626-2640).

## 4. Fragment layout, operand mapping, x staging (sk2)

mma.m16n8k16.row.col: A = x (16 tokens x 16 k), B = weights (8 rows x 16 k); tokens per mma = 16 (MT=2 at TOKR=32);
a warp has NT n8 tiles. Orientation chosen because the C fragment gives (token, 2 consecutive n): float2 smem stores
and 8-B bf16x4 global stores along n, and 8-row granularity for R. B tile-unit (`sk2_tileunits`, L3414-3419):
[N/8][K/64][32 lanes][16]; lane (g,t) holds row g, value v = 4j+2h+p is k = 16j+8h+2t+p. Compressed tile-unit =
768 B: bytes 0-511 low bytes lane-major 16 B per lane (lw.x = k-step 0 (b0,b1 pairs), .y = step 1, ...); bytes
512-767 codes 8 B per lane (word0 = v0..7, word1 = v8..15, byte b = code[b] | code[4+b]<<4). Each lane gets 24 B per
8x64 tile; two decode8 calls give b[nt][0..7]. x in smem: TMA box [UPS][Mb rows][128 B] 128B-swizzled; A fragments via
ldmatrix.x4 with swizzle row (ui*Mb + row) & 7 (L1238-1243). PK mode (M < TOKR or TOKR==8): the TMA box has exactly M
rows; rows >= M read a 128-B zero row (L1051, L1107). TOKR = 8/16/32 for m <= 8/16/32; at m=1/8 half of each mma is
wasted (memory-bound, fine).

## 5. Split-K

sk2: only across warps. Epilogue (L1290-1401): wait cbar (escape warps) + named barrier; each consumer thread sums its
NACC accumulator sets and writes float2 pairs into red[wk][tok][RP] overlaying the idle ring (RP = 8 mod 32:
conflict-free); second named barrier; threads loop over (token, 4-row group) reading WSK float4 partials + NESC x 4
scalar corrections (+ (CL-1)*WSK peer float4s for the router) and write one 8-B bf16x4 (float4 for the router).
Cost: 2 named barriers + a few LDS/STS per thread; no global partials, fences, atomics, or second kernel.
Router across CTAs (CL=3 cluster, L1296-1364): ranks 1,2 push per-warp sums into rank 0's `peer` buffer with
st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32; rank 0 expect_tx((CL-1)*NC*MT*NT*16) on qbar at init;
barrier.cluster.arrive.relaxed right after init, wait.acquire only before the remote stores / final reduce.
Legacy global split-K (skinny, wide, gemm90): monotonic per-row-block counter, `dist` mode when the whole grid is
co-resident (every CTA waits then reduces its 1/splits slice) else last-arriving CTA reduces; fences:
__stcg partials -> __threadfence -> named barrier -> atomicAdd -> spin -> __threadfence (skinny);
atom.add.release.gpu + ld.acquire.gpu spin (wide); prefetch.global.L2 on the counter at kernel start; reduce loads
__ldcg float4 8 in flight. rk (router m>=33): each CTA zeroes its output slice in-kernel, generation counter,
red.relaxed.gpu.global.add.v4.f32 of an smem-transposed partial tile (grid <= nsm checked).

Why cheap where ours costs 10+ us: (1) no cross-CTA reduction in the 12-bit m<=32 path (R chosen so N/R = 96-256 CTAs
each streaming full K); (2) escapes off the critical path on dedicated warps with private accumulators and an uncapped
table; (3) the only reduction is intra-CTA through the dead ring; (4) the one cross-CTA split (router N=256) uses DSMEM
with mbarrier transaction counting; (5) where global split-K remains, counters are monotonic and co-resident grids
reduce in parallel.
Compared with our t12c/t12e: weights are our A operand (R multiples of 16); x via cp.async by compute warps; one
__syncthreads per chunk; escapes on the single producer warp; table capped at 64 per tile; w12/g12 use global
partials + ticket + separate reduce.

## 6. Router

hi = w.bf16(), lo = (w - hi.float()).bf16(). m=1: rgemv on the fp32 weights (8 MB, exact fp32 FMA, fixed-order
reductions). m<=32 sk2 `router`: tile-unit = bf16 hi tile (1024 B) + lo tile, uncompressed; two mmas per (k-step,
tile) into alternating accumulator sets (NACC=4) to break dependency chains; CL=3 cluster, 32 row blocks of 8 rows,
WSK=8, 96 CTAs; fp32 float4 output. m=128/256: autotuned among rk (router_cat [hi|lo] [256,16384], 6 stages, SS-wgmma
m64n64k16 on hi then lo), wide (TMA hi/lo tiles with evict_first, SS-wgmma, global split-K), gemm90, cuBLAS x2.

## 7. Tunables and recorded measurements

SK2_CONFIGS/SK2_PICK per-(layer,TOKR) geometry with ring caps ("okc_tune retimed winners, Oct 25": pooler16 -0.43,
unpooler16 -0.60, o16 -0.15, qkv8 -0.12, qkv16 -0.08 us); NSTAGE = max(2, min(16, nst, ring//sst)); NACC forced to 1
when MT=2, compressed, NSTAGE>=4, NT=4 (register pressure); SK2_BALANCE for qkv only (o/unpooler m32 lost);
CG_CONFIGS 2 rows/CTA (v137: pooler m1 2x512 -0.34 us vs sk2; 4/8 rows or 1 row lose); GQ pooler only; GQX qkv
BN=128 CL=3 (16.4 -> 14.0 us at m128); TORCHT_LAYERS = (unpooler, o) (unpooler m=240 +11%); CUTLASS pooler m256
Pingpong 128x64x64 (31.3 vs 34.5 raw; 54 configs x 8 shapes sweep, 170 s build timeout in a background thread);
RK_SPLITS {1:16,2:12,3:10,4:8}; WIDE_BF16=False; autotune 24 reps, 256 MB flush, logged times include ~4 us replay
cost; mbar wait suspend hint 0x989680; __nanosleep(20/32/128) in spins; SK2_DIAG globaltimer tracing compiled out.
Notes: the wide kernel has no producer warp because 17 warps would cap registers at 96/thread (L1488); ~0.7 us
cluster-release stall (L1073).

## 8. Host / prepare

One load_inline TU with no_implicit_headers (torch/types.h alone costs ~45 s of nvcc; main build ~25 s); CUTLASS built
in a background thread (~70-80 s) with a 170 s join inside prepare's 300 s. Repack is torch GPU ops (bincount, argmax,
nonzero, argsort, gather); sk2_stream builds one 12-bit copy per distinct geometry key (about 3 for pooler/o).
Workspaces: ws_shared 4 MB, mid_ws_shared 16 MB, per-config counters, rk_zc 8 int32, cut_ws 1 MB. Graph capture:
outputs via torch.empty only; tensor maps encoded on the host per launch from x.data_ptr() and passed as
__grid_constant__ (baked into the graph node; fine since replays reuse the captured input buffer);
cudaFuncSetAttribute once per template before any capture; monotonic counters need no reset node; rk/dist spins
require co-residency (checked on the host); the autotuner times exactly the judge's mode.

## 9. Takeaways for our kernel

1. Remove cross-CTA split-K from the 12-bit m<=32 path: weights-as-B orientation (8-row tiles), R so N/R ~ 96-128 at
   1 CTA/SM or 256 at 2 CTAs/SM with the ring capped near 70-100 KB (qkv R=24; pooler R=16 at m8, 32 at m16/32;
   o R=32; unpooler R=32 at m16, 64 at m32; slice R=8).
2. Move escapes to 3-4 dedicated warps taking turns on stages, reading x from the smem stage, private per-warp
   [tok][R+1] accumulators, k-sorted uncapped per-CTA table by one bulk copy; count them in the empty barrier; qkv row
   permutation for escape balance.
3. Weights by one bulk copy per stage + x by 3-D TMA on the same mbarrier; stage-major global layout; packed M rows
   for M < TOKR; issue stage 0 before initializing the other barriers.
4. In the consumer: issue every smem load of a unit, release the slot, then decode and MMA; consumers spin,
   producer/escape warps wait with a suspend hint.
5. For m>=128 stop pushing 12-bit; add raw-BF16 candidates: gqx (qkv m128, CL=3), gq (pooler m128, CL=2), CUTLASS
   Pingpong (pooler m256), torchT (o, unpooler), rk (router).
6. Choose between candidates at prepare time with a cold-L2 graph-replay autotuner + an FP64 self-test gate.
