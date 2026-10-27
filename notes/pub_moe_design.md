# Design memo: published `muse_moe_w4a16.py` (vjeux/okc-2026, Apache-2.0; 2.32x vs SGLang Marlin on H200)

Source: /tmp/okc_pub/muse_moe_w4a16.py. Line references are to that file. Reference design only: re-implement.

## 0. Headline facts

- ONE kernel launch per call (per chunk of <= 384 tokens, L4175-4178): routing prologue (device function `route()`
  L478, run redundantly by every CTA with identical result), GEMM1+SwiGLU, GEMM2+norm stats, software grid barrier,
  final norm/sum. The docstring's route_kernel/final_kernel are out of date.
- `OKC_NS` = 3 on sm90 (host passes -DOKC_NS=3, L3926/L3950). FPF/FNST are dead knobs.
- Four compiled variants chosen by M (L3909-3918, L4167-4174):
  | M | build | engine |
  |---|---|---|
  | <=4 (bs1) | TINY | mma.sync "skinny" units: 8 compute warps + 1 helper warp (WS) |
  | 5-8 (bs8) | TINY2 | same, WSBLK=1, L2 hints on |
  | 9-16 (bs16) | MID | same + SPLIT=1 (experts with > 8 tokens become several jobs of <= 8) |
  | >=17 (bs32/128/256) | MAIN = NW=16, FAST1=0, EPI4=0, WGS=1 | RS-wgmma (A from registers): 4 consumer warpgroups + 2 TMA producer warps, stream-K |
- Fallback build (mma.sync skinny + fat) only if a scale byte lies outside 0x08..0x7E (PRMT scale decode inexact);
  the judge's weights clamp scales to [8, 448] so the PRMT path / WGS is what is scored.
- Weights are ALWAYS the MMA A operand (M side), tokens are N (n8 tiles). Same orientation as ours.
- All weights stream through TMA bulk copies of 36,864 B (one 16-row x 4096-K unit incl. scales), 1 CTA per SM,
  grid = SM count (132).

## 1. Kernel structure

Grid = multi_processor_count (L4122), `__launch_bounds__(NTL, 1)`; no cooperative launch (co-residency guaranteed by
smem size); spin-waits on doneA/doneAs and the grid barrier. NTL = NT + (WGS ? 64 : WS ? 32 : 0) (L279):
tiny builds 256 compute + 32 helper = 288 (warp 7 also acts as FAST1 issuer; helper warp runs `ws_helper` L3036:
finishes every unit's epilogue and issues the ring prologue); WGS 512 consumers (4 warpgroups) + warp 16 lane 0 =
weights TMA producer (`wgs_wproducer` L2518) + warp 17 lane 0 = x TMA producer (`wgs_xproducer` L2546, also computes
stage ranges). Named barriers: 1 = CTA_SYNC over NT compute threads; 2 = WS_JOIN over NTL; 7 = consumer -> x-producer
handoff after routing; 3..6 one per warpgroup. Dynamic smem: WGS ~204 KB (3 x 36 KB weight ring + 3 x 32 KB x ring)
+ ~21 KB Smem; others ~174 KB + Smem.

Timeline of `moe_fused` (L3026-3437): griddepcontrol.wait (no-op without PDL) -> role split -> tiny: FAST1 (warp 7
reads topk_ids, builds touched-expert mask; lane 0 inits mbarriers and issues the CTA's first 2 phase-A unit TMAs
BEFORE routing, L3045-3143) -> XPF2: prefetch.global.L2 over every x row grid-strided -> (LUT if !PRMTSC) -> tid 0:
mbarrier inits, seq, generation gen = ctr[3] selecting the ss copy -> route() -> zero the other ss copy and the next
launch's barrier counter (this CTA's share) -> work phases (WS: tid 0 computes phase A/B unit ranges into S.wsr then
run_phase<false>/<true>; WGS: job padded-row scan, x gather, WS_JOIN, wgs_phase<false>/<true>) ->
griddepcontrol.launch_dependents -> (FINPF: first final item's pos/topk_w loads issued before the barrier) ->
grid barrier: `bc = ctr + 4 + (gen & 1); __threadfence(); old = atomicAdd(bc,1); if old == grid-1: atomicAdd(ctr+3,1);
while (ld_acquire(bc) < grid) __nanosleep(64);` -> zero per-launch flags (ctr[0..1], doneA, xready, doneAs) -> final
norm/sum pass.

Work decomposition: jobs (`route` L537-561): expert with tot > 8 tokens -> ceil(tot/64) "fat" jobs of <= 64 tokens;
1..8 tokens -> one "skinny" job; with SPLIT each expert -> ceil(tot/8) jobs of <= 8 (a split expert re-streams its
weights). Fat jobs first then skinny, expert order; job = (expert, first pair row, token count); pair rows contiguous
per expert and index inter/ybuf/ss.
Tiny builds: unit = (job, 16-row block) x full K = 36,864 B; RB_A=512 units/job phase A (8192 w13 rows/16), RB_B=256
phase B; each CTA gets a STATIC contiguous equal slice of the concatenated unit list per phase (L3203-3209); `muldiv_g`
avoids integer division ("divide subroutines cost ~1 us from a cold instruction cache"); no queue (TDYN=0). Each weight
byte read once except SPLIT experts / fat jobs > 64 tokens.
WGS: unit = (job, 256-row block) streamed in 16 stages of 256 K; 32 units/job phase A, 16 phase B; global stage index
gs = (j<<9)|(u*16+st) (A) / (j<<8)|... (B); each CTA gets a contiguous cost-weighted stage range (stream-K; per-stage
cost tables WGSA/WGSB by n-tile count, L2333-2350; prefix sums `wgs_prefix` L2911; `wgs_range` L2929); units cut
between CTAs go through a global fixup; WGSREORD: a CTA runs its trailing partial unit first in phase A.

Chaining GEMM1 -> GEMM2 (intermediate via global/L2, not smem). Tiny: phase-A epilogue writes SwiGLU output to
inter [EMAX, 4096] bf16 with st.global.L2::cache_hint evict_last (0x14F0000000000000; weights use evict_first
0x12F0000000000000); readiness per job in doneA[j] (helper credits after the CTA's last phase-A epilogue:
__threadfence + atomicAdd; phase-B consumers spin `while (ld_acquire(doneA+j) < RB_A) __nanosleep(64)`); the ring index
runs over A units then B units without a break so TMA already fetches phase-B weights while warps wait on doneA; with
contiguous static split, job j's A and B units sit on the same CTA band. WGS: phase-A epilogue writes SwiGLU output
DIRECTLY into xB in phase B's per-stage wgmma B layout (L2657-2671); readiness per (job, phase-B stage) doneAs[j*16+st]
(8 releases per stage: `red.release.gpu.global.add.s32`; x producer waits `ld_acquire >= 8`; weights producer ungated
"so the weights run ahead by the whole ring even while x waits"). Phase-B outputs: y = acc * gsB*2^62 * (1+post_norm_w)
fp32 to ybuf [EMAX,4096] (evict_last); tiny/fallback also atomicAdd per-pair sum of squares into ss; WGS final pass
computes the norm. Stream-K fixup (WGS L2693-2755): partial share does red.global.add.v4.f32 into wgs_fix(slot) (slot =
CTA holding the unit's first stage), counts stages (phase A atom.acq_rel, last arriver finishes; phase B designated
finisher = share holding stage 0 waits fc >= 16-nst), finisher fence.acq_rel, __ldcg the sums (two n-tiles in
flight), zeroes them, runs the epilogue; the count/release is deferred until after commit group 3 of the next segment's
first stage ("+0.5-1%").

## 2. Weight layout and load path

prepare (L4094-4138): bit/byte permutations only; fp32 global scale applied in the epilogue as gs x 2^62 (TWO62).
Row order: w13 16-row block = 8 gate rows 8rb+g followed by their 8 up rows I+8rb+g, so in any m16 fragment rows g are
gate and g+8 are up; SwiGLU formed from accumulator registers, output column rb*8+g; w2 rows natural.
`_pack_codes` (L3995-4011), mma.sync lane order with K permuted: one 32-bit word = 8 E2M1 codes = one lane's
m16n8k16 A fragment for one k16 step; layout [rb][w (512-K slice = warp)][c (64-K chunk)][g][t][s (k16 step)], K =
512w + 64c + 16t + 4s + j; register R -> (h, j0) = (0,0),(1,0),(0,2),(1,2): a0 = row g K j=0,1; a1 = row g+8; a2 = row g
j=2,3; a3 = row g+8. So mma-k 2t+e <-> K 16t+4s+e and mma-k 2t+8+e <-> 16t+4s+2+e. WHY: a lane's 4 k16 steps in a
chunk cover exactly one 16-K scale block (one scale per row per lane per 4 MMAs), and the B fragments become
contiguous 16-B loads of x (no ldmatrix, no smem for x).
Bit layout inside a word (`_stored_pos`, DQ7=1 forced by the host): code bits (m,e0,e1,s) land at bf16 bits
(6,7,8,15) (+16 for the upper half); R0 stored in place {6,7,8,15,22,23,24,31}; R1 stored 4 lower (decode <<4); R2 at
rotr 10 (decode rotl 10); R3 keeps m,e0,s at -6 (<<6) and e1 at -14 (rotl 14); all 32 positions covered.
Scales (`_pack_scales`): bytes [rb][w][g][t][c][h]; each lane gets 16 bytes (8 chunks x 2 rows) = one uint4 per warp
per unit. Unit order (tiny): per 16-row block 32,768 B codes [w][c][lane][16 B] + 4,096 B scales [w][lane][16 B] =
36,864 B contiguous = one TMA per unit. WGS copy (`_wgs_copy`): natural mma k order k = 64c+16s+8q+2t+e, layout
[256-row unit][stage 16][warpgroup 4][9216 B] = 8 KB codes [grp 4][warp 4][lane 32][4 words] + 1 KB scales
[grp][warp][g][j 4][h 2].

Load path, tiny: ring of 3 slots x 36,864 B; one thread issues each unit: `mbarrier.arrive.expect_tx` +
`cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [slot],[src],UNIT_BYTES,[bar],policy`;
consumers `mbarrier.try_wait.parity`; FAST1 issues 2 units before routing, helper the rest up to NS; refill after the
per-unit CTA barrier by thread 0 into the slot just consumed. x goes straight from global to REGISTERS (`load_x_regs`
L777-799, __ldg for x, __ldcg for inter): each lane holds 16 uint4 = 8 tokens x 512 K per warp (~64 regs), reloaded
only when the job changes (XPF2 already pulled x into L2).
WGS: weights producer issues a 36,864 B TMA per stage into a 3-slot ring (empty barrier count 16: lane 0 of each
consumer warp arrives as soon as the stage's codes and scales are in registers, so the slot is held briefly); x
producer issues one TMA of npad x 512 B per stage from xA/xB into a 3-slot 32 KB x ring (slot freed when the stage's
last wgmma group retires). Phase-A x gather (`wgs_gather`): every CTA copies its share of all jobs' padded token rows
into xA in the per-stage wgmma layout, then red.release xready[j]; GATHB=8 batches 8 rows of loads before stores
("one x row at a time was one dependent HBM round trip per row ... 1.3-7 us at bs256").

## 3. The MMA

Tiny: `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`; A = dequantized weights (8 gate + 8 up rows in phase A),
B = x (n8 = up to 8 tokens); per unit the CTA covers 16 rows x 8 tokens x 4096 K; 8 warps split K (512 each = 8 chunks
x 4 k16 steps = 32 MMAs per warp per unit); b0 = token g's K {16t+4st, +1}, b1 = {16t+4st+2, +3} offset by 512*warp +
64c; c0,c1 = (row g, tokens 2t,2t+1), c2,c3 = (row g+8, ...). NACC=4: one accumulator per k-step -> 4 independent
HMMA chains summed at the end. Cross-warp K reduction: each warp writes a float4 to S.red[q%4][warp][lane]; the helper
sums 8 float4.
WGS: RS wgmma `wgmma.mma_async.sync.aligned.m64nNk16.f32.bf16.bf16 {d0..d3}, {a0..a3}, desc, 1,1,1,0`; A =
dequantized weights in registers, B = x in smem via descriptor; N = 8*ntt, ntt = ceil(job tokens/8) in 1..8 (template
switch); each warpgroup owns 64 rows of the 256-row unit, all 4 read the same x stage; per stage 16 k16 steps = 16
wgmma per warpgroup; WGSGS=2 steps per commit group then wait_group 1; af[2][GS][4] double-buffered; first wgmma of a
segment uses scale-d=0 instead of zeroing ("no zeroing: ptxas would serialize the wgmmas, C7513"); B descriptor
wg_desc(xring, LBO=128, SBO=4096), no swizzle, x slot layout [n8][k8 x32][8 rows][16 B], +256 B per k16 step;
WGSDESC0 builds the slot-0 descriptor once per segment (~8 uniform ops per stage saved). Register budget: "one
register budget (96 at 544 threads) and ptxas serializes every wgmma if any pipeline does not fit (C7512)"; NTL 576 ->
~112 cap; ss_cur() re-read via volatile shared load; wgs_phase __forceinline__ ("a CALL makes ptxas serialize every
wgmma, C7510").

## 4. The dequant

Encoding trick: dq4 writes the E2M1 exponent into the bf16 exponent field and m into the top mantissa bit: exactly
e2m1 x 2^-126 (e=0 -> bf16 subnormal). Scale becomes bf16x2(scale x 2^64); HMUL2 gives e2m1*scale*2^-62, exact (<= 2+4
significant bits); epilogue multiplies by gs x 2^62.
dq4 DQ7 (L448-455): M = 0x81C081C0; r0 = q & M; r1 = (q*16) & M (IMAD.SHL on the FMA pipe); r2 = funnelshift_l(q,q,10)
& M; r3 = ((q*64) & 0x80C080C0) | (funnelshift_l(q,q,14) & 0x01000100) -> "7 ALU + 2 IMAD.SHL per 8 weights".
Scale build (PRMT, active): SCDK(word,b,k16) = __byte_perm(word, 0, 0x4040 | b*0x0101) * k16 + 0x5C005C00: PRMT
duplicates the E4M3 byte into both halves, x16 moves eeee into bf16 exponent bits 10..7 and mmm into mantissa 6..4,
+0x5C00 rebiases by 184 (x2^64); k16 = 16 + p.zero stays opaque so ptxas emits IMAD on the FMA pipe; exact for bytes
0x08..0x7E. Tiny: per lane per chunk (4 MMAs) 2 x (PRMT+IMAD) build sg (row g) and sg8 (row g+8), then a0=r0*sg,
a1=r1*sg8, a2=r2*sg, a3=r3*sg8 (4 HMUL2 per word). WGS (WGSSC2): one PRMT + one IMAD per k16 step giving (s_g, s_g8)
in one word, then bmul2l/bmul2h half-broadcast HMUL2s ("2 instructions per k16 fewer").
Instruction tally per lane: tiny per m16n8k16: 7 ALU + 2 IMAD + 4 HMUL2 + 0.5 PRMT + 0.5 IMAD + 0.25 LDS.128 (codes) +
1 HMMA, NO LDS/ldmatrix for x. WGS per k16: 7 ALU + 2 IMAD + 4 HMUL2 + 1 PRMT + 1 IMAD + ~0.25 LDS.128 + 0.25 LDS.64;
the wgmma amortized over N up to 64.

## 5. Small vs large batches

| | bs1 TINY | bs8 TINY2 | bs16 MID | bs32/128/256 WGS |
|---|---|---|---|---|
| warps | 8 compute + helper | same | same | 16 consumers + 2 producers |
| unit | 16 rows x 4096 K | same | same | 256 rows x 256-K stages |
| tokens/job | <= 8 | <= 8 | <= 8 (SPLIT) | <= 64 (n8..n64) |
| split-K | inside the CTA: 8 warps x 512 K, smem reduce | same | same | stream-K across CTAs, 256-K granularity |
| scheduling | static equal slices | same | same | cost-weighted stage ranges |
| flags | FAST1=2, F1EARLY, WS/WSB, WSBL=2, WSBLK=2, NACC=4, L2H=0, UWARP=0, FINPF | WSBLK=1, CPF=0, L2H on | SPLIT=1, WSBLK=2 | PRMTSC, FINCOL, WGSFINSS |
| measured | "bs1 +2.5% / +3.5%" | "bs8 +2.0..2.5%" | (Oct 26) | WGSA table "bs32 +0.45, bs128 +0.94, bs256 +0.17%" |
FAST1: unit v -> expert = ((v>>9)+1)-th set bit of the touched-expert mask (__fns); the last warp issues these TMAs
"before routing, off the cold-start critical path"; F1EARLY: topk loads first, lane 0 inits mbarriers while in flight.
WS details: compute warps per unit: wait TMA mbarrier; dequant + 32 HMMA; wait wsre[bb]; store float4 partials;
CTA_SYNC; thread 0 refills slot q+NS and arrives on wsrf[bb]. Helper: wait wsrf, sum 8 float4, release wsre, epilogue
(A: SwiGLU -> bf16 inter; B: y -> ybuf + ss atomic). NRB = WSNR = 4 partial buffers; WSBL=2: last WSBLK phase-B units
keep their epilogue on compute warps 0..K-1 with gsB / norm weights prefetched (shortens the tail); "WSBLK > WSNR
deadlocks the partial ring". Load balance at bs1: each touched expert = 28.31 MB; with 2 experts 1024 phase-A units /
132 CTAs ~ 7.8 units (~280 KB) per CTA and ~3.9 phase-B units; ring keeps 108 KB in flight per SM.
WGS for small N: at bs32 most jobs have 1-3 tokens -> m64n8k16 ("M 17..63 runs on the WGS MAIN build: faster than the
MID kernel at bs32").

## 6. Routing prologue and final pass

route() (L478-596), E = 8M <= 3072: eid = topk_id - lo or -1 (WGS: all loads int4 issued before smem stores);
per-warp histograms with __match_any_sync + __popc; warp 0 (lane = expert): cnt/off scans, per-warp write offsets,
job table (fat first then skinny), LPT forder for fat jobs (non-WGS); positions pass: ent[q] = token*8+slot sorted by
expert, pos[i] = q or -1 (q = pair row).
Final pass: out[t] = sum_s topk_w * rsqrt(ss_s/4096 + 1e-8) * y_s (y includes 1+w). WGS: one token per CTA iteration,
8 columns per thread; pass 1 sums y^2 per route (shuffles + 16 warp partials through the freed ring smem); pass 2
re-reads rows (L2 hits), stores bf16. Tiny (FINCOL): item = (token, 1024-column segment), 4 items per token, 4 columns
per thread; FINPF: first item's pos/topk_w loaded before the barrier, its 8 ss + 8 ybuf loads batched.

## 7. Switches with recorded measurements (shipped values)

NS 3 (host); L2H 1 (TINY 0): weights evict_first, inter/ybuf evict_last; XPF2 1 ("pull every token row of x into L2
now"); EARLYN 1; SPLIT MID only; SKT 8 (16 would add a second n-tile via smem x); FT_T 64; FATSTATIC 1; DYN off in tiny;
WS/WSB tiny 1; WSBL tiny 2; WSBLK tiny/MID 2, TINY2 1; WSNR 4; FAST1 tiny 2, WGS 0; F1EARLY 1; NW 8 (WGS 16); NACC tiny
4; DQ7 forced 1; PRMTSC when scales in 0x08..0x7E; SKSC2 0 ("MID +0.8% at bs16 with FTSC2; TINY -0.5%"); UWARP 1 (tiny
0) ("warp index through a shuffle so ptxas keeps warp-derived values in uniform registers"); FDIV 1 ("SwiGLU's divide
as MUFU.RCP + FMUL"); EPI4 0 in shipped builds; FINCOL 1; FINPF tiny 1; RLOAD WGS only ("small builds lose 1-2%");
WGSNX 3; WGSA "112,135,140,158,173,185,200,207" (tuner Oct 26: bs32 +0.45, bs128 +0.94, bs256 +0.17%); WGSB
"100,104,115,123,132,144,156,174"; WGSREORD 1; GATHB 8; WGSPR 1; WGSDFIN 2; WGSFCI 3 (+0.5-1%); WGSGS 2 ("4 serializes
the N=16..32 variants at 96 registers: C7512"); WGSSC2 1; WGSDESC0 1; WGSFINSS 1. muldiv_g: "~1 us" for divide
subroutines from a cold i-cache; wgs_prefix 32-bit arithmetic; wgs_flush keeps two n-tile loads in flight ("ntt
serialized L2 round trips under full HBM load").
What won: bs1/8/16 = mma.sync K-split + helper-warp epilogue + FAST1 pre-routing TMA + NACC=4 + DQ7 decode with
PRMT+IMAD scales + L2 prefetch of x; >= 17 tokens = RS-wgmma with separate weights/x TMA producers, stream-K cost
tables, phase-A epilogue writing straight into phase-B's wgmma B layout, per-stage doneAs gating.

## 8. Host side

prepare compiles every variant with load_inline: -O3 -gencode=arch=compute_90a,code=sm_90a -lineinfo -DOKC_NS=3
-DOKC_FNST=4 -DOKC_NSF=3 + variant defines + -DOKC_DQ7=1. Scratch allocated once (`_make_scratch`): int32
[8+1024+EMAX] (ctr[0..1] pools, ctr[3] generation, ctr[4..5] barrier; doneA at +8; xready at +520; ss1 at +1032);
fp32 ss, ybuf [EMAX x 4096], WGS fixup + doneAs; bf16 inter, WGS xA/xB. kernel_fn: variant by M; ids.clone() only if
topk_ids not 16-B aligned; out = torch.empty; okc_run once per chunk of <= 384 tokens: cudaFuncSetAttribute once,
launch moe_fused<M>=64> with grid = SM count, block = NTL, dynamic smem = okc_smem_bytes(). PDL infra exists but PDL =
False ("one launch per call gains nothing"). No cluster launch attributes (.shared::cluster is just PTX spelling).
Graph safety: no host syncs or value branching; per-launch state resets inside the kernel (ss double-buffered by the
generation counter bumped at the grid barrier; barrier counter alternates ctr[4]/ctr[5]; other copies zeroed during the
launch; flags zeroed after the barrier; fixup slots zeroed by their finisher). DIAG (off): graph-replay timing with L2
flush, TMA/cp.async bandwidth microbenchmarks ("tma 3x36KB").

## 9. Differences from our kernel (cuda_moe.py)

1. Scale build: their K permutation makes one scale block cover 4 consecutive words of a lane -> 0.5 PRMT + 0.5 IMAD
   per word; ours: cvt + 2 mov per word (our conversion-free dequant since 2026-10-26 differs; recount).
2. Decode: 7 ALU + 2 IMAD (FMA pipe) vs ours ~5 ALU + 4 PRMT.
3. B operand: their x fragments stay in registers for the whole job (no ldmatrix per step); ours reads x via ldmatrix
   from smem.
4. Latency hiding: 4 independent accumulators, epilogue on a helper warp, TMA issued by one thread.
5. Launches: ONE launch per call (routing + both GEMMs + norm/sum), first-unit TMA issued before routing; at bs1
   against an 11.4 us floor, launches and cold start are a large share (ours: 4 launches + PDL).
6. Large batches: RS-wgmma (m64nNk16, N up to 64) with stream-K; GEMM1 epilogue writes straight into GEMM2's wgmma B
   layout, gated per 256-K stage.
