# muse_decode_layer: map of the derived layer file, per-shape time budget, composition tool

Judged: `muse_decode_layer/submission.py` at commit 5c08f5b = job f662ee18cbb842eb, 2.092x (CI 2.061-2.123), 13/13,
rank 50. Line references "old L" are to that file (13,774 lines); "new L" to the regenerated file of commit c6ca340
(14,422 lines, same components, produced by `tools/compose_decode_layer.py`).

## 1. Layout of the layer file and which component version is embedded

| block | old L | new L | what |
|---|---|---|---|
| header, constants, PDL rule | 1-79 | 1-82 | `PDL = True`, `PDL_MAX_T = 32` (L59/62), `PIN_PICK` (L73/76), `_pdl(T)` |
| attention CUDA (`ATT7_CUDA_SRC`) | 80-4925 | 85-4983 (`att_CUDA_SRC`) | BYTE-IDENTICAL to `muse_attention_decode/submission.py` L53-4898 (236,836 chars; `okc_attn_v298`) |
| attention host | 4926-5123 | 4984-5284 | identical to the standalone L4910-5138 modulo names (`_WS7`, `_plan7`, `_TUNED7`, `OKC_*`, `_cluster_bufs7`); the layer added its own builder `_att7` (old L4938) with flags `-DOKC_QDIRECT=0 -DOKC_PK2=0 -DOKC_W0ENABLE=1` (standalone defaults: 1 / 1 / 0, CUDA L77/358/225): SAME SOURCE, DIFFERENT BUILD, ext name `muse_att_v298w1_{arch}` |
| MoE CUDA (`MOE_CUDA_SRC`) | 5124-8626 | 5288-8836 (`moe_CUDA_SRC`) | BYTE-IDENTICAL to `muse_moe_w4a16/submission.py` L47-3549 (167,206 chars; `okc_moe_w4a16_v216x`) |
| MoE host | 8627-8890 | 8837-9618 | the layer re-implemented the build choice (`_moe(M)`, `_moe_tiny*`, `_moe_mid`, old L8702-8756) and prepare (`_ma_prepare`, old L12172); the define tuples are the standalone's EXCEPT `TINY_DEFINES` (old L8640) lacks `-DOKC_FINPF=1` (standalone L3914: "Oct 26: bs1 +2.5% / +3.5%"); TINY2/MID carry FINPF in both. So the standalone MoE is one flag NEWER. |
| proj (`bfp_*`, every name prefixed) | 8891-12126 | 9622-12852 | a DIVERGED FORK of the bf16_proj module, not an older copy (see 1.1) |
| layer glue | 12127-13774 | 12853-14422 | bfp bridge, Triton kernels, tuner, prepare, kernel_fn, _after_attention |

### 1.1 The proj fork vs `muse_bf16_proj/submission.py` (diff: 291 lines only in the layer, 1,169 only in the standalone)

Layer-only (needed by the layer; the standalone has none of it):
* sk2 PDL: `sk2_set_pdl(n)` / `g_sk2_pdl` (SK2_SRC), the kernel takes `pdl`, issues the first `P = min(pdl, NSTAGE, nst)`
  weight stages before `griddepcontrol.wait`, every thread waits before touching x, `griddepcontrol.launch_dependents`
  after its main loop; `launch_sk2` adds `cudaLaunchAttributeProgrammaticStreamSerialization`. Host: `bfp_SK2_PDL`
  (new L12474), set per call by `_bfp_pdl(T)` (new L12871) to `BFP_PDL_N = 2`.
* fused pooler + router (`FUSE` template flag, `sk2f` entry point, `bfp_fused_pooler_router` new L12454): after its pooler
  stages every CTA streams one router piece (8 rows x quarter K, hi|lo) through the same ring; quarter partials to
  `fused_part`, last-of-4 CTA (release/acquire counter `fused_cnt`) writes the fp32 router output. Only with the
  `pooler` config (NT 4, WSK 8, UPW 1) at tokr 8/16; at T = 32 `PIN_PICK[(32,"pr")] = "bfp"` runs pooler32_r144 and
  the 3-CTA-cluster router kernel separately.
* bounded cross-CTA spins (`okc_spin_ge`, trap after 2^24 polls), L2 `evict_first` hints on weight loads
  (`createpolicy`), `bulk_g2s` / `tma_load_3d` with `.L2::cache_hint`.
* escape table entries carry the precomputed fp32 delta (`comp["delta"]`).

Standalone-only (newer, v132-v141): escapes corrected from the original bf16 bits in the kernel (`esc_delta`, `wbits`);
escape-balanced row permutation for qkv (`SK2_BALANCE`, v135/136); retimed sk2 configs `pooler16t_r192`,
`unpooler16t_r72`, `qkv8t_r144`, `qkv16t_r200`, `o16t_r68` (-0.1..-0.6 us each at tokr 8/16); m = 1 GEMVs (`cgemv`
12-bit, `sgemv`, `rgemv`: pooler -0.34 us); m 33..256 routes `gqx`, `rk` (router), `torchT`, CUTLASS pingpong (pooler
m=256); router wide path via TMA'd hi|lo tiles + SS-wgmma; release/acquire counters in the wide kernel; selftest/tune
prints on. None of these reach the layer today. **To take a newer proj into the layer, port sk2 PDL + sk2f onto it
(the diff above is ~150 lines of SK2_SRC + 40 host lines) or port the standalone's gains into `proj_fork.py`.**

### 1.2 Per-call launch sequence (kernel_fn new L14258, _after_attention new L14329)

T <= 32 (decode 1/16/32 = T 1/16/32, verify 1 = T 8); every launch PDL (`_pdl(T)` true; Triton `launch_pdl`,
sk2 `pdl=2`, attention `tma` bit 16, MoE `okc_set_pdl(1)`); single stream, no memset in the graph:

| # | kernel (profile name) | source | notes |
|---|---|---|---|
| 1 | `sk2_kernel<1,8,3,1,4,2,6,...>` (qkv) | bfp sk2 `qkv8/16/32_r*` | `mm_pick[(T,"qkv")] = "bfp"` (tuner) |
| 2 | `attn_kernel<L,FLAT,MODE>` | att | `_decode_attention7` -> `att__launch(pdl=True)`; plan per 1.3 |
| 3 | `sk2_kernel<..,4,1,4,1,..>` (o) | bfp sk2 `o8h_r88`/`o32h_r100` | gate already applied in attention's epilogue |
| 4 | `_boundary1` | Triton, one program per token, 16 warps | residual gate 1, rmsnorm, (1+w): s1, h |
| 5 | `sk2_kernel<1,8,4,1,8,1,6,..>` = sk2f fused pooler+router (T 1/8/16); at T=32: `pooler32_r144` + `router` (CL 3) | bfp | `mm_pick[(T,"pr")]` = "bfpf" / "bfp" |
| 6 | `_route_split` | Triton, one program per token | sigmoid(0.1768 logits)+bias, top-8, local ids, pre-expert rmsnorm -> xm |
| 7 | `okc::moe_fused<false>` | moe TINY (T<=4) / TINY2 (<=8) / MID (<=16) / WGS main (17..) | one launch, self-resetting counters |
| 8 | `sk2_kernel<1,8,1,1,8,2,4,..>` (unpooler slice, T<=8) / `unpooler16h_r100` / `unpooler32_r144` | bfp | T <= 8 uses the 1024-row slice |
| 9 | `_boundary2` | Triton, one program per token | residual gate 2, post-ffn norm, next-layer norm: out, s2 |

T > 32 (verify 16 = 128 tokens, verify 32 = 256): no PDL anywhere (`PDL_MAX_T`), cuBLAS for every dense GEMM:
qkv `torch.mm` (`PIN_PICK` True) -> attention (flat kernel, no PDL) -> `_normgate` (Triton) -> o `torch.mm` fp32 out ->
`_boundary1` -> pooler+router as ONE `torch.mm` on `st["pr"]` = cat(pooler, router_hi, router_lo) [4608 x 8192] ->
`_route` -> `okc::moe_fused<true>` (WGS main build) -> unpooler `torch.mm` -> `_boundary2`. cuBLAS kernels in the
profile: `nvjet_sm90_tss_128x64..TNT` x2 (qkv + unpooler), `..96x64..TNN` (pooler+router), `..splitK..` + `splitKreduce`
(o) at T = 128; `128x128` x2, `80x128`, `96x64` at T = 256.

Attention plan per scored shape (`att__plan` new L5078, `att__TUNED` L5057): sliding q1 B=1 -> plain split-KV + SPEC
(`attn_kernel<1,false,13>`); full q1 B=1 -> coop 252 splits (`<1,false,11>`); sliding q1 B=16/32 -> plain
(`<1,false,1>`); full q1 B=16/32 -> flat (`<1,true,0>`); verify sliding B=1 -> `<8,false,15>`, B=16/32 -> plain
(`<8,false,3>`, OKC_PLAIN8); verify full B=1 -> `<8,false,11>`, B=16/32 -> flat `<8,true,0>`.

Layer-specific kernels (all Triton, new L12979-13800): `_gemm_splitk`, `_gemm_splitk_ng`, `_normgate`, `_qk_prep`,
`_attn_split`, `_attn_combine` (FA3-less fallback path only), `_gemm_o` + `_bnd1_apply`, `_route_a`/`_route_b`,
`_bnd2_sums`/`_bnd2_apply` (the "v2" parallel variants: `state["gluev2"]` False at every T in practice), `_boundary1`,
`_route`, `_route_split`, `_boundary2`, `_moe_align`/`_moe_w13`/`_moe_w2`/`_moe_combine` (dead), `_pexp_norm_sum`
(Marlin path only), `_marlin_experts` (tuner candidate when sgl_kernel exists; loses to the moe kernel).

prepare (new L14057): bfp prepare (compress 5 matrices, its own autotune), attention + 4 MoE builds in threads,
`_ma_prepare` (moe_prepare + guarded scratch), `_tune` (per T in 1..256: dense backend per matrix via graph timing
with a cold-L2 read sweep, marlin vs agent, gluev1/v2), `PIN_PICK` overrides, `_attn_selftest` (11 plan families),
`_check_guards`. prepare prints nothing (the f662 log is empty).

## 2. Per-shape time budget (profile of job f662ee18, GPU time per kernel in us; judged time is the median paired replay)

Caveat: under PDL (T <= 32) "an overlapping kernel's time includes any wait for the one before it", so the kernel
rows over-count; the judged time is BELOW the kernel sum on 10 of 12 shapes and the gaps are not resolvable from this
profile. At T > 32 (no PDL) sum - judged = 3..16 us = profiler overhead over 9-10 launches: no launch gaps in the graph.

| shape | judged | attention | MoE | dense GEMMs (qkv / o / pool+router / unpool) | glue (bnd1 / route / bnd2) | sum |
|---|---|---|---|---|---|---|
| d1 sliding | 77.1 | 4.9 | 36.7 | 39.3 (10.1 / 5.8 / 17.7f / 5.7) | 13.2 (4.5/4.3/4.4) | 94.1 |
| d1 full | 83.1 | 11.0 | 29.0 | 38.9 (10.2 / 5.8 / 18.3f / 4.6) | 13.0 | 91.9 |
| d16 sliding | 136.0 | 7.4 | 60.4 | 51.1 (10.3 / 6.1 / 18.8f / 15.9) | 12.8 | 131.7 |
| d16 full | 214.8 | 88.1 | 77.2 | 50.6 (10.1 / 7.7 / 17.7f / 15.1) | 13.8 | 229.7 |
| d32 sliding | 161.7 | 9.3 | 86.3 | 59.9 (11.6 / 6.7 / 16.3+5.4 / 19.9) | 13.3 | 168.8 |
| d32 full | 291.8 | 169.8 | 85.6 | 61.1 (11.2 / 7.2 / 16.4+5.4 / 20.9) | 14.1 | 330.5 |
| v1 sliding (T=8) | 109.1 | 7.0 | 37.9 | 40.8 (10.3 / 5.9 / 19.8f / 4.8) | 12.6 | 98.1 |
| v1 full | 122.2 | 14.7 | 72.8 | 40.6 (10.2 / 6.3 / 18.6f / 5.5) | 12.7 | 140.9 |
| v16 sliding (T=128) | 259.9 | 11.9 | 162.6 | 76.5 cuBLAS (30.8 x2-call / 16.1+3.1 / 26.5) | 16.3 (5.8/4.7/5.8) | 267.3 |
| v16 full | 345.4 | 105.3 | 160.2 | 76.6 | 16.1 | 358.2 |
| v32 sliding (T=256) | 331.2 | 15.4 | 198.3 | 96.6 cuBLAS (38.5 x2 / 34.0 / 24.1) | 25.6 (9.6/6.2/9.8) | 335.9 |
| v32 full | 497.6 | 186.3 | 207.6 | 94.8 | 25.0 | 513.8 |

(f = fused pooler+router sk2f launch.) Attention vs its board (same shapes): parity everywhere (4.9/5.0, 11.0/11.0,
7.4/8.0, 88.1/87.0, 9.3/10.0, 169.8/165.8, 7.0/6.9, 14.7/14.0, 11.9/11.3, 105.3/101.5, 15.4/14.6, 186.3/~200).
MoE vs its board is NOT comparable: the layer's routing (judge: ~1.6 local routes per token, busiest range) differs
from the MoE board's inputs (bs16 board 99.3 vs 60-77 in-layer; bs1 board 24.9 vs 29-37).

Largest gaps between in-layer component time and the standalone kernels' judged times:
1. Dense GEMMs at T > 32 through cuBLAS: 76.5 vs 69.3 (standalone proj m=128 sum: qkv 14.9 + o 7.2 + pooler 22.3 +
   un 18.7 + router 6.2) = ~7 us on both verify-16 shapes (2.7% / 2.0%); 96.6 / 94.8 vs 89.2 (m=256) = ~6-7 us on
   verify-32 (2.0% / 1.4%). Needs the standalone's m >= 33 routes (gqx / rk / torchT / CUTLASS) in the layer.
2. sk2 unpooler at T = 32: 19.9 / 20.9 vs 14.9 standalone (+5-6 us = 3-4% of d32 sliding); T = 16: 15.9 / 15.1 vs
   13.8 (+1.3-2.1). Part is PDL accounting (the unpooler launches during the MoE tail and its time includes that
   wait); the standalone's `unpooler16t_r72` config is -0.6 us. qkv/o/pool at T <= 32 are within +0.5..1.2 us of
   the board (PDL wait included), un slice at T <= 8 +1.1..2.2.
3. The three Triton glue kernels: 12.6-14.1 us per T <= 32 shape (8-17% of the judged time), 16 us at T = 128,
   25 us at T = 256. Each is one program per token over 8192 (boundaries) or 4096 + 256 (route) columns: ~4.4 us
   for a kernel that moves ~100 KB at T = 1 (latency floor ~2 us each).

## 3. Layer-level opportunities (estimated us per shape)

* Fuse / shrink the glue kernels (item 3 above): `_route_split` into the sk2f epilogue (the last-of-4 CTA already
  reduces the router logits; top-8 of 256 + the pre-expert norm of the pooled row is a few hundred FLOPs per token)
  and `_boundary2` into the unpooler sk2 epilogue / `_boundary1` into the o-projection epilogue (each needs one
  row-wide sum of squares: last-CTA-per-row-block reduction, the sk2 kernels already have release/acquire grid
  counters in the FUSE path): -2..-2.5 us per fused kernel, -6..-8 us per T <= 32 shape (5-9%), -10..-15 us at
  T = 128/256 if the same is done on the cuBLAS path (a `torch.mm` cannot fuse: use the proj module's kernels there).
  Cheaper first step: multi-CTA boundary kernels with the per-token sums done by a 2-stage (sums, apply) split only
  where it wins (the v2 path exists, new L13321-13483, but adds a launch; the tuner rejected it at every T).
* Dense GEMMs at T = 128/256 (item 1): adopt the standalone proj module's m 33..256 routes (-7 us each on the four
  verify-16/32 shapes, 1.4-2.7%). Requires either porting sk2 PDL + sk2f into the standalone proj (preferred: then
  `--proj muse_bf16_proj/submission.py` composes directly) or porting the routes into `proj_fork.py`.
* MoE TINY build with `-DOKC_FINPF=1` (standalone, "bs1 +2.5% / +3.5%"): in the layer's d1/v1 shapes the MoE is
  29-38 us -> -0.7..-1.3 us (check c8b5aab9 measures exactly this).
* PDL at T > 32: off by the layer's measurement (-1% at T = 256 for the Triton glue); the sk2/MoE kernels' PDL would
  only matter if the dense GEMMs left cuBLAS (nvjet kernels take no programmatic launch).
* No redundant memsets in the T <= 32 graph (attention workspace and MoE scratch self-reset; sk2f counter is a
  monotonic ticket); the v2 glue path's `torch.zeros` are never captured.
* Attention build flags: the layer compiles the attention with QDIRECT=0 PK2=0 W0ENABLE=1 (ATT_BUILD_FLAGS, new
  L14142); the board's 2.588x build uses the defaults. Worth one A/B check in the layer (q8 shapes only).

## 4. Composition tool

    python3 tools/compose_decode_layer.py --check            # regenerate muse_decode_layer/submission.py
    python3 tools/compose_decode_layer.py --check --moe PATH --attn PATH --proj PATH -o OUT

Inputs: `muse_attention_decode/submission.py` (prefix att_), `muse_moe_w4a16/submission.py` (moe_),
`muse_decode_layer/proj_fork.py` (bfp_; the layer's proj fork, un-prefixed so it diffs against the standalone),
`muse_decode_layer/layer_glue.py` (header + `# {{COMPONENT attn|moe|proj}}` markers + glue). Each component is
embedded whole; every module-level name is prefixed by an AST walk with scope resolution (locals, parameters,
attributes, keyword arguments, strings untouched; `global` statements and f-string expressions handled). Verified:
re-prefixing the un-prefixed proj fork reproduces the judged file's bfp_ block byte-for-byte (97 names);
`--check` compiles the output and lists module-level references that nothing binds (one pre-existing: `MID_SRC`
in the proj module's dead `_mid_src`). The glue calls the components' own entry points: `att__build("w1",
ATT_BUILD_FLAGS)`, `att__launch(..., pdl=)`, `att__supported`, guarded `att__workspace`/`att__cluster_bufs`
(rebound, new L14198-14214); `moe_prepare` (+ guarded scratch), `moe__ext` prebuilt in threads, `moe_kernel_fn`
with `okc_set_pdl` per call (`_ma_run`, new L12917); `bfp_prepare`, `bfp_kernel_fn`, `bfp_fused_pooler_router`,
`bfp_SK2_PDL`. The attention CUDA/host and the MoE CUDA are byte-identical to the standalone files, so a newer
standalone attention or MoE composes in with no edit as long as `_launch(data, pdl=)`, `_build(tag, flags)`,
`_workspace`/`_cluster_bufs`/`_WS`/`_CL`, `prepare`/`kernel_fn`/`_ext`/`_EXTS`/`_main_defines`/the *_DEFINES
names keep their meaning (`--check` flags a removed name).

Validation runs: job 3e87d0e5862f4f29 = regenerated layer with the SAME component versions as f662ee18 (standalone
attention; standalone MoE with the layer's TINY_DEFINES, i.e. no FINPF; proj fork); job c8b5aab94fd34546 = the
regenerated layer with the standalone MoE as is (TINY FINPF=1). Results: see section 5 (filled in when they land).

## 5. Results (all 13/13; the judge's CI is +-1.5%)

| run | layer | speedup (CI) | d1 s/f | d16 s/f | d32 s/f | v1 s/f | v16 s/f | v32 s/f |
|---|---|---|---|---|---|---|---|---|
| f662ee18 | hand-built file (commit 5c08f5b) | 2.092 (2.061-2.123) | 77.1/83.1 | 136.0/214.8 | 161.7/291.8 | 109.1/122.2 | 259.9/345.4 | 331.2/497.6 |
| 3e87d0e5 | regenerated, same component versions | 2.0885 (2.057-2.120) | 76.5/81.9 | 130.3/209.8 | 155.0/312.4 | 115.2/121.6 | 263.6/352.8 | 331.6/- |
| c8b5aab9 | regenerated, standalone MoE as is (TINY FINPF=1) | 2.0911 (2.060-2.122) | 75.0/81.5 | 133.0/207.0 | 147.7/302.7 | 115.8/119.5 | 267.7/353.7 | 339.0/- |

The composition round-trips (all three within one CI). The MoE kernel's time on the SAME code moves by 10-20 us
between runs on half the shapes (profiles: d1 full 22.6 vs 28.8 vs 29.0; v16 sliding 146.7 vs 165.7 vs 162.6;
v32 sliding 236.0 vs 231.7 vs 198.3), i.e. the judge's fresh inputs route differently each run; a component change
below ~3% on the MoE cannot be resolved by one layer check. Profiles: `muse_decode_layer/profiles/
submission_samever.3e87d0e5862f4f29.txt` and `submission.c8b5aab94fd34546.txt`. Current `submission.py` = c8b5aab9.

## 6. Round 2 (layer-level levers), judged A/Bs (all 13/13 unless noted; baseline c8b5aab9 2.0911)

| job | change | speedup (CI) | verdict |
|---|---|---|---|
| 7d23a37c | ATT_BUILD_FLAGS = () (standalone defaults QDIRECT=1 PK2=1 W0ENABLE=0) | 2.0905 (2.059-2.122) | geomean noise; attention rows 0.3-3 us faster on 10/12 shapes (flat q1 87.3 vs 90.4, 169.3 vs 172.6) -> adopted |
| e1500783 | "mm16": bf16-output cuBLAS candidate for qkv/o/un, tuned per T | 2.0983 (2.067-2.130) | best; verify-16/32 -6..-17 us vs c8b5 (noise-sized) |
| 19ea8e60 | routing fused into sk2f's last CTA (v1) | 2.051 | REGRESSION: tuner bias (routing timed only in the bfp candidates) picked Triton split-K pr (21.6 + 6.0 us); tail's partial reduction was a 128-load chain |
| 0ac3dcbb | route tail v2 (token-major partials, warp tree) + tuner consistency + pr pinned to bfp paths | 2.0975 (2.066-2.129) | separate pooler+router still won at every T (prB:27); fused tail never ran |
| 91ec8f0a | standalone-proj composition (v1) | failed | glue passed route=None to the standalone's 2-arg fused_pooler_router |
| 55cdc71f | standalone-proj composition (muse_bf16_proj with the sk2 PDL + sk2f port, CUTLASS off) | 2.0305 (2.000-2.061) | loses on every T<=32 shape (qkv sk2 picks 16-18 us vs the fork's 14-15, un 19-21 vs 18-19); T>32 unchanged (the layer never routes through the proj module above BFP_MAX_T). proj_fork.py stays |
| d467a5c5 | route tail + boundary tails (sk2b: o->boundary1, unpooler slice->boundary2, T<=8) behind prepare-time self-tests | 2.011 | self-tests PASS (ids identical, wts/xm 1e-8) but the single-CTA tails are LATENCY-BOUND: fused+route 29/35/43/64 us at T 1/8/16/32 vs 27 separate; the boundary tails made the tuner pick the v2 glue (d1 88 vs 75 us). Both OFF (28fda91), code + self-tests kept |
| fc83beb0 | clean: mm16 + ATT flags + tuner fixes, tails off (HEAD 28fda91) | 2.0765 (2.045-2.108) | picks as intended (prF T1-16, prB T32, mm16 qkv/o/un at T>=128); per shape vs e150: d16 -4/-7, d32 +4/+19 (attention/MoE noise), verify +-4 us. Best judged file stays e1500783 (commit e36a421, 2.0983); the three clean runs e150/0ac3/fc83 span 2.077-2.098 on near-identical code |

Lessons: (1) a "last CTA finishes the glue" tail has one 16-B load in flight per thread per round (~0.7 us a round):
T=32 x 4096 bf16 needs 64 rounds; the Triton per-token kernels issue a whole row at once. A tail would need all loads
hoisted (8+ in flight per thread) and/or the last K CTAs cooperating; expected gain even then <= 1 us per shape at T=1.
(2) The tuner must time exactly what dense()/the glue will run for a pick (bfp above BFP_MAX_T ran as fp32 cuBLAS;
pr candidates must include their routing kernel). (3) At T=128/256 the o projection runs a cuBLAS split-K kernel
(15.7+3.2 / 23.7 us) in the layer even as bf16 F.linear, while the proj module's isolated timing of the same call is
10.6 / 12.5 us: unexplained, worth a dedicated look. (4) MoE and attention kernel times move 5-20 us run to run on
identical code: judge component changes by their kernel rows, never by the geomean.

## 7. Follow-up: the T = 128/256 dense GEMMs (job ba80dfcdbc2c48ee, 2.0963x, 13/13; commit fa2fc10)

The "o projection runs a 19-24 us split-K kernel" in section 6 was a misattribution: at T = 128 the judge profile's
`splitK` kernel (15.4 + 3.2 us) is the qkv GEMM (N = 2304 = 24 tiles of 96 x 64, K = 8192: cuBLAS splits K to fill the
GPU); o and the unpooler share the `128x64` kernel (29 us for both = ~10.6 + ~19), exactly the proj module's isolated
timings. The tuner's DIAG (every candidate's graph time per matrix, judge .log `T128/.../T256/...` lines):

| T | qkv (floor 7.9/7.9) | o (3.5) | pooler+router cat (15.7) | unpooler (14) |
|---|---|---|---|---|
| 128 | Triton 20.8, cuBLAS fp32 20.5, bf16 20.6, K-major 20.4, proj-gq 20.3 | 11.4 / 11.4 / 10.9 / 10.8 / 11.6 | 32.2 / 32.2 / K-major 33.1 / proj pool+router 41.2 | 23.6 / 23.4 / 23.4 / 22.1 / 23.3 |
| 256 | 23.7 / 23.3 / 23.1 / 24.2 / 23.0 | 13.6 / 13.7 / 12.8 / 12.7 / 12.7 | 39.2 / 38.6 / 43.3 / 50.8 | 29.6 / 29.5 / 27.8 / 28.2 / 27.6 |

Every candidate is within ~1 us of cuBLAS on every matrix; hypotheses (a) workspace/heuristics, (b) input layout,
(c) PDL neighbours are moot (no discrepancy exists); (d) the proj module's m >= 33 routes gain <= 1 us. The judged
T > 32 shapes are unchanged within noise (verify-16 255.3/345.1, verify-32 322.1/490.5). The remaining headroom there
is cuBLAS vs the bandwidth floor (qkv 20 vs 7.9 us, pr 32 vs 15.7, un 22 vs 14, o 11 vs 3.5: ~45 us per verify-16
shape, ~55 per verify-32), which needs a custom skinny-M (M = 128/256) GEMM, not a dispatch change.

## 8. Round 4 (launch removal under PDL), judged (all 13/13 unless noted)

| job | change | speedup | finding |
|---|---|---|---|
| 30d1ae0f | boundary-1 prologue v1 (every sk2f CTA writes the full h to global, then TMA) + PDL depth sweep | 2.053 (d1 85.6/92.8) | 128 CTAs writing the same 16 KB serialize in L2 (~20 us in the chain); tuner fell back to the v2 glue. PDL depth 2 = 3 = 4 = 6 at every T |
| 960a1e77 | prologue v2: h in 16 KB shared memory (swizzled unit layout, no x boxes) + MoE-side routing (872689c) with a grid-barrier norm | 2.0855 | prologue correct (pooled/logits bit-identical) but my self-test compared the unused global h and disabled it; MoE-side routing a wash: rim[off/on] 72/76, 100/104, 157/157, 233/233 |
| 0f57b859 | prologue v2 enabled | 2.0516 (d1 87.1/92.0) | the v1 chain measured +12 us at T=1 (age1 84 vs 72) -> v2 glue fallback again |
| 1a4846de | prologue v3: staggered column sweep + isolated gate; routing modes split / sk2f-side (CTA t after the barrier) / MoE-side | 2.0967 | isolated: fused launch 23.4 -> 34.0 us (+10.6): not a hot-spot. rmode T1 73/78/77, T8 114/117/118, T16 126/126/125, T32 180/181/181: a grid barrier at the END of a PDL-chained kernel costs 3-5 us (SMs stay held until the last CTA; the early-launched MoE CTAs cannot start) |
| 2ac47839 | attention -DOKC_PDLKV=1 | 2.0852 | attention rows +0.4..+1.1 us on 10/12 shapes vs the same-base run; not adopted |
| 200a83ac, 8c7152a2 | + MoE a388dff (pre_norm_partials), prologue v4 (L2 prefetch of the input rows before the wait) | failed in prepare | the MoE module's OWN routing self-test deadlocked after bs=32 (its distributed-selection barrier; fixed in dc4ca7a, self-test off) |
| 711bbae5 | HEAD 2e669e4: dc4ca7a + prologue v4 chain-timed + 4 routing modes | 2.0762 | prologue isolated 22.8 -> 36.1 (+13.3) even with the prefetch: the cost is the prologue's own load issue (128 live floats per thread under the 128-register cap -> register-bound batches, ~25 dependent rounds; ptxas spills were the symptom). rmode T1 70/70/71/70, T8 107/108/107/107, T16 158/159/158/157, T32 183/182/182/181: the no-barrier MoE mode (self-test: ids exact, output 5e-3) repays the 4.3 us launch exactly, no more |

Net: no lever produced a judged kernel-row win; every new path is self-tested and auto-gated (off unless it wins
on the chain), so HEAD is safe but not faster. Lessons: (1) under PDL a kernel must not end with a grid barrier
(it blocks the dependent's early start: 3-5 us); (2) a CTA-local prologue that needs a whole 8192-wide row is
load-issue-bound unless it keeps <= ~16 elements per thread in flight (the Triton kernels' layout), i.e. it needs
>= 512 threads or a streamed two-pass structure; (3) the per-token top-8 costs ~2.3 us wherever it runs, so moving
it only pays if nothing else is added; (4) the judge flushes L2 before each replay: anything an input row is read
by must be prefetched before the kernel's gdc_wait.

## 9. Overnight (launch removal, m = 1 GEMVs), judged

| job | change | speedup | finding |
|---|---|---|---|
| f046f778 | cooperative boundary-1 prologue (CTA c: its 64 columns of all rows, 2 grid tickets at the kernel start, producer streams the ring meanwhile, x boxes after h) | 2.0899 | isolated cost +3.9 us at T=1 (was +13.3 whole-row), +6.8 at T=8; registers 101-130, zero spills; gated off by an all-or-nothing self-test bug (fixed per T) |
| 89687f36 | + m = 1 GEMVs ported (cgemv 12-bit planes qkv/o/pooler/unpooler slice, sgemv, rgemv router) | 1.847 (build failure) | the ported block needed the standalone's `esc_delta`; the offline harness had skipped the new block; the bf16_proj backend was disabled everywhere |
| 47921ff6 | fixed; pooler+router form at T <= 8 chain-timed among separate GEMVs / fused / fused+prologue | 2.0952 (2.064-2.127) | rows at T=1: qkv cgemv 9.2 (sk2 10.3-10.8), o 5.2 (6.0-6.5), unpooler 3.5 (4.3-4.4): ~-3 us of GEMM rows; prform T1 bfp 90 / bfpf 90 / bfpf+bnd1 91 -> fused, prologue a wash (as its +3.9 us isolated cost predicted); T8 bfp 102 / bfpf 99. Judged d1 78.2/83.6: noise band (the MoE row moved 22 -> 28.5 us between runs on the same code) |

State: HEAD carries every lever self-tested and tuner-picked (cgemv/sgemv/rgemv kept where the fork's autotune
picks them; prologue, routing modes and pr form chosen on the chain per T). Judged band of the last 9 runs on
near-identical code: 2.076-2.098. The remaining layer time at T=1 is kernel-side: MoE 22-29 us (routing-dependent),
fused pooler+router 19, qkv cgemv 9.2 (floor 7.9), attention 4.7, three glue kernels ~13 us of rows (~8 exposed).
| 1d2eac51 | + MoE aa9a8a4 (TINY/TINY2 builds with their L2 hints on; the phase-B prefetch OKC_PBPF measured a loss in the MoE agent's DIAG, dormant) | **2.1055 (2.074-2.137)** | highest judged layer: MoE rows d1 full 28.4 -> 22.0, d16 75.8/60.0 -> 61.8/52.4, v1 sliding 65.4 -> 56.7; at T=1 the separate cgemv pooler 13.7 + rgemv router 3.7 beat the fused launch 19.1 on the chain (prform bfp 70 / bfpf 73); per shape 77.8/83.0, 131.5/209.7, 154.8/295.8, 109.1/119.1, 259.1/335.7, 326.9/493.2. HEAD a9da5cd = the file to score |
| b73aff83 | the standalone's retimed sk2 configs as the tokr 8/16 picks; the pooler+router form chain-timed per T to 32 | **2.1112 (2.080-2.143)** | highest judged; rows: unpooler16t_r72 18.2 us vs the old unpooler16h_r100 15.5 at d16 (reverted: the standalone's -0.6 does not transfer to this kernel), qkv16t 10.2 vs 10.3, o16t 6.2 vs 6.2, qkv8t 10.0 vs 10.1 (neutral); prform: fused at T <= 16 (T1 69 vs 71: a +-3 us coin flip between runs), separate at T=32 (214 vs 223); per shape 76.6/83.0, 136.8/208.5, 152.6/294.4, 114.5/116.5, 265.1/347.7, 329.8/493.7. Scored run of the 1d2eac51 file: 9f733a01 2.0754 (the MoE row is a +-10-20 us routing draw per run on half the shapes) |
| ea7cacb1 | unpooler-16 pick reverted + SK2_BALANCE (qkv) + MoE 548cbc2 (hints revert) | 2.1064 (2.075-2.138) | rows as predicted: unpooler d16 18.2 -> 15.3, qkv T8 10.0 -> 9.8 / T16 10.2 -> 10.0; the rest is the MoE draw (d32 sliding 72 -> 94, v16 175 -> 159). The last three runs (2.1055 / 2.1112 / 2.1064) sit above the earlier 2.076-2.098 band: the per-row wins (m = 1 GEMVs, configs, BALANCE) moved it ~1%. HEAD d73566d = the file to score (b73aff83's content + the certain unpooler fix + BALANCE) |

## 10. Ticket tails v2 (overnight, 2 checks): exact, chain-gated off; mechanism = single-CTA compute throughput

| job | ticket | isolated with/without tail (T=1) | chain btail[none/un/o/both] | speedup |
|---|---|---|---|---|
| a15e0698 | atom.add.release.gpu | o launch 9.0 -> 49.2 us (256 CTAs); unpooler slice 7.5 -> 21.6 (128) | 70/83/106/118 -> none | 2.1049 |
| a3816fbb | __threadfence + relaxed atomicAdd | o 8.8 -> 46.0; unpooler 7.2 -> 21.5 | 70/82/104/117 -> none | 2.0839 |

Numerics: s1 5.8e-8 / h 0 / s2 6.4e-8 / out 0 (rel) vs the Triton kernels, both runs. The ticket flavor is not the cost
(identical). The loads are in flight (zero spills, 154 / 103 registers = the whole row's loads + the row in registers).
The cost is the tail body on ONE CTA: the same per-element gate math as the Triton kernel (2 sigmoid, sqrt, 2 fp32
divisions, ~100+ instructions per element) over 8192 elements on 4 consumer warps (o kernel: 64 elements per thread)
or 8 warps (slice kernel: 32 per thread) vs the Triton kernel's 16 warps x 16 elements: a 4-8x longer serial
instruction stream with one warp per scheduler (~6.5 us x 2 and x 5 reproduce +14 / +37 us). Precomputed alpha/beta
would cut ~70% of the instructions but adds 16-32 loads per thread (spills at the slice kernel's 157-register cap) and
lands at ~2.5-3 us = the separate kernel's exposed cost: no win in the last-CTA structure at T=1. Left off
(FUSE_BOUNDARY = False); code, self-test and chain pick kept.

## 11. Chain-timed picks re-decided with statistics on a rented H200 (judge image), Oct 28

Instrument: `muse_decode_layer/ab_vm.py` (one prepare per seed; variants = runtime overrides of the tuner's picks, one
CUDA graph each, replayed round-robin after a 256 MB L2 read flush; medians + paired deltas) driven by
`muse_decode_layer/vm_ab/ab*.sh`. VM medians run 5-12% lighter than the judge and the MoE row is a per-seed routing draw,
so only fixed-seed paired comparisons count; every call below holds at seeds 0 and 1.

Baseline 8e96589 (VM, seed 0 / 1, us): d1s 72.0/61.0, d1f 86.3/80.8, d16s 131.4/121.5, d16f 189.5/195.3, d32s
146.4/157.0, d32f 301.6/322.8, v1s 105.9/105.6, v1f 112.0/105.2, v16s 249.0/255.7, v16f 327.1/350.6, v32s 314.2/301.6,
v32f 477.6/495.1. Prepare 245-304 s on the VM (builds cached): the 300 s budget is tight.

The tuner's single-sample chain timings flipped between prepares: pr form at T=1/16, o at T=16 (sk2 vs cuBLAS bf16),
PDL depth 2/3/4/6, rmode, and at T=128/256 qkv (cuBLAS fp32 / bf16 out), o, pr (Triton / cuBLAS / torchT), un.

ab1 (pr form x rmode, bnd1, PDL; 8 shapes, 12 variants):
- T=8: fused pooler+router + _route_split beats separate by 3.2/4.1 (s0) and 4.1/4.3 (s1) us.
- T=32: separate beats fused by 1.9/1.8 and 3.3/2.8 us. T=16: fused+split beats separate by 0.7/0.5 and 1.2/0.5.
- T=1: fused+split beats separate by ~1.8 us; the bnd1 prologue costs +0.8-1.3 on top. (Artifact seen: the FIRST
  captured variant of a run came out 2.6-3.7 us faster at T=1 only; hence the round-robin harness and ab5.)
- rmode sk2f / moe / moe_nb: +3-10 us at every T vs _route_split (the grid barrier / the MoE-side top-8 on every CTA).
- PDL depth 3 costs +1.5-2.2 us at T=16/32, +0.4-1.3 at T=8; 4 and 6 worse; nothing at T=1 (the m=1 GEMVs).

ab4 (T=128/256 dense routes, 5 per layer): qkv cuBLAS fp32-out wins by 1.2-3.5 us vs the other forms; un torchT best
(bfq/mm16 +0-2.4, Triton / cuBLAS fp32 +3-4 at T=256); o: mm16 / torchT / bfq tie (+-1); pr: Triton == cuBLAS, torchT
+2-4, the proj module's bfq route +12-14 us.

Locks (layer_glue): PDL_LOCK = 2 at every T; ROUTE_MODE_LOCK = split; PR_FORM_LOCK T8 fused / T32 separate (T1, T16
after ab5); PIN_PICK at 128/256: qkv True, o mm16/torchT, pr True, un torchT. FUSE_ROUTE / ROUTE_IN_MOE / FUSE_BND1 off
(code kept; their self-tests and the 11 chain timings per T no longer run in prepare).
