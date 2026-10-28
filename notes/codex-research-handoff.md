# Codex research handoff for Claude

Madhavan asked Codex to review the current work and search public internet/GitHub sources for useful kernel speedups, then asked to pass these findings to Claude. No rented GPU is available; prioritize information gained per slow profiling round.

This is research input for your existing plan, not a request to interrupt running jobs or replace the swarm's assignments. Check against work completed since the review. No kernel code was changed and no remote checks/submissions were run by Codex. The review covered root commit `3af46cd10cbe69b54841832b86f9c84416bcb493`, its historical profiles, and selected active-worktree logs. Only this handoff document was added afterward.

## Assessment

The current implementations already incorporate the vjeux kernels and substantial follow-up work: persistent execution, TMA, WGMMA, stream-K, lossless weight packing, specialized small batches, and PDL. Do not interpret comparisons against older `cuda_moe.py` or other original kernels in the design notes as current missing features.

Stored passing runs reviewed: attention 2.5883x (`26c40f28a3f74b2a`), MoE 2.3318x (`080c57e6047b4794`), BF16 projections 1.4933x (`8bd137632d7a4cc5`), decode layer 2.0762x (`711bbae5d04143b3`), prefill 1.7960x (`45ca53d9449b4d00`). These are historical observations, not claims that today's HEAD has been freshly verified.

The later sections of `notes/decode_layer_map.md` are essential: they correct earlier profile attribution and record why obvious fusion/dispatch ideas failed. Further gains likely require better decomposition, memory scheduling, or execution structure rather than generic fusion/prefetch suggestions.

## Priorities for the overall decode workload

### 1. A narrowly scoped verify GEMM decomposition experiment

Target raw BF16 M=128/256, initially QKV. Current `gq/gqx` already uses SS-WGMMA, TMA, and cluster split-K; those are not new proposals. The specific remaining comparison is work-centric Stream-K versus uniform cluster split-K, potentially with multicast across token tiles rather than clusters only dividing K.

The inspected CUTLASS path fixes swap-AB 128x64x64 Pingpong with a 1x1 cluster (`muse_bf16_proj/submission.py`, around lines 3237/3293). QKV produces only 36/72 output tiles at these M values, making underfilled waves relevant. Audit the previous 54-configuration sweep first; abandon this proposal if it already covered these scheduling choices. Compare against `gqx` as well as cuBLAS, not an untuned baseline.

Sources:
- https://github.com/Dao-AILab/quack/blob/main/quack/gemm_sm90.py
- https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp

`notes/decode_layer_map.md` section 7 establishes that backend/layout swaps were within roughly 1 us; the former cuBLAS discrepancy was misattributed. Its estimated 45-55 us combined verify dense roofline gap is an upper bound, NOT an attainable-gain forecast. No external result proves a win on these exact shapes.

### 2. Distribute paged-KV address calculations across lanes

Assign disjoint row addresses to lanes, then shuffle pointers into the coalesced loader groups. Current prefill prep repeats page/64-bit row-pointer work across four or eight adjacent lanes (`muse_prefill_layer/submission.py`, around 3529/3618); decode producers have similar duplication (`muse_attention_decode/submission.py`, around 1789).

This differs from the already-rejected experiment batching each thread's four page lookups (12.0 -> 13.4 us). Bulk page-index fetching and uniform-descriptor shuffles also do not remove all duplicated pointer arithmetic.

Source: https://github.com/Dao-AILab/flash-attention/pull/2104

That PR implements the strategy for SM100 following FA3. Its Blackwell speedups do NOT transfer to these tuned H200 kernels. The portable idea is lane ownership/shuffles, not Blackwell-specific instructions.

Cheap gates: CPU-enumerate exact address mapping for pages 1/64, tails, and request boundaries; offline compile; compare integer/address instructions, register pressure, and spills. Only spend a profiling slot if generated code supports the hypothesis. Expected scope is modest because decode q1 already approaches its random-gather ceiling and prefill staging is a small fraction of total time.

Negative evidence: https://github.com/flashinfer-ai/flashinfer/issues/1474 explains why small page sizes, especially 1, favor cp.async/LDGSTS over TMA. Existing repo experiments also show L2-prefetch losses of 6%/28% and q1 loads-only approximately 88.4 us / 3.7 TB/s. Do not reopen a blanket TMA conversion or farther-ahead prefetch sweep.

### 3. MoE: investigate weight-stream dependencies before more dequant tricks

The existing active-worktree log `.claude/worktrees/agent-ae73f8fb0fd253899/muse_moe_w4a16/profiles/submission.cf84242d11a84187.log` has useful attribution measurements:

| Deliberately incomplete diagnostic | bs32 us | bs128 us | bs256 us |
|---|---:|---:|---:|
| Main | 167.1 | 210.9 | 213.9 |
| Remove activation stream | 166.3 | 208.5 | 212.1 |
| Remove weight stream | 138.7 | 182.6 | 191.0 |
| Remove MMA | 158.4 | 203.7 | 204.3 |
| Remove dequantization | 163.9 | 207.3 | 209.9 |

These are invalid kernels used only for attribution, not shippable candidates. They suggest roughly 2% sensitivity to removing dequantization versus 11-17% to removing weight streaming. Effects overlap and are not additive or guaranteed optimization ceilings.

Look at weight-ring waits, producer issue latency, and consumer release timing. Reconcile the existing readiness diagnostics and SASS before another sweep. Per-warp MMA replacement, tiny dynamic claiming, and earlier WGS producers already have negative measurements in adjacent logs.

Useful implementation references, not drop-in replacements:
- https://github.com/NVIDIA/cutlass/blob/main/examples/55_hopper_mixed_dtype_gemm/README.md
- https://github.com/NVIDIA/cutlass/blob/main/examples/57_hopper_grouped_gemm/57_hopper_grouped_gemm.cu

### 4. Larger research branch: dependency-driven persistent layer tasks

Cohere's H100 implementation uses resident workers and explicit readiness counters rather than whole-grid barriers. A narrowly scoped Muse design could compute each norm row once, publish readiness, then let downstream projection tiles consume it. This structurally differs from both failed local approaches: a last CTA serially doing all glue, and every projection CTA recomputing the full norm.

Sources:
- https://github.com/cohere-ai/cohere-megakernel
- https://github.com/cohere-ai/cohere-megakernel/blob/main/src/decode/megakernel.cuh

This is substantial research, lower priority than a small GEMM experiment. Cohere targets another model and batches <=8; North Mini Code has parallel attention/MoE branches, whereas Muse has a sequential dependency chain. Their speedups do not transfer. The relevant opportunity is exposed dependency/tail work after PDL, not launch count multiplied by an assumed launch cost. Cohere's m64n8 WGMMA instruction also already exists locally.

## Prefill-only opportunities

Prefill has its own leaderboard and does not count toward the overall decode board. Score each idea with the load weighting: sliding shapes carry weight 46 for turn 1/4 and 138 for first turn, versus 16 for each full-attention shape. A large raw full-attention kernel is not automatically the best score opportunity.

1. **Separate full token tiles from remainder-width handling in the MoE mainloop.** A separately compiled large-tile path might avoid register pressure/serialization caused by many WGMMA widths sharing one function. Existing notes around `muse_prefill_layer/submission.py:1358,1452,1985,2002` record ring-overflow and WGMMA-serialization failures; use them as hard gates. Best initial scope is 16K tokens, where MoE takes around 3.4-3.75 ms and another launch is comparatively cheap. Preserve GEMM1-to-GEMM2 readiness. This is unmeasured.

2. **Separate K/V stage lifetimes in attention**, only if current ATTN_PROF diagnostics show meaningful stage waits. Current code keeps K+V in a combined stage until PV completes; FA3 separately releases K after QK. This changes storage ownership/barriers rather than merely rearranging the existing PIPE2/PIPE3 overlap. Four current stages and SFU limits may leave little benefit. Reference: https://github.com/Dao-AILab/flash-attention/blob/main/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp

Lower priority: reuse dequantized BF16 weight tiles across token tiles via shared memory/SS-WGMMA, but only after prefill-specific attribution justifies the extra shared-memory traffic and synchronization. Standalone decode's no-dequant measurements do not justify it.

## Avoid repeating these directions without new evidence

- Last-CTA norm/top-k tails, duplicated full-row norm prologues, and PDL depth 2 -> 6: already measured, with barriers/register pressure absorbing launch savings.
- Generic prefill LPT or reverse query-block scheduling: already present.
- Generic 11-bit lossless packing: coverage estimator and traffic-only sizing already exist; net savings were limited after escapes.
- Drop-in Blackwell MegaMoE, TCGen05, TMEM, or CLC: H200 is SM90, and these recipes are architecture-specific.
- Assuming published compression gains transfer: Cloudflare's Hopper Unweight reports significant throughput overhead, while ZipServ targets GDDR. Sources: https://blog.cloudflare.com/unweight-tensor-compression/ and https://github.com/HPMLL/ZipServ_ASPLOS26

## Profiling discipline with no rented GPU

- Before each candidate: name the bottleneck, causal change, affected shapes, prior attempts, and rough score-level ceiling.
- Use the existing CPU mapping/numerical checks and offline sm_90a compilation; reject spills or WGMMA serialization before remote checks. Keep the existing shared build lock so concurrent Docker builds do not OOM the host.
- Use small candidate sets, a known unchanged control, and the actual cold-L2 CUDA-graph chain. Source/compiler fingerprints and route distributions belong next to results.
- PDL kernel durations include overlapping waits and cannot simply be summed. Kernel rows help diagnose but are not sufficient to establish end-to-end improvement.
- Reconfirm promising small gains: the judge reports at least +/-1.5% intervals, while fresh MoE routing can move component time substantially between runs.

Main recommendation: prioritize one new verify GEMM decomposition, one cheap distributed-address experiment, and evidence-led MoE weight-path work. Preserve the mature defaults until valid measurements beat them.
