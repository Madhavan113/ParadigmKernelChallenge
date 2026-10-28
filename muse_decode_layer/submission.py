# okc: problem=muse_decode_layer
# Derived work. Base: https://github.com/vjeux/okc-2026 (muse_decode_layer.py), Copyright (c) 2026 vjeux,
# licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
# Modifications Copyright (c) 2026 Madhavan113, same license. See NOTICE in the repository root.
"""Muse Spark decode layer: one CUDA-graph-capturable kernel_fn composed from three kernel modules and this glue.

GENERATED FILE: tools/compose_decode_layer.py embeds the three component files whole (every module-level name of a
component prefixed att_ / moe_ / bfp_) into muse_decode_layer/layer_glue.py. Edit those sources, not this file.

Provenance (all derived from vjeux/okc-2026, Apache-2.0; see the headers of the component sections):
  * attention (prefix att_, from muse_attention_decode/submission.py): the fused decode kernel (QK-norm, RoPE, FP8
    cache write, split-KV attention, merge, per-head RMSNorm * sigmoid(gate)), driven by its own launcher, plan and
    workspace allocator (att__launch, att__plan, att__workspace, att__supported); this layer builds it with
    ATT_BUILD_FLAGS (the layer's measured variant of the same source) and guards its buffers;
  * experts (prefix moe_, from muse_moe_w4a16/submission.py): the NVFP4 MoE kernel in its builds (TINY M <= 4,
    TINY2 <= 8, MID <= 16, WGS main), through its own prepare (weight packing, PRMT-scale build check) and
    kernel_fn (build choice per token count); its scratch is guarded here;
  * dense projections (prefix bfp_, from muse_decode_layer/proj_fork.py): the layer's fork of the muse_bf16_proj
    module, incl. the sk2 PDL prologue and the fused pooler + router launch (sk2f);
  * this file's glue: residual/norm boundaries, routing (top-8 of biased sigmoid scores; router weights as
    BF16 hi + lo with FP32 accumulation; pre-expert RMSNorm), Triton split-K GEMMs, and SGLang's own FA3,
    Marlin and cuBLAS paths from the judge image.

Selection: prepare() times each backend per (token count, matrix), and the expert kernels per token count,
on the judge's own GPU with synthetic activations (it reads only weights, never data), then fixes the
choices; untuned token counts take the next tuned bucket's. Attention backend by shape (q_len, window, batch).

Precision: BF16 weights stay BF16 (or bf16_proj's lossless 12-bit encoding, exact escape correction);
NVFP4 experts stay NVFP4; norms apply (1 + w) formed in FP32 at use; FP8 queries and FP8 KV cache as
SGLang; softmax P in f16, or e4m3 where SGLang's FA3 rounds it to e4m3 too; FP32 accumulation throughout;
activations BF16 wherever SGLang keeps BF16.

Launch order: at T <= 32 tokens every kernel of the layer is launched with programmatic dependent launch
(PDL). Each one waits (griddepcontrol.wait) before its first read of data produced in this call; only
weights, and inputs that no kernel in this call writes, are loaded before the wait. Above 32 tokens all
launches are plain stream-ordered.

Safety: every scratch/workspace buffer this file allocates or caches for a component is followed by
sentinel words; prepare() checks them after the tuner and an attention self-test have run every path.
"""

import math

import torch
import triton
import triton.language as tl

HIDDEN = 8192
D_EXPERT = 4096
LOCAL_EXPERTS = 32
N_EXPERTS = 256
TOP_K = 8
QKV = 2304
ROUTER_MULT = 0.1767766952966369
LOG2E = 1.4426950408889634

_TUNE_LOG = True   # prepare() prints the tuner's candidate times and picks per token count (untimed; judge .log)
FORCE_PBMAX = None
# Programmatic Dependent Launch for this file's Triton glue: each kernel launches while its producer drains and
# waits (griddepcontrol.wait) right before its first read of producer data; weights and older inputs load before.
PDL = True
PDL_MAX_T = 32   # measured: a launch-latency win at decode sizes (d1 +2%), a ~1% loss at T = 256 (big grids)


def _pdl(T):
    return PDL and T <= PDL_MAX_T
# The moe agent's kernel wins at every token count we tune (their board is 1.76x); the tuner may only
# depart from it when another backend is clearly faster, so a noisy measurement cannot flip a pick.
DEFAULT_MOE = "agent"
TUNE_MARGIN = 0.95
# Dense picks the isolated tuner gets wrong part of the time (it cannot see PDL overlap in the chain), pinned to the
# winner measured in the whole layer (pickab.py, v132, 31 reps): o at T=32 bfp 166.7 vs cuBLAS 176.1 us (32x1 sliding);
# qkv at T=128 cuBLAS 249.4 vs split-K 251.0 (16x8 sliding); qkv at T=256 cuBLAS 351.1 vs split-K 353.5 (32x8 sliding).
# pr at T=32: separate bfp 166.8 / 318.7 vs fused bfpf 167.9 / 320.2 us (32x1 sliding / full). T=256 pr/un: 4-way tie.
# un at T=1/8 (the 1024-row slice): bfp 80.7 / 80.1 / 116.7 / 138.2 vs split-K 81.9 / 81.2 / 118.8 / 140.1 us (1x1 s/f, 1x8 s/f).
PIN_PICK = {(1, "un"): "bfp", (8, "un"): "bfp", (1, "o"): "bfp", (8, "o"): "bfp", (32, "o"): "bfp",
            (1, "pr"): ("bfpf", "bfp"), (8, "pr"): ("bfpf", "bfp"), (16, "pr"): ("bfpf", "bfp"), (32, "pr"): ("bfpf", "bfp"),
            (128, "qkv"): (True, "mm16"), (256, "qkv"): (True, "mm16")}   # (8, "o"): in-chain sk2 112.5/133.4 vs cuBLAS 114.7/136.3 us (Oct 26)
# pr at T <= 32: the bf16_proj kernels (fused launch incl. the fused routing tail, or separate + _route_split), by the
# tuner's times; never the Triton split-K / cuBLAS cat GEMM (check 19ea8e60: Triton pr picked by a biased timing cost
# 6-8 us per decode shape)
# a tuple pins the choice to the faster of those picks by the tuner's own times (here: cuBLAS fp32 out vs bf16 out,
# never the Triton split-K the isolated tuner prefers)
FORCE_MM = None
FORCE_GLUEV2 = None  # None: tuner's choice; True/False: force the parallel boundary/router kernels      # None: the tuner's per-(T, matrix) dense choice; else force one backend everywhere   # set to 0/1/2 to override the tuner's megakernel threshold (pod experiments)

# Fused decode attention (QK-norm, RoPE, FP8 cache write, split-KV FP8 attention, merge, rmsnorm * sigmoid(gate)):
# copied from our muse_attention_decode entry (same author), compiled as its own extension.

# ==============================================================================================================
# component: attn (muse_attention_decode/submission.py); every module-level name of that file carries the prefix `att_`
# ==============================================================================================================
# Derived work. Base: https://github.com/vjeux/okc-2026 (muse_attention_decode.py), Copyright (c) 2026 vjeux,
# licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
# Modifications Copyright (c) 2026 Madhavan113, same license. See NOTICE in the repository root.
"""Gated attention decode for Muse Spark Open: one fused CUDA kernel per call.

Each CTA takes one request, one pair of KV heads (its 16 query heads) and one contiguous range of
cached positions (split-KV). It normalizes the queries (and RoPE on sliding layers), builds the
block's new keys/values (normalized, RoPE, FP8) and writes them to the cache, streams its range of
the paged FP8 cache through shared memory (cp.async, 4 stages), computes QK^T with FP8 tensor-core
MMAs (queries rounded to E4M3, as FA3 does) and PV with FP16 MMAs (P in FP16, V exact in FP16), with
an online softmax. Splits are merged with FP32 atomics after rescaling to a fixed reference (QK-norm
bounds the logits); the last CTA of each (request, KV pair) normalizes, applies rmsnorm *
sigmoid(gate) and writes the output, then zeroes the workspace for the next call.
"""

import math
from typing import TypedDict

import torch


class att_Config(TypedDict):
    batch: int
    q_len: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    window: int
    rope_theta: float
    qk_scale_factor: float
    sm_scale: float
    kv_scale: float


class att_Inputs(TypedDict):
    qkv: torch.Tensor
    positions: torch.Tensor
    cache_loc: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    scheduler_metadata: torch.Tensor | None
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    config: att_Config


att_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda.h>
#include <stdint.h>

namespace {

#ifndef OKC_SM90
#define OKC_SM90 0
#endif
#ifndef OKC_NP
#define OKC_NP 128      // producer threads (q_len 1 flat; 128 since v111: f16 -0.7%, f32 -0.4% vs 256)
#endif
#ifndef OKC_WS
#define OKC_WS 1        // producer warpgroup for the flat (B > 1, full layer) kernel; 0 = v10's flat kernel
#endif
#ifndef OKC_WS8
#define OKC_WS8 1       // producer warps for the q_len 8 flat kernel too
#endif
#ifndef OKC_NP8
#define OKC_NP8 128     // q_len 8: 4 producer warps (its consumers need the registers)
#endif
#ifndef OKC_WSMINB
#define OKC_WSMINB 2
#endif
#ifndef OKC_WSMINB1
#define OKC_WSMINB1 OKC_WSMINB   // q_len 1 flat kernel: CTAs per SM the registers must allow (3: 396-CTA grids)
#endif
#ifndef OKC_SETMAXNREG
#define OKC_SETMAXNREG 1
#endif
#ifndef OKC_NOCOMPUTE
#define OKC_NOCOMPUTE 0
#endif
#ifndef OKC_PREGS
#define OKC_PREGS 0
#endif
#ifndef OKC_TMAPG
#define OKC_TMAPG 1     // producer mode: page indices by TMA (one copy per tile) instead of per-thread loads
#endif
#ifndef OKC_XHALFPV
#define OKC_XHALFPV 0   // (probe) q_len 1: half the PV work, to see whether the q1 flat loop is consumer-bound
#endif
#ifndef OKC_XNOWAIT
#define OKC_XNOWAIT 0   // (probe) per-request loop: no wait for the tile's copies (garbage), the compute-bound tile time
#endif
#ifndef OKC_XNOWAITF
#define OKC_XNOWAITF 0  // (probe) flat producers issue no K/V copies (stale data): the consumer-bound tile time
#endif
#ifndef OKC_CPHINT
#define OKC_CPHINT 0    // flat producer: K/V copies with an L2 evict_first cache hint (the rows are read once)
#endif
#ifndef OKC_EXH8F
#define OKC_EXH8F 0     // flat q8 (PK2 path): the softmax exponentials as f16x2 pairs (half the MUFU ops; the P bytes are e4m3
                        // anyway); the lazy-offset byte test, the split PV and the late sync are unchanged
#endif
#ifndef OKC_PFL2
#define OKC_PFL2 0      // flat q1 producer (page indices by TMA): prefetch.global.L2 of the K/V rows OKC_PFL2 tiles ahead of
                        // the copies (more rows in flight without more stages); 0 = off
#endif
#ifndef OKC_PIPE8
#define OKC_PIPE8 0     // flat q8: the next tile's stage wait + QK^T wgmma issued right after this tile's PV wgmma (both in
                        // flight across the loop edge: the QK^T latency and the PV end-of-tile wait leave the chain)
#endif
#ifndef OKC_FLATONLY
#define OKC_FLATONLY 0  // bench builds: only the flat kernels carry code (the per-request instances return at once)
#endif
#ifndef OKC_FENCEAR
#define OKC_FENCEAR 0   // merge / barrier fences as fence.acq_rel.gpu instead of fence.sc.gpu
#endif
#ifndef OKC_MRGDBG
#define OKC_MRGDBG 0    // (debug builds) phase marks around the merge's fence (columns precl1 / cl2)
#endif
#ifndef OKC_MERGE2
#define OKC_MERGE2 1    // last-CTA merges: one acq_rel atomic after the CTA barrier instead of per-thread fences
#endif
#ifndef OKC_MERGE3
#define OKC_MERGE3 1    // cooperative merges: one acq_rel barrier arrive after the CTA barrier instead of per-thread fences
#endif
#ifndef OKC_IDXFIRST
#define OKC_IDXFIRST 1  // per-request kernels: K/V issued once the page indices land (not after the q rows too)
#endif
#ifndef OKC_ONETILE
#define OKC_ONETILE 1   // per-request ONE builds: single fully cached tile -> straight-line prologue copies
#endif
#ifndef OKC_MAGICDIV
#define OKC_MAGICDIV 1  // per-request chunk: host magic multiplier instead of fdiv_i
#endif
#ifndef OKC_SC1
#define OKC_SC1 1       // coop finalize: sum_copies without the copy loop when nc == 1
#endif
#ifndef OKC_ONE
#define OKC_ONE 1       // per-request: builds for single-tile splits (MODE | 4), chosen by the host
#endif
#ifndef OKC_QDIRECT
#define OKC_QDIRECT 1   // per-request q8: the q pass writes the e4m3 wgmma A tile directly
#endif
#ifndef OKC_SPEC
#define OKC_SPEC 1      // one request, page size 1: speculate seq_len = page-table width - q_len (verified on device)
#endif
#ifndef OKC_PDLLATE
#define OKC_PDLLATE 0   // PDL builds (decode_layer): skip the early launch_dependents trigger at the tile-loop end
#endif
#ifndef OKC_QP16
#define OKC_QP16 1      // q_len 8 QDIRECT q pass: four threads per row (16 dims each)
#endif
#ifndef OKC_ONENEW
#define OKC_ONENEW 1    // ONE builds: the straight-line prologue copies also for the tile holding new rows
#endif
#ifndef OKC_SC8
#define OKC_SC8 1       // coop finalize: the eight workspace copies' loads in one round trip
#endif
#ifndef OKC_PINDIV
#define OKC_PINDIV 1    // coop finalize: the slice divisions pinned above the barrier wait
#endif
#ifndef OKC_ONEPRED
#define OKC_ONEPRED 1   // ONE builds: predicated (branch-free) straight-line prologue copies
#endif
#ifndef OKC_LATETID
#define OKC_LATETID 1   // per-request prologue: tid re-read (volatile) after the page-index copies, so the q-pass math stays late
#endif
#ifndef OKC_PFASTPR
#define OKC_PFASTPR 1   // per-request issue_kv: branch-free shift-add addressing on fully cached tiles (page size 1)
#endif
#ifndef OKC_PFAST
#define OKC_PFAST 1     // flat producer: branch-free shift-add addressing on common tiles (page size 1, < 4 GB)
#endif
#ifndef OKC_PGCONT
#define OKC_PGCONT 0    // (racecheck control) page-index TMA barriers once per launch, slots continued across segments
#endif
#ifndef OKC_TMAPG8
#define OKC_TMAPG8 0    // ... for the q_len 8 flat kernel too (measured ~+0.7%)
#endif
#ifndef OKC_ST8F
#define OKC_ST8F 4      // pipeline stages of the q_len 8 flat kernel
#endif
#ifndef OKC_COOP
#define OKC_COOP 1      // cooperative finalize for single-request grids (0: always the last-CTA finalize)
#endif
#ifndef OKC_PVW
#define OKC_PVW 0       // q_len 8 on sm_90: PV as an asynchronous wgmma from a double-buffered f16 V tile
#endif
#ifndef OKC_WGDEC
#define OKC_WGDEC 1     // q_len 8 producer mode: the two consumer warpgroups sync separately (one per head)
#endif
#ifndef OKC_SEGDBG
#define OKC_SEGDBG 0
#endif
#define DBGSTRIDE (OKC_SEGDBG ? 64 : 16)
#ifndef OKC_EXH2
#define OKC_EXH2 0      // q_len 8 on sm_90: softmax exponentials as packed f16 pairs (half the MUFU ops)
#endif
#ifndef OKC_PV8
#define OKC_PV8 1       // q_len 8 on sm_90: PV as an FP8 wgmma (P in e4m3, as SGLang's FA3 FP8 path does: pod/p16probe*.py)
#endif
#ifndef OKC_PROF
#define OKC_PROF 0      // per-warp phase profile of the main loops (diagnostic builds)
#endif
#ifndef OKC_PROFPV8
#define OKC_PROFPV8 0   // profile builds: keep the FP8 PV path
#endif
#define PV8ON (OKC_PV8 && (!OKC_PROF || OKC_PROFPV8) && !OKC_PVW)
#ifndef OKC_PV8FLAT
#define OKC_PV8FLAT 1   // FP8 PV in the flat (B > 1 full) kernel too (needs OKC_QSS: 104-register consumers)
#endif
#ifndef OKC_QSS
#define OKC_QSS 1       // with FP8 PV: q in shared memory (QK^T as an SS wgmma), 8 fewer live registers
#endif
#ifndef OKC_PV8SYNC
#define OKC_PV8SYNC 0   // wait for each PV wgmma inside its tile (fewer live registers)
#endif
#ifndef OKC_Q8RAW
#define OKC_Q8RAW 1     // q_len 8 sm_90: FP8 query from the unit-RMS query (scores scaled after QK^T)
#endif
#ifndef OKC_PROBE_O8
#define OKC_PROBE_O8 0
#endif
#ifndef OKC_P8EMU
#define OKC_P8EMU 0
#endif
#ifndef OKC_FLATCOOP1C
#define OKC_FLATCOOP1C 0  // compile the grid-wide flat finalize for q_len 1 too
#endif
#ifndef OKC_RAWTHR
#define OKC_RAWTHR 0    // q8 lazy max: compare raw scores with (offset + 8) / ssc (measured slower: +10 static fp32 instr)
#endif
#ifndef OKC_PCHK
#define OKC_PCHK 1      // q8: the lazy-max offset test on the packed e4m3 P (no per-tile row-max scan)
#endif
#ifndef OKC_TURN
#define OKC_TURN 0      // q8 flat: the heads' warpgroups alternate in the exponential phase (1: every tile, measured
                        // +1.3%; 2: offset once, on the first tile)
#endif
#ifndef OKC_CLSM
#define OKC_CLSM 0      // (experiment) minimum dynamic shared memory of cluster launches (forces one CTA per SM)
#endif
#ifndef OKC_PDLKV
#define OKC_PDLKV 0     // PDL launches: K/V issued before griddepcontrol.wait, q / new k,v as direct loads after it
                        // (decode_layer: build with 1; standalone it costs ~0.3% from code layout)
#endif
#ifndef OKC_BRFREE
#define OKC_BRFREE 0    // per-request K/V and page-index issue without per-row branches (predicated, zero-fill; neutral)
#endif
#ifndef OKC_RARE
#define OKC_RARE 1      // q8 flat: common tiles (no mask, no new rows) found once; one compare per tile
#endif
#ifndef OKC_PREUNR
#define OKC_PREUNR 1    // per-request prologue: the first STAGES-1 tiles' K/V issue unrolled (no loop branches; s1 -2%)
#endif
#ifndef OKC_FIN1
#define OKC_FIN1 0      // (experiment) coop finalize row loop kept rolled
#endif
#ifndef OKC_PCHKTEST
#define OKC_PCHKTEST 0  // test hook for OKC_PCHK (offset moves on most tiles)
#endif
#ifndef OKC_UNEV
#define OKC_UNEV 0      // per-request coop grids with more splits than SMs: uneven tiles by dispatch wave (w0; no gain
                        // on b1 q8 full: whole-tile granularity, 5 tiles per split, cannot shorten the slow wave)
#endif
#ifndef OKC_W0L1
#define OKC_W0L1 0      // per-SM uneven shares for the non-cooperative (q_len 1) flat grids too (measured +3..5 us: the
                        // generation-tagged registration costs more than the q_len 1 balance gains)
#endif
#ifndef OKC_SEGK8
#define OKC_SEGK8 600   // flat q_len 8: virtual keys per request boundary in the partition (b16 107.4 -> 103, b32 187.4 -> 184)
#endif
#ifndef OKC_SEGK1
#define OKC_SEGK1 400   // flat q_len 1: the same (b16 90.4 -> 88.7)
#endif
#ifndef OKC_W0SPREAD
#define OKC_W0SPREAD 0  // W0 registration barrier over 8 arrival counters (measured slower: 8 sequential polls)
#endif
#ifndef OKC_W0EARLY
#define OKC_W0EARLY 1   // the W0 registration atomics issued before the prefix sums (warp 1), the wait after them
#endif
#ifndef OKC_W0FORCEBAD
#define OKC_W0FORCEBAD 0  // test only: force the W0 registration's fallback (every CTA takes the even split)
#endif
#ifndef OKC_W0BID
#define OKC_W0BID 1     // rank the two CTAs of an SM by blockIdx (the lower one gets the warp schedulers' priority:
                        // it finishes first on every SM) instead of by arrival
#endif
#ifndef OKC_W0ENABLE
#define OKC_W0ENABLE 0  // flat coop (q_len 8): uneven shares for the two CTAs of each SM (see OKC_W0_L8)
#endif
#ifndef OKC_CLR2
#define OKC_CLR2 1      // cluster epilogue: compact (item, peer) DSMEM reduction with shuffles
#endif
#ifndef OKC_DBGON
#define OKC_DBGON 0     // phase timestamps (diagnostic builds): compiled out, the checks cost cold instructions
#endif
#ifndef OKC_BALPART
#define OKC_BALPART 0   // flat: balanced integer shares (one 32-bit division) instead of c*T/G
#endif
#ifndef OKC_CONT
#define OKC_CONT 0      // flat producer mode: one continuous stage pipeline across segments (no drain per segment)
#endif
#ifndef OKC_DYN
#define OKC_DYN 0       // flat producer mode: dynamic chunks claimed from a global counter (needs OKC_CONT)
#endif
#ifndef OKC_DYNK
#define OKC_DYNK 3      // chunks per CTA
#endif
#if OKC_DYN && !OKC_CONT
#error "OKC_DYN needs OKC_CONT"
#endif
#ifndef OKC_FINWARM
#define OKC_FINWARM 0   // dry call of the last-CTA finalize during the prologue (instruction cache warm-up)
#endif
#ifndef OKC_PVW1
#define OKC_PVW1 1      // flat q_len 8: wait for the previous PV wgmma before the V transpose (frees its registers)
#endif
#ifndef OKC_NOXCH1
#define OKC_NOXCH1 0    // q_len 1, workspace merge: no smem exchange of the two key halves (both reduce; measured slower)
#endif
#ifndef OKC_TANHSIG
#define OKC_TANHSIG 0   // gate sigmoid via tanh.approx (fewer cold instructions in the finalizes)
#endif
#ifndef OKC_PVSPLIT
#define OKC_PVSPLIT 1   // q8 PV: each half issued as soon as its P is packed; 1: waited at the tile end (no ptxas C7515
                        // wgmma serialization: nothing in flight across the loop edge), 2: at the next tile's top
#endif
#ifndef OKC_QKZ
#define OKC_QKZ 0       // q8 QK^T: first wgmma with write-only accumulators (not needed for C7515; QKZ=0 vf32 -0.4%)
#endif
#ifndef OKC_XP_NORESC
#define OKC_XP_NORESC 0 // timing/compile experiment only (wrong results): no O rescale on an offset move
#endif
#ifndef OKC_INV2
#define OKC_INV2 1      // per-tile invariants: hk / wsub / hi_all as lane-0 broadcasts, vld_off1 as vld_off0 ^ 32
#endif
#ifndef OKC_OZF
#define OKC_OZF 1       // q8: fence the O / row-sum zero init before the tile loop (scout: the C7515 trigger)
#endif
#ifndef OKC_XP_ARRALL
#define OKC_XP_ARRALL 1 // every consumer thread arrives on the stage-empty barriers (compute-sanitizer racecheck clean; the one-lane
                        // arrive after bar.sync / __syncwarp was flagged on every ring reuse; vf32 -0.8%)
#endif
#ifndef OKC_XP_MASKRO
#define OKC_XP_MASKRO 0 // compile experiment only
#endif
#ifndef OKC_XP_NOMASK
#define OKC_XP_NOMASK 0 // timing/compile experiment only (wrong results): no per-element masks
#endif
#ifndef OKC_IRANGE
#define OKC_IRANGE 0    // q8 loop: interior-tile test as one unsigned compare on the tile index (range found once; neutral)
#endif
#ifndef OKC_PATCHNI
#define OKC_PATCHNI 1   // q8 flat loop: the rare new-row patch as a noinline call (no hoisted bounds math per tile; -1.5%)
#endif
#ifndef OKC_DSHFL
#define OKC_DSHFL 1     // q8 flat loop: wgmma descriptor bases broadcast via shuffles (kept in uniform registers)
#endif
#ifndef OKC_PARITY
#define OKC_PARITY 1    // per-request coop barrier: per-launch alternating counters (no departure counter)
#endif
#ifndef OKC_SENSE2
#define OKC_SENSE2 0    // per-request coop barrier: sense-reversing with the flag read in the prologue (slower: the flag store delays the waiters)
#endif
#ifndef OKC_UNRN
#define OKC_UNRN 1      // q_len 8 loop unroll in the plain split-KV per-request kernel (b32 q8 sliding -0.5 us)
#endif
#ifndef OKC_ADDRN
#define OKC_ADDRN 1     // OKC_ADDR in the per-request kernels too (few tiles per CTA: the hoisting does not pay)
#endif
#ifndef OKC_OFF32
#define OKC_OFF32 1     // 32-bit index math for qkv / workspace / caches (host-checked sizes)
#endif
#ifndef OKC_UNR8
#define OKC_UNR8 2      // q_len 8 flat loop unroll (2: the V^T buffer / stage parity become constants per copy; -4.5%)
#endif
#ifndef OKC_RED4
#define OKC_RED4 1      // q_len 8 workspace merge with 16-byte atomics (lane-pair swaps)
#endif
#ifndef OKC_SWAPHALF
#define OKC_SWAPHALF 0
#endif
#ifndef OKC_LATESYNC
#define OKC_LATESYNC 1  // with OKC_PK2 (flat): the V^T proxy fence and warpgroup barrier move from after the transpose to
                        // just before the PV issue (each warp runs its exponentials without waiting for the others' V^T)
#endif
#ifndef OKC_LATESYNCN
#define OKC_LATESYNCN 1 // OKC_LATESYNC for the per-request q8 kernels (split PV): the V^T proxy fence and warpgroup barrier
                        // just before the first PV half instead of right after the transpose
#endif
#ifndef OKC_QKFIRST
#define OKC_QKFIRST 1   // per-request kernels (one tile stream): the tile's QK^T wgmma issued right after the stage barrier,
                        // before this thread's K/V copies for tile i + STAGES - 1 (they refill tile i-1's stage, which no
                        // wgmma of tile i reads), so the tensor core starts while the copy instructions issue
#endif
#ifndef OKC_FPART
#define OKC_FPART 1     // q_len 1 flat grids (<= 32 requests: ballots): warp 0 computes this CTA's key range and segment list
                        // right after its prefix sums (no all-thread binary search, no serial scan by thread 0, one barrier
                        // less: v288 vf16 spent 1.18 us between the prefix sums and the segment list)
#endif
#ifndef OKC_EARLYBAR
#define OKC_EARLYBAR 1  // flat producer mode: the first segment's stage barriers set up at kernel start (in the shadow of the
                        // seq_lens load) instead of after the partition (one init + CTA barrier off the critical path)
#endif
#ifndef OKC_PTPF
#define OKC_PTPF 1      // (with OKC_FPART) warp 0 prefetches the first segment's first STAGES tiles of page indices into L2
                        // as soon as the segment list is known (the producers' index loads then hit L2)
#endif
#ifndef OKC_FPRO
#define OKC_FPRO 0      // (debug) flat prologue stamps in dbg slots 40-44 (needs OKC_SEGDBG=1: stride 64)
#endif
#define FPRO_STAMP(k) do { if (OKC_FPRO && p.dbg) { long long* _dp = p.dbg + (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + (k); if (*_dp == 0) *_dp = gtime(); } } while (0)
#ifndef OKC_COOPGUARD
#define OKC_COOPGUARD 1 // host: a cooperative (grid-barrier) launch only when the instance actually launched (ONE / SPEC
                        // builds have their own register counts) holds the whole grid, else plain split-KV; device: the
                        // grid barriers trap after 1 s instead of spinning forever (never a hang on any data or node)
#endif
__device__ __noinline__ void spin_fail() { __trap(); }   // (out of line: keeps the trap off the polling loop's code)
#define SPIN_T0 unsigned _spin_n = 0   /* (an iteration count: a %globaltimer read per poll slowed the coop barriers ~1%) */
#define SPIN_CHECK do { if (OKC_COOPGUARD && ++_spin_n > (1u << 24)) spin_fail(); } while (0)   /* (>= 2^24 global polls: seconds) */
#ifndef OKC_PK2
#define OKC_PK2 1       // q8 FP8 PV with the packed-P offset test: both halves' exponentials as one straight-line block
                        // (32 independent FMA -> MUFU -> pack chains instead of two 16-chain passes split by a vote and
                        // a PV issue), one offset test for the tile, both PV halves issued together
#endif
#ifndef OKC_YIELD
#define OKC_YIELD 0     // flat kernels: ns the first-dispatched CTA's consumers sleep per tile (fairness experiment)
#endif
#ifndef OKC_ST1F
#define OKC_ST1F 3      // q_len 1 flat kernel: pipeline stages (3 measured best: b16 full 93 -> 91.5 us vs 6)
#endif
#ifndef OKC_STN
#define OKC_STN 3       // per-request kernels: pipeline stages
#endif
#ifndef OKC_PACK4
#define OKC_PACK4 0     // e4m3x4 packing in PTX (same SASS: ptxas already merges the halves with F2FP.MERGE_C)
#endif
#ifndef OKC_UARGS
#define OKC_UARGS 1
#endif
#ifndef OKC_UWARP
#define OKC_UWARP 1
#endif
#ifndef OKC_INEW
#define OKC_INEW 0      // q_len 8 flat loop: the new-row patch test as one compare (first such tile found per segment; measured slower)
#endif
#ifndef OKC_ADDR
#define OKC_ADDR 1      // q_len 8 loop: loop-carried stage / phase / descriptors / V^T buffer (fewer address ops per tile;
                        // slower before OKC_UWARP/UARGS (vector registers), -1% since (uniform registers))
#endif
#ifndef OKC_FASTSC
#define OKC_FASTSC 1    // RoPE: reduced angle + hardware sin / cos instead of the accurate sincosf
#endif
#ifndef OKC_STSM
#define OKC_STSM 0      // q_len 8 FP8 PV: V^T stores via stmatrix (2 per thread per tile instead of 8; measured ~1.5% slower)
#endif
#ifndef OKC_STSMPR
#define OKC_STSMPR 1    // ... in the per-request kernels (where it wins)
#endif
#ifndef OKC_TCHUNK
#define OKC_TCHUNK 1    // per-request splits: whole 64-key tiles per CTA
#endif
#ifndef OKC_EXPPOLY
#define OKC_EXPPOLY 0   // q_len 8 FP8 PV: how many of every 4 exponentials run on the FMA pipe (0..4; measured slower: issue-bound)
#endif
#ifndef OKC_EXH8
#define OKC_EXH8 0      // q_len 8 FP8 PV: exponentials as packed f16 pairs (experiment)
#endif
#ifndef OKC_LAZY1
#define OKC_LAZY1 0     // q_len 1: lazily moved softmax offset (no per-tile rescale or row-max shuffles; measured neutral)
#endif
#ifndef OKC_NOSYNC0
#define OKC_NOSYNC0 1   // flat q_len 8 FP8 PV: no warpgroup barrier at the start of each tile
#endif
#ifndef OKC_LAZYMAX
#define OKC_LAZYMAX 1   // q_len 8 sm_90: no per-tile row-max shuffles (only for the rare offset move)
#endif
#ifndef OKC_QROLL
#define OKC_QROLL 0     // prologue: the query passes as a loop instead of unrolled
#endif
#ifndef OKC_CLR4
#define OKC_CLR4 1      // q_len 8 cluster epilogue: DSMEM loads in groups of 4 peers (compact code)
#endif
#ifndef OKC_SENSE
#define OKC_SENSE 0     // coop finalizes: sense-reversing barriers (no departure counter round trip at the end)
#endif
#ifndef OKC_WSPLIT
#define OKC_WSPLIT 500  // flat: per mille of the keys for the first half of the grid (500 = even)
#endif
#ifndef OKC_F2
#define OKC_F2 0        // q_len 8 flat: one CTA per SM, two streams claiming tiles per KV head (measured slower: one producer warpgroup per SM limits)
#endif
#ifndef OKC_S2
#define OKC_S2 0        // q_len 8 per-request kernels: two tile streams per CTA, one CTA per SM (measured slower)
#endif
#ifndef OKC_PTR
#define OKC_PTR 0       // flat q_len 8: the producer warps build V^T (measured much slower: the producers stall on their copies)
#endif
#ifndef OKC_PTRD
#define OKC_PTRD 2      // OKC_PTR: tile y is transposed after the copies of tile y + OKC_PTRD are issued
#endif
#ifndef OKC_XP_NOTR
#define OKC_XP_NOTR 0      // timing experiments only (wrong results): skip the V transpose
#endif
#ifndef OKC_XP_NOFENCE
#define OKC_XP_NOFENCE 0   // timing experiments only: skip the proxy fence after the transpose
#endif
#ifndef OKC_XP_NOEXP
#define OKC_XP_NOEXP 0     // timing experiments only: P = scores (no exponentials)
#endif
#ifndef OKC_PROFPV8
#define OKC_PROFPV8 0   // profile builds: keep the FP8 PV path
#endif
#ifndef OKC_SMDBG
#define OKC_SMDBG 0     // diagnostic builds: record %smid of every CTA in debug slot 15
#endif
#ifndef OKC_CLDBG
#define OKC_CLDBG 0
#endif
#ifndef OKC_PROFDEP
#define OKC_PROFDEP 0
#endif
#ifndef OKC_SPINDBG
#define OKC_SPINDBG 0
#endif
template <int L, bool F>
struct Cfg {
  // q_len 8 per-request kernels on sm_90 with FP8 PV: two tile streams per CTA (warps 8..15 take the odd tiles), one
  // CTA per SM -- half the tile latency chain for the latency-bound per-request grids
  static constexpr bool S2 = (L == 8) && !F && OKC_S2 && OKC_SM90 && PV8ON;
  // q_len 8 flat kernel on sm_90: one CTA per SM with two consumer streams; each KV head's two warpgroups claim
  // tiles from a shared counter, so the faster one takes more (two CTAs per SM do not progress equally)
  static constexpr bool F2 = (L == 8) && F && OKC_F2 && OKC_SM90 && PV8ON && OKC_PV8FLAT && OKC_QSS && !OKC_CONT && !OKC_PTR;
  static constexpr int NT = (L == 1) ? 128 : ((S2 || F2) ? 512 : 256);   // consumer threads per CTA
  // flat kernel on sm_90: one extra producer warp issues every K/V copy (warp specialization), so the
  // consumers never stall in cp.async issue; the stages are handed over with mbarriers
  static constexpr bool WS = F && OKC_SM90 && OKC_WS && (L == 1 || OKC_WS8);
  static constexpr int NP = WS ? ((L == 1) ? OKC_NP : OKC_NP8) : 0;   // producer threads
  static constexpr int NTX = NT + NP;                     // threads per CTA
  static constexpr int MINB = WS ? (F2 ? 1 : (L == 1 ? OKC_WSMINB1 : OKC_WSMINB)) : (S2 ? 1 : 2);   // CTAs per SM the registers must allow
  // setmaxnreg split of the CTA's registers (launch: 65536 / (MINB * NTX), rounded down to 8)
  static constexpr int BASEREGS = (65536 / (MINB * NTX)) / 8 * 8 > 255 ? 248 : (65536 / (MINB * NTX)) / 8 * 8;
  static constexpr int PREGS = OKC_PREGS > 0 ? OKC_PREGS : ((L == 1 && NP <= 128) ? 56 : 32);
  static constexpr int CREGS = ((BASEREGS * NTX - PREGS * NP) / (NT > 0 ? NT : 1)) / 8 * 8 > 248
                                   ? 248 : ((BASEREGS * NTX - PREGS * NP) / (NT > 0 ? NT : 1)) / 8 * 8;
  // cp.async pipeline depth (64-key tiles): deep for the long flat runs, shallow (smaller shared memory,
  // more CTAs per SM, all clusters resident) for the per-request grids of sliding windows / one request
  static constexpr int STAGES = F ? ((L == 1) ? OKC_ST1F : (F2 ? 8 : OKC_ST8F)) : (S2 ? 6 : OKC_STN);
  // producer passes: NP/8 rows each must tile the 64-row tile exactly (else rows go missing and the consumers wait
  // forever on the stage barrier)
  static_assert(!WS || (NP % 8 == 0 && NP >= 8 && 64 % (NP >= 8 ? NP / 8 : 1) == 0),
                "producer threads must cover a tile in whole passes");
};
constexpr int TILE = 64;
constexpr int STAGES = 4;
constexpr int ROWB = 128;                    // bytes per tile row: 2 KV heads x 64
constexpr int STAGE_BYTES = TILE * ROWB;     // 8 KB per stage for K (same for V)
constexpr int IDX_SLOTS = 2 * STAGES;
constexpr int MAXB = 512;               // flat partition: requests per call
constexpr int MAXSEG = 64;              // flat partition: segments per CTA

struct Params {
  const __nv_bfloat16* qkv;
  const int* seq_lens;
  const int64_t* cache_loc;
  uint8_t* k_cache;
  uint8_t* v_cache;
  const int* page_table;
  const float* k_scale;
  const float* v_scale;
  const float* inv_freq;
  __nv_bfloat16* out;
  float* ws_acc;
  float* ws_l;
  int* ws_cnt;
  int B, Hq, Hkv, P, max_pages, window, rope, n_splits, flat, nc;
  long long* dbg;      // optional per-CTA phase timestamps (globaltimer), null in normal runs
  int cs, ncl;          // cluster mode: cs CTAs per cluster (DSMEM reduction), ncl clusters per request
  float* scratch;       // sm_80 emulation of DSMEM (local testing only)
  int* bar;             // sm_80 emulation of the cluster barrier
  int ld_qkv, row_bytes;
  float scale_log2;   // sm_scale * qk_scale_factor * log2(e)
  int tma;              // sm_90: K/V rows by TMA (one 128-byte swizzled row per copy) instead of cp.async
  int coop;             // per-request grids that are all resident: every CTA finalizes a slice of the rows
  int lp;               // log2(page size)
  int lrb;              // log2(row bytes of a cache row: Hkv * 64)
  int off32;            // OKC_OFF32: both caches below 4 GB (32-bit cache offsets); else the 64-bit path
  int w0;               // flat grids, 2 CTAs per SM: key share of the first CTA on each SM (1024 = even)
  int resident;         // the host checked: the grid is exactly 2 CTAs per SM and all of them are resident
  unsigned ns_magic;    // OKC_MAGICDIV: ceil(2^32 / n_splits) (0: n_splits == 1)
  int unev;             // OKC_UNEV: per-request coop grid of one request with more splits than SMs: the SM count
  int pdlf;             // launched with programmatic dependent launch (OKC_PDLKV prologue order)
  CUtensorMap tk, tv;   // [rows, row_bytes] views of the K / V caches, 128x1-byte boxes, 128B swizzle
  CUtensorMap tpt;      // producer mode: the page table as one int32 vector, 64-entry boxes (zero past the end)
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(src_bytes));
}
// (OKC_CPHINT) the same with an L2 cache policy (createpolicy)
__device__ __forceinline__ void cp_async16_hint(uint32_t dst, const void* src, int src_bytes, uint64_t pol) {
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;\n" ::"r"(dst), "l"(src), "r"(src_bytes), "l"(pol));
}
__device__ __forceinline__ uint64_t l2_evict_first_policy() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ void cp_async16_pred(uint32_t dst, const void* src, int src_bytes, bool pred) {
  // predicated copy (no branch): pred = 0 issues nothing, src_bytes = 0 zero-fills
  asm volatile("{\n .reg .pred q;\n setp.ne.b32 q, %3, 0;\n @q cp.async.cg.shared.global [%0], [%1], 16, %2;\n}\n"
               ::"r"(dst), "l"(src), "r"(src_bytes), "r"((int)pred));
}
__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_async4z(uint32_t dst, const void* src, int src_bytes) {   // (0: zero-fill)
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(dst), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

#ifndef OKC_TMA
#define OKC_TMA 0
#endif
// ---- sm_90 TMA row loads tracked by one mbarrier per pipeline stage ----
__device__ __forceinline__ void mbar_init(uint64_t* bar) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(smem_u32(bar)) : "memory");
#endif
}
__device__ __forceinline__ void mbar_fence_init() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, int bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
#endif
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, int parity) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  for (int spin = 0; spin < (1 << 16); ++spin) {   // bounded: a broken phase gives wrong results, never a hang
    uint32_t ok;
    asm volatile("{\n.reg .pred q;\nmbarrier.try_wait.parity.shared::cta.b64 q, [%1], %2;\nselp.u32 %0, 1, 0, q;\n}\n"
                 : "=r"(ok) : "r"(smem_u32(bar)), "r"(parity) : "memory");
    if (ok) break;
  }
#endif
}
__device__ __forceinline__ void mbar_init_n(uint64_t* bar, int count) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count) : "memory");
#endif
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(bar)) : "memory");
#endif
}
// unbounded wait for a phase (producer / consumer hand-off)
__device__ __forceinline__ void mbar_wait_full(uint64_t* bar, int parity) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("{\n.reg .pred q;\nWAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 q, [%0], %1;\n@!q bra WAIT_%=;\n}\n"
               ::"r"(smem_u32(bar)), "r"(parity) : "memory");
#endif
}
// the mbarrier receives one arrival when all of this thread's earlier cp.async copies have landed
__device__ __forceinline__ void cp_async_mbar_arrive(uint64_t* bar) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(smem_u32(bar)) : "memory");
#endif
}
// plain bulk copy global -> shared (16-byte aligned, multiple of 16 bytes), completion on an mbarrier
__device__ __forceinline__ void bulk_load(void* dst, const void* src, int bytes, uint64_t* bar) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
#endif
}
// one 64-entry box of a 1-D int32 tensor map into shared memory (TMA), completion on an mbarrier
__device__ __forceinline__ void tma_load_1d(void* dst, const CUtensorMap* map, int c0, uint64_t* bar) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2}], [%3];\n"
               ::"r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(smem_u32(bar)) : "memory");
#endif
}
// 16 zero bytes into shared memory through the same cp.async path (no global read)
__device__ __forceinline__ void cp_async16_zero(uint32_t dst, const void* any_src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 0;\n" ::"r"(dst), "l"(any_src));
}
// barrier over the NT consumer threads only (named barrier 1); __syncthreads when there is no producer
// barrier over one warpgroup (named barriers 3 and 4)
__device__ __forceinline__ void bar_sync_n(int id, int n) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive_n(int id, int n) { asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void wgsync(int wg) {
  asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
}
template <bool WS, int NT>
__device__ __forceinline__ void csync() {
  if constexpr (WS) asm volatile("bar.sync 1, %0;\n" ::"n"(NT) : "memory");
  else __syncthreads();
}
__device__ __forceinline__ void tma_row(void* dst, const CUtensorMap* map, int x, int y, uint64_t* bar) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
               ::"r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(map)), "r"(x), "r"(y), "r"(smem_u32(bar)) : "memory");
#endif
}

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}
// two e4m3 (HIGH 16 bits of x) -> f16x2: ptxas folds the half select into F2FP's operand (.H1), no SHF
__device__ __forceinline__ uint32_t cvt2h(uint32_t x) {
  uint32_t d;
  asm("{\n\t.reg .b16 t;\n\tmov.b32 {_, t}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, t;\n\t}" : "=r"(d) : "r"(x));
  return d;
}
// two e4m3 (low 16 bits of x) -> f16x2 (low byte -> low half)
__device__ __forceinline__ uint32_t cvt2(uint32_t x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  uint32_t d;
  asm("{\n\t.reg .b16 t;\n\tmov.b32 {t, _}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, t;\n\t}" : "=r"(d) : "r"(x));
  return d;
#else
  // exact bit trick (no NaN handling): e4m3 s.eeee.mmm -> f16 s.0eeee.mmm0000000, then x 2^8
  const uint32_t h = prmt(x, 0u, 0x7170);              // bytes -> [b0, 0, b1, 0]
  const uint32_t bits = ((h & 0x007f007fu) << 7) | ((h & 0x00800080u) << 8);
  __half2 v = *reinterpret_cast<const __half2*>(&bits);
  v = __hmul2(v, __floats2half2_rn(256.f, 256.f));
  return *reinterpret_cast<uint32_t*>(&v);
#endif
}
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}
// 2^lo, 2^hi as a packed f16 pair: the inputs rounded to f16 (|x| <= 8 here: error <= 2^-8), one MUFU op
__device__ __forceinline__ uint32_t ex2h2(float lo, float hi) {
#if OKC_P8EMU
  // accuracy experiment: P rounded to e4m3 (as an FP8 PV MMA would see it), then back to f16
  {
    uint16_t q8;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(q8) : "f"(ex2(lo)), "f"(ex2(hi)));
    uint32_t h;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"(q8));
    return h;
  }
#endif
#if OKC_EXH2
  uint32_t x = pack_h2(lo, hi), y;
  asm("ex2.approx.f16x2 %0, %1;" : "=r"(y) : "r"(x));
  return y;
#else
  return pack_h2(ex2(lo), ex2(hi));
#endif
}
// conformance sensitivity probe only (-DOKC_PROBE_O8=1): round the attention output, which SGLang keeps in
// BF16 before rmsnorm * gate, to e4m3 -- the precision check must reject that build
__device__ __forceinline__ float pr8(float x) {
#if OKC_PROBE_O8
  return __half2float(__nv_cvt_fp8_to_halfraw(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3), __NV_E4M3));
#else
  return x;
#endif
}
__device__ __forceinline__ uint8_t to_e4m3(float x) {
  return (uint8_t)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ void mma_f16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// m16n8k32 e4m3 x e4m3 -> f32 (sm_89+); emulated with two f16 MMAs elsewhere.
__device__ __forceinline__ void mma_fp8(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
  mma_f16(c, cvt2(a0), cvt2(a1), cvt2(a0 >> 16), cvt2(a1 >> 16), cvt2(b0), cvt2(b0 >> 16));
  mma_f16(c, cvt2(a2), cvt2(a3), cvt2(a2 >> 16), cvt2(a3 >> 16), cvt2(b1), cvt2(b1 >> 16));
#endif
}

__device__ __forceinline__ uint4 lds128(const uint8_t* p) { return *reinterpret_cast<const uint4*>(p); }
__device__ __forceinline__ uint2 lds64(const uint8_t* p) { return *reinterpret_cast<const uint2*>(p); }

// V B-fragments for 16 keys starting at kb (rows kb+2tig, +1, +8, +9), head hk, dims 8g..8g+7.
__device__ __forceinline__ void load_v_frags(const uint8_t* V, int kb, int hk, int g, int tig, uint32_t* b0,
                                             uint32_t* b1) {
  const int ch = hk * 4 + (g >> 1), off = (g & 1) * 8;
  const int k0 = kb + 2 * tig;
  const uint2 v0 = lds64(V + k0 * ROWB + ((ch ^ (k0 & 7)) << 4) + off);
  const uint2 v1 = lds64(V + (k0 + 1) * ROWB + ((ch ^ ((k0 + 1) & 7)) << 4) + off);
  const uint2 v2 = lds64(V + (k0 + 8) * ROWB + ((ch ^ ((k0 + 8) & 7)) << 4) + off);
  const uint2 v3 = lds64(V + (k0 + 9) * ROWB + ((ch ^ ((k0 + 9) & 7)) << 4) + off);
  uint32_t x;
  x = prmt(v0.x, v1.x, 0x5140); b0[0] = cvt2(x); b0[1] = cvt2h(x);
  x = prmt(v0.x, v1.x, 0x7362); b0[2] = cvt2(x); b0[3] = cvt2h(x);
  x = prmt(v0.y, v1.y, 0x5140); b0[4] = cvt2(x); b0[5] = cvt2h(x);
  x = prmt(v0.y, v1.y, 0x7362); b0[6] = cvt2(x); b0[7] = cvt2h(x);
  x = prmt(v2.x, v3.x, 0x5140); b1[0] = cvt2(x); b1[1] = cvt2h(x);
  x = prmt(v2.x, v3.x, 0x7362); b1[2] = cvt2(x); b1[3] = cvt2h(x);
  x = prmt(v2.y, v3.y, 0x5140); b1[4] = cvt2(x); b1[5] = cvt2h(x);
  x = prmt(v2.y, v3.y, 0x7362); b1[6] = cvt2(x); b1[7] = cvt2h(x);
}

// OKC_FENCEAR: the merge / barrier fences as fence.acq_rel.gpu (MEMBAR.ALL.GPU) instead of __threadfence()'s
// fence.sc.gpu (MEMBAR.SC.GPU + ERRBAR + CCTL.IVALL): the message-passing pattern needs release / acquire only
__device__ __forceinline__ void gpu_fence() {
#if OKC_FENCEAR
  asm volatile("fence.acq_rel.gpu;" ::: "memory");
#else
  __threadfence();
#endif
}
__device__ __forceinline__ void red_f32(float* a, float x) {
  asm volatile("red.global.add.f32 [%0], %1;" ::"l"(a), "f"(x) : "memory");
}
__device__ __forceinline__ void red_v4(float* a, float x, float y, float z, float w) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(a), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");
#else
  red_f32(a, x); red_f32(a + 1, y); red_f32(a + 2, z); red_f32(a + 3, w);
#endif
}

__device__ __forceinline__ long long gtime() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
#define DBG(k)                                                                                      \
  if (OKC_DBGON && p.dbg && threadIdx.x == 0)                                                       \
    p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + (k)] = gtime();

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
// four 8x8 b16 matrices from registers to shared memory (each lane gives one row address: lanes 8m..8m+7 the
// rows of matrix m; thread t holds row t/4, bytes 4(t%4)..4(t%4)+3 of each matrix)
__device__ __forceinline__ void stsm_x4(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\n" ::"r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3)
               : "memory");
#endif
}

// ---- per-warp phase profile of the main loops (diagnostic builds only: -DOKC_PROF=1) ----
#ifndef OKC_PROF
#define OKC_PROF 0
#endif
#if OKC_PROF
#define PROF_INIT long long _pt = clock64(); long long _pa[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#define PROF_MARK(k) do { const long long _t = clock64(); _pa[k] += _t - _pt; _pt = _t; } while (0)
#define PROF_TILE (_pa[7] += 1)
#define PROF_DUMP do { if (p.dbg && lane == 0) { \
    const int _w = (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * (blockDim.x / 32) + warp; \
    for (int _k = 0; _k < 8; ++_k) atomicAdd(reinterpret_cast<unsigned long long*>(p.dbg + 65536 + _w * 8 + _k), \
                                             (unsigned long long)_pa[_k]); } } while (0)
#else
#define PROF_INIT
#define PROF_MARK(k)
#define PROF_TILE
#define PROF_DUMP
#endif

// ---- sm_90 warpgroup MMA helpers ----
__device__ __forceinline__ void fence_proxy_async() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void wgmma_fence() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void wgmma_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#endif
}
template <int N>
__device__ __forceinline__ void wgmma_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
#endif
}
__device__ __forceinline__ void reg_fence(float& r) { asm volatile("" : "+f"(r)::"memory"); }
// K-major, 128-byte swizzled shared-memory matrix descriptor (8-row groups 1024 bytes apart)
__device__ __forceinline__ uint64_t desc_sw128(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
// D[64x64] (+)= A[64x32] (e4m3, registers) * B[64x32]^T (e4m3, shared memory descriptor)
__device__ __forceinline__ void wgmma_m64n64k32_e4m3(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                     uint64_t desc, int scale_d) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
      "{%32, %33, %34, %35}, %36, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(scale_d));
#endif
}
// D[64x64] (+)= A[64x32] (e4m3, shared memory descriptor) * B[64x32]^T (e4m3, shared memory descriptor)
__device__ __forceinline__ void wgmma_m64n64k32_e4m3_ss(float* d, uint64_t da, uint64_t db, int scale_d) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
      "%32, %33, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(da), "l"(db), "r"(scale_d));
#endif
}
// the same with scale-d = 0 and write-only accumulators: the previous tile's (masked) scores do not feed this
// wgmma, so ptxas sees no non-wgmma definition of its accumulators inside the pipeline (warning C7515, which
// serializes every wgmma of the function)
__device__ __forceinline__ void wgmma_m64n64k32_e4m3_ss_z(float* d, uint64_t da, uint64_t db) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
      "%32, %33, p, 1, 1;\n}\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]),
        "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]),
        "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]),
        "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31])
      : "l"(da), "l"(db), "r"(0));
#endif
}
// K-major operand with 64-byte rows, 64B swizzle (8-row atoms of 512 bytes)
__device__ __forceinline__ uint64_t desc_sw64(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(512 >> 4) << 32) | ((uint64_t)2 << 62);
}
// D[64x72] += A[64x32] (e4m3, registers) * B[72x32]^T (e4m3, K-major shared memory): d = 32 + 4 accumulators
__device__ __forceinline__ void wgmma_m64n72k32_e4m3(float* d, float* e, const uint32_t* a, uint64_t desc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35}, "
      "{%36, %37, %38, %39}, %40, 1, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
        "+f"(e[0]), "+f"(e[1]), "+f"(e[2]), "+f"(e[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
#endif
}
// floor(x / d) for 0 <= x < 2^24 and 1 <= d < 2^24 through the float reciprocal (one correction step):
// ~8 instructions instead of the ~20 of an integer division, which matters in once-per-launch (cold) code
__device__ __forceinline__ int fdiv_i(int x, int d) {
  // approximate reciprocal (MUFU.RCP, no IEEE slow-path call: the correction below absorbs its error)
  const float rd = __fdividef(1.f, __int2float_rn(d));
  int q = __float2int_rz(__int2float_rn(x) * rd);
  const int r = x - q * d;
  q += (r >= d) - (r < 0);
  return q;
}
// 2^x for four floats -> four e4m3 bytes, through packed f16 exponentials (2 MUFU ops instead of 4). The f16
// exponent result (10-bit mantissa) is rounded to e4m3 (3-bit mantissa) right after, like the f32 path's result.
__device__ __forceinline__ uint32_t ex2_e4m3x4_h(float x0, float x1, float x2, float x3) {
  uint32_t a, b, ea, eb;
  asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(a) : "f"(x0), "f"(x1));
  asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(b) : "f"(x2), "f"(x3));
  asm("ex2.approx.f16x2 %0, %1;" : "=r"(ea) : "r"(a));
  asm("ex2.approx.f16x2 %0, %1;" : "=r"(eb) : "r"(b));
  uint16_t lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo) : "r"(ea));
  asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi) : "r"(eb));
  return (uint32_t)lo | ((uint32_t)hi << 16);
}
// (OKC_EXH8F) four exponentials packed to e4m3: f16x2 pairs or fp32 (the PK2 pack)
#if OKC_EXH8F
#define PKX(a, b, c, d) ex2_e4m3x4_h(a, b, c, d)
#else
#define PKX(a, b, c, d) e4m3x4f(ex2(a), ex2(b), ex2(c), ex2(d))
#endif
// sigmoid for the output gate: 0.5 + 0.5 tanh(x / 2) with the hardware tanh (3 instructions; abs error ~2^-12,
// below the BF16 output's rounding) or exp + fast reciprocal
__device__ __forceinline__ float sigm(float x) {
#if OKC_TANHSIG
  float t;
  asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * x));
  return fmaf(0.5f, t, 0.5f);
#else
  return __fdividef(1.f, 1.f + __expf(-x));
#endif
}
// 2^x on the FMA pipe (FlashAttention-4 style, to balance the MUFU unit): x clamped to >= -127, x = j + f with
// j = floor(x), 2^f by a degree-3 minimax polynomial on [0, 1) (rel. error ~9e-5, far below the e4m3 rounding of P),
// j added into the exponent bits
__device__ __forceinline__ float ex2_poly(float x) {
  x = fmaxf(x, -127.f);
  const float j = floorf(x);
  const float f = x - j;
  float p = fmaf(f, 0.0790209f, 0.2249424f);
  p = fmaf(f, p, 0.6960656f);
  p = fmaf(f, p, 0.9999893f);
  return __int_as_float(__float_as_int(p) + ((int)j << 23));
}
// Programmatic dependent launch (sm_90): wait for the preceding grid (the one writing qkv) before the first qkv read;
// let the next grid start its own prologue once the main loop is done. Both are no-ops for a plain launch.
__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_launch_dependents() {
  // (OKC_PDLLATE: no early trigger at the end of the tile loop -- the dependent kernel launches when this grid
  // exits; for decode_layer's chain A/B, no effect on standalone launches)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !OKC_PDLLATE
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}
// OKC_OFF32: 32-bit offsets for qkv, the workspace and the caches (the host checks the sizes): no 64-bit multiplies
// in the once-per-launch address math
#if OKC_OFF32
typedef int ix_t;
#else
typedef int64_t ix_t;
#endif
__device__ __forceinline__ const uint8_t* cache_row(const uint8_t* base, const Params& p, int64_t row, int col_bytes) {
#if OKC_OFF32
  if (p.off32) return base + (((uint32_t)row << p.lrb) + (uint32_t)col_bytes);
#endif
  return base + row * p.row_bytes + col_bytes;
}
// `for (e = tid; e < N; e += NTH)` with a compile-time trip count: fully unrolled, the tail predicated (the
// once-executed prologue loops otherwise keep 3-4 branches each; a taken branch costs 60-100 cycles cold)
template <int N, int NTH, class F>
__device__ __forceinline__ void tfor(int tid, F&& f) {
#pragma unroll
  for (int e0 = 0; e0 < N; e0 += NTH) {
    const int e = e0 + tid;
    if ((N % NTH) == 0 || e < N) f(e);
  }
}
// four floats -> four e4m3 bytes (x0 in the low byte), saturating
__device__ __forceinline__ uint32_t e4m3x4f(float x0, float x1, float x2, float x3) {
#if OKC_PACK4
  // both halves packed in PTX (mov.b32 {lo, hi}): ptxas can place the two conversions in one register's halves
  // instead of a shift + or
  uint32_t r;
  asm("{\n.reg .b16 l, h;\n"
      "cvt.rn.satfinite.e4m3x2.f32 l, %2, %1;\n"
      "cvt.rn.satfinite.e4m3x2.f32 h, %4, %3;\n"
      "mov.b32 %0, {l, h};\n}"
      : "=r"(r) : "f"(x0), "f"(x1), "f"(x2), "f"(x3));
  return r;
#else
  uint16_t lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(lo) : "f"(x0), "f"(x1));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(hi) : "f"(x2), "f"(x3));
  return (uint32_t)lo | ((uint32_t)hi << 16);
#endif
}
// D[64x64] (+)= A[64x16] (f16, registers) * B[16x64] (f16, shared memory, MN-major 128B swizzle: tnspB = 1)
__device__ __forceinline__ void wgmma_m64n64k16_f16_tb(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                       uint64_t desc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
      "{%32, %33, %34, %35}, %36, 1, 1, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
#endif
}
// four f16 (two f16x2 words, low first) -> four e4m3 bytes (low byte first)
__device__ __forceinline__ uint32_t pack_e4m3x4(uint32_t h01, uint32_t h23) {
  __half2_raw a, b;
  a.x = (unsigned short)(h01 & 0xffffu); a.y = (unsigned short)(h01 >> 16);
  b.x = (unsigned short)(h23 & 0xffffu); b.y = (unsigned short)(h23 >> 16);
  const uint32_t lo = __nv_cvt_halfraw2_to_fp8x2(a, __NV_SATFINITE, __NV_E4M3);
  const uint32_t hi = __nv_cvt_halfraw2_to_fp8x2(b, __NV_SATFINITE, __NV_E4M3);
  return (lo & 0xffffu) | (hi << 16);
}

__device__ __forceinline__ void red_v2(float* a, float x, float y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" ::"l"(a), "f"(x), "f"(y) : "memory");
#else
  red_f32(a, x); red_f32(a + 1, y);
#endif
}

__device__ __forceinline__ float quad_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
  return v;
}
__device__ __forceinline__ float quad_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  return v;
}
__device__ __forceinline__ float oct_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  return v;
}

// ---- thread-block clusters (sm_90: DSMEM; elsewhere emulated through global memory for testing) ----
__device__ __forceinline__ int cl_rank(const Params& p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return (int)r;
#else
  return blockIdx.x % p.cs;
#endif
}
__device__ __forceinline__ int cl_gid(const Params& p) {
  return (blockIdx.z * gridDim.y + blockIdx.y) * (gridDim.x / p.cs) + blockIdx.x / p.cs;
}
__device__ __forceinline__ void cl_sync(const Params& p, int phase) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
#else
  __syncthreads();
  if (threadIdx.x == 0) {
    int* bar = p.bar + cl_gid(p);
    __threadfence();
    atomicAdd(bar, 1);
    const int target = phase * p.cs;
    SPIN_T0;
    while (true) {
      int v;
      asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(bar));
      if (v >= target) break;
      SPIN_CHECK;
    }
    __threadfence();
  }
  __syncthreads();
#endif
}
__device__ __forceinline__ void cl_done(const Params& p) {
#if !(defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
  if (threadIdx.x == 0) {
    int* bar = p.bar + cl_gid(p);
    if (atomicAdd(bar, 1) == 3 * p.cs - 1) *bar = 0;
  }
#endif
}
// float4 at byte offset `off` of peer `rank`'s partial buffer (`local` = this CTA's copy)
__device__ __forceinline__ float4 cl_ld4(const Params& p, const float* local, int rank, int off_floats, int region_floats) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t a = smem_u32(local + off_floats), r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
  float4 v;
  asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(r));
  return v;
#else
  const float* g = p.scratch + ((int64_t)cl_gid(p) * p.cs + rank) * region_floats + off_floats;
  return __ldcg(reinterpret_cast<const float4*>(g));
#endif
}
__device__ __forceinline__ float cl_ld1(const Params& p, const float* local, int rank, int off_floats, int region_floats) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t a = smem_u32(local + off_floats), r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
  float v;
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(r));
  return v;
#else
  return __ldcg(p.scratch + ((int64_t)cl_gid(p) * p.cs + rank) * region_floats + off_floats);
#endif
}

template <int L, bool F>
struct Smem {
  uint8_t k[Cfg<L, F>::STAGES * STAGE_BYTES];
  uint8_t v[Cfg<L, F>::STAGES * STAGE_BYTES];
  uint64_t mbar[8];   // TMA mode: one per stage (64 bytes keeps the members after it 16-byte aligned)
  uint64_t pbar[4];   // producer mode: page-index slots landed (TMA)
  int pgc;            // OKC_PGCONT: page-index fills issued in this CTA's earlier segments
  int dbgn;           // debug builds: progress-mark count
  int vidx;           // flat weighted partition: this CTA's slot (2 * SM + arrival rank), -1 = uniform
  alignas(128) int pidx[4][80];   // producer mode: page indices of upcoming tiles (16-byte aligned window)
  uint64_t full[8];   // producer mode: stage landed (32 producer lanes arrive as their copies complete)
  uint64_t empty[8];  // producer mode: stage consumed (one arrival per consumer warp)
  uint64_t vtf[2];    // OKC_PTR: V^T slot written by the producers (one arrival per producer thread)
  uint64_t vte[2];    // OKC_PTR: V^T slot read by both heads' PV wgmmas (one arrival per consumer warp)
  uint64_t qdone;     // OKC_PTR: consumers have read the q staging (S.kvq) and staged S.kn / S.vn
  int claim[2];       // Cfg::F2: next unclaimed tile per KV head
  int nxt[4][2];      // Cfg::F2: each warpgroup's next claimed tile (double buffered)
  int idx[2 * Cfg<L, F>::STAGES * TILE];
  // raw q rows of the block (bf16, prologue only) | L==8: the current tile's K and V in f16 (2 x 2 x 64 x 64)
#if OKC_SM90
  // L == 8: q staging (producer mode) before the loop, then the per-tile f16 V copy -- or, FP8 PV, per tile
  // (double buffer) and KV head, V^T as 72 rows (64 dims + 8 rows of ones for the row sums) x 64 keys (permuted to
  // the P register order), K-major with 64B swizzle: 4 x 4608 bytes
  alignas(1024) uint8_t kvq[(L == 1) ? L * 16 * 64 * 2 : (OKC_PVW ? 32768 : (PV8ON ? ((Cfg<L, F>::S2 || Cfg<L, F>::F2) ? 36864 : 18432) : 16384))];
  // L == 8 with FP8 PV and OKC_QSS: the block's e4m3 queries per KV head (64 rows x 64 dims, K-major, 64B swizzle)
  alignas(1024) uint8_t q8s[(L == 8 && PV8ON && OKC_QSS && (!F || OKC_PV8FLAT)) ? 8192 : 16];
#else
  uint8_t kvq[(L == 1) ? L * 16 * 64 * 2 : 32768];
#endif
  __device__ __forceinline__ __nv_bfloat16* qst() { return reinterpret_cast<__nv_bfloat16*>(kvq); }
  __nv_bfloat16 kvst[L * 4 * 64];   // raw k (2 heads) then v (2 heads) rows per token
  __nv_bfloat16 gst[L * 4 * 64];    // cluster mode: gate rows of this CTA's output slice (<= L*16/4 rows)
  uint8_t kn[L * ROWB];        // new keys of the block, [j][2 heads x 64] e4m3
  uint8_t vn[L * ROWB];
  float cs[L * 32];
  float sn[L * 32];
  float ifreq[32];          // RoPE inverse frequencies (staged with cp.async)
  float scl[4];             // k_scale, v_scale
  long long cloc[L];        // cache rows of the block's tokens
  int flag;
  int sense;          // OKC_SENSE2: the cooperative barrier's flag, read early (prologue) with the constants
  int cum[F ? MAXB + 1 : 1];
  int seg[F ? MAXSEG : 1][3];
  int nseg;
  int fcs, fce;             // OKC_FPART: this CTA's key range (warp 0 -> everyone)
  // flat dynamic chunks (OKC_DYN): the producer's segment list of its current chunk, and a queue that hands the
  // segments (request, first key, end key, copy; request -1 = done) to the consumers
  int pseg[F ? MAXSEG : 1][3];
  int pnseg, pchunk;
  int dq[F ? 16 : 1][4];
  uint64_t dqf[F ? 16 : 1];
  uint64_t dqe[F ? 16 : 1];
};

// Cluster mode, several clusters per request (Params::ncl > 1): add this CTA's reduced slice into the global
// workspace; the last cluster to arrive sums the copies back into RB/RL and finalizes. Out of line: no scored shape
// uses it, and keeping it out of the one-cluster epilogue keeps that code contiguous (cold instruction fetch).
template <int L, typename SM>
__device__ __noinline__ bool cluster_multi(const Params& p, SM& S, const int CS, const int rank, const int RPC,
                                           const int row0, const int bp, float* RB, float* RL) {
  constexpr int NT = Cfg<L, false>::NT;
  const int tid = threadIdx.x;
  bool fin = true;
    const int clu = fdiv_i(blockIdx.x, CS);
    const int copy = clu - fdiv_i(clu, p.nc) * p.nc;
    for (int e = tid; e < RPC * 16; e += NT) {
      const int row = row0 + (e >> 4);
      float* acc = p.ws_acc + ((ix_t)(bp * p.nc + copy) * L * 16 + row) * 64 + (e & 15) * 4;
      const float4 a = *reinterpret_cast<const float4*>(&RB[e * 4]);
      red_v4(acc, a.x, a.y, a.z, a.w);
    }
    for (int e = tid; e < RPC; e += NT) red_f32(p.ws_l + (bp * p.nc + copy) * L * 16 + row0 + e, RL[e]);
    __threadfence();
    __syncthreads();
    if (tid == 0) {
      // every cluster of the request arrives (empty ones with zero partials)
      const int old = atomicAdd(p.ws_cnt + bp * CS + rank, 1);
      S.flag = (old == p.ncl - 1);
    }
    __syncthreads();
    fin = S.flag;
    if (fin) {
      __threadfence();
      for (int e = tid; e < RPC * 16; e += NT) {
        const int row = row0 + (e >> 4);
        constexpr int NCM = 4;
        float4 v[NCM];
#pragma unroll
        for (int c = 0; c < NCM; ++c)
          if (c < p.nc) v[c] = __ldcg(reinterpret_cast<const float4*>(p.ws_acc + ((ix_t)(bp * p.nc + c) * L * 16 + row) * 64 + (e & 15) * 4));
        float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int c = 0; c < NCM; ++c) {
          if (c < p.nc) {
            a.x += v[c].x; a.y += v[c].y; a.z += v[c].z; a.w += v[c].w;
            *reinterpret_cast<float4*>(p.ws_acc + ((ix_t)(bp * p.nc + c) * L * 16 + row) * 64 + (e & 15) * 4) = make_float4(0.f, 0.f, 0.f, 0.f);
          }
        }
        *reinterpret_cast<float4*>(&RB[e * 4]) = a;
      }
      for (int e = tid; e < RPC; e += NT) {
        float a = 0.f;
        for (int c = 0; c < p.nc; ++c) {
          float* lw = p.ws_l + (bp * p.nc + c) * L * 16 + row0 + e;
          a += __ldcg(lw);
          *lw = 0.f;
        }
        RL[e] = a;
      }
      if (tid == 0) p.ws_cnt[bp * CS + rank] = 0;
      __syncthreads();
    }
  return fin;
}

// Cluster mode epilogue. Every CTA of the cluster has written its rescaled partial (PB: [L*16][64] o * 2^m,
// PL: [L*16] l * 2^m) into its own shared memory. CTA `rank` sums rows [rank*RPC, (rank+1)*RPC) over the
// cluster through DSMEM, then either finalizes them (one cluster per request) or accumulates them into the
// global workspace, where the last cluster to arrive for this slice finalizes.
template <int L, typename SM>
__device__ __forceinline__ void cluster_finish(const Params& p, SM& S, const int b, const int pair,
                                               const int n_keys, const int chunk) {
  constexpr int NT = Cfg<L, false>::NT;
  constexpr int STAGES = Cfg<L, false>::STAGES;
  constexpr int IDX_SLOTS = 2 * STAGES;
  const int tid = threadIdx.x;
  const int CS = p.cs, rank = cl_rank(p);
  const int RPC = fdiv_i(L * 16, CS);
  const int pairs = p.Hkv >> 1;
  const int bp = b * pairs + pair;
  // S.k and S.v are adjacent: PB [L*16*64] | PL [L*16, padded to 1024] | RB [RPC*64] | RL [RPC]
  float* PB = reinterpret_cast<float*>(S.k);
  float* PL = PB + L * 16 * 64;
  float* RB = PL + 1024;
  float* RL = RB + RPC * 64;
  constexpr int REGION = L * 16 * 64 + 1024;   // emulation: per-CTA scratch floats (PB, then PL at +L*16*64)
#if !(defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
  {  // publish this CTA's partial for the emulated DSMEM reads
    float* g = p.scratch + ((int64_t)cl_gid(p) * CS + rank) * REGION;
    for (int e = tid; e < L * 16 * 64; e += NT) g[e] = PB[e];
    for (int e = tid; e < L * 16; e += NT) g[L * 16 * 64 + e] = PL[e];
    __threadfence();
  }
#endif
  if (OKC_CLDBG) DBG(7);
  cl_sync(p, 1);
  DBG(4);
  long long dbgc0 = 0;
  if (OKC_CLDBG) asm volatile("mov.u64 %0, %%clock64;" : "=l"(dbgc0));
  const int row0 = rank * RPC;
#if OKC_CLR2
  if constexpr (L == 1) {   // compact reduction (this code runs once per launch from a cold instruction cache, so it is kept short):
      // (row, slot) items -- 16 float4 slots of o and one for l -- times the CS peers; CS consecutive lanes load
      // one item from every peer and sum it with shuffles. Two items per thread per pass keep loads in flight.
    const int TOT0 = RPC * 17 * CS;
    const int lcs = 31 - __clz(CS);
#pragma unroll 1
    for (int rep2 = (OKC_CLDBG == 2 ? 0 : 1); rep2 < 2; ++rep2) {
    int TOT = TOT0;
    if (rep2 == 0) asm volatile("mov.u32 %0, 0;" : "=r"(TOT));   // dry pass: same code, no work
    if (OKC_CLDBG == 2 && rep2 == 1) asm volatile("mov.u64 %0, %%clock64;" : "=l"(dbgc0));
    for (int k0 = 0; k0 < (rep2 == 0 ? 2 * NT : TOT); k0 += 2 * NT) {   // warp-uniform trip count
      float4 v[2];
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int k = k0 + u * NT + tid;
        const int item = k >> lcs, q = k & (CS - 1);
        const int rl = item / 17, slot = item - rl * 17;
        v[u] = make_float4(0.f, 0.f, 0.f, 0.f);
        if (k < TOT) {
          if (slot < 16) v[u] = cl_ld4(p, PB, q, (row0 + rl) * 64 + slot * 4, REGION);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
          else v[u].x = cl_ld1(p, PL, q, row0 + rl, REGION);
#else
          else v[u].x = cl_ld1(p, PL, q, L * 16 * 64 + row0 + rl, REGION);
#endif
        }
      }
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        for (int off = 1; off < CS; off <<= 1) {
          v[u].x += __shfl_xor_sync(0xffffffffu, v[u].x, off);
          v[u].y += __shfl_xor_sync(0xffffffffu, v[u].y, off);
          v[u].z += __shfl_xor_sync(0xffffffffu, v[u].z, off);
          v[u].w += __shfl_xor_sync(0xffffffffu, v[u].w, off);
        }
        const int k = k0 + u * NT + tid;
        const int item = k >> lcs, q = k & (CS - 1);
        const int rl = item / 17, slot = item - rl * 17;
        if (k < TOT && q == 0) {
          if (slot < 16) *reinterpret_cast<float4*>(&RB[(rl * 16 + slot) * 4]) = v[u];
          else RL[rl] = v[u].x;
        }
      }
    }
    }
  } else
#endif
  {
  constexpr int CSMAX = 16;
#if OKC_CLR4
  // compact: peers in groups of 4 (one loop body, warm after its first pass: this code runs once per launch)
  for (int e = tid; e < RPC * 16; e += NT) {
    const int off = (row0 + (e >> 4)) * 64 + (e & 15) * 4;
    const bool lrow = (e & 15) == 0;
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
    float la = 0.f;
#pragma unroll 1
    for (int q0 = 0; q0 < CS; q0 += 4) {
      float4 v[4];
      float lv[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        v[u] = cl_ld4(p, PB, q0 + u, off, REGION);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        lv[u] = lrow ? cl_ld1(p, PL, q0 + u, row0 + (e >> 4), REGION) : 0.f;
#else
        lv[u] = lrow ? cl_ld1(p, PL, q0 + u, L * 16 * 64 + row0 + (e >> 4), REGION) : 0.f;
#endif
      }
#pragma unroll
      for (int u = 0; u < 4; ++u) { a.x += v[u].x; a.y += v[u].y; a.z += v[u].z; a.w += v[u].w; la += lv[u]; }
    }
    *reinterpret_cast<float4*>(&RB[e * 4]) = a;
    if (lrow) RL[e >> 4] = la;
  }
  if (false)
#endif
  for (int e = tid; e < RPC * 16; e += NT) {
    const int off = (row0 + (e >> 4)) * 64 + (e & 15) * 4;
    const bool lrow = (e & 15) == 0;   // the row's first thread also sums the row's l (same DSMEM round)
    float4 v[CSMAX];
    float lv[CSMAX];
#pragma unroll
    for (int q = 0; q < CSMAX; ++q) {   // all peers' loads in flight at once
      if (q < CS) {
        v[q] = cl_ld4(p, PB, q, off, REGION);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        lv[q] = lrow ? cl_ld1(p, PL, q, row0 + (e >> 4), REGION) : 0.f;
#else
        lv[q] = lrow ? cl_ld1(p, PL, q, L * 16 * 64 + row0 + (e >> 4), REGION) : 0.f;
#endif
      }
    }
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
    float la = 0.f;
#pragma unroll
    for (int q = 0; q < CSMAX; ++q)
      if (q < CS) { a.x += v[q].x; a.y += v[q].y; a.z += v[q].z; a.w += v[q].w; la += lv[q]; }
    *reinterpret_cast<float4*>(&RB[e * 4]) = a;
    if (lrow) RL[e >> 4] = la;
  }
  }
  if (OKC_CLDBG) {
    long long c1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(c1));
    if (p.dbg && threadIdx.x == 0) p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + 11] = c1 - dbgc0;
  }
  if (OKC_CLDBG) DBG(8);
  DBG(14);
  cl_sync(p, 2);   // peers are done reading this CTA's partial
  cl_done(p);
  DBG(6);

  bool fin = true;
  if (OKC_CLDBG) DBG(9);
  if (p.ncl > 1) {
    fin = cluster_multi<L>(p, S, CS, rank, RPC, row0, bp, RB, RL);
  }
  if (fin) {
    // rmsnorm * sigmoid(gate) for the slice: 16 threads per row, 4 dims each
    if (OKC_CLDBG) DBG(10);
    long long dbgc2 = 0;
    if (OKC_CLDBG) asm volatile("mov.u64 %0, %%clock64;" : "=l"(dbgc2));
    const float vscale = S.scl[1];
    for (int eb = tid - (tid & 31); eb < RPC * 16; eb += NT) {   // warp-uniform trip count (shuffles below)
      const int e = eb + (tid & 31);
      const bool act = e < RPC * 16;
      const int rl = act ? (e >> 4) : 0, c4 = e & 15;
      const int row = row0 + rl;
      const int j = row >> 4, hh = row & 15;
      const float lsum = act ? RL[rl] : 0.f;
      const float inv = lsum > 0.f ? __fdividef(vscale, lsum) : 0.f;
      const float4 a = act ? *reinterpret_cast<const float4*>(&RB[e * 4]) : make_float4(0.f, 0.f, 0.f, 0.f);
      float y[4] = {pr8(a.x * inv), pr8(a.y * inv), pr8(a.z * inv), pr8(a.w * inv)};
      float ss = y[0] * y[0] + y[1] * y[1] + y[2] * y[2] + y[3] * y[3];
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      ss += __shfl_xor_sync(0xffffffffu, ss, 4);
      ss += __shfl_xor_sync(0xffffffffu, ss, 8);
      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      const uint2 graw = *reinterpret_cast<const uint2*>(&S.gst[rl * 64 + c4 * 4]);
      const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw);
      uint2 ov;
      uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const float2 gt = __bfloat1622float2(g2[k]);
        const float s0 = sigm(gt.x), s1 = sigm(gt.y);
        __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * k] * rn * s0, y[2 * k + 1] * rn * s1);
        ow[k] = *reinterpret_cast<uint32_t*>(&r2);
      }
      if (act) *reinterpret_cast<uint2*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + c4 * 4) = ov;
    }
    if (OKC_CLDBG) {
      long long c3;
      asm volatile("mov.u64 %0, %%clock64;" : "=l"(c3));
      if (p.dbg && threadIdx.x == 0) p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + 15] = c3 - dbgc2;
    }
    DBG(5);
  }
}

// Gate rows of this CTA's cluster slice -> S.gst (cp.async, in the caller's commit group)
template <int L, typename SM>
__device__ __forceinline__ void cluster_stage_gate(const Params& p, SM& S, const int b, const int pair) {
  constexpr int NT = Cfg<L, false>::NT;
  constexpr int STAGES = Cfg<L, false>::STAGES;
  constexpr int IDX_SLOTS = 2 * STAGES;
  const int RPC = fdiv_i(L * 16, p.cs), row0 = cl_rank(p) * RPC;
  for (int e = threadIdx.x; e < RPC * 8; e += NT) {
    const int rl = e >> 3, ch = e & 7;
    const int row = row0 + rl, j = row >> 4, hh = row & 15;
    cp_async16(smem_u32(&S.gst[rl * 64 + ch * 8]),
               p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + ch * 8, 16);
  }
}

// One (request b, KV pair) segment [start, end) of the request's key positions.
// Sum the nc workspace copies of one (row, 8-dim part) of request-pair bp, token j, head hh, and zero them. Copies in
// groups of 4 (loads of a group in flight together): one compact loop body, since this runs once per launch from a
// cold instruction cache (an 8-way unrolled version cost ~2x the instructions).
template <int L>
__device__ __forceinline__ void sum_copies(const Params& p, const int bp, const int j, const int hh, const int part,
                                           const bool act, float (&y)[8], float& lsum) {
  if (OKC_SC1 && p.nc == 1) {   // one copy (the common case): no copy loop, no per-copy branches (scout's census)
    if (act) {
      float* acc = p.ws_acc + ((ix_t)(bp * L + j) * 16 + hh) * 64 + part * 8;
      const float4 xa = __ldcg(reinterpret_cast<const float4*>(acc));
      const float4 xb = __ldcg(reinterpret_cast<const float4*>(acc) + 1);
      const float lc = __ldcg(p.ws_l + (bp * L + j) * 16 + hh);
      y[0] += xa.x; y[1] += xa.y; y[2] += xa.z; y[3] += xa.w;
      y[4] += xb.x; y[5] += xb.y; y[6] += xb.z; y[7] += xb.w;
      lsum += lc;
      *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);
      *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);
      if (part == 0) p.ws_l[(bp * L + j) * 16 + hh] = 0.f;
    }
    return;
  }
  if (OKC_SC8 && p.nc == 8) {   // eight copies (the one-request q_len 8 plan): every copy's loads in one round trip
    if (act) {                   // (the rolled 4-copy loop below took two dependent L2 round trips)
      float4 xa[8], xb[8];
      float lc[8];
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        const float* acc = p.ws_acc + ((ix_t)((bp * 8 + c) * L + j) * 16 + hh) * 64 + part * 8;
        xa[c] = __ldcg(reinterpret_cast<const float4*>(acc));
        xb[c] = __ldcg(reinterpret_cast<const float4*>(acc) + 1);
        lc[c] = __ldcg(p.ws_l + ((bp * 8 + c) * L + j) * 16 + hh);
      }
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        y[0] += xa[c].x; y[1] += xa[c].y; y[2] += xa[c].z; y[3] += xa[c].w;
        y[4] += xb[c].x; y[5] += xb[c].y; y[6] += xb[c].z; y[7] += xb[c].w;
        lsum += lc[c];
        float* acc = p.ws_acc + ((ix_t)((bp * 8 + c) * L + j) * 16 + hh) * 64 + part * 8;
        *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);
        *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);
        if (part == 0) p.ws_l[((bp * 8 + c) * L + j) * 16 + hh] = 0.f;
      }
    }
    return;
  }
#pragma unroll 1
  for (int c0 = 0; c0 < p.nc; c0 += 4) {
    float4 xa[4], xb[4];
    float lc[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int c = c0 + u;
      xa[u] = make_float4(0.f, 0.f, 0.f, 0.f);
      xb[u] = make_float4(0.f, 0.f, 0.f, 0.f);
      lc[u] = 0.f;
      if (act && c < p.nc) {
        const float* acc = p.ws_acc + ((ix_t)((bp * p.nc + c) * L + j) * 16 + hh) * 64 + part * 8;
        xa[u] = __ldcg(reinterpret_cast<const float4*>(acc));
        xb[u] = __ldcg(reinterpret_cast<const float4*>(acc) + 1);
        lc[u] = __ldcg(p.ws_l + ((bp * p.nc + c) * L + j) * 16 + hh);
      }
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int c = c0 + u;
      if (act && c < p.nc) {
        y[0] += xa[u].x; y[1] += xa[u].y; y[2] += xa[u].z; y[3] += xa[u].w;
        y[4] += xb[u].x; y[5] += xb[u].y; y[6] += xb[u].z; y[7] += xb[u].w;
        lsum += lc[u];
        float* acc = p.ws_acc + ((ix_t)((bp * p.nc + c) * L + j) * 16 + hh) * 64 + part * 8;
        *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);
        *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);
        if (part == 0) p.ws_l[((bp * p.nc + c) * L + j) * 16 + hh] = 0.f;
      }
    }
  }
}

// Last-CTA finalize of a request with one workspace copy: sum, normalize, rmsnorm * sigmoid(gate), store,
// reset the workspace. Not inlined, so a dry call early in the kernel (dry = 1: same instructions, stores off)
// pulls its code into the SM's instruction cache: this runs once per launch and is otherwise fetched cold.
template <int L, int NT>
__device__ __forceinline__ void fin_nc1(const Params& p, const int b, const int pair, const int bp, const float vscale,
                                     const int dry) {
  const int tid = threadIdx.x;
  // loads of a group of passes first (one L2 round trip per group), then math and stores; groups of 2 passes in a
  // rolled loop (this runs once per launch from a cold instruction cache: q_len 8 has 4 passes)
  constexpr int PASSES = L * 16 / (NT / 8);
  constexpr int GP = PASSES >= 2 ? 2 : 1;
  const int part = tid & 7;
#pragma unroll 1
  for (int p0 = 0; p0 < PASSES; p0 += GP) {
    float4 xa[GP], xb[GP];
    float lc[GP];
    uint4 graw[GP];
#pragma unroll
    for (int u = 0; u < GP; ++u) {
      const int rr = (p0 + u) * (NT / 8) + (tid >> 3);
      const int j = rr >> 4, hh = rr & 15;
      const float* acc = p.ws_acc + ((ix_t)(bp * L + j) * 16 + hh) * 64 + part * 8;
      xa[u] = __ldcg(reinterpret_cast<const float4*>(acc));
      xb[u] = __ldcg(reinterpret_cast<const float4*>(acc) + 1);
      lc[u] = __ldcg(p.ws_l + (bp * L + j) * 16 + hh);
      graw[u] = *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8);
    }
#pragma unroll
    for (int u = 0; u < GP; ++u) {
      const int rr = (p0 + u) * (NT / 8) + (tid >> 3);
      const int j = rr >> 4, hh = rr & 15;
      float* acc = p.ws_acc + ((ix_t)(bp * L + j) * 16 + hh) * 64 + part * 8;
      if (!dry) {
        *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);
        *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);
        if (part == 0) p.ws_l[(bp * L + j) * 16 + hh] = 0.f;
      }
      const float inv = lc[u] > 0.f ? __fdividef(vscale, lc[u]) : 0.f;
      float y[8] = {pr8(xa[u].x * inv), pr8(xa[u].y * inv), pr8(xa[u].z * inv), pr8(xa[u].w * inv),
                    pr8(xb[u].x * inv), pr8(xb[u].y * inv), pr8(xb[u].z * inv), pr8(xb[u].w * inv)};
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) ss += y[e] * y[e];
      ss = oct_sum(ss);
      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw[u]);
      uint4 ov;
      uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 gt = __bfloat1622float2(g2[e]);
        const float s0 = sigm(gt.x), s1 = sigm(gt.y);
        __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * e] * rn * s0, y[2 * e + 1] * rn * s1);
        ow[e] = *reinterpret_cast<uint32_t*>(&r2);
      }
      if (!dry) *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + part * 8) = ov;
    }
  }
}

// OKC_PATCHNI: the q_len 8 flat loop's rare new-row patch as a call, so ptxas cannot hoist its bounds math into the
// common path of every tile
template <int L, int NT, bool PTR, class SM>
__device__ __noinline__ bool ws_patch_ni(SM* Sp, int i, int st, int start, int seq, int end, int tid) {
  SM& S = *Sp;
  const int t0 = start + i * TILE;
  const int lo = max(t0, seq), hi = min(t0 + TILE, end);
  if (lo < hi) {
    for (int e = tid; e < (hi - lo) * 8; e += NT) {
      const int pos = lo + (e >> 3), c = e & 7, r = pos - t0;
      const int sw = ((c ^ (r & 7)) << 4);
      *reinterpret_cast<uint4*>(S.k + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + c * 16);
      if constexpr (!PTR)
        *reinterpret_cast<uint4*>(S.v + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + c * 16);
    }
    if (L == 8) fence_proxy_async();
  }
  return lo < hi;
}

template <int L, bool FLAT, int ROLE, int MODE = 0>
__device__ __forceinline__ bool segment(const Params& p, Smem<L, FLAT>& S, const int b_, const int pair_, const bool first,
                                        const int start_, const int end_, const int seq_, const int n_keys_,
                                        const int copy_, const int chunk_, const int tb_ = 0, const int seq_real_ = 0,
                                        const bool skip_init_ = false) {
  // (returns false only in the OKC_SPEC builds, when the speculated length missed: nothing global was written yet and
  // the caller re-runs the segment with the real length)
  // per-segment values as lane-0 broadcasts (OKC_UARGS): provably warp-uniform, so ptxas keeps the address math
  // derived from them in uniform registers (they come from shared memory or per-CTA math, uniform but unproven)
#if OKC_UARGS
#define UNI(x) __shfl_sync(0xffffffffu, (x), 0)
#else
#define UNI(x) (x)
#endif
  const int b = UNI(b_), pair = UNI(pair_), start = UNI(start_), end = UNI(end_), seq = UNI(seq_);
  const int n_keys = UNI(n_keys_), copy = UNI(copy_), chunk = UNI(chunk_), tb = UNI(tb_);
#undef UNI
  // tb: tiles this CTA's pipeline has already run (flat producer mode: the stage ring and its mbarrier phases
  // continue across segments, so the producer runs ahead into the next segment instead of draining)
  // MODE: the per-request (non-flat) kernel is built once per launch mode, so each build carries only its own
  // epilogue (once-per-launch code runs from a cold instruction cache): 1 plain split-KV, 2 clusters,
  // 3 cooperative; 0 = decided at run time (flat kernels)
  // OKC_ONE: MODE | 4 = a build for grids whose every split holds at most one tile (the host checks): the tile loop,
  // the in-loop copies and the first-tile tests fold at compile time (scout's once-path census, s1 / vs1)
  constexpr bool ONE = !FLAT && (MODE & 4) != 0;
  const bool M_CL = !FLAT && ((MODE & 3) == 0 ? p.cs > 1 : (MODE & 3) == 2);
  const bool M_COOP = (MODE & 3) == 0 ? p.coop != 0 : (MODE & 3) == 3;
  constexpr int NT = Cfg<L, FLAT>::NT;
  constexpr int STAGES = Cfg<L, FLAT>::STAGES;
  constexpr bool CONT = FLAT && Cfg<L, FLAT>::WS && OKC_CONT;
  // flat q_len 8 with FP8 PV: the producer warps (idle most of the time) build each tile's V^T for the PV wgmma
  constexpr bool PTR = (L == 8) && FLAT && Cfg<L, FLAT>::WS && OKC_PTR && PV8ON && OKC_PV8FLAT && OKC_QSS && !OKC_CONT;
  constexpr int IDX_SLOTS = 2 * STAGES;
  constexpr bool WS = Cfg<L, FLAT>::WS;
  // (OKC_UWARP: the warp index as a lane-0 broadcast, so ptxas can prove it warp-uniform and keep the addresses
  // derived from it (KV head, descriptors, buffers) in uniform registers; CUTLASS's canonical_warp_idx_sync idiom)
  const int tid = threadIdx.x, lane = tid & 31, warp = OKC_UWARP ? __shfl_sync(0xffffffffu, tid >> 5, 0) : tid >> 5;
  const int g = lane >> 2, tig = lane & 3;
  const int split = first ? 0 : 1;
  const int pairs = p.Hkv >> 1;
  const int bp = b * pairs + pair;
  const int ntiles = ONE ? 1 : (end - start + TILE - 1) / TILE;   // (ONE: a CTA without keys returned earlier)

  if constexpr (WS) {
    constexpr int NP = Cfg<L, FLAT>::NP;
    constexpr int PR = NP / 8;            // rows per producer pass
    constexpr int PK = TILE / PR;         // passes per tile
    if (!CONT && !skip_init_) {   // (OKC_EARLYBAR: the first segment's barriers were set up at kernel start)
      if (ROLE == 2 && tid == 0) {   // fresh phases for this segment (the previous one ended with every stage consumed)
        for (int st = 0; st < STAGES; ++st) {
          mbar_init_n(&S.full[st], NP);
          mbar_init_n(&S.empty[st], Cfg<L, FLAT>::F2 ? 8 : (OKC_XP_ARRALL ? NT : NT / 32));   // F2: one warpgroup per head takes each tile
        }
        if constexpr (PTR) {
          for (int k = 0; k < 2; ++k) {
            mbar_init_n(&S.vtf[k], NP);
            mbar_init_n(&S.vte[k], NT / 32);
          }
          mbar_init_n(&S.qdone, NT / 32);
        }
        mbar_fence_init();
      }
      __syncthreads();
    }
    if constexpr (ROLE == 1) {
      // ---- producer threads: 16-byte chunk (q & 7) of rows (q >> 3) + PR*k; the cache rows of every tile,
      // zeros past the range; the block's own new keys are patched in by the consumers
      const int q = tid - NT;
      if (q == 0) FPRO_STAMP(42);
      const int ch = q & 7, r0 = q >> 3;
      const int* ptr = p.page_table + (ix_t)b * p.max_pages;
      const int cache_end = min(end, seq);
      int pg[PK];
#if OKC_PFAST
      static_assert(PR % 8 == 0, "PFAST: the swizzle chunk is the same for all of a thread's rows");
      const bool pfast = OKC_OFF32 && p.off32 && p.lp == 0;
      const uint32_t pcol = (uint32_t)(pair * 128 + ch * 16);
      const uint32_t pso = r0 * ROWB + ((ch ^ (r0 & 7)) << 4);
      const uint64_t cpol = OKC_CPHINT ? l2_evict_first_policy() : 0ull;
#endif
      // OKC_TMAPG: page indices by TMA (q_len 1 flat only: f16 -2.4%, f32 -1.7%; the q_len 8 flat kernel lost ~0.7%)
      constexpr bool TMAPG_ON = OKC_TMAPG && (L == 1 || OKC_TMAPG8);
      // page indices by TMA: one 64-entry box per tile, PD tiles ahead, PS slots, producer barrier per tile
      constexpr int PS = 4, PD = 3;
      int pgb = 0;   // OKC_PGCONT: fills issued by earlier segments (slot / phase base)
      if constexpr (TMAPG_ON) {
        if constexpr (OKC_PGCONT) {
          pgb = S.pgc;
        } else {
          if (q == 0) {
            for (int st = 0; st < PS; ++st) mbar_init_n(&S.pbar[st], 1);
            mbar_fence_init();
          }
          asm volatile("bar.sync 2, %0;\n" ::"n"(NP) : "memory");
        }
      }
      // the 16-byte aligned window holding the tile's page indices; reads stay inside the table's allocation
      // (CUDA allocations are 256-byte aligned and sized in larger multiples than 16 bytes)
      const long long pt_total = (long long)p.B * p.max_pages;
      const long long pt_lim = (pt_total + 3) & ~3LL;
      auto fetch_pages = [&](int x) {
        if (q == 0 && x < ntiles) {
          const int slot = (pgb + x) % PS;
          const long long e0 = (long long)b * p.max_pages + ((start + x * TILE) >> p.lp);
          const long long w0 = e0 & ~3LL;
          const long long w1 = min(pt_lim, (e0 + 64 + 3) & ~3LL);
          const int bytes = (int)(w1 - w0) * 4;
          mbar_expect_tx(&S.pbar[slot], bytes);
          bulk_load(&S.pidx[slot][0], p.page_table + w0, bytes, &S.pbar[slot]);
        }
      };
      if constexpr (TMAPG_ON) {
        for (int x = 0; x < PD; ++x) fetch_pages(x);
      }
      auto load_pages_t = [&](int x, int* dst) {
        const int slot = (pgb + x) % PS;
        mbar_wait_full(&S.pbar[slot], ((pgb + x) / PS) & 1);
        const int fp = (start + x * TILE) >> p.lp;
        const int dl = (int)(((long long)b * p.max_pages + fp) & 3);   // window offset of the first page
#pragma unroll
        for (int k = 0; k < PK; ++k) {
          const int pos = start + x * TILE + r0 + PR * k;
          dst[k] = (pos < cache_end) ? S.pidx[slot][(pos >> p.lp) - fp + dl] : 0;
        }
      };
      auto load_pages = [&](int x, int* dst) {
#pragma unroll
        for (int k = 0; k < PK; ++k) {
          const int pos = start + x * TILE + r0 + PR * k;
          dst[k] = (x < ntiles && pos < cache_end) ? __ldg(ptr + (pos >> p.lp)) : 0;
        }
      };
      if constexpr (!TMAPG_ON) load_pages(0, pg);
      if (OKC_FPRO >= 2 && q == 0) {   // (debug) when the first tile's page indices have landed
        asm volatile("" :: "r"(pg[0]), "r"(pg[PK - 1]));
        volatile int sink = pg[0] + pg[PK - 1];
        (void)sink;
        FPRO_STAMP(45);
      }
      // OKC_PTR: tile y's V (both KV heads) -> V^T slot y & 1 (72 x 64 per head: 64 dims + 8 rows of e4m3 ones for the
      // row sums, keys permuted to the P register order, K-major, 64B swizzle), the layout the PV wgmma reads
      auto ptr_transpose = [&](int y) {
        if constexpr (PTR) {
          const int sty = (tb + y) % STAGES;
          if (y == 0) {   // the q staging in S.kvq is read and S.kn / S.vn are staged: write the ones rows
            mbar_wait_full(&S.qdone, 0);
            for (int e = q; e < 4 * 32; e += NP)
              *reinterpret_cast<uint4*>(S.kvq + (e >> 5) * 4608 + 4096 + (e & 31) * 16) =
                  make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
          }
          mbar_wait_full(&S.full[sty], ((tb + y) / STAGES) & 1);   // tile y landed
          const int t0 = start + y * TILE;
          const int nlo = max(t0, seq), nhi = min(t0 + TILE, end);
          if (nlo < nhi) {   // the block's own new values (the consumers patch the new keys)
            for (int e = q; e < (nhi - nlo) * 8; e += NP) {
              const int pos = nlo + (e >> 3), c = e & 7, r = pos - t0;
              *reinterpret_cast<uint4*>(S.v + sty * STAGE_BYTES + r * ROWB + ((c ^ (r & 7)) << 4)) =
                  *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + c * 16);
            }
            asm volatile("bar.sync 2, %0;\n" ::"n"(NP) : "memory");
          }
          if (y >= 2) mbar_wait_full(&S.vte[y & 1], ((y >> 1) - 1) & 1);   // PV of tile y - 2 done with the slot
          const int pl = q & 31, pw = (q >> 5) & 3, pvq = pl & 3, pvw = pl >> 2;
          const uint32_t s0 = (pvw >= 4) ? 0x7531u : 0x6420u, s1 = (pvw >= 4) ? 0x6420u : 0x7531u;
          const int pr0 = (pvw >= 4) ? 1 : 0;
          const uint32_t o0 = 128 * pvw + ((pw ^ (pvw & 3)) << 4) + 4 * pvq + 64 * pr0;
          const uint32_t o1 = 128 * pvw + ((pw ^ (pvw & 3)) << 4) + 4 * pvq + 64 * (1 - pr0);
          const int pkey = 16 * pw + 8 * ((pl >> 3) & 1) + (pl & 7);
          const uint32_t vsrc = smem_u32(S.v + sty * STAGE_BYTES) + pkey * ROWB;
          uint8_t* vt = S.kvq + (y & 1) * 2 * 4608;
#pragma unroll
          for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int u = 0; u < 2; ++u) {
              uint32_t r0, r1, r2, r3;
              ldsm_x4_t(vsrc + (((4 * h + 2 * u + ((pl >> 3) >> 1)) ^ (pl & 7)) << 4), r0, r1, r2, r3);
#pragma unroll
              for (int g2 = 0; g2 < 2; ++g2) {
                const uint32_t R0 = g2 ? r2 : r0, R1 = g2 ? r3 : r1;
                *reinterpret_cast<uint32_t*>(vt + h * 4608 + o0 + 1024 * (2 * u + g2)) = prmt(R0, R1, s0);
                *reinterpret_cast<uint32_t*>(vt + h * 4608 + o1 + 1024 * (2 * u + g2)) = prmt(R0, R1, s1);
              }
            }
          }
          fence_proxy_async();   // V^T written with st.shared, read by the PV wgmma
          mbar_arrive(&S.vtf[y & 1]);
        }
      };
      PROF_INIT
      for (int x = 0; x < ntiles; ++x) {
        const int st = (tb + x) % STAGES;
        int pgn[PK];
        if constexpr (TMAPG_ON) {
          fetch_pages(x + PD);
          load_pages_t(x, pg);
#if OKC_PFL2 > 0 && OKC_PFAST
          // (OKC_PFL2) the rows of tile x + OKC_PFL2 toward L2 (its page indices landed PD - OKC_PFL2 tiles ago); one
          // lane per row, common tiles only
          if (OKC_PFL2 < PD && pfast && ch == 0 && x + OKC_PFL2 < ntiles && start + (x + OKC_PFL2) * TILE + TILE <= cache_end) {
            int pgf[PK];
            load_pages_t(x + OKC_PFL2, pgf);
#pragma unroll
            for (int k = 0; k < PK; ++k) {
              const uint32_t off = ((uint32_t)pgf[k] << p.lrb) + pcol;
              asm volatile("prefetch.global.L2 [%0];" ::"l"(p.k_cache + off));
              asm volatile("prefetch.global.L2 [%0];" ::"l"(p.v_cache + off));
            }
          }
#endif
        } else {
          load_pages(x + 1, pgn);   // in flight while this tile's copies issue
        }
        PROF_MARK(2);
        if (tb + x >= STAGES) mbar_wait_full(&S.empty[st], (((tb + x) / STAGES) - 1) & 1);
        PROF_MARK(0);
        if (x == 0 && q == 0) FPRO_STAMP(46);
        const uint32_t kb = smem_u32(S.k + st * STAGE_BYTES), vb = smem_u32(S.v + st * STAGE_BYTES);
        if constexpr (!OKC_XNOWAITF) {   // (OKC_XNOWAITF probe: no copies, the stage barrier completes at once)
#if OKC_PFAST
        // common tiles (every row cached, page size 1, caches below 4 GB): one shift-add per row, no per-row
        // branches (the producer's instructions take issue slots from the consumers on the same SMSPs)
        if (pfast && start + x * TILE + TILE <= cache_end) {
#pragma unroll
          for (int k = 0; k < PK; ++k) {
            const uint32_t off = ((uint32_t)pg[k] << p.lrb) + pcol;
            if constexpr (OKC_CPHINT) {
              cp_async16_hint(kb + pso + k * (PR * ROWB), p.k_cache + off, 16, cpol);
              cp_async16_hint(vb + pso + k * (PR * ROWB), p.v_cache + off, 16, cpol);
            } else {
              cp_async16(kb + pso + k * (PR * ROWB), p.k_cache + off, 16);
              cp_async16(vb + pso + k * (PR * ROWB), p.v_cache + off, 16);
            }
          }
        } else
#endif
#pragma unroll
        for (int k = 0; k < PK; ++k) {
          const int r = r0 + PR * k;
          const int pos = start + x * TILE + r;
          const uint32_t so = r * ROWB + ((ch ^ (r & 7)) << 4);
          if (pos < cache_end) {
#if OKC_OFF32
            if (p.off32) {   // caches below 4 GB: 32-bit offsets (uniform branch)
              const uint32_t off = ((((uint32_t)pg[k] << p.lp) + (uint32_t)(pos & (p.P - 1))) << p.lrb) + (uint32_t)(pair * 128 + ch * 16);
              cp_async16(kb + so, p.k_cache + off, 16);
              cp_async16(vb + so, p.v_cache + off, 16);
            } else
#endif
            {
              const int64_t off = (((int64_t)pg[k] << p.lp) + (pos & (p.P - 1))) * p.row_bytes + pair * 128 + ch * 16;
              cp_async16(kb + so, p.k_cache + off, 16);
              cp_async16(vb + so, p.v_cache + off, 16);
            }
          } else if (pos >= end) {
            cp_async16_zero(kb + so, p.k_cache);
            cp_async16_zero(vb + so, p.v_cache);
          }
        }
        }
        cp_async_mbar_arrive(&S.full[st]);
        if (x == 0 && q == 0) FPRO_STAMP(43);
        if constexpr (PTR) {
          if (x >= OKC_PTRD) ptr_transpose(x - OKC_PTRD);
        }
        PROF_MARK(1);
        PROF_TILE;
        if constexpr (TMAPG_ON) {
          asm volatile("bar.sync 2, %0;\n" ::"n"(NP) : "memory");   // slot x % PS is free for tile x + PS
        } else {
#pragma unroll
          for (int k = 0; k < PK; ++k) pg[k] = pgn[k];
        }
      }
      if constexpr (PTR) {
        for (int y = max(0, ntiles - OKC_PTRD); y < ntiles; ++y) ptr_transpose(y);
      }
      if constexpr (TMAPG_ON && OKC_PGCONT) {   // every producer read pgb at the segment start (tile barriers since)
        if (q == 0) S.pgc = pgb + ntiles;
      }
      PROF_DUMP;
      if constexpr (!CONT) {
        cp_wait<0>();
        __syncthreads();   // matches the consumers' end-of-segment barrier
      }
      return true;
    }
  }
  if constexpr (ROLE != 1) {

  const int prow = tid >> 1, phalf = tid & 1;
  const int* pt_row = p.page_table + (ix_t)b * p.max_pages;
  const int part = tid & 7;

  // ---- producer helpers ----
  // K/V: thread -> 16-byte chunk (tid & 7) of rows (tid >> 3) + 16*k: a warp copies 4 whole 128-byte rows.
  // New-token rows come from S.kn / S.vn; rows past the range are zeroed.
  const int cchunk = tid & 7, crow0 = tid >> 3;
  constexpr int RPP = NT / 8;   // rows per pass
  auto issue_kv = [&](int x, bool with_new) {
    const int st = x % STAGES;
    const int* idxs = &S.idx[(x % IDX_SLOTS) * TILE];
    bool generic = false;   // st.shared into a K stage: the wgmma (async proxy) needs a proxy fence
#if OKC_TMA
    const int nv = max(0, min(TILE, min(end, seq) - (start + x * TILE)));   // rows read from the cache
    if (p.tma) {
      if (tid == 0) mbar_expect_tx(&S.mbar[st], nv * 2 * ROWB);
      if (tid < nv) {
        const int pos = start + x * TILE + tid;
        const int row = (idxs[tid] << p.lp) + (pos & (p.P - 1));
        tma_row(S.k + st * STAGE_BYTES + tid * ROWB, &p.tk, pair * 128, row, &S.mbar[st]);
        tma_row(S.v + st * STAGE_BYTES + tid * ROWB, &p.tv, pair * 128, row, &S.mbar[st]);
      }
    }
#endif
#if OKC_BRFREE && !OKC_TMA
    // branch-free on the common path (a taken branch costs 60-100 cycles from a cold instruction cache): cached
    // rows by cp.async, rows past the range zero-filled by cp.async (src-size 0), the block's own new rows (rare)
    // by stores; the 32/64-bit offset choice once per call
    {
      const uint32_t kst = smem_u32(S.k + st * STAGE_BYTES), vst = smem_u32(S.v + st * STAGE_BYTES);
      const int t0x = start + x * TILE;
      auto rows = [&](auto o32) {
#pragma unroll
        for (int k = 0; k < TILE / RPP; ++k) {
          const int r = crow0 + RPP * k;
          const int pos = t0x + r;
          const uint32_t so = r * ROWB + ((cchunk ^ (r & 7)) << 4);
          const bool cached = pos < end && pos < seq;
          const bool issue = cached || pos >= end;
          const bool fresh = with_new && !cached && pos < end;
          const int ix = idxs[r];
          const uint8_t* ks;
          const uint8_t* vs;
          if constexpr (decltype(o32)::value) {
            const uint32_t off = ((((uint32_t)ix << p.lp) + (uint32_t)(pos & (p.P - 1))) << p.lrb) + (uint32_t)(pair * 128 + cchunk * 16);
            ks = p.k_cache + off;
            vs = p.v_cache + off;
          } else {
            const int64_t off = (((int64_t)ix << p.lp) + (pos & (p.P - 1))) * p.row_bytes + pair * 128 + cchunk * 16;
            ks = p.k_cache + off;
            vs = p.v_cache + off;
          }
          if (issue) {
            cp_async16(kst + so, cached ? ks : p.k_cache, cached ? 16 : 0);
            cp_async16(vst + so, cached ? vs : p.v_cache, cached ? 16 : 0);
          }
          if (fresh) {
            *reinterpret_cast<uint4*>(S.k + st * STAGE_BYTES + so) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + cchunk * 16);
            *reinterpret_cast<uint4*>(S.v + st * STAGE_BYTES + so) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + cchunk * 16);
            generic = true;
          }
        }
      };
#if OKC_OFF32
      if (p.off32) rows(std::true_type{});
      else
#endif
        rows(std::false_type{});
      if (L == 8 && generic) fence_proxy_async();
      return;
    }
#endif
#if OKC_PFASTPR && OKC_OFF32 && !OKC_TMA
    // OKC_PFASTPR: every row of the tile cached (pos < min(end, seq)), page size 1, caches below 4 GB: one shift-add
    // per row, no per-row branches (the per-request issue runs from a cold instruction cache, mostly once)
    if (with_new && p.off32 && p.lp == 0 && start + x * TILE + TILE <= min(end, seq)) {   // (in-loop issues only: the
      // prologue's cold copy of the tile-0 issue stays the plain one, s1 measured +1.5% with it)
      const uint32_t kst = smem_u32(S.k + st * STAGE_BYTES), vst = smem_u32(S.v + st * STAGE_BYTES);
      const uint32_t col = (uint32_t)(pair * 128 + cchunk * 16);
#pragma unroll
      for (int k = 0; k < TILE / RPP; ++k) {
        const int r = crow0 + RPP * k;
        const uint32_t so = r * ROWB + ((cchunk ^ (r & 7)) << 4);
        const uint32_t off = ((uint32_t)idxs[r] << p.lrb) + col;
        cp_async16(kst + so, p.k_cache + off, 16);
        cp_async16(vst + so, p.v_cache + off, 16);
      }
      return;
    }
#endif
#pragma unroll
    for (int k = 0; k < TILE / RPP; ++k) {
      const int r = crow0 + RPP * k;
      const int pos = start + x * TILE + r;
      const int sw = ((cchunk ^ (r & 7)) << 4);
      uint8_t* dk = S.k + st * STAGE_BYTES + r * ROWB + sw;
      uint8_t* dv = S.v + st * STAGE_BYTES + r * ROWB + sw;
      if (pos < end && pos < seq) {
#if OKC_TMA
        if (p.tma) continue;
#endif
#if OKC_OFF32
        if (p.off32) {   // caches below 4 GB: 32-bit offsets (uniform branch)
          const uint32_t off = ((((uint32_t)idxs[r] << p.lp) + (uint32_t)(pos & (p.P - 1))) << p.lrb) + (uint32_t)(pair * 128 + cchunk * 16);
          cp_async16(smem_u32(dk), p.k_cache + off, 16);
          cp_async16(smem_u32(dv), p.v_cache + off, 16);
        } else
#endif
        {
          const int64_t row = ((int64_t)idxs[r] << p.lp) + (pos & (p.P - 1));
          const int64_t off = row * p.row_bytes + pair * 128 + cchunk * 16;
          cp_async16(smem_u32(dk), p.k_cache + off, 16);
          cp_async16(smem_u32(dv), p.v_cache + off, 16);
        }
      } else if (pos < end) {
        if (with_new) {
          *reinterpret_cast<uint4*>(dk) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + cchunk * 16);
          *reinterpret_cast<uint4*>(dv) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + cchunk * 16);
          generic = true;
        }
      } else {   // past the range: masked out of S; V must be finite (P = 0 times V)
        *reinterpret_cast<uint4*>(dk) = make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(dv) = make_uint4(0, 0, 0, 0);
      }
    }
    if (L == 8 && generic) fence_proxy_async();
  };
  auto issue_idx = [&](int y) {
#if OKC_BRFREE
    // predicated: one select-built copy per thread, invalid entries zero-filled (never read)
    const int pos = start + y * TILE + prow;
    const bool val = pos < end && pos < seq;
    if (phalf == 0 && prow < TILE && y < ntiles)
      cp_async4z(smem_u32(&S.idx[(y % IDX_SLOTS) * TILE + prow]), val ? pt_row + (pos >> p.lp) : pt_row, val ? 4 : 0);
#else
    if (phalf == 0 && prow < TILE && y < ntiles) {
      const int pos = start + y * TILE + prow;
      if (pos < end && pos < seq) cp_async4(smem_u32(&S.idx[(y % IDX_SLOTS) * TILE + prow]), pt_row + (pos >> p.lp));
    }
#endif
  };
  // producer mode: wait for stage `st` (tile i) and write the block's own new keys / values into it
  auto ws_patch = [&](int i, int st) {   // the block's own new rows (positions >= seq) into tile i's stage
    const int t0 = start + i * TILE;
    const int lo = max(t0, seq), hi = min(t0 + TILE, end);
    if (lo < hi) {   // same for every consumer thread
      for (int e = tid; e < (hi - lo) * 8; e += NT) {
        const int pos = lo + (e >> 3), c = e & 7, r = pos - t0;
        const int sw = ((c ^ (r & 7)) << 4);
        *reinterpret_cast<uint4*>(S.k + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + c * 16);
        if constexpr (!PTR)
          *reinterpret_cast<uint4*>(S.v + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + c * 16);
      }
      if (L == 8) fence_proxy_async();
    }
    return lo < hi;
  };
  auto ws_acquire = [&](int i, int st) {
    mbar_wait_full(&S.full[st], ((tb + i) / STAGES) & 1);
    const int t0 = start + i * TILE;
    const int lo = max(t0, seq), hi = min(t0 + TILE, end);
    if (lo < hi) {   // same for every consumer thread
      for (int e = tid; e < (hi - lo) * 8; e += NT) {
        const int pos = lo + (e >> 3), c = e & 7, r = pos - t0;
        const int sw = ((c ^ (r & 7)) << 4);
        *reinterpret_cast<uint4*>(S.k + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + c * 16);
        if constexpr (!PTR)   // (OKC_PTR: the producers patch the values before their transpose)
          *reinterpret_cast<uint4*>(S.v + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + c * 16);
      }
      if (L == 8) fence_proxy_async();
    }
    return lo < hi;
  };

  // ---- 1. page indices of the first STAGES-1 tiles; raw q / k / v rows into registers ----
#if OKC_TMA
  if (p.tma && tid == 0) {   // fresh phases per segment (every earlier fill was consumed)
    for (int st = 0; st < STAGES; ++st) mbar_init(&S.mbar[st]);
    mbar_fence_init();
  }
#endif
  int tidL = tid, partL = part;   // (OKC_LATETID: re-read after the page-index copies are issued)
  constexpr int QPASS = L * 16 / RPP;            // L*16 rows, RPP rows per pass
  const bool has_new = end > seq;
  constexpr int KPASS = (L * 4 + RPP - 1) / RPP; // L*4 rows (k, v of 2 heads), RPP rows per pass
  constexpr bool QDIR = OKC_QDIRECT && L == 8 && !WS && !Cfg<L, FLAT>::S2 && PV8ON && OKC_QSS && OKC_SM90 && OKC_Q8RAW &&
                        (!FLAT || OKC_PV8FLAT);
  constexpr bool IDXF = OKC_IDXFIRST && !WS && !Cfg<L, FLAT>::S2 && !OKC_PDLKV;
#if OKC_PDLKV
  if (__builtin_expect(p.pdlf != 0, 0)) {   // (unlikely: the standalone path falls through, no taken branch)
    // OKC_PDLKV, PDL launches only (decode_layer): everything the preceding kernel does not write first -- page
    // indices, the small constants, the first tiles' K/V (past keys) -- then griddepcontrol.wait, so the K/V flight
    // overlaps the qkv kernel's tail; q and the new k / v (its output) as direct 16-byte loads (each thread stages
    // the chunk it processes), outside the cp.async groups. (Standalone this order loses: q lands later.)
    if constexpr (!WS) {
#pragma unroll
      for (int y = 0; y < STAGES - 1; ++y) issue_idx(y);
    }
    if (tid < 8) cp_async16(smem_u32(&S.ifreq[tid * 4]), p.inv_freq + tid * 4, 16);
    else if (tid == 8) cp_async4(smem_u32(&S.scl[0]), p.k_scale);
    else if (tid == 9) cp_async4(smem_u32(&S.scl[1]), p.v_scale);
    else if ((OKC_SENSE2 || OKC_PARITY) && !FLAT && M_COOP && tid == 10)
      cp_async4(smem_u32(&S.sense), p.ws_cnt + 24576 + bp);
    else if (tid >= 16 && tid < 16 + L) {
      const int jj = tid - 16;
      asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(smem_u32(&S.cloc[jj])), "l"(p.cache_loc + b * L + jj));
    }
    cp_commit();
    cp_wait<0>();
    csync<WS, NT>();
    DBG(1);
    if constexpr (!WS) {
      constexpr int PRE = Cfg<L, FLAT>::S2 ? STAGES - 2 : STAGES - 1;
#pragma unroll 1
      for (int k = 0; k < STAGES - 1; ++k) {
        if (k < PRE && k < ntiles) issue_kv(k, false);
        issue_idx(k + STAGES - 1);
        if (k < PRE - 1 || k == STAGES - 2) cp_commit();
      }
    }
    DBG(8);
    pdl_wait();
    if (M_CL) cluster_stage_gate<L>(p, S, b, pair);
#pragma unroll
    for (int ps = 0; ps < QPASS; ++ps) {
      const int rr = ps * RPP + (tid >> 3);
      const uint4 v = __ldcg(reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + (rr >> 4)) * p.ld_qkv + (16 * pair + (rr & 15)) * 64 + (tid & 7) * 8));
      *reinterpret_cast<uint4*>(&S.qst()[rr * 64 + (tid & 7) * 8]) = v;
    }
    if (has_new) {
#pragma unroll
      for (int ps = 0; ps < KPASS; ++ps) {
        const int rr = ps * RPP + (tid >> 3);
        if (rr < L * 4) {
          const int kind = rr / (2 * L), jj = (rr % (2 * L)) >> 1, hh = rr & 1;
          const int col = (kind == 0 ? p.Hq : p.Hq + p.Hkv) * 64 + (2 * pair + hh) * 64 + (tid & 7) * 8;
          *reinterpret_cast<uint4*>(&S.kvst[rr * 64 + (tid & 7) * 8]) =
              __ldcg(reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + jj) * p.ld_qkv + col));
        }
      }
    }
    if (split == 0 && (FLAT || !M_CL)) {
      tfor<L * 16, NT>(tid, [&](int e) {
        const __nv_bfloat16* gp = p.qkv + (ix_t)(b * L + (e >> 4)) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + (e & 15)) * 64;
        asm volatile("prefetch.global.L2 [%0];" ::"l"(gp));
      });
    }
  } else
#endif
  {
  if constexpr (!WS) {
#pragma unroll
    for (int y = 0; y < STAGES - 1; ++y) issue_idx(y);
  }
  cp_commit();
  if (OKC_LATETID && !WS && (MODE & 8)) {   // (speculative builds only: Oct 26 01:33, vs1 -3%, vf1 -1.6%, but s32/vs32 +1.5%) a late copy of tid: the q / new-row / table math below cannot be hoisted above the page-index
    // copies (or out of the speculative retry loop to the kernel top: scout's v200 census, segment head 21 -> 62/97)
    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tidL));
    partL = tidL & 7;
  }

  pdl_wait();   // qkv (q, new k / v, gate) is the preceding kernel's output; nothing above reads or writes it
  if (tid == 0) FPRO_STAMP(44);
  // raw q rows: L*16 rows x 128 B = 8 chunks of 16 B per row
  tfor<L * 16 * 8, NT>(tidL, [&](int e) {   // (straight-line: no loop branches on the once-path)
    const int rr = e >> 3, ch = e & 7;
    const int j = rr >> 4, hh = rr & 15;
    cp_async16(smem_u32(&S.qst()[rr * 64 + ch * 8]), p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (16 * pair + hh) * 64 + ch * 8, 16);
  });
  if (has_new) {
    tfor<L * 4 * 8, NT>(tidL, [&](int e) {   // (straight-line: no loop branches on the once-path)
      const int rr = e >> 3, ch = e & 7;
      const int kind = rr / (2 * L), jj = (rr % (2 * L)) >> 1, hh = rr & 1;
      const int col = (kind == 0 ? p.Hq : p.Hq + p.Hkv) * 64 + (2 * pair + hh) * 64 + ch * 8;
      cp_async16(smem_u32(&S.kvst[rr * 64 + ch * 8]), p.qkv + (ix_t)(b * L + jj) * p.ld_qkv + col, 16);
    });
  }
  if (M_CL) cluster_stage_gate<L>(p, S, b, pair);
  // small per-call constants: staged with the same group instead of cold loads in the middle of the prologue
  if (tidL < 8) cp_async16(smem_u32(&S.ifreq[tidL * 4]), p.inv_freq + tidL * 4, 16);
  else if (tidL == 8) cp_async4(smem_u32(&S.scl[0]), p.k_scale);
  else if (tidL == 9) cp_async4(smem_u32(&S.scl[1]), p.v_scale);
  else if ((OKC_SENSE2 || OKC_PARITY) && !FLAT && M_COOP && tidL == 10)   // the coop barrier's flag / epoch, early
    cp_async4(smem_u32(&S.sense), p.ws_cnt + 24576 + bp);
  else if (tidL >= 16 && tidL < 16 + L) {
    const int jj = tidL - 16;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(smem_u32(&S.cloc[jj])), "l"(p.cache_loc + b * L + jj));
  }
  cp_commit();
  if (split == 0 && (FLAT || !M_CL)) {  // warm L2 with the gate rows for whichever CTA finishes last
    tfor<L * 16, NT>(tidL, [&](int e) {   // (straight-line: no loop branches on the once-path)
      const __nv_bfloat16* gp = p.qkv + (ix_t)(b * L + (e >> 4)) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + (e & 15)) * 64;
      asm volatile("prefetch.global.L2 [%0];" ::"l"(gp));
    });
  }

  // ---- 2. K/V of the first STAGES-1 tiles in flight as soon as their page indices land ----
  // OKC_IDXFIRST (per-request kernels): wait for the page-index group only (committed first, 256 B); the q / new
  // k,v / constants group (up to 16 KB) may still be in flight while the K/V copies go out
  cp_wait<IDXF ? 1 : 0>();
  csync<WS, NT>();
  if constexpr (!FLAT && (MODE & 8) != 0) {
    // OKC_SPEC: this CTA's ranges and page indices came from the speculated length (see the kernel); the real
    // cache_seqlens value has landed by now (it was loaded alongside the indices). On a miss, nothing global has been
    // written: drain the copies and let the caller redo the segment with the real length (today's path)
    if (seq_real_ != seq) {
      cp_wait<0>();
      __syncthreads();
      return false;
    }
  }
  DBG(1);
  if constexpr (!WS) {
    constexpr int PRE = Cfg<L, FLAT>::S2 ? STAGES - 2 : STAGES - 1;   // two streams: one cp.async group per tile, the loop issues pairs
    bool one = false;
#if OKC_ONETILE && OKC_OFF32 && !OKC_TMA
    // OKC_ONETILE: a CTA with a single fully cached tile (page size 1, caches < 4 GB): its copies straight-line (one
    // shift-add per row), none of the loop / per-row tests / issue_idx past the last tile, then the commits the
    // pipeline's group count expects (scout's census: ~168 instr, 11 taken branches here on s1, cold)
    if (ONE && !Cfg<L, FLAT>::S2 && p.off32 && p.lp == 0 && (OKC_ONENEW || start + TILE <= min(end, seq))) {
      one = true;
      const uint32_t kst = smem_u32(S.k), vst = smem_u32(S.v);
      const uint32_t col = (uint32_t)(pair * 128 + cchunk * 16);
      const int cend = min(end, seq);
#pragma unroll
      for (int k = 0; k < TILE / RPP; ++k) {
        const int r = crow0 + RPP * k;
        const uint32_t so = r * ROWB + ((cchunk ^ (r & 7)) << 4);
        if constexpr (OKC_ONEPRED) {
          // branch-free (scout's v200 census: the per-row branches cost 16 instr + 4 taken branches per warp, cold):
          // cached rows copy, rows past the range zero-fill (masked out of S; V must be finite), the block's new
          // rows are skipped (written by the patch pass after the new-k/v pass)
          const int pr = start + r;
          const bool cached = !OKC_ONENEW || pr < cend;
          const bool doit = cached || pr >= end;
          const uint32_t off = ((uint32_t)S.idx[r] << p.lrb) + col;   // (zero-filled index past the cached rows)
          cp_async16_pred(kst + so, p.k_cache + off, cached ? 16 : 0, doit);
          cp_async16_pred(vst + so, p.v_cache + off, cached ? 16 : 0, doit);
        } else if (!OKC_ONENEW || start + r < cend) {
          const uint32_t off = ((uint32_t)S.idx[r] << p.lrb) + col;
          cp_async16(kst + so, p.k_cache + off, 16);
          cp_async16(vst + so, p.v_cache + off, 16);
        } else if (start + r >= end) {   // (OKC_ONENEW) past the range: zero (masked out of S; V must be finite)
          cp_async16_zero(kst + so, p.k_cache);
          cp_async16_zero(vst + so, p.v_cache);
        }   // (the block's new rows: written by the patch pass after the new-k/v pass, as with the general issue)
      }
#pragma unroll
      for (int k = 0; k < STAGES - 1; ++k) cp_commit();
    }
#endif
    if (!one) {
#if OKC_PREUNR
#pragma unroll
#else
#pragma unroll 1
#endif
    for (int k = 0; k < STAGES - 1; ++k) {   // (unrolled: no loop branches; rolled: one copy of the issue code)
      if (k < PRE && k < ntiles) issue_kv(k, false);
      issue_idx(k + STAGES - 1);
      if (k < PRE - 1 || k == STAGES - 2) cp_commit();
    }
    }
  }
  DBG(8);
  }

  if constexpr (IDXF) {   // the q group (all but the STAGES-1 K/V groups just committed), visible to every thread
    cp_wait<STAGES - 1>();
    csync<WS, NT>();
  }
  // ---- 3. while they load: RoPE table, queries, the block's new keys / values ----
  if (p.rope) {
    tfor<L * 32, NT>(tidL, [&](int e) {   // (straight-line: no loop branches on the once-path)
      const int j = e >> 5, i = e & 31;
      const float ang = (float)(seq + j) * S.ifreq[i];   // fp32 angle, as the reference builds it
      float sv, cv;
#if OKC_FASTSC
      // Cody-Waite reduction mod 2 pi with a 3-part constant (6.25 has 5 significant bits: exact products for
      // |k| < 2^19, angles up to ~3e6), then the hardware sin / cos on [-pi, pi]: ~5e-7 absolute, against
      // ~30 cold instructions (and a Payne-Hanek call above |x| ~ 1e5) for the accurate sincosf
      const float kq = rintf(ang * 0.15915493667125702f);
      float rr = fmaf(-kq, 6.25f, ang);
      rr = fmaf(-kq, 0.03318530693650246f, rr);
      rr = fmaf(-kq, 2.430837753308879e-10f, rr);
      __sincosf(rr, &sv, &cv);
#else
      sincosf(ang, &sv, &cv);
#endif
      S.cs[e] = cv;
      S.sn[e] = sv;
    });
  }
  csync<WS, NT>();
  DBG(9);

  const float kscale = S.scl[0], vscale = S.scl[1];
  const float sl2 = p.scale_log2 * kscale;

  if constexpr (QDIR && OKC_QP16) {
    // OKC_QP16 (q_len 8, QDIRECT): four threads per query row, 16 dims each (half the passes of the 8-thread
    // layout, two shuffle levels instead of three, one 16-byte e4m3 store); same math per element
    constexpr int RP4 = NT / 4;                    // rows per pass
    constexpr int QP4 = L * 16 / RP4;
    const int q4 = tidL & 3;
#pragma unroll
    for (int ps = 0; ps < QP4; ++ps) {
      const int rr = ps * RP4 + (tidL >> 2);
      const int j = rr >> 4, hh = rr & 15;
      const uint4 qa = *reinterpret_cast<const uint4*>(&S.qst()[rr * 64 + q4 * 16]);
      const uint4 qb = *reinterpret_cast<const uint4*>(&S.qst()[rr * 64 + q4 * 16 + 8]);
      const __nv_bfloat162* ha = reinterpret_cast<const __nv_bfloat162*>(&qa);
      const __nv_bfloat162* hb = reinterpret_cast<const __nv_bfloat162*>(&qb);
      float x[16];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 fa = __bfloat1622float2(ha[e]);
        const float2 fb = __bfloat1622float2(hb[e]);
        x[2 * e] = fa.x; x[2 * e + 1] = fa.y;
        x[8 + 2 * e] = fb.x; x[8 + 2 * e + 1] = fb.y;
      }
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 16; ++e) ss += x[e] * x[e];
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      const float r = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
#pragma unroll
      for (int e = 0; e < 16; ++e) x[e] *= r;
      if (p.rope) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int i = q4 * 8 + e;
          const float c = S.cs[j * 32 + i], sn = S.sn[j * 32 + i];
          const float a0 = x[2 * e], a1 = x[2 * e + 1];
          x[2 * e] = a0 * c - a1 * sn;
          x[2 * e + 1] = a0 * sn + a1 * c;
        }
      }
      const int hk4 = hh >> 3, gg4 = hh & 7;
      const int r4 = j * 8 + gg4, kb = q4 * 16;
      *reinterpret_cast<uint4*>(S.q8s + hk4 * 4096 + r4 * 64 + (((kb >> 4) ^ ((r4 >> 1) & 3)) << 4)) =
          make_uint4(e4m3x4f(x[0], x[1], x[2], x[3]), e4m3x4f(x[4], x[5], x[6], x[7]),
                     e4m3x4f(x[8], x[9], x[10], x[11]), e4m3x4f(x[12], x[13], x[14], x[15]));
    }
  } else
#if OKC_QROLL
#pragma unroll 1   // one copy of the pass (it runs once per launch from a cold instruction cache)
#else
#pragma unroll
#endif
  for (int ps = 0; ps < QPASS; ++ps) {
    const int rr = ps * RPP + (tidL >> 3);
    const int j = rr >> 4, hh = rr & 15;
    const uint4 qv = *reinterpret_cast<const uint4*>(&S.qst()[rr * 64 + partL * 8]);
    const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&qv);
    float x[8];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 f = __bfloat1622float2(h2[e]);
      x[2 * e] = f.x;
      x[2 * e + 1] = f.y;
    }
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) ss += x[e] * x[e];
    ss = oct_sum(ss);
    const float r = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
#pragma unroll
    for (int e = 0; e < 8; ++e) x[e] *= r;
    if (p.rope) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int i = partL * 4 + e;
        const float c = S.cs[j * 32 + i], sn = S.sn[j * 32 + i];
        const float a0 = x[2 * e], a1 = x[2 * e + 1];
        x[2 * e] = a0 * c - a1 * sn;
        x[2 * e + 1] = a0 * sn + a1 * c;
      }
    }
    // f16 query pre-scaled to log2 units, staged in the (still free) last pipeline stage
    uint4 qw;
    uint32_t* qwp = reinterpret_cast<uint32_t*>(&qw);
    // (q_len 8 on sm_90 rounds the query to e4m3: keep it unit-RMS like SGLang does -- pre-scaling it by the
    // kv scale would push it into e4m3's subnormals -- and scale the scores instead)
    const float qsc = (L == 8 && OKC_SM90 && OKC_Q8RAW) ? 1.f : sl2;
    const int hk = hh >> 3, gg = hh & 7;
    if constexpr (QDIR) {
      // OKC_QDIRECT: straight to the e4m3 A tile of the SS QK^T wgmma (64-byte rows, 16-byte chunks swizzled by
      // (row >> 1) & 3), one fp32 -> e4m3 rounding, no f16 staging / re-read (scout's census: ~90 instr / warp)
      const int r = j * 8 + gg, kb = partL * 8;
      *reinterpret_cast<uint2*>(S.q8s + hk * 4096 + r * 64 + (((kb >> 4) ^ ((r >> 1) & 3)) << 4) + (kb & 15)) =
          make_uint2(e4m3x4f(x[0], x[1], x[2], x[3]), e4m3x4f(x[4], x[5], x[6], x[7]));
      continue;
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) qwp[e] = pack_h2(x[2 * e] * qsc, x[2 * e + 1] * qsc);
    if constexpr (L == 1) { const uint32_t t1 = qwp[1]; qwp[1] = qwp[2]; qwp[2] = t1; }   // (scout: quad order 0,2,1,3)
    if constexpr (WS) {   // in place over the raw row (same thread, same 16 bytes)
      *reinterpret_cast<uint4*>(S.kvq + rr * 128 + partL * 16) = qw;
    } else {
      uint8_t* qs = (hk ? S.v : S.k) + (STAGES - 1) * STAGE_BYTES;
      *reinterpret_cast<uint4*>(qs + (j * 8 + gg) * 128 + partL * 16) = qw;
    }
  }
  if constexpr (QDIR) fence_proxy_async();   // the e4m3 q tile (st.shared) is read by the wgmma after the loop's first barrier
  DBG(10);

  if (has_new) {
#pragma unroll
    for (int ps = 0; ps < KPASS; ++ps) {
      const int rr = ps * RPP + (tidL >> 3);
      const bool live = rr < L * 4;
      const int kind = rr / (2 * L);
      const int jj = (rr % (2 * L)) >> 1, hh = rr & 1;
      float x[8];
      const uint4 kvv = live ? *reinterpret_cast<const uint4*>(&S.kvst[rr * 64 + partL * 8]) : make_uint4(0, 0, 0, 0);
      const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&kvv);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 f = __bfloat1622float2(h2[e]);
        x[2 * e] = f.x;
        x[2 * e + 1] = f.y;
      }
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) ss += x[e] * x[e];
      ss = oct_sum(ss);   // every lane of the warp (rows of both kinds share warps)
      float sc;
      if (kind == 0) {
        const float r = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
#pragma unroll
        for (int e = 0; e < 8; ++e) x[e] *= r;
        if (p.rope && live) {
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int i = partL * 4 + e;
            const float c = S.cs[jj * 32 + i], sn = S.sn[jj * 32 + i];
            const float a0 = x[2 * e], a1 = x[2 * e + 1];
            x[2 * e] = a0 * c - a1 * sn;
            x[2 * e + 1] = a0 * sn + a1 * c;
          }
        }
        sc = __fdividef(1.f, kscale);   // (MUFU.RCP: no IEEE slow-path call and its branches on the once-path;
      } else {                          //  exact for the power-of-two scales)
        sc = __fdividef(1.f, vscale);
      }
      if (live) {
        uint32_t w0 = 0, w1 = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          w0 |= (uint32_t)to_e4m3(x[e] * sc) << (8 * e);
          w1 |= (uint32_t)to_e4m3(x[4 + e] * sc) << (8 * e);
        }
        uint8_t* dst = (kind == 0 ? S.kn : S.vn) + jj * ROWB + hh * 64 + partL * 8;
        *reinterpret_cast<uint2*>(dst) = make_uint2(w0, w1);
        const int pos = seq + jj;
        if (pos >= start && pos < end) {
          const int64_t row = S.cloc[jj];
          uint8_t* g8 = const_cast<uint8_t*>(cache_row(kind == 0 ? p.k_cache : p.v_cache, p, row, (2 * pair + hh) * 64 + partL * 8));
          *reinterpret_cast<uint2*>(g8) = make_uint2(w0, w1);
        }
      }
    }
  }
  csync<WS, NT>();
  DBG(11);
  if (!WS && has_new) {
    // new-token rows of the tiles already issued (their cached rows are in flight with cp.async)
    for (int x = 0; x < (Cfg<L, FLAT>::S2 ? STAGES - 2 : STAGES - 1) && x < ntiles; ++x) {
      const int st = x % STAGES;
#pragma unroll
      for (int k = 0; k < TILE / RPP; ++k) {
        const int r = crow0 + RPP * k;
        const int pos = start + x * TILE + r;
        if (pos >= seq && pos < end) {
          const int sw = ((cchunk ^ (r & 7)) << 4);
          *reinterpret_cast<uint4*>(S.k + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + cchunk * 16);
          *reinterpret_cast<uint4*>(S.v + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + cchunk * 16);
        }
      }
    }
    if (L == 8) fence_proxy_async();   // new keys written with st.shared, read by the QK wgmma
  }

  if constexpr (!FLAT && OKC_FINWARM) {   // warm the cold last-CTA finalize code while the first tiles load
    if (p.cs <= 1 && !p.coop && p.nc == 1) fin_nc1<L, NT>(p, b, pair, bp, 0.f, 1);
  }
  DBG(2);
  // ---- consumer state ----
  constexpr int MT = 1;                    // m-tiles per warp
  constexpr int NROW = (L == 1) ? 1 : 2;   // rows per thread
  constexpr bool S2 = Cfg<L, FLAT>::S2;
  constexpr bool F2 = Cfg<L, FLAT>::F2;
  const int sid = (S2 || F2) ? (warp >> 3) : 0;            // two streams: warps 8..15
  // (OKC_INV2: lane-0 broadcasts, so ptxas keeps them in uniform registers instead of re-deriving them every tile)
  const int hk = OKC_INV2 ? __shfl_sync(0xffffffffu, (L == 1) ? (warp & 1) : ((warp & 7) >> 2), 0)
                          : ((L == 1) ? (warp & 1) : ((warp & 7) >> 2));      // L==8: warpgroup = KV head
  const int wsub = OKC_INV2 ? __shfl_sync(0xffffffffu, (L == 1) ? (warp >> 1) : (warp & 3), 0)
                            : ((L == 1) ? (warp >> 1) : (warp & 3));    // L==1: key half; L==8: m-tile (tokens 2*wsub, 2*wsub+1)
  const int wgi = (S2 || F2) ? 2 * sid + hk : hk;          // named barrier of this warpgroup
  float o[MT][8][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int t = 0; t < 8; ++t)
#pragma unroll
      for (int e = 0; e < 4; ++e) o[mt][t][e] = 0.f;
  float m_r[NROW], l_r[NROW];
#pragma unroll
  for (int r = 0; r < NROW; ++r) { m_r[r] = -INFINITY; l_r[r] = 0.f; }
  // tiles with no window / causal / range edge skip the per-element mask
  const int lo_all = (L == 1 || !p.window) ? start : max(start, seq + (L - 1) - p.window + 1);
  const int hi_all = OKC_INV2 ? __shfl_sync(0xffffffffu, min(end - 1, seq), 0) : min(end - 1, seq);

  if constexpr (L == 1) {
  // Q A-fragments (f16, pre-scaled to log2 units) for 4 k-steps of 16 dims: thread holds dims
  // 16*tig + 4*ks + {0,1} (a0) and + {2,3} (a2); the K B-fragments use the same dims.
  uint32_t qa[4][2];
  {
    const uint8_t* qs = WS ? S.kvq + hk * 8 * 128 : (hk ? S.v : S.k) + (STAGES - 1) * STAGE_BYTES;
    const uint4 l0 = *reinterpret_cast<const uint4*>(qs + g * 128 + 32 * tig);
    const uint4 l1 = *reinterpret_cast<const uint4*>(qs + g * 128 + 32 * tig + 16);
    // (scout: chunk words stored (0, 2, 1, 3): l0 = {ks0 a0, ks1 a0, ks0 a2, ks1 a2}, l1 the same for ks 2, 3)
    qa[0][0] = l0.x; qa[0][1] = l0.y; qa[1][0] = l0.z; qa[1][1] = l0.w;
    qa[2][0] = l1.x; qa[2][1] = l1.y; qa[3][0] = l1.z; qa[3][1] = l1.w;
  }

  PROF_INIT
  for (int i = 0; i < ntiles; ++i) {
    if constexpr (WS) {
      if (ws_acquire(i, (tb + i) % STAGES)) csync<WS, NT>();
      PROF_MARK(0);
      if (i == 0) DBG(12);
    } else {
      cp_wait<STAGES - 2>();
#if OKC_TMA
      if (p.tma) mbar_wait(&S.mbar[i % STAGES], (i / STAGES) & 1);
#endif
      __syncthreads();
      PROF_MARK(0);
      if (i == 0) DBG(12);
      if (i + STAGES - 1 < ntiles) issue_kv(i + STAGES - 1, true);
      issue_idx(i + 2 * STAGES - 2);
      cp_commit();
      PROF_MARK(1);
    }

#if OKC_NOCOMPUTE
    if constexpr (WS) { __syncwarp(); if (OKC_XP_ARRALL || lane == 0) mbar_arrive(&S.empty[(tb + i) % STAGES]); }
    continue;
#endif
    const uint8_t* K = S.k + ((tb + i) % STAGES) * STAGE_BYTES;
    const uint8_t* V = S.v + ((tb + i) % STAGES) * STAGE_BYTES;
    const int t0 = start + i * TILE;
    const bool interior = (t0 >= lo_all) && (t0 + TILE - 1 <= hi_all);
    // warp: kv head hk, keys [32*wsub, 32*wsub+32) of the tile
    float s[4][4];
    uint32_t kw[4][4];
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int key = 32 * wsub + 8 * nt + g;
      const uint4 kb = lds128(K + key * ROWB + (((hk * 4 + tig) ^ (key & 7)) << 4));
      kw[nt][0] = kb.x; kw[nt][1] = kb.y; kw[nt][2] = kb.z; kw[nt][3] = kb.w;
    }
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {   // (scout: unpadded quads; even k-steps -> rows g of se, odd -> rows g+8 of so)
      float se[4] = {0.f, 0.f, 0.f, 0.f}, so[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        mma_f16(se, qa[2 * h][0], qa[2 * h][1], qa[2 * h + 1][0], qa[2 * h + 1][1], cvt2(kw[nt][2 * h]), cvt2h(kw[nt][2 * h]));
        mma_f16(so, qa[2 * h][0], qa[2 * h][1], qa[2 * h + 1][0], qa[2 * h + 1][1], cvt2(kw[nt][2 * h + 1]), cvt2h(kw[nt][2 * h + 1]));
      }
      s[nt][0] = se[0] + so[2]; s[nt][1] = se[1] + so[3]; s[nt][2] = 0.f; s[nt][3] = 0.f;
    }
    PROF_MARK(2);
    if (!interior) {
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int pos = t0 + 32 * wsub + 8 * nt + 2 * tig + e;
          if (pos >= end) s[nt][e] = -INFINITY;
        }
    }
    float mx = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) mx = fmaxf(mx, fmaxf(s[nt][0], s[nt][1]));
#if OKC_LAZY1
    // lazy offset (log2 units, in m_r): P = 2^(s - m_r) <= 2^8 in f16; the offset moves (with a rescale of o and l)
    // only when a score passes it by 8 or on the first tile with keys. The check needs no row reduction: any such
    // score shows in its own thread's partial max (warp-uniform vote, then quad shuffles only when needed).
    if (__any_sync(0xffffffffu, (m_r[0] == -INFINITY) || (mx > m_r[0] + 8.f))) {
      const float mq = quad_max(mx);
      const float nd = (m_r[0] == -INFINITY) ? mq : fmaxf(mq, m_r[0]);
      if (m_r[0] != -INFINITY && nd != m_r[0]) {
        const float alpha = ex2(m_r[0] - nd);
        l_r[0] *= alpha;
#pragma unroll
        for (int t = 0; t < 8; ++t) { o[0][t][0] *= alpha; o[0][t][1] *= alpha; }
      }
      m_r[0] = nd;
    }
    const float m_new = m_r[0];
    if (m_new != -INFINITY) {
      float ls = 0.f;
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float pv = ex2(s[nt][e] - m_new);
          s[nt][e] = pv;
          ls += pv;
        }
      l_r[0] += ls;
#else
    mx = quad_max(mx);
    const float m_new = fmaxf(m_r[0], mx);
    if (m_new != -INFINITY) {
      const float alpha = ex2(m_r[0] - m_new);
      m_r[0] = m_new;
      float ls = 0.f;
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float pv = ex2(s[nt][e] - m_new);
          s[nt][e] = pv;
          ls += pv;
        }
      if constexpr (ONE) {   // (scout: the only tile: o = 0 and l = 0, alpha would only multiply zeros)
        l_r[0] = ls;
      } else {
        l_r[0] = l_r[0] * alpha + ls;
#pragma unroll
        for (int t = 0; t < 8; ++t) { o[0][t][0] *= alpha; o[0][t][1] *= alpha; }
      }
#endif
#pragma unroll
      for (int ks = 0; ks < (OKC_XHALFPV ? 1 : 2); ++ks) {   // (OKC_XHALFPV: timing probe only, wrong results)
        uint32_t b0[8], b1[8];
        load_v_frags(V, 32 * wsub + 16 * ks, hk, g, tig, b0, b1);
        const uint32_t a0 = pack_h2(s[2 * ks][0], s[2 * ks][1]);
        const uint32_t a2 = pack_h2(s[2 * ks + 1][0], s[2 * ks + 1][1]);
#pragma unroll
        for (int t = 0; t < 8; ++t) mma_f16(o[0][t], a0, 0u, a2, 0u, b0[t], b1[t]);
      }
    }
    if constexpr (WS) {   // this warp is done with the stage
      __syncwarp();
      if (OKC_XP_ARRALL || lane == 0) mbar_arrive(&S.empty[(tb + i) % STAGES]);
    }
    PROF_MARK(3);
    PROF_TILE;
    if (i == 0) DBG(13);
#if OKC_SEGDBG
    if (p.dbg && tid == 0 && (i & 15) == 15) {   // progress marks: time and (request, key) every 16 tiles
      const int cta = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
      const int n = S.dbgn;
      if (n < 16) {
        p.dbg[cta * DBGSTRIDE + 32 + n] = gtime();
        p.dbg[cta * DBGSTRIDE + 48 + n] = (long long)b * 1000000 + start + (i + 1) * TILE;
        S.dbgn = n + 1;
      }
    }
#endif
  }
  PROF_DUMP;
  pdl_launch_dependents();
  } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  // ---- L == 8 on sm_90: QK^T with FP8 wgmma straight from the FP8 K tile in shared memory (warpgroup =
  // KV head, 64 rows = 8 tokens x 8 heads, K-major 128B-swizzled B operand), PV with f16 mma.sync from a
  // per-tile f16 copy of V. Softmax with a lazily updated per-row offset; row sums from a ones column.
  uint8_t* V16 = S.kvq;
  const uint32_t v16_base = smem_u32(V16) + hk * 8192;
  uint32_t qa8[2][4];   // e4m3 A fragments, k-steps of 32 dims
  if constexpr (!QDIR) {
    const uint8_t* qs = (hk ? S.v : S.k) + (STAGES - 1) * STAGE_BYTES;   // f16 staging, row = j*8 + g
    const int r0 = 16 * wsub + g, r1 = r0 + 8;
    // producer mode: in place in S.kvq, row j*16 + hk*8 + g
    const uint8_t* q0 = WS ? S.kvq + ((r0 >> 3) * 16 + hk * 8 + (r0 & 7)) * 128 : qs + r0 * 128;
    const uint8_t* q1 = WS ? S.kvq + ((r1 >> 3) * 16 + hk * 8 + (r1 & 7)) * 128 : qs + r1 * 128;
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {        // dims 32*ks + 16*hf + 4*tig .. +3
        const int dofs = (32 * ks + 16 * hf + 4 * tig) * 2;
        const uint2 lo = *reinterpret_cast<const uint2*>(q0 + dofs);
        const uint2 hi = *reinterpret_cast<const uint2*>(q1 + dofs);
        qa8[ks][2 * hf] = pack_e4m3x4(lo.x, lo.y);
        qa8[ks][2 * hf + 1] = pack_e4m3x4(hi.x, hi.y);
      }
    }
    if constexpr (PV8ON && OKC_QSS && (!FLAT || OKC_PV8FLAT)) {   // the A tile for the SS QK^T wgmma
#pragma unroll
      for (int ks = 0; ks < 2; ++ks)
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          const int kb = 32 * ks + 16 * hf + 4 * tig;
          *reinterpret_cast<uint32_t*>(S.q8s + hk * 4096 + r0 * 64 + (((kb >> 4) ^ ((r0 >> 1) & 3)) << 4) + (kb & 15)) =
              qa8[ks][2 * hf];
          *reinterpret_cast<uint32_t*>(S.q8s + hk * 4096 + r1 * 64 + (((kb >> 4) ^ ((r1 >> 1) & 3)) << 4) + (kb & 15)) =
              qa8[ks][2 * hf + 1];
        }
      fence_proxy_async();   // read by the wgmma after the loop's first barrier
    }
    if constexpr (PTR) {   // q staging read, S.kn / S.vn staged: the producers may write V^T into S.kvq
      __syncwarp();
      if (lane == 0) mbar_arrive(&S.qdone);
    }
  }
  float cr[2] = {0.f, 0.f};   // per-row offsets (log2 units), rows g and g+8
  const float ssc = OKC_Q8RAW ? sl2 : 1.f;   // raw score -> log2 units
  // OKC_RAWTHR: the lazy-max test on raw scores against (offset + 8) / ssc (ssc > 0), updated with the offset
  const float rssc = OKC_RAWTHR ? __fdividef(1.f, ssc) : 1.f;
  float thr[2] = {0.f, 0.f};
  float lacc[4] = {0.f, 0.f, 0.f, 0.f};
  bool first = true;
  if constexpr (OKC_OZF) {   // the zero init stays before the loop: ptxas had sunk these moves into the first QK^T
    float* oz = &o[0][0][0];   // wgmma's pipeline stage (warning C7515: every wgmma of the kernel serialized)
#pragma unroll
    for (int e = 0; e < 32; ++e) reg_fence(oz[e]);
#pragma unroll
    for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
  }
  const uint32_t ONES = 0x3C003C00u;   // f16 {1, 1}
  const int ckey = tid >> 2, cq = tid & 3;   // V conversion: key, two 16-byte chunks
  // FP8 PV: the per-CTA kernels (the flat kernel's register budget serializes its wgmmas: measured slower)
  constexpr bool PV8 = PV8ON && (!FLAT || OKC_PV8FLAT);
  // transpose roles (warpgroup = head): warp wq takes keys 16wq..16wq+15, lane = (q, w) of the ldmatrix fragments
  const int vwq = warp & 3, vq = lane & 3, vw = lane >> 2;   // (warp: uniform, OKC_UWARP)
  const uint32_t vsel0 = (vw >= 4) ? 0x7531u : 0x6420u, vsel1 = (vw >= 4) ? 0x6420u : 0x7531u;
  const int vr0 = (vw >= 4) ? 1 : 0;   // rows d + vr0 then d + 1 - vr0 (bank spread)
  // hoisted addressing (registers are tight in the flat kernel): V^T row n = 16*dgrp + 2*vw + r has swizzle chunk
  // vwq ^ (vw & 3) for every dgrp, so the 8 stores per tile are one per-thread base plus immediates; the two
  // ldmatrix rows are key = 16*vwq + 8*(mat & 1) + (lane & 7) with chunk (4*hk + 2u + (mat >> 1)) ^ (lane & 7)
  const uint32_t vt_off0 = 128 * vw + ((vwq ^ (vw & 3)) << 4) + 4 * vq + 64 * vr0;
  const uint32_t vt_off1 = 128 * vw + ((vwq ^ (vw & 3)) << 4) + 4 * vq + 64 * (1 - vr0);
  const int vkey_t = 16 * vwq + 8 * ((lane >> 3) & 1) + (lane & 7);
  const uint32_t vld_off0 = vkey_t * ROWB + (((4 * hk + 0 + ((lane >> 3) >> 1)) ^ (lane & 7)) << 4);
  const uint32_t vld_off1 = vkey_t * ROWB + (((4 * hk + 2 + ((lane >> 3) >> 1)) ^ (lane & 7)) << 4);
  // OKC_STSM: the V^T stores as two stmatrix.x4 per thread (instead of eight 4-byte stores). For ldmatrix group u,
  // matrix m = 2*g2 + ab holds, in row j, the 16 key bytes of V^T row 16(2u+g2) + 2j + (ab ^ (j >= 4)) (the same
  // rows and the same half swap for bank spread as the per-thread stores: thread (q, w) supplies row j = w)
  const int stj = lane & 7, stm = lane >> 3;
  const uint32_t vst_off = 1024 * (stm >> 1) + 128 * stj + 64 * ((stm & 1) ^ (stj >> 2)) + ((vwq ^ (stj & 3)) << 4);
  // (per-request kernels: stmatrix measured vs16 -2.5%, vs1/vs32 -1.5%, vf1 -1%; the flat kernel: +0..1.5%)
  constexpr bool STSM_ON = OKC_STSM || (OKC_STSMPR && !FLAT);
  // P fragments of the in-flight PV wgmma: kept live (fenced) until the next wgmma wait, so the compiler cannot
  // hand their registers to anything the next tile writes before that
  uint32_t pa[2][4] = {{0u, 0u, 0u, 0u}, {0u, 0u, 0u, 0u}};
  // OKC_ADDR: per-tile addressing as loop-carried values instead of re-derived from (tb + i) every tile: stage and
  // full-barrier phase counters, wgmma descriptors as base + stage offset (the 14-bit address field takes it without
  // a carry: shared memory < 256 KB), the V^T buffer and its descriptor toggled, new-row tiles found once
  constexpr bool ADDR = OKC_ADDR && !S2 && !F2 && (WS || OKC_ADDRN);   // (OKC_ADDRN: per-request kernels too)
  constexpr bool PVSPLIT = OKC_PVSPLIT && !S2 && !F2 && PV8 && !PTR && !OKC_PROF;
  // OKC_LATESYNCN: per-request (non-producer) q8 kernels with the split PV and the consumers' own V^T
  constexpr bool LSYNCN = OKC_LATESYNCN && !WS && !FLAT && PV8 && !PTR && PVSPLIT && !OKC_XP_NOFENCE && !S2 && !F2;
  constexpr bool LSYNC = OKC_LATESYNC && OKC_PK2 && (OKC_PCHK && OKC_LAZYMAX && !OKC_RAWTHR && PV8 && !OKC_EXH8 && !OKC_PROF &&
                         !PTR && WS) && PVSPLIT && OKC_PVSPLIT == 1 && !OKC_XP_NOEXP && OKC_EXPPOLY == 0 &&
                         !(OKC_TURN && WS && PV8 && !S2 && !F2 && !OKC_PROF) && !OKC_XP_NOFENCE;
  int st_c = tb % STAGES;
  uint32_t ph_c = (uint32_t)(tb / STAGES) & 1u;
#if OKC_DSHFL
  // descriptor low words broadcast from lane 0: a shuffle result is uniform and is not rematerialized, so the
  // bases stay in uniform registers (the high words are constants); +st*512 / +2 never carry out of the low word
  const uint32_t qlo0 = __shfl_sync(0xffffffffu, (uint32_t)((smem_u32(S.q8s) + hk * 4096) >> 4) | 0x10000u, 0);
  const uint32_t klo0 = __shfl_sync(0xffffffffu, (uint32_t)((smem_u32(S.k) + hk * 64) >> 4) | 0x10000u, 0);
  const uint64_t qdesc0 = ((uint64_t)((512 >> 4) | (2u << 30)) << 32) | qlo0;
  const uint64_t kdesc0 = ((uint64_t)((1024 >> 4) | (1u << 30)) << 32) | klo0;
#else
  const uint64_t qdesc0 = desc_sw64(smem_u32(S.q8s) + hk * 4096);
  const uint64_t kdesc0 = desc_sw128(smem_u32(S.k) + hk * 64);
#endif
  const uint32_t vsrc0 = smem_u32(S.v);
  uint8_t* vtb_c = S.kvq + hk * 4608;
  const uintptr_t vtb_x = (uintptr_t)(S.kvq + hk * 4608) ^ (uintptr_t)(S.kvq + (2 + hk) * 4608);
  uint64_t vdesc_c = desc_sw64(smem_u32(S.kvq + hk * 4608));
  const uint64_t vdesc_x = vdesc_c ^ desc_sw64(smem_u32(S.kvq + (2 + hk) * 4608));
  const int i_new = seq > start ? (seq - start) / TILE : 0;   // first tile that can hold the block's new rows
  // OKC_IRANGE: interior tiles (no mask) are i in [i_lo, i_lo + i_span], found once (shuffled: kept uniform)
  uint32_t i_lo = 0, i_span = 0;
  if constexpr (OKC_IRANGE) {
    const int a = (lo_all - start + TILE - 1) / TILE;              // first tile with t0 >= lo_all (lo_all >= start)
    const int z = hi_all - (TILE - 1) - start;                      // last tile with t0 + 63 <= hi_all
    const int bz = z >= 0 ? z / TILE : -1;
    i_lo = __shfl_sync(0xffffffffu, bz >= a ? (uint32_t)a : 0x7fffffffu, 0);
    i_span = __shfl_sync(0xffffffffu, bz >= a ? (uint32_t)(bz - a) : 0u, 0);
  }

  // OKC_RARE (producer mode): the common tiles -- no mask, no new rows (every key before the block's first new
  // position seq) -- are i in [r_lo, r_lo + r_span], found once; the per-tile test is then one compare, and t0, the
  // exact mask test and the new-row patch run only on the rare tiles
  uint32_t r_lo = 0x7fffffffu, r_span = 0u;
  if constexpr (OKC_RARE && WS && ADDR) {
    const int hi2 = min(end - 1, seq - 1);
    const int a = (lo_all - start + TILE - 1) / TILE;
    const int z = hi2 - (TILE - 1) - start;
    const int bz = z >= 0 ? z / TILE : -1;
    r_lo = __shfl_sync(0xffffffffu, bz >= a ? (uint32_t)a : 0x7fffffffu, 0);
    r_span = __shfl_sync(0xffffffffu, bz >= a ? (uint32_t)(bz - a) : 0u, 0);
  }
  PROF_INIT
  // OKC_TURN (flat producer mode): the two KV heads' warpgroups take turns in the exponential phase (named barriers
  // 5 / 6), so the 4 consumer warps of an SMSP do not all queue on the MUFU unit at once (scout: the exps segment
  // takes ~650-820 cycles for ~110 instructions; both warpgroups start each tile together on the same stage)
  constexpr bool TURN = OKC_TURN && WS && PV8 && !S2 && !F2 && !OKC_PROF;
  if constexpr (TURN && OKC_TURN == 1) {
    if (hk == 1 && ntiles > 0) bar_arrive_n(5, 256);   // head 0 goes first
  }
  int i_cur = 0, cnt2 = 0;   // F2: this warpgroup's claimed tile, and how many it has taken
  if constexpr (F2) {
    if (tid == 0) { S.claim[0] = 0; S.claim[1] = 0; }
    csync<WS, NT>();   // the q staging (S.q8s) is complete for both streams; counters reset
    if ((tid & 127) == 0) S.nxt[wgi][0] = atomicAdd(&S.claim[hk], 1);
    wgsync(wgi);
    i_cur = S.nxt[wgi][0];
  }
  const int nit = S2 ? ((ntiles + 1) >> 1) : (F2 ? 0x7fffffff : ntiles);
  // OKC_PIPE8 (flat producer mode, FP8 PV, loop-carried addressing): tile i+1's stage wait and QK^T go out right after
  // tile i's PV; the scores live across the loop edge, so they are declared outside the loop in that build only
  constexpr bool PIPE8 = OKC_PIPE8 && WS && FLAT && ADDR && PV8 && OKC_QSS && OKC_RARE && PVSPLIT && OKC_PVSPLIT == 1 &&
                         LSYNC && !PTR && !F2 && !S2 && !OKC_PROF && !OKC_TURN && !OKC_NOCOMPUTE && OKC_WGDEC;
  float s_outer[PIPE8 ? 32 : 1];
  auto issue_qk = [&](int st_) {   // (PIPE8) the QK^T wgmmas of the tile in stage st_ into s_outer
    if constexpr (PIPE8) {
      wgmma_fence();
#if OKC_DSHFL
      const uint32_t kl = klo0 + (uint32_t)(st_ * (STAGE_BYTES >> 4));
      const uint64_t kd = ((uint64_t)((1024 >> 4) | (1u << 30)) << 32) | kl;
      const uint64_t kd2 = ((uint64_t)((1024 >> 4) | (1u << 30)) << 32) | (kl + 2u);
      const uint64_t qd2 = ((uint64_t)((512 >> 4) | (2u << 30)) << 32) | (qlo0 + 2u);
      wgmma_m64n64k32_e4m3_ss(s_outer, qdesc0, kd, 0);
      wgmma_m64n64k32_e4m3_ss(s_outer, qd2, kd2, 1);
#else
      const uint64_t kd = kdesc0 + (uint64_t)(st_ * (STAGE_BYTES >> 4));
      wgmma_m64n64k32_e4m3_ss(s_outer, qdesc0, kd, 0);
      wgmma_m64n64k32_e4m3_ss(s_outer, qdesc0 + 2, kd + 2, 1);
#endif
      wgmma_commit();
    }
  };
  if constexpr (PIPE8) {
    if (ntiles > 0) {   // tile 0: landed (+ new rows), both warpgroups meet (the q staging), its QK^T issued ahead
      mbar_wait_full(&S.full[st_c], ph_c);
      if (((uint32_t)0 - r_lo > r_span) && (0 >= i_new)) ws_patch_ni<L, NT, PTR>(&S, 0, st_c, start, seq, end, tid);
      csync<WS, NT>();
      issue_qk(st_c);
    }
  }
  constexpr int UNR8 = WS ? OKC_UNR8 : ((MODE & 3) == 1 ? OKC_UNRN : 1);   // (pragma arguments are not macro-expanded;
  // plain split-KV with many tiles per CTA (b32 q8 sliding: 9) gains from the unroll, 1-tile coop grids lose)
#pragma unroll (UNR8)
  for (int jj = 0; jj < nit; ++jj) {
    const int i = S2 ? 2 * jj + sid : (F2 ? i_cur : jj);   // S2: tiles 2jj and 2jj + 1; F2: claimed
    if (F2 && i >= ntiles) break;
    if constexpr (F2) {
      // the next claim, read after this tile's warpgroup barrier
      if ((tid & 127) == 0) S.nxt[wgi][(cnt2 + 1) & 1] = atomicAdd(&S.claim[hk], 1);
      const int st0 = (tb + i) % STAGES;
      mbar_wait_full(&S.full[st0], ((tb + i) / STAGES) & 1);   // tile i landed
      const int t0n = start + i * TILE;
      const int nlo = max(t0n, seq), nhi = min(t0n + TILE, end);
      if (nlo < nhi) {   // this head's half of the block's own new rows (same for the whole warpgroup)
        for (int e = (tid & 127); e < (nhi - nlo) * 4; e += 128) {
          const int pos = nlo + (e >> 2), c = 4 * hk + (e & 3), r = pos - t0n;
          const int sw = ((c ^ (r & 7)) << 4);
          *reinterpret_cast<uint4*>(S.k + st0 * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + c * 16);
          *reinterpret_cast<uint4*>(S.v + st0 * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + c * 16);
        }
        fence_proxy_async();
        wgsync(wgi);
      }
      PROF_MARK(0);
    } else if constexpr (WS) {
      // tile i landed (+ new rows); previous tile's PV of this warpgroup done (its V16 half is free). The two
      // warpgroups (one per KV head) only meet when both must: tile 0 (q staging) and tiles with new rows.
      bool patched;
      if constexpr (PIPE8) {
        patched = false;   // (waited for, patched and its QK^T issued when the previous tile's PV went out)
      } else if constexpr (ADDR) {
        mbar_wait_full(&S.full[st_c], ph_c);
        if constexpr (OKC_RARE) patched = ((uint32_t)i - r_lo > r_span) && (i >= i_new) && ws_patch_ni<L, NT, PTR>(&S, i, st_c, start, seq, end, tid);
        else if constexpr (OKC_PATCHNI && WS) patched = (i >= i_new) && ws_patch_ni<L, NT, PTR>(&S, i, st_c, start, seq, end, tid);
        else patched = (i >= i_new) && ws_patch(i, st_c);
      } else if constexpr (OKC_INEW) {   // new rows only in tiles >= i_new (found once per segment)
        mbar_wait_full(&S.full[(tb + i) % STAGES], ((tb + i) / STAGES) & 1);
        patched = (i >= i_new) && ws_patch(i, (tb + i) % STAGES);
      } else {
        patched = ws_acquire(i, (tb + i) % STAGES);
      }
      if constexpr (!PIPE8) {
        if (patched || i == 0 || !OKC_WGDEC) csync<WS, NT>();
        else if (!(PV8ON && OKC_PV8FLAT && OKC_NOSYNC0)) wgsync(hk);
      }
      // (FP8 PV: V^T is double-buffered and each warp's wgmma wait already covers the PV that read this tile's
      // buffer, so the tile needs no warpgroup barrier before its transpose)
      PROF_MARK(0);
      if (i == 0) DBG(12);
    } else if constexpr (S2) {
      cp_wait<2>();          // tiles 2jj and 2jj + 1 landed (one group per tile)
      __syncthreads();       // and both streams' previous PVs read their V^T (the stages of tiles 2jj-2, 2jj-1 are free)
      PROF_MARK(0);
      if (jj == 0) DBG(12);
      if (2 * jj + 4 < ntiles) issue_kv(2 * jj + 4, true);
      issue_idx(2 * jj + 2 * STAGES - 2);
      cp_commit();
      if (2 * jj + 5 < ntiles) issue_kv(2 * jj + 5, true);
      issue_idx(2 * jj + 2 * STAGES - 1);
      cp_commit();
      PROF_MARK(1);
      if (i >= ntiles) continue;   // odd tile count: the second stream sits out the last pair
    } else {
      if constexpr (!OKC_XNOWAIT) cp_wait<STAGES - 2>();   // (OKC_XNOWAIT: timing probe only, wrong results)
#if OKC_TMA
      if (p.tma) mbar_wait(&S.mbar[i % STAGES], (i / STAGES) & 1);
#endif
      __syncthreads();       // tile i landed; previous tile's PV done (V16 free)
      PROF_MARK(0);
      if (i == 0) DBG(12);
      if constexpr (!OKC_QKFIRST) {
        if (i + STAGES - 1 < ntiles) issue_kv(i + STAGES - 1, true);
        issue_idx(i + 2 * STAGES - 2);
        cp_commit();
      }
      PROF_MARK(1);
    }

#if OKC_NOCOMPUTE
    if constexpr (WS && ADDR) {   // (probe) the stage is released untouched: the load-bound tile time
      __syncwarp();
      if (OKC_XP_ARRALL || lane == 0) mbar_arrive(&S.empty[st_c]);
      if (++st_c == STAGES) { st_c = 0; ph_c ^= 1u; }
      vtb_c = reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(vtb_c) ^ vtb_x);
      vdesc_c ^= vdesc_x;
      continue;
    }
#endif
    const int st = ADDR ? st_c : (tb + i) % STAGES;
    const uint32_t kaddr = smem_u32(S.k + st * STAGE_BYTES) + hk * 64;
    if constexpr (PVSPLIT && OKC_PVSPLIT == 2) {   // the previous tile's PV ran across the stage wait
      wgmma_wait<0>();
      float* od = &o[0][0][0];
#pragma unroll
      for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
      for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
#pragma unroll
      for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
    }
    float s_inner[PIPE8 ? 1 : 32];
    float* s_sel = PIPE8 ? s_outer : s_inner;
    float (&s)[32] = *reinterpret_cast<float(*)[32]>(s_sel);
    if constexpr (!PIPE8) {
    wgmma_fence();
    if constexpr (PV8 && OKC_QSS && ADDR) {
#if OKC_DSHFL
      const uint32_t kl = klo0 + (uint32_t)(st * (STAGE_BYTES >> 4));
      const uint64_t kd = ((uint64_t)((1024 >> 4) | (1u << 30)) << 32) | kl;
      const uint64_t kd2 = ((uint64_t)((1024 >> 4) | (1u << 30)) << 32) | (kl + 2u);
      const uint64_t qd2 = ((uint64_t)((512 >> 4) | (2u << 30)) << 32) | (qlo0 + 2u);
      if constexpr (OKC_QKZ) wgmma_m64n64k32_e4m3_ss_z(s, qdesc0, kd);
      else wgmma_m64n64k32_e4m3_ss(s, qdesc0, kd, 0);
      wgmma_m64n64k32_e4m3_ss(s, qd2, kd2, 1);
#else
      const uint64_t kd = kdesc0 + (uint64_t)(st * (STAGE_BYTES >> 4));
      if constexpr (OKC_QKZ) wgmma_m64n64k32_e4m3_ss_z(s, qdesc0, kd);
      else wgmma_m64n64k32_e4m3_ss(s, qdesc0, kd, 0);
      wgmma_m64n64k32_e4m3_ss(s, qdesc0 + 2, kd + 2, 1);
#endif
    } else if constexpr (PV8 && OKC_QSS) {
      const uint32_t qaddr = smem_u32(S.q8s) + hk * 4096;
      if constexpr (OKC_QKZ) wgmma_m64n64k32_e4m3_ss_z(s, desc_sw64(qaddr), desc_sw128(kaddr));
      else wgmma_m64n64k32_e4m3_ss(s, desc_sw64(qaddr), desc_sw128(kaddr), 0);
      wgmma_m64n64k32_e4m3_ss(s, desc_sw64(qaddr + 32), desc_sw128(kaddr + 32), 1);
    } else {
      wgmma_m64n64k32_e4m3(s, qa8[0][0], qa8[0][1], qa8[0][2], qa8[0][3], desc_sw128(kaddr), 0);
      wgmma_m64n64k32_e4m3(s, qa8[1][0], qa8[1][1], qa8[1][2], qa8[1][3], desc_sw128(kaddr + 32), 1);
    }
    wgmma_commit();
    }   // (!PIPE8)
    if constexpr (OKC_QKFIRST && !WS && !Cfg<L, FLAT>::S2) {   // (OKC_QKFIRST) the copies for tile i + STAGES - 1
      if (i + STAGES - 1 < ntiles) issue_kv(i + STAGES - 1, true);
      issue_idx(i + 2 * STAGES - 2);
      cp_commit();
    }
    if constexpr (PV8 && FLAT && OKC_PVW1 && !PIPE8) {
      // the previous tile's PV (committed before this QK^T) is done: its P registers are free for the transpose
      // below (the flat kernel's 104-register consumers spilled there) and o / lacc are settled
      wgmma_wait<1>();
#pragma unroll
      for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
      float* od0 = &o[0][0][0];
#pragma unroll
      for (int e = 0; e < 32; ++e) reg_fence(od0[e]);
#pragma unroll
      for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
    }
    uint8_t* V16t = V16 + (OKC_PVW ? (i & 1) * 16384 : 0);   // PV reads it asynchronously: double buffer
    const uint32_t v16b = smem_u32(V16t) + hk * 8192;
    // this head's V^T for this tile (PV reads it asynchronously: double buffer per stream)
    uint8_t* vtb = ADDR ? vtb_c : S.kvq + ((S2 ? 2 * sid + (jj & 1) : (F2 ? 2 * sid + (cnt2 & 1) : (i & 1))) * 2 + hk) * 4608;
    if constexpr (PV8 && !PTR) {
      if (F2 ? cnt2 == 0 : jj == 0) {   // rows 64..71 of this head's two V^T buffers: e4m3 ones (0x38), so the PV
                                     // wgmma also produces the row sums (after the first barrier: the q staging shares this space)
        const int wt = tid & 127;
        *reinterpret_cast<uint2*>(S.kvq + ((((S2 || F2) ? 2 * sid : 0) + (wt >> 6)) * 2 + hk) * 4608 + 4096 + (wt & 63) * 8) =
            make_uint2(0x38383838u, 0x38383838u);
      }
    }
    if constexpr (PTR) {
      // the producers build V^T
    } else if constexpr (PV8) {   // meanwhile: this head's V tile -> V^T (FP8, 72 x 64, keys in P-register order: k = 4q + 2h + b for
        // key = 8h + 2q + b within each 16-key block), two ldmatrix.trans (b16 pairs) + byte permutes per thread
      const uint32_t vsrc = ADDR ? vsrc0 + st * STAGE_BYTES : smem_u32(S.v + st * STAGE_BYTES);
#pragma unroll
      for (int u = 0; u < (OKC_XP_NOTR ? 0 : 2); ++u) {
        uint32_t r0, r1, r2, r3;
        ldsm_x4_t(vsrc + (u ? (OKC_INV2 ? (vld_off0 ^ 32u) : vld_off1) : vld_off0), r0, r1, r2, r3);   // (scout: off1 == off0 ^ 32)
        // r0/r1: dims group 2u (h = 0 / 1), r2/r3: dims group 2u+1
        if constexpr (STSM_ON) {
          stsm_x4(smem_u32(vtb) + vst_off + 2048 * u, prmt(r0, r1, vsel0), prmt(r0, r1, vsel1), prmt(r2, r3, vsel0),
                  prmt(r2, r3, vsel1));
        } else {
#pragma unroll
        for (int g2 = 0; g2 < 2; ++g2) {
          const uint32_t R0 = g2 ? r2 : r0, R1 = g2 ? r3 : r1;
          const uint32_t w0 = prmt(R0, R1, vsel0), w1 = prmt(R0, R1, vsel1);
          *reinterpret_cast<uint32_t*>(vtb + vt_off0 + 1024 * (2 * u + g2)) = w0;   // row 16(2u+g2) + 2vw + vr0
          *reinterpret_cast<uint32_t*>(vtb + vt_off1 + 1024 * (2 * u + g2)) = w1;   // row 16(2u+g2) + 2vw + 1 - vr0
        }
        }
      }
      if (!OKC_XP_NOFENCE && !LSYNC && !LSYNCN) fence_proxy_async();   // V^T written with st.shared, read by the PV wgmma
    } else {   // meanwhile: this tile's V -> f16 (both heads; in producer mode each warpgroup converts its own head)
      const uint8_t* V = S.v + st * STAGE_BYTES;
      const bool dec = WS && OKC_WGDEC;
      const int wt = tid & 127;
      const int vkey = dec ? (wt >> 1) : ckey;
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int c = dec ? (hk * 4 + (wt & 1) * 2 + u) : (2 * cq + u);
        const int hd = c >> 2, fc = (c & 3) * 2;
        const int ckey = vkey;
        const uint4 vb = lds128(V + ckey * ROWB + ((c ^ (ckey & 7)) << 4));
        *reinterpret_cast<uint4*>(V16t + hd * 8192 + ckey * 128 + ((fc ^ (ckey & 7)) << 4)) =
            make_uint4(cvt2(vb.x), cvt2(vb.x >> 16), cvt2(vb.y), cvt2(vb.y >> 16));
        *reinterpret_cast<uint4*>(V16t + hd * 8192 + ckey * 128 + (((fc + 1) ^ (ckey & 7)) << 4)) =
            make_uint4(cvt2(vb.z), cvt2(vb.z >> 16), cvt2(vb.w), cvt2(vb.w >> 16));
      }
      if (OKC_PVW) fence_proxy_async();   // V16 written with st.shared, read by the PV wgmma
    }
    if constexpr (PTR) {
    } else if (LSYNC || LSYNCN) {   // (OKC_LATESYNC / N: fence + barrier just before the PV)
    } else if ((WS && OKC_WGDEC) || PV8) wgsync(wgi);   // this head's V16 / V^T ready
    else csync<WS, NT>();                          // V16 ready
    PROF_MARK(2);
    wgmma_wait<0>();
#pragma unroll
    for (int e = 0; e < 32; ++e) reg_fence(s[e]);
    if constexpr (PV8) {
#pragma unroll
      for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
    }
    if constexpr (WS) {   // K (wgmma) and V (conversion) of the stage are consumed
      if (OKC_XP_ARRALL || lane == 0) mbar_arrive(&S.empty[st]);
    }
    if constexpr (PTR) {   // the previous tile's PV is done with its V^T slot
      if (i >= 1 && lane == 0) mbar_arrive(&S.vte[(i - 1) & 1]);
    }
    PROF_MARK(3);

    const bool common_t = OKC_RARE && WS && ADDR && ((uint32_t)i - r_lo <= r_span);
    const int t0 = start + i * TILE;
    const bool interior = OKC_XP_NOMASK || common_t ||
                          (OKC_IRANGE ? ((uint32_t)i - i_lo <= i_span) : ((t0 >= lo_all) && (t0 + TILE - 1 <= hi_all)));
    if (!interior) {
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const int pj = seq + 2 * wsub + hf;
        const int vlo = p.window ? max(start, pj - p.window + 1) : start;
        const int vhi = min(end - 1, pj);
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int pos = t0 + 8 * nt + 2 * tig + e;
#if OKC_XP_MASKRO   // compile experiment only: read the scores instead of writing them
            if (pos < vlo || pos > vhi) asm volatile("" ::"f"(s[4 * nt + 2 * hf + e]));
#else
            if (pos < vlo || pos > vhi) s[4 * nt + 2 * hf + e] = -INFINITY;
#endif
          }
      }
    }
#if OKC_PROF && OKC_PROFDEP
    {   // diagnostic: when do the QK results actually arrive?
      float t = __shfl_sync(0xffffffffu, fmaxf(s[0], s[31]), 0);
      asm volatile("" ::"f"(t) : "memory");
      PROF_MARK(3);
    }
#endif
    // OKC_PCHK: after a segment's first tile the offset test runs on the packed P instead of on 32 row-max steps: an
    // e4m3 byte above 0x78 (P > 256, i.e. a score past the offset + 8) sets bit 7 when 0x07 is added to every byte
    // (P >= 0: no carry between bytes); the (rare) hit moves the offset and repacks that half (see the PV loop)
    constexpr bool PCHK = OKC_PCHK && OKC_LAZYMAX && !OKC_RAWTHR && PV8 && !OKC_EXH8 && !OKC_PROF && !PTR && WS;
    const bool chk = PCHK && !first;
    float mlo = -INFINITY, mhi = -INFINITY;
    if (!chk) {
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      mlo = fmaxf(mlo, fmaxf(s[4 * nt], s[4 * nt + 1]));
      mhi = fmaxf(mhi, fmaxf(s[4 * nt + 2], s[4 * nt + 3]));
    }
    }
#if OKC_LAZYMAX
    // the threshold check needs no row reduction: any element past the offset + 8 shows in its own thread's
    // partial max; only the (rare) offset move and the first tile reduce over the row's quad
    if (!OKC_RAWTHR && !chk) {
      mlo *= ssc;   // scores -> log2 units (ssc > 0; ssc = 1 when the query was pre-scaled)
      mhi *= ssc;
    }
#else
    mlo = quad_max(mlo) * ssc;   // scores -> log2 units (ssc = 1 when the query was pre-scaled)
    mhi = quad_max(mhi) * ssc;
#endif
    float dlo, dhi;
    if (first) {
      if (OKC_LAZYMAX) { mlo = quad_max(mlo); mhi = quad_max(mhi); }
      if (OKC_LAZYMAX && OKC_RAWTHR) { mlo *= ssc; mhi *= ssc; }
      dlo = (mlo == -INFINITY) ? 0.f : mlo;
      dhi = (mhi == -INFINITY) ? 0.f : mhi;
      cr[0] = dlo; cr[1] = dhi;
      if (OKC_RAWTHR) { thr[0] = (dlo + 8.f) * rssc; thr[1] = (dhi + 8.f) * rssc; }
      first = false;
    } else {
      dlo = cr[0]; dhi = cr[1];
      if (!PCHK && __any_sync(0xffffffffu, (OKC_LAZYMAX && OKC_RAWTHR) ? (mlo > thr[0] || mhi > thr[1])
                                                                       : (mlo > cr[0] + 8.f || mhi > cr[1] + 8.f))) {
        // rare: a score grew past 2^8 relative to the offset -> move the offset up, rescale the state
        if (OKC_LAZYMAX) { mlo = quad_max(mlo); mhi = quad_max(mhi); }   // (warp-uniform branch: full-mask shuffles)
        if (OKC_LAZYMAX && OKC_RAWTHR) { mlo *= ssc; mhi *= ssc; }
        const float nlo = fmaxf(mlo, cr[0]), nhi = fmaxf(mhi, cr[1]);
        const float alo = ex2(cr[0] - nlo), ahi = ex2(cr[1] - nhi);
        // (the previous tile's PV wgmma completed at this tile's wgmma wait)
#if !OKC_XP_NORESC
#pragma unroll
        for (int t = 0; t < 8; ++t) { o[0][t][0] *= alo; o[0][t][1] *= alo; o[0][t][2] *= ahi; o[0][t][3] *= ahi; }
        lacc[0] *= alo; lacc[1] *= alo; lacc[2] *= ahi; lacc[3] *= ahi;   // (FP8 PV: the row sums live here too)
#endif
        cr[0] = nlo; cr[1] = nhi;
        dlo = nlo; dhi = nhi;
        if (OKC_RAWTHR) { thr[0] = (nlo + 8.f) * rssc; thr[1] = (nhi + 8.f) * rssc; }
      }
    }
    PROF_MARK(4);
    if constexpr (TURN && OKC_TURN == 1) bar_sync_n(5 + hk, 256);   // my turn for the exponentials
    if constexpr (TURN && OKC_TURN == 2) {   // (2: one offset only: head 1's first exponentials after head 0's)
      if (hk == 1 && jj == 0) bar_sync_n(6, 256);
    }
#if OKC_PROF && !OKC_PROFPV8
    // diagnostic split: all exponentials first (p5), then the MMAs (p6)
    {
      uint32_t pa[4][4];
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        pa[ks][0] = ex2h2(fmaf(s[8 * ks], ssc, -dlo), fmaf(s[8 * ks + 1], ssc, -dlo));
        pa[ks][1] = ex2h2(fmaf(s[8 * ks + 2], ssc, -dhi), fmaf(s[8 * ks + 3], ssc, -dhi));
        pa[ks][2] = ex2h2(fmaf(s[8 * ks + 4], ssc, -dlo), fmaf(s[8 * ks + 5], ssc, -dlo));
        pa[ks][3] = ex2h2(fmaf(s[8 * ks + 6], ssc, -dhi), fmaf(s[8 * ks + 7], ssc, -dhi));
      }
#pragma unroll
      for (int ks = 0; ks < 4; ++ks)
#pragma unroll
        for (int e = 0; e < 4; ++e) asm volatile("" : "+r"(pa[ks][e]));
      PROF_MARK(5);
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        const int mi = lane >> 3;
        const int key = 16 * ks + (mi & 1) * 8 + (lane & 7);
#pragma unroll
        for (int dp = 0; dp < 4; ++dp) {
          uint32_t b0a, b1a, b0b, b1b;
          ldsm_x4_t(v16b + key * 128 + (((2 * dp + (mi >> 1)) ^ (key & 7)) << 4), b0a, b1a, b0b, b1b);
          mma_f16(o[0][2 * dp], pa[ks][0], pa[ks][1], pa[ks][2], pa[ks][3], b0a, b1a);
          mma_f16(o[0][2 * dp + 1], pa[ks][0], pa[ks][1], pa[ks][2], pa[ks][3], b0b, b1b);
        }
        mma_f16(lacc, pa[ks][0], pa[ks][1], pa[ks][2], pa[ks][3], ONES, ONES);
      }
#pragma unroll
      for (int t = 0; t < 8; ++t)
#pragma unroll
        for (int e = 0; e < 4; ++e) asm volatile("" : "+f"(o[0][t][e]));
      PROF_MARK(6);
    }
#else
    if constexpr (PV8) {   // P (e4m3, registers, k-steps of 32 permuted keys) x V^T (e4m3, K-major, 72 columns: 64 dims + 8 copies of
        // the row sum) as an asynchronous wgmma; the next tile's wgmma wait covers it before P, o or V^T are reused
      constexpr bool PK2 = OKC_PK2 && PCHK && PVSPLIT && OKC_PVSPLIT == 1 && !OKC_EXH8 && !OKC_XP_NOEXP && OKC_EXPPOLY == 0 &&
                           !TURN && !PTR;
      if constexpr (PK2) {
        auto pack2 = [&]() {   // both halves of P from the scores (32 independent chains)
#pragma unroll
          for (int kk = 0; kk < 2; ++kk) {
            const float* sv = s + 16 * kk;
            pa[kk][0] = PKX(fmaf(sv[0], ssc, -dlo), fmaf(sv[1], ssc, -dlo), fmaf(sv[4], ssc, -dlo), fmaf(sv[5], ssc, -dlo));
            pa[kk][1] = PKX(fmaf(sv[2], ssc, -dhi), fmaf(sv[3], ssc, -dhi), fmaf(sv[6], ssc, -dhi), fmaf(sv[7], ssc, -dhi));
            pa[kk][2] = PKX(fmaf(sv[8], ssc, -dlo), fmaf(sv[9], ssc, -dlo), fmaf(sv[12], ssc, -dlo), fmaf(sv[13], ssc, -dlo));
            pa[kk][3] = PKX(fmaf(sv[10], ssc, -dhi), fmaf(sv[11], ssc, -dhi), fmaf(sv[14], ssc, -dhi), fmaf(sv[15], ssc, -dhi));
          }
        };
        pack2();
        if (chk) {
          // an e4m3 byte above 0x78 (P > 256: a score past its row's offset + 8) sets bit 7 when 0x07 is added
          constexpr uint32_t PB = OKC_PCHKTEST ? 0x47474747u : 0x07070707u;
          const uint32_t ov = ((pa[0][0] + PB) | (pa[0][1] + PB) | (pa[0][2] + PB) | (pa[0][3] + PB) |
                               (pa[1][0] + PB) | (pa[1][1] + PB) | (pa[1][2] + PB) | (pa[1][3] + PB)) & 0x80808080u;
          if (__any_sync(0xffffffffu, ov != 0u)) {
            // rare: offsets up to the tile's row maxima, state rescaled (no PV of this tile issued yet; the previous
            // tile's finished at its end), both halves repacked (P <= 1 afterwards: one pass)
            float hlo = -INFINITY, hhi = -INFINITY;
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
              hlo = fmaxf(hlo, fmaxf(s[4 * nt], s[4 * nt + 1]));
              hhi = fmaxf(hhi, fmaxf(s[4 * nt + 2], s[4 * nt + 3]));
            }
            hlo = quad_max(hlo) * ssc;
            hhi = quad_max(hhi) * ssc;
            const float nlo = fmaxf(hlo, cr[0]), nhi = fmaxf(hhi, cr[1]);
            const float alo = ex2(cr[0] - nlo), ahi = ex2(cr[1] - nhi);
#pragma unroll
            for (int t = 0; t < 8; ++t) { o[0][t][0] *= alo; o[0][t][1] *= alo; o[0][t][2] *= ahi; o[0][t][3] *= ahi; }
            lacc[0] *= alo; lacc[1] *= alo; lacc[2] *= ahi; lacc[3] *= ahi;
            cr[0] = nlo; cr[1] = nhi;
            dlo = nlo; dhi = nhi;
            pack2();
          }
        }
        float* odk = &o[0][0][0];
#pragma unroll
        for (int e = 0; e < 32; ++e) reg_fence(odk[e]);
#pragma unroll
        for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
#pragma unroll
        for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
        if constexpr (LSYNC) {   // every warp's V^T for this tile written (and visible to the async proxy)
          fence_proxy_async();
          wgsync(wgi);
        }
        wgmma_fence();
        if constexpr (ADDR) {
          wgmma_m64n72k32_e4m3(odk, lacc, pa[0], vdesc_c);
          wgmma_m64n72k32_e4m3(odk, lacc, pa[1], vdesc_c + 2);
        } else {
          wgmma_m64n72k32_e4m3(odk, lacc, pa[0], desc_sw64(smem_u32(vtb)));
          wgmma_m64n72k32_e4m3(odk, lacc, pa[1], desc_sw64(smem_u32(vtb) + 32));
        }
        wgmma_commit();
      }
#pragma unroll
      for (int kk = 0; kk < (PK2 ? 0 : 2); ++kk) {
        const float* sv = s + 16 * kk;
#if OKC_EXH8
        pa[kk][0] = ex2_e4m3x4_h(fmaf(sv[0], ssc, -dlo), fmaf(sv[1], ssc, -dlo), fmaf(sv[4], ssc, -dlo), fmaf(sv[5], ssc, -dlo));
        pa[kk][1] = ex2_e4m3x4_h(fmaf(sv[2], ssc, -dhi), fmaf(sv[3], ssc, -dhi), fmaf(sv[6], ssc, -dhi), fmaf(sv[7], ssc, -dhi));
        pa[kk][2] = ex2_e4m3x4_h(fmaf(sv[8], ssc, -dlo), fmaf(sv[9], ssc, -dlo), fmaf(sv[12], ssc, -dlo), fmaf(sv[13], ssc, -dlo));
        pa[kk][3] = ex2_e4m3x4_h(fmaf(sv[10], ssc, -dhi), fmaf(sv[11], ssc, -dhi), fmaf(sv[14], ssc, -dhi), fmaf(sv[15], ssc, -dhi));
        continue;
#endif
#if OKC_XP_NOEXP
#define ex2(x) (x)
#endif
#if OKC_EXPPOLY > 0
        // OKC_EXPPOLY of every 4 exponentials on the FMA pipe (positions 0..k-1 of each e4m3x4 group)
#define EXQ(x, q) ((q) < OKC_EXPPOLY ? ex2_poly(x) : ex2(x))
#else
#define EXQ(x, q) ex2(x)
#endif
        bool rchk = chk;   // one retry at most: after the offset move every finite P is <= 1 (NaN / garbage scores
                           // would otherwise retrigger the check forever: the SASS oracle's value-perturbing probe hung)
#pragma unroll 1
        for (;;) {   // (OKC_PCHK: repeated once after a rare offset move; otherwise a single pass)
        pa[kk][0] = e4m3x4f(EXQ(fmaf(sv[0], ssc, -dlo), 0), EXQ(fmaf(sv[1], ssc, -dlo), 1), EXQ(fmaf(sv[4], ssc, -dlo), 2), EXQ(fmaf(sv[5], ssc, -dlo), 3));
        pa[kk][1] = e4m3x4f(EXQ(fmaf(sv[2], ssc, -dhi), 0), EXQ(fmaf(sv[3], ssc, -dhi), 1), EXQ(fmaf(sv[6], ssc, -dhi), 2), EXQ(fmaf(sv[7], ssc, -dhi), 3));
        pa[kk][2] = e4m3x4f(EXQ(fmaf(sv[8], ssc, -dlo), 0), EXQ(fmaf(sv[9], ssc, -dlo), 1), EXQ(fmaf(sv[12], ssc, -dlo), 2), EXQ(fmaf(sv[13], ssc, -dlo), 3));
        pa[kk][3] = e4m3x4f(EXQ(fmaf(sv[10], ssc, -dhi), 0), EXQ(fmaf(sv[11], ssc, -dhi), 1), EXQ(fmaf(sv[14], ssc, -dhi), 2), EXQ(fmaf(sv[15], ssc, -dhi), 3));
        if constexpr (!PCHK) break;
        else {
          if (!rchk) break;
          {
            // (OKC_PCHKTEST: test hook, a hit at P > 1 instead of P > 256: exercises the offset move on most tiles)
            constexpr uint32_t PB = OKC_PCHKTEST ? 0x47474747u : 0x07070707u;
            const uint32_t ov = ((pa[kk][0] + PB) | (pa[kk][1] + PB) | (pa[kk][2] + PB) | (pa[kk][3] + PB)) & 0x80808080u;
            if (!__any_sync(0xffffffffu, ov != 0u)) break;
            {
              // rare: a score of this half passed its row's offset + 8: offsets up to the half's row maxima,
              // state rescaled (half 0's PV wgmma finished first), this half repacked
              float hlo = fmaxf(fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[4], sv[5])), fmaxf(fmaxf(sv[8], sv[9]), fmaxf(sv[12], sv[13])));
              float hhi = fmaxf(fmaxf(fmaxf(sv[2], sv[3]), fmaxf(sv[6], sv[7])), fmaxf(fmaxf(sv[10], sv[11]), fmaxf(sv[14], sv[15])));
              hlo = quad_max(hlo) * ssc;
              hhi = quad_max(hhi) * ssc;
              const float nlo = fmaxf(hlo, cr[0]), nhi = fmaxf(hhi, cr[1]);
              const float alo = ex2(cr[0] - nlo), ahi = ex2(cr[1] - nhi);
              if (PVSPLIT && (kk == 1 || OKC_PVSPLIT == 2)) {   // (a PV wgmma may still accumulate into o)
                wgmma_wait<0>();
                float* odw = &o[0][0][0];
#pragma unroll
                for (int e = 0; e < 32; ++e) reg_fence(odw[e]);
#pragma unroll
                for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
              }
#pragma unroll
              for (int t = 0; t < 8; ++t) { o[0][t][0] *= alo; o[0][t][1] *= alo; o[0][t][2] *= ahi; o[0][t][3] *= ahi; }
              lacc[0] *= alo; lacc[1] *= alo; lacc[2] *= ahi; lacc[3] *= ahi;
              cr[0] = nlo; cr[1] = nhi;
              dlo = nlo; dhi = nhi;
              rchk = false;
            }
          }
        }
        }   // (pack / check loop)
#undef EXQ
#if OKC_XP_NOEXP
#undef ex2
#endif
        if constexpr (PVSPLIT) {   // each half of P goes to the tensor cores as soon as it is packed
          float* od = &o[0][0][0];
          if constexpr (LSYNCN) {   // every warp's V^T for this tile written (and visible to the async proxy)
            if (kk == 0) {
              fence_proxy_async();
              wgsync(wgi);
            }
          }
          if (kk == 0) {
#pragma unroll
            for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
            for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
          }
#pragma unroll
          for (int e = 0; e < 4; ++e) asm volatile("" : "+r"(pa[kk][e]));
          wgmma_fence();
          if constexpr (ADDR) wgmma_m64n72k32_e4m3(od, lacc, pa[kk], vdesc_c + 2 * kk);
          else wgmma_m64n72k32_e4m3(od, lacc, pa[kk], desc_sw64(smem_u32(vtb) + 32 * kk));
          wgmma_commit();
        }
      }
      if constexpr (TURN && OKC_TURN == 1) {   // the other head's turn (head 1 skips its last hand-back)
        if (!(hk == 1 && i == ntiles - 1)) bar_arrive_n(6 - hk, 256);
      }
      if constexpr (TURN && OKC_TURN == 2) {
        if (hk == 0 && jj == 0) bar_arrive_n(6, 256);
      }
      float* od = &o[0][0][0];
      if constexpr (PVSPLIT) {
        if constexpr (PIPE8) {
          // next tile: stage advance, its wait (+ new rows: both warpgroups meet) and its QK^T, across this tile's PV
          // (the wgmma wait after the next transpose covers both; o / lacc / pa are fenced there)
          if (++st_c == STAGES) { st_c = 0; ph_c ^= 1u; }
          vtb_c = reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(vtb_c) ^ vtb_x);
          vdesc_c ^= vdesc_x;
          if (i + 1 < ntiles) {
            mbar_wait_full(&S.full[st_c], ph_c);
            const bool patched_n = ((uint32_t)(i + 1) - r_lo > r_span) && (i + 1 >= i_new) &&
                                   ws_patch_ni<L, NT, PTR>(&S, i + 1, st_c, start, seq, end, tid);
            if (patched_n) csync<WS, NT>();
            issue_qk(st_c);
          }
        } else if constexpr (OKC_PVSPLIT == 1) {   // done before the next tile (no wgmma in flight across the loop edge)
          wgmma_wait<0>();
#pragma unroll
          for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
          for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
#pragma unroll
          for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
        }
      } else {
#pragma unroll
      for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
      for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
#pragma unroll
      for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
      if constexpr (PTR) mbar_wait_full(&S.vtf[i & 1], (i >> 1) & 1);   // this tile's V^T is written
      wgmma_fence();
      if constexpr (ADDR) {
        wgmma_m64n72k32_e4m3(od, lacc, pa[0], vdesc_c);
        wgmma_m64n72k32_e4m3(od, lacc, pa[1], vdesc_c + 2);
      } else {
        const uint32_t vtaddr = smem_u32(vtb);
        wgmma_m64n72k32_e4m3(od, lacc, pa[0], desc_sw64(vtaddr));
        wgmma_m64n72k32_e4m3(od, lacc, pa[1], desc_sw64(vtaddr + 32));
      }
      wgmma_commit();
      if (OKC_PV8SYNC) {
        wgmma_wait<0>();
#pragma unroll
        for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
        for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
      }
      }   // (!PVSPLIT)
    } else {
#if OKC_PVW
    {   // P (registers) x V16 (shared, MN-major) as an asynchronous wgmma: it completes while the next tile's
        // QK^T and V conversion run (the wgmma wait there covers it before P, o or this V16 buffer are reused)
      uint32_t pa[4][4];
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        pa[ks][0] = ex2h2(fmaf(s[8 * ks], ssc, -dlo), fmaf(s[8 * ks + 1], ssc, -dlo));
        pa[ks][1] = ex2h2(fmaf(s[8 * ks + 2], ssc, -dhi), fmaf(s[8 * ks + 3], ssc, -dhi));
        pa[ks][2] = ex2h2(fmaf(s[8 * ks + 4], ssc, -dlo), fmaf(s[8 * ks + 5], ssc, -dlo));
        pa[ks][3] = ex2h2(fmaf(s[8 * ks + 6], ssc, -dhi), fmaf(s[8 * ks + 7], ssc, -dhi));
      }
      float* od = &o[0][0][0];
#pragma unroll
      for (int e = 0; e < 32; ++e) reg_fence(od[e]);
      wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < 4; ++ks)
        wgmma_m64n64k16_f16_tb(od, pa[ks][0], pa[ks][1], pa[ks][2], pa[ks][3], desc_sw128(v16b + ks * 2048));
      wgmma_commit();
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) mma_f16(lacc, pa[ks][0], pa[ks][1], pa[ks][2], pa[ks][3], ONES, ONES);
    }
    PROF_MARK(5);
#else
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const uint32_t a0 = ex2h2(fmaf(s[8 * ks], ssc, -dlo), fmaf(s[8 * ks + 1], ssc, -dlo));
      const uint32_t a1 = ex2h2(fmaf(s[8 * ks + 2], ssc, -dhi), fmaf(s[8 * ks + 3], ssc, -dhi));
      const uint32_t a2 = ex2h2(fmaf(s[8 * ks + 4], ssc, -dlo), fmaf(s[8 * ks + 5], ssc, -dlo));
      const uint32_t a3 = ex2h2(fmaf(s[8 * ks + 6], ssc, -dhi), fmaf(s[8 * ks + 7], ssc, -dhi));
      const int mi = lane >> 3;
      const int key = 16 * ks + (mi & 1) * 8 + (lane & 7);
#pragma unroll
      for (int dp = 0; dp < 4; ++dp) {
        uint32_t b0a, b1a, b0b, b1b;
        ldsm_x4_t(v16b + key * 128 + (((2 * dp + (mi >> 1)) ^ (key & 7)) << 4), b0a, b1a, b0b, b1b);
        mma_f16(o[0][2 * dp], a0, a1, a2, a3, b0a, b1a);
        mma_f16(o[0][2 * dp + 1], a0, a1, a2, a3, b0b, b1b);
      }
      mma_f16(lacc, a0, a1, a2, a3, ONES, ONES);
    }
#endif
    }
    PROF_MARK(5);
#endif
    PROF_TILE;
    if (i == 0) DBG(13);
    if constexpr (F2) {
      i_cur = S.nxt[wgi][(cnt2 + 1) & 1];   // written before this tile's warpgroup barrier
      ++cnt2;
    }
#if OKC_YIELD > 0
    // flat grids, two CTAs per SM: the first-dispatched CTA (lower half of the grid) wins the warp schedulers and
    // the other one runs mostly after it (measured slow/fast 1.3 on every SM); its consumers yield a little per
    // tile so both finish together
    if constexpr (FLAT && WS) {
      if (blockIdx.x * 2 < gridDim.x) __nanosleep(OKC_YIELD);
    }
#endif
    if constexpr (ADDR && !PIPE8) {
      if (++st_c == STAGES) { st_c = 0; ph_c ^= 1u; }
      vtb_c = reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(vtb_c) ^ vtb_x);
      vdesc_c ^= vdesc_x;
    }
  }
  PROF_DUMP;
  pdl_launch_dependents();
  if constexpr ((OKC_PVW && !OKC_PROF) || PV8) {
  wgmma_wait<0>();
  if constexpr (PV8) {
#pragma unroll
    for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
  }
  {
    float* od = &o[0][0][0];
#pragma unroll
    for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
    for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
  }
  }
  m_r[0] = first ? -INFINITY : cr[0];
  m_r[1] = first ? -INFINITY : cr[1];
  l_r[0] = 0.25f * lacc[0];   // every lane of the quad holds the full row sum; the epilogue quad-sums
  l_r[1] = 0.25f * lacc[2];
  if constexpr (F2) {   // stream 1's warpgroup of each head hands its state to stream 0's (same rows)
    csync<WS, NT>();    // every tile of the segment is consumed: the K stages are free scratch
    float* xs = reinterpret_cast<float*>(S.k) + (hk * 128 + (tid & 127)) * 37;
    if (sid == 1) {
      xs[0] = m_r[0]; xs[1] = m_r[1]; xs[2] = l_r[0]; xs[3] = l_r[1];
#pragma unroll
      for (int t = 0; t < 8; ++t)
#pragma unroll
        for (int e = 0; e < 4; ++e) xs[4 + 4 * t + e] = o[0][t][e];
    }
    csync<WS, NT>();
    if (sid == 0) {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const float m1 = xs[r], mm = fmaxf(m_r[r], m1);
        const float f0 = (m_r[r] == -INFINITY) ? 0.f : ex2(m_r[r] - mm);
        const float f1 = (m1 == -INFINITY) ? 0.f : ex2(m1 - mm);
        m_r[r] = mm;
        l_r[r] = l_r[r] * f0 + xs[2 + r] * f1;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
          o[0][t][2 * r] = o[0][t][2 * r] * f0 + xs[4 + 4 * t + 2 * r] * f1;
          o[0][t][2 * r + 1] = o[0][t][2 * r + 1] * f0 + xs[4 + 4 * t + 2 * r + 1] * f1;
        }
      }
    }
  }
#else
  // ---- L == 8: 8 warps, warp = (kv head hk, m-tile wsub: tokens 2*wsub, 2*wsub+1 x 8 heads).
  // Each tile's FP8 K/V are converted once per CTA to f16 (S.kv16) and read with ldmatrix.
  // Softmax with a lazily updated per-row offset C (P = 2^(s - C) in f16, C only moves up when a
  // tile's max exceeds it by > 8); the row sums come from an extra MMA against a ones column.
  uint8_t* K16 = S.kvq;
  uint8_t* V16 = S.kvq + 2 * 64 * 128;
  const uint32_t k16_base = smem_u32(K16) + hk * 8192, v16_base = smem_u32(V16) + hk * 8192;
  uint32_t qa[4][4];
  {
    const uint32_t qs = smem_u32((hk ? S.v : S.k) + (STAGES - 1) * STAGE_BYTES);
    const int mi = lane >> 3, rr = lane & 7;
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const int row = 16 * wsub + (mi & 1) * 8 + rr;
      ldsm_x4(qs + row * 128 + (2 * ks + (mi >> 1)) * 16, qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3]);
    }
  }
  __syncthreads();   // q staging (last stage) is overwritten by the first issue below
  float cr[2] = {0.f, 0.f};   // per-row offsets (log2 units), rows g and g+8
  float lacc[4] = {0.f, 0.f, 0.f, 0.f};
  bool first = true;
  const uint32_t ONES = 0x3C003C00u;   // f16 {1, 1}
  // conversion: thread -> key (tid >> 2), FP8 chunks 2*(tid&3), 2*(tid&3)+1 of its 128-byte row
  const int ckey = tid >> 2, cq = tid & 3;

  for (int i = 0; i < ntiles; ++i) {
    cp_wait<STAGES - 2>();
    __syncthreads();   // tile i landed; previous tile's compute done (kv16 free)
    if (i + STAGES - 1 < ntiles) issue_kv(i + STAGES - 1, true);
    issue_idx(i + 2 * STAGES - 2);
    cp_commit();
    {
      const uint8_t* K = S.k + (i % STAGES) * STAGE_BYTES;
      const uint8_t* V = S.v + (i % STAGES) * STAGE_BYTES;
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int c = 2 * cq + u;               // fp8 chunk: head c >> 2, dims (c & 3)*16 .. +16
        const int hd = c >> 2, fc = (c & 3) * 2;
        const int src = ckey * ROWB + ((c ^ (ckey & 7)) << 4);
        const int dst0 = hd * 8192 + ckey * 128 + ((fc ^ (ckey & 7)) << 4);
        const int dst1 = hd * 8192 + ckey * 128 + (((fc + 1) ^ (ckey & 7)) << 4);
        const uint4 kb = lds128(K + src);
        const uint4 vb = lds128(V + src);
        *reinterpret_cast<uint4*>(K16 + dst0) = make_uint4(cvt2(kb.x), cvt2(kb.x >> 16), cvt2(kb.y), cvt2(kb.y >> 16));
        *reinterpret_cast<uint4*>(K16 + dst1) = make_uint4(cvt2(kb.z), cvt2(kb.z >> 16), cvt2(kb.w), cvt2(kb.w >> 16));
        *reinterpret_cast<uint4*>(V16 + dst0) = make_uint4(cvt2(vb.x), cvt2(vb.x >> 16), cvt2(vb.y), cvt2(vb.y >> 16));
        *reinterpret_cast<uint4*>(V16 + dst1) = make_uint4(cvt2(vb.z), cvt2(vb.z >> 16), cvt2(vb.w), cvt2(vb.w >> 16));
      }
    }
    __syncthreads();   // kv16 ready

    const int t0 = start + i * TILE;
    const bool interior = (t0 >= lo_all) && (t0 + TILE - 1 <= hi_all);
#pragma unroll 1
    for (int kh = 0; kh < 2; ++kh) {     // two halves of 32 keys
    float s[4][4];
    const float ilo = first ? 0.f : -cr[0], ihi = first ? 0.f : -cr[1];
#pragma unroll
    for (int nn = 0; nn < 4; ++nn) {
      const int nt = 4 * kh + nn;
      s[nn][0] = ilo; s[nn][1] = ilo; s[nn][2] = ihi; s[nn][3] = ihi;
      const int key = 8 * nt + (lane & 7);
#pragma unroll
      for (int h2 = 0; h2 < 2; ++h2) {
        uint32_t b0a, b1a, b0b, b1b;
        ldsm_x4(k16_base + key * 128 + (((4 * h2 + (lane >> 3)) ^ (key & 7)) << 4), b0a, b1a, b0b, b1b);
        mma_f16(s[nn], qa[2 * h2][0], qa[2 * h2][1], qa[2 * h2][2], qa[2 * h2][3], b0a, b1a);
        mma_f16(s[nn], qa[2 * h2 + 1][0], qa[2 * h2 + 1][1], qa[2 * h2 + 1][2], qa[2 * h2 + 1][3], b0b, b1b);
      }
    }
    if (!interior) {
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const int pj = seq + 2 * wsub + hf;
        const int vlo = p.window ? max(start, pj - p.window + 1) : start;
        const int vhi = min(end - 1, pj);
#pragma unroll
        for (int nn = 0; nn < 4; ++nn)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int pos = t0 + 32 * kh + 8 * nn + 2 * tig + e;
            if (pos < vlo || pos > vhi) s[nn][2 * hf + e] = -INFINITY;
          }
      }
    }
    float mlo = -INFINITY, mhi = -INFINITY;
#pragma unroll
    for (int nn = 0; nn < 4; ++nn) {
      mlo = fmaxf(mlo, fmaxf(s[nn][0], s[nn][1]));
      mhi = fmaxf(mhi, fmaxf(s[nn][2], s[nn][3]));
    }
    mlo = quad_max(mlo);
    mhi = quad_max(mhi);
    if (first) {
      // offsets from the first half-tile's row maxima (raw scores): every later score is relative to them
      const float nlo = (mlo == -INFINITY) ? 0.f : mlo, nhi = (mhi == -INFINITY) ? 0.f : mhi;
      cr[0] = nlo; cr[1] = nhi;
#pragma unroll
      for (int nn = 0; nn < 4; ++nn) { s[nn][0] -= nlo; s[nn][1] -= nlo; s[nn][2] -= nhi; s[nn][3] -= nhi; }
      first = false;
    } else if (__any_sync(0xffffffffu, mlo > 8.f || mhi > 8.f)) {
      // rare: a score grew past 2^8 relative to the offset -> move the offset up, rescale the state
      const float dlo = fmaxf(mlo, 0.f), dhi = fmaxf(mhi, 0.f);
      const float alo = ex2(-dlo), ahi = ex2(-dhi);
      cr[0] += dlo; cr[1] += dhi;
#pragma unroll
      for (int t = 0; t < 8; ++t) { o[0][t][0] *= alo; o[0][t][1] *= alo; o[0][t][2] *= ahi; o[0][t][3] *= ahi; }
      lacc[0] *= alo; lacc[1] *= alo; lacc[2] *= ahi; lacc[3] *= ahi;
#pragma unroll
      for (int nn = 0; nn < 4; ++nn) { s[nn][0] -= dlo; s[nn][1] -= dlo; s[nn][2] -= dhi; s[nn][3] -= dhi; }
    }
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
      const int ks = 2 * kh + kk;
      const uint32_t a0 = pack_h2(ex2(s[2 * kk][0]), ex2(s[2 * kk][1]));
      const uint32_t a1 = pack_h2(ex2(s[2 * kk][2]), ex2(s[2 * kk][3]));
      const uint32_t a2 = pack_h2(ex2(s[2 * kk + 1][0]), ex2(s[2 * kk + 1][1]));
      const uint32_t a3 = pack_h2(ex2(s[2 * kk + 1][2]), ex2(s[2 * kk + 1][3]));
      const int mi = lane >> 3;
      const int key = 16 * ks + (mi & 1) * 8 + (lane & 7);
#pragma unroll
      for (int dp = 0; dp < 4; ++dp) {
        uint32_t b0a, b1a, b0b, b1b;
        ldsm_x4_t(v16_base + key * 128 + (((2 * dp + (mi >> 1)) ^ (key & 7)) << 4), b0a, b1a, b0b, b1b);
        mma_f16(o[0][2 * dp], a0, a1, a2, a3, b0a, b1a);
        mma_f16(o[0][2 * dp + 1], a0, a1, a2, a3, b0b, b1b);
      }
      mma_f16(lacc, a0, a1, a2, a3, ONES, ONES);
    }
    }
  }
  m_r[0] = first ? -INFINITY : cr[0];
  m_r[1] = first ? -INFINITY : cr[1];
  l_r[0] = 0.25f * lacc[0];   // every lane of the quad holds the full row sum; the epilogue quad-sums
  l_r[1] = 0.25f * lacc[2];
#endif
  }

  DBG(3);
  // ---- merge: rescale to the fixed reference 2^0 and accumulate with FP32 atomics ----
  bool do_red = true;
  if constexpr (Cfg<L, FLAT>::F2) do_red = (warp >> 3) == 0;   // stream 1's state was merged into stream 0
  if (L == 1 && !CONT && !(OKC_NOXCH1 && !M_CL)) {
    // warps 2,3 (second key half) hand their state to warps 0,1 (same heads) through shared memory
    // (OKC_NOXCH1: outside cluster mode both halves add their own rescaled state to the workspace instead)
    cp_wait<0>();
    csync<WS, NT>();
    float* xs = reinterpret_cast<float*>(S.k) + ((warp & 1) * 32 + lane) * 18;
    if (wsub == 1) {
      xs[0] = m_r[0];
      xs[1] = l_r[0];
#pragma unroll
      for (int t = 0; t < 8; ++t) { xs[2 + 2 * t] = o[0][t][0]; xs[3 + 2 * t] = o[0][t][1]; }
    }
    csync<WS, NT>();
    if (wsub == 0) {
      const float m1 = xs[0], l1 = xs[1];
      const float mm = fmaxf(m_r[0], m1);
      const float f0 = (m_r[0] == -INFINITY) ? 0.f : ex2(m_r[0] - mm);
      const float f1 = (m1 == -INFINITY) ? 0.f : ex2(m1 - mm);
      m_r[0] = mm;
      l_r[0] = l_r[0] * f0 + l1 * f1;
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        o[0][t][0] = o[0][t][0] * f0 + xs[2 + 2 * t] * f1;
        o[0][t][1] = o[0][t][1] * f0 + xs[3 + 2 * t] * f1;
      }
    } else {
      do_red = false;
    }
  }
  if (M_CL) {
    cp_wait<0>();
    csync<WS, NT>();
    float* PB = reinterpret_cast<float*>(S.k);
    float* PL = PB + L * 16 * 64;
    // two tile streams (Cfg::S2): stream 0 writes its rows, then stream 1 (same rows) adds its own
#pragma unroll 1
    for (int ph = 0; ph < (Cfg<L, FLAT>::S2 ? 2 : 1); ++ph) {
    if (ph == 1) csync<WS, NT>();
    const bool mine = !Cfg<L, FLAT>::S2 || (warp >> 3) == ph;
#pragma unroll
    for (int r = 0; r < NROW; ++r) {
      const int mt = r >> 1, hf = r & 1;
      const int j = (L == 1) ? 0 : 2 * (wsub + mt) + hf;
      const int row = j * 16 + hk * 8 + g;
      const float lsum = quad_sum(l_r[r]);
      if (do_red && mine) {
        const float f = (m_r[r] == -INFINITY) ? 0.f : ex2(fminf(m_r[r], 126.f));
        if constexpr (L == 1) {   // permuted dims: n-tile t, column c <-> dim 16*tig + 8*c + t
#pragma unroll
          for (int c = 0; c < 2; ++c)
#pragma unroll
            for (int t4 = 0; t4 < 8; t4 += 4)
              *reinterpret_cast<float4*>(&PB[row * 64 + 16 * tig + 8 * c + t4]) =
                  make_float4(o[mt][t4][2 * hf + c] * f, o[mt][t4 + 1][2 * hf + c] * f,
                              o[mt][t4 + 2][2 * hf + c] * f, o[mt][t4 + 3][2 * hf + c] * f);
        } else {                  // natural dims: n-tile t, column c <-> dim 8*t + 2*tig + c
#pragma unroll
          for (int t = 0; t < 8; ++t) {
            float2* d2 = reinterpret_cast<float2*>(&PB[row * 64 + 8 * t + 2 * tig]);
            const float2 pv = ph ? *d2 : make_float2(0.f, 0.f);
            *d2 = make_float2(pv.x + o[mt][t][2 * hf] * f, pv.y + o[mt][t][2 * hf + 1] * f);
          }
        }
        if (tig == 0) PL[row] = (ph ? PL[row] : 0.f) + lsum * f;
      }
    }
    }
    csync<WS, NT>();
    cluster_finish<L>(p, S, b, pair, n_keys, chunk);
    return true;
  }
  if (do_red) {   // (one branch, not one per row: the post-loop merge runs straight-line from a cold cache)
#pragma unroll
  for (int r = 0; r < NROW; ++r) {
    const int mt = r >> 1, hf = r & 1;
    const int j = (L == 1) ? 0 : 2 * (wsub + mt) + hf;
    const int hh = hk * 8 + g;
    const float lsum = quad_sum(l_r[r]);
    const float f = (m_r[r] == -INFINITY) ? 0.f : ex2(fminf(m_r[r], 126.f));
    float* acc = p.ws_acc + ((ix_t)((bp * p.nc + copy) * L + j) * 16 + hh) * 64;
    if constexpr (L == 1) {
#pragma unroll
      for (int c = 0; c < 2; ++c)
#pragma unroll
        for (int t4 = 0; t4 < 8; t4 += 4)
          red_v4(acc + 16 * tig + 8 * c + t4, o[mt][t4][2 * hf + c] * f, o[mt][t4 + 1][2 * hf + c] * f,
                 o[mt][t4 + 2][2 * hf + c] * f, o[mt][t4 + 3][2 * hf + c] * f);
    } else if constexpr (OKC_RED4) {
      // 16-byte atomics (half the L2 atomic operations of 8-byte ones): lane pairs (tig, tig ^ 1) swap halves so
      // each lane holds 4 consecutive dims: even tig keeps t even, odd tig keeps t odd (dims 8t + 4(tig >> 1) + 0..3)
      const bool odd = tig & 1;
#pragma unroll
      for (int t = 0; t < 8; t += 2) {
        // mine: (t, 2tig..2tig+1) and (t+1, same); partner's: same t's at 2(tig^1)..
        const float a0 = o[mt][t][2 * hf] * f, a1 = o[mt][t][2 * hf + 1] * f;
        const float b0 = o[mt][t + 1][2 * hf] * f, b1 = o[mt][t + 1][2 * hf + 1] * f;
        // even lane sends its t+1 pair, odd lane sends its t pair
        const float s0 = odd ? a0 : b0, s1 = odd ? a1 : b1;
        const float r0 = __shfl_xor_sync(0xffffffffu, s0, 1), r1 = __shfl_xor_sync(0xffffffffu, s1, 1);
        // even lane: t (dims 2tig, 2tig+1, then the partner's); odd lane: t+1 (the partner's, then mine) -- selects,
        // one atomic, no branch
        red_v4(acc + 8 * (t + (odd ? 1 : 0)) + 4 * (tig >> 1), odd ? r0 : a0, odd ? r1 : a1, odd ? b0 : r0, odd ? b1 : r1);
      }
    } else {
#pragma unroll
      for (int t = 0; t < 8; ++t) red_v2(acc + 8 * t + 2 * tig, o[mt][t][2 * hf] * f, o[mt][t][2 * hf + 1] * f);
    }
    if (tig == 0) red_f32(p.ws_l + ((bp * p.nc + copy) * L + j) * 16 + hh, lsum * f);
  }
  }
  if (OKC_MRGDBG) DBG(7);   // (debug: reds issued)
  // OKC_MERGE2 (last-CTA finalizes): no per-thread fences; one acq_rel atomic by thread 0 after the CTA barrier
  // releases every thread's reds (cumulativity through bar.sync, as CUTLASS's semaphore release) and acquires the
  // others' for the last CTA's reads
  constexpr bool MRG2 = OKC_MERGE2;   // (the cluster path returned above)
  const bool mode1 = !((FLAT && (L == 8 || OKC_FLATCOOP1C)) && M_COOP) && !(!FLAT && M_COOP);
  // OKC_MERGE3: the same for the cooperative grids (the barrier's arrive is an acq_rel atomic after a CTA barrier)
  if (!((MRG2 && mode1) || (OKC_MERGE3 && !mode1))) gpu_fence();
  csync<WS, NT>();
  if (OKC_MRGDBG) DBG(6);   // (debug: every thread's reds performed)
  if ((FLAT && (L == 8 || OKC_FLATCOOP1C)) && M_COOP) {
    // flat cooperative mode: every request is finalized after all CTAs have merged all their segments
    DBG(4);
  } else if (!FLAT && M_COOP) {
    // ---- all splits of this (request, kv pair) are resident: wait for every split's merge, then each
    // CTA finalizes its own slice of the rows (instead of one CTA doing all of them)
    // (the slice arithmetic first: it does not depend on the merge, so its divisions leave the barrier's chain)
    const int split_idx = (int)blockIdx.x;
    int nact = (OKC_UNEV && (MODE & 3) == 3 && p.unev > 0 && ((n_keys + 63) >> 6) >= 2 * p.n_splits)
                   ? p.n_splits : fdiv_i(n_keys + chunk - 1, chunk);    // splits with keys
    int R = fdiv_i(L * 16 + nact - 1, nact);          // rows per CTA
    // OKC_PINDIV: keep both divisions above the barrier wait (ptxas otherwise sinks them below the spin, where
    // they are first used: scout's v200 census found them after the poll); volatile asm keeps program order
    if (OKC_PINDIV) asm volatile("" : "+r"(nact), "+r"(R));
    if (tid == 0 && (OKC_SENSE || OKC_SENSE2)) {
      // sense-reversing barrier: the last split to arrive resets the key counter and flips this request's flag;
      // the others wait for the flip (no departure counter: no extra atomic round trip after the finalize).
      // OKC_SENSE2: the flag was read in the prologue (it can only flip after every split has arrived)
      int* flag = p.ws_cnt + 24576 + bp;
      int sense;
      if (OKC_SENSE2) sense = S.sense;
      else asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(sense) : "l"(flag) : "memory");
      const int mine = end - start;
      int old;   // acq_rel: the last arrival synchronizes with every earlier one, then releases the flag
      asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(p.ws_cnt + bp), "r"(mine) : "memory");
      if (old + mine == n_keys) {
        p.ws_cnt[bp] = 0;
        asm volatile("st.release.gpu.global.s32 [%0], %1;\n" ::"l"(flag), "r"(sense ^ 1) : "memory");
      } else {
        int v;
        SPIN_T0;
        for (;;) {
          asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(flag) : "memory");
          if (v != sense) break;
          SPIN_CHECK;
          __nanosleep(64);
        }
      }
    } else if (tid == 0 && OKC_PARITY) {
      // two key counters alternating per launch (parity = this slot's epoch, read in the prologue): the last split
      // to arrive resets the OTHER counter (unused until the next launch) and bumps the epoch, so nobody needs a
      // departure counter (no atomic round trip at the very end of the kernel)
      const int ep = S.sense;
      int* c = p.ws_cnt + ((ep & 1) ? 12288 : 8192) + bp;
      const int mine = end - start;
      int old;
      if constexpr (OKC_MERGE3) asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(c), "r"(mine) : "memory");
      else old = atomicAdd(c, mine);
      if (old + mine == n_keys) {
        p.ws_cnt[((ep & 1) ? 8192 : 12288) + bp] = 0;
        p.ws_cnt[24576 + bp] = ep + 1;
      } else {
        int v;
        SPIN_T0;
        for (;;) {
          asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(c) : "memory");
#ifdef OKC_SPINTEST
          if (v >= n_keys + 1) break;   // (test builds: an unreachable target -> the spin guard must trap)
#else
          if (v >= n_keys) break;
#endif
          SPIN_CHECK;
          __nanosleep(64);
        }
      }
    } else if (tid == 0) {
      atomicAdd(p.ws_cnt + bp, end - start);
      int v;
      SPIN_T0;
      for (int spin = 0;; ++spin) {
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p.ws_cnt + bp) : "memory");
        if (v >= n_keys) break;
        SPIN_CHECK;
        if (OKC_SPINDBG && spin > (1 << 20)) {   // debug builds: report and give up instead of hanging
          printf("coop wait stuck: block (%d,%d,%d) bp %d counter %d n_keys %d start %d end %d\n", blockIdx.x,
                 blockIdx.y, blockIdx.z, bp, v, n_keys, start, end);
          break;
        }
        __nanosleep(64);
      }
    }
    csync<WS, NT>();
    DBG(4);
    const int r_lo = split_idx * R, r_hi = min(L * 16, r_lo + R);
    const int part = tid & 7;
#if OKC_FIN1
#pragma unroll 1
    for (int rbase = r_lo; rbase < r_hi; rbase += NT / 8) {   // warp-uniform trip count (shuffles below)
#else
    for (int rbase = r_lo; rbase < r_hi; rbase += NT / 8) {   // warp-uniform trip count (shuffles below)
#endif
      const int rr = rbase + (tid >> 3);
      const bool act = rr < r_hi;
      const int j = act ? rr >> 4 : 0, hh = act ? rr & 15 : 0;
      const uint4 graw = act ? *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8)
                             : make_uint4(0, 0, 0, 0);
      float y[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      float lsum = 0.f;
      sum_copies<L>(p, bp, j, hh, part, act, y, lsum);
      const float inv = lsum > 0.f ? __fdividef(vscale, lsum) : 0.f;
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) { y[e] = pr8(y[e] * inv); ss += y[e] * y[e]; }
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      ss += __shfl_xor_sync(0xffffffffu, ss, 4);
      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw);
      uint4 ov;
      uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 gt = __bfloat1622float2(g2[e]);
        const float s0 = sigm(gt.x), s1 = sigm(gt.y);
        __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * e] * rn * s0, y[2 * e + 1] * rn * s1);
        ow[e] = *reinterpret_cast<uint32_t*>(&r2);
      }
      if (act) *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + part * 8) = ov;
    }
    if (!OKC_SENSE && !OKC_SENSE2 && !OKC_PARITY) {
      csync<WS, NT>();
      if (tid == 0) {   // the last CTA out resets both counters (every CTA has left the wait by then)
        const int old = atomicAdd(p.ws_cnt + 16384 + bp, 1);
        if (old + 1 == nact) {
          p.ws_cnt[bp] = 0;
          p.ws_cnt[16384 + bp] = 0;
        }
      }
    }
    DBG(5);
  } else {
  if (tid == 0) {
    const int mine = end - start;
    int old;
    if constexpr (MRG2) asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(p.ws_cnt + bp), "r"(mine) : "memory");
    else old = atomicAdd(p.ws_cnt + bp, mine);
    S.flag = (old + mine == n_keys);
  }
  csync<WS, NT>();
  DBG(4);
  if (S.flag) {
  if constexpr (!MRG2) gpu_fence();
  DBG(14);

  // ---- last CTA: sum the copies, normalize, rmsnorm * sigmoid(gate), store; reset the workspace ----
  if (p.nc == 1) {
    fin_nc1<L, NT>(p, b, pair, bp, vscale, 0);
  } else
  {
    const int part = tid & 7;
    for (int rr = tid >> 3; rr < L * 16; rr += NT / 8) {
      const int j = rr >> 4, hh = rr & 15;
      float y[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      float lsum = 0.f;
      // gate row in flight with the copies (one L2 round trip per row instead of two)
      const uint4 graw = *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8);
      sum_copies<L>(p, bp, j, hh, part, true, y, lsum);
      const float inv = lsum > 0.f ? __fdividef(vscale, lsum) : 0.f;
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) { y[e] = pr8(y[e] * inv); ss += y[e] * y[e]; }
      ss = oct_sum(ss);
      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw);
      uint4 ov;
      uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 gt = __bfloat1622float2(g2[e]);
        const float s0 = sigm(gt.x), s1 = sigm(gt.y);
        __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * e] * rn * s0, y[2 * e + 1] * rn * s1);
        ow[e] = *reinterpret_cast<uint32_t*>(&r2);
      }
      *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + part * 8) = ov;
    }
  }
  if (tid == 0) p.ws_cnt[bp] = 0;
  DBG(5);
  if (p.dbg && tid == 0) p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + 6] = 1;
  }
  }
  if constexpr (CONT) csync<WS, NT>();   // consumers' shared memory is reused by the next segment
  else __syncthreads();                   // shared memory is reused by the next segment
  }
  return true;
}

// Flat grids whose CTAs are all resident: once every CTA has merged all its segments (grid-wide count),
// each CTA finalizes its own slice of all requests' rows (sum the copies, normalize, rmsnorm * gate, store,
// zero the workspace). Runs on the NT consumer threads; the last CTA out resets the two counters.
template <int L, bool WS, int NT>
__device__ __forceinline__ void flat_finalize(const Params& p, const int* sense_s) {
  const int tid = threadIdx.x;
  pdl_wait();   // a CTA without segments has not waited yet (the gate rows are qkv)
  const int G = gridDim.x * gridDim.y * gridDim.z;
  const int c = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  int* cnt = p.ws_cnt + 30000;
  if constexpr (!OKC_MERGE3) gpu_fence();   // (OKC_MERGE3: thread 0's acq_rel arrive releases the CTA's reds)
  csync<WS, NT>();
  constexpr bool PAR = OKC_PARITY;
  if (tid == 0 && PAR) {
    // two grid counters alternating per launch (parity = the epoch thread 0 read at kernel start): the last CTA
    // to arrive resets the other counter and bumps the epoch; no departure counter at the end of the kernel
    cp_wait<0>();
    const int ep = *reinterpret_cast<const volatile int*>(sense_s);
    int* ac = p.ws_cnt + 30020 + (ep & 1);
    int old;
    if constexpr (OKC_MERGE3) asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;\n" : "=r"(old) : "l"(ac) : "memory");
    else old = atomicAdd(ac, 1);
    if (old + 1 == G) {
      p.ws_cnt[30020 + ((ep & 1) ^ 1)] = 0;
      p.ws_cnt[30022] = ep + 1;
      if (OKC_W0ENABLE) {   // the per-SM rank registration (read by every CTA at kernel start, before arriving here)
        cnt[2] = 0;
        cnt[3] = 0;
        for (int k = 0; k < 8; ++k) p.ws_cnt[30030 + k] = 0;
        for (int i = 0; i < G / 2 && i < 1700; ++i) p.ws_cnt[31000 + i] = 0;
      }
    } else {
      int v;
      SPIN_T0;
      for (;;) {
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(ac) : "memory");
        if (v >= G) break;
        SPIN_CHECK;
        __nanosleep(64);
      }
    }
  } else if (tid == 0 && OKC_SENSE) {   // sense-reversing grid barrier (flag at cnt[8]): no departure counter
    int sense;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(sense) : "l"(cnt + 8) : "memory");
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;\n" : "=r"(old) : "l"(cnt) : "memory");
    if (old + 1 == G) {
      cnt[0] = 0;
      asm volatile("st.release.gpu.global.s32 [%0], %1;\n" ::"l"(cnt + 8), "r"(sense ^ 1) : "memory");
    } else {
      int v;
      SPIN_T0;
      for (;;) {
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(cnt + 8) : "memory");
        if (v != sense) break;
        SPIN_CHECK;
        __nanosleep(64);
      }
    }
  } else if (tid == 0) {
    atomicAdd(cnt, 1);
    int v;
    SPIN_T0;
    for (;;) {
      asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(cnt) : "memory");
      if (v >= G) break;
      SPIN_CHECK;
      __nanosleep(64);
    }
  }
  csync<WS, NT>();
  const int pairs = p.Hkv >> 1;
  const int rows = p.B * pairs * L * 16;
  const int R = fdiv_i(rows + G - 1, G);
  const int r_lo = c * R, r_hi = min(rows, r_lo + R);
  const float vscale = __ldg(p.v_scale);
  const int part = tid & 7;
  for (int rbase = r_lo; rbase < r_hi; rbase += NT / 8) {   // warp-uniform trip count (shuffles below)
    const int r = rbase + (tid >> 3);
    const bool act = r < r_hi;
    const int bp = act ? r / (L * 16) : 0, rr = act ? r % (L * 16) : 0;
    const int j = rr >> 4, hh = rr & 15;
    const int b = (pairs == 1) ? bp : fdiv_i(bp, pairs), pair = bp - b * pairs;
    const uint4 graw = act ? *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8)
                           : make_uint4(0, 0, 0, 0);
    float y[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float lsum = 0.f;
    sum_copies<L>(p, bp, j, hh, part, act, y, lsum);
    const float inv = lsum > 0.f ? __fdividef(vscale, lsum) : 0.f;
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) { y[e] = pr8(y[e] * inv); ss += y[e] * y[e]; }
    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
    ss += __shfl_xor_sync(0xffffffffu, ss, 2);
    ss += __shfl_xor_sync(0xffffffffu, ss, 4);
    const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
    const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw);
    uint4 ov;
    uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 gt = __bfloat1622float2(g2[e]);
      const float s0 = sigm(gt.x), s1 = sigm(gt.y);
      __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * e] * rn * s0, y[2 * e + 1] * rn * s1);
      ow[e] = *reinterpret_cast<uint32_t*>(&r2);
    }
    if (act) *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + part * 8) = ov;
  }
  if (!OKC_SENSE && !PAR) {
    csync<WS, NT>();
    if (tid == 0) {
      const int old = atomicAdd(cnt + 1, 1);
      if (old + 1 == G) {
        cnt[0] = 0;
        cnt[1] = 0;
        cnt[2] = 0;   // weighted partition: arrivals, placement flag, per-SM ranks
        cnt[3] = 0;
        for (int i = 0; i < G / 2 && i < 1700; ++i) p.ws_cnt[31000 + i] = 0;
      }
    }
  }
}

template <int L, bool FLAT, int MODE = 0>
__global__ void __launch_bounds__(Cfg<L, FLAT>::NTX, Cfg<L, FLAT>::MINB) attn_kernel(const __grid_constant__ Params p) {
  constexpr int NT = Cfg<L, FLAT>::NT;
  constexpr int STAGES = Cfg<L, FLAT>::STAGES;
  constexpr int IDX_SLOTS = 2 * STAGES;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  // 1024-byte aligned base: the 128B-swizzled tiles must start on swizzle-pattern boundaries (wgmma)
  Smem<L, FLAT>& S = *reinterpret_cast<Smem<L, FLAT>*>(smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u));
  if constexpr (!FLAT && OKC_FLATONLY) return;   // (bench builds of the flat kernels only)
  const int pair = blockIdx.z;
  DBG(0);
#if OKC_SMDBG
  if (p.dbg && threadIdx.x == 0) {
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + 15] = smid;
  }
#endif
  if (!FLAT) {
    // grid (n_splits, B): request b, split s of its window / context
    const int split = blockIdx.x, b = blockIdx.y;
    // OKC_SPEC (MODE | 8 builds, chosen by the host for one request with page size 1): SGLang's FA3 backend builds
    // page_table = req_to_token[req_pool_indices, :max_seq_len_k] (sglang/srt/layers/attention/
    // flashattention_backend.py, init_forward_metadata: decode / target-verify paths), so with one request its row
    // is exactly seq_len + q_len entries wide. The page table's width is a shape: speculate seq = width - q_len and
    // issue the window's page indices in the same trip as cache_seqlens; the real length is checked (segment) before
    // the indices are used for any K/V address or any softmax work, and a miss re-runs everything with the real one
    const int seq_ld = __ldg(p.seq_lens + b);
    constexpr bool SPEC = (MODE & 8) != 0;
    int seq = SPEC ? p.max_pages - L : seq_ld;
    for (;;) {
    const int lo = p.window ? max(0, seq + 1 - p.window) : 0;
    const int n_keys = seq + L - lo;
    int chunk = fdiv_i(n_keys + p.n_splits - 1, p.n_splits);
    chunk = (chunk + 15) & ~15;
#if OKC_TCHUNK
    // plain split-KV: whole tiles per split (the per-CTA tile count is what costs); trailing splits may get no keys:
    // they skip the workspace merge (the merge counts keys, the cooperative finalize counts the splits with keys).
    // Not for clusters (an empty cluster CTA still runs the cluster epilogue: measured slower)
    if ((MODE & 3) == 0 ? p.cs <= 1 : (MODE & 3) != 2) {
      const int ntl = ((n_keys + 63) >> 6) + p.n_splits - 1;
      // OKC_MAGICDIV: ceil(2^32 / n_splits) from the host (exact for ntl < 2^20, n_splits <= 1024): one IMAD.HI instead
      // of the ~20-instruction fdiv_i on the once-path (scout's census)
      if constexpr (OKC_MAGICDIV) chunk = (p.ns_magic ? (int)__umulhi((unsigned)ntl, p.ns_magic) : ntl) << 6;
      else chunk = fdiv_i(ntl, p.n_splits) << 6;
    }
#endif
    int start = lo + split * chunk;
    int end = min(seq + L, start + chunk);
    // OKC_UNEV: splits beyond the SM count share an SM with an earlier split, which gets the warp schedulers'
    // priority: the first-wave splits take w0/1024 of a second-wave split's tiles (only when every split still gets
    // at least one tile, so all splits have keys and the cooperative finalize counts all of them)
    const int unev_tiles = (n_keys + 63) >> 6;
    const bool unev = OKC_UNEV && (MODE & 3) == 3 && p.unev > 0 && unev_tiles >= 2 * p.n_splits;
    if (unev) {
      const int H = p.unev;
      const long long wn = (long long)p.w0 * min(p.n_splits, H) + 1024LL * max(0, p.n_splits - H);
      const long long w_s = (long long)p.w0 * min(split, H) + 1024LL * max(0, split - H);
      const long long w_e = (long long)p.w0 * min(split + 1, H) + 1024LL * max(0, split + 1 - H);
      start = lo + (int)(unev_tiles * w_s / wn) * 64;
      end = min(seq + L, lo + (int)(unev_tiles * w_e / wn) * 64);
    }
    if (start >= end) {
      if (SPEC && seq != seq_ld) { seq = seq_ld; continue; }   // (OKC_SPEC: an empty split only on the real length)
      if ((MODE & 3) == 0 ? p.cs > 1 : (MODE & 3) == 2) {   // still part of its cluster: contributes an empty partial, finalizes its slice
        pdl_wait();     // (the gate rows are qkv)
        cluster_stage_gate<L>(p, S, b, pair);
        if (threadIdx.x == 0) cp_async4(smem_u32(&S.scl[1]), p.v_scale);
        cp_commit();
        float* PB = reinterpret_cast<float*>(S.k);
        float* PL = PB + L * 16 * 64;
        for (int e = threadIdx.x; e < L * 16 * 64; e += NT) PB[e] = 0.f;
        for (int e = threadIdx.x; e < L * 16; e += NT) PL[e] = 0.f;
        cp_wait<0>();
        __syncthreads();
        cluster_finish<L>(p, S, b, pair, n_keys, chunk);
      }
      return;
    }
    if (segment<L, FLAT, 0, MODE>(p, S, b, pair, split == 0, start, end, seq, n_keys, split - fdiv_i(split, p.nc) * p.nc,
                                  chunk, 0, SPEC ? seq_ld : seq))
      return;
    seq = seq_ld;   // (OKC_SPEC miss: today's path with the real length)
    }
  }
  if constexpr (FLAT && (L == 8 || OKC_FLATCOOP1C) && OKC_PARITY) {
    if (p.coop && threadIdx.x == 0) {   // the grid barrier's epoch, long before it is needed
      cp_async4(smem_u32(&S.sense), p.ws_cnt + 30022);
      cp_commit();
    }
  }
  // OKC_W0EARLY: the per-SM registration's atomics go out from warp 1 while warp 0 builds the prefix sums
  const bool w0on = OKC_W0ENABLE && p.coop && p.resident && p.w0 != 1024 && gridDim.y * gridDim.z == 1 && (gridDim.x & 1) == 0;
  // OKC_W0L1: the same uneven per-SM shares for the non-cooperative (q_len 1) flat grids. No grid barrier ends those,
  // so the registration state is generation-tagged: a monotonic launch counter picks one of two state copies, and
  // the launch's last registrant clears the other copy (the previous launch's) for the next launch
  const bool w0g = OKC_W0L1 && OKC_W0ENABLE && !p.coop && p.resident && p.w0 != 1024;
  int w0g_rank = 2, w0g_par = 0, w0g_gen = 0;
  unsigned w0_smid = 0;
  int w0_rank = 0;
  if (OKC_W0EARLY && w0on && threadIdx.x == 32) {
    asm volatile("mov.u32 %0, %%smid;" : "=r"(w0_smid));
    const int Gx = gridDim.x;
    w0_rank = ((int)w0_smid < Gx / 2) ? atomicAdd(p.ws_cnt + 31000 + w0_smid, 1) : 2;
    if (w0_rank >= 2 || OKC_W0FORCEBAD) atomicExch(p.ws_cnt + 30003, 1);
    if (OKC_W0BID && w0_rank < 2) p.ws_cnt[31200 + 2 * w0_smid + w0_rank] = (int)blockIdx.x + 1;
    gpu_fence();
    atomicAdd(p.ws_cnt + (OKC_W0SPREAD ? 30030 + (blockIdx.x & 7) : 30002), 1);   // (8 counters: no single hot address)
  }
  if (w0g && threadIdx.x == 32) {
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    const int Gx = gridDim.x;
    const unsigned og = (unsigned)atomicAdd(p.ws_cnt + 32000, 1);   // launch generation (G registrants per launch)
    w0g_gen = (int)(og / (unsigned)Gx);
    w0g_par = w0g_gen & 1;
    if (og % (unsigned)Gx == (unsigned)Gx - 1) {   // the launch's last registrant: clear the copy the next launch uses
      for (int i = 0; i < Gx / 2; ++i) p.ws_cnt[32010 + (w0g_par ^ 1) * 136 + i] = 0;
      p.ws_cnt[32002 + (w0g_par ^ 1)] = 0;
    }
    w0g_rank = ((int)smid < Gx / 2) ? atomicAdd(p.ws_cnt + 32010 + w0g_par * 136 + smid, 1) : 2;
    if (w0g_rank >= 2 || OKC_W0FORCEBAD) atomicExch(p.ws_cnt + 32002 + w0g_par, 1);
    if (w0g_rank < 2) p.ws_cnt[32300 + 2 * smid + w0g_rank] = (int)blockIdx.x + 1;
    __threadfence();
    atomicAdd(p.ws_cnt + 32001, 1);   // registered (monotonic: launch g is complete at (g + 1) * G)
    w0g_rank = (w0g_rank & 3) | ((int)smid << 2);
  }
  // grid (G): CTA c takes keys [c*T/G, (c+1)*T/G) of all requests' keys laid end to end (equal work)
  // OKC_FPART: warp 0 computes the key range and the segment list right after its prefix sums (one request per lane)
  constexpr bool fpart = OKC_FPART && L == 1 && !OKC_W0ENABLE && !OKC_W0L1 && OKC_WSPLIT == 500 && !OKC_BALPART &&
                         !OKC_SWAPHALF && !OKC_DYN;   // (compile time: no run-time branch in the main flow)
  if constexpr (OKC_EARLYBAR && L == 1 && Cfg<L, FLAT>::WS && !OKC_CONT && !OKC_PTR) {
    if (threadIdx.x == 32) {   // the first segment's stage barriers (segment() skips its init for that segment)
      for (int st = 0; st < Cfg<L, FLAT>::STAGES; ++st) {
        mbar_init_n(&S.full[st], Cfg<L, FLAT>::NP);
        mbar_init_n(&S.empty[st], Cfg<L, FLAT>::F2 ? 8 : (OKC_XP_ARRALL ? Cfg<L, FLAT>::NT : Cfg<L, FLAT>::NT / 32));
      }
      mbar_fence_init();
    }
  }
  if (threadIdx.x < 32) {
    int run = 0;
    int incl_l = 0, n_l = 0, seq_l = 0;
    for (int b0 = 0; b0 < p.B; b0 += 32) {
      const int b = b0 + (int)threadIdx.x;
      int n = 0;
      if (b < p.B) {
        const int seq = __ldg(p.seq_lens + b);
        const int lo = p.window ? max(0, seq + 1 - p.window) : 0;
        n = seq + L - lo;
        seq_l = seq;
      }
      int incl = n;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int t = __shfl_up_sync(0xffffffffu, incl, o);
        if ((int)threadIdx.x >= o) incl += t;
      }
      if (b < p.B) S.cum[b + 1] = run + incl;
      run += __shfl_sync(0xffffffffu, incl, 31);
      incl_l = incl;
      n_l = n;
    }
    if (threadIdx.x == 0) S.cum[0] = 0;
    if (fpart && p.B > 32) {   // many requests: the generic partition by lane 0 (after the whole prefix sum)
      __syncwarp();
      if (threadIdx.x == 0) {
        constexpr int SEGKF = (L == 8) ? OKC_SEGK8 : OKC_SEGK1;
        const int64_t Tg = (int64_t)S.cum[p.B] + (SEGKF > 0 ? (int64_t)(p.B - 1) * SEGKF : 0);
        int csg = (int)((int64_t)blockIdx.x * Tg / gridDim.x), ceg = (int)((int64_t)(blockIdx.x + 1) * Tg / gridDim.x);
        if (SEGKF > 0) {
          auto vmapg = [&](int v) -> int {
            int lo_b = 0, hi_b = p.B - 1;
            while (lo_b < hi_b) {
              const int mid = (lo_b + hi_b + 1) >> 1;
              if (S.cum[mid] + mid * SEGKF <= v) lo_b = mid; else hi_b = mid - 1;
            }
            return min(v - lo_b * SEGKF, S.cum[lo_b + 1]);
          };
          csg = vmapg(csg);
          ceg = vmapg(ceg);
        }
        int b = 0, n = 0;
        while (S.cum[b + 1] <= csg) ++b;
        for (; b < p.B && S.cum[b] < ceg && n < MAXSEG; ++b) {
          S.seg[n][0] = b;
          S.seg[n][1] = max(csg, S.cum[b]) - S.cum[b];
          S.seg[n][2] = min(ceg, S.cum[b + 1]) - S.cum[b];
          ++n;
        }
        S.nseg = n;
        S.fcs = csg;
        S.fce = ceg;
      }
    } else if (fpart) {   // (one iteration above: lane b holds cum[b + 1] = incl_l, cum[b] = incl_l - n_l; lanes >= B: n = 0)
      const unsigned FULL = 0xffffffffu;
      const int lane_f = (int)threadIdx.x;
      constexpr int SEGKF = (L == 8) ? OKC_SEGK8 : OKC_SEGK1;
      const int excl = incl_l - n_l;
      const int64_t Tf = (int64_t)__shfl_sync(FULL, incl_l, 31) + (SEGKF > 0 ? (int64_t)(p.B - 1) * SEGKF : 0);
      const int cf = blockIdx.x, Gf = gridDim.x;
      int csf = (int)((int64_t)cf * Tf / Gf), cef = (int)((int64_t)(cf + 1) * Tf / Gf);
      if (SEGKF > 0) {   // virtual -> real: the last request whose virtual start (cum[b] + b * SEGK) is <= v
        const int vstart = excl + lane_f * SEGKF;
        const unsigned ms = __ballot_sync(FULL, lane_f < p.B && vstart <= csf);
        const unsigned me = __ballot_sync(FULL, lane_f < p.B && vstart <= cef);
        const int bs = 31 - __clz(ms), be = 31 - __clz(me);   // (lane 0 always qualifies: virtual start 0)
        const int is = __shfl_sync(FULL, incl_l, bs), ie = __shfl_sync(FULL, incl_l, be);
        csf = min(csf - bs * SEGKF, is);
        cef = min(cef - be * SEGKF, ie);
      }
      // segment list: the requests with cum[b + 1] > cs and cum[b] < ce (a contiguous run of lanes)
      const bool in = lane_f < p.B && incl_l > csf && excl < cef;
      const unsigned mi = __ballot_sync(FULL, in);
      const int rank = __popc(mi & ((1u << lane_f) - 1u));
      if (in && rank < MAXSEG) {
        S.seg[rank][0] = lane_f;
        S.seg[rank][1] = max(csf, excl) - excl;
        S.seg[rank][2] = min(cef, incl_l) - excl;
      }
      if (lane_f == 0) {
        S.nseg = min(__popc(mi), MAXSEG);
        S.fcs = csf;
        S.fce = cef;
      }
      if (OKC_PTPF && mi != 0u && csf < cef) {   // the first segment's first STAGES tiles of page indices toward L2
        const int b0 = __ffs(mi) - 1;
        const int seqb = __shfl_sync(FULL, seq_l, b0), exb = __shfl_sync(FULL, excl, b0), inb = __shfl_sync(FULL, incl_l, b0);
        const int lob = p.window ? max(0, seqb + 1 - p.window) : 0;
        const int s0 = max(csf, exb) - exb + lob, s1 = min(cef, inb) - exb + lob;
        const int cend = min(min(s1, seqb), s0 + Cfg<L, FLAT>::STAGES * TILE);
        if (cend > s0) {
          const long long e0 = (long long)b0 * p.max_pages + (s0 >> p.lp);
          const long long e1 = (long long)b0 * p.max_pages + ((cend - 1) >> p.lp);
          const long long l0 = e0 >> 5, l1 = e1 >> 5;   // 128-byte lines of int32 page indices
          if (l0 + lane_f <= l1) asm volatile("prefetch.global.L2 [%0];" ::"l"(p.page_table + ((l0 + lane_f) << 5)));
        }
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) FPRO_STAMP(40);
  const int64_t T0 = S.cum[p.B];
  // OKC_SEGK: every request boundary costs a CTA that crosses it a whole extra segment (q staging, pipeline refill,
  // merge: ~7 us at q_len 8), so the shares are cut in a virtual key space with SEGK keys inserted before each
  // request after the first, then mapped back (a CTA crossing a boundary gets SEGK fewer real keys)
  constexpr int SEGK = (L == 8) ? OKC_SEGK8 : OKC_SEGK1;
  const int64_t T = T0 + (SEGK > 0 ? (int64_t)(p.B - 1) * SEGK : 0);
  const int c = blockIdx.x, G = gridDim.x;
#if OKC_BALPART
  // balanced shares with one 32-bit division (T < 2^31: at most 512 requests of ~1M keys)
  const unsigned Tq = (unsigned)T / (unsigned)G, Tr = (unsigned)T - Tq * (unsigned)G;
  int cs = (int)(c * Tq + min((unsigned)c, Tr)), ce = cs + (int)Tq + ((unsigned)c < Tr ? 1 : 0);
#else
  // (OKC_SWAPHALF, experiment: the first-dispatched half of the grid takes the second half of the keys)
  const int cpart = (OKC_SWAPHALF && (G & 1) == 0) ? (c + (G >> 1)) % G : c;
  int cs, ce;
  if constexpr (fpart) {   // (OKC_FPART: computed by warp 0)
    cs = S.fcs;
    ce = S.fce;
  } else {
    cs = (int)((int64_t)cpart * T / G);
    ce = (int)((int64_t)(cpart + 1) * T / G);
  }   // (a float-estimate + correction
  // version without the 64-bit division call measured slower: q_len 1 b16 full +3 us, likely code layout)
#endif
  if (OKC_WSPLIT != 500 && (G & 1) == 0 && gridDim.y * gridDim.z == 1) {
    // CTAs c and c + G/2 are dispatched to the same SM, and the first one dispatched runs faster: give the first
    // half of the grid OKC_WSPLIT per mille of the keys
    const int64_t Th = T * OKC_WSPLIT / 1000, H = G >> 1;
    if (c < H) { cs = (int)((int64_t)c * Th / H); ce = (int)((int64_t)(c + 1) * Th / H); }
    else { cs = (int)(Th + (int64_t)(c - H) * (T - Th) / H); ce = (int)(Th + (int64_t)(c + 1 - H) * (T - Th) / H); }
  }
  int vc = c;
  if (w0on) {
    // The two CTAs sharing an SM do not progress equally (the first one dispatched gets more issue slots), so
    // give the first one a larger share: slot 2 * SM + arrival rank. Every CTA waits until all have registered;
    // any SM with more than two (another kernel holding SMs) falls back to the even split.
    if (threadIdx.x == (OKC_W0EARLY ? 32 : 0)) {
      unsigned smid = w0_smid;
      int* reg = p.ws_cnt + 31000;
      int* arrive = p.ws_cnt + 30002;
      int* bad = p.ws_cnt + 30003;
      int rank = w0_rank;
      if (!OKC_W0EARLY) {
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        rank = ((int)smid < G / 2) ? atomicAdd(reg + smid, 1) : 2;
        if (rank >= 2 || OKC_W0FORCEBAD) atomicExch(bad, 1);   // (OKC_W0FORCEBAD: test the even-split fallback)
        if (OKC_W0BID && rank < 2) p.ws_cnt[31200 + 2 * smid + rank] = c + 1;   // who took this SM's slot
        gpu_fence();
        atomicAdd(OKC_W0SPREAD ? p.ws_cnt + 30030 + (c & 7) : arrive, 1);
      }
      int v;
      SPIN_T0;
      if (OKC_W0SPREAD) {   // every CTA registered: counter k holds the CTAs c with c % 8 == k
        for (int k = 0; k < 8; ++k) {
          const int need = (G - k + 7) >> 3;
          for (;;) {
            asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p.ws_cnt + 30030 + k) : "memory");
            if (v >= need) break;
            SPIN_CHECK;
          }
        }
      } else {
        for (;;) {
          asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(arrive) : "memory");
          if (v >= G) break;
          SPIN_CHECK;
        }
      }
      asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(bad) : "memory");
      if (OKC_W0BID && !v) {   // rank by blockIdx order on the SM (the lower index gets the issue-slot priority)
        int other;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(other) : "l"(p.ws_cnt + 31200 + 2 * smid + (rank ^ 1)) : "memory");
        rank = (other > 0 && other - 1 < c) ? 1 : 0;
      }
      S.vidx = v ? -1 : (int)(smid * 2 + rank);
    }
    __syncthreads();
    if (S.vidx >= 0) {
      vc = S.vidx;
      const int64_t sp = vc >> 1, r = vc & 1;
      cs = (int)(T * (2048 * sp + r * p.w0) / (1024LL * G));
      ce = (int)(T * (2048 * sp + (r ? 2048 : p.w0)) / (1024LL * G));
    }
  }
  if (w0g) {
    if (threadIdx.x == 32) {
      const unsigned smid = (unsigned)w0g_rank >> 2;
      int rank = w0g_rank & 3;
      int v;
      const unsigned need = ((unsigned)w0g_gen + 1u) * (unsigned)G;
      SPIN_T0;
      for (;;) {
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p.ws_cnt + 32001) : "memory");
        if ((int)((unsigned)v - need) >= 0) break;
        SPIN_CHECK;
      }
      asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p.ws_cnt + 32002 + w0g_par) : "memory");
      if (!v) {
        int other;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(other) : "l"(p.ws_cnt + 32300 + 2 * smid + (rank ^ 1)) : "memory");
        rank = (other > 0 && other - 1 < c) ? 1 : 0;
      }
      S.vidx = v ? -1 : (int)(smid * 2 + rank);
    }
    __syncthreads();
    if (S.vidx >= 0) {
      vc = S.vidx;
      const int64_t sp = vc >> 1, r = vc & 1;
      cs = (int)(T * (2048 * sp + r * p.w0) / (1024LL * G));
      ce = (int)(T * (2048 * sp + (r ? 2048 : p.w0)) / (1024LL * G));
    }
  }
  if (SEGK > 0 && !fpart) {   // virtual -> real key positions
    auto vmap = [&](int v) -> int {
      int lo_b = 0, hi_b = p.B - 1;   // the last request whose virtual start is <= v
      while (lo_b < hi_b) {
        const int mid = (lo_b + hi_b + 1) >> 1;
        if (S.cum[mid] + mid * SEGK <= v) lo_b = mid; else hi_b = mid - 1;
      }
      return min(v - lo_b * SEGK, S.cum[lo_b + 1]);
    };
    cs = vmap(cs);
    ce = vmap(ce);
  }
  if (!(OKC_DYN && Cfg<L, FLAT>::WS) && cs >= ce) {
    if constexpr (L == 8 || OKC_FLATCOOP1C) {
      if (p.coop) {   // still part of the grid-wide finalize
        if constexpr (Cfg<L, FLAT>::WS) {
          if (threadIdx.x < Cfg<L, FLAT>::NT) flat_finalize<L, true, Cfg<L, FLAT>::NT>(p, &S.sense);
        } else {
          flat_finalize<L, false, Cfg<L, FLAT>::NTX>(p, &S.sense);
        }
      }
    }
    return;
  }
  if constexpr (Cfg<L, FLAT>::WS && OKC_TMAPG && OKC_PGCONT && (L == 1 || OKC_TMAPG8)) {
    // OKC_PGCONT: the page-index TMA ring's barriers set up once per launch; slots and phases continue across this
    // CTA's segments (no per-segment re-init of an mbarrier an earlier async copy completed)
    if (threadIdx.x == Cfg<L, FLAT>::NT) {
      for (int st = 0; st < 4; ++st) mbar_init_n(&S.pbar[st], 1);
      S.pgc = 0;
      mbar_fence_init();
    }
    __syncthreads();
  }
  if constexpr (Cfg<L, FLAT>::WS && OKC_CONT) {   // continuous pipeline: stage barriers set up once per launch
    if (threadIdx.x == 0) {
      for (int st = 0; st < Cfg<L, FLAT>::STAGES; ++st) {
        mbar_init_n(&S.full[st], Cfg<L, FLAT>::NP);
        mbar_init_n(&S.empty[st], OKC_XP_ARRALL ? Cfg<L, FLAT>::NT : Cfg<L, FLAT>::NT / 32);
      }
#if OKC_DYN
      for (int k = 0; k < 16; ++k) {
        mbar_init_n(&S.dqf[k], 1);
        mbar_init_n(&S.dqe[k], Cfg<L, FLAT>::NT / 32);
      }
#endif
      mbar_fence_init();
    }
  }
  // segment list in shared memory (keeps the per-segment loop state out of registers)
  if (threadIdx.x == 0 && !fpart) {
    int b = 0, n = 0;
    while (S.cum[b + 1] <= cs) ++b;
    for (; b < p.B && S.cum[b] < ce && n < MAXSEG; ++b) {
      S.seg[n][0] = b;
      S.seg[n][1] = max(cs, S.cum[b]) - S.cum[b];
      S.seg[n][2] = min(ce, S.cum[b + 1]) - S.cum[b];
      ++n;
    }
    S.nseg = n;
  }
  if (!fpart) __syncthreads();   // (OKC_FPART: warp 0 wrote the list before the barrier above)
  if (threadIdx.x == 0) FPRO_STAMP(41);
  const int nseg = S.nseg;
#if OKC_SEGDBG
  if (p.dbg && threadIdx.x == 0) p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + 7] = nseg;
  if (threadIdx.x == 0) S.dbgn = 0;
#endif
#if OKC_SEGDBG
  if (p.dbg && threadIdx.x == 0) {
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + 24] = smid;
    p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + 25] = vc;
  }
#endif
  auto run_segments = [&](auto role) {
    constexpr int R = decltype(role)::value;
    int tb = 0;
#pragma unroll 1
    for (int k = 0; k < nseg; ++k) {
      const int b = S.seg[k][0];
      const int seq = __ldg(p.seq_lens + b);
      const int lo = p.window ? max(0, seq + 1 - p.window) : 0;
      const int n_keys = seq + L - lo;
      const int s0 = S.seg[k][1] + lo, s1 = S.seg[k][2] + lo;
      const bool lead = OKC_SEGDBG && p.dbg && threadIdx.x == 0 && k < 2;
      const int dbase = (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + 16 + 4 * k;
      if (lead) { p.dbg[dbase] = gtime(); p.dbg[dbase + 2] = s1 - s0; p.dbg[dbase + 3] = b; }
      if (s0 < s1) {
        segment<L, FLAT, R>(p, S, b, pair, s0 == lo, s0, s1, seq, n_keys, vc - fdiv_i(vc, p.nc) * p.nc, 0,
                            OKC_CONT ? tb : 0, 0,
                            OKC_EARLYBAR && L == 1 && Cfg<L, FLAT>::WS && !OKC_CONT && !OKC_PTR && k == 0);
        tb += (s1 - s0 + TILE - 1) / TILE;
      }
      if (lead) p.dbg[dbase + 1] = gtime();
    }
    if constexpr (R == 1 && OKC_CONT) cp_wait<0>();   // the producer leaves with its copies landed
  };
#if OKC_DYN
  // Dynamic chunks (producer mode with a continuous pipeline): the concatenated keys are cut into G * OKC_DYNK
  // equal chunks; CTA c starts with chunk c and claims further chunks from a global counter, so the CTA that
  // runs faster (two share each SM, and they do not progress equally) takes more of them. The producer's first
  // thread turns each chunk into segments and hands them to the consumers through a shared-memory queue.
  auto run_dynamic = [&](auto role) {
    constexpr int R = decltype(role)::value;
    constexpr int QN = 16;
    const int NCH = G * OKC_DYNK;
    int tb = 0, qk = 0;
    if constexpr (R == 1) {
      const int q = threadIdx.x - Cfg<L, FLAT>::NT;
      int c = blockIdx.x;
#pragma unroll 1
      for (;;) {
        if (q == 0) {
          int n = 0;
          if (c < NCH) {
            const int cs = (int)((int64_t)c * T / NCH), ce = (int)((int64_t)(c + 1) * T / NCH);
            int lo_b = 0, hi_b = p.B - 1;   // first request with cum[b + 1] > cs
            while (lo_b < hi_b) {
              const int mid = (lo_b + hi_b) >> 1;
              if (S.cum[mid + 1] <= cs) lo_b = mid + 1; else hi_b = mid;
            }
            for (int b = lo_b; b < p.B && S.cum[b] < ce && n < MAXSEG; ++b) {
              S.pseg[n][0] = b;
              S.pseg[n][1] = max(cs, S.cum[b]) - S.cum[b];
              S.pseg[n][2] = min(ce, S.cum[b + 1]) - S.cum[b];
              ++n;
            }
          }
          S.pnseg = n;
          S.pchunk = c;
          for (int k = 0; k < n + (c < NCH ? 0 : 1); ++k) {   // the segments, or the end marker
            const int slot = qk % QN;
            if (qk >= QN) mbar_wait_full(&S.dqe[slot], ((qk / QN) - 1) & 1);
            if (c < NCH) {
              S.dq[slot][0] = S.pseg[k][0];
              S.dq[slot][1] = S.pseg[k][1];
              S.dq[slot][2] = S.pseg[k][2];
              S.dq[slot][3] = c;
            } else {
              S.dq[slot][0] = -1;
            }
            mbar_arrive(&S.dqf[slot]);
            ++qk;
          }
        }
        asm volatile("bar.sync 2, %0;\n" ::"n"(Cfg<L, FLAT>::NP) : "memory");
        const int n = S.pnseg, cc = S.pchunk;
        if (cc >= NCH) break;
#pragma unroll 1
        for (int k = 0; k < n; ++k) {
          const int b = S.pseg[k][0];
          const int seq = __ldg(p.seq_lens + b);
          const int lo = p.window ? max(0, seq + 1 - p.window) : 0;
          const int s0 = S.pseg[k][1] + lo, s1 = S.pseg[k][2] + lo;
          if (s0 < s1) {
            segment<L, FLAT, 1>(p, S, b, pair, s0 == lo, s0, s1, seq, seq + L - lo, cc - fdiv_i(cc, p.nc) * p.nc, 0, tb);
            tb += (s1 - s0 + TILE - 1) / TILE;
          }
        }
        if (q == 0) {   // every copy of this chunk is issued: claim the next one
          const int old = atomicAdd(p.ws_cnt + 30010 + pair, 1);   // one counter per KV pair (grid z)
          if (old == NCH - 1) p.ws_cnt[30010 + pair] = 0;   // the launch's last claim resets the counter
          c = G + old;
        }
        asm volatile("bar.sync 2, %0;\n" ::"n"(Cfg<L, FLAT>::NP) : "memory");   // pseg is rewritten next
      }
      cp_wait<0>();
    } else {
#pragma unroll 1
      for (;;) {
        const int slot = qk % QN;
        mbar_wait_full(&S.dqf[slot], (qk / QN) & 1);
        const int b = S.dq[slot][0], r0 = S.dq[slot][1], r1 = S.dq[slot][2], cc = S.dq[slot][3];
        __syncwarp();
        if ((threadIdx.x & 31) == 0) mbar_arrive(&S.dqe[slot]);
        ++qk;
        if (b < 0) break;
        const int seq = __ldg(p.seq_lens + b);
        const int lo = p.window ? max(0, seq + 1 - p.window) : 0;
        const int s0 = r0 + lo, s1 = r1 + lo;
        if (s0 < s1) {
          segment<L, FLAT, R>(p, S, b, pair, s0 == lo, s0, s1, seq, seq + L - lo, cc - fdiv_i(cc, p.nc) * p.nc, 0, tb);
          tb += (s1 - s0 + TILE - 1) / TILE;
        }
      }
    }
  };
#endif
  if constexpr (Cfg<L, FLAT>::WS) {
    // separate code for the two roles, so the producer warpgroup can hand its registers to the consumers
    if (threadIdx.x >= Cfg<L, FLAT>::NT) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
      if constexpr (OKC_SETMAXNREG) asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(Cfg<L, FLAT>::PREGS));
#endif
#if OKC_DYN
      run_dynamic(std::integral_constant<int, 1>());
#else
      run_segments(std::integral_constant<int, 1>());
#endif
    } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
      if constexpr (OKC_SETMAXNREG) asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(Cfg<L, FLAT>::CREGS));
#endif
#if OKC_DYN
      run_dynamic(std::integral_constant<int, 2>());
#else
      run_segments(std::integral_constant<int, 2>());
#endif
      if constexpr (L == 8 || OKC_FLATCOOP1C) {
        if (p.coop) flat_finalize<L, true, Cfg<L, FLAT>::NT>(p, &S.sense);
      }
    }
  } else {
    run_segments(std::integral_constant<int, 0>());
    if constexpr (FLAT && (L == 8 || OKC_FLATCOOP1C)) {
      if (p.coop) flat_finalize<L, false, Cfg<L, FLAT>::NTX>(p, &S.sense);
    }
  }
}

}  // namespace

template <int L>
void launch(const Params& p0, dim3 grid, cudaStream_t stream, int cs, bool clusters, bool pdl) {
  Params p = p0;
  p.pdlf = pdl ? 1 : 0;
  static bool init = false;
  const int sm_f = (int)sizeof(Smem<L, true>) + 1024, sm_n = (int)sizeof(Smem<L, false>) + 1024;
  const int sm = p.flat ? sm_f : sm_n;
  static int occ_c = 0, occ_f = 0, nsm = 0;
  static int occ_c1 = 0, occ_sp = 0, occ_sp1 = 0;   // (OKC_COOPGUARD) the coop ONE / SPEC / SPEC-ONE instances
  typedef void (*KernFn)(const Params);
  // per-request kernels, one build per launch mode (1 plain split-KV, 2 clusters, 3 cooperative); flat kernel
  // (slot 0, the run-time-mode build, only serves the sm_80 emulation of clusters: not built for sm_90)
  static const KernFn kn[4] = {attn_kernel<L, false, OKC_SM90 ? 2 : 0>, attn_kernel<L, false, 1>,
                               attn_kernel<L, false, 2>, attn_kernel<L, false, 3>};
#if OKC_ONE
  static const KernFn kn1[4] = {nullptr, attn_kernel<L, false, 5>, nullptr, attn_kernel<L, false, 7>};
#endif
#if OKC_SPEC
  static const KernFn ksp[4] = {nullptr, attn_kernel<L, false, 9>, nullptr, attn_kernel<L, false, 11>};
  static const KernFn ksp1[4] = {nullptr, attn_kernel<L, false, 13>, nullptr, attn_kernel<L, false, 15>};
#endif
  if (!init) {
    for (int m = 1; m < 4; ++m)
      cudaFuncSetAttribute(kn[m], cudaFuncAttributeMaxDynamicSharedMemorySize, (m == 2 && OKC_CLSM > sm_n) ? OKC_CLSM : sm_n);
#if OKC_ONE
    cudaFuncSetAttribute(kn1[1], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);
    cudaFuncSetAttribute(kn1[3], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);
#endif
#if OKC_SPEC
    for (int m = 1; m < 4; m += 2) {
      cudaFuncSetAttribute(ksp[m], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);
      cudaFuncSetAttribute(ksp1[m], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);
    }
#endif
    cudaFuncSetAttribute(attn_kernel<L, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_f);
    if (clusters) cudaFuncSetAttribute(kn[2], cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_c, kn[3], Cfg<L, false>::NTX, sm_n);
#if OKC_ONE
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_c1, kn1[3], Cfg<L, false>::NTX, sm_n);
#endif
#if OKC_SPEC
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_sp, ksp[3], Cfg<L, false>::NTX, sm_n);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_sp1, ksp1[3], Cfg<L, false>::NTX, sm_n);
#endif
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_f, attn_kernel<L, true>, Cfg<L, true>::NTX, sm_f);
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    init = true;
  }
  // the cooperative finalize waits for every split of a request: only when the whole grid is resident
  Params q = p;
  if (q.coop && (long long)grid.x * grid.y * grid.z > (long long)(p.flat ? occ_f : occ_c) * nsm) q.coop = 0;
  q.resident = (p.flat && grid.y * grid.z == 1 && (int)grid.x == 2 * nsm && occ_f >= 2) ? 1 : 0;
  q.unev = (OKC_UNEV && !p.flat && q.coop && p.w0 != 1024 && grid.y * grid.z == 1 && (int)grid.x > nsm &&
            (int)grid.x <= 2 * nsm) ? nsm : 0;
  const bool cl = cs > 1 && clusters;
  KernFn kern = p.flat ? (KernFn)attn_kernel<L, true> : kn[cl ? 2 : (q.coop ? 3 : 1)];
#if OKC_ONE
  if (!p.flat && !cl && !(OKC_UNEV && q.unev)) {
    // every split holds at most one tile: whole tiles per split (TCHUNK) and at least as many splits as the longest
    // request's tiles (sliding windows: window + L - 1 keys)
    const long long kmax = p.window ? std::min<long long>((long long)p.window + L - 1, (long long)p.max_pages * p.P)
                                    : (long long)p.max_pages * p.P;
    if (OKC_TCHUNK && (long long)p.n_splits * 64 >= kmax) kern = kn1[q.coop ? 3 : 1];
  }
#endif
#if OKC_SPEC
  // one request, page size 1: the speculative-length builds (OKC_SPEC; the kernel verifies and falls back)
  if (!p.flat && !cl && !(OKC_UNEV && q.unev) && p.B == 1 && p.P == 1 && p.max_pages >= L && (int)grid.y == 1) {
    const bool one = kern != kn[q.coop ? 3 : 1];
    kern = one ? ksp1[q.coop ? 3 : 1] : ksp[q.coop ? 3 : 1];
  }
#endif
  if (OKC_COOPGUARD && q.coop && !p.flat && !cl) {
    // the instance actually launched must hold the whole grid, else its grid barrier would wait forever
    int occ_k = occ_c;
    KernFn plain = kn[1];
#if OKC_ONE
    if (kern == kn1[3]) { occ_k = occ_c1; plain = kn1[1]; }
#endif
#if OKC_SPEC
    if (kern == ksp[3]) { occ_k = occ_sp; plain = ksp[1]; }
    if (kern == ksp1[3]) { occ_k = occ_sp1; plain = ksp1[1]; }
#endif
#ifdef OKC_COOPGUARD_TEST
    occ_k = 0;   // (test builds: always take the fallback)
#endif
    if ((long long)grid.x * grid.y * grid.z > (long long)occ_k * nsm) {
      q.coop = 0;
      kern = plain;
    }
  }
  if (!p.flat && cs > 1 && !clusters) kern = kn[0];   // (sm_80 emulation of clusters: run-time mode)
  if (cl || pdl) {
    // cluster dims and / or programmatic stream serialization (PDL: this grid may start while the preceding
    // kernel finishes; the kernel waits with griddepcontrol.wait before reading qkv)
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(p.flat ? Cfg<L, true>::NTX : Cfg<L, false>::NTX, 1, 1);
    cfg.dynamicSmemBytes = (cl && OKC_CLSM > sm) ? OKC_CLSM : sm;   // (OKC_CLSM: one cluster CTA per SM)
    cfg.stream = stream;
    cudaLaunchAttribute attr[2];
    int na = 0;
    if (cl) {
      attr[na].id = cudaLaunchAttributeClusterDimension;
      attr[na].val.clusterDim.x = cs;
      attr[na].val.clusterDim.y = 1;
      attr[na].val.clusterDim.z = 1;
      ++na;
    }
    if (pdl) {
      attr[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr[na].val.programmaticStreamSerializationAllowed = 1;
      ++na;
    }
    cfg.attrs = attr;
    cfg.numAttrs = na;
    cudaLaunchKernelEx(&cfg, kern, q);
  } else {
    kern<<<grid, p.flat ? Cfg<L, true>::NTX : Cfg<L, false>::NTX, sm, stream>>>(q);
  }
}

typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                   const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                   CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeTiledFn get_encode() {
  static EncodeTiledFn encode = nullptr;
  if (!encode) {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &q);
    TORCH_CHECK(fn != nullptr, "cuTensorMapEncodeTiled unavailable");
    encode = reinterpret_cast<EncodeTiledFn>(fn);
  }
  return encode;
}
// the page table as one int32 vector, 64-entry boxes, zeros past the end
static CUtensorMap pt_map(void* base, long long n) {
  CUtensorMap m;
  cuuint64_t dims[1] = {(cuuint64_t)n};
  cuuint64_t strides[1] = {0};
  cuuint32_t box[1] = {64};
  cuuint32_t es[1] = {1};
  const CUresult r = get_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_INT32, 1, base, dims, strides, box, es,
                                  CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                  CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (page table) failed: ", (int)r);
  return m;
}
// [rows, row_bytes] byte view of a cache, boxes of one 128-byte row (one KV pair), 128-byte swizzle
static CUtensorMap row_map(void* base, long long rows, long long row_bytes) {
  static EncodeTiledFn encode = nullptr;
  if (!encode) {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &q);
    TORCH_CHECK(fn != nullptr, "cuTensorMapEncodeTiled unavailable");
    encode = reinterpret_cast<EncodeTiledFn>(fn);
  }
  CUtensorMap m;
  cuuint64_t dims[2] = {(cuuint64_t)row_bytes, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)row_bytes};
  cuuint32_t box[2] = {128, 1};
  cuuint32_t es[2] = {1, 1};
  const CUresult r = encode(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, base, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}

void run(torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor cache_loc, torch::Tensor k_cache,
         torch::Tensor v_cache, torch::Tensor page_table, torch::Tensor k_scale, torch::Tensor v_scale,
         torch::Tensor inv_freq, torch::Tensor out, torch::Tensor ws_acc, torch::Tensor ws_l,
         torch::Tensor ws_cnt, int64_t B, int64_t L, int64_t Hq, int64_t Hkv, int64_t P, int64_t window,
         int64_t rope, int64_t n_splits, int64_t flat, int64_t nc, double scale_log2,
         std::optional<torch::Tensor> dbg, int64_t cs, int64_t ncl, torch::Tensor scratch, torch::Tensor bar,
         int64_t tma) {
  Params p;
  p.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  p.seq_lens = seq_lens.data_ptr<int>();
  p.cache_loc = cache_loc.data_ptr<int64_t>();
  p.k_cache = reinterpret_cast<uint8_t*>(k_cache.data_ptr());
  p.v_cache = reinterpret_cast<uint8_t*>(v_cache.data_ptr());
  p.page_table = page_table.data_ptr<int>();
  p.k_scale = k_scale.data_ptr<float>();
  p.v_scale = v_scale.data_ptr<float>();
  p.inv_freq = inv_freq.data_ptr<float>();
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.ws_acc = ws_acc.data_ptr<float>();
  p.ws_l = ws_l.data_ptr<float>();
  p.ws_cnt = ws_cnt.data_ptr<int>();
  p.B = (int)B; p.Hq = (int)Hq; p.Hkv = (int)Hkv; p.P = (int)P;
  TORCH_CHECK(P > 0 && (P & (P - 1)) == 0, "page size must be a power of two");
  TORCH_CHECK(k_cache.numel() / (Hkv * 64) < (1LL << 31), "cache rows must fit in 31 bits");
  p.lp = 0;
  while ((1 << p.lp) < P) ++p.lp;
  p.max_pages = (int)page_table.size(1);
  p.window = (int)window; p.rope = (int)rope; p.n_splits = (int)n_splits; p.flat = (int)flat; p.nc = (int)nc;
  p.ns_magic = (n_splits >= 2 && n_splits <= 1024) ? (unsigned)(((1ull << 32) + (uint64_t)n_splits - 1) / (uint64_t)n_splits) : 0u;
  TORCH_CHECK(n_splits <= 1024 || !OKC_MAGICDIV, "OKC_MAGICDIV: n_splits <= 1024");
  p.dbg = dbg.has_value() ? reinterpret_cast<long long*>(dbg->data_ptr<int64_t>()) : nullptr;
  p.cs = (int)cs; p.ncl = (int)ncl;
  p.scratch = scratch.data_ptr<float>();
  p.bar = bar.data_ptr<int>();
  p.ld_qkv = (int)qkv.size(-1);
  p.row_bytes = (int)(Hkv * 64);
  TORCH_CHECK((p.row_bytes & (p.row_bytes - 1)) == 0, "Hkv * 64 must be a power of two");
  p.lrb = 0;
  while ((1 << p.lrb) < p.row_bytes) ++p.lrb;
  p.off32 = (k_cache.numel() < (1LL << 32) && v_cache.numel() < (1LL << 32)) ? 1 : 0;   // larger caches: 64-bit offsets
  TORCH_CHECK(!OKC_OFF32 || ws_acc.numel() < (1LL << 31), "OKC_OFF32: workspace too large");
  TORCH_CHECK(B * (Hkv / 2) <= 4096, "at most 4096 (request, kv pair) slots in the counter workspace");
  TORCH_CHECK(!OKC_OFF32 || qkv.numel() < (1LL << 31), "OKC_OFF32: qkv too large");
  TORCH_CHECK(!OKC_OFF32 || (out.numel() < (1LL << 31) && page_table.numel() < (1LL << 31)), "OKC_OFF32: out / page table too large");
  p.scale_log2 = (float)scale_log2;
  dim3 grid((unsigned)n_splits, flat ? 1u : (unsigned)B, (unsigned)(Hkv / 2));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  static int major = -1;
  if (major < 0) {
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  }
  p.tma = (OKC_TMA && (tma & 1) && major >= 9) ? 1 : 0;
  p.coop = OKC_COOP ? (int)((tma >> 1) & 1) : 0;
  p.w0 = (int)((tma >> 2) & 4095);
  if (p.w0 == 0) p.w0 = 1024;
  if (p.tma) {
    const long long rows = k_cache.numel() / (Hkv * 64);
    p.tk = row_map(k_cache.data_ptr(), rows, Hkv * 64);
    p.tv = row_map(v_cache.data_ptr(), rows, Hkv * 64);
  } else {
    memset(&p.tk, 0, sizeof(p.tk));
    memset(&p.tv, 0, sizeof(p.tv));
  }
  memset(&p.tpt, 0, sizeof(p.tpt));
  const bool pdl = ((tma >> 16) & 1) && major >= 9;   // programmatic dependent launch (off by default)
  if (L == 1) launch<1>(p, grid, stream, (int)cs, major >= 9, pdl);
  else launch<8>(p, grid, stream, (int)cs, major >= 9, pdl);
}
"""

att_CPP_SRC = r"""
void run(torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor cache_loc, torch::Tensor k_cache,
         torch::Tensor v_cache, torch::Tensor page_table, torch::Tensor k_scale, torch::Tensor v_scale,
         torch::Tensor inv_freq, torch::Tensor out, torch::Tensor ws_acc, torch::Tensor ws_l,
         torch::Tensor ws_cnt, int64_t B, int64_t L, int64_t Hq, int64_t Hkv, int64_t P, int64_t window,
         int64_t rope, int64_t n_splits, int64_t flat, int64_t nc, double scale_log2,
         std::optional<torch::Tensor> dbg, int64_t cs, int64_t ncl, torch::Tensor scratch, torch::Tensor bar,
         int64_t tma);
"""

att__MOD = None
att__INV_FREQ = {}
att__WS = {}
att__NUM_SMS = {}
att_LOG2E = 1.4426950408889634


def att__build(tag="", flags=()):
    from torch.utils.cpp_extension import load_inline
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}" + ("a" if major >= 9 else "")
    return load_inline(
        name=f"okc_attn_v298{tag}_{arch}",
        cpp_sources=att_CPP_SRC,
        cuda_sources=att_CUDA_SRC,
        functions=["run"],
        extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", f"-DOKC_SM90={1 if major >= 9 else 0}",
                           "--expt-relaxed-constexpr", *flags],
        verbose=False,
    )


def att__module():
    global att__MOD
    if att__MOD is None:
        att__MOD = att__build()
    return att__MOD


def att__inv_freq(theta, device):
    key = (float(theta), str(device))
    t = att__INV_FREQ.get(key)
    if t is None:
        if theta:
            t = 1.0 / (theta ** (torch.arange(0, 64, 2, device=device).float() / 64))
        else:
            t = torch.zeros(32, device=device)
        att__INV_FREQ[key] = t.float().contiguous()
    return t


def att__workspace(n_rows, device):
    """Persistent FP32 accumulators (zero between calls: the last CTA resets them)."""
    key = str(device)
    ws = att__WS.get(key)
    if ws is None or ws[0].numel() < n_rows * 64:
        n = max(n_rows, 1 << 15)
        ws = (torch.zeros(n * 64, dtype=torch.float32, device=device),
              torch.zeros(n, dtype=torch.float32, device=device),
              torch.zeros(n, dtype=torch.int32, device=device))
        att__WS[key] = ws
    return ws


def att__num_sms(device):
    key = str(device)
    if key not in att__NUM_SMS:
        att__NUM_SMS[key] = torch.cuda.get_device_properties(device).multi_processor_count
    return att__NUM_SMS[key]


# per-shape launch overrides measured on the judge's H200: (batch, q_len, sliding, kv pairs) -> plan fields
att__TUNED = {
    # (Oct 26 14:16-14:21, podbench_multi 15 rounds x2: b32 q8 sliding at 4 splits per request -- 128 CTAs, one per SM --
    # 16.10 -> 15.50 us; 3 splits +13%, 5 +10.6%, 6 +3.1%: since the faster per-request q8 loop (LATESYNCN, QKFIRST))
    (32, 8, True, 1): dict(n_splits=4, nc=1, coop=1),
    # (Oct 26 14:44-14:48: b1 q1 full with the cooperative finalize over 8 copies: GPU-1 plan sweep -1.6%, podbench_multi
    # 12 rounds 12.30 -> 12.00 us; coop with 1/2/4 copies +26/+16/+4%: same-address red serialization of 250 splits)
    (1, 1, False, 1): dict(nc=8, coop=1, n_splits=252),   # (experiment: 252 instead of 264 splits)
}


att_OKC_FLATCOOP = 1
att_OKC_PDL = 0      # launch with programmatic dependent launch (for a fused layer: overlaps the qkv GEMM's tail)
att_OKC_FLAT1B = 0   # q_len 1, one full-layer request: flat kernel (Oct 26 00:14: the per-request coop plan, 264 splits, 4 copies, is now faster: f1 13.2 -> 12.6 us)
att_OKC_PLAIN8 = 1   # q_len 8 sliding, >= 16 requests: plain split-KV, ~1 CTA per SM
att_OKC_F2 = 0   # must match the CUDA macro OKC_F2 (q_len 8 flat: 1 CTA per SM, two streams)
att_OKC_S2 = 0   # must match the CUDA macro OKC_S2 (q_len 8 per-request kernels: 2 streams, 1 CTA per SM)
att_OKC_W0_L1 = 1024   # flat: first CTA's share on each SM, /1024 (uneven shares measured slower: the faster CTA is not always the first)
att_OKC_W0_PR = 1024   # per-request coop grids, one request, more splits than SMs: first-wave tile weight /1024
att_OKC_W0_L8 = 1135   # q_len 8 flat: the lower-blockIdx CTA of each SM takes 1135/2048 (Oct 25 re-sweeps: bottom 1130-1140; 1180 overloads it)


def att__plan(B, pairs, L, window, max_pages, P, device):
    """Launch plan for a shape: n_splits CTAs per (request, KV pair) or, flat, per KV pair over all
    requests; clusters of cs CTAs reduce through DSMEM (ncl clusters per request); nc workspace copies."""
    sms = att__num_sms(device)
    target = 2 * sms
    if not window and 1 < B <= 512:
        # full layer, several requests: equal shares of all requests' keys, one CTA pass per SM pair;
        # several workspace copies spread the end-of-run atomics (H200: b32 L1 206 us with 8 vs 237 with 1)
        ft = sms if (L == 8 and att_OKC_F2 and torch.cuda.get_device_capability(device)[0] >= 9) else target
        # one workspace copy since the 16-byte merge atomics and the parity barriers (v101: q_len 1 b16 91.4 -> 90.0,
        # b32 167.7 -> 166.1 with 264 CTAs; q_len 8 b32 199.6 -> 198.0); single requests keep their copies (contention)
        return dict(n_splits=max(1, ft // pairs), flat=1, cs=0, ncl=1, nc=1)
    keys_max = min(window + L - 1, max_pages * P) if window else max_pages * P
    tiles = max(1, (keys_max + 63) // 64)
    reqs = B * pairs
    if L == 8 and att_OKC_S2 and torch.cuda.get_device_capability(device)[0] >= 9:
        target = sms   # two tile streams per CTA, one CTA per SM (the kernel's OKC_S2)
    if not window:
        if L == 1 and B == 1 and att_OKC_FLAT1B:
            # one long request, q_len 1: the flat kernel (producer warps) over 2 CTAs per SM (H200: 14.4 -> 13.9 us)
            return dict(n_splits=max(1, ((target * 21) // 22) // pairs), flat=1, cs=0, ncl=1, nc=4)
        # one long request: plain split-KV over 2 CTAs per SM (clusters measured slower here); q_len 8: 252 CTAs
        # and 8 workspace copies (H200 b1: 20.7 -> 19.75 us vs 264 and 4)
        if L == 8:
            n = max(1, min(tiles, ((target * 21) // 22) // reqs))
            return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=max(1, min(n, 8)))
        n = max(1, min(tiles, target // reqs))
        # (q_len 1: the last CTA finalizes, over 4 workspace copies -- Oct 26 00:26, f1 per-request: coop 13.0-13.1 us,
        # last-CTA 12.6-12.9, one copy 15.2, two 14.0, eight 12.9-13.2; the flat kernel 13.6)
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=max(1, min(n, 4)), coop=0)
    if L == 1 and reqs < 24:
        # judge timings: plain split-KV with one tile per CTA and a single workspace copy beats clusters here
        # (Oct 25 21:36, after the cheaper merges: b1 33 splits (one tile each) 6.6 -> 6.5 us vs 32, b16 16 splits
        # 9.4 -> 9.2 us vs 12)
        n = max(1, min(tiles, target // reqs))
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1)
    if L == 8 and reqs < 16 and att_OKC_PLAIN8:
        # q_len 8, few requests: plain split-KV with one tile per CTA and the cooperative finalize (every CTA
        # normalizes a slice of the 128 rows) beats clusters since the inlined finalizes (H200 b1: 13.0 -> 10.2 us)
        n = max(1, min(tiles, target // reqs))
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1, coop=1)   # (one copy since v101: b1 9.8 -> 9.6 us)
    if L == 8 and reqs >= 16 and att_OKC_PLAIN8:
        # q_len 8, many requests: plain split-KV with about one CTA per SM (H200 since the inlined last-CTA
        # finalize: b16 19.0 -> 17.2 us vs clusters of 8, b32 22.0 -> 21.1 us vs two CTAs per SM); the cooperative
        # finalize helps at 8 splits per request (b16 16.2 -> 15.85 us), and at 4 (b32) since the parity barriers
        # (v102: 20.15 -> 19.55 us); b32: two CTAs per SM (8 splits) since v126 (19.4 -> 18.25 us)
        n = max(1, min(tiles, (target if reqs >= 24 else sms) // reqs))
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1, coop=1)
    if reqs >= 24:
        # many requests: clusters no longer fit one CTA per SM (GPC sizes: 32 clusters of 4 put 16 CTAs two to
        # an SM, which then finish last); plain split-KV spreads evenly (H200: b32 q8 27.7 -> 23.5, q1 14.4 -> 13.1)
        n = max(1, min(tiles, target // reqs))
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1)
    # sliding window: one cluster per (request, KV pair); keep the grid small enough that all clusters
    # are resident at once
    cs = 16
    limit = 2 * sms if L == 1 else sms
    if L == 8 and target == sms:
        limit = (sms * 15) // 16   # one CTA per SM: clusters of 8 fit 15 per 132 SMs (GPC sizes), so stay below
    while cs > 4 and reqs * cs > limit:
        cs //= 2
    ncl = 1
    while cs * ncl > tiles and ncl > 1:
        ncl -= 1
    while cs * ncl > tiles and cs > 4:
        cs //= 2
    if cs * ncl > tiles or cs < 4:
        n = max(1, min(tiles, target // reqs))
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=max(1, min(n, 4)))
    return dict(n_splits=cs * ncl, flat=0, cs=cs, ncl=ncl, nc=max(1, min(ncl, 4)))


att__CL = {}


def att__cluster_bufs(n_ctas, L, device):
    """sm_80 emulation buffers for the cluster path (unused on sm_90 beyond a pointer)."""
    key = str(device)
    need = n_ctas * (L * 16 * 64 + 1024)
    cur = att__CL.get(key)
    if cur is None or cur[0].numel() < need or cur[1].numel() < n_ctas:
        cur = (torch.zeros(max(need, 1 << 20), dtype=torch.float32, device=device),
               torch.zeros(max(n_ctas, 1 << 12), dtype=torch.int32, device=device))
        att__CL[key] = cur
    return cur


def att__launch(data, dbg=None, _mod=None, pdl=None, **override):
    cfg = data["config"]
    B, L = cfg["batch"], cfg["q_len"]
    hq, hkv, d, P = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["page_size"]
    qkv = data["qkv"]
    device = qkv.device
    pairs = hkv // 2
    window = cfg["window"]
    page_table = data["page_table"]
    out = torch.empty((B, L, hq * d), dtype=torch.bfloat16, device=device)
    plan = att__plan(B, pairs, L, window, page_table.shape[1], P, device)
    tuned = att__TUNED.get((B, L, bool(window), pairs))
    if tuned and not override:
        override = tuned
    if override:
        plan = dict(plan, **override)
        if "cs" in override and override["cs"] > 1:
            plan["flat"] = 0
            if "ncl" not in override:
                plan["ncl"] = max(1, plan["n_splits"] // override["cs"])
            plan["n_splits"] = plan["cs"] * plan["ncl"]
            if "nc" not in override:
                plan["nc"] = max(1, min(plan["ncl"], 4))
        elif "n_splits" in override and "nc" not in override:
            plan["nc"] = 1 if plan["flat"] else max(1, min(plan["n_splits"], 4))
    n_splits, flat, cs, ncl, nc = plan["n_splits"], plan["flat"], plan["cs"], plan["ncl"], plan["nc"]
    # bit 0: TMA loads (compiled out by default); bit 1: cooperative finalize (all CTAs resident)
    # cooperative finalize: only for the long single-request grids (their last-CTA finalize reads and zeroes
    # tens of KB from one SM); the short sliding grids finalize faster in one CTA (measured on H200)
    if flat:   # grid-wide finalize after all segments (the kernel re-checks residency)
        coop = int(plan.get("coop", int(att_OKC_FLATCOOP and L == 8 and
                                        n_splits * pairs <= 2 * att__num_sms(device))))
    else:
        coop = int(plan.get("coop", 0 if (cs > 1 or n_splits < 64) else
                            int(n_splits * B * pairs <= 2 * att__num_sms(device))))
    w0 = int(plan.get("w0", (att_OKC_W0_L1 if L == 1 else att_OKC_W0_L8) if (flat and (coop or L == 1)) else
                      (att_OKC_W0_PR if (coop and not flat and B * pairs == 1) else 1024)))
    use_pdl = att_OKC_PDL if pdl is None else bool(pdl)
    tma = int(plan.get("tma", 0)) | (coop << 1) | (min(max(w0, 0), 4095) << 2) | (int(use_pdl) << 16)
    if flat or cs <= 1:
        cs, ncl = 0, 1
    ws_acc, ws_l, ws_cnt = att__workspace(B * pairs * nc * L * 16, device)
    n_ctas = n_splits * (1 if flat else B) * pairs
    scratch, bar = att__cluster_bufs(n_ctas if cs > 1 else 1, L, device)
    scale_log2 = cfg["sm_scale"] * cfg["qk_scale_factor"] * att_LOG2E
    (_mod or att__module()).run(qkv, data["seq_lens"], data["cache_loc"], data["k_cache"], data["v_cache"], page_table,
                  data["k_scale"], data["v_scale"], att__inv_freq(cfg["rope_theta"], device), out, ws_acc, ws_l,
                  ws_cnt, B, L, hq, hkv, P, window, 1 if cfg["rope_theta"] else 0, n_splits, flat, nc, scale_log2,
                  dbg, cs, ncl, scratch, bar, tma)
    return out


def att__supported(cfg):
    P = cfg["page_size"]
    return (cfg["head_dim"] == 64 and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"] and cfg["num_kv_heads"] % 2 == 0
            and cfg["q_len"] in (1, 8) and cfg["batch"] <= 512 and P > 0 and (P & (P - 1)) == 0
            and cfg["batch"] * (cfg["num_kv_heads"] // 2) <= 4096)


def att__rms(x, eps=1e-5):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def att__rope(x, pos, theta):
    d = x.shape[-1]
    inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device).float() / d))
    ang = pos.float()[..., None] * inv_freq
    cos, sin = ang.cos().unsqueeze(-2), ang.sin().unsqueeze(-2)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack([even * cos - odd * sin, even * sin + odd * cos], -1).flatten(-2)


def att__fallback(data):
    """Plain PyTorch path for shapes the CUDA kernel does not cover (other q_len / head layouts)."""
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d, page = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["page_size"]
    q_in, k_in, v_in, gate = data["qkv"].split([hq * d, hkv * d, hkv * d, hq * d], -1)
    pos = data["positions"].view(batch, length)
    q = att__rms(q_in.reshape(batch, length, hq, d)) * cfg["qk_scale_factor"]
    k = att__rms(k_in.reshape(batch, length, hkv, d))
    if cfg["rope_theta"]:
        q, k = att__rope(q, pos, cfg["rope_theta"]), att__rope(k, pos, cfg["rope_theta"])
    q = q.to(torch.bfloat16)
    table = data["page_table"].long()
    slots = data["cache_loc"].view(batch, length)
    k_cache = data["k_cache"].view(-1, hkv, d)
    v_cache = data["v_cache"].view(-1, hkv, d)
    k_scale, v_scale = data["k_scale"], data["v_scale"]
    k_cache[slots] = (k / k_scale).to(torch.float8_e4m3fn)
    v_cache[slots] = (v_in.reshape(batch, length, hkv, d).float() / v_scale).to(torch.float8_e4m3fn)
    t = torch.arange(table.shape[1] * page, device=q.device)
    out = torch.empty((batch, length, hq, d), dtype=torch.bfloat16, device=q.device)
    for b in range(batch):
        rows = table[b].clamp(min=0)[t // page] * page + t % page
        keys = k_cache[rows].float() * k_scale
        vals = v_cache[rows].float() * v_scale
        visible = t[None, :] <= pos[b][:, None]
        if cfg["window"]:
            visible &= t[None, :] > pos[b][:, None] - cfg["window"]
        att = torch.einsum("lkgd,skd->lkgs", q[b].view(length, hkv, hq // hkv, d).float(), keys) * cfg["sm_scale"]
        att = att.masked_fill(~visible[:, None, None, :], float("-inf")).softmax(-1)
        out[b] = torch.einsum("lkgs,skd->lkgd", att, vals).reshape(length, hq, d)
    out = att__rms(out) * torch.sigmoid(gate.float()).view(batch, length, hq, d)
    return out.reshape(batch, length, hq * d).to(torch.bfloat16)


def att__fits(data):
    """Sizes the CUDA path's 32-bit indexing covers (caches of any size are fine: 64-bit offsets past 4 GB)."""
    cfg = data["config"]
    out_elems = cfg["batch"] * cfg["q_len"] * cfg["num_q_heads"] * cfg["head_dim"]
    return (data["qkv"].numel() < (1 << 31) and data["page_table"].numel() < (1 << 31) and out_elems < (1 << 31))


att_OKC_BENCH = 0      # 1: one on-device sweep of flat-kernel builds in the first eager call (never under graph capture); prints to the log
att__BENCH_DONE = False
# (name, extra nvcc flags) of the builds the sweep times on synthetic inputs shaped like the scored full-context cases
att_BENCH_VARIANTS = [
    ("base", ()),
    ("pfl2_1", ("-DOKC_PFL2=1",)),
    ("pfl2_2", ("-DOKC_PFL2=2",)),
]
att_BENCH_SHAPES = [(16, 1), (32, 1)]   # (batch, q_len), full layer, ctx 80000 agentic
# launch-plan sweeps on the base build: ((batch, q_len, window), [override dicts]) ({} = today's plan)
att_W2K = 2048
att_BENCH_PLANS = []
att_BENCH_BUDGET_S = 100
import os as _os
att_BENCH_WORKERS = max(1, min(6, (_os.cpu_count() or 1)))


def att__bench_inputs(data, B, L, ctx=80000, seed=0, window=0):
    """Synthetic inputs like the judge's generator: scattered pages, agentic lengths (fixed at batch 1)."""
    dev = data["qkv"].device
    cfg0 = dict(data["config"])
    HQ, HKV, D = cfg0["num_q_heads"], cfg0["num_kv_heads"], cfg0["head_dim"]
    g = torch.Generator(device=dev).manual_seed(seed)
    longest = math.ceil(ctx * 1.075)
    max_pages = longest + L
    u = torch.rand((B,), generator=g, device=dev)
    seq = (ctx * (1 + 0.075 * (2 * u - 1))).long().clamp(1, longest)
    if B == 1:
        longest = ctx
        max_pages = ctx + L
        seq = torch.full((1,), ctx, dtype=torch.long, device=dev)
    total = B * max_pages
    perm = torch.randperm(total, generator=g, device=dev).to(torch.int32)
    page_table = torch.full((B, max_pages), -1, dtype=torch.int32, device=dev)
    pages_per = seq + L
    offs = torch.cumsum(pages_per, 0) - pages_per
    for b in range(B):
        n, o = int(pages_per[b]), int(offs[b])
        page_table[b, :n] = perm[o:o + n]
    k_cache = torch.randn((total, 1, HKV, D), generator=g, device=dev).to(torch.float8_e4m3fn)
    v_cache = torch.randn((total, 1, HKV, D), generator=g, device=dev).to(torch.float8_e4m3fn)
    pos = (seq[:, None] + torch.arange(L, device=dev)).reshape(-1)
    loc = torch.gather(page_table.long(), 1, pos.view(B, L)).reshape(-1)
    qkv = (2 * torch.randn((B * L, (2 * HQ + 2 * HKV) * D), generator=g, device=dev)).to(torch.bfloat16)
    c = dict(cfg0, batch=B, q_len=L, page_size=1, window=window, rope_theta=500000.0 if window else 0.0)
    return {"qkv": qkv, "positions": pos, "cache_loc": loc, "cache_seqlens": seq.int(), "cu_seqlens_q": None,
            "scheduler_metadata": None, "k_cache": k_cache, "v_cache": v_cache, "k_scale": data["k_scale"],
            "v_scale": data["v_scale"], "page_table": page_table, "seq_lens": seq.int(), "config": c}


def att__bench_timeit(fn, flush, sink, reps=9):
    """us per CUDA-graph replay after an L2 flush (median, best), as the judge times."""
    fn(); fn()
    torch.cuda.synchronize()
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        gph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gph, stream=st):
            fn()
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(reps):
        sink.copy_(flush.view(-1, 1 << 16).sum(1, dtype=torch.int32))   # cold L2 by reading, as the judge does
        e0.record()
        gph.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2], ts[0]


def att__bench(data):
    """The sweep (in process, as the judge allows no child processes): variants over BENCH_SHAPES, then the plan
    sweeps on the base build. Stops at the first CUDA error (the context is sticky-dead then)."""
    import time, hashlib
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    t_end = t0 + att_BENCH_BUDGET_S
    dev = data["qkv"].device
    flush = torch.zeros((160 << 20,), dtype=torch.uint8, device=dev)
    sink = torch.empty(((160 << 20) >> 16,), dtype=torch.int32, device=dev)
    nv = len(att_BENCH_VARIANTS)
    print(f"BENCH attention: {nv} flat builds x {att_BENCH_SHAPES} + {len(att_BENCH_PLANS)} plan sweeps (us per graph replay after an L2 flush: median / best)")

    def build(item):
        name, flags = item[0], item[1]
        if not flags:
            return name, att__module(), None
        tag = "_b" + hashlib.md5(" ".join(flags).encode()).hexdigest()[:8]
        try:
            return name, att__build(tag=tag, flags=("-DOKC_FLATONLY=1", *flags)), None
        except Exception as e:   # noqa: BLE001
            return name, None, str(e)[-500:]

    with ThreadPoolExecutor(max_workers=att_BENCH_WORKERS) as ex:
        built = {n: (m, err) for n, m, err in ex.map(build, att_BENCH_VARIANTS)}
    print(f"BENCH builds done in {time.time() - t0:.0f}s: " +
          ", ".join(f"{n}:{'ok' if m else 'FAIL ' + (err or '')[-200:]}" for n, (m, err) in built.items()))
    base_mod = att__module()
    inputs = {(B, L): att__bench_inputs(data, B, L) for (B, L) in att_BENCH_SHAPES}
    refs = {}
    for v in att_BENCH_VARIANTS:
        name, flags = v[0], v[1]
        ov_by_L = v[2] if len(v) > 2 else {}
        mod = built.get(name, (None, None))[0]
        if mod is None:
            continue
        for (B, L) in att_BENCH_SHAPES:
            if time.time() > t_end:
                print("BENCH budget reached"); break
            d = inputs[(B, L)]
            ov = ov_by_L.get(L, {})
            try:
                o = att__launch(d, _mod=mod, **ov)
                torch.cuda.synchronize()
                if (B, L) not in refs:
                    refs[(B, L)] = att__launch(d, _mod=base_mod).float()
                    torch.cuda.synchronize()
                ref = refs[(B, L)]
                err = float((o.float() - ref).norm() / (ref.norm() + 1e-20))
                us, best = att__bench_timeit(lambda: att__launch(d, _mod=mod, **ov), flush, sink)
                print(f"BENCH B{B} L{L} full {name:12s} {str(flags) + (str(ov) if ov else ''):58s} {us:8.1f} / {best:8.1f} us  relerr {err:.2e}")
            except Exception as e:   # noqa: BLE001
                print(f"BENCH B{B} L{L} full {name:12s} failed: {str(e)[-200:]}")
                if "CUDA" in str(e) or "illegal" in str(e):
                    print(f"BENCH stopped: sticky CUDA error in {name}")
                    return
    del inputs, refs
    for (B, L, window), plans in att_BENCH_PLANS:
        if time.time() > t_end:
            print("BENCH budget reached"); break
        d = att__bench_inputs(data, B, L, window=window)
        ref = None
        for ov in plans:
            try:
                o = att__launch(d, _mod=base_mod, **ov)
                torch.cuda.synchronize()
                if ref is None:
                    ref, err = o.float(), 0.0
                else:
                    err = float((o.float() - ref).norm() / (ref.norm() + 1e-20))
                us, best = att__bench_timeit(lambda: att__launch(d, _mod=base_mod, **ov), flush, sink)
                print(f"BENCH B{B} L{L} {'sliding' if window else 'full':7s} plan {str(ov):50s} {us:8.1f} / {best:8.1f} us  relerr {err:.2e}")
            except Exception as e:   # noqa: BLE001
                print(f"BENCH B{B} L{L} plan {ov} failed: {str(e)[-200:]}")
                if "CUDA" in str(e) or "illegal" in str(e):
                    print("BENCH stopped: sticky CUDA error in the plan sweeps")
                    return
        del d
    # probe builds may leave garbage in the shared workspace: back to the all-zero state
    for t in att__WS.get(str(dev), ()):
        t.zero_()
    torch.cuda.synchronize()
    print(f"BENCH done in {time.time() - t0:.0f}s")


def att_kernel_fn(data: att_Inputs) -> torch.Tensor:
    global att__BENCH_DONE
    if att_OKC_BENCH and not att__BENCH_DONE and torch.cuda.is_available() and not torch.cuda.is_current_stream_capturing():
        att__BENCH_DONE = True
        try:
            att__bench(data)
        except Exception as e:   # noqa: BLE001
            print("BENCH failed:", repr(e)[-400:])
    if not att__supported(data["config"]) or not att__fits(data):
        return att__fallback(data)
    return att__launch(data)
# ---- end of component: attn ----


# ==============================================================================================================
# component: moe (muse_moe_w4a16/submission.py); every module-level name of that file carries the prefix `moe_`
# ==============================================================================================================
# Derived work. Base: https://github.com/vjeux/okc-2026 (muse_moe_w4a16.py), Copyright (c) 2026 vjeux,
# licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
# Modifications Copyright (c) 2026 Madhavan113, same license. See NOTICE in the repository root.
"""Routed experts (NVFP4 W4A16 MoE) as one persistent CUDA kernel plus two tiny ones.

route_kernel builds the per-expert token lists on device; moe_main streams each touched expert's
prepacked NVFP4 weights once (GEMM1 + SwiGLU, then GEMM2 + scaled output + sum of squares for the
post-expert RMSNorm); final_kernel applies the norm and the routing weights and sums per token.
Weights are only repacked (bit/byte permutations), never requantized.
"""

from typing import TypedDict

import torch
from torch.utils.cpp_extension import load_inline


class moe_Weights(TypedDict):
    w13: torch.Tensor
    w13_sf: torch.Tensor
    w13_gs: torch.Tensor
    w2: torch.Tensor
    w2_sf: torch.Tensor
    w2_gs: torch.Tensor
    post_norm_w: torch.Tensor


class moe_Config(TypedDict):
    bs: int
    d_hidden: int
    d_expert: int
    n_experts: int
    local_start: int
    local_count: int
    top_k: int
    sharding: str


class moe_Inputs(TypedDict):
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    config: moe_Config


moe_CUDA_SRC = r'''
// muse_moe_w4a16 v2: one persistent kernel: routing prologue, GEMM1+SwiGLU, GEMM2+norm stats, final norm/sum.
#include <cuda_bf16.h>
#include <type_traits>
#include <cuda_runtime.h>
#include <stdint.h>
#ifndef OKC_DEBUG
#define OKC_DEBUG 0
#endif
#ifndef OKC_NS
#define OKC_NS 4
#endif
#ifndef OKC_FNST
#define OKC_FNST 4
#endif
#ifndef OKC_USE_TMA
#define OKC_USE_TMA 1
#endif
#ifndef OKC_TWOACC
#define OKC_TWOACC 1
#endif
#ifndef OKC_NACC
#define OKC_NACC 2
#endif
#ifndef OKC_BEPI
#define OKC_BEPI 0
#endif
#ifndef OKC_FPF
#define OKC_FPF 4
#endif
#ifndef OKC_RLOAD
#define OKC_RLOAD 1  // routing: all topk_ids loads issued before the smem stores
#endif
#ifndef OKC_SKSC2
#define OKC_SKSC2 0  // skinny dequant: one scale build per row pair, HMUL2 half broadcasts (MID: +0.8% at bs16 with
                     // OKC_FTSC2; the TINY builds measured -0.5%, so off by default)
#endif
#ifndef OKC_FTSC2
#define OKC_FTSC2 0  // fat dequant (WGALU scales): HMUL2 half broadcasts instead of byte_perm duplicates
#endif
#ifndef OKC_F1EARLY
#define OKC_F1EARLY 1  // tiny builds: FAST1 warp issues its topk loads before lane 0's barrier inits
#endif
#ifndef OKC_WGSFINSS
#define OKC_WGSFINSS 1  // WGS: the final pass forms each route's sum of squares (the phase-B epilogue only writes y)
#endif
#ifndef OKC_WGSDESC0
#define OKC_WGSDESC0 1  // WGS: x descriptor / code pointer from a per-segment slot-0 base + slot offset
#endif
#ifndef OKC_WGSSC2
#define OKC_WGSSC2 1  // WGS: scale layout [j][h] (template_v34 _wgs_copy) and one scale build per k16 step
#endif
#ifndef OKC_FINCOL
#define OKC_FINCOL 1
#endif
#ifndef OKC_WGMIN
#define OKC_WGMIN 9
#endif
#ifndef OKC_WGALU
#define OKC_WGALU 1
#endif
#ifndef OKC_XPRE
#define OKC_XPRE 1
#endif
#ifndef OKC_PROBE
#define OKC_PROBE 0
#endif
#ifndef OKC_EPI4
#define OKC_EPI4 1
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define FAST1_ARCH 1
#else
#define FAST1_ARCH 0
#endif
#ifndef OKC_FAST1
#define OKC_FAST1 3
#endif
#define FAST1_ON (OKC_FAST1 > 0 && OKC_TINY && FAST1_ARCH && OKC_USE_TMA)
#ifndef OKC_CPFF
#define OKC_CPFF 1
#endif
#ifndef OKC_FCOSTBB
#define OKC_FCOSTBB OKC_FCOSTB
#endif
#ifndef OKC_FCOSTSB
#define OKC_FCOSTSB OKC_FCOST
#endif
#ifndef OKC_DYN
#define OKC_DYN 1
#endif
#ifndef OKC_DYNK
#define OKC_DYNK 4
#endif
#ifndef OKC_TDYN
#define OKC_TDYN 0
#endif
#define SKDYN (OKC_DYN && (!OKC_TINY || OKC_TDYN) && FAST1_ARCH && OKC_USE_TMA && OKC_SKT == 8 && !OKC_XPF && !OKC_NOLOAD)
#ifndef OKC_FT_T
#define OKC_FT_T 64
#endif
#ifndef OKC_SPLIT
#define OKC_SPLIT 0  // skinny-only build with experts of > SK_T tokens split into <= SK_T-token jobs (MID, Oct 26)
#endif
#ifndef OKC_TINY
#define OKC_TINY 0
#endif
#ifndef OKC_NW
#define OKC_NW 8
#endif
#ifndef OKC_EARLYN
#define OKC_EARLYN 1
#endif
#ifndef OKC_BIGM
#define OKC_BIGM 64
#endif
#ifndef OKC_XPF2
#define OKC_XPF2 1
#endif
#ifndef OKC_XPF
#define OKC_XPF 0
#endif
#ifndef OKC_CPF
#define OKC_CPF 1
#endif
#ifndef OKC_FATSTATIC
#define OKC_FATSTATIC 1
#endif
#ifndef OKC_FCOST
#define OKC_FCOST 40
#endif
#ifndef OKC_FCOST2
#define OKC_FCOST2 OKC_FCOST
#endif
#ifndef OKC_FCOSTB
#define OKC_FCOSTB 40
#endif
#ifndef OKC_FCOSTB2
#define OKC_FCOSTB2 OKC_FCOSTB
#endif
#ifndef OKC_WS
#define OKC_WS 0  // tiny build: one extra helper warp issues the ring's TMA copies and runs every skinny epilogue
#endif
#ifndef OKC_WSDYN
#define OKC_WSDYN 0  // WS: dynamic unit claiming: each phase = static equal slices of the first units, then a pool
                     // of the last max(G, nS >> OKC_WSDYNS) units claimed one at a time (one atomic in flight) by the
                     // helper warp, which owns the whole unit sequence (both phases) and publishes it in S.uid
#endif
#ifndef OKC_WSDYNS
#define OKC_WSDYNS 2
#endif
#ifndef OKC_LAUNCHCHK
#define OKC_LAUNCHCHK 0  // dev: check the launch and synchronize after it (reports smem size and the CUDA error)
#endif
#ifndef OKC_WGSX
#define OKC_WGSX 0  // dev: 1 = WGS without the wgmma, 2 = WGS phases skipped
#endif
#ifndef OKC_WGSREL32
#define OKC_WGSREL32 0  // WGS: every consumer lane arrives on the empty slot (else lane 0 of each warp)
#endif
#ifndef OKC_WGS
#define OKC_WGS 0  // tiny build on RS-wgmma: 4 consumer warpgroups split K of 64-row units, plus a TMA producer warp
#endif
#if OKC_WGS && !(OKC_NW == 16 && !OKC_FAST1 && OKC_PRMTSC && !OKC_WS && OKC_FINCOL)
#error "OKC_WGS: 16 warps, PRMT scales, column final pass, no FAST1 / WS"
#endif
#ifndef OKC_FDIV
#define OKC_FDIV 1  // SwiGLU's divide as MUFU.RCP + FMUL (no IEEE div.rn slow-path call in the epilogue)
#endif
#if OKC_FDIV
#define SILU_DIV(a, b) __fdividef((a), (b))
#else
#define SILU_DIV(a, b) ((a) / (b))
#endif
#ifndef OKC_UWARP
#define OKC_UWARP 1  // warp index through a shuffle, so ptxas can keep warp-derived values in uniform registers
#endif
#if OKC_UWARP
#define WARP_IDX(t) __shfl_sync(0xffffffffu, (t) >> 5, 0)
#else
#define WARP_IDX(t) ((t) >> 5)
#endif
#ifndef OKC_DQ7
#define OKC_DQ7 0
#endif
#ifndef OKC_DQIMAD
#define OKC_DQIMAD 1
#endif
#ifndef OKC_PRMTSC
#define OKC_PRMTSC 0
#endif
#ifndef OKC_FINPF
#define OKC_FINPF 0   // skinny builds: final item's topk_w / pos before the barrier, ss + ybuf loads batched (Oct 26, bf16)
#endif
#ifndef OKC_XNOLUT
#define OKC_XNOLUT 0
#endif
#ifndef OKC_XNODQ
#define OKC_XNODQ 0
#endif
#ifndef OKC_XNOMMA
#define OKC_XNOMMA 0
#endif
#ifndef OKC_WSNR
#define OKC_WSNR 4  // partial-sum buffers between the compute warps and the helper warp
#endif
#if OKC_WS && !(OKC_TINY && OKC_FAST1 && OKC_USE_TMA && OKC_FATSTATIC)
#error "OKC_WS: tiny TMA build with FAST1 only"
#endif
#ifndef OKC_NOLOAD
#define OKC_NOLOAD 0
#endif
#ifndef OKC_RLDIST
#define OKC_RLDIST 32  // OKC_ROUTE_LOGITS: above this many tokens the selection is distributed over the CTAs (token t by
                       // CTA t mod G), published in global memory and followed by a grid barrier; up to it every CTA
                       // selects every token itself (the tiny FAST1 builds always do)
#endif
#ifndef OKC_ROUTE_LOGITS
#define OKC_ROUTE_LOGITS 0  // routing in the kernel from the router logits (fp32 [M, 256]) and bias (fp32 [256]):
                            // scores = sigmoid(logit * ROUTER_MULTIPLIER), top-8 of scores + bias (ties: lower id),
                            // weights = chosen scores / (sum + 1e-15), local id = (id - local_start) mod 256 if < 32;
                            // the decode layer's _route_split formula, bit-for-bit (ex2.approx + div.full as Triton)
#endif
#ifndef OKC_NOCOMPUTE
#define OKC_NOCOMPUTE 0
#endif

namespace okc {

#ifndef OKC_L2H
#define OKC_L2H 1
#endif
#if OKC_L2H
// L2 policies (CUTLASS CacheHintSm90): weights stream through once (evict first); inter and ybuf are re-read later in
// the same call (evict last). Hints only: the math and the memory traffic's results are unchanged.
#define L2H_SUFFIX ".L2::cache_hint"
#define L2H_OPERAND ", %4"
#define L2H_ARG , "l"(0x12F0000000000000ull)
__device__ __forceinline__ void st_keep_bf16(__nv_bfloat16* a, __nv_bfloat16 v) {
  asm volatile("st.global.L2::cache_hint.b16 [%0], %1, %2;\n" ::"l"(a), "h"(*reinterpret_cast<unsigned short*>(&v)), "l"(0x14F0000000000000ull) : "memory");
}
__device__ __forceinline__ void st_keep_f32(float* a, float v) {
  asm volatile("st.global.L2::cache_hint.b32 [%0], %1, %2;\n" ::"l"(a), "f"(v), "l"(0x14F0000000000000ull) : "memory");
}
#else
#define L2H_SUFFIX ""
#define L2H_OPERAND ""
#define L2H_ARG
__device__ __forceinline__ void st_keep_bf16(__nv_bfloat16* a, __nv_bfloat16 v) { *a = v; }
__device__ __forceinline__ void st_keep_f32(float* a, float v) { *a = v; }
#endif

constexpr int HD = 4096, ID = 4096;
constexpr int NW = OKC_NW, NT = NW * 32;
constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? 32 : 0);  // + helper warp (WS) | weights and x producer warps (WGS)  // launched: NT compute threads (+ helper warp | WGS producer warp + epilogue warpgroup)
constexpr int NRB = OKC_WS ? OKC_WSNR : 2;   // cross-warp partial-sum buffers
#if OKC_WS || OKC_WGS
// CTA barrier over the NT compute threads only (the helper warp syncs with them through mbarriers and barrier 2)
#define CTA_SYNC() asm volatile("bar.sync 1, %0;\n" ::"n"(NT) : "memory")
#define WS_JOIN() asm volatile("bar.sync 2, %0;\n" ::"n"(NTL) : "memory")
#else
#define CTA_SYNC() __syncthreads()
#endif
constexpr int NCH = 64 / NW;     // 64-K chunks per warp in a skinny unit (4096 / NW K per warp)
#ifndef OKC_SKT
#define OKC_SKT 8
#endif
#if OKC_SKT > 8 && OKC_EPI4
#error "OKC_EPI4 assumes single-tile skinny jobs"
#endif
constexpr int SK_T = OKC_SKT;   // max tokens of a skinny job (8: 1 n-tile; 16: 2 n-tiles via smem x)
constexpr int RED_W = OKC_SKT > 8 ? 8 : 4;
static_assert(!OKC_WS || OKC_SKT == 8, "OKC_WS assumes single-tile skinny jobs");
constexpr int FT_T = OKC_FT_T;   // max tokens of a fat job (longer jobs are split into chunks)
constexpr int RB_A = 2 * ID / 16, RB_B = HD / 16;   // 16-row blocks
constexpr int UB_A = RB_A / 16, UB_B = RB_B / 16;
constexpr int RB_A_LOG = 9, RB_B_LOG = 8;
static_assert((1 << RB_B_LOG) == RB_B, "RB_B must be 256");
constexpr int BFLAG = 1 << 30;  // uslot: a phase-B unit prefetched at the end of phase A
// dynamic tail: the last pool units of a phase's skinny work are claimed at run time (OKC_DYNK per CTA)
__device__ __forceinline__ int dyn_pool(int nS, int G) {
#if OKC_WSDYN
  const int d = max(G, nS >> OKC_WSDYNS);
  return d < nS ? d : nS;
#else
  const int d = G * OKC_DYNK;
  return SKDYN ? (d < nS ? d : nS) : 0;
#endif
}
static_assert((1 << RB_A_LOG) == RB_A, "RB_A must be 512");
static_assert(OKC_NSF <= 7 && OKC_NS <= 5, "barrier slots");   // 256-row blocks (fat units)
constexpr int MMAX = 384, EMAX = MMAX * 8, JMAX = 32 + EMAX / FT_T + 8;
constexpr int XP_SK = HD + 8;   // skinny x row pitch (bf16)
constexpr int KS = 256;         // fat: K per x stage (4 chunks)
constexpr int XP_FT = KS + 8;   // fat x row pitch (bf16)
constexpr int NST = 3;          // fat x stages
constexpr float TWO62 = 4.611686018427388e18f;

struct Params {
  const uint4* wA; const uint4* sA; const float* gsA;
  const uint4* wB; const uint4* sB; const float* gsB;
  const uint32_t* lut; const __nv_bfloat16* normw;  // post_norm_w as given (bf16); 1 + w is formed in FP32 here
  const unsigned char* wAf; const unsigned char* wBf;  // fat-order weight copies (36 KB per k-stage)
  const unsigned char* wAu; const unsigned char* wBu;  // unit-contiguous copies (36 KB per 16-row block)
  const unsigned char* wAn; const unsigned char* wBn;  // natural-k fat copies (wgmma path)
  const unsigned char* wAs; const unsigned char* wBs;  // WGS copies: [256-row unit][stage][warpgroup][9 KB]
  __nv_bfloat16* xA; __nv_bfloat16* xB;  // WGS: per job, per stage x / inter in wgmma B layout [stage][n8][k8][8][8]
  int* xready;                           // WGS: per job, rows of xA written
  int* doneAs;                           // WGS: per (job, phase-B stage), phase-A warpgroup epilogues done (8 = ready)
  uint32_t zero;  // always 0 (opaque to the compiler)
  const __nv_bfloat16* x; const int* topk_ids; const float* topk_w;
  const float* logits;   // OKC_ROUTE_LOGITS: router logits [M, 256] fp32
  const float* rbias;    // OKC_ROUTE_LOGITS: router bias [256] fp32
  const int* lsp;        // OKC_ROUTE_LOGITS: local_start (device int32), or null -> lo
  int* gid_out;          // OKC_ROUTE_LOGITS: chosen global expert ids [M, 8] (written by CTA 0)
  const float* rsp;      // OKC_ROUTE_LOGITS: pre-expert RMS partials [M, 128] fp32 (sum of squares of the pooled row
                         // over 128 slices) or null; rs_t = rsqrt(sum / 4096 + 1e-5) scales GEMM1's output per token
  int M, lo, nloc;
  int* ctr;        // [8]: 0 ctrA, 1 ctrB, 2 barrier, 3 exit
  int* doneA;      // [JMAX]
  __nv_bfloat16* inter;  // [EMAX, ID]
  float* ybuf;     // [EMAX, HD]
  float* ss;       // [EMAX] copy 0 of the per-token sums of squares (copy 1: ss1); a launch uses copy gen & 1
  float* ss1;      // [EMAX]
  __nv_bfloat16* out;
  unsigned long long* dbg;
};

struct Smem {
  uint32_t lut[256];
#if OKC_ROUTE_LOGITS && !OKC_PRMTSC
  float rs_[256];   // per-token GEMM1 scale (PRMT builds alias the unused LUT)
#endif
#if !OKC_WGS
  float red[NRB][NW][32][RED_W];  // per-unit cross-warp partial sums
#else
  float red[1][1][1][RED_W];      // (the skinny paths that use it are not run in WGS builds)
#endif
#if OKC_WS
  uint64_t wse[8], wsrf[8], wsre[8];  // WS: ring slot released by all compute warps / partials ready / buffer free
  int wsr[4];                          // WS: this CTA's phase A and phase B unit ranges
  int wsd[4];                          // WSDYN: pool bounds [nAs, nA) and [nBs, nB)
  int uid[8];                          // WSDYN: unit of sequence index q at q & 7: (q << 16) | (PB << 15) | v
#endif
#if OKC_WGS
  uint64_t bars[24];  // WGS: 0..3 weights full, 4..7 weights empty, 8..15 x full, 16..23 x empty
  int wglast[4];      // WGS: this warpgroup completed a split unit
  int wcp[JMAX + 1];  // WGS: phase-A cost prefix per job
  int wcpB[JMAX + 1]; // WGS: phase-B cost prefix per job
  int jrow[JMAX + 1]; // WGS: first padded row of each job in xA / xB (rows padded to 8 per job)
  uint32_t wtot;      // WGS: phase-A total cost
  uint32_t wtotB;     // WGS: phase-B total cost
#else
  uint64_t bars[12];  // 0..4 skinny ring (NS <= 5), 5..11 fat ring (NSF <= 7)
#endif
#if OKC_WGS
  int wgr[4];         // WGS: this CTA's phase A and phase B unit ranges
  int wgd[4];         // WGSDYN: pool stage ranges [Ts, T) per phase
  int wseg[16];       // WGSDYN: published pieces (tag << 20 | u << 8 | s0 << 4 | s1 - 1), index k & 15
#endif
  int seq, fseq, preissued, preissuedB;
  int fcnt[32];     // OKC_SPLIT: FAST1 per-expert entry counts (the preissue's job -> expert map)
  void* xbuf_ptr;
  int cnt[64], off[64];
  int wc[NW][32];
  int jE[JMAX], jS[JMAX], jC[JMAX], forder[JMAX];
  int njF, njS, unit;
  int gen;          // the grid barrier's generation at launch start (+1 per launch): picks this launch's ss copy
  float* ssp;       // this launch's ss copy
  int uslot[8];    // unit id per skinny ring slot (-1: none; BFLAG | v: phase-B unit prefetched in phase A)
  int dync[JMAX];  // phase A: pool units this CTA processed, per job (credited to doneA at the end of phase A)
  int unit_next, unit_next2;
  signed char eid[EMAX];
  short ent[EMAX];   // sorted entries (token*8+slot) by expert
  short pos[EMAX];   // entry -> pair position, -1 if not local
};

#if OKC_ROUTE_LOGITS
__device__ __forceinline__ float* rs_ptr(Smem& S) {
#if OKC_PRMTSC
  return reinterpret_cast<float*>(S.lut);
#else
  return S.rs_;
#endif
}
#define RS_T(tok) (rs_ptr(const_cast<Smem&>(S))[(tok)])
#else
#define RS_T(tok) 1.f
#endif
// floor(a * c / G) without an integer divide (per-launch code runs from a cold instruction cache, where the divide
// subroutines cost ~1 us): a * c < 2^24, rG ~ 1 / G, then an exact +-1 fix-up
__device__ __forceinline__ int muldiv_g(int a, int c, int G, float rG) {
  const int p = a * c;
  int q = __float2int_rz((float)p * rG);
  if (q * G > p) --q;
  else if ((q + 1) * G <= p) ++q;
  return q;
}
__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}
// a x (b.lo, b.lo) and a x (b.hi, b.hi): the broadcast folds into the HMUL2 operand (.H0_H0 / .H1_H1)
__device__ __forceinline__ uint32_t bmul2l(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), __low2bfloat162(*reinterpret_cast<__nv_bfloat162*>(&b)));
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ uint32_t bmul2h(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), __high2bfloat162(*reinterpret_cast<__nv_bfloat162*>(&b)));
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ uint32_t dq_hi(uint32_t q) { return (q & 0x80008000u) | ((q & 0x70007000u) >> 6); }

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint4 ldg_stream(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
  uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
  int sz = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(sz));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
#if OKC_ROUTE_LOGITS
#define TOPKW(i) __ldcg(p.topk_w + (i))  // written by CTA 0 in this launch (visible after the grid barrier)
#else
#define TOPKW(i) p.topk_w[i]
#endif
#define DBG(k) do { if (OKC_DEBUG && threadIdx.x == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)
extern __shared__ __align__(16) unsigned char smem_raw[];
__device__ __forceinline__ float* ss_cur();  // this launch's ss copy (defined after Smem users)
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p));
  return v;
}

// Dequantize one 32-bit word (8 E2M1 codes) into the 4 bf16x2 A-fragment registers (values e2m1 * 2^-126).
// Prepacked bit layout (see _pack_codes): r0 = q & M, r1 = rotr(q,6) & M, r2 = rotr(q,10) & M,
// r3 = (rotl(q,4) & 0x81808180) | (rotr(q,4) & 0x00400040), M = 0x81C081C0.
__device__ __forceinline__ void dq4(uint32_t q, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  constexpr uint32_t M = 0x81C081C0u;
  r0 = q & M;
  r1 = __funnelshift_r(q, q, 6) & M;
  r2 = __funnelshift_r(q, q, 10) & M;
#if OKC_DQ7
  // DQ7 bit layout (template DQ7 = True): 7 ALU + 2 IMAD.SHL per 8 weights. Register sources partition the word:
  // r0 bits as stored, r1 = word << 4, r2 = rotl 10, r3 = word << 6 (6 bits) | rotl 14 (the two e1 bits)
  r1 = (q * 16u) & M;
  r2 = __funnelshift_l(q, q, 10) & M;
  r3 = ((q * 64u) & 0x80C080C0u) | (__funnelshift_l(q, q, 14) & 0x01000100u);
  return;
#endif
#if OKC_DQIMAD
  // bits kept by 0x81808180 never wrap, so the left rotate is a plain shift: as a multiply it runs on the FMA pipe
  // (IMAD.SHL) instead of the ALU pipe the rest of the decode saturates
  r3 = ((q * 16u) & 0x81808180u) | (__funnelshift_r(q, q, 4) & 0x00400040u);
#else
  r3 = (__funnelshift_l(q, q, 4) & 0x81808180u) | (__funnelshift_r(q, q, 4) & 0x00400040u);
#endif
}

// dequantize one 16-row x 64-K chunk word set and multiply-accumulate against b regs
__device__ __forceinline__ void chunk_mma(uint4 w, uint32_t sg, uint32_t sg8, const uint32_t* b, float* acc) {
  const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
  for (int st = 0; st < 4; ++st) {
    uint32_t r0, r1, r2, r3;
    dq4(ww[st], r0, r1, r2, r3);
    mma_bf16(acc, bmul2(r0, sg), bmul2(r1, sg8), bmul2(r2, sg), bmul2(r3, sg8), b[2 * st], b[2 * st + 1]);
  }
}

#if OKC_ROUTE_LOGITS
// ---------------------------------------------------------------- in-kernel top-8 routing from the router logits
// Triton's tl.sigmoid(x) = 1 / (1 + exp(-x)) with exp lowered to ex2.approx.f32(x * log2e) and / to div.full.f32:
// the same instructions here, so the scores (and the selection) match the layer's _route_split kernel bit for bit.
__device__ __forceinline__ float ex2_approx(float x) { float y; asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float div_full(float a, float b) { float y; asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y; }
__device__ __forceinline__ float sigmoid_tl(float x) { return div_full(1.f, 1.f + ex2_approx((-x) * 1.4426950408889634f)); }
// one warp selects token t's experts: lane l holds experts 8 l .. 8 l + 7; 8 rounds of (lane-local strict max, lowest k
// on ties; butterfly max over the lanes keeping value desc, index asc; the winner masked to -inf). Writes S.eid (local
// ids) for this CTA; CTA 0 also writes the global ids and the renormalized weights (read by the final pass of every
// CTA after the grid barrier).
__device__ __forceinline__ void route_select(const Params& p, Smem& S, int t, int lo, const float4 (&lq)[2], const float (&bv)[8],
                                             bool store_eid, bool store_out) {
  const int lane = threadIdx.x & 31;
  const float lv[8] = {lq[0].x, lq[0].y, lq[0].z, lq[0].w, lq[1].x, lq[1].y, lq[1].z, lq[1].w};
  float sc[8], sel[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) { sc[k] = sigmoid_tl(lv[k] * 0.1767766952966369f); sel[k] = sc[k] + bv[k]; }
  int chosen[8];
  float w[8];
#pragma unroll
  for (int r = 0; r < 8; ++r) {
    float m = sel[0];
    int mi = 0;
#pragma unroll
    for (int k = 1; k < 8; ++k) if (sel[k] > m) { m = sel[k]; mi = k; }
    int gi = lane * 8 + mi;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const float om = __shfl_xor_sync(0xffffffffu, m, off);
      const int oi = __shfl_xor_sync(0xffffffffu, gi, off);
      if (om > m || (om == m && oi < gi)) { m = om; gi = oi; }
    }
    // gi is warp-uniform: the winner's score from its lane, then masked
    const int kw = gi & 7;
    float v = sc[0];
#pragma unroll
    for (int k = 1; k < 8; ++k) v = (k == kw) ? sc[k] : v;
    w[r] = __shfl_sync(0xffffffffu, v, gi >> 3);
    chosen[r] = gi;
    if (lane == (gi >> 3)) {
#pragma unroll
      for (int k = 0; k < 8; ++k) sel[k] = (k == kw) ? -INFINITY : sel[k];
    }
  }
  const float sum = ((w[0] + w[1]) + (w[2] + w[3])) + ((w[4] + w[5]) + (w[6] + w[7]));
  const float den = sum + 1e-15f;
  if (lane < 8) {
    int g = chosen[0];
    float ww = w[0];
#pragma unroll
    for (int k = 1; k < 8; ++k) { g = (lane == k) ? chosen[k] : g; ww = (lane == k) ? w[k] : ww; }
    const int el = (g - lo + 256) & 255;  // the layer's local id: (id - local_start) mod 256, local if < 32
    if (store_eid) S.eid[t * 8 + lane] = (signed char)(el < p.nloc ? el : -1);
    if (store_out) {
      p.gid_out[t * 8 + lane] = g;
      const_cast<float*>(p.topk_w)[t * 8 + lane] = div_full(ww, den);
    }
  }
}
#endif

// ---------------------------------------------------------------- routing prologue (every CTA, identical result)
__device__ void early_issue(const Params& p, Smem& S);
__device__ void route(const Params& p, Smem& S) {
  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const int E = p.M * 8;
  const int CH = ((E + NW * 32 - 1) / (NW * 32)) * 32;
  const int i0 = warp * CH, i1 = min(E, i0 + CH);
  if (tid < NW * 32) S.wc[tid >> 5][tid & 31] = 0;
#if OKC_ROUTE_LOGITS
  if (!OKC_TINY && p.M > OKC_RLDIST) {  // distributed selection: the global ids from every CTA (else S.eid is filled)
    const int lo = p.lsp ? __ldg(p.lsp) : p.lo;
#pragma unroll 4
    for (int i = tid; i < E; i += NT) {
      const int el = (__ldcg(p.gid_out + i) - lo + 256) & 255;
      S.eid[i] = (signed char)(el < p.nloc ? el : -1);
    }
  }
#elif OKC_RLOAD && OKC_WGS  // (the small builds lose 1-2% with it: their E is at most 256)
  {  // 16 B loads (E = 8 M; the launcher checks the alignment): one or two per thread, all issued before the stores
    constexpr int LPT = (EMAX / 4 + NT - 1) / NT;
    int4 ev[LPT];
    const int E4 = E >> 2;
#pragma unroll
    for (int k = 0; k < LPT; ++k) {
      const int i4 = tid + k * NT;
      ev[k] = i4 < E4 ? __ldg(reinterpret_cast<const int4*>(p.topk_ids) + i4) : make_int4(0, 0, 0, 0);
    }
#pragma unroll
    for (int k = 0; k < LPT; ++k) {
      const int i4 = tid + k * NT;
      if (i4 < E4) {
        const int v[4] = {ev[k].x - p.lo, ev[k].y - p.lo, ev[k].z - p.lo, ev[k].w - p.lo};
        char4 c;
        c.x = (signed char)((v[0] >= 0 && v[0] < p.nloc) ? v[0] : -1);
        c.y = (signed char)((v[1] >= 0 && v[1] < p.nloc) ? v[1] : -1);
        c.z = (signed char)((v[2] >= 0 && v[2] < p.nloc) ? v[2] : -1);
        c.w = (signed char)((v[3] >= 0 && v[3] < p.nloc) ? v[3] : -1);
        *reinterpret_cast<char4*>(&S.eid[i4 * 4]) = c;
      }
    }
  }
#else
#pragma unroll 8
  for (int i = tid; i < E; i += NT) {
    int e = __ldg(p.topk_ids + i) - p.lo;
    S.eid[i] = (signed char)((e >= 0 && e < p.nloc) ? e : -1);
  }
#endif
  CTA_SYNC();
  for (int base = i0; base < i1; base += 32) {
    const int i = base + lane;
    const int e = (i < i1) ? (int)S.eid[i] : -1;
    const unsigned peers = __match_any_sync(0xffffffffu, e);
    if (e >= 0 && lane == __ffs(peers) - 1) S.wc[warp][e] += __popc(peers);
  }
  CTA_SYNC();
  if (warp == 0) {
    int c[NW];
    int tot = 0;
#pragma unroll
    for (int w = 0; w < NW; ++w) { c[w] = (lane < p.nloc) ? S.wc[w][lane] : 0; tot += c[w]; }
    int incl = tot;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { int v = __shfl_up_sync(0xffffffffu, incl, d); if (lane >= d) incl += v; }
    int off = incl - tot;
    S.cnt[lane] = tot;
    S.off[lane] = off;
    int o2 = off;
#pragma unroll
    for (int w = 0; w < NW; ++w) { S.wc[w][lane] = o2; o2 += c[w]; }
    // jobs: fat (cnt > SK_T, chunks of FT_T) first, then skinny (0 < cnt <= SK_T); expert order within each
#if OKC_SPLIT
    static_assert(OKC_TINY, "OKC_SPLIT: skinny-only (tiny-style) build");
    const int nfj = 0;
    const int nsj = (tot + SK_T - 1) / SK_T;   // ceil(tot / 8) jobs of <= 8 tokens; a split expert re-streams its weights
#else
    const int nfj = tot > SK_T ? (tot + FT_T - 1) / FT_T : 0;
    const int nsj = (tot > 0 && tot <= SK_T) ? 1 : 0;
#endif
    int fi = nfj, si = nsj;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      int v1 = __shfl_up_sync(0xffffffffu, fi, d), v2 = __shfl_up_sync(0xffffffffu, si, d);
      if (lane >= d) { fi += v1; si += v2; }
    }
    const int nfTot = __shfl_sync(0xffffffffu, fi, 31), nsTot = __shfl_sync(0xffffffffu, si, 31);
    int fj = fi - nfj, sj = nfTot + si - nsj;
    for (int q = 0; q < nfj; ++q) {
      S.jE[fj + q] = lane; S.jS[fj + q] = off + q * FT_T; S.jC[fj + q] = min(FT_T, tot - q * FT_T);
    }
#if OKC_SPLIT
    for (int q = 0; q < nsj; ++q) { S.jE[sj + q] = lane; S.jS[sj + q] = off + q * SK_T; S.jC[sj + q] = min(SK_T, tot - q * SK_T); }
#else
    if (nsj) { S.jE[sj] = lane; S.jS[sj] = off; S.jC[sj] = tot; }
#endif
    if (lane == 0) { S.njF = nfTot; S.njS = nsTot; }
    __syncwarp();
#if !OKC_WGS  // (the WGS build has no fat queue)
    // fat queue order: largest n-tile count first (LPT), stable by job index
    int pos = 0;
    for (int key = FT_T / 8; key >= 1; --key) {
      for (int f0 = 0; f0 < nfTot; f0 += 32) {
        const int f = f0 + lane;
        const bool m = f < nfTot && ((S.jC[f] + 7) >> 3) == key;
        const unsigned bal = __ballot_sync(0xffffffffu, m);
        if (m) S.forder[pos + __popc(bal & ((1u << lane) - 1u))] = f;
        pos += __popc(bal);
      }
    }
#endif
  }
  CTA_SYNC();
  early_issue(p, S);
  for (int base = i0; base < i1; base += 32) {
    const int i = base + lane;
    const int e = (i < i1) ? (int)S.eid[i] : -1;
    const unsigned peers = __match_any_sync(0xffffffffu, e);
    if (e >= 0) {
      const int q = S.wc[warp][e] + __popc(peers & ((1u << lane) - 1u));
      S.ent[q] = (short)i;
      S.pos[i] = (short)q;
    } else if (i < i1) {
      S.pos[i] = -1;
    }
    __syncwarp();
    if (e >= 0 && lane == __ffs(peers) - 1) S.wc[warp][e] += __popc(peers);
    __syncwarp();
  }
  CTA_SYNC();
}

// ---------------------------------------------------------------- skinny unit: CTA = 16 rows x full K, warps split K
template <bool PB>
__device__ __forceinline__ void sk_load_w(const Params& p, int e, int rb, int lane, int warp, uint4 (&w)[NCH], uint4& s) {
  const int rbs = PB ? RB_B : RB_A;
  const uint4* wb = (PB ? p.wB : p.wA) + ((size_t)(e * rbs + rb) * NW + warp) * (NCH * 32) + lane;
  const uint4* sb = (PB ? p.sB : p.sA) + ((size_t)(e * rbs + rb) * NW + warp) * 32 + lane;
#pragma unroll
  for (int c = 0; c < NCH; ++c) w[c] = ldg_stream(wb + c * 32);
  s = ldg_stream(sb);
}

// stage x rows of a skinny job into xs [nrows][XP_SK] (rows >= cnt zero-filled)
template <bool PB>
__device__ __forceinline__ void sk_stage_x(const Params& p, const Smem& S, int j, __nv_bfloat16* xs, int nrows) {
  const int tid = threadIdx.x;
  const int cnt = S.jC[j], st = S.jS[j];
  const int tot = nrows * (HD / 8);
#pragma unroll 4
  for (int i = tid; i < tot; i += NT) {
    int r = i >> 9, c = i & 511;
    bool v = r < cnt;
    const __nv_bfloat16* src;
    if (PB) src = p.inter + (size_t)(st + (v ? r : 0)) * ID + c * 8;
    else src = p.x + (size_t)(v ? (S.ent[st + r] >> 3) : 0) * HD + c * 8;
    cp_async16(xs + r * XP_SK + c * 8, src, v);
  }
  cp_commit();
}

// ---------------------------------------------------------------- epilogues (one 16-row block, tokens 2t,2t+1 of n-tile)
// r[0..1] rows g (gate / out row n0), r[2..3] rows g+8 (up / out row n1), tokens tk0 = 2t, 2t+1 relative to tile
__device__ __forceinline__ float probe_q(float v) {
#if OKC_PROBE
  // sensitivity probe for the precision check: keep only E4M3's 3 mantissa bits (round to nearest even)
  uint32_t b = __float_as_uint(v);
  b += 0x7FFFFu + ((b >> 20) & 1u);
  return __uint_as_float(b & 0xFFF00000u);
#else
  return v;
#endif
}
__device__ __forceinline__ void epi_A(const Params& p, int pstart, int ntok, int tok0, int col, float gsc, const float* r) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    int tk = tok0 + h;
    if (tk < ntok) {
      float gt = r[h] * gsc, up = r[2 + h] * gsc;
      float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;
      st_keep_bf16(p.inter + ((size_t)(pstart + tk) * ID + col), __float2bfloat16(probe_q(v)));
    }
  }
}
// returns per-token partial sum of squares (2 values) for this lane's two tokens
__device__ __forceinline__ void epi_B(const Params& p, int pstart, int ntok, int tok0, int n0, float gsc, const float* r,
                                      float* sq) {
  const float w0 = 1.f + __bfloat162float(p.normw[n0]), w1 = 1.f + __bfloat162float(p.normw[n0 + 8]);
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    int tk = tok0 + h;
    float v0 = r[h] * gsc * w0, v1 = r[2 + h] * gsc * w1;
    sq[h] = v0 * v0 + v1 * v1;
    if (tk < ntok) {
      st_keep_f32(p.ybuf + ((size_t)(pstart + tk) * HD + n0), v0);
      st_keep_f32(p.ybuf + ((size_t)(pstart + tk) * HD + n0 + 8), v1);
    }
  }
}
__device__ __forceinline__ void ss_reduce_add(const Params& p, int pstart, int ntok, int tok0, float* sq, int g) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    sq[h] += __shfl_xor_sync(0xffffffffu, sq[h], 4);
    sq[h] += __shfl_xor_sync(0xffffffffu, sq[h], 8);
    sq[h] += __shfl_xor_sync(0xffffffffu, sq[h], 16);
  }
  if (g == 0) {
#pragma unroll
    for (int h = 0; h < 2; ++h)
      if (tok0 + h < ntok) atomicAdd(&ss_cur()[pstart + tok0 + h], sq[h]);
  }
}

// ---------------------------------------------------------------- the persistent kernel
// cost of one unit in quarter skinny-units: skinny = 4; fat (128 rows, nt n-tiles) ~ max(8, 5.5 + 2 nt) skinny units
__device__ __forceinline__ int fat_cost(int cnt) {
  int nt = (cnt + 7) >> 3;
  int c = 22 + 8 * nt;
  return c < 32 ? 32 : c;
}

struct UnitIter {
  int j, nj, njF, ubs, rbs, ul, uh;
  long long Jc, lo, hi;
  __device__ int cost(const Smem& S, int jj) const { return jj < njF ? fat_cost(S.jC[jj]) : 4; }
  __device__ int count(int jj) const { return jj < njF ? ubs : rbs; }
  __device__ bool seek(const Smem& S) {
    while (j < nj && Jc < hi) {
      int c = cost(S, j), n = count(j);
      long long a = lo - Jc, b = hi - Jc;
      int l = a <= 0 ? 0 : (int)min((long long)n, (a + c - 1) / c);
      int h = b <= 0 ? 0 : (int)min((long long)n, (b + c - 1) / c);
      if (l < h) { ul = l; uh = h; return true; }
      Jc += (long long)c * n;
      ++j;
    }
    j = nj;
    return false;
  }
  // advance to the unit after (j, u); returns false at the end
  __device__ bool next(const Smem& S, int u, int& jn, int& un) {
    if (u + 1 < uh) { jn = j; un = u + 1; return true; }
    Jc += (long long)cost(S, j) * count(j);
    ++j;
    if (seek(S)) { jn = j; un = ul; return true; }
    return false;
  }
};

// ---------------------------------------------------------------- skinny path: shared-memory weight ring
// A unit = (job <= 16 tokens, 16-row block) x full K = 32 KB codes + 4 KB scales, contiguous in global memory.
// Units of the CTA's range stream through OKC_NS smem slots (TMA bulk copies on sm_90, cp.async elsewhere).
// All 8 warps work on the same unit, each on its own 512-K range, with x for that K range held in registers.
constexpr int UNIT_CODES = 32768, UNIT_SC = 4096, UNIT_BYTES = UNIT_CODES + UNIT_SC;


__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
#if OKC_PRMTSC
// e4m3 scale byte b of `word` -> bf16x2(scale * 2^64) in both halves with one PRMT and one IMAD: (byte << 4) puts the
// e4m3 exponent and mantissa into the bf16 fields, + 184 << 7 rebiases the exponent. Exact for positive normal bytes
// 0x08..0x7E only; prepare() checks every scale byte and uses the LUT build otherwise.
#define SCD(word, b) (__byte_perm((word), 0u, 0x4040u | ((b) * 0x0101u)) * 16u + 0x5C005C00u)
// same, with the multiplier opaque to ptxas (k16 = 16 + p.zero): an IMAD on the FMA pipe instead of a LEA on the
// ALU pipe, which the FP4 decode saturates
#define SCDK(word, b, k16) (__byte_perm((word), 0u, 0x4040u | ((b) * 0x0101u)) * (k16) + 0x5C005C00u)
#endif
__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive1(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}\n"
      ::"r"(smem_u32(bar)), "r"(parity) : "memory");
#else
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.test_wait.parity.shared.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}\n"
      ::"r"(smem_u32(bar)), "r"(parity) : "memory");
#endif
}

template <bool PB>
__device__ __forceinline__ void ring_issue(const Params& p, const Smem& S, int v, unsigned char* slot, uint64_t* bar) {
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;
  const size_t blk = (size_t)S.jE[j] * rbs + rb;
  const unsigned char* src = (PB ? p.wBu : p.wAu) + blk * (size_t)UNIT_BYTES;  // [32 KB codes | 4 KB scales]
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && OKC_USE_TMA
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
               ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");
#else
  for (int i = threadIdx.x; i < UNIT_BYTES / 16; i += NT)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(slot + i * 16)), "l"(src + i * 16));
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(smem_u32(bar)) : "memory");
#endif
}

template <bool PB>
__device__ __forceinline__ void ring_issue_all(const Params& p, const Smem& S, int v, unsigned char* slot, uint64_t* bar) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && OKC_USE_TMA
  if (threadIdx.x == 0) ring_issue<PB>(p, S, v, slot, bar);
#else
  ring_issue<PB>(p, S, v, slot, bar);
#endif
}

template <bool PB>
__device__ __forceinline__ void load_x_regs(const Params& p, const Smem& S, int j, int lane, int warp,
                                            uint4 (&x)[2 * NCH]) {
  const int g = lane >> 2, t = lane & 3;
  const int cnt = S.jC[j], st = S.jS[j];
  {
    const int r = g;
    if (r < cnt) {
      const __nv_bfloat16* row;
      if (PB) row = p.inter + (size_t)(st + r) * ID;
      else row = p.x + (size_t)(S.ent[st + r] >> 3) * HD;
      const uint4* src = reinterpret_cast<const uint4*>(row + warp * (64 * NCH) + 16 * t);
#pragma unroll
      for (int c = 0; c < NCH; ++c) {
        x[2 * c] = PB ? __ldcg(src + c * 8) : __ldg(src + c * 8);
        x[2 * c + 1] = PB ? __ldcg(src + c * 8 + 1) : __ldg(src + c * 8 + 1);
      }
    } else {
#pragma unroll
      for (int c = 0; c < 2 * NCH; ++c) x[c] = make_uint4(0, 0, 0, 0);
    }
  }
}

constexpr int X1P = 512 + 8;  // per-warp smem x rows for the 2nd n-tile (tokens 8..15), pitch in bf16
template <bool PB>
__device__ __forceinline__ void load_x1_smem(const Params& p, const Smem& S, int j, int lane, int warp,
                                             __nv_bfloat16* x1s) {
  // warp-private: rows 8..15 of the job, K range [512*warp, +512): 8 rows x 64 uint4
  const int cnt = S.jC[j], st = S.jS[j];
  for (int i = lane; i < 8 * 64; i += 32) {
    const int r = i >> 6, c = i & 63;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (8 + r < cnt) {
      const __nv_bfloat16* row = PB ? p.inter + (size_t)(st + 8 + r) * ID : p.x + (size_t)(S.ent[st + 8 + r] >> 3) * HD;
      const uint4* src = reinterpret_cast<const uint4*>(row + warp * 512) + c;
      v = PB ? __ldcg(src) : __ldg(src);
    }
    *reinterpret_cast<uint4*>(x1s + r * X1P + c * 8) = v;
  }
  __syncwarp();
}

#if SKDYN
template <bool PB>
__device__ void skinny_ring(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq,
                            __nv_bfloat16* x1base, bool cpf_ok, int cB0, int cBn, int nSall, int poolBase) {
  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = 16u + p.zero;  // opaque 16 (scale decode IMAD)
  (void)k16;
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int nU = v1 - v0;  // static units [v0, v1); the run-time pool [poolBase, nSall) follows
  const int rsh = PB ? RB_B_LOG : RB_A_LOG;
  const int seq0 = seq;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  // order earlier generic-proxy accesses of this smem region (fat path) before async-proxy (TMA) writes
  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif
  // cross-phase prefill: with no fat jobs, phase A's last ring refills already fetch this CTA's first B units
  int preB = 0;
  const bool cpf = OKC_CPF && !PB && cpf_ok;
  const int v0B = cB0, nB = cBn;
  const int pre = PB ? S.preissuedB : S.preissued;
  // thread 0 owns the unit source: static units first, then pool units claimed with one atomic in flight
  int snext = v0 + pre, pend = -2, jdmin = 1 << 30, jdmax = -1;
  bool exh = poolBase >= nSall;
  int* pctr = p.ctr + (PB ? 1 : 0);
  auto take = [&]() -> int {
    if (snext < v1) return snext++;
    if (exh) return -1;
    const int cl = pend >= 0 ? pend : atomicAdd(pctr, 1);
    const int v = poolBase + cl;
    if (v >= nSall) { exh = true; pend = -2; return -1; }
    pend = atomicAdd(pctr, 1);  // the next claim; its value is first needed at the next refill
    if (!PB) {
      const int jj = njF + (v >> rsh);
      S.dync[jj] += 1;
      jdmin = min(jdmin, jj); jdmax = max(jdmax, jj);
    }
    return v;
  };
  auto refill = [&](int sl) {
    const int v = take();
    if (v >= 0) { ring_issue<PB>(p, S, v, ring + sl * UNIT_BYTES, bars + sl); S.uslot[sl] = v; }
    else if (cpf && preB < nB) {
      ring_issue<true>(p, S, v0B + preB, ring + sl * UNIT_BYTES, bars + sl);
      S.uslot[sl] = BFLAG | (v0B + preB);
      ++preB;
    } else {
      S.uslot[sl] = -1;
    }
  };
  if (tid == 0)
    for (int i = pre; i < OKC_NS; ++i) {
      const int sl = (seq0 + i) % OKC_NS;
      if (i < nU) { ring_issue<PB>(p, S, v0 + i, ring + sl * UNIT_BYTES, bars + sl); ++snext; }
      else refill(sl);
    }
  if (nU < OKC_NS) CTA_SYNC();  // pool slots filled by the prologue: their unit ids must be visible
  uint4 x0[2 * NCH];
  int curj = -1;
  bool two = false;
  __nv_bfloat16* x1s = x1base + warp * 8 * X1P;
  // phase B epilogue: prefetch (1 + post_norm_w) of the next unit's rows one unit ahead
  float nwc0 = 0.f, nwc1 = 0.f;
#if OKC_EPI4
  const int egg = ((tid >> 5) & 1) * 4 + ((tid & 15) >> 2);  // epilogue row group of this thread (4-warp layout)
#else
  const int egg = (tid & 31) >> 2;
#endif
  if (PB && tid < 128) {
    const int u0 = nU > 0 ? v0 : (S.uslot[seq0 % OKC_NS] & ~BFLAG);
    const int rbq = (u0 < 0 ? 0 : u0) & (rbs - 1);
    nwc0 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg)); nwc1 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8));
  }
  int nproc = 0;
  for (int i = 0;; ++i) {
    int v;
    if (i < nU) {
      v = v0 + i;  // static units: no slot lookup
    } else {
      v = S.uslot[(seq0 + i) % OKC_NS];
      if (v < 0 || (!PB && (v & BFLAG))) break;  // pool exhausted (or phase B's prefetched units follow)
      v &= ~BFLAG;
    }
    const int j = njF + (v >> rsh), rb = v & (rbs - 1);
    float nwn0 = 0.f, nwn1 = 0.f;
    int vn = -1;
    if (PB) vn = i + 1 < nU ? v + 1 : S.uslot[(seq0 + i + 1) % OKC_NS];
    if (PB && tid < 128 && vn >= 0) {
      const int rbq = (vn & ~BFLAG) & (rbs - 1);
      nwn0 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg)); nwn1 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8));
    }
#if OKC_XPF
    if (!PB) {  // L2 prefetch of the x rows of the job starting OKC_NS units ahead
      const int va = v + OKC_NS;
      if (i + OKC_NS < nU && va / rbs != v / rbs) {
        const int ja = njF + va / rbs;
        const int ga = lane >> 2, ta = lane & 3;
        if (ga < S.jC[ja]) {
          // the warp's 1 KB slice of this token row = 8 lines; lanes t take lines 2t, 2t+1
          const char* row = reinterpret_cast<const char*>(p.x + (size_t)(S.ent[S.jS[ja] + ga] >> 3) * HD + warp * 512) + 256 * ta;
          asm volatile("prefetch.global.L2 [%0];\n" ::"l"(row));
          asm volatile("prefetch.global.L2 [%0];\n" ::"l"(row + 128));
        }
      }
    }
#endif
    if (j != curj) {
      curj = j;
      two = (OKC_SKT > 8) && S.jC[j] > 8;
      if (PB) {
        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
        __syncwarp();
      }
      load_x_regs<PB>(p, S, j, lane, warp, x0);
      if (two) load_x1_smem<PB>(p, S, j, lane, warp, x1s);
    }
    const int sl = (seq0 + i) % OKC_NS;
    const uint32_t par = (uint32_t)(((seq0 + i) / OKC_NS) & 1);
#if !OKC_NOLOAD
    mbar_wait(bars + sl, par);
#else
    if (i < OKC_NS) mbar_wait(bars + sl, par);
#endif
    const unsigned char* slot = ring + sl * UNIT_BYTES;
#if OKC_NW == 16
    // a warp covers half of an 8-warp K slice: its 4 chunks' scale bytes are 8 of the slice lane's 16
    const uint2 sc2 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (warp >> 1) * 512 + lane * 16 + (warp & 1) * 8);
    const uint4 sc = make_uint4(sc2.x, sc2.y, 0u, 0u);
#else
    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);
#endif
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
#if OKC_NACC == 4
    float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};
#endif
    const __nv_bfloat16* x1r = x1s + (lane >> 2) * X1P + 16 * (lane & 3);
#if !OKC_NOCOMPUTE
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);
#if OKC_PRMTSC
      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);
#if OKC_SKSC2
      // rows g / g+8 of this chunk are adjacent scale bytes: one PRMT + one IMAD build both, the multiplies take the
      // halves (HMUL2 .H0_H0 / .H1_H1)
      const uint32_t sgp = __byte_perm(swd, 0u, (c & 1) ? 0x4342u : 0x4140u) * k16 + 0x5C005C00u;
#define MULG(r) bmul2l((r), sgp)
#define MULG8(r) bmul2h((r), sgp)
#else
      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);
#define MULG(r) bmul2((r), sg)
#define MULG8(r) bmul2((r), sg8)
#endif
#else
      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);
      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];
#define MULG(r) bmul2((r), sg)
#define MULG8(r) bmul2((r), sg8)
#endif
      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,
                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};
      if (!two) {
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          uint32_t r0, r1, r2, r3;
          dq4(ww[st], r0, r1, r2, r3);
#if OKC_NACC == 4
          mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, MULG(r0), MULG8(r1), MULG(r2), MULG8(r3), b[2 * st], b[2 * st + 1]);
#elif OKC_TWOACC
          mma_bf16((st & 1) ? acc2 : acc, MULG(r0), MULG8(r1), MULG(r2), MULG8(r3), b[2 * st], b[2 * st + 1]);
#else
          mma_bf16(acc, MULG(r0), MULG8(r1), MULG(r2), MULG8(r3), b[2 * st], b[2 * st + 1]);
#endif
        }
      } else {
        const uint4 c0 = *reinterpret_cast<const uint4*>(x1r + c * 64);
        const uint4 c1 = *reinterpret_cast<const uint4*>(x1r + c * 64 + 8);
        const uint32_t bb[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          uint32_t r0, r1, r2, r3;
          dq4(ww[st], r0, r1, r2, r3);
          const uint32_t a0 = MULG(r0), a1 = MULG8(r1), a2 = MULG(r2), a3 = MULG8(r3);
          mma_bf16(acc, a0, a1, a2, a3, b[2 * st], b[2 * st + 1]);
          mma_bf16(acc2, a0, a1, a2, a3, bb[2 * st], bb[2 * st + 1]);
        }
      }
    }
#endif
#undef MULG
#undef MULG8
#if OKC_NACC == 4
    if (!two) {
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];
    }
#elif OKC_TWOACC
    if (!two) { acc[0] += acc2[0]; acc[1] += acc2[1]; acc[2] += acc2[2]; acc[3] += acc2[3]; }
#endif
    float* rd = &S.red[i & 1][warp][lane][0];
    *reinterpret_cast<float4*>(rd) = make_float4(acc[0], acc[1], acc[2], acc[3]);
#if OKC_SKT > 8
    if (two) *reinterpret_cast<float4*>(rd + 4) = make_float4(acc2[0], acc2[1], acc2[2], acc2[3]);
#endif
    CTA_SYNC();  // slot sl fully consumed; partials visible
    if (tid == 0) {
      if (i + OKC_NS < nU) { ring_issue<PB>(p, S, v + OKC_NS, ring + sl * UNIT_BYTES, bars + sl); ++snext; }
      else refill(sl);
    }
#if OKC_BEPI
    {
      // balanced epilogue: every thread reduces 4 of the 8 warp partials of one (output pair, half)
      const int hw = tid >> 5, h = hw >> 2, tt = hw & 3, gg = (tid >> 2) & 7, q = tid & 3;
      const int ln = gg * 4 + tt;
      const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
      const int ntiles = two ? 2 : 1;
      for (int nt = 0; nt < ntiles; ++nt) {
        const int kk = nt * 4 + h + ((q & 2) ? 2 : 0);
        const int w0 = (q & 1) * 4;
        float v = S.red[i & 1][w0][ln][kk] + S.red[i & 1][w0 + 1][ln][kk] + S.red[i & 1][w0 + 2][ln][kk] +
                  S.red[i & 1][w0 + 3][ln][kk];
        v += __shfl_xor_sync(0xffffffffu, v, 1);
        const float s1 = __shfl_xor_sync(0xffffffffu, v, 2);
        const float s0 = v;
        const int tk = nt * 8 + 2 * tt + h;
        if (!PB) {
          if (q == 0 && tk < cnt) {
            const float gsc = p.gsA[ce] * TWO62;
            const float gt = s0 * gsc, up = s1 * gsc;
            st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));
          }
        } else {
          const float gsc = p.gsB[ce] * TWO62;
          const int n0 = rb * 16 + gg;
          const float v0f = s0 * gsc * (1.f + __bfloat162float(p.normw[n0])), v1f = s1 * gsc * (1.f + __bfloat162float(p.normw[n0 + 8]));
          float sq = v0f * v0f + v1f * v1f;
          if (q == 0 && tk < cnt) {
            st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
            st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
          }
          sq += __shfl_xor_sync(0xffffffffu, sq, 4);
          sq += __shfl_xor_sync(0xffffffffu, sq, 8);
          sq += __shfl_xor_sync(0xffffffffu, sq, 16);
          if (q == 0 && gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
        }
      }
    }
#elif OKC_EPI4
    // epilogue on 4 warps: each output's 8 warp partials are split over a lane pair (lane, lane ^ 16)
    if (tid < 128) {
      const int half = (tid >> 4) & 1;
      const int o = (tid >> 5) * 16 + (tid & 15);
      const int h = o >> 5, ln = o & 31;
      const int gg = ln >> 2, tt = ln & 3;
      float s0 = 0.f, s1 = 0.f;
#pragma unroll
      for (int k = 0; k < NW / 2; ++k) {
        const float* q = &S.red[i & 1][half * (NW / 2) + k][ln][0];
        s0 += q[h];
        s1 += q[2 + h];
      }
      s0 += __shfl_xor_sync(0xffffffffu, s0, 16);
      s1 += __shfl_xor_sync(0xffffffffu, s1, 16);
      const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
      const int tk = 2 * tt + h;
      if (!PB) {
        if (half == 0 && tk < cnt) {
          const float gsc = p.gsA[ce] * TWO62;
          const float gt = s0 * gsc, up = s1 * gsc;
          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));
        }
      } else {
        const float gsc = p.gsB[ce] * TWO62;
        const int n0 = rb * 16 + gg;
        const float v0f = s0 * gsc * nwc0, v1f = s1 * gsc * nwc1;
        float sq = v0f * v0f + v1f * v1f;
        if (half == 0 && tk < cnt) {
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
        }
        sq += __shfl_xor_sync(0xffffffffu, sq, 4);
        sq += __shfl_xor_sync(0xffffffffu, sq, 8);
        if (half == 0 && (tid & 12) == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);  // two warps per token
      }
    }
#else
    if (tid < (two ? 128 : 64)) {
      const int nt = tid >> 6, h = (tid >> 5) & 1, ln = tid & 31;
      const int gg = ln >> 2, tt = ln & 3;
      float s0 = 0.f, s1 = 0.f;
#pragma unroll
      for (int ww = 0; ww < NW; ++ww) {
        const float* q = &S.red[i & 1][ww][ln][nt * 4];
        s0 += q[h];
        s1 += q[2 + h];
      }
      const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
      const int tk = nt * 8 + 2 * tt + h;
      if (!PB) {
        if (tk < cnt) {
          const float gsc = p.gsA[ce] * TWO62;
          const float gt = s0 * gsc, up = s1 * gsc;
          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));
        }
      } else {
        const float gsc = p.gsB[ce] * TWO62;
        const int n0 = rb * 16 + gg;
        const float v0f = s0 * gsc * nwc0, v1f = s1 * gsc * nwc1;
        float sq = v0f * v0f + v1f * v1f;
        if (tk < cnt) {
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
        }
        sq += __shfl_xor_sync(0xffffffffu, sq, 4);
        sq += __shfl_xor_sync(0xffffffffu, sq, 8);
        sq += __shfl_xor_sync(0xffffffffu, sq, 16);
        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
      }
    }
#endif
    nwc0 = nwn0; nwc1 = nwn1;
    ++nproc;
  }
  seq = seq0 + nproc;
  if (!PB) {
    __threadfence();
    CTA_SYNC();
    if (tid == 0) {
      if (nU > 0) {
        const int jf = njF + (v0 >> rsh), jl = njF + ((v1 - 1) >> rsh);
        for (int jj = jf; jj <= jl; ++jj) {
          const int a = max(v0, (jj - njF) * rbs), b = min(v1, (jj - njF + 1) * rbs);
          atomicAdd(p.doneA + jj, b - a);
        }
      }
      for (int jj = jdmin; jj <= jdmax; ++jj) {  // pool units
        const int k = S.dync[jj];
        if (k) { atomicAdd(p.doneA + jj, k); S.dync[jj] = 0; }
      }
    }
  }
  if (!PB && tid == 0) S.preissuedB = preB;
  if (PB && tid == 0) S.preissuedB = 0;
  CTA_SYNC();
}
#else
template <bool PB>
__device__ void skinny_ring(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq,
                            __nv_bfloat16* x1base, bool cpf_ok, int cB0, int cBn, int nSall, int poolBase) {
  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = 16u + p.zero;  // opaque 16 (scale decode IMAD)
  (void)k16;
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int nU = v1 - v0;
  if (nU <= 0) return;
  const int seq0 = seq;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  // order earlier generic-proxy accesses of this smem region (fat path) before async-proxy (TMA) writes
  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif
  // cross-phase prefill: with no fat jobs, phase A's last ring refills already fetch this CTA's first B units
  int preB = 0;
  const bool cpf = OKC_CPF && !PB && cpf_ok;
  const int v0B = cB0, nB = cBn;
  const int pre = PB ? S.preissuedB : S.preissued;
  for (int i = pre; i < OKC_NS; ++i) {
    const int sl = (seq0 + i) % OKC_NS;
    if (i < nU) ring_issue_all<PB>(p, S, v0 + i, ring + sl * UNIT_BYTES, bars + sl);
    else if (cpf && i - nU < nB) { ring_issue_all<true>(p, S, v0B + (i - nU), ring + sl * UNIT_BYTES, bars + sl); ++preB; }
  }
  uint4 x0[2 * NCH];
  int curj = -1;
  bool two = false;
  __nv_bfloat16* x1s = x1base + warp * 8 * X1P;
  // phase B epilogue: prefetch (1 + post_norm_w) of the next unit's rows one unit ahead
  float nwc0 = 0.f, nwc1 = 0.f;
#if OKC_EPI4
  const int egg = ((tid >> 5) & 1) * 4 + ((tid & 15) >> 2);  // epilogue row group of this thread (4-warp layout)
#else
  const int egg = (tid & 31) >> 2;
#endif
  if (PB && tid < 128) { const int rbq = v0 % rbs; nwc0 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg)); nwc1 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8)); }
  for (int i = 0; i < nU; ++i) {
    const int v = v0 + i;
    const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;
    float nwn0 = 0.f, nwn1 = 0.f;
    if (PB && tid < 128 && i + 1 < nU) {
      const int rbq = (v + 1) % rbs;
      nwn0 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg)); nwn1 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8));
    }
#if OKC_XPF
    if (!PB) {  // L2 prefetch of the x rows of the job starting OKC_NS units ahead
      const int va = v + OKC_NS;
      if (i + OKC_NS < nU && va / rbs != v / rbs) {
        const int ja = njF + va / rbs;
        const int ga = lane >> 2, ta = lane & 3;
        if (ga < S.jC[ja]) {
          // the warp's 1 KB slice of this token row = 8 lines; lanes t take lines 2t, 2t+1
          const char* row = reinterpret_cast<const char*>(p.x + (size_t)(S.ent[S.jS[ja] + ga] >> 3) * HD + warp * 512) + 256 * ta;
          asm volatile("prefetch.global.L2 [%0];\n" ::"l"(row));
          asm volatile("prefetch.global.L2 [%0];\n" ::"l"(row + 128));
        }
      }
    }
#endif
    if (j != curj) {
      curj = j;
      two = (OKC_SKT > 8) && S.jC[j] > 8;
      if (PB) {
        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
        __syncwarp();
      }
      load_x_regs<PB>(p, S, j, lane, warp, x0);
      if (two) load_x1_smem<PB>(p, S, j, lane, warp, x1s);
    }
    const int sl = (seq0 + i) % OKC_NS;
    const uint32_t par = (uint32_t)(((seq0 + i) / OKC_NS) & 1);
#if !OKC_NOLOAD
    mbar_wait(bars + sl, par);
#else
    if (i < OKC_NS) mbar_wait(bars + sl, par);
#endif
    const unsigned char* slot = ring + sl * UNIT_BYTES;
#if OKC_NW == 16
    // a warp covers half of an 8-warp K slice: its 4 chunks' scale bytes are 8 of the slice lane's 16
    const uint2 sc2 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (warp >> 1) * 512 + lane * 16 + (warp & 1) * 8);
    const uint4 sc = make_uint4(sc2.x, sc2.y, 0u, 0u);
#else
    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);
#endif
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
#if OKC_NACC == 4
    float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};
#endif
    const __nv_bfloat16* x1r = x1s + (lane >> 2) * X1P + 16 * (lane & 3);
#if !OKC_NOCOMPUTE
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);
#if OKC_PRMTSC
      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);
#if OKC_SKSC2
      // rows g / g+8 of this chunk are adjacent scale bytes: one PRMT + one IMAD build both, the multiplies take the
      // halves (HMUL2 .H0_H0 / .H1_H1)
      const uint32_t sgp = __byte_perm(swd, 0u, (c & 1) ? 0x4342u : 0x4140u) * k16 + 0x5C005C00u;
#define MULG(r) bmul2l((r), sgp)
#define MULG8(r) bmul2h((r), sgp)
#else
      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);
#define MULG(r) bmul2((r), sg)
#define MULG8(r) bmul2((r), sg8)
#endif
#else
      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);
      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];
#define MULG(r) bmul2((r), sg)
#define MULG8(r) bmul2((r), sg8)
#endif
      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,
                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};
      if (!two) {
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          uint32_t r0, r1, r2, r3;
          dq4(ww[st], r0, r1, r2, r3);
#if OKC_NACC == 4
          mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, MULG(r0), MULG8(r1), MULG(r2), MULG8(r3), b[2 * st], b[2 * st + 1]);
#elif OKC_TWOACC
          mma_bf16((st & 1) ? acc2 : acc, MULG(r0), MULG8(r1), MULG(r2), MULG8(r3), b[2 * st], b[2 * st + 1]);
#else
          mma_bf16(acc, MULG(r0), MULG8(r1), MULG(r2), MULG8(r3), b[2 * st], b[2 * st + 1]);
#endif
        }
      } else {
        const uint4 c0 = *reinterpret_cast<const uint4*>(x1r + c * 64);
        const uint4 c1 = *reinterpret_cast<const uint4*>(x1r + c * 64 + 8);
        const uint32_t bb[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          uint32_t r0, r1, r2, r3;
          dq4(ww[st], r0, r1, r2, r3);
          const uint32_t a0 = MULG(r0), a1 = MULG8(r1), a2 = MULG(r2), a3 = MULG8(r3);
          mma_bf16(acc, a0, a1, a2, a3, b[2 * st], b[2 * st + 1]);
          mma_bf16(acc2, a0, a1, a2, a3, bb[2 * st], bb[2 * st + 1]);
        }
      }
    }
#endif
#undef MULG
#undef MULG8
#if OKC_NACC == 4
    if (!two) {
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];
    }
#elif OKC_TWOACC
    if (!two) { acc[0] += acc2[0]; acc[1] += acc2[1]; acc[2] += acc2[2]; acc[3] += acc2[3]; }
#endif
    float* rd = &S.red[i & 1][warp][lane][0];
    *reinterpret_cast<float4*>(rd) = make_float4(acc[0], acc[1], acc[2], acc[3]);
#if OKC_SKT > 8
    if (two) *reinterpret_cast<float4*>(rd + 4) = make_float4(acc2[0], acc2[1], acc2[2], acc2[3]);
#endif
    CTA_SYNC();  // slot sl fully consumed; partials visible
#if !OKC_NOLOAD
    if (i + OKC_NS < nU) ring_issue_all<PB>(p, S, v + OKC_NS, ring + sl * UNIT_BYTES, bars + sl);
    else if (cpf && i + OKC_NS - nU < nB) {
      ring_issue_all<true>(p, S, v0B + (i + OKC_NS - nU), ring + sl * UNIT_BYTES, bars + sl);
      ++preB;
    }
#endif
#if OKC_BEPI
    {
      // balanced epilogue: every thread reduces 4 of the 8 warp partials of one (output pair, half)
      const int hw = tid >> 5, h = hw >> 2, tt = hw & 3, gg = (tid >> 2) & 7, q = tid & 3;
      const int ln = gg * 4 + tt;
      const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
      const int ntiles = two ? 2 : 1;
      for (int nt = 0; nt < ntiles; ++nt) {
        const int kk = nt * 4 + h + ((q & 2) ? 2 : 0);
        const int w0 = (q & 1) * 4;
        float v = S.red[i & 1][w0][ln][kk] + S.red[i & 1][w0 + 1][ln][kk] + S.red[i & 1][w0 + 2][ln][kk] +
                  S.red[i & 1][w0 + 3][ln][kk];
        v += __shfl_xor_sync(0xffffffffu, v, 1);
        const float s1 = __shfl_xor_sync(0xffffffffu, v, 2);
        const float s0 = v;
        const int tk = nt * 8 + 2 * tt + h;
        if (!PB) {
          if (q == 0 && tk < cnt) {
            const float gsc = p.gsA[ce] * TWO62;
            const float gt = s0 * gsc, up = s1 * gsc;
            st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));
          }
        } else {
          const float gsc = p.gsB[ce] * TWO62;
          const int n0 = rb * 16 + gg;
          const float v0f = s0 * gsc * (1.f + __bfloat162float(p.normw[n0])), v1f = s1 * gsc * (1.f + __bfloat162float(p.normw[n0 + 8]));
          float sq = v0f * v0f + v1f * v1f;
          if (q == 0 && tk < cnt) {
            st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
            st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
          }
          sq += __shfl_xor_sync(0xffffffffu, sq, 4);
          sq += __shfl_xor_sync(0xffffffffu, sq, 8);
          sq += __shfl_xor_sync(0xffffffffu, sq, 16);
          if (q == 0 && gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
        }
      }
    }
#elif OKC_EPI4
    // epilogue on 4 warps: each output's 8 warp partials are split over a lane pair (lane, lane ^ 16)
    if (tid < 128) {
      const int half = (tid >> 4) & 1;
      const int o = (tid >> 5) * 16 + (tid & 15);
      const int h = o >> 5, ln = o & 31;
      const int gg = ln >> 2, tt = ln & 3;
      float s0 = 0.f, s1 = 0.f;
#pragma unroll
      for (int k = 0; k < NW / 2; ++k) {
        const float* q = &S.red[i & 1][half * (NW / 2) + k][ln][0];
        s0 += q[h];
        s1 += q[2 + h];
      }
      s0 += __shfl_xor_sync(0xffffffffu, s0, 16);
      s1 += __shfl_xor_sync(0xffffffffu, s1, 16);
      const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
      const int tk = 2 * tt + h;
      if (!PB) {
        if (half == 0 && tk < cnt) {
          const float gsc = p.gsA[ce] * TWO62;
          const float gt = s0 * gsc, up = s1 * gsc;
          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));
        }
      } else {
        const float gsc = p.gsB[ce] * TWO62;
        const int n0 = rb * 16 + gg;
        const float v0f = s0 * gsc * nwc0, v1f = s1 * gsc * nwc1;
        float sq = v0f * v0f + v1f * v1f;
        if (half == 0 && tk < cnt) {
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
        }
        sq += __shfl_xor_sync(0xffffffffu, sq, 4);
        sq += __shfl_xor_sync(0xffffffffu, sq, 8);
        if (half == 0 && (tid & 12) == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);  // two warps per token
      }
    }
#else
    if (tid < (two ? 128 : 64)) {
      const int nt = tid >> 6, h = (tid >> 5) & 1, ln = tid & 31;
      const int gg = ln >> 2, tt = ln & 3;
      float s0 = 0.f, s1 = 0.f;
#pragma unroll
      for (int ww = 0; ww < NW; ++ww) {
        const float* q = &S.red[i & 1][ww][ln][nt * 4];
        s0 += q[h];
        s1 += q[2 + h];
      }
      const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
      const int tk = nt * 8 + 2 * tt + h;
      if (!PB) {
        if (tk < cnt) {
          const float gsc = p.gsA[ce] * TWO62;
          const float gt = s0 * gsc, up = s1 * gsc;
          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));
        }
      } else {
        const float gsc = p.gsB[ce] * TWO62;
        const int n0 = rb * 16 + gg;
        const float v0f = s0 * gsc * nwc0, v1f = s1 * gsc * nwc1;
        float sq = v0f * v0f + v1f * v1f;
        if (tk < cnt) {
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
        }
        sq += __shfl_xor_sync(0xffffffffu, sq, 4);
        sq += __shfl_xor_sync(0xffffffffu, sq, 8);
        sq += __shfl_xor_sync(0xffffffffu, sq, 16);
        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
      }
    }
#endif
    nwc0 = nwn0; nwc1 = nwn1;
  }
#if OKC_NOLOAD
  seq = seq0 + min(nU, OKC_NS);
#else
  seq = seq0 + nU;
#endif
  if (!PB) {
    __threadfence();
    CTA_SYNC();
    if (tid == 0) {
      const int jf = njF + v0 / rbs, jl = njF + (v1 - 1) / rbs;
      for (int jj = jf; jj <= jl; ++jj) {
        const int a = max(v0, (jj - njF) * rbs), b = min(v1, (jj - njF + 1) * rbs);
        atomicAdd(p.doneA + jj, b - a);
      }
    }
  }
  if (!PB && tid == 0) S.preissuedB = preB;
  if (PB && tid == 0) S.preissuedB = 0;
  CTA_SYNC();
}
#endif

// ---------------------------------------------------------------- fat unit: CTA = 256 rows (warp = 2 x 16 rows) x full K, <=64 tokens
// ---------------------------------------------------------------- fat unit through the smem ring
// A fat unit is 256 rows x full K for <= 64 tokens. Its weights arrive in 16 k-stages of 36 KB (16 row blocks x
// 4 chunks + their scales, contiguous in the fat-order copy) through slots 0..2 of the ring region; the x tile
// for the stage is staged by cp.async into two buffers that live in slots 3..4.
#ifndef OKC_NSF
#define OKC_NSF 3
#endif
constexpr int NSF = OKC_NSF;
template <bool PB>
__device__ __forceinline__ void fat_issue(const Params& p, int e, int ub, int ks, unsigned char* slot, uint64_t* bar, bool nat = false) {
  const int nub = PB ? UB_B : UB_A;
  const unsigned char* src = (nat ? (PB ? p.wBn : p.wAn) : (PB ? p.wBf : p.wAf)) + (((size_t)e * nub + ub) * 16 + ks) * (size_t)UNIT_BYTES;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && OKC_USE_TMA
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
                 ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");
  }
#else
  for (int i = threadIdx.x; i < UNIT_BYTES / 16; i += NT)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(slot + i * 16)), "l"(src + i * 16));
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(smem_u32(bar)) : "memory");
#endif
}

// ---------------------------------------------------------------- fat units: one continuous stream per CTA
// Units come from a dynamic queue; the next unit is claimed when the current one starts, so the ring keeps
// streaming across unit boundaries (the refills of the last k-stages already fetch the next unit's weights).
// ---------------------------------------------------------------- wgmma (sm90a): A from registers, B (x) from smem
__device__ __forceinline__ uint64_t wg_desc_sw128(uint32_t saddr) {  // K-major, 128 B swizzle, 8-row groups 1 KB apart
  return (uint64_t)((saddr >> 4) & 0x3FFF) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 62);
}
__device__ __forceinline__ void wg_pin(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
  asm volatile("" : "+r"(a), "+r"(b), "+r"(c), "+r"(d));
}
__device__ __forceinline__ uint64_t wg_desc(uint32_t saddr, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((saddr >> 4) & 0x3FFF) | ((uint64_t)((lbo >> 4) & 0x3FFF) << 16) | ((uint64_t)((sbo >> 4) & 0x3FFF) << 32);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }
template <int NTT> struct WG;
template <> struct WG<1> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
#endif
  }
};
template <> struct WG<2> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
#endif
  }
};
template <> struct WG<3> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
#endif
  }
};
template <> struct WG<4> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
#endif
  }
};
template <> struct WG<5> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
#endif
  }
};
template <> struct WG<6> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
#endif
  }
};
template <> struct WG<7> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
#endif
  }
};
template <> struct WG<8> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3]), "+f"(acc[7][0]), "+f"(acc[7][1]), "+f"(acc[7][2]), "+f"(acc[7][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
#endif
  }
};

// one 36 KB k-stage of a fat unit with wgmma: A = dequantized weights (natural k order) from registers, B = x tile from
// shared memory in the K-major no-swizzle core-matrix layout [n8][k8][8 rows][16 B] (LBO 128 B along K, SBO 4 KB along N)
template <int NTT>
__device__ __forceinline__ void fat_compute_wg(const Smem& S, const unsigned char* slot, const __nv_bfloat16* xs, int warp,
                                               int lane, float (&acc0)[FT_T / 8][4], float (&acc1)[FT_T / 8][4], uint32_t zero) {
  // x stage: [kb 4][64 rows][128 B], 128 B swizzle (chunk c of row n at c ^ (n & 7)); k-step st of chunk cc reads
  // k block cc at byte offset 32 st
  const int g = lane >> 2;
  const uint32_t xb = smem_u32(xs);
  uint32_t A[2][8][4];  // [chunk parity][k-step x rb][fragment]; a set stays untouched until its group retired
  uint32_t keep = 0;    // folds every fragment after its group retired, so ptxas keeps the registers live (no reuse)
#pragma unroll
  for (int cc = 0; cc < 4; ++cc) {
    const int sb = cc & 1;
    const uint4 w0 = *reinterpret_cast<const uint4*>(slot + ((warp * 4 + cc) * 32 + lane) * 16);
    const uint4 w1 = *reinterpret_cast<const uint4*>(slot + (((warp + 8) * 4 + cc) * 32 + lane) * 16);
    const uint2 sc0 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + ((warp * 4 + cc) * 8 + g) * 8);
    const uint2 sc1 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (((warp + 8) * 4 + cc) * 8 + g) * 8);
#pragma unroll
    for (int st = 0; st < 4; ++st) {
#if OKC_WGALU
      // e4m3 -> bf16 without the LUT: bf16 bits (b << 4) hold value * 2^-120 (subnormal codes included), two exact
      // multiplies by 2^92 give value * 2^64 like the LUT; rows g | g+8 travel as one bf16x2
      const uint32_t K92 = 0x6D806D80u;  // bf16x2(2^92, 2^92)
      const uint32_t p0 = bmul2(bmul2((__byte_perm(sc0.x, sc0.y, 0x0400u | st | ((st + 4) << 8)) & 0x00FF00FFu) << 4, K92), K92);
      const uint32_t p1 = bmul2(bmul2((__byte_perm(sc1.x, sc1.y, 0x0400u | st | ((st + 4) << 8)) & 0x00FF00FFu) << 4, K92), K92);
#if OKC_FTSC2
#define FMG0(r) bmul2l((r), p0)
#define FMG80(r) bmul2h((r), p0)
#define FMG1(r) bmul2l((r), p1)
#define FMG81(r) bmul2h((r), p1)
#else
      const uint32_t sg0 = __byte_perm(p0, p0, 0x1010u), sg80 = __byte_perm(p0, p0, 0x3232u);
      const uint32_t sg1 = __byte_perm(p1, p1, 0x1010u), sg81 = __byte_perm(p1, p1, 0x3232u);
#define FMG0(r) bmul2((r), sg0)
#define FMG80(r) bmul2((r), sg80)
#define FMG1(r) bmul2((r), sg1)
#define FMG81(r) bmul2((r), sg81)
#endif
#else
      const uint32_t sg0 = S.lut[(sc0.x >> (8 * st)) & 0xff], sg80 = S.lut[(sc0.y >> (8 * st)) & 0xff];
      const uint32_t sg1 = S.lut[(sc1.x >> (8 * st)) & 0xff], sg81 = S.lut[(sc1.y >> (8 * st)) & 0xff];
#define FMG0(r) bmul2((r), sg0)
#define FMG80(r) bmul2((r), sg80)
#define FMG1(r) bmul2((r), sg1)
#define FMG81(r) bmul2((r), sg81)
#endif
      uint32_t r0, r1, r2, r3, q0, q1, q2, q3;
      dq4((&w0.x)[st], r0, r1, r2, r3);
      dq4((&w1.x)[st], q0, q1, q2, q3);
      A[sb][2 * st][0] = FMG0(r0); A[sb][2 * st][1] = FMG80(r1);
      A[sb][2 * st][2] = FMG0(r2); A[sb][2 * st][3] = FMG80(r3);
      A[sb][2 * st + 1][0] = FMG1(q0); A[sb][2 * st + 1][1] = FMG81(q1);
      A[sb][2 * st + 1][2] = FMG1(q2); A[sb][2 * st + 1][3] = FMG81(q3);
#undef FMG0
#undef FMG80
#undef FMG1
#undef FMG81
    }
    wg_fence();
#pragma unroll
    for (int st = 0; st < 4; ++st) {
      const uint64_t dk = wg_desc_sw128(xb + cc * 8192 + st * 32);
      WG<NTT>::mma(acc0, A[sb][2 * st][0], A[sb][2 * st][1], A[sb][2 * st][2], A[sb][2 * st][3], dk);
      WG<NTT>::mma(acc1, A[sb][2 * st + 1][0], A[sb][2 * st + 1][1], A[sb][2 * st + 1][2], A[sb][2 * st + 1][3], dk);
    }
    wg_commit();
    if (cc >= 1) {
      wg_wait<1>();  // the previous chunk's group retired: its register set may be rewritten
#pragma unroll
      for (int k = 0; k < 8; ++k) keep ^= A[sb ^ 1][k][0] ^ A[sb ^ 1][k][1] ^ A[sb ^ 1][k][2] ^ A[sb ^ 1][k][3];
    }
  }
  wg_wait<0>();
#pragma unroll
  for (int k = 0; k < 8; ++k) keep ^= A[1][k][0] ^ A[1][k][1] ^ A[1][k][2] ^ A[1][k][3];
  acc0[0][0] += __uint_as_float(keep & zero);  // adds +0.0f
}

// one 36 KB k-stage of a fat unit: 2 row blocks per warp x NTT token tiles (exact tile count, no predicated MMAs)
template <int NTT>
__device__ __forceinline__ void fat_compute(const Smem& S, const unsigned char* slot, const __nv_bfloat16* xs, int warp,
                                            int lane, float (&acc0)[FT_T / 8][4], float (&acc1)[FT_T / 8][4]) {
  const int g = lane >> 2, t = lane & 3;
  const uint2 sc0 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (warp * 32 + lane) * 8);
#if OKC_NW == 8
  const uint2 sc1 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + ((warp + 8) * 32 + lane) * 8);
#endif
#pragma unroll
  for (int cc = 0; cc < 4; ++cc) {
    const uint4 w0 = *reinterpret_cast<const uint4*>(slot + ((warp * 4 + cc) * 32 + lane) * 16);
#if OKC_NW == 8
    const uint4 w1 = *reinterpret_cast<const uint4*>(slot + (((warp + 8) * 4 + cc) * 32 + lane) * 16);
#endif
#if OKC_PRMTSC
    const uint32_t sg0 = SCD(cc < 2 ? sc0.x : sc0.y, (cc & 1) * 2), sg80 = SCD(cc < 2 ? sc0.x : sc0.y, (cc & 1) * 2 + 1);
#if OKC_NW == 8
    const uint32_t sg1 = SCD(cc < 2 ? sc1.x : sc1.y, (cc & 1) * 2), sg81 = SCD(cc < 2 ? sc1.x : sc1.y, (cc & 1) * 2 + 1);
#endif
#else
    const int sh = (cc & 1) * 16;
    const uint32_t sw0 = (cc < 2 ? sc0.x : sc0.y) >> sh;
#if OKC_NW == 8
    const uint32_t sw1 = (cc < 2 ? sc1.x : sc1.y) >> sh;
#endif
    const uint32_t sg0 = S.lut[sw0 & 0xff], sg80 = S.lut[(sw0 >> 8) & 0xff];
#if OKC_NW == 8
    const uint32_t sg1 = S.lut[sw1 & 0xff], sg81 = S.lut[(sw1 >> 8) & 0xff];
#endif
#endif
    const __nv_bfloat16* xr = xs + g * XP_FT + cc * 64 + 16 * t;
#pragma unroll
    for (int sp = 0; sp < 2; ++sp) {
      uint32_t a0[2][4], a1[2][4];
      (void)a1;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int s2 = 2 * sp + h;
        uint32_t r0, r1, r2, r3;
        dq4((&w0.x)[s2], r0, r1, r2, r3);
        a0[h][0] = bmul2(r0, sg0); a0[h][1] = bmul2(r1, sg80); a0[h][2] = bmul2(r2, sg0); a0[h][3] = bmul2(r3, sg80);
#if OKC_NW == 8
        dq4((&w1.x)[s2], r0, r1, r2, r3);
        a1[h][0] = bmul2(r0, sg1); a1[h][1] = bmul2(r1, sg81); a1[h][2] = bmul2(r2, sg1); a1[h][3] = bmul2(r3, sg81);
#endif
      }
#pragma unroll
      for (int nt = 0; nt < NTT; ++nt) {
        const uint4 bv = *reinterpret_cast<const uint4*>(xr + nt * 8 * XP_FT + 8 * sp);
        mma_bf16(acc0[nt], a0[0][0], a0[0][1], a0[0][2], a0[0][3], bv.x, bv.y);
#if OKC_NW == 8
        mma_bf16(acc1[nt], a1[0][0], a1[0][1], a1[0][2], a1[0][3], bv.x, bv.y);
#endif
        mma_bf16(acc0[nt], a0[1][0], a0[1][1], a0[1][2], a0[1][3], bv.z, bv.w);
#if OKC_NW == 8
        mma_bf16(acc1[nt], a1[1][0], a1[1][1], a1[1][2], a1[1][3], bv.z, bv.w);
#endif
      }
    }
  }
}

template <bool PB, bool BIG>
__device__ __forceinline__ void fat_stream(const Params& p, Smem& S, __nv_bfloat16* xbuf, int nF, int ubs) {
  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const int g = lane >> 2, t = lane & 3;
  int* ctr = p.ctr + (PB ? 1 : 0);
  unsigned char* ring = reinterpret_cast<unsigned char*>(xbuf);
  uint64_t* fb = S.bars + 5;
  __nv_bfloat16* xs0 = reinterpret_cast<__nv_bfloat16*>(ring + NSF * UNIT_BYTES);
  constexpr int NKS = HD / KS;  // 16
#if OKC_FATSTATIC
  int u = blockIdx.x;
  (void)ctr;
#else
  if (tid == 0) S.unit = atomicAdd(ctr, 1);
  CTA_SYNC();
  int u = S.unit;
  CTA_SYNC();
#endif
  if (u >= nF) return;
  constexpr int UBS = PB ? UB_B : UB_A;  // == ubs, as a constant (shift instead of a divide)
  (void)ubs;
  auto job_of = [&](int uu, int& j, int& ub) { const int jq = uu / UBS; ub = uu - jq * UBS; j = S.forder[jq]; };
  auto wait_A = [&](int j) {  // phase B: inter rows of job j complete
    if (PB) {
      if (tid == 0) { while (ld_acquire(p.doneA + j) < UB_A) __nanosleep(64); }
      CTA_SYNC();
    }
  };
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  auto is_nat = [&](int j) { return BIG && OKC_WGMIN <= 8 && ((S.jC[j] + 7) >> 3) >= OKC_WGMIN; };
#else
  auto is_nat = [&](int j) { (void)j; return false; };  // wgmma needs sm90a
#endif
  auto stage_x = [&](int j, int ks, int xsl) {
    const int cnt = S.jC[j], st = S.jS[j];
    const int ntl = (cnt + 7) >> 3;
    const int nrow = ntl * 8;
    __nv_bfloat16* xs = xs0 + xsl * (FT_T * XP_FT);
    const int per = nrow * (KS / 8);
    if (is_nat(j)) {
      // 128 B swizzled K-major layout [kb 4][64 rows][128 B]: a warp copies one 512 B row (conflict-free)
      unsigned char* xsb = reinterpret_cast<unsigned char*>(xs);
      for (int i = tid; i < per; i += NT) {
        const int r = i >> 5, ch = i & 31;
        const bool v = r < cnt;
        const __nv_bfloat16* src;
        if (PB) src = p.inter + (size_t)(st + (v ? r : 0)) * ID + ks * KS + ch * 8;
        else src = p.x + (size_t)(v ? (S.ent[st + r] >> 3) : 0) * HD + ks * KS + ch * 8;
        cp_async16(reinterpret_cast<__nv_bfloat16*>(xsb + (ch >> 3) * 8192 + r * 128 + (((ch & 7) ^ (r & 7)) << 4)), src, v);
      }
    } else {
      for (int i = tid; i < per; i += NT) {
        int r = i / (KS / 8), c = i - r * (KS / 8);
        bool v = r < cnt;
        const __nv_bfloat16* src;
        if (PB) src = p.inter + (size_t)(st + (v ? r : 0)) * ID + ks * KS + c * 8;
        else src = p.x + (size_t)(v ? (S.ent[st + r] >> 3) : 0) * HD + ks * KS + c * 8;
        cp_async16(xs + r * XP_FT + c * 8, src, v);
      }
    }
    cp_commit();
  };
  // BIG kernel: per-thread x staging addresses computed once per unit (a stage only adds its k offset)
  const __nv_bfloat16* xsrc[8];
  uint32_t xdst[8];
  uint32_t xval = 0;
  int xn = 0;
  auto prep_x = [&](int j) {
    const int cnt = S.jC[j], st = S.jS[j];
    const int per = ((cnt + 7) >> 3) * 8 * (KS / 8);
    xval = 0;
    xn = per > tid ? min(8, (per - tid + NT - 1) / NT) : 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int i = tid + k * NT;
      const int r = i >> 5, c = i & 31;
      const bool v = i < per && r < cnt;
      if (PB) xsrc[k] = p.inter + (size_t)(st + (v ? r : 0)) * ID + c * 8;
      else xsrc[k] = p.x + (size_t)(v ? (S.ent[st + r] >> 3) : 0) * HD + c * 8;
      xdst[k] = r * XP_FT + c * 8;
      if (v) xval |= 1u << k;
    }
  };
  auto stage_xp = [&](int ks, int xsl) {
    __nv_bfloat16* xs = xs0 + xsl * (FT_T * XP_FT);
#pragma unroll
    for (int k = 0; k < 8; ++k)
      if (k < xn) cp_async16(xs + xdst[k], xsrc[k] + ks * KS, (xval >> k) & 1u);
    cp_commit();
  };
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif
  int f = S.fseq;  // ring position of the current unit's k-stage 0
  int xq = 0;      // x stage parity counter
  int j, ub;
  job_of(u, j, ub);
#pragma unroll
  for (int k = 0; k < NSF; ++k) fat_issue<PB>(p, S.jE[j], ub, k, ring + ((f + k) % NSF) * UNIT_BYTES, fb + (f + k) % NSF, is_nat(j));
  wait_A(j);
  if (OKC_XPRE) { prep_x(j); stage_xp(0, 0); } else stage_x(j, 0, 0);
  while (true) {
#if OKC_FATSTATIC
    const int un = u + gridDim.x;
#else
    if (tid == 0) S.unit_next = atomicAdd(ctr, 1);
    CTA_SYNC();
    const int un = S.unit_next;
#endif
    int jn = 0, ubn = 0;
    if (un < nF) job_of(un, jn, ubn);
    const int e = S.jE[j], cnt = S.jC[j], st = S.jS[j];
    const int ntile = (cnt + 7) >> 3;
    const int ntc = ntile;  // exact token-tile count (rows past cnt are zero-filled)
    const bool nat = is_nat(j);
    const int rb0 = ub * 16 + warp, rb1 = rb0 + 8;
    float acc0[FT_T / 8][4], acc1[FT_T / 8][4];
#pragma unroll
    for (int i = 0; i < FT_T / 8; ++i)
#pragma unroll
      for (int k = 0; k < 4; ++k) { acc0[i][k] = 0.f; acc1[i][k] = 0.f; }
    for (int ks = 0; ks < NKS; ++ks) {
      if (OKC_XPRE) {
        if (ks + 1 < NKS) stage_xp(ks + 1, (xq + 1) & 1);
        else if (un < nF) { wait_A(jn); prep_x(jn); stage_xp(0, (xq + 1) & 1); }
        else cp_commit();
      } else {
        if (ks + 1 < NKS) stage_x(j, ks + 1, (xq + 1) & 1);
        else if (un < nF) { wait_A(jn); stage_x(jn, 0, (xq + 1) & 1); }
        else cp_commit();
      }
      cp_wait<1>();
      const int q = f + ks, sl = q % NSF;
      mbar_wait(fb + sl, (uint32_t)((q / NSF) & 1));
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
      if (nat) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");  // cp.async x writes -> wgmma reads
#endif
      CTA_SYNC();  // x stage visible to every warp
      const unsigned char* slot = ring + sl * UNIT_BYTES;
      const __nv_bfloat16* xs = xs0 + (xq & 1) * (FT_T * XP_FT);
      if (nat) {
        switch (ntc) {
          case 1: fat_compute_wg<1>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 2: fat_compute_wg<2>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 3: fat_compute_wg<3>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 4: fat_compute_wg<4>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 5: fat_compute_wg<5>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 6: fat_compute_wg<6>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 7: fat_compute_wg<7>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          default: fat_compute_wg<8>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
        }
      } else {
        switch (ntc) {
          case 1: fat_compute<1>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 2: fat_compute<2>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 3: fat_compute<3>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 4: fat_compute<4>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 5: fat_compute<5>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 6: fat_compute<6>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 7: fat_compute<7>(S, slot, xs, warp, lane, acc0, acc1); break;
          default: fat_compute<8>(S, slot, xs, warp, lane, acc0, acc1); break;
        }
      }
      CTA_SYNC();  // slot sl and the x stage consumed
      ++xq;
      // refill: the stage NSF ahead, in this unit or the next one
      const int kn = ks + NSF;
      if (kn < NKS) fat_issue<PB>(p, e, ub, kn, ring + sl * UNIT_BYTES, fb + sl, nat);
      else if (un < nF) fat_issue<PB>(p, S.jE[jn], ubn, kn - NKS, ring + sl * UNIT_BYTES, fb + sl, is_nat(jn));
    }
    // epilogue of unit u (registers -> global)
    const float gsc = (PB ? p.gsB[e] : p.gsA[e]) * TWO62;
    if (!PB) {
#pragma unroll
      for (int nt = 0; nt < FT_T / 8; ++nt)
        if (nt < ntile) {
          epi_A(p, st, cnt, nt * 8 + 2 * t, rb0 * 8 + g, gsc, acc0[nt]);
#if OKC_NW == 8
          epi_A(p, st, cnt, nt * 8 + 2 * t, rb1 * 8 + g, gsc, acc1[nt]);
#endif
        }
      __threadfence();
      CTA_SYNC();
      if (tid == 0) atomicAdd(p.doneA + j, 1);
    } else {
#pragma unroll
      for (int nt = 0; nt < FT_T / 8; ++nt)
        if (nt < ntile) {
          float sq0[2], sq1[2];
          epi_B(p, st, cnt, nt * 8 + 2 * t, rb0 * 16 + g, gsc, acc0[nt], sq0);
#if OKC_NW == 8
          epi_B(p, st, cnt, nt * 8 + 2 * t, rb1 * 16 + g, gsc, acc1[nt], sq1);
          sq0[0] += sq1[0]; sq0[1] += sq1[1];
#else
          (void)sq1;
#endif
          ss_reduce_add(p, st, cnt, nt * 8 + 2 * t, sq0, g);
        }
    }
    f += NKS;
    if (un >= nF) break;
    u = un; j = jn; ub = ubn;
  }
  cp_wait<0>();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif
  if (tid == 0) S.fseq = f;
  CTA_SYNC();
}

template <bool PB>
__device__ __noinline__ void fat_stream_ni(const Params& p, Smem& S, __nv_bfloat16* xbuf, int nF, int ubs) {
  fat_stream<PB, false>(p, S, xbuf, nF, ubs);
}



// issue the phase-A skinny ring prologue as soon as the job table exists (before the positions pass)
__device__ void early_issue(const Params& p, Smem& S) {
#if FAST1_ON || OKC_WGS
  return;  // the first units were issued at kernel start (FAST1), or the WGS producer warp owns the ring
#endif
  if (OKC_FATSTATIC && S.njF > 0) {  // fat units run first and own the ring region
    if (threadIdx.x == 0) S.preissued = 0;
    return;
  }
  const int nS = S.njS * RB_A - dyn_pool(S.njS * RB_A, gridDim.x);  // static part (the pool tail is claimed at run time)
  const float rG = __fdividef(1.f, (float)gridDim.x);
  const int v0 = muldiv_g(nS, blockIdx.x, gridDim.x, rG), v1 = muldiv_g(nS, blockIdx.x + 1, gridDim.x, rG);
  unsigned char* ring = reinterpret_cast<unsigned char*>(S.xbuf_ptr);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && OKC_USE_TMA
  // only the first unit, issued by the last warp so warp 0 (routing) is not held up; the ring prologue in
  // skinny_ring issues the rest after routing, so every CTA's first unit is served ahead of the second ones
  const int n = min(OKC_EARLYN, v1 - v0);
  if (threadIdx.x == NT - 32) {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    for (int i = 0; i < n; ++i) { ring_issue<false>(p, S, v0 + i, ring + i * UNIT_BYTES, S.bars + i); S.uslot[i] = v0 + i; }
    S.preissued = n > 0 ? n : 0;
  }
#else
  const int n = min(OKC_NS, v1 - v0);
  for (int i = 0; i < n; ++i) ring_issue_all<false>(p, S, v0 + i, ring + i * UNIT_BYTES, S.bars + i);
  if (threadIdx.x == 0) S.preissued = n > 0 ? n : 0;
#endif
}

// skinny share [v0, v1) of CTA c when nF fat units go round robin first: every CTA ends at the same modeled cost
#if OKC_WS
// ---------------------------------------------------------------- warp-specialized skinny ring (tiny build)
// The NW compute warps only wait for data, compute, release the slot and hand their partial sums over; the helper
// warp (threads NT..NT+31) issues every TMA copy as soon as all compute warps released a slot and runs every unit's
// epilogue, so no CTA-wide barrier sits in the per-unit loop.
#ifndef OKC_WSBL
#define OKC_WSBL 0
#endif
#ifndef OKC_WSBLK
#define OKC_WSBLK 1
#endif
// the last WSBLK units keep their partial buffers until the compute warps' own epilogues: WSBLK > WSNR deadlocks the
// partial ring (tuner, Oct 26: every WSNR=2 / WSBLK=3 config hung)
static_assert(!OKC_WS || !OKC_WSBL || OKC_WSBLK <= OKC_WSNR, "OKC_WSBLK must not exceed OKC_WSNR");
#ifndef OKC_WSB
#define OKC_WSB 0
#endif
#ifndef OKC_WSRF
#define OKC_WSRF 0
#endif
#ifndef OKC_WSER
#define OKC_WSER 0  // WSB: release the ring slot as soon as every warp holds the unit's codes and scales in registers
                    // (warp 0 refills it before the compute), so OKC_NS units stream while a unit is computed, not NS-1
#endif
static_assert(!OKC_WSER || OKC_WSB, "OKC_WSER needs the WSB loop");
__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring);
template <bool PB>
__device__ void skinny_ring_ws(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq) {
  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = 16u + p.zero;  // opaque 16 (scale decode IMAD)
  (void)k16;
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int nU = v1 - v0;
  const int seq0 = seq;
  uint4 x0[2 * NCH];
  int curj = -1;
#if OKC_WSB && OKC_WSBL
  float lgs = 0.f;                             // the last phase-B unit's epilogue constants (warp 0)
  uint32_t lnw0 = 0u, lnw1 = 0u;                // (norm weights as raw bf16 bits)
#endif
  for (int i = 0; i < nU; ++i) {
    const int v = v0 + i;
    const int j = njF + v / rbs;
#if OKC_WSB && OKC_WSBL == 2
    if (PB && warp < OKC_WSBLK && i == nU - 1 - warp) {
      const int rbl = v - (v / RB_B) * RB_B, n0l = rbl * 16 + (lane >> 2);
      lgs = __ldg(p.gsB + S.jE[j]); lnw0 = (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l);
      lnw1 = (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l + 8);
    }
#endif
    if (j != curj) {
      curj = j;
      if (PB) {
        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
        __syncwarp();
      }
      load_x_regs<PB>(p, S, j, lane, warp, x0);
    }
    const int q = seq0 + i;
    const int sl = q % OKC_NS;
#if OKC_NOLOAD
    if (q < OKC_NS)
#endif
    mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));
    const unsigned char* slot = ring + sl * UNIT_BYTES;
#if OKC_NW == 16
    const uint2 sc2 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (warp >> 1) * 512 + lane * 16 + (warp & 1) * 8);
    const uint4 sc = make_uint4(sc2.x, sc2.y, 0u, 0u);
#else
    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);
#endif
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
#if OKC_NACC == 4
    float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};
#endif
#if OKC_WSER
    // the unit's codes into registers first, then the slot is released (mbarrier arrive: release semantics order the
    // loads before it) and warp 0 refills it with unit q + NS while every warp computes
    uint4 wq[NCH];
#pragma unroll
    for (int c = 0; c < NCH; ++c) wq[c] = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);
    __syncwarp();
    if (lane == 0) mbar_arrive1(S.wse + sl);
    if (warp == 0 && lane == 0) {
      const int nA = S.wsr[1] - S.wsr[0], N = nA + (S.wsr[3] - S.wsr[2]);
      if (q + OKC_NS < N) {
        mbar_wait(S.wse + sl, (uint32_t)((q / OKC_NS) & 1));  // every warp holds its codes and scales
        ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);
      }
    }
#endif
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
#if OKC_WSER
      const uint4 w = wq[c];
#else
      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);
#endif
#if OKC_PRMTSC
      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);
      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);
#else
      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);
#if OKC_XNOLUT
      const uint32_t sg = 0x3F803F80u ^ (sw & 1), sg8 = 0x3F803F80u ^ (sw & 2);
#else
      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];
#endif
#endif
      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,
                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};
#pragma unroll
      for (int st = 0; st < 4; ++st) {
        uint32_t r0, r1, r2, r3;
#if OKC_XNODQ
        r0 = ww[st]; r1 = ww[st] >> 1; r2 = ww[st] >> 2; r3 = ww[st] >> 3;
#else
        dq4(ww[st], r0, r1, r2, r3);
#endif
#if OKC_XNOMMA
        float* aa = (st & 1) ? acc2 : acc;
        aa[0] += __uint_as_float(bmul2(r0, sg) ^ b[2 * st]); aa[1] += __uint_as_float(bmul2(r1, sg8) ^ b[2 * st + 1]);
        aa[2] += __uint_as_float(bmul2(r2, sg)); aa[3] += __uint_as_float(bmul2(r3, sg8));
#elif OKC_NACC == 4
        mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, bmul2(r0, sg), bmul2(r1, sg8), bmul2(r2, sg), bmul2(r3, sg8), b[2 * st], b[2 * st + 1]);
#else
        mma_bf16((st & 1) ? acc2 : acc, bmul2(r0, sg), bmul2(r1, sg8), bmul2(r2, sg), bmul2(r3, sg8), b[2 * st], b[2 * st + 1]);
#endif
      }
    }
#if OKC_NACC == 4 && !OKC_XNOMMA
#pragma unroll
    for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];
#else
    acc[0] += acc2[0]; acc[1] += acc2[1]; acc[2] += acc2[2]; acc[3] += acc2[3];
#endif
#if OKC_WSB
    // WSB: the compute warps keep the base loop's CTA barrier per unit (slot consumed + partials visible), thread 0
    // refills the slot (unless OKC_WSER refilled it before the compute) and hands the partial buffer to the helper
    // warp, which runs every epilogue off this loop
    const int bb = q % NRB;
    mbar_wait(S.wsre + bb, (uint32_t)(((q / NRB) & 1) ^ 1));  // the helper has consumed this buffer's previous unit
    *reinterpret_cast<float4*>(&S.red[bb][warp][lane][0]) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    CTA_SYNC();
    if (tid == 0) {
#if !OKC_WSER
      const int nA = S.wsr[1] - S.wsr[0], N = nA + (S.wsr[3] - S.wsr[2]);
      if (q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);
#endif
      if (!(OKC_WSBL && PB && i >= nU - OKC_WSBLK)) mbar_arrive1(S.wsrf + bb);   // WSBL: the last units stay with warps 0..K-1
    }
#else
    __syncwarp();
    if (lane == 0) mbar_arrive1(S.wse + sl);  // this warp is done reading the slot
#if OKC_WSRF
    if (warp == 0 && lane == 0) {  // refill this slot as soon as every compute warp released it (not the helper)
      const int nA = S.wsr[1] - S.wsr[0], N = nA + (S.wsr[3] - S.wsr[2]);
      if (q + OKC_NS < N) { mbar_wait(S.wse + sl, (uint32_t)((q / OKC_NS) & 1)); ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring); }
    }
#endif
    const int bb = q % NRB;
    mbar_wait(S.wsre + bb, (uint32_t)(((q / NRB) & 1) ^ 1));  // the helper has consumed this buffer's previous unit
    *reinterpret_cast<float4*>(&S.red[bb][warp][lane][0]) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    __syncwarp();
    if (lane == 0) mbar_arrive1(S.wsrf + bb);
#endif
  }
#if OKC_WSB && OKC_WSBL
  if (PB && warp < OKC_WSBLK && warp < nU) {  // the CTA's last units: epilogues here, not behind the helper's queue
    const int ul = nU - 1 - warp;
    const int q = seq0 + ul, bb = q % NRB;
    const int v = v0 + ul, j = njF + v / RB_B, rb = v - (v / RB_B) * RB_B;
    const int gg = lane >> 2, tt = lane & 3;
    float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int w = 0; w < NW; ++w) {
      const float4 t = *reinterpret_cast<const float4*>(&S.red[bb][w][lane][0]);
      sm.x += t.x; sm.y += t.y; sm.z += t.z; sm.w += t.w;
    }
    const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
    const int n0 = rb * 16 + gg;
#if OKC_WSBL == 2
    const float gsc = lgs * TWO62;
    const float nw0 = 1.f + __uint_as_float(lnw0 << 16), nw1 = 1.f + __uint_as_float(lnw1 << 16);
#else
    (void)lgs; (void)lnw0; (void)lnw1;
    const float gsc = p.gsB[ce] * TWO62;
    const float nw0 = 1.f + __bfloat162float(p.normw[n0]), nw1 = 1.f + __bfloat162float(p.normw[n0 + 8]);
#endif
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = 2 * tt + h;
      const float v0f = (h ? sm.y : sm.x) * gsc * nw0, v1f = (h ? sm.w : sm.z) * gsc * nw1;
      float sq = v0f * v0f + v1f * v1f;
      if (tk < cnt) {
        st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
        st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
      }
      sq += __shfl_xor_sync(0xffffffffu, sq, 4);
      sq += __shfl_xor_sync(0xffffffffu, sq, 8);
      sq += __shfl_xor_sync(0xffffffffu, sq, 16);
      if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
    }
  }
#endif
  seq = seq0 + nU;
}

__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring) {
  const int sl = q % OKC_NS;
  if (q < nA) ring_issue<false>(p, S, a0 + q, ring + sl * UNIT_BYTES, S.bars + sl);
  else ring_issue<true>(p, S, b0 + (q - nA), ring + sl * UNIT_BYTES, S.bars + sl);
}

__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {
  const int lane = threadIdx.x & 31;
  WS_JOIN();  // routing tables and this CTA's ranges are ready
  const int a0 = S.wsr[0], a1 = S.wsr[1], b0 = S.wsr[2], b1 = S.wsr[3];
  const int nA = a1 - a0, N = nA + (b1 - b0);
  const int njF = S.njF;
  if (lane == 0)
    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);
  const int gg = lane >> 2, tt = lane & 3;
  const int Nh = (OKC_WSB && OKC_WSBL) ? N - min(OKC_WSBLK, b1 - b0) : N;
  for (int q = 0; q < Nh; ++q) {
#if !OKC_NOLOAD
    if (!OKC_WSRF && !OKC_WSB && lane == 0 && q + OKC_NS < N) {  // unit q + NS goes into unit q's slot once every compute warp released it
      mbar_wait(S.wse + q % OKC_NS, (uint32_t)((q / OKC_NS) & 1));
      ws_issue(p, S, q + OKC_NS, nA, a0, b0, ring);
    }
#endif
    const bool PB = q >= nA;
    const int v = PB ? b0 + (q - nA) : a0 + q;
    const int rbs = PB ? RB_B : RB_A;
    const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;
    const int bb = q % NRB;
    mbar_wait(S.wsrf + bb, (uint32_t)((q / NRB) & 1));
    float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int w = 0; w < NW; ++w) {
      const float4 t = *reinterpret_cast<const float4*>(&S.red[bb][w][lane][0]);
      sm.x += t.x; sm.y += t.y; sm.z += t.z; sm.w += t.w;
    }
    __syncwarp();
    if (lane == 0) mbar_arrive1(S.wsre + bb);  // buffer free again
    // sm: (row gg, token 2tt), (row gg, token 2tt+1), (row gg+8, token 2tt), (row gg+8, token 2tt+1)
    const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
    if (!PB) {
      const float gsc = p.gsA[ce] * TWO62;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tk = 2 * tt + h;
        if (tk < cnt) {
          const float gsct = gsc * RS_T(S.ent[st + tk] >> 3);  // (OKC_ROUTE_LOGITS: per-token pre-expert RMS scale)
          const float gt = (h ? sm.y : sm.x) * gsct, up = (h ? sm.w : sm.z) * gsct;
          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));
        }
      }
      if (q == nA - 1) {  // this CTA's phase A is done: credit its units to the jobs' counters
        __threadfence();
        __syncwarp();
        if (lane == 0) {
          const int jf = njF + a0 / RB_A, jl = njF + (a1 - 1) / RB_A;
          for (int jj = jf; jj <= jl; ++jj) {
            const int a = max(a0, (jj - njF) * RB_A), b = min(a1, (jj - njF + 1) * RB_A);
            atomicAdd(p.doneA + jj, b - a);
          }
        }
      }
    } else {
      const float gsc = p.gsB[ce] * TWO62;
      const int n0 = rb * 16 + gg;
      const float nw0 = 1.f + __bfloat162float(p.normw[n0]), nw1 = 1.f + __bfloat162float(p.normw[n0 + 8]);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tk = 2 * tt + h;
        const float v0f = (h ? sm.y : sm.x) * gsc * nw0, v1f = (h ? sm.w : sm.z) * gsc * nw1;
        float sq = v0f * v0f + v1f * v1f;
        if (tk < cnt) {
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
        }
        sq += __shfl_xor_sync(0xffffffffu, sq, 4);
        sq += __shfl_xor_sync(0xffffffffu, sq, 8);
        sq += __shfl_xor_sync(0xffffffffu, sq, 16);
        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
      }
    }
  }
  __threadfence();
  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier
}
#endif

#if OKC_WSDYN
// ---------------------------------------------------------------- WSDYN: dynamic unit claiming on the WS ring
// A CTA's unit sequence q = 0, 1, ... is: its static phase-A slice [wsr[0], wsr[1]), then phase-A pool units claimed
// from p.ctr[0] (pool = [wsd[0], wsd[1])), then its static phase-B slice, then phase-B pool units (p.ctr[1]). The
// helper warp's lane 0 owns the sequence: it claims (one atomic kept in flight), publishes S.uid[q & 7] and issues the
// TMA into slot q % NS as soon as every compute warp holds unit q's codes (S.wse). The compute warps follow S.uid and
// never issue. Phase-A credits to doneA[j] come from the helper (per-job counts in S.dync) once its last phase-A
// epilogue is done. The last sequence entry is WSD_END.
static_assert(!OKC_WSDYN || (OKC_WS && OKC_WSB && OKC_NW == 8 && OKC_SKT == 8 && OKC_FAST1 > 0 && OKC_TINY), "OKC_WSDYN: WSB tiny build");
constexpr int WSD_END = 0x7FFF;
__device__ __forceinline__ int wsd_uid_read(const Smem& S, int q) {
  const volatile int* u = &S.uid[q & 7];
  int x;
  while (((x = *u) >> 16) != q) __nanosleep(20);
  return x & 0xFFFF;
}
__device__ __forceinline__ void wsd_publish(Smem& S, int q, int pb, int v) {
  *reinterpret_cast<volatile int*>(&S.uid[q & 7]) = (q << 16) | (pb << 15) | v;
}
#ifndef OKC_WSDYNC
#define OKC_WSDYNC 1  // pool units per claim
#endif
#ifndef OKC_WSDYNLATE
#define OKC_WSDYNLATE 0  // dev: release the ring slot after the compute (as the static WSB loop) instead of after the loads
#endif
struct WsdSrc {       // the helper lane's claim state
  int ph;             // 0 static A, 1 pool A, 2 static B, 3 pool B, 4 done
  int vnext, vend;    // static cursor (also the current claimed chunk in the pool phases)
  int pend;           // pool claim in flight (-1: none)
  __device__ __forceinline__ bool next(const Params& p, const Smem& S, int& pb, int& v) {
    for (;;) {
      if (ph == 0 || ph == 2) {
        if (vnext < vend) { pb = ph >> 1; v = vnext++; return true; }
        ++ph;
        pend = -1;
        vnext = vend = 0;
      } else if (ph == 1 || ph == 3) {
        if (vnext < vend) { pb = ph >> 1; v = vnext++; return true; }  // rest of the claimed chunk
        const int base = S.wsd[ph == 1 ? 0 : 2], end = S.wsd[ph == 1 ? 1 : 3];
        if (base < end) {
          int* ctr = p.ctr + (ph >> 1);
          const int cl = pend >= 0 ? pend : atomicAdd(ctr, 1);
          pend = atomicAdd(ctr, 1);  // the next claim: its value is first needed one chunk later
          const int a = base + cl * OKC_WSDYNC;
          if (a < end) { pb = ph >> 1; v = a; vnext = a + 1; vend = min(end, a + OKC_WSDYNC); return true; }
        }
        ++ph;
        if (ph == 2) { vnext = S.wsr[2]; vend = S.wsr[3]; }
      } else {
        return false;
      }
    }
  }
};
__device__ __forceinline__ void wsd_issue(const Params& p, Smem& S, int q, int pb, int v, unsigned char* ring) {
  const int sl = q % OKC_NS;
  wsd_publish(S, q, pb, v);
  if (pb) ring_issue<true>(p, S, v, ring + sl * UNIT_BYTES, S.bars + sl);
  else ring_issue<false>(p, S, v, ring + sl * UNIT_BYTES, S.bars + sl);
}
// compute warps: follow the published sequence; per unit: x for the job (registers), wait the slot, codes to registers,
// release the slot, 32 MMAs, partials to the helper
__device__ void skinny_ring_wsd(const Params& p, Smem& S, unsigned char* ring) {
  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = 16u + p.zero;  // opaque 16 (scale decode IMAD)
  (void)k16;
  const int njF = S.njF;
  uint4 x0[2 * NCH];
  int curj = -1;
  bool curpb = false;
  for (int q = 0;; ++q) {
    const int u = wsd_uid_read(S, q);
    if ((u & 0x7FFF) == WSD_END) break;
    const bool PB = (u >> 15) & 1;
    const int v = u & 0x7FFF;
    const int j = njF + (PB ? (v >> RB_B_LOG) : (v >> RB_A_LOG));
    if (j != curj || PB != curpb) {
      curj = j; curpb = PB;
      if (PB) {
        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
        __syncwarp();
        load_x_regs<true>(p, S, j, lane, warp, x0);
      } else {
        load_x_regs<false>(p, S, j, lane, warp, x0);
      }
    }
    const int sl = q % OKC_NS;
    mbar_wait(S.bars + sl, (uint32_t)((q / OKC_NS) & 1));
    const unsigned char* slot = ring + sl * UNIT_BYTES;
    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);
    uint4 wq[NCH];
#pragma unroll
    for (int c = 0; c < NCH; ++c) wq[c] = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);
#if !OKC_WSDYNLATE
    __syncwarp();
    if (lane == 0) mbar_arrive1(S.wse + sl);  // codes and scales in registers: the helper may refill the slot
#endif
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
#if OKC_NACC == 4
    float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};
#endif
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      const uint4 w = wq[c];
#if OKC_PRMTSC
      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);
      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);
#else
      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);
      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];
#endif
      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,
                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};
#pragma unroll
      for (int st = 0; st < 4; ++st) {
        uint32_t r0, r1, r2, r3;
        dq4(ww[st], r0, r1, r2, r3);
#if OKC_NACC == 4
        mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, bmul2(r0, sg), bmul2(r1, sg8), bmul2(r2, sg), bmul2(r3, sg8), b[2 * st], b[2 * st + 1]);
#else
        mma_bf16((st & 1) ? acc2 : acc, bmul2(r0, sg), bmul2(r1, sg8), bmul2(r2, sg), bmul2(r3, sg8), b[2 * st], b[2 * st + 1]);
#endif
      }
    }
#if OKC_NACC == 4
#pragma unroll
    for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];
#else
    acc[0] += acc2[0]; acc[1] += acc2[1]; acc[2] += acc2[2]; acc[3] += acc2[3];
#endif
#if OKC_WSDYNLATE
    __syncwarp();
    if (lane == 0) mbar_arrive1(S.wse + sl);  // this warp is done with the slot
#endif
    const int bb = q % NRB;
    mbar_wait(S.wsre + bb, (uint32_t)(((q / NRB) & 1) ^ 1));  // the helper has consumed this buffer's previous unit
    *reinterpret_cast<float4*>(&S.red[bb][warp][lane][0]) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    CTA_SYNC();  // partials visible
    if (tid == 0) mbar_arrive1(S.wsrf + bb);
  }
}
// helper warp: claims and issues the sequence (lane 0), runs every epilogue, credits doneA
__device__ void ws_helper_d(const Params& p, Smem& S, unsigned char* ring) {
  const int lane = threadIdx.x & 31;
  WS_JOIN();  // routing tables, ranges and pool bounds are ready
  const int njF = S.njF;
  WsdSrc src{0, S.wsr[0] + S.preissued, S.wsr[1], -1};
  bool ended = false;
  if (lane == 0) {
    for (int q = S.preissued; q < OKC_NS && !ended; ++q) {
      int pb, v;
      if (src.next(p, S, pb, v)) wsd_issue(p, S, q, pb, v, ring);
      else { wsd_publish(S, q, 0, WSD_END); ended = true; }
    }
  }
  const int gg = lane >> 2, tt = lane & 3;
  bool credited = false;
  auto credit = [&]() {  // this CTA's phase-A units, per job, into doneA (after its epilogue stores)
    __threadfence();
    __syncwarp();
    for (int jj = lane; jj < S.njS; jj += 32) {
      const int k = S.dync[njF + jj];
      if (k) atomicAdd(p.doneA + njF + jj, k);
    }
    credited = true;
  };
  for (int q = 0;; ++q) {
    const int u = wsd_uid_read(S, q);
    if ((u & 0x7FFF) == WSD_END) break;
    const bool PB = (u >> 15) & 1;
    const int v = u & 0x7FFF;
    if (lane == 0 && !ended) {  // unit q + NS into unit q's slot once every compute warp holds unit q's codes
      mbar_wait(S.wse + q % OKC_NS, (uint32_t)((q / OKC_NS) & 1));
      int pb, v2;
      if (src.next(p, S, pb, v2)) wsd_issue(p, S, q + OKC_NS, pb, v2, ring);
      else { wsd_publish(S, q + OKC_NS, 0, WSD_END); ended = true; }
    }
    const int rbs = PB ? RB_B : RB_A;
    const int j = njF + (PB ? (v >> RB_B_LOG) : (v >> RB_A_LOG)), rb = v & (rbs - 1);
    const int bb = q % NRB;
    mbar_wait(S.wsrf + bb, (uint32_t)((q / NRB) & 1));
    float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int w = 0; w < NW; ++w) {
      const float4 t = *reinterpret_cast<const float4*>(&S.red[bb][w][lane][0]);
      sm.x += t.x; sm.y += t.y; sm.z += t.z; sm.w += t.w;
    }
    __syncwarp();
    if (lane == 0) mbar_arrive1(S.wsre + bb);  // buffer free again
    const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
    if (!PB) {
      const float gsc = p.gsA[ce] * TWO62;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tk = 2 * tt + h;
        if (tk < cnt) {
          const float gsct = gsc * RS_T(S.ent[st + tk] >> 3);
          const float gt = (h ? sm.y : sm.x) * gsct, up = (h ? sm.w : sm.z) * gsct;
          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));
        }
      }
      if (lane == 0) S.dync[j] += 1;
      if (!credited) {  // last phase-A unit of this CTA (the next entry is phase B or the end): credit now, before
        const int un = wsd_uid_read(S, q + 1);  // the compute warps block on doneA for the first phase-B unit
        if ((un & 0x7FFF) == WSD_END || ((un >> 15) & 1)) credit();
      }
    } else {
      const float gsc = p.gsB[ce] * TWO62;
      const int n0 = rb * 16 + gg;
      const float nw0 = 1.f + __bfloat162float(p.normw[n0]), nw1 = 1.f + __bfloat162float(p.normw[n0 + 8]);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tk = 2 * tt + h;
        const float v0f = (h ? sm.y : sm.x) * gsc * nw0, v1f = (h ? sm.w : sm.z) * gsc * nw1;
        float sq = v0f * v0f + v1f * v1f;
        if (tk < cnt) {
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
        }
        sq += __shfl_xor_sync(0xffffffffu, sq, 4);
        sq += __shfl_xor_sync(0xffffffffu, sq, 8);
        sq += __shfl_xor_sync(0xffffffffu, sq, 16);
        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
      }
    }
  }
  if (!credited) credit();
  __threadfence();
  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier
}
#endif

template <bool BIG, bool PB>
__device__ __forceinline__ void skinny_split(int nS, int nF, int G, int c, int& v0, int& v1) {
  const float rG = __fdividef(1.f, (float)G);
  const int q = muldiv_g(nF, 1, G, rG), r = nF - q * G;
  auto fcost = [](long long k) -> long long {
    // phase-specific cost of one fat unit in skinny-unit equivalents (phase B's skinny units are slower, so its ratio is lower)
    const int fc = BIG ? (PB ? OKC_FCOSTBB : OKC_FCOSTB) : (PB ? OKC_FCOSTSB : OKC_FCOST);
    return k > 0 ? (long long)fc * k : 0;
  };
  const long long T = (long long)nS + (long long)r * fcost(q + 1) + (long long)(G - r) * fcost(q);  // total cost = G x level
  long long wA = T - (long long)G * fcost(q + 1); if (wA < 0) wA = 0;
  long long wB = T - (long long)G * fcost(q); if (wB < 0) wB = 0;
  auto W = [&](int cc) -> long long { return cc <= r ? (long long)cc * wA : (long long)r * wA + (long long)(cc - r) * wB; };
  const long long WG = W(G);
  if (nF > 0 && WG > 0) {
    // split point of CTA x: floor(nS * W(x) / WG) in float (one formula for both neighbours, so the shares tile [0, nS))
    const float sc = __fdividef((float)nS, (float)WG);
    auto split = [&](int x) -> int {
      if (x >= G) return nS;
      const int v = __float2int_rz((float)W(x) * sc);
      return v < nS ? v : nS;
    };
    v0 = split(c); v1 = split(c + 1);
  } else {
    v0 = muldiv_g(nS, c, G, rG); v1 = muldiv_g(nS, c + 1, G, rG);
  }
}

#if OKC_WGS
// ---------------------------------------------------------------- WGS: RS-wgmma for every job (row split, stream-K)
// Unit = (job, 256-row block): phase A 32 per job, phase B 16, each 16 stages of 256 K. Stage = 36 KB of weights,
// for each warpgroup q its 64 rows: 8 KB codes [grp][warp][lane][4 words] (natural mma k order) + 1 KB scales
// [grp][warp][g][h][j]; the stage's x slice (the job's token rows, rounded up to 8, x 256 K as wgmma B core matrices
// [n8][32 k8][8 rows][16 B]) sits in its own ring. Warpgroup q accumulates its 64 rows over K with wgmma m64nNk16
// (A = dequantized codes in registers, N = 8 * n-tiles) and runs its own epilogue (rows g / g + 8 are gate / up).
// Work is split by cost-weighted stage ranges (stream-K): a unit cut between two CTAs adds its partial sums into a
// global buffer (indexed by the CTA boundary) and the second contributor runs the epilogue.
#ifndef OKC_WGSNX
#define OKC_WGSNX 3                             // x ring slots
#endif
#ifndef OKC_WGSNS
#define OKC_WGSNS 3
#endif
#ifndef OKC_WGSEARLY
#define OKC_WGSEARLY 0  // WGS: the producers start right after routing (+ ranges), not after the consumers' x gather
#endif
constexpr int WGS_NS = OKC_WGSNS;               // weight ring slots (36 KB; freed as soon as the codes are in registers)
constexpr int WGS_NX = OKC_WGSNX;               // x ring slots (freed when the stage's last wgmma retired)
constexpr int WGS_NT = FT_T / 8;                // max n-tiles of a job
constexpr int WGS_PIECE = 9216;                 // one warpgroup's weights in a stage
constexpr int WGS_XS = WGS_NT * 4096;           // an x slot: up to FT_T rows x 256 K
constexpr int BW_F = 0, BW_E = 3, BX_F = 6, BX_E = 6 + WGS_NX;  // barrier indices
static_assert(BX_E + WGS_NX <= 24, "WGS barriers");
#ifndef OKC_WGSDYN
#define OKC_WGSDYN 0  // WGS: dynamic stage claiming: cost-weighted static ranges over the first (1 - 2^-WGSDYNS) of
                      // each phase's cost, then OKC_WGSSEG-stage segments claimed from a global counter by the weights
                      // producer, which publishes the (unit, stage range) pieces for the consumers and the x producer;
                      // cut units use unit-indexed fixup slots and the last-arriving share runs the epilogue
#endif
#ifndef OKC_WGSDYNS
#define OKC_WGSDYNS 2
#endif
#ifndef OKC_WGSSEG
#define OKC_WGSSEG 4
#endif
constexpr int WGS_SLOTS = OKC_WGSDYN ? JMAX * 32 : 160;  // fixup slots per phase (>= grid + 1 | one per unit)
constexpr int WGS_FIXW = WGS_NT * 512;          // floats per (slot, warpgroup): [n-tile][128 threads][4]
constexpr size_t WGS_FIXF = (size_t)2 * WGS_SLOTS * 4 * WGS_FIXW + 2 * WGS_SLOTS * 4 + 136 * 16;  // + doneAs
__device__ __forceinline__ float* wgs_fix(const Params& p, int pb) {
  return p.ybuf + (size_t)EMAX * HD + (size_t)pb * WGS_SLOTS * 4 * WGS_FIXW;
}
__device__ __forceinline__ int* wgs_cnt(const Params& p, int pb) {
  return reinterpret_cast<int*>(p.ybuf + (size_t)EMAX * HD + (size_t)2 * WGS_SLOTS * 4 * WGS_FIXW) + pb * WGS_SLOTS * 4;
}
// per-stage cost by n-tiles (1..8), phase A and phase B (relative units, <= 255; fit on per-CTA phase times at
// bs 96..256, t_fit2.py): tuner knobs OKC_WGSA / OKC_WGSB = 8 comma-separated values each
#ifndef OKC_WGSA
#define OKC_WGSA 112, 135, 140, 158, 173, 185, 200, 207  // tuner combo_m798 on v182x (Oct 26: bs32 +0.45, bs128 +0.94, bs256 +0.17%)
#endif
#ifndef OKC_WGSB
#define OKC_WGSB 100, 104, 115, 123, 132, 144, 156, 174
#endif
constexpr unsigned long long wgs_pack8(const int (&c)[8]) {
  unsigned long long r = 0;
  for (int i = 0; i < 8; ++i) r |= (unsigned long long)(c[i] & 255) << (8 * i);
  return r;
}
constexpr int WGS_CA[8] = {OKC_WGSA};
constexpr int WGS_CB[8] = {OKC_WGSB};
constexpr unsigned long long WGS_PA = wgs_pack8(WGS_CA), WGS_PB = wgs_pack8(WGS_CB);
static_assert(WGS_CA[0] > 0 && WGS_CA[7] < 256 && WGS_CB[0] > 0 && WGS_CB[7] < 256, "WGS cost tables: 1..255");
__device__ __forceinline__ int wgs_cost(int pb, int cnt) {
  return (int)(((pb ? WGS_PB : WGS_PA) >> (8 * (((cnt + 7) >> 3) - 1))) & 255ull);
}
#ifndef OKC_WGSPROF
#define OKC_WGSPROF 0  // dev: per-warpgroup cycle counters (drain, fixup, epilogue) in p.dbg after the timeline slots
#endif
#ifndef OKC_WGSREORD
#define OKC_WGSREORD 1  // WGS: phase A starts with the trailing partial unit (see wgs_split)
#endif
#ifndef OKC_GATHB
#define OKC_GATHB 8  // WGS: x rows gathered per batch (loads first, then stores); 0 = one row at a time
#endif
#ifndef OKC_WGSPR
#define OKC_WGSPR 1  // WGS: the x producer warp computes the stage ranges while the consumers gather
#endif
#if OKC_WGSDYN
#undef OKC_WGSDFIN
#define OKC_WGSDFIN 3  // (shares are claimed in stage order: the share holding the last stage finishes, both phases)
#endif
#ifndef OKC_WGSDFIN
#define OKC_WGSDFIN 2  // WGS: stream-K split units finished by a designated share (see wgs_flush): bit0 phase A, bit1 phase B
#endif
#ifndef OKC_WGSFCI
#define OKC_WGSFCI 3
#endif
#ifndef OKC_WGSMMAN
#define OKC_WGSMMAN 0  // WGS: jobs with <= this many n-tiles run on per-warp mma.sync (wgs_seg_mma) instead of wgmma
#endif
#ifndef OKC_WGSMMANA
#define OKC_WGSMMANA 4  // wgs_seg_mma: accumulator chains for 1-n-tile jobs
#endif
#ifndef OKC_WGSGS4N
#define OKC_WGSGS4N 0  // WGS: jobs with <= this many n-tiles commit 4 k16 steps per wgmma group (small accumulators)
#endif
#ifndef OKC_WGSWD
#define OKC_WGSWD 1    // WGS: wgmma groups kept in flight (wait_group depth); OKC_WGSWD + 1 A-fragment buffers
#endif
#ifndef OKC_XNOSC
#define OKC_XNOSC 0    // dev: WGS without the block-scale multiplies (timing only)
#endif
#ifndef OKC_WGSGS
#define OKC_WGSGS 2  // WGS: k16 steps per wgmma commit group (4 serializes the N=16..32 variants at 96 registers: C7512)
#endif
#ifndef OKC_WGSFIT
#define OKC_WGSFIT 0  // dev (debug builds): dbg slots 6/7 = per-phase stage counts by n-tiles (else: ranges / gather times)
#endif
#ifndef OKC_WGSDBG
#define OKC_WGSDBG 0  // dev: WGS waits time out with a printf + trap (tag says which wait)
#endif
__device__ __forceinline__ void wgs_wait(uint64_t* bar, uint32_t parity, int tag) {
#if OKC_WGSDBG
  uint32_t ok = 0;
  for (long long it = 0; it < (1ll << 22); ++it) {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(smem_u32(bar)), "r"(parity) : "memory");
    if (ok) return;
  }
  printf("WGS hang: block %d thread %d tag %d parity %u\n", (int)blockIdx.x, (int)threadIdx.x, tag, parity);
  __trap();
#else
  (void)tag;
  mbar_wait(bar, parity);
#endif
}
template <int NTT> struct WGSMMA;
template <> struct WGSMMA<1> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !(OKC_WGSX & 1)
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
#endif
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !(OKC_WGSX & 1)
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
#endif
  }
};
template <> struct WGSMMA<2> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !(OKC_WGSX & 1)
    asm volatile("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
#endif
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !(OKC_WGSX & 1)
    asm volatile("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
#endif
  }
};
template <> struct WGSMMA<3> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !(OKC_WGSX & 1)
    asm volatile("wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
#endif
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !(OKC_WGSX & 1)
    asm volatile("wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
#endif
  }
};
template <> struct WGSMMA<4> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !(OKC_WGSX & 1)
    asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
#endif
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !(OKC_WGSX & 1)
    asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
#endif
  }
};
template <> struct WGSMMA<5> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !(OKC_WGSX & 1)
    asm volatile("wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
#endif
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !(OKC_WGSX & 1)
    asm volatile("wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
#endif
  }
};
template <> struct WGSMMA<6> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !(OKC_WGSX & 1)
    asm volatile("wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]), "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
#endif
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !(OKC_WGSX & 1)
    asm volatile("wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
#endif
  }
};
template <> struct WGSMMA<7> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !(OKC_WGSX & 1)
    asm volatile("wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]), "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3]), "=f"(d[6][0]), "=f"(d[6][1]), "=f"(d[6][2]), "=f"(d[6][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
#endif
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !(OKC_WGSX & 1)
    asm volatile("wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
#endif
  }
};
template <> struct WGSMMA<8> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !(OKC_WGSX & 1)
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]), "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3]), "=f"(d[6][0]), "=f"(d[6][1]), "=f"(d[6][2]), "=f"(d[6][3]), "=f"(d[7][0]), "=f"(d[7][1]), "=f"(d[7][2]), "=f"(d[7][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
#endif
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !(OKC_WGSX & 1)
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
#endif
  }
};
// two producer warps: the weights warp streams every stage's 36 KB through the weight ring (TMA, lane 0), the x warp
// gathers every stage's x slice into the x ring (cp.async) and, in phase B, first waits for the job's phase-A units;
// so the weights run ahead by the whole ring even while x waits
// Phase A runs a CTA's trailing partial unit FIRST, then the rest of its range in order: the CTA that holds a unit's
// first stages otherwise reaches them last, finishes the unit (read back + epilogue) and pays it exposed at the phase
// end. Returns where the first sub-range starts: [sp, g1) then [g0, sp) (sp == g0: no reordering). Phase B keeps the
// order (its first stages wait for phase A's units in order).
__device__ __forceinline__ int wgs_split(int pb, int g0, int g1) {
  const int ue = ((g1 - 1) >> 4) << 4;  // first stage of the unit holding the range's last stage
  return (OKC_WGSREORD && pb == 0 && (g1 & 15) != 0 && ue > g0) ? ue : g0;
}
__device__ void wgs_ranges(const Params& p, Smem& S);  // below (needs the cost tables)
#if OKC_WGSDYN
// piece ring: entry k = (k & 0xFFF) << 20 | u << 8 | s0 << 4 | (s1 - 1); the end of a phase is u = 0xFFF
constexpr int WGS_PEND = 0xFFF00;
__device__ __forceinline__ void wgs_pub(Smem& S, int k, int u, int s0, int s1) {
  *reinterpret_cast<volatile uint32_t*>(&S.wseg[k & 15]) = ((uint32_t)(k & 0xFFF) << 20) | ((uint32_t)u << 8) | ((uint32_t)s0 << 4) | (uint32_t)(s1 - 1);
}
__device__ __forceinline__ int wgs_piece(const Smem& S, int k) {
  const volatile uint32_t* e = reinterpret_cast<const volatile uint32_t*>(&S.wseg[k & 15]);
  uint32_t x;
  while (((x = *e) >> 20) != (uint32_t)(k & 0xFFF)) __nanosleep(20);
  return (int)(x & 0xFFFFFu);
}
// weights producer: walks the static ranges, then claims pool segments; publishes every (unit, s0, s1) piece before
// issuing its stages
__device__ void wgs_wproducer(const Params& p, Smem& S, unsigned char* ring) {
#if OKC_WGSEARLY
  asm volatile("bar.sync 7, %0;\n" ::"n"(NT + 64) : "memory");  // routing tables
  asm volatile("bar.sync 8, 64;\n" ::: "memory");               // ranges (from the x producer warp)
#else
  WS_JOIN();  // routing tables and this CTA's stage ranges are ready
#endif
  if ((threadIdx.x & 31) != 0) return;
  int ws = 0, n = 0, k = 0;
  uint32_t wph = 0;
  unsigned long long wring_wait = 0;
  auto stage = [&](int pb, int gs) {
    const int ubl = pb ? 4 : 5;
    const unsigned char* base = pb ? p.wBs : p.wAs;
    const int u = gs >> 4, st = gs & 15;
    const int j = u >> ubl, ub = u & ((1 << ubl) - 1);
    if (n >= WGS_NS) {
      unsigned long long tr0 = 0;
      if (OKC_DEBUG) tr0 = gtimer();
      wgs_wait(S.bars + BW_E + ws, wph ^ 1, 1000 + gs);
      if (OKC_DEBUG) wring_wait += gtimer() - tr0;
    }
    uint64_t* bar = S.bars + BW_F + ws;
    const unsigned char* src = base + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + st)) * (size_t)UNIT_BYTES;
#if OKC_WGSX & 32
    (void)src;
    mbar_arrive1(bar);
#else
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
                 ::"r"(smem_u32(ring + ws * UNIT_BYTES)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");
#endif
    if (++ws == WGS_NS) { ws = 0; wph ^= 1; }
    ++n;
  };
  auto range = [&](int pb, int a, int b) {  // stages [a, b) as unit pieces
    for (int gs = a; gs < b;) {
      const int u = gs >> 4, s0 = gs & 15, s1 = min(16, s0 + (b - gs));
      wgs_pub(S, k++, u, s0, s1);
      for (int st = s0; st < s1; ++st) stage(pb, (u << 4) + st);
      gs += s1 - s0;
    }
  };
#pragma unroll 1
  for (int pb = 0; pb < 2; ++pb) {
    const int g0 = S.wgr[2 * pb], g1 = S.wgr[2 * pb + 1], sp = wgs_split(pb, g0, g1);
    range(pb, sp, g1);
    if (sp > g0) range(pb, g0, sp);
    const int Ts = S.wgd[2 * pb], T = S.wgd[2 * pb + 1];
    if (Ts < T) {
      int* ctr = p.ctr + pb;
      int pend = atomicAdd(ctr, 1);
      for (;;) {
        const int cl = pend;
        pend = atomicAdd(ctr, 1);  // the next claim, in flight while this segment streams
        const int a = Ts + cl * OKC_WGSSEG;
        if (a >= T) break;
        range(pb, a, min(T, a + OKC_WGSSEG));
      }
    }
    wgs_pub(S, k++, 0xFFF, 0, 1);  // end of phase pb
  }
  if (OKC_DEBUG && p.dbg) p.dbg[(size_t)gridDim.x * 8 + (size_t)gridDim.x * 4 + blockIdx.x] = wring_wait;
}
#else
__device__ void wgs_wproducer(const Params& p, Smem& S, unsigned char* ring) {
#if OKC_WGSEARLY
  asm volatile("bar.sync 7, %0;\n" ::"n"(NT + 64) : "memory");  // routing tables
  asm volatile("bar.sync 8, 64;\n" ::: "memory");               // ranges (from the x producer warp)
#else
  WS_JOIN();  // routing tables and this CTA's stage ranges are ready
#endif
  if ((threadIdx.x & 31) != 0) return;
  int ws = 0, n = 0;
  uint32_t wph = 0;
  unsigned long long wring_wait = 0;
#pragma unroll 1
  for (int pb = 0; pb < 2; ++pb) {
    const int g0 = S.wgr[2 * pb], g1 = S.wgr[2 * pb + 1], sp = wgs_split(pb, g0, g1);
    const int ubl = pb ? 4 : 5;  // log2 of 256-row units per job
    const unsigned char* base = pb ? p.wBs : p.wAs;
#pragma unroll 1
    for (int i = 0, gs = sp; i < g1 - g0; ++i, ++gs, ++n) {
      if (gs == g1) gs = g0;  // [sp, g1) done: then [g0, sp)
      const int u = gs >> 4, st = gs & 15;
      const int j = u >> ubl, ub = u & ((1 << ubl) - 1);
      if (n >= WGS_NS) {
        unsigned long long tr0 = 0;
        if (OKC_DEBUG) tr0 = gtimer();
        wgs_wait(S.bars + BW_E + ws, wph ^ 1, 1000 + gs);
        if (OKC_DEBUG) wring_wait += gtimer() - tr0;
      }
      uint64_t* bar = S.bars + BW_F + ws;
      const unsigned char* src = base + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + st)) * (size_t)UNIT_BYTES;
#if OKC_WGSX & 32
      (void)src;
      mbar_arrive1(bar);  // dev: no weight copy (the consumers decode whatever the slot holds)
#else
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
                   ::"r"(smem_u32(ring + ws * UNIT_BYTES)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");
#endif
      if (++ws == WGS_NS) { ws = 0; wph ^= 1; }
    }
  }
  if (OKC_DEBUG && p.dbg) p.dbg[(size_t)gridDim.x * 8 + (size_t)gridDim.x * 4 + blockIdx.x] = wring_wait;  // weight-ring empty waits (ns)
}
#endif
// x producer: one bulk copy per stage (lane 0): the job's x slice for the stage is contiguous in xA (phase A, written
// by every CTA's gather pass at the start) or xB (phase B, written by the phase-A epilogues), N_pad x 512 B
constexpr int WGS_XROWS = EMAX + 8 * JMAX;  // padded rows of all jobs, upper bound
__device__ void wgs_xproducer(const Params& p, Smem& S, unsigned char* ring) {
#if OKC_WGSPR
  // the routing tables are ready (the consumers arrive on barrier 7 right after routing): this idle warp computes the
  // stage ranges while the consumers gather phase A's x
  asm volatile("bar.sync 7, %0;\n" ::"n"(NT + (OKC_WGSEARLY ? 64 : 32)) : "memory");
  if ((threadIdx.x & 31) == 0) wgs_ranges(p, S);
  __syncwarp();
#endif
#if OKC_WGSEARLY
  asm volatile("bar.arrive 9, %0;\n" ::"n"(NT + 32) : "memory");  // ranges ready for the consumers (they sync after the gather)
  asm volatile("bar.sync 8, 64;\n" ::: "memory");                 // ranges ready: both producers go (the x stages wait on xready)
#else
  WS_JOIN();  // routing tables and this CTA's stage ranges are ready
#endif
  if ((threadIdx.x & 31) != 0) return;
  unsigned char* xring = ring + WGS_NS * UNIT_BYTES;
  int xs = 0, n = 0;
  uint32_t xph = 0;
  unsigned long long xwaitA = 0, xwaitB = 0, xstallB = 0, xring_wait = 0;
#if OKC_WGSDYN
  int k = 0;  // piece sequence (published by the weights producer)
#endif
#pragma unroll 1
  for (int pb = 0; pb < 2; ++pb) {
    const int g0 = S.wgr[2 * pb], g1 = S.wgr[2 * pb + 1], sp = wgs_split(pb, g0, g1);
    const int ubl = pb ? 4 : 5;
    const __nv_bfloat16* xsrc = pb ? p.xB : p.xA;
    int curj = -1, npad = 8;
    const __nv_bfloat16* jb = xsrc;
#if OKC_WGSDYN
    (void)g0; (void)g1; (void)sp;
    int ps0 = 0, ps1 = 0, pu = 0;  // current piece
#pragma unroll 1
    for (int st = 0;; ++st, ++n) {
      if (st >= ps1) {
        const int e = wgs_piece(S, k++);
        if (e == WGS_PEND) break;
        pu = e >> 8; ps0 = (e >> 4) & 15; ps1 = (e & 15) + 1;
        st = ps0;
      }
      const int u = pu;
      const int j = u >> ubl;
      const int gs = (u << 4) + st;
#else
#pragma unroll 1
    for (int i = 0, gs = sp; i < g1 - g0; ++i, ++gs, ++n) {
      if (gs == g1) gs = g0;  // [sp, g1) done: then [g0, sp)
      const int u = gs >> 4, st = gs & 15;
      const int j = u >> ubl;
#endif
      if (n >= WGS_NX) {
        unsigned long long tr0 = 0;
        if (OKC_DEBUG) tr0 = gtimer();
        wgs_wait(S.bars + BX_E + xs, xph ^ 1, 2000000 + gs);
        if (OKC_DEBUG) xring_wait += gtimer() - tr0;
      }
      unsigned long long tw0 = 0;
      if (OKC_DEBUG) tw0 = gtimer();
      if (j != curj) {
        curj = j;
        npad = ((S.jC[j] + 7) >> 3) << 3;
        jb = xsrc + (size_t)S.jrow[j] * HD;
        if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.xready + j) < npad) __nanosleep(32); }  // every row gathered
      }
      // phase B stage st reads inter columns 256 st .. +255: the 2 phase-A units (x 4 warpgroups) that wrote them
      if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.doneAs + j * 16 + st) < 8) __nanosleep(32); }
      if (OKC_DEBUG) { const unsigned long long dt = gtimer() - tw0; if (pb) { xwaitB += dt; xstallB += dt > 500; } else xwaitA += dt; }
      uint64_t* bar = S.bars + BX_F + xs;
      const uint32_t bytes = (uint32_t)npad * 512u;
#if OKC_WGSX & 16
      (void)bytes; (void)jb;
      mbar_arrive1(bar);  // dev: no x copy (the consumers read whatever the slot holds)
#else
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                   ::"r"(smem_u32(xring + xs * WGS_XS)), "l"(jb + (size_t)st * npad * 256), "r"(bytes), "r"(smem_u32(bar)) : "memory");
#endif
      if (++xs == WGS_NX) { xs = 0; xph ^= 1; }
    }
  }
  if (OKC_DEBUG && p.dbg) {  // x producer stalls: dbg[G * 8 + cta * 4 + (0 phase-A gating ns, 1 phase-B gating ns, 2 phase-B stalls > 0.5 us, 3 x-ring empty waits ns)]
    unsigned long long* d = p.dbg + (size_t)gridDim.x * 8 + (size_t)blockIdx.x * 4;
    d[0] = xwaitA; d[1] = xwaitB; d[2] = xstallB; d[3] = xring_wait;
  }
}
// phase A's x gather (all consumer threads of every CTA, once, at the start): this CTA's share of the jobs' padded rows
// copied into xA's per-stage wgmma layout, then counted into xready (padding rows are counted, not written)
__device__ void wgs_gather(const Params& p, Smem& S) {
  const int tid = threadIdx.x, nj = S.njF + S.njS;
  const int R = S.jrow[nj];  // R * G < 2^24: exact float muldiv, no divide routine
  const float rG = __fdividef(1.f, (float)gridDim.x);
  const int q0 = muldiv_g(R, blockIdx.x, gridDim.x, rG), q1 = muldiv_g(R, blockIdx.x + 1, gridDim.x, rG);
  int j = 0;
#if OKC_GATHB
  // all of a batch's row loads first, then the stores: one x row at a time was one dependent HBM round trip per row
  // (the next row's load could not move above the previous row's store: they may alias), 1.3-7 us at bs256
  static_assert(NT * 8 == HD, "one 16 B chunk per thread per row");
  constexpr int GB = OKC_GATHB;
  for (int qb = q0; qb < q1; qb += GB) {
    uint4 v[GB];
    uint4* d[GB];
#pragma unroll
    for (int k = 0; k < GB; ++k) {
      const int q = qb + k;
      d[k] = nullptr;
      if (q < q1) {
        while (S.jrow[j + 1] <= q) ++j;
        const int r = q - S.jrow[j], cnt = S.jC[j];
        if (r < cnt) {
          const int ntt = (cnt + 7) >> 3;
          d[k] = reinterpret_cast<uint4*>(p.xA + (size_t)S.jrow[j] * HD) + ((((tid >> 5) * ntt + (r >> 3)) * 32 + (tid & 31)) * 8 + (r & 7));
          v[k] = __ldg(reinterpret_cast<const uint4*>(p.x + (size_t)(S.ent[S.jS[j] + r] >> 3) * HD) + tid);
        }
      }
    }
#pragma unroll
    for (int k = 0; k < GB; ++k)
      if (d[k]) *d[k] = v[k];
  }
#else
  for (int q = q0; q < q1; ++q) {
    while (S.jrow[j + 1] <= q) ++j;
    const int r = q - S.jrow[j], cnt = S.jC[j];
    if (r < cnt) {
      const int ntt = (cnt + 7) >> 3;
      const uint4* src = reinterpret_cast<const uint4*>(p.x + (size_t)(S.ent[S.jS[j] + r] >> 3) * HD);
      uint4* dst = reinterpret_cast<uint4*>(p.xA + (size_t)S.jrow[j] * HD);
      for (int kc = tid; kc < HD / 8; kc += NT) {  // 16 B chunk kc: stage kc >> 5, k8 kc & 31
        const uint4 v = __ldg(src + kc);
        dst[(((kc >> 5) * ntt + (r >> 3)) * 32 + (kc & 31)) * 8 + (r & 7)] = v;
      }
    }
  }
#endif
  CTA_SYNC();
  if (tid == 0) {  // count this CTA's rows per job (release: the CTA's stores are ordered before by the barrier)
    int jj = 0;
    for (int q = q0; q < q1;) {
      while (S.jrow[jj + 1] <= q) ++jj;
      const int e = min(q1, S.jrow[jj + 1]);
      asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.xready + jj), "r"(e - q) : "memory");
      q = e;
    }
  }
}
// Work left from the previous segment, done after the next segment's first wgmma group (so the stores and atomics it
// waits for have long completed): kind 1 = release doneA for job j (phase A), kind 2 = add the stage count of a split
// unit's share and, if that completes the unit, read the summed partials back and run its epilogue.
struct WgsPend { int kind, u, slot, nst; };
template <bool PB>
__device__ __forceinline__ void wgs_epi_tile(const Params& p, const Smem& S, int j, int r16, int nt, const float* d4) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int cnt = S.jC[j], jst = S.jS[j], e = S.jE[j];
  if (!PB) {
    // SwiGLU straight into xB (phase B's per-stage wgmma B layout) instead of row-major inter
    const int col = r16 * 8 + g, ntt = (cnt + 7) >> 3;
    const float gsc = p.gsA[e] * TWO62;
    __nv_bfloat16* xb = p.xB + (size_t)S.jrow[j] * HD + (col & 7);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = nt * 8 + 2 * t + h;
      if (tk < cnt) {
        const float gsct = gsc * RS_T(S.ent[jst + tk] >> 3);  // (OKC_ROUTE_LOGITS: per-token pre-expert RMS scale)
        const float gt = d4[h] * gsct, up = d4[2 + h] * gsct;
        const float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;
        st_keep_bf16(xb + (size_t)((((col >> 8) * ntt + (tk >> 3)) * 32 + ((col & 255) >> 3)) * 64 + (tk & 7) * 8),
                     __float2bfloat16(probe_q(v)));
      }
    }
  } else {
#if OKC_WGSFINSS
    // y rows only: the per-route sum of squares is formed by the final pass from these rows (no shuffles / atomics)
    const int n0 = r16 * 16 + g, tok0 = nt * 8 + 2 * t;
    const float gsc = p.gsB[e] * TWO62;
    const float w0 = (1.f + __bfloat162float(__ldg(p.normw + n0))) * gsc, w1 = (1.f + __bfloat162float(__ldg(p.normw + n0 + 8))) * gsc;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = tok0 + h;
      if (tk < cnt) {
        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0), d4[h] * w0);
        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0 + 8), d4[2 + h] * w1);
      }
    }
#else
    float sq[2];
    epi_B(p, jst, cnt, nt * 8 + 2 * t, r16 * 16 + g, p.gsB[e] * TWO62, d4, sq);
    ss_reduce_add(p, jst, cnt, nt * 8 + 2 * t, sq, g);
#endif
  }
}
template <bool PB>
__device__ __forceinline__ void wgs_flush(const Params& p, Smem& S, WgsPend& pd) {
  if (pd.kind == 0) return;
  const int tid = threadIdx.x, warp = WARP_IDX(tid), wg = warp >> 2, wp = warp & 3, wt = tid & 127;
  constexpr int UBL = PB ? 4 : 5;
  const int j = pd.u >> UBL, ub = pd.u & ((1 << UBL) - 1);
  asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
  if (pd.kind == 1) {
    // release (cumulative over the warpgroup's epilogue stores ordered by the barrier); pairs with the producer's
    // acquire. Unit ub wrote inter columns ub*128 .. +127: half of phase B stage ub / 2
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");
    pd.kind = 0;
    return;
  }
  int* fc = wgs_cnt(p, PB) + pd.slot * 4 + wg;
  constexpr bool DF = (OKC_WGSDFIN >> (PB ? 1 : 0)) & 1;  // designated finisher in this phase
  if constexpr (DF) {
  // designated finisher (kind 3): the share processed last in time by construction (phase A: the one holding the
  // unit's last stage, run early by the reorder; phase B: the one holding stage 0, at its CTA's end). The others just
  // count with a fire-and-forget release (no atomic round trip, no drain wait); the finisher waits for their counts
  if (pd.kind == 2) {
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(fc), "r"(pd.nst) : "memory");
    pd.kind = 0;
    return;
  }
  if (wt == 0) { while (ld_acquire(fc) < 16 - pd.nst) __nanosleep(32); }
  asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
  pd.kind = 0;
  } else {
  if (wt == 0) {
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(fc), "r"(pd.nst) : "memory");
    S.wglast[wg] = old + pd.nst == 16;
  }
  asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
  const bool last = S.wglast[wg];
  pd.kind = 0;
  if (!last) return;
  }
  asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
  float* fb = wgs_fix(p, PB) + (size_t)(pd.slot * 4 + wg) * WGS_FIXW + wt * 4;
  const int ntt = (S.jC[j] + 7) >> 3, r16 = ub * 16 + wg * 4 + wp;
  // two n-tiles' loads in flight before their zeroing stores and epilogues (the scout): one at a time, each load
  // waits behind the previous tile's stores (they may alias), ntt serialized L2 round trips under full HBM load
  for (int nt0 = 0; nt0 < ntt; nt0 += 2) {
    float4 v[2];
#pragma unroll
    for (int k = 0; k < 2; ++k)
      if (nt0 + k < ntt) v[k] = __ldcg(reinterpret_cast<const float4*>(fb + (nt0 + k) * 512));
#pragma unroll
    for (int k = 0; k < 2; ++k)
      if (nt0 + k < ntt) {
        *reinterpret_cast<float4*>(fb + (nt0 + k) * 512) = make_float4(0.f, 0.f, 0.f, 0.f);
        const float d4[4] = {v[k].x, v[k].y, v[k].z, v[k].w};
        wgs_epi_tile<PB>(p, S, j, r16, nt0 + k, d4);
      }
  }
  if (wt == 0) *fc = 0;
  if (!PB) {
    asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");
  }
}
// one segment (stages [s0, s1) of unit u) on NTT n-tiles: accumulate, then finish (whole unit) or hand off (split)
template <bool PB, int NTT>
__device__ __forceinline__ void wgs_seg(const Params& p, Smem& S, unsigned char* ring, int u, int s0, int s1, int slot,
                                        int& seq, WgsPend& pd) {
  const int tid = threadIdx.x, lane = tid & 31;
  const int warp = WARP_IDX(tid);
  const int wg = warp >> 2, wp = warp & 3, g = lane >> 2, wt = tid & 127;
  constexpr int UBL = PB ? 4 : 5;
  const uint32_t k16c = 16u + p.zero;  // opaque 16: the scale decode's multiply stays an IMAD (FMA pipe)
  const uint32_t xring_s = smem_u32(ring + WGS_NS * UNIT_BYTES);
  float d[WGS_NT][4];  // written first by a scale-d = 0 wgmma (no zeroing: ptxas would serialize the wgmmas, C7513)
  // k16 steps per commit group: the whole function shares one register budget (96 at 544 threads) and ptxas
  // serializes every wgmma if any pipeline does not fit (C7512), so the wide-N variants commit 2 steps at a time
  constexpr int GS = (NTT <= OKC_WGSGS4N) ? 4 : OKC_WGSGS;  // 4 steps per commit group where the accumulators are small
  constexpr int NB = OKC_WGSWD + 1;                           // A-fragment buffers: OKC_WGSWD groups stay in flight
  uint32_t af[NB][GS][4];
  int prev = -1;  // x slot the last committed group may still read
#if OKC_WGSDESC0
  // slot-0 descriptor and code pointer, once per segment: a stage adds its slot's offset (the address field cannot
  // carry: shared addresses >> 4 stay below 2^14) instead of rebuilding the descriptor (~8 uniform ops per stage)
  const uint64_t xd0 = wg_desc(xring_s, 128, 4096);
  const unsigned char* pc0 = ring + wg * WGS_PIECE;
#endif
  auto stage = [&](auto first) {
    constexpr bool FIRST = decltype(first)::value;
    const int ws = seq % WGS_NS, xs = seq % WGS_NX;
    wgs_wait(S.bars + BW_F + ws, (uint32_t)(seq / WGS_NS) & 1u, 3000000 + seq);
#if OKC_WGSDESC0
    const unsigned char* pc = pc0 + ws * UNIT_BYTES;
#else
    const unsigned char* pc = ring + ws * UNIT_BYTES + wg * WGS_PIECE;
#endif
    wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);
#if OKC_WGSDESC0
    const uint64_t xd = xd0 + (uint64_t)(uint32_t)(xs * (WGS_XS >> 4));
#else
    const uint64_t xd = wg_desc(xring_s + xs * WGS_XS, 128, 4096);
#endif
    // codes and scales one 4-step group at a time (fewer live registers)
    uint4 cwn = *reinterpret_cast<const uint4*>(pc + (wp * 32 + lane) * 16);
    uint2 scn = *reinterpret_cast<const uint2*>(pc + 8192 + (wp * 8 + g) * 8);
#pragma unroll
    for (int lg = 0; lg < 4; ++lg) {
      const uint4 cwg = cwn;
      const uint2 scg = scn;
      if (lg < 3) {
        cwn = *reinterpret_cast<const uint4*>(pc + (((lg + 1) * 4 + wp) * 32 + lane) * 16);
        scn = *reinterpret_cast<const uint2*>(pc + 8192 + (((lg + 1) * 4 + wp) * 8 + g) * 8);
      } else if (lane == 0) {
        mbar_arrive1(S.bars + BW_E + ws);  // every code and scale of this stage is in registers: weights slot free
      }
#pragma unroll
      for (int h = 0; h < 4 / GS; ++h) {
        const int ci = lg * (4 / GS) + h;  // commit group index in the stage
#pragma unroll
        for (int jj = 0; jj < GS; ++jj) {
          const int js = h * GS + jj;      // k16 step within the 4-step load group
          const uint32_t q = js == 0 ? cwg.x : js == 1 ? cwg.y : js == 2 ? cwg.z : cwg.w;
          uint32_t r0, r1, r2, r3;
#if OKC_XNODQ
          r0 = q; r1 = q >> 1; r2 = q >> 2; r3 = q >> 3;  // dev: no decode
#else
          dq4(q, r0, r1, r2, r3);
#endif
#if OKC_WGSSC2
          // scales stored [j][h] (rows g, g+8 of step j side by side): one PRMT + one IMAD build both rows' scales in
          // one bf16x2, and the multiplies broadcast its halves (HMUL2 .H0_H0 / .H1_H1): 2 instructions per k16 fewer
          const uint32_t sw = js < 2 ? scg.x : scg.y, sb = (uint32_t)(js & 1) * 2u;
          const uint32_t sgp = __byte_perm(sw, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16c + 0x5C005C00u;
#if OKC_XNOSC
          (void)sgp;
          af[ci % NB][jj][0] = r0; af[ci % NB][jj][1] = r1; af[ci % NB][jj][2] = r2; af[ci % NB][jj][3] = r3;  // dev: no scale
#else
          af[ci % NB][jj][0] = bmul2l(r0, sgp); af[ci % NB][jj][1] = bmul2h(r1, sgp);
          af[ci % NB][jj][2] = bmul2l(r2, sgp); af[ci % NB][jj][3] = bmul2h(r3, sgp);
#endif
#else
          const uint32_t sg = SCDK(scg.x, js, k16c), s8 = SCDK(scg.y, js, k16c);
          af[ci % NB][jj][0] = bmul2(r0, sg); af[ci % NB][jj][1] = bmul2(r1, s8);
          af[ci % NB][jj][2] = bmul2(r2, sg); af[ci % NB][jj][3] = bmul2(r3, s8);
#endif
        }
        wg_fence();
#pragma unroll
        for (int jj = 0; jj < GS; ++jj) {
          if (FIRST && ci == 0 && jj == 0)
            WGSMMA<NTT>::mma0(d, af[0][0][0], af[0][0][1], af[0][0][2], af[0][0][3], xd);
          else
            WGSMMA<NTT>::mma(d, af[ci % NB][jj][0], af[ci % NB][jj][1], af[ci % NB][jj][2], af[ci % NB][jj][3],
                             xd + (uint64_t)((lg * 4 + h * GS + jj) * 16));
        }
        wg_commit();
        wg_wait<OKC_WGSWD>();
        if (ci == OKC_WGSWD - 1) {  // the previous stage's last group has retired: its x slot is free
          if (prev >= 0) {  // the previous stage's last group just retired: its x slot is free
            if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);
            prev = -1;
          }
        }
        // the previous segment's deferred work, behind group OKC_WGSFCI of the first stage (scout: CI 3 lets this WG's
        // own red.adds / epilogue stores drain a little longer before the count / release atomics: +0.5-1%)
        if (FIRST && ci == OKC_WGSFCI && pd.kind) wgs_flush<PB>(p, S, pd);
      }
    }
    prev = xs;
    ++seq;
  };
  stage(std::true_type{});
#pragma unroll 1
  for (int st = s0 + 1; st < s1; ++st) stage(std::false_type{});
  wg_wait<0>();
  if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);
  if (OKC_WGSX & 8) { if (d[0][0] == 1234.5f) S.wglast[0] = 7; return; }
  if (slot >= 0) {  // a share of a unit split between CTAs: add it now, count it later (deferred)
    float* fb = wgs_fix(p, PB) + (size_t)(slot * 4 + wg) * WGS_FIXW + wt * 4;
#pragma unroll
    for (int nt = 0; nt < NTT; ++nt)
      asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" ::"l"(fb + nt * 512), "f"(d[nt][0]), "f"(d[nt][1]),
                   "f"(d[nt][2]), "f"(d[nt][3]) : "memory");
    // kind 3: this share finishes the unit (designated finisher, OKC_WGSDFIN): phase A the share holding stage 15,
    // phase B the one holding stage 0
    pd = WgsPend{(((OKC_WGSDFIN >> (PB ? 1 : 0)) & 1) && (OKC_WGSDYN ? s1 == 16 : (PB ? s0 == 0 : s1 == 16))) ? 3 : 2, u, slot, s1 - s0};
    return;
  }
  const int j = u >> UBL, r16 = (u & ((1 << UBL) - 1)) * 16 + wg * 4 + wp;
#pragma unroll
  for (int nt = 0; nt < NTT; ++nt) wgs_epi_tile<PB>(p, S, j, r16, nt, d[nt]);
  if (!PB) pd = WgsPend{1, u, -1, 0};  // the doneA release waits until the stores are done: deferred
}
#if OKC_WGSMMAN > 0
// wgs_seg on per-warp mma.sync (m16n8k16) for jobs of <= OKC_WGSMMAN n-tiles: the same loads, dequant, scales and
// epilogue / fixup paths as the wgmma version (the dequantized registers are an m16n8k16 A fragment, warp wp of the
// warpgroup owns rows 16 wp.. of its 64; the wgmma D fragment of a warp is the mma.sync C fragment), but no
// warpgroup-wide synchronization, descriptor or commit/wait per step: B fragments are 32-bit LDS from the x ring's
// core matrices ([n8][k8][8 rows][16 B]: row g at +16 g, k = 2t at +4 t), 4 independent accumulator chains
template <bool PB, int NTT>
__device__ __forceinline__ void wgs_seg_mma(const Params& p, Smem& S, unsigned char* ring, int u, int s0, int s1, int slot,
                                            int& seq, WgsPend& pd) {
  const int tid = threadIdx.x, lane = tid & 31;
  const int warp = WARP_IDX(tid);
  const int wg = warp >> 2, wp = warp & 3, g = lane >> 2, t = lane & 3, wt = tid & 127;
  constexpr int UBL = PB ? 4 : 5;
  constexpr int NA = NTT == 1 ? OKC_WGSMMANA : 2;  // independent accumulator chains per n-tile
  const uint32_t k16c = 16u + p.zero;
  const unsigned char* xring = ring + WGS_NS * UNIT_BYTES + g * 16 + t * 4;
  float acc[NA][NTT][4];
#pragma unroll
  for (int a = 0; a < NA; ++a)
#pragma unroll
    for (int nt = 0; nt < NTT; ++nt) { acc[a][nt][0] = 0.f; acc[a][nt][1] = 0.f; acc[a][nt][2] = 0.f; acc[a][nt][3] = 0.f; }
#pragma unroll 1
  for (int st = s0; st < s1; ++st) {
    const int ws = seq % WGS_NS, xs = seq % WGS_NX;
    wgs_wait(S.bars + BW_F + ws, (uint32_t)(seq / WGS_NS) & 1u, 3000000 + seq);
    const unsigned char* pc = ring + ws * UNIT_BYTES + wg * WGS_PIECE;
    wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);
    const unsigned char* xb = xring + xs * WGS_XS;
    uint4 cwn = *reinterpret_cast<const uint4*>(pc + (wp * 32 + lane) * 16);
    uint2 scn = *reinterpret_cast<const uint2*>(pc + 8192 + (wp * 8 + g) * 8);
#pragma unroll
    for (int lg = 0; lg < 4; ++lg) {
      const uint4 cwg = cwn;
      const uint2 scg = scn;
      if (lg < 3) {
        cwn = *reinterpret_cast<const uint4*>(pc + (((lg + 1) * 4 + wp) * 32 + lane) * 16);
        scn = *reinterpret_cast<const uint2*>(pc + 8192 + (((lg + 1) * 4 + wp) * 8 + g) * 8);
      } else if (lane == 0) {
        mbar_arrive1(S.bars + BW_E + ws);  // every code and scale of this stage is in registers: weights slot free
      }
#pragma unroll
      for (int js = 0; js < 4; ++js) {
        const uint32_t q = js == 0 ? cwg.x : js == 1 ? cwg.y : js == 2 ? cwg.z : cwg.w;
        uint32_t r0, r1, r2, r3;
        dq4(q, r0, r1, r2, r3);
        const uint32_t sw = js < 2 ? scg.x : scg.y, sb = (uint32_t)(js & 1) * 2u;
        const uint32_t sgp = __byte_perm(sw, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16c + 0x5C005C00u;
        const uint32_t a0 = bmul2l(r0, sgp), a1 = bmul2h(r1, sgp), a2 = bmul2l(r2, sgp), a3 = bmul2h(r3, sgp);
        const int step = lg * 4 + js;  // k16 step in the stage: k8 blocks 2 step, 2 step + 1 (128 B each)
#pragma unroll
        for (int nt = 0; nt < NTT; ++nt) {
          const uint32_t b0 = *reinterpret_cast<const uint32_t*>(xb + nt * 4096 + step * 256);
          const uint32_t b1 = *reinterpret_cast<const uint32_t*>(xb + nt * 4096 + step * 256 + 128);
          mma_bf16(acc[step % NA][nt], a0, a1, a2, a3, b0, b1);
        }
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive1(S.bars + BX_E + xs);  // the stage's x slot is free (mma.sync is synchronous)
    if (st == s0 && pd.kind) wgs_flush<PB>(p, S, pd);  // the previous segment's deferred work
    ++seq;
  }
  float d[NTT][4];
#pragma unroll
  for (int nt = 0; nt < NTT; ++nt)
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      float v = acc[0][nt][kk];
#pragma unroll
      for (int a = 1; a < NA; ++a) v += acc[a][nt][kk];
      d[nt][kk] = v;
    }
  if (slot >= 0) {  // a share of a unit split between CTAs: add it now, count it later (deferred)
    float* fb = wgs_fix(p, PB) + (size_t)(slot * 4 + wg) * WGS_FIXW + wt * 4;
#pragma unroll
    for (int nt = 0; nt < NTT; ++nt)
      asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" ::"l"(fb + nt * 512), "f"(d[nt][0]), "f"(d[nt][1]),
                   "f"(d[nt][2]), "f"(d[nt][3]) : "memory");
    pd = WgsPend{(((OKC_WGSDFIN >> (PB ? 1 : 0)) & 1) && (OKC_WGSDYN ? s1 == 16 : (PB ? s0 == 0 : s1 == 16))) ? 3 : 2, u, slot, s1 - s0};
    return;
  }
  const int j = u >> UBL, r16 = (u & ((1 << UBL) - 1)) * 16 + wg * 4 + wp;
#pragma unroll
  for (int nt = 0; nt < NTT; ++nt) wgs_epi_tile<PB>(p, S, j, r16, nt, d[nt]);
  if (!PB) pd = WgsPend{1, u, -1, 0};
}
#endif
template <bool PB>
__device__ __forceinline__ void wgs_phase(const Params& p, Smem& S, unsigned char* ring, int g0, int g1, int& seq, int& k) {
  // (__forceinline__: outlined by NVVM when it grows, a CALL that makes ptxas serialize every wgmma, C7510)
  constexpr int UBL = PB ? 4 : 5;
  WgsPend pd{0, 0, 0, 0};
#if OKC_WGSDYN
  (void)g0; (void)g1;
#pragma unroll 1
  for (;;) {
    const int e = wgs_piece(S, k++);  // the weights producer's published piece
    if (e == WGS_PEND) break;
    const int u = e >> 8, s0 = (e >> 4) & 15, s1 = (e & 15) + 1;
    const int slot = (s1 - s0 == 16) ? -1 : u;  // a unit split between CTAs: fixup slot = the unit
    const int ntt = (S.jC[u >> UBL] + 7) >> 3;
#else
  (void)k;
#pragma unroll 1
  for (int gs = wgs_split(PB, g0, g1), ge = g1;;) {
    if (gs >= ge) {  // [sp, g1) done: then [g0, sp); the bounds come back from smem (no extra live state)
      const int a0 = S.wgr[2 * PB], a1 = S.wgr[2 * PB + 1], sp = wgs_split(PB, a0, a1);
      if (ge == a1 && sp > a0) { gs = a0; ge = sp; } else break;
    }
    const int u = gs >> 4, s0 = gs & 15, s1 = min(16, s0 + (ge - gs));
    // a unit split between CTAs: its fixup slot is the CTA holding its first stage
    const int slot = (s1 - s0 == 16) ? -1 : (s0 == 0 ? (int)blockIdx.x : wgs_cta_of(S, PB, u << 4, gridDim.x));
    const int ntt = (S.jC[u >> UBL] + 7) >> 3;
#endif
    switch (ntt) {
#if OKC_WGSMMAN >= 1
      case 1: wgs_seg_mma<PB, 1>(p, S, ring, u, s0, s1, slot, seq, pd); break;
#else
      case 1: wgs_seg<PB, 1>(p, S, ring, u, s0, s1, slot, seq, pd); break;
#endif
#if OKC_FT_T > 8
#if OKC_WGSMMAN >= 2
      case 2: wgs_seg_mma<PB, 2>(p, S, ring, u, s0, s1, slot, seq, pd); break;
#else
      case 2: wgs_seg<PB, 2>(p, S, ring, u, s0, s1, slot, seq, pd); break;
#endif
      case 3: wgs_seg<PB, 3>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 4: wgs_seg<PB, 4>(p, S, ring, u, s0, s1, slot, seq, pd); break;
#endif
#if OKC_FT_T > 32
      case 5: wgs_seg<PB, 5>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 6: wgs_seg<PB, 6>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 7: wgs_seg<PB, 7>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      default: wgs_seg<PB, 8>(p, S, ring, u, s0, s1, slot, seq, pd); break;
#endif
    }
#if !OKC_WGSDYN
    gs += s1 - s0;
#endif
  }
  wgs_flush<PB>(p, S, pd);
}
// cost-weighted stage ranges: per-stage cost by the job's n-tiles, a prefix per job and phase (S.wcp, S.wcpB)
// all 32-bit (tot * c < 2^32: tot <= 136 jobs x 512 stages x 255 x 132): no 64-bit divide subroutine calls, which run
// from a cold instruction cache at the start of every launch
__device__ __forceinline__ void wgs_prefix(Smem& S, int nj) {
  uint32_t acc = 0, accB = 0;
  for (int j = 0; j < nj; ++j) {
    S.wcp[j] = (int)acc; S.wcpB[j] = (int)accB;
    acc += (uint32_t)wgs_cost(0, S.jC[j]) << 9;
    accB += (uint32_t)wgs_cost(1, S.jC[j]) << 8;
  }
  S.wcp[nj] = (int)acc; S.wcpB[nj] = (int)accB;
  S.wtot = acc; S.wtotB = accB;
}
// stage at cost position x (phase pb), the stage whose cost interval holds x
__device__ __forceinline__ int wgs_stage_at(const Smem& S, int nj, int pb, uint32_t x) {
  const int* pre = pb ? S.wcpB : S.wcp;
  int lo = 0, hi = nj;  // job: last j with prefix(j) <= x
  while (hi - lo > 1) { const int mid = (lo + hi) >> 1; if ((uint32_t)pre[mid] <= x) lo = mid; else hi = mid; }
  const uint32_t cs = (uint32_t)wgs_cost(pb, S.jC[lo]);
  return (lo << (9 - pb)) + (int)((x - (uint32_t)pre[lo]) / cs);
}
__device__ __forceinline__ void wgs_range(const Smem& S, int nj, int pb, int c, int G, int& g0, int& g1) {
  const uint32_t tot = pb ? S.wtotB : S.wtot;
#if OKC_WGSDYN
  // static ranges over the first (1 - 2^-WGSDYNS) of the cost; the stages from stage_at(tots) on are the pool
  if (nj == 0) { g0 = g1 = 0; return; }
  const uint32_t tots = tot - (tot >> OKC_WGSDYNS);
  g0 = wgs_stage_at(S, nj, pb, tots * (uint32_t)c / (uint32_t)G);
  g1 = wgs_stage_at(S, nj, pb, tots * (uint32_t)(c + 1) / (uint32_t)G);
#else
  g0 = wgs_stage_at(S, nj, pb, tot * (uint32_t)c / (uint32_t)G);
  g1 = c == G - 1 ? (nj << (9 - pb)) : wgs_stage_at(S, nj, pb, tot * (uint32_t)(c + 1) / (uint32_t)G);
#endif
}
// this CTA's stage ranges (stream-K over 256-row units x 16 stages), one thread: the producers stream the same
__device__ __noinline__ void wgs_ranges(const Params& p, Smem& S) {
  const int nj = S.njF + S.njS;
  wgs_prefix(S, nj);
  wgs_range(S, nj, 0, blockIdx.x, gridDim.x, S.wgr[0], S.wgr[1]);
  wgs_range(S, nj, 1, blockIdx.x, gridDim.x, S.wgr[2], S.wgr[3]);
#if OKC_WGSDYN
  S.wgd[0] = nj ? wgs_stage_at(S, nj, 0, S.wtot - (S.wtot >> OKC_WGSDYNS)) : 0;
  S.wgd[1] = nj << 9;
  S.wgd[2] = nj ? wgs_stage_at(S, nj, 1, S.wtotB - (S.wtotB >> OKC_WGSDYNS)) : 0;
  S.wgd[3] = nj << 8;
#endif
#if OKC_DEBUG && OKC_WGSFIT
  if (p.dbg) {  // per-phase stage counts by n-tiles (bytes 0..7), for the cost-model fit
    for (int pb = 0; pb < 2; ++pb) {
      unsigned long long pk = 0;
      for (int gs = S.wgr[2 * pb]; gs < S.wgr[2 * pb + 1]; ++gs) {
        const int nt = (S.jC[gs >> (9 - pb)] + 7) >> 3;
        pk += 1ull << (8 * (nt - 1));
      }
      p.dbg[blockIdx.x * 8 + 6 + pb] = pk;
    }
  }
#endif
}
// padded-row base of every job in xA / xB (one warp: lanes = jobs)
__device__ __forceinline__ void wgs_jrow(Smem& S) {
  const int tid = threadIdx.x;
  const int nj = S.njF + S.njS;
  int carry = 0;
  for (int j0 = 0; j0 < nj; j0 += 32) {
    const int j = j0 + tid;
    const int v = j < nj ? (((S.jC[j] + 7) >> 3) << 3) : 0;
    int incl = v;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { const int t = __shfl_up_sync(0xffffffffu, incl, d); if (tid >= d) incl += t; }
    if (j < nj) S.jrow[j] = carry + incl - v;
    carry += __shfl_sync(0xffffffffu, incl, 31);
  }
  if (tid == 0) S.jrow[nj] = carry;
}
// the CTA whose range holds stage s (the fixup slot of a split unit is the CTA holding its first stage)
__device__ __forceinline__ int wgs_cta_of(const Smem& S, int pb, int s, int G) {
  const uint32_t tot = pb ? S.wtotB : S.wtot;
  const int j = s >> (9 - pb);
  const uint32_t xe = (uint32_t)(pb ? S.wcpB : S.wcp)[j] + (uint32_t)((s & ((1 << (9 - pb)) - 1)) + 1) * (uint32_t)wgs_cost(pb, S.jC[j]);
  // largest c with floor(tot * c / G) < xe
  int c = (int)(((xe - 1) * (unsigned long long)G) / tot);
  if (c > G - 1) c = G - 1;
  while (c + 1 < G && tot * (uint32_t)(c + 1) / (uint32_t)G < xe) ++c;
  while (c > 0 && tot * (uint32_t)c / (uint32_t)G >= xe) --c;
  return c;
}
#endif

template <bool PB, bool BIG>
__device__ __forceinline__ void run_phase(const Params& p, Smem& S, __nv_bfloat16* xbuf) {
  const int njF = S.njF, njS = S.njS;
  const int ubs = PB ? UB_B : UB_A, rbs = PB ? RB_B : RB_A;
  const int nS = njS * rbs, nF = njF * ubs;
#if OKC_FATSTATIC
  // fat units first (round robin over CTAs), then a skinny share sized so every CTA carries the same cost
#if OKC_TINY
  if (nF > 0) __trap();  // tiny build (M <= 8): a job never has more than 8 tokens, so there are no fat units
#else
  if (nF > 0) { if (BIG) fat_stream<PB, true>(p, S, xbuf, nF, ubs); else fat_stream_ni<PB>(p, S, xbuf, nF, ubs); }
#endif
  const int G = gridDim.x, c = blockIdx.x;
  int v0, v1;
  const int nSs = nS - dyn_pool(nS, G);  // static skinny units; [nSs, nS) is the run-time pool
  skinny_split<BIG, PB>(nSs, nF, G, c, v0, v1);
  // cross-phase prefill: a CTA without phase-B fat units starts phase B with its skinny share, so phase A's last ring
  // refills already fetch those units (same split function as phase B uses)
  int cB0 = 0, cBn = 0;
  bool cpf = false;
  if (!PB) {
    const int nFB = njF * UB_B;
    if (c >= nFB) {
      int b1;
      skinny_split<BIG, true>(njS * RB_B - dyn_pool(njS * RB_B, G), nFB, G, c, cB0, b1);
      cBn = b1 - cB0;
      cpf = OKC_CPFF || njF == 0;
    }
  }
#if OKC_WSDYN
  (void)cpf; (void)cBn; (void)v0; (void)v1;
  if (!PB) skinny_ring_wsd(p, S, reinterpret_cast<unsigned char*>(xbuf));  // both phases in one dynamic sequence
#elif OKC_WS
  (void)cpf; (void)cBn;
  skinny_ring_ws<PB>(p, S, v0, v1, reinterpret_cast<unsigned char*>(xbuf), S.bars, S.seq);
#else
  skinny_ring<PB>(p, S, v0, v1, reinterpret_cast<unsigned char*>(xbuf), S.bars, S.seq,
                  reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<unsigned char*>(xbuf) + OKC_NS * UNIT_BYTES), cpf, cB0, cBn, nS, nSs);
#endif
#else
  const int v0 = (int)((long long)nS * blockIdx.x / gridDim.x), v1 = (int)((long long)nS * (blockIdx.x + 1) / gridDim.x);
  skinny_ring<PB>(p, S, v0, v1, reinterpret_cast<unsigned char*>(xbuf), S.bars, S.seq,
                  reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<unsigned char*>(xbuf) + OKC_NS * UNIT_BYTES), false, 0, 0, nS, nS);
#if OKC_TINY
  if (nF > 0) __trap();  // tiny build (M <= 8): a job never has more than 8 tokens, so there are no fat units
#else
  if (nF > 0) { if (BIG) fat_stream<PB, true>(p, S, xbuf, nF, ubs); else fat_stream_ni<PB>(p, S, xbuf, nF, ubs); }
#endif
#endif
}

// a volatile shared load at every use: hoisted, the pointer would stay live across the accumulate loops (96-register
// cap: spills)
__device__ __forceinline__ float* ss_cur() {
#if OKC_WGS
  unsigned long long v;
  asm volatile("ld.volatile.shared.u64 %0, [%1];\n" : "=l"(v) : "r"(smem_u32(&reinterpret_cast<const Smem*>(smem_raw)->ssp)));
  return reinterpret_cast<float*>(v);
#else
  return reinterpret_cast<const Smem*>(smem_raw)->ssp;  // the skinny / fat builds have registers to spare: let it hoist
#endif
}
template <bool BIG>
__global__ void __launch_bounds__(NTL, 1) moe_fused(Params p) {
  Smem& S = *reinterpret_cast<Smem*>(smem_raw);
  __nv_bfloat16* xbuf = reinterpret_cast<__nv_bfloat16*>(smem_raw + ((sizeof(Smem) + 127) / 128) * 128);
  const int tid = threadIdx.x;
  DBG(0);
  // programmatic dependent launch: everything below reads the previous kernel's outputs (routing, x) or writes
  // scratch, so wait for it here; without the launch attribute this returns at once
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#if OKC_WSDYN
  if (tid >= NT) { ws_helper_d(p, S, reinterpret_cast<unsigned char*>(xbuf)); return; }
#elif OKC_WS
  if (tid >= NT) { ws_helper(p, S, reinterpret_cast<unsigned char*>(xbuf)); return; }
#endif
#if OKC_WGS
  if (tid >= NT) {
    if (tid < NT + 32) wgs_wproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
    else wgs_xproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
    return;
  }
#endif
#if OKC_ROUTE_LOGITS
  {  // the top-8 selection, one warp per token (every CTA, identical result): S.eid for the routing below; the
     // logits of the warp's next two tokens are kept in flight (L2 latency overlaps the selection)
    const int lo = p.lsp ? __ldg(p.lsp) : p.lo;
    const int lane = tid & 31, warp = tid >> 5;
    const float4* bb = reinterpret_cast<const float4*>(p.rbias + lane * 8);
    const float4 b0 = __ldg(bb), b1 = __ldg(bb + 1);
    const float bv[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
    float4 q0[2], q1[2], q2[2];
    auto ld = [&](int t, float4 (&q)[2]) {
      if (t < p.M) {
        const float4* lg = reinterpret_cast<const float4*>(p.logits + (size_t)t * 256 + lane * 8);
        q[0] = __ldg(lg); q[1] = __ldg(lg + 1);
      }
    };
    const bool dist = !OKC_TINY && p.M > OKC_RLDIST;  // (the tiny builds' FAST1 needs S.eid before routing)
    const int G = (int)gridDim.x;
    const int t0 = dist ? (int)blockIdx.x + warp * G : warp, ts = dist ? NW * G : NW;
    ld(t0, q0);
    ld(t0 + ts, q1);
    for (int t = t0; t < p.M; t += ts) {
      ld(t + 2 * ts, q2);
      route_select(p, S, t, lo, q0, bv, !dist, dist || blockIdx.x == 0);
      q0[0] = q1[0]; q0[1] = q1[1]; q1[0] = q2[0]; q1[1] = q2[1];
    }
    {  // per-token GEMM1 scale from the pre-expert RMS partials (all tokens, every CTA); 1 without partials
      float* rs = rs_ptr(S);
      if (p.rsp) {
        for (int t = warp; t < p.M; t += NW) {
          const float4 v = __ldcg(reinterpret_cast<const float4*>(p.rsp + (size_t)t * 128) + lane);
          float sum = (v.x + v.y) + (v.z + v.w);
#pragma unroll
          for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
          if (lane == 0) rs[t] = rsqrtf(sum * (1.f / 4096.f) + 1e-5f);
        }
      } else {
        for (int t = tid; t < p.M; t += NT) rs[t] = 1.f;
      }
    }
    CTA_SYNC();
    if (dist) {  // every CTA's selections are visible after this barrier: a monotonic epoch barrier (ctr[6] = distributed
                 // launches completed, bumped by CTA 0 after the final barrier; ctr[7] = arrivals), so interleaved
                 // launches of other builds, which never touch these counters, cannot leave a stale count
      if (tid == 0) {
        const int e = *reinterpret_cast<volatile const int*>(p.ctr + 6);
        __threadfence();
        atomicAdd(p.ctr + 7, 1);
        while (ld_acquire(p.ctr + 7) < (e + 1) * G) __nanosleep(64);
      }
      CTA_SYNC();
    }
  }
#endif
#if FAST1_ON
  // tiny build (M <= 8): every touched expert is one skinny job, in expert order, so this CTA's first units follow
  // from the touched-expert mask alone; the last warp issues them before routing, off the cold-start critical path
  if (tid >= NT - 32) {
    const int ln = tid & 31;
#if OKC_SPLIT
    // per-expert entry counts -> jobs per expert (ceil(c / 8)) -> inclusive prefix over experts (lane = expert)
    S.fcnt[ln] = 0;
    __syncwarp();
    for (int i0 = 0; i0 < p.M * 8; i0 += 32) {
      const int i = i0 + ln;
      int e = -1;
#if OKC_ROUTE_LOGITS
      if (i < p.M * 8) e = S.eid[i];
#else
      if (i < p.M * 8) { e = __ldg(p.topk_ids + i) - p.lo; if (e < 0 || e >= p.nloc) e = -1; }
#endif
      const unsigned peers = __match_any_sync(0xffffffffu, e);
      if (e >= 0 && ln == __ffs(peers) - 1) atomicAdd(&S.fcnt[e], __popc(peers));
    }
    __syncwarp();
    int jincl = (S.fcnt[ln] + SK_T - 1) / SK_T;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { const int v = __shfl_up_sync(0xffffffffu, jincl, d); if (ln >= d) jincl += v; }
    const int njob = __shfl_sync(0xffffffffu, jincl, 31);
    const int nS1 = njob * RB_A - dyn_pool(njob * RB_A, gridDim.x);
    const float rG1 = __fdividef(1.f, (float)gridDim.x);
    const int v01 = muldiv_g(nS1, blockIdx.x, gridDim.x, rG1), v11 = muldiv_g(nS1, blockIdx.x + 1, gridDim.x, rG1);
    const int n1 = min(OKC_FAST1, v11 - v01);
    int fex[OKC_FAST1 > 0 ? OKC_FAST1 : 1];
#pragma unroll
    for (int i = 0; i < OKC_FAST1; ++i) {
      const int jj = (v01 + i) >> RB_A_LOG;   // (n1 is warp-uniform: every lane runs the same ballots)
      fex[i] = i < n1 ? __ffs(__ballot_sync(0xffffffffu, jincl > jj)) - 1 : 0;
    }
#elif OKC_F1EARLY
    // the topk loads go out first (E <= 64 in the tiny builds: two per lane at most) and lane 0 initializes the
    // barriers while they are in flight (the inits used to wait behind the cold loads)
    uint32_t m = 0;
    const int E8 = p.M * 8;
#if OKC_ROUTE_LOGITS
    const int ev0 = 0, ev1 = 0;
    (void)ev0; (void)ev1;
#else
    const int ev0 = ln < E8 ? __ldg(p.topk_ids + ln) : 0, ev1 = ln + 32 < E8 ? __ldg(p.topk_ids + ln + 32) : 0;
#endif
    if (ln == 0) {
      for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);
#if OKC_WS
      for (int k = 0; k < OKC_NS; ++k) mbar_init(S.wse + k, NW);
      for (int k = 0; k < NRB; ++k) { mbar_init(S.wsrf + k, OKC_WSB ? 1 : NW); mbar_init(S.wsre + k, 1); }
#endif
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
#if OKC_ROUTE_LOGITS
    for (int i = ln; i < E8; i += 32) { const int e = S.eid[i]; if (e >= 0) m |= 1u << e; }
#else
    if (ln < E8) { const int e = ev0 - p.lo; if (e >= 0 && e < p.nloc) m |= 1u << e; }
    if (ln + 32 < E8) { const int e = ev1 - p.lo; if (e >= 0 && e < p.nloc) m |= 1u << e; }
    for (int i = ln + 64; i < E8; i += 32) {  // (never in the tiny builds: M <= 8)
      const int e = __ldg(p.topk_ids + i) - p.lo;
      if (e >= 0 && e < p.nloc) m |= 1u << e;
    }
#endif
    m = __reduce_or_sync(0xffffffffu, m);
#else
    uint32_t m = 0;
    for (int i = ln; i < p.M * 8; i += 32) {
#if OKC_ROUTE_LOGITS
      const int e = S.eid[i];
#else
      const int e = __ldg(p.topk_ids + i) - p.lo;
#endif
      if (e >= 0 && e < p.nloc) m |= 1u << e;
    }
    m = __reduce_or_sync(0xffffffffu, m);
#endif
    if (ln == 0) {
#if OKC_SPLIT || !OKC_F1EARLY
      for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);
#if OKC_WS
      for (int k = 0; k < OKC_NS; ++k) mbar_init(S.wse + k, NW);
      for (int k = 0; k < NRB; ++k) { mbar_init(S.wsrf + k, OKC_WSB ? 1 : NW); mbar_init(S.wsre + k, 1); }
#endif
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
#if OKC_SPLIT
      const int v0 = v01, n = n1;
#else
      const int nS = __popc(m) * RB_A - dyn_pool(__popc(m) * RB_A, gridDim.x);  // static part (pool tail claimed at run time)
      const float rG = __fdividef(1.f, (float)gridDim.x);
      const int v0 = muldiv_g(nS, blockIdx.x, gridDim.x, rG), v1 = muldiv_g(nS, blockIdx.x + 1, gridDim.x, rG);
      const int n = min(OKC_FAST1, v1 - v0);
#endif
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#if OKC_WSDYN
      for (int k = 0; k < 8; ++k) *reinterpret_cast<volatile int*>(&S.uid[k]) = -1;  // no stale sequence tags
#endif
#if OKC_SPLIT
#pragma unroll
      for (int i = 0; i < OKC_FAST1; ++i) {
        if (i >= n) break;
        const int v = v0 + i;
        const int e = fex[i];
#else
      for (int i = 0; i < n; ++i) {
        const int v = v0 + i;
        const int e = (int)__fns(m, 0, (v >> RB_A_LOG) + 1);
#endif
        const unsigned char* src = p.wAu + ((size_t)e * RB_A + (v & (RB_A - 1))) * (size_t)UNIT_BYTES;
        unsigned char* slot = reinterpret_cast<unsigned char*>(xbuf) + i * UNIT_BYTES;
#if OKC_WSDYN
        *reinterpret_cast<volatile int*>(&S.uid[i]) = (i << 16) | v;  // sequence index i = phase-A unit v
#endif
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(S.bars + i)), "r"(UNIT_BYTES) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
                     ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(S.bars + i)) L2H_ARG : "memory");
      }
      S.preissued = n > 0 ? n : 0;
    }
  }
#endif
#if OKC_XPF2
  // pull every token row of x into L2 now (one 128 B line per thread-iteration); the skinny jobs read them later
  for (int li = blockIdx.x * NT + tid; li < p.M * (HD * 2 / 128); li += gridDim.x * NT)
    asm volatile("prefetch.global.L2 [%0];\n" ::"l"(reinterpret_cast<const char*>(p.x) + (size_t)li * 128));
#endif
  for (int i = tid; i < (OKC_PRMTSC ? 0 : 256); i += NT) {  // fp8 e4m3 code -> bf16x2(value * 2^64), computed (no global load)
    const int sgn = i >> 7, ex = (i >> 3) & 15, mt = i & 7;
    uint32_t b;
    if (ex == 15 && mt == 7) {
      b = 0xFFFFu;
    } else {
      float v = ex == 0 ? (float)mt * 0x1p-9f : (1.f + (float)mt * 0.125f) * __int_as_float((ex - 7 + 127) << 23);
      v *= 0x1p64f;
      b = __float_as_uint(sgn ? -v : v) >> 16;
    }
    S.lut[i] = b | (b << 16);
  }
  if (tid == 0) {
#if !FAST1_ON
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && OKC_USE_TMA
#if OKC_WGS
    // full slots (TMA tx), empty slots (one arrival per consumer warp), partials ready (every consumer thread),
    // partial buffer free (the epilogue warpgroup)
    for (int k = 0; k < BX_E + WGS_NX; ++k) mbar_init(S.bars + k, k < BW_E ? 1 : k < BX_F ? NW : k < BX_E ? 1 : NW);
#else
    for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);
#endif
#else
    for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, NT);
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
    S.preissued = 0;
#endif
    S.seq = 0;
    S.fseq = 0;
    S.preissuedB = 0;
    S.xbuf_ptr = xbuf;
#if OKC_WGSDYN
    for (int k = 0; k < 16; ++k) S.wseg[k] = -1;  // no stale piece tags from an earlier launch
#endif
    // the generation can only change at this launch's grid barrier, which every CTA reaches after this point
    const int gen = *reinterpret_cast<volatile const int*>(p.ctr + 3);  // launch generation (bumped at the barrier)
    S.gen = gen;
    S.ssp = (gen & 1) ? p.ss1 : p.ss;
  }
#if SKDYN || OKC_WSDYN
  for (int i = tid; i < JMAX; i += NT) S.dync[i] = 0;
#endif
  route(p, S);
#if OKC_WGS && OKC_WGSEARLY
  static_assert(OKC_WGSPR, "OKC_WGSEARLY: the x producer computes the ranges");
  if (tid < 32) wgs_jrow(S);  // (the x producer needs jrow; the gather below needs it too)
  CTA_SYNC();
  __threadfence_block();
  asm volatile("bar.arrive 7, %0;\n" ::"n"(NT + 64) : "memory");  // both producers start now: weights stream during the gather
#elif OKC_WGS && OKC_WGSPR
  __threadfence_block();
  asm volatile("bar.arrive 7, %0;\n" ::"n"(NT + 32) : "memory");  // the x producer computes the stage ranges
#endif
  DBG(1);
  {  // the previous launch's ss copy and barrier counter are the next launch's: zero them now (this CTA's share)
    float* other = (S.gen & 1) ? p.ss : p.ss1;
    for (int i = blockIdx.x * NT + tid; i < EMAX; i += gridDim.x * NT) other[i] = 0.f;
    if (blockIdx.x == 0 && tid == 0) p.ctr[4 + ((S.gen + 1) & 1)] = 0;
  }
#if OKC_WS
  if (tid == 0) {  // this CTA's unit ranges for the helper warp (the same exact split run_phase uses: no fat jobs)
    const int G = gridDim.x, c = blockIdx.x;
    const float rG = __fdividef(1.f, (float)G);
    const int nA = S.njS * RB_A, nB = S.njS * RB_B;
#if OKC_WSDYN
    const int nAs = nA - dyn_pool(nA, G), nBs = nB - dyn_pool(nB, G);  // static parts (the same split FAST1 used)
    S.wsr[0] = muldiv_g(nAs, c, G, rG); S.wsr[1] = muldiv_g(nAs, c + 1, G, rG);
    S.wsr[2] = muldiv_g(nBs, c, G, rG); S.wsr[3] = muldiv_g(nBs, c + 1, G, rG);
    S.wsd[0] = nAs; S.wsd[1] = nA; S.wsd[2] = nBs; S.wsd[3] = nB;
#else
    S.wsr[0] = muldiv_g(nA, c, G, rG); S.wsr[1] = muldiv_g(nA, c + 1, G, rG);
    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);
#endif
  }
  WS_JOIN();
#endif
#if OKC_WGS
#if !OKC_WGSEARLY
  if (tid < 32) wgs_jrow(S);
  CTA_SYNC();
#endif
  // phase A's x into xA (this CTA's share of the rows), before the producers start: its x reads do not queue behind
  // every CTA's first weight stages (OKC_WGSEARLY: the producers are already streaming)
  wgs_gather(p, S);
#if !OKC_WGSFIT
  DBG(7);
#endif
#if !OKC_WGSPR
  if (tid == 0) wgs_ranges(p, S);
#endif
#if OKC_WGSEARLY
  asm volatile("bar.sync 9, %0;\n" ::"n"(NT + 32) : "memory");  // the x producer's stage ranges (S.wgr, cost prefix) are written
#else
  WS_JOIN();
#endif
#if !OKC_WGSFIT
  DBG(6);
#endif
  {
    unsigned char* ring = reinterpret_cast<unsigned char*>(xbuf);
    int seq = 0, wk = 0;
    wgs_phase<false>(p, S, ring, S.wgr[0], S.wgr[1], seq, wk);
    DBG(2);
    wgs_phase<true>(p, S, ring, S.wgr[2], S.wgr[3], seq, wk);
    DBG(3);
  }
#else
  run_phase<false, BIG>(p, S, xbuf);
  DBG(2);
  run_phase<true, BIG>(p, S, xbuf);
  DBG(3);
#endif
#if OKC_WS
  WS_JOIN();  // the helper's last epilogue stores are visible
#endif
  // programmatic dependent launch: the next kernel may start its own prologue now (its griddepcontrol.wait still
  // waits for this whole grid to finish and its writes to be visible)
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  // grid barrier on this launch's counter (two, by generation: the other one was zeroed by this launch for the next);
  // the last arriver bumps the generation, which no CTA of this launch reads again
#if OKC_FINPF && OKC_FINCOL && !OKC_WGS
  // the CTA's first final item: its routing (pos) and route weights are inputs, so they load before the barrier
  constexpr int FSEGP = HD / (NT * 4);
  const int itp = blockIdx.x;
  const bool hasp = itp < p.M * FSEGP;
  const int tkp = hasp ? itp / FSEGP : 0;
  int qvp[8];
  float wvp[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    qvp[s] = hasp ? (int)S.pos[tkp * 8 + s] : -1;
#if OKC_ROUTE_LOGITS
    wvp[s] = 0.f;  // (CTA 0's weights: loaded after the grid barrier)
#else
    wvp[s] = (qvp[s] >= 0) ? __ldg(p.topk_w + tkp * 8 + s) : 0.f;
#endif
  }
#endif
  CTA_SYNC();
  if (tid == 0) {
    int* bc = p.ctr + 4 + (S.gen & 1);
    __threadfence();
    const int old = atomicAdd(bc, 1);
    if (old == (int)gridDim.x - 1) atomicAdd(p.ctr + 3, 1);
    while (ld_acquire(bc) < (int)gridDim.x) __nanosleep(64);
  }
  CTA_SYNC();
  DBG(4);
#if OKC_ROUTE_LOGITS
  if (!OKC_TINY && p.M > OKC_RLDIST && blockIdx.x == 0 && tid == 0) p.ctr[6] += 1;  // distributed-routing epoch
#endif
#if OKC_ROUTE_LOGITS && OKC_FINPF && OKC_FINCOL && !OKC_WGS
#pragma unroll
  for (int s = 0; s < 8; ++s) wvp[s] = (qvp[s] >= 0) ? __ldcg(p.topk_w + tkp * 8 + s) : 0.f;
#endif
  // phases A and B are over everywhere: zero the per-launch flags for the next launch (this CTA's share)
  {
    const int G = gridDim.x, i0 = blockIdx.x * NT + tid;
    if (i0 < 2) p.ctr[i0] = 0;  // dynamic pool counters
    for (int i = i0; i < JMAX; i += G * NT) p.doneA[i] = 0;
#if OKC_WGS
    for (int i = i0; i < JMAX; i += G * NT) p.xready[i] = 0;
    for (int i = i0; i < JMAX * 16; i += G * NT) p.doneAs[i] = 0;
#endif
  }
  // final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]
#if OKC_WGS && OKC_WGSFINSS
  // item = one token's whole row (thread = 8 columns): pass 1 sums each local route's y^2 over the row (warp
  // shuffles, then the 16 warps' partials through smem: the ring is free now), pass 2 re-reads the rows (L2 hits)
  {
    static_assert(HD == NT * 8, "8 columns per thread");
    float* red = reinterpret_cast<float*>(xbuf);
    const int lane = tid & 31, warp = WARP_IDX(tid);
    for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {
      int qv[8];
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) qv[sl] = S.pos[tk * 8 + sl];
      float ps[8];
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) {
        ps[sl] = 0.f;
        if (qv[sl] >= 0) {
          const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;
          const float4 a = __ldcg(src), b = __ldcg(src + 1);
          ps[sl] = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w + b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;
        }
      }
#pragma unroll
      for (int sl = 0; sl < 8; ++sl)
        if (qv[sl] >= 0) {  // uniform over the CTA (one token)
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) ps[sl] += __shfl_xor_sync(0xffffffffu, ps[sl], o);
        }
      if (lane == 0) {
#pragma unroll
        for (int sl = 0; sl < 8; ++sl) red[warp * 8 + sl] = ps[sl];
      }
      CTA_SYNC();
      if (tid < 8) {
        float tsum = 0.f;
        for (int w = 0; w < NW; ++w) tsum += red[w * 8 + tid];
        red[NW * 8 + tid] = tsum;
      }
      CTA_SYNC();
      float acc[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[i] = 0.f;
#pragma unroll
      for (int sl = 0; sl < 8; ++sl)
        if (qv[sl] >= 0) {
          const float r = TOPKW(tk * 8 + sl) * rsqrtf(red[NW * 8 + sl] * (1.f / HD) + 1e-8f);
          const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;
          const float4 a = __ldcg(src), b = __ldcg(src + 1);
          acc[0] += r * a.x; acc[1] += r * a.y; acc[2] += r * a.z; acc[3] += r * a.w;
          acc[4] += r * b.x; acc[5] += r * b.y; acc[6] += r * b.z; acc[7] += r * b.w;
        }
      uint4 o;
      uint32_t* ow = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        __nv_bfloat162 v = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
        ow[i] = *reinterpret_cast<uint32_t*>(&v);
      }
      *reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + tid * 8) = o;
      CTA_SYNC();  // red is reused by the next token
    }
  }
#elif OKC_FINCOL
  // column-parallel: item = (token, 1024-column segment); thread = 4 columns
  constexpr int FSEG = HD / (NT * 4);  // column segments per token (thread = 4 columns)
  int it0 = blockIdx.x;
#if OKC_FINPF
  if (hasp) {   // first item: weights / positions from before the barrier; all 16 loads issued before the math
    const int col = (itp % FSEG) * (NT * 4) + tid * 4;
    float ssv[8];
    float4 yv[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      ssv[s] = qvp[s] >= 0 ? __ldcg(ss_cur() + qvp[s]) : 0.f;
      yv[s] = qvp[s] >= 0 ? __ldcg(reinterpret_cast<const float4*>(p.ybuf + (size_t)qvp[s] * HD + col)) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      if (qvp[s] >= 0) {
        const float r = wvp[s] * rsqrtf(ssv[s] * (1.f / HD) + 1e-8f);
        acc.x += r * yv[s].x; acc.y += r * yv[s].y; acc.z += r * yv[s].z; acc.w += r * yv[s].w;
      }
    }
    __nv_bfloat162 o0 = __floats2bfloat162_rn(acc.x, acc.y), o1 = __floats2bfloat162_rn(acc.z, acc.w);
    uint2 o;
    o.x = *reinterpret_cast<uint32_t*>(&o0);
    o.y = *reinterpret_cast<uint32_t*>(&o1);
    *reinterpret_cast<uint2*>(p.out + (size_t)tkp * HD + col) = o;
  }
  it0 += gridDim.x;
#endif
  for (int it = it0; it < p.M * FSEG; it += gridDim.x) {
    const int tk = it / FSEG, col = (it % FSEG) * (NT * 4) + tid * 4;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int q = S.pos[tk * 8 + s];
      if (q >= 0) {
        const float r = TOPKW(tk * 8 + s) * rsqrtf(__ldcg(ss_cur() + q) * (1.f / HD) + 1e-8f);
        const float4 v = __ldcg(reinterpret_cast<const float4*>(p.ybuf + (size_t)q * HD + col));
        acc.x += r * v.x; acc.y += r * v.y; acc.z += r * v.z; acc.w += r * v.w;
      }
    }
    __nv_bfloat162 o0 = __floats2bfloat162_rn(acc.x, acc.y), o1 = __floats2bfloat162_rn(acc.z, acc.w);
    uint2 o;
    o.x = *reinterpret_cast<uint32_t*>(&o0);
    o.y = *reinterpret_cast<uint32_t*>(&o1);
    *reinterpret_cast<uint2*>(p.out + (size_t)tk * HD + col) = o;
  }
#else
  for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {
    float acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.f;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int q = S.pos[tk * 8 + s];
      if (q >= 0) {
        const float r = TOPKW(tk * 8 + s) * rsqrtf(__ldcg(ss_cur() + q) * (1.f / HD) + 1e-8f);
        const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)q * HD) + tid * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float4 v = __ldcg(src + i);
          acc[4 * i] += r * v.x; acc[4 * i + 1] += r * v.y; acc[4 * i + 2] += r * v.z; acc[4 * i + 3] += r * v.w;
        }
      }
    }
    uint4 o[2];
    uint32_t* ow = reinterpret_cast<uint32_t*>(o);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      __nv_bfloat162 v = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
      ow[i] = *reinterpret_cast<uint32_t*>(&v);
    }
    uint4* dst = reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + tid * 16);
    dst[0] = o[0]; dst[1] = o[1];
  }
#endif
  DBG(5);  // nothing to reset at exit: the flags were zeroed after the barrier, ss is double buffered
}

}  // namespace okc

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

static size_t okc_smem_bytes() {
  size_t base = ((sizeof(okc::Smem) + 127) / 128) * 128;
#if OKC_WGS
  return base + (size_t)okc::WGS_NS * okc::UNIT_BYTES + (size_t)okc::WGS_NX * okc::WGS_XS;
#endif
  size_t xa = (size_t)OKC_NS * okc::UNIT_BYTES + (OKC_SKT > 8 ? (size_t)okc::NW * 8 * okc::X1P * 2 : 0);
  size_t xb = (size_t)okc::NSF * okc::UNIT_BYTES + 2 * (size_t)okc::FT_T * okc::XP_FT * 2;
  return base + (xa > xb ? xa : xb);
}

static int g_okc_pdl = 0;  // launch with programmatic stream serialization (PDL); off unless okc_set_pdl(1)

void okc_set_pdl(int64_t on) { g_okc_pdl = on ? 1 : 0; }

static void okc_run_impl(std::vector<torch::Tensor>& w, std::vector<torch::Tensor>& scratch, torch::Tensor x,
                         const int* ids_ptr, const float* tw_ptr, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid,
                         const float* logits, const float* rbias, const int* lsp, int* gid_out, const float* rsp) {
  okc::Params p;
  p.logits = logits; p.rbias = rbias; p.lsp = lsp; p.gid_out = gid_out; p.rsp = rsp;
  p.wA = (const uint4*)w[0].data_ptr(); p.sA = (const uint4*)w[1].data_ptr(); p.gsA = (const float*)w[2].data_ptr();
  p.wB = (const uint4*)w[3].data_ptr(); p.sB = (const uint4*)w[4].data_ptr(); p.gsB = (const float*)w[5].data_ptr();
  p.lut = (const uint32_t*)w[6].data_ptr(); p.normw = (const __nv_bfloat16*)w[7].data_ptr();
  p.wAf = (const unsigned char*)w[8].data_ptr(); p.wBf = (const unsigned char*)w[9].data_ptr();
  p.wAu = (const unsigned char*)w[10].data_ptr(); p.wBu = (const unsigned char*)w[11].data_ptr();
  if (w.size() > 13) { p.wAn = (const unsigned char*)w[12].data_ptr(); p.wBn = (const unsigned char*)w[13].data_ptr(); }
  else { p.wAn = p.wAf; p.wBn = p.wBf; }  // no natural copies: the wgmma path must stay off (OKC_WGMIN 9)
#if OKC_WGS
  TORCH_CHECK(w.size() > 13, "OKC_WGS needs the WGS weight copies (w[12], w[13])");
  TORCH_CHECK((size_t)scratch[1].numel() >= (size_t)okc::EMAX + (size_t)okc::EMAX * okc::HD + okc::WGS_FIXF,
              "OKC_WGS needs the stream-K fixup area after ybuf (scratch[1])");
  p.wAs = (const unsigned char*)w[12].data_ptr(); p.wBs = (const unsigned char*)w[13].data_ptr();
#else
  p.wAs = nullptr; p.wBs = nullptr;
#endif
  p.zero = 0;
  p.x = (const __nv_bfloat16*)x.data_ptr(); p.topk_ids = ids_ptr; p.topk_w = tw_ptr;
  p.M = (int)x.size(0); p.lo = (int)lo; p.nloc = (int)nloc;
#if OKC_WGS && OKC_RLOAD && !OKC_ROUTE_LOGITS
  TORCH_CHECK(((uintptr_t)ids_ptr & 15) == 0, "topk_ids must be 16 B aligned and contiguous");
#endif
  int* ib = (int*)scratch[0].data_ptr();
  p.ctr = ib; p.doneA = ib + 8;
  p.ss = (float*)scratch[1].data_ptr();
  TORCH_CHECK(scratch[0].numel() >= 8 + 1024 + okc::EMAX, "scratch[0] needs the second ss copy");
  p.ss1 = reinterpret_cast<float*>(ib + 8 + 1024);
  p.ybuf = p.ss + okc::EMAX;
  p.inter = (__nv_bfloat16*)scratch[2].data_ptr();
#if OKC_WGS
  TORCH_CHECK((size_t)scratch[2].numel() >= ((size_t)okc::EMAX + 2 * (size_t)okc::WGS_XROWS) * okc::HD,
              "OKC_WGS needs xA / xB after inter (scratch[2])");
  p.xA = p.inter + (size_t)okc::EMAX * okc::HD;
  p.xB = p.xA + (size_t)okc::WGS_XROWS * okc::HD;
  p.xready = ib + 8 + 512;
  p.doneAs = reinterpret_cast<int*>(p.ybuf + (size_t)okc::EMAX * okc::HD + (size_t)2 * okc::WGS_SLOTS * 4 * okc::WGS_FIXW) + 2 * okc::WGS_SLOTS * 4;
#else
  p.xA = p.xB = nullptr;
  p.xready = nullptr;
  p.doneAs = nullptr;
#endif
  p.out = (__nv_bfloat16*)out.data_ptr();
  p.dbg = scratch.size() > 3 ? (unsigned long long*)scratch[3].data_ptr() : nullptr;
  static bool attr = false;
  size_t sm = okc_smem_bytes();
  if (!attr) {
    cudaError_t e0 = cudaFuncSetAttribute(okc::moe_fused<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    cudaError_t e1 = cudaFuncSetAttribute(okc::moe_fused<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
#if OKC_LAUNCHCHK
    TORCH_CHECK(e0 == cudaSuccess && e1 == cudaSuccess, "smem attribute ", (int)sm, ": ", cudaGetErrorString(e0 != cudaSuccess ? e0 : e1));
#else
    (void)e0; (void)e1;
#endif
    attr = true;
  }
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  // large batches run the variant with the fat path inlined (and wgmma); small ones keep it out of line, which
  // leaves the skinny loop's code and registers as tuned
  if (g_okc_pdl) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)grid);
    cfg.blockDim = dim3(okc::NTL);
    cfg.dynamicSmemBytes = sm;
    cfg.stream = st;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    if (p.M >= OKC_BIGM) cudaLaunchKernelEx(&cfg, okc::moe_fused<true>, p);
    else cudaLaunchKernelEx(&cfg, okc::moe_fused<false>, p);
    return;
  }
  if (p.M >= OKC_BIGM) okc::moe_fused<true><<<(int)grid, okc::NTL, sm, st>>>(p);
  else okc::moe_fused<false><<<(int)grid, okc::NTL, sm, st>>>(p);
#if OKC_LAUNCHCHK
  {
    cudaError_t e = cudaGetLastError();
    TORCH_CHECK(e == cudaSuccess, "launch (smem ", (int)sm, ", threads ", okc::NTL, "): ", cudaGetErrorString(e));
    e = cudaStreamSynchronize(st);
    TORCH_CHECK(e == cudaSuccess, "kernel (smem ", (int)sm, ", threads ", okc::NTL, "): ", cudaGetErrorString(e));
  }
#endif
}

void okc_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,
             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid) {
#if OKC_ROUTE_LOGITS
  TORCH_CHECK(false, "an OKC_ROUTE_LOGITS build: use okc_run_logits");
#else
  TORCH_CHECK(ids.is_contiguous() && tw.is_contiguous(), "topk_ids / topk_weights must be contiguous");
  okc_run_impl(w, scratch, x, (const int*)ids.data_ptr(), (const float*)tw.data_ptr(), out, lo, nloc, grid,
               nullptr, nullptr, nullptr, nullptr, nullptr);
#endif
}

// routing in the kernel: logits [M, 256] fp32, rbias [256] fp32, ls int32 [1] (local_start on the device), gid int32
// [M, 8] (chosen global ids, output), wts fp32 [>= M * 8] scratch (the renormalized route weights, written by CTA 0)
void okc_run_logits(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor logits,
                    torch::Tensor rbias, torch::Tensor ls, torch::Tensor out, torch::Tensor gid, torch::Tensor wts,
                    torch::Tensor rsp, int64_t nloc, int64_t grid) {
#if OKC_ROUTE_LOGITS
  const float* rsp_ptr = nullptr;
  if (rsp.numel() > 0) {
    TORCH_CHECK(rsp.is_contiguous() && rsp.dtype() == torch::kFloat32 && rsp.size(0) == x.size(0) && rsp.size(1) == 128,
                "pre_norm_partials: contiguous fp32 [M, 128]");
    TORCH_CHECK(x.size(0) <= 256, "pre_norm_partials: at most 256 tokens per launch");
    TORCH_CHECK(OKC_WS || OKC_WGS, "pre_norm_partials: WS / WGS builds only");
    rsp_ptr = (const float*)rsp.data_ptr();
  }
  TORCH_CHECK(logits.is_contiguous() && logits.dtype() == torch::kFloat32 && logits.size(1) == 256 && logits.size(0) == x.size(0),
              "router logits must be contiguous fp32 [M, 256]");
  TORCH_CHECK(rbias.is_contiguous() && rbias.dtype() == torch::kFloat32 && rbias.numel() == 256, "router bias: fp32 [256]");
  TORCH_CHECK(ls.dtype() == torch::kInt32 && ls.numel() >= 1, "local_start: int32 device tensor");
  TORCH_CHECK(gid.is_contiguous() && gid.dtype() == torch::kInt32 && gid.numel() >= x.size(0) * 8, "topk ids out: int32 [M, 8]");
  TORCH_CHECK(wts.is_contiguous() && wts.dtype() == torch::kFloat32 && wts.numel() >= x.size(0) * 8, "weights scratch: fp32 [M * 8]");
  okc_run_impl(w, scratch, x, nullptr, (const float*)wts.data_ptr(), out, 0, nloc, grid, (const float*)logits.data_ptr(),
               (const float*)rbias.data_ptr(), (const int*)ls.data_ptr(), (int*)gid.data_ptr(), rsp_ptr);
#else
  TORCH_CHECK(false, "not an OKC_ROUTE_LOGITS build");
#endif
}


'''

moe_BW_SRC = r'''
// H200 memory-pipeline microbenchmarks (diagnostics only).
#include <cuda_runtime.h>
#include <stdint.h>

namespace bw {

__device__ __forceinline__ uint4 ldg_stream(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

// per-warp contiguous streaming with a register ring of PF 512-byte chunks (like the MoE kernel)
template <int PF>
__global__ void __launch_bounds__(256, 1) ring_kernel(const uint4* __restrict__ src, size_t n16, unsigned* out) {
  const int lane = threadIdx.x & 31;
  const size_t warps = (size_t)gridDim.x * 8;
  const size_t wid = (size_t)blockIdx.x * 8 + (threadIdx.x >> 5);
  const size_t chunks = n16 / 32;                // 512-byte chunks
  const size_t per = (chunks + warps - 1) / warps;
  const size_t c0 = wid * per, c1 = c0 + per < chunks ? c0 + per : chunks;
  unsigned acc = 0;
  uint4 ring[PF];
#pragma unroll
  for (int r = 0; r < PF; ++r) ring[r] = (c0 + r < c1) ? ldg_stream(src + (c0 + r) * 32 + lane) : make_uint4(0, 0, 0, 0);
  for (size_t c = c0; c < c1; c += PF) {
#pragma unroll
    for (int r = 0; r < PF; ++r) {
      uint4 v = ring[r];
      if (c + PF + r < c1) ring[r] = ldg_stream(src + (c + PF + r) * 32 + lane);
      acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
  }
  if (acc == 0x12345678u) out[0] = acc;
}

// high occupancy grid-stride copy-like reader
__global__ void __launch_bounds__(256) occ_kernel(const uint4* __restrict__ src, size_t n16, unsigned* out) {
  unsigned acc = 0;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (; i + 3 * stride < n16; i += 4 * stride) {
    uint4 a = ldg_stream(src + i), b = ldg_stream(src + i + stride), c = ldg_stream(src + i + 2 * stride),
          d = ldg_stream(src + i + 3 * stride);
    acc ^= a.x ^ b.y ^ c.z ^ d.w ^ a.w ^ b.x ^ c.y ^ d.z;
  }
  for (; i < n16; i += stride) { uint4 a = ldg_stream(src + i); acc ^= a.x ^ a.w; }
  if (acc == 0x12345678u) out[0] = acc;
}

// cp.async (LDGSTS) ring into shared memory: 1 CTA/SM, NS stages of 16 KB (each thread copies 64 B per stage)
template <int NS>
__global__ void __launch_bounds__(256, 1) cpasync_kernel(const uint4* __restrict__ src, size_t n16, unsigned* out) {
  extern __shared__ __align__(16) uint4 sm[];
  const int tid = threadIdx.x;
  constexpr int ST16 = 1024;  // 16 KB per stage
  const size_t stages = n16 / ST16;
  const size_t per = (stages + gridDim.x - 1) / gridDim.x;
  const size_t s0 = blockIdx.x * per, s1 = s0 + per < stages ? s0 + per : stages;
  unsigned acc = 0;
  auto issue = [&](size_t s, int slot) {
    if (s < s1) {
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        uint32_t d = (uint32_t)__cvta_generic_to_shared(sm + slot * ST16 + k * 256 + tid);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src + s * ST16 + k * 256 + tid));
      }
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };
#pragma unroll
  for (int k = 0; k < NS - 1; ++k) issue(s0 + k, k);
  for (size_t s = s0; s < s1; ++s) {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(NS - 2));
    __syncthreads();
    issue(s + NS - 1, (int)((s - s0 + NS - 1) % NS));
    const uint4* st = sm + ((s - s0) % NS) * ST16;
#pragma unroll
    for (int k = 0; k < 4; ++k) { uint4 v = st[k * 256 + tid]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
  }
  if (acc == 0x12345678u) out[0] = acc;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define BW_HAS_TMA 1
#endif

// TMA bulk (cp.async.bulk) ring: thread 0 issues 16 KB bulk copies into NS stages, full/empty mbarriers.
template <int NS>
__global__ void __launch_bounds__(256, 1) tma_kernel(const uint4* __restrict__ src, size_t n16, unsigned* out) {
#ifdef BW_HAS_TMA
  extern __shared__ __align__(128) uint4 sm[];
  __shared__ __align__(8) uint64_t full[NS];
  const int tid = threadIdx.x;
  constexpr int ST16 = 1024;
  const size_t stages = n16 / ST16;
  const size_t per = (stages + gridDim.x - 1) / gridDim.x;
  const size_t s0 = blockIdx.x * per, s1 = s0 + per < stages ? s0 + per : stages;
  if (tid == 0) {
    for (int k = 0; k < NS; ++k) {
      uint32_t b = (uint32_t)__cvta_generic_to_shared(&full[k]);
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(b));
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::);
  }
  __syncthreads();
  auto issue = [&](size_t s, int slot) {
    if (s < s1) {
      uint32_t b = (uint32_t)__cvta_generic_to_shared(&full[slot]);
      uint32_t d = (uint32_t)__cvta_generic_to_shared(sm + slot * ST16);
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(b), "r"(16384));
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                   ::"r"(d), "l"(src + s * ST16), "r"(16384), "r"(b) : "memory");
    }
  };
  if (tid == 0)
    for (int k = 0; k < NS; ++k) issue(s0 + k, k);
  unsigned acc = 0;
  for (size_t s = s0; s < s1; ++s) {
    const int slot = (int)((s - s0) % NS);
    const uint32_t phase = (uint32_t)(((s - s0) / NS) & 1);
    uint32_t b = (uint32_t)__cvta_generic_to_shared(&full[slot]);
    asm volatile(
        "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}\n"
        ::"r"(b), "r"(phase));
    const uint4* st = sm + slot * ST16;
#pragma unroll
    for (int k = 0; k < 4; ++k) { uint4 v = st[k * 256 + tid]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    __syncthreads();
    if (tid == 0) issue(s + NS, slot);
  }
  if (acc == 0x12345678u) out[0] = acc;
#endif
}


template <int SB, int NS, int P, bool MIX>
__global__ void __launch_bounds__(256) tma2_kernel(const uint4* __restrict__ src, size_t n16, unsigned* out) {
#ifdef BW_HAS_TMA
  extern __shared__ __align__(128) uint4 sm[];
  __shared__ __align__(8) uint64_t full[NS];
  const int tid = threadIdx.x;
  constexpr int ST16 = SB / 16;
  const size_t stages = n16 / ST16;
  const size_t per = (stages + gridDim.x - 1) / gridDim.x;
  const size_t s0 = blockIdx.x * per, s1 = s0 + per < stages ? s0 + per : stages;
  if (tid == 0) {
    for (int k = 0; k < NS; ++k) {
      uint32_t b = (uint32_t)__cvta_generic_to_shared(&full[k]);
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(b), "r"(MIX ? 257 : 1));
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::);
  }
  __syncthreads();
  auto issue = [&](size_t s, int slot) {
    if (s < s1) {
      uint32_t b = (uint32_t)__cvta_generic_to_shared(&full[slot]);
      constexpr int TB = MIX ? SB / 2 : SB;  // bytes via TMA
      if (tid == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(b), "r"(TB));
        for (int k = 0; k < P; ++k) {
          uint32_t d = (uint32_t)__cvta_generic_to_shared(sm + slot * ST16 + k * (TB / 16 / P));
          asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                       ::"r"(d), "l"(src + s * ST16 + k * (TB / 16 / P)), "r"(TB / P), "r"(b) : "memory");
        }
      }
      if (MIX) {
        for (int i = tid; i < (SB - TB) / 16; i += 256) {
          uint32_t d = (uint32_t)__cvta_generic_to_shared(sm + slot * ST16 + TB / 16 + i);
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src + s * ST16 + TB / 16 + i));
        }
        asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(b) : "memory");
      }
    } else if (MIX) {
      uint32_t b = (uint32_t)__cvta_generic_to_shared(&full[slot]);
      (void)b;
    }
  };
  for (int k = 0; k < NS; ++k) issue(s0 + k, k);
  unsigned acc = 0;
  for (size_t s = s0; s < s1; ++s) {
    const int slot = (int)((s - s0) % NS);
    const uint32_t phase = (uint32_t)(((s - s0) / NS) & 1);
    uint32_t b = (uint32_t)__cvta_generic_to_shared(&full[slot]);
    asm volatile(
        "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}\n"
        ::"r"(b), "r"(phase));
    const uint4* st = sm + slot * ST16;
    for (int k = tid; k < ST16; k += 256) { uint4 v = st[k]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    __syncthreads();
    issue(s + NS, slot);
  }
  if (acc == 0x12345678u) out[0] = acc;
#endif
}


template <int SB, int NS, int PD>
__global__ void __launch_bounds__(256) tmapf_kernel(const uint4* __restrict__ src, size_t n16, unsigned* out) {
#ifdef BW_HAS_TMA
  extern __shared__ __align__(128) uint4 sm[];
  __shared__ __align__(8) uint64_t full[NS];
  const int tid = threadIdx.x;
  constexpr int ST16 = SB / 16;
  const size_t stages = n16 / ST16;
  const size_t per = (stages + gridDim.x - 1) / gridDim.x;
  const size_t s0 = blockIdx.x * per, s1 = s0 + per < stages ? s0 + per : stages;
  if (tid == 0) {
    for (int k = 0; k < NS; ++k) {
      uint32_t b = (uint32_t)__cvta_generic_to_shared(&full[k]);
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(b), "r"(1));
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::);
  }
  __syncthreads();
  auto issue = [&](size_t s, int slot) {
    if (tid == 0 && s < s1) {
      uint32_t b = (uint32_t)__cvta_generic_to_shared(&full[slot]);
      uint32_t d = (uint32_t)__cvta_generic_to_shared(sm + slot * ST16);
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(b), "r"(SB));
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                   ::"r"(d), "l"(src + s * ST16), "r"(SB), "r"(b) : "memory");
      if (s + PD < s1) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(src + (s + PD) * ST16), "r"(SB) : "memory");
    }
  };
  if (tid == 0) for (size_t s = s0 + NS; s < s0 + PD && s < s1; ++s)
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(src + s * ST16), "r"(SB) : "memory");
  for (int k = 0; k < NS; ++k) issue(s0 + k, k);
  unsigned acc = 0;
  for (size_t s = s0; s < s1; ++s) {
    const int slot = (int)((s - s0) % NS);
    const uint32_t phase = (uint32_t)(((s - s0) / NS) & 1);
    uint32_t b = (uint32_t)__cvta_generic_to_shared(&full[slot]);
    asm volatile(
        "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}\n"
        ::"r"(b), "r"(phase));
    const uint4* st = sm + slot * ST16;
    for (int k = tid; k < ST16; k += 256) { uint4 v = st[k]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    __syncthreads();
    issue(s + NS, slot);
  }
  if (acc == 0x12345678u) out[0] = acc;
#endif
}

}  // namespace bw

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void bw_run(torch::Tensor src, torch::Tensor out, int64_t mode, int64_t grid) {
  const uint4* p = (const uint4*)src.data_ptr();
  size_t n16 = src.numel() * src.element_size() / 16;
  unsigned* o = (unsigned*)out.data_ptr();
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  switch (mode) {
    case 0: bw::ring_kernel<16><<<(int)grid, 256, 0, st>>>(p, n16, o); break;
    case 1: bw::ring_kernel<32><<<(int)grid, 256, 0, st>>>(p, n16, o); break;
    case 2: bw::occ_kernel<<<(int)grid * 8, 256, 0, st>>>(p, n16, o); break;
    case 3: {
      size_t sm = 6 * 16384;
      cudaFuncSetAttribute(bw::cpasync_kernel<6>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::cpasync_kernel<6><<<(int)grid, 256, sm, st>>>(p, n16, o); break;
    }
    case 4: {
      size_t sm = 10 * 16384;
      cudaFuncSetAttribute(bw::cpasync_kernel<10>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::cpasync_kernel<10><<<(int)grid, 256, sm, st>>>(p, n16, o); break;
    }
    case 5: {
      size_t sm = 6 * 16384;
      cudaFuncSetAttribute(bw::tma_kernel<6>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma_kernel<6><<<(int)grid, 256, sm, st>>>(p, n16, o); break;
    }
    case 6: {
      size_t sm = 12 * 16384;
      cudaFuncSetAttribute(bw::tma_kernel<12>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma_kernel<12><<<(int)grid, 256, sm, st>>>(p, n16, o); break;
    }
    case 7: bw::ring_kernel<8><<<(int)grid * 2, 256, 0, st>>>(p, n16, o); break;

    case 8: { constexpr int SB = 32768, NS = 4; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 9: { constexpr int SB = 8192, NS = 12; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 10: { constexpr int SB = 16384, NS = 6; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 16, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 16, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 11: { constexpr int SB = 32768, NS = 4; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, true><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 12: { constexpr int SB = 16384, NS = 4; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid * 2, 256, sm, st>>>(p, n16, o); break; }
    case 14: { constexpr int SB = 16384, NS = 6; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tmapf_kernel<SB, NS, 12>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tmapf_kernel<SB, NS, 12><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 15: { constexpr int SB = 36864, NS = 5; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tmapf_kernel<SB, NS, 9>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tmapf_kernel<SB, NS, 9><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 16: { constexpr int SB = 36864, NS = 5; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 17: { constexpr int SB = 36864, NS = 4; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 18: { constexpr int SB = 36864, NS = 4; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 2, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 2, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 19: { constexpr int SB = 36864, NS = 3; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 20: { constexpr int SB = 32768, NS = 5; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 21: { constexpr int SB = 16384, NS = 8; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 22: { constexpr int SB = 16384, NS = 4; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 23: { constexpr int SB = 16384, NS = 6; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 24: { constexpr int SB = 36864, NS = 2; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid * 2, 256, sm, st>>>(p, n16, o); break; }
    case 13: { constexpr int SB = 36864, NS = 3; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, true><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
  }
}

'''

moe_CPP_SRC = r'''
#include <torch/extension.h>
void okc_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,
             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid);
void okc_run_logits(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor logits,
                    torch::Tensor rbias, torch::Tensor ls, torch::Tensor out, torch::Tensor gid, torch::Tensor wts,
                    torch::Tensor rsp, int64_t nloc, int64_t grid);
void okc_set_pdl(int64_t on);
'''
# Programmatic dependent launch for embedders whose graph runs this kernel right after other kernels (the kernel
# waits on griddepcontrol.wait before touching their outputs). Off here: one launch per call gains nothing.
moe_PDL = False
# Routing in the kernel from the router logits (the decode layer: no _route_split launch). ROUTE_LOGITS_BUILDS makes
# prepare() also compile the OKC_ROUTE_LOGITS=1 variants of the selected builds (the layer sets it before prepare);
# kernel_fn_logits(data, state) is the entry. ROUTE_LOGITS_SELFTEST runs the in-kernel selection against the host
# selection on synthetic logits in prepare() (untimed) and prints the result.
moe_ROUTE_LOGITS_BUILDS = False
moe_ROUTE_LOGITS_SELFTEST = False
moe_ROUTER_MULTIPLIER = 0.1767766952966369
# Scale decode with PRMT + IMAD instead of the smem LUT; exact only when every block-scale byte of the local experts
# is a positive normal e4m3 value (0x08..0x7E). prepare() checks the actual weights and keeps the LUT build otherwise.
moe_PRMT_SCALES = True


def moe__scales_prmt_exact(*sfs):
    for sf in sfs:
        b = sf.contiguous().view(torch.uint8)
        if bool(((b < 0x08) | (b > 0x7E)).any()):
            return False
    return True

moe__EXTS = {}
moe_MAIN_DEFINES = ("-DOKC_NW=16", "-DOKC_FAST1=0", "-DOKC_EPI4=0", "-DOKC_WGS=1")          # compile-time variant used by kernel_fn
# M <= TINY_MAX_M runs a skinny-only build with 16 warps per CTA (a job never has more than M tokens, so no fat units)
moe_TINY_DEFINES = ("-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_EPI4=0", "-DOKC_FAST1=2", "-DOKC_L2H=0", "-DOKC_NACC=4", "-DOKC_UWARP=0", "-DOKC_WS=1", "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=2", "-DOKC_FINPF=1")  # bf16 WSB at NW=8 + WSBL tail (Oct 26: bs1 +2.5% / +3.5%)
moe_TINY_MAX_M = 4
# TINY_MAX_M < M <= TINY2_MAX_M runs a second skinny-only build (empty: skipped); TINY2_MAX_M must stay <= 8
moe_TINY2_DEFINES = ('-DOKC_NW=8', '-DOKC_TINY=1', '-DOKC_CPF=0', '-DOKC_EPI4=0', '-DOKC_FAST1=2', '-DOKC_NACC=4', '-DOKC_UWARP=0', '-DOKC_WS=1', '-DOKC_WSB=1', '-DOKC_WSBL=2', '-DOKC_WSBLK=1', '-DOKC_FINPF=1')  # bf16 WSB (Oct 26: bs8 +2.0..2.5%)
moe_TINY2_MAX_M = 8
# TINY_MAX_M < M <= MID_MAX_M runs MID_DEFINES (empty: the main module)
moe_MID_DEFINES = ('-DOKC_NW=8', '-DOKC_TINY=1', '-DOKC_SPLIT=1', '-DOKC_CPF=0', '-DOKC_EPI4=0', '-DOKC_FAST1=2', '-DOKC_NACC=4', '-DOKC_UWARP=0', '-DOKC_WS=1', '-DOKC_WSB=1', '-DOKC_WSBL=2', '-DOKC_WSBLK=2', '-DOKC_FINPF=1')  # bf16 split-job MID (Oct 26): tiny-style, experts > 8 tokens split
moe_MID_MAX_M = 16  # M 17..63 runs on the WGS MAIN build (v179: faster than the v122c MID kernel at bs32)
moe_DIAG = False               # prepare() prints an H200 timing/diagnostic table (untimed; for development only)
moe_DIAG_SET = "I"             # which variant set below (a copy of this file with another set runs as a second check)
moe__WGS_SHAPES = [32, 128, 256]
# name -> defines (all DIAG_SHAPES) or {"defines": ..., "shapes": [...]}, compiled and timed in diag mode
moe_DIAG_VARIANT_SETS = {
    "A": {  # WGS attribution: weight stream / x stream / wgmma / decode, producer stall timers
        "main": moe_MAIN_DEFINES,
        "wgs_dbg": {"defines": moe_MAIN_DEFINES + ("-DOKC_DEBUG=1",), "shapes": moe__WGS_SHAPES},
        "wgs_nox": {"defines": moe_MAIN_DEFINES + ("-DOKC_WGSX=16",), "shapes": moe__WGS_SHAPES},
        "wgs_now": {"defines": moe_MAIN_DEFINES + ("-DOKC_WGSX=32",), "shapes": moe__WGS_SHAPES},
        "wgs_nomma": {"defines": moe_MAIN_DEFINES + ("-DOKC_WGSX=1",), "shapes": moe__WGS_SHAPES},
        "wgs_nodq": {"defines": moe_MAIN_DEFINES + ("-DOKC_XNODQ=1",), "shapes": moe__WGS_SHAPES},
    },
    "I": {  # WGS: per-warp mma.sync for small-N jobs (no warpgroup sync / descriptor / commit per step) vs wgmma
        "main": {"defines": moe_MAIN_DEFINES, "shapes": moe__WGS_SHAPES},
        "wgs_mma1": {"defines": moe_MAIN_DEFINES + ("-DOKC_WGSMMAN=1",), "shapes": moe__WGS_SHAPES},
        "wgs_mma2": {"defines": moe_MAIN_DEFINES + ("-DOKC_WGSMMAN=2",), "shapes": moe__WGS_SHAPES},
        "wgs_mma2_dbg": {"defines": moe_MAIN_DEFINES + ("-DOKC_WGSMMAN=2", "-DOKC_DEBUG=1"), "shapes": moe__WGS_SHAPES},
        "wgs_mma2_na2": {"defines": moe_MAIN_DEFINES + ("-DOKC_WGSMMAN=2", "-DOKC_WGSMMANA=2"), "shapes": moe__WGS_SHAPES},
    },
    "H": {  # WGS for M <= 64: FT_T=32 frees accumulator registers (x slots 16 KB): deeper wgmma pipeline / weight ring
        "main": {"defines": moe_MAIN_DEFINES, "shapes": [32, 128]},
        "wgs_ft32": {"defines": moe_MAIN_DEFINES + ("-DOKC_FT_T=32",), "shapes": [32, 128]},
        "wgs_ft32_wd2": {"defines": moe_MAIN_DEFINES + ("-DOKC_FT_T=32", "-DOKC_WGSWD=2"), "shapes": [32]},
        "wgs_ft32_gs4": {"defines": moe_MAIN_DEFINES + ("-DOKC_FT_T=32", "-DOKC_WGSGS4N=4"), "shapes": [32]},
        "wgs_ft32_ns4": {"defines": moe_MAIN_DEFINES + ("-DOKC_FT_T=32", "-DOKC_WGSNS=4"), "shapes": [32]},
        "wgs_ft32_ns4_wd2": {"defines": moe_MAIN_DEFINES + ("-DOKC_FT_T=32", "-DOKC_WGSNS=4", "-DOKC_WGSWD=2"), "shapes": [32]},
    },
    "G": {  # WGS wgmma pipeline: commit-group length for small N, wait depth, no-scale ablation (bs32/128/256)
        "main": {"defines": moe_MAIN_DEFINES, "shapes": moe__WGS_SHAPES},
        "wgs_gs4n2": {"defines": moe_MAIN_DEFINES + ("-DOKC_WGSGS4N=2",), "shapes": moe__WGS_SHAPES},
        "wgs_wd2": {"defines": moe_MAIN_DEFINES + ("-DOKC_WGSWD=2",), "shapes": moe__WGS_SHAPES},
        "wgs_gs4n2_wd2": {"defines": moe_MAIN_DEFINES + ("-DOKC_WGSGS4N=2", "-DOKC_WGSWD=2"), "shapes": moe__WGS_SHAPES},
        "wgs_gs4": {"defines": moe_MAIN_DEFINES + ("-DOKC_WGSGS=4",), "shapes": moe__WGS_SHAPES},
        "wgs_noscale": {"defines": moe_MAIN_DEFINES + ("-DOKC_XNOSC=1",), "shapes": moe__WGS_SHAPES},
    },
    "F": {  # WGS early producer start (fixed consumer/range sync), bs32/128/256
        "main": {"defines": moe_MAIN_DEFINES, "shapes": moe__WGS_SHAPES},
        "wgs_early_dbg": {"defines": moe_MAIN_DEFINES + ("-DOKC_WGSEARLY=1", "-DOKC_DEBUG=1", "-DOKC_WGSDBG=1"), "shapes": moe__WGS_SHAPES},
        "wgs_early": {"defines": moe_MAIN_DEFINES + ("-DOKC_WGSEARLY=1",), "shapes": moe__WGS_SHAPES},
    },
    "E": {  # WSDYN cause split (pool off / chunked claims / late slot release) + WGS early producer start
        "tiny": {"defines": moe_TINY_DEFINES, "shapes": [1]},
        "tiny_dyn_s30": {"defines": moe_TINY_DEFINES + ("-DOKC_WSDYN=1", "-DOKC_WSDYNS=30"), "shapes": [1]},
        "mid": {"defines": moe_MID_DEFINES, "shapes": [16]},
        "mid_dyn_s30": {"defines": moe_MID_DEFINES + ("-DOKC_WSDYN=1", "-DOKC_WSDYNS=30"), "shapes": [16]},
        "mid_dyn_c4": {"defines": moe_MID_DEFINES + ("-DOKC_WSDYN=1", "-DOKC_WSDYNC=4"), "shapes": [16]},
        "mid_dyn_late": {"defines": moe_MID_DEFINES + ("-DOKC_WSDYN=1", "-DOKC_WSDYNLATE=1"), "shapes": [16]},
        "main": {"defines": moe_MAIN_DEFINES, "shapes": moe__WGS_SHAPES},
        "wgs_early": {"defines": moe_MAIN_DEFINES + ("-DOKC_WGSEARLY=1",), "shapes": moe__WGS_SHAPES},
        "wgs_early_dbg": {"defines": moe_MAIN_DEFINES + ("-DOKC_WGSEARLY=1", "-DOKC_DEBUG=1"), "shapes": moe__WGS_SHAPES},
    },
    "D": {  # WGS: dynamic stage claiming (WGSDYN) vs stream-K static ranges, pool size / segment length, timelines
        "main": {"defines": moe_MAIN_DEFINES, "shapes": moe__WGS_SHAPES},
        "wgs_dyn": {"defines": moe_MAIN_DEFINES + ("-DOKC_WGSDYN=1",), "shapes": moe__WGS_SHAPES},
        "wgs_dyn_s1": {"defines": moe_MAIN_DEFINES + ("-DOKC_WGSDYN=1", "-DOKC_WGSDYNS=1"), "shapes": moe__WGS_SHAPES},
        "wgs_dyn_seg8": {"defines": moe_MAIN_DEFINES + ("-DOKC_WGSDYN=1", "-DOKC_WGSSEG=8"), "shapes": moe__WGS_SHAPES},
        "wgs_dyn_dbg": {"defines": moe_MAIN_DEFINES + ("-DOKC_WGSDYN=1", "-DOKC_DEBUG=1"), "shapes": moe__WGS_SHAPES},
    },
    "C": {  # tiny builds: dynamic unit claiming (WSDYN) vs static slices, pool size, timelines
        "tiny": {"defines": moe_TINY_DEFINES, "shapes": [1]},
        "tiny_dyn": {"defines": moe_TINY_DEFINES + ("-DOKC_WSDYN=1",), "shapes": [1]},
        "tiny_dyn_s1": {"defines": moe_TINY_DEFINES + ("-DOKC_WSDYN=1", "-DOKC_WSDYNS=1"), "shapes": [1]},
        "tiny_dyn_dbg": {"defines": moe_TINY_DEFINES + ("-DOKC_WSDYN=1", "-DOKC_DEBUG=1"), "shapes": [1]},
        "tiny2": {"defines": moe_TINY2_DEFINES, "shapes": [8]},
        "tiny2_dyn": {"defines": moe_TINY2_DEFINES + ("-DOKC_WSDYN=1",), "shapes": [8]},
        "mid": {"defines": moe_MID_DEFINES, "shapes": [8, 16]},
        "mid_dyn": {"defines": moe_MID_DEFINES + ("-DOKC_WSDYN=1",), "shapes": [8, 16]},
        "mid_dyn_s1": {"defines": moe_MID_DEFINES + ("-DOKC_WSDYN=1", "-DOKC_WSDYNS=1"), "shapes": [16]},
        "mid_dyn_dbg": {"defines": moe_MID_DEFINES + ("-DOKC_WSDYN=1", "-DOKC_DEBUG=1"), "shapes": [16]},
    },
    "B": {  # tiny builds: early slot release (WSER), ring depth, timelines
        "main": {"defines": moe_MAIN_DEFINES, "shapes": [8, 16]},
        "tiny": {"defines": moe_TINY_DEFINES, "shapes": [1]},
        "tiny_wser": {"defines": moe_TINY_DEFINES + ("-DOKC_WSER=1",), "shapes": [1]},
        "tiny_dbg": {"defines": moe_TINY_DEFINES + ("-DOKC_DEBUG=1",), "shapes": [1]},
        "tiny2": {"defines": moe_TINY2_DEFINES, "shapes": [8]},
        "mid": {"defines": moe_MID_DEFINES, "shapes": [8, 16]},
        "mid_wser": {"defines": moe_MID_DEFINES + ("-DOKC_WSER=1",), "shapes": [8, 16]},
        "mid_ns4": {"defines": moe_MID_DEFINES + ("-DOKC_NS=4",), "shapes": [8, 16]},
        "mid_dbg": {"defines": moe_MID_DEFINES + ("-DOKC_DEBUG=1",), "shapes": [16]},
    },
}
moe_DIAG_VARIANTS = moe_DIAG_VARIANT_SETS[moe_DIAG_SET]
moe_DIAG_SHAPES = [1, 8, 16, 32, 128, 256] if moe_DIAG_SET == "A" else moe__WGS_SHAPES if moe_DIAG_SET in ("D", "F", "G", "I") else [32, 128] if moe_DIAG_SET == "H" else [1, 16, 32, 128, 256] if moe_DIAG_SET == "E" else [1, 8, 16]
moe_DIAG_SEED = 1234
moe_DIAG_BW = moe_DIAG_SET == "A"  # also run the bandwidth microbenchmarks
moe_DIAG_BW_MODES = (8, 16, 17, 19, 24)


moe_ALT_SRCS = {}
moe_MAIN_NS = 3


def moe__ext(defines=None):
    defines = tuple(moe_MAIN_DEFINES if defines is None else defines)
    if moe_DQ7 and "-DOKC_DQ7=1" not in defines:
        defines = defines + ("-DOKC_DQ7=1",)
    if defines not in moe__EXTS:
        src = moe_CUDA_SRC
        defs = [d for d in defines if not d.startswith("@")]
        if torch.cuda.get_device_capability()[0] < 9:  # small-smem dev GPUs: keep the ring at 1 slot
            defs = [d for d in defs if not d.startswith("-DOKC_NS=") and not d.startswith("-DOKC_NSF=")]
        for d in defines:
            if d.startswith("@"):
                src = moe_ALT_SRCS[d[1:]]
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}" + ("a" if major == 9 else "")
        import hashlib  # short unique name: a ~190-char module name produced a .so without its PyInit symbol
        tag = hashlib.sha1("_".join(defines).encode()).hexdigest()[:16]
        moe__EXTS[defines] = load_inline(
            name=f"okc_moe_w4a16_v216x_{tag}_sm{arch}",
            cpp_sources=moe_CPP_SRC,
            cuda_sources=src,
            functions=["okc_run", "okc_set_pdl", "okc_run_logits"],
            extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-lineinfo",
                               *([] if any(d.startswith("-DOKC_NS=") for d in defs) else [f"-DOKC_NS={moe_MAIN_NS if major >= 9 else 1}"]),
                               f"-DOKC_FNST={4 if major >= 9 else 3}",
                               *([] if any(d.startswith("-DOKC_NSF=") for d in defs) else [f"-DOKC_NSF={3 if major >= 9 else 1}"]), *defs],
            verbose=False,
        )
        moe__EXTS[defines].okc_set_pdl(1 if moe_PDL else 0)
    return moe__EXTS[defines]


# Bit layout of a 32-bit word holding 8 E2M1 codes for one MMA k-step of one lane: register R (0..3) is
# (h, j0) = (0,0) (1,0) (0,2) (1,2): rows g+8h, k = base + 4s + j0 (+1 for the upper bf16). Code bits (m, e0, e1, s)
# land at output bits (6, 7, 8, 15) (+16 for the upper half) after the per-register rotation used by dq4().
moe__REG_HJ = [(0, 0), (1, 0), (0, 2), (1, 2)]


# DQ7: a bit layout whose decode is 7 ALU ops + 2 IMAD.SHL per 8 weights (the kernel gets -DOKC_DQ7=1)
moe_DQ7 = True


def moe__stored_pos():
    pos = {}
    for R in range(4):
        for up in range(2):
            outb = {0: 6 + 16 * up, 1: 7 + 16 * up, 2: 8 + 16 * up, 3: 15 + 16 * up}  # code bit -> output bit
            for b, o in outb.items():
                if moe_DQ7:
                    # register R reads source bit (o + d) mod 32: r0 as stored, r1 = word << 4, r2 = rotl 10,
                    # r3 = word << 6 except the e1 bits (output 8 / 24), which come from rotl 14
                    d = (0, 28, 22, 18 if b == 2 else 26)[R]
                elif R == 0:
                    d = 0
                elif R == 1:
                    d = 6
                elif R == 2:
                    d = 10
                else:
                    d = 4 if b == 0 else 28   # m via rotr 4, (e0, e1, s) via rotl 4 == rotr 28
                pos[(R, up, b)] = (o + d) % 32
    assert sorted(pos.values()) == list(range(32))
    return pos


moe__POS = moe__stored_pos()


def moe__pack_codes(codes, rows):
    """codes uint8 [R0, K/2] (one expert), rows: LongTensor row order (R). Returns int32 words
    laid out [rb][w][c][g][t][s] (uint4 per lane per chunk)."""
    c = codes.index_select(0, rows)
    R = c.shape[0]
    nib = torch.stack((c & 0xF, c >> 4), -1).reshape(R, -1)  # [R, K]
    nb = R // 16
    n = nib.view(nb, 2, 8, 8, 8, 4, 4, 4)  # rb, h, g, w, c, t, s, j
    n = n.permute(0, 3, 4, 2, 5, 6, 1, 7).to(torch.int64)  # rb, w, c, g, t, s, h, j
    word = torch.zeros(n.shape[:-2], dtype=torch.int64, device=codes.device)
    for Rg, (h, j0) in enumerate(moe__REG_HJ):
        for up in range(2):
            code = n[..., h, j0 + up]
            for b in range(4):
                word |= ((code >> b) & 1) << moe__POS[(Rg, up, b)]
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    return word.contiguous().view(-1)


def moe__pack_scales(sf, rows):
    """sf fp8 [R0, K/16] -> bytes [rb][w][g][t][c][h]."""
    s = sf.view(torch.uint8).index_select(0, rows)
    R = s.shape[0]
    nb = R // 16
    s = s.view(nb, 2, 8, 8, 8, 4)  # rb, h, g, w, c, t
    s = s.permute(0, 3, 2, 5, 4, 1)  # rb, w, g, t, c, h
    return s.contiguous().view(-1)


def moe__fat_order(codes, scales):
    """Packed per-expert codes (int32 [rb][w][c][lane][4]) and scales (uint8 [rb][w][lane][c][h]) -> fat order:
    [ub (256-row unit)][ks (16 k-stages of 256)][32 KB codes [rb16][cc][lane][16 B] + 4 KB scales [rb16][lane][cc][h]]."""
    nub = codes.numel() // (16 * 8 * 8 * 32 * 4)
    c = codes.view(nub, 16, 8, 2, 4, 32, 4).permute(0, 2, 3, 1, 4, 5, 6).contiguous()
    c = c.view(torch.uint8).view(nub, 16, 32768)
    s = scales.view(nub, 16, 8, 32, 2, 4, 2).permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(nub, 16, 4096)
    return torch.cat([c, s], dim=2).contiguous().view(-1)


def moe__unit_order(codes, scales):
    """Per 16-row block: its 32 KB of codes followed by its 4 KB of scales (one TMA copy per skinny unit)."""
    nblk = codes.numel() // (8 * 8 * 32 * 4)
    c = codes.contiguous().view(torch.uint8).view(nblk, 32768)
    s = scales.contiguous().view(nblk, 4096)
    return torch.cat([c, s], dim=1).contiguous().view(-1)


def moe__pack_codes_nat(codes, rows):
    """Like _pack_codes but with the natural mma k order inside each 16-k step (for the wgmma fat path):
    word (rb, w, c, g, t, s) holds k = 64c + 16s + 8q + 2t + e of rows g (h=0) / g+8 (h=1)."""
    c = codes.index_select(0, rows)
    R = c.shape[0]
    nib = torch.stack((c & 0xF, c >> 4), -1).reshape(R, -1)  # [R, K]
    nb = R // 16
    n = nib.view(nb, 2, 8, 8, 8, 4, 2, 4, 2)  # rb, h, g, w, c, s, q, t, e
    n = n.permute(0, 3, 4, 2, 7, 5, 1, 6, 8).to(torch.int64)  # rb, w, c, g, t, s, h, q, e
    word = torch.zeros(n.shape[:-3], dtype=torch.int64, device=codes.device)
    for Rg, (h, q) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        for up in range(2):
            code = n[..., h, q, up]
            for b in range(4):
                word |= ((code >> b) & 1) << moe__POS[(Rg, up, b)]
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    return word.contiguous().view(-1)


def moe__wgs_copy(codes, sf, rows):
    """WGS copy (RS-wgmma tiny band, moe -DOKC_WGS=1): [256-row unit][stage (16)][warpgroup q (4)][9216 B]. Warpgroup q
    owns rows 64q..64q+63 of the unit; stage st holds k16 steps st*16 + (0..15): codes [grp (4)][warp (4)][lane (32)]
    [4 words] (word (grp, j) = k16 step st*16 + grp*4 + j, natural mma k order), then scales [grp][warp][g][j][h] (the kernel's OKC_WGSSC2 layout)."""
    wn = moe__pack_codes_nat(codes, rows)
    R = rows.numel()
    nb, nub = R // 16, R // 256
    w = wn.view(nb, 8, 8, 8, 4, 4)                                    # rb16, w8, c8, g, t, s
    w = w.permute(0, 1, 2, 5, 3, 4).contiguous().view(nb, 256, 32)   # rb16, k16 step (w8, c8, s), lane (g, t)
    w = w.view(nub, 4, 4, 16, 4, 4, 32)                               # ub, q, warp, st, grp, j, lane
    w = w.permute(0, 3, 1, 4, 2, 6, 5).contiguous()                   # ub, st, q, grp, warp, lane, j
    cb = w.view(torch.uint8).view(nub, 16, 4, 8192)
    s = sf.view(torch.uint8).index_select(0, rows)                    # [R, 256]: one byte per row per 16 K
    s = s.view(nub, 4, 4, 2, 8, 16, 4, 4)                             # ub, q, warp, h, g, st, grp, j
    s = s.permute(0, 5, 1, 6, 2, 4, 7, 3).contiguous().view(nub, 16, 4, 1024)  # ub, st, q, grp, warp, g, j, h (OKC_WGSSC2)
    return torch.cat([cb, s], dim=3).contiguous().view(-1)


# the M >= 64 build when the scale bytes rule out the PRMT decode (the WGS kernel needs it): the mma.sync kernel
moe_MAIN_FALLBACK_DEFINES = ("-DOKC_CPF=0", "-DOKC_CPFF=0", "-DOKC_DYNK=7", "-DOKC_EARLYN=2", "-DOKC_EPI4=0", "-DOKC_FCOSTB=35",
                         "-DOKC_FCOSTBB=36", "-DOKC_FINCOL=0", "-DOKC_FPF=5")


def moe__main_defines(sfx):
    if not sfx and "-DOKC_WGS=1" in tuple(moe_MAIN_DEFINES):
        return moe_MAIN_FALLBACK_DEFINES
    return moe_MAIN_DEFINES


def moe__wgs_on():
    return any("-DOKC_WGS=1" in tuple(d) for d in (moe_MAIN_DEFINES, moe_TINY_DEFINES, moe_TINY2_DEFINES, moe_MID_DEFINES))


def moe_prepare(weights):
    dev = weights["w13"].device
    lo, cnt = 0, 32
    I = weights["w13"].shape[1] // 2
    H = weights["w2"].shape[1]
    assert I == 4096 and H == 4096
    assert weights["post_norm_w"].dtype == torch.bfloat16  # kept as given (TASK.md: BF16 weights stay BF16)
    assert weights["w13_gs"].dtype == torch.float32 and weights["w2_gs"].dtype == torch.float32  # stored as given
    rb = torch.arange(I // 8, device=dev)
    g = torch.arange(8, device=dev)
    rowsA = torch.stack((8 * rb[:, None] + g[None, :], I + 8 * rb[:, None] + g[None, :]), 1).reshape(-1)
    rowsB = torch.arange(H, device=dev)
    wA, sA, wB, sB = [], [], [], []
    for e in range(lo, lo + cnt):
        wA.append(moe__pack_codes(weights["w13"][e], rowsA))
        sA.append(moe__pack_scales(weights["w13_sf"][e], rowsA))
        wB.append(moe__pack_codes(weights["w2"][e], rowsB))
        sB.append(moe__pack_scales(weights["w2_sf"][e], rowsB))
    lut = (torch.arange(256, dtype=torch.int32, device=dev).to(torch.uint8).view(torch.float8_e4m3fn).float()
           * 2.0**64).to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    lut = (lut | (lut << 16)).to(torch.int32)
    state = {
        "w": [torch.cat(wA), torch.cat(sA), weights["w13_gs"][lo:lo + cnt].contiguous(),
              torch.cat(wB), torch.cat(sB), weights["w2_gs"][lo:lo + cnt].contiguous(),
              lut.contiguous(), weights["post_norm_w"].contiguous(),  # bf16 as given; the kernel forms 1 + w in FP32
              torch.cat([moe__fat_order(a, b) for a, b in zip(wA, sA)]), torch.cat([moe__fat_order(a, b) for a, b in zip(wB, sB)]),
              torch.cat([moe__unit_order(a, b) for a, b in zip(wA, sA)]), torch.cat([moe__unit_order(a, b) for a, b in zip(wB, sB)])],
        "lo": lo, "cnt": cnt, "scratch": moe__make_scratch(dev),
        "grid": torch.cuda.get_device_properties(dev).multi_processor_count,
    }
    if moe__wgs_on():  # w[12], w[13]: the WGS copies (one more copy of the weights, laid out per 64-row unit)
        state["w"].append(torch.cat([moe__wgs_copy(weights["w13"][e], weights["w13_sf"][e], rowsA) for e in range(lo, lo + cnt)]))
        state["w"].append(torch.cat([moe__wgs_copy(weights["w2"][e], weights["w2_sf"][e], rowsB) for e in range(lo, lo + cnt)]))
    prmt = moe_PRMT_SCALES and moe__scales_prmt_exact(weights["w13_sf"][lo:lo + cnt], weights["w2_sf"][lo:lo + cnt])
    state["sfx"] = ("-DOKC_PRMTSC=1",) if prmt else ()
    mods = [tuple(moe__main_defines(state["sfx"])) + state["sfx"]]
    for d in (moe_TINY_DEFINES, moe_TINY2_DEFINES, moe_MID_DEFINES):
        if d and (state["sfx"] or "-DOKC_WGS=1" not in tuple(d)):  # the WGS build needs the PRMT scale decode
            mods.append(tuple(d) + state["sfx"])
    state["wts"] = torch.zeros(moe_EMAX, dtype=torch.float32, device=dev)  # in-kernel routing: the route weights (CTA 0)
    state["norsp"] = torch.empty(0, dtype=torch.float32, device=dev)  # (no pre-expert RMS partials)
    if moe_ROUTE_LOGITS_BUILDS or moe_ROUTE_LOGITS_SELFTEST:
        mods += [moe__logits_defines(d) for d in list(mods)]
    extra = []
    if moe_DIAG:  # the diagnostic variants compile alongside (one pool; nvcc runs in subprocesses); a variant that fails
        for v in moe_DIAG_VARIANTS.values():  # to build only loses its column
            d = tuple(v["defines"] if isinstance(v, dict) else v) + state["sfx"]
            if d not in mods and d not in extra:
                extra.append(d)

    def _build(d):
        try:
            moe__ext(d)
        except Exception as ex:
            if d in mods:
                raise
            print(f"[diag] variant {d} failed to build: {str(ex)[-300:]}", flush=True)

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_build, mods + extra))
    torch.cuda.synchronize()
    if moe_ROUTE_LOGITS_SELFTEST:
        try:
            moe__route_logits_selftest(state)
        except Exception as ex:  # a self-test must never break the run
            print(f"[route-logits] self-test failed: {ex}", flush=True)
    if moe_DIAG:
        moe__diag(state, weights)
    return state


def moe__logits_defines(d):
    return tuple(d) + ("-DOKC_ROUTE_LOGITS=1",)


def moe__select_ext(M, sfx, logits=False):
    """The build for M tokens (the shipped policy), optionally its OKC_ROUTE_LOGITS variant."""
    if moe_TINY_DEFINES and M <= moe_TINY_MAX_M:
        d = tuple(moe_TINY_DEFINES) + sfx
    elif moe_TINY2_DEFINES and M <= moe_TINY2_MAX_M and (sfx or "-DOKC_WGS=1" not in tuple(moe_TINY2_DEFINES)):
        d = tuple(moe_TINY2_DEFINES) + sfx  # (the WGS build needs the PRMT scale decode: else MID/MAIN)
    elif moe_MID_DEFINES and M <= moe_MID_MAX_M:
        d = tuple(moe_MID_DEFINES) + sfx
    else:
        d = tuple(moe__main_defines(sfx)) + sfx
    return moe__ext(moe__logits_defines(d) if logits else d)


def moe_kernel_fn_logits(data, state):
    """The MoE with the routing in the kernel. data: hidden_states bf16 [T, 4096]; router_logits fp32 [T, 256]
    (contiguous); router_bias fp32 [256]; local_start: an int32 device tensor [1] (graph-safe) or an int (untimed use);
    optional topk_out int32 [T, 8]; optional pre_norm_partials fp32 [T, 128] (partial sums of squares of the pooled row
    over 128 slices: the per-token pre-expert RMS scale rs_t = rsqrt(sum / 4096 + 1e-5) is then applied in GEMM1's
    epilogue, so hidden_states carries the column gain only, T <= 256). Returns (out bf16 [T, 4096], topk_ids int32 [T, 8]: the chosen GLOBAL expert ids,
    slots in descending biased-score order). The selection and the route weights follow the decode layer's
    _route_split kernel (sigmoid(logit * ROUTER_MULTIPLIER) + bias, top-8 with ties to the lower id, weights
    renormalized with eps 1e-15; local = (id - local_start) mod 256 < local_count). Needs the OKC_ROUTE_LOGITS builds
    (ROUTE_LOGITS_BUILDS = True before prepare())."""
    x = data["hidden_states"].contiguous()
    M = x.shape[0]
    logits = data["router_logits"]
    if logits.dtype != torch.float32 or not logits.is_contiguous():
        logits = logits.float().contiguous()
    bias = data["router_bias"]
    if bias.dtype != torch.float32 or not bias.is_contiguous():
        bias = bias.float().contiguous()
    ls = data.get("local_start", 0)
    if not torch.is_tensor(ls):
        ls = torch.tensor([int(ls)], dtype=torch.int32, device=x.device)
    elif ls.dtype != torch.int32:
        ls = ls.to(torch.int32)
    cnt = data.get("config", {}).get("local_count", state["cnt"])
    out = torch.empty((M, 4096), dtype=torch.bfloat16, device=x.device)
    gid = data.get("topk_out")
    if gid is None:
        gid = torch.empty((M, 8), dtype=torch.int32, device=x.device)
    rsp = data.get("pre_norm_partials")
    if rsp is None:
        rsp = state["norsp"]
    elif rsp.dtype != torch.float32 or not rsp.is_contiguous():
        rsp = rsp.float().contiguous()
    ext = moe__select_ext(M, state.get("sfx", ()), logits=True)
    for m0 in range(0, M, moe_MMAX):
        m1 = min(M, m0 + moe_MMAX)
        ext.okc_run_logits(state["w"], state["scratch"], x[m0:m1], logits[m0:m1], bias, ls, out[m0:m1], gid[m0:m1],
                           state["wts"], rsp[m0:m1] if rsp.numel() else rsp, cnt, state["grid"])
    return out, gid


def moe__route_logits_selftest(state):
    """In-kernel routing vs the host selection: synthetic logits whose top-8 reproduce _gen_input's ids (so the MoE
    output can be compared with kernel_fn on the same routes), random logits (ids / weights vs torch), timing."""
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(4321)
    bias = (0.05 * torch.randn((256,), generator=gen, device=dev)).float().contiguous()
    ls = torch.tensor([0], dtype=torch.int32, device=dev)
    for bs in (1, 4, 8, 16, 32, 33, 128, 256):
        d = moe__gen_input(bs, 777 + bs, dev)
        ids = d["topk_ids"].long()
        # logits: the given ids get large, slot-ordered logits (their biased scores dominate every other expert)
        logits = 0.5 * torch.randn((bs, 256), generator=gen, device=dev)
        logits.scatter_(1, ids, (20.0 - 0.5 * torch.arange(8, device=dev, dtype=torch.float32)).expand(bs, 8))
        logits = logits.float().contiguous()
        sc = torch.sigmoid(logits * moe_ROUTER_MULTIPLIER)
        sel = sc + bias
        ref_ids = sel.topk(8, dim=-1).indices
        picked = sc.gather(1, ref_ids)
        ref_w = picked / (picked.sum(-1, keepdim=True) + 1e-15)
        out2, gid = moe_kernel_fn_logits({"hidden_states": d["hidden_states"], "router_logits": logits, "router_bias": bias,
                                      "local_start": ls, "config": {"local_count": 32}}, state)
        torch.cuda.synchronize()
        kw = state["wts"][:bs * 8].view(bs, 8)
        same_ids = bool((gid.long() == ref_ids).all())
        same_set = bool((gid.long().sort(-1).values == ids.sort(-1).values).all())
        werr = float((kw - ref_w).abs().max())
        # the MoE output with the same routes through the plain entry (host ids / weights)
        lids = ((gid.long() - 0) % 256)
        lids = torch.where(lids < 32, lids, torch.full_like(lids, -1)).to(torch.int32)
        out1 = moe_kernel_fn({"hidden_states": d["hidden_states"], "topk_ids": gid, "topk_weights": kw.contiguous(),
                          "config": {"local_start": 0, "local_count": 32}}, state)
        torch.cuda.synchronize()
        oerr = float((out2.float() - out1.float()).norm() / out1.float().norm().clamp(min=1e-20))
        # per-token pre-expert RMS scale in the kernel (partials) vs the scale applied to x on the host
        rerr = float("nan")
        if bs <= 256:
            pooled = 3.0 * torch.randn((bs, 4096), generator=gen, device=dev)
            part = (pooled * pooled).view(bs, 128, 32).sum(-1).contiguous()
            rs = torch.rsqrt(part.sum(-1, keepdim=True) / 4096.0 + 1e-5)
            o_k, _ = moe_kernel_fn_logits({"hidden_states": pooled.to(torch.bfloat16), "router_logits": logits, "router_bias": bias,
                                       "local_start": ls, "pre_norm_partials": part, "config": {"local_count": 32}}, state)
            o_h, _ = moe_kernel_fn_logits({"hidden_states": (pooled * rs).to(torch.bfloat16), "router_logits": logits,
                                       "router_bias": bias, "local_start": ls, "config": {"local_count": 32}}, state)
            torch.cuda.synchronize()
            rerr = float((o_k.float() - o_h.float()).norm() / o_h.float().norm().clamp(min=1e-20))
        # random logits (no structure): ids vs torch's top-8 (ties improbable), weights
        logits2 = (3.0 * torch.randn((bs, 256), generator=gen, device=dev)).float().contiguous()
        sc2 = torch.sigmoid(logits2 * moe_ROUTER_MULTIPLIER)
        ref2 = (sc2 + bias).topk(8, dim=-1).indices
        _, gid2 = moe_kernel_fn_logits({"hidden_states": d["hidden_states"], "router_logits": logits2, "router_bias": bias,
                                    "local_start": ls, "config": {"local_count": 32}}, state)
        torch.cuda.synchronize()
        nd = int((gid2.long() != ref2).any(-1).sum())
        # timing: in-kernel routing vs the plain entry (graph replays, cold L2)
        tw = ref_w.float().contiguous()

        def f_plain():
            moe_kernel_fn({"hidden_states": d["hidden_states"], "topk_ids": gid, "topk_weights": tw,
                       "config": {"local_start": 0, "local_count": 32}}, state)

        def f_logits():
            moe_kernel_fn_logits({"hidden_states": d["hidden_states"], "router_logits": logits, "router_bias": bias,
                              "local_start": ls, "config": {"local_count": 32}}, state)

        t_plain, _ = moe__time_graph(f_plain)
        t_logits, _ = moe__time_graph(f_logits)
        print(f"[route-logits] bs={bs:3d}: ids == host top-8 {same_ids} (same set as the input routing {same_set}), "
              f"max |w - w_ref| {werr:.1e}, out rel L2 vs plain entry {oerr:.1e}, rs-in-epilogue vs host-scaled x {rerr:.1e}, "
              f"random-logit tokens differing {nd}/{bs}, "
              f"time plain {t_plain:6.1f} us vs in-kernel routing {t_logits:6.1f} us", flush=True)


moe_MMAX = 384
moe_EMAX = moe_MMAX * 8
moe_JMAX = 32 + moe_EMAX // 64 + 8


def moe__wgs_fixf():
    """moe WGS_FIXF: stream-K fixups + doneAs after ybuf (zero between calls); unit-indexed fixup slots (JMAX * 32) only
    when a selected build uses OKC_WGSDYN, else the 160 CTA-indexed slots."""
    sets = [moe_MAIN_DEFINES, moe_TINY_DEFINES, moe_TINY2_DEFINES, moe_MID_DEFINES]
    if moe_DIAG:
        sets += [v["defines"] if isinstance(v, dict) else v for v in moe_DIAG_VARIANTS.values()]
    slots = moe_JMAX * 32 if any("-DOKC_WGSDYN=1" in tuple(d) for d in sets) else 160
    return 2 * slots * 4 * 4096 + 2 * slots * 4 + 136 * 16


moe_WGS_FIXF = moe__wgs_fixf()
moe_WGS_XROWS = moe_EMAX + 8 * (32 + moe_EMAX // 32 + 8)  # >= moe WGS_XROWS for FT_T >= 32


def moe__make_scratch(dev):
    return [torch.zeros(8 + 1024 + moe_EMAX, dtype=torch.int32, device=dev),  # counters + doneA (>= any build's JMAX) + ss copy 1
            torch.zeros(moe_EMAX + moe_EMAX * 4096 + moe_WGS_FIXF, dtype=torch.float32, device=dev),
            torch.zeros((moe_EMAX + 2 * moe_WGS_XROWS) * 4096, dtype=torch.bfloat16, device=dev)]  # inter, then WGS xA / xB


def moe_kernel_fn(data, state):
    cfg = data["config"]
    x = data["hidden_states"].contiguous()
    M = x.shape[0]
    assert cfg["local_start"] == state["lo"] and cfg["local_count"] == state["cnt"]
    ids = data["topk_ids"].to(torch.int32).contiguous()
    if ids.data_ptr() & 15:  # the WGS routing loads topk_ids 16 B at a time (an offset view gets an aligned copy)
        ids = ids.clone()
    tw = data["topk_weights"].to(torch.float32).contiguous()
    out = torch.empty((M, 4096), dtype=torch.bfloat16, device=x.device)
    sfx = state.get("sfx", ())
    if moe_TINY_DEFINES and M <= moe_TINY_MAX_M:
        ext = moe__ext(tuple(moe_TINY_DEFINES) + sfx)
    elif moe_TINY2_DEFINES and M <= moe_TINY2_MAX_M and (sfx or "-DOKC_WGS=1" not in tuple(moe_TINY2_DEFINES)):
        ext = moe__ext(tuple(moe_TINY2_DEFINES) + sfx)  # (the WGS build needs the PRMT scale decode: else MID/MAIN)
    elif moe_MID_DEFINES and M <= moe_MID_MAX_M:
        ext = moe__ext(tuple(moe_MID_DEFINES) + sfx)
    else:
        ext = moe__ext(tuple(moe__main_defines(sfx)) + sfx)
    for m0 in range(0, M, moe_MMAX):
        m1 = min(M, m0 + moe_MMAX)
        ext.okc_run(state["w"], state["scratch"], x[m0:m1], ids[m0:m1], tw[m0:m1], out[m0:m1],
                    state["lo"], state["cnt"], state["grid"])
    return out


# ---------------------------------------------------------------------------------------------- diagnostics
def moe__gen_input(bs, seed, dev):
    """Same distribution as the task's generate_input (normal dist, busiest-GPU relabel), written here."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn((bs, 4096), generator=gen, device=dev).to(torch.bfloat16)
    scores = torch.rand((bs, 256), generator=gen, device=dev)
    z = torch.randn((256,), generator=gen, device=dev)
    u = torch.rand((256,), generator=gen, device=dev).clamp(min=1e-12)
    scores = scores + 0.03 * z / (-u.log()).sqrt()
    routed = scores.topk(8, dim=-1)
    ids = routed.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    return {"hidden_states": x, "topk_ids": ids.to(torch.int32),
            "topk_weights": (routed.values / routed.values.sum(-1, keepdim=True)).float(),
            "config": {"bs": bs, "d_hidden": 4096, "d_expert": 4096, "n_experts": 256, "local_start": 0,
                       "local_count": 32, "top_k": 8, "sharding": "ep8"}}


def moe__time_graph(fn, reps=40):
    dev = torch.device("cuda")
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=dev)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(reps):
        flush.add_(1)
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0)
    ts.sort()
    del g
    return ts[len(ts) // 2], ts[0]


def moe__diag_bw(state):
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}" + ("a" if major == 9 else "")
    ext = load_inline(name=f"okc_bw_v216x_sm{arch}", cpp_sources="#include <torch/extension.h>\nvoid bw_run(torch::Tensor src, torch::Tensor out, int64_t mode, int64_t grid);",
                      cuda_sources=moe_BW_SRC, functions=["bw_run"],
                      extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"], verbose=False)
    src = state["w"][0]
    nbytes = src.numel() * src.element_size()
    out = torch.zeros(4, dtype=torch.int32, device=src.device)
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=src.device)
    names = {0: "ring16 1cta/sm", 1: "ring32 1cta/sm", 2: "occ 8cta/sm", 3: "cp.async 6x16KB", 4: "cp.async 10x16KB",
             5: "tma 6x16KB", 6: "tma 12x16KB", 7: "ring8 2cta/sm", 8: "tma 4x32KB", 9: "tma 12x8KB",
             10: "tma 6x16KB as 16x1KB", 11: "mix 4x32KB half tma", 12: "tma 4x16KB 2cta/sm", 13: "mix 3x36KB half tma",
             14: "tma 6x16KB + L2 pf 12 ahead", 15: "tma 5x36KB + L2 pf 9 ahead", 16: "tma 5x36KB",
             17: "tma 4x36KB", 18: "tma 4x36KB as 2x18KB", 19: "tma 3x36KB", 20: "tma 5x32KB", 21: "tma2 8x16KB",
             22: "tma2 4x16KB", 23: "tma2 6x16KB", 24: "tma 2x36KB 2cta/sm"}
    for mode in moe_DIAG_BW_MODES:
        if mode >= 5 and mode != 7 and major < 9:
            continue
        ts = []
        for _ in range(5):
            flush.add_(1)
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            ext.bw_run(src, out, mode, state["grid"])
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1000.0)
        ts.sort()
        print(f"[diag-bw] {names[mode]:18s}: {ts[2]:8.1f} us  {nbytes / ts[2] / 1e6:6.2f} TB/s", flush=True)


def moe__diag(state, weights):
    """Untimed development table (DIAG): every variant in DIAG_VARIANTS is compiled (concurrently), checked against the
    shipped build on the same input (relative L2) and timed as a CUDA graph replay with a cold L2, per shape."""
    import time
    dev = weights["w13"].device
    t0 = time.time()
    if moe_DIAG_BW:
        try:
            moe__diag_bw(state)
        except Exception as ex:  # diagnostics must never break the run
            print(f"[diag-bw] failed: {ex}", flush=True)
    sfx = state.get("sfx", ())
    try:  # block-scale statistics (E4M3 bytes): share of powers of two (mantissa 0), exponent span
        for nm in ("w13_sf", "w2_sf"):
            b = weights[nm][:32].contiguous().view(torch.uint8)
            pow2 = float(((b & 7) == 0).float().mean())
            ex = (b.int() >> 3) & 15
            print(f"[diag] {nm}: power-of-two scales {pow2:.3f}, exponent min/max {int(ex.min())}/{int(ex.max())}, "
                  f"bytes min/max {int(b.min()):#x}/{int(b.max()):#x}", flush=True)
    except Exception as ex:
        print(f"[diag] scale stats failed: {ex}", flush=True)
    shapes = list(moe_DIAG_SHAPES)
    variants = {}  # name -> (defines, shapes)
    for name, v in moe_DIAG_VARIANTS.items():
        if isinstance(v, dict):
            variants[name] = (tuple(v["defines"]) + sfx, list(v.get("shapes") or shapes))
        else:
            variants[name] = (tuple(v) + sfx, shapes)
    # compile the variants concurrently (separate extension dirs; nvcc runs in subprocesses)
    import concurrent.futures
    todo = []
    for d, _ in variants.values():
        if d not in todo and (d + (("-DOKC_DQ7=1",) if moe_DQ7 and "-DOKC_DQ7=1" not in d else ())) not in moe__EXTS:
            todo.append(d)
    def _try_ext(d):
        try:
            moe__ext(d)
        except Exception as ex:
            print(f"[diag] variant {d} failed to build: {str(ex)[-200:]}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_try_ext, todo))
    print(f"[diag] compiled {len(todo)} extra variants in {time.time() - t0:.0f}s", flush=True)

    def _built(d):  # (an unbuilt variant only loses its column)
        return (d + (("-DOKC_DQ7=1",) if moe_DQ7 and "-DOKC_DQ7=1" not in d else ())) in moe__EXTS
    for bs in shapes:
        d = moe__gen_input(bs, moe_DIAG_SEED + bs, dev)
        ids = d["topk_ids"]
        touched = int(torch.unique(ids[(ids >= 0) & (ids < 32)]).numel())
        ref = moe_kernel_fn(d, state)  # the shipped policy's output for this input
        torch.cuda.synchronize()
        row = [f"bs={bs:3d} touched={touched:2d}"]
        for name, (defs, vshapes) in variants.items():
            if bs not in vshapes or not _built(defs):
                continue
            try:
                ext = moe__ext(defs)
                x = d["hidden_states"]
                idt = ids.to(torch.int32).contiguous()
                tw = d["topk_weights"].float().contiguous()
                out = torch.empty((bs, 4096), dtype=torch.bfloat16, device=dev)

                def fn():
                    ext.okc_run(state["w"], state["scratch"], x, idt, tw, out, 0, 32, state["grid"])

                fn()
                torch.cuda.synchronize()
                err = float((out.float() - ref.float()).norm() / ref.float().norm().clamp(min=1e-20))
                med, mn = moe__time_graph(fn)
                row.append(f"{name}: {med:7.1f}us ({touched * 28.31 / med:4.2f} TB/s, min {mn:6.1f}, err {err:.1e})")
            except Exception as ex:
                row.append(f"{name}: FAILED {str(ex)[:80]}")
                torch.cuda.synchronize()
        print("[diag] " + "  ".join(row), flush=True)
    # timeline of the debug variants (-DOKC_DEBUG=1): per-CTA %globaltimer stamps, min-max over CTAs
    for dbgname in [k for k, (v, _) in variants.items() if "-DOKC_DEBUG=1" in v and _built(v)]:
        defs, vshapes = variants[dbgname]
        ext = moe__ext(defs)
        G = state["grid"]
        dbg = torch.zeros(G * 8 + G * 8 * 4, dtype=torch.int64, device=dev)
        sc = list(state["scratch"]) + [dbg]
        names = ["start", "routed", "A done", "B done", "barrier", "final", "ranges", "gathered"]
        for bs in vshapes:
            d = moe__gen_input(bs, moe_DIAG_SEED + bs, dev)
            x = d["hidden_states"]
            idt = d["topk_ids"].to(torch.int32).contiguous()
            tw = d["topk_weights"].float().contiguous()
            out = torch.empty((bs, 4096), dtype=torch.bfloat16, device=dev)
            flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=dev)
            for _ in range(3):
                flush.add_(1)
                dbg.zero_()
                torch.cuda.synchronize()
                ext.okc_run(state["w"], sc, x, idt, tw, out, 0, 32, G)
                torch.cuda.synchronize()
            tt = dbg[:G * 8].view(G, 8).cpu().double()
            t00 = tt[:, 0][tt[:, 0] > 0].min()
            parts = []
            for k, n in enumerate(names):
                col = tt[:, k][tt[:, k] > 0]
                if col.numel():
                    parts.append(f"{n} {(col.min() - t00) / 1000:.1f}-{(col.max() - t00) / 1000:.1f}")
            if "-DOKC_WGS=1" in defs:  # producer stall counters (us): x gating phase A / B, phase-B stalls > 0.5 us, x-ring empty, weight-ring empty
                xs_ = dbg[G * 8:G * 12].view(G, 4).cpu().double() / 1000
                ws_ = dbg[G * 12:G * 13].cpu().double() / 1000
                parts.append(f"xgateA mean/max {xs_[:, 0].mean():.1f}/{xs_[:, 0].max():.1f}  xgateB {xs_[:, 1].mean():.1f}/{xs_[:, 1].max():.1f}"
                             f" (stalls {xs_[:, 2].mean() * 1000:.0f}/{xs_[:, 2].max() * 1000:.0f})  xring {xs_[:, 3].mean():.1f}/{xs_[:, 3].max():.1f}  wring {ws_.mean():.1f}/{ws_.max():.1f}")
            print(f"[diag] timeline {dbgname} bs={bs}: " + "  ".join(parts), flush=True)
    print(f"[diag] total {time.time() - t0:.0f}s", flush=True)
# ---- end of component: moe ----


# ==============================================================================================================
# component: proj (muse_decode_layer/proj_fork.py); every module-level name of that file carries the prefix `bfp_`
# ==============================================================================================================
"""Dense projections for Muse Spark Open: skinny swap-AB tensor-core GEMV/GEMM with
losslessly compressed BF16 weights.

out[m, n] = x[m, k] @ W[n, k]^T.  For m <= 64 every weight byte is streamed once from HBM
straight into registers, in mma.m16n8k16 A-fragment order (prepared once), and multiplied
against x (staged in shared memory) on BF16 tensor cores with FP32 accumulation.

Lossless 12-bit weights (BF16 projections): each BF16 weight is stored as its low byte
(exponent bit 0 + 7 mantissa bits, raw) and a 4-bit code for its high byte (sign + exponent
bits 7..1, as an offset into an 8-value window chosen per matrix).  The kernel rebuilds the
exact BF16 bits in registers (a few integer ops per weight).  The rare weights whose exponent
falls outside the window (tiny magnitudes) are exact-corrected: the kernel adds
x[:, k] * (w - w_decoded) for each of them (both factors exact in FP32), so every output
equals the product with the original weights up to FP32 accumulation order.

Split-K across warps (shared-memory reduction) and across CTAs (FP32 partials + a
self-resetting semaphore; the last CTA of a row block reduces) keeps ~128 CTAs streaming on
every shape.  The router's FP32 weights run as a BF16 hi + lo pair (concatenated along K)
with FP32 accumulation, as SGLang does above 8 tokens.
"""

from typing import TypedDict

import torch
import torch.nn.functional as F
import sys

from torch.utils.cpp_extension import load_inline
import os as _os
bfp__SK2_DELAY = int(_os.environ.get("SK2_DELAY_ESC", "0"))   # debug only: delay (ns) of escape warp 0 + ring spin-limit trap


class bfp_Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class bfp_Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class bfp_Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: bfp_Config


bfp_CUDA_SRC = r"""
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ uint4 ldg_stream(const uint4* p) {
  uint4 r;
#if __CUDA_ARCH__ >= 800
  uint64_t pol; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p), "l"(pol));
#else
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
#endif
  return r;
}

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}

__device__ __forceinline__ uint32_t lop3_sel(uint32_t a, uint32_t b, uint32_t c) {
  // c ? b : a, bitwise
  uint32_t d;
  asm("lop3.b32 %0, %1, %2, %3, 0xD8;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

// Decode 8 weights: low bytes of weights 0..3 in lw0, 4..7 in lw1, codes in nw
// (low nibble of byte b -> weight b, high nibble -> weight 4+b; code = s<<3 | o).
// Output: 4 bf16x2 registers r[0..3] = (w0,w1) (w2,w3) (w4,w5) (w6,w7).
__device__ __forceinline__ void decode8(uint32_t lw0, uint32_t lw1, uint32_t nw, uint32_t base4, uint32_t* r) {
  uint32_t ha = lop3_sel(nw & 0x07070707u, nw << 4, 0x80808080u) + base4;
  uint32_t hb = lop3_sel((nw >> 4) & 0x07070707u, nw, 0x80808080u) + base4;
  r[0] = prmt(lw0, ha, 0x5140);
  r[1] = prmt(lw0, ha, 0x7362);
  r[2] = prmt(lw1, hb, 0x5140);
  r[3] = prmt(lw1, hb, 0x7362);
}

template <typename T> __device__ __forceinline__ T cvt_out(float v);
template <> __device__ __forceinline__ float cvt_out<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 cvt_out<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

// Weights, uncompressed (COMP=0): [num_rg][S][32 lanes][16 B] A fragments, S = Kp / 16.
// Weights, compressed (COMP=1): [num_rg][S/4 units][3 parts][32 lanes][16 B]: parts 0,1 = low
//   bytes of the lane's 32 weights of 4 k-steps, part 2 = their 32 codes.
// x: [M][Kx] bf16.  Weight column k uses x column (k % Kx) (router hi|lo concat).
// CTA (rb, sp): row groups rb*RG .. +RG, k range [sp*Kc, (sp+1)*Kc).
// Warp w: row group rb*RG + w / WS, k sub-range (w % WS).
// D: prefetch depth in loads of 16 B per lane (k-steps if !COMP, unit parts if COMP).
@@MBAR@@
// cross-CTA spins are bounded (POD.md Oct 26 17:50) in the measured-free form (attention v298, POD.md 18:31): count
// polls, no timer read per poll, and past 2^24 polls call an out-of-line trap (keeps the hot path's codegen unchanged)
__device__ __noinline__ void okc_spin_trap() { asm volatile("trap;"); }
__device__ __forceinline__ void okc_spin_ge(const int* p, int target, int ns) {
  for (unsigned it = 0;; ++it) {
    if (*((volatile const int*)p) - target >= 0) return;
    if (ns > 0) __nanosleep(ns);
    if (it >= (1u << 24)) okc_spin_trap();
  }
}
@@BODY@@
"""

bfp_MBAR_HELPERS = r"""
// ------------------------------------------------------------------ mbarrier / bulk copy helpers
__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ bool mbar_test(uint32_t bar, uint32_t parity) {
  uint32_t ok;
#if __CUDA_ARCH__ >= 900
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
#else
  asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
#endif
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  while (!mbar_test(bar, parity)) {}
}
// wait with a suspend-time hint: the warp sleeps in hardware until the phase completes instead of re-polling
__device__ __forceinline__ void mbar_wait_sleep(uint32_t bar, uint32_t parity) {
#if __CUDA_ARCH__ >= 900
  uint32_t ok = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity), "r"(0x989680u) : "memory");
  } while (!ok);
#else
  mbar_wait(bar, parity);
#endif
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared.b64 st, [%0];\n}\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void named_bar(int id, int n) {
  asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory");
}
#if __CUDA_ARCH__ >= 900
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared.b64 st, [%0], %1;\n}\n" :: "r"(bar), "r"(bytes) : "memory");
}
// L2 policies (v86y = v86r + bf16_proj v131's hints): weight streams are read once per call -> evict_first; x tiles
// are re-read by every CTA -> evict_last
__device__ __forceinline__ uint64_t l2_pol_ef() {
  uint64_t p; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p)); return p;
}
__device__ __forceinline__ uint64_t l2_pol_el() {
  uint64_t p; asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p)); return p;
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(l2_pol_ef()) : "memory");
}
// wait on a local mbarrier whose phase another CTA of the cluster completes (acquire at cluster scope)
__device__ __forceinline__ void mbar_wait_cluster(uint32_t bar, uint32_t parity) {
  uint32_t ok = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n"
                 " selp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  } while (!ok);
}
#endif

"""

bfp_WG_HELPERS = r"""
// ---------------------------------------------------------------- wgmma helpers (sm_90a)
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }

// smem matrix descriptor: K-major, 128B swizzle, SBO = 1024 B (8 rows x 128 B)
__device__ __forceinline__ uint64_t make_desc_sw128(uint32_t saddr) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr & 0x3FFFF) >> 4);
  d |= (uint64_t)1 << 16;                 // LBO (ignored for swizzled K-major)
  d |= (uint64_t)(1024 >> 4) << 32;       // SBO
  d |= (uint64_t)1 << 62;                 // 128B swizzle
  return d;
}


"""


def bfp_wgmma_defs(bns):
    """Generate wgmma_call overloads for m64nBNk16 bf16 with A from registers."""
    out = []
    for bn in bns:
        nacc = bn // 2
        regs = ", ".join(f"%{i}" for i in range(nacc))
        a0 = nacc
        cons = ", ".join(f'"+f"(d[{i}])' for i in range(nacc))
        pi = a0 + 5
        out.append(f"""
template <int BN_> __device__ __forceinline__ typename std::enable_if<BN_ == {bn}>::type
wgmma_bn(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {{
  asm volatile("{{\\n"
      ".reg .pred p;\\n"
      "setp.ne.b32 p, %{pi}, 0;\\n"
      "wgmma.mma_async.sync.aligned.m64n{bn}k16.f32.bf16.bf16 "
      "{{{regs}}}, {{%{a0}, %{a0+1}, %{a0+2}, %{a0+3}}}, %{a0+4}, p, 1, 1, 0;\\n"
      "}}\\n"
      : {cons}
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}}""")
    return "\n".join(out)


bfp_TMA_SRC = r"""
// Weights: per-warp contiguous layout, uncompressed [nrg][S][32][16B], compressed [nrg][S/4][3][32][16B].
// One stage = KS k-steps (uncompressed) or KS/4 units (compressed) for every consumer warp.
// Warps 0..NW-1 consume, warp NW produces (one elected lane issues bulk copies on sm_90;
// on sm_80 all 32 producer lanes issue 16-byte cp.async and arrive with cp.async.mbarrier.arrive).
template <int MT, int RG, int WS, int KS, int NSTAGE, bool COMP, bool HL, bool WGM, typename OutT>
__global__ void __launch_bounds__((RG * WS + 1) * 32, 1)
skinny_tma_kernel(const uint4* __restrict__ Wp, const __nv_bfloat16* __restrict__ x,
                  OutT* __restrict__ out, float* __restrict__ ws, int* __restrict__ counters,
                  const int2* __restrict__ ptab, int pmax, uint32_t base4,
                  int M, int N, int Kp, int Kx, int splits, int nrg, int dist, int dg) {
  // dg (diagnostics only, 0 in production): 1 = skip epilogue, 2 = also skip x / escape-table loads,
  // 3 = also skip all compute (pure weight streaming through the pipeline)
  constexpr int NW = RG * WS;
  constexpr int NC = NW * 32;                       // consumer threads
  constexpr int TOK = MT * 8;
  constexpr int TOKP = TOK + 1;  // padded red row (bank-conflict-free epilogue reads)
  // bytes per warp per stage (HL: router hi and lo tiles of the same k-step back to back)
  constexpr int CHUNK = COMP ? (KS / 4) * 1536 : KS * 512 * (HL ? 2 : 1);
  constexpr int SLOT = NW * CHUNK;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  // layout: [bars: full NSTAGE, empty NSTAGE, xbar XP (pad 256)] [ring NSTAGE*SLOT] [x rows M*XS*2 + zero row] [red]
  constexpr int XP = 4;
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t full0 = sb, empty0 = sb + NSTAGE * 8, xbar0 = sb + 2 * NSTAGE * 8;
  const uint32_t ring = sb + 256;
  const int Kc = Kp / splits;
  const int XS = Kc + 8;
  // WGM: x as 128B-swizzled K-major tiles [Kc/64][TOK rows][128 B] (wgmma B operand), 1024-aligned
  const size_t x_off = WGM ? (size_t)((((sb + 256u + (uint32_t)(NSTAGE * SLOT)) + 1023u) & ~1023u) - sb)
                           : (size_t)256 + NSTAGE * SLOT;   // WGM: absolute shared address 1024-aligned
  const size_t x_bytes = WGM ? (size_t)Kc * TOK * 2 : (((size_t)M * XS * 2 + 16 + 15) & ~(size_t)15);
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem_raw + x_off);
  float* red = reinterpret_cast<float*>(smem_raw + x_off + x_bytes);
  int2* ptab_s = reinterpret_cast<int2*>(red + NW * 16 * TOKP + ((NW * 16 * TOKP) & 1));

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int rb = blockIdx.x;
  const int sp = blockIdx.y;
  const int k0 = sp * Kc;
  const int S = Kp >> 4;
  const int NS = Kc / (16 * WS);       // k-steps per consumer warp
  const int nst = NS / KS;              // stages
  // WS == 1: stream is [rb][stage over all K][warp][chunk] (shared with the wide kernel); else [cta][stage][warp][chunk]
  const char* wsrc = reinterpret_cast<const char*>(Wp) +
      (WS == 1 ? ((size_t)blockIdx.x * (S / KS) + (size_t)blockIdx.y * nst) : (size_t)(blockIdx.y * gridDim.x + blockIdx.x) * nst) * SLOT;

  if (threadIdx.x == 0) {
    for (int i = 0; i < NSTAGE; ++i) {
#if __CUDA_ARCH__ >= 900
      mbar_init(full0 + 8 * i, 1);
#else
      mbar_init(full0 + 8 * i, 32);
#endif
      mbar_init(empty0 + 8 * i, NW);
    }
    for (int i = 0; i < XP; ++i) mbar_init(xbar0 + 8 * i, NC);
#if __CUDA_ARCH__ >= 900
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    // first stages go out right away (no wait for the CTA barrier)
    for (int st = 0; st < NSTAGE && st < nst; ++st) {
      mbar_expect_tx(full0 + 8 * st, SLOT);
      bulk_g2s(ring + st * SLOT, wsrc + (size_t)st * SLOT, SLOT, full0 + 8 * st);
    }
#endif
  }
  if (!WGM && threadIdx.x < 4) reinterpret_cast<uint32_t*>(xs + M * XS)[threadIdx.x] = 0u;  // zero row
  __syncthreads();

  if (warp == NW) {
    // ---------------- producer: this CTA's weight stream is contiguous
#if __CUDA_ARCH__ >= 900
    const int st0 = NSTAGE < nst ? NSTAGE : nst;   // already issued by thread 0
#else
    const int st0 = 0;
#endif
    for (int st = st0; st < nst; ++st) {
      const int slot = st % NSTAGE;
      if (st >= NSTAGE) mbar_wait(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
      const uint32_t dst = ring + slot * SLOT;
      const char* src = wsrc + (size_t)st * SLOT;
#if __CUDA_ARCH__ >= 900
      if (lane == 0) {
        mbar_expect_tx(full0 + 8 * slot, SLOT);
        bulk_g2s(dst, src, SLOT, full0 + 8 * slot);
      }
#else
      for (int off = lane * 16; off < SLOT; off += 512)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst + off), "l"(src + off) : "memory");
      asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(full0 + 8 * slot) : "memory");
#endif
    }
    return;
  }

  // ---------------- consumers
  const int rg = rb * RG + warp % RG;   // warps of a warpgroup: 4 consecutive row groups, same k range
  const int wsub = warp / RG;
  if (threadIdx.x == 0 && splits > 1) asm volatile("prefetch.global.L2 [%0];" :: "l"(counters + rb));
  if (COMP && dg < 2) {
    // this CTA's exact-correction entries -> smem (tracked by x part 0's mbarrier below)
    const int cta = blockIdx.y * gridDim.x + blockIdx.x;
    const char* src = reinterpret_cast<const char*>(ptab + (size_t)cta * pmax);
    const uint32_t dst = (uint32_t)__cvta_generic_to_shared(ptab_s);
    for (int i = threadIdx.x; i < pmax / 2; i += NC)
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst + i * 16), "l"(src + i * 16) : "memory");
  }
  // x chunk in XP parts, each completion tracked by its own mbarrier (no CTA-wide wait)
  if (dg < 2) {
    const int kx0 = k0 % Kx;
    const int pch = Kc / 8 / XP;  // 16-byte chunks per token row per part
    const uint32_t xs_s = (uint32_t)__cvta_generic_to_shared(xs);
    for (int p = 0; p < XP; ++p) {
      if constexpr (WGM) {
        for (int idx = threadIdx.x; idx < TOK * pch; idx += NC) {
          int t = idx / pch, c = p * pch + (idx - t * pch);   // c: 16-byte chunk index along k
          const uint32_t dst = xs_s + (uint32_t)((((c >> 3) * TOK + t) << 7) + ((((c & 7) ^ (t & 7))) << 4));
          const __nv_bfloat16* src = x + (size_t)(t < M ? t : 0) * Kx + kx0 + c * 8;
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(t < M ? 16 : 0) : "memory");
        }
      } else {
        for (int idx = threadIdx.x; idx < M * pch; idx += NC) {
          int t = idx / pch, c = p * pch + (idx - t * pch);
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
                       "r"(xs_s + (uint32_t)((t * XS + c * 8) * 2)), "l"(x + (size_t)t * Kx + kx0 + c * 8) : "memory");
        }
      }
      asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(xbar0 + 8 * p) : "memory");
    }
  }
  int xp_ready = -1;
  const int kpart = Kc / XP;  // k per part

  float acc[MT][4];
#pragma unroll
  for (int j = 0; j < MT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  float wacc[WGM ? TOK / 2 : 1];
#pragma unroll
  for (int q = 0; q < (WGM ? TOK / 2 : 1); ++q) wacc[q] = 0.f;
  uint32_t afrag[2][4];

  const uint32_t xs_base = (uint32_t)__cvta_generic_to_shared(xs);
  const int lrow = lane & 7, lmat = lane >> 3;
  const int kw0 = wsub * NS * 16;
  const uint32_t zero_row = xs_base + (uint32_t)(M * XS * 2);
  uint32_t xs_row[MT];
  uint32_t xs_adv[MT];
#pragma unroll
  for (int j = 0; j < MT; ++j) {
    const int t = j * 8 + lrow;
    xs_row[j] = t < M ? xs_base + (uint32_t)((t * XS + kw0 + lmat * 8) * 2) : zero_row;
    xs_adv[j] = t < M ? 32u : 0u;   // bytes per k-step pair... per k-step: 32 B
  }

  for (int st = 0; st < nst; ++st) {
    const int slot = st % NSTAGE;
    {
      const int need = (kw0 + (st + 1) * KS * 16 - 1) / kpart;  // x part holding this stage's last k
      if (need > xp_ready && dg < 2) {
        mbar_wait(xbar0 + 8 * need, 0);
        xp_ready = need;
#if __CUDA_ARCH__ >= 900
        if constexpr (WGM) fence_async_smem();   // cp.async (generic proxy) writes -> wgmma (async proxy) reads
#endif
      }
    }
    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
    // pull this warp's stage data into registers, then hand the slot back to the producer at once
    constexpr int NL = CHUNK / 512;  // 16-byte loads per lane per stage
    uint4 wr[NL];
    {
      const uint32_t src = ring + slot * SLOT + warp * CHUNK + lane * 16;
#pragma unroll
      for (int q = 0; q < NL; ++q)
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(wr[q].x), "=r"(wr[q].y), "=r"(wr[q].z), "=r"(wr[q].w) : "r"(src + q * 512));
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    if (dg == 3) {
      if (wr[0].x == 0x9e3779b9u && wr[NL - 1].w == 0x7f4a7c15u) acc[0][0] += 1.f;   // keep the loads
      continue;
    }
    if constexpr (!COMP && HL) {
#pragma unroll
      for (int j2 = 0; j2 < KS; j2 += 2) {
        const uint4 h0 = wr[2 * j2], l0 = wr[2 * j2 + 1], h1 = wr[2 * j2 + 2], l1 = wr[2 * j2 + 3];
        const int s = st * KS + j2;
        uint32_t b[MT][4];
#pragma unroll
        for (int j = 0; j < MT; ++j)
          ldsm_x4(xs_row[j] + xs_adv[j] * (uint32_t)s, b[j][0], b[j][1], b[j][2], b[j][3]);
#pragma unroll
        for (int j = 0; j < MT; ++j) {
          mma_bf16(acc[j], h0.x, h0.y, h0.z, h0.w, b[j][0], b[j][1]);
          mma_bf16(acc[j], l0.x, l0.y, l0.z, l0.w, b[j][0], b[j][1]);
          mma_bf16(acc[j], h1.x, h1.y, h1.z, h1.w, b[j][2], b[j][3]);
          mma_bf16(acc[j], l1.x, l1.y, l1.z, l1.w, b[j][2], b[j][3]);
        }
      }
    } else if constexpr (!COMP) {
#pragma unroll
      for (int j2 = 0; j2 < KS; j2 += 2) {
        const uint4 a0 = wr[j2], a1 = wr[j2 + 1];
        const int s = st * KS + j2;
        uint32_t b[MT][4];
#pragma unroll
        for (int j = 0; j < MT; ++j)
          ldsm_x4(xs_row[j] + xs_adv[j] * (uint32_t)s, b[j][0], b[j][1], b[j][2], b[j][3]);
#pragma unroll
        for (int j = 0; j < MT; ++j) {
          mma_bf16(acc[j], a0.x, a0.y, a0.z, a0.w, b[j][0], b[j][1]);
          mma_bf16(acc[j], a1.x, a1.y, a1.z, a1.w, b[j][2], b[j][3]);
        }
      }
    } else if constexpr (WGM) {
#if __CUDA_ARCH__ >= 900
#pragma unroll
      for (int u = 0; u < KS / 4; ++u) {
        const uint4 L0 = wr[3 * u], L1 = wr[3 * u + 1], C = wr[3 * u + 2];
        const int kl = kw0 + (st * KS + u * 4) * 16;   // local k of this unit's first k-step (multiple of 64)
        const uint32_t xt = xs_base + (uint32_t)((kl >> 6) * TOK * 128);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          uint32_t* af = afrag[j & 1];
          decode8(j < 2 ? (j == 0 ? L0.x : L0.z) : (j == 2 ? L1.x : L1.z),
                  j < 2 ? (j == 0 ? L0.y : L0.w) : (j == 2 ? L1.y : L1.w),
                  j == 0 ? C.x : j == 1 ? C.y : j == 2 ? C.z : C.w, base4, af);
#pragma unroll
          for (int q = 0; q < TOK / 2; ++q) asm volatile("" : "+f"(wacc[q]) :: "memory");
          wg_fence();
          wgmma_bn<TOK>(wacc, af[0], af[1], af[2], af[3], make_desc_sw128(xt + j * 32));
          wg_commit();
          wg_wait<1>();   // the wgmma that used afrag[(j+1)&1] is done -> it can be overwritten next step
#pragma unroll
          for (int q = 0; q < 4; ++q) asm volatile("" : "+r"(afrag[0][q]), "+r"(afrag[1][q]) :: "memory");
        }
      }
#else
      // sm_80 check path: same data layout, mma.sync with B fragments read from the swizzled x tile
      const int g8 = lane >> 2, t8 = lane & 3;
#pragma unroll
      for (int u = 0; u < KS / 4; ++u) {
        const uint4 L0 = wr[3 * u], L1 = wr[3 * u + 1], C = wr[3 * u + 2];
        const int kl = kw0 + (st * KS + u * 4) * 16;
        const unsigned char* tile = reinterpret_cast<const unsigned char*>(xs) + (kl >> 6) * TOK * 128;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          uint32_t af[4];
          decode8(j < 2 ? (j == 0 ? L0.x : L0.z) : (j == 2 ? L1.x : L1.z),
                  j < 2 ? (j == 0 ? L0.y : L0.w) : (j == 2 ? L1.y : L1.w),
                  j == 0 ? C.x : j == 1 ? C.y : j == 2 ? C.z : C.w, base4, af);
#pragma unroll
          for (int jj = 0; jj < MT; ++jj) {
            const int tok = jj * 8 + g8;
            const unsigned char* row = tile + tok * 128;
            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(row + (((2 * j) ^ (tok & 7)) << 4) + 4 * t8);
            const uint32_t b1 = *reinterpret_cast<const uint32_t*>(row + (((2 * j + 1) ^ (tok & 7)) << 4) + 4 * t8);
            mma_bf16(wacc + 4 * jj, af[0], af[1], af[2], af[3], b0, b1);
          }
        }
      }
#endif
    } else {
#pragma unroll
      for (int u = 0; u < KS / 4; ++u) {
        const uint4 L0 = wr[3 * u], L1 = wr[3 * u + 1], C = wr[3 * u + 2];
        uint32_t a[16];
        decode8(L0.x, L0.y, C.x, base4, a + 0);
        decode8(L0.z, L0.w, C.y, base4, a + 4);
        decode8(L1.x, L1.y, C.z, base4, a + 8);
        decode8(L1.z, L1.w, C.w, base4, a + 12);
        const int s = st * KS + u * 4;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          uint32_t b[MT][4];
#pragma unroll
          for (int j = 0; j < MT; ++j)
            ldsm_x4(xs_row[j] + xs_adv[j] * (uint32_t)(s + 2 * h), b[j][0], b[j][1], b[j][2], b[j][3]);
#pragma unroll
          for (int j = 0; j < MT; ++j) {
            const uint32_t* a0 = a + 8 * h;
            mma_bf16(acc[j], a0[0], a0[1], a0[2], a0[3], b[j][0], b[j][1]);
            mma_bf16(acc[j], a0[4], a0[5], a0[6], a0[7], b[j][2], b[j][3]);
          }
        }
      }
    }
  }
#if __CUDA_ARCH__ >= 900
  if constexpr (WGM) {
    wg_wait<0>();
#pragma unroll
    for (int q = 0; q < TOK / 2; ++q) asm volatile("" : "+f"(wacc[q]) :: "memory");
  }
#endif

  if constexpr (WGM) {
#pragma unroll
    for (int j = 0; j < MT; ++j) { acc[j][0] = wacc[4 * j]; acc[j][1] = wacc[4 * j + 1]; acc[j][2] = wacc[4 * j + 2]; acc[j][3] = wacc[4 * j + 3]; }
  }
  if (dg >= 1) {
    if (acc[0][0] == 1.2345e-30f) out[threadIdx.x] = cvt_out<OutT>(acc[0][1]);
    return;   // (the producer warp has already returned)
  }
  // per-warp partials -> shared memory: red[w][row16][TOK]
  {
    const int g = lane >> 2, t = lane & 3;
    float* r = red + warp * 16 * TOKP;
#pragma unroll
    for (int j = 0; j < MT; ++j) {
      r[g * TOKP + j * 8 + 2 * t] = acc[j][0];
      r[g * TOKP + j * 8 + 2 * t + 1] = acc[j][1];
      r[(g + 8) * TOKP + j * 8 + 2 * t] = acc[j][2];
      r[(g + 8) * TOKP + j * 8 + 2 * t + 1] = acc[j][3];
    }
  }
  if constexpr (COMP) {
    // all x parts and the patch table are in smem; apply exact corrections in parallel (smem atomics)
    mbar_wait(xbar0 + 8 * (XP - 1), 0);
    named_bar(1, NC);
    const int cnt = ptab_s[0].x;
    for (int p = 1 + warp; p <= cnt; p += NW) {
      const int2 e = ptab_s[p];
      const int rl = e.x >> 16, kl = e.x & 0xFFFF;
      const float d = __int_as_float(e.y);
      float* r = red + ((rl >> 4) * 16 + (rl & 15)) * TOKP;   // warp (wsub 0, row group rl/16)
      for (int t = lane; t < M; t += 32) {
        float xv;
        if constexpr (WGM) {   // swizzled K-major tiles [Kc/64][TOK][128 B]
          const int c = kl >> 3;
          xv = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
              reinterpret_cast<const unsigned char*>(xs) + (((c >> 3) * TOK + t) << 7) + (((c & 7) ^ (t & 7)) << 4) + (kl & 7) * 2));
        } else {
          xv = __bfloat162float(xs[t * XS + kl]);
        }
        atomicAdd(r + t, xv * d);
      }
    }
  }
  named_bar(1, NC);
  const int rows = RG * 16;
  const int row0 = rb * rows;
  if (splits == 1) {
    for (int idx = threadIdx.x; idx < rows * M; idx += NC) {
      int t = idx / rows, rr = idx - t * rows;
      int grp = rr >> 4, r16 = rr & 15;
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < WS; ++w) v += red[((w * RG + grp) * 16 + r16) * TOKP + t];
      out[(size_t)t * N + row0 + rr] = cvt_out<OutT>(v);
    }
    return;
  }
  for (int idx = threadIdx.x; idx < rows * M; idx += NC) {
    int t = idx / rows, rr = idx - t * rows;
    int grp = rr >> 4, r16 = rr & 15;
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < WS; ++w) v += red[((w * RG + grp) * 16 + r16) * TOKP + t];
    __stcg(ws + ((size_t)sp * M + t) * N + row0 + rr, v);
  }
  __threadfence();
  named_bar(1, NC);
  // dist: all `splits` CTAs of this row block are co-resident (grid fits the GPU at once): monotonic-counter
  // barrier, then each reduces its own slice of the row block.  Otherwise the last-arriving CTA reduces all.
  // Counters only ever increase (both modes), so the modes can share them.
  __shared__ int s_last;
  if (threadIdx.x == 0) {
    const int old = atomicAdd(counters + rb, 1);
    s_last = (old % splits) == splits - 1;
    if (dist) {
      const int target = (old / splits + 1) * splits;
      okc_spin_ge(counters + rb, target, 20);
    }
  }
  named_bar(1, NC);
  if (dist || s_last) {
  __threadfence();
  {
    const int r4n = rows / 4;
    const int nch = M * r4n;
    const int c_lo = dist ? (int)(((long long)sp * nch) / splits) : 0;
    const int c_hi = dist ? (int)(((long long)(sp + 1) * nch) / splits) : nch;
    for (int c0 = c_lo + threadIdx.x; c0 < c_hi; c0 += 2 * NC) {
      float4 acc4[2];
      int tt[2], rr[2];
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int c = c0 + u * NC;
        tt[u] = c / r4n; rr[u] = (c - tt[u] * r4n) * 4;
        acc4[u] = make_float4(0.f, 0.f, 0.f, 0.f);
      }
      for (int q = 0; q < splits; q += 4) {
        float4 v[2][4];
#pragma unroll
        for (int u = 0; u < 2; ++u)
#pragma unroll
          for (int qq = 0; qq < 4; ++qq) {
            v[u][qq] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (c0 + u * NC < c_hi && q + qq < splits)
              v[u][qq] = __ldcg(reinterpret_cast<const float4*>(ws + ((size_t)(q + qq) * M + tt[u]) * N + row0 + rr[u]));
          }
#pragma unroll
        for (int u = 0; u < 2; ++u)
#pragma unroll
          for (int qq = 0; qq < 4; ++qq) {
            acc4[u].x += v[u][qq].x; acc4[u].y += v[u][qq].y; acc4[u].z += v[u][qq].z; acc4[u].w += v[u][qq].w;
          }
      }
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        if (c0 + u * NC < c_hi) {
          OutT* o = out + (size_t)tt[u] * N + row0 + rr[u];
          o[0] = cvt_out<OutT>(acc4[u].x); o[1] = cvt_out<OutT>(acc4[u].y);
          o[2] = cvt_out<OutT>(acc4[u].z); o[3] = cvt_out<OutT>(acc4[u].w);
        }
      }
    }
  }
  }  // dist || s_last
}

template <int MT, int RG, int WS, int KS, int NSTAGE, bool COMP, bool HL, bool WGM, typename OutT>
static void launch_tma(const at::Tensor& Wp, const at::Tensor& x, at::Tensor& out,
                       at::Tensor& ws, at::Tensor& counters, const at::Tensor& ptab, int64_t pmax,
                       int64_t base4, int M, int N, int Kp, int Kx, int splits, int nrg, int dg) {
  constexpr int NW = RG * WS;
  constexpr int CHUNK = COMP ? (KS / 4) * 1536 : KS * 512 * (HL ? 2 : 1);
  int Kc = Kp / splits;
  size_t x_bytes = WGM ? (size_t)Kc * MT * 8 * 2 + 1024 : (((size_t)M * (Kc + 8) * 2 + 16 + 15) & ~(size_t)15);
  size_t smem = 256 + (size_t)NSTAGE * NW * CHUNK + x_bytes + (size_t)NW * 16 * (MT * 8 + 1) * 4 + 8 + (size_t)pmax * 8;
  auto kern = skinny_tma_kernel<MT, RG, WS, KS, NSTAGE, COMP, HL, WGM, OutT>;
  static bool attr_done = false;
  if (!attr_done) {
    int dev = 0, optin = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);
    attr_done = true;
  }
  dim3 grid(N / (RG * 16), splits);
  int dev = 0, nsm = 0, per_sm = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, (NW + 1) * 32, smem);
  const int dist = (int)(grid.x * grid.y) <= per_sm * nsm ? 1 : 0;
  kern<<<grid, (NW + 1) * 32, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(Wp.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<OutT*>(out.data_ptr()), ws.data_ptr<float>(), counters.data_ptr<int>(),
      reinterpret_cast<const int2*>(ptab.data_ptr()), (int)pmax, (uint32_t)base4,
      M, N, Kp, Kx, splits, nrg, dist, dg);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void @@TMA_NAME@@(at::Tensor Wp, at::Tensor x, at::Tensor out, at::Tensor ws, at::Tensor counters,
                at::Tensor ptab, int64_t pmax, int64_t base4,
                int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t mt, int64_t rg, int64_t wsn,
                int64_t comp, int64_t nrg, int64_t ks, int64_t dg) {
  bool out_f32 = out.scalar_type() == at::kFloat;
  @@TMA_DISPATCH@@
  TORCH_CHECK(false, "skinny_tma: no config");
}
"""

bfp_MAP_SRC = r"""
#include <cudaTypedefs.h>
#include <cstring>
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void tma_load_2d_pol(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1, uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5;"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "l"(pol) : "memory");
}
__device__ __forceinline__ void tma_load_2d_ef(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  tma_load_2d_pol(dst, map, bar, c0, c1, l2_pol_ef());
}
__device__ __forceinline__ void tma_load_2d_el(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  tma_load_2d_pol(dst, map, bar, c0, c1, l2_pol_el());
}
__device__ __forceinline__ void tma_load_3d(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4, %5}], [%2], %6;"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2), "l"(l2_pol_el()) : "memory");
}
#endif
static PFN_cuTensorMapEncodeTiled_v12000 get_encode() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult q;
    void* p = nullptr;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);
    TORCH_CHECK(p != nullptr, "cuTensorMapEncodeTiled not found");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}

// x [M, K] bf16 as 3-D {64 (k in unit), M (token), K/64 (unit)}: one TMA brings `units` tiles of [rows x 128 B]
static CUtensorMap make_map_x3d(const void* ptr, uint64_t m, uint64_t k, uint32_t box_rows, uint32_t units) {
  CUtensorMap mp;
  cuuint64_t dims[3] = {64, m, k / 64};
  cuuint64_t strides[2] = {k * 2, 128};
  cuuint32_t box[3] = {64, box_rows, units};
  cuuint32_t estr[3] = {1, 1, 1};
  CUresult r = get_encode()(&mp, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(ptr), dims, strides, box, estr,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (3d) failed: ", (int)r);
  return mp;
}

static CUtensorMap make_map_2d(const void* ptr, uint64_t rows, uint64_t cols, uint32_t box_rows) {
  CUtensorMap m;
  cuuint64_t dims[2] = {cols, rows};
  cuuint64_t strides[1] = {cols * 2};
  cuuint32_t box[2] = {64, box_rows};
  cuuint32_t estr[2] = {1, 1};
  CUresult r = get_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, estr,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}

"""

bfp_GEMM_SRC = r"""

@@WGMMA_SS_DEFS@@

template <int NWG, int BN, int STAGES, bool AS, typename OutT>
__global__ void __launch_bounds__(NWG * 128 + 128, 1)
gemm90_kernel(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
              const uint8_t* __restrict__ Wst,
              OutT* __restrict__ out, float* __restrict__ ws, int* __restrict__ counters,
              int M, int N, int Kp, int Kx, int splits) {
  constexpr int BM = NWG * 64;
  constexpr int A_BYTES = BM * 128;       // BM rows x 64 bf16
  constexpr int B_BYTES = BN * 128;       // BN rows x 64 bf16
  constexpr int ST_BYTES = A_BYTES + B_BYTES;
  constexpr int NACC = BN / 2;
  constexpr int NT = NWG * 128;           // consumer threads
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  const uint32_t bars = sbase + STAGES * ST_BYTES;          // full[STAGES], empty[STAGES]
  const uint32_t full0 = bars, empty0 = bars + 8 * STAGES;

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int rb = blockIdx.x, sp = blockIdx.y;
  const int nkb_all = Kp >> 6;
  const int kb0 = (int)(((long long)sp * nkb_all) / splits);
  const int kb1 = (int)(((long long)(sp + 1) * nkb_all) / splits);
  const int nkb = kb1 - kb0;

  if (tid == 0) {
    for (int i = 0; i < STAGES; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NWG); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if (splits > 1) asm volatile("prefetch.global.L2 [%0];" :: "l"(counters + rb));
  }
  __syncthreads();

  if (warp >= NWG * 4) {
    // ---------------- producer warp (one lane)
    if (warp == NWG * 4 && lane == 0) {
      asm volatile("prefetch.tensormap [%0];" :: "l"(&mapA) : "memory");
      asm volatile("prefetch.tensormap [%0];" :: "l"(&mapB) : "memory");
      for (int i = 0; i < nkb; ++i) {
        const int slot = i % STAGES;
        if (i >= STAGES) mbar_wait(empty0 + 8 * slot, ((i / STAGES) - 1) & 1);
        mbar_expect_tx(full0 + 8 * slot, ST_BYTES);
        const int kg = (kb0 + i) * 64;
        if constexpr (AS)   // A tile pre-swizzled and contiguous: [rb][kb][BM x 128 B]
          bulk_g2s(sbase + slot * ST_BYTES, Wst + ((size_t)rb * nkb_all + kb0 + i) * A_BYTES, A_BYTES, full0 + 8 * slot);
        else
          tma_load_2d_ef(sbase + slot * ST_BYTES, &mapA, full0 + 8 * slot, kg, rb * BM);
        tma_load_2d_el(sbase + slot * ST_BYTES + A_BYTES, &mapB, full0 + 8 * slot, kg % Kx, 0);
      }
    }
    return;
  }

  // ---------------- consumers
  const int wg = warp >> 2;
  float acc[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) acc[i] = 0.f;

  for (int i = 0; i < nkb; ++i) {
    const int slot = i % STAGES;
    mbar_wait(full0 + 8 * slot, (i / STAGES) & 1);
    const uint32_t a_st = sbase + slot * ST_BYTES + wg * (64 * 128);
    const uint32_t b_st = sbase + slot * ST_BYTES + A_BYTES;
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j)
      wgmma_ss<BN>(acc, make_desc_sw128(a_st + j * 32), make_desc_sw128(b_st + j * 32));
    wg_commit();
    // release the previous stage once its wgmmas are done (keep one group in flight)
    wg_wait<1>();
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    if (i > 0 && (tid & 127) == 0) mbar_arrive(empty0 + 8 * ((i - 1) % STAGES));
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");

  const int g = lane >> 2, t4 = lane & 3;
  const int row_w = rb * BM + (warp & 3) * 16 + wg * 64;
  if (splits == 1) {
#pragma unroll
    for (int j = 0; j < BN / 8; ++j) {
      const int t0 = 8 * j + 2 * t4;
      if (t0 < M) {
        out[(size_t)t0 * N + row_w + g] = cvt_out<OutT>(acc[4 * j]);
        out[(size_t)t0 * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 2]);
      }
      if (t0 + 1 < M) {
        out[(size_t)(t0 + 1) * N + row_w + g] = cvt_out<OutT>(acc[4 * j + 1]);
        out[(size_t)(t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 3]);
      }
    }
    return;
  }
  {
    float4* wsp = reinterpret_cast<float4*>(ws) + (size_t)(rb * splits + sp) * (NACC / 4) * NT;
#pragma unroll
    for (int j = 0; j < NACC / 4; ++j)
      __stcg(wsp + j * NT + tid, make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]));
  }
  __threadfence();
  named_bar(1, NT);
  if (tid == 0) {
    int old = atomicAdd(counters + rb, 1);
    int target = (old / splits + 1) * splits;
    okc_spin_ge(counters + rb, target, 32);
  }
  named_bar(1, NT);
  __threadfence();
  const int j0 = (int)(((long long)sp * (NACC / 4)) / splits);
  const int j1 = (int)(((long long)(sp + 1) * (NACC / 4)) / splits);
  const float4* base = reinterpret_cast<const float4*>(ws) + (size_t)rb * splits * (NACC / 4) * NT;
  for (int j = j0; j < j1; ++j) {
    float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
    int q = 0;
    for (; q + 4 <= splits; q += 4) {
      float4 v0 = __ldcg(base + ((size_t)(q + 0) * (NACC / 4) + j) * NT + tid);
      float4 v1 = __ldcg(base + ((size_t)(q + 1) * (NACC / 4) + j) * NT + tid);
      float4 v2 = __ldcg(base + ((size_t)(q + 2) * (NACC / 4) + j) * NT + tid);
      float4 v3 = __ldcg(base + ((size_t)(q + 3) * (NACC / 4) + j) * NT + tid);
      s.x += v0.x + v1.x + v2.x + v3.x; s.y += v0.y + v1.y + v2.y + v3.y;
      s.z += v0.z + v1.z + v2.z + v3.z; s.w += v0.w + v1.w + v2.w + v3.w;
    }
    for (; q < splits; ++q) {
      float4 v = __ldcg(base + ((size_t)q * (NACC / 4) + j) * NT + tid);
      s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
    }
    const int t0 = 8 * j + 2 * t4;
    if (t0 < M) {
      out[(size_t)t0 * N + row_w + g] = cvt_out<OutT>(s.x);
      out[(size_t)t0 * N + row_w + g + 8] = cvt_out<OutT>(s.z);
    }
    if (t0 + 1 < M) {
      out[(size_t)(t0 + 1) * N + row_w + g] = cvt_out<OutT>(s.y);
      out[(size_t)(t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(s.w);
    }
  }
}

template <int NWG, int BN, int STAGES, bool AS, typename OutT>
static void launch_gemm90(const at::Tensor& W, const at::Tensor& Wst, const at::Tensor& x, at::Tensor& out, at::Tensor& ws,
                          at::Tensor& counters, int M, int N, int Kp, int Kx, int splits) {
  constexpr int BM = NWG * 64;
  size_t smem = 1024 + (size_t)STAGES * (BM * 128 + BN * 128) + 16 * STAGES + 64;
  auto kern = gemm90_kernel<NWG, BN, STAGES, AS, OutT>;
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr_done = true;
  }
  CUtensorMap mA = make_map_2d(W.data_ptr(), (uint64_t)W.size(0), (uint64_t)W.size(1), BM);
  CUtensorMap mB = make_map_2d(x.data_ptr(), (uint64_t)M, (uint64_t)Kx, BN);
  dim3 grid(N / BM, splits);
  kern<<<grid, NWG * 128 + 128, smem, at::cuda::getCurrentCUDAStream()>>>(
      mA, mB, reinterpret_cast<const uint8_t*>(Wst.data_ptr()),
      reinterpret_cast<OutT*>(out.data_ptr()), ws.data_ptr<float>(), counters.data_ptr<int>(),
      M, N, Kp, Kx, splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gemm90(at::Tensor W, at::Tensor Wst, at::Tensor x, at::Tensor out, at::Tensor ws, at::Tensor counters,
            int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t nwg, int64_t bn, int64_t as) {
  bool out_f32 = out.scalar_type() == at::kFloat;
  @@GEMM_DISPATCH@@
  TORCH_CHECK(false, "gemm90: no config");
}
"""


bfp_SK2_SRC = r"""

#ifndef SK2_DELAY_ESC
#define SK2_DELAY_ESC 0
#endif
#if SK2_DELAY_ESC
// debug builds only: every ring wait of the escape and producer warps gives up after 2 s (globaltimer) with a trap, so
// a wedged ring aborts the launch instead of hanging the GPU
__device__ __forceinline__ void sk2_wait_dbg(uint32_t bar, uint32_t parity) {
  unsigned long long t0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  uint32_t ok = 0;
  for (;;) {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity), "r"(1000u) : "memory");
    if (ok) return;
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    if (t - t0 > 2000000000ull) {
      printf("sk2 ring wedged: block %d warp %d bar 0x%x parity %u\n", (int)blockIdx.x, (int)(threadIdx.x >> 5), bar, parity);
      __trap();
    }
  }
}
#define SK2_WAIT(bar, par) sk2_wait_dbg((bar), (par))
#else
#define SK2_WAIT(bar, par) mbar_wait_sleep((bar), (par))
#endif
#ifndef SK2_NESC
#define SK2_NESC 3
#endif
// PDL (programmatic dependent launch) for the sk2 launches: 0 = off (plain launches); N > 0 = launch with programmatic
// stream serialization and issue the first N weight stages before griddepcontrol.wait (set by sk2_set_pdl)
static int g_sk2_pdl = 0;
void sk2_set_pdl(int64_t n) { g_sk2_pdl = (int)n; }
#ifndef SK2_DIAG
#define SK2_DIAG 0
#endif
// ============================================================================ sk2 (m <= 32): swapped operands
// A = x tile (16 tokens x 16 k; ldmatrix from the TMA-loaded, 128B-swizzled tile), B = weights (8 rows x 16 k),
// decoded to registers from the 12-bit stream (BF16 hi + lo for the router).  Each CTA owns R rows over the whole
// K: no cross-CTA reduction.  Warps split the CTA's rows (NWR x NT n8 tiles) and, inside every stage, K (WSK).
// A stage = UPS 64-k units: x tiles [UPS][TOKR rows x 128 B] (TMA, rows >= M zero-filled) + this CTA's weights.
template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT,
          bool PK, bool FUSE, bool BT>
__global__ void __launch_bounds__((NWR * WSK + 1 + (COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1)) * 32, 1)
sk2_kernel(const __grid_constant__ CUtensorMap xmap, const uint8_t* __restrict__ Ws,
           const __nv_bfloat16* __restrict__ x, OutT* __restrict__ out,
           const int2* __restrict__ ptab, int pmax, uint32_t base4, int M, int N, int K, int dg,
           unsigned long long* __restrict__ dbg, const uint8_t* __restrict__ Wsr, float* __restrict__ outr,
           float* __restrict__ part, int* __restrict__ rcnt, int pdl,
           const float* __restrict__ rbias, const __nv_bfloat16* __restrict__ gpre, int* __restrict__ rids,
           int* __restrict__ rlids, float* __restrict__ rwts, __nv_bfloat16* __restrict__ xm, const int* __restrict__ rls,
           float* __restrict__ part2, int* __restrict__ dcnt,
           const float* __restrict__ b_alpha, const float* __restrict__ b_beta, const float* __restrict__ b_sin,
           const __nv_bfloat16* __restrict__ b_gp, const __nv_bfloat16* __restrict__ b_gn, float* __restrict__ b_sout,
           __nv_bfloat16* __restrict__ b_hout, float* __restrict__ b_part, int* __restrict__ b_cnt) {
#if __CUDA_ARCH__ >= 900
  const int dgv = SK2_DIAG ? dg : 0;   // diagnostics (dg) compiled out of production builds
  auto gtime = []() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; };
  const unsigned other_cta = CL > 1 ? 1u : gridDim.x / 2;
  const bool trace = (dgv & 32) && dbg != nullptr && (blockIdx.x == 0 || blockIdx.x == other_cta);
  unsigned long long* tr = trace ? dbg + (blockIdx.x == 0 ? 0 : 4096) : nullptr;
  const unsigned long long t_start = (dgv & 32) ? gtime() : 0ull;
  if (trace && threadIdx.x == 0) tr[0] = t_start;
  constexpr int NW = NWR * WSK;
  constexpr int NC = NW * 32;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int TU = COMP ? 768 : (HL ? 2048 : 1024);
  constexpr int WBY = NW * UPW * NT * TU;
  constexpr int XT = TOKR * 128;
  constexpr int XBY = UPS * XT;
  constexpr int SST = ((XBY + WBY) + 1023) & ~1023;
  // partial sums [warp][token][RP] (rows fastest, padded): 64-bit stores in, 128-bit loads out
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);   // >= R, = 8 (mod 32): conflict-free 64-bit stores
  constexpr int CP = R + 1;                         // corr[token][CP]: odd stride, conflict-free per-token updates
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t bars = sbase + NSTAGE * SST;
  const uint32_t full0 = bars, empty0 = bars + 8 * NSTAGE, pbar = bars + 16 * NSTAGE, cbar = pbar + 8, qbar = pbar + 16;
  // corrections, one copy per escape warp: [NESC][TOKS tokens][CP]; escape warps take turns on the stages
  constexpr int NESC = COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1;
  static_assert(NESC <= NSTAGE, "sk2: escape warps need NESC <= NSTAGE");
  float* corr = reinterpret_cast<float*>(gbase + NSTAGE * SST + 16 * NSTAGE + 32);
  constexpr int CSZ = (TOKS * CP + 3) & ~3;
  float* peer = corr + NESC * CSZ;                       // [CL-1][WSK][TOKS][RP]: ranks > 0 per-warp sums
  int2* ptab_s = reinterpret_cast<int2*>(peer + (CL > 1 ? (CL - 1) * WSK * TOKS * RP : 0));   // [pmax]: escapes
  const uint32_t zrow = (uint32_t)__cvta_generic_to_shared(ptab_s + pmax);   // 128 B of zeros (x rows >= M)
  // HSM: x lives in shared memory for the whole launch (16 KB after zrow, M = 1), written by the boundary-1 prologue
  // in the unit/swizzle layout the TMA boxes would have had; no x boxes, the stage barriers count weights only
  const bool HSM = false;   // (the shared-memory-x prologue, retired: M = 1 only and load-issue bound)
  const uint32_t hsm = zrow + 128;
  // BND1: the cooperative boundary-1 prologue (decode layer, M <= 8): CTA c computes columns 64c..64c+63 of
  // h = rmsnorm(alpha s + beta rmsnorm(a)) (1 + w) and of s1 for every row from the o-projection output a (b_gp),
  // the input stream (b_sin), alpha / beta (fp32) and the gain (b_gn); 4 partial row sums per CTA go through
  // b_part [128 CTAs][8 rows][4], two grid tickets on b_cnt (at the kernel START: the predecessor is done, no
  // dependent can be waiting); the producer streams the whole weight ring meanwhile and issues the x boxes
  // (this launch's x IS h, b_hout) once every CTA's slice is written. No per-CTA whole-row work.
  const bool BND1 = FUSE && (b_hout != nullptr) && (b_part != nullptr) && (b_cnt != nullptr);
  __shared__ int s_hready;
  // m <= 8: the x box has exactly M rows per 64-k unit (packed, M*128 B per unit; no zero rows moved)
  // PK (M < TOKR, or TOKR == 8): packed x, exactly M rows per 64-k unit (no out-of-bounds TMA rows; rows >= M
  // read zrow); else the full TOKR-row tile with compile-time addressing
  const int Mb = PK ? M : TOKR;
  const uint32_t XU = (uint32_t)Mb * 128u;
  const uint32_t XBYr = UPS * XU;
  float* red = reinterpret_cast<float*>(gbase);   // reuses the ring once every stage is consumed

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  // CL = 2: a cluster of two CTAs shares a row block; each streams one half of K, rank 1 hands its sums to rank 0
  const int cta = blockIdx.x / CL, crank = CL > 1 ? (int)(blockIdx.x % CL) : 0;
  const int tst = K / (UPS * 64);                      // stages over all of K
  const int gs0 = (crank * tst) / CL;                  // this CTA's first stage (uneven split for CL = 3)
  const int nst = ((crank + 1) * tst) / CL - gs0;      // and its stage count
  // stage-major weight stream: [stage][row block][...] (at any moment all CTAs read one contiguous region)
  const uint8_t* wsrc = Ws + (size_t)cta * WBY;
  const size_t wstride = (size_t)(gridDim.x / CL) * WBY;
  // FUSE (pooler + router in one launch): after its pooler stages every CTA streams one router piece: 8 router rows
  // (tile rt = cta / 4) over one quarter of K (rq = cta % 4), from the router's own stage-major hi|lo stream
  // ([16 stages][32 tiles][8 warps][2048 B]); 4 more stages through the same ring
  constexpr int NSTR = FUSE ? 4 : 0;
  constexpr int WBYR = NW * 2048;
  static_assert(!FUSE || (COMP && CL == 1 && UPW == 1 && UPS == 8 && WBYR <= WBY), "sk2 fused: pooler-like config only");
  const int rt = cta >> 2, rq = cta & 3;
  const uint8_t* wsr = FUSE ? Wsr + (size_t)rt * WBYR : nullptr;
  const int nstt = nst + NSTR;
  // stages issued by the prologue: 1, or under PDL up to pdl (CL == 1 only; never past this CTA's own stages)
  const int P = (CL == 1 && pdl > 0) ? min(pdl, min(NSTAGE, nst)) : 1;

  if (tid == 0) {
    // stage 0 first of all (the first bytes' latency is the ramp of the whole kernel): the weight copy needs only
    // its barrier; the tensor map prefetch goes out before it so the x box follows right behind. In a real cluster
    // (CL > 1) a cluster-scope release issued AFTER a copy is in flight waits for it (~0.7 us): there every barrier
    // is initialized and fenced before the first copy.
    asm volatile("prefetch.tensormap [%0];" :: "l"(&xmap) : "memory");
    mbar_init(full0, 1);
    if constexpr (CL == 1)   // PDL: the first P stages' weights go out before the wait; their barriers first
      for (int i = 1; i < P; ++i) mbar_init(full0 + 8 * i, 1);
    if constexpr (CL > 1) {
      mbar_init(empty0, NW + (COMP ? NESC : 0));
      for (int i = 1; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NW + (COMP ? NESC : 0)); }
      mbar_init(pbar, 1);
      mbar_init(cbar, 32 * NESC);
      mbar_init(qbar, 1);
      if (crank == 0) mbar_expect_tx(qbar, (CL - 1) * NC * MT * NT * 16);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    // weights never depend on the previous kernel: under PDL (pdl > 0) they stream while it finishes; x (its output)
    // only after griddepcontrol.wait (a no-op when the launch has no programmatic dependency)
    for (int st = 0; st < P; ++st) {
      mbar_expect_tx(full0 + 8 * st, (((dgv & 4) || HSM) ? 0 : XBYr) + WBY);
      bulk_g2s(sbase + st * SST + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * st);
    }
    if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
    if (!(dgv & 4) && !HSM && !BND1)   // (BND1: the producer issues every x box once h exists in global memory)
      for (int st = 0; st < P; ++st) tma_load_3d(sbase + st * SST, &xmap, full0 + 8 * st, 0, 0, (gs0 + st) * UPS);
    if (COMP) {   // this CTA's escape table
      if constexpr (CL == 1) {
        mbar_init(pbar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      }
      mbar_expect_tx(pbar, pmax * 8);
      bulk_g2s((uint32_t)__cvta_generic_to_shared(ptab_s), ptab + (size_t)cta * pmax, pmax * 8, pbar);
    }
    if constexpr (CL == 1) {
      mbar_init(empty0, NW + (COMP ? NESC : 0));
      for (int i = 1; i < NSTAGE; ++i) {
        if (i >= P) mbar_init(full0 + 8 * i, 1);
        mbar_init(empty0 + 8 * i, NW + (COMP ? NESC : 0));
      }
      if (!COMP) mbar_init(pbar, 1);
      mbar_init(cbar, 32 * NESC);
      mbar_init(qbar, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
  }
  if (tid >= 32 && tid < 64) reinterpret_cast<uint32_t*>(ptab_s + pmax)[tid - 32] = 0u;
  if constexpr (FUSE) {
    if (BND1) {
      // this CTA's 64-column slice of the rows the prologue reads after the wait and that no kernel of this call
      // writes (the input stream per row, alpha, beta, the gain): cold in L2 on a judged replay -> prefetch now
      if (tid == 0) s_hready = 0;
      if (tid < 2 * M + 5) {
        const char* ptr;
        if (tid < 2 * M) ptr = reinterpret_cast<const char*>(b_sin + (size_t)(tid >> 1) * 8192 + cta * 64 + (tid & 1) * 32);
        else if (tid < 2 * M + 2) ptr = reinterpret_cast<const char*>(b_alpha + cta * 64 + (tid - 2 * M) * 32);
        else if (tid < 2 * M + 4) ptr = reinterpret_cast<const char*>(b_beta + cta * 64 + (tid - 2 * M - 2) * 32);
        else ptr = reinterpret_cast<const char*>(b_gn + cta * 64);
        asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr));
      }
    }
  }
  // local barriers are all this CTA needs to start streaming; the cluster barrier (rank 0's qbar initialized before
  // ranks > 0 send into it) is only arrived at here and waited on right before the remote stores
  __syncthreads();
  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");   // every thread: x and the outputs are ours from here
  if constexpr (CL > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");   // inits fenced above

  if (warp == NW) {
    // producer warp: lane 0 refills the ring
    if (lane == 0) {
      int st0 = P;
      if (BND1) {
        // the ring's remaining weight stages go out now (fresh slots, no empty wait) so the weight stream never
        // waits for the prologue; the x boxes of stages 0 .. NSTAGE-1 follow once h is complete (s_hready)
        for (int st = P; st < NSTAGE && st < nst; ++st) {
          mbar_expect_tx(full0 + 8 * st, XBYr + WBY);
          bulk_g2s(sbase + st * SST + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * st);
        }
        while (*reinterpret_cast<volatile int*>(&s_hready) == 0) __nanosleep(64);
        for (int st = 0; st < NSTAGE && st < nst; ++st)
          tma_load_3d(sbase + st * SST, &xmap, full0 + 8 * st, 0, 0, (gs0 + st) * UPS);
        st0 = NSTAGE < nst ? NSTAGE : nst;
      }
      for (int st = st0; st < nstt; ++st) {
        const int slot = st % NSTAGE;
        if (st >= NSTAGE) SK2_WAIT(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
        if (trace) tr[16 + 3 * st] = gtime();
        const uint32_t s0 = sbase + slot * SST;
        if (!FUSE || st < nst) {
          mbar_expect_tx(full0 + 8 * slot, (((dgv & 4) || HSM) ? 0 : XBYr) + WBY);
          bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * slot);
          if (!(dgv & 4) && !HSM) tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, (gs0 + st) * UPS);
        } else {   // router piece: global router stage rq * 4 + j of tile rt
          const int gsr = rq * NSTR + (st - nst);
          mbar_expect_tx(full0 + 8 * slot, (HSM ? 0 : XBYr) + WBYR);
          bulk_g2s(s0 + XBY, wsr + (size_t)gsr * (32 * WBYR), WBYR, full0 + 8 * slot);
          if (!HSM) tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, gsr * UPS);
        }
      }
    }
    __syncwarp();
    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }
  if (warp >= NW + 1) {
    // NESC escape warps take turns on the stages (each has NESC stage-times per stage): exact corrections for the
    // escapes, read from the x tiles of the ring as each stage lands (the table is in k order); lane t owns token t,
    // so corr[t][row] accumulates with no atomics; each warp has its own copy
    const int ew = warp - (NW + 1);
    if (COMP) {
      float* cr = corr + ew * CSZ;
      for (int i = lane; i < TOKS * CP; i += 32) cr[i] = 0.f;
      mbar_wait_sleep(pbar, 0);
      const int cnt = ptab_s[0].x;
      constexpr int KST = UPS * 64;
      int pe = 1;
      for (int st = 0; st < nstt; ++st) {
        const int slot = st % NSTAGE;
        if (FUSE && st >= nst) {   // router stages: nothing to correct, only the slot hand-back (in order, as below)
          SK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);
          __syncwarp();
          if (lane == 0) mbar_arrive(empty0 + 8 * slot);   // every escape warp hands every stage back
          continue;
        }
        const int kbeg = (gs0 + st) * KST, kend = kbeg + KST;
        // Every escape warp observes every stage, in order, works only on its own (st % NESC == ew), and hands EVERY
        // stage back (the empty barriers count NW + NESC arrivals). The producer therefore never refills a slot
        // before every escape warp has seen the stage it held, so no warp can find its slot two phases ahead (a
        // parity wait that aliases and wedges the ring: astra review 2B, reproduced with a delayed escape warp).
        SK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);
        if ((st % NESC) != ew) {   // another warp's stage: skip its escapes, hand the slot back
          while (pe <= cnt && (ptab_s[pe].x & 0xFFFF) < kend) ++pe;
          __syncwarp();
          if (lane == 0) mbar_arrive(empty0 + 8 * slot);
          continue;
        }
        const unsigned char* xs = HSM ? gbase + (hsm - sbase) + (uint32_t)(gs0 + st) * XBYr : gbase + slot * SST;
        while (pe <= cnt) {
          const int2 e = ptab_s[pe];
          const int kk = e.x & 0xFFFF;
          if (kk >= kend) break;
          if (lane < M && kk >= kbeg) {
            const int kl = kk - kbeg, u = kl >> 6, kc = kl & 63;
            const __nv_bfloat16 v = *reinterpret_cast<const __nv_bfloat16*>(
                xs + u * XU + lane * 128 + (((kc >> 3) ^ ((u * Mb + lane) & 7)) << 4) + (kc & 7) * 2);
            cr[lane * CP + (e.x >> 16)] += __int_as_float(e.y) * __bfloat162float(v);
          }
          ++pe;
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
#if SK2_DELAY_ESC
        if (ew == 0 && st == 0) {   // debug: escape warp 0 stalls after its first hand-back (the ring must not wedge)
          const unsigned long long t0 = gtime();
          while (gtime() - t0 < (unsigned long long)SK2_DELAY_ESC) __nanosleep(500);
        }
#endif
      }
    }
    mbar_arrive(cbar);
    __syncwarp();
    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }

  // ---------------- consumers
  const int wr = warp / WSK, wk = warp % WSK;
  const int g = lane >> 2, t4 = lane & 3;
  const int q = lane >> 3;
  const int chi = q >> 1;
  int trow[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) trow[mt] = TOKR >= 16 ? mt * 16 + (lane & 7) + 8 * (q & 1) : (lane & 7);
  float acc[NACC][MT][NT][4];
#pragma unroll
  for (int c = 0; c < NACC; ++c)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) acc[c][mt][nt][0] = acc[c][mt][nt][1] = acc[c][mt][nt][2] = acc[c][mt][nt][3] = 0.f;

  if constexpr (FUSE) {
    if (BND1) {
      // ---- cooperative boundary-1 prologue: warp w handles row w (M <= 8 rows), lane l the columns
      // 64 cta + 2 l, 2 l + 1. Phase A: partials C = sum a^2, A = sum (alpha s)^2, B = sum (alpha s)(beta a),
      // D = sum (beta a)^2 over this slice -> b_part[cta][w][0..3]. Ticket 1. Phase B: every CTA sums the 128
      // partials of its rows in the same order (identical r, c in every CTA), writes its h and s1 columns
      // (h = (alpha s + r beta a) c (1 + w), r = rsqrt(C/8192 + 1e-8), c = rsqrt((A + 2rB + r^2 D)/8192 + 1e-5)),
      // fences the generic writes for the TMA proxy, ticket 2, then s_hready lets the producer issue the x boxes.
      static_assert(NC == 256, "sk2 boundary prologue: 8 consumer warps");
      const int col = cta * 64 + lane * 2;
      float a0 = 0.f, a1 = 0.f, s0v = 0.f, s1v = 0.f, al0 = 0.f, al1 = 0.f, be0 = 0.f, be1 = 0.f;
      if (warp < M) {
        const uint32_t av = __ldcg(reinterpret_cast<const uint32_t*>(b_gp + (size_t)warp * 8192 + col));
        const float2 af = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&av));
        const float2 sv = __ldg(reinterpret_cast<const float2*>(b_sin + (size_t)warp * 8192 + col));
        const float2 lv = __ldg(reinterpret_cast<const float2*>(b_alpha + col));
        const float2 bv = __ldg(reinterpret_cast<const float2*>(b_beta + col));
        a0 = af.x; a1 = af.y; s0v = sv.x; s1v = sv.y; al0 = lv.x; al1 = lv.y; be0 = bv.x; be1 = bv.y;
      }
      float pC = a0 * a0 + a1 * a1;
      float pA = (al0 * s0v) * (al0 * s0v) + (al1 * s1v) * (al1 * s1v);
      float pB = (al0 * s0v) * (be0 * a0) + (al1 * s1v) * (be1 * a1);
      float pD = (be0 * a0) * (be0 * a0) + (be1 * a1) * (be1 * a1);
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        pC += __shfl_xor_sync(0xffffffffu, pC, off); pA += __shfl_xor_sync(0xffffffffu, pA, off);
        pB += __shfl_xor_sync(0xffffffffu, pB, off); pD += __shfl_xor_sync(0xffffffffu, pD, off);
      }
      if (warp < M && lane == 0)
        __stcg(reinterpret_cast<float4*>(b_part + ((size_t)cta * 8 + warp) * 4), make_float4(pC, pA, pB, pD));
      __shared__ int s_btick;
      named_bar(1, NC);
      if (tid == 0) {   // ticket 1: this CTA's partials released; wait for every CTA's (one wave, co-resident)
        int old;
        asm volatile("atom.add.release.gpu.global.s32 %0, [%1], 1;" : "=r"(old) : "l"(b_cnt) : "memory");
        const int G = (int)gridDim.x;
        okc_spin_ge(b_cnt, (old / G + 1) * G, 32);
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
        s_btick = old;
      }
      named_bar(1, NC);
      if (warp < M) {
        // the row's 128 partial float4s: lane l sums CTAs l, l+32, l+64, l+96 (4 independent loads), then a tree
        float C = 0.f, A = 0.f, B = 0.f, D = 0.f;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float4 v = __ldcg(reinterpret_cast<const float4*>(b_part + ((size_t)(lane + 32 * k) * 8 + warp) * 4));
          C += v.x; A += v.y; B += v.z; D += v.w;
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
          C += __shfl_xor_sync(0xffffffffu, C, off); A += __shfl_xor_sync(0xffffffffu, A, off);
          B += __shfl_xor_sync(0xffffffffu, B, off); D += __shfl_xor_sync(0xffffffffu, D, off);
        }
        const float r = rsqrtf(C * (1.f / 8192.f) + 1e-8f);
        const float c = rsqrtf((A + 2.f * r * B + r * r * D) * (1.f / 8192.f) + 1e-5f);
        const float t0 = al0 * s0v + be0 * (a0 * r), t1 = al1 * s1v + be1 * (a1 * r);
        const uint32_t gv = __ldg(reinterpret_cast<const uint32_t*>(b_gn + col));
        const float2 gf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gv));
        const __nv_bfloat162 hv = __floats2bfloat162_rn(t0 * c * (1.f + gf.x), t1 * c * (1.f + gf.y));
        *reinterpret_cast<__nv_bfloat162*>(b_hout + (size_t)warp * 8192 + col) = hv;
        *reinterpret_cast<float2*>(b_sout + (size_t)warp * 8192 + col) = make_float2(t0, t1);
      }
      asm volatile("fence.proxy.async.global;" ::: "memory");   // h (generic stores) -> the TMA boxes of every CTA
      named_bar(1, NC);
      if (tid == 0) {   // ticket 2: every CTA's h slice written; then the producer may issue the x boxes
        int old;
        asm volatile("atom.add.release.gpu.global.s32 %0, [%1], 1;" : "=r"(old) : "l"(b_cnt) : "memory");
        const int G = (int)gridDim.x;
        okc_spin_ge(b_cnt, (old / G + 1) * G, 32);
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
        *reinterpret_cast<volatile int*>(&s_hready) = 1;
      }
      named_bar(1, NC);
    }
  }

  for (int st = 0; st < nst; ++st) {
    const int slot = st % NSTAGE;
    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
    if (trace && warp == 0 && lane == 0) tr[16 + 3 * st + 1] = gtime();
    const uint32_t s0 = sbase + slot * SST;
    if (dgv & 2) { __syncwarp(); if (lane == 0) mbar_arrive(empty0 + 8 * slot); continue; }
#pragma unroll
    for (int uu = 0; uu < UPW; ++uu) {
      const int ui = wk * UPW + uu;
      const uint32_t xt = (HSM ? hsm + (uint32_t)(gs0 + st) * XBYr : s0) + ui * XU;
      const uint32_t wb = s0 + XBY + (warp * UPW + uu) * NT * TU;
      // 1) every shared-memory load of this unit first (weights of all tiles, x fragments of all 4 k-steps)
      uint4 lw[NT];
      uint2 cw[NT];
      uint32_t b[NT][8];
      uint32_t bl[HL ? NT : 1][8];
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        if constexpr (COMP) {
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(lw[nt].x), "=r"(lw[nt].y), "=r"(lw[nt].z), "=r"(lw[nt].w) : "r"(wb + nt * TU + lane * 16));
          asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(cw[nt].x), "=r"(cw[nt].y) : "r"(wb + nt * TU + 512 + lane * 8));
        } else {
          const uint32_t a0 = wb + nt * TU + lane * 32;
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(b[nt][0]), "=r"(b[nt][1]), "=r"(b[nt][2]), "=r"(b[nt][3]) : "r"(a0));
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(b[nt][4]), "=r"(b[nt][5]), "=r"(b[nt][6]), "=r"(b[nt][7]) : "r"(a0 + 16));
          if constexpr (HL) {
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bl[nt][0]), "=r"(bl[nt][1]), "=r"(bl[nt][2]), "=r"(bl[nt][3]) : "r"(a0 + 1024));
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bl[nt][4]), "=r"(bl[nt][5]), "=r"(bl[nt][6]), "=r"(bl[nt][7]) : "r"(a0 + 1040));
          }
        }
      }
      uint32_t a[4][MT][4];
      if (dgv & 16) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) a[j][mt][0] = a[j][mt][1] = a[j][mt][2] = a[j][mt][3] = 0x3f803f80u;
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int mt = 0; mt < MT; ++mt)
            ldsm_x4((PK && trow[mt] >= M) ? zrow
                        : xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * Mb + trow[mt]) & 7)) << 4),
                    a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
      }
      // the stage's shared memory is in registers now: hand the slot back before decoding and multiplying
      if (UPW == 1 && uu == 0) {
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      }
      // 2) decode
      if constexpr (COMP) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          decode8(lw[nt].x, lw[nt].y, cw[nt].x, base4, &b[nt][0]);
          decode8(lw[nt].z, lw[nt].w, cw[nt].y, base4, &b[nt][4]);
        }
      }
      // 3) tensor-core work: independent accumulators per k-step
      if (dgv & 8) {   // diagnostic: replace the tensor-core instructions by ALU work
        const int nh = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            if (j < nh) mma_bf16(acc[j % NACC][0][nt], a[j][0][0], a[j][0][1], a[j][0][2], a[j][0][3], b[nt][2 * j], b[nt][2 * j + 1]);
            else acc[j % NACC][0][nt][0] += __uint_as_float((b[nt][2 * j] ^ b[nt][2 * j + 1]) & a[j][0][0] & 0x007fffffu);
          }
        continue;
      }
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            constexpr int HM = HL ? 2 : 1;
            mma_bf16(acc[((uu * 4 + j) * HM) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], b[nt][2 * j], b[nt][2 * j + 1]);
            if constexpr (HL)
              mma_bf16(acc[((uu * 4 + j) * HM + 1) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], bl[nt][2 * j], bl[nt][2 * j + 1]);
          }
    }
    if (UPW > 1) {
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
    if (trace && warp == 0 && lane == 0) tr[16 + 3 * st + 2] = gtime();
  }
  if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");   // the next kernel may start its prologue
  if constexpr (FUSE) {
    // router piece: 8 rows (one tile) over 4 stages of 8 units; hi and lo BF16 tiles into two FP32 accumulators
    float accr[2][MT][4];
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) accr[h][mt][0] = accr[h][mt][1] = accr[h][mt][2] = accr[h][mt][3] = 0.f;
    for (int st = nst; st < nstt; ++st) {
      const int slot = st % NSTAGE;
      mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
      const uint32_t s0 = sbase + slot * SST;
      const int ui = wk;
      const uint32_t xt = (HSM ? hsm + (uint32_t)(rq * NSTR + (st - nst)) * XBYr : s0) + ui * XU;
      const uint32_t a0 = s0 + XBY + warp * 2048 + lane * 32;
      uint32_t bh[8], bo[8];
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bh[0]), "=r"(bh[1]), "=r"(bh[2]), "=r"(bh[3]) : "r"(a0));
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bh[4]), "=r"(bh[5]), "=r"(bh[6]), "=r"(bh[7]) : "r"(a0 + 16));
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bo[0]), "=r"(bo[1]), "=r"(bo[2]), "=r"(bo[3]) : "r"(a0 + 1024));
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bo[4]), "=r"(bo[5]), "=r"(bo[6]), "=r"(bo[7]) : "r"(a0 + 1040));
      uint32_t a[4][MT][4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
          ldsm_x4((PK && trow[mt] >= M) ? zrow
                      : xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * Mb + trow[mt]) & 7)) << 4),
                  a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          mma_bf16(accr[0][mt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], bh[2 * j], bh[2 * j + 1]);
          mma_bf16(accr[1][mt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], bo[2 * j], bo[2 * j + 1]);
        }
    }
    // k-warp reduction of the 8 x TOKS piece through the (now idle) ring: redr[wk][tok][8]
    if constexpr (COMP) mbar_wait(cbar, 0);   // the escape warps are done with the ring too
    named_bar(1, NC);
    float* redr = reinterpret_cast<float*>(gbase);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      const int row = 2 * t4, tok = mt * 16 + g;
      redr[(wk * TOKS + tok) * 8 + row] = accr[0][mt][0] + accr[1][mt][0];
      redr[(wk * TOKS + tok) * 8 + row + 1] = accr[0][mt][1] + accr[1][mt][1];
      redr[(wk * TOKS + tok + 8) * 8 + row] = accr[0][mt][2] + accr[1][mt][2];
      redr[(wk * TOKS + tok + 8) * 8 + row + 1] = accr[0][mt][3] + accr[1][mt][3];
    }
    named_bar(1, NC);
    // this CTA's quarter-K partial -> part[rt][rq][TOKS][8]; the last of the 4 quarters (release/acquire counter)
    // adds them in quarter order (deterministic) and writes the router output
    float* pp = part + ((size_t)rt * 4 + rq) * 32 * 8;
    for (int i = tid; i < TOKS * 8; i += NC) {
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < WSK; ++w) v += redr[(w * TOKS + (i >> 3)) * 8 + (i & 7)];
      __stcg(pp + i, v);
    }
    named_bar(1, NC);
    __shared__ int s_rlast;
    if (tid == 0) {
      int old;
      asm volatile("atom.add.release.gpu.global.s32 %0, [%1], 1;" : "=r"(old) : "l"(rcnt + rt) : "memory");
      s_rlast = (old & 3) == 3;
      if (s_rlast) asm volatile("fence.acq_rel.gpu;" ::: "memory");
    }
    named_bar(1, NC);
    if (s_rlast) {
      const float* pr = part + (size_t)rt * 4 * 32 * 8;
      for (int i = tid; i < M * 8; i += NC) {
        const float v = __ldcg(pr + i) + __ldcg(pr + 256 + i) + __ldcg(pr + 512 + i) + __ldcg(pr + 768 + i);
        outr[(size_t)(i >> 3) * 256 + rt * 8 + (i & 7)] = v;
      }
    }
    named_bar(1, NC);   // the ring (redr) is reused by the pooler epilogue below
  }
  if (trace && lane == 0) tr[1 + warp] = gtime();   // per consumer warp: main loop done

  // ---------------- epilogue: K-warps reduce through shared memory (the ring is free now)
  if (dgv & 1) {
    if (acc[0][0][0][0] == 1.2345e-30f) out[tid] = cvt_out<OutT>(acc[NACC - 1][0][0][1]);
    return;
  }
  if constexpr (COMP) mbar_wait(cbar, 0);   // the escape warp reads the ring's x tiles until it is done
  if (CL > 1 && crank > 0) {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");   // rank 0's qbar is initialized
    // ranks > 0: every accumulator goes straight into rank 0's peer buffer (slot crank - 1) by st.async
    // (completes rank 0's qbar transaction count; no fence, no wait), escape corrections folded into k-warp 0's
    const uint32_t peer_l = (uint32_t)__cvta_generic_to_shared(peer + (crank - 1) * WSK * TOKS * RP);
    uint32_t peer_r, qbar_r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(peer_r) : "r"(peer_l));
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(qbar_r) : "r"(qbar));
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int row = (wr * NT + nt) * 8 + 2 * t4;
        const int tok = mt * 16 + g;
        float v[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          v[e] = acc[0][mt][nt][e];
#pragma unroll
          for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
        }
        if (COMP && wk == 0) {
          v[0] += corr[tok * CP + row];
          v[1] += corr[tok * CP + row + 1];
          if (tok + 8 < 32) {
            v[2] += corr[(tok + 8) * CP + row];
            v[3] += corr[(tok + 8) * CP + row + 1];
          }
        }
        const uint32_t a0 = peer_r + (uint32_t)(((wk * TOKS + tok) * RP + row) * 4);
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];"
                     :: "r"(a0), "r"(__float_as_uint(v[0])), "r"(__float_as_uint(v[1])), "r"(qbar_r) : "memory");
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];"
                     :: "r"(a0 + 8 * RP * 4), "r"(__float_as_uint(v[2])), "r"(__float_as_uint(v[3])), "r"(qbar_r) : "memory");
      }
    if (trace && tid == 0) tr[15] = gtime();
    return;
  }
  named_bar(1, NC);
  if (trace && tid == 0) tr[9] = gtime();
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int row = (wr * NT + nt) * 8 + 2 * t4;
      const int tok = mt * 16 + g;
      float* r0 = red + (wk * TOKS + tok) * RP + row;
      float v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        v[e] = acc[0][mt][nt][e];
#pragma unroll
        for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
      }
      if (trace && mt == 0 && nt == 0) {
        asm volatile("" :: "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]));
        if (tid == 0) tr[12] = gtime();
      }
      *reinterpret_cast<float2*>(r0) = make_float2(v[0], v[1]);
      *reinterpret_cast<float2*>(r0 + 8 * RP) = make_float2(v[2], v[3]);
    }
  if (trace && tid == 0) tr[11] = gtime();
  named_bar(1, NC);
  if (trace && tid == 0) tr[10] = gtime();
  if (trace && tid == 0) tr[13] = gtime();
  if constexpr (CL > 1) {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    mbar_wait_cluster(qbar, 0);
  }
  if (trace && tid == 0) tr[14] = gtime();
  constexpr int R4 = R / 4;
  float ssq = 0.f;   // FUSE route: this thread's sum of squares of its bf16 pooled outputs
  __nv_bfloat162 keep_lo = __floats2bfloat162_rn(0.f, 0.f), keep_hi = keep_lo;   // FUSE norm: this thread's 4 bf16
  int keep_tt = -1, keep_row = 0;                                                  // pooled values and their position
  for (int idx = tid; idx < R4 * M; idx += NC) {
    const int tt = idx / R4, row = (idx - tt * R4) * 4;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if constexpr (COMP) {
#pragma unroll
      for (int c = 0; c < NESC; ++c) {
        v.x += corr[c * CSZ + tt * CP + row];
        v.y += corr[c * CSZ + tt * CP + row + 1];
        v.z += corr[c * CSZ + tt * CP + row + 2];
        v.w += corr[c * CSZ + tt * CP + row + 3];
      }
    }
#pragma unroll
    for (int w = 0; w < WSK; ++w) {
      const float4 p = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row);
      v.x += p.x; v.y += p.y; v.z += p.z; v.w += p.w;
#pragma unroll
      for (int c = 0; c < CL - 1; ++c) {
        const float4 q = *reinterpret_cast<const float4*>(peer + ((c * WSK + w) * TOKS + tt) * RP + row);
        v.x += q.x; v.y += q.y; v.z += q.z; v.w += q.w;
      }
    }
    OutT* o = out + (size_t)tt * N + cta * R + row;
    if constexpr (sizeof(OutT) == 4) {
      *reinterpret_cast<float4*>(o) = v;
    } else {
      if constexpr (FUSE) {
        if (gpre != nullptr && dcnt == nullptr) {
          // mode 3 (routing + the per-token RMS scale in the MoE kernel): the output is pooled * (1 + w) in bf16 (column
          // gain only); the fp32 partials of sum(pooled^2) per token (this CTA's 32 columns) go to part2[tok][cta], the
          // MoE kernel forms rs_t = rsqrt(mean + 1e-5) and applies it in its GEMM1 epilogue. No grid barrier.
          ssq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
          const uint2 gq = __ldg(reinterpret_cast<const uint2*>(gpre + cta * R + row));
          const float2 g0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gq.x));
          const float2 g1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gq.y));
          v.x *= 1.f + g0.x; v.y *= 1.f + g0.y; v.z *= 1.f + g1.x; v.w *= 1.f + g1.y;
        }
      }
      __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y), hi = __floats2bfloat162_rn(v.z, v.w);
      uint2 pk;
      pk.x = *reinterpret_cast<uint32_t*>(&lo);
      pk.y = *reinterpret_cast<uint32_t*>(&hi);
      *reinterpret_cast<uint2*>(o) = pk;
      if constexpr (FUSE) {
        if (!(gpre != nullptr && dcnt == nullptr)) {   // modes 1/2: the bf16-rounded pooled values, as _route_split
          const float2 l2 = __bfloat1622float2(lo), h2 = __bfloat1622float2(hi);
          ssq += l2.x * l2.x + l2.y * l2.y + h2.x * h2.x + h2.y * h2.y;
        }
        keep_lo = lo; keep_hi = hi; keep_tt = tt; keep_row = row;
      }
      if constexpr (BT) {   // boundary tail: sum of squares of (a * (1 + gp)) over this CTA's columns, bf16-rounded a
        float2 l2 = __bfloat1622float2(lo), h2 = __bfloat1622float2(hi);
        if (b_gp != nullptr) {
          const uint2 gq = __ldg(reinterpret_cast<const uint2*>(b_gp + cta * R + row));
          const float2 g0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gq.x));
          const float2 g1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gq.y));
          l2.x *= 1.f + g0.x; l2.y *= 1.f + g0.y; h2.x *= 1.f + g1.x; h2.y *= 1.f + g1.y;
        }
        ssq += l2.x * l2.x + l2.y * l2.y + h2.x * h2.x + h2.y * h2.y;
      }
    }
  }
  if constexpr (BT) {
    if (b_cnt != nullptr) {
      // ---- boundary tail (decode layer, M <= 8): every CTA leaves its per-token partial sum of squares of the GEMM
      // output a (times (1 + gp) for the post-FFN boundary), takes a grid ticket, and the LAST CTA computes
      //   s = alpha * sin + beta * a * (1 + gp) * rsqrt(mean((a (1 + gp))^2) + 1e-8)         (rows of width 8192; a = 0
      //   h = s * rsqrt(mean(s^2) + 1e-5) * (1 + gn)                                          beyond the GEMM's N columns)
      // i.e. the layer's _boundary1 (o projection -> residual stream s1, pre-FFN norm h) or _boundary2 (unpooler slice
      // -> stream s2, next layer's norm) without a separate launch.
      static_assert(sizeof(OutT) == 2 && !HL && CL == 1, "sk2 boundary tail: bf16 output, one CTA per row block");
      constexpr int R4c = R / 4;
      static_assert((R4c & (R4c - 1)) == 0 && R4c <= 32 && NC >= 128, "sk2 boundary tail: lane groups");
#pragma unroll
      for (int off = 1; off < R4c; off <<= 1) ssq += __shfl_xor_sync(0xffffffffu, ssq, off);
      if ((tid % R4c) == 0 && (tid / R4c) < M) __stcg(b_part + (tid / R4c) * 256 + cta, ssq);   // token-major [32][256 CTAs]
      __shared__ int s_blast;
      named_bar(1, NC);
      if (tid == 0) {
        int old;
        asm volatile("atom.add.release.gpu.global.s32 %0, [%1], 1;" : "=r"(old) : "l"(b_cnt) : "memory");
        s_blast = (old % (int)gridDim.x) == (int)gridDim.x - 1;
        if (s_blast) asm volatile("fence.acq_rel.gpu;" ::: "memory");
      }
      named_bar(1, NC);
      if (!s_blast) return;
      float* ssm = red;          // [32] per-token sum of squares of a (1 + gp), the CTA partials in a fixed (tree) order
      float* ssm2 = red + 32;    // [32] per-token sum of squares of s
      float* wred = red + 64;    // [NW][32] per-warp partials of the latter
      for (int tt = warp; tt < M; tt += NW) {   // one warp per token: <= 8 independent loads per lane, shuffle tree
        float s = 0.f;
        for (int c = lane; c < (int)gridDim.x; c += 32) s += __ldcg(b_part + tt * 256 + c);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
        if (lane == 0) ssm[tt] = s;
      }
      named_bar(1, NC);
      float acc[8];
#pragma unroll
      for (int q = 0; q < 8; ++q) acc[q] = 0.f;
      for (int i = tid; i < M * 1024; i += NC) {   // 8 columns per step over rows of 8192
        const int tt = i >> 10, c8 = (i & 1023) * 8;
        const float rs = rsqrtf(ssm[tt] * (1.f / 8192.f) + 1e-8f);
        float a[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) a[j] = 0.f;
        if (c8 < N) {
          const uint4 pv = __ldcg(reinterpret_cast<const uint4*>(out + (size_t)tt * N + c8));
          const __nv_bfloat162* p2 = reinterpret_cast<const __nv_bfloat162*>(&pv);
#pragma unroll
          for (int j = 0; j < 4; ++j) { const float2 p = __bfloat1622float2(p2[j]); a[2 * j] = p.x; a[2 * j + 1] = p.y; }
          if (b_gp != nullptr) {
            const uint4 gv = __ldg(reinterpret_cast<const uint4*>(b_gp + c8));
            const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&gv);
#pragma unroll
            for (int j = 0; j < 4; ++j) { const float2 g = __bfloat1622float2(g2[j]); a[2 * j] *= 1.f + g.x; a[2 * j + 1] *= 1.f + g.y; }
          }
        }
        const float4 al0 = __ldg(reinterpret_cast<const float4*>(b_alpha + c8)), al1 = __ldg(reinterpret_cast<const float4*>(b_alpha + c8 + 4));
        const float4 be0 = __ldg(reinterpret_cast<const float4*>(b_beta + c8)), be1 = __ldg(reinterpret_cast<const float4*>(b_beta + c8 + 4));
        const float4 si0 = __ldg(reinterpret_cast<const float4*>(b_sin + (size_t)tt * 8192 + c8));
        const float4 si1 = __ldg(reinterpret_cast<const float4*>(b_sin + (size_t)tt * 8192 + c8 + 4));
        const float al[8] = {al0.x, al0.y, al0.z, al0.w, al1.x, al1.y, al1.z, al1.w};
        const float be[8] = {be0.x, be0.y, be0.z, be0.w, be1.x, be1.y, be1.z, be1.w};
        const float si[8] = {si0.x, si0.y, si0.z, si0.w, si1.x, si1.y, si1.z, si1.w};
        float s[8], sq = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) { s[j] = al[j] * si[j] + be[j] * (a[j] * rs); sq += s[j] * s[j]; }
        *reinterpret_cast<float4*>(b_sout + (size_t)tt * 8192 + c8) = make_float4(s[0], s[1], s[2], s[3]);
        *reinterpret_cast<float4*>(b_sout + (size_t)tt * 8192 + c8 + 4) = make_float4(s[4], s[5], s[6], s[7]);
#pragma unroll
        for (int q = 0; q < 8; ++q) if (q == tt) acc[q] += sq;
      }
#pragma unroll
      for (int q = 0; q < 8; ++q) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc[q] += __shfl_xor_sync(0xffffffffu, acc[q], off);
        if (lane == 0) wred[warp * 32 + q] = acc[q];
      }
      named_bar(1, NC);
      if (tid < M) {
        float s = 0.f;
#pragma unroll
        for (int w = 0; w < NW; ++w) s += wred[w * 32 + tid];
        ssm2[tid] = s;
      }
      named_bar(1, NC);
      for (int i = tid; i < M * 1024; i += NC) {
        const int tt = i >> 10, c8 = (i & 1023) * 8;
        const float rs = rsqrtf(ssm2[tt] * (1.f / 8192.f) + 1e-5f);
        const float4 s0 = __ldcg(reinterpret_cast<const float4*>(b_sout + (size_t)tt * 8192 + c8));
        const float4 s1 = __ldcg(reinterpret_cast<const float4*>(b_sout + (size_t)tt * 8192 + c8 + 4));
        const uint4 gv = __ldg(reinterpret_cast<const uint4*>(b_gn + c8));
        const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&gv);
        const float s[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
        uint4 ov;
        __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&ov);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 g = __bfloat1622float2(g2[j]);
          o2[j] = __floats2bfloat162_rn(s[2 * j] * rs * (1.f + g.x), s[2 * j + 1] * rs * (1.f + g.y));
        }
        *reinterpret_cast<uint4*>(b_hout + (size_t)tt * 8192 + c8) = ov;
      }
    }
  }
  if constexpr (FUSE) {
    if (dcnt == nullptr && part2 != nullptr) {   // mode 3: the partials only (token-major [32][128 CTAs]); no barrier
      ssq += __shfl_xor_sync(0xffffffffu, ssq, 1);
      ssq += __shfl_xor_sync(0xffffffffu, ssq, 2);
      ssq += __shfl_xor_sync(0xffffffffu, ssq, 4);
      if ((tid & 7) == 0 && (tid >> 3) < M) __stcg(part2 + (tid >> 3) * 128 + cta, ssq);
    }
    if (dcnt != nullptr) {
      // ---- fused pre-expert norm (decode layer; routing by the MoE kernel from the logits, or by CTA t below): every CTA
      // leaves its per-token partial sum of squares of its 32 bf16 pooled columns in part2[tok][cta], arrives on the
      // grid ticket and WAITS for the whole launch (128 CTAs = one wave, co-resident: nothing in this launch can be
      // scheduled after a dependent), then every CTA sums the 128 partials in the same order (identical bits) and
      // scales its own 4 columns per token, still in registers: xm = pooled * rsqrt(mean(pooled^2) + 1e-5) * (1 + w).
      // No serial tail; the only cost is the barrier (~the spread of the CTAs' finish times).
      static_assert(R == 32 && NC >= 256, "sk2 fused norm: pooler config (R = 32, 8 consumer warps)");
      ssq += __shfl_xor_sync(0xffffffffu, ssq, 1);
      ssq += __shfl_xor_sync(0xffffffffu, ssq, 2);
      ssq += __shfl_xor_sync(0xffffffffu, ssq, 4);
      if ((tid & 7) == 0 && (tid >> 3) < M) __stcg(part2 + (tid >> 3) * 128 + cta, ssq);   // token-major [32][128 CTAs]
      __shared__ int s_ntarget;
      named_bar(1, NC);
      if (tid == 0) {   // release this CTA's partial (and its pooled / router rows), then wait for every CTA's
        int old;
        asm volatile("atom.add.release.gpu.global.s32 %0, [%1], 1;" : "=r"(old) : "l"(dcnt) : "memory");
        const int G = (int)gridDim.x;
        const int target = (old / G + 1) * G;   // the counter grows by G per launch: this launch completes at target
        okc_spin_ge(dcnt, target, 32);
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
        s_ntarget = target;
      }
      named_bar(1, NC);
      {
        // the token's 8 lanes each sum 16 partials, then a 3-step tree: the same order in every CTA
        const int tt = keep_tt < 0 ? 0 : keep_tt;
        float s = 0.f;
        for (int c = (tid & 7); c < (int)gridDim.x; c += 8) s += __ldcg(part2 + tt * 128 + c);
        s += __shfl_xor_sync(0xffffffffu, s, 1);
        s += __shfl_xor_sync(0xffffffffu, s, 2);
        s += __shfl_xor_sync(0xffffffffu, s, 4);
        if (keep_tt >= 0) {
          const float rs = rsqrtf(s * (1.f / 4096.f) + 1e-5f);
          const int col = cta * R + keep_row;
          const uint2 gq = __ldg(reinterpret_cast<const uint2*>(gpre + col));
          const float2 g0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gq.x));
          const float2 g1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gq.y));
          const float2 p0 = __bfloat1622float2(keep_lo), p1 = __bfloat1622float2(keep_hi);
          const __nv_bfloat162 o0 = __floats2bfloat162_rn(p0.x * rs * (1.f + g0.x), p0.y * rs * (1.f + g0.y));
          const __nv_bfloat162 o1 = __floats2bfloat162_rn(p1.x * rs * (1.f + g1.x), p1.y * rs * (1.f + g1.y));
          uint2 ov;
          ov.x = *reinterpret_cast<const uint32_t*>(&o0);
          ov.y = *reinterpret_cast<const uint32_t*>(&o1);
          *reinterpret_cast<uint2*>(xm + (size_t)keep_tt * 4096 + col) = ov;
        }
      }
      if (rids != nullptr && cta < M && warp == 0) {
        // routing of token `cta` by this CTA's warp 0 (tokens in parallel across CTAs; every router tile's logits
        // are complete after the barrier): the layer's _route_split selection -- top-8 of sigmoid(0.1768 logit) +
        // bias, ties to the lower id, weights = the selected sigmoid scores over their sum, local id = (id - start)
        // mod 256 if < 32 else -1
        const float NEG_INF = __int_as_float(0xff800000u);
        const int ls = __ldcg(rls);
        const int tt = cta;
        float sc[8], sel[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float l = __ldcg(outr + (size_t)tt * 256 + lane * 8 + j);
          sc[j] = 1.f / (1.f + __expf(-l * 0.1767766952966369f));
          sel[j] = sc[j] + __ldg(rbias + lane * 8 + j);
        }
        float wsum = 0.f, myw = 0.f;
        int myid = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          float bm = NEG_INF;
          int bi = lane * 8;
#pragma unroll
          for (int j = 0; j < 8; ++j)
            if (sel[j] > bm) { bm = sel[j]; bi = lane * 8 + j; }
#pragma unroll
          for (int off = 16; off > 0; off >>= 1) {
            const float om = __shfl_xor_sync(0xffffffffu, bm, off);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (om > bm || (om == bm && oi < bi)) { bm = om; bi = oi; }
          }
          float mine = 0.f;
          if ((bi >> 3) == lane) {
#pragma unroll
            for (int j = 0; j < 8; ++j)
              if ((bi & 7) == j) { mine = sc[j]; sel[j] = NEG_INF; }
          }
          const float val = __shfl_sync(0xffffffffu, mine, bi >> 3);
          wsum += val;
          if (lane == k) { myid = bi; myw = val; }
        }
        if (lane < 8) {
          rids[tt * 8 + lane] = myid;
          rwts[tt * 8 + lane] = myw / (wsum + 1e-15f);
          const int lid = (myid - ls + 256) % 256;
          rlids[tt * 8 + lane] = lid < 32 ? lid : -1;
        }
      }
    }
  }
  if (trace && tid == 0) tr[15] = gtime();
  if ((dgv & 32) && dbg != nullptr && tid == 0) {   // every CTA: SM id, start and end time
    unsigned smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    dbg[8192 + 3 * cta] = smid; dbg[8192 + 3 * cta + 1] = t_start; dbg[8192 + 3 * cta + 2] = gtime();
  }
#endif
}

template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT,
          bool FUSE = false, bool BT = false>
static void launch_sk2(const at::Tensor& Ws, const at::Tensor& x, at::Tensor& out, const at::Tensor& ptab,
                       int64_t pmax, int64_t base4, int M, int N, int K, int64_t dg, unsigned long long* dbg,
                       const uint8_t* wsr = nullptr, float* outr = nullptr, float* part = nullptr, int* rcnt = nullptr,
                       const float* rbias = nullptr, const __nv_bfloat16* gpre = nullptr, int* rids = nullptr,
                       int* rlids = nullptr, float* rwts = nullptr, __nv_bfloat16* xmo = nullptr, const int* rls = nullptr,
                       float* part2 = nullptr, int* dcnt = nullptr,
                       const float* b_alpha = nullptr, const float* b_beta = nullptr, const float* b_sin = nullptr,
                       const __nv_bfloat16* b_gp = nullptr, const __nv_bfloat16* b_gn = nullptr, float* b_sout = nullptr,
                       __nv_bfloat16* b_hout = nullptr, float* b_part = nullptr, int* b_cnt = nullptr) {
  constexpr int NW = NWR * WSK;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int TU = COMP ? 768 : (HL ? 2048 : 1024);
  constexpr int WBY = NW * UPW * NT * TU;
  constexpr int XBY = UPS * TOKR * 128;
  constexpr int SST = ((XBY + WBY) + 1023) & ~1023;
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);
  constexpr int CP = R + 1;
  constexpr int NESC = COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1;
  const size_t smem = 1024 + (size_t)NSTAGE * SST + 16 * NSTAGE + 32 + (size_t)NESC * ((TOKS * CP + 3) & ~3) * 4 +
                      (CL > 1 ? (size_t)(CL - 1) * WSK * TOKS * RP * 4 : 0) + (size_t)pmax * 8 + 128;
  auto go = [&](auto pkc) {
    constexpr bool PK = decltype(pkc)::value;
    auto kern = sk2_kernel<MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL, OutT, PK, FUSE, BT>;
    static bool attr_done = false;
    if (!attr_done) {
      int dev = 0, optin = 0;
      cudaGetDevice(&dev);
      cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
      cudaFuncAttributes fa;
      cudaFuncGetAttributes(&fa, kern);
      cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);
      attr_done = true;
    }
    CUtensorMap xm = make_map_x3d(x.data_ptr(), (uint64_t)M, (uint64_t)K, PK ? (uint32_t)M : (uint32_t)TOKR, UPS);
    const uint8_t* wsp = reinterpret_cast<const uint8_t*>(Ws.data_ptr());
    const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    OutT* op = reinterpret_cast<OutT*>(out.data_ptr());
    const int2* tp = reinterpret_cast<const int2*>(ptab.data_ptr());
    const int pdl = g_sk2_pdl;
    if (CL > 1 || pdl > 0) {
      cudaLaunchConfig_t lc = {};
      lc.gridDim = dim3(N / R * CL);
      lc.blockDim = dim3((NW + 1 + NESC) * 32);
      lc.dynamicSmemBytes = smem;
      lc.stream = at::cuda::getCurrentCUDAStream();
      cudaLaunchAttribute at_[2];
      int na = 0;
      if (CL > 1) {
        at_[na].id = cudaLaunchAttributeClusterDimension;
        at_[na].val.clusterDim.x = CL;
        at_[na].val.clusterDim.y = 1;
        at_[na].val.clusterDim.z = 1;
        ++na;
      }
      if (pdl > 0) {   // programmatic dependent launch: may start while the previous kernel in the stream finishes
        at_[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        at_[na].val.programmaticStreamSerializationAllowed = 1;
        ++na;
      }
      lc.attrs = at_;
      lc.numAttrs = na;
      C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, xm, wsp, xp, op, tp, (int)pmax, (uint32_t)base4, M, N, K, (int)dg, dbg,
                                        wsr, outr, part, rcnt, pdl, rbias, gpre, rids, rlids, rwts, xmo, rls, part2, dcnt,
                                        b_alpha, b_beta, b_sin, b_gp, b_gn, b_sout, b_hout, b_part, b_cnt));
    } else {
      kern<<<N / R, (NW + 1 + NESC) * 32, smem, at::cuda::getCurrentCUDAStream()>>>(xm, wsp, xp, op, tp, (int)pmax,
                                                                               (uint32_t)base4, M, N, K, (int)dg, dbg,
                                                                               wsr, outr, part, rcnt, 0, rbias, gpre, rids,
                                                                               rlids, rwts, xmo, rls, part2, dcnt,
                                                                               b_alpha, b_beta, b_sin, b_gp, b_gn, b_sout,
                                                                               b_hout, b_part, b_cnt);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  };
  if (TOKR == 8 || M < TOKR) go(std::true_type{}); else go(std::false_type{});
}

void sk2f(at::Tensor Ws, at::Tensor Wsr, at::Tensor x, at::Tensor out, at::Tensor outr, at::Tensor ptab, int64_t pmax,
          int64_t base4, at::Tensor part, at::Tensor rcnt, int64_t M, int64_t tokr, at::Tensor rbias, at::Tensor gpre,
          at::Tensor rids, at::Tensor rlids, at::Tensor rwts, at::Tensor xm, at::Tensor rls, at::Tensor part2,
          at::Tensor dcnt, int64_t fuse_route, at::Tensor ba, at::Tensor bs, at::Tensor b1a, at::Tensor b1b,
          at::Tensor b1g, at::Tensor b1s, int64_t fuse_bnd1, at::Tensor b1p, at::Tensor b1c) {
  TORCH_CHECK(!fuse_bnd1 || M <= 8, "sk2f: the boundary-1 prologue handles M <= 8 rows");
  @@SK2F_DISPATCH@@
  TORCH_CHECK(false, "sk2f: no config");
}

// sk2 launch whose last CTA also computes a residual/norm boundary of the decode layer (see the BT tail): M <= 8 only
void sk2b(at::Tensor Ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4, int64_t M, int64_t N,
          int64_t K, int64_t cfg, int64_t tokr, at::Tensor alpha, at::Tensor beta, at::Tensor sin_, at::Tensor gp,
          at::Tensor gn, at::Tensor sout, at::Tensor hout, at::Tensor part, at::Tensor cnt, int64_t has_gp) {
  TORCH_CHECK(M >= 1 && M <= 8, "sk2b: M <= 8");
  @@SK2B_DISPATCH@@
  TORCH_CHECK(false, "sk2b: no config");
}

void sk2(at::Tensor Ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4,
         int64_t M, int64_t N, int64_t K, int64_t cfg, int64_t tokr, int64_t dg, at::Tensor dbgt) {
  unsigned long long* dbg = dbgt.numel() >= 8192 + 3 * 1024 ? reinterpret_cast<unsigned long long*>(dbgt.data_ptr()) : nullptr;
  @@SK2_DISPATCH@@
  TORCH_CHECK(false, "sk2: no config");
}
"""


bfp_WIDE_SRC = r"""
// ============================================================================ wide kernel (m 33..128)
// Weights: the per-row-block compressed stream [rb][unit][warp][1536 B] (unit = 16 rows x 64 k per warp),
// one bulk copy per stage; x: one TMA tile [TOK x 64] (128B swizzle) per stage.  Consumers decode the
// weights into registers and run RS-wgmma m64nTOKk16 (sm_90) / mma.sync on the swizzled tile (sm_80 check
// path).  Split-K over units (uneven splits allowed); escapes corrected from a per-CTA table.
template <int NWG, int TOK, int STAGES, bool HLW, typename OutT>
__global__ void __launch_bounds__(NWG * 128, 1)
wide_kernel(const __grid_constant__ CUtensorMap mapB, const uint8_t* __restrict__ Wst,
            const __nv_bfloat16* __restrict__ x, OutT* __restrict__ out, float* __restrict__ ws,
            int* __restrict__ counters, const int2* __restrict__ ptab, int pmax, uint32_t base4,
            int M, int N, int K, int splits, int dist) {
  // No separate producer warp (17 warps would cap registers at 96 per thread: 5 warps on one SM sub-partition):
  // thread 0 issues the stage copies.
  constexpr int NWARP = NWG * 4;
  constexpr int NT = NWG * 128;
  // per warp per 64-k unit: compressed 1536 B, or (HLW, router) BF16 hi tile 2 KB + lo tile 2 KB in fragment order
  constexpr int WB = HLW ? 4096 : 1536;
  constexpr int A_BYTES = NWARP * WB;
  constexpr int B_BYTES = TOK * 128;
  constexpr int ST_BYTES = A_BYTES + B_BYTES;
  constexpr int NACC = TOK / 2;
  static_assert(ST_BYTES % 1024 == 0, "stage alignment");
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  __shared__ int s_last;
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t full0 = sbase + STAGES * ST_BYTES, empty0 = full0 + 8 * STAGES;
  int2* ptab_s = reinterpret_cast<int2*>(gbase + STAGES * ST_BYTES + 16 * STAGES);
  __nv_bfloat16* xpatch = reinterpret_cast<__nv_bfloat16*>(ptab_s + pmax);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rb = blockIdx.x, sp = blockIdx.y, nrb = gridDim.x;
  const int zh = blockIdx.z;                    // token block: rows [zh*TOK, zh*TOK + Mz) of x / out
  const int t0z = zh * TOK;
  const int Mz = min(TOK, M - t0z);
  const int U = K >> 6;
  const int u0 = (int)(((long long)sp * U) / splits);
  const int u1 = (int)(((long long)(sp + 1) * U) / splits);
  const int nst = u1 - u0;
  const int cta = sp * nrb + rb;
  const uint8_t* wsrc = Wst + ((size_t)rb * U + u0) * A_BYTES;

  if (tid == 0) {
    for (int i = 0; i < STAGES; ++i) {
#if __CUDA_ARCH__ >= 900
      mbar_init(full0 + 8 * i, 1);
#else
      mbar_init(full0 + 8 * i, 32);
#endif
      mbar_init(empty0 + 8 * i, NWG);
    }
#if __CUDA_ARCH__ >= 900
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
    if (splits > 1) asm volatile("prefetch.global.L2 [%0];" :: "l"(counters + zh * nrb + rb));
  }

  // stage issue (sm_90: thread 0, TMA + one bulk copy; sm_80 check path: warp 0, cp.async)
  auto issue = [&](int i) {
    const int slot = i % STAGES;
    const uint32_t st = sbase + slot * ST_BYTES;
#if __CUDA_ARCH__ >= 900
    mbar_expect_tx(full0 + 8 * slot, ST_BYTES);
    tma_load_2d_el(st, &mapB, full0 + 8 * slot, (u0 + i) * 64, t0z);
    bulk_g2s(st + B_BYTES, wsrc + (size_t)i * A_BYTES, A_BYTES, full0 + 8 * slot);
#else
    for (int idx = lane; idx < TOK * 8; idx += 32) {
      const int t = idx >> 3, c = idx & 7;
      const uint32_t dst = st + t * 128 + ((c ^ (t & 7)) << 4);
      const __nv_bfloat16* src = x + (size_t)(t0z + (t < Mz ? t : 0)) * K + (u0 + i) * 64 + c * 8;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(t < Mz ? 16 : 0) : "memory");
    }
    const char* wa = reinterpret_cast<const char*>(wsrc) + (size_t)i * A_BYTES;
    for (int off = lane * 16; off < A_BYTES; off += 512)
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(st + B_BYTES + off), "l"(wa + off) : "memory");
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(full0 + 8 * slot) : "memory");
#endif
  };
#if __CUDA_ARCH__ >= 900
  const bool producer = tid == 0;
#else
  const bool producer = warp == 0;
#endif
#if __CUDA_ARCH__ >= 900
  if (producer) {   // thread 0: barriers are initialized (program order) -> issue before the CTA barrier
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapB) : "memory");
    for (int i = 0; i < STAGES && i < nst; ++i) issue(i);
  }
  __syncthreads();
#else
  __syncthreads();
  if (producer)
    for (int i = 0; i < STAGES && i < nst; ++i) issue(i);
#endif
  // escape table + the x values it needs -> shared memory (used after the main loop)
  {
    const int2* tab = ptab + (size_t)cta * pmax;
    const int cnt = pmax ? tab[0].x : 0;
    if (pmax) for (int p = tid; p <= cnt; p += NT) ptab_s[p] = tab[p];
    for (int idx = tid; idx < cnt * TOK; idx += NT) {
      const int p = idx / TOK + 1, t = idx - (p - 1) * TOK;
      const int kl = tab[p].x & 0xFFFF;
      xpatch[p * TOK + t] = t < Mz ? x[(size_t)(t0z + t) * K + u0 * 64 + kl] : __float2bfloat16(0.f);
    }
  }

  // ---------------- consumers
  const int g = lane >> 2, t4 = lane & 3;
  float acc[NACC];
#pragma unroll
  for (int q = 0; q < NACC; ++q) acc[q] = 0.f;
  uint32_t afrag[2][4];
  for (int i = 0; i < nst; ++i) {
    const int slot = i % STAGES;
    mbar_wait(full0 + 8 * slot, (i / STAGES) & 1);
    const uint32_t st = sbase + slot * ST_BYTES;
    const uint32_t wa = st + B_BYTES + warp * WB + lane * 16;
    if constexpr (HLW) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {   // h = 0: hi tile, 1: lo tile (same B) -> FP32 sum of both
          uint32_t* af = afrag[h];
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(af[0]), "=r"(af[1]), "=r"(af[2]), "=r"(af[3])
                       : "r"(wa + h * 2048 + j * 512));
#if __CUDA_ARCH__ >= 900
#pragma unroll
          for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
          wg_fence();
          wgmma_bn<TOK>(acc, af[0], af[1], af[2], af[3], make_desc_sw128(st + j * 32));
          wg_commit();
          wg_wait<1>();
#pragma unroll
          for (int q = 0; q < 4; ++q) asm volatile("" : "+r"(afrag[0][q]), "+r"(afrag[1][q]) :: "memory");
#else
          const unsigned char* tile = gbase + slot * ST_BYTES;
#pragma unroll
          for (int jj = 0; jj < TOK / 8; ++jj) {
            const int tok = jj * 8 + g;
            const unsigned char* row = tile + tok * 128;
            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(row + (((2 * j) ^ (tok & 7)) << 4) + 4 * t4);
            const uint32_t b1 = *reinterpret_cast<const uint32_t*>(row + (((2 * j + 1) ^ (tok & 7)) << 4) + 4 * t4);
            mma_bf16(acc + 4 * jj, af[0], af[1], af[2], af[3], b0, b1);
          }
#endif
        }
      }
    } else {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      // k-step j: low bytes = 8 B at (j/2)*512 + (j%2)*8, codes = 4 B at 1024 + 4j (per-k-step loads: low pressure)
      uint32_t lw0, lw1, cw;
      asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(lw0), "=r"(lw1) : "r"(wa + (j >> 1) * 512 + (j & 1) * 8));
      asm volatile("ld.shared.u32 %0, [%1];" : "=r"(cw) : "r"(wa + 1024 + 4 * j));
      uint32_t* af = afrag[j & 1];
      decode8(lw0, lw1, cw, base4, af);
#if __CUDA_ARCH__ >= 900
#pragma unroll
      for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
      wg_fence();
      wgmma_bn<TOK>(acc, af[0], af[1], af[2], af[3], make_desc_sw128(st + j * 32));
      wg_commit();
      wg_wait<1>();
#pragma unroll
      for (int q = 0; q < 4; ++q) asm volatile("" : "+r"(afrag[0][q]), "+r"(afrag[1][q]) :: "memory");
#else
      const unsigned char* tile = gbase + slot * ST_BYTES;
#pragma unroll
      for (int jj = 0; jj < TOK / 8; ++jj) {
        const int tok = jj * 8 + g;
        const unsigned char* row = tile + tok * 128;
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(row + (((2 * j) ^ (tok & 7)) << 4) + 4 * t4);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(row + (((2 * j + 1) ^ (tok & 7)) << 4) + 4 * t4);
        mma_bf16(acc + 4 * jj, af[0], af[1], af[2], af[3], b0, b1);
      }
#endif
    }
    }  // !HLW
#if __CUDA_ARCH__ >= 900
    // stage i-1's wgmmas are complete (one group in flight) -> its slot can be refilled
    if (i > 0) {
      if ((tid & 127) == 0) mbar_arrive(empty0 + 8 * ((i - 1) % STAGES));
      if (producer && i - 1 + STAGES < nst) {
        mbar_wait(empty0 + 8 * ((i - 1) % STAGES), ((i - 1) / STAGES) & 1);
        issue(i - 1 + STAGES);
      }
    }
#else
    named_bar(1, NT);   // check path: every warp is done with this slot -> refill it
    if (producer && i + STAGES < nst) issue(i + STAGES);
#endif
  }
#if __CUDA_ARCH__ >= 900
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
#endif

  // ---------------- exact corrections for escapes (table + x values staged at kernel start)
  named_bar(1, NT);
  {
    const int cnt = pmax ? ptab_s[0].x : 0;
    for (int p = 1; p <= cnt; ++p) {
      const int2 e = ptab_s[p];
      const int rl = e.x >> 16;
      if ((rl >> 4) != warp) continue;
      const int r16 = rl & 15;
      if (g != (r16 & 7)) continue;
      const float d = __int_as_float(e.y);
      const __nv_bfloat16* xv = xpatch + p * TOK;
      const bool hi = r16 >= 8;
#pragma unroll
      for (int jj = 0; jj < TOK / 8; ++jj) {
        const float v0 = d * __bfloat162float(xv[8 * jj + 2 * t4]);
        const float v1 = d * __bfloat162float(xv[8 * jj + 2 * t4 + 1]);
        if (hi) { acc[4 * jj + 2] += v0; acc[4 * jj + 3] += v1; }
        else { acc[4 * jj] += v0; acc[4 * jj + 1] += v1; }
      }
    }
  }

  const int row_w = rb * (NWARP * 16) + warp * 16;
  if (splits == 1) {
#pragma unroll
    for (int j = 0; j < TOK / 8; ++j) {
      const int t0 = 8 * j + 2 * t4;
      if (t0 < Mz) {
        out[(size_t)(t0z + t0) * N + row_w + g] = cvt_out<OutT>(acc[4 * j]);
        out[(size_t)(t0z + t0) * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 2]);
      }
      if (t0 + 1 < Mz) {
        out[(size_t)(t0z + t0 + 1) * N + row_w + g] = cvt_out<OutT>(acc[4 * j + 1]);
        out[(size_t)(t0z + t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 3]);
      }
    }
    return;
  }
  {
    float4* wsp = reinterpret_cast<float4*>(ws) + ((size_t)(zh * nrb + rb) * splits + sp) * (NACC / 4) * NT;
#pragma unroll
    for (int j = 0; j < NACC / 4; ++j)
      __stcg(wsp + j * NT + tid, make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]));
  }
  __threadfence();
  named_bar(1, NT);
  if (tid == 0) {
    const int old = atomicAdd(counters + zh * nrb + rb, 1);
    s_last = (old % splits) == splits - 1;
    if (dist) {
      const int target = (old / splits + 1) * splits;
      okc_spin_ge(counters + zh * nrb + rb, target, 32);   // bounded (POD.md Oct 26 17:50)
    }
  }
  named_bar(1, NT);
  if (!dist && !s_last) return;
  __threadfence();
  const int j0 = dist ? (int)(((long long)sp * (NACC / 4)) / splits) : 0;
  const int j1 = dist ? (int)(((long long)(sp + 1) * (NACC / 4)) / splits) : NACC / 4;
  const float4* base = reinterpret_cast<const float4*>(ws) + (size_t)(zh * nrb + rb) * splits * (NACC / 4) * NT;
  for (int j = j0; j < j1; ++j) {
    float4 s4 = make_float4(0.f, 0.f, 0.f, 0.f);
    int q = 0;
    for (; q + 4 <= splits; q += 4) {
      const float4 v0 = __ldcg(base + ((size_t)(q + 0) * (NACC / 4) + j) * NT + tid);
      const float4 v1 = __ldcg(base + ((size_t)(q + 1) * (NACC / 4) + j) * NT + tid);
      const float4 v2 = __ldcg(base + ((size_t)(q + 2) * (NACC / 4) + j) * NT + tid);
      const float4 v3 = __ldcg(base + ((size_t)(q + 3) * (NACC / 4) + j) * NT + tid);
      s4.x += v0.x + v1.x + v2.x + v3.x; s4.y += v0.y + v1.y + v2.y + v3.y;
      s4.z += v0.z + v1.z + v2.z + v3.z; s4.w += v0.w + v1.w + v2.w + v3.w;
    }
    for (; q < splits; ++q) {
      const float4 v = __ldcg(base + ((size_t)q * (NACC / 4) + j) * NT + tid);
      s4.x += v.x; s4.y += v.y; s4.z += v.z; s4.w += v.w;
    }
    const int t0 = 8 * j + 2 * t4;
    if (t0 < Mz) {
      out[(size_t)(t0z + t0) * N + row_w + g] = cvt_out<OutT>(s4.x);
      out[(size_t)(t0z + t0) * N + row_w + g + 8] = cvt_out<OutT>(s4.z);
    }
    if (t0 + 1 < Mz) {
      out[(size_t)(t0z + t0 + 1) * N + row_w + g] = cvt_out<OutT>(s4.y);
      out[(size_t)(t0z + t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(s4.w);
    }
  }
}

template <int NWG, int TOK, int STAGES, bool HLW, typename OutT>
static void launch_wide(const at::Tensor& Wst, const at::Tensor& x, at::Tensor& out, at::Tensor& ws,
                        at::Tensor& counters, const at::Tensor& ptab, int64_t pmax, int64_t base4,
                        int M, int N, int K, int splits) {
  constexpr int ST_BYTES = NWG * 4 * (HLW ? 4096 : 1536) + TOK * 128;
  const int nz = (M + TOK - 1) / TOK;   // token blocks (same weight stream: concurrent CTAs share it in L2)
  const size_t smem = 1024 + (size_t)STAGES * ST_BYTES + 16 * STAGES + (size_t)pmax * 8 + (size_t)pmax * TOK * 2 + 64;
  auto kern = wide_kernel<NWG, TOK, STAGES, HLW, OutT>;
  int dev = 0;
  cudaGetDevice(&dev);
  static bool attr_done = false;
  if (!attr_done) {
    int optin = 0;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);
    attr_done = true;
  }
  int major = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  CUtensorMap mB;
  memset(&mB, 0, sizeof(mB));
  if (major >= 9) mB = make_map_2d(x.data_ptr(), (uint64_t)M, (uint64_t)K, TOK);
  dim3 grid(N / (NWG * 64), splits, nz);
  int nsm = 0, per_sm = 0;
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, NWG * 128, smem);
  const int dist = (int)(grid.x * grid.y * grid.z) <= per_sm * nsm ? 1 : 0;
  kern<<<grid, NWG * 128, smem, at::cuda::getCurrentCUDAStream()>>>(
      mB, reinterpret_cast<const uint8_t*>(Wst.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<OutT*>(out.data_ptr()), ws.data_ptr<float>(), counters.data_ptr<int>(),
      reinterpret_cast<const int2*>(ptab.data_ptr()), (int)pmax, (uint32_t)base4, M, N, K, splits, dist);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void wide(at::Tensor Wst, at::Tensor x, at::Tensor out, at::Tensor ws, at::Tensor counters, at::Tensor ptab,
          int64_t pmax, int64_t base4, int64_t M, int64_t N, int64_t K, int64_t splits, int64_t nwg, int64_t tok) {
  const int hl = out.scalar_type() == at::kFloat ? 1 : 0;
  @@WIDE_DISPATCH@@
  TORCH_CHECK(false, "wide: no config");
}
"""


def bfp_wgmma_ss_defs(bns):
    out = []
    for bn in bns:
        nacc = bn // 2
        regs = ", ".join(f"%{i}" for i in range(nacc))
        cons = ", ".join(f'"+f"(d[{i}])' for i in range(nacc))
        a, b, p = nacc, nacc + 1, nacc + 2
        out.append(f"""
template <int BN_> __device__ __forceinline__ typename std::enable_if<BN_ == {bn}>::type
wgmma_ss(float* d, uint64_t da, uint64_t db) {{
  asm volatile("{{\\n"
      ".reg .pred p;\\n"
      "setp.ne.b32 p, %{p}, 0;\\n"
      "wgmma.mma_async.sync.aligned.m64n{bn}k16.f32.bf16.bf16 "
      "{{{regs}}}, %{a}, %{b}, p, 1, 1, 0, 0;\\n"
      "}}\\n"
      : {cons}
      : "l"(da), "l"(db), "r"(1));
}}""")
    return "\n".join(out)


bfp_CPP_SRC = """
void skinny_tma_a(torch::Tensor Wp, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor counters,
                torch::Tensor ptab, int64_t pmax, int64_t base4,
                int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t mt, int64_t rg, int64_t wsn,
                int64_t comp, int64_t nrg, int64_t ks, int64_t dg);
void gemm90(torch::Tensor W, torch::Tensor Wst, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor counters,
            int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t nwg, int64_t bn, int64_t as);
void wide(torch::Tensor Wst, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor counters, torch::Tensor ptab,
          int64_t pmax, int64_t base4, int64_t M, int64_t N, int64_t K, int64_t splits, int64_t nwg, int64_t tok);
void gq(torch::Tensor W, torch::Tensor x, torch::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
        int64_t cl, int64_t nstage);
void sk2(torch::Tensor Ws, torch::Tensor x, torch::Tensor out, torch::Tensor ptab, int64_t pmax, int64_t base4,
         int64_t M, int64_t N, int64_t K, int64_t cfg, int64_t tokr, int64_t dg, torch::Tensor dbgt);
void sk2f(torch::Tensor Ws, torch::Tensor Wsr, torch::Tensor x, torch::Tensor out, torch::Tensor outr, torch::Tensor ptab,
          int64_t pmax, int64_t base4, torch::Tensor part, torch::Tensor rcnt, int64_t M, int64_t tokr, torch::Tensor rbias,
          torch::Tensor gpre, torch::Tensor rids, torch::Tensor rlids, torch::Tensor rwts, torch::Tensor xm, torch::Tensor rls,
          torch::Tensor part2, torch::Tensor dcnt, int64_t fuse_route, torch::Tensor ba, torch::Tensor bs, torch::Tensor b1a,
          torch::Tensor b1b, torch::Tensor b1g, torch::Tensor b1s, int64_t fuse_bnd1, torch::Tensor b1p, torch::Tensor b1c);
void sk2b(torch::Tensor Ws, torch::Tensor x, torch::Tensor out, torch::Tensor ptab, int64_t pmax, int64_t base4, int64_t M,
          int64_t N, int64_t K, int64_t cfg, int64_t tokr, torch::Tensor alpha, torch::Tensor beta, torch::Tensor sin_,
          torch::Tensor gp, torch::Tensor gn, torch::Tensor sout, torch::Tensor hout, torch::Tensor part, torch::Tensor cnt,
          int64_t has_gp);
void sk2_set_pdl(int64_t n);
"""

# (layer, mt) -> (splits, RG, WS); grid = (N / (16 RG), splits) ~ 128 CTAs
bfp_CONFIGS = {
    ("pooler", 1): (1, 2, 8), ("pooler", 2): (2, 4, 4), ("pooler", 4): (4, 8, 2), ("pooler", 8): (16, 8, 1),
    ("unpooler", 1): (2, 1, 8), ("unpooler", 2): (1, 4, 4), ("unpooler", 4): (2, 8, 2), ("unpooler", 8): (8, 8, 1),
    ("qkv_gate", 1): (8, 9, 2), ("qkv_gate", 2): (8, 9, 2), ("qkv_gate", 4): (8, 9, 1), ("qkv_gate", 8): (16, 9, 1),
    ("o", 1): (1, 4, 4), ("o", 2): (1, 4, 4), ("o", 4): (1, 4, 4), ("o", 8): (2, 8, 1),
    ("router", 1): (8, 1, 8), ("router", 2): (8, 1, 8), ("router", 4): (8, 1, 8), ("router", 8): (16, 1, 4),
}
bfp_MAX_SKINNY_M = 32
bfp_w_rows = {"qkv_gate": 2304, "o": 8192, "pooler": 4096, "unpooler": 8192, "router": 256}
bfp_COMPRESS = {"qkv_gate", "o", "pooler", "unpooler"}
bfp_D_PLAIN = 16
bfp_D_COMP = 12
bfp_PF = 0

bfp__ext = None


def bfp__dispatch_src():
    return ""  # LDG skinny kernel unused in v6 (TMA path); keep the template uninstantiated
    lines = []
    for (layer, mt), (splits, rg, wsn) in sorted(bfp_CONFIGS.items()):
        f32 = layer == "router"
        ot = "float" if f32 else "__nv_bfloat16"
        for comp in ((0,) if f32 else (0, 1)):
            d = bfp_D_COMP if comp else (bfp_D_PLAIN if rg * wsn <= 9 and mt <= 4 else 8)
            lines.append(f"if (mt == {mt} && rg == {rg} && wsn == {wsn} && comp == {comp} && out_f32 == {str(f32).lower()}) "
                         f"{{ launch<{mt}, {rg}, {wsn}, {d}, {str(bool(comp)).lower()}, {bfp_PF}, {ot}>(Wp, x, out, ws, counters, "
                         f"poff, pk, pr, pd, base4, M, N, Kp, Kx, splits, nrg); return; }}")
    return "\n  ".join(dict.fromkeys(lines))


bfp_MID_CONFIGS = {  # (layer, bn) -> (nwg, splits); grid = (N / (64 nwg), splits) <= 132 CTAs (co-resident)
    ("pooler", 128): (2, 4), ("unpooler", 128): (2, 2), ("qkv_gate", 128): (2, 7), ("o", 128): (2, 2),
    ("router", 128): (2, 64),
    ("pooler", 256): (2, 4), ("unpooler", 256): (2, 2), ("qkv_gate", 256): (2, 7), ("o", 256): (2, 2),
    ("router", 256): (2, 64),
}
bfp_TMA_CONFIGS = {  # (layer, mt) -> (splits, RG, WS, KS); x chunk M*Kc*2 small, ~128 CTAs; mt 0 = m == 1
    ("pooler", 0): (1, 2, 8, 4), ("pooler", 1): (2, 4, 4, 4), ("pooler", 2): (4, 8, 2, 4), ("pooler", 4): (8, 16, 1, 4),
    ("unpooler", 0): (2, 1, 8, 4), ("unpooler", 1): (2, 1, 8, 4), ("unpooler", 2): (2, 8, 2, 4), ("unpooler", 4): (4, 16, 1, 4),
    ("qkv_gate", 0): (8, 9, 2, 4), ("qkv_gate", 1): (8, 9, 2, 4), ("qkv_gate", 2): (8, 9, 2, 4), ("qkv_gate", 4): (8, 9, 1, 8),
    ("o", 0): (1, 4, 4, 4), ("o", 1): (1, 4, 4, 4), ("o", 2): (1, 4, 4, 4), ("o", 4): (1, 4, 4, 4),
    ("router", 0): (8, 1, 8, 4), ("router", 1): (8, 1, 8, 4), ("router", 2): (8, 1, 8, 4), ("router", 4): (8, 1, 8, 4),
}
bfp_TMA_KS = 4
bfp_USE_TMA = True
bfp_USE_WGM = True   # m >= 9 compressed configs with RG % 4 == 0 use RS-wgmma (sm_90)
bfp_WGM_SM80_TEST = False  # local check of the wgmma path's data layout (mma.sync emulation on sm_80)


def bfp__tma_nstage(nw, comp, sm90, ks, hl=False):
    chunk = (ks // 4) * 1536 if comp else ks * 512 * (2 if hl else 1)
    budget = (112 if sm90 else 40) * 1024
    return max(2, min(8, budget // (nw * chunk)))


bfp_TMA_TU = {}  # (layer, mtk) -> 0 / 1 (which translation unit instantiates it)


# alternative skinny configs (more splits -> fewer x bytes per CTA); the prepare()-time autotuner times them
# against the default config and cuBLAS on every scored shape
bfp_TMA_ALT = {("pooler", 8): (8, 16, 1, 4), ("pooler", 16): (8, 16, 1, 4), ("unpooler", 1): (8, 4, 1, 4),
           ("unpooler", 8): (8, 4, 1, 4), ("unpooler", 16): (4, 16, 1, 4), ("o", 16): (4, 16, 1, 4),
           ("o", 32): (4, 16, 1, 4)}


def bfp__tma_cfg_items():
    items = [((layer, mtk), cfg) for (layer, mtk), cfg in sorted(bfp_TMA_CONFIGS.items())]
    items += [(("alt", layer, m), cfg) for (layer, m), cfg in sorted(bfp_TMA_ALT.items())]
    return items


def bfp__tma_srcs(sm90):
    lines = {}
    for key, (splits, rg, wsn, ks) in bfp__tma_cfg_items():
        if key[0] == "alt":
            layer, mtk = key[1], bfp__mt_key(key[2])
        else:
            layer, mtk = key
        mt = max(mtk, 1)
        f32 = layer == "router"
        ot = "float" if f32 else "__nv_bfloat16"
        comp = 0 if f32 else 1
        hl = f32
        ns = bfp__tma_nstage(rg * wsn, comp, sm90, ks, hl)
        if f32 and mt >= 4:
            ns = min(ns, 3)  # x chunk (32 x 1024) + ring must fit in shared memory
        n_cfg = 1024 if (layer == "unpooler" and mtk <= 1) else bfp_w_rows[layer]
        wgm = bool((sm90 or bfp_WGM_SM80_TEST) and comp and bfp_USE_WGM and rg % 4 == 0 and mtk >= 2)
        line = (f"if (mt == {mt} && rg == {rg} && wsn == {wsn} && ks == {ks} && comp == {comp} && out_f32 == {str(f32).lower()}) "
                f"{{ launch_tma<{mt}, {rg}, {wsn}, {ks}, {ns}, {str(bool(comp)).lower()}, {str(hl).lower()}, {str(wgm).lower()}, {ot}>(Wp, x, out, ws, "
                f"counters, ptab, pmax, base4, M, N, Kp, Kx, splits, nrg, (int)dg); return; }}")
        lines.setdefault(line, []).append(key)
    keys = list(lines)
    for ln in keys:
        for cfg in lines[ln]:
            bfp_TMA_TU[cfg] = 0
    return bfp_TMA_SRC.replace("@@TMA_DISPATCH@@", "\n  ".join(keys)).replace("@@TMA_NAME@@", "skinny_tma_a")


bfp_GEMM_CONFIGS = {  # (layer, bn) -> (nwg, splits, stages); grid = (N / (64 nwg), splits) <= 132 (co-resident)
    ("pooler", 64): (2, 4, 8), ("unpooler", 64): (2, 2, 8), ("qkv_gate", 64): (2, 7, 8), ("o", 64): (2, 2, 8),
    ("router", 64): (2, 32, 8),
    ("pooler", 128): (2, 4, 6), ("unpooler", 128): (2, 2, 6), ("qkv_gate", 128): (2, 7, 6), ("o", 128): (2, 2, 6),
    ("router", 128): (2, 32, 6),
    ("pooler", 256): (2, 4, 4), ("unpooler", 256): (2, 2, 4), ("qkv_gate", 256): (2, 7, 4), ("o", 256): (2, 2, 4),
    ("router", 256): (2, 32, 4),
}


bfp_DIAG_LAYERS = ()
bfp_GEMM_DIAG = False   # diagnostic: also time gemm90 variants (TMA-2D A vs contiguous pre-swizzled A) on m 65..256


def bfp_swizzle_tiles(w, bm):
    """W [N, K] -> [N/bm][K/64][bm rows x 128 B] with the 128B swizzle (16-byte chunk c of row r at c ^ (r % 8))."""
    n, k = w.shape
    t = w.view(n // bm, bm, k // 64, 8, 8).permute(0, 2, 1, 3, 4)  # [rb, kb, r, c, 8]
    r = torch.arange(bm, device=w.device).view(bm, 1)
    p = torch.arange(8, device=w.device).view(1, 8)
    src_c = (p ^ (r % 8)).view(1, 1, bm, 8, 1).expand(t.shape[0], t.shape[1], bm, 8, 8)
    return torch.gather(t, 3, src_c).contiguous()


def bfp__gemm_src(sm90):
    if not sm90:
        return ('void gemm90(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, '
                'int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t) { TORCH_CHECK(false, "gemm90 needs sm_90"); }')
    lines = []
    for (layer, bn), (nwg, splits, stages) in sorted(bfp_GEMM_CONFIGS.items()):
        f32 = layer == "router"
        ot = "float" if f32 else "__nv_bfloat16"
        for a_s in ((0, 1) if (not f32 and bfp_GEMM_DIAG) else (0,)):
            lines.append(f"if (nwg == {nwg} && bn == {bn} && out_f32 == {str(f32).lower()} && as == {a_s}) "
                         f"{{ launch_gemm90<{nwg}, {bn}, {stages}, {str(bool(a_s)).lower()}, {ot}>(W, Wst, x, out, ws, counters, M, N, Kp, Kx, splits); return; }}")
    bns = sorted({bn for (_, bn) in bfp_GEMM_CONFIGS})
    return (bfp_GEMM_SRC.replace("@@WGMMA_SS_DEFS@@", bfp_wgmma_ss_defs(bns))
            .replace("@@GEMM_DISPATCH@@", "\n  ".join(dict.fromkeys(lines))))


bfp_GQ_SRC = r"""
// ============================================================================ gq (m 33..256): BF16 weights, SS-wgmma
// One CTA = BM = NWG*64 weight rows x BN tokens: W tile [BM x 64 k] and x tile [BN x 64 k] arrive by TMA (128B
// swizzle) into a ring fed by a producer warp; warpgroup w multiplies rows [64w, 64w+64) of every stage (all share
// the x tile, so x is read from L2 once per BM rows).  CL > 1: a cluster of CL CTAs splits K; ranks > 0 hand their
// accumulators to rank 0 by st.async (completing rank 0's barrier transaction count); rank 0 adds and stores BF16.
template <int BN, int NWG, int NSTAGE, int CL>
__global__ void __launch_bounds__(NWG * 128 + 32, 1)
gq_kernel(const __grid_constant__ CUtensorMap mapW, const __grid_constant__ CUtensorMap mapX,
          __nv_bfloat16* __restrict__ out, int M, int N, int K) {
#if __CUDA_ARCH__ >= 900
  constexpr int BM = NWG * 64;
  constexpr int A_BYTES = BM * 128;
  constexpr int B_BYTES = BN * 128;
  constexpr int ST = A_BYTES + B_BYTES;
  constexpr int NACC = BN / 2;
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t bars = sbase + NSTAGE * ST;
  const uint32_t full0 = bars, empty0 = bars + 8 * NSTAGE, qbar = bars + 16 * NSTAGE;
  float* peer = reinterpret_cast<float*>(gbase + NSTAGE * ST + 16 * NSTAGE + 16);   // [CL-1][NWG][NACC][128]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rb = blockIdx.x / CL, crank = CL > 1 ? (int)(blockIdx.x % CL) : 0;
  const int U = K >> 6;
  const int u0 = (crank * U) / CL, u1 = ((crank + 1) * U) / CL;
  const int nkb = u1 - u0;
  const int xrows = min(M, BN);   // x rows actually loaded (rows >= M of the tile only feed outputs that are dropped)

  if (tid == 0) {
    for (int i = 0; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NWG); }
    mbar_init(qbar, 1);
    if (CL == 2) mbar_expect_tx(qbar, NWG * 64 * NACC * 4);   // split epilogue: the peer's half of our rows
    else if (CL > 1 && crank == 0) mbar_expect_tx(qbar, (CL - 1) * NWG * NACC * 128 * 4);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    // the first ring of stages goes out right away (no cluster barrier in front of it: a cluster-scope release
    // issued after a copy is in flight waits for it, and the stages only need this CTA's barriers)
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapW) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapX) : "memory");
    for (int i = 0; i < NSTAGE && i < nkb; ++i) {
      mbar_expect_tx(full0 + 8 * i, A_BYTES + xrows * 128);
      const int kg = (u0 + i) * 64;
      tma_load_2d_ef(sbase + i * ST, &mapW, full0 + 8 * i, kg, rb * BM);
      tma_load_2d_el(sbase + i * ST + A_BYTES, &mapX, full0 + 8 * i, kg, 0);
    }
  }
  __syncthreads();
  // cluster barrier: arrived here (relaxed: the barrier inits are fenced above), waited on only before remote stores
  if constexpr (CL > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

  if (warp == NWG * 4) {
    if (lane == 0) {
      for (int i = NSTAGE; i < nkb; ++i) {
        const int slot = i % NSTAGE;
        if (i >= NSTAGE) mbar_wait_sleep(empty0 + 8 * slot, ((i / NSTAGE) - 1) & 1);
        mbar_expect_tx(full0 + 8 * slot, A_BYTES + xrows * 128);
        const int kg = (u0 + i) * 64;
        tma_load_2d_ef(sbase + slot * ST, &mapW, full0 + 8 * slot, kg, rb * BM);
        tma_load_2d_el(sbase + slot * ST + A_BYTES, &mapX, full0 + 8 * slot, kg, 0);
      }
    }
    __syncwarp();
    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }

  const int wg = warp >> 2, t = tid & 127;
  float acc[NACC];
#pragma unroll
  for (int q = 0; q < NACC; ++q) acc[q] = 0.f;
  for (int i = 0; i < nkb; ++i) {
    const int slot = i % NSTAGE;
    mbar_wait_sleep(full0 + 8 * slot, (i / NSTAGE) & 1);
    const uint32_t a_st = sbase + slot * ST + wg * (64 * 128);
    const uint32_t b_st = sbase + slot * ST + A_BYTES;
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j) wgmma_ss<BN>(acc, make_desc_sw128(a_st + j * 32), make_desc_sw128(b_st + j * 32));
    wg_commit();
    wg_wait<1>();   // the previous stage's group is done: hand its slot back
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    if (i > 0 && t == 0) mbar_arrive(empty0 + 8 * ((i - 1) % NSTAGE));
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");

  if constexpr (CL <= 2) {
    // CL = 2: rank r finishes rows half r of every warpgroup (warps 2r, 2r+1); the other two warps hand their sums
    // to the peer by st.async (into [wg][warp][lane][NACC + 4]). The kept rows go out through a shared-memory
    // staging tile [wg][token][rows] so that each token's rows are written as contiguous 16-byte chunks.
    constexpr int RS = NACC + 4;
    constexpr int KR = CL == 2 ? 32 : 64;          // kept rows per warpgroup
    const int q = warp & 3;
    const int keep_lo = CL == 2 ? 2 * crank : 0;   // first kept warp of the warpgroup
    const bool keep = CL == 1 || (q >> 1) == crank;
    if (CL == 2 && !keep) {
      const int ql = q & 1;
      const uint32_t rl = (uint32_t)__cvta_generic_to_shared(peer + ((wg * 2 + ql) * 32 + lane) * RS);
      uint32_t rr, qb;
      asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");   // the peer's qbar is initialized
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rr) : "r"(rl), "r"(crank ^ 1));
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(qb) : "r"(qbar), "r"(crank ^ 1));
#pragma unroll
      for (int i = 0; i < NACC; i += 4)
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                     :: "r"(rr + i * 4), "r"(__float_as_uint(acc[i])), "r"(__float_as_uint(acc[i + 1])),
                        "r"(__float_as_uint(acc[i + 2])), "r"(__float_as_uint(acc[i + 3])), "r"(qb) : "memory");
      return;
    }
    if constexpr (CL == 2) {
      asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
      mbar_wait_cluster(qbar, 0);
      const float* rv = peer + ((wg * 2 + (q & 1)) * 32 + lane) * RS;
#pragma unroll
      for (int i = 0; i < NACC; i += 4) {
        const float4 p4 = *reinterpret_cast<const float4*>(rv + i);
        acc[i] += p4.x; acc[i + 1] += p4.y; acc[i + 2] += p4.z; acc[i + 3] += p4.w;
      }
    }
    // stage BF16 [wg][token][KR rows] in the idle ring (every warpgroup's ring reads are over: wg_wait<0> above;
    // the TMA writes all landed before the last full-barrier wait)
    if constexpr (NWG > 1) named_bar(1, NWG * (CL == 2 ? 64 : 128));
    __nv_bfloat16* stg = reinterpret_cast<__nv_bfloat16*>(gbase);
    {
      const int g = lane >> 2, t4 = lane & 3;
      const int rk = (q - keep_lo) * 16;   // this warp's first kept row
#pragma unroll
      for (int j = 0; j < BN / 8; ++j) {
        const int t0 = 8 * j + 2 * t4;
        __nv_bfloat16* s0 = stg + ((size_t)wg * BN + t0) * KR + rk + g;
        s0[0] = __float2bfloat16_rn(acc[4 * j]);
        s0[8] = __float2bfloat16_rn(acc[4 * j + 2]);
        s0[KR] = __float2bfloat16_rn(acc[4 * j + 1]);
        s0[KR + 8] = __float2bfloat16_rn(acc[4 * j + 3]);
      }
    }
    named_bar(2, NWG * (CL == 2 ? 64 : 128));
    {
      constexpr int CH = KR / 8;               // 16-byte chunks per token row
      const int nkt = NWG * (CL == 2 ? 64 : 128);
      const int kt = (q - keep_lo) * 32 + lane + wg * (CL == 2 ? 64 : 128);
      for (int idx = kt; idx < NWG * BN * CH; idx += nkt) {
        const int c = idx % CH, tt = (idx / CH) % BN, w2 = idx / (CH * BN);
        if (tt < M) {
          const uint4 v = *reinterpret_cast<const uint4*>(stg + ((size_t)w2 * BN + tt) * KR + c * 8);
          *reinterpret_cast<uint4*>(out + (size_t)tt * N + rb * BM + w2 * 64 + (CL == 2 ? crank * 32 : 0) + c * 8) = v;
        }
      }
    }
    return;
  }
  if (CL > 1 && crank > 0) {
    // stage the accumulators in this CTA's (now idle) ring, then one bulk shared::cluster copy into rank 0's buffer
    float* mine = reinterpret_cast<float*>(gbase);   // [NWG][NACC][128]
    if constexpr (NWG > 1) named_bar(1, NWG * 128);   // every warpgroup is done reading the ring
#pragma unroll
    for (int q = 0; q < NACC; ++q) mine[(wg * NACC + q) * 128 + t] = acc[q];
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    named_bar(1, NWG * 128);
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");   // rank 0's qbar is initialized
    if (tid == 0) {
      const uint32_t peer_l = (uint32_t)__cvta_generic_to_shared(peer + (crank - 1) * NWG * NACC * 128);
      uint32_t peer_r, qbar_r;
      asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(peer_r) : "r"(peer_l));
      asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(qbar_r) : "r"(qbar));
      asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                   :: "r"(peer_r), "r"((uint32_t)__cvta_generic_to_shared(mine)), "r"(NWG * NACC * 128 * 4), "r"(qbar_r)
                   : "memory");
      asm volatile("cp.async.bulk.commit_group;\n\tcp.async.bulk.wait_group.read 0;" ::: "memory");
    }
    return;
  }
  if constexpr (CL > 1) {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    mbar_wait_cluster(qbar, 0);
#pragma unroll
    for (int r = 0; r < CL - 1; ++r)
#pragma unroll
      for (int q = 0; q < NACC; ++q) acc[q] += peer[((r * NWG + wg) * NACC + q) * 128 + t];
  }
  const int g = lane >> 2, t4 = lane & 3;
  const int row_w = rb * BM + wg * 64 + (warp & 3) * 16;
#pragma unroll
  for (int j = 0; j < BN / 8; ++j) {
    const int t0 = 8 * j + 2 * t4;
    if (t0 < M) {
      out[(size_t)t0 * N + row_w + g] = __float2bfloat16_rn(acc[4 * j]);
      out[(size_t)t0 * N + row_w + g + 8] = __float2bfloat16_rn(acc[4 * j + 2]);
    }
    if (t0 + 1 < M) {
      out[(size_t)(t0 + 1) * N + row_w + g] = __float2bfloat16_rn(acc[4 * j + 1]);
      out[(size_t)(t0 + 1) * N + row_w + g + 8] = __float2bfloat16_rn(acc[4 * j + 3]);
    }
  }
#endif
}

template <int BN, int NWG, int NSTAGE, int CL>
static void launch_gq(const at::Tensor& W, const at::Tensor& x, at::Tensor& out, int M, int N, int K) {
  constexpr int BM = NWG * 64;
  constexpr int ST = BM * 128 + BN * 128;
  const size_t smem = 1024 + (size_t)NSTAGE * ST + 16 * NSTAGE + 16 +
                      (CL == 2 ? (size_t)NWG * 64 * (BN / 2 + 4) * 4 : (size_t)(CL - 1) * NWG * (BN / 2) * 128 * 4);
  auto kern = gq_kernel<BN, NWG, NSTAGE, CL>;
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr_done = true;
  }
  CUtensorMap mW = make_map_2d(W.data_ptr(), (uint64_t)W.size(0), (uint64_t)W.size(1), BM);
  CUtensorMap mX = make_map_2d(x.data_ptr(), (uint64_t)M, (uint64_t)K, (uint32_t)(M < BN ? M : BN));
  __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  if constexpr (CL > 1) {
    cudaLaunchConfig_t lc = {};
    lc.gridDim = dim3(N / BM * CL);
    lc.blockDim = dim3(NWG * 128 + 32);
    lc.dynamicSmemBytes = smem;
    lc.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at_[1];
    at_[0].id = cudaLaunchAttributeClusterDimension;
    at_[0].val.clusterDim.x = CL;
    at_[0].val.clusterDim.y = 1;
    at_[0].val.clusterDim.z = 1;
    lc.attrs = at_;
    lc.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, mW, mX, op, M, N, K));
  } else {
    kern<<<N / BM, NWG * 128 + 32, smem, at::cuda::getCurrentCUDAStream()>>>(mW, mX, op, M, N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void gq(at::Tensor W, at::Tensor x, at::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
        int64_t cl, int64_t nstage) {
  @@GQ_DISPATCH@@
  TORCH_CHECK(false, "gq: no config");
}
"""

bfp_GQ_CONFIGS = {  # (layer, bn) -> (nwg, cl, nstage): BM = 64*nwg rows per CTA, grid = N/BM * cl CTAs (cl-CTA clusters split K)
    # measured on H200 against cuBLAS (graph replay, cold L2): only the pooler shape wins (64 row blocks x 2 K halves)
    ("pooler", 64): (1, 2, 10), ("pooler", 128): (1, 2, 7),
}
bfp_USE_GQ = True


def bfp__gq_src(sm90):
    if not sm90:
        return ('void gq(at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, '
                'int64_t) { TORCH_CHECK(false, "gq needs sm_90"); }')
    lines = []
    for (layer, bn), (nwg, cl, ns) in sorted(bfp_GQ_CONFIGS.items()):
        lines.append(f"if (bn == {bn} && nwg == {nwg} && cl == {cl} && nstage == {ns}) "
                     f"{{ launch_gq<{bn}, {nwg}, {ns}, {cl}>(W, x, out, M, N, K); return; }}")
    return bfp_GQ_SRC.replace("@@GQ_DISPATCH@@", "\n  ".join(dict.fromkeys(lines)))


def bfp__gq_bn(m):
    return 64 if m <= 64 else 128 if m <= 128 else 256


def bfp__run_gq(x, layer, m, n, k, state):
    nwg, cl, ns = bfp_GQ_CONFIGS[(layer, bfp__gq_bn(m))]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    bfp__get_ext().gq(state["raw"][layer], x, out, m, n, k, bfp__gq_bn(m), nwg, cl, ns)
    return out



bfp_WIDE_CONFIGS = {  # (layer, tok, nz) -> (nwg, splits, stages); grid = (N / (64 nwg), splits, nz) <= 132 (co-resident)
    ("pooler", 64, 1): (1, 2, 8), ("unpooler", 64, 1): (1, 1, 8), ("qkv_gate", 64, 1): (1, 3, 8), ("o", 64, 1): (1, 1, 8),
    ("pooler", 128, 1): (1, 2, 7), ("unpooler", 128, 1): (1, 1, 7), ("qkv_gate", 128, 1): (1, 3, 7), ("o", 128, 1): (1, 1, 7),
    ("pooler", 128, 2): (2, 1, 5), ("unpooler", 128, 2): (4, 1, 4), ("qkv_gate", 128, 2): (1, 1, 7), ("o", 128, 2): (4, 1, 4),
    # router: BF16 hi|lo tiles (plain), 64-row blocks
    ("router", 64, 1): (1, 32, 4), ("router", 128, 1): (1, 32, 4), ("router", 128, 2): (1, 16, 4),
}


def bfp__wide_key(m):
    return (64, 1) if m <= 64 else (128, 1) if m <= 128 else (128, 2)


def bfp__wide_src(sm90):
    lines = []
    for (layer, tok, nz), (nwg, splits, stages) in sorted(bfp_WIDE_CONFIGS.items()):
        hl = layer == "router"
        if not sm90:
            stages = min(stages, 3 if not hl else 2)  # sm_80 check path: 163 KB of shared memory
        ot = "float" if hl else "__nv_bfloat16"
        lines.append(f"if (nwg == {nwg} && tok == {tok} && hl == {int(hl)}) "
                     f"{{ launch_wide<{nwg}, {tok}, {stages}, {str(hl).lower()}, {ot}>(Wst, x, out, ws, counters, ptab, pmax, base4, "
                     f"M, N, K, splits); return; }}")
    return bfp_WIDE_SRC.replace("@@WIDE_DISPATCH@@", "\n  ".join(dict.fromkeys(lines)))


bfp_MID_STAGES = 4
bfp_MID_DU = 2


def bfp__mid_src(sm90):
    if not sm90:
        return ('void mid(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, '
                'torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, '
                'int64_t, int64_t, int64_t, int64_t) { TORCH_CHECK(false, "mid needs sm_90"); }')
    lines = []
    for (layer, bn), (nwg, splits) in []:
        f32 = layer == "router"
        ot = "float" if f32 else "__nv_bfloat16"
        for comp in ((0,) if f32 else (0, 1)):
            lines.append(f"if (nwg == {nwg} && bn == {bn} && comp == {comp} && out_f32 == {str(f32).lower()}) "
                         f"{{ launch_mid<{nwg}, {bn}, {bfp_MID_STAGES}, {bfp_MID_DU}, {str(bool(comp)).lower()}, {ot}>(Wp, x, out, ws, "
                         f"counters, poff, pk, pr, pd, base4, M, N, Kp, Kx, splits, nrg); return; }}")
    bns = sorted({bn for (_, bn) in bfp_MID_CONFIGS})
    return (MID_SRC.replace("@@WGMMA_DEFS@@", bfp_wgmma_defs(bns))
            .replace("@@MID_DISPATCH@@", "\n  ".join(dict.fromkeys(lines))))


def bfp__get_ext():
    """Compile everything as one CUDA translation unit with load_inline.  no_implicit_headers: the CUDA file includes
    only the ATen headers it needs (torch/types.h alone costs ~45 s of nvcc time)."""
    global bfp__ext
    if bfp__ext is None:
        major, minor = torch.cuda.get_device_capability()
        sm90 = major >= 9
        arch = f"{major}{minor}a" if sm90 else f"{major}{minor}"
        common = bfp_CUDA_SRC.replace("@@MBAR@@", bfp_MBAR_HELPERS).replace("@@BODY@@", bfp_WG_HELPERS if sm90 else "")
        defs = bfp_wgmma_defs([8, 16, 32, 64, 128]) if sm90 else ""
        cuda_src = (common + defs + bfp__tma_srcs(sm90) + bfp_MAP_SRC + bfp__gemm_src(sm90) + bfp__wide_src(sm90) + bfp__sk2_src(sm90)
                    + bfp__gq_src(sm90))
        bfp__ext = load_inline(
            name=f"dl_okc_bf16proj_v86y4_{arch}" + (f"_dly{bfp__SK2_DELAY}" if bfp__SK2_DELAY else ""),
            cpp_sources="#include <torch/extension.h>\n" + bfp_CPP_SRC,
            cuda_sources=cuda_src,
            functions=["skinny_tma_a", "gemm90", "wide", "sk2", "gq", "sk2f", "sk2b", "sk2_set_pdl"],
            extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"] + ([f"-DSK2_DELAY_ESC={bfp__SK2_DELAY}"] if bfp__SK2_DELAY else []),
            no_implicit_headers=True,
            verbose=False,
        )
    return bfp__ext


def bfp_pack_fragments(w: torch.Tensor) -> torch.Tensor:
    """W [N, K] bf16 -> [N/16, K/16, 32 lanes, 8] in mma.m16n8k16 A-fragment order."""
    n, k = w.shape
    t = w.view(n // 16, 2, 8, k // 16, 2, 4, 2)  # [rg, hr, g, s, hk, t, e]: row = hr*8+g, col = hk*8+2t+e
    t = t.permute(0, 3, 2, 5, 4, 1, 6)  # [rg, s, g, t, hk, hr, e]
    return t.contiguous().view(n // 16, k // 16, 32, 8)


def bfp_pack_compressed(w: torch.Tensor):
    """Lossless 12-bit packing of BF16 W [N, K].  Returns None if the matrix doesn't suit it."""
    n, k = w.shape
    if n % 16 or k % 64:
        return None
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hi = bits >> 8                  # sign | exponent bits 7..1
    e71 = hi & 0x7F
    hist = torch.bincount(e71.flatten(), minlength=128)
    win = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(win.argmax())
    o = (e71 - base).clamp(0, 7)
    code = ((hi >> 7) << 3) | o
    dec_hi = ((hi >> 7) << 7) | (base + o)
    dec = ((dec_hi << 8) | (bits & 0xFF)).to(torch.int16).view(torch.bfloat16)
    esc = (e71 < base) | (e71 > base + 7)
    n_esc = int(esc.sum())
    if n_esc > n * k // 200:
        return None
    rows, cols = esc.nonzero(as_tuple=True)
    wtrue = w[rows, cols].float()
    wdec = dec[rows, cols].float()
    delta = wtrue - wdec
    if not bool(((wdec + delta) == wtrue).all()):
        return None
    # fragment order for low bytes and codes: [rg, s, lane, 8]
    lo_f = bfp_pack_fragments((bits & 0xFF).to(torch.int16).view(torch.bfloat16)).view(torch.int16) & 0xFF
    cd_f = bfp_pack_fragments(code.to(torch.int16).view(torch.bfloat16)).view(torch.int16) & 0xF
    nrg, S = n // 16, k // 16
    lo_f = lo_f.view(nrg, S // 4, 4, 32, 8).to(torch.uint8)   # [rg, u, ks, lane, 8]
    cd_f = cd_f.view(nrg, S // 4, 4, 32, 8).to(torch.uint8)
    # parts 0,1: lane's low bytes for k-steps (0,1) and (2,3) -> [rg, u, 2, lane, 16]
    lo_p = lo_f.view(nrg, S // 4, 2, 2, 32, 8).permute(0, 1, 2, 4, 3, 5).reshape(nrg, S // 4, 2, 32, 16)
    # part 2: word i (k-step i): byte b = code[8i + b] | code[8i + 4 + b] << 4
    cw = (cd_f[..., 0:4] | (cd_f[..., 4:8] << 4))           # [rg, u, ks, lane, 4]
    cw = cw.permute(0, 1, 3, 2, 4).reshape(nrg, S // 4, 1, 32, 16)
    packed = torch.cat([lo_p, cw], dim=2).transpose(0, 1).contiguous()  # [u, rg, 3, lane, 16]
    order = torch.argsort(rows * k + cols)
    rows, cols, delta = rows[order], cols[order], delta[order]
    rg_idx = rows // 16
    poff = torch.zeros(nrg + 1, dtype=torch.int32, device=w.device)
    poff[1:] = torch.cumsum(torch.bincount(rg_idx, minlength=nrg), 0).to(torch.int32)
    return {
        "packed": packed,
        "base4": base * 0x01010101,
        "poff": poff,
        "pk": cols.to(torch.int32).contiguous(),
        "pr": (rows % 16).to(torch.int32).contiguous(),
        "pd": delta.to(torch.float32).contiguous(),
        "n_esc": n_esc,
    }


def bfp_build_patch_table(c, n_cfg, k, splits, rg):
    """Per-CTA exact-correction table for a (splits, RG) decomposition over the first n_cfg rows:
    int2 [num_ctas][pmax]; entry 0 = (count, 0); entries = ((row_local << 16) | k_local, delta bits)."""
    dev = c["pk"].device
    rows_cta = rg * 16
    nrb = n_cfg // rows_cta
    nunits = k // 64
    poff = c["poff"].long()
    nrg_all = poff.numel() - 1
    rgi = torch.repeat_interleave(torch.arange(nrg_all, device=dev), poff[1:] - poff[:-1])
    rows = rgi * 16 + c["pr"].long()
    cols = c["pk"].long()
    keep = rows < n_cfg
    rows, cols, delta = rows[keep], cols[keep], c["pd"][keep]
    unit = cols // 64
    spl = ((unit + 1) * splits + nunits - 1) // nunits - 1          # split owning this 64-k unit
    k_start = (spl * nunits // splits) * 64
    cta = spl * nrb + rows // rows_cta
    ncta = nrb * splits
    counts = torch.bincount(cta, minlength=ncta)
    pmax = int(counts.max().item()) + 1 if rows.numel() else 1
    pmax += pmax % 2
    order = torch.argsort(cta, stable=True)
    cta, rows, cols, delta, k_start = cta[order], rows[order], cols[order], delta[order], k_start[order]
    start = torch.zeros(ncta + 1, dtype=torch.long, device=dev)
    start[1:] = torch.cumsum(counts, 0)
    slot = torch.arange(cta.numel(), device=dev) - start[cta] + 1
    tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=dev)
    tab[:, 0, 0] = counts.to(torch.int32)
    tab[cta, slot, 0] = (((rows % rows_cta) << 16) | (cols - k_start)).to(torch.int32)
    tab[cta, slot, 1] = delta.contiguous().view(torch.int32)
    return tab.contiguous(), pmax


def bfp_build_stream(src, comp, n_cfg, k, splits, rg, wsn, ks):
    """Per-CTA contiguous weight stream for a (splits, RG, WS, KS) decomposition of the first n_cfg rows.
    WS > 1: [cta = sp * nrb + rb][stage][warp][chunk].  WS == 1: [rb][stage over all K][warp][chunk] (independent of
    splits; a CTA reads its contiguous stage range).  src: compressed [nrg, K/64, 3, 32, 16] uint8 or uncompressed
    fragments [nrg, K/16, 32, 8] bf16."""
    dev = src.device
    nw = rg * wsn
    nrb = (n_cfg // 16) // rg
    ar = lambda n: torch.arange(n, device=dev)
    if wsn == 1:
        s_all = (k // 16) // ks
        rb = ar(nrb).view(-1, 1, 1, 1)
        st = ar(s_all).view(1, -1, 1, 1)
        w = ar(nw).view(1, 1, -1, 1)
        rg_idx = rb * rg + w
        if comp:
            u_idx = st * (ks // 4) + ar(ks // 4).view(1, 1, 1, -1)
            return src[rg_idx, u_idx].contiguous()
        s_idx = st * ks + ar(ks).view(1, 1, 1, -1)
        return src[rg_idx, s_idx].contiguous()
    kc = k // splits
    ns = kc // (16 * wsn)
    nst = ns // ks
    sp = ar(splits).view(-1, 1, 1, 1, 1)
    rb = ar(nrb).view(1, -1, 1, 1, 1)
    st = ar(nst).view(1, 1, -1, 1, 1)
    w = ar(nw).view(1, 1, 1, -1, 1)
    rg_idx = rb * rg + w % rg
    if comp:
        ku = ar(ks // 4).view(1, 1, 1, 1, -1)
        u_idx = sp * (kc // 64) + (w // rg) * (ns // 4) + st * (ks // 4) + ku
        out = src[rg_idx, u_idx]
    else:
        j = ar(ks).view(1, 1, 1, 1, -1)
        s_idx = sp * (kc // 16) + (w // rg) * ns + st * ks + j
        out = src[rg_idx, s_idx]
    return out.contiguous()


# ---------------------------------------------------------------- sk2: m <= 32, swapped operands, no cross-CTA reduction
bfp_SK2_CONFIGS = [  # (name, layer, n_rows, R rows/CTA, NT, NWR, WSK, UPW, NACC, CL); a name ending in _rNNN caps the ring at NNN KB
    ('o32h_r100', 'o', 8192, 32, 2, 2, 2, 2, 1, 1),
    ('o8h_r88', 'o', 8192, 32, 4, 1, 4, 1, 2, 1),
    ('pooler', 'pooler', 4096, 32, 4, 1, 8, 1, 2, 1),
    ('pooler32_r144', 'pooler', 4096, 32, 2, 2, 2, 2, 1, 1),
    ('pooler8h_r72', 'pooler', 4096, 16, 1, 2, 2, 4, 1, 1),
    ('qkv16_r176', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 1),
    ('qkv32_r208', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 1),
    ('qkv8_r176', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 1),
    ('router', 'router', 256, 8, 1, 1, 8, 1, 4, 3),
    ('slice8_r192', 'unpooler', 1024, 8, 1, 1, 8, 2, 2, 1),
    ('unpooler16h_r100', 'unpooler', 8192, 32, 4, 1, 4, 1, 2, 1),
    ('unpooler32_r144', 'unpooler', 8192, 64, 4, 2, 2, 2, 1, 1),
]
bfp_SK2_PICK = {  # (layer, tokr) -> config name
    ('o', 8): 'o8h_r88',
    ('o', 16): 'o8h_r88',
    ('o', 32): 'o32h_r100',
    ('pooler', 8): 'pooler8h_r72',
    ('pooler', 16): 'pooler',
    ('pooler', 32): 'pooler32_r144',
    ('qkv_gate', 8): 'qkv8_r176',
    ('qkv_gate', 16): 'qkv16_r176',
    ('qkv_gate', 32): 'qkv32_r208',
    ('router', 8): 'router',
    ('router', 16): 'router',
    ('router', 32): 'router',
    ('unpooler', 8): 'slice8_r192',
    ('unpooler', 16): 'unpooler16h_r100',
    ('unpooler', 32): 'unpooler32_r144',
}
bfp_SK2_ROT = False
bfp_SK2_K = {"pooler": 8192, "unpooler": 4096, "o": 1024, "qkv_gate": 8192, "router": 8192}
bfp_USE_SK2 = True
bfp_WIDE_BF16 = False  # the wide (RS-wgmma) kernel never beats cuBLAS for BF16 m 33..256


def bfp__sk2_name(layer, m):
    return bfp_SK2_PICK[(layer, bfp__sk2_tokr(m))]


def bfp__sk2_tokr(m):
    return 8 if m <= 8 else 16 if m <= 16 else 32


def bfp__sk2_geom(cfg, tokr):
    name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
    hl = layer == "router"
    tu = 2048 if hl else 768
    ups = WSK * UPW
    wby = NWR * WSK * UPW * NT * tu
    xby = ups * tokr * 128
    sst = (xby + wby + 1023) // 1024 * 1024
    nst = -(-(bfp_SK2_K[layer] // (ups * 64)) // CL)   # (the largest rank's stages)
    mt = 2 if tokr == 32 else 1
    toks = 16 * mt
    cp = R + 1
    nesc = 4 if not hl else 1   # (3 when the ring has 3 stages; sized for 4 here)
    rp = R + ((40 - (R % 32)) % 32)
    # everything but the ring: alignment pad, barriers, corrections, peer sums, escape table (<= 64 entries), zero row
    extra = (1024 + 16 * 16 + 32 + nesc * ((toks * cp + 3) // 4 * 4) * 4 + ((CL - 1) * WSK * toks * rp * 4 if CL > 1 else 0)
             + 64 * 8 + 128)
    cap = 200
    tail = name.rsplit("_r", 1)
    if len(tail) == 2 and tail[1].isdigit():
        cap = int(tail[1])
    ring = min(cap * 1024, 232448 - extra)
    nstage = max(2, min(16, nst, ring // sst))
    return hl, nstage


def bfp__sk2_src(sm90):
    if not sm90:
        return ('void sk2(at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, '
                'int64_t, int64_t, int64_t, at::Tensor) { TORCH_CHECK(false, "sk2 needs sm_90"); }\n'
                'void sk2f(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, '
                'at::Tensor, at::Tensor, int64_t, int64_t, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, '
                'at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, at::Tensor, at::Tensor, at::Tensor, at::Tensor, '
                'at::Tensor, at::Tensor, int64_t, at::Tensor, at::Tensor) { TORCH_CHECK(false, "sk2f needs sm_90"); }\n'
                'void sk2b(at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, '
                'int64_t, int64_t, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, '
                'at::Tensor, at::Tensor, int64_t) { TORCH_CHECK(false, "sk2b needs sm_90"); }\n'
                'void sk2_set_pdl(int64_t) {}')
    lines = []
    for ci, cfg in enumerate(bfp_SK2_CONFIGS):
        name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
        for tokr in (8, 16, 32):
            if bfp_SK2_PICK.get((layer, tokr)) != name:
                continue
            hl, nstage = bfp__sk2_geom(cfg, tokr)
            mt = 2 if tokr == 32 else 1
            ot = "float" if hl else "__nv_bfloat16"
            lines.append(f"if (cfg == {ci} && tokr == {tokr}) {{ launch_sk2<{mt}, {tokr}, {NT}, {NWR}, {WSK}, {UPW}, {nstage}, "
                         f"{str(not hl).lower()}, {str(hl).lower()}, {1 if (mt == 2 and not hl and nstage >= 4 and NT == 4) else NACC}, {CL}, {ot}>(Ws, x, out, ptab, pmax, base4, M, N, K, dg, dbg); return; }}")
    flines = []
    for ci, cfg in enumerate(bfp_SK2_CONFIGS):
        name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
        if name != "pooler":
            continue
        for tokr in (8, 16, 32):
            hl, nstage = bfp__sk2_geom(cfg, tokr)
            mt = 2 if tokr == 32 else 1
            nacc = 1 if (mt == 2 and nstage >= 4 and NT == 4) else NACC
            flines.append(f"if (tokr == {tokr}) {{ launch_sk2<{mt}, {tokr}, {NT}, {NWR}, {WSK}, {UPW}, {nstage}, true, false, "
                          f"{nacc}, 1, __nv_bfloat16, true>(Ws, x, out, ptab, pmax, base4, M, 4096, 8192, 0, nullptr, "
                          f"reinterpret_cast<const uint8_t*>(Wsr.data_ptr()), outr.data_ptr<float>(), part.data_ptr<float>(), "
                          # fuse_route 1: the routing tail (rids != nullptr); 2: the pre-expert norm only (grid barrier,
                          # routing in the MoE kernel): gpre / xm / part2 / dcnt, no rids
                          f"rcnt.data_ptr<int>(), fuse_route == 1 ? rbias.data_ptr<float>() : nullptr, "
                          f"fuse_route ? reinterpret_cast<const __nv_bfloat16*>(gpre.data_ptr()) : nullptr, "
                          f"fuse_route == 1 ? rids.data_ptr<int>() : nullptr, fuse_route == 1 ? rlids.data_ptr<int>() : nullptr, "
                          f"fuse_route == 1 ? rwts.data_ptr<float>() : nullptr, "
                          f"(fuse_route == 1 || fuse_route == 2) ? reinterpret_cast<__nv_bfloat16*>(xm.data_ptr()) : nullptr, "
                          f"fuse_route == 1 ? rls.data_ptr<int>() : nullptr, fuse_route ? part2.data_ptr<float>() : nullptr, "
                          f"(fuse_route == 1 || fuse_route == 2) ? dcnt.data_ptr<int>() : nullptr, "
                          # boundary-1 prologue through the b_* slots: alpha, beta, stream (b_sin), a (b_gp), 1+gw source (b_gn),
                          # s1 out (b_sout), h = this launch's x buffer (b_hout); no b_part / b_cnt
                          f"fuse_bnd1 ? b1a.data_ptr<float>() : nullptr, fuse_bnd1 ? b1b.data_ptr<float>() : nullptr, "
                          f"fuse_bnd1 ? bs.data_ptr<float>() : nullptr, "
                          f"fuse_bnd1 ? reinterpret_cast<const __nv_bfloat16*>(ba.data_ptr()) : nullptr, "
                          f"fuse_bnd1 ? reinterpret_cast<const __nv_bfloat16*>(b1g.data_ptr()) : nullptr, "
                          f"fuse_bnd1 ? b1s.data_ptr<float>() : nullptr, "
                          f"fuse_bnd1 ? reinterpret_cast<__nv_bfloat16*>(x.data_ptr()) : nullptr, "
                          f"fuse_bnd1 ? b1p.data_ptr<float>() : nullptr, fuse_bnd1 ? b1c.data_ptr<int>() : nullptr); return; }}")
    blines = []   # boundary tails: the o projection and the unpooler slice at tokr 8 (M <= 8), bf16 out, CL == 1
    for ci, cfg in enumerate(bfp_SK2_CONFIGS):
        name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
        if layer not in ("o", "unpooler") or bfp_SK2_PICK.get((layer, 8)) != name or CL != 1:
            continue
        hl, nstage = bfp__sk2_geom(cfg, 8)
        blines.append(f"if (cfg == {ci} && tokr == 8) {{ launch_sk2<1, 8, {NT}, {NWR}, {WSK}, {UPW}, {nstage}, true, false, "
                      f"{NACC}, 1, __nv_bfloat16, false, true>(Ws, x, out, ptab, pmax, base4, M, N, K, 0, nullptr, "
                      f"nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, "
                      f"nullptr, nullptr, alpha.data_ptr<float>(), beta.data_ptr<float>(), sin_.data_ptr<float>(), "
                      f"has_gp ? reinterpret_cast<const __nv_bfloat16*>(gp.data_ptr()) : nullptr, "
                      f"reinterpret_cast<const __nv_bfloat16*>(gn.data_ptr()), sout.data_ptr<float>(), "
                      f"reinterpret_cast<__nv_bfloat16*>(hout.data_ptr()), part.data_ptr<float>(), cnt.data_ptr<int>()); return; }}")
    return (bfp_SK2_SRC.replace("@@SK2_DISPATCH@@", "\n  ".join(lines))
            .replace("@@SK2F_DISPATCH@@", "\n  ".join(flines))
            .replace("@@SK2B_DISPATCH@@", "\n  ".join(blines)))


def bfp_sk2_tileunits(w):
    """W [N, K] -> [N/8, K/64, 32 lanes, 16] in mma.m16n8k16 B-fragment order: lane (g, t) holds row g and, for each
    k-step j of the 64-k unit, k = 16j + 2t, 2t+1, 2t+8, 2t+9."""
    n, k = w.shape
    t = w.view(n // 8, 8, k // 64, 4, 2, 4, 2)          # [tile, g, unit, j, h, t, p]: k = 64u + 16j + 8h + 2t + p
    return t.permute(0, 2, 1, 5, 3, 4, 6).reshape(n // 8, k // 64, 32, 16)


def bfp_sk2_compress(w):
    """Lossless 12-bit packing of BF16 W [N, K] into B-fragment tile-units [N/8, K/64, 768] (512 B low bytes, lane-major,
    then 256 B of 4-bit codes) + escapes (row, k, delta) that the kernel corrects exactly.  Returns None if unsuitable."""
    n, k = w.shape
    tu = bfp_sk2_tileunits(w)
    bits = tu.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hi = bits >> 8
    low = bits & 0xFF
    e71 = hi & 0x7F
    sign = hi >> 7
    hist = torch.bincount(e71.flatten(), minlength=128)
    win = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(win.argmax())
    o = (e71 - base).clamp(0, 7)
    code = (sign << 3) | o
    esc = (e71 < base) | (e71 > base + 7)
    if int(esc.sum()) > n * k // 200:
        return None
    T, U = n // 8, k // 64
    c8 = code.to(torch.uint8)
    w0 = c8[..., 0:4] | (c8[..., 4:8] << 4)
    w1 = c8[..., 8:12] | (c8[..., 12:16] << 4)
    codes = torch.cat([w0, w1], dim=-1)                  # [T, U, 32, 8]
    packed = torch.cat([low.to(torch.uint8).reshape(T, U, 512), codes.reshape(T, U, 256)], dim=-1).contiguous()
    idx = esc.nonzero()                                   # [E, 4]: tile, unit, lane, v
    dec_bits = ((((sign << 7) | (base + o)) << 8) | low)[esc]
    wdec = dec_bits.to(torch.int16).view(torch.bfloat16).float()
    wtrue = tu[esc].float()
    delta = wtrue - wdec
    if not bool(((wdec + delta) == wtrue).all()):
        return None
    lane, v = idx[:, 2], idx[:, 3]
    rows = idx[:, 0] * 8 + (lane >> 2)
    cols = idx[:, 1] * 64 + 16 * (v >> 2) + 8 * ((v >> 1) & 1) + 2 * (lane & 3) + (v & 1)
    return {"packed": packed, "base4": base * 0x01010101, "rows": rows, "cols": cols, "delta": delta.contiguous()}


def bfp_sk2_stream(tub, n_cfg, R, NT, NWR, WSK, UPW):
    """Stage-major stream [stage][cta][warp][unit][tile][bytes] for the first n_cfg rows."""
    dev = tub.device
    U = tub.shape[1]
    ups = WSK * UPW
    ar = lambda v: torch.arange(v, device=dev)
    s = ar(U // ups).view(-1, 1, 1, 1, 1)
    c = ar(n_cfg // R).view(1, -1, 1, 1, 1)
    w = ar(NWR * WSK).view(1, 1, -1, 1, 1)
    uu = ar(UPW).view(1, 1, 1, -1, 1)
    nt = ar(NT).view(1, 1, 1, 1, -1)
    tile = c * (R // 8) + (w // WSK) * NT + nt
    unit = s * ups + (w % WSK) * UPW + uu
    return tub[tile, unit].contiguous()


def bfp_sk2_table(comp, n_cfg, R):
    """Per-CTA escape table int2 [ncta][pmax]: entry 0 = (count, 0); entries ((row_local << 16) | k, delta bits)."""
    dev = comp["rows"].device
    rows, cols, delta = comp["rows"], comp["cols"], comp["delta"]
    keep = rows < n_cfg
    rows, cols, delta = rows[keep], cols[keep], delta[keep]
    ncta = n_cfg // R
    cta = rows // R
    counts = torch.bincount(cta, minlength=ncta)
    pmax = int(counts.max().item()) + 1 if rows.numel() else 1
    pmax += pmax % 2
    order = torch.argsort(cta.long() * 65536 + cols.long())   # per CTA in k order (the kernel walks K once)
    cta, rows, cols, delta = cta[order], rows[order], cols[order], delta[order]
    start = torch.zeros(ncta + 1, dtype=torch.long, device=dev)
    start[1:] = torch.cumsum(counts, 0)
    slot = torch.arange(cta.numel(), device=dev) - start[cta] + 1
    tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=dev)
    tab[:, 0, 0] = counts.to(torch.int32)
    tab[cta, slot, 0] = (((rows % R) << 16) | cols).to(torch.int32)
    tab[cta, slot, 1] = delta.view(torch.int32)
    return tab.contiguous(), pmax


def bfp_sk2_prepare(weights, state):
    state["sk2"] = {}
    if not (bfp_USE_SK2 and state["sm90"]):
        return
    dev = weights["o"].device
    shared = {}
    for ci, cfg in enumerate(bfp_SK2_CONFIGS):
        name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
        w = weights[layer]
        if layer == "router":
            hi = w.to(torch.bfloat16)
            lo = (w - hi.float()).to(torch.bfloat16)
            T, U = w.shape[0] // 8, w.shape[1] // 64
            tub = torch.cat([bfp_sk2_tileunits(hi).contiguous().view(torch.uint8).reshape(T, U, 1024),
                             bfp_sk2_tileunits(lo).contiguous().view(torch.uint8).reshape(T, U, 1024)], dim=-1)
            stream = bfp_sk2_stream(tub, n_rows, R, NT, NWR, WSK, UPW)
            tab, pmax, b4 = state["dummy"], 0, 0
        else:
            comp = state.setdefault("sk2_comp", {}).get(layer)
            if comp is None:
                comp = bfp_sk2_compress(w.contiguous())
                state["sk2_comp"][layer] = comp
            if comp is None:
                continue
            lkey = (layer, n_rows, R, NT, NWR, WSK, UPW)   # configs differing only in NACC / ring depth share a stream
            if lkey not in shared:
                shared[lkey] = (bfp_sk2_stream(comp["packed"], n_rows, R, NT, NWR, WSK, UPW),) + tuple(bfp_sk2_table(comp, n_rows, R))
            stream, tab, pmax = shared[lkey]
            b4 = comp["base4"]
        state["sk2"][name] = (ci, stream, tab, pmax, b4, n_rows)
    state.pop("sk2_comp", None)


def bfp__run_sk2(x, layer, m, n, k, state, dg=0, dbg=None):
    name = bfp__sk2_name(layer, m)
    ci, stream, tab, pmax, b4, n_rows = state["sk2"][name]
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    ext = bfp__get_ext()
    ext.sk2_set_pdl(bfp_SK2_PDL)
    ext.sk2(stream, x, out, tab, pmax, b4, m, n, k, ci, bfp__sk2_tokr(m), dg, state["dummy"] if dbg is None else dbg)
    return out


def bfp_run_sk2_tail(x, layer, m, n, k, state, tail):
    """sk2 GEMM (o projection or the unpooler's 1024-row slice, m <= 8) whose last CTA also computes the decode
    layer's residual/norm boundary: tail = (alpha fp32 [8192], beta fp32 [8192], sin fp32 [m, 8192] (the stream the
    gate mixes), gp bf16 [N] or None (post-FFN norm weight applied to the GEMM output before its norm), gn bf16 [8192]
    (the norm weight applied to the output), sout fp32 [m, 8192], hout bf16 [m, 8192]). Returns the GEMM output."""
    assert m <= 8 and layer in ("o", "unpooler")
    name = bfp__sk2_name(layer, m)
    ci, stream, tab, pmax, b4, n_rows = state["sk2"][name]
    alpha, beta, sin_, gp, gn, sout, hout = tail
    key = "btail_" + layer
    if key not in state:
        state[key] = (torch.zeros(32 * 256, dtype=torch.float32, device=x.device),   # [token][CTA] partial sums
                      torch.zeros(1, dtype=torch.int32, device=x.device))            # grid ticket (never reset)
    part, cnt = state[key]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    ext = bfp__get_ext()
    ext.sk2_set_pdl(bfp_SK2_PDL)
    ext.sk2b(stream, x, out, tab, pmax, b4, m, n, k, ci, bfp__sk2_tokr(m), alpha, beta, sin_, gn if gp is None else gp, gn,
             sout, hout, part, cnt, 0 if gp is None else 1)
    return out


def bfp_fused_pooler_router(x, state, route=None, bnd1=None, norm=None, gain=None):
    """Pooler and router of one decode step in ONE launch (m <= 32, sm_90): x [m, 8192] bf16 ->
    (pooler [m, 4096] bf16, router [m, 256] fp32). Same numerics as the separate sk2 kernels: 12-bit lossless pooler
    weights with exact escape corrections, router as BF16 hi + lo with FP32 accumulation (quarter-K partials added in
    a fixed order).
    route = (bias fp32 [256], pre-expert norm weight bf16 [4096], ids int32 [m, 8], local ids int32 [m, 8], weights
    fp32 [m, 8], xm bf16 [m, 4096], local_start int32 [1]): the last CTA of the launch also does the routing (top-8 of
    sigmoid(0.1768 logit) + bias; weights = the selected sigmoid scores over their sum; local id = (id - start) mod 256
    if < 32 else -1) and the pre-expert RMSNorm of the pooled rows into xm (the decode layer's _route_split)."""
    m = x.shape[0]
    assert m <= 32 and state.get("sm90") and "pooler" in state["sk2"] and "router" in state["sk2"]
    ci, stream, tab, pmax, b4, n_rows = state["sk2"]["pooler"]
    rstream = state["sk2"]["router"][1]
    if "fused_part" not in state:
        state["fused_part"] = torch.empty(32 * 4 * 32 * 8, dtype=torch.float32, device=x.device)
        state["fused_cnt"] = torch.zeros(32, dtype=torch.int32, device=x.device)
        state["fused_part2"] = torch.zeros(128 * 32, dtype=torch.float32, device=x.device)   # [pooler CTA][token] sum sq
        state["fused_dcnt"] = torch.zeros(1, dtype=torch.int32, device=x.device)             # grid ticket (never reset)
    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
    outr = torch.empty((m, 256), dtype=torch.float32, device=x.device)
    ext = bfp__get_ext()
    ext.sk2_set_pdl(bfp_SK2_PDL)
    d = state["dummy"]
    fuse_route = 0
    if route is not None:
        rbias, gpre, rids, rlids, rwts, xm, rls = route
        fuse_route = 1
    elif norm is not None:
        # norm = (gpre bf16 [4096]: the pre-expert norm weight, xm bf16 [m, 4096] out): every CTA normalizes its own
        # pooled columns after a grid barrier (routing is then the MoE kernel's, from the router logits returned here)
        gpre, xm = norm
        rbias = rids = rlids = rwts = rls = d
        fuse_route = 2
    elif gain is not None:
        # gain = gpre bf16 [4096]: the pooler output is pooled * (1 + gpre) in bf16 and state["fused_part2"] ([32][128]
        # fp32, token-major) holds the partial sums of pooled^2; the MoE kernel forms the per-token RMS scale itself
        # (its pre_norm_partials key) and routes from the logits. No grid barrier.
        gpre = gain
        rbias = rids = rlids = rwts = rls = xm = d
        fuse_route = 3
    else:
        rbias = gpre = rids = rlids = rwts = xm = rls = d
    # bnd1 = (a bf16 [m, 8192]: the o-projection output, s fp32 [m, 8192]: the input stream, alpha fp32 [8192],
    # beta fp32 [8192], gw bf16 [8192]: the pre-FFN norm weight, s1_out fp32 [m, 8192]): the launch computes its own
    # x = h = rmsnorm(alpha s + beta rmsnorm(a)) (1 + gw) into x (an empty buffer) in its prologue (m <= 8) and CTA 0
    # stores s1; the decode layer's _boundary1 without a launch
    if bnd1 is None:
        ba = bs = b1a = b1b = b1g = b1s = d
    else:
        ba, bs, b1a, b1b, b1g, b1s = bnd1
        assert m <= 8 and ba.dtype == torch.bfloat16 and bs.dtype == torch.float32 and b1g.dtype == torch.bfloat16
        if "bnd1_part" not in state:
            state["bnd1_part"] = torch.zeros(128 * 8 * 4, dtype=torch.float32, device=x.device)   # [CTA][row][4] partials
            state["bnd1_cnt"] = torch.zeros(1, dtype=torch.int32, device=x.device)                # grid tickets (never reset)
    b1p, b1c = (d, d) if bnd1 is None else (state["bnd1_part"], state["bnd1_cnt"])
    ext.sk2f(stream, rstream, x, out, outr, tab, pmax, b4, state["fused_part"], state["fused_cnt"], m, bfp__sk2_tokr(m),
             rbias, gpre, rids, rlids, rwts, xm, rls, state["fused_part2"], state["fused_dcnt"], fuse_route,
             ba, bs, b1a, b1b, b1g, b1s, 0 if bnd1 is None else 1, b1p, b1c)
    return out, outr


bfp_SK2_PDL = 0   # decode_layer: set to N > 0 for programmatic dependent launches (N weight stages before the wait)


def bfp__mt_key(m):
    return 0 if m == 1 else 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 8


def bfp_prepare(weights):
    ext = bfp__get_ext()
    dev = weights["qkv_gate"].device
    sm90_ = torch.cuda.get_device_capability()[0] >= 9
    # on sm_90 the sk2 kernels replace the older skinny kernels and the BF16 wide kernel (never faster than cuBLAS)
    old_paths = not (sm90_ and bfp_USE_SK2)
    state = {"raw": {}, "cnt": {}, "comp": {}, "mid_cnt": {}, "streams": {}, "tabs": {}, "base4": {}}
    state["dummy"] = torch.zeros(2, dtype=torch.int32, device=dev)
    # split-K partials: kernels run one at a time, so every layer shares one workspace
    ws_floats = 0
    for (lname, mtk), (splits, rgc, wsn, ks) in bfp_TMA_CONFIGS.items():
        mmax = {0: 1, 1: 8, 2: 16, 4: 32, 8: 64}[mtk]
        ws_floats = max(ws_floats, splits * mmax * bfp_w_rows[lname])
    for (lname, mm), (splits, rgc, wsn, ks) in bfp_TMA_ALT.items():
        ws_floats = max(ws_floats, splits * mm * bfp_w_rows[lname])
    state["ws_shared"] = torch.empty(ws_floats, dtype=torch.float32, device=dev)
    gemm_floats = 0
    for (lname, bn), (nwg, splits, stages) in bfp_GEMM_CONFIGS.items():
        gemm_floats = max(gemm_floats, (bfp_w_rows[lname] // (64 * nwg)) * splits * (nwg * 128) * (bn // 2))
    for (lname, tok, nz), (nwg, splits, stages) in bfp_WIDE_CONFIGS.items():
        gemm_floats = max(gemm_floats, nz * (bfp_w_rows[lname] // (64 * nwg)) * splits * (nwg * 64) * tok)
    state["mid_ws_shared"] = torch.empty(gemm_floats, dtype=torch.float32, device=dev)
    if bfp_GEMM_DIAG:
        state["ws_shared_diag"] = torch.empty(4 * 1024 * 1024, dtype=torch.float32, device=dev)
    for name, w in weights.items():
        if name == "router":
            hi = w.to(torch.bfloat16)
            lo = (w - hi.float()).to(torch.bfloat16)
            state["router_hi_t"] = hi.t().contiguous()
            state["router_lo_t"] = lo.t().contiguous()
            state["router_cat"] = torch.cat([hi, lo], dim=1).contiguous()   # gemm90 fallback (m 33..256)
            # hi and lo tiles of the same k-step back to back: [rg, s, 2, 32, 8]
            src = torch.stack([bfp_pack_fragments(hi), bfp_pack_fragments(lo)], dim=2).contiguous()
            comp, kk = False, w.shape[1]
            c = None
            # wide kernel stream: [rb][unit][warp][hi 4 k-steps x 512 B | lo 4 k-steps x 512 B], 64-row blocks
            nrg_r = w.shape[0] // 16
            u_all = w.shape[1] // 64
            t = src.view(nrg_r, u_all, 4, 2, 32, 8).permute(0, 1, 3, 2, 4, 5).contiguous()   # [rg, u, hl, 4, 32, 8]
            nwg_r = bfp_WIDE_CONFIGS[("router", 128, 1)][0]
            t = t.view(nrg_r // (4 * nwg_r), 4 * nwg_r, u_all, 2 * 4 * 32 * 8).permute(0, 2, 1, 3).contiguous()
            state["streams"][("router", 0, 4 * nwg_r, 1, 4, w.shape[0])] = t
            for (lname, tok, nz), (nwg, splits, stages) in bfp_WIDE_CONFIGS.items():
                if lname == "router":
                    state["cnt"][("router", "wide", tok, nz)] = torch.zeros(w.shape[0] // 16, dtype=torch.int32, device=dev)
        else:
            w = w.contiguous()
            state["raw"][name] = w
            c = bfp_pack_compressed(w) if name in bfp_COMPRESS else None
            if c is not None:
                src, comp = c["packed"].transpose(0, 1).contiguous(), True  # [rg, u, 3, 32, 16]
                state["base4"][name] = c["base4"]
                del c["packed"]
            else:
                src, comp = bfp_pack_fragments(w), False
            kk = w.shape[1]
        n = w.shape[0]
        state["comp"][name] = comp
        for (lname, mtk), (splits, rgc, wsn, ks) in bfp_TMA_CONFIGS.items():
            if lname != name or not old_paths:
                continue
            n_cfg = 1024 if (name == "unpooler" and mtk <= 1) else n
            key = (name, 0 if wsn == 1 else splits, rgc, wsn, ks, n_cfg)
            if key not in state["streams"]:
                state["streams"][key] = bfp_build_stream(src, comp, n_cfg, kk, splits, rgc, wsn, ks)
            ckey = (name, splits, rgc, wsn, ks, n_cfg)
            if ckey not in state["cnt"]:
                state["cnt"][ckey] = torch.zeros(n // 16, dtype=torch.int32, device=dev)  # per-config counters
            if comp and (name, splits, rgc, n_cfg) not in state["tabs"]:
                state["tabs"][(name, splits, rgc, n_cfg)] = bfp_build_patch_table(c, n_cfg, kk, splits, rgc)
        if comp and old_paths:
            for (lname, mm), (splits, rgc, wsn, ks) in bfp_TMA_ALT.items():
                if lname != name:
                    continue
                n_cfg = 1024 if (name == "unpooler" and mm <= 8) else n
                key = (name, 0 if wsn == 1 else splits, rgc, wsn, ks, n_cfg)
                if key not in state["streams"]:
                    state["streams"][key] = bfp_build_stream(src, comp, n_cfg, kk, splits, rgc, wsn, ks)
                ckey = (name, splits, rgc, wsn, ks, n_cfg)
                if ckey not in state["cnt"]:
                    state["cnt"][ckey] = torch.zeros(n // 16, dtype=torch.int32, device=dev)
                if (name, splits, rgc, n_cfg) not in state["tabs"]:
                    state["tabs"][(name, splits, rgc, n_cfg)] = bfp_build_patch_table(c, n_cfg, kk, splits, rgc)
        for (lname, tok, nz), (nwg, splits, stages) in bfp_WIDE_CONFIGS.items():
            if lname != name or not comp:
                continue
            rgc = 4 * nwg
            key = (name, 0, rgc, 1, 4, n)
            if key not in state["streams"]:
                state["streams"][key] = bfp_build_stream(src, comp, n, kk, splits, rgc, 1, 4)
            if (name, splits, rgc, n) not in state["tabs"]:
                state["tabs"][(name, splits, rgc, n)] = bfp_build_patch_table(c, n, kk, splits, rgc)
            state["cnt"][(name, "wide", tok, nz)] = torch.zeros(n // 16, dtype=torch.int32, device=dev)
        del src
        state["mid_cnt"][name] = torch.zeros(n // 16, dtype=torch.int32, device=dev)
    state["sm90"] = sm90_
    state["old_paths"] = old_paths
    bfp_sk2_prepare(weights, state)
    if bfp_GEMM_DIAG and state["sm90"]:
        state["wtile"], state["diag_cnt"] = {}, {}
        for layer in bfp_DIAG_LAYERS:
            state["wtile"][layer] = bfp_swizzle_tiles(state["raw"][layer], 128)
            for a_s in (0, 1):
                for sp_ in sorted({bfp_GEMM_CONFIGS[(layer, 128)][1], bfp_GEMM_CONFIGS[(layer, 256)][1], 1}):
                    state["diag_cnt"][(layer, a_s, sp_)] = torch.zeros(bfp_w_rows[layer] // 16, dtype=torch.int32, device=dev)
    bfp__selftest(weights, state)
    bfp__autotune(weights, state)
    torch.cuda.empty_cache()
    return state


# Which implementation runs a (layer, m): "skinny" (TMA-streamed tensor-core GEMV, m <= 32),
# "gemm" (TMA + wgmma split-K GEMM, sm_90, m <= 256) or "torch" (cuBLAS, as SGLang).
bfp_ROUTE_BF16 = [(32, "skinny"), (256, "wide")]
bfp_WIDE_SM80_TEST = False  # local check of the wide kernel's sm_80 path
bfp_ROUTE_ROUTER = [(32, "skinny"), (256, "wide")]


def bfp__route(layer, m, sm90):
    for mmax, r in (bfp_ROUTE_ROUTER if layer == "router" else bfp_ROUTE_BF16):
        if m <= mmax:
            if r == "gemm" and not sm90:
                return "torch"
            if r == "wide" and not (sm90 or bfp_WIDE_SM80_TEST):
                return "torch"
            return r
    return "torch"


def bfp__route_key(layer, m, route):
    if route == "sk2":
        return (layer, route, bfp__sk2_name(layer, m), bfp__sk2_tokr(m))
    if route == "skinny":
        return (layer, route, bfp__mt_key(m))
    if route == "wide":
        return (layer, route) + bfp__wide_key(m)
    return (layer, route, 64 if m <= 64 else 128 if m <= 128 else 256)


bfp_SELFTEST_M = (1, 8, 16, 32, 33, 128, 200, 256)


def bfp__selftest(weights, state):
    """Run every custom path once on a random input against an FP64 reference; disable (-> cuBLAS) any path that
    does not match.  Guards against a kernel that misbehaves on a GPU this file was not tested on."""
    state["disabled"] = set()
    g = torch.Generator(device=weights["o"].device)
    g.manual_seed(1234)
    for layer, w in weights.items():
        n, k = w.shape
        for m in bfp_SELFTEST_M:
            nn = 1024 if (layer == "unpooler" and m <= 8) else n
            x = torch.randn(m, k, device=w.device, generator=g).to(torch.bfloat16)
            ref = x.double() @ w[:nn].double().T
            if layer != "router":
                ref = ref.to(torch.bfloat16).double()
            tol = 2e-5 if layer == "router" else 5e-3
            for ch in bfp__candidates(layer, m, state):
                if ch[0] == "torch":
                    continue
                key = (("skinny_alt", layer, m) if (ch[0] == "skinny" and ch[2][0] == "alt")
                       else bfp__route_key(layer, m, ch[0]))
                try:
                    out = bfp__run(ch, x, layer, m, nn, k, state).double()
                    torch.cuda.synchronize()
                    err = ((out - ref).norm() / ref.norm()).item()
                except RuntimeError as e:  # launch failure -> fall back
                    err = float("inf")
                    if False: print(f"[okc] selftest {layer} m={m} {ch[0]}: {e}", file=sys.stderr)
                if not err < tol:
                    state["disabled"].add(key)
                    if False: print(f"[okc] selftest {layer} m={m} {ch[0]}: rel err {err:.3e} -> disabled", file=sys.stderr)


def bfp__run_skinny(x, layer, m, n, k, state, cfg, tu_key, dg=0):
    splits, rg, wsn, ks = cfg
    mtk = bfp__mt_key(m)
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    stream = state["streams"][(layer, 0 if wsn == 1 else splits, rg, wsn, ks, n)]
    cnt = state["cnt"][(layer, splits, rg, wsn, ks, n)]
    fn = bfp__get_ext().skinny_tma_a
    if state["comp"][layer]:
        tab, pmax = state["tabs"][(layer, splits, rg, n)]
        fn(stream, x, out, state["ws_shared"], cnt, tab, pmax, state["base4"][layer], m, n, k, k, splits,
           max(mtk, 1), rg, wsn, 1, bfp_w_rows[layer] // 16, ks, dg)
    else:
        fn(stream, x, out, state["ws_shared"], cnt, state["dummy"], 2, 0, m, n, k, k, splits,
           max(mtk, 1), rg, wsn, 0, bfp_w_rows[layer] // 16, ks, dg)
    return out


def bfp__run_wide(x, layer, m, n, k, state):
    tok, nz = bfp__wide_key(m)
    nwg, splits, _ = bfp_WIDE_CONFIGS[(layer, tok, nz)]
    rgc = 4 * nwg
    if layer == "router":
        out = torch.empty((m, n), dtype=torch.float32, device=x.device)
        tab, pmax, b4 = state["dummy"], 0, 0
    else:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        (tab, pmax), b4 = state["tabs"][(layer, splits, rgc, n)], state["base4"][layer]
    bfp__get_ext().wide(state["streams"][(layer, 0, rgc, 1, 4, n)], x, out, state["mid_ws_shared"],
                    state["cnt"][(layer, "wide", tok, nz)], tab, pmax, b4, m, n, k, splits, nwg, tok)
    return out


def bfp__run_gemm(x, layer, m, n, k, state):
    bn = 64 if m <= 64 else 128 if m <= 128 else 256
    nwg, splits, _ = bfp_GEMM_CONFIGS[(layer, bn)]
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    w = state["router_cat"] if layer == "router" else state["raw"][layer]
    bfp__get_ext().gemm90(w, state["dummy"], x, out, state["mid_ws_shared"], state["mid_cnt"][layer], m, n, w.shape[1], k,
                      splits, nwg, bn, 0)
    return out


def bfp__run_torch(x, layer, m, n, k, state):
    if layer == "router":
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc
    return F.linear(x, state["raw"][layer][:n])   # what SGLang's bf16 dispatch runs on H200


def bfp__run(choice, x, layer, m, n, k, state):
    kind = choice[0]
    if kind == "skinny":
        return bfp__run_skinny(x, layer, m, n, k, state, choice[1], choice[2])
    if kind == "wide":
        return bfp__run_wide(x, layer, m, n, k, state)
    if kind == "sk2":
        return bfp__run_sk2(x, layer, m, n, k, state)
    if kind == "gemm":
        return bfp__run_gemm(x, layer, m, n, k, state)
    if kind == "gq":
        return bfp__run_gq(x, layer, m, n, k, state)
    return bfp__run_torch(x, layer, m, n, k, state)


def bfp__static_choice(layer, m, state):
    route = bfp__route(layer, m, state["sm90"])
    if not state.get("old_paths", True):
        if route == "skinny":
            route = "sk2" if (bfp__sk2_name(layer, m) in state.get("sk2", {})
                              and bfp__route_key(layer, m, "sk2") not in state.get("disabled", ())) else "torch"
        elif route == "wide" and layer != "router" and not bfp_WIDE_BF16:
            route = "torch"
    if route != "torch" and bfp__route_key(layer, m, route) in state.get("disabled", ()):
        # fall back: router -> the wgmma GEMM (validated earlier) when possible, else cuBLAS
        route = "gemm" if (layer == "router" and route == "wide" and state["sm90"]
                           and bfp__route_key(layer, m, "gemm") not in state["disabled"]) else "torch"
    if route == "skinny":
        mtk = bfp__mt_key(m)
        return ("skinny", bfp_TMA_CONFIGS[(layer, mtk)], (layer, mtk))
    return (route,)


def bfp__candidates(layer, m, state):
    """Implementations worth timing for (layer, m): the static choice, alternative skinny configs, cuBLAS."""
    c = [bfp__static_choice(layer, m, state)]
    if c[0][0] == "skinny" and (layer, m) in bfp_TMA_ALT and ("skinny_alt", layer, m) not in state["disabled"]:
        c.append(("skinny", bfp_TMA_ALT[(layer, m)], ("alt", layer, m)))
    if (c[0][0] == "wide" and layer == "router" and state["sm90"]
            and bfp__route_key(layer, m, "gemm") not in state["disabled"]):
        c.append(("gemm",))
    if (m <= 32 and c[0][0] != "sk2" and bfp__sk2_name(layer, m) in state.get("sk2", {})
            and bfp__route_key(layer, m, "sk2") not in state["disabled"]):
        c.insert(0, ("sk2",))
    if (bfp_USE_GQ and state["sm90"] and 32 < m <= 256 and layer != "router" and (layer, bfp__gq_bn(m)) in bfp_GQ_CONFIGS
            and bfp__route_key(layer, m, "gq") not in state["disabled"]):
        c.insert(0, ("gq",))
    if c[-1][0] != "torch" and all(ch[0] != "torch" for ch in c):
        c.append(("torch",))
    return c


bfp_AUTOTUNE = True
bfp_TUNE_M = (1, 16, 32, 8, 128, 256)   # the scored token counts


def bfp__autotune(weights, state):
    """Time each candidate as a CUDA graph replay from a cold L2 (a read sweep larger than L2 before every replay,
    per-replay events, candidates interleaved), on synthetic activations (reads only the weights), and keep the
    fastest per (layer, m)."""
    state["choice"] = {}
    if not bfp_AUTOTUNE:
        return
    dev = weights["o"].device
    flush = torch.ones(64 * 1024 * 1024 // 1024, 1024, dtype=torch.float32, device=dev)   # 256 MB, read-only sweep
    sink = torch.empty(1024, dtype=torch.float32, device=dev)
    g = torch.Generator(device=dev)
    g.manual_seed(4321)
    reps = 24
    for m in bfp_TUNE_M:
        for layer, w in weights.items():
            n = 1024 if (layer == "unpooler" and m <= 8) else w.shape[0]
            k = w.shape[1]
            cands = bfp__candidates(layer, m, state)
            if len(cands) < 2:
                continue
            x = torch.randn(m, k, device=dev, generator=g).to(torch.bfloat16)
            graphs = []
            for ch in cands:
                bfp__run(ch, x, layer, m, n, k, state)
                torch.cuda.synchronize()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    bfp__run(ch, x, layer, m, n, k, state)
                graphs.append(gr)
            evs = []
            for r in range(reps):
                order = range(len(cands)) if r % 2 == 0 else reversed(range(len(cands)))
                for i in order:
                    torch.sum(flush, dim=0, out=sink)
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                    graphs[i].replay()
                    e1.record()
                    evs.append((i, e0, e1))
            torch.cuda.synchronize()
            per = [[] for _ in cands]
            for i, e0, e1 in evs:
                per[i].append(e0.elapsed_time(e1))
            med = [sorted(v)[len(v) // 2] for v in per]
            best = min(range(len(cands)), key=lambda i: med[i])
            state["choice"][(layer, m)] = cands[best]
            state.setdefault("tune_log", []).append((layer, m, [(c[0], round(1000 * v, 2)) for c, v in zip(cands, med)]))
            del graphs
    del flush
    torch.cuda.synchronize()
    # one line per shape in the run's log: candidate times (us, raw graph replay incl. ~4 us replay cost) and the pick
    for layer, m, tl in state.get("tune_log", []):
        ch = state["choice"][(layer, m)]
        desc = " ".join(f"{name}{'/' + 'x'.join(map(str, c[1])) if c[0] == 'skinny' else ''}={t}"
                        for (name, t), c in zip(tl, bfp__candidates(layer, m, state)))
        pick = ch[0] + ("/" + "x".join(map(str, ch[1])) if ch[0] == "skinny" else "")
        if False: print(f"[okc] tune {layer} m={m}: {desc} -> {pick}", file=sys.stderr)
    if False: print(f"[okc] disabled: {sorted(state['disabled'])}", file=sys.stderr)


def bfp__nearby_choice(layer, m, state):
    """An untuned m takes the choice tuned for the scored m of its bucket (1 | 2-8 | 9-16 | 17-32 | 33-128 |
    129-256): every implementation handles any m of its bucket."""
    tm = 1 if m <= 1 else 8 if m <= 8 else 16 if m <= 16 else 32 if m <= 32 else 128 if m <= 128 else 256 if m <= 256 else 0
    if not tm:
        return None
    ch = state.get("choice", {}).get((layer, tm))
    if ch is None:
        return None
    if ch[0] == "gq" and (layer, bfp__gq_bn(m)) not in bfp_GQ_CONFIGS:
        return None
    return ch


def bfp_kernel_fn(data, state):
    x = data["x"]
    cfg = data["config"]
    layer, m, n, k = cfg["layer"], cfg["m"], cfg["n"], cfg["k"]
    choice = state.get("choice", {}).get((layer, m)) or bfp__nearby_choice(layer, m, state) or bfp__static_choice(layer, m, state)
    out = bfp__run(choice, x, layer, m, n, k, state)
    if bfp_GEMM_DIAG:
        bfp__diag(x, layer, m, n, k, state, choice)
    return out


def bfp__diag(x, layer, m, n, k, state, choice):
    """Extra kernels for the judge's per-kernel profile (outputs discarded)."""
    if choice[0] == "skinny" and m == 1 and layer in ("o", "unpooler") and state["comp"][layer]:
        for dg in (1, 2, 3):   # same kernel without epilogue / x loads / compute
            bfp__run_skinny(x, layer, m, n, k, state, choice[1], choice[2], dg)
# ---- end of component: proj ----



if "bfp_CUTLASS_LAYERS" in globals():   # standalone proj composition: no CUTLASS build inside this layer's 300 s prepare
    bfp_CUTLASS_LAYERS = set()


def _bfp_prepare(weights, st):
    """Their prepare() on the five dense matrices, guarded: any failure just disables the backend."""
    st["bfp"] = None
    try:
        w5 = {k: weights[k].contiguous() for k in ("qkv_gate", "o", "pooler", "unpooler", "router")}
        st["bfp"] = bfp_prepare(w5)
    except Exception as ex:  # noqa: BLE001
        print("bfp disabled:", repr(ex)[:160], flush=True)


_BFP_LAYER = {"qkv": "qkv_gate", "o": "o", "un": "unpooler", "pool": "pooler", "router": "router"}
_T_COPY = {"qkv": "qkvT", "o": "oT", "pr": "prT", "un": "unpoolT"}   # dense name -> state key of the K-major weight copy


BFP_PDL_N = 2   # bf16_proj's sk2 / fused kernels: weight stages issued before griddepcontrol.wait (0 = off)


PDL_N_SWEEP = (2, 3, 4, 6)   # prepare(): per T, the sk2 pre-wait weight-stage count timed on the whole post-attention chain
_PDL_N_BY_T = {}             # T -> the winner (the ring has 4-6 stages at tokr 8; 6 = the whole ring streams before the wait)


def _bfp_pdl(T):
    """Their launchers read SK2_PDL (bfp_SK2_PDL once inlined) at every launch: same T rule as this file's glue."""
    global bfp_SK2_PDL
    bfp_SK2_PDL = _PDL_N_BY_T.get(T, BFP_PDL_N) if _pdl(T) else 0


def _bfp_gemm(st, name, x, m, n, k):
    _bfp_pdl(m)
    return bfp_kernel_fn({"x": x, "config": {"layer": _BFP_LAYER[name], "m": m, "n": n, "k": k}}, st["bfp"])


FUSE_ROUTE = True   # the fused pooler + router launch (sk2f) also does the pre-expert norm after a grid barrier (every CTA its
                    # own columns) and the routing of token t by CTA t (parallel, no serial tail; the d467a5c5 "last CTA does
                    # everything" tail was latency-bound: 29/35/43/64 us at T 1/8/16/32). Per T the tuner picks among
                    # _route_split / routing in the MoE kernel / this, on the chain time. The fused launch's last
                    # CTA, replacing the _route_split launch (T <= 32, agent/Marlin experts: plain xm layout)


def _fused_route(state, data, ids, lids, wts, xm, selftest=False):
    """The route tuple for bfp_fused_pooler_router, or None when the separate _route_split kernel must run (also
    when the composed proj module's fused launch has no routing tail: the standalone muse_bf16_proj, or when the
    prepare-time self-test of the tail failed)."""
    ls = data["local_start"]
    if (not FUSE_ROUTE or ls.dtype != torch.int32 or state["g_preexp"].dtype != torch.bfloat16
            or "route" not in bfp_fused_pooler_router.__code__.co_varnames
            or not (selftest or (state.get("fuse_route_ok") and state.get("route_mode", {}).get(T_of(ids)) == "sk2f"))):
        return None
    return (state["bias"], state["g_preexp"], ids, lids, wts, xm, ls)


def T_of(t):
    return int(t.shape[0])


FUSE_BOUNDARY = False   # OFF (d467a5c5: correct, same latency-bound tail; the tuner fell back to the v2 glue, d1 88 vs 75 us). T <= 8: the o-projection / unpooler-slice sk2 launches' last CTA computes _boundary1 / _boundary2
                        # (proj_fork's sk2b); enabled only when the prepare-time self-test matches the Triton kernels


def _route_selftest(st):
    """The fused routing tail against _route_split on synthetic activations (T = 1, 8, 16, 32): ids must match, the
    weights and xm within rounding. Any mismatch or exception disables the tail (st["fuse_route_ok"]); printed."""
    st["fuse_route_ok"] = False
    if st.get("bfp") is None or not FUSE_ROUTE or "route" not in bfp_fused_pooler_router.__code__.co_varnames:
        print("route tail: not available", flush=True)
        return
    dev = st["qkv"].device
    g = torch.Generator(device=dev).manual_seed(777)
    ls = torch.zeros(1, dtype=torch.int32, device=dev)
    worst = []
    for T in (1, 8, 16, 32):
        h = (torch.randn(T, HIDDEN, generator=g, device=dev) * 0.1).to(torch.bfloat16)
        a = [torch.empty((T, 8), dtype=torch.int32, device=dev), torch.empty((T, 8), dtype=torch.int32, device=dev),
             torch.empty((T, 8), dtype=torch.float32, device=dev), torch.empty((T, D_EXPERT), dtype=torch.bfloat16, device=dev)]
        b = [torch.empty_like(t) for t in a]
        rt = _fused_route(st, {"local_start": ls}, *a, selftest=True)
        try:
            _bfp_pdl(T)
            pooled, logits = bfp_fused_pooler_router(h, st["bfp"], rt)
            torch.cuda.synchronize()
        except Exception as ex:  # noqa: BLE001
            print(f"route tail T{T}: FAILED {repr(ex)[:300]}", flush=True)
            return
        _route_split[(T,)](pooled, logits, st["bias"], st["g_preexp"], b[0], b[2], b[3], ls, b[1], PERMUTED=False, num_warps=8)
        torch.cuda.synchronize()
        d_ids = int((a[0] != b[0]).sum()) + int((a[1] != b[1]).sum())
        d_w = float((a[2] - b[2]).abs().max())
        d_x = float((a[3].float() - b[3].float()).abs().max()) / max(1e-6, float(b[3].float().abs().max()))
        worst.append((T, d_ids, d_w, d_x))
        if d_ids or not d_w < 1e-4 or not d_x < 2e-2:
            print(f"route tail T{T}: MISMATCH ids {d_ids} wts {d_w:.2e} xm {d_x:.2e}", flush=True)
            return
    print("route tail ok: " + " ".join(f"T{T} ids0 w{w:.1e} x{x:.1e}" for T, _, w, x in worst), flush=True)
    st["fuse_route_ok"] = True


def _boundary_selftest(st):
    """The sk2b boundary tails (o projection -> _boundary1, unpooler slice -> _boundary2) against the Triton kernels
    at T = 1, 8 on synthetic inputs; any mismatch or exception disables them (st["fuse_bnd_ok"]); printed."""
    st["fuse_bnd_ok"] = False
    if (st.get("bfp") is None or not FUSE_BOUNDARY or "bfp_run_sk2_tail" not in globals()
            or "ab1" not in st):
        print("boundary tails: not available", flush=True)
        return
    dev = st["qkv"].device
    g = torch.Generator(device=dev).manual_seed(778)
    f32 = dict(device=dev, dtype=torch.float32)
    report = []
    for T in (1, 8):
        stream = torch.randn(T, HIDDEN, generator=g, device=dev)
        o_in = (torch.randn(T, 1024, generator=g, device=dev) * 0.1).to(torch.bfloat16)
        m = (torch.randn(T, D_EXPERT, generator=g, device=dev) * 0.1).to(torch.bfloat16)
        try:
            _bfp_pdl(T)
            s1a, ha = torch.empty((T, HIDDEN), **f32), torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
            P = globals()["bfp_run_sk2_tail"](o_in, "o", T, HIDDEN, 1024, st["bfp"],
                                              (st["ab1"][0], st["ab1"][1], stream, None, st["g_preffn"], s1a, ha))
            s2a, outa = torch.empty((T, HIDDEN), **f32), torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
            Pu = globals()["bfp_run_sk2_tail"](m, "unpooler", T, 1024, D_EXPERT, st["bfp"],
                                               (st["ab2"][0], st["ab2"][1], s1a, st["g_postffn"], st["g_next"], s2a, outa))
            torch.cuda.synchronize()
        except Exception as ex:  # noqa: BLE001
            print(f"boundary tails T{T}: FAILED {repr(ex)[:300]}", flush=True)
            return
        # the Triton kernels on the SAME GEMM outputs (P, Pu are the bf16 products the tails saw)
        s1b, hb = torch.empty((T, HIDDEN), **f32), torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
        _boundary1[(T,)](P.view(1, T, HIDDEN), 1, T, stream, st["gate1"], st["gate1"], st["g_preffn"], s1b, hb, num_warps=16)
        s2b, outb = torch.empty((T, HIDDEN), **f32), torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
        _boundary2[(T,)](Pu.view(1, T, 1024), 1, T, 1024, s1a, st["gate2"], st["gate2"], st["g_postffn"], st["g_next"],
                         s2b, outb, num_warps=16)
        torch.cuda.synchronize()
        errs = []
        for x, y in ((s1a, s1b), (ha, hb), (s2a, s2b), (outa, outb)):
            errs.append(float((x.float() - y.float()).norm() / max(1e-6, float(y.float().norm()))))
        report.append((T, errs))
        if not all(e < 2e-3 for e in errs):
            print(f"boundary tails T{T}: MISMATCH rel err s1 {errs[0]:.2e} h {errs[1]:.2e} s2 {errs[2]:.2e} out {errs[3]:.2e}", flush=True)
            return
    print("boundary tails ok: " + " ".join(f"T{T} " + "/".join(f"{e:.1e}" for e in errs) for T, errs in report), flush=True)
    st["fuse_bnd_ok"] = True


FUSE_BND1 = True   # the fused pooler + router launch computes h (its own x) and s1 in its PROLOGUE from the o-projection
                   # output (every CTA redundantly, before its x boxes; the weight ring is already streaming): no
                   # _boundary1 launch at T <= BND1_MAX_T. Enabled only when the prepare-time self-test matches.
BND1_MAX_T = 8     # cooperative prologue: CTA c owns its 64 columns of every row (<= 3 KB per CTA), two grid tickets
                   # at the kernel start, the weight ring streams meanwhile (the per-CTA whole-row versions were
                   # load-issue bound: +10..13 us isolated)


def _bnd1_fuse_ok(state, T):
    return (FUSE_BND1 and T <= BND1_MAX_T and state.get("fuse_bnd1_ok") and state.get("bfp") is not None
            and "bnd1" in bfp_fused_pooler_router.__code__.co_varnames and "ab1" in state
            and state["g_preffn"].dtype == torch.bfloat16 and state.get("bnd1_on", {}).get(T, True))


def _bnd1_selftest(st):
    """The fused launch's boundary-1 prologue against _boundary1 (+ the plain fused launch on _boundary1's h) at
    T = 1 and 8 on synthetic inputs; any mismatch or exception disables it (st["fuse_bnd1_ok"]); printed."""
    st["fuse_bnd1_ok"] = False
    if (not FUSE_BND1 or st.get("bfp") is None or "bnd1" not in bfp_fused_pooler_router.__code__.co_varnames
            or "ab1" not in st):
        print("boundary-1 prologue: not available", flush=True)
        return
    dev = st["qkv"].device
    g = torch.Generator(device=dev).manual_seed(779)
    f32 = dict(device=dev, dtype=torch.float32)
    report = []
    for T in (1, 8):   # the cooperative prologue handles M <= 8 rows
        a = (torch.randn(T, HIDDEN, generator=g, device=dev) * 0.1).to(torch.bfloat16)
        s = torch.randn(T, HIDDEN, generator=g, device=dev)
        h = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
        s1 = torch.empty((T, HIDDEN), **f32)
        try:
            _bfp_pdl(T)
            pooled, logits = bfp_fused_pooler_router(h, st["bfp"], None, (a, s, st["ab1"][0], st["ab1"][1], st["g_preffn"], s1))
            torch.cuda.synchronize()
        except Exception as ex:  # noqa: BLE001
            print(f"boundary-1 prologue T{T}: FAILED {repr(ex)[:300]}", flush=True)
            return
        s1r = torch.empty((T, HIDDEN), **f32)
        hr = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
        _boundary1[(T,)](a.view(1, T, HIDDEN), 1, T, s, st["gate1"], st["gate1"], st["g_preffn"], s1r, hr, num_warps=16)
        pooled_r, logits_r = bfp_fused_pooler_router(hr, st["bfp"])
        torch.cuda.synchronize()
        # h itself lives only in the launch's shared memory (the global x buffer is not written): judged through the
        # pooler / router outputs computed from it, against the plain launch on _boundary1's h
        errs = [float((x.float() - y.float()).norm() / max(1e-6, float(y.float().norm())))
                for x, y in ((s1, s1r), (pooled, pooled_r), (logits, logits_r))]
        report.append((T, errs))
        if not all(e < 2e-3 for e in errs):
            print(f"boundary-1 prologue T{T}: MISMATCH rel err s1 {errs[0]:.2e} pooled {errs[1]:.2e} "
                  f"logits {errs[2]:.2e}", flush=True)
            return
        # the prologue's own cost, isolated (graph replays, cold L2): it replaces a ~4.4 us launch, so it is enabled
        # only when it adds less than BND1_MAX_COST_US to the fused launch (check 0f57b859: +12 us in the chain)
        flush_ = torch.zeros(int(60e6), dtype=torch.float32, device=dev)
        global _FLUSH_SINK
        if _FLUSH_SINK is None:
            _FLUSH_SINK = torch.zeros((), dtype=torch.float32, device=dev)
        t_plain = _graph_time(lambda: bfp_fused_pooler_router(hr, st["bfp"]), flush_, reps=9)
        t_bnd1 = _graph_time(lambda: bfp_fused_pooler_router(h, st["bfp"], None, (a, s, st["ab1"][0], st["ab1"][1], st["g_preffn"], s1)),
                             flush_, reps=9)
        del flush_
        print(f"boundary-1 prologue T{T}: fused launch {t_plain:.1f} us -> with the prologue {t_bnd1:.1f} us", flush=True)
        if not (t_bnd1 - t_plain < BND1_MAX_COST_US):   # (the chain timing in _tune makes the final on/off choice)
            print(f"boundary-1 prologue T{T}: too costly (+{t_bnd1 - t_plain:.1f} us), off", flush=True)
            return
    print("boundary-1 prologue ok: " + " ".join(f"T{T} " + "/".join(f"{e:.1e}" for e in errs) for T, errs in report), flush=True)
    st["fuse_bnd1_ok"] = True


BND1_MAX_COST_US = 6.0   # isolated gate (a guard against a broken prologue); the chain timing per T decides below it


def _norm_selftest(st):
    """Routing in the MoE kernel + the fused launch's grid-barrier pre-expert norm, against _route_split + the MoE's
    plain entry on synthetic activations (T = 1, 8, 16, 32): xm must match within rounding, the chosen ids exactly,
    the expert output within rounding. Any mismatch or exception disables the path (st["route_moe_ok"]); printed."""
    st["route_moe_ok"] = False
    if (not ROUTE_IN_MOE or st.get("bfp") is None or "moe_kernel_fn_logits" not in globals()
            or "norm" not in bfp_fused_pooler_router.__code__.co_varnames or st.get("ma") is None
            or "wts" not in st["ma"]):
        print("routing in the MoE kernel: not available", flush=True)
        return
    dev = st["qkv"].device
    g = torch.Generator(device=dev).manual_seed(780)
    ls = torch.zeros(1, dtype=torch.int32, device=dev)
    report = []
    for T in (1, 8, 16, 32):
        h = (torch.randn(T, HIDDEN, generator=g, device=dev) * 0.1).to(torch.bfloat16)
        xm = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
        ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
        try:
            _bfp_pdl(T)
            pooled, logits = bfp_fused_pooler_router(h, st["bfp"], None, None, (st["g_preexp"], xm))
            # a synthetic skew so that some routes land on the 32 local experts (as the judge's inputs do)
            logits = logits + 0.25 * torch.randn(T, N_EXPERTS, generator=g, device=dev)
            out = _ma_run_logits(st, xm, logits, st["bias"], ls, ids, T)
            torch.cuda.synchronize()
        except Exception as ex:  # noqa: BLE001
            print(f"routing in the MoE kernel T{T}: FAILED {repr(ex)[:300]}", flush=True)
            return
        ids2 = torch.empty_like(ids)
        lids2 = torch.empty_like(ids)
        wts2 = torch.empty((T, 8), device=dev, dtype=torch.float32)
        xm2 = torch.empty_like(xm)
        _route_split[(T,)](pooled, logits, st["bias"], st["g_preexp"], ids2, wts2, xm2, ls, lids2, PERMUTED=False, num_warps=8)
        out2 = _ma_run(st, xm2, lids2, wts2, T)
        torch.cuda.synchronize()
        d_ids = int((ids != ids2).sum())
        d_x = float((xm.float() - xm2.float()).norm() / max(1e-6, float(xm2.float().norm())))
        d_o = float((out.float() - out2.float()).norm() / max(1e-6, float(out2.float().norm())))
        report.append((T, d_ids, d_x, d_o))
        if d_ids or not d_x < 2e-3 or not d_o < 2e-2:
            print(f"routing in the MoE kernel T{T}: MISMATCH ids {d_ids} xm {d_x:.2e} out {d_o:.2e}", flush=True)
            return
    print("routing in the MoE kernel ok: " + " ".join(f"T{T} ids0 x{x:.1e} o{o:.1e}" for T, _, x, o in report), flush=True)
    st["route_moe_ok"] = True
    # mode 3 (no barrier): the MoE kernel also forms the per-token RMS scale from the fused launch's partial sums; the
    # experts' input is rounded to bf16 before that scale, so the output differs from the _route_split path at the
    # bf16-rounding level (ids stay exact)
    st["route_moe_nb_ok"] = False
    if not (_moe_partials_ok() and "gain" in bfp_fused_pooler_router.__code__.co_varnames):
        print("routing in the MoE kernel (no barrier): not available", flush=True)
        return
    report = []
    for T in (1, 8, 16, 32):
        h = (torch.randn(T, HIDDEN, generator=g, device=dev) * 0.1).to(torch.bfloat16)
        ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
        try:
            _bfp_pdl(T)
            pooled, logits = bfp_fused_pooler_router(h, st["bfp"], None, None, None, st["g_preexp"])
            logits = logits + 0.25 * torch.randn(T, N_EXPERTS, generator=g, device=dev)
            out = _ma_run_logits(st, pooled, logits, st["bias"], ls, ids, T,
                                 partials=st["bfp"]["fused_part2"][:T * 128].view(T, 128))
            torch.cuda.synchronize()
        except Exception as ex:  # noqa: BLE001
            print(f"routing in the MoE kernel (no barrier) T{T}: FAILED {repr(ex)[:300]}", flush=True)
            return
        pooled2, logits2 = bfp_fused_pooler_router(h, st["bfp"])
        logits2 = logits   # the same perturbed logits
        ids2 = torch.empty_like(ids)
        lids2 = torch.empty_like(ids)
        wts2 = torch.empty((T, 8), device=dev, dtype=torch.float32)
        xm2 = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
        _route_split[(T,)](pooled2, logits2, st["bias"], st["g_preexp"], ids2, wts2, xm2, ls, lids2, PERMUTED=False, num_warps=8)
        out2 = _ma_run(st, xm2, lids2, wts2, T)
        torch.cuda.synchronize()
        d_ids = int((ids != ids2).sum())
        d_o = float((out.float() - out2.float()).norm() / max(1e-6, float(out2.float().norm())))
        report.append((T, d_ids, d_o))
        if d_ids or not d_o < 2e-2:
            print(f"routing in the MoE kernel (no barrier) T{T}: MISMATCH ids {d_ids} out {d_o:.2e}", flush=True)
            return
    print("routing in the MoE kernel (no barrier) ok: " + " ".join(f"T{T} ids0 o{o:.1e}" for T, _, o in report), flush=True)
    st["route_moe_nb_ok"] = True


def _gate_coeffs(g):
    """alpha, beta of a residual gate from its BF16 weight, exactly as _gate_ab forms them (prepare-time, fp32)."""
    w = g.float() / 0.3
    beta = torch.sigmoid(w)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=1e-3))
    return alpha.contiguous(), beta.contiguous()


def _bfp_tail_ok(state, T):
    return (FUSE_BOUNDARY and T <= 8 and state.get("bfp") is not None and "bfp_run_sk2_tail" in globals()
            and state.get("fuse_bnd_ok") and "ab1" in state and state["g_preffn"].dtype == torch.bfloat16
            and state["g_postffn"].dtype == torch.bfloat16 and state["g_next"].dtype == torch.bfloat16)


SCRATCH_GUARD = 1 << 16   # sentinel words after each buffer my glue allocates: wide enough for far overruns (v154m-style state ~4K words out)
SCRATCH_SENTINEL = 0x5EADBEEF


def _moe_weights(weights):
    """The expert weights in the moe module's own Weights layout (uint8 codes, e4m3 block scales, fp32 global
    scales, post_norm_w = this layer's post_expert_norm)."""
    return {"w13": weights["w13"].view(torch.uint8), "w13_sf": weights["w13_sf"], "w13_gs": weights["w13_gs"].float(),
            "w2": weights["w2"].view(torch.uint8), "w2_sf": weights["w2_sf"], "w2_gs": weights["w2_gs"].float(),
            "post_norm_w": weights["post_expert_norm"]}


def _moe_sfx(weights):
    """The moe module's build suffix (its prepare() computes the same): the PRMT scale decode only when every local
    block-scale byte is a positive normal e4m3."""
    return ("-DOKC_PRMTSC=1",) if (moe_PRMT_SCALES and moe__scales_prmt_exact(weights["w13_sf"], weights["w2_sf"])) else ()


if "moe_ROUTE_LOGITS_BUILDS" in globals():   # the MoE module's in-kernel routing from the router logits (its prepare()
    moe_ROUTE_LOGITS_BUILDS = True             # then also builds the OKC_ROUTE_LOGITS variants; see _ma_run_logits)
ROUTE_IN_MOE = True   # offer "routing in the MoE kernel" (no _route_split launch; the fused pooler+router launch
                      # normalizes xm after a grid barrier); the tuner picks per T on the chain time, self-test gated


def _moe_builds(sfx):
    """The builds the moe module's prepare() will ask for (its rule), so prepare() here can run them in parallel."""
    mods = [tuple(moe__main_defines(sfx)) + sfx]
    for d in (moe_TINY_DEFINES, moe_TINY2_DEFINES, moe_MID_DEFINES):
        if d and (sfx or "-DOKC_WGS=1" not in tuple(d)):   # the WGS build needs the PRMT scale decode
            mods.append(tuple(d) + sfx)
    if globals().get("moe_ROUTE_LOGITS_BUILDS") and "moe__logits_defines" in globals():
        mods += [moe__logits_defines(d) for d in list(mods)]
    return mods


def _route_mode(state, T):
    """Where this call's routing runs: "split" (the _route_split kernel), "sk2f" (CTA t of the fused pooler+router
    launch after its grid barrier), "moe" (every MoE CTA, xm normalized by the fused launch after a grid barrier) or
    "moe_nb" (every MoE CTA, which also forms the RMS scale from the fused launch's partial sums: no barrier)."""
    mode = state.get("route_mode", {}).get(T, "split")
    if mode == "moe" and not (ROUTE_IN_MOE and state.get("route_moe_ok") and "moe_kernel_fn_logits" in globals()):
        return "split"
    if mode == "moe_nb" and not (ROUTE_IN_MOE and state.get("route_moe_nb_ok") and _moe_partials_ok()):
        return "split"
    return mode


def _route_in_moe_ok(state, T):
    return _route_mode(state, T) == "moe"


def _moe_partials_ok():
    """Does the MoE module's logits entry take pre_norm_partials (the per-token RMS scale formed in its GEMM1 epilogue)?"""
    return ("moe_kernel_fn_logits" in globals()
            and "pre_norm_partials" in (moe_kernel_fn_logits.__code__.co_consts or ()))


def _ma_run_logits(state, x, logits, bias, ls, ids, T, partials=None):
    """The MoE module's in-kernel routing entry: top-8 of sigmoid(logit * 0.1768) + bias per token (ties to the lower
    id), renormalized weights, local ids, computed by every CTA from the router logits; CTA 0 writes the chosen global
    ids into `ids`. With partials (fp32 [T, 128], token-major partial sums of pooled^2), x is pooled * (1 + w) in bf16
    and the kernel applies rs_t = rsqrt(mean + 1e-5) in its GEMM1 epilogue. Returns the expert output bf16 [T, 4096]."""
    pdl = 1 if _pdl(T) else 0
    for ext in moe__EXTS.values():
        ext.okc_set_pdl(pdl)
    data = {"hidden_states": x, "router_logits": logits, "router_bias": bias, "local_start": ls, "topk_out": ids,
            "config": {"local_count": LOCAL_EXPERTS}}
    if partials is not None:
        data["pre_norm_partials"] = partials
    out, _ = moe_kernel_fn_logits(data, state["ma"])
    return out


def _ma_prepare(weights, st):
    """The moe module's own prepare() (weight packing in its layouts, build check, scratch), its scratch guarded:
    the scratch size depends on every build its kernel_fn may pick, so only its allocator is trusted to size it."""
    st["ma"] = moe_prepare(_moe_weights(weights))
    st["ma"]["scratch"] = [_guarded(t) for t in st["ma"]["scratch"]]


def _ma_run(state, x, lids, wts, T):
    """Their kernel_fn: the build for this token count, at most moe_MMAX tokens per launch. PDL as this call's rule
    (their extensions keep one flag each, set at every call; capture-time Python only)."""
    pdl = 1 if _pdl(T) else 0
    for ext in moe__EXTS.values():
        ext.okc_set_pdl(pdl)
    return moe_kernel_fn({"hidden_states": x, "topk_ids": lids, "topk_weights": wts,
                          "config": {"local_start": 0, "local_count": LOCAL_EXPERTS}}, state["ma"])






def _to_i32(x64):
    x64 = x64 & 0xFFFFFFFF
    return (x64 - (x64 >= 2**31).to(torch.int64) * 2**32).to(torch.int32)


def _repack_q(wp, interleave):
    """wp uint8 [E, R, 2048] (low nibble = even k) -> int32 [E, R/16, 64, 32, 4] mma-fragment words."""
    E, R, _ = wp.shape
    if interleave:
        wp = wp.view(E, 2, R // 32, 16, 2048).transpose(1, 2).reshape(E, R, 2048)
    x = wp.view(E, R // 16, 16, 256, 8).permute(0, 1, 3, 2, 4).to(torch.int64)  # [E, RB, kb, row, j]
    B3 = x[..., :8, :4]
    B2 = x[..., 8:, :4]
    B1 = x[..., :8, 4:]
    B0 = x[..., 8:, 4:]
    w = ((B0 & 15) | ((B1 & 15) << 4) | ((B2 & 15) << 8) | ((B3 & 15) << 12)
         | ((B0 >> 4) << 16) | ((B1 >> 4) << 20) | ((B2 >> 4) << 24) | ((B3 >> 4) << 28))
    w = _to_i32(w).reshape(E, R // 16, 64, 4, 32).permute(0, 1, 2, 4, 3).contiguous()
    return w


def _prepack_frag(w):
    """bf16 [N, K] -> int32 [N/16, K/16, 32 lanes, 4]: mma.m16n8k16 A-fragment order per (rowblock, k16)."""
    N, K = w.shape
    dev = w.device
    g = torch.arange(8, device=dev).view(8, 1, 1)
    c = torch.arange(4, device=dev).view(1, 4, 1)
    j = torch.arange(8, device=dev).view(1, 1, 8)
    rows = g + 8 * ((j // 2) % 2)
    cols = 2 * c + (j % 2) + 8 * (j // 4)
    x = w.view(N // 16, 16, K // 16, 16).permute(0, 2, 1, 3)  # [rb, kb, row, col]
    f = x[:, :, rows, cols]  # [rb, kb, 8, 4, 8]
    return f.reshape(N // 16, K // 16, 32, 8).contiguous().view(torch.int32)


def _repack_s(sf, interleave):
    """sf uint8 [E, R, 256] -> uint8 [E, R/16, 64, 8, 8]: (g, [rows g kb0..3, rows g+8 kb0..3])."""
    E, R, _ = sf.shape
    if interleave:
        sf = sf.view(E, 2, R // 32, 16, 256).transpose(1, 2).reshape(E, R, 256)
    x = sf.view(E, R // 16, 16, 64, 4).permute(0, 1, 3, 2, 4)  # [E, RB, ch, row, kk]
    return torch.cat([x[:, :, :, :8, :], x[:, :, :, 8:, :]], dim=-1).contiguous()


# ----------------------------------------------------------------------------------------------
# dense skinny GEMM: P[s, t, n] = sum_{k in split s} X[t, k] * W[n, k]
# ----------------------------------------------------------------------------------------------
@triton.jit
def _gemm_splitk(X, W, P, T, N, K, KSPLIT: tl.constexpr, BN: tl.constexpr, BT: tl.constexpr,
                 BK: tl.constexpr, USE_PDL: tl.constexpr = False):
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    pt = tl.program_id(2)
    rn = pn * BN + tl.arange(0, BN)
    rt = pt * BT + tl.arange(0, BT)
    rk = tl.arange(0, BK)
    kbase = ps * KSPLIT
    w_ptr = W + rn[:, None].to(tl.int64) * K + (kbase + rk)[None, :]
    x_ptr = X + rt[None, :] * K + (kbase + rk)[:, None]
    tmask = rt[None, :] < T
    acc = tl.zeros([BN, BT], dtype=tl.float32)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()   # X is the producer's output
    for _k in range(0, KSPLIT, BK):
        w = tl.load(w_ptr)
        x = tl.load(x_ptr, mask=tmask, other=0.0)
        acc = tl.dot(w, x, acc)
        w_ptr += BK
        x_ptr += BK
    tl.store(P + ps * T * N + rt[None, :] * N + rn[:, None], acc, mask=tmask)


@triton.jit
def _gemm_splitk_ng(X, G, W, P, T, N, K, KSPLIT: tl.constexpr, BN: tl.constexpr, BT: tl.constexpr,
                    BK: tl.constexpr, USE_PDL: tl.constexpr = False):
    """Like _gemm_splitk with x = rmsnorm_per_64(X) * sigmoid(G) applied on the fly (X bf16, G fp32)."""
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    pt = tl.program_id(2)
    rn = pn * BN + tl.arange(0, BN)
    rt = pt * BT + tl.arange(0, BT)
    rk = tl.arange(0, BK)
    kbase = ps * KSPLIT
    w_ptr = W + rn[:, None].to(tl.int64) * K + (kbase + rk)[None, :]
    x_ptr = X + rt[None, :] * K + (kbase + rk)[:, None]
    g_ptr = G + rt[None, :] * K + (kbase + rk)[:, None]
    tmask = rt[None, :] < T
    acc = tl.zeros([BN, BT], dtype=tl.float32)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()
    for _k in range(0, KSPLIT, BK):
        w = tl.load(w_ptr)
        x = tl.load(x_ptr, mask=tmask, other=0.0).to(tl.float32)
        gt = tl.load(g_ptr, mask=tmask, other=0.0)
        x3 = tl.reshape(x, [BK // 64, 64, BT])
        r = tl.rsqrt(tl.sum(x3 * x3, 1) / 64.0 + 1e-5)
        xn = tl.reshape(x3 * r[:, None, :], [BK, BT]) * tl.sigmoid(gt)
        acc = tl.dot(w, xn.to(tl.bfloat16), acc)
        w_ptr += BK
        x_ptr += BK
        g_ptr += BK
    tl.store(P + ps * T * N + rt[None, :] * N + rn[:, None], acc, mask=tmask)


@triton.jit
def _normgate(X, G, OUT, USE_PDL: tl.constexpr = False):
    t = tl.program_id(0)
    h = tl.arange(0, 16)[:, None]
    d = tl.arange(0, 64)[None, :]
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()
    x = tl.load(X + t * 1024 + h * 64 + d).to(tl.float32)
    x = x * tl.rsqrt(tl.sum(x * x, 1) / 64.0 + 1e-5)[:, None]
    x = x * tl.sigmoid(tl.load(G + t * 1024 + h * 64 + d))
    tl.store(OUT + t * 1024 + h * 64 + d, x.to(tl.bfloat16))


def dense_ng(x, gate, w, T, state=None):
    N, K = w.shape
    if T > 32 or (state is not None and state.get("mm_pick", {}).get((T, "o"), False)):
        xn = torch.empty((T, K), device=x.device, dtype=torch.bfloat16)
        _normgate[(T,)](x, gate, xn, num_warps=4, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
        return dense(xn, w, T, state, "o")
    BN, BK = 64, 128
    BT = 16 if T <= 16 else 32
    S = 1
    P = torch.empty((S, T, N), device=x.device, dtype=torch.float32)
    grid = (N // BN, S, triton.cdiv(T, BT))
    _gemm_splitk_ng[grid](x, gate, w, P, T, N, K, KSPLIT=K // S, BN=BN, BT=BT, BK=BK, num_warps=4, num_stages=4, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    return P, S


TUNED_T = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def _bucket(T):
    """The tuned token count whose choices apply to T (the next one at or above it)."""
    for t in TUNED_T:
        if T <= t:
            return t
    return TUNED_T[-1]


BFP_MAX_T = 32   # bf16_proj's qkv / o / unpooler kernels where they win inside the chain (T=1..32: +4-9% on those
                 # loads); at T=256 their routes measured 10 us slower than cuBLAS on 32x8 sliding, at T=128 a wash


def dense(x, w, T, state=None, name=None):
    """Returns (partials [S, T, N] fp32, S). The tuner picks cuBLAS or the split-K Triton kernel per
    (token count, weight matrix): cuBLAS wins on the o projection, the Triton one on the pooler+router."""
    N, K = w.shape
    pick = state.get("mm_pick", {}).get((T, name), False) if state is not None else False
    if pick == "bfp" and T <= BFP_MAX_T and name in _BFP_LAYER and state.get("bfp") is not None:
        # bf16_proj's kernel (BF16 out, as SGLang's GEMM): the consumers read P as one [1, T, N] partial
        return _bfp_gemm(state, name, x, T, N, K).view(1, T, N), 1
    if pick == "mm16":
        # cuBLAS with BF16 output (SGLang's own bf16 GEMM call), a different kernel choice from the fp32-output call;
        # the consumers read P in any dtype (the bf16_proj paths at T <= 32 already hand them bf16). (At T = 128 the
        # judge profile's split-K kernel is the qkv GEMM: N = 2304 is 24 tiles of 96 x 64; o and unpooler share the
        # 128 x 64 kernel at ~10.6 / ~19 us, the same as the proj module's isolated timings.)
        return torch.nn.functional.linear(x, w).view(1, T, N), 1
    if pick == "torchT" and name in _T_COPY and _T_COPY[name] in state and not (name == "un" and T <= 8):
        # cuBLAS on a K-major copy of W (x [T, K] @ W^T stored [K, N]): a different kernel choice again (the proj
        # module measured it faster for the unpooler at m = 128); fp32 out only for the router logits ("pr")
        wT = state[_T_COPY[name]]
        if name == "pr":
            return torch.mm(x, wT, out_dtype=torch.float32).view(1, T, N), 1
        return torch.mm(x, wT).view(1, T, N), 1
    if pick == "bfq" and name in _BFP_LAYER and state.get("bfp") is not None:
        # the proj module's own autotuned route at this token count (gq / torch / ... at m >= 33), any T
        return _bfp_gemm(state, name, x, T, N, K).view(1, T, N), 1
    if pick is True or pick == "bfp":       # above BFP_MAX_T a bfp pick runs as cuBLAS (measured in the chain)
        out = torch.mm(x, w.t(), out_dtype=torch.float32)
        return out.view(1, T, N), 1
    if T > 32:
        out = torch.mm(x, w.t(), out_dtype=torch.float32)
        return out.view(1, T, N), 1
    BN, BK = 64, 128
    BT = 16 if T <= 16 else 32
    tiles = (N // BN) * triton.cdiv(T, BT)
    S = 1
    while tiles * S * 2 <= 600 and K // (S * 2) >= 2 * BK:
        S *= 2
    P = torch.empty((S, T, N), device=x.device, dtype=torch.float32)
    grid = (N // BN, S, triton.cdiv(T, BT))
    _gemm_splitk[grid](x, w, P, T, N, K, KSPLIT=K // S, BN=BN, BT=BT, BK=BK, num_warps=4, num_stages=4, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    return P, S


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
@triton.jit
def _e4m3_to_f32(b):
    """uint8/int32 e4m3fn bits -> float32 (exact)."""
    b = b.to(tl.int32)
    h = (((b & 0x7F) << 7) | ((b & 0x80) << 8)).to(tl.int16).to(tl.float16, bitcast=True)
    return h.to(tl.float32) * 256.0


@triton.jit
def _f32_to_e4m3(x):
    """float32 -> e4m3fn bits (int32), round to nearest even, saturating to +-448."""
    x = tl.minimum(tl.maximum(x, -448.0), 448.0)
    bits = x.to(tl.int32, bitcast=True)
    sign = (bits >> 24) & 0x80
    a = tl.abs(x)
    # subnormal: multiples of 2^-9
    sub = (a * 512.0 + 0.5).to(tl.int32)  # round half up (ties are measure-zero)
    ab = a.to(tl.int32, bitcast=True)
    r = ab + 0x7FFFF + ((ab >> 20) & 1)
    r = r & 0x7FF00000
    e = (r >> 23) - 120
    m = (r >> 20) & 7
    nrm = (e << 3) | m
    nrm = tl.minimum(nrm, 0x7E)
    code = tl.where(a < 0.015625, sub, nrm)
    return code | sign


# ----------------------------------------------------------------------------------------------
# QK norm + RoPE + cache write
# ----------------------------------------------------------------------------------------------
@triton.jit
def _qk_prep(P, S: tl.constexpr, T, POS, LOC, INVF, KC, VC, KS, VS, QOUT, Q8OUT, GOUT, qk_factor,
             ROPE: tl.constexpr, USE_PDL: tl.constexpr = False):
    t = tl.program_id(0)
    base = P + t * 2304
    hq = tl.arange(0, 16)[:, None]
    hk = tl.arange(0, 2)[:, None]
    i = tl.arange(0, 32)[None, :]
    d = tl.arange(0, 64)[None, :]
    qe = tl.zeros([16, 32], dtype=tl.float32)
    qo = tl.zeros([16, 32], dtype=tl.float32)
    ke = tl.zeros([2, 32], dtype=tl.float32)
    ko = tl.zeros([2, 32], dtype=tl.float32)
    v = tl.zeros([2, 64], dtype=tl.float32)
    g = tl.zeros([1024], dtype=tl.float32)
    gi = tl.arange(0, 1024)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()
    for s in range(S):
        b = base + s * T * 2304
        qe += tl.load(b + hq * 64 + 2 * i)
        qo += tl.load(b + hq * 64 + 2 * i + 1)
        ke += tl.load(b + 1024 + hk * 64 + 2 * i)
        ko += tl.load(b + 1024 + hk * 64 + 2 * i + 1)
        v += tl.load(b + 1152 + hk * 64 + d)
        g += tl.load(b + 1280 + gi)
    rq = tl.rsqrt(tl.sum(qe * qe + qo * qo, 1) / 64.0 + 1e-5)[:, None]
    rk = tl.rsqrt(tl.sum(ke * ke + ko * ko, 1) / 64.0 + 1e-5)[:, None]
    qe = qe * rq
    qo = qo * rq
    ke = ke * rk
    ko = ko * rk
    if ROPE:
        pos = tl.load(POS + t).to(tl.float32)
        ang = pos * tl.load(INVF + i)
        c = tl.cos(ang)
        sn = tl.sin(ang)
        qe, qo = qe * c - qo * sn, qe * sn + qo * c
        ke, ko = ke * c - ko * sn, ke * sn + ko * c
    qb = QOUT + t * 1024 + hq * 64 + 2 * i
    tl.store(qb, (qe * qk_factor).to(tl.bfloat16))
    tl.store(qb + 1, (qo * qk_factor).to(tl.bfloat16))
    q8b = Q8OUT + t * 1024 + hq * 64 + 2 * i
    tl.store(q8b, _f32_to_e4m3(qe).to(tl.uint8))
    tl.store(q8b + 1, _f32_to_e4m3(qo).to(tl.uint8))
    tl.store(GOUT + t * 1024 + gi, g)
    loc = tl.load(LOC + t).to(tl.int64)
    kinv = 1.0 / tl.load(KS)
    vinv = 1.0 / tl.load(VS)
    kb = KC + loc * 128 + hk * 64 + 2 * i
    tl.store(kb, _f32_to_e4m3(ke * kinv).to(tl.uint8))
    tl.store(kb + 1, _f32_to_e4m3(ko * kinv).to(tl.uint8))
    tl.store(VC + loc * 128 + hk * 64 + d, _f32_to_e4m3(v * vinv).to(tl.uint8))


# ----------------------------------------------------------------------------------------------
# attention (split-KV flash decoding over a paged FP8 cache)
# ----------------------------------------------------------------------------------------------
@triton.jit
def _attn_split(Q, KC, VC, PT, SEQ, KS, ACC, ML, pt_stride, page_size, window, sm_scale,
                L: tl.constexpr, NSPLIT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                SLIDING: tl.constexpr):
    """One program = one request, both KV heads (128-byte K/V rows), one key split."""
    b = tl.program_id(0)
    s = tl.program_id(1)
    seq = tl.load(SEQ + b)
    m_off = tl.arange(0, BM)
    l_idx = m_off // 8
    hh = m_off % 8
    row_ok = m_off < L * 8
    qpos = seq + l_idx
    d = tl.arange(0, 64)
    qrow = (b * L + l_idx) * 16 + hh
    q0 = tl.load(Q + qrow[:, None] * 64 + d[None, :], mask=row_ok[:, None], other=0.0).to(tl.float16)
    q1 = tl.load(Q + (qrow + 8)[:, None] * 64 + d[None, :], mask=row_ok[:, None], other=0.0).to(tl.float16)
    total = seq + L
    if SLIDING:
        lo = tl.maximum(seq - window + 1, 0)
    else:
        lo = seq * 0
    n = total - lo
    per = tl.cdiv(tl.cdiv(n, NSPLIT), BN) * BN
    kst = lo + s * per
    ken = tl.minimum(total, kst + per)
    scale = sm_scale * 256.0 * tl.load(KS) * 1.4426950408889634
    m0 = tl.full([BM], float("-inf"), dtype=tl.float32)
    m1 = tl.full([BM], float("-inf"), dtype=tl.float32)
    l0 = tl.zeros([BM], dtype=tl.float32)
    l1 = tl.zeros([BM], dtype=tl.float32)
    a0 = tl.zeros([BM, 64], dtype=tl.float32)
    a1 = tl.zeros([BM, 64], dtype=tl.float32)
    for kb in range(kst, ken, BN):
        offs = kb + tl.arange(0, BN)
        kv_ok = offs < ken
        page = tl.load(PT + b * pt_stride + offs // page_size, mask=kv_ok, other=0)
        row = page.to(tl.int64) * page_size + offs % page_size
        kptr = row[:, None] * 128 + d[None, :]
        ok = kv_ok[None, :] & (offs[None, :] <= qpos[:, None])
        if SLIDING:
            ok = ok & (offs[None, :] > qpos[:, None] - window)
        k8 = tl.load(KC + kptr, mask=kv_ok[:, None], other=0).to(tl.int32)
        k16 = (((k8 & 0x7F) << 7) | ((k8 & 0x80) << 8)).to(tl.int16).to(tl.float16, bitcast=True)
        k8b = tl.load(KC + kptr + 64, mask=kv_ok[:, None], other=0).to(tl.int32)
        k16b = (((k8b & 0x7F) << 7) | ((k8b & 0x80) << 8)).to(tl.int16).to(tl.float16, bitcast=True)
        v8 = tl.load(VC + kptr, mask=kv_ok[:, None], other=0).to(tl.int32)
        v16 = (((v8 & 0x7F) << 7) | ((v8 & 0x80) << 8)).to(tl.int16).to(tl.float16, bitcast=True)
        v8b = tl.load(VC + kptr + 64, mask=kv_ok[:, None], other=0).to(tl.int32)
        v16b = (((v8b & 0x7F) << 7) | ((v8b & 0x80) << 8)).to(tl.int16).to(tl.float16, bitcast=True)
        # head 0
        sc = tl.where(ok, tl.dot(q0, tl.trans(k16)) * scale, float("-inf"))
        mn = tl.maximum(m0, tl.max(sc, 1))
        mu = tl.where(mn == float("-inf"), 0.0, mn)
        p = tl.exp2(sc - mu[:, None])
        al = tl.exp2(m0 - mu)
        l0 = l0 * al + tl.sum(p, 1)
        a0 = a0 * al[:, None] + tl.dot(p.to(tl.float16), v16)
        m0 = mn
        # head 1
        sc = tl.where(ok, tl.dot(q1, tl.trans(k16b)) * scale, float("-inf"))
        mn = tl.maximum(m1, tl.max(sc, 1))
        mu = tl.where(mn == float("-inf"), 0.0, mn)
        p = tl.exp2(sc - mu[:, None])
        al = tl.exp2(m1 - mu)
        l1 = l1 * al + tl.sum(p, 1)
        a1 = a1 * al[:, None] + tl.dot(p.to(tl.float16), v16b)
        m1 = mn
    ob0 = ((b * 2 + 0) * NSPLIT + s) * BM + m_off
    ob1 = ((b * 2 + 1) * NSPLIT + s) * BM + m_off
    tl.store(ACC + ob0[:, None] * 64 + d[None, :], a0)
    tl.store(ACC + ob1[:, None] * 64 + d[None, :], a1)
    tl.store(ML + ob0 * 2, m0)
    tl.store(ML + ob0 * 2 + 1, l0)
    tl.store(ML + ob1 * 2, m1)
    tl.store(ML + ob1 * 2 + 1, l1)


@triton.jit
def _attn_combine(ACC, ML, GATE, VS, OUT, L: tl.constexpr, NSPLIT: tl.constexpr, BM: tl.constexpr,
                  SB: tl.constexpr, NORMGATE: tl.constexpr = True):
    """One program per (token, q head): merge the splits, per-head RMSNorm, * sigmoid(gate)."""
    t = tl.program_id(0)
    h = tl.program_id(1)
    b = t // L
    l = t % L
    g = h // 8
    m_row = l * 8 + h % 8
    d = tl.arange(0, 64)
    base = (b * 2 + g) * NSPLIT
    M = tl.full([SB], float("-inf"), dtype=tl.float32)
    for s0 in range(0, NSPLIT, SB):
        sidx = s0 + tl.arange(0, SB)
        sm = sidx < NSPLIT
        M = tl.maximum(M, tl.load(ML + ((base + sidx) * BM + m_row) * 2, mask=sm, other=float("-inf")))
    Mx = tl.max(M, 0)
    Mx = tl.where(Mx == float("-inf"), 0.0, Mx)
    den = tl.zeros([SB], dtype=tl.float32)
    num = tl.zeros([64], dtype=tl.float32)
    for s0 in range(0, NSPLIT, SB):
        sidx = s0 + tl.arange(0, SB)
        sm = sidx < NSPLIT
        rows = (base + sidx) * BM + m_row
        m_s = tl.load(ML + rows * 2, mask=sm, other=float("-inf"))
        l_s = tl.load(ML + rows * 2 + 1, mask=sm, other=0.0)
        w = tl.exp2(m_s - Mx)
        den += w * l_s
        a = tl.load(ACC + rows[:, None] * 64 + d[None, :], mask=sm[:, None], other=0.0)
        num += tl.sum(w[:, None] * a, 0)
    o = num / tl.sum(den, 0) * (256.0 * tl.load(VS))
    if NORMGATE:
        o = o * tl.rsqrt(tl.sum(o * o, 0) / 64.0 + 1e-5)
        gt = tl.load(GATE + t * 1024 + h * 64 + d)
        o = o * tl.sigmoid(gt)
    tl.store(OUT + t * 1024 + h * 64 + d, o.to(tl.bfloat16))


# ----------------------------------------------------------------------------------------------
# boundaries, routing
# ----------------------------------------------------------------------------------------------
@triton.jit
def _gemm_o(X, G, W, AOUT, SUMS, STREAM, A1, B1, T, N, K: tl.constexpr, BN: tl.constexpr, BT: tl.constexpr,
            BK: tl.constexpr, NORMGATE: tl.constexpr):
    """o projection without a K split (K = 1024), so each program owns complete rows: it writes a and
    accumulates the four per-token sums the post-attention boundary needs (a^2, (a1 s)^2, (a1 s)(b1 a), (b1 a)^2)."""
    pn = tl.program_id(0)
    pt = tl.program_id(1)
    rn = pn * BN + tl.arange(0, BN)
    rt = pt * BT + tl.arange(0, BT)
    rk = tl.arange(0, BK)
    tmask = rt[None, :] < T
    acc = tl.zeros([BN, BT], dtype=tl.float32)
    w_ptr = W + rn[:, None].to(tl.int64) * K + rk[None, :]
    x_ptr = X + rt[None, :] * K + rk[:, None]
    for _k in range(0, K, BK):
        w = tl.load(w_ptr)
        if NORMGATE:
            xv = tl.load(x_ptr, mask=tmask, other=0.0).to(tl.float32)
            gv = tl.load(G + rt[None, :] * K + (rk[:, None] + _k))
            # per-64 rmsnorm * sigmoid(gate) on the fly (BK is a multiple of 64)
            xr = tl.reshape(xv, [BK // 64, 64, BT])
            rs = tl.rsqrt(tl.sum(xr * xr, 1) / 64.0 + 1e-5)
            xv = tl.reshape(xr * rs[:, None, :], [BK, BT]) * tl.sigmoid(gv)
            x = xv.to(tl.bfloat16)
        else:
            x = tl.load(x_ptr, mask=tmask, other=0.0)
        acc = tl.dot(w, x, acc)
        w_ptr += BK
        x_ptr += BK
    tl.store(AOUT + rt[None, :] * N + rn[:, None], acc, mask=tmask)
    st = tl.load(STREAM + rt[None, :] * N + rn[:, None], mask=tmask, other=0.0)
    a1v, b1v = _gate_ab(A1, rn)
    as_ = a1v[:, None] * st
    ba = b1v[:, None] * acc
    ok = (rt[None, :] < T) + tl.zeros([BN, BT], tl.int1)
    q0 = tl.sum(tl.where(ok, acc * acc, 0.0), 0)
    q1 = tl.sum(tl.where(ok, as_ * as_, 0.0), 0)
    q2 = tl.sum(tl.where(ok, as_ * ba, 0.0), 0)
    q3 = tl.sum(tl.where(ok, ba * ba, 0.0), 0)
    m = rt < T
    tl.atomic_add(SUMS + rt * 4 + 0, q0, mask=m)
    tl.atomic_add(SUMS + rt * 4 + 1, q1, mask=m)
    tl.atomic_add(SUMS + rt * 4 + 2, q2, mask=m)
    tl.atomic_add(SUMS + rt * 4 + 3, q3, mask=m)


@triton.jit
def _bnd1_apply(AOUT, SUMS, STREAM, A1, B1, G, S1OUT, HOUT, N: tl.constexpr, BC: tl.constexpr):
    """s1 = a1 * stream + b1 * a * rstd(a), h = s1 * rstd(s1) * g, from the four sums (one block per column slice)."""
    t = tl.program_id(0)
    c0 = tl.program_id(1) * BC
    offs = c0 + tl.arange(0, BC)
    s0 = tl.load(SUMS + t * 4 + 0)
    s1q = tl.load(SUMS + t * 4 + 1)
    s2q = tl.load(SUMS + t * 4 + 2)
    s3q = tl.load(SUMS + t * 4 + 3)
    r = tl.rsqrt(s0 / N + 1e-8)
    r1 = tl.rsqrt((s1q + 2.0 * r * s2q + r * r * s3q) / N + 1e-5)
    a = tl.load(AOUT + t * N + offs)
    a1v, b1v = _gate_ab(A1, offs)
    s1 = a1v * tl.load(STREAM + t * N + offs) + b1v * a * r
    tl.store(S1OUT + t * N + offs, s1)
    tl.store(HOUT + t * N + offs, (s1 * r1 * (1.0 + tl.load(G + offs).to(tl.float32))).to(tl.bfloat16))


@triton.jit
def _route_a(P, S: tl.constexpr, T, BIAS, IDS, WTS, LS, LIDS, POOL, PSUM, BC: tl.constexpr):
    """Reduce the pooler/router partials: one block per (token, pooled slice); slice 0 also does the top-8."""
    t = tl.program_id(0)
    j = tl.program_id(1)
    c = j * BC + tl.arange(0, BC)
    pooled = tl.zeros([BC], dtype=tl.float32)
    for s in range(S):
        pooled += tl.load(P + (s * T + t) * 4608 + c)
    tl.store(POOL + t * 4096 + c, pooled)
    tl.atomic_add(PSUM + t, tl.sum(pooled * pooled, 0))
    if j == 0:
        e = tl.arange(0, 256)
        logit = tl.zeros([256], dtype=tl.float32)
        for s in range(S):
            b = P + (s * T + t) * 4608
            logit += tl.load(b + 4096 + e) + tl.load(b + 4352 + e)
        sc = tl.sigmoid(logit * 0.1767766952966369)
        sel = sc + tl.load(BIAS + e)
        k8 = tl.arange(0, 8)
        ids = tl.zeros([8], dtype=tl.int32)
        wv = tl.zeros([8], dtype=tl.float32)
        for k in tl.static_range(8):
            idx = tl.argmax(sel, 0)
            val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
            ids = tl.where(k8 == k, idx, ids)
            wv = tl.where(k8 == k, val, wv)
            sel = tl.where(e == idx, float("-inf"), sel)
        wv = wv / (tl.sum(wv, 0) + 1e-15)
        tl.store(IDS + t * 8 + k8, ids)
        lid = (ids - tl.load(LS) + 256) % 256
        tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
        tl.store(WTS + t * 8 + k8, wv)


@triton.jit
def _route_b(POOL, PSUM, GPRE, XOUT, BC: tl.constexpr, PERMUTED: tl.constexpr):
    """x = rmsnorm(pooled) * (1 + pre_expert_norm), in the expert kernel's layout."""
    t = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    r = tl.rsqrt(tl.load(PSUM + t) / 4096.0 + 1e-5)
    x = tl.load(POOL + t * 4096 + c) * r * (1.0 + tl.load(GPRE + c).to(tl.float32))
    if PERMUTED:
        ch = c // 64
        kk = c % 64
        kb = kk // 16
        rr = kk % 16
        po = ((t * 64 + ch) * 4 + (rr % 8) // 2) * 16 + kb * 4 + (rr // 8) * 2 + (rr % 2)
        tl.store(XOUT + po, x.to(tl.bfloat16).to(tl.float16))
    else:
        tl.store(XOUT + t * 4096 + c, x.to(tl.bfloat16))


@triton.jit
def _bnd2_sums(P, S: tl.constexpr, T, NU: tl.constexpr, S1, A2, B2, GPOST, FOUT, SUMS, BC: tl.constexpr):
    """f = (unpooler partials summed) * g_postffn, and the four sums for the FFN boundary."""
    t = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    m = c < NU
    f = tl.zeros([BC], dtype=tl.float32)
    for s in range(S):
        f += tl.load(P + (s * T + t) * NU + c, mask=m, other=0.0)
    f = f * (1.0 + tl.load(GPOST + c).to(tl.float32))
    tl.store(FOUT + t * 8192 + c, f)
    st = tl.load(S1 + t * 8192 + c)
    a2v, b2v = _gate_ab(A2, c)
    as_ = a2v * st
    bf = b2v * f
    tl.atomic_add(SUMS + t * 4 + 0, tl.sum(f * f, 0))
    tl.atomic_add(SUMS + t * 4 + 1, tl.sum(as_ * as_, 0))
    tl.atomic_add(SUMS + t * 4 + 2, tl.sum(as_ * bf, 0))
    tl.atomic_add(SUMS + t * 4 + 3, tl.sum(bf * bf, 0))


@triton.jit
def _bnd2_apply(FOUT, SUMS, S1, A2, B2, GNEXT, SOUT, HOUT, N: tl.constexpr, BC: tl.constexpr):
    t = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    s0 = tl.load(SUMS + t * 4 + 0)
    s1q = tl.load(SUMS + t * 4 + 1)
    s2q = tl.load(SUMS + t * 4 + 2)
    s3q = tl.load(SUMS + t * 4 + 3)
    r = tl.rsqrt(s0 / N + 1e-8)
    r2 = tl.rsqrt((s1q + 2.0 * r * s2q + r * r * s3q) / N + 1e-5)
    a2v, b2v = _gate_ab(A2, c)
    s2 = a2v * tl.load(S1 + t * N + c) + b2v * tl.load(FOUT + t * N + c) * r
    tl.store(SOUT + t * N + c, s2)
    tl.store(HOUT + t * N + c, (s2 * r2 * (1.0 + tl.load(GNEXT + c).to(tl.float32))).to(tl.bfloat16))


@triton.jit
def _gate_ab(G, offs):
    """alpha, beta of a residual gate, from the checkpoint's BF16 weight (as SGLang's fused module does)."""
    w = tl.load(G + offs).to(tl.float32) / 0.3
    beta = tl.sigmoid(w)
    alpha = tl.sqrt(tl.maximum(tl.sigmoid(-w) * (1.0 + beta), 1e-3))
    return alpha, beta


@triton.jit
def _boundary1(P, S: tl.constexpr, T, STREAM, A1, B1, G, S1OUT, HOUT, USE_PDL: tl.constexpr = False):
    t = tl.program_id(0)
    offs = tl.arange(0, 8192)
    a1v, b1v = _gate_ab(A1, offs)              # weights and the layer's input stream: no producer dependency
    stv = tl.load(STREAM + t * 8192 + offs)
    gw = 1.0 + tl.load(G + offs).to(tl.float32)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()   # P (the o-projection) is the producer's output
    a = tl.zeros([8192], dtype=tl.float32)
    for s in range(S):
        a += tl.load(P + (s * T + t) * 8192 + offs)
    n = a * tl.rsqrt(tl.sum(a * a, 0) / 8192.0 + 1e-8)
    s1 = a1v * stv + b1v * n
    tl.store(S1OUT + t * 8192 + offs, s1)
    hh = s1 * tl.rsqrt(tl.sum(s1 * s1, 0) / 8192.0 + 1e-5) * gw
    tl.store(HOUT + t * 8192 + offs, hh.to(tl.bfloat16))


@triton.jit
def _route(P, S: tl.constexpr, T, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, PERMUTED: tl.constexpr, USE_PDL: tl.constexpr = False):
    t = tl.program_id(0)
    e = tl.arange(0, 256)
    logit = tl.zeros([256], dtype=tl.float32)
    pooled = tl.zeros([4096], dtype=tl.float32)
    c = tl.arange(0, 4096)
    biasv = tl.load(BIAS + e)                      # weights and local_start: no producer dependency
    gpre = 1.0 + tl.load(GPRE + c).to(tl.float32)
    lsv = tl.load(LS)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()   # P (pooler + router) is the producer's output
    for s in range(S):
        b = P + (s * T + t) * 4608
        logit += tl.load(b + 4096 + e) + tl.load(b + 4352 + e)
        pooled += tl.load(b + c)
    sc = tl.sigmoid(logit * 0.1767766952966369)
    sel = sc + biasv
    k8 = tl.arange(0, 8)
    ids = tl.zeros([8], dtype=tl.int32)
    wv = tl.zeros([8], dtype=tl.float32)
    for k in tl.static_range(8):
        idx = tl.argmax(sel, 0)
        val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
        ids = tl.where(k8 == k, idx, ids)
        wv = tl.where(k8 == k, val, wv)
        sel = tl.where(e == idx, float("-inf"), sel)
    wv = wv / (tl.sum(wv, 0) + 1e-15)
    tl.store(IDS + t * 8 + k8, ids)
    lid = (ids - lsv + 256) % 256
    tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
    tl.store(WTS + t * 8 + k8, wv)
    x = pooled * tl.rsqrt(tl.sum(pooled * pooled, 0) / 4096.0 + 1e-5) * gpre
    # BF16 activation, stored as exact fp16 in the MoE kernel's fragment-permuted layout
    ch = c // 64
    kk = c % 64
    kb = kk // 16
    r = kk % 16
    if PERMUTED:
        po = ((t * 64 + ch) * 4 + (r % 8) // 2) * 16 + kb * 4 + (r // 8) * 2 + (r % 2)
        tl.store(XOUT + po, x.to(tl.bfloat16).to(tl.float16))
    else:
        tl.store(XOUT + t * 4096 + c, x.to(tl.bfloat16))


@triton.jit
def _route_split(POOLED, LOGITS, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, PERMUTED: tl.constexpr, USE_PDL: tl.constexpr = False):
    """Same as _route but the pooler output (bf16) and the router logits (fp32) arrive as separate tensors."""
    t = tl.program_id(0)
    e = tl.arange(0, 256)
    c = tl.arange(0, 4096)
    biasv = tl.load(BIAS + e)                      # weights and local_start: no producer dependency
    gpre = 1.0 + tl.load(GPRE + c).to(tl.float32)
    lsv = tl.load(LS)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()   # pooled / logits are the producer's output
    logit = tl.load(LOGITS + t * 256 + e)
    pooled = tl.load(POOLED + t * 4096 + c).to(tl.float32)
    sc = tl.sigmoid(logit * 0.1767766952966369)
    sel = sc + biasv
    k8 = tl.arange(0, 8)
    ids = tl.zeros([8], dtype=tl.int32)
    wv = tl.zeros([8], dtype=tl.float32)
    for k in tl.static_range(8):
        idx = tl.argmax(sel, 0)
        val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
        ids = tl.where(k8 == k, idx, ids)
        wv = tl.where(k8 == k, val, wv)
        sel = tl.where(e == idx, float("-inf"), sel)
    wv = wv / (tl.sum(wv, 0) + 1e-15)
    tl.store(IDS + t * 8 + k8, ids)
    lid = (ids - lsv + 256) % 256
    tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
    tl.store(WTS + t * 8 + k8, wv)
    x = pooled * tl.rsqrt(tl.sum(pooled * pooled, 0) / 4096.0 + 1e-5) * gpre
    ch = c // 64
    kk = c % 64
    kb = kk // 16
    r = kk % 16
    if PERMUTED:
        po = ((t * 64 + ch) * 4 + (r % 8) // 2) * 16 + kb * 4 + (r // 8) * 2 + (r % 2)
        tl.store(XOUT + po, x.to(tl.bfloat16).to(tl.float16))
    else:
        tl.store(XOUT + t * 4096 + c, x.to(tl.bfloat16))


@triton.jit
def _moe_align(IDS, LS, R, CNT, OFF, SORTED, RPOS, ACTIVE, NACT, CH: tl.constexpr):
    start = tl.load(LS)
    ex = tl.arange(0, 32)
    cnt = tl.zeros([32], dtype=tl.int32)
    for c0 in range(0, R, CH):
        r = c0 + tl.arange(0, CH)
        rv = r < R
        ids = tl.load(IDS + r, mask=rv, other=0)
        local = (ids - start + 256) % 256
        isl = rv & (local < 32)
        oh = ((local[:, None] == ex[None, :]) & isl[:, None]).to(tl.int32)
        cnt += tl.sum(oh, 0)
    off = tl.cumsum(cnt, 0) - cnt
    run = off
    for c0 in range(0, R, CH):
        r = c0 + tl.arange(0, CH)
        rv = r < R
        ids = tl.load(IDS + r, mask=rv, other=0)
        local = (ids - start + 256) % 256
        isl = rv & (local < 32)
        oh = ((local[:, None] == ex[None, :]) & isl[:, None]).to(tl.int32)
        rank = tl.sum((tl.cumsum(oh, 0) - oh) * oh, 1)
        pos = tl.sum(run[None, :] * oh, 1) + rank
        tl.store(RPOS + r, tl.where(isl, pos, -1), mask=rv)
        tl.store(SORTED + pos, r, mask=isl)
        run += tl.sum(oh, 0)
    tl.store(CNT + ex, cnt)
    tl.store(OFF + ex, off)
    act = (cnt > 0).to(tl.int32)
    arank = tl.cumsum(act, 0) - act
    tl.store(ACTIVE + arank, ex, mask=act > 0)
    tl.store(NACT, tl.sum(act, 0))


# ----------------------------------------------------------------------------------------------
# MoE experts (NVFP4 W4A16)
# ----------------------------------------------------------------------------------------------
@triton.jit
def _dequant(w8, s8, BN: tl.constexpr, BK: tl.constexpr):
    """w8 uint8 [BN, BK/2] packed e2m1 (low nibble first), s8 uint8 [BN, BK/16] e4m3 -> bf16 [BN, BK]."""
    w = w8.to(tl.int32)
    codes = tl.reshape(tl.join(w & 0xF, w >> 4), [BN, BK])
    v = (((codes & 7) << 9) | ((codes & 8) << 12)).to(tl.int16).to(tl.float16, bitcast=True)
    sc = _e4m3_to_f32(s8) * 16384.0
    scf = tl.reshape(tl.broadcast_to(sc[:, :, None], [BN, BK // 16, 16]), [BN, BK])
    return (v.to(tl.float32) * scf).to(tl.bfloat16)


@triton.jit
def _moe_w13(X, W13, SF13, GS13, SORTED, CNT, OFF, ACTIVE, NACT, INTER,
             BN: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr):
    a = tl.program_id(0)
    j = tl.program_id(1)
    if a >= tl.load(NACT):
        return
    e = tl.load(ACTIVE + a)
    n_e = tl.load(CNT + e)
    off = tl.load(OFF + e)
    rows = j * BN + tl.arange(0, BN)
    wbase = W13 + e.to(tl.int64) * (8192 * 2048)
    sbase = SF13 + e.to(tl.int64) * (8192 * 256)
    gs = tl.load(GS13 + e)
    tt = tl.arange(0, BT)
    for t0 in range(0, n_e, BT):
        rv = (t0 + tt) < n_e
        route = tl.load(SORTED + off + t0 + tt, mask=rv, other=0)
        tok = route // 8
        accg = tl.zeros([BN, BT], dtype=tl.float32)
        accu = tl.zeros([BN, BT], dtype=tl.float32)
        for k in range(0, 4096, BK):
            x = tl.load(X + tok[None, :] * 4096 + (k + tl.arange(0, BK))[:, None], mask=rv[None, :], other=0.0)
            cb = k // 2 + tl.arange(0, BK // 2)
            cs = k // 16 + tl.arange(0, BK // 16)
            wg = tl.load(wbase + rows[:, None] * 2048 + cb[None, :])
            wu = tl.load(wbase + (rows + 4096)[:, None] * 2048 + cb[None, :])
            sg = tl.load(sbase + rows[:, None] * 256 + cs[None, :])
            su = tl.load(sbase + (rows + 4096)[:, None] * 256 + cs[None, :])
            accg = tl.dot(_dequant(wg, sg, BN, BK), x, accg)
            accu = tl.dot(_dequant(wu, su, BN, BK), x, accu)
        gg = accg * gs
        uu = accu * gs
        inter = gg / (1.0 + tl.exp(-gg)) * uu
        tl.store(INTER + (off + t0 + tt)[None, :] * 4096 + rows[:, None], inter.to(tl.bfloat16), mask=rv[None, :])


@triton.jit
def _moe_w2(INTER, W2, SF2, GS2, POSTW, CNT, OFF, ACTIVE, NACT, Z, SSP,
            BN: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, NT: tl.constexpr):
    a = tl.program_id(0)
    j = tl.program_id(1)
    if a >= tl.load(NACT):
        return
    e = tl.load(ACTIVE + a)
    n_e = tl.load(CNT + e)
    off = tl.load(OFF + e)
    rows = j * BN + tl.arange(0, BN)
    wbase = W2 + e.to(tl.int64) * (4096 * 2048)
    sbase = SF2 + e.to(tl.int64) * (4096 * 256)
    gs = tl.load(GS2 + e)
    pw = tl.load(POSTW + rows).to(tl.float32)
    tt = tl.arange(0, BT)
    for t0 in range(0, n_e, BT):
        rv = (t0 + tt) < n_e
        rr = off + t0 + tt
        acc = tl.zeros([BN, BT], dtype=tl.float32)
        for k in range(0, 4096, BK):
            x = tl.load(INTER + rr[None, :] * 4096 + (k + tl.arange(0, BK))[:, None], mask=rv[None, :], other=0.0)
            cb = k // 2 + tl.arange(0, BK // 2)
            cs = k // 16 + tl.arange(0, BK // 16)
            w = tl.load(wbase + rows[:, None] * 2048 + cb[None, :])
            s = tl.load(sbase + rows[:, None] * 256 + cs[None, :])
            acc = tl.dot(_dequant(w, s, BN, BK), x, acc)
        z = acc * gs * pw[:, None]
        tl.store(Z + rr[None, :] * 4096 + rows[:, None], z.to(tl.bfloat16), mask=rv[None, :])
        tl.store(SSP + rr * NT + j, tl.sum(z * z, 0), mask=rv)


@triton.jit
def _pexp_norm_sum(x_ptr, w_ptr, s_ptr, out_ptr, ids_ptr, TOPK: tl.constexpr, H: tl.constexpr, eps,
                   BLOCK_H: tl.constexpr):
    """post-expert norm (weight first) * routing weight, summed over slots. Same kernel as SGLang's
    fused_post_expert_norm_weighted_sum (Apache 2.0), called directly: its wrapper falls back to a chain of
    ~8 torch ops below 32 tokens, which costs 39-43 us on the judge instead of ~6."""
    tok = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    w = 1.0 + tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)   # the checkpoint's BF16 w; 1 + w formed here
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for k in tl.static_range(TOPK):
        if tl.load(ids_ptr + tok * TOPK + k) >= 0:
            x = tl.load(x_ptr + (tok * TOPK + k) * H + offs, mask=mask, other=0.0).to(tl.float32)
            y = x * w
            y = y * tl.rsqrt(tl.sum(y * y, axis=0) / H + eps)
            acc += y * tl.load(s_ptr + tok * TOPK + k).to(tl.float32)
    tl.store(out_ptr + tok * H + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _moe_combine(Z, SSP, RPOS, WTS, M, NT: tl.constexpr):
    t = tl.program_id(0)
    c = tl.arange(0, 4096)
    jj = tl.arange(0, NT)
    acc = tl.zeros([4096], dtype=tl.float32)
    for k in range(8):
        p = tl.load(RPOS + t * 8 + k)
        if p >= 0:
            ss = tl.sum(tl.load(SSP + p * NT + jj), 0)
            w = tl.load(WTS + t * 8 + k) * tl.rsqrt(ss / 4096.0 + 1e-8)
            acc += w * tl.load(Z + p * 4096 + c).to(tl.float32)
    tl.store(M + t * 4096 + c, acc.to(tl.bfloat16))


@triton.jit
def _boundary2(P, S: tl.constexpr, T, NU: tl.constexpr, S1, A2, B2, GPOST, GNEXT, SOUT, HOUT,
               USE_PDL: tl.constexpr = False):
    t = tl.program_id(0)
    offs = tl.arange(0, 8192)
    m = offs < NU
    gp = 1.0 + tl.load(GPOST + offs).to(tl.float32)      # weights, and S1 from _boundary1: every kernel between them (pooler/router, route,
    gn = 1.0 + tl.load(GNEXT + offs).to(tl.float32)      # moe, unpooler) triggers only after its own wait, so _boundary1 completed before this kernel began
    a2v, b2v = _gate_ab(A2, offs)
    s1v = tl.load(S1 + t * 8192 + offs)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()   # P (the unpooler) is the producer's output
    f = tl.zeros([8192], dtype=tl.float32)
    for s in range(S):
        f += tl.load(P + (s * T + t) * NU + offs, mask=m, other=0.0)
    f = f * gp
    n = f * tl.rsqrt(tl.sum(f * f, 0) / 8192.0 + 1e-8)
    s2 = a2v * s1v + b2v * n
    tl.store(SOUT + t * 8192 + offs, s2)
    hh = s2 * tl.rsqrt(tl.sum(s2 * s2, 0) / 8192.0 + 1e-5) * gn
    tl.store(HOUT + t * 8192 + offs, hh.to(tl.bfloat16))


# ----------------------------------------------------------------------------------------------

def _marlin_bank(weights):
    """The local experts in Marlin's layout (SGLang's own loader step), as baseline.py builds them."""
    import types
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
    from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
    device = weights["w13"].device
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous().clone(),
        w2_weight=weights["w2"].contiguous().clone(),
        w13_weight_scale=weights["w13_sf"].contiguous().clone(),
        w2_weight_scale=weights["w2_sf"].contiguous().clone(),
        w13_weight_scale_2=weights["w13_gs"].contiguous().clone(),
        w2_weight_scale_2=weights["w2_gs"].contiguous().clone(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
    expert_map[:LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "expert_map": expert_map, "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
    }


def _marlin_experts(x, topk_weights, topk_ids, state):
    """MuseSpark1xFp4Experts._marlin_expert_outputs + post-expert-norm finalize (as baseline.py).
    moe_block_size and use_atomic_add come from the prepare()-time sweep in _tune."""
    from sgl_kernel.scalar_type import scalar_types
    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    local_ids = topk_ids  # already (expert - local_start) % 256 mapped to 0..31 / -1 by the router kernel
    mc = state.get("marlin_cfg", {})
    cfgm = mc.get(num_tokens, mc.get(_bucket(num_tokens)))
    if cfgm is None:
        for block_size_m in (8, 16, 32, 48, 64):
            if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
                break
        atomic = False
    else:
        block_size_m, atomic = cfgm
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=atomic,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    m = torch.empty((num_tokens, size_k), device=x.device, dtype=x.dtype)
    _pexp_norm_sum[(num_tokens,)](y, state["g_postexp"], topk_weights, m, local_ids, TOPK=TOP_K,
                                  H=size_k, eps=1e-8, BLOCK_H=4096, num_warps=8, num_stages=2)
    return m


def _graph_time(fn, flush, reps=9):
    """Median graph-replay time with a cold L2. The flush is a READ sweep: zeroing a buffer leaves ~120 MB of
    dirty lines to write back, which inflates the next kernel by 15-25% (measured on the judge's H200) and not
    evenly across kernels, so it would skew every choice below."""
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(reps):
        _FLUSH_SINK.copy_(flush.sum())
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]


_FLUSH_SINK = None


def _calib_bias(st, dev, g, log=False):
    """Give the 32 local experts a skewed bias so that synthetic routing matches the judge's busiest-range load:
    ~1.6 local routes per token and ~15.7 distinct local experts at 32 tokens (reference statistics). Two
    parameters (skew width, bias size) found by a torch simulation of the real router; no input data involved."""
    router = st["pr"][D_EXPERT:].float()  # [256 hi; 256 lo] rows of width HIDDEN
    router = router[:N_EXPERTS] + router[N_EXPERTS:]
    gc = torch.Generator(device=dev).manual_seed(987654321)
    h = torch.randn(512, HIDDEN, generator=gc, device=dev)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-5)
    sc = torch.sigmoid(h.double() @ router.double().T * 0.1767766952966369).float()
    idx = torch.arange(LOCAL_EXPERTS, device=dev).float()

    def stats(shape, delta):
        sel = sc + st["bias"]
        sel[:, :LOCAL_EXPERTS] += delta * shape
        ids = sel.topk(TOP_K, dim=-1).indices
        loc = ids < LOCAL_EXPERTS
        rpt = float(loc.float().sum(-1).mean())
        counts = [torch.unique(ids[i:i + 32][loc[i:i + 32]]).numel() for i in range(0, 512, 32)]
        d32 = sum(counts) / len(counts)
        return rpt, d32

    best = None
    for width in (2.0, 3.0, 4.5, 6.0, 9.0):
        shape = torch.exp(-idx / width)
        lo, hi = 0.0, 3.0
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            if stats(shape, mid)[0] < 1.6:
                lo = mid
            else:
                hi = mid
        rpt, d32 = stats(shape, hi)
        if best is None or abs(d32 - 15.7) < abs(best[3] - 15.7):
            best = (shape, hi, rpt, d32)
    st["bias"][:LOCAL_EXPERTS] += best[1] * best[0]
    if log:
        print(f"calib delta {best[1]:.3f} routes/token {best[2]:.2f} experts@32 {best[3]:.1f}", flush=True)


def _tune(st, log=False):
    """prepare()-time choice of the post-attention path per token count: the persistent megakernel or the
    multi-kernel pipeline, and which routed-expert kernel. Uses synthetic activations only (never input data)."""
    dev = st["qkv"].device
    global _FLUSH_SINK
    flush = torch.zeros(int(60e6), dtype=torch.float32, device=dev)  # 240 MB, read-swept between replays
    _FLUSH_SINK = torch.zeros((), dtype=torch.float32, device=dev)
    g = torch.Generator(device=dev).manual_seed(12345)
    # Synthetic routing is uniform over all 256 experts, but the judge picks each input's local range as the
    # busiest one: ~1.6 local routes per token (2.6 at T=1) over few distinct experts. Calibrate a skewed bias
    # on the local experts (torch-side simulation with the real router weights) so the tuned load is realistic.
    bias0 = st["bias"].clone()
    _calib_bias(st, dev, g, log)
    st["moe_pick"] = {}
    st["gluev2"] = {}
    st["marlin_cfg"] = {}
    st["mm_pick"] = {}
    pb_ok = {}
    for T in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        data = {"stream": torch.randn(T, HIDDEN, generator=g, device=dev),
                "local_start": torch.zeros(1, dtype=torch.int32, device=dev)}
        o_in = (torch.randn(T, 1024, generator=g, device=dev) * 0.1).to(torch.bfloat16)
        if T <= 4:  # redraw until the token(s) route to about as many local experts as the judge's inputs do
            want = max(2, int(round(2.57 * T / 1.0)) if T == 1 else 2 * T)
            for _ in range(40):
                _o, _s, _ids = _after_attention(data, st, o_in, None, T)
                if int(((_ids >= 0) & (_ids < LOCAL_EXPERTS)).sum()) >= min(want, 8 * T):
                    break
                data["stream"] = torch.randn(T, HIDDEN, generator=g, device=dev)
                o_in = (torch.randn(T, 1024, generator=g, device=dev) * 0.1).to(torch.bfloat16)
        best, bestt, bestv2 = None, 1e30, True
        times, dfltv2 = {}, False
        msg = []
        nloc = ndis = -1
        for name, w, K in (("qkv", st["qkv"], HIDDEN), ("o", st["o"], 1024), ("pr", st["pr"], HIDDEN),
                           ("un", st["unpool_slice"] if T <= 8 else st["unpool"], D_EXPERT)):
            xd = (torch.randn(T, K, generator=g, device=dev) * 0.1).to(torch.bfloat16)
            # the pooler + router candidates are timed WITH the routing kernel / the fused routing tail they imply
            r_ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
            r_lids = torch.empty((T, 8), device=dev, dtype=torch.int32)
            r_wts = torch.empty((T, 8), device=dev, dtype=torch.float32)
            r_xm = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)

            def _pr_split(pooled, logits):
                _route_split[(T,)](pooled, logits, st["bias"], st["g_preexp"], r_ids, r_wts, r_xm, data["local_start"],
                                   r_lids, PERMUTED=False, num_warps=8, USE_PDL=_pdl(T), launch_pdl=_pdl(T))

            def _pr_route(PS):   # the combined pooler+router GEMM output -> the routing kernel that consumes it
                P_, S_ = PS
                _route[(T,)](P_, S_, T, st["bias"], st["g_preexp"], r_ids, r_wts, r_xm, data["local_start"], r_lids,
                             PERMUTED=False, num_warps=8, USE_PDL=_pdl(T), launch_pdl=_pdl(T))

            def _dense_t(pick):   # time what dense() runs for this pick (+ the routing kernel for "pr")
                st["mm_pick"][(T, name)] = pick
                if name == "pr":
                    return _graph_time(lambda: _pr_route(dense(xd, w, T, st, name)), flush, reps=5)
                return _graph_time(lambda: dense(xd, w, T, st, name), flush, reps=5)

            if st.get("bfp") is not None and T <= BFP_MAX_T:   # above BFP_MAX_T a "bfp" pick runs as cuBLAS: not a candidate
                st["mm_pick"][(T, name)] = "bfp"
                try:
                    if name == "pr":
                        t_bfp = _graph_time(lambda: _pr_split(_bfp_gemm(st, "pool", xd, T, D_EXPERT, HIDDEN),
                                                              _bfp_gemm(st, "router", xd, T, N_EXPERTS, HIDDEN)),
                                            flush, reps=5)
                    else:
                        nn = w.shape[0]
                        t_bfp = _graph_time(lambda: _bfp_gemm(st, name, xd, T, nn, K), flush, reps=5)
                except Exception:  # noqa: BLE001
                    t_bfp = 1e30
            else:
                t_bfp = 1e30
            try:
                t_tri = _dense_t(False)
            except Exception:  # noqa: BLE001
                t_tri = 1e30
            try:
                t_mm = _dense_t(True)
            except Exception:  # noqa: BLE001
                t_mm = 1e30
            t_bfpf = 1e30
            if name == "pr" and T <= 32 and st.get("bfp") is not None and "bfp_fused_pooler_router" in globals():
                try:
                    _bfp_pdl(T)
                    rt = _fused_route(st, data, r_ids, r_lids, r_wts, r_xm)
                    if rt is not None:
                        t_bfpf = _graph_time(lambda: bfp_fused_pooler_router(xd, st["bfp"], rt), flush, reps=5)
                    else:
                        t_bfpf = _graph_time(lambda: _pr_split(*bfp_fused_pooler_router(xd, st["bfp"])), flush, reps=5)
                except Exception as ex:  # noqa: BLE001  (sm_80 or an older bf16_proj file: not offered)
                    t_bfpf = 1e30
                    if log:
                        msg.append(f"bfpfX:{repr(ex)[:120]}")
            if log and name == "pr":
                msg.append(f"pr[T{t_tri:.0f} M{t_mm:.0f} B{t_bfp:.0f} F{t_bfpf:.0f}]")
            t_mm16 = 1e30
            if name != "pr":   # the router logits stay fp32 (hi + lo rows); the other three may come out bf16 as SGLang's
                try:
                    t_mm16 = _dense_t("mm16")
                except Exception:  # noqa: BLE001
                    t_mm16 = 1e30
            t_torchT = t_bfq = 1e30
            if T > 32:   # the T > 32 candidates: cuBLAS on the K-major copy, and the proj module's own tuned route
                try:
                    t_torchT = _dense_t("torchT")
                except Exception:  # noqa: BLE001
                    t_torchT = 1e30
                if st.get("bfp") is not None:
                    try:
                        if name == "pr":
                            t_bfq = _graph_time(lambda: _pr_split(_bfp_gemm(st, "pool", xd, T, D_EXPERT, HIDDEN),
                                                                  _bfp_gemm(st, "router", xd, T, N_EXPERTS, HIDDEN)),
                                                flush, reps=5)
                        else:
                            t_bfq = _dense_t("bfq")
                    except Exception as ex:  # noqa: BLE001
                        t_bfq = 1e30
                        if log:
                            msg.append(f"bfqX:{repr(ex)[:100]}")
            opts = {False: t_tri, True: t_mm, "bfp": t_bfp, "bfpf": t_bfpf, "mm16": t_mm16, "torchT": t_torchT, "bfq": t_bfq}
            st.setdefault("mm_times", {})[(T, name)] = opts
            bestp = min(opts, key=lambda p: opts[p])
            st["mm_pick"][(T, name)] = bestp
            if log:
                letters = {False: 'T', True: 'M', 'bfp': 'B', 'bfpf': 'F', 'mm16': 'm', 'torchT': 't', 'bfq': 'q'}
                msg.append(f"{name}{letters[bestp]}:{opts[bestp]:.0f}")
                if T >= 128:   # DIAG: every candidate's graph time (us) for the T > 32 dense GEMMs
                    msg.append(f"{name}[" + " ".join(f"{letters[p]}{v:.1f}" for p, v in opts.items() if v < 1e29) + "]")
        if st["bank"] is not None and T <= 384:
            xm = (torch.randn(T, D_EXPERT, generator=g, device=dev) * 0.1).to(torch.bfloat16)
            raw = torch.randint(0, N_EXPERTS, (T, 8), generator=g, device=dev, dtype=torch.int32)
            # the convention the router kernel writes: local index 0..31, -1 for another GPU's expert
            mids = torch.where(raw < LOCAL_EXPERTS, raw, torch.full_like(raw, -1)).contiguous()
            mwt = torch.full((T, 8), 0.125, device=dev)
            bestm, bestmt = None, 1e30
            for bs in (8, 16, 32, 48, 64):
                for atomic in (False, True):
                    st["marlin_cfg"][T] = (bs, atomic)
                    try:
                        tm = _graph_time(lambda: _marlin_experts(xm, mwt, mids, st), flush, reps=5)
                    except Exception:  # noqa: BLE001
                        continue
                    if tm < bestmt:
                        bestm, bestmt = (bs, atomic), tm
            if bestm is not None:
                st["marlin_cfg"][T] = bestm
                if log:
                    msg.append(f"mcfg{bestm[0]}{'a' if bestm[1] else ''}:{bestmt:.0f}")
            else:
                st["marlin_cfg"].pop(T, None)
        # my own MoE kernel (FP16 expert input) is offered only where neither SGLang's Marlin nor the moe agent's
        # kernel is available (never on the judge image): both beat it everywhere, and SGLang keeps BF16 there
        cands = [c for c in ("marlin", "agent") if (c != "marlin" or st["bank"] is not None)
                 and (c != "agent" or st["ma_ok"])]
        for moe in cands:
            if moe == "marlin" and st["bank"] is None:
                continue
            if moe == "agent" and not st["ma_ok"]:
                continue
            st["moe_pick"][T] = moe
            for v2 in (True, False):
                st["gluev2"][T] = v2
                try:
                    t = _graph_time(lambda: _after_attention(data, st, o_in, None, T), flush)
                except Exception as ex:  # noqa: BLE001
                    if log:
                        msg.append(f"{moe[:3]}{2 if v2 else 1}:X{repr(ex)[:40]}")
                    continue
                if log:
                    msg.append(f"{moe[:3]}{2 if v2 else 1}:{t:.0f}")
                    if nloc < 0:
                        _o, _s, _ids = _after_attention(data, st, o_in, None, T)
                        loc = _ids[(_ids >= 0) & (_ids < LOCAL_EXPERTS)]
                        nloc, ndis = int(loc.numel()), int(torch.unique(loc).numel())
                if t < times.get(moe, 1e30):
                    times[moe] = t
                    if moe == DEFAULT_MOE:
                        dfltv2 = v2
                if t < bestt:
                    best, bestt, bestv2 = moe, t, v2
        if (DEFAULT_MOE in times and best is not None and best != DEFAULT_MOE
                and times[best] > TUNE_MARGIN * times[DEFAULT_MOE]):
            best, bestv2 = DEFAULT_MOE, dfltv2
        st["moe_pick"][T] = best if best is not None else DEFAULT_MOE
        st["gluev2"][T] = bestv2
        if _pdl(T) and st.get("bfp") is not None:   # sk2 PDL depth: weight stages issued before griddepcontrol.wait,
            ptimes = {}                               # timed on the whole post-attention chain (where the overlap is)
            for n in PDL_N_SWEEP:
                _PDL_N_BY_T[T] = n
                try:
                    ptimes[n] = _graph_time(lambda: _after_attention(data, st, o_in, None, T), flush)
                except Exception:  # noqa: BLE001
                    ptimes[n] = 1e30
            nbest = min(ptimes, key=lambda n: ptimes[n])
            _PDL_N_BY_T[T] = nbest
            if log:
                msg.append("pdl[" + " ".join(f"{n}:{t:.0f}" for n, t in ptimes.items()) + f"]->{nbest}")
        modes = (["split"] + (["sk2f"] if st.get("fuse_route_ok") else []) + (["moe"] if st.get("route_moe_ok") else [])
                 + (["moe_nb"] if st.get("route_moe_nb_ok") else []))
        if len(modes) > 1 and T <= 32:   # where the routing runs: the _route_split kernel, CTA t of the fused pooler +
            rtimes = {}                   # router launch after its grid barrier, or every CTA of the MoE kernel
            for mode in modes:
                st.setdefault("route_mode", {})[T] = mode
                try:
                    rtimes[mode] = _graph_time(lambda: _after_attention(data, st, o_in, None, T), flush)
                except Exception as ex:  # noqa: BLE001
                    rtimes[mode] = 1e30
                    if log:
                        msg.append(f"rmodeX:{mode}:{repr(ex)[:80]}")
            st["route_mode"][T] = min(rtimes, key=lambda m_: rtimes[m_])
            if log:
                msg.append("rmode[" + " ".join(f"{m_}:{t:.0f}" for m_, t in rtimes.items()) + f"]->{st['route_mode'][T]}")
        if st.get("fuse_bnd1_ok") and T <= BND1_MAX_T:   # the boundary-1 prologue vs the _boundary1 launch, on the chain
            btimes = {}
            for flag in (False, True):
                st.setdefault("bnd1_on", {})[T] = flag
                try:
                    btimes[flag] = _graph_time(lambda: _after_attention(data, st, o_in, None, T), flush)
                except Exception as ex:  # noqa: BLE001
                    btimes[flag] = 1e30
                    if log:
                        msg.append(f"bnd1X:{repr(ex)[:80]}")
            st["bnd1_on"][T] = btimes[True] < btimes[False]
            if log:
                msg.append(f"bnd1[{btimes[False]:.0f}/{btimes[True]:.0f}]->{int(st['bnd1_on'][T])}")
        tb = 1e30
        if log:
            print(f"T{T}/{nloc}r{ndis}e " + " ".join(msg) + (f" pb:{tb:.0f}" if tb < 1e9 else ""), flush=True)
        pb_ok[T] = tb < bestt
    pbmax = 0
    for T in (1, 2, 4, 8, 16, 32):
        if pb_ok.get(T):
            pbmax = T
        else:
            break
    st["bias"].copy_(bias0)
    st["pbmax"] = pbmax if FORCE_PBMAX is None else FORCE_PBMAX
    if log:
        print("tune: pbmax", pbmax, "moe", st["moe_pick"], "gluev2", st["gluev2"], flush=True)
    del flush


def prepare(weights):
    dev = weights["qkv_gate"].device
    router = weights["router"].float()
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    st = {}
    st["qkv"] = weights["qkv_gate"].contiguous()
    st["o"] = weights["o"].contiguous()
    st["pr"] = torch.cat([weights["pooler"], hi, lo], 0).contiguous()
    st["unpool"] = weights["unpooler"].contiguous()
    st["unpool_slice"] = weights["unpooler"][:1024].contiguous()
    for name, key in _T_COPY.items():   # K-major copies for the "torchT" cuBLAS candidate at T > 32 (~200 MB)
        st[key] = st[{"qkv": "qkv", "o": "o", "pr": "pr", "un": "unpool"}[name]].t().contiguous()
    # Weight storage, to the letter of TASK.md: every BF16 checkpoint weight stays BF16. The norms are
    # applied as (1 + w), which my kernels now form in FP32 at use; the one tensor SGLang itself
    # precomputes (the post-expert effective weight) is stored exactly as SGLang stores it, in BF16.
    st["gate1"] = weights["attn_residual_gate"].contiguous()
    st["gate2"] = weights["ffn_residual_gate"].contiguous()
    st["ab1"] = _gate_coeffs(st["gate1"])   # (alpha, beta) fp32 for the fused boundary tails (FUSE_BOUNDARY)
    st["ab2"] = _gate_coeffs(st["gate2"])
    st["g_preffn"] = weights["pre_ffn_norm"].contiguous()
    st["g_preexp"] = weights["pre_expert_norm"].contiguous()
    st["g_postexp"] = weights["post_expert_norm"].contiguous()   # BF16 as given; _pexp_norm_sum forms 1 + w
    st["g_postffn"] = weights["post_ffn_norm"].contiguous()
    st["g_next"] = weights["next_pre_attn_norm"].contiguous()
    st["bias"] = weights["router_bias"].float().contiguous()
    st["invf"] = (1.0 / (500000.0 ** (torch.arange(0, 64, 2, device=dev).float() / 64))).contiguous()
    import threading
    errs = []

    def _build(fn):
        try:
            fn()
        except Exception as ex:  # noqa: BLE001
            errs.append(ex)

    sfx = _moe_sfx(weights)
    builds = [_att7] + [(lambda d=d: moe__ext(d)) for d in _moe_builds(sfx)]
    ths = [threading.Thread(target=_build, args=(f,)) for f in builds]  # ~85 s of nvcc, in parallel
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    if errs:
        raise errs[0]
    _ma_prepare(weights, st)
    _bfp_prepare(weights, st)
    for selftest in (_route_selftest, _boundary_selftest, _bnd1_selftest, _norm_selftest):   # the fused paths, each enabled only when it matches
        try:
            selftest(st)
        except Exception as ex:  # noqa: BLE001
            print(f"{selftest.__name__} failed: {repr(ex)[:300]}", flush=True)
    st["nsm"] = torch.cuda.get_device_properties(dev).multi_processor_count
    st["fa3"] = None
    st["bank"] = None
    st["att_ok"] = True
    st["ma_ok"] = True
    st["pbmax"] = 2
    st["moe_pick"] = {}
    try:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache
        st["fa3"] = flash_attn_with_kvcache
        st["bank"] = _marlin_bank(weights)
    except Exception as ex:  # noqa: BLE001  (local GPUs without sgl_kernel: fall back to our own kernels)
        print("fallback (no sgl_kernel/sglang):", repr(ex)[:200])
    try:
        _tune(st, log=_TUNE_LOG)
    except Exception as ex:  # noqa: BLE001
        print("tune failed:", repr(ex)[:200], flush=True)
        st["pbmax"] = 2
        st["moe_pick"] = {}
        st["gluev2"] = {}
        st["marlin_cfg"] = {}
        st["mm_pick"] = {}
    for k, v in PIN_PICK.items():      # in-chain winners the isolated tuner sometimes misses (no PDL in its timing)
        if isinstance(v, tuple):
            times = st.get("mm_times", {}).get(k, {})
            v = min(v, key=lambda p: times.get(p, 1e30))
        st.setdefault("mm_pick", {})[k] = v
    if st.get("fuse_bnd1_ok"):   # the boundary-1 prologue lives in the fused pooler + router launch: pin it at those T
        for T_ in TUNED_T:
            if T_ <= BND1_MAX_T:
                st["mm_pick"][(T_, "pr")] = "bfpf"
                st["mm_pick"][(T_, "o")] = "bfp"
    torch.cuda.synchronize()
    if st.get("att_ok"):
        _attn_selftest(st)     # every attention plan family once, on synthetic inputs
    _check_guards()            # after the tuner (every moe build) and the attention self-test
    return st


def _attn_cfg(B, L, sliding):
    BM = 16 if L == 1 else 64
    if sliding:
        nsplit = max(1, min(32, 264 // B))
    else:
        nsplit = max(1, min(132, 528 // B))
    return BM, nsplit


# The attention module's extension, built once for this layer: its source with the layer's measured flags (the
# standalone entry's defaults are QDIRECT=1, PK2=1, W0ENABLE=0; this build is their "v298w1").
ATT_BUILD_FLAGS = ()   # A/B 7d23a37c vs c8b5aab9: the standalone defaults (QDIRECT=1 PK2=1 W0ENABLE=0) measured 0.3-3 us
                       # faster per attention launch on 10 of 12 shapes than the layer's old build (0/0/1)


def _att7():
    global att__MOD
    if att__MOD is None:
        att__MOD = att__build("w1", ATT_BUILD_FLAGS)
    return att__MOD


ATT_HAS_PDL = True   # does the embedded attention launcher take pdl= (their v63+)?


def _decode_attention7(data, state, qkv):
    """The attention module's own launcher (att__launch): plan, workspace sizes and the coop / w0 bits always match
    the kernel version composed in. This layer's config already carries their keys (heads, head_dim, page_size,
    window, rope_theta, scales)."""
    cfg = data["config"]
    pdl = {"pdl": _pdl(cfg["batch"] * cfg["q_len"])} if ATT_HAS_PDL else {}   # their launcher's PDL switch
    out = att__launch({"config": cfg, "qkv": qkv, "seq_lens": data["seq_lens"], "cache_loc": data["cache_loc"],
                    "k_cache": data["k_cache"], "v_cache": data["v_cache"], "page_table": data["page_table"],
                    "k_scale": data["k_scale"], "v_scale": data["v_scale"]}, **pdl)
    return out.view(cfg["batch"] * cfg["q_len"], 1024)


# Guards around every component buffer this glue allocates or caches (POD.md item 6 follow-up): the moe
# scratch, the attention workspace and its cluster buffers. Each is a view into a larger allocation whose
# tail is SCRATCH_GUARD sentinel words; prepare() checks every tail after the tuner and an attention self-test
# have run each path, so an out-of-bounds write fails loudly there instead of corrupting a neighbour.
_GUARDS = []
_GUARDED_IDS = set()


def _guarded(t):
    nbytes = t.numel() * t.element_size()
    words = (nbytes + 3) // 4
    full = torch.full((words + SCRATCH_GUARD,), SCRATCH_SENTINEL, dtype=torch.int32, device=t.device)
    full.view(torch.uint8)[:nbytes].copy_(t.reshape(-1).view(torch.uint8))
    v = full.view(torch.uint8)[:nbytes].view(t.dtype).view(t.shape)
    _GUARDS.append((full, words))
    _GUARDED_IDS.add(id(v))
    return v


def _check_guards():
    torch.cuda.synchronize()
    for full, words in _GUARDS:
        bad = int((full[words:] != SCRATCH_SENTINEL).sum())
        if bad:
            raise RuntimeError(f"a component kernel wrote {bad} words past a {words * 4}-byte buffer")


# The attention module's allocators are rebound (its launcher looks them up by name at every call) so that whatever
# they newly allocate is guarded and cached back into their own dictionaries.
_att_workspace_raw, _att_cluster_bufs_raw = att__workspace, att__cluster_bufs


def att__workspace(n_rows, device):          # their allocator, then guard whatever it newly allocated
    ws = _att_workspace_raw(n_rows, device)
    if id(ws[0]) not in _GUARDED_IDS:
        ws = tuple(_guarded(t) for t in ws)
        att__WS[str(device)] = ws
    return ws


def att__cluster_bufs(n_ctas, L, device):
    cb = _att_cluster_bufs_raw(n_ctas, L, device)
    if id(cb[0]) not in _GUARDED_IDS:
        cb = tuple(_guarded(t) for t in cb)
        att__CL[str(device)] = cb
    return cb


def _attn_selftest(st):
    """Run the attention kernel once per plan family on synthetic inputs (problem constants, no data), so its
    lazily allocated buffers exist and are guarded before any timed call, and the guards see every path."""
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(4242)
    nq = st["qkv"].shape[0]
    for window, L, B, ctx, P in ((2048, 1, 1, 3000, 1), (2048, 1, 16, 2500, 1), (2048, 1, 32, 2500, 1),
                                 (0, 1, 1, 3000, 1), (0, 1, 16, 1500, 1), (0, 1, 33, 1000, 1),
                                 (0, 8, 1, 3000, 1), (0, 8, 16, 1500, 1), (0, 8, 32, 800, 1),
                                 (2048, 1, 4, 3000, 64), (0, 8, 4, 3000, 64)):
        cfg = dict(batch=B, q_len=L, num_q_heads=16, num_kv_heads=2, head_dim=64, page_size=P, window=window,
                   rope_theta=500000.0 if window else 0.0, qk_scale_factor=5.4730064863875, sm_scale=0.125)
        if not att__supported(cfg):
            continue
        seq = torch.full((B,), ctx, dtype=torch.int32, device=dev)
        per = (ctx + L + P - 1) // P
        pt = torch.arange(B * per, dtype=torch.int32, device=dev).view(B, per)
        pos = (seq[:, None].long() + torch.arange(L, device=dev)).view(B, L)
        loc = (torch.gather(pt.long(), 1, pos // P) * P + pos % P).reshape(-1)
        kc = torch.zeros((B * per, P, 2, 64), dtype=torch.float8_e4m3fn, device=dev)
        vc = torch.zeros_like(kc)
        qkv = (torch.randn(B * L, nq, generator=g, device=dev) * 0.1).to(torch.bfloat16)
        one = torch.ones((), dtype=torch.float32, device=dev)
        _decode_attention7({"config": cfg, "seq_lens": seq, "cache_loc": loc, "k_cache": kc, "v_cache": vc,
                            "page_table": pt, "k_scale": one, "v_scale": one}, st, qkv)

def _att_fits7(data, cfg):
    """The attention entry's launcher host-checks these sizes (32-bit index math, OKC_OFF32) and raises otherwise;
    mirrored here so such shapes take SGLang's FA3 instead (fuzz: batch 64 x 150k context exceeds the caches' 4 GB)."""
    kc, vc = data["k_cache"], data["v_cache"]
    rows = kc.numel() // max(1, cfg["num_kv_heads"] * cfg["head_dim"])
    return (kc.numel() < (1 << 32) and vc.numel() < (1 << 32) and rows < (1 << 31)
            and data["page_table"].numel() < (1 << 31)
            and cfg["batch"] * cfg["q_len"] * 2304 < (1 << 31)
            and cfg["batch"] * (cfg["num_kv_heads"] // 2) <= 4096)


def _kernel_fn_fa3(data, state):
    st2 = dict(state)
    st2["att_ok"] = False
    return kernel_fn(data, st2)


def kernel_fn(data, state):
    cfg = data["config"]
    B, L, T = cfg["batch"], cfg["q_len"], cfg["tokens"]
    # q_len 1 only: on the kv_scale = 1/64 test their q_len 8 path's query scaling rounds coarser than FA3
    # (conformance test 9: hidden/stream 1.41x SGLang's error), so FA3 keeps every q_len 8 shape.
    # q_len 8: their kernel on full layers, and on sliding only for a single request, where fusing QK-norm, RoPE,
    # the cache write and the norm-gate into one launch beats FA3 + my two glue kernels by 5.6% in-layer;
    # at B >= 16 sliding FA3 still matches or wins (16x8 -1.2%, 32x8 +0.8%), below my 4% bar for q_len 8.
    agent_attn = (state["att_ok"] and att__supported(cfg) and _att_fits7(data, cfg)
                 )
    if agent_attn:
        # fused decode attention (QK-norm, RoPE, cache write, split-KV, merge, rmsnorm * sigmoid(gate))
        qpick = state.get("mm_pick", {}).get((T, "qkv"), True)
        if qpick == "bfp" and T <= BFP_MAX_T and state.get("bfp") is not None:
            qkv = _bfp_gemm(state, "qkv", data["attn_input"], T, state["qkv"].shape[0], HIDDEN)
        elif qpick is True or qpick == "bfp":
            qkv = torch.mm(data["attn_input"], state["qkv"].t())
        else:
            Pq, Sq = dense(data["attn_input"], state["qkv"], T)
            qkv = Pq[0] if Sq == 1 else Pq.sum(0)
            qkv = qkv.to(torch.bfloat16)
        try:
            o_in = _decode_attention7(data, state, qkv)
        except RuntimeError:
            # their launcher's host-side size checks run before any launch: nothing half-captured, take FA3
            return _kernel_fn_fa3(data, state)
        return _after_attention(data, state, o_in, None, T)
    dev = data["attn_input"].device
    x = data["attn_input"]
    sliding = cfg["window"] > 0
    use_fa3 = state["fa3"] is not None
    # 1. qkv projection, QK-norm, RoPE, FP8 cache write
    P, S = dense(x, state["qkv"], T)
    q = torch.empty((T, 1024), device=dev, dtype=torch.bfloat16)
    q8 = torch.empty((T, 1024), device=dev, dtype=torch.uint8)
    gate = torch.empty((T, 1024), device=dev, dtype=torch.float32)
    kc = data["k_cache"].view(torch.uint8)
    vc = data["v_cache"].view(torch.uint8)
    _qk_prep[(T,)](P, S, T, data["positions"], data["cache_loc"], state["invf"], kc, vc,
                   data["k_scale"], data["v_scale"], q, q8, gate, cfg["qk_scale_factor"],
                   ROPE=bool(cfg["rope_theta"]), num_warps=4, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    # 2. attention
    if use_fa3:
        descale = {}
        if cfg["kv_scale"] != 1.0:
            descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(B, 2),
                           v_descale=data["v_scale"].reshape(1, 1).expand(B, 2))
        window = cfg["window"]
        out = state["fa3"](q=q8.view(torch.float8_e4m3fn).view(T, 16, 64), k_cache=data["k_cache"],
                           v_cache=data["v_cache"], page_table=data["page_table"],
                           cache_seqlens=data["cache_seqlens"], cu_seqlens_q=data["cu_seqlens_q"],
                           max_seqlen_q=L, softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
                           window_size=(window - 1, 0) if window else (-1, -1),
                           scheduler_metadata=data["scheduler_metadata"], **descale)
        if isinstance(out, tuple):
            out = out[0]
        o_in = out.reshape(T, 1024)
    else:
        BM, NS = _attn_cfg(B, L, sliding)
        acc = torch.empty((B * 2 * NS * BM, 64), device=dev, dtype=torch.float32)
        ml = torch.empty((B * 2 * NS * BM, 2), device=dev, dtype=torch.float32)
        pt = data["page_table"]
        _attn_split[(B, NS)](q, kc, vc, pt, data["seq_lens"], data["k_scale"], acc, ml,
                             pt.stride(0), cfg["page_size"], cfg["window"], cfg["sm_scale"],
                             L=L, NSPLIT=NS, BM=BM, BN=64, SLIDING=sliding, num_warps=4, num_stages=3)
        o_in = torch.empty((T, 1024), device=dev, dtype=torch.bfloat16)
        _attn_combine[(T, 16)](acc, ml, gate, data["v_scale"], o_in, L=L, NSPLIT=NS, BM=BM,
                               SB=min(64, triton.next_power_of_2(NS)), NORMGATE=False, num_warps=4)
    return _after_attention(data, state, o_in, gate, T)


def _after_attention(data, state, o_in, gate, T):
    """o projection onwards: residual boundary, router, experts, unpooler, final boundary.
    gate is None when o_in already carries rmsnorm(o) * sigmoid(gate). state["gluev2"] picks between the
    one-program-per-token boundary kernels and the parallel (atomic-sum) ones; the tuner chooses per T."""
    dev = o_in.device
    f32 = dict(device=dev, dtype=torch.float32)
    gv = state.get("gluev2", {})
    v2 = gv.get(T, gv.get(_bucket(T), False)) if FORCE_GLUEV2 is None else FORCE_GLUEV2
    s1 = torch.empty((T, HIDDEN), **f32)
    h = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    bnd1 = None   # set when _boundary1 is deferred into the fused pooler + router launch's prologue
    if v2:
        # o projection with no K split (each program owns whole rows) + the boundary's four per-token sums
        a_out = torch.empty((T, HIDDEN), **f32)
        sums = torch.zeros((T, 4), **f32)
        BN, BT, BK = 64, (16 if T <= 16 else 32), 128
        _gemm_o[(HIDDEN // BN, triton.cdiv(T, BT))](
            o_in, gate, state["o"], a_out, sums, data["stream"], state["gate1"], state["gate1"], T, HIDDEN,
            K=1024, BN=BN, BT=BT, BK=BK, NORMGATE=gate is not None, num_warps=4, num_stages=3)
        BC1 = 2048
        _bnd1_apply[(T, HIDDEN // BC1)](a_out, sums, data["stream"], state["gate1"], state["gate1"], state["g_preffn"],
                                        s1, h, N=HIDDEN, BC=BC1, num_warps=8)
    elif (gate is None and _bfp_tail_ok(state, T) and state.get("mm_pick", {}).get((T, "o")) == "bfp"
          and data["stream"].dtype == torch.float32 and data["stream"].is_contiguous()):
        # o projection whose last CTA also computes the post-attention boundary (s1, h): no _boundary1 launch
        _bfp_pdl(T)
        globals()["bfp_run_sk2_tail"](o_in, "o", T, HIDDEN, 1024, state["bfp"],
                         (state["ab1"][0], state["ab1"][1], data["stream"], None, state["g_preffn"], s1, h))
    else:
        if gate is None:
            P, S = dense(o_in, state["o"], T, state, "o")
        else:
            P, S = dense_ng(o_in, gate, state["o"], T, state)
        if (gate is None and S == 1 and P.dtype == torch.bfloat16 and _bnd1_fuse_ok(state, T)
                and data["stream"].dtype == torch.float32 and data["stream"].is_contiguous()):
            # deferred: the fused pooler + router launch computes h (its own x) and s1 in its prologue (no launch)
            bnd1 = (P.view(T, HIDDEN), data["stream"], state["ab1"][0], state["ab1"][1], state["g_preffn"], s1)
        else:
            _boundary1[(T,)](P, S, T, data["stream"], state["gate1"], state["gate1"], state["g_preffn"], s1, h,
                             num_warps=16, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    # 4. pooler + router (BF16 hi + lo rows, FP32 accumulation), routing, pre-expert norm
    ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    lids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    wts = torch.empty((T, 8), **f32)
    moe = state["moe_pick"].get(T, state["moe_pick"].get(_bucket(T), "agent"))
    if moe == "marlin" and state["bank"] is None:
        moe = "agent"
    if moe != "marlin" and not state["ma_ok"]:
        moe = "marlin"
    if moe == "marlin" and state["bank"] is None:
        moe = "agent"
    use_marlin = moe == "marlin"
    use_ma = moe == "agent"
    if not (use_marlin or use_ma):
        raise RuntimeError("no expert backend: neither the moe agent's kernel nor SGLang's Marlin is available")
    xm = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
    mmp = state.get("mm_pick", {})
    ppick = mmp.get((T, "pr"), mmp.get((_bucket(T), "pr"), False))
    use_fused_pr = ppick == "bfpf" and T <= 32 and state.get("bfp") is not None
    if bnd1 is not None and not use_fused_pr:   # the deferred boundary has no fused launch to live in: run it
        _boundary1[(T,)](bnd1[0].view(1, T, HIDDEN), 1, T, bnd1[1], state["gate1"], state["gate1"], state["g_preffn"],
                         s1, h, num_warps=16, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
        bnd1 = None
    if use_fused_pr:
        # bf16_proj's fused pooler + router: one launch, both read h once (bf16 [T, 4096], fp32 [T, 256]); with
        # FUSE_ROUTE its last CTA also does the routing and the pre-expert norm (no _route_split launch); with
        # bnd1 its prologue computes h (= its x) and s1 from the o-projection output (no _boundary1 launch)
        _bfp_pdl(T)
        route = _fused_route(state, data, ids, lids, wts, xm) if (use_marlin or use_ma) else None
        if route is None and use_ma and _route_in_moe_ok(state, T):
            # the launch also normalizes xm (grid barrier, every CTA its own columns); the routing is the MoE
            # kernel's, from the router logits: no _route_split launch
            P, S = bfp_fused_pooler_router(h, state["bfp"], None, bnd1, (state["g_preexp"], xm)), -3
        elif route is None and use_ma and _route_mode(state, T) == "moe_nb":
            # no barrier: the launch writes pooled * (1 + w) and the per-token partial sums; the MoE kernel forms the
            # RMS scale in its GEMM1 epilogue and routes from the logits (pooled IS the experts' input)
            P, S = bfp_fused_pooler_router(h, state["bfp"], None, bnd1, None, state["g_preexp"]), -4
        elif route is None:   # (the standalone proj module's launch takes no route argument)
            if bnd1 is not None:
                P, S = bfp_fused_pooler_router(h, state["bfp"], None, bnd1), -1
            else:
                P, S = bfp_fused_pooler_router(h, state["bfp"]), -1
        else:
            P, S = bfp_fused_pooler_router(h, state["bfp"], route, bnd1), -2
    elif ppick in ("bfp", "bfq") and state.get("bfp") is not None:   # ("bfq": the proj module's m >= 33 routes)
        pooled = _bfp_gemm(state, "pool", h, T, D_EXPERT, HIDDEN)          # bf16 [T, 4096]
        logits = _bfp_gemm(state, "router", h, T, N_EXPERTS, HIDDEN)       # fp32 [T, 256], hi+lo already summed
        P, S = (pooled, logits), -1
    else:
        P, S = dense(h, state["pr"], T, state, "pr")
    if S == -2:  # the routing and xm were produced inside the fused pooler + router launch
        pass
    elif S == -3:  # xm was normalized inside the fused launch; the MoE kernel routes from the logits (below)
        pooled, logits = P
    elif S == -4:  # pooled * (1 + w) + partial sums from the fused launch; the MoE kernel scales and routes (below)
        pooled, logits = P
    elif S == -1:  # pooler and router came from the bf16_proj kernels as two separate tensors
        pooled, logits = P
        _route_split[(T,)](pooled, logits, state["bias"], state["g_preexp"], ids, wts, xm,
                           data["local_start"], lids, PERMUTED=not (use_marlin or use_ma), num_warps=8, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    elif v2:
        pool = torch.empty((T, D_EXPERT), **f32)
        psum = torch.zeros((T,), **f32)
        BCR = 1024
        _route_a[(T, D_EXPERT // BCR)](P, S, T, state["bias"], ids, wts, data["local_start"], lids, pool, psum,
                                       BC=BCR, num_warps=8)
        _route_b[(T, D_EXPERT // BCR)](pool, psum, state["g_preexp"], xm, BC=BCR,
                                       PERMUTED=not (use_marlin or use_ma), num_warps=8)
    else:
        _route[(T,)](P, S, T, state["bias"], state["g_preexp"], ids, wts, xm, data["local_start"], lids,
                     PERMUTED=not (use_marlin or use_ma), num_warps=8, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    # 5. experts
    if use_ma and S == -3:
        # routing in the MoE kernel from the router logits (every CTA; CTA 0 writes the chosen global ids into ids)
        m = _ma_run_logits(state, xm, logits, state["bias"], data["local_start"], ids, T)
    elif use_ma and S == -4:
        # as above, plus the per-token RMS scale from the fused launch's partial sums (token-major [T, 128] fp32)
        m = _ma_run_logits(state, pooled, logits, state["bias"], data["local_start"], ids, T,
                           partials=state["bfp"]["fused_part2"][:T * 128].view(T, 128))
    elif use_ma:
        m = _ma_run(state, xm, lids, wts, T)   # lids: local expert index, -1 = another GPU's expert
    elif use_marlin:
        m = _marlin_experts(xm, wts, lids, state)
    # 6. unpooler + post-FFN boundary + next layer's norm
    wu = state["unpool_slice"] if T <= 8 else state["unpool"]
    nu = wu.shape[0]
    s2 = torch.empty((T, HIDDEN), **f32)
    out = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    if not v2 and _bfp_tail_ok(state, T) and state.get("mm_pick", {}).get((T, "un")) == "bfp":
        # unpooler slice whose last CTA also computes the post-FFN boundary (s2, out): no _boundary2 launch
        _bfp_pdl(T)
        globals()["bfp_run_sk2_tail"](m, "unpooler", T, nu, D_EXPERT, state["bfp"],
                         (state["ab2"][0], state["ab2"][1], s1, state["g_postffn"], state["g_next"], s2, out))
        return out, s2, ids
    P, S = dense(m, wu, T, state, "un")
    if v2:
        fbuf = torch.zeros((T, HIDDEN), **f32)
        sums2 = torch.zeros((T, 4), **f32)
        BC2 = 1024
        _bnd2_sums[(T, HIDDEN // BC2)](P, S, T, nu, s1, state["gate2"], state["gate2"], state["g_postffn"], fbuf, sums2,
                                       BC=BC2, num_warps=8)
        _bnd2_apply[(T, HIDDEN // BC2)](fbuf, sums2, s1, state["gate2"], state["gate2"], state["g_next"], s2, out,
                                        N=HIDDEN, BC=BC2, num_warps=8)
    else:
        _boundary2[(T,)](P, S, T, nu, s1, state["gate2"], state["gate2"], state["g_postffn"], state["g_next"],
                         s2, out, num_warps=16, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    return out, s2, ids
