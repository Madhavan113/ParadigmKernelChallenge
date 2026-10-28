# okc: problem=muse_moe_w4a16
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


class Weights(TypedDict):
    w13: torch.Tensor
    w13_sf: torch.Tensor
    w13_gs: torch.Tensor
    w2: torch.Tensor
    w2_sf: torch.Tensor
    w2_gs: torch.Tensor
    post_norm_w: torch.Tensor


class Config(TypedDict):
    bs: int
    d_hidden: int
    d_expert: int
    n_experts: int
    local_start: int
    local_count: int
    top_k: int
    sharding: str


class Inputs(TypedDict):
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    config: Config


CUDA_SRC = r'''
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
#ifndef OKC_WGSREG
#define OKC_WGSREG 0   // WGS: consumer registers per thread via setmaxnreg: the producers form a full warpgroup (warps
                       // 16 weights, 17 x, 18-19 idle) that releases its registers (dec 24) so the 4 consumer
                       // warpgroups can grow (inc OKC_WGSREG): per SM quadrant 4 x OKC_WGSREG + 24 <= 512 -> up to 120
#endif
#ifndef OKC_WGSREGP
#define OKC_WGSREGP 32  // registers kept by the producer warpgroup (its TMA loops use R0..R15). The setmaxnreg pool holds
                        // only what the CTA released: 4 x (96 - OKC_WGSREGP) must cover 16 x (OKC_WGSREG - 96), i.e.
                        // OKC_WGSREG <= 112 at 32 (120 with 24 released only 288 of the 384 needed: TRY_ALLOC spins forever)
#endif
#if OKC_WGSREG
#undef OKC_WGSPR
#define OKC_WGSPR 0    // (the x producer must not call wgs_ranges after its dealloc: thread 0 computes the ranges)
static_assert(16 * (OKC_WGSREG - 96) <= 4 * (96 - OKC_WGSREGP) && OKC_WGSREG % 8 == 0 && OKC_WGSREGP % 8 == 0 && OKC_WGSREGP >= 24,
              "OKC_WGSREG: the consumers can only take what the producer warpgroup released (launch allocation 96)");
#endif
#ifndef OKC_WGSFOLD
#define OKC_WGSFOLD 0  // WGS: no producer warps: lane 0 of warp 0 issues the weight and x TMAs between its own k16
                       // steps (polls, never blocking except for the stage about to be consumed), so the CTA is
                       // exactly 4 warpgroups (512 threads) and ptxas can allocate 128 registers per thread
#endif
#if OKC_WGSFOLD
#undef OKC_WGSPR
#define OKC_WGSPR 0    // (no x producer warp: thread 0 computes the stage ranges after the gather)
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
#ifndef OKC_WSNOC
#define OKC_WSNOC 0  // dev: tiny WS builds without the dequant / MMA loop: the ring, x loads, handoffs and epilogues
                     // run as usual over zero partials (the weights-only stream time of the same expert set)
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
constexpr int NTL = NT + (OKC_WGS ? (OKC_WGSFOLD ? 0 : OKC_WGSREG ? 128 : 64) : OKC_WS ? 32 : 0);  // + helper warp (WS) | weights and x producer warps (WGS)  // launched: NT compute threads (+ helper warp | WGS producer warp + epilogue warpgroup)
constexpr int NRB = OKC_WS ? OKC_WSNR : 2;   // cross-warp partial-sum buffers
#ifndef OKC_WSH0
#define OKC_WSH0 0  // WS tiny builds: the helper warp is HARDWARE warp 0 (the oldest warp of the CTA, which a greedy-then-
                    // oldest scheduler prefers over the ALU-saturating compute warps; as the youngest warp its first
                    // epilogue took ~5 us and its credits lagged the hand-offs by 1-5 us: set V/W stamps); the logical
                    // thread index TIDX is rotated so that hardware warps 1.. are the compute warps 0..
#endif
#if OKC_WSH0 && !OKC_WS
#error "OKC_WSH0 needs the WS helper warp"
#endif
#if OKC_WSH0
#define TIDX ((int)((threadIdx.x + NT) % NTL))
#else
#define TIDX ((int)threadIdx.x)
#endif
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
#if OKC_WSHP
  float hgs[2][32];                    // WSHP: per-expert gsA / gsB (all local experts)
  alignas(16) __nv_bfloat16 hnw[HD];   // WSHP: post-norm weights (cp.async 16 B copies)
#endif
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
#define DBGP(k) do { if (OKC_DEBUG && TIDX == 0 && p.dbg) p.dbg[(size_t)gridDim.x * 13 + (size_t)blockIdx.x * 4 + (k)] = gtimer(); } while (0)
#define DBG(k) do { if (OKC_DEBUG && TIDX == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)
// tiny (WS) builds: the same slots as DBGP (prologue) and the WGS producer-stall counters, stamped by whichever single
// thread runs the site (the caller guards): DBGPX(0) first weight unit issued, (1) first phase-B unit landed, (2) first
// phase-B gate passed, (3) first phase-B unit issued; DBGT(0) helper epilogues done, (1) grid barrier arrival,
// (3) last phase-A unit landed
#define DBGPX(k) do { if (OKC_DEBUG && p.dbg) p.dbg[(size_t)gridDim.x * 13 + (size_t)blockIdx.x * 4 + (k)] = gtimer(); } while (0)
#define DBGT(k) do { if (OKC_DEBUG && !OKC_WGS && p.dbg) p.dbg[(size_t)gridDim.x * 8 + (size_t)blockIdx.x * 4 + (k)] = gtimer(); } while (0)
// DBGU(k), k < 8: tiny builds, slots after the WGS per-warpgroup region: (0) helper: first phase-A credit, (1) last credit,
// (2) consumer: second phase-B job gate passed, (3) helper: first phase-A epilogue done; (4) / (5) consumer thread 0:
// ns waited on the helper (partial buffer free) in phase A / B, (6) / (7) ns waited on the ring (unit landing) in A / B
#define DBGU_PTR(k) (p.dbg + (size_t)gridDim.x * 17 + (size_t)blockIdx.x * 8 + (k))
#define DBGU(k) do { if (OKC_DEBUG && p.dbg) *DBGU_PTR(k) = gtimer(); } while (0)
// DBGV(k), k < 8: helper warp accounting (tiny builds): (0) ns in __threadfence before the doneA credits, (1) gtimer after
// the phase-A epilogue of unit q = 3, (2) consumer thread 0: gtimer after the hand-off (CTA barrier) of unit q = 3,
// (3) / (6) helper ns waiting for partials (wsrf) in phase A / B, (4) / (7) helper ns in epilogues A / B, (5) helper ready
#define DBGV_PTR(k) (p.dbg + (size_t)gridDim.x * 25 + (size_t)blockIdx.x * 8 + (k))
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
  const int lane = TIDX & 31;
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
  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);
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
  const int tid = TIDX;
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
  for (int i = TIDX; i < UNIT_BYTES / 16; i += NT)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(slot + i * 16)), "l"(src + i * 16));
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(smem_u32(bar)) : "memory");
#endif
}

template <bool PB>
__device__ __forceinline__ void ring_issue_all(const Params& p, const Smem& S, int v, unsigned char* slot, uint64_t* bar) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && OKC_USE_TMA
  if (TIDX == 0) ring_issue<PB>(p, S, v, slot, bar);
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
  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);
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
  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);
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
  if (TIDX == 0) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
                 ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");
  }
#else
  for (int i = TIDX; i < UNIT_BYTES / 16; i += NT)
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
  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);
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
    if (TIDX == 0) S.preissued = 0;
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
  if (TIDX == NT - 32) {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    for (int i = 0; i < n; ++i) { ring_issue<false>(p, S, v0 + i, ring + i * UNIT_BYTES, S.bars + i); S.uslot[i] = v0 + i; }
    S.preissued = n > 0 ? n : 0;
  }
#else
  const int n = min(OKC_NS, v1 - v0);
  for (int i = 0; i < n; ++i) ring_issue_all<false>(p, S, v0 + i, ring + i * UNIT_BYTES, S.bars + i);
  if (TIDX == 0) S.preissued = n > 0 ? n : 0;
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
#ifndef OKC_WSHP
#define OKC_WSHP 0  // WS tiny builds: the helper warp copies its epilogue constants (per-job gsA / gsB, the 8 KB post-norm
                    // weights) into smem while the first units are in flight; measured without it the helper's first
                    // epilogue waits ~5 us on cold global loads under a saturated memory system and the backlog delays
                    // the doneA credits (the phase-B gate) by 2-4 us (set V stamps)
#endif
#ifndef OKC_WSSTR
#define OKC_WSSTR 0  // WS tiny builds: with up to this many skinny jobs, a CTA's units of each phase are the strided set
                     // {c + i G} of the job-major unit space (its share of EVERY job, in job order; phase-A credits per job
                     // as soon as the CTA's units of that job are done) instead of the contiguous range
                     // [nS c / G, nS (c + 1) / G): the B units of all jobs but the last stream and compute while the last
                     // job's phase A is still ending (set Q: at bs1 the 4 resident B units cost 5.4 us of compute after a
                     // gate that opens 1.3 us after the slowest CTA's phase A)
#endif
#ifndef OKC_PBPF
#define OKC_PBPF 0  // WS: L2 prefetch (evict_last) of this CTA's own phase-B units, in consumption order, issued by the
                    // helper warp a few lines per lane per phase-A unit; OKC_PBPF = grid-wide byte cap (per CTA: / G)
#endif
#ifndef OKC_WSER
#define OKC_WSER 0  // WSB: release the ring slot as soon as every warp holds the unit's codes and scales in registers
                    // (warp 0 refills it before the compute), so OKC_NS units stream while a unit is computed, not NS-1
#endif
static_assert(!OKC_WSER || OKC_WSB, "OKC_WSER needs the WSB loop");
__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring);
// OKC_WSSTR: the strided unit order (see the define); ws_str(S): active for this launch
__device__ __forceinline__ bool ws_str(const Smem& S) { return OKC_WSSTR && !OKC_SPLIT && S.njS <= OKC_WSSTR; }
__device__ __forceinline__ int ws_str_cnt(int nS) {  // this CTA's unit count of a phase of nS units
  const int c = blockIdx.x, G = gridDim.x;
  return c < nS ? (nS - c + G - 1) / G : 0;
}
__device__ __forceinline__ int ws_str_jobcnt(int jr, int rbs) {  // this CTA's units of (skinny-relative) job jr
  const int c = blockIdx.x, G = gridDim.x;
  const int x0 = jr * rbs, x1 = x0 + rbs;  // (x - c + G - 1) / G = ceil((x - c) / G) for x > c - G
  return (x1 - c + G - 1) / G - (x0 - c + G - 1) / G;
}
template <bool PB>
__device__ void skinny_ring_ws(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq) {
  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);
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
  unsigned long long dbg_wland = 0, dbg_whelp = 0, dbg_t0 = 0;  // (debug) thread 0: ns waited on the ring / the helper
  bool dbg_g2 = false;
  for (int i = 0; i < nU; ++i) {
    const int v = (OKC_WSSTR && ws_str(S)) ? (int)blockIdx.x + i * (int)gridDim.x : v0 + i;
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
        if (i == 0 && tid == 0) DBGPX(2);  // (debug) first phase-B job gate passed
        if (OKC_DEBUG && i > 0 && tid == 0 && !dbg_g2) { dbg_g2 = true; DBGU(2); }  // (debug) second job gate passed
      }
      load_x_regs<PB>(p, S, j, lane, warp, x0);
    }
    const int q = seq0 + i;
    const int sl = q % OKC_NS;
#if OKC_NOLOAD
    if (q < OKC_NS)
#endif
    if (OKC_DEBUG && tid == 0) dbg_t0 = gtimer();
    mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));
    if (OKC_DEBUG && tid == 0) dbg_wland += gtimer() - dbg_t0;
    if (OKC_DEBUG && tid == 0) {  // (debug) first unit of the phase landed / last phase-A unit landed
      if (i == 0) { if (PB) DBGPX(1); else DBG(7); }
      if (!PB && i == nU - 1) DBGT(3);
    }
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
#if !OKC_WSNOC
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
#else
    (void)sc; (void)x0; (void)k16;
#endif
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
    if (OKC_DEBUG && tid == 0) dbg_t0 = gtimer();
    mbar_wait(S.wsre + bb, (uint32_t)(((q / NRB) & 1) ^ 1));  // the helper has consumed this buffer's previous unit
    if (OKC_DEBUG && tid == 0) dbg_whelp += gtimer() - dbg_t0;
    *reinterpret_cast<float4*>(&S.red[bb][warp][lane][0]) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    CTA_SYNC();
    if (OKC_DEBUG && !PB && q == 3 && tid == 0 && p.dbg) *DBGV_PTR(2) = gtimer();  // (debug) unit 3 handed off
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
    const int v = (OKC_WSSTR && ws_str(S)) ? (int)blockIdx.x + ul * (int)gridDim.x : v0 + ul, j = njF + v / RB_B, rb = v - (v / RB_B) * RB_B;
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
  if (OKC_DEBUG && tid == 0 && p.dbg) { *DBGU_PTR(PB ? 5 : 4) = dbg_whelp; *DBGU_PTR(PB ? 7 : 6) = dbg_wland; }
  seq = seq0 + nU;
}

__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring) {
  const int sl = q % OKC_NS;
  if (OKC_DEBUG) { if (q == 0) DBGPX(0); if (q == nA) DBGPX(3); }  // (debug) first unit / first phase-B unit issued
  const bool st = OKC_WSSTR && ws_str(S);
  const int G = (int)gridDim.x, c = (int)blockIdx.x;
  if (q < nA) ring_issue<false>(p, S, st ? c + q * G : a0 + q, ring + sl * UNIT_BYTES, S.bars + sl);
  else ring_issue<true>(p, S, st ? c + (q - nA) * G : b0 + (q - nA), ring + sl * UNIT_BYTES, S.bars + sl);
}

__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {
  const int lane = TIDX & 31;
#if OKC_WSHP
  {  // the epilogue constants into smem, issued now (kernel start) as asynchronous copies: a cold global load issued
     // during the weight stream waits 5-7 us (set V/W stamps), and a synchronous load here delays the CTA barrier;
     // all local experts (no routing needed); waited for just before the first epilogue
    const uint32_t s_nw = smem_u32(S.hnw), s_gs = smem_u32(&S.hgs[0][0]);
    const char* nw = reinterpret_cast<const char*>(p.normw);
    for (int i = lane; i < HD * 2 / 16; i += 32)
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s_nw + i * 16), "l"(nw + i * 16) : "memory");
    if (lane < p.nloc && lane < 32) {
      asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(s_gs + lane * 4), "l"(reinterpret_cast<const char*>(p.gsA + lane)) : "memory");
      asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(s_gs + 128 + lane * 4), "l"(reinterpret_cast<const char*>(p.gsB + lane)) : "memory");
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  }
#endif
  WS_JOIN();  // routing tables and this CTA's ranges are ready
  const int a0 = S.wsr[0], a1 = S.wsr[1], b0 = S.wsr[2], b1 = S.wsr[3];
  const int nA = a1 - a0, N = nA + (b1 - b0);
  const int njF = S.njF;
  if (lane == 0)
    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);
  const int gg = lane >> 2, tt = lane & 3;
  const int Nh = (OKC_WSB && OKC_WSBL) ? N - min(OKC_WSBLK, b1 - b0) : N;
#if OKC_PBPF
  // phase-B L2 prefetch: the lines of this CTA's phase-B units [b0, b1) in consumption order, up to the byte cap,
  // spread over the phase-A units (pf_per lines per lane per unit); the weights' own TMAs stay evict_first / normal
  constexpr int PF_LPU = UNIT_BYTES / 128;  // 288 lines per unit
  const int pf_n = min((b1 - b0) * PF_LPU, (int)((OKC_PBPF / (long long)gridDim.x) / 128));
  const int pf_per = nA > 0 ? (pf_n + nA * 32 - 1) / (nA * 32) : 0;
  int pf_i = 0;
  auto pf_issue = [&](int cnt) {
    for (int k = 0; k < cnt; ++k) {
      const int i = pf_i + lane + k * 32;
      if (i < pf_n) {
        const int u = i / PF_LPU, v = b0 + u;
        const int j = njF + (v >> RB_B_LOG), rb = v & (RB_B - 1);
        const unsigned char* src = p.wBu + ((size_t)S.jE[j] * RB_B + rb) * (size_t)UNIT_BYTES + (size_t)(i - u * PF_LPU) * 128;
        asm volatile("prefetch.global.L2::evict_last [%0];\n" ::"l"(src));
      }
    }
    pf_i += cnt * 32;
  };
  if (nA == 0) pf_issue((pf_n + 31) / 32);
#endif
#if OKC_WSHP
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");  // the epilogue constants have landed (every lane its own copies)
  __syncwarp();
#endif
  unsigned long long dv_fence = 0, dv_wA = 0, dv_wB = 0, dv_eA = 0, dv_eB = 0, dv_t = 0;  // (debug) helper accounting
  if (OKC_DEBUG && lane == 0 && p.dbg) *DBGV_PTR(5) = gtimer();
  for (int q = 0; q < Nh; ++q) {
#if !OKC_NOLOAD
    if (!OKC_WSRF && !OKC_WSB && lane == 0 && q + OKC_NS < N) {  // unit q + NS goes into unit q's slot once every compute warp released it
      mbar_wait(S.wse + q % OKC_NS, (uint32_t)((q / OKC_NS) & 1));
      ws_issue(p, S, q + OKC_NS, nA, a0, b0, ring);
    }
#endif
#if OKC_PBPF
    if (q < nA) pf_issue(q == nA - 1 ? (pf_n - pf_i + 31) / 32 : pf_per);
#endif
    const bool PB = q >= nA;
    const int v = (OKC_WSSTR && ws_str(S)) ? (int)blockIdx.x + (PB ? q - nA : q) * (int)gridDim.x : (PB ? b0 + (q - nA) : a0 + q);
    const int rbs = PB ? RB_B : RB_A;
    const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;
    const int bb = q % NRB;
    if (OKC_DEBUG) dv_t = gtimer();
    mbar_wait(S.wsrf + bb, (uint32_t)((q / NRB) & 1));
    if (OKC_DEBUG) { const unsigned long long t1 = gtimer(); if (PB) dv_wB += t1 - dv_t; else dv_wA += t1 - dv_t; dv_t = t1; }
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
#if OKC_WSHP
      const float gsc = S.hgs[0][ce] * TWO62;
#else
      const float gsc = p.gsA[ce] * TWO62;
#endif
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tk = 2 * tt + h;
        if (tk < cnt) {
          const float gsct = gsc * RS_T(S.ent[st + tk] >> 3);  // (OKC_ROUTE_LOGITS: per-token pre-expert RMS scale)
          const float gt = (h ? sm.y : sm.x) * gsct, up = (h ? sm.w : sm.z) * gsct;
          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));
        }
      }
#if OKC_WSSTR
      if (ws_str(S)) {  // strided order: job j is credited as soon as this CTA's last unit of it is done (its B can start)
        if (q == nA - 1 || (((int)blockIdx.x + (q + 1) * (int)gridDim.x) >> RB_A_LOG) != (v >> RB_A_LOG)) {
          if (OKC_DEBUG) dv_t = gtimer();
          __threadfence();
          if (OKC_DEBUG) dv_fence += gtimer() - dv_t;
          __syncwarp();
          if (lane == 0) {
            atomicAdd(p.doneA + j, ws_str_jobcnt(v >> RB_A_LOG, RB_A));
            if (OKC_DEBUG && p.dbg) { if (*DBGU_PTR(0) == 0ull) DBGU(0); DBGU(1); }  // (p.dbg is null in timing runs)
          }
        }
      } else
#endif
      if (q == nA - 1) {  // this CTA's phase A is done: credit its units to the jobs' counters
        if (OKC_DEBUG) dv_t = gtimer();
        __threadfence();
        if (OKC_DEBUG) dv_fence += gtimer() - dv_t;
        __syncwarp();
        if (lane == 0) {
          const int jf = njF + a0 / RB_A, jl = njF + (a1 - 1) / RB_A;
          for (int jj = jf; jj <= jl; ++jj) {
            const int a = max(a0, (jj - njF) * RB_A), b = min(a1, (jj - njF + 1) * RB_A);
            atomicAdd(p.doneA + jj, b - a);
          }
          if (OKC_DEBUG) { DBGU(0); DBGU(1); }
        }
      }
      if (OKC_DEBUG && q == 0 && lane == 0) DBGU(3);  // (debug) the helper's first phase-A epilogue done
    } else {
      const int n0 = rb * 16 + gg;
#if OKC_WSHP
      const float gsc = S.hgs[1][ce] * TWO62;
      const float nw0 = 1.f + __bfloat162float(S.hnw[n0]), nw1 = 1.f + __bfloat162float(S.hnw[n0 + 8]);
#else
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
    if (OKC_DEBUG) {  // (debug) epilogue time of this unit; the unit-3 phase-A epilogue end stamp
      const unsigned long long t2 = gtimer();
      if (PB) dv_eB += t2 - dv_t; else dv_eA += t2 - dv_t;
      if (!PB && q == 3 && lane == 0 && p.dbg) *DBGV_PTR(1) = t2;
    }
  }
  if (OKC_DEBUG && lane == 0 && p.dbg) { *DBGV_PTR(0) = dv_fence; *DBGV_PTR(3) = dv_wA; *DBGV_PTR(4) = dv_eA; *DBGV_PTR(6) = dv_wB; *DBGV_PTR(7) = dv_eB; }
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
  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);
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
  const int lane = TIDX & 31;
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
constexpr int BW_F = 0, BW_E = WGS_NS, BX_F = 2 * WGS_NS, BX_E = BX_F + WGS_NX;  // barrier indices (weights full / empty, x full / empty)
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
#ifndef OKC_GATHT
#define OKC_GATHT 0  // WGS: phase-A x gather by whole n8 tiles (CTA c: tiles c, c + G, ...), the 8 rows staged through
                     // the idle ring smem (coalesced 16 B loads, padded row stride) and written as 16 contiguous 4 KB
                     // blocks, one release per tile: the per-row gather's release fence over scattered 16 B stores
                     // ended 4-11 us after launch and gated every CTA's first x stage
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
#if OKC_WGSDBG == 1
  printf("WGS hang: block %d thread %d tag %d parity %u\n", (int)blockIdx.x, (int)TIDX, tag, parity);
#endif
  __trap();  // (OKC_WGSDBG 2: no printf, so no CALL inside the wgmma pipeline or a deallocated producer region)
#else
  (void)tag;
  mbar_wait(bar, parity);
#endif
}
// bounded global-memory spins in the DIAG builds (a hang traps within seconds instead of the judge's 900 s)
#if OKC_WGSDBG
#define SPIN_GUARD(cond) do { long long it_ = 0; while (cond) { __nanosleep(32); if (++it_ > (1ll << 24)) __trap(); } } while (0)
#else
#define SPIN_GUARD(cond) do { while (cond) __nanosleep(32); } while (0)
#endif
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
  if ((TIDX & 31) != 0) return;
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
  if ((TIDX & 31) != 0) return;
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
  if ((TIDX & 31) == 0) { wgs_ranges(p, S); if (OKC_DEBUG && p.dbg) p.dbg[(size_t)gridDim.x * 13 + (size_t)blockIdx.x * 4 + 3] = gtimer(); }
  __syncwarp();
#endif
#if OKC_WGSEARLY
  asm volatile("bar.arrive 9, %0;\n" ::"n"(NT + 32) : "memory");  // ranges ready for the consumers (they sync after the gather)
  asm volatile("bar.sync 8, 64;\n" ::: "memory");                 // ranges ready: both producers go (the x stages wait on xready)
#else
  WS_JOIN();  // routing tables and this CTA's stage ranges are ready
#endif
  if ((TIDX & 31) != 0) return;
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
        if (!pb && !(OKC_WGSX & 8)) SPIN_GUARD(ld_acquire(p.xready + j) < npad);  // every row gathered
      }
      // phase B stage st reads inter columns 256 st .. +255: the 2 phase-A units (x 4 warpgroups) that wrote them
      if (pb && !(OKC_WGSX & 8)) SPIN_GUARD(ld_acquire(p.doneAs + j * 16 + st) < 8);
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
#if OKC_WGSFOLD
static_assert(!OKC_WGSDYN && !OKC_WGSEARLY && !OKC_WGSPR, "OKC_WGSFOLD: static ranges, no producer warps");
__device__ __forceinline__ bool mbar_try(uint64_t* bar, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(smem_u32(bar)), "r"(parity) : "memory");
  return ok != 0;
}
// a cursor over this CTA's stage sequence, as the producer warps walked it: phase A [sp, g1) then [g0, sp)
// (wgs_split), then phase B; n = ring position (the consumers' seq)
struct WgsCur {
  int pb, gs, g0, g1, sp, n;
  bool done;
  __device__ __forceinline__ void next_phase(const Smem& S) {
    for (;;) {
      ++pb;
      if (pb > 1) { done = true; return; }
      g0 = S.wgr[2 * pb]; g1 = S.wgr[2 * pb + 1]; sp = wgs_split(pb, g0, g1);
      if (g1 > g0) { gs = sp; return; }
    }
  }
  __device__ __forceinline__ void init(const Smem& S) { n = 0; pb = -1; done = false; next_phase(S); }
  __device__ __forceinline__ void advance(const Smem& S) {
    ++n;
    if (++gs == g1) gs = g0;      // [sp, g1) done: then [g0, sp)
    if (gs == sp) next_phase(S);  // back at the split point: the phase's stages are all issued
  }
};
__device__ __forceinline__ void wgs_issue_w(const Params& p, Smem& S, unsigned char* ring, const WgsCur& c) {
  const int ubl = c.pb ? 4 : 5;
  const unsigned char* base = c.pb ? p.wBs : p.wAs;
  const int u = c.gs >> 4, st = c.gs & 15, j = u >> ubl, ub = u & ((1 << ubl) - 1);
  const int ws = c.n % WGS_NS;
  uint64_t* bar = S.bars + BW_F + ws;
  const unsigned char* src = base + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + st)) * (size_t)UNIT_BYTES;
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
               ::"r"(smem_u32(ring + ws * UNIT_BYTES)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");
}
__device__ __forceinline__ bool wgs_x_ready(const Params& p, const Smem& S, const WgsCur& c) {
  const int ubl = c.pb ? 4 : 5;
  const int u = c.gs >> 4, st = c.gs & 15, j = u >> ubl;
  if (!c.pb) { const int npad = ((S.jC[j] + 7) >> 3) << 3; return ld_acquire(p.xready + j) >= npad; }
  return ld_acquire(p.doneAs + j * 16 + st) >= 8;
}
__device__ __forceinline__ void wgs_issue_x(const Params& p, Smem& S, unsigned char* ring, const WgsCur& c) {
  const int ubl = c.pb ? 4 : 5;
  const int u = c.gs >> 4, st = c.gs & 15, j = u >> ubl;
  const int npad = ((S.jC[j] + 7) >> 3) << 3;
  const __nv_bfloat16* jb = (c.pb ? p.xB : p.xA) + (size_t)S.jrow[j] * HD;
  const int xs = c.n % WGS_NX;
  uint64_t* bar = S.bars + BX_F + xs;
  const uint32_t bytes = (uint32_t)npad * 512u;
  unsigned char* xring = ring + WGS_NS * UNIT_BYTES;
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               ::"r"(smem_u32(xring + xs * WGS_XS)), "l"(jb + (size_t)st * npad * 256), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
// the folded producers' step (lane 0 of warp 0, s = the stage about to be consumed or in progress): weights and x of
// stage s must be issued (blocking only then); later stages up to the ring depths when their slot is free (and, for x,
// its gating is satisfied)
__device__ __forceinline__ void wgs_fold_poll(const Params& p, Smem& S, unsigned char* ring, WgsCur& wc, WgsCur& xc, int s) {
  while (!wc.done && wc.n < s + WGS_NS) {
    if (wc.n >= WGS_NS) {
      uint64_t* eb = S.bars + BW_E + wc.n % WGS_NS;
      const uint32_t par = (uint32_t)((wc.n / WGS_NS) & 1) ^ 1u;
      if (wc.n <= s) mbar_wait(eb, par);
      else if (!mbar_try(eb, par)) break;
    }
    wgs_issue_w(p, S, ring, wc);
    wc.advance(S);
  }
  while (!xc.done && xc.n < s + WGS_NX) {
    const bool must = xc.n <= s;
    if (xc.n >= WGS_NX) {
      uint64_t* eb = S.bars + BX_E + xc.n % WGS_NX;
      const uint32_t par = (uint32_t)((xc.n / WGS_NX) & 1) ^ 1u;
      if (must) mbar_wait(eb, par);
      else if (!mbar_try(eb, par)) break;
    }
    if (!wgs_x_ready(p, S, xc)) {
      if (!must) break;
      while (!wgs_x_ready(p, S, xc)) __nanosleep(32);
    }
    wgs_issue_x(p, S, ring, xc);
    xc.advance(S);
  }
}
#else
struct WgsCur { int n; };
#endif
// phase A's x gather (all consumer threads of every CTA, once, at the start): this CTA's share of the jobs' padded rows
// copied into xA's per-stage wgmma layout, then counted into xready (padding rows are counted, not written)
__device__ void wgs_gather(const Params& p, Smem& S) {
  const int tid = TIDX, nj = S.njF + S.njS;
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
  DBGP(1);  // this thread's gather copies done
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
  DBGP(2);  // gather released
}
#if OKC_GATHT
__device__ void wgs_gather_t(const Params& p, Smem& S, unsigned char* stg) {
  const int tid = TIDX, nj = S.njF + S.njS;
  const int ntiles = S.jrow[nj] >> 3;
  constexpr int RS = HD * 2 + 16;  // staged row stride (bytes): the 8 rows of a tile hit different bank groups
  static_assert(NT * 16 == HD * 2, "one 16 B chunk per thread per row");
  static_assert(8 * RS <= WGS_NS * UNIT_BYTES, "the staging tile fits in the weight ring");
  int j = 0;
  for (int tl = blockIdx.x; tl < ntiles; tl += gridDim.x) {
    const int r0 = tl << 3;
    while (S.jrow[j + 1] <= r0) ++j;
    const int cnt = S.jC[j], ntt = (cnt + 7) >> 3, nt = (r0 - S.jrow[j]) >> 3;
    uint4 v[8];  // the tile's rows (padding rows zero), coalesced; all eight loads in flight before the stores
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int r = nt * 8 + i;
      v[i] = make_uint4(0u, 0u, 0u, 0u);
      if (r < cnt) v[i] = __ldg(reinterpret_cast<const uint4*>(p.x + (size_t)(S.ent[S.jS[j] + r] >> 3) * HD) + tid);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) *reinterpret_cast<uint4*>(stg + i * RS + tid * 16) = v[i];
    CTA_SYNC();
    uint4* dst = reinterpret_cast<uint4*>(p.xA + (size_t)S.jrow[j] * HD);
#pragma unroll 2
    for (int m = 0; m < 8; ++m) {  // out in the per-stage wgmma B layout: per stage 32 k8 x 8 rows x 16 B contiguous
      const int L = tid + m * NT;
      const int st = L >> 8, kb = (L >> 3) & 31, i = L & 7;
      const uint4 v = *reinterpret_cast<const uint4*>(stg + i * RS + (st * 32 + kb) * 16);
      dst[((st * ntt + nt) * 32 + kb) * 8 + i] = v;
    }
    CTA_SYNC();  // (the staging buffer is reused; the stores are ordered before the release)
    if (tid == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.xready + j), "r"(8) : "memory");
  }
  DBGP(2);
}
#endif
// Work left from the previous segment, done after the next segment's first wgmma group (so the stores and atomics it
// waits for have long completed): kind 1 = release doneA for job j (phase A), kind 2 = add the stage count of a split
// unit's share and, if that completes the unit, read the summed partials back and run its epilogue.
struct WgsPend { int kind, u, slot, nst; };
template <bool PB>
__device__ __forceinline__ void wgs_epi_tile(const Params& p, const Smem& S, int j, int r16, int nt, const float* d4) {
  const int lane = TIDX & 31, g = lane >> 2, t = lane & 3;
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
  const int tid = TIDX, warp = WARP_IDX(tid), wg = warp >> 2, wp = warp & 3, wt = tid & 127;
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
  if (wt == 0) SPIN_GUARD(ld_acquire(fc) < 16 - pd.nst);
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
// the designated finisher of a split unit: phase A the share holding its last stage (the next CTA's first piece, run
// first there by the reorder), phase B the share holding stage 0 (the previous CTA's last piece); OKC_WGSDYN pool
// pieces (stage >= the pool start) are claimed in stage order, so the share holding the last stage finishes in both phases
template <bool PB>
__device__ __forceinline__ bool wgs_is_fin(const Smem& S, int u, int s0, int s1) {
#if OKC_WGSDYN
  if ((u << 4) + s0 >= S.wgd[2 * (PB ? 1 : 0)]) return s1 == 16;
#else
  (void)S; (void)u;
#endif
  return PB ? s0 == 0 : s1 == 16;
}
// one segment (stages [s0, s1) of unit u) on NTT n-tiles: accumulate, then finish (whole unit) or hand off (split)
template <bool PB, int NTT>
__device__ __forceinline__ void wgs_seg(const Params& p, Smem& S, unsigned char* ring, int u, int s0, int s1, int slot,
                                        int& seq, WgsPend& pd, WgsCur& wc, WgsCur& xc) {
  const int tid = TIDX, lane = tid & 31;
  const int warp = WARP_IDX(tid);
  (void)wc; (void)xc;
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
#if OKC_WGSFOLD
    if (warp == 0 && lane == 0) wgs_fold_poll(p, S, ring, wc, xc, seq);  // this stage's copies issued; later ones if free
    __syncwarp();
#endif
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
#if OKC_WGSFOLD
        // after the last group: this stage's weight slot is released by every warp that loaded it and the previous
        // stage's x slot by every warp past its group OKC_WGSWD - 1: refill them (non-blocking)
        if (lg == 3 && h == 4 / GS - 1 && warp == 0 && lane == 0) wgs_fold_poll(p, S, ring, wc, xc, seq);
#endif
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
    pd = WgsPend{(((OKC_WGSDFIN >> (PB ? 1 : 0)) & 1) && wgs_is_fin<PB>(S, u, s0, s1)) ? 3 : 2, u, slot, s1 - s0};
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
                                            int& seq, WgsPend& pd, WgsCur& wc, WgsCur& xc) {
  static_assert(!OKC_WGSFOLD || OKC_WGSMMAN == 0, "OKC_WGSFOLD: wgmma segments only");
  (void)wc; (void)xc;
  const int tid = TIDX, lane = tid & 31;
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
    pd = WgsPend{(((OKC_WGSDFIN >> (PB ? 1 : 0)) & 1) && wgs_is_fin<PB>(S, u, s0, s1)) ? 3 : 2, u, slot, s1 - s0};
    return;
  }
  const int j = u >> UBL, r16 = (u & ((1 << UBL) - 1)) * 16 + wg * 4 + wp;
#pragma unroll
  for (int nt = 0; nt < NTT; ++nt) wgs_epi_tile<PB>(p, S, j, r16, nt, d[nt]);
  if (!PB) pd = WgsPend{1, u, -1, 0};
}
#endif
template <bool PB>
__device__ __forceinline__ void wgs_phase(const Params& p, Smem& S, unsigned char* ring, int g0, int g1, int& seq, int& k,
                                          WgsCur& wc, WgsCur& xc) {
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
      case 1: wgs_seg_mma<PB, 1>(p, S, ring, u, s0, s1, slot, seq, pd, wc, xc); break;
#else
      case 1: wgs_seg<PB, 1>(p, S, ring, u, s0, s1, slot, seq, pd, wc, xc); break;
#endif
#if OKC_FT_T > 8
#if OKC_WGSMMAN >= 2
      case 2: wgs_seg_mma<PB, 2>(p, S, ring, u, s0, s1, slot, seq, pd, wc, xc); break;
#else
      case 2: wgs_seg<PB, 2>(p, S, ring, u, s0, s1, slot, seq, pd, wc, xc); break;
#endif
      case 3: wgs_seg<PB, 3>(p, S, ring, u, s0, s1, slot, seq, pd, wc, xc); break;
      case 4: wgs_seg<PB, 4>(p, S, ring, u, s0, s1, slot, seq, pd, wc, xc); break;
#endif
#if OKC_FT_T > 32
      case 5: wgs_seg<PB, 5>(p, S, ring, u, s0, s1, slot, seq, pd, wc, xc); break;
      case 6: wgs_seg<PB, 6>(p, S, ring, u, s0, s1, slot, seq, pd, wc, xc); break;
      case 7: wgs_seg<PB, 7>(p, S, ring, u, s0, s1, slot, seq, pd, wc, xc); break;
      default: wgs_seg<PB, 8>(p, S, ring, u, s0, s1, slot, seq, pd, wc, xc); break;
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
  // static ranges over the first (1 - 2^-WGSDYNS) of the cost; the pool starts at the next UNIT boundary after
  // stage_at(tots) (a unit straddling the boundary would get two designated finishers: deadlock), and the last
  // CTA's static range extends to it so no stage is left unissued
  if (nj == 0) { g0 = g1 = 0; return; }
  const uint32_t tots = tot - (tot >> OKC_WGSDYNS);
  const int T = nj << (9 - pb);
  int Ts = ((wgs_stage_at(S, nj, pb, tots) + 15) >> 4) << 4;
  if (Ts > T) Ts = T;
  g0 = wgs_stage_at(S, nj, pb, tots * (uint32_t)c / (uint32_t)G);
  g1 = c == G - 1 ? Ts : wgs_stage_at(S, nj, pb, tots * (uint32_t)(c + 1) / (uint32_t)G);
  if (g0 > Ts) g0 = Ts;
  if (g1 > Ts) g1 = Ts;
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
  S.wgd[1] = nj << 9;
  S.wgd[3] = nj << 8;
  S.wgd[0] = nj ? min(S.wgd[1], ((wgs_stage_at(S, nj, 0, S.wtot - (S.wtot >> OKC_WGSDYNS)) + 15) >> 4) << 4) : 0;
  S.wgd[2] = nj ? min(S.wgd[3], ((wgs_stage_at(S, nj, 1, S.wtotB - (S.wtotB >> OKC_WGSDYNS)) + 15) >> 4) << 4) : 0;
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
  const int tid = TIDX;
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
#if OKC_WSSTR
  if (ws_str(S)) { v0 = 0; v1 = ws_str_cnt(nS); }  // strided order: the loop indexes the CTA's units 0..count
#endif
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
  const int tid = TIDX;
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
#if OKC_WGSREG
    // the producer warpgroup gives its registers up at once; only the two TMA loops follow (R0..R15 in SASS): the
    // stage ranges, a CALL into kernel-budget code, are computed by thread 0 after the gather (OKC_WGSPR = 0)
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(OKC_WGSREGP) : "memory");
    if (tid < NT + 32) wgs_wproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
    else if (tid < NT + 64) wgs_xproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
    else WS_JOIN();  // (the two idle warps of the producer warpgroup join the one CTA-wide barrier, then leave)
#else
    if (tid < NT + 32) wgs_wproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
    else wgs_xproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
#endif
    return;
  }
#if OKC_WGSREG
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(OKC_WGSREG) : "memory");  // consumer warpgroups grow (blocks until the producers' dealloc)
#endif
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
      int n = min(OKC_FAST1, v1 - v0);
#if OKC_WSSTR
      const bool fstr = __popc(m) <= OKC_WSSTR;  // (= ws_str(S) once routing has set S.njS)
      if (fstr) n = min(OKC_FAST1, ws_str_cnt(nS));
#endif
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
#if OKC_WSSTR
        const int v = fstr ? (int)blockIdx.x + i * (int)gridDim.x : v0 + i;
#else
        const int v = v0 + i;
#endif
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
      if (n > 0) DBGPX(0);  // (debug) first weight unit issued
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
#if OKC_WSSTR
    if (ws_str(S)) { S.wsr[0] = 0; S.wsr[1] = ws_str_cnt(nA); S.wsr[2] = 0; S.wsr[3] = ws_str_cnt(nB); }  // strided: counts
#endif
#endif
  }
  WS_JOIN();
  DBG(6);  // (debug) ranges done: the helper issues the rest of the ring
#endif
#if OKC_WGS
#if !OKC_WGSEARLY
  if (tid < 32) wgs_jrow(S);
  CTA_SYNC();
  DBGP(0);  // jrow scan done
#endif
  // phase A's x into xA (this CTA's share of the rows), before the producers start: its x reads do not queue behind
  // every CTA's first weight stages (OKC_WGSEARLY: the producers are already streaming)
#if OKC_GATHT
  wgs_gather_t(p, S, reinterpret_cast<unsigned char*>(xbuf));  // (the weight ring is idle until the producers start)
#else
  wgs_gather(p, S);
#endif
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
    WgsCur wc, xc;  // OKC_WGSFOLD: the folded producers' cursors (lane 0 of warp 0 drives them)
#if OKC_WGSFOLD
    wc.init(S); xc.init(S);
#else
    wc.n = xc.n = 0;
#endif
    wgs_phase<false>(p, S, ring, S.wgr[0], S.wgr[1], seq, wk, wc, xc);
    DBG(2);
    wgs_phase<true>(p, S, ring, S.wgr[2], S.wgr[3], seq, wk, wc, xc);
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
  if (tid == 0) DBGT(0);  // (debug) helper epilogues done
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
    DBGT(1);  // (debug) grid barrier arrival
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

BW_SRC = r'''
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

CPP_SRC = r'''
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
PDL = False
# Routing in the kernel from the router logits (the decode layer: no _route_split launch). ROUTE_LOGITS_BUILDS makes
# prepare() also compile the OKC_ROUTE_LOGITS=1 variants of the selected builds (the layer sets it before prepare);
# kernel_fn_logits(data, state) is the entry. ROUTE_LOGITS_SELFTEST runs the in-kernel selection against the host
# selection on synthetic logits in prepare() (untimed) and prints the result.
ROUTE_LOGITS_BUILDS = False
ROUTE_LOGITS_SELFTEST = False
ROUTER_MULTIPLIER = 0.1767766952966369
# Scale decode with PRMT + IMAD instead of the smem LUT; exact only when every block-scale byte of the local experts
# is a positive normal e4m3 value (0x08..0x7E). prepare() checks the actual weights and keeps the LUT build otherwise.
PRMT_SCALES = True


def _scales_prmt_exact(*sfs):
    for sf in sfs:
        b = sf.contiguous().view(torch.uint8)
        if bool(((b < 0x08) | (b > 0x7E)).any()):
            return False
    return True

_EXTS = {}
MAIN_DEFINES = ("-DOKC_NW=16", "-DOKC_FAST1=0", "-DOKC_EPI4=0", "-DOKC_WGS=1")          # compile-time variant used by kernel_fn
# M <= TINY_MAX_M runs a skinny-only build with 16 warps per CTA (a job never has more than M tokens, so no fat units)
TINY_DEFINES = ("-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_EPI4=0", "-DOKC_FAST1=2", "-DOKC_L2H=0", "-DOKC_NACC=4", "-DOKC_UWARP=0", "-DOKC_WS=1", "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=2", "-DOKC_FINPF=1")  # bf16 WSB at NW=8 + WSBL tail (Oct 26: bs1 +2.5% / +3.5%). Oct 27: without -DOKC_L2H=0 the DIAG harness is 2.4 us faster at bs1 (28.6 vs 31.0) but two judged samples gave bs1 24.89 / 24.62 vs the shipped 24.6-26.1 band (~-0.5 us): below the lock threshold, kept as the author scored it
TINY_MAX_M = 4
# TINY_MAX_M < M <= TINY2_MAX_M runs a second skinny-only build (empty: skipped); TINY2_MAX_M must stay <= 8
TINY2_DEFINES = ('-DOKC_NW=8', '-DOKC_TINY=1', '-DOKC_CPF=0', '-DOKC_EPI4=0', '-DOKC_FAST1=2', '-DOKC_NACC=4', '-DOKC_UWARP=0', '-DOKC_WS=1', '-DOKC_WSB=1', '-DOKC_WSBL=2', '-DOKC_WSBLK=1', '-DOKC_FINPF=1')  # bf16 WSB (Oct 26: bs8 +2.0..2.5%)
TINY2_MAX_M = 8
# TINY_MAX_M < M <= MID_MAX_M runs MID_DEFINES (empty: the main module)
MID_DEFINES = ('-DOKC_NW=8', '-DOKC_TINY=1', '-DOKC_SPLIT=1', '-DOKC_CPF=0', '-DOKC_EPI4=0', '-DOKC_FAST1=2', '-DOKC_NACC=4', '-DOKC_UWARP=0', '-DOKC_WS=1', '-DOKC_WSB=1', '-DOKC_WSBL=2', '-DOKC_WSBLK=2', '-DOKC_FINPF=1')  # bf16 split-job MID (Oct 26): tiny-style, experts > 8 tokens split
MID_MAX_M = 16  # M 17..63 runs on the WGS MAIN build (v179: faster than the v122c MID kernel at bs32)
DIAG = True                # prepare() prints an H200 timing/diagnostic table (untimed; for development only)
DIAG_SET = "X"             # which variant set below (a copy of this file with another set runs as a second check)
_WGS_SHAPES = [32, 128, 256]
# name -> defines (all DIAG_SHAPES) or {"defines": ..., "shapes": [...]}, compiled and timed in diag mode
def _fast1(defs, n):  # a tiny define tuple with another FAST1 pre-issue depth (units issued before routing, <= OKC_NS)
    return tuple(f"-DOKC_FAST1={n}" if d.startswith("-DOKC_FAST1=") else d for d in defs)


def _nw(defs, n):  # a tiny define tuple with n compute warps (16: half a K slice per warp, 4 warps per SM quadrant)
    return tuple(f"-DOKC_NW={n}" if d.startswith("-DOKC_NW=") else d for d in defs)


DIAG_VARIANT_SETS = {
    "A": {  # WGS attribution: weight stream / x stream / wgmma / decode, producer stall timers
        "main": MAIN_DEFINES,
        "wgs_dbg": {"defines": MAIN_DEFINES + ("-DOKC_DEBUG=1",), "shapes": _WGS_SHAPES},
        "wgs_nox": {"defines": MAIN_DEFINES + ("-DOKC_WGSX=16",), "shapes": _WGS_SHAPES},
        "wgs_now": {"defines": MAIN_DEFINES + ("-DOKC_WGSX=32",), "shapes": _WGS_SHAPES},
        "wgs_nomma": {"defines": MAIN_DEFINES + ("-DOKC_WGSX=1",), "shapes": _WGS_SHAPES},
        "wgs_nodq": {"defines": MAIN_DEFINES + ("-DOKC_XNODQ=1",), "shapes": _WGS_SHAPES},
    },
    "X": {  # OKC_WSH0: the helper as hardware warp 0 (scheduler age priority), alone and with the strided order
        "tiny": {"defines": TINY_DEFINES, "shapes": [1]},
        "tiny_h0": {"defines": TINY_DEFINES + ("-DOKC_WSH0=1",), "shapes": [1]},
        "tiny_h0_dbg": {"defines": TINY_DEFINES + ("-DOKC_WSH0=1", "-DOKC_DEBUG=1"), "shapes": [1]},
        "tiny_h0_str": {"defines": TINY_DEFINES + ("-DOKC_WSH0=1", "-DOKC_WSSTR=4"), "shapes": [1]},
        "tiny_h0_str_dbg": {"defines": TINY_DEFINES + ("-DOKC_WSH0=1", "-DOKC_WSSTR=4", "-DOKC_DEBUG=1"), "shapes": [1]},
        "tiny2": {"defines": TINY2_DEFINES, "shapes": [8]},
        "tiny2_h0": {"defines": TINY2_DEFINES + ("-DOKC_WSH0=1",), "shapes": [8]},
        "tiny2_h0_dbg": {"defines": TINY2_DEFINES + ("-DOKC_WSH0=1", "-DOKC_DEBUG=1"), "shapes": [8]},
        "mid": {"defines": MID_DEFINES, "shapes": [16]},
        "mid_h0": {"defines": MID_DEFINES + ("-DOKC_WSH0=1",), "shapes": [16]},
    },
    "W": {  # OKC_WSHP: the helper's epilogue constants prefetched into smem (set V: its first epilogue took ~5 us on cold
            # loads and the backlog delayed the doneA credits 2-4 us), alone and with the strided order
        "tiny": {"defines": TINY_DEFINES, "shapes": [1]},
        "tiny_hp": {"defines": TINY_DEFINES + ("-DOKC_WSHP=1",), "shapes": [1]},
        "tiny_hp_dbg": {"defines": TINY_DEFINES + ("-DOKC_WSHP=1", "-DOKC_DEBUG=1"), "shapes": [1]},
        "tiny_hp_str": {"defines": TINY_DEFINES + ("-DOKC_WSHP=1", "-DOKC_WSSTR=4"), "shapes": [1]},
        "tiny_hp_str_dbg": {"defines": TINY_DEFINES + ("-DOKC_WSHP=1", "-DOKC_WSSTR=4", "-DOKC_DEBUG=1"), "shapes": [1]},
        "tiny2": {"defines": TINY2_DEFINES, "shapes": [8]},
        "tiny2_hp": {"defines": TINY2_DEFINES + ("-DOKC_WSHP=1",), "shapes": [8]},
        "tiny2_hp_dbg": {"defines": TINY2_DEFINES + ("-DOKC_WSHP=1", "-DOKC_DEBUG=1"), "shapes": [8]},
        "mid": {"defines": MID_DEFINES, "shapes": [16]},
        "mid_hp": {"defines": MID_DEFINES + ("-DOKC_WSHP=1",), "shapes": [16]},
    },
    "V": {  # helper-warp accounting at bs1: fence time before the doneA credits, backlog (unit-3 hand-off vs its epilogue),
            # partials waits vs epilogue time; shipped contiguous order vs the strided order (set T: the strided order's
            # per-job credits landed 4-6 us after the units, so its early B gate never materialized)
        "tiny": {"defines": TINY_DEFINES, "shapes": [1]},
        "tiny_dbg": {"defines": TINY_DEFINES + ("-DOKC_DEBUG=1",), "shapes": [1]},
        "tiny_str_dbg": {"defines": TINY_DEFINES + ("-DOKC_WSSTR=4", "-DOKC_DEBUG=1"), "shapes": [1]},
        "tiny2_dbg": {"defines": TINY2_DEFINES + ("-DOKC_DEBUG=1",), "shapes": [8]},
    },
    "U": {  # WGS bytes in flight: 3 x 36 KB weight slots = 108 KB per SM (ncu bs32: DRAM 69% of peak, torch reaches 94%);
            # the x ring (3 x 32 KB for FT_T = 64 rows) blocks a 4th weight slot, with FT_T = 32 (16 KB x slots) 4 fit
        "main": {"defines": MAIN_DEFINES, "shapes": [32, 128]},
        "main_dbg": {"defines": MAIN_DEFINES + ("-DOKC_DEBUG=1",), "shapes": [32]},
        "main_ft32": {"defines": MAIN_DEFINES + ("-DOKC_FT_T=32",), "shapes": [32, 128]},
        "main_ft32_ns4": {"defines": MAIN_DEFINES + ("-DOKC_FT_T=32", "-DOKC_WGSNS=4"), "shapes": [32, 128]},
        "main_ft32_ns4_dbg": {"defines": MAIN_DEFINES + ("-DOKC_FT_T=32", "-DOKC_WGSNS=4", "-DOKC_DEBUG=1"), "shapes": [32]},
        "main_ft32_ns4_nx2": {"defines": MAIN_DEFINES + ("-DOKC_FT_T=32", "-DOKC_WGSNS=4", "-DOKC_WGSNX=2"), "shapes": [32]},
    },
    "T": {  # bytes in flight: the tiny builds run the skinny ring at OKC_NS = MAIN_NS = 3 slots (110 KB per SM; the fat-ring
            # budget the launch reserves holds 4, and 5 fits the 227 KB limit); the ncu capture of the shipped kernel shows
            # DRAM at 65% (bs1) / 75% (bs8) of peak while torch reaches 94%; plus the strided order with credit stamps
        "tiny": {"defines": TINY_DEFINES, "shapes": [1, 4]},
        "tiny_ns4": {"defines": TINY_DEFINES + ("-DOKC_NS=4",), "shapes": [1, 4]},
        "tiny_ns5": {"defines": TINY_DEFINES + ("-DOKC_NS=5",), "shapes": [1, 4]},
        "tiny_ns5_dbg": {"defines": TINY_DEFINES + ("-DOKC_NS=5", "-DOKC_DEBUG=1"), "shapes": [1]},
        "tiny_dbg": {"defines": TINY_DEFINES + ("-DOKC_DEBUG=1",), "shapes": [1]},
        "tiny_str_dbg": {"defines": TINY_DEFINES + ("-DOKC_WSSTR=4", "-DOKC_DEBUG=1"), "shapes": [1]},
        "tiny2": {"defines": TINY2_DEFINES, "shapes": [8]},
        "tiny2_ns4": {"defines": TINY2_DEFINES + ("-DOKC_NS=4",), "shapes": [8]},
        "tiny2_ns5": {"defines": TINY2_DEFINES + ("-DOKC_NS=5",), "shapes": [8]},
        "tiny2_ns5_dbg": {"defines": TINY2_DEFINES + ("-DOKC_NS=5", "-DOKC_DEBUG=1"), "shapes": [8]},
        "mid": {"defines": MID_DEFINES, "shapes": [16]},
        "mid_ns5": {"defines": MID_DEFINES + ("-DOKC_NS=5",), "shapes": [16]},
    },
    "S": {  # small-batch item 2: OKC_WSSTR strided job-major unit order (phase-B units of all jobs but the last overlap the
            # last job's phase-A tail; set R: 16 warps did not cut the per-unit compute, the loop is issue-bound near its floor)
        "tiny": {"defines": TINY_DEFINES, "shapes": [1, 4]},
        "tiny_str": {"defines": TINY_DEFINES + ("-DOKC_WSSTR=4",), "shapes": [1, 4]},   # (bs4 x6 jobs: contiguous = control)
        "tiny_str_dbg": {"defines": TINY_DEFINES + ("-DOKC_WSSTR=4", "-DOKC_DEBUG=1"), "shapes": [1]},
        "tiny_str6": {"defines": TINY_DEFINES + ("-DOKC_WSSTR=6",), "shapes": [4]},
        "tiny2": {"defines": TINY2_DEFINES, "shapes": [8]},
        "tiny2_str8": {"defines": TINY2_DEFINES + ("-DOKC_WSSTR=8",), "shapes": [8]},
        "tiny2_str8_dbg": {"defines": TINY2_DEFINES + ("-DOKC_WSSTR=8", "-DOKC_DEBUG=1"), "shapes": [8]},
    },
    "R": {  # small-batch item: the per-unit compute of the 8-warp tiny loop (set Q: 1.0-1.35 us per 36 KB unit, 2 warps per
            # SM quadrant, latency-bound; the stream runs 1.35 us/unit): 16 compute warps (96 registers, no spills), NACC 2/4,
            # dequant ablation; the MID build (bs16) with 16 warps too
        "tiny": {"defines": TINY_DEFINES, "shapes": [1, 4]},
        "tiny_nw16": {"defines": _nw(TINY_DEFINES, 16), "shapes": [1, 4]},
        "tiny_nw16_dbg": {"defines": _nw(TINY_DEFINES, 16) + ("-DOKC_DEBUG=1",), "shapes": [1, 4]},
        "tiny_nw16_na2": {"defines": tuple("-DOKC_NACC=2" if d == "-DOKC_NACC=4" else d for d in _nw(TINY_DEFINES, 16)), "shapes": [1]},
        "tiny_xnodq": {"defines": TINY_DEFINES + ("-DOKC_XNODQ=1",), "shapes": [1]},  # (wrong results: the dequant's share)
        "tiny2": {"defines": TINY2_DEFINES, "shapes": [8]},
        "tiny2_nw16": {"defines": _nw(TINY2_DEFINES, 16), "shapes": [8]},
        "tiny2_nw16_dbg": {"defines": _nw(TINY2_DEFINES, 16) + ("-DOKC_DEBUG=1",), "shapes": [8]},
        "mid": {"defines": MID_DEFINES, "shapes": [16]},
        "mid_nw16": {"defines": _nw(MID_DEFINES, 16), "shapes": [16]},
    },
    "Q": {  # small-batch latency: TINY (bs1, bs4) and TINY2 (bs8) stamps (min/med/max over CTAs) and the weights-only
            # stream floor of the same expert set (OKC_WSNOC: ring, x loads, handoffs, epilogues, no dequant / MMA)
        "tiny": {"defines": TINY_DEFINES, "shapes": [1, 4]},
        "tiny_dbg": {"defines": TINY_DEFINES + ("-DOKC_DEBUG=1",), "shapes": [1, 4]},
        "tiny_noc": {"defines": TINY_DEFINES + ("-DOKC_WSNOC=1",), "shapes": [1, 4]},
        "tiny_noc_dbg": {"defines": TINY_DEFINES + ("-DOKC_WSNOC=1", "-DOKC_DEBUG=1"), "shapes": [1, 4]},
        "tiny2": {"defines": TINY2_DEFINES, "shapes": [8]},
        "tiny2_dbg": {"defines": TINY2_DEFINES + ("-DOKC_DEBUG=1",), "shapes": [8]},
        "tiny2_noc": {"defines": TINY2_DEFINES + ("-DOKC_WSNOC=1",), "shapes": [8]},
        "tiny2_noc_dbg": {"defines": TINY2_DEFINES + ("-DOKC_WSNOC=1", "-DOKC_DEBUG=1"), "shapes": [8]},
        "tiny_f4": {"defines": _fast1(TINY_DEFINES, 4), "shapes": [1, 4]},    # the whole ring issued before routing
        "tiny2_f4": {"defines": _fast1(TINY2_DEFINES, 4), "shapes": [8]},
    },
    "P": {  # tails: WGSDYN pool segments (static cost-weighted ranges + segments claimed by the weights producer) with the
            # last-stage finisher and unit-indexed fixup slots (check D ran the last-arriver finisher), SEG 4 / 8
        "main": {"defines": MAIN_DEFINES, "shapes": _WGS_SHAPES},
        "dyn4": {"defines": MAIN_DEFINES + ("-DOKC_WGSDYN=1", "-DOKC_WGSSEG=4", "-DOKC_WGSDBG=2"), "shapes": _WGS_SHAPES},
        "dyn8": {"defines": MAIN_DEFINES + ("-DOKC_WGSDYN=1", "-DOKC_WGSSEG=8", "-DOKC_WGSDBG=2"), "shapes": _WGS_SHAPES},
        "dyn8_dbg": {"defines": MAIN_DEFINES + ("-DOKC_WGSDYN=1", "-DOKC_WGSSEG=8", "-DOKC_WGSDBG=2", "-DOKC_DEBUG=1"), "shapes": _WGS_SHAPES},
    },
    "O": {  # WGS phase-A x gather by whole n8 tiles through smem (OKC_GATHT) vs the per-row gather
        "main": {"defines": MAIN_DEFINES, "shapes": _WGS_SHAPES},
        "gatht": {"defines": MAIN_DEFINES + ("-DOKC_GATHT=1",), "shapes": _WGS_SHAPES},
        "gatht_dbg": {"defines": MAIN_DEFINES + ("-DOKC_GATHT=1", "-DOKC_DEBUG=1"), "shapes": _WGS_SHAPES},
    },
    "N": {  # WGS final pass: sum of squares fused into phase B's epilogue (OKC_WGSFINSS=0: one ybuf read) vs the shipped two-read pass
        "main": {"defines": MAIN_DEFINES, "shapes": _WGS_SHAPES},
        "wgs_finss0": {"defines": MAIN_DEFINES + ("-DOKC_WGSFINSS=0",), "shapes": _WGS_SHAPES},
        "wgs_finss0_dbg": {"defines": MAIN_DEFINES + ("-DOKC_WGSFINSS=0", "-DOKC_DEBUG=1"), "shapes": _WGS_SHAPES},
    },
    "M": {  # WGS with setmaxnreg: producer warpgroup dec 24, consumers inc 120; wgmma pipeline depth variants
        "main": {"defines": MAIN_DEFINES, "shapes": _WGS_SHAPES},
        "reg112": {"defines": MAIN_DEFINES + ("-DOKC_WGSREG=112", "-DOKC_WGSDBG=2"), "shapes": _WGS_SHAPES},
        "reg112_dbg": {"defines": MAIN_DEFINES + ("-DOKC_WGSREG=112", "-DOKC_WGSDBG=2", "-DOKC_DEBUG=1"), "shapes": _WGS_SHAPES},
    },
    "L": {  # WGS with the producers folded into warp 0 (16-warp CTA, 128 registers): wgmma pipeline depth variants
        "main": {"defines": MAIN_DEFINES, "shapes": _WGS_SHAPES},
        "fold": {"defines": MAIN_DEFINES + ("-DOKC_WGSFOLD=1",), "shapes": _WGS_SHAPES},
        "fold_gs4": {"defines": MAIN_DEFINES + ("-DOKC_WGSFOLD=1", "-DOKC_WGSGS=4"), "shapes": _WGS_SHAPES},
        "fold_dbg": {"defines": MAIN_DEFINES + ("-DOKC_WGSFOLD=1", "-DOKC_DEBUG=1"), "shapes": _WGS_SHAPES},
        "fold_gs4_dbg": {"defines": MAIN_DEFINES + ("-DOKC_WGSFOLD=1", "-DOKC_WGSGS=4", "-DOKC_DEBUG=1"), "shapes": _WGS_SHAPES},
    },
    "K": {  # bs1: L2H (evict_first stream + evict_last inter) with / without the phase-B prefetch, prefetch caps
        "tiny": {"defines": TINY_DEFINES, "shapes": [1]},
        "tiny_l2h": {"defines": tuple(d for d in TINY_DEFINES if d != "-DOKC_L2H=0"), "shapes": [1]},
        "tiny_pf_l2h": {"defines": tuple(d for d in TINY_DEFINES if d != "-DOKC_L2H=0") + ("-DOKC_PBPF=64000000",), "shapes": [1]},
        "tiny_pf_l2h_c32": {"defines": tuple(d for d in TINY_DEFINES if d != "-DOKC_L2H=0") + ("-DOKC_PBPF=32000000",), "shapes": [1]},
        "tiny_pf_l2h_c16": {"defines": tuple(d for d in TINY_DEFINES if d != "-DOKC_L2H=0") + ("-DOKC_PBPF=16000000",), "shapes": [1]},
        "tiny_l2h_dbg": {"defines": tuple(d for d in TINY_DEFINES if d != "-DOKC_L2H=0") + ("-DOKC_DEBUG=1",), "shapes": [1]},
        "tiny_pf_l2h_dbg": {"defines": tuple(d for d in TINY_DEFINES if d != "-DOKC_L2H=0") + ("-DOKC_PBPF=64000000", "-DOKC_DEBUG=1"), "shapes": [1]},
    },
    "J": {  # tiny builds: L2 prefetch of each CTA's phase-B units during phase A (OKC_PBPF byte cap), bs1 / bs8
        "tiny": {"defines": TINY_DEFINES, "shapes": [1]},
        "tiny_pf": {"defines": TINY_DEFINES + ("-DOKC_PBPF=64000000",), "shapes": [1]},
        "tiny_pf_l2h": {"defines": tuple(d for d in TINY_DEFINES if d != "-DOKC_L2H=0") + ("-DOKC_PBPF=64000000",), "shapes": [1]},
        "tiny_pf_dbg": {"defines": TINY_DEFINES + ("-DOKC_PBPF=64000000", "-DOKC_DEBUG=1"), "shapes": [1]},
        "tiny_dbg": {"defines": TINY_DEFINES + ("-DOKC_DEBUG=1",), "shapes": [1]},
        "tiny2": {"defines": TINY2_DEFINES, "shapes": [8]},
        "tiny2_pf40": {"defines": TINY2_DEFINES + ("-DOKC_PBPF=40000000",), "shapes": [8]},
        "tiny2_pf40_dbg": {"defines": TINY2_DEFINES + ("-DOKC_PBPF=40000000", "-DOKC_DEBUG=1"), "shapes": [8]},
    },
    "I": {  # WGS: per-warp mma.sync for small-N jobs (no warpgroup sync / descriptor / commit per step) vs wgmma
        "main": {"defines": MAIN_DEFINES, "shapes": _WGS_SHAPES},
        "wgs_mma1": {"defines": MAIN_DEFINES + ("-DOKC_WGSMMAN=1",), "shapes": _WGS_SHAPES},
        "wgs_mma2": {"defines": MAIN_DEFINES + ("-DOKC_WGSMMAN=2",), "shapes": _WGS_SHAPES},
        "wgs_mma2_dbg": {"defines": MAIN_DEFINES + ("-DOKC_WGSMMAN=2", "-DOKC_DEBUG=1"), "shapes": _WGS_SHAPES},
        "wgs_mma2_na2": {"defines": MAIN_DEFINES + ("-DOKC_WGSMMAN=2", "-DOKC_WGSMMANA=2"), "shapes": _WGS_SHAPES},
    },
    "H": {  # WGS for M <= 64: FT_T=32 frees accumulator registers (x slots 16 KB): deeper wgmma pipeline / weight ring
        "main": {"defines": MAIN_DEFINES, "shapes": [32, 128]},
        "wgs_ft32": {"defines": MAIN_DEFINES + ("-DOKC_FT_T=32",), "shapes": [32, 128]},
        "wgs_ft32_wd2": {"defines": MAIN_DEFINES + ("-DOKC_FT_T=32", "-DOKC_WGSWD=2"), "shapes": [32]},
        "wgs_ft32_gs4": {"defines": MAIN_DEFINES + ("-DOKC_FT_T=32", "-DOKC_WGSGS4N=4"), "shapes": [32]},
        "wgs_ft32_ns4": {"defines": MAIN_DEFINES + ("-DOKC_FT_T=32", "-DOKC_WGSNS=4"), "shapes": [32]},
        "wgs_ft32_ns4_wd2": {"defines": MAIN_DEFINES + ("-DOKC_FT_T=32", "-DOKC_WGSNS=4", "-DOKC_WGSWD=2"), "shapes": [32]},
    },
    "G": {  # WGS wgmma pipeline: commit-group length for small N, wait depth, no-scale ablation (bs32/128/256)
        "main": {"defines": MAIN_DEFINES, "shapes": _WGS_SHAPES},
        "wgs_gs4n2": {"defines": MAIN_DEFINES + ("-DOKC_WGSGS4N=2",), "shapes": _WGS_SHAPES},
        "wgs_wd2": {"defines": MAIN_DEFINES + ("-DOKC_WGSWD=2",), "shapes": _WGS_SHAPES},
        "wgs_gs4n2_wd2": {"defines": MAIN_DEFINES + ("-DOKC_WGSGS4N=2", "-DOKC_WGSWD=2"), "shapes": _WGS_SHAPES},
        "wgs_gs4": {"defines": MAIN_DEFINES + ("-DOKC_WGSGS=4",), "shapes": _WGS_SHAPES},
        "wgs_noscale": {"defines": MAIN_DEFINES + ("-DOKC_XNOSC=1",), "shapes": _WGS_SHAPES},
    },
    "F": {  # WGS early producer start (fixed consumer/range sync), bs32/128/256
        "main": {"defines": MAIN_DEFINES, "shapes": _WGS_SHAPES},
        "wgs_early_dbg": {"defines": MAIN_DEFINES + ("-DOKC_WGSEARLY=1", "-DOKC_DEBUG=1", "-DOKC_WGSDBG=2"), "shapes": _WGS_SHAPES},
        "wgs_early": {"defines": MAIN_DEFINES + ("-DOKC_WGSEARLY=1",), "shapes": _WGS_SHAPES},
    },
    "E": {  # WSDYN cause split (pool off / chunked claims / late slot release) + WGS early producer start
        "tiny": {"defines": TINY_DEFINES, "shapes": [1]},
        "tiny_dyn_s30": {"defines": TINY_DEFINES + ("-DOKC_WSDYN=1", "-DOKC_WSDYNS=30"), "shapes": [1]},
        "mid": {"defines": MID_DEFINES, "shapes": [16]},
        "mid_dyn_s30": {"defines": MID_DEFINES + ("-DOKC_WSDYN=1", "-DOKC_WSDYNS=30"), "shapes": [16]},
        "mid_dyn_c4": {"defines": MID_DEFINES + ("-DOKC_WSDYN=1", "-DOKC_WSDYNC=4"), "shapes": [16]},
        "mid_dyn_late": {"defines": MID_DEFINES + ("-DOKC_WSDYN=1", "-DOKC_WSDYNLATE=1"), "shapes": [16]},
        "main": {"defines": MAIN_DEFINES, "shapes": _WGS_SHAPES},
        "wgs_early": {"defines": MAIN_DEFINES + ("-DOKC_WGSEARLY=1",), "shapes": _WGS_SHAPES},
        "wgs_early_dbg": {"defines": MAIN_DEFINES + ("-DOKC_WGSEARLY=1", "-DOKC_DEBUG=1"), "shapes": _WGS_SHAPES},
    },
    "D": {  # WGS: dynamic stage claiming (WGSDYN) vs stream-K static ranges, pool size / segment length, timelines
        "main": {"defines": MAIN_DEFINES, "shapes": _WGS_SHAPES},
        "wgs_dyn": {"defines": MAIN_DEFINES + ("-DOKC_WGSDYN=1",), "shapes": _WGS_SHAPES},
        "wgs_dyn_s1": {"defines": MAIN_DEFINES + ("-DOKC_WGSDYN=1", "-DOKC_WGSDYNS=1"), "shapes": _WGS_SHAPES},
        "wgs_dyn_seg8": {"defines": MAIN_DEFINES + ("-DOKC_WGSDYN=1", "-DOKC_WGSSEG=8"), "shapes": _WGS_SHAPES},
        "wgs_dyn_dbg": {"defines": MAIN_DEFINES + ("-DOKC_WGSDYN=1", "-DOKC_DEBUG=1"), "shapes": _WGS_SHAPES},
    },
    "C": {  # tiny builds: dynamic unit claiming (WSDYN) vs static slices, pool size, timelines
        "tiny": {"defines": TINY_DEFINES, "shapes": [1]},
        "tiny_dyn": {"defines": TINY_DEFINES + ("-DOKC_WSDYN=1",), "shapes": [1]},
        "tiny_dyn_s1": {"defines": TINY_DEFINES + ("-DOKC_WSDYN=1", "-DOKC_WSDYNS=1"), "shapes": [1]},
        "tiny_dyn_dbg": {"defines": TINY_DEFINES + ("-DOKC_WSDYN=1", "-DOKC_DEBUG=1"), "shapes": [1]},
        "tiny2": {"defines": TINY2_DEFINES, "shapes": [8]},
        "tiny2_dyn": {"defines": TINY2_DEFINES + ("-DOKC_WSDYN=1",), "shapes": [8]},
        "mid": {"defines": MID_DEFINES, "shapes": [8, 16]},
        "mid_dyn": {"defines": MID_DEFINES + ("-DOKC_WSDYN=1",), "shapes": [8, 16]},
        "mid_dyn_s1": {"defines": MID_DEFINES + ("-DOKC_WSDYN=1", "-DOKC_WSDYNS=1"), "shapes": [16]},
        "mid_dyn_dbg": {"defines": MID_DEFINES + ("-DOKC_WSDYN=1", "-DOKC_DEBUG=1"), "shapes": [16]},
    },
    "B": {  # tiny builds: early slot release (WSER), ring depth, timelines
        "main": {"defines": MAIN_DEFINES, "shapes": [8, 16]},
        "tiny": {"defines": TINY_DEFINES, "shapes": [1]},
        "tiny_wser": {"defines": TINY_DEFINES + ("-DOKC_WSER=1",), "shapes": [1]},
        "tiny_dbg": {"defines": TINY_DEFINES + ("-DOKC_DEBUG=1",), "shapes": [1]},
        "tiny2": {"defines": TINY2_DEFINES, "shapes": [8]},
        "mid": {"defines": MID_DEFINES, "shapes": [8, 16]},
        "mid_wser": {"defines": MID_DEFINES + ("-DOKC_WSER=1",), "shapes": [8, 16]},
        "mid_ns4": {"defines": MID_DEFINES + ("-DOKC_NS=4",), "shapes": [8, 16]},
        "mid_dbg": {"defines": MID_DEFINES + ("-DOKC_DEBUG=1",), "shapes": [16]},
    },
}
DIAG_VARIANTS = DIAG_VARIANT_SETS[DIAG_SET]
DIAG_SHAPES = [1, 8, 16, 32, 128, 256] if DIAG_SET == "A" else [1, 4, 8] if DIAG_SET in ("Q", "S", "V") else [1, 8, 16] if DIAG_SET in ("W", "X") else [1, 4, 8, 16] if DIAG_SET in ("R", "T") else [32, 128] if DIAG_SET == "U" else [1, 8] if DIAG_SET == "J" else [1] if DIAG_SET == "K" else _WGS_SHAPES if DIAG_SET in ("D", "F", "G", "I", "L", "M", "N", "O", "P") else [32, 128] if DIAG_SET == "H" else [1, 16, 32, 128, 256] if DIAG_SET == "E" else [1, 8, 16]
DIAG_SEED = 1234
# (bs, touched local experts) cases: the input seed is searched (DIAG_SEED + bs + 1000 k) for the judge's typical draw
# (tools/moe_routing_stats.py, 60 seeds: bs1 p10/p50/p90 2/3/3, bs4 4/6/8, bs8 5/8/11); None: the fixed seed
DIAG_CASES = [(1, 2), (1, 3), (8, 8)] if DIAG_SET == "V" else [(1, 2), (1, 3), (8, 8), (16, None)] if DIAG_SET in ("W", "X") else [(1, 2), (1, 3), (4, 6), (8, 8)] if DIAG_SET in ("Q", "S") else [(1, 2), (1, 3), (4, 6), (8, 8), (16, None)] if DIAG_SET in ("R", "T") else [(bs, None) for bs in DIAG_SHAPES]
DIAG_BW = DIAG_SET == "A"  # also run the bandwidth microbenchmarks
DIAG_BW_MODES = (8, 16, 17, 19, 24)


ALT_SRCS = {}
MAIN_NS = 3


def _ext(defines=None):
    defines = tuple(MAIN_DEFINES if defines is None else defines)
    if DQ7 and "-DOKC_DQ7=1" not in defines:
        defines = defines + ("-DOKC_DQ7=1",)
    if defines not in _EXTS:
        src = CUDA_SRC
        defs = [d for d in defines if not d.startswith("@")]
        if torch.cuda.get_device_capability()[0] < 9:  # small-smem dev GPUs: keep the ring at 1 slot
            defs = [d for d in defs if not d.startswith("-DOKC_NS=") and not d.startswith("-DOKC_NSF=")]
        for d in defines:
            if d.startswith("@"):
                src = ALT_SRCS[d[1:]]
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}" + ("a" if major == 9 else "")
        import hashlib  # short unique name: a ~190-char module name produced a .so without its PyInit symbol
        tag = hashlib.sha1("_".join(defines).encode()).hexdigest()[:16]
        _EXTS[defines] = load_inline(
            name=f"okc_moe_w4a16_v216x_{tag}_sm{arch}",
            cpp_sources=CPP_SRC,
            cuda_sources=src,
            functions=["okc_run", "okc_set_pdl", "okc_run_logits"],
            extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-lineinfo",
                               *([] if any(d.startswith("-DOKC_NS=") for d in defs) else [f"-DOKC_NS={MAIN_NS if major >= 9 else 1}"]),
                               f"-DOKC_FNST={4 if major >= 9 else 3}",
                               *([] if any(d.startswith("-DOKC_NSF=") for d in defs) else [f"-DOKC_NSF={3 if major >= 9 else 1}"]), *defs],
            verbose=False,
        )
        _EXTS[defines].okc_set_pdl(1 if PDL else 0)
    return _EXTS[defines]


# Bit layout of a 32-bit word holding 8 E2M1 codes for one MMA k-step of one lane: register R (0..3) is
# (h, j0) = (0,0) (1,0) (0,2) (1,2): rows g+8h, k = base + 4s + j0 (+1 for the upper bf16). Code bits (m, e0, e1, s)
# land at output bits (6, 7, 8, 15) (+16 for the upper half) after the per-register rotation used by dq4().
_REG_HJ = [(0, 0), (1, 0), (0, 2), (1, 2)]


# DQ7: a bit layout whose decode is 7 ALU ops + 2 IMAD.SHL per 8 weights (the kernel gets -DOKC_DQ7=1)
DQ7 = True


def _stored_pos():
    pos = {}
    for R in range(4):
        for up in range(2):
            outb = {0: 6 + 16 * up, 1: 7 + 16 * up, 2: 8 + 16 * up, 3: 15 + 16 * up}  # code bit -> output bit
            for b, o in outb.items():
                if DQ7:
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


_POS = _stored_pos()


def _pack_codes(codes, rows):
    """codes uint8 [R0, K/2] (one expert), rows: LongTensor row order (R). Returns int32 words
    laid out [rb][w][c][g][t][s] (uint4 per lane per chunk)."""
    c = codes.index_select(0, rows)
    R = c.shape[0]
    nib = torch.stack((c & 0xF, c >> 4), -1).reshape(R, -1)  # [R, K]
    nb = R // 16
    n = nib.view(nb, 2, 8, 8, 8, 4, 4, 4)  # rb, h, g, w, c, t, s, j
    n = n.permute(0, 3, 4, 2, 5, 6, 1, 7).to(torch.int64)  # rb, w, c, g, t, s, h, j
    word = torch.zeros(n.shape[:-2], dtype=torch.int64, device=codes.device)
    for Rg, (h, j0) in enumerate(_REG_HJ):
        for up in range(2):
            code = n[..., h, j0 + up]
            for b in range(4):
                word |= ((code >> b) & 1) << _POS[(Rg, up, b)]
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    return word.contiguous().view(-1)


def _pack_scales(sf, rows):
    """sf fp8 [R0, K/16] -> bytes [rb][w][g][t][c][h]."""
    s = sf.view(torch.uint8).index_select(0, rows)
    R = s.shape[0]
    nb = R // 16
    s = s.view(nb, 2, 8, 8, 8, 4)  # rb, h, g, w, c, t
    s = s.permute(0, 3, 2, 5, 4, 1)  # rb, w, g, t, c, h
    return s.contiguous().view(-1)


def _fat_order(codes, scales):
    """Packed per-expert codes (int32 [rb][w][c][lane][4]) and scales (uint8 [rb][w][lane][c][h]) -> fat order:
    [ub (256-row unit)][ks (16 k-stages of 256)][32 KB codes [rb16][cc][lane][16 B] + 4 KB scales [rb16][lane][cc][h]]."""
    nub = codes.numel() // (16 * 8 * 8 * 32 * 4)
    c = codes.view(nub, 16, 8, 2, 4, 32, 4).permute(0, 2, 3, 1, 4, 5, 6).contiguous()
    c = c.view(torch.uint8).view(nub, 16, 32768)
    s = scales.view(nub, 16, 8, 32, 2, 4, 2).permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(nub, 16, 4096)
    return torch.cat([c, s], dim=2).contiguous().view(-1)


def _unit_order(codes, scales):
    """Per 16-row block: its 32 KB of codes followed by its 4 KB of scales (one TMA copy per skinny unit)."""
    nblk = codes.numel() // (8 * 8 * 32 * 4)
    c = codes.contiguous().view(torch.uint8).view(nblk, 32768)
    s = scales.contiguous().view(nblk, 4096)
    return torch.cat([c, s], dim=1).contiguous().view(-1)


def _pack_codes_nat(codes, rows):
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
                word |= ((code >> b) & 1) << _POS[(Rg, up, b)]
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    return word.contiguous().view(-1)


def _wgs_copy(codes, sf, rows):
    """WGS copy (RS-wgmma tiny band, moe -DOKC_WGS=1): [256-row unit][stage (16)][warpgroup q (4)][9216 B]. Warpgroup q
    owns rows 64q..64q+63 of the unit; stage st holds k16 steps st*16 + (0..15): codes [grp (4)][warp (4)][lane (32)]
    [4 words] (word (grp, j) = k16 step st*16 + grp*4 + j, natural mma k order), then scales [grp][warp][g][j][h] (the kernel's OKC_WGSSC2 layout)."""
    wn = _pack_codes_nat(codes, rows)
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
MAIN_FALLBACK_DEFINES = ("-DOKC_CPF=0", "-DOKC_CPFF=0", "-DOKC_DYNK=7", "-DOKC_EARLYN=2", "-DOKC_EPI4=0", "-DOKC_FCOSTB=35",
                         "-DOKC_FCOSTBB=36", "-DOKC_FINCOL=0", "-DOKC_FPF=5")


def _main_defines(sfx):
    if not sfx and "-DOKC_WGS=1" in tuple(MAIN_DEFINES):
        return MAIN_FALLBACK_DEFINES
    return MAIN_DEFINES


def _wgs_on():
    return any("-DOKC_WGS=1" in tuple(d) for d in (MAIN_DEFINES, TINY_DEFINES, TINY2_DEFINES, MID_DEFINES))


def prepare(weights):
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
        wA.append(_pack_codes(weights["w13"][e], rowsA))
        sA.append(_pack_scales(weights["w13_sf"][e], rowsA))
        wB.append(_pack_codes(weights["w2"][e], rowsB))
        sB.append(_pack_scales(weights["w2_sf"][e], rowsB))
    lut = (torch.arange(256, dtype=torch.int32, device=dev).to(torch.uint8).view(torch.float8_e4m3fn).float()
           * 2.0**64).to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    lut = (lut | (lut << 16)).to(torch.int32)
    state = {
        "w": [torch.cat(wA), torch.cat(sA), weights["w13_gs"][lo:lo + cnt].contiguous(),
              torch.cat(wB), torch.cat(sB), weights["w2_gs"][lo:lo + cnt].contiguous(),
              lut.contiguous(), weights["post_norm_w"].contiguous(),  # bf16 as given; the kernel forms 1 + w in FP32
              torch.cat([_fat_order(a, b) for a, b in zip(wA, sA)]), torch.cat([_fat_order(a, b) for a, b in zip(wB, sB)]),
              torch.cat([_unit_order(a, b) for a, b in zip(wA, sA)]), torch.cat([_unit_order(a, b) for a, b in zip(wB, sB)])],
        "lo": lo, "cnt": cnt, "scratch": _make_scratch(dev),
        "grid": torch.cuda.get_device_properties(dev).multi_processor_count,
    }
    if _wgs_on():  # w[12], w[13]: the WGS copies (one more copy of the weights, laid out per 64-row unit)
        state["w"].append(torch.cat([_wgs_copy(weights["w13"][e], weights["w13_sf"][e], rowsA) for e in range(lo, lo + cnt)]))
        state["w"].append(torch.cat([_wgs_copy(weights["w2"][e], weights["w2_sf"][e], rowsB) for e in range(lo, lo + cnt)]))
    prmt = PRMT_SCALES and _scales_prmt_exact(weights["w13_sf"][lo:lo + cnt], weights["w2_sf"][lo:lo + cnt])
    state["sfx"] = ("-DOKC_PRMTSC=1",) if prmt else ()
    mods = [tuple(_main_defines(state["sfx"])) + state["sfx"]]
    for d in (TINY_DEFINES, TINY2_DEFINES, MID_DEFINES):
        if d and (state["sfx"] or "-DOKC_WGS=1" not in tuple(d)):  # the WGS build needs the PRMT scale decode
            mods.append(tuple(d) + state["sfx"])
    state["wts"] = torch.zeros(EMAX, dtype=torch.float32, device=dev)  # in-kernel routing: the route weights (CTA 0)
    state["norsp"] = torch.empty(0, dtype=torch.float32, device=dev)  # (no pre-expert RMS partials)
    if ROUTE_LOGITS_BUILDS or ROUTE_LOGITS_SELFTEST:
        mods += [_logits_defines(d) for d in list(mods)]
    extra = []
    if DIAG:  # the diagnostic variants compile alongside (one pool; nvcc runs in subprocesses); a variant that fails
        for v in DIAG_VARIANTS.values():  # to build only loses its column
            d = tuple(v["defines"] if isinstance(v, dict) else v) + state["sfx"]
            if d not in mods and d not in extra:
                extra.append(d)

    def _build(d):
        try:
            _ext(d)
        except Exception as ex:
            if d in mods:
                raise
            print(f"[diag] variant {d} failed to build: {str(ex)[-300:]}", flush=True)

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_build, mods + extra))
    torch.cuda.synchronize()
    if ROUTE_LOGITS_SELFTEST:
        try:
            _route_logits_selftest(state)
        except Exception as ex:  # a self-test must never break the run
            print(f"[route-logits] self-test failed: {ex}", flush=True)
    if DIAG:
        _diag(state, weights)
    return state


def _logits_defines(d):
    return tuple(d) + ("-DOKC_ROUTE_LOGITS=1",)


def _select_ext(M, sfx, logits=False):
    """The build for M tokens (the shipped policy), optionally its OKC_ROUTE_LOGITS variant."""
    if TINY_DEFINES and M <= TINY_MAX_M:
        d = tuple(TINY_DEFINES) + sfx
    elif TINY2_DEFINES and M <= TINY2_MAX_M and (sfx or "-DOKC_WGS=1" not in tuple(TINY2_DEFINES)):
        d = tuple(TINY2_DEFINES) + sfx  # (the WGS build needs the PRMT scale decode: else MID/MAIN)
    elif MID_DEFINES and M <= MID_MAX_M:
        d = tuple(MID_DEFINES) + sfx
    else:
        d = tuple(_main_defines(sfx)) + sfx
    return _ext(_logits_defines(d) if logits else d)


def kernel_fn_logits(data, state):
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
    ext = _select_ext(M, state.get("sfx", ()), logits=True)
    for m0 in range(0, M, MMAX):
        m1 = min(M, m0 + MMAX)
        ext.okc_run_logits(state["w"], state["scratch"], x[m0:m1], logits[m0:m1], bias, ls, out[m0:m1], gid[m0:m1],
                           state["wts"], rsp[m0:m1] if rsp.numel() else rsp, cnt, state["grid"])
    return out, gid


def _route_logits_selftest(state):
    """In-kernel routing vs the host selection: synthetic logits whose top-8 reproduce _gen_input's ids (so the MoE
    output can be compared with kernel_fn on the same routes), random logits (ids / weights vs torch), timing."""
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(4321)
    bias = (0.05 * torch.randn((256,), generator=gen, device=dev)).float().contiguous()
    ls = torch.tensor([0], dtype=torch.int32, device=dev)
    for bs in (1, 4, 8, 16, 32, 33, 128, 256):
        d = _gen_input(bs, 777 + bs, dev)
        ids = d["topk_ids"].long()
        # logits: the given ids get large, slot-ordered logits (their biased scores dominate every other expert)
        logits = 0.5 * torch.randn((bs, 256), generator=gen, device=dev)
        logits.scatter_(1, ids, (20.0 - 0.5 * torch.arange(8, device=dev, dtype=torch.float32)).expand(bs, 8))
        logits = logits.float().contiguous()
        sc = torch.sigmoid(logits * ROUTER_MULTIPLIER)
        sel = sc + bias
        ref_ids = sel.topk(8, dim=-1).indices
        picked = sc.gather(1, ref_ids)
        ref_w = picked / (picked.sum(-1, keepdim=True) + 1e-15)
        out2, gid = kernel_fn_logits({"hidden_states": d["hidden_states"], "router_logits": logits, "router_bias": bias,
                                      "local_start": ls, "config": {"local_count": 32}}, state)
        torch.cuda.synchronize()
        kw = state["wts"][:bs * 8].view(bs, 8)
        same_ids = bool((gid.long() == ref_ids).all())
        same_set = bool((gid.long().sort(-1).values == ids.sort(-1).values).all())
        werr = float((kw - ref_w).abs().max())
        # the MoE output with the same routes through the plain entry (host ids / weights)
        lids = ((gid.long() - 0) % 256)
        lids = torch.where(lids < 32, lids, torch.full_like(lids, -1)).to(torch.int32)
        out1 = kernel_fn({"hidden_states": d["hidden_states"], "topk_ids": gid, "topk_weights": kw.contiguous(),
                          "config": {"local_start": 0, "local_count": 32}}, state)
        torch.cuda.synchronize()
        oerr = float((out2.float() - out1.float()).norm() / out1.float().norm().clamp(min=1e-20))
        # per-token pre-expert RMS scale in the kernel (partials) vs the scale applied to x on the host
        rerr = float("nan")
        if bs <= 256:
            pooled = 3.0 * torch.randn((bs, 4096), generator=gen, device=dev)
            part = (pooled * pooled).view(bs, 128, 32).sum(-1).contiguous()
            rs = torch.rsqrt(part.sum(-1, keepdim=True) / 4096.0 + 1e-5)
            o_k, _ = kernel_fn_logits({"hidden_states": pooled.to(torch.bfloat16), "router_logits": logits, "router_bias": bias,
                                       "local_start": ls, "pre_norm_partials": part, "config": {"local_count": 32}}, state)
            o_h, _ = kernel_fn_logits({"hidden_states": (pooled * rs).to(torch.bfloat16), "router_logits": logits,
                                       "router_bias": bias, "local_start": ls, "config": {"local_count": 32}}, state)
            torch.cuda.synchronize()
            rerr = float((o_k.float() - o_h.float()).norm() / o_h.float().norm().clamp(min=1e-20))
        # random logits (no structure): ids vs torch's top-8 (ties improbable), weights
        logits2 = (3.0 * torch.randn((bs, 256), generator=gen, device=dev)).float().contiguous()
        sc2 = torch.sigmoid(logits2 * ROUTER_MULTIPLIER)
        ref2 = (sc2 + bias).topk(8, dim=-1).indices
        _, gid2 = kernel_fn_logits({"hidden_states": d["hidden_states"], "router_logits": logits2, "router_bias": bias,
                                    "local_start": ls, "config": {"local_count": 32}}, state)
        torch.cuda.synchronize()
        nd = int((gid2.long() != ref2).any(-1).sum())
        # timing: in-kernel routing vs the plain entry (graph replays, cold L2)
        tw = ref_w.float().contiguous()

        def f_plain():
            kernel_fn({"hidden_states": d["hidden_states"], "topk_ids": gid, "topk_weights": tw,
                       "config": {"local_start": 0, "local_count": 32}}, state)

        def f_logits():
            kernel_fn_logits({"hidden_states": d["hidden_states"], "router_logits": logits, "router_bias": bias,
                              "local_start": ls, "config": {"local_count": 32}}, state)

        t_plain, _ = _time_graph(f_plain)
        t_logits, _ = _time_graph(f_logits)
        print(f"[route-logits] bs={bs:3d}: ids == host top-8 {same_ids} (same set as the input routing {same_set}), "
              f"max |w - w_ref| {werr:.1e}, out rel L2 vs plain entry {oerr:.1e}, rs-in-epilogue vs host-scaled x {rerr:.1e}, "
              f"random-logit tokens differing {nd}/{bs}, "
              f"time plain {t_plain:6.1f} us vs in-kernel routing {t_logits:6.1f} us", flush=True)


MMAX = 384
EMAX = MMAX * 8
JMAX = 32 + EMAX // 64 + 8


def _wgs_fixf():
    """moe WGS_FIXF: stream-K fixups + doneAs after ybuf (zero between calls); unit-indexed fixup slots (JMAX * 32) only
    when a selected build uses OKC_WGSDYN, else the 160 CTA-indexed slots."""
    sets = [MAIN_DEFINES, TINY_DEFINES, TINY2_DEFINES, MID_DEFINES]
    if DIAG:
        sets += [v["defines"] if isinstance(v, dict) else v for v in DIAG_VARIANTS.values()]
    slots = JMAX * 32 if any("-DOKC_WGSDYN=1" in tuple(d) for d in sets) else 160
    return 2 * slots * 4 * 4096 + 2 * slots * 4 + 136 * 16


WGS_FIXF = _wgs_fixf()
WGS_XROWS = EMAX + 8 * (32 + EMAX // 32 + 8)  # >= moe WGS_XROWS for FT_T >= 32


def _make_scratch(dev):
    return [torch.zeros(8 + 1024 + EMAX, dtype=torch.int32, device=dev),  # counters + doneA (>= any build's JMAX) + ss copy 1
            torch.zeros(EMAX + EMAX * 4096 + WGS_FIXF, dtype=torch.float32, device=dev),
            torch.zeros((EMAX + 2 * WGS_XROWS) * 4096, dtype=torch.bfloat16, device=dev)]  # inter, then WGS xA / xB


def kernel_fn(data, state):
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
    if TINY_DEFINES and M <= TINY_MAX_M:
        ext = _ext(tuple(TINY_DEFINES) + sfx)
    elif TINY2_DEFINES and M <= TINY2_MAX_M and (sfx or "-DOKC_WGS=1" not in tuple(TINY2_DEFINES)):
        ext = _ext(tuple(TINY2_DEFINES) + sfx)  # (the WGS build needs the PRMT scale decode: else MID/MAIN)
    elif MID_DEFINES and M <= MID_MAX_M:
        ext = _ext(tuple(MID_DEFINES) + sfx)
    else:
        ext = _ext(tuple(_main_defines(sfx)) + sfx)
    for m0 in range(0, M, MMAX):
        m1 = min(M, m0 + MMAX)
        ext.okc_run(state["w"], state["scratch"], x[m0:m1], ids[m0:m1], tw[m0:m1], out[m0:m1],
                    state["lo"], state["cnt"], state["grid"])
    return out


# ---------------------------------------------------------------------------------------------- diagnostics
def _gen_input(bs, seed, dev):
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


def _time_graph(fn, reps=40):
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
        flush.view(torch.int32).sum()  # read flush, as the judge
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


def _diag_bw(state):
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}" + ("a" if major == 9 else "")
    ext = load_inline(name=f"okc_bw_v216x_sm{arch}", cpp_sources="#include <torch/extension.h>\nvoid bw_run(torch::Tensor src, torch::Tensor out, int64_t mode, int64_t grid);",
                      cuda_sources=BW_SRC, functions=["bw_run"],
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
    for mode in DIAG_BW_MODES:
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


def _diag(state, weights):
    """Untimed development table (DIAG): every variant in DIAG_VARIANTS is compiled (concurrently), checked against the
    shipped build on the same input (relative L2) and timed as a CUDA graph replay with a cold L2, per shape."""
    import time
    dev = weights["w13"].device
    t0 = time.time()
    if DIAG_BW:
        try:
            _diag_bw(state)
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
    shapes = list(DIAG_SHAPES)
    variants = {}  # name -> (defines, shapes)
    for name, v in DIAG_VARIANTS.items():
        if isinstance(v, dict):
            variants[name] = (tuple(v["defines"]) + sfx, list(v.get("shapes") or shapes))
        else:
            variants[name] = (tuple(v) + sfx, shapes)
    # compile the variants concurrently (separate extension dirs; nvcc runs in subprocesses)
    import concurrent.futures
    todo = []
    for d, _ in variants.values():
        if d not in todo and (d + (("-DOKC_DQ7=1",) if DQ7 and "-DOKC_DQ7=1" not in d else ())) not in _EXTS:
            todo.append(d)
    def _try_ext(d):
        try:
            _ext(d)
        except Exception as ex:
            print(f"[diag] variant {d} failed to build: {str(ex)[-200:]}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_try_ext, todo))
    print(f"[diag] compiled {len(todo)} extra variants in {time.time() - t0:.0f}s", flush=True)

    def _built(d):  # (an unbuilt variant only loses its column)
        return (d + (("-DOKC_DQ7=1",) if DQ7 and "-DOKC_DQ7=1" not in d else ())) in _EXTS
    def _case_input(bs, target):
        for k in range(64 if target is not None else 1):
            d = _gen_input(bs, DIAG_SEED + bs + 1000 * k, dev)
            ids = d["topk_ids"]
            touched = int(torch.unique(ids[(ids >= 0) & (ids < 32)]).numel())
            if target is None or touched == target:
                break
        return d, touched

    for bs, target in DIAG_CASES:
        d, touched = _case_input(bs, target)
        ids = d["topk_ids"]
        ref = kernel_fn(d, state)  # the shipped policy's output for this input
        torch.cuda.synchronize()
        row = [f"bs={bs:3d} touched={touched:2d}"]
        for name, (defs, vshapes) in variants.items():
            if bs not in vshapes or not _built(defs):
                continue
            try:
                ext = _ext(defs)
                x = d["hidden_states"]
                idt = ids.to(torch.int32).contiguous()
                tw = d["topk_weights"].float().contiguous()
                out = torch.empty((bs, 4096), dtype=torch.bfloat16, device=dev)

                def fn():
                    ext.okc_run(state["w"], state["scratch"], x, idt, tw, out, 0, 32, state["grid"])

                fn()
                torch.cuda.synchronize()
                err = float((out.float() - ref.float()).norm() / ref.float().norm().clamp(min=1e-20))
                med, mn = _time_graph(fn)
                row.append(f"{name}: {med:7.1f}us ({touched * 28.31 / med:4.2f} TB/s, min {mn:6.1f}, err {err:.1e})")
            except Exception as ex:
                row.append(f"{name}: FAILED {str(ex)[:80]}")
                torch.cuda.synchronize()
        print("[diag] " + "  ".join(row), flush=True)
    # timeline of the debug variants (-DOKC_DEBUG=1): per-CTA %globaltimer stamps, min-max over CTAs
    for dbgname in [k for k, (v, _) in variants.items() if "-DOKC_DEBUG=1" in v and _built(v)]:
        defs, vshapes = variants[dbgname]
        ext = _ext(defs)
        G = state["grid"]
        dbg = torch.zeros(G * 17 + G * 8 * 4, dtype=torch.int64, device=dev)
        sc = list(state["scratch"]) + [dbg]
        names = ["start", "routed", "A done", "B done", "barrier", "final", "ranges", "gathered"]
        tiny = "-DOKC_WS=1" in defs and "-DOKC_WGS=1" not in defs
        for bs, target in DIAG_CASES:
            if bs not in vshapes:
                continue
            d, touched = _case_input(bs, target)
            x = d["hidden_states"]
            idt = d["topk_ids"].to(torch.int32).contiguous()
            tw = d["topk_weights"].float().contiguous()
            out = torch.empty((bs, 4096), dtype=torch.bfloat16, device=dev)
            flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=dev)
            for _ in range(3):
                flush.view(torch.int32).sum()
                dbg.zero_()
                torch.cuda.synchronize()
                ext.okc_run(state["w"], sc, x, idt, tw, out, 0, 32, G)
                torch.cuda.synchronize()
            tt = dbg[:G * 8].view(G, 8).cpu().double()
            t00 = tt[:, 0][tt[:, 0] > 0].min()
            parts = []
            if tiny:  # per-CTA stamps, min / median / max (us from the first CTA's start), then per-CTA waits
                try:
                    pr = dbg[G * 13:G * 17].view(G, 4).cpu().double()
                    xt = dbg[G * 8:G * 12].view(G, 4).cpu().double()
                    cols = [("start", tt[:, 0]), ("1st issue", pr[:, 0]), ("routed", tt[:, 1]), ("ranges", tt[:, 6]),
                            ("A 1st landed", tt[:, 7]), ("A last landed", xt[:, 3]), ("A done", tt[:, 2]),
                            ("B 1st issue", pr[:, 3]), ("B gate", pr[:, 2]), ("B 1st landed", pr[:, 1]), ("B done", tt[:, 3]),
                            ("helper done", xt[:, 0]), ("bar arrive", xt[:, 1]), ("barrier", tt[:, 4]), ("end", tt[:, 5])]
                    def mmm(col):
                        col = col[col > 0]
                        if not col.numel():
                            return "-"
                        col = (col - t00) / 1000
                        return f"{col.min():.1f}/{col.median():.1f}/{col.max():.1f}"
                    def dmmm(a, b):  # per-CTA difference b - a where both stamps exist
                        m = (a > 0) & (b > 0)
                        if not int(m.sum()):
                            return "-"
                        dd = (b[m] - a[m]) / 1000
                        return f"{dd.min():.1f}/{dd.median():.1f}/{dd.max():.1f}"
                    print(f"[diag] stamps {dbgname} bs={bs} touched={touched} (us from the first CTA start, min/med/max over CTAs): "
                          + "  ".join(f"{n} {mmm(c)}" for n, c in cols), flush=True)
                    print(f"[diag] waits  {dbgname} bs={bs}: issue->A 1st landed {dmmm(pr[:, 0], tt[:, 7])}  A done->B gate {dmmm(tt[:, 2], pr[:, 2])}"
                          f"  B gate->B 1st landed {dmmm(pr[:, 2], pr[:, 1])}  B done->helper done {dmmm(tt[:, 3], xt[:, 0])}"
                          f"  bar arrive->barrier {dmmm(xt[:, 1], tt[:, 4])}  barrier->end {dmmm(tt[:, 4], tt[:, 5])}", flush=True)
                    xu = dbg[G * 17:G * 17 + G * 8].view(G, 8).cpu().double()
                    def us(col):  # accumulated ns -> us, min/med/max over CTAs
                        col = col / 1000
                        return f"{col.min():.1f}/{col.median():.1f}/{col.max():.1f}"
                    print(f"[diag] helper {dbgname} bs={bs}: 1st epilogue {mmm(xu[:, 3])}  credit first {mmm(xu[:, 0])}  credit last {mmm(xu[:, 1])}"
                          f"  gate job2 {mmm(xu[:, 2])}  | thread-0 waits (us): A ring {us(xu[:, 6])}  A helper {us(xu[:, 4])}"
                          f"  B ring {us(xu[:, 7])}  B helper {us(xu[:, 5])}", flush=True)
                    xv = dbg[G * 25:G * 25 + G * 8].view(G, 8).cpu().double()
                    print(f"[diag] helper2 {dbgname} bs={bs}: ready {mmm(xv[:, 5])}  unit3 handoff {mmm(xv[:, 2])}  unit3 epilogue end {mmm(xv[:, 1])}"
                          f"  (lag {dmmm(xv[:, 2], xv[:, 1])})  | helper (us): fence {us(xv[:, 0])}  A wait partials {us(xv[:, 3])}  A epilogues {us(xv[:, 4])}"
                          f"  B wait partials {us(xv[:, 6])}  B epilogues {us(xv[:, 7])}", flush=True)
                except Exception as ex:
                    print(f"[diag] stamps {dbgname} bs={bs} failed: {ex}", flush=True)
            for k, n in enumerate(names):
                col = tt[:, k][tt[:, k] > 0]
                if col.numel():
                    parts.append(f"{n} {(col.min() - t00) / 1000:.1f}-{(col.max() - t00) / 1000:.1f}")
            if "-DOKC_WGS=1" in defs:  # prologue stamps (us from start, min-max over CTAs): jrow / gather copies / gather release / ranges
                pr = dbg[G * 13:G * 17].view(G, 4).cpu().double()
                pp = []
                for k, n in enumerate(["jrow", "gath", "rel", "ranges"]):
                    col = pr[:, k][pr[:, k] > 0]
                    if col.numel():
                        pp.append(f"{n} {(col.min() - t00) / 1000:.1f}-{(col.max() - t00) / 1000:.1f}")
                parts.append("prologue[" + " ".join(pp) + "]")
            if "-DOKC_WGS=1" in defs:  # producer stall counters (us): x gating phase A / B, phase-B stalls > 0.5 us, x-ring empty, weight-ring empty
                xs_ = dbg[G * 8:G * 12].view(G, 4).cpu().double() / 1000
                ws_ = dbg[G * 12:G * 13].cpu().double() / 1000
                parts.append(f"xgateA mean/max {xs_[:, 0].mean():.1f}/{xs_[:, 0].max():.1f}  xgateB {xs_[:, 1].mean():.1f}/{xs_[:, 1].max():.1f}"
                             f" (stalls {xs_[:, 2].mean() * 1000:.0f}/{xs_[:, 2].max() * 1000:.0f})  xring {xs_[:, 3].mean():.1f}/{xs_[:, 3].max():.1f}  wring {ws_.mean():.1f}/{ws_.max():.1f}")
            print(f"[diag] timeline {dbgname} bs={bs}: " + "  ".join(parts), flush=True)
    print(f"[diag] total {time.time() - t0:.0f}s", flush=True)
