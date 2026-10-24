# okc: problem=muse_moe_w4a16
"""Routed NVFP4 experts (W4A16) for Muse Spark Open: four launches, no host sync.

  1. _route_kernel      one CTA sorts the (token, slot) pairs by local expert (tl.sort), pads
                        each expert's rows to BLOCK_M and writes the block list; the other CTAs
                        convert x to fp16 once (exact for BF16 values).
  2. _moe_gemm_kernel   gate/up GEMM + SiLU: persistent CTAs walk (block, n-tile) work items,
                        gather the block's x rows, stream the expert's packed E2M1 tile and
                        dequantize it in registers, tensor-core dot with fp32 accumulation.
  3. _moe_gemm_kernel   down GEMM: inter @ W2^T -> y (BF16).
  4. _finalize_kernel   per token: RMSNorm(y * (1 + w)) per local route, weighted sum.

Dequantization: an E2M1 nibble c = (s, e1, e0, m) placed into an fp16 as
  bits = s << 15 | (e1 e0) << 10 | m << 9
is exactly value(c) * 2^-14 for every code (subnormals included), so the 2^14 is folded into the
expert's global scale once in prepare() and the tile needs only integer ops and one fp16
multiply per element. Weights are used at their stored precision (E2M1 x E4M3 block scale x FP32
global scale); only the fp16 product rounds (11 bits, finer than BF16).
"""

from typing import TypedDict

import torch
import triton
import triton.language as tl


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


# == BEGIN SHARED moe ==
TOPK = 8
LOCAL = 32
POST_NORM_EPS = 1e-8
MAX_M = 1024
NUM_SMS = 132
MOE_OUT_DTYPE = torch.bfloat16
MOE_BACKEND = "marlin"   # "marlin": SGLang's W4A16 GEMMs with this file's routing/finalize; "triton": this file's GEMMs
DEQUANT_ASM = True          # inline PTX for the nibble -> fp16 placement (False: plain Triton ops)
SCALE_3D = tl.constexpr(False)   # scale expansion: 3-D broadcast instead of 4 interleaves
SCALE_VARIANT = tl.constexpr(4)  # scaled-asm dequant: 2 = interleave-expanded scales, 3 = broadcast-reshape
SCALE_FIX = 2.0 ** 14       # folded into the global scales: the placement trick yields value * 2^-14


# ----------------------------------------------------------------------------------------------
# 1. routing
# ----------------------------------------------------------------------------------------------


@triton.jit
def _route_kernel(
    ids_ptr, x_ptr, x16_ptr,
    sorted_tok_ptr, row_of_pair_ptr, blk_e_ptr, blk_row0_ptr, blk_end_ptr, nblk_ptr,
    sorted_pair_ptr, lids_ptr, npad_ptr,
    M, local_start, n_elems, n_route_ctas,
    PB: tl.constexpr, CH: tl.constexpr, BM: tl.constexpr, MAXB: tl.constexpr, E: tl.constexpr,
    TOPK: tl.constexpr, XBLOCK: tl.constexpr, MARLIN: tl.constexpr,
):
    """Sort-free routing. CTA c < n_route_ctas owns pairs [c*PB, (c+1)*PB): each pair's rank among
    the pairs of its expert (how many earlier pairs route there) gives its padded row; every CTA
    recounts the per-expert totals (cheap), CTA 0 also writes the block list. The remaining
    CTAs convert x to fp16 (exact for BF16 values)."""
    pid = tl.program_id(0)
    if pid >= n_route_ctas:
        offs = (pid - n_route_ctas) * XBLOCK + tl.arange(0, XBLOCK)
        m = offs < n_elems
        v = tl.load(x_ptr + offs, mask=m, other=0.0)
        tl.store(x16_ptr + offs, v.to(tl.float16), mask=m)
        return
    npairs = M * TOPK
    ex = tl.arange(0, E)
    i = pid * PB + tl.arange(0, PB)
    valid = i < npairs
    e = tl.load(ids_ptr + i, mask=valid, other=0) - local_start
    is_local = valid & (e >= 0) & (e < E)
    e_key = tl.where(is_local, e, E)                       # E: not this GPU's
    counts = tl.zeros((E,), dtype=tl.int32)
    rank = tl.zeros((PB,), dtype=tl.int32)
    for c0 in range(0, npairs, CH):
        ci = c0 + tl.arange(0, CH)
        cv = ci < npairs
        ce = tl.load(ids_ptr + ci, mask=cv, other=0) - local_start
        cl = cv & (ce >= 0) & (ce < E)
        ce_key = tl.where(cl, ce, E)
        counts += tl.sum((ce_key[None, :] == ex[:, None]).to(tl.int32), 1)
        rank += tl.sum(((ce_key[None, :] == e_key[:, None]) & (ci[None, :] < i[:, None])).to(tl.int32), 1)
    padded = (counts + BM - 1) // BM * BM
    pad_off = tl.cumsum(padded, 0) - padded                 # padded row offset per expert
    onehot = (e_key[:, None] == ex[None, :]).to(tl.int32)   # [PB, E]
    row = tl.sum(onehot * pad_off[None, :], 1) + rank
    tl.store(sorted_tok_ptr + row, i // TOPK, mask=is_local)
    tl.store(row_of_pair_ptr + i, tl.where(is_local, row, -1), mask=valid)
    if MARLIN:
        # Marlin's moe_align_block_size format: pair indices grouped by expert, padding = npairs
        tl.store(sorted_pair_ptr + row, i, mask=is_local)
        tl.store(lids_ptr + i, tl.where(is_local, e, -1), mask=valid)
    if pid == 0:
        nb = padded // BM
        blk_off = tl.cumsum(nb, 0) - nb
        total = tl.sum(nb, 0)
        j = tl.arange(0, MAXB)
        ej = tl.sum(((blk_off + nb)[None, :] <= j[:, None]).to(tl.int32), 1)     # expert of block j
        jm = j < total
        oh = (ej[:, None] == ex[None, :]).to(tl.int32)
        bo = tl.sum(oh * blk_off[None, :], 1)
        po = tl.sum(oh * pad_off[None, :], 1)
        ce2 = tl.sum(oh * counts[None, :], 1)
        tl.store(blk_e_ptr + j, ej + local_start, mask=jm)
        tl.store(blk_row0_ptr + j, po + (j - bo) * BM, mask=jm)
        tl.store(blk_end_ptr + j, po + ce2, mask=jm)
        tl.store(nblk_ptr, total)
        if MARLIN:
            tl.store(npad_ptr, total * BM)
            pj = tl.arange(0, BM)
            prow = (pad_off + counts)[:, None] + pj[None, :]                       # [E, BM]
            pm = pj[None, :] < (padded - counts)[:, None]
            tl.store(sorted_pair_ptr + prow, tl.full((E, BM), 0, tl.int32) + npairs, mask=pm)


# ----------------------------------------------------------------------------------------------
# 2./3. dequantizing GEMM
# ----------------------------------------------------------------------------------------------

# 4 packed bytes (8 nibbles) -> 8 fp16 (x 2^-14). Byte j holds elements 2j (low nibble) and
# 2j+1 (high nibble). Outputs, as Triton packs 4 fp16 into 2 registers in element order:
#   $0 = (e0, e2)  $1 = (e4, e6)   [low nibbles]     $2 = (e1, e3)  $3 = (e5, e7)   [high nibbles]
# prmt first spreads bytes (0, 1) and (2, 3) into the two 16-bit halves of a word, so one shift
# serves both halves: magnitude bits (e1 e0 m) land at fp16 bits [11:9], the sign at bit 15.
_NIBBLE_ASM = tl.constexpr("""
{
    .reg .b32 a, b, t, u, z;
    mov.b32 z, 0;
    prmt.b32 a, $4, z, 0x4140;
    prmt.b32 b, $4, z, 0x4342;
    shl.b32 t, a, 9;    and.b32 t, t, 0x0E000E00;
    shl.b32 u, a, 12;   and.b32 u, u, 0x80008000;  or.b32 $0, t, u;
    shl.b32 t, b, 9;    and.b32 t, t, 0x0E000E00;
    shl.b32 u, b, 12;   and.b32 u, u, 0x80008000;  or.b32 $1, t, u;
    shl.b32 t, a, 5;    and.b32 t, t, 0x0E000E00;
    shl.b32 u, a, 8;    and.b32 u, u, 0x80008000;  or.b32 $2, t, u;
    shl.b32 t, b, 5;    and.b32 t, t, 0x0E000E00;
    shl.b32 u, b, 8;    and.b32 u, u, 0x80008000;  or.b32 $3, t, u;
}
""")


_NIBBLE_SCALED_ASM = tl.constexpr("""
{
    .reg .b32 a, b, t, u, z;
    mov.b32 z, 0;
    prmt.b32 a, $4, z, 0x4140;
    prmt.b32 b, $4, z, 0x4342;
    shl.b32 t, a, 9;    and.b32 t, t, 0x0E000E00;
    shl.b32 u, a, 12;   and.b32 u, u, 0x80008000;  or.b32 t, t, u;   mul.f16x2 $0, t, $5;
    shl.b32 t, b, 9;    and.b32 t, t, 0x0E000E00;
    shl.b32 u, b, 12;   and.b32 u, u, 0x80008000;  or.b32 t, t, u;   mul.f16x2 $1, t, $5;
    shl.b32 t, a, 5;    and.b32 t, t, 0x0E000E00;
    shl.b32 u, a, 8;    and.b32 u, u, 0x80008000;  or.b32 t, t, u;   mul.f16x2 $2, t, $5;
    shl.b32 t, b, 5;    and.b32 t, t, 0x0E000E00;
    shl.b32 u, b, 8;    and.b32 u, u, 0x80008000;  or.b32 t, t, u;   mul.f16x2 $3, t, $5;
}
""")


@triton.jit
def _dequant_scaled(wq, s8, gs16, VARIANT: tl.constexpr):
    """Scale applied inside the nibble asm (packed f16x2), so the result keeps the asm output
    layout and feeds the MMA without a shared-memory round trip.
    wq uint8 [BN, BK/2]; s8: VARIANT 1 -> fp8 [BN, BK/2] (each group's scale gathered per byte),
    VARIANT 2 -> fp8 [BN, BK/16] expanded here by interleaves. gs16: fp16 global scale x 2^14."""
    if VARIANT == 1:
        s = _cache_scale_to_f16(s8) * gs16                       # [BN, BK/2] fp16
    elif VARIANT == 3:
        s = (s8.to(tl.float32).to(tl.float16)) * gs16            # [BN, BK/16]
        s = tl.reshape(tl.broadcast_to(s[:, :, None], (s.shape[0], s.shape[1], 8)), (s.shape[0], s.shape[1] * 8))
    else:   # 2 and 4 (4: the scales were loaded pair-blocked, see SF_BLOCKED)
        s = (s8.to(tl.float32).to(tl.float16)) * gs16            # [BN, BK/16]
        s = tl.interleave(s, s)
        s = tl.interleave(s, s)
        s = tl.interleave(s, s)                                  # [BN, BK/2]
    lo, hi = tl.inline_asm_elementwise(
        asm=_NIBBLE_SCALED_ASM, constraints="=r,=r,=r,=r,r,r,r", args=[wq, s],
        dtype=(tl.float16, tl.float16), is_pure=True, pack=4)
    return tl.interleave(lo, hi)


@triton.jit
def _cache_scale_to_f16(x8):
    """fp8 e4m3 -> fp16, one cvt per pair, keeping the input's layout."""
    b = x8.to(tl.uint8, bitcast=True)
    return tl.inline_asm_elementwise(
        asm="{ .reg .b16 lo, hi; mov.b32 {lo, hi}, $2; cvt.rn.f16x2.e4m3x2 $0, lo; cvt.rn.f16x2.e4m3x2 $1, hi; }",
        constraints="=r,=r,r", args=[b], dtype=tl.float16, is_pure=True, pack=4)


@triton.jit
def _dequant(wq, sf, gs, ASM: tl.constexpr, MODE: tl.constexpr = 0):
    """wq uint8 [BN, BK/2] packed E2M1, sf fp8 [BN, BK/16], gs fp32 (x 2^14) -> fp16 [BN, BK]."""
    if MODE == 5:
        lo = wq.to(tl.float16)
        hi = lo
    elif ASM:
        lo, hi = tl.inline_asm_elementwise(
            asm=_NIBBLE_ASM, constraints="=r,=r,=r,=r,r", args=[wq],
            dtype=(tl.float16, tl.float16), is_pure=True, pack=4)
    else:
        lo = (wq & 0x0F).to(tl.int16)
        hi = (wq >> 4).to(tl.int16)
        lo = (((lo & 7) << 9) | ((lo & 8) << 12)).to(tl.float16, bitcast=True)
        hi = (((hi & 7) << 9) | ((hi & 8) << 12)).to(tl.float16, bitcast=True)
    w = tl.interleave(lo, hi)                                   # [BN, BK]
    if MODE == 4:
        return w
    s = (sf.to(tl.float32) * gs).to(tl.float16)                 # [BN, BK/16]
    if MODE == 6 or SCALE_3D:
        # one scale per 16 consecutive K elements: broadcast over a [BN, BK/16, 16] view
        w3 = tl.reshape(w, (w.shape[0], w.shape[1] // 16, 16))
        return tl.reshape(w3 * s[:, :, None], (w.shape[0], w.shape[1]))
    s = tl.interleave(s, s)
    s = tl.interleave(s, s)
    s = tl.interleave(s, s)
    s = tl.interleave(s, s)                                     # [BN, BK]: each scale x16
    return w * s


@triton.jit
def _moe_gemm_kernel(
    a_ptr, w_ptr, sf_ptr, gs_ptr, out_ptr,
    sorted_tok_ptr, blk_e_ptr, blk_row0_ptr, blk_end_ptr, nblk_ptr,
    stride_am, stride_om, K, N, stride_we, stride_wn, stride_se, stride_sn, up_rows,
    NUM_CTAS, n_tiles,
    GATHER: tl.constexpr, GATED: tl.constexpr, ASM: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, MODE: tl.constexpr, SF_BLOCKED: tl.constexpr,
):
    # MODE (benchmark ablations; outputs are wrong unless 0): 1 = fp16 weights read from w_ptr
    # (no dequant), 2 = dequant but no dot, 3 = loads only
    pid = tl.program_id(0)
    nblk = tl.load(nblk_ptr)
    total = nblk * n_tiles
    offs_m = tl.arange(0, BM)
    offs_k = tl.arange(0, BK)
    offs_kb = tl.arange(0, BK // 2)
    offs_ks = tl.arange(0, BK // 16)
    for item in range(pid, total, NUM_CTAS):
        j = item // n_tiles
        nt = item % n_tiles
        e = tl.load(blk_e_ptr + j)
        row0 = tl.load(blk_row0_ptr + j)
        end = tl.load(blk_end_ptr + j)
        rows = row0 + offs_m
        rmask = rows < end
        if GATHER:
            arow = tl.load(sorted_tok_ptr + rows, mask=rmask, other=0)
        else:
            arow = rows
        offs_n = nt * BN + tl.arange(0, BN)
        gs = tl.load(gs_ptr + e)
        a_ptrs = a_ptr + arow[:, None] * stride_am + offs_k[None, :]
        w_ptrs = w_ptr + e * stride_we + offs_n[:, None] * stride_wn + offs_kb[None, :]
        if MODE == 7:   # scale gathered per packed byte: [BN, BK/2]
            s_ptrs = sf_ptr + e * stride_se + offs_n[:, None] * stride_sn + (offs_kb // 8)[None, :]
        elif SF_BLOCKED:
            # scales stored as [N/8, G/2, 8, 2] (16-byte blocks): pairs are contiguous, so the
            # load vectorizes by 2 and, after three interleaves, lands in the packed tile's layout
            G = K // 16
            s_ptrs = (sf_ptr + e * stride_se + (offs_n // 8)[:, None] * (G * 8) + (offs_n % 8)[:, None] * 2
                      + (offs_ks // 2)[None, :] * 16 + (offs_ks % 2)[None, :])
        else:
            s_ptrs = sf_ptr + e * stride_se + offs_n[:, None] * stride_sn + offs_ks[None, :]
        gs16 = gs.to(tl.float16)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        if GATED:
            w2_ptrs = w_ptrs + up_rows * stride_wn
            s2_ptrs = s_ptrs + up_rows * stride_sn
            acc2 = tl.zeros((BM, BN), dtype=tl.float32)
        if MODE == 1:
            wf_ptrs = w_ptr.to(tl.pointer_type(tl.float16)) + e * (stride_we // 2) + offs_n[:, None] * (stride_wn // 2) + offs_k[None, :]
            if GATED:
                wf2_ptrs = wf_ptrs + up_rows * (stride_wn // 2)
        for _ in range(0, K, BK):
            a = tl.load(a_ptrs, mask=rmask[:, None], other=0.0)
            if MODE == 1:
                w = tl.load(wf_ptrs)
                acc = tl.dot(a, tl.trans(w), acc)
                wf_ptrs += BK
                if GATED:
                    w2 = tl.load(wf2_ptrs)
                    acc2 = tl.dot(a, tl.trans(w2), acc2)
                    wf2_ptrs += BK
            else:
                wq = tl.load(w_ptrs)
                sf = tl.load(s_ptrs)
                if MODE == 3:
                    acc += tl.sum(wq.to(tl.float32), 1)[None, :] + tl.sum(sf.to(tl.float32), 1)[None, :]
                elif MODE == 11:
                    # per-16-group loads and dots; the group's scale multiplies the fp32 dot output
                    for g in tl.static_range(BK // 16):
                        wq_g = tl.load(w_ptrs + g * 8)                                      # [BN, 8] bytes
                        a_g = tl.load(a_ptrs + g * 16, mask=rmask[:, None], other=0.0)      # [BM, 16]
                        s_g = tl.sum(tl.where(offs_ks[None, :] == g, sf.to(tl.float32), 0.0), 1) * gs   # [BN]
                        w_g = _dequant(wq_g, sf, gs, ASM, 4)                                # [BN, 16] unscaled
                        acc += tl.dot(a_g, tl.trans(w_g)) * s_g[None, :]
                elif MODE == 9:
                    # per-16-group batched dot; scales applied to the dot output (fp32)
                    w = _dequant(wq, sf, gs, ASM, 4)                                  # [BN, BK] unscaled
                    w3 = tl.permute(tl.reshape(w, (BN, BK // 16, 16)), (1, 2, 0))     # [G, 16, BN]
                    a3 = tl.permute(tl.reshape(a, (BM, BK // 16, 16)), (1, 0, 2))     # [G, BM, 16]
                    o3 = tl.dot(a3, w3)                                               # [G, BM, BN]
                    s3 = (sf.to(tl.float32) * gs)                                     # [BN, G]
                    acc += tl.sum(o3 * tl.permute(s3, (1, 0))[:, None, :], 0)
                elif MODE == 7 or MODE == 8 or MODE == 10:
                    w = _dequant_scaled(wq, sf, gs16, 1 if MODE == 7 else (3 if MODE == 10 else 2))
                    acc = tl.dot(a, tl.trans(w), acc)
                elif MODE == 0 and ASM:
                    w = _dequant_scaled(wq, sf, gs16, SCALE_VARIANT)
                    acc = tl.dot(a, tl.trans(w), acc)
                else:
                    w = _dequant(wq, sf, gs, ASM, MODE)
                    if MODE == 2:
                        acc += tl.sum(w.to(tl.float32), 1)[None, :]
                    else:
                        acc = tl.dot(a, tl.trans(w), acc)
                if GATED:
                    wq2 = tl.load(w2_ptrs)
                    sf2 = tl.load(s2_ptrs)
                    if MODE == 3:
                        acc2 += tl.sum(wq2.to(tl.float32), 1)[None, :] + tl.sum(sf2.to(tl.float32), 1)[None, :]
                    elif MODE == 11:
                        for g in tl.static_range(BK // 16):
                            wq_g = tl.load(w2_ptrs + g * 8)
                            a_g = tl.load(a_ptrs + g * 16, mask=rmask[:, None], other=0.0)
                            s_g = tl.sum(tl.where(offs_ks[None, :] == g, sf2.to(tl.float32), 0.0), 1) * gs
                            w_g = _dequant(wq_g, sf2, gs, ASM, 4)
                            acc2 += tl.dot(a_g, tl.trans(w_g)) * s_g[None, :]
                    elif MODE == 9:
                        w2 = _dequant(wq2, sf2, gs, ASM, 4)
                        w3 = tl.permute(tl.reshape(w2, (BN, BK // 16, 16)), (1, 2, 0))
                        a3 = tl.permute(tl.reshape(a, (BM, BK // 16, 16)), (1, 0, 2))
                        o3 = tl.dot(a3, w3)
                        s3 = (sf2.to(tl.float32) * gs)
                        acc2 += tl.sum(o3 * tl.permute(s3, (1, 0))[:, None, :], 0)
                    elif MODE == 7 or MODE == 8 or MODE == 10:
                        w2 = _dequant_scaled(wq2, sf2, gs16, 1 if MODE == 7 else (3 if MODE == 10 else 2))
                        acc2 = tl.dot(a, tl.trans(w2), acc2)
                    elif MODE == 0 and ASM:
                        w2 = _dequant_scaled(wq2, sf2, gs16, SCALE_VARIANT)
                        acc2 = tl.dot(a, tl.trans(w2), acc2)
                    else:
                        w2 = _dequant(wq2, sf2, gs, ASM, MODE)
                        if MODE == 2:
                            acc2 += tl.sum(w2.to(tl.float32), 1)[None, :]
                        else:
                            acc2 = tl.dot(a, tl.trans(w2), acc2)
                    w2_ptrs += BK // 2
                    s2_ptrs += (BK // 2) // 8 if MODE == 7 else (BK // 2 if SF_BLOCKED else BK // 16)
                w_ptrs += BK // 2
                s_ptrs += (BK // 2) // 8 if MODE == 7 else (BK // 2 if SF_BLOCKED else BK // 16)
            a_ptrs += BK
        o_ptrs = out_ptr + rows[:, None] * stride_om + offs_n[None, :]
        if GATED:
            h = acc * tl.sigmoid(acc) * acc2
            # SiLU(gate) * up kept at BF16 precision (as SGLang's path does), stored in fp16
            tl.store(o_ptrs, h.to(tl.bfloat16).to(tl.float16), mask=rmask[:, None])
        else:
            tl.store(o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=rmask[:, None])


# ----------------------------------------------------------------------------------------------
# 4. post-expert norm and weighted sum
# ----------------------------------------------------------------------------------------------


@triton.jit
def _finalize_kernel(y_ptr, row_of_pair_ptr, tw_ptr, nw_ptr, out_ptr, eps,
                     H: tl.constexpr, TOPK: tl.constexpr):
    t = tl.program_id(0)
    d = tl.arange(0, H)
    nw = tl.load(nw_ptr + d)
    acc = tl.zeros((H,), dtype=tl.float32)
    for s in tl.static_range(TOPK):
        r = tl.load(row_of_pair_ptr + t * TOPK + s)
        ok = r >= 0
        y = tl.load(y_ptr + r * H + d, mask=ok & (d < H), other=0.0).to(tl.float32) * nw
        ms = tl.sum(y * y, 0) / H
        y = y * tl.rsqrt(ms + eps)
        g = tl.load(tw_ptr + t * TOPK + s)
        acc += tl.where(ok, g, 0.0) * y
    tl.store(out_ptr + t * H + d, acc.to(out_ptr.dtype.element_ty))


@triton.jit
def _finalize_pairs_kernel(y_ptr, lids_ptr, tw_ptr, nw_ptr, out_ptr, eps, H: tl.constexpr, TOPK: tl.constexpr):
    t = tl.program_id(0)
    d = tl.arange(0, H)
    nw = tl.load(nw_ptr + d)
    acc = tl.zeros((H,), dtype=tl.float32)
    for s in tl.static_range(TOPK):
        ok = tl.load(lids_ptr + t * TOPK + s) >= 0
        y = tl.load(y_ptr + (t * TOPK + s) * H + d, mask=ok & (d < H), other=0.0).to(tl.float32) * nw
        ms = tl.sum(y * y, 0) / H
        y = y * tl.rsqrt(ms + eps)
        g = tl.load(tw_ptr + t * TOPK + s)
        acc += tl.where(ok, g, 0.0) * y
    tl.store(out_ptr + t * H + d, acc.to(out_ptr.dtype.element_ty))


# ----------------------------------------------------------------------------------------------
# host side
# ----------------------------------------------------------------------------------------------


# Sweep (one hypothesis per scored shape): (BM, BN, BK, num_warps, num_stages)
GEMM_SWEEP = {
    1: (16, 32, 256, 4, 4),
    8: (16, 64, 128, 4, 4),
    16: (16, 64, 128, 4, 4),
    32: (16, 64, 128, 4, 4),
    128: (64, 128, 128, 8, 3),
    256: (64, 128, 128, 8, 3),
}


_GEMM_OVERRIDE = None   # (BM, BN, BK, num_warps, num_stages) during in-prepare benchmarking
_MODE = 0               # benchmark ablation mode (see _moe_gemm_kernel)
_CTAS_PER_SM = 3


def _gemm_config(bs: int):
    """(BM, BN, BK, num_warps, num_stages) by tokens per step."""
    if _GEMM_OVERRIDE is not None:
        return _GEMM_OVERRIDE
    if bs in GEMM_SWEEP:
        return GEMM_SWEEP[bs]
    if bs <= 8:
        return 16, 32, 256, 4, 4
    if bs <= 64:
        return 16, 64, 128, 4, 4
    return 64, 128, 128, 8, 3   # (64, 128, 64, 4, 3) crashes Triton 3.8's pass pipeline with the scaled asm


def _moe_state(weights) -> dict:
    dev = weights["w13"].device
    def block_scales(sf):   # [E, N, G] fp8 -> [E, N/8, G/2, 8, 2] contiguous (same bytes, pair-blocked)
        E_, N_, G_ = sf.shape
        return sf.view(E_, N_ // 8, 8, G_ // 2, 2).permute(0, 1, 3, 2, 4).contiguous().view(E_, N_, G_)

    state = {
        "w13_sfb": block_scales(weights["w13_sf"]), "w2_sfb": block_scales(weights["w2_sf"]),
        "w13": weights["w13"].contiguous(), "w13_sf": weights["w13_sf"].contiguous(),
        "w2": weights["w2"].contiguous(), "w2_sf": weights["w2_sf"].contiguous(),
        # the 2^14 of the fp16 placement trick, folded in (an exact power-of-two scaling)
        "w13_gs": (weights["w13_gs"].float() * SCALE_FIX).contiguous(),
        "w2_gs": (weights["w2_gs"].float() * SCALE_FIX).contiguous(),
        "norm_w": (weights["post_norm_w"].float() + 1.0).contiguous(),
    }
    H = weights["w2"].shape[1]
    I = weights["w2"].shape[2] * 2
    NP = MAX_M * TOPK
    rows_max = NP + LOCAL * 64
    maxb = NP // 16 + LOCAL
    state.update({
        "x16": torch.empty((MAX_M, H), dtype=torch.float16, device=dev),
        "sorted_tok": torch.zeros((rows_max,), dtype=torch.int32, device=dev),
        "row_of_pair": torch.full((NP,), -1, dtype=torch.int32, device=dev),
        "blk_e": torch.zeros((maxb,), dtype=torch.int32, device=dev),
        "blk_row0": torch.zeros((maxb,), dtype=torch.int32, device=dev),
        "blk_end": torch.zeros((maxb,), dtype=torch.int32, device=dev),
        "nblk": torch.zeros((1,), dtype=torch.int32, device=dev),
        "inter": torch.empty((rows_max, I), dtype=torch.float16, device=dev),
        "y": torch.empty((rows_max, H), dtype=torch.bfloat16, device=dev),
        "sorted_pair": torch.zeros((rows_max,), dtype=torch.int32, device=dev),
        "lids": torch.full((NP,), -1, dtype=torch.int32, device=dev),
        "npad": torch.zeros((1,), dtype=torch.int32, device=dev),
    })
    if MOE_BACKEND == "marlin":
        try:
            state["bank"] = _marlin_bank(weights)
            state["cache13"] = torch.empty((MAX_M * TOPK * max(2 * I, H),), dtype=torch.bfloat16, device=dev)
            state["cache2"] = torch.empty((MAX_M * TOPK * I,), dtype=torch.bfloat16, device=dev)
        except Exception as e:   # not in the image: the Triton experts
            print("marlin unavailable:", repr(e)[:120])
    return state


def _marlin_bank(weights):
    """The local experts in Marlin's layout, through SGLang's own loader step (the stock path)."""
    import types
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
    from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
    w13 = weights["w13"][:LOCAL]
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=w13.shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=w13.contiguous(), w2_weight=weights["w2"][:LOCAL].contiguous(),
        w13_weight_scale=weights["w13_sf"][:LOCAL].contiguous(), w2_weight_scale=weights["w2_sf"][:LOCAL].contiguous(),
        w13_weight_scale_2=weights["w13_gs"][:LOCAL].contiguous(), w2_weight_scale_2=weights["w2_gs"][:LOCAL].contiguous())
    prepare_moe_nvfp4_layer_for_marlin(layer)
    device = w13.device
    torch.cuda.synchronize()
    return {"w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
            "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
            "size_n": layer.intermediate_size_per_partition,
            "workspace": marlin_make_workspace(device, max_blocks_per_sm=4)}


def _route(x, ids, local_start: int, BM: int, state, marlin: bool):
    """Runs the routing kernel (plus the fp16 copy of x for the Triton GEMMs); returns (NP, MAXB)."""
    M, H = x.shape
    NP = max(16, triton.next_power_of_2(M * TOPK))
    MAXB = triton.next_power_of_2(NP // BM + LOCAL)
    x16 = state["x16"][:M]
    XBLOCK = 4096
    n_elems = M * H
    PB = min(NP, 64)                      # pairs per routing CTA
    CH = min(NP, 256)                     # pairs scanned per step
    n_route = triton.cdiv(M * TOPK, PB)
    grid0 = (n_route + (0 if marlin else triton.cdiv(n_elems, XBLOCK)),)
    _route_kernel[grid0](
        ids, x, x16,
        state["sorted_tok"], state["row_of_pair"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
        state["sorted_pair"], state["lids"], state["npad"],
        M, local_start, n_elems, n_route,
        PB=PB, CH=CH, BM=BM, MAXB=MAXB, E=LOCAL, TOPK=TOPK, XBLOCK=XBLOCK, MARLIN=marlin, num_warps=4,
    )
    return NP, MAXB


MARLIN_BLOCK_M = (8, 16, 32, 48, 64)   # MuseSpark1xFp4Experts' candidates, smallest first


_MARLIN_BM_OVERRIDE = None   # in-prepare benchmarking


def _marlin_block_m(num_tokens: int) -> int:
    if _MARLIN_BM_OVERRIDE is not None:
        return _MARLIN_BM_OVERRIDE
    bm = MARLIN_BLOCK_M[0]
    for b in MARLIN_BLOCK_M:
        if num_tokens * TOPK / LOCAL / b < 0.9:
            break
        bm = b
    return bm


def _moe_forward_marlin(x, ids, tw, local_start: int, I: int, state) -> torch.Tensor:
    """Experts through SGLang's Marlin W4A16 GEMMs, with this file's routing (one launch, in
    Marlin's sorted-ids format) and finalize."""
    from sgl_kernel.scalar_type import scalar_types
    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
    M, H = x.shape
    bank = state["bank"]
    x = x.contiguous()
    if ids.dtype != torch.int32:
        ids = ids.to(torch.int32)
    ids = ids.contiguous()
    tw = tw.contiguous()
    BM = _marlin_block_m(M)
    _route(x, ids, local_start, BM, state, marlin=True)
    out = torch.empty((M, H), dtype=MOE_OUT_DTYPE, device=x.device)
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    lids = state["lids"][: M * TOPK].view(M, TOPK)
    sorted_ids = state["sorted_pair"]
    cache1 = state["cache13"][: M * TOPK * gemm1_n].view(-1, gemm1_n)
    cache2 = state["cache2"][: M * TOPK * size_n].view(-1, size_n)
    cache3 = state["cache13"][: M * TOPK * H].view(-1, H)
    common = dict(moe_block_size=BM, mul_topk_weights=False, is_ep=True, b_q_type=scalar_types.float4_e2m1f,
                  is_k_full=True, use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, state["blk_e"], state["npad"], tw, top_k=TOPK,
        size_m=M, size_n=gemm1_n, size_k=H, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=lids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, state["blk_e"], state["npad"], tw, top_k=1,
        size_m=M * TOPK, size_n=H, size_k=size_n, **common).view(M * TOPK, H)
    _finalize_pairs_kernel[(M,)](y, lids, tw, state["norm_w"], out, POST_NORM_EPS, H=H, TOPK=TOPK, num_warps=4)
    return out


def _moe_forward(x, ids, tw, local_start: int, I: int, state) -> torch.Tensor:
    if "bank" in state:
        return _moe_forward_marlin(x, ids, tw, local_start, I, state)
    return _moe_forward_triton(x, ids, tw, local_start, I, state)


def _moe_forward_triton(x, ids, tw, local_start: int, I: int, state) -> torch.Tensor:
    """x [M, H] (bf16), ids [M, 8] global expert ids, tw [M, 8] fp32: the local experts'
    weighted post-normed outputs summed per token, bf16 [M, H]. Experts local_start .. +31 use
    w13[0..31]."""
    M, H = x.shape
    if ids.dtype != torch.int32:
        ids = ids.to(torch.int32)
    if not ids.is_contiguous():
        ids = ids.contiguous()
    tw = tw.contiguous()
    dev = x.device
    out = torch.empty((M, H), dtype=MOE_OUT_DTYPE, device=dev)

    BM, BN, BK, nw, ns = _gemm_config(M)
    NP, MAXB = _route(x, ids, local_start, BM, state, marlin=False)
    x16 = state["x16"][:M]
    w13, w13_sf, w2, w2_sf = state["w13"], state["w13_sf"], state["w2"], state["w2_sf"]
    max_items = (NP // BM + LOCAL) * (max(I, H) // BN)
    num_ctas = min(_CTAS_PER_SM * NUM_SMS, max_items)
    blocked = DEQUANT_ASM and _MODE == 0 and SCALE_VARIANT == 4
    sf13 = state["w13_sfb"] if blocked else w13_sf
    sf2 = state["w2_sfb"] if blocked else w2_sf
    common = dict(GATHER=True, GATED=True, ASM=DEQUANT_ASM, BM=BM, BN=BN, BK=BK, MODE=_MODE, SF_BLOCKED=blocked,
                  num_warps=nw, num_stages=ns)
    # gate/up GEMM + SiLU: A = x16 [M, H], W13[e] rows [0, I) gate and [I, 2I) up, K = H, N = I
    _moe_gemm_kernel[(num_ctas,)](
        x16, w13, sf13, state["w13_gs"], state["inter"],
        state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
        x16.stride(0), state["inter"].stride(0), H, I,
        w13.stride(0), w13.stride(1), w13_sf.stride(0), w13_sf.stride(1), I,
        num_ctas, I // BN, **common,
    )
    # down GEMM: A = inter [rows, I], W2[e] [H, I/2], K = I, N = H
    common.update(GATHER=False, GATED=False)
    _moe_gemm_kernel[(num_ctas,)](
        state["inter"], w2, sf2, state["w2_gs"], state["y"],
        state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
        state["inter"].stride(0), state["y"].stride(0), I, H,
        w2.stride(0), w2.stride(1), w2_sf.stride(0), w2_sf.stride(1), 0,
        num_ctas, H // BN, **common,
    )
    _finalize_kernel[(M,)](
        state["y"], state["row_of_pair"], tw, state["norm_w"], out, POST_NORM_EPS,
        H=H, TOPK=TOPK, num_warps=4,
    )
    return out
# == END SHARED moe ==


BENCH = True       # microbenchmarks in prepare (untimed; results go to the run log)
BENCH_BUDGET_S = 170


def _bench_inputs(bs, device, local_frac=0.35, hot=4, seed=0):
    """Synthetic routing: each token picks 8 distinct experts; a few hot local experts take
    most of the local routes, like the judge's skewed routers."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((bs, 4096), generator=g, device=device).to(torch.bfloat16)
    scores = torch.rand((bs, 256), generator=g, device=device)
    heat = torch.zeros((256,), device=device)
    heat[:hot] = 2.0                      # hot local experts
    heat[32:] -= 0.3                      # the other GPUs' experts a little colder
    ids = (scores + heat).topk(8, dim=-1).indices.to(torch.int32)
    w = torch.rand((bs, 8), generator=g, device=device)
    return x, ids, (w / w.sum(-1, keepdim=True)).float()


def _bench(state):
    import time
    global _GEMM_OVERRIDE, _CTAS_PER_SM
    t_start = time.time()
    dev = state["w13"].device
    data = {bs: _bench_inputs(bs, dev) for bs in (1, 8, 16, 32, 128, 256)}

    def timeit(fn, reps=20):
        # as the judge: CUDA-graph replays (host launch cost, e.g. TMA descriptor encoding, excluded)
        fn(); fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(reps):
                fn()
        g.replay()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(3):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) * 1000.0 / (3 * reps)

    global _MODE
    print("BENCH moe (graph replays): ablations, dot kept in every mode. MODE 0 full | 1 fp16 weights, no dequant"
          " (2x bytes) | 4 no scale multiply | 5 numeric nibble->fp16 (no asm) | 2 dequant, no dot | 3 loads only")
    E = state["w13"].shape[0]
    saved = (state["w13"], state["w2"], _GEMM_OVERRIDE)
    try:
        dummy13 = torch.randn((min(E, 32), 8192, 4096), device=dev).to(torch.float16).view(torch.uint8)
        dummy2 = torch.randn((min(E, 32), 4096, 4096), device=dev).to(torch.float16).view(torch.uint8)
    except Exception as e:
        dummy13 = dummy2 = None
        print("  (no fp16 dummy weights:", str(e)[:60], ")")
    for mode in (0, 4, 8, 10, 5):
        _MODE = mode
        _GEMM_OVERRIDE = (16, 64, 64, 4, 4) if mode == 1 else None
        if mode == 1:
            if dummy13 is None:
                continue
            state["w13"], state["w2"] = dummy13, dummy2
        try:
            r = [timeit(lambda bs=bs: _moe_forward(*data[bs][:3], 0, 4096, state)) for bs in (1, 8, 16, 32, 128, 256)]
            print(f"  MODE {mode}: " + "  ".join(f"bs{bs} {t:7.1f}" for bs, t in zip((1, 8, 16, 32, 128, 256), r)))
        except Exception as e:
            print(f"  MODE {mode} failed: {str(e)[:100]}")
        state["w13"], state["w2"] = saved[0], saved[1]
    del dummy13, dummy2
    _GEMM_OVERRIDE = saved[2]
    _MODE = 0
    cands = [(16, 32, 256, 4, 4), (16, 64, 128, 4, 4), (16, 32, 128, 4, 6), (16, 64, 128, 8, 4), (16, 32, 256, 4, 6)]
    big = [(64, 64, 128, 4, 4), (64, 128, 128, 8, 3), (64, 64, 128, 8, 3)]
    for ctas in (3,):
        _CTAS_PER_SM = ctas
        for cfg in cands:
            if time.time() - t_start > BENCH_BUDGET_S:
                break
            _GEMM_OVERRIDE = cfg
            try:
                r = [timeit(lambda bs=bs: _moe_forward(*data[bs][:3], 0, 4096, state)) for bs in (1, 8, 16, 32)]
                print(f"  ctas/sm={ctas} cfg={cfg}  bs1 {r[0]:7.1f}  bs8 {r[1]:7.1f}  bs16 {r[2]:7.1f}  bs32 {r[3]:7.1f}")
            except Exception as e:
                print(f"  ctas/sm={ctas} cfg={cfg}  failed: {str(e)[:80]}")
        for cfg in big:
            if time.time() - t_start > BENCH_BUDGET_S:
                break
            _GEMM_OVERRIDE = cfg
            try:
                r = [timeit(lambda bs=bs: _moe_forward(*data[bs][:3], 0, 4096, state)) for bs in (128, 256)]
                print(f"  ctas/sm={ctas} cfg={cfg}  bs128 {r[0]:7.1f}  bs256 {r[1]:7.1f}")
            except Exception as e:
                print(f"  ctas/sm={ctas} cfg={cfg}  failed: {str(e)[:80]}")
    _GEMM_OVERRIDE = None
    _CTAS_PER_SM = 3
    print(f"BENCH moe done in {time.time() - t_start:.0f}s")


def _bench_marlin(state):
    import time
    global _MARLIN_BM_OVERRIDE
    t0 = time.time()
    dev = state["w13"].device
    data = {bs: _bench_inputs(bs, dev) for bs in (1, 8, 16, 32, 128, 256)}

    def timeit(fn, reps=10):
        fn(); fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(reps):
                fn()
        g.replay(); torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(3):
            g.replay()
        e1.record(); torch.cuda.synchronize()
        return e0.elapsed_time(e1) * 1000.0 / (3 * reps)

    print("BENCH marlin block size (graph replays, us): rule = the stock density rule")
    for bm in (None, 8, 16, 32, 48, 64):
        _MARLIN_BM_OVERRIDE = bm
        try:
            r = [timeit(lambda bs=bs: _moe_forward(*data[bs][:3], 0, 4096, state)) for bs in (1, 8, 16, 32, 128, 256)]
            print(f"  BM={str(bm):4s}: " + "  ".join(f"bs{bs} {t:7.1f}" for bs, t in zip((1, 8, 16, 32, 128, 256), r)))
        except Exception as e:
            print(f"  BM={bm} failed: {str(e)[:100]}")
    _MARLIN_BM_OVERRIDE = None
    print(f"BENCH marlin done in {time.time() - t0:.0f}s")


def prepare(weights: Weights):
    state = _moe_state(weights)
    if BENCH and "bank" in state:
        try:
            _bench_marlin(state)
        except Exception as e:
            print("BENCH marlin failed:", repr(e))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    if BENCH and "bank" not in state:
        try:
            _bench(state)
        except Exception as e:
            print("BENCH failed:", repr(e))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    return state


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    cfg = data["config"]
    return _moe_forward(data["hidden_states"], data["topk_ids"], data["topk_weights"], cfg["local_start"],
                        cfg["d_expert"], state)
