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
DEQUANT_ASM = True          # inline PTX for the nibble -> fp16 placement (False: plain Triton ops)
SCALE_FIX = 2.0 ** 14       # folded into the global scales: the placement trick yields value * 2^-14


# ----------------------------------------------------------------------------------------------
# 1. routing
# ----------------------------------------------------------------------------------------------


@triton.jit
def _route_kernel(
    ids_ptr, x_ptr, x16_ptr,
    sorted_tok_ptr, row_of_pair_ptr, blk_e_ptr, blk_row0_ptr, blk_end_ptr, nblk_ptr,
    M, local_start, n_elems, n_route_ctas,
    PB: tl.constexpr, CH: tl.constexpr, BM: tl.constexpr, MAXB: tl.constexpr, E: tl.constexpr,
    TOPK: tl.constexpr, XBLOCK: tl.constexpr,
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


@triton.jit
def _dequant(wq, sf, gs, ASM: tl.constexpr):
    """wq uint8 [BN, BK/2] packed E2M1, sf fp8 [BN, BK/16], gs fp32 (x 2^14) -> fp16 [BN, BK]."""
    if ASM:
        lo, hi = tl.inline_asm_elementwise(
            asm=_NIBBLE_ASM, constraints="=r,=r,=r,=r,r", args=[wq],
            dtype=(tl.float16, tl.float16), is_pure=True, pack=4)
    else:
        lo = (wq & 0x0F).to(tl.int16)
        hi = (wq >> 4).to(tl.int16)
        lo = (((lo & 7) << 9) | ((lo & 8) << 12)).to(tl.float16, bitcast=True)
        hi = (((hi & 7) << 9) | ((hi & 8) << 12)).to(tl.float16, bitcast=True)
    w = tl.interleave(lo, hi)                                   # [BN, BK]
    s = (sf.to(tl.float32) * gs).to(tl.float16)                 # [BN, BK/16]
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
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
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
        s_ptrs = sf_ptr + e * stride_se + offs_n[:, None] * stride_sn + offs_ks[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        if GATED:
            w2_ptrs = w_ptrs + up_rows * stride_wn
            s2_ptrs = s_ptrs + up_rows * stride_sn
            acc2 = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, K, BK):
            a = tl.load(a_ptrs, mask=rmask[:, None], other=0.0)
            wq = tl.load(w_ptrs)
            sf = tl.load(s_ptrs)
            w = _dequant(wq, sf, gs, ASM)
            acc = tl.dot(a, tl.trans(w), acc)
            if GATED:
                wq2 = tl.load(w2_ptrs)
                sf2 = tl.load(s2_ptrs)
                w2 = _dequant(wq2, sf2, gs, ASM)
                acc2 = tl.dot(a, tl.trans(w2), acc2)
                w2_ptrs += BK // 2
                s2_ptrs += BK // 16
            a_ptrs += BK
            w_ptrs += BK // 2
            s_ptrs += BK // 16
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


# ----------------------------------------------------------------------------------------------
# host side
# ----------------------------------------------------------------------------------------------


def _gemm_config(bs: int):
    """(BM, BN, BK, num_warps, num_stages) by tokens per step."""
    if bs <= 8:
        return 16, 32, 256, 4, 4
    if bs <= 64:
        return 16, 64, 128, 4, 4
    return 64, 128, 64, 4, 3


def _moe_state(weights) -> dict:
    dev = weights["w13"].device
    state = {
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
    })
    return state


def _moe_forward(x, ids, tw, local_start: int, I: int, state) -> torch.Tensor:
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
    NP = max(16, triton.next_power_of_2(M * TOPK))
    MAXB = triton.next_power_of_2(NP // BM + LOCAL)
    x16 = state["x16"][:M]
    XBLOCK = 4096
    n_elems = M * H
    PB = min(NP, 64)                      # pairs per routing CTA
    CH = min(NP, 256)                     # pairs scanned per step
    n_route = triton.cdiv(M * TOPK, PB)
    grid0 = (n_route + triton.cdiv(n_elems, XBLOCK),)
    _route_kernel[grid0](
        ids, x, x16,
        state["sorted_tok"], state["row_of_pair"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
        M, local_start, n_elems, n_route,
        PB=PB, CH=CH, BM=BM, MAXB=MAXB, E=LOCAL, TOPK=TOPK, XBLOCK=XBLOCK, num_warps=4,
    )
    w13, w13_sf, w2, w2_sf = state["w13"], state["w13_sf"], state["w2"], state["w2_sf"]
    max_items = (NP // BM + LOCAL) * (max(I, H) // BN)
    num_ctas = min(2 * NUM_SMS, max_items)
    common = dict(GATHER=True, GATED=True, ASM=DEQUANT_ASM, BM=BM, BN=BN, BK=BK, num_warps=nw, num_stages=ns)
    # gate/up GEMM + SiLU: A = x16 [M, H], W13[e] rows [0, I) gate and [I, 2I) up, K = H, N = I
    _moe_gemm_kernel[(num_ctas,)](
        x16, w13, w13_sf, state["w13_gs"], state["inter"],
        state["sorted_tok"], state["blk_e"], state["blk_row0"], state["blk_end"], state["nblk"],
        x16.stride(0), state["inter"].stride(0), H, I,
        w13.stride(0), w13.stride(1), w13_sf.stride(0), w13_sf.stride(1), I,
        num_ctas, I // BN, **common,
    )
    # down GEMM: A = inter [rows, I], W2[e] [H, I/2], K = I, N = H
    common.update(GATHER=False, GATED=False)
    _moe_gemm_kernel[(num_ctas,)](
        state["inter"], w2, w2_sf, state["w2_gs"], state["y"],
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


def prepare(weights: Weights):
    return _moe_state(weights)


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    cfg = data["config"]
    return _moe_forward(data["hidden_states"], data["topk_ids"], data["topk_weights"], cfg["local_start"],
                        cfg["d_expert"], state)
