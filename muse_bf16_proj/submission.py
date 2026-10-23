# okc: problem=muse_bf16_proj
"""Dense projections for Muse Spark Open: x @ W.T at skinny M.

Small M (the decode / verify shapes) is a bandwidth problem: the whole weight is streamed
once and x stays in L2. Each shape runs as ONE Triton kernel:

  * the grid is (N tiles) x (split-K); each CTA streams a [BLOCK_N, K/SPLIT_K] slab of W
    through a multi-stage cp.async pipeline and accumulates in fp32 with tensor cores;
  * with split-K, the partials go to an fp32 workspace and the LAST CTA of a tile (an atomic
    ticket, acq_rel at gpu scope, as CUTLASS's split-K semaphore does) sums them and writes
    the output, so there is no second launch and no memset; it also resets its ticket.

The router's FP32 weight runs as a BF16 hi + lo pair with FP32 accumulation (both weights are
dotted in the same kernel), which TASK.md allows at any M. Large-M BF16 shapes go to cuBLAS.
"""

from typing import TypedDict

import torch
import triton
import triton.language as tl


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int      # output width; the unpooler's is 1024 up to 8 tokens (this GPU's slice), else 8192
    k: int
    layer: str  # which weight: "qkv_gate", "o", "pooler", "unpooler" or "router"


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


# ----------------------------------------------------------------------------------------------
# Triton kernel
# ----------------------------------------------------------------------------------------------


@triton.jit
def _skinny_gemm_kernel(
    x_ptr, w_ptr, w2_ptr, out_ptr, part_ptr, cnt_ptr,
    M, N, K,
    stride_xm, stride_wn, stride_om,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr, TWO_W: tl.constexpr,
):
    """out[M, N] = x[M, K] @ W[N, K].T (+ x @ W2.T when TWO_W), fp32 accumulate."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    n_tiles = tl.num_programs(0)
    m_tiles = tl.num_programs(2)

    # Each split's K range is a multiple of BLOCK_K (the host checks it): say so, or the
    # pointer alignment is unknown and the loads degrade to scalar 2-byte copies (no cp.async).
    k_chunk = tl.multiple_of(K // SPLIT_K, BLOCK_K)
    k0 = tl.multiple_of(pid_k * k_chunk, BLOCK_K)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M

    # x tile [BLOCK_M, BLOCK_K] (rows past M read as zero); W tile as [BLOCK_K, BLOCK_N] so the
    # dot needs no transpose. W is [N, K] row-major: K is the contiguous axis.
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :]
    w_ptrs = w_ptr + (k0 + offs_k)[:, None] + offs_n[None, :] * stride_wn
    if TWO_W:
        w2_ptrs = w2_ptr + (k0 + offs_k)[:, None] + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, k_chunk, BLOCK_K):
        x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptrs)
        acc = tl.dot(x, w, acc)
        if TWO_W:
            w2 = tl.load(w2_ptrs)
            acc = tl.dot(x, w2, acc)
            w2_ptrs += BLOCK_K
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    if SPLIT_K == 1:
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None])
    else:
        # Partials: [SPLIT_K, m_tiles * BLOCK_M, N] fp32, whole tiles (no masking needed).
        m_pad = m_tiles * BLOCK_M
        part_ptrs = part_ptr + (pid_k * m_pad + offs_m)[:, None] * N + offs_n[None, :]
        tl.store(part_ptrs, acc)
        # All of this CTA's partial stores are issued before one thread takes the ticket
        # (release, gpu scope: the stores become visible before the increment does).
        tl.debug_barrier()
        tile = pid_m * n_tiles + pid_n
        ticket = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel", scope="gpu")
        tl.debug_barrier()
        if ticket == SPLIT_K - 1:
            # Last CTA of this tile: every other split's partial is visible (acquire). Read
            # them from L2 (.cg bypasses L1, so no stale line, and the loads still vectorize).
            total = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for s in tl.static_range(SPLIT_K):
                p = part_ptr + (s * m_pad + offs_m)[:, None] * N + offs_n[None, :]
                total += tl.load(p, cache_modifier=".cg")
            tl.store(out_ptrs, total.to(out_ptr.dtype.element_ty), mask=m_mask[:, None])
            # Reset the ticket for the next launch (graph replays reuse the counters).
            tl.atomic_xchg(cnt_ptr + tile, 0)


# ----------------------------------------------------------------------------------------------
# Configs
# ----------------------------------------------------------------------------------------------

# (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages) per (layer, N, K, m-bucket).
# Aims: >= ~256 CTAs so every SM has 2+ resident, each with ~48 KB of weight in flight, and
# as little split-K partial traffic as that allows.
MAX_TRITON_M = 64        # BF16 layers above this go to cuBLAS
COUNTERS = 8192
PARTIALS_FLOATS = 8 << 20  # 32 MB fp32 workspace


def _block_m(m: int) -> int:
    if m <= 16:
        return 16
    if m <= 32:
        return 32
    if m <= 64:
        return 64
    return 128


# Sweep table: (BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages) per (layer, n, m). The m <= 32
# shapes move the same bytes, so giving each m a different configuration measures four
# pipeline shapes per layer in one run.
SWEEP = {
    # pooler: N 4096, K 8192
    ("pooler", 4096, 1):   (32, 256, 1, 4, 4),
    ("pooler", 4096, 8):   (32, 256, 2, 4, 4),
    ("pooler", 4096, 16):  (64, 128, 2, 4, 4),
    ("pooler", 4096, 32):  (32, 256, 1, 4, 5),
    # unpooler, whole weight: N 8192, K 4096
    ("unpooler", 8192, 16): (32, 256, 1, 4, 4),
    ("unpooler", 8192, 32): (32, 256, 2, 4, 4),
    # unpooler, this GPU's slice: N 1024, K 4096
    ("unpooler", 1024, 1): (32, 256, 4, 4, 4),
    ("unpooler", 1024, 8): (16, 256, 4, 4, 4),
    # o: N 8192, K 1024
    ("o", 8192, 1):   (32, 256, 1, 4, 4),
    ("o", 8192, 8):   (64, 256, 1, 4, 4),
    ("o", 8192, 16):  (32, 512, 1, 4, 2),
    ("o", 8192, 32):  (32, 256, 1, 8, 4),
    # qkv_gate: N 2304, K 8192
    ("qkv_gate", 2304, 1):  (32, 256, 4, 4, 4),
    ("qkv_gate", 2304, 8):  (32, 256, 2, 4, 4),
    ("qkv_gate", 2304, 16): (16, 256, 4, 4, 4),
    ("qkv_gate", 2304, 32): (32, 256, 8, 4, 4),
    # router: N 256, K 8192, hi + lo (two weight streams)
    ("router", 256, 1):   (16, 256, 8, 4, 4),
    ("router", 256, 8):   (32, 256, 16, 4, 3),
    ("router", 256, 16):  (16, 256, 16, 4, 4),
    ("router", 256, 32):  (16, 256, 8, 4, 4),
}


def _config(layer: str, m: int, n: int, k: int):
    bm = _block_m(m)
    if layer == "router":
        if m > 64:
            return None   # cuBLAS, one GEMM over [hi | lo]
        bn, bk, sk, nw, ns = SWEEP.get((layer, n, m), (16, 256, 8, 4, 4))
        return dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=sk, num_warps=nw, num_stages=ns)
    if m > MAX_TRITON_M:
        return None
    default = {
        "qkv_gate": (16, 256, 1, 4, 4),
        "o": (32, 256, 1, 4, 4),
        "pooler": (32, 256, 1, 4, 4),
        "unpooler": (32, 256, 1, 4, 4) if n > 1024 else (32, 256, 4, 4, 4),
    }[layer]
    bn, bk, sk, nw, ns = SWEEP.get((layer, n, m), default)
    if bm > 32 and bk > 256:   # keep shared memory in budget for the 64-row tiles
        bk = 256
    return dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=sk, num_warps=nw, num_stages=ns)


def _usable(cfg, m: int, n: int, k: int) -> bool:
    if cfg is None:
        return False
    if n % cfg["BLOCK_N"] or k % cfg["SPLIT_K"]:
        return False
    chunk = k // cfg["SPLIT_K"]
    if chunk % cfg["BLOCK_K"]:
        return False
    m_tiles = triton.cdiv(m, cfg["BLOCK_M"])
    n_tiles = n // cfg["BLOCK_N"]
    if m_tiles * n_tiles > COUNTERS:
        return False
    if cfg["SPLIT_K"] > 1 and cfg["SPLIT_K"] * m_tiles * cfg["BLOCK_M"] * n > PARTIALS_FLOATS:
        return False
    return True


# ----------------------------------------------------------------------------------------------
# prepare / kernel_fn
# ----------------------------------------------------------------------------------------------


def prepare(weights: Weights, warm: bool = True):
    """Called once, untimed. Contiguous copies, the router's hi/lo split, and the split-K
    workspace; then every scored shape is run once so the judge never compiles under capture."""
    state = {name: w.contiguous() for name, w in weights.items()}
    router = state["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi"] = hi.contiguous()
    state["router_lo"] = lo.contiguous()
    state["router_hilo_t"] = torch.cat((hi, lo), dim=1).t().contiguous()   # [2K, N]
    device = router.device
    state["counters"] = torch.zeros((COUNTERS,), dtype=torch.int32, device=device)
    state["partials"] = torch.empty((PARTIALS_FLOATS,), dtype=torch.float32, device=device)

    if not warm:
        return state
    # Warm-up / compile for the shapes the judge will run (tests and scored), with throwaway x.
    layers = {"qkv_gate": 8192, "o": 1024, "pooler": 8192, "unpooler": 4096, "router": 8192}
    try:
        for m in (1, 5, 7, 8, 9, 16, 32, 33, 128, 130, 256):
            for layer, k in layers.items():
                n = state[layer].shape[0]
                if layer == "unpooler" and m <= 8:
                    n = 1024
                x = torch.randn((m, k), device=device).to(torch.bfloat16)
                kernel_fn({"x": x, "config": {"m": m, "n": n, "k": k, "layer": layer}}, state)
        torch.cuda.synchronize()
    except Exception as e:  # never let warm-up sink the run
        print("warm-up failed:", repr(e))
    return state


def _launch(x, w, w2, out, state, cfg, m, n, k):
    m_tiles = triton.cdiv(m, cfg["BLOCK_M"])
    n_tiles = n // cfg["BLOCK_N"]
    grid = (n_tiles, cfg["SPLIT_K"], m_tiles)
    _skinny_gemm_kernel[grid](
        x, w, w2 if w2 is not None else w, out, state["partials"], state["counters"],
        m, n, k,
        x.stride(0), w.stride(0), out.stride(0),
        BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
        SPLIT_K=cfg["SPLIT_K"], TWO_W=w2 is not None,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )
    return out


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    x = data["x"]
    cfg_in = data["config"]
    layer, n = cfg_in["layer"], cfg_in["n"]
    m, k = x.shape
    if x.stride(1) != 1:
        x = x.contiguous()

    if layer == "router":
        cfg = _config(layer, m, n, k)
        if _usable(cfg, m, n, k):
            out = torch.empty((m, n), dtype=torch.float32, device=x.device)
            return _launch(x, state["router_hi"], state["router_lo"], out, state, cfg, m, n, k)
        # one GEMM: [x | x] @ [hi | lo]^T == x @ hi^T + x @ lo^T, fp32 accumulation
        x2 = torch.cat((x, x), dim=1)
        return torch.mm(x2, state["router_hilo_t"], out_dtype=torch.float32)

    w = state[layer]
    if w.shape[0] != n:
        w = w[:n]
    cfg = _config(layer, m, n, k)
    if _usable(cfg, m, n, k):
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        return _launch(x, w, None, out, state, cfg, m, n, k)
    return torch.matmul(x, w.t())
