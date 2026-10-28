#!/usr/bin/env python3
"""Streaming-read controls on the real GPU, timed like the judge (CUDA-graph replay, a 256 MB read flush of L2 before
every replay, min / median of N replays, graph-replay launch cost included like gpu_run's numbers):
  1. torch's reduction over float buffers of the proj layers' weight-stream sizes (6.3 .. 1000 MB);
  2. a pure stream kernel over 50.3 / 28.3 MB with three load paths - LDG.128 into registers (MODE 0), cp.async 16 B
     into a per-thread shared-memory ring (MODE 1), TMA 1-D bulk copies into an mbarrier ring (MODE 2) - at depth D
     (bytes in flight per thread / CTA) and G CTAs of 512 (+32) threads.
    python3 tools/bw_ctl.py [--emit]      (--emit: write the .cu and an nvcc command for the offline gate, no GPU)"""
import os, pathlib, statistics, sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "muse_bf16_proj"))
import submission as S  # noqa: E402  (its mbarrier helper source only; nothing is built)

SRC = r"""
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <stdint.h>
@@MBAR@@
__device__ __forceinline__ uint4 ldg_nc(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ void cpa16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cpa_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }

// step = 512 threads x 16 B = 8 KB (MODE 0/1); 16 KB per slot for MODE 2. CTA b takes steps b, b + G, ...
template <int D, int MODE>
__global__ void __launch_bounds__(544) stream_kernel(const uint8_t* __restrict__ src, uint32_t* __restrict__ out, int nsteps) {
  extern __shared__ __align__(1024) unsigned char smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  uint32_t acc = 0;
  if constexpr (MODE == 0) {
    if (warp == 16) return;
    const uint4* p = reinterpret_cast<const uint4*>(src) + tid;
    uint4 buf[D];
#pragma unroll
    for (int i = 0; i < D; ++i) {
      const int s = blockIdx.x + i * gridDim.x;
      buf[i] = s < nsteps ? ldg_nc(p + (size_t)s * 512) : make_uint4(0, 0, 0, 0);
    }
    for (int base = blockIdx.x; base < nsteps; base += D * gridDim.x) {
#pragma unroll
      for (int i = 0; i < D; ++i) {
        const int cur = base + i * gridDim.x;
        if (cur < nsteps) {
          acc ^= buf[i].x ^ buf[i].y ^ buf[i].z ^ buf[i].w;
          const int nxt = cur + D * gridDim.x;
          if (nxt < nsteps) buf[i] = ldg_nc(p + (size_t)nxt * 512);
        }
      }
    }
  } else if constexpr (MODE == 1) {
    if (warp == 16) return;
    const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem) + 16 * tid;
    const uint8_t* p = src + 16 * tid;
#pragma unroll
    for (int i = 0; i < D; ++i) {
      const int s = blockIdx.x + i * gridDim.x;
      if (s < nsteps) cpa16(sb + i * 8192, p + (size_t)s * 8192);
      cpa_commit();
    }
    for (int base = blockIdx.x; base < nsteps; base += D * gridDim.x) {
#pragma unroll
      for (int i = 0; i < D; ++i) {
        const int cur = base + i * gridDim.x;
        cpa_wait<D - 1>();
        if (cur < nsteps) {
          uint32_t a, b, c, d;
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(sb + i * 8192));
          acc ^= a ^ b ^ c ^ d;
          asm volatile("" :: "r"(acc));
          const int nxt = cur + D * gridDim.x;
          if (nxt < nsteps) cpa16(sb + i * 8192, p + (size_t)nxt * 8192);
        }
        cpa_commit();
      }
    }
    cpa_wait<0>();
  } else if constexpr (MODE == 3) {
    // burst (cgemv-like): CTA b loads steps b D .. b D + D - 1 into registers at once and exits (grid = nsteps / D)
    if (warp == 16) return;
    const uint4* p = reinterpret_cast<const uint4*>(src) + (size_t)blockIdx.x * D * 512 + tid;
    uint4 buf[D];
#pragma unroll
    for (int i = 0; i < D; ++i) buf[i] = ldg_nc(p + i * 512);
#pragma unroll
    for (int i = 0; i < D; ++i) acc ^= buf[i].x ^ buf[i].y ^ buf[i].z ^ buf[i].w;
  } else {
    // MODE 2: 16 KB slots filled by one bulk copy each; warp 16 lane 0 produces, warps 0..15 consume 32 B per thread
    const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem);
    const uint32_t bars = sb + D * 16384;
    const uint32_t full0 = bars, empty0 = bars + 8 * D;
    if (tid == 0) {
      for (int i = 0; i < D; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, 16); }
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (warp == 16) {
      if (lane == 0) {
        int it = 0;
        for (int s = blockIdx.x; s < nsteps; s += gridDim.x, ++it) {
          const int slot = it % D;
          if (it >= D) mbar_wait(empty0 + 8 * slot, ((it / D) - 1) & 1);
          mbar_expect_tx(full0 + 8 * slot, 16384);
          bulk_g2s(sb + slot * 16384, src + (size_t)s * 16384, 16384, full0 + 8 * slot);
        }
      }
      return;
    }
    int it = 0;
    for (int s = blockIdx.x; s < nsteps; s += gridDim.x, ++it) {
      const int slot = it % D;
      mbar_wait(full0 + 8 * slot, (it / D) & 1);
      uint32_t a, b, c, d, e, f, g, h;
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(sb + slot * 16384 + 32 * tid));
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(e), "=r"(f), "=r"(g), "=r"(h) : "r"(sb + slot * 16384 + 32 * tid + 16));
      acc ^= a ^ b ^ c ^ d ^ e ^ f ^ g ^ h;
      asm volatile("" :: "r"(acc));
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
  }
  out[blockIdx.x * 512 + tid] = acc;
}

template <int D, int MODE>
static void launch(const at::Tensor& src, at::Tensor& out, int64_t nbytes, int64_t grid) {
  constexpr int step = MODE == 2 ? 16384 : 8192;
  constexpr int smem = (MODE == 0 || MODE == 3) ? 0 : (MODE == 1 ? D * 8192 : D * 16384 + 16 * D + 64);
  auto kern = stream_kernel<D, MODE>;
  static bool attr_done = false;
  if (!attr_done) { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr_done = true; }
  const int nsteps = (int)(nbytes / step);
  const int g = MODE == 3 ? nsteps / D : (int)grid;
  kern<<<g, 544, smem, at::cuda::getCurrentCUDAStream()>>>(src.data_ptr<uint8_t>(), reinterpret_cast<uint32_t*>(out.data_ptr<int>()), nsteps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void stream(at::Tensor src, at::Tensor out, int64_t nbytes, int64_t grid, int64_t d, int64_t mode) {
  @@DISPATCH@@
  TORCH_CHECK(false, "stream: no instantiation for d=", d, " mode=", mode);
}
"""
CFG = [(d, m) for m in (0, 1, 2) for d in (2, 4, 8, 12)] + [(d, 3) for d in (2, 3, 4, 6, 8)]
disp = "\n  ".join(f"if (d == {d} && mode == {m}) {{ launch<{d}, {m}>(src, out, nbytes, grid); return; }}" for d, m in CFG)
src = SRC.replace("@@MBAR@@", S.MBAR_HELPERS).replace("@@DISPATCH@@", disp)

if "--emit" in sys.argv:
    import torch.utils.cpp_extension as E
    import sysconfig
    p = pathlib.Path("/work/tools/bw_ctl.cu")
    p.write_text("#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n" + src)
    inc = sum([["-I", d] for d in E.include_paths(device_type="cuda") + [sysconfig.get_paths()["include"]]], [])
    cmd = ["nvcc", "-gencode", "arch=compute_90a,code=sm_90a", "-O3", "-std=c++20", "-c", "-o", "/tmp/bw_ctl.o", "-Xptxas", "-v",
           "-DTORCH_API_INCLUDE_EXTENSION_H", "-D_GLIBCXX_USE_CXX11_ABI=1"] + inc + [str(p)]
    pathlib.Path("/work/tools/pub_nvcc_cmd.txt").write_text("\n".join(cmd))
    print("emitted", p)
    sys.exit(0)

from torch.utils.cpp_extension import load_inline  # noqa: E402

ext = load_inline(name="okc_bw_ctl_v1", cpp_sources="#include <torch/extension.h>\nvoid stream(torch::Tensor src, torch::Tensor out, int64_t nbytes, int64_t grid, int64_t d, int64_t mode);",
                  cuda_sources=src, functions=["stream"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                  no_implicit_headers=True, verbose=False)
dev = "cuda"
flush_buf = torch.empty(256 << 20, dtype=torch.uint8, device=dev)


def timed(fn, iters=30):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        flush_buf.view(torch.int32).sum()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0)
    ts.sort()
    return ts[0], statistics.median(ts)


OFF = 3.7   # us: graph-replay launch cost (judge calibration) - the TB/s column is bytes / (median - OFF)


def opt(name, default):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return default


sizes = [float(v) for v in opt("sizes", "50.3,28.3").split(",")]
modes = [int(v) for v in opt("modes", "0,1,2").split(",")]
grids = [int(v) for v in opt("grids", "132,264,396").split(",")]
depths = [int(v) for v in opt("depths", "2,4,8,12").split(",")]
if "--torch" in sys.argv:
    for mb in (6.3, 12.6, 28.3, 50.3, 67.1, 100.0, 200.0, 1000.0):
        buf = torch.randn(int(mb * 1e6) // 4, dtype=torch.float32, device=dev)
        mn, md = timed(lambda: torch.sum(buf))
        print(f"[torch.sum] {mb:7.1f} MB: min {mn:7.1f} med {md:7.1f} us -> {mb * 1e6 / (md - OFF) / 1e6:.2f} TB/s (med - {OFF} us)", flush=True)
        del buf
names = {0: "LDG.128 regs", 1: "cp.async smem", 2: "TMA bulk 16K", 3: "LDG burst CTAs"}
for mb in sizes:
    nbytes = int(mb * 1e6) // 16384 * 16384
    src_t = torch.randint(0, 255, (nbytes,), dtype=torch.uint8, device=dev)
    for mode in modes:
        for grid in (grids if mode != 3 else [0]):
            for d in depths:
                if (d, mode) not in CFG:
                    continue
                if mode == 1 and d * 8192 * (grid / 132) > 220 * 1024:
                    continue
                if mode == 2 and d * 16384 * (grid / 132) > 220 * 1024:
                    continue
                step = 16384 if mode == 2 else 8192
                g = nbytes // step // d if mode == 3 else grid
                out = torch.zeros(max(g, 1) * 512, dtype=torch.int32, device=dev)
                mn, md = timed(lambda: ext.stream(src_t, out, nbytes, grid, d, mode))
                infl = d * step * (grid / 132) if mode != 3 else d * 8192 * 3
                print(f"[stream] {mb:5.1f} MB {names[mode]:14s} G={g:4d} D={d:2d} (~{infl / 1024:5.0f} KB/SM in flight): "
                      f"min {mn:6.1f} med {md:6.1f} us -> {nbytes / (md - OFF) / 1e6:.2f} TB/s", flush=True)
    del src_t
