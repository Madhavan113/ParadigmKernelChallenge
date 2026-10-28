"""Round-2 item 2 continued: the sgemv (unpooler slice, m = 1, bf16 weights) gets the same PDL treatment as cgemv
(weights before griddepcontrol.wait, x after it through L2, launch_dependents at entry; a PDL launch when the fork's
PDL word has bit 9), so the flag covers whichever m = 1 route the fork's autotuner picked; ab_vm `m1:<layer>=<kind>`
pins a layer's m = 1 route (cgemv / sgemv / sk2) for a paired A/B.  Run once from the worktree root."""
import pathlib

root = pathlib.Path(__file__).resolve().parents[2]


def patch(path, edits):
    p = root / path
    s = p.read_text()
    for old, new, cnt in edits:
        n = s.count(old)
        assert n == cnt, (path, n, cnt, old[:90])
        s = s.replace(old, new)
    p.write_text(s)


patch("muse_decode_layer/proj_fork.py", [
    ("""  for (int j = 0; j < PER; ++j) xv[j] = __ldg(reinterpret_cast<const uint4*>(x + j * (NTH * 8) + tid * 8));""",
     """  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");   // x is the previous kernel's output
  for (int j = 0; j < PER; ++j) xv[j] = __ldcg(reinterpret_cast<const uint4*>(x + j * (NTH * 8) + tid * 8));""", 1),
])
# the sgemv kernel's signature, entry trigger and launcher: edit by line scan (the signature spans two lines)
p = root / "muse_decode_layer/proj_fork.py"
s = p.read_text()
i = s.index("__global__ void __launch_bounds__(NTH) sgemv_kernel(")
j = s.index("{", i)
sig = s[i:j]
assert "int pdl" not in sig
s = s[:i] + sig.rstrip().rstrip(")") + ", int pdl)" + s[j:]
k = s.index("constexpr int PER = K / NTH / 8;", i)
s = s[:k] + "if (pdl) asm volatile(\"griddepcontrol.launch_dependents;\" ::: \"memory\");   // dependents may prefetch now\n  " + s[k:]
old = """  sgemv_kernel<4, 512, 4096><<<(int)(N / 4), 512, 0, at::cuda::getCurrentCUDAStream()>>>("""
assert s.count(old) == 1, s.count(old)
m = s.index(old)
e = s.index(";", m)
call = s[m:e + 1]
args = call[call.index(">>>(") + 4:-2]   # the argument list between >>>( and );
new = ("""  {
    auto st = at::cuda::getCurrentCUDAStream();
    const int pdl = (g_sk2_pdl & 0x200) ? 1 : 0;   // PDL launch of the m = 1 GEMV (sk2_set_pdl bit 9)
    if (pdl) {
      cudaLaunchConfig_t lc = {}; lc.gridDim = dim3((int)(N / 4)); lc.blockDim = dim3(512); lc.dynamicSmemBytes = 0; lc.stream = st;
      cudaLaunchAttribute at_[1]; at_[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      at_[0].val.programmaticStreamSerializationAllowed = 1; lc.attrs = at_; lc.numAttrs = 1;
      C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, sgemv_kernel<4, 512, 4096>, """ + args + """, 1));
    } else {
      sgemv_kernel<4, 512, 4096><<<(int)(N / 4), 512, 0, st>>>(""" + args + """, 0);
    }
  }""")
s = s[:m] + new + s[e + 1:]
# the sgemv run path publishes the PDL word like cgemv
old2 = """    if kind == "sgemv":   # unpooler slice, m = 1: BF16 weights straight from memory (4 rows per CTA)
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        _get_ext().sgemv(state["raw"][layer], x, out, n)"""
assert s.count(old2) == 1
s = s.replace(old2, """    if kind == "sgemv":   # unpooler slice, m = 1: BF16 weights straight from memory (4 rows per CTA)
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        ext = _get_ext()
        ext.sk2_set_pdl(SK2_PDL)
        ext.sgemv(state["raw"][layer], x, out, n)""")
p.write_text(s)

patch("muse_decode_layer/ab_vm.py", [
    ("""        elif kind == "tinyh0":   # the MoE TINY build with OKC_WSH0 (pre-built when MOE_TINY_H0_BUILD)""",
     """        elif kind == "m1":   # pin a layer's m = 1 route in the proj fork: m1:unpooler=sgemv|cgemv|sk2 (qkv_gate / o / unpooler)
            ch = st["bfp"]["choice"]
            saved.append(("m1", (sel, 1), ch.get((sel, 1))))
            ch[(sel, 1)] = (val,)
        elif kind == "tinyh0":   # the MoE TINY build with OKC_WSH0 (pre-built when MOE_TINY_H0_BUILD)""", 1),
    ("""        elif kind == "tinyh0":
            S.moe_TINY_DEFINES = old
        else:""", """        elif kind == "tinyh0":
            S.moe_TINY_DEFINES = old
        elif kind == "m1":
            if old is None:
                st["bfp"]["choice"].pop(k, None)
            else:
                st["bfp"]["choice"][k] = old
        else:""", 1),
])
print("patched")
