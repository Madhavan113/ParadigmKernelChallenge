"""Probe Triton lowering questions for sm_90 without a GPU.

  python3 probe_dot3d.py 3d   # batched (3D) dots: fp8 QK, fp16 PV, 8 warps -> wgmma?
  python3 probe_dot3d.py pv8  # tl.dot(p8, v8) with V [T, 64] loaded row-major: transpose cost?
  python3 probe_dot3d.py ex2  # ex2.approx.f16x2 inline asm compiles
"""
import sys, re
import triton, triton.language as tl
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget


@triton.jit
def k3d(q_ptr, k_ptr, v_ptr, o_ptr, n, RB: tl.constexpr, D: tl.constexpr, T: tl.constexpr, QK8: tl.constexpr, PV8: tl.constexpr):
    hh = tl.arange(0, 2)
    r = tl.arange(0, RB)
    d = tl.arange(0, D)
    t = tl.arange(0, T)
    qp = q_ptr + hh[:, None, None] * RB * D + r[None, :, None] * D + d[None, None, :]
    q = tl.load(qp)                      # [2, RB, D] fp8 or fp16
    acc = tl.zeros((2, RB, D), tl.float32)
    m_i = tl.full((2, RB), -1e30, tl.float32)
    l_i = tl.zeros((2, RB), tl.float32)
    for t0 in range(0, n, T):
        kp = k_ptr + (t0 + t)[None, :, None] * (2 * D) + hh[:, None, None] * D + d[None, None, :]
        k8 = tl.load(kp)                 # [2, T, D]
        if QK8:
            sc = tl.dot(q, tl.permute(k8, (0, 2, 1)))
        else:
            sc = tl.dot(q, tl.permute(k8.to(tl.float16), (0, 2, 1)))
        m_new = tl.maximum(m_i, tl.max(sc, 2))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(sc - m_new[:, :, None])
        l_i = l_i * alpha + tl.sum(p, 2)
        v8 = tl.load(v_ptr + (t0 + t)[None, :, None] * (2 * D) + hh[:, None, None] * D + d[None, None, :])
        if PV8:
            acc = acc * alpha[:, :, None] + tl.dot(p.to(tl.float8e4nv), v8)
        else:
            acc = acc * alpha[:, :, None] + tl.dot(p.to(tl.float16), v8.to(tl.float16))
        m_i = m_new
    tl.store(o_ptr + hh[:, None, None] * RB * D + r[None, :, None] * D + d[None, None, :], acc / l_i[:, :, None])


@triton.jit
def kpv8(q_ptr, k_ptr, v_ptr, o_ptr, n, RB: tl.constexpr, D: tl.constexpr, T: tl.constexpr, PV8: tl.constexpr, EX2H: tl.constexpr):
    r = tl.arange(0, RB)
    d = tl.arange(0, D)
    t = tl.arange(0, T)
    q = tl.load(q_ptr + r[:, None] * D + d[None, :])   # fp8
    acc = tl.zeros((RB, D), tl.float32)
    m_i = tl.full((RB,), -1e30, tl.float32)
    l_i = tl.zeros((RB,), tl.float32)
    for t0 in range(0, n, T):
        k8 = tl.load(k_ptr + (t0 + t)[:, None] * (2 * D) + d[None, :])
        sc = tl.dot(q, tl.trans(k8))
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        alpha = tl.exp2(m_i - m_new)
        if EX2H:
            x = (sc - m_new[:, None]).to(tl.float16)
            p = tl.inline_asm_elementwise(asm="ex2.approx.f16x2 $0, $1;", constraints="=r,r", args=[x],
                                          dtype=tl.float16, is_pure=True, pack=2)
            l_i = l_i * alpha + tl.sum(p, 1).to(tl.float32)
        else:
            p = tl.exp2(sc - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
        v8 = tl.load(v_ptr + (t0 + t)[:, None] * (2 * D) + d[None, :])
        if PV8:
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float8e4nv), v8)
        else:
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v8.to(tl.float16))
        m_i = m_new
    tl.store(o_ptr + r[:, None] * D + d[None, :], acc / l_i[:, None])


def count(pat, s):
    return len(re.findall(pat, s))


def report(k, tag):
    ttgir, ptx = k.asm["ttgir"], k.asm["ptx"]
    print(f"[{tag}] ttgir: warp_group_dot {count('warp_group_dot', ttgir)} | dot(mma2) {count(r'tt.dot ', ttgir) + count(r'tt.dot<', ttgir)} "
          f"| convert_layout {count('convert_layout', ttgir)} | memdesc_trans {count('memdesc_trans', ttgir)} | local_alloc {count('local_alloc', ttgir)}")
    print(f"[{tag}] ptx: wgmma {count(r'wgmma.mma_async', ptx)} ({sorted(set(re.findall(r'wgmma.mma_async[.\w]+', ptx)))}) | mma.sync {count('mma.sync', ptx)} "
          f"| ex2 {count('ex2', ptx)} | cvt {count(r'cvt\.', ptx)} | prmt {count('prmt', ptx)} | shfl {count('shfl', ptx)} | bar.sync {count('bar.sync', ptx)} "
          f"| movmatrix {count('movmatrix', ptx)} | stmatrix {count('stmatrix', ptx)} | smem {k.metadata.shared}")
    open(f"/work/tools/out_probe_{tag}.ptx", "w").write(ptx)


mode = sys.argv[1]
tgt = GPUTarget("cuda", 90, 32)
if mode == "3d":
    for RB, QK8, PV8, nw in ((64, True, False, 8), (64, True, False, 4), (16, False, False, 4), (64, True, True, 8)):
        sig = {"q_ptr": "*fp8e4nv" if QK8 else "*fp16", "k_ptr": "*fp8e4nv", "v_ptr": "*fp8e4nv", "o_ptr": "*fp32", "n": "i32",
               "RB": "constexpr", "D": "constexpr", "T": "constexpr", "QK8": "constexpr", "PV8": "constexpr"}
        attrs = {(i,): [["tt.divisibility", 16]] for i in range(4)}
        try:
            k = triton.compile(ASTSource(fn=k3d, signature=sig, constexprs={"RB": RB, "D": 64, "T": 128, "QK8": QK8, "PV8": PV8}, attrs=attrs),
                               target=tgt, options={"num_warps": nw, "num_stages": 3})
            report(k, f"3d_rb{RB}_qk8{int(QK8)}_pv8{int(PV8)}_w{nw}")
        except Exception as e:
            print(f"[3d RB={RB} QK8={QK8} PV8={PV8} warps={nw}] FAILED: {str(e)[:300]}")
elif mode == "pv8":
    for PV8, EX2H in ((False, False), (True, False), (False, True)):
        sig = {"q_ptr": "*fp8e4nv", "k_ptr": "*fp8e4nv", "v_ptr": "*fp8e4nv", "o_ptr": "*fp32", "n": "i32",
               "RB": "constexpr", "D": "constexpr", "T": "constexpr", "PV8": "constexpr", "EX2H": "constexpr"}
        attrs = {(i,): [["tt.divisibility", 16]] for i in range(4)}
        try:
            k = triton.compile(ASTSource(fn=kpv8, signature=sig, constexprs={"RB": 64, "D": 64, "T": 128, "PV8": PV8, "EX2H": EX2H}, attrs=attrs),
                               target=tgt, options={"num_warps": 4, "num_stages": 3})
            report(k, f"pv8{int(PV8)}_ex2h{int(EX2H)}")
        except Exception as e:
            print(f"[pv8={PV8} ex2h={EX2H}] FAILED: {str(e)[:400]}")
