"""Edit: cover the fused pooler + router launch (sk2f) in prepare's FP64 self-test (both outputs, m in 1..32); a failure
only prints (the standalone never calls it; the decode layer's composition does). Run: python3 tools/_edit_fused_selftest.py"""
import pathlib
p = pathlib.Path("muse_bf16_proj/submission.py")
s = p.read_text()
old = '''                if not err < tol:
                    state["disabled"].add(key)
                    print(f"[okc] selftest {layer} m={m} {ch[0]}: rel err {err:.3e} -> disabled", file=sys.stderr)
'''
new = '''                if not err < tol:
                    state["disabled"].add(key)
                    print(f"[okc] selftest {layer} m={m} {ch[0]}: rel err {err:.3e} -> disabled", file=sys.stderr)
    # the fused pooler + router launch (decode layer only): both outputs against FP64, every token bucket
    state["fused_ok"] = False
    if state.get("sm90") and SK2_FUSED_CONFIG in state.get("sk2", {}) and "router" in state.get("sk2", {}):
        try:
            worst = 0.0
            for m in (1, 8, 16, 32):
                x = torch.randn(m, 8192, device=weights["pooler"].device, generator=g).to(torch.bfloat16)
                out, outr = fused_pooler_router(x, state)
                torch.cuda.synchronize()
                ref = (x.double() @ weights["pooler"].double().T).to(torch.bfloat16).double()
                refr = x.double() @ weights["router"].double().T
                e1 = ((out.double() - ref).norm() / ref.norm()).item()
                e2 = ((outr.double() - refr).norm() / refr.norm()).item()
                worst = max(worst, e1 / 5e-3, e2 / 2e-5)
            state["fused_ok"] = worst < 1.0
            print(f"[okc] selftest fused pooler+router: worst err/tol {worst:.3f} -> {'ok' if state['fused_ok'] else 'FAIL'}",
                  file=sys.stderr)
        except RuntimeError as e:   # noqa: BLE001
            print(f"[okc] selftest fused pooler+router: {str(e)[:200]}", file=sys.stderr)
'''
assert s.count(old) == 1
s = s.replace(old, new)
p.write_text(s)
print("ok")
