"""One-off edit: 2-stage variants (more resident CTAs per SM for the ALU-bound shapes) in the switch and the bench lists."""
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
p = root / "muse_moe_w4a16/cuda_moe.py"
s = p.read_text()
old = s[s.index("        case 111104: OKC_W(true, 1, 1, 1, 4); break;"):s.index('        default: TORCH_CHECK(false, "unsupported wt variant (mw, nt, ks, nstage)");')]
new = """        case 111102: OKC_W(true, 1, 1, 1, 2); break;   case 111104: OKC_W(true, 1, 1, 1, 4); break;
        case 111108: OKC_W(true, 1, 1, 1, 8); break;
        case 112102: OKC_W(true, 1, 2, 1, 2); break;   case 112104: OKC_W(true, 1, 2, 1, 4); break;
        case 114102: OKC_W(true, 1, 4, 1, 2); break;   case 114104: OKC_W(true, 1, 4, 1, 4); break;
        case 11102: OKC_W(false, 1, 1, 1, 2); break;   case 11104: OKC_W(false, 1, 1, 1, 4); break;
        case 11108: OKC_W(false, 1, 1, 1, 8); break;
        case 21102: OKC_W(false, 2, 1, 1, 2); break;   case 21104: OKC_W(false, 2, 1, 1, 4); break;
        case 12104: OKC_W(false, 1, 2, 1, 4); break;   case 22102: OKC_W(false, 2, 2, 1, 2); break;
        case 22104: OKC_W(false, 2, 2, 1, 4); break;   case 14104: OKC_W(false, 1, 4, 1, 4); break;
        case 24102: OKC_W(false, 2, 4, 1, 2); break;   case 24104: OKC_W(false, 2, 4, 1, 4); break;
"""
s = s.replace(old, new)
p.write_text(s)

q = root / "muse_moe_w4a16/submission.py"
t = q.read_text()
old = "    small = [(1, 1, 1, 4, 1), (1, 1, 1, 8, 1), (1, 1, 1, 8, 2), (1, 1, 1, 8, 4), (1, 1, 1, 4, 4), (1, 2, 1, 8, 2), (1, 2, 1, 4, 2), (1, 2, 1, 4, 4)]\n    big = [(2, 2, 1, 4, 1), (4, 2, 1, 4, 1), (2, 2, 1, 4, 2), (2, 1, 1, 4, 1), (4, 1, 1, 4, 1)]"
new = "    small = [(1, 1, 1, 2, 1), (1, 1, 1, 4, 1), (1, 1, 1, 8, 1), (1, 1, 1, 8, 2), (1, 1, 1, 8, 4), (1, 1, 1, 4, 4), (1, 2, 1, 8, 2), (1, 2, 1, 4, 2), (1, 2, 1, 2, 1), (1, 1, 1, 2, 2)]\n    big = [(2, 2, 1, 4, 1), (2, 2, 1, 2, 1), (4, 2, 1, 4, 1), (4, 2, 1, 2, 1), (2, 1, 1, 2, 1), (4, 1, 1, 4, 1), (2, 2, 1, 4, 2)]"
assert t.count(old) == 1
t = t.replace(old, new)
old = """WT_TEST_VARIANTS = ((1, 1, 1, 4, 1), (1, 1, 1, 8, 1), (1, 1, 1, 8, 2), (1, 1, 1, 8, 4), (1, 1, 1, 4, 4), (1, 2, 1, 8, 2),
                    (1, 2, 1, 4, 2), (1, 2, 1, 4, 4), (2, 2, 1, 4, 1), (2, 2, 1, 4, 2), (4, 2, 1, 4, 1), (2, 1, 1, 4, 1), (4, 1, 1, 4, 1))"""
new = """WT_TEST_VARIANTS = ((1, 1, 1, 4, 1), (1, 1, 1, 2, 1), (1, 1, 1, 8, 1), (1, 1, 1, 8, 2), (1, 1, 1, 8, 4), (1, 1, 1, 4, 4), (1, 1, 1, 2, 2),
                    (1, 2, 1, 8, 2), (1, 2, 1, 4, 2), (1, 2, 1, 2, 1), (2, 2, 1, 4, 1), (2, 2, 1, 2, 1), (2, 2, 1, 4, 2),
                    (4, 2, 1, 4, 1), (4, 2, 1, 2, 1), (2, 1, 1, 2, 1), (4, 1, 1, 4, 1))"""
assert t.count(old) == 1
t = t.replace(old, new)
q.write_text(t)
print("edited")
