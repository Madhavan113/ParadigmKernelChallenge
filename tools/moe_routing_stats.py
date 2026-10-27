"""Routing statistics of the judge's generate_input (CPU, inside the judge image): touched local experts, tokens per
expert (n-tiles), experts with > 8 tokens (MID SPLIT re-streams), bytes streamed. Usage (in the image):
  python3 /work/tools/moe_routing_stats.py [nseeds]"""
import sys, collections
sys.path.insert(0, "/work/muse_moe_w4a16")
import torch
from reference import generate_input
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
MB = 28.31
for bs, dist in [(1, "normal"), (8, "normal"), (16, "normal"), (32, "normal"), (128, "normal"), (256, "normal"), (128, "skewed"), (4, "normal"), (33, "normal"), (257, "normal")]:
    touched, routes, over8, mb_once, mb_split, ntile_hist, maxtok, jobs16 = [], [], [], [], [], collections.Counter(), [], []
    for seed in range(N):
        d = generate_input(bs, "ep8", 1000 + seed, dist=dist, device="cpu")
        ids = d["topk_ids"]
        loc = ids[(ids >= 0) & (ids < 32)]
        cnt = torch.bincount(loc, minlength=32)
        t = int((cnt > 0).sum()); touched.append(t); routes.append(int(loc.numel()))
        c = cnt[cnt > 0]
        over8.append(int((c > 8).sum()))
        mb_once.append(t * MB)
        mb_split.append(float(((c + 7) // 8).sum()) * MB)
        jobs16.append(float(((c + 63) // 64).sum()) * MB)
        maxtok.append(int(c.max()))
        for v in c.tolist():
            ntile_hist[(v + 7) // 8] += 1
    def q(a, f):
        a = sorted(a); return a[int(f * (len(a) - 1))]
    tot = sum(ntile_hist.values())
    print(f"bs={bs:3d} {dist:6s} routes med {q(routes,.5):3d}  touched p10/p50/p90 {q(touched,.1)}/{q(touched,.5)}/{q(touched,.9)}  "
          f"experts>8tok p50/p90 {q(over8,.5)}/{q(over8,.9)}  maxtok p50/p90 {q(maxtok,.5)}/{q(maxtok,.9)}  "
          f"MB once {q(mb_once,.5):6.0f} split8 {q(mb_split,.5):6.0f} split64 {q(jobs16,.5):6.0f} (p90 {q(jobs16,.9):5.0f})  "
          f"ntile share {{{', '.join(f'{k}:{v/tot:.2f}' for k, v in sorted(ntile_hist.items()))}}}", flush=True)
