#!/bin/bash
# Round 4 chain: --diag switch in the bench generator, apply the stream-rate edit, gates, geometry, locked offline build,
# regenerate the prefill bench file with the sk2 diagnostics on, submit it as a prefill check and wait.
# Usage: bash tools/run_round4.sh <tag>
set -e
TAG="$1"
W=/Users/madhavanp/kerneloptimizer/.claude/worktrees/agent-a03a0e505e0547c6d
cd "$W"
python3 - <<'EOF'
import pathlib
p = pathlib.Path("tools/make_prefill_bench.py"); s = p.read_text()
if "--diag" not in s:
    s = s.replace('''    global bfp_CUTLASS_LAYERS
    bfp_CUTLASS_LAYERS = set()   # no CUTLASS build in the bench slot (170 s join); the proj check measures it
''', '''    global bfp_CUTLASS_LAYERS, bfp_DIAG_SK2
    bfp_CUTLASS_LAYERS = set()   # no CUTLASS build in the bench slot (170 s join); the proj check measures it
    bfp_DIAG_SK2 = @@DIAG@@      # sk2 diagnostics (dg modes) compiled in and timed from prepare
''')
    s = s.replace('''src = pre.rstrip("\\n") +''', '''bench = bench.replace("@@DIAG@@", "True" if "--diag" in sys.argv else "False")
src = pre.rstrip("\\n") +''')
    p.write_text(s)
print("generator ok")
EOF
python3 tools/_edit_round4.py
python3 -c "compile(open('muse_bf16_proj/submission.py').read(),'s','exec'); print('compile ok')"
python3 -m pyflakes muse_bf16_proj/submission.py | grep -v "imported but unused\|redefinition of unused\|n_cfg\|MID_SRC\|'dev'\|'ext' is assigned" || true
python3 - <<'EOF'
src = open("muse_bf16_proj/submission.py").read()
def grab(a, b):
    i = src.index(a); j = src.index(b, i); return src[i:j]
ns = {}
exec(grab("SK2_CONFIGS = [", "SK2_ROT = False"), ns)
exec(grab("SK2_K = {", "USE_SK2 = True"), ns)
exec(grab("def _sk2_geom(cfg, tokr):", "def _sk2_src(sm90):"), ns)
for cfg in ns["SK2_CONFIGS"]:
    name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
    for tokr in (8, 16, 32):
        if name not in ns["SK2_ALT"].get((layer, tokr), ()):
            continue
        hl, nstage = ns["_sk2_geom"](cfg, tokr)
        ups = WSK * UPW; wby = NWR * WSK * UPW * NT * 768; xby = ups * tokr * 128
        sst = (xby + wby + 1023) // 1024 * 1024
        toks = 32 if tokr == 32 else 16
        nesc = min(nstage, 4)
        smem = 1024 + nstage * sst + 16 * nstage + 32 + nesc * ((toks * (R + 1) + 3) // 4 * 4) * 4 + 64 * 8 + 128
        thr = (NWR * WSK + 1 + nesc) * 32
        per_sm = min(232448 // (smem + 1024), 2048 // thr, 32)
        print(f"{name:18s} tokr{tokr:2d} R={R:3d} copy={wby // 1024:3d}KB stage={sst // 1024:3d}KB nstage={nstage} inflight={nstage * wby // 1024:3d}KB smem={smem // 1024:3d}KB grid={n_rows // R * CL} CTAs/SM<={per_sm}")
EOF
python3 tools/okc_lock.py -w 1800 /tmp/okc_nvcc.lock docker run --rm --platform linux/amd64 -v "$W":/work -w /work/tools lmsysorg/sglang:dev-muse-spark-open bash -c "python3 nvcc_pub.py --only sk2 --sk2-alt-only --emit && python3 nvcc_pub.py --build 'r100|r192|r216|r220'; echo rc=\$?" > /tmp/nvcc_r4.txt 2>&1
grep -v "deprecated\|13101\|No CUDA runtime\|^==\|CUDA Version\|Container\|NVIDIA\|license\|^$\|container\|nvidia" /tmp/nvcc_r4.txt | tail -40
grep -q "^rc=0" /tmp/nvcc_r4.txt
if grep -q "spill" /tmp/nvcc_r4.txt; then echo "SPILLS FOUND"; exit 1; fi
python3 tools/make_prefill_bench.py --diag -o muse_prefill_layer/bench_proj.py
grep -c "bfp_DIAG_SK2 = True" muse_prefill_layer/bench_proj.py
B=$(python3 -m pyflakes muse_prefill_layer/bench_proj.py | grep "undefined name" | grep -vc "MID_SRC")
echo "undefined=$B"
test "$B" = "31"
~/.okc/bin/okc check muse_prefill_layer/bench_proj.py --no-wait --json > "/tmp/okc_${TAG}_submit.json" 2>&1
cat "/tmp/okc_${TAG}_submit.json"
JOB=$(python3 -c "import json; print(json.load(open('/tmp/okc_${TAG}_submit.json'))['job_id'])")
~/.okc/bin/okc status "$JOB" --wait --json > "/tmp/okc_${TAG}.json" 2>&1
python3 -c "import json; d=json.load(open('/tmp/okc_${TAG}.json')); r=d.get('result') or {}; print('status', d.get('status'), 'speedup', r.get('speedup'), 'error', (r.get('error') or '')[:400]); print(d.get('output'))"
