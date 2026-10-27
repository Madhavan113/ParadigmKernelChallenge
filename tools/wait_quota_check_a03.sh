#!/bin/bash
# Poll the proj check quota; when a check is available, submit this worktree's submission.py as a check and wait
# for its result. Usage: wait_quota_check_a03.sh <tag>   (writes /tmp/okc_<tag>_submit.json and /tmp/okc_<tag>.json)
TAG="$1"
OKC="$HOME/.okc/bin/okc"
DIR="/Users/madhavanp/kerneloptimizer/.claude/worktrees/agent-a03a0e505e0547c6d"
while true; do
  LEFT=$("$OKC" problems --json 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); d=d if isinstance(d,list) else d.get('problems',d); print([p for p in d if p['name']=='muse_bf16_proj'][0]['limits']['checks']['left'])" 2>/dev/null)
  if [ -n "$LEFT" ] && [ "$LEFT" -ge 1 ] 2>/dev/null; then
    break
  fi
  sleep 60
done
echo "quota available ($LEFT): submitting at $(date)"
"$OKC" check "$DIR/muse_bf16_proj/submission.py" --no-wait --json > "/tmp/okc_${TAG}_submit.json" 2>&1
JOB=$(python3 -c "import json,sys; d=json.load(open('/tmp/okc_${TAG}_submit.json')); print(d.get('job_id') or d.get('id') or d.get('job'))" 2>/dev/null)
echo "job $JOB"
if [ -z "$JOB" ] || [ "$JOB" = "None" ]; then
  cat "/tmp/okc_${TAG}_submit.json"
  exit 1
fi
"$OKC" status "$JOB" --wait --json > "/tmp/okc_${TAG}.json" 2>&1
echo "finished $JOB at $(date)"
python3 -c "import json; d=json.load(open('/tmp/okc_${TAG}.json')); r=d.get('result') or {}; print('status', d.get('status'), 'speedup', r.get('speedup'), 'error', (r.get('error') or '')[:300])"
