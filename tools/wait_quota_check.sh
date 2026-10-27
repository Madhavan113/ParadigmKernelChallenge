#!/bin/bash
# Poll the proj check quota; when a check is available, submit the current submission.py as a check and
# wait for its result. Usage: wait_quota_check.sh <tag>   (writes /tmp/okc_<tag>_submit.json and /tmp/okc_<tag>.json)
TAG="$1"
OKC="$HOME/.okc/bin/okc"
DIR="/Users/madhavanp/kerneloptimizer/.claude/worktrees/agent-ae3cdecd65bf7abee"
while true; do
  LEFT=$("$OKC" problems 2>/dev/null | grep -m1 "^muse_bf16_proj " | sed -E 's/.* ([0-9]+) of 20 checks.*/\1/')
  if [ -n "$LEFT" ] && [ "$LEFT" -ge 1 ] 2>/dev/null; then
    break
  fi
  sleep 300
done
echo "quota available ($LEFT): submitting at $(date)"
"$OKC" check "$DIR/muse_bf16_proj/submission.py" --no-wait --json > "/tmp/okc_${TAG}_submit.json" 2>&1
JOB=$(python3 -c "import json,sys; print(json.load(open('/tmp/okc_${TAG}_submit.json'))['job_id'])" 2>/dev/null)
echo "job $JOB"
if [ -z "$JOB" ]; then
  cat "/tmp/okc_${TAG}_submit.json"
  exit 1
fi
"$OKC" status "$JOB" --wait --json > "/tmp/okc_${TAG}.json" 2>&1
echo "finished $JOB at $(date)"
