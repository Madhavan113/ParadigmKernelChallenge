#!/bin/bash
# Run a command chain on the H200 box detached from the ssh session (survives link drops), with retries on connect.
#   tools/attn_remote_run.sh <name> '<remote command chain>'
# Syncs muse_attention_decode/submission.py and the two tools first; the remote log is
# ~/kerneloptimizer/muse_attention_decode/_logs/<name>.log (also fetched to ~/<name>.log here on completion).
# Poll: tools/attn_remote_run.sh --poll <name>   (prints the remote log tail; exits 0 when the DONE marker is present)
set -u
HOST=shiba@204.12.171.27
SSH="ssh -o ConnectTimeout=20 -o ServerAliveInterval=15 -o BatchMode=yes"
REPO=/Users/madhavanp/kerneloptimizer/.claude/worktrees/agent-af7174b5096001399
retry() {   # retry a command up to 40 times, 30 s apart (flaky link)
  local n=0
  until "$@"; do n=$((n + 1)); [ $n -ge ${RETRIES:-400} ] && return 1; sleep 30; done
}
if [ "$1" = "--poll" ]; then
  name=$2
  retry $SSH $HOST "cat ~/kerneloptimizer/muse_attention_decode/_logs/$name.log 2>/dev/null" > "$HOME/$name.log" || exit 2
  grep -q "^DONE" "$HOME/$name.log" && exit 0
  exit 1
fi
name=$1; chain=$2
retry rsync -az -e "$SSH" "$REPO/muse_attention_decode/submission.py" "$HOST:~/kerneloptimizer/muse_attention_decode/submission.py" || exit 2
retry rsync -az -e "$SSH" "$REPO/tools/attn_ab.py" "$HOST:~/kerneloptimizer/muse_attention_decode/_attn_ab.py" || exit 2
retry rsync -az -e "$SSH" "$REPO/tools/attn_timeline.py" "$HOST:~/kerneloptimizer/muse_attention_decode/_attn_timeline.py" || exit 2
printf '%s\n' "$chain" > "/tmp/_remote_chain_$name.sh"
retry $SSH $HOST "mkdir -p ~/kerneloptimizer/muse_attention_decode/_logs" || exit 2
retry rsync -az -e "$SSH" "/tmp/_remote_chain_$name.sh" "$HOST:~/kerneloptimizer/muse_attention_decode/_logs/$name.sh" || exit 2
retry $SSH $HOST "cd ~/kerneloptimizer; nohup bash -c 'bash muse_attention_decode/_logs/$name.sh; echo DONE' > ~/kerneloptimizer/muse_attention_decode/_logs/$name.log 2>&1 < /dev/null & echo launched" || exit 2
