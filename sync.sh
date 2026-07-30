#!/usr/bin/env bash
# Pod self-update loop. Push -> every pod has it within a minute. Public repo,
# zero credentials. HF token + GitHub token are placed BY HAND (see README).
export GIT_LFS_SKIP_SMUDGE=1
export GIT_TERMINAL_PROMPT=0
cd /workspace/pod-runtime
while true; do
  timeout 60 git fetch --quiet origin main 2>/dev/null
  LOCAL=$(git rev-parse @); REMOTE=$(git rev-parse origin/main 2>/dev/null || echo "$LOCAL")
  if [ "$LOCAL" != "$REMOTE" ]; then
    git reset --hard origin/main --quiet && echo "[sync] pod-runtime -> $(git rev-parse --short HEAD) $(date -u +%H:%M:%S)"
    while [ -f /tmp/worker-busy ]; do sleep 30; done  # never behead a working worker
    pkill -f "pod-runtime/worker.py" 2>/dev/null   # new code -> new worker
    setsid nohup bash /workspace/pod-runtime/onstart.sh >> /workspace/onstart.log 2>&1 < /dev/null &
    exec bash /workspace/pod-runtime/sync.sh        # reload THIS loop too (absolute path)
  fi
  # daemon guard every cycle: a dead worker resurrects within a minute
  pgrep -f "pod-runtime/worker.py" >/dev/null || setsid nohup python3 -u /workspace/pod-runtime/worker.py >> /workspace/worker.log 2>&1 < /dev/null &
  sleep 60
done
