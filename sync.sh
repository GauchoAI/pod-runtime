#!/usr/bin/env bash
export GIT_LFS_SKIP_SMUDGE=1  # media lives on HF, never via LFS (budget exhausted by design)
# Pod self-update loop: the CI/CD delivery end. Push -> every pod has it
# within a minute. This public repo needs no credentials; private repos are
# pulled too when the admin has placed the GitHub token by hand (see README).
# HF credentials are NEVER in this repo: HF_TOKEN env or
# ~/.cache/huggingface/token, also placed by hand.
cd "$(dirname "$0")"
while true; do
  git fetch --quiet origin main 2>/dev/null
  LOCAL=$(git rev-parse @); REMOTE=$(git rev-parse origin/main 2>/dev/null || echo "$LOCAL")
  if [ "$LOCAL" != "$REMOTE" ]; then
    git reset --hard origin/main --quiet && echo "[sync] pod-runtime -> $(git rev-parse --short HEAD) $(date -u +%H:%M:%S)"
    pkill -f "pod-runtime/worker.py" 2>/dev/null   # new code -> new worker
    bash onstart.sh                                 # restart daemons, repull repos
  fi
  if [ -s /root/.config/pod-secrets/github-token ]; then
    for D in /workspace/image-generation /workspace/neural-landscape; do
      [ -d "$D/.git" ] && git -C "$D" pull --ff-only --quiet 2>/dev/null || true
    done
  fi
  sleep 60
done
