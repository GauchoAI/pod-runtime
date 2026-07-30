#!/usr/bin/env bash
export GIT_LFS_SKIP_SMUDGE=1  # media lives on HF, never via LFS (budget exhausted by design)
# Wake-up hook: a stopped pod that restarts only needs this.
cd "$(dirname "$0")" && git pull --ff-only --quiet || true
# Private repos: readable IFF the admin placed the GitHub token by hand.
GHTOK=/root/.config/pod-secrets/github-token
if [ -s "$GHTOK" ]; then
  CRED=/root/.config/pod-secrets/git-credentials
  echo "https://x-access-token:$(cat "$GHTOK")@github.com" > "$CRED" && chmod 600 "$CRED"
  git config --global credential.helper "store --file=$CRED"
  for R in GauchoAI/image-generation miguelemosreverte/neural-landscape; do
    D="/workspace/$(basename "$R")"
    if [ -d "$D/.git" ]; then git -C "$D" pull --ff-only --quiet || true
    else git clone --quiet "https://github.com/$R" "$D" || true; fi
  done
fi
pgrep -f "pod-runtime/sync.sh" >/dev/null || setsid nohup bash sync.sh >> /workspace/sync.log 2>&1 < /dev/null &
pgrep -f "pod-runtime/worker.py" >/dev/null || setsid nohup python3 worker.py >> /workspace/worker.log 2>&1 < /dev/null &
