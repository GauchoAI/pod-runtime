#!/usr/bin/env bash
# Pod self-update loop: the CI/CD delivery end. Push to this repo -> every
# pod has it within a minute. No credentials needed (repo is public).
# HF credentials are NEVER in this repo: the pod expects HF_TOKEN in env or
# ~/.cache/huggingface/token, placed there BY HAND at provision time.
cd "$(dirname "$0")"
while true; do
  git fetch --quiet origin main 2>/dev/null
  LOCAL=$(git rev-parse @); REMOTE=$(git rev-parse origin/main 2>/dev/null || echo "$LOCAL")
  if [ "$LOCAL" != "$REMOTE" ]; then
    git reset --hard origin/main --quiet && echo "[sync] updated to $(git rev-parse --short HEAD) $(date -u +%H:%M:%S)"
  fi
  sleep 60
done
