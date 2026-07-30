#!/usr/bin/env bash
# Wake-up hook: a stopped pod that restarts only needs this.
cd "$(dirname "$0")" && git pull --ff-only --quiet || true
pgrep -f "pod-runtime/sync.sh" >/dev/null || setsid nohup bash sync.sh >> /workspace/sync.log 2>&1 < /dev/null &
