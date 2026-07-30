#!/usr/bin/env bash
# Air-side, run at EVERY pod creation: ships the two hand-delivered secrets.
#   pod/push_secrets.sh <ssh-host> <ssh-port>
# HF token from ~/.cache/huggingface/token; GitHub token from the admin
# repo's .env (GITHUB_TOKEN=...). Tokens travel over stdin, never argv.
set -e
HOST=${1:?ssh host}; PORT=${2:?ssh port}
HF=$(cat "$HOME/.cache/huggingface/token")
GH=$(grep '^GITHUB_TOKEN=' "$HOME/Desktop/image-generation/.env" | cut -d= -f2)
ssh -p "$PORT" "root@$HOST" 'mkdir -p /root/.config/pod-secrets /root/.cache/huggingface
read -r HF; read -r GH
printf "%s" "$HF" > /root/.cache/huggingface/token && chmod 600 /root/.cache/huggingface/token
printf "%s" "$GH" > /root/.config/pod-secrets/github-token && chmod 600 /root/.config/pod-secrets/github-token
echo "secrets installed"' <<< "$HF
$GH"
