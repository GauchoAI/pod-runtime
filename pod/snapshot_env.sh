#!/usr/bin/env bash
# Persist the expensive-to-rebuild parts of the pod env to Hugging Face,
# so a brand-new box restores in minutes instead of recompiling.
#
# What gets snapshotted (to $HF_REPO_ID under pod-env/igen-splat/):
#   wheelhouse/*.whl   — gsplat, fused-ssim, fused-bilagrid, asmk, built
#                        with TORCH_CUDA_ARCH_LIST spanning 3090/4090/H100
#   manifest.json      — torch/cuda versions, gsplat commit, build date
#
# MASt3R checkpoints are NOT snapshotted — naver's CDN is fast and free.
# The pip-index packages (torch, the lockfile graph) are NOT snapshotted —
# uv reinstalls them in ~3 min on datacenter bandwidth.
#
# Run on the pod AFTER a green setup_pod.sh:
#   bash /workspace/scripts/snapshot_env.sh
set -euo pipefail
WS=/workspace
WHEELHOUSE=$WS/wheelhouse
export PATH="/root/.local/bin:$PATH"
[ -d /usr/local/cuda-12.8 ] && export CUDA_HOME=/usr/local/cuda-12.8 && export PATH="$CUDA_HOME/bin:$PATH"

# Fat binaries: sm_86 (3090), sm_89 (4090), sm_90 (H100), +PTX for newer.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6;8.9;9.0+PTX}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"

mkdir -p "$WHEELHOUSE"
cd "$WHEELHOUSE"

echo "== building wheels (TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST) =="
build() {  # build <src> unless a wheel matching <pattern> already exists
    local pattern=$1 src=$2
    if ls "$WHEELHOUSE"/$pattern >/dev/null 2>&1; then
        echo "  [ok] $pattern already built"
    else
        python3 -m pip wheel --no-build-isolation --no-deps -w "$WHEELHOUSE" "$src"
    fi
}
LOCK=$WS/scripts/requirements.lock.txt
build 'gsplat-*.whl'         "$WS/gsplat"
build 'fused_ssim-*.whl'     "$(grep -o 'git+https://github.com/rahul-goel/fused-ssim@[0-9a-f]*' "$LOCK")"
build 'fused_bilagrid-*.whl' "$(grep -o 'git+https://github.com/harry7557558/fused-bilagrid@[0-9a-f]*' "$LOCK")"
build 'ppisp-*.whl'          "$(grep -o 'git+https://github.com/nv-tlabs/ppisp@[0-9a-f]*' "$LOCK")"
if [ -d /tmp/asmk ]; then
    build 'asmk-*.whl' /tmp/asmk
else
    echo "  [!!] /tmp/asmk missing (setup_pod.sh clones it) — skipping asmk wheel"
fi
ls -lh "$WHEELHOUSE"

echo "== manifest =="
python3 - "$WHEELHOUSE/manifest.json" <<'PY'
import json, subprocess, sys, torch
gsplat_commit = subprocess.run(
    ["git", "-C", "/workspace/gsplat", "rev-parse", "HEAD"],
    capture_output=True, text=True).stdout.strip()
json.dump({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "python": ".".join(map(str, sys.version_info[:3])),
    "gsplat_commit": gsplat_commit,
    "arch_list": __import__("os").environ.get("TORCH_CUDA_ARCH_LIST"),
}, open(sys.argv[1], "w"), indent=2)
print(open(sys.argv[1]).read())
PY

echo "== uploading to HF =="
python3 "$WS/scripts/hf_upload.py" "$WHEELHOUSE"/*.whl "$WHEELHOUSE/manifest.json" \
    --prefix pod-env/igen-splat/wheelhouse
echo "== snapshot done =="
