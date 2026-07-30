#!/usr/bin/env bash
# From-zero provisioning of a fresh GPU box (vast.ai base image or similar).
# Everything setup_pod.sh assumes to exist gets created here. Idempotent;
# setup_pod.sh calls it automatically when the layout is missing.
#
# Cost profile on a fresh box: ~2.7 GB checkpoint download + two clones +
# (once) the cuda-12.8 toolkit. The expensive part of a fresh pod is NOT
# this — it is compiling gsplat/fused-*/asmk, which snapshot_env.sh
# persists to HF as wheels so setup_pod.sh can restore instead of rebuild.
set -euo pipefail
WS=/workspace

echo "== provision: tools =="
if ! { command -v git && command -v curl && command -v ffmpeg && command -v rsync; } >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -qq ffmpeg git curl rsync >/dev/null
fi

echo "== provision: layout =="
mkdir -p "$WS/sitecustom" "$WS/inputs"
# torch 2.8 defaults weights_only=True; MASt3R checkpoints contain
# argparse.Namespace and won't load without this shim (see docs/pod-setup.md).
cat > "$WS/sitecustom/sitecustomize.py" <<'PY'
import torch
_orig = torch.load
def _patched(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig(*a, **kw)
torch.load = _patched
PY
grep -q sitecustom ~/.bashrc 2>/dev/null || \
    echo 'export PYTHONPATH=/workspace/sitecustom:$PYTHONPATH' >> ~/.bashrc

echo "== provision: sources from HF (one pull instead of clones + CDN) =="
# snapshot_env.sh's sibling bundle: mast3r (with checkpoints) + pinned gsplat.
# Plain curl — runs before any Python tooling exists. Fallbacks below cover
# a missing bundle, missing token, or partial extraction (steps are guarded).
HF_SRC_REPO="${HF_REPO_ID:-miguelemosreverte/alambique-datasets}"
if [ ! -d "$WS/mast3r/.git" ] && [ -s "$HOME/.cache/huggingface/token" ]; then
    if curl -sfL -m 900 -H "Authorization: Bearer $(cat "$HOME/.cache/huggingface/token")" \
        "https://huggingface.co/datasets/$HF_SRC_REPO/resolve/main/pod-env/igen-splat/sources/igen-sources.tar.gz" \
        -o /tmp/igen-sources.tar.gz; then
        tar xzf /tmp/igen-sources.tar.gz -C "$WS" && rm -f /tmp/igen-sources.tar.gz
        echo "  [ok] mast3r + gsplat + checkpoints restored from HF"
    else
        echo "  [!!] HF sources bundle unavailable — falling back to github/naver"
    fi
fi

echo "== provision: mast3r =="
[ -d "$WS/mast3r/.git" ] || \
    git clone --quiet --recursive https://github.com/naver/mast3r "$WS/mast3r"
mkdir -p "$WS/mast3r/checkpoints"
cd "$WS/mast3r/checkpoints"
for f in MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
         MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth \
         MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl; do
    [ -s "$f" ] || {
        echo "  downloading $f"
        curl -sL -o "$f" "https://download.europe.naverlabs.com/ComputerVision/MASt3R/$f"
    }
done

echo "== provision: gsplat =="
# Pinned: last upstream commit with gsplat/color_correct.py (which the PyPI
# 1.5.3 wheel lacks and simple_trainer.py imports) BEFORE the CUDA-13 API
# migration (4561ac47 "Add mGPU support" switches cudaEventCreate signature,
# breaking builds against the CUDA 12.8 toolkit that matches torch cu128).
GSPLAT_REF=608e19ad1657815b685a14f1735ac830838c240e
[ -d "$WS/gsplat/.git" ] || \
    git clone --quiet --recursive https://github.com/nerfstudio-project/gsplat "$WS/gsplat"
( cd "$WS/gsplat" && git fetch -q --tags && git checkout -q "$GSPLAT_REF" \
    && git submodule update --init --recursive --quiet )

# NOTE: the CUDA 12.8 toolkit (needed only for SOURCE builds — torch cu128
# requires a matching-major nvcc) is installed lazily by setup_pod.sh's
# ensure_cuda128, and only when the HF wheelhouse can't provide a wheel.
# A wheelhouse-restored pod never compiles and never pays the ~5 min apt.

echo "== provision done =="
