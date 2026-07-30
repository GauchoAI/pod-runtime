#!/usr/bin/env bash
# One-shot pod setup for the video -> PLY pipeline.
#
# Idempotent. Run it after any event that may have drifted the pod:
#   - first provisioning
#   - RunPod volume resize (REIMAGES the container: loses /usr/local pip
#     packages, keeps /workspace)
#   - container restart for any reason
#   - manually broken environment
#
# What it does, in order:
#   1. ensure core /workspace dirs exist (mast3r, gsplat, sitecustom, checkpoints)
#   2. install `uv` (the fast package manager) if missing
#   3. install every Python dep we need via
#        uv pip install --system --no-build-isolation -r requirements.txt
#      pinned against `requirements.lock.txt` for reproducibility
#   4. override the PyPI gsplat wheel with an editable install from
#      /workspace/gsplat (the PyPI wheel is missing the `color_correct`
#      submodule that simple_trainer.py imports)
#   5. patch pycolmap for numpy ≥ 2 compatibility
#   6. install asmk (MASt3R retrieval fallback) — requires regenerating
#      its pre-shipped Cython C file on Python 3.12 because
#      longintrepr.h was removed
#   7. copy this dir's scripts into /workspace/scripts/ and /workspace/mast3r/
#   8. sanity-check imports of mast3r, gsplat, pycolmap, torch, fused_ssim
#   9. clean stale outputs
#  10. report free disk
#
# Usage (on the pod):
#   bash /workspace/scripts/setup_pod.sh
# Or from your laptop:
#   rsync -avz -e "ssh -p $POD_PORT" scripts/pod/ "$POD:/workspace/scripts/"
#   ssh -p $POD_PORT $POD "bash /workspace/scripts/setup_pod.sh"

set -euo pipefail

WS=/workspace
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ok()   { printf "  [ok]  %s\n" "$*"; }
warn() { printf "  [!!]  %s\n" "$*"; }
fail() { printf "  [FAIL] %s\n" "$*"; exit 1; }

# -----------------------------------------------------------------------------
# 1. pod layout
# -----------------------------------------------------------------------------
echo "== pod layout =="
# A fresh box (vast.ai base image) has none of this — provision it rather
# than fail. provision_pod.sh is idempotent, so partial layouts heal too.
if [ ! -d "$WS/mast3r/checkpoints" ] || [ ! -d "$WS/gsplat" ] || [ ! -d "$WS/sitecustom" ]; then
    bash "$SCRIPT_DIR/provision_pod.sh" || fail "provisioning failed"
fi
[ -d "$WS/mast3r" ]            || fail "$WS/mast3r not present after provisioning"
[ -d "$WS/mast3r/checkpoints" ]|| fail "MASt3R checkpoints missing after provisioning"
[ -d "$WS/gsplat" ]            || fail "$WS/gsplat not present after provisioning"
[ -d "$WS/sitecustom" ]        || fail "$WS/sitecustom not present after provisioning"
ok "core dirs present"

# -----------------------------------------------------------------------------
# 2. uv
# -----------------------------------------------------------------------------
echo "== uv =="
export PATH="/root/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || fail "uv install failed"
    ok "uv installed"
else
    ok "uv already present ($(uv --version))"
fi

# -----------------------------------------------------------------------------
# 2b. wheelhouse restore (HF-persisted compiled binaries)
# -----------------------------------------------------------------------------
# snapshot_env.sh uploads the compiled wheels (gsplat, fused-*, asmk) to the
# HF dataset repo. Restoring them here turns a ~20 min compile into a
# ~30 s download. Absence of a snapshot (or of a token) is not an error —
# everything below falls back to building from source.
echo "== wheelhouse =="
WHEELHOUSE=$WS/wheelhouse
HF_SNAPSHOT_REPO="${HF_REPO_ID:-miguelemosreverte/alambique-datasets}"
if ls "$WHEELHOUSE"/*.whl >/dev/null 2>&1; then
    ok "wheelhouse already present ($(ls "$WHEELHOUSE"/*.whl | wc -l) wheels)"
else
    python3 -c 'import huggingface_hub' >/dev/null 2>&1 || \
        uv pip install --system --break-system-packages huggingface_hub >/dev/null 2>&1 || true
    mkdir -p "$WHEELHOUSE"
    python3 - "$HF_SNAPSHOT_REPO" "$WHEELHOUSE" <<'PY' && ok "wheelhouse restored from HF" || warn "no HF wheelhouse (will build from source)"
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], repo_type="dataset",
                  allow_patterns="pod-env/igen-splat/wheelhouse/*",
                  local_dir="/tmp/hf-wheelhouse")
import glob, shutil
whls = glob.glob("/tmp/hf-wheelhouse/pod-env/igen-splat/wheelhouse/*.whl")
if not whls:
    sys.exit(1)
for w in whls:
    shutil.copy(w, sys.argv[2])
print(f"  {len(whls)} wheels")
PY
fi

# -----------------------------------------------------------------------------
# 3. Python deps
# -----------------------------------------------------------------------------
echo "== python deps =="
# gsplat and the fused-* pins compile CUDA kernels at install; torch requires
# nvcc's major CUDA version to match its own (cu128 → CUDA 12.x). CUDA-13
# base images (vast.ai) ship only nvcc 13, so source builds need the 12.8
# toolkit — installed LAZILY, only when a wheelhouse wheel isn't available.
ensure_cuda128() {
    if [ ! -d /usr/local/cuda-12.8 ]; then
        warn "source build required — installing cuda-toolkit-12-8 (~5 min)"
        apt-get update -qq && apt-get install -y -qq cuda-toolkit-12-8 >/dev/null \
            || fail "cuda-toolkit-12-8 install failed (needed for source builds)"
    fi
    export CUDA_HOME=/usr/local/cuda-12.8
    export PATH="$CUDA_HOME/bin:$PATH"
    ok "CUDA_HOME=$CUDA_HOME (matching torch cu128)"
}
if [ -d /usr/local/cuda-12.8 ]; then
    export CUDA_HOME=/usr/local/cuda-12.8
    export PATH="$CUDA_HOME/bin:$PATH"
fi
# Prefer the lockfile when present (exact transitive pins). Fall back to
# the top-level requirements.txt if the lockfile is missing or the user
# explicitly asked for a fresh resolve.
FRESH_RESOLVE="${FRESH_RESOLVE:-0}"
REQ_TXT="$SCRIPT_DIR/requirements.txt"
REQ_LOCK="$SCRIPT_DIR/requirements.lock.txt"
[ -f "$REQ_TXT" ] || fail "missing $REQ_TXT"

# The +cu128 torch pins live on the PyTorch index, not PyPI. On RunPod
# torch came preinstalled so this never resolved remotely; on a bare image
# (e.g. vast.ai base) uv must be able to fetch it.
# unsafe-best-match: the PyTorch index also republishes common packages
# (certifi, …) at stale versions; uv's default first-index-wins strategy
# would pin to those and fail the lockfile. Both indexes are trusted.
TORCH_INDEX="--extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match"

# Two-phase install: the fused-* pins are source builds whose setup.py
# imports torch, and --no-build-isolation means torch must already be in
# the environment before they build. RunPod images preinstalled torch and
# masked this; bare images (vast.ai) hit it. Install torch first.
if ! python3 -c 'import torch' >/dev/null 2>&1; then
    if [ -f "$REQ_LOCK" ]; then
        TORCH_PINS=$(grep -E '^torch(vision|audio)?==' "$REQ_LOCK" || true)
    fi
    # shellcheck disable=SC2086
    uv pip install --system --break-system-packages $TORCH_INDEX \
        ${TORCH_PINS:-torch torchvision torchaudio} || fail "torch preinstall failed"
    ok "torch preinstalled for source builds"
fi

# Wheels restored from HF satisfy the source-built git+ pins; install them
# first and drop those lines from the effective lockfile so uv doesn't
# rebuild (with a possibly-mismatched nvcc) what the wheelhouse provides.
EFFECTIVE_LOCK="$REQ_LOCK"
PREINSTALLED=""
for W in fused_ssim fused_bilagrid ppisp; do
    if ls "$WHEELHOUSE"/${W}-*.whl >/dev/null 2>&1 && \
       uv pip install --system --break-system-packages "$WHEELHOUSE"/${W}-*.whl >/dev/null 2>&1; then
        PREINSTALLED="$PREINSTALLED $W"
    fi
done
if [ -n "$PREINSTALLED" ]; then
    EFFECTIVE_LOCK=$(mktemp)
    cp "$REQ_LOCK" "$EFFECTIVE_LOCK"
    for W in $PREINSTALLED; do
        sed -i "/^${W//_/[-_]} @ git+/d" "$EFFECTIVE_LOCK"
    done
    ok "wheelhouse preinstalled:$PREINSTALLED"
fi
# Source-built fused-* pins remain in the effective lock → nvcc needed.
if [ "$EFFECTIVE_LOCK" = "$REQ_LOCK" ] && grep -q 'git+' "$REQ_LOCK"; then
    ensure_cuda128
fi

if [ "$FRESH_RESOLVE" = "1" ] || [ ! -f "$REQ_LOCK" ]; then
    warn "fresh resolve from $REQ_TXT (this will overwrite the lockfile)"
    uv pip install --system --break-system-packages --no-build-isolation \
        $TORCH_INDEX -r "$REQ_TXT" || fail "pip install from requirements.txt failed"
    uv pip freeze --system 2>/dev/null | grep -v '^-e' > "$REQ_LOCK" || true
    ok "froze $(wc -l <"$REQ_LOCK") packages into $(basename "$REQ_LOCK")"
else
    uv pip install --system --break-system-packages --no-build-isolation \
        $TORCH_INDEX -r "$EFFECTIVE_LOCK" || fail "pip install from lockfile failed"
    ok "installed from lockfile ($(wc -l <"$EFFECTIVE_LOCK") packages)"
fi

# -----------------------------------------------------------------------------
# 4. gsplat from source tree (overrides PyPI wheel)
# -----------------------------------------------------------------------------
echo "== gsplat (source override) =="
# The PyPI gsplat 1.5.3 wheel is missing `gsplat.color_correct`, which
# simple_trainer.py imports. Preference order:
#   1. already importable with color_correct (idempotent re-run)
#   2. wheelhouse wheel from HF (compiled once by snapshot_env.sh)
#   3. editable build from /workspace/gsplat (slow: full CUDA compile)
if python3 -c 'from gsplat.color_correct import color_correct_affine' >/dev/null 2>&1; then
    ok "gsplat with color_correct already installed"
elif ls "$WHEELHOUSE"/gsplat-*.whl >/dev/null 2>&1; then
    uv pip install --system --break-system-packages "$WHEELHOUSE"/gsplat-*.whl \
        >/dev/null 2>&1 || fail "gsplat wheelhouse install failed"
    ok "gsplat installed from wheelhouse"
else
    ensure_cuda128
    uv pip install --system --break-system-packages --no-build-isolation \
        -e "$WS/gsplat" >/dev/null 2>&1 || fail "gsplat editable install failed"
    ok "gsplat built from /workspace/gsplat (editable)"
fi
python3 - <<'PY' || fail "gsplat.color_correct import failed"
from gsplat.color_correct import color_correct_affine, color_correct_quadratic
PY

# -----------------------------------------------------------------------------
# 5. pycolmap patch
# -----------------------------------------------------------------------------
echo "== pycolmap =="
# gsplat @ our pin (608e19ad) reads COLMAP models via the OFFICIAL pycolmap
# (pycolmap.Reconstruction). The old rmbrualla fork (SceneManager) served the
# April-era examples and is gone from the lockfile; this guard heals any pod
# that still carries it.
if ! python3 -c 'import pycolmap; assert hasattr(pycolmap, "Reconstruction")' >/dev/null 2>&1; then
    uv pip install --system --break-system-packages "pycolmap==3.11.1" >/dev/null 2>&1 \
        || fail "official pycolmap install failed"
fi
ok "official pycolmap with Reconstruction API"

# -----------------------------------------------------------------------------
# 5b. huggingface_hub (result uploads via hf_upload.py — kept out of the lockfile)
# -----------------------------------------------------------------------------
echo "== huggingface_hub =="
if python3 -c 'import huggingface_hub' >/dev/null 2>&1; then
    ok "huggingface_hub already installed"
else
    uv pip install --system --break-system-packages huggingface_hub >/dev/null 2>&1 \
        || fail "huggingface_hub install failed"
    ok "huggingface_hub installed"
fi

# -----------------------------------------------------------------------------
# 6. asmk (MASt3R retrieval)
# -----------------------------------------------------------------------------
echo "== asmk (MASt3R retrieval) =="
if python3 -c 'import asmk' >/dev/null 2>&1; then
    ok "asmk already installed"
elif ls "$WHEELHOUSE"/asmk-*.whl >/dev/null 2>&1 && \
     uv pip install --system --break-system-packages "$WHEELHOUSE"/asmk-*.whl >/dev/null 2>&1 && \
     python3 -c 'import asmk' >/dev/null 2>&1; then
    ok "asmk installed from wheelhouse"
else
    ASMK_DIR=/tmp/asmk
    if [ ! -d "$ASMK_DIR" ]; then
        git clone --quiet https://github.com/jenicek/asmk "$ASMK_DIR" || fail "asmk clone failed"
    fi
    # asmk ships a Cython-generated hamming.c that references longintrepr.h,
    # which was removed in Python 3.12. Regenerate it with the installed
    # Cython 3.x before building.
    if [ -f "$ASMK_DIR/cython/hamming.pyx" ] && command -v cython >/dev/null 2>&1; then
        (cd "$ASMK_DIR" && cython cython/hamming.pyx -o cython/hamming.c) >/dev/null 2>&1 \
            || warn "cython regenerate failed (may still work if hamming.c is fresh)"
    fi
    uv pip install --system --break-system-packages --no-build-isolation "$ASMK_DIR" \
        >/dev/null 2>&1 || fail "asmk install failed"
    python3 -c 'import asmk' || fail "asmk import still failing after install"
    ok "asmk installed from $ASMK_DIR"
fi

# -----------------------------------------------------------------------------
# 7. install scripts
# -----------------------------------------------------------------------------
echo "== install scripts =="
mkdir -p "$WS/scripts"
copy_if_different() {
    local src="$1" dst="$2"
    if [ "$(readlink -f "$src")" = "$(readlink -f "$dst" 2>/dev/null || echo x)" ]; then
        return 0
    fi
    cp "$src" "$dst"
}
copy_if_different "$SCRIPT_DIR/pipeline.sh"        "$WS/scripts/pipeline.sh"
copy_if_different "$SCRIPT_DIR/colmap_txt2bin.py"  "$WS/scripts/colmap_txt2bin.py"
copy_if_different "$SCRIPT_DIR/setup_pod.sh"       "$WS/scripts/setup_pod.sh"
[ -f "$SCRIPT_DIR/requirements.txt" ]      && copy_if_different "$SCRIPT_DIR/requirements.txt"      "$WS/scripts/requirements.txt"
[ -f "$SCRIPT_DIR/requirements.lock.txt" ] && copy_if_different "$SCRIPT_DIR/requirements.lock.txt" "$WS/scripts/requirements.lock.txt"
chmod +x "$WS/scripts/pipeline.sh" "$WS/scripts/setup_pod.sh"
copy_if_different "$SCRIPT_DIR/run_mast3r_sfm.py"  "$WS/mast3r/run_mast3r_sfm.py"
ok "scripts installed under $WS/scripts and $WS/mast3r"

# -----------------------------------------------------------------------------
# 8. sanity imports
# -----------------------------------------------------------------------------
echo "== python imports =="
export PYTHONPATH="$WS/sitecustom:$WS/mast3r${PYTHONPATH:+:$PYTHONPATH}"
python3 - <<'PY' || fail "mast3r import failed"
from mast3r.model import AsymmetricMASt3R
from mast3r.retrieval.processor import Retriever
from mast3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
print("  mast3r ok")
PY
python3 - <<'PY' || fail "gsplat import failed"
from gsplat.utils import save_ply
from gsplat.color_correct import color_correct_affine
print("  gsplat ok")
PY
python3 - <<'PY' || fail "pycolmap Reconstruction API missing (need official pycolmap, not the rmbrualla fork)"
import pycolmap
assert hasattr(pycolmap, "Reconstruction")
print("  pycolmap Reconstruction ok")
PY
python3 - <<'PY' || fail "fused_ssim import failed"
from fused_ssim import fused_ssim
print("  fused_ssim ok")
PY
python3 - <<'PY' || fail "torch/cuda check failed"
import torch
print(f"  torch {torch.__version__} cuda={torch.cuda.is_available()}")
PY
# Full trainer-import probe: catches any downstream dep drift before a run.
( cd "$WS/gsplat/examples" && python3 - <<'PY' ) || fail "simple_trainer.py import probe failed"
import sys; sys.path.insert(0, ".")
import ast, pathlib
src = pathlib.Path("simple_trainer.py").read_text()
# Just execute the module up to the first function/class definition.
for node in ast.parse(src).body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        break
    exec(compile(ast.Module(body=[node], type_ignores=[]), "simple_trainer.py", "exec"))
print("  simple_trainer imports ok")
PY
ok "mast3r / gsplat / pycolmap / fused_ssim / torch / simple_trainer all import"

# -----------------------------------------------------------------------------
# 9. cleanup
# -----------------------------------------------------------------------------
echo "== cleanup stale outputs =="
removed=0
for p in /tmp/gaucho-*.ply; do
    [ -e "$p" ] || continue
    size=$(du -sh "$p" | cut -f1)
    rm -f "$p" && removed=$((removed+1))
    ok "removed $p ($size)"
done
[ "$removed" -eq 0 ] && ok "no /tmp/gaucho-*.ply to clean"

for d in "$WS/results/triangle_gaucho" "$WS/results/triangle_gaucho_hifi"; do
    if [ -d "$d" ]; then
        size=$(du -sh "$d" | cut -f1)
        rm -rf "$d"
        ok "removed $d ($size)"
    fi
done

# -----------------------------------------------------------------------------
# 10. disk
# -----------------------------------------------------------------------------
echo "== disk =="
df -h "$WS" | awk 'NR==1 || NR==2'
free_g=$(df -BG --output=avail "$WS" | tail -1 | tr -dc '0-9')
if [ "${free_g:-0}" -lt 20 ]; then
    warn "free disk < 20G on $WS — a single 120-frame run needs ~6G; delete more result dirs if tight"
else
    ok "free disk: ${free_g}G"
fi

echo
echo "pod setup complete. next:"
echo "  ssh \$POD '/workspace/scripts/pipeline.sh <video_or_images_dir> <scene_name>'"
echo "  # or, from your laptop:"
echo "  igen splat <input> [--from <stage>] [--pod user@host:port]"
