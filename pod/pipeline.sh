#!/usr/bin/env bash
# End-to-end pod-side pipeline: video (or image dir) -> PLY.
#
# Stages (each idempotent, each skippable via --from <stage>):
#   1. extract         ffmpeg frames -> scenes/<name>/images_raw/
#   2. subsample       keep N frames -> scenes/<name>/images/
#   3. mast3r          run MASt3R-SfM (CPU) -> scenes/<name>/sparse/0/*.txt
#   4. sanity          validate single-cluster poses
#   5. scale           rewrite cameras.bin with intrinsics*2 if train_res>image_size
#   6. train           gsplat simple_trainer default, 30k iters
#   7. export          save_ply -> /tmp/<name>.ply
#   8. report          print PSNR/SSIM/LPIPS + paths
#
# Usage:
#   pipeline.sh <video_or_images_dir> <scene_name>
#              [--frames 120] [--iters 30000]
#              [--image_size 512] [--train_res 1024x768]
#              [--from extract|subsample|mast3r|sanity|scale|train|export|report]

set -euo pipefail

WS=/workspace
MAST3R_DIR="$WS/mast3r"
GSPLAT_DIR="$WS/gsplat"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -------- args --------
if [ "$#" -lt 2 ]; then
    echo "usage: $(basename "$0") <video_or_images_dir> <scene_name> [--frames N] [--iters N] [--image_size N] [--train_res WxH] [--from stage]" >&2
    exit 2
fi
INPUT="$1"; shift
NAME="$1";  shift

FRAMES=120
ITERS=30000
IMAGE_SIZE=512
TRAIN_RES="1024x768"
FROM_STAGE="extract"
MAST3R_DEVICE="cpu"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --frames)     FRAMES="$2";     shift 2;;
        --iters)      ITERS="$2";      shift 2;;
        --image_size) IMAGE_SIZE="$2"; shift 2;;
        --train_res)  TRAIN_RES="$2";  shift 2;;
        --from)       FROM_STAGE="$2"; shift 2;;
        --device)     MAST3R_DEVICE="$2"; shift 2;;
        *) echo "unknown flag: $1" >&2; exit 2;;
    esac
done

SCENE_DIR="$WS/scenes/$NAME"
IMAGES_RAW="$SCENE_DIR/images_raw"
IMAGES="$SCENE_DIR/images"
SPARSE_DIR="$SCENE_DIR/sparse/0"
RESULT_DIR="$WS/results/gsplat_$NAME"
LOG="$SCENE_DIR/pipeline.log"
PLY_OUT="/tmp/$NAME.ply"

mkdir -p "$SCENE_DIR"
: > "$LOG" || true

log() { printf "[%s] %s\n" "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

# Stage ordering: map name -> index. We run stages whose index >= from_stage.
declare -A STAGE_IDX=( [extract]=1 [subsample]=2 [mast3r]=3 [sanity]=4 [scale]=5 [train]=6 [export]=7 [report]=8 )
if [ -z "${STAGE_IDX[$FROM_STAGE]:-}" ]; then
    echo "unknown --from stage: $FROM_STAGE (valid: ${!STAGE_IDX[*]})" >&2
    exit 2
fi
START_IDX="${STAGE_IDX[$FROM_STAGE]}"
should_run() { [ "${STAGE_IDX[$1]}" -ge "$START_IDX" ]; }

log "scene        : $NAME"
log "input        : $INPUT"
log "frames       : $FRAMES"
log "iters        : $ITERS"
log "image_size   : $IMAGE_SIZE"
log "train_res    : $TRAIN_RES"
log "from stage   : $FROM_STAGE"
log "scene dir    : $SCENE_DIR"
log "ply out      : $PLY_OUT"

# -------- 1. extract --------
if should_run extract; then
    log "== 1. extract =="
    rm -rf "$IMAGES_RAW"
    mkdir -p "$IMAGES_RAW"
    if [ -d "$INPUT" ]; then
        count=0
        for f in "$INPUT"/*.{jpg,jpeg,png,JPG,JPEG,PNG} ; do
            [ -e "$f" ] || continue
            ln -sf "$(readlink -f "$f")" "$IMAGES_RAW/$(basename "$f")"
            count=$((count+1))
        done
        log "linked $count images from $INPUT"
    elif [ -f "$INPUT" ]; then
        log "ffmpeg decode @30fps -> $IMAGES_RAW"
        ffmpeg -nostdin -hide_banner -loglevel error -y -i "$INPUT" \
            -vf fps=30 -qscale:v 2 \
            "$IMAGES_RAW/frame_%04d.jpg" 2>&1 | tee -a "$LOG"
        count=$(ls "$IMAGES_RAW" | wc -l | tr -d ' ')
        log "extracted $count frames"
    else
        log "ERROR: input not a file or directory: $INPUT"
        exit 1
    fi
fi

# -------- 2. subsample --------
if should_run subsample; then
    log "== 2. subsample to $FRAMES =="
    rm -rf "$IMAGES"
    mkdir -p "$IMAGES"
    # shellcheck disable=SC2207
    all=( $(ls "$IMAGES_RAW" | sort) )
    total="${#all[@]}"
    if [ "$total" -eq 0 ]; then
        log "ERROR: no files in $IMAGES_RAW"
        exit 1
    fi
    if [ "$total" -le "$FRAMES" ]; then
        log "only $total frames available (<= target $FRAMES), using all"
        for f in "${all[@]}"; do
            ln -sf "$IMAGES_RAW/$f" "$IMAGES/$f"
        done
    else
        # Evenly spaced subsample. awk avoids bc.
        step=$(awk -v t="$total" -v f="$FRAMES" 'BEGIN { printf "%.6f", t/f }')
        log "step=$step total=$total"
        for i in $(seq 0 $((FRAMES-1))); do
            idx=$(awk -v i="$i" -v s="$step" 'BEGIN { printf "%d", i*s }')
            src="${all[$idx]}"
            ln -sf "$IMAGES_RAW/$src" "$IMAGES/$src"
        done
    fi
    got=$(ls "$IMAGES" | wc -l | tr -d ' ')
    log "subsampled to $got frames"
fi

# -------- 3. mast3r --------
if should_run mast3r; then
    log "== 3. mast3r ($MAST3R_DEVICE) =="
    # Wipe prior sparse outputs for a clean run.
    # Preserve cache_mast3r/ — it contains hashed pair-forward outputs that
    # MASt3R reuses verbatim. Nuking it forces a full rerun of the slow
    # pair-matching phase for no benefit (cache keys are content-hashed, so
    # stale entries just go unread). Resume-friendly.
    rm -rf "$SCENE_DIR/sparse" "$SCENE_DIR/sparse_points.ply"
    export PYTHONPATH="$WS/sitecustom:$MAST3R_DIR${PYTHONPATH:+:$PYTHONPATH}"
    cd "$MAST3R_DIR"
    python3 run_mast3r_sfm.py \
        --device "$MAST3R_DEVICE" \
        --image_size "$IMAGE_SIZE" \
        --scene_graph retrieval-20-10 \
        --images_dir "$IMAGES" \
        --out_dir "$SCENE_DIR" 2>&1 | tee -a "$LOG"
    cd - >/dev/null
    [ -f "$SPARSE_DIR/images.txt" ] || { log "ERROR: mast3r did not produce $SPARSE_DIR/images.txt"; exit 1; }

    log "converting COLMAP text -> binary"
    python3 "$SCRIPT_DIR/colmap_txt2bin.py" "$SPARSE_DIR" "$SPARSE_DIR" 2>&1 | tee -a "$LOG"
fi

# -------- 4. sanity --------
if should_run sanity; then
    log "== 4. sanity =="
    # Single sparse/0 dir (multi-cluster would produce sparse/1, sparse/2, ...)
    extras=$(find "$SCENE_DIR/sparse" -mindepth 1 -maxdepth 1 -type d ! -name "0" | wc -l | tr -d ' ')
    if [ "$extras" -ne 0 ]; then
        log "ERROR: multi-cluster pose graph detected ($extras extra sparse/* dirs)"
        find "$SCENE_DIR/sparse" -mindepth 1 -maxdepth 1 -type d | tee -a "$LOG"
        exit 1
    fi
    log "single sparse/0/ cluster: ok"

    # Preview first few poses.
    log "first 6 poses (IMAGE_ID QW QX QY QZ TX TY TZ CAM_ID NAME):"
    awk 'NR>0 && $0!~/^#/ && NF>=10 { print; n++; if (n==6) exit }' "$SPARSE_DIR/images.txt" | tee -a "$LOG"

    # Count registered images vs input.
    regs=$(awk 'NR>0 && $0!~/^#/ && NF>=10 { n++ } END { print n+0 }' "$SPARSE_DIR/images.txt")
    inputs=$(ls "$IMAGES" | wc -l | tr -d ' ')
    log "registered $regs / $inputs images"
    if [ "$regs" -lt "$((inputs * 9 / 10))" ]; then
        log "WARN: fewer than 90% of images registered"
    fi
fi

# -------- 5. scale intrinsics --------
if should_run scale; then
    log "== 5. scale intrinsics =="
    tw="${TRAIN_RES%x*}"
    th="${TRAIN_RES#*x}"
    # image_size is the long side in MASt3R; short side is derived from aspect ratio.
    # We simply compare the train width to image_size to decide the factor.
    if [ "$tw" -eq "$IMAGE_SIZE" ]; then
        log "train_res width == image_size, no scaling needed"
    else
        factor=$(awk -v a="$tw" -v b="$IMAGE_SIZE" 'BEGIN { printf "%.6f", a/b }')
        log "scaling cameras.bin by factor $factor (image_size $IMAGE_SIZE -> train width $tw)"
        python3 - "$SPARSE_DIR/cameras.bin" "$factor" <<'PY' 2>&1 | tee -a "$LOG"
import os, shutil, struct, sys
src = sys.argv[1]
factor = float(sys.argv[2])
orig = src + ".orig"
if not os.path.exists(orig):
    shutil.copy(src, orig)
with open(orig, "rb") as f:
    data = f.read()
off = 0
(num,) = struct.unpack_from("<Q", data, off); off += 8
out = bytearray(struct.pack("<Q", num))
MODEL_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8}  # SIMPLE_PINHOLE..OPENCV_FISHEYE
for _ in range(num):
    (cam_id,)   = struct.unpack_from("<i", data, off); off += 4
    (model_id,) = struct.unpack_from("<i", data, off); off += 4
    (w,)        = struct.unpack_from("<Q", data, off); off += 8
    (h,)        = struct.unpack_from("<Q", data, off); off += 8
    np_ = MODEL_PARAMS[model_id]
    params = struct.unpack_from(f"<{np_}d", data, off); off += 8 * np_
    scaled = [p * factor for p in params]
    new_w = int(round(w * factor))
    new_h = int(round(h * factor))
    out += struct.pack("<iiQQ", cam_id, model_id, new_w, new_h)
    out += struct.pack(f"<{np_}d", *scaled)
with open(src, "wb") as f:
    f.write(bytes(out))
print(f"rewrote {src} ({num} cams, factor {factor})")
PY
    fi
fi

# -------- 6. train --------
if should_run train; then
    log "== 6. train gsplat ($ITERS steps) =="
    mkdir -p "$RESULT_DIR"
    half_iters=$((ITERS / 2))
    last_step=$((ITERS - 1))
    cd "$GSPLAT_DIR"
    # expandable_segments helps fragmentation when we're close to VRAM limits
    # (e.g. 10 M+ initial splats on a 32 GB card). --packed halves peak VRAM
    # on the SH backward at a small throughput cost; required for anything
    # beyond ~7 M init points on RTX 5090.
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 examples/simple_trainer.py default \
        --data_dir "$SCENE_DIR" \
        --data_factor 1 \
        --result_dir "$RESULT_DIR" \
        --max_steps "$ITERS" \
        --save_steps "$half_iters" "$ITERS" \
        --eval_steps "$ITERS" \
        --packed \
        --disable_viewer 2>&1 | tee -a "$LOG"
    cd - >/dev/null
fi

# -------- 7. export --------
if should_run export; then
    log "== 7. export PLY -> $PLY_OUT =="
    last_step=$((ITERS - 1))
    ckpt="$RESULT_DIR/ckpts/ckpt_${last_step}_rank0.pt"
    if [ ! -f "$ckpt" ]; then
        # fall back to whatever's there
        ckpt=$(ls -t "$RESULT_DIR"/ckpts/ckpt_*_rank0.pt 2>/dev/null | head -1 || true)
    fi
    [ -n "$ckpt" ] && [ -f "$ckpt" ] || { log "ERROR: no checkpoint found in $RESULT_DIR/ckpts"; exit 1; }
    log "using ckpt: $ckpt"
    python3 - "$ckpt" "$PLY_OUT" <<'PY' 2>&1 | tee -a "$LOG"
import sys, torch
from gsplat.utils import save_ply
ckpt_path, ply_path = sys.argv[1], sys.argv[2]
ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
splats = ckpt["splats"]
save_ply(splats, ply_path)
print(f"wrote {ply_path}")
PY
    ls -lh "$PLY_OUT" | tee -a "$LOG"
fi

# -------- 8. report --------
if should_run report; then
    log "== 8. report =="
    last_step=$((ITERS - 1))
    stats="$RESULT_DIR/stats/val_step${last_step}.json"
    if [ -f "$stats" ]; then
        log "val stats ($stats):"
        python3 - "$stats" <<'PY' 2>&1 | tee -a "$LOG"
import json, sys
s = json.load(open(sys.argv[1]))
keys = ["psnr", "ssim", "lpips", "num_GS"]
for k in keys:
    if k in s:
        print(f"  {k}: {s[k]}")
PY
    else
        log "WARN: $stats missing; skipping metrics"
    fi
    log "PLY: $PLY_OUT"
    log "Log: $LOG"
    log "done."
fi
