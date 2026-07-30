#!/usr/bin/env python3
# Voxel-grid subsample of a COLMAP points3D.bin.
#
# Why: MASt3R on 300+ frames produces 20-30 M sparse points (dense
# depth projection), way more than a 32 GB GPU can train with gsplat's
# SH backward. We need to drop 50-80% of them.
#
# Random subsampling preserves MASt3R's per-frame density bias — areas
# seen by many frames stay over-represented, background gets thinned.
# Voxel downsampling gives even spatial coverage: bin by grid, keep
# one representative point per cell. That's the right prior when we
# don't have reprojection-error or track-length signals (MASt3R
# leaves both at zero because its points come from dense depth maps,
# not sparse feature tracks).
#
# Strategy:
#   1. Compute bounding box of all points.
#   2. Binary-search a voxel size that yields approximately `--keep`
#      occupied cells.
#   3. For each occupied cell, keep the point closest to the cell
#      centroid (more stable than picking the first one).
#
# Writes in-place; backs up the original as points3D.bin.full if not
# already present so repeated runs with different --keep values all
# work from the same baseline.
#
# Usage:
#   subsample_sfm.py <sparse_dir> --keep 10000000 [--random-tail 500000]

import argparse
import os
import shutil
import struct
import sys

import numpy as np
from pycolmap import SceneManager


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sparse_dir", help="dir containing cameras.bin / images.bin / points3D.bin")
    ap.add_argument("--keep", type=int, required=True, help="target number of points to keep")
    ap.add_argument("--random-tail", type=int, default=0,
                    help="additional random points drawn from the high-error tail")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sparse = args.sparse_dir.rstrip("/")
    pts_bin = f"{sparse}/points3D.bin"
    full_bin = f"{sparse}/points3D.bin.full"

    # Preserve the original once so we can re-subsample later.
    if not os.path.exists(full_bin):
        shutil.copy(pts_bin, full_bin)
        print(f"backed up original -> {full_bin}")
    else:
        # Ensure we operate on the original, not a prior subsample.
        shutil.copy(full_bin, pts_bin)
        print(f"restored {pts_bin} from {full_bin}")

    sm = SceneManager(sparse)
    sm.load_points3D()
    n = len(sm.point3D_ids)
    xyz = sm.points3D.astype(np.float32)
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    extent = hi - lo
    print(f"loaded {n:,} points; bbox {extent[0]:.2f} x {extent[1]:.2f} x {extent[2]:.2f}")

    if args.keep >= n:
        print(f"keep ({args.keep:,}) >= total ({n:,}); nothing to do")
        return

    # Binary-search a voxel size that yields ~keep_top occupied cells.
    # The count is monotonic in voxel size: bigger voxels -> fewer cells.
    keep_top = args.keep - args.random_tail
    def count_cells(vsize):
        idx = np.floor((xyz - lo) / vsize).astype(np.int64)
        # Hash ijk to a single int64 for fast unique()
        mx = idx.max(axis=0) + 1
        key = idx[:, 0] * (mx[1] * mx[2]) + idx[:, 1] * mx[2] + idx[:, 2]
        return np.unique(key).size, idx, key

    # Start with a very small v_lo (close to 0, gives ~n cells) and
    # v_hi = max extent (gives ~1 cell). MASt3R dense points are surfaces
    # not volumes, so the density heuristic undershoots badly — just
    # brackets the full range and lets binary search do the work.
    v_lo = float(extent.max()) * 1e-5
    v_hi = float(extent.max())
    chosen_vsize = None
    chosen_idx = None
    chosen_key = None
    for _ in range(40):
        v_mid = (v_lo + v_hi) / 2
        c, idx_arr, key_arr = count_cells(v_mid)
        if c > keep_top * 1.02:
            v_lo = v_mid  # too many cells -> need bigger voxels
        elif c < keep_top * 0.98:
            v_hi = v_mid  # too few cells -> need smaller voxels
        else:
            chosen_vsize, chosen_idx, chosen_key = v_mid, idx_arr, key_arr
            break
    if chosen_vsize is None:
        # Converged to the window. Use whichever endpoint is closer to target.
        c_lo, idx_lo, key_lo = count_cells(v_lo)
        c_hi, idx_hi, key_hi = count_cells(v_hi)
        if abs(c_lo - keep_top) < abs(c_hi - keep_top):
            chosen_vsize, chosen_idx, chosen_key = v_lo, idx_lo, key_lo
        else:
            chosen_vsize, chosen_idx, chosen_key = v_hi, idx_hi, key_hi

    # For each unique cell, pick the point closest to the cell centroid.
    sort = np.argsort(chosen_key, kind="stable")
    chosen_key_sorted = chosen_key[sort]
    # Cell boundaries (indices into sort where key changes).
    starts = np.concatenate([[0], np.where(np.diff(chosen_key_sorted) != 0)[0] + 1])
    cell_centers = lo + (chosen_idx[sort] + 0.5) * chosen_vsize
    xyz_sorted = xyz[sort]
    dist2 = np.sum((xyz_sorted - cell_centers) ** 2, axis=1)
    # numpy segment argmin via groupby: we take the index of min dist within
    # each run of equal keys.
    keep_rel = np.empty(len(starts), dtype=np.int64)
    ends = np.concatenate([starts[1:], [len(chosen_key_sorted)]])
    for i, (s, e) in enumerate(zip(starts, ends)):
        keep_rel[i] = s + int(np.argmin(dist2[s:e]))
    keep_idx = sort[keep_rel]

    print(f"voxel size {chosen_vsize:.4f}: kept {len(keep_idx):,} cells")

    if args.random_tail > 0:
        # Draw from points not already selected.
        mask = np.ones(n, dtype=bool)
        mask[keep_idx] = False
        remaining = np.flatnonzero(mask)
        rng = np.random.default_rng(args.seed)
        if args.random_tail >= remaining.size:
            tail_pick = remaining
        else:
            tail_pick = rng.choice(remaining, args.random_tail, replace=False)
        keep_idx = np.concatenate([keep_idx, tail_pick])

    keep_idx.sort()
    print(f"keeping {len(keep_idx):,} points "
          f"(voxel + {args.random_tail:,} random-tail)")

    ids = sm.point3D_ids[keep_idx]
    xyz = sm.points3D[keep_idx]
    rgb = sm.point3D_colors[keep_idx]
    err = sm.point3D_errors[keep_idx]

    with open(pts_bin, "wb") as f:
        f.write(struct.pack("<Q", len(keep_idx)))
        for i in range(len(keep_idx)):
            pid = int(ids[i])
            f.write(struct.pack("<Q3d3Bd",
                                pid,
                                *xyz[i].tolist(),
                                *[int(c) for c in rgb[i]],
                                float(err[i])))
            track = sm.point3D_id_to_images.get(pid, [])
            f.write(struct.pack("<Q", len(track)))
            for (img_id, p2d_idx) in track:
                f.write(struct.pack("<II", int(img_id), int(p2d_idx)))

    size = os.path.getsize(pts_bin)
    print(f"wrote {pts_bin} ({size:,} bytes)")


if __name__ == "__main__":
    main()
