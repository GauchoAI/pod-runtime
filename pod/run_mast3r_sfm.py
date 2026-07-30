#!/usr/bin/env python3
"""
Drive MASt3R-SfM (sparse_global_alignment) and export COLMAP text format.
MPS-compatible wrapper for Apple Silicon.
"""
import argparse, glob, os, sys, gc, time, numpy as np, torch
from pathlib import Path

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import load_images
from mast3r.model import AsymmetricMASt3R
from mast3r.retrieval.processor import Retriever
from mast3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment


def rotmat_to_quat(R):
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return qw, qx, qy, qz


def write_colmap_text(out_dir, intrinsics_np, cam2w_np, names, sizes_wh,
                      pts3d_np, colors_np):
    """
    intrinsics_np: (N, 3, 3) at working resolution
    cam2w_np:      (N, 4, 4) camera-to-world (MASt3R convention)
    names:         list[str] of basenames
    sizes_wh:      (N, 2) working-resolution (W, H)
    pts3d_np:      (M, 3) world points
    colors_np:     (M, 3) uint8
    """
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cameras.txt"), "w") as f:
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i, (K, sz) in enumerate(zip(intrinsics_np, sizes_wh)):
            W, H = int(sz[0]), int(sz[1])
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])
            f.write(f"{i+1} PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n")

    with open(os.path.join(out_dir, "images.txt"), "w") as f:
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i, (cam2w, name) in enumerate(zip(cam2w_np, names)):
            # world->camera = inv(cam2w)
            w2c = np.linalg.inv(cam2w)
            R, t = w2c[:3, :3], w2c[:3, 3]
            qw, qx, qy, qz = rotmat_to_quat(R)
            f.write(f"{i+1} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} {i+1} {name}\n\n")

    with open(os.path.join(out_dir, "points3D.txt"), "w") as f:
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for j, (p, c) in enumerate(zip(pts3d_np, colors_np)):
            r, g, b = int(c[0]), int(c[1]), int(c[2])
            f.write(f"{j+1} {p[0]} {p[1]} {p[2]} {r} {g} {b} 0.0\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_dir", required=True, help="scene dir; writes sparse/0/ + points.ply")
    ap.add_argument("--weights", default=str(Path(__file__).with_name("checkpoints") /
                                            "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"))
    ap.add_argument("--retrieval_model", default=str(Path(__file__).with_name("checkpoints") /
                                            "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"))
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--scene_graph", default="retrieval-20-10",
                    help="e.g. swin-5, swin-10, retrieval-Na-k, complete")
    ap.add_argument("--lr1", type=float, default=0.07)
    ap.add_argument("--niter1", type=int, default=300)
    ap.add_argument("--lr2", type=float, default=0.014)
    ap.add_argument("--niter2", type=int, default=300)
    ap.add_argument("--matching_conf_thr", type=float, default=0.0)
    ap.add_argument("--shared_intrinsics", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max_images", type=int, default=0)
    args = ap.parse_args()

    if args.device:
        device = args.device
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Gather images
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(args.images_dir, e))
    paths.sort()
    if args.max_images and args.max_images < len(paths):
        step = len(paths) / args.max_images
        paths = [paths[int(i * step)] for i in range(args.max_images)]
    if not paths:
        print(f"no images in {args.images_dir}")
        sys.exit(1)
    print(f"Found {len(paths)} images")

    # Load MASt3R
    print(f"Loading MASt3R from {args.weights}")
    model = AsymmetricMASt3R.from_pretrained(args.weights).to(device)
    model.eval()

    # Load images (as dust3r dicts)
    imgs = load_images(paths, size=args.image_size, verbose=True)

    # Build scene graph
    sim_mat = None
    if "retrieval" in args.scene_graph:
        print(f"Running retrieval with {args.retrieval_model}")
        retriever = Retriever(args.retrieval_model, backbone=model, device=device)
        with torch.no_grad():
            sim_mat = retriever(paths)
        del retriever
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    pairs = make_pairs(imgs, scene_graph=args.scene_graph, prefilter=None,
                       symmetrize=True, sim_mat=sim_mat)
    print(f"Number of pairs: {len(pairs)}")

    # Run sparse global alignment
    os.makedirs(args.out_dir, exist_ok=True)
    cache_dir = os.path.join(args.out_dir, "cache_mast3r")
    os.makedirs(cache_dir, exist_ok=True)

    t0 = time.time()
    scene = sparse_global_alignment(
        paths, pairs, cache_dir, model,
        lr1=args.lr1, niter1=args.niter1,
        lr2=args.lr2, niter2=args.niter2,
        device=device, opt_depth=True,
        shared_intrinsics=args.shared_intrinsics,
        matching_conf_thr=args.matching_conf_thr,
    )
    print(f"Sparse global alignment: {time.time()-t0:.1f}s")

    # Extract outputs
    intrinsics = torch.stack(list(scene.intrinsics)).detach().cpu().numpy() \
                 if isinstance(scene.intrinsics, (list, tuple)) \
                 else scene.intrinsics.detach().cpu().numpy()
    cam2w = scene.cam2w.detach().cpu().numpy()
    sparse_pts = [p.detach().cpu().numpy() for p in scene.get_sparse_pts3d()]
    sparse_cols = [c if isinstance(c, np.ndarray) else c.detach().cpu().numpy()
                   for c in scene.get_pts3d_colors()]

    pts = np.concatenate(sparse_pts, axis=0) if sparse_pts else np.zeros((0, 3))
    cols = np.concatenate(sparse_cols, axis=0) if sparse_cols else np.zeros((0, 3))
    cols_u8 = np.clip(cols * 255.0, 0, 255).astype(np.uint8)

    # working-resolution sizes (from the loaded img tensors)
    sizes_wh = []
    for im in imgs:
        H, W = im["true_shape"][0] if hasattr(im["true_shape"], "shape") else im["true_shape"]
        sizes_wh.append((int(W), int(H)))
    sizes_wh = np.array(sizes_wh)

    names = [os.path.basename(p) for p in paths]
    sparse_out = os.path.join(args.out_dir, "sparse", "0")
    print(f"Writing COLMAP text to {sparse_out}")
    write_colmap_text(sparse_out, intrinsics, cam2w, names, sizes_wh, pts, cols_u8)

    # PLY for quick inspection
    try:
        import trimesh
        ply_path = os.path.join(args.out_dir, "sparse_points.ply")
        trimesh.PointCloud(pts, colors=cols_u8).export(ply_path)
        print(f"Wrote {ply_path}")
    except Exception as e:
        print(f"ply export failed: {e}")

    # Quick pose stats
    centers = -cam2w[:, :3, :3].transpose(0, 2, 1) @ cam2w[:, :3, 3:4]
    centers = centers.squeeze(-1)
    # Actually for cam2w matrix, center is the translation part directly:
    centers = cam2w[:, :3, 3]
    centroid = centers.mean(0)
    spread = np.linalg.norm(centers - centroid, axis=1)
    print(f"Camera centers: {len(cam2w)}, centroid {centroid}, "
          f"mean dist {spread.mean():.3f}, max dist {spread.max():.3f}")
    print(f"Wrote {len(pts)} sparse points, {len(cam2w)} cameras.")


if __name__ == "__main__":
    main()
