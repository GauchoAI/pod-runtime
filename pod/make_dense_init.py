#!/usr/bin/env python3
"""Dense ground-truth initialization: backproject engine depth+class+RGB into
surface-seeded colored points, APPENDED to the existing points3D.bin (the
original tracked points stay first, so images.txt tracks remain valid for the
sparse depth loss).

    make_dense_init.py <dataset_dir> <gt_root> <anchor_flight_json> [--cap 4000000]
"""
import json, math, os, struct, sys

import numpy as np
import imageio.v2 as imageio

ds, gt_root, flight_json = sys.argv[1], sys.argv[2], sys.argv[3]
CAP = int(sys.argv[sys.argv.index("--cap") + 1]) if "--cap" in sys.argv else 4_000_000

anchor = json.load(open(flight_json))["anchor"]
M_LAT = 111320.0
m_lon = M_LAT * math.cos(math.radians(anchor["latitude"]))

def enu(p):
    return np.array([(p["longitude"] - anchor["longitude"]) * m_lon,
                     (p["latitude"] - anchor["latitude"]) * M_LAT,
                     p["altitudeAbsoluteMetres"] - anchor["altitudeAbsoluteMetres"]])

def basis(yaw, pitch, roll):
    f = np.array([math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), -math.sin(pitch)])
    r = np.array([f[1], -f[0], 0.0]); r /= np.linalg.norm(r) or 1
    d = np.cross(f, r)
    cr, sr = math.cos(roll), math.sin(roll)
    return r * cr + d * sr, -r * sr + d * cr, f  # right, down, forward

gt_index = json.load(open(f"{gt_root}/gt-index.json"))  # passN -> flight dir
pass_of = {v: k for k, v in gt_index.items()}

pts, cols = [], []
FRAME_STRIDE, PX_STRIDE = 8, 5
for flight, pass_ in ((v, k) for k, v in gt_index.items()):
    meta = json.load(open(f"{gt_root}/{flight}/meta.json"))
    W, H = meta["width"], meta["height"]
    scale = meta["depthEncoding"]["metresPerUnit"]
    fx = 1.5 * H / 2; cx, cy = W / 2, H / 2
    for pose in meta["poses"][::FRAME_STRIDE]:
        n = str(pose["frame"] + 1).zfill(4)
        dpath = f"{gt_root}/{flight}/depth_{n}.png"
        ipath = f"{ds}/images/{pass_}_frame_{n}.jpg"
        if not (os.path.isfile(dpath) and os.path.isfile(ipath)):
            continue
        d16 = imageio.imread(dpath).astype(np.float32)
        cls = imageio.imread(f"{gt_root}/{flight}/class_{n}.png") % 4
        rgb = imageio.imread(ipath)
        C = enu(pose)
        right, down, fwd = basis(pose["yaw"], pose["pitch"], pose["roll"])
        ys, xs = np.mgrid[0:H:PX_STRIDE, 0:W:PX_STRIDE]
        z = d16[ys, xs] * scale
        k = cls[ys, xs]
        ok = (d16[ys, xs] < 65535) & (k >= 1)  # any surface: terrain + vegetation
        xs, ys, z, k = xs[ok], ys[ok], z[ok], k[ok]
        cxr = (xs - cx) / fx; cyr = (ys - cy) / fx
        dirs = (right[None, :] * cxr[:, None] + down[None, :] * cyr[:, None] + fwd[None, :])
        P = C[None, :] + dirs * z[:, None]  # viewDepth along forward: dirs has fwd-comp 1
        pts.append(P.astype(np.float32))
        cols.append(rgb[ys, xs])
P = np.concatenate(pts); Cb = np.concatenate(cols)
if len(P) > CAP:
    sel = np.random.default_rng(19).choice(len(P), CAP, replace=False)
    P, Cb = P[sel], Cb[sel]
print(f"dense init: {len(P):,} surface points from GT")

pts_bin = f"{ds}/sparse/0/points3D.bin"
orig = open(pts_bin, "rb").read()
n_orig = struct.unpack("<Q", orig[:8])[0]
with open(pts_bin, "wb") as f:
    f.write(struct.pack("<Q", n_orig + len(P)))
    f.write(orig[8:])
    pid = 10_000_000  # far above original ids; no tracks
    for i in range(len(P)):
        f.write(struct.pack("<Q3d3Bd", pid + i, *P[i].tolist(), *(int(c) for c in Cb[i][:3]), 0.5))
        f.write(struct.pack("<Q", 0))
print(f"points3D.bin: {n_orig:,} tracked + {len(P):,} GT-seeded = {n_orig + len(P):,}")
print("DENSE-INIT-OK")
