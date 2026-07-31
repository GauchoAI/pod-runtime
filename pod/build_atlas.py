#!/usr/bin/env python3
"""The Kazbek Atlas: a perfectly-stitched orthophoto pyramid of our mountain,
built from the engine's own eyes (RGB + exact depth + normals + class).

Every GT pixel is unprojected through its exact depth into ENU world space and
scattered into one global grid — seams are geometrically impossible. Layers:
  albedo  RGB      best-view weighted mean color
  normal  RGB8     world-space normal, n*0.5+0.5
  height  16-bit   surface height above anchor, decimetres (offset +2000 m)
  meta    RGB8     R=class bucket (0 none,1 terrain,2+ vegetation), G=coverage 0..255

    build_atlas.py <dataset_dir> <gt_root> <out_dir>
        [--cell-m 0.5] [--tile 512] [--zooms 4] [--frame-stride 2] [--px-stride 2]

Tiles: <out>/<layer>/<z>/<x>/<y>.png, z0 = base resolution, z+1 = half.
Manifest: <out>/atlas.json (bounds ENU metres, anchor geodetic, encodings).
"""
import json, math, os, sys
import numpy as np
import imageio.v2 as imageio

ds, gt_root, out = sys.argv[1], sys.argv[2], sys.argv[3]
def arg(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default
CELL = arg("--cell-m", 0.5)
TILE = arg("--tile", 512)
ZOOMS = arg("--zooms", 4)
FSTRIDE = arg("--frame-stride", 2)
PSTRIDE = arg("--px-stride", 2)

# ---- dataset cameras + poses (COLMAP text; world = raw ENU metres) ----------
cams = {}
for line in open(f"{ds}/sparse/0/cameras.txt"):
    if line.startswith("#"): continue
    t = line.split()
    if len(t) < 8: continue
    cid, model, W, H = int(t[0]), t[1], int(t[2]), int(t[3])
    p = list(map(float, t[4:]))
    fx, fy, cx, cy = (p[0], p[0], p[1], p[2]) if model == "SIMPLE_PINHOLE" else (p[0], p[1], p[2], p[3])
    cams[cid] = (W, H, fx, fy, cx, cy)

imgs = []  # (name, R(3x3), t(3), cam_id)
lines = [l for l in open(f"{ds}/sparse/0/images.txt") if not l.startswith("#")]
for l in lines:
    t = l.strip().split()
    if len(t) >= 10 and t[9].endswith((".jpg", ".png")):
        qw, qx, qy, qz = map(float, t[1:5])
        tx, ty, tz = map(float, t[5:8])
        R = np.array([
            [1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
            [2*(qx*qy+qw*qz),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
            [2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx),   1-2*(qx*qx+qy*qy)],
        ])
        imgs.append((t[9], R, np.array([tx, ty, tz]), int(t[8])))

gt_index = json.load(open(f"{gt_root}/gt-index.json"))
meta0 = json.load(open(f"{gt_root}/{next(iter(gt_index.values()))}/meta.json"))
print(f"atlas: {len(imgs)} posed frames, {len(gt_index)} passes, encodings "
      f"d={meta0.get('depthEncoding')} n={meta0.get('normalEncoding')}", flush=True)

# ---- pass 1: bounds from a sparse sample of unprojections -------------------
def frame_paths(name):
    p, f = name.split("_frame_")
    n = f.split(".")[0]
    fl = gt_index.get(p)
    if fl is None: return None
    d = f"{gt_root}/{fl}"
    paths = (f"{ds}/images/{name}", f"{d}/depth_{n}.png", f"{d}/normal_{n}.png", f"{d}/class_{n}.png")
    return paths if all(os.path.isfile(x) for x in paths) else None

def unproject(name, R, t, cid, stride):
    paths = frame_paths(name)
    if paths is None: return None
    W, H, fx, fy, cx, cy = cams[cid]
    d16 = imageio.imread(paths[1]).astype(np.float32)
    z = d16 * 0.032
    valid = d16 < 65535
    vs, us = np.mgrid[0:H:stride, 0:W:stride]
    z = z[::stride, ::stride]; valid = valid[::stride, ::stride]
    x = (us - cx) / fx * z
    y = (vs - cy) / fy * z
    P_cam = np.stack([x, y, z], -1)[valid]                      # camera space
    Xw = (P_cam - t) @ R                                        # R^T (P - t): world ENU
    return Xw, valid, z[valid], paths

sample_pts = []
for name, R, t, cid in imgs[::25]:
    r = unproject(name, R, t, cid, 16)
    if r is not None: sample_pts.append(r[0])
allp = np.concatenate(sample_pts)
lo = np.percentile(allp, 0.5, axis=0) - 5
hi = np.percentile(allp, 99.5, axis=0) + 5
# ENU: axes 0=east 1=north 2=up
E0, N0 = lo[0], lo[1]
GW = int(math.ceil((hi[0] - E0) / CELL))
GH = int(math.ceil((hi[1] - N0) / CELL))
GW += (-GW) % TILE; GH += (-GH) % TILE                          # pad to tile grid
print(f"grid {GW}x{GH} cells at {CELL} m ({GW*CELL:.0f} x {GH*CELL:.0f} m)", flush=True)

# ---- pass 2: weighted scatter ----------------------------------------------
NC = GW * GH
acc_rgb = np.zeros((NC, 3), np.float64)
acc_n   = np.zeros((NC, 3), np.float64)
acc_h   = np.zeros(NC, np.float64)
acc_w   = np.zeros(NC, np.float64)
cls_best = np.zeros(NC, np.uint8)
cls_w    = np.zeros(NC, np.float32)

done = 0
for name, R, t, cid in imgs[::FSTRIDE]:
    r = unproject(name, R, t, cid, PSTRIDE)
    if r is None: continue
    Xw, valid, zs, paths = r
    rgb = imageio.imread(paths[0])[::PSTRIDE, ::PSTRIDE][valid].astype(np.float64)
    nrm = imageio.imread(paths[2])[::PSTRIDE, ::PSTRIDE][valid].astype(np.float64) / 255.0 * 2 - 1
    cls = imageio.imread(paths[3])[::PSTRIDE, ::PSTRIDE][valid]
    gx = ((Xw[:, 0] - E0) / CELL).astype(np.int64)
    gy = ((Xw[:, 1] - N0) / CELL).astype(np.int64)
    inb = (gx >= 0) & (gx < GW) & (gy >= 0) & (gy < GH)
    if not inb.any(): continue
    gx, gy, Xw, rgb, nrm, cls, zs = gx[inb], gy[inb], Xw[inb], rgb[inb], nrm[inb], cls[inb], zs[inb]
    cell = gy * GW + gx
    # weight: nearer view = finer texels; face-on = less smear (up-normal proxy)
    w = 1.0 / np.maximum(zs, 1.0) ** 2 * np.maximum(np.abs(nrm[:, 2]), 0.15)
    np.add.at(acc_w, cell, w)
    for c in range(3):
        np.add.at(acc_rgb[:, c], cell, rgb[:, c] * w)
        np.add.at(acc_n[:, c], cell, nrm[:, c] * w)
    np.add.at(acc_h, cell, Xw[:, 2] * w)
    stronger = w.astype(np.float32) > cls_w[cell]
    cls_best[cell[stronger]] = (cls[stronger] % 4).astype(np.uint8)
    cls_w[cell[stronger]] = w[stronger].astype(np.float32)
    done += 1
    if done % 100 == 0: print(f"  {done} frames scattered", flush=True)
print(f"scatter complete: {done} frames, {int((acc_w > 0).sum()):,}/{NC:,} cells observed", flush=True)

seen = acc_w > 0
alb = np.zeros((NC, 3), np.float32); alb[seen] = acc_rgb[seen] / acc_w[seen, None]
nn  = np.zeros((NC, 3), np.float32); nn[seen] = acc_n[seen] / acc_w[seen, None]
ln = np.linalg.norm(nn, axis=1); nz = ln > 1e-6
nn[nz] /= ln[nz, None]
hgt = np.zeros(NC, np.float32); hgt[seen] = acc_h[seen] / acc_w[seen]
cov = np.clip(np.log1p(acc_w / np.median(acc_w[seen])) * 85, 0, 255) if seen.any() else np.zeros(NC)

def grid(a, ch=None):
    return a.reshape(GH, GW) if ch is None else a.reshape(GH, GW, ch)

layers = {
    "albedo": grid(alb, 3).astype(np.float32),
    "normal": grid(nn, 3).astype(np.float32),
    "height": grid(hgt),
    "meta":   np.stack([grid(cls_best.astype(np.float32)), grid(cov.astype(np.float32)),
                        np.zeros((GH, GW), np.float32)], -1),
    "_w":     grid(acc_w.astype(np.float32)),
}

# ---- pyramid (weight-aware box filter: coarse is MADE good) -----------------
def enc(layer, a):
    if layer == "albedo": return np.clip(a, 0, 255).astype(np.uint8)
    if layer == "normal": return np.clip((a * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    if layer == "height": return np.clip((a + 2000) * 10, 0, 65535).astype(np.uint16)
    if layer == "meta":   return np.clip(a, 0, 255).astype(np.uint8)

os.makedirs(out, exist_ok=True)
cur = layers
for z in range(ZOOMS):
    w = cur["_w"]
    gh, gw = w.shape
    for layer in ("albedo", "normal", "height", "meta"):
        a = cur[layer]
        for ty in range(gh // TILE):
            for tx in range(gw // TILE):
                td = f"{out}/{layer}/{z}/{tx}"
                os.makedirs(td, exist_ok=True)
                patch = a[ty*TILE:(ty+1)*TILE, tx*TILE:(tx+1)*TILE]
                imageio.imwrite(f"{td}/{ty}.png", enc(layer, patch))
    print(f"z{z}: {gw//TILE}x{gh//TILE} tiles per layer", flush=True)
    if z == ZOOMS - 1 or gh // 2 < TILE or gw // 2 < TILE: break
    w4 = w.reshape(gh//2, 2, gw//2, 2)
    ws = w4.sum((1, 3))
    nxt = {"_w": ws.astype(np.float32)}
    for layer in ("albedo", "normal", "height"):
        a = cur[layer]
        if a.ndim == 2:
            s = (a * w).reshape(gh//2, 2, gw//2, 2).sum((1, 3))
            nxt[layer] = np.where(ws > 0, s / np.maximum(ws, 1e-9), 0).astype(np.float32)
        else:
            s = (a * w[..., None]).reshape(gh//2, 2, gw//2, 2, a.shape[2]).sum((1, 3))
            nxt[layer] = np.where(ws[..., None] > 0, s / np.maximum(ws[..., None], 1e-9), 0).astype(np.float32)
    m4 = cur["meta"].reshape(gh//2, 2, gw//2, 2, 3)
    nxt["meta"] = np.concatenate([
        m4[..., 0].max((1, 3))[..., None],            # class: strongest claim survives
        m4[..., 1].mean((1, 3))[..., None],           # coverage: average
        m4[..., 2].mean((1, 3))[..., None]], -1).astype(np.float32)
    cur = nxt

json.dump({
    "schema": "kazbek-atlas-v1",
    "boundsEnuMetres": {"e0": float(E0), "n0": float(N0),
                        "e1": float(E0 + GW * CELL), "n1": float(N0 + GH * CELL)},
    "cellMetres": CELL, "tile": TILE, "zooms": ZOOMS,
    "gridCells": [GW, GH],
    "layers": {
        "albedo": "RGB8 weighted-mean color",
        "normal": "RGB8 world ENU normal, n*0.5+0.5",
        "height": "u16 (h_m + 2000) * 10 (decimetre precision)",
        "meta":   "R=class (0 none,1 terrain,2+ vegetation) G=coverage 0..255",
    },
    "framesUsed": done, "cellsObserved": int(seen.sum()),
    "source": "engine GT (exact depth/normals/class) via COLMAP poses",
}, open(f"{out}/atlas.json", "w"), indent=1)
print("ATLAS-OK", flush=True)
