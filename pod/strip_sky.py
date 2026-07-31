#!/usr/bin/env python3
"""Sky ban enforcement (Leaflet Covenant): delete every splat hovering above
the terrain surface + margin. Models contain terrain only; sky is environment.
Operates on our own save_ply output (known binary layout).

    strip_sky.py <model.ply> <dataset_dir> <normalization.json> [--margin-m 80]
"""
import json, struct, sys
import numpy as np

ply_path, ds, norm_path = sys.argv[1], sys.argv[2], sys.argv[3]
margin_m = float(sys.argv[sys.argv.index("--margin-m") + 1]) if "--margin-m" in sys.argv else 80.0

T = np.array(json.load(open(norm_path))["T"])
scale = float(np.linalg.norm(T[0, :3]))

# terrain points: original tracked entries (pid < 10M) of points3D.bin, in ENU
data = open(f"{ds}/sparse/0/points3D.bin", "rb").read()
n = struct.unpack("<Q", data[:8])[0]
off, pts = 8, []
for _ in range(min(n, 30000)):
    pid, x, y, z = struct.unpack_from("<Qddd", data, off)
    off += 8 + 24 + 3 + 8
    tlen = struct.unpack_from("<Q", data, off)[0]
    off += 8 + tlen * 8
    if pid < 10_000_000:
        pts.append((x, y, z))
P_enu = np.array(pts)
P_n = P_enu @ T[:3, :3].T + T[:3, 3]
span = P_n.max(0) - P_n.min(0)
v = int(np.argmin(span))                       # vertical axis in normalized frame
h = [a for a in range(3) if a != v]
G = 64
lo = P_n.min(0)
gx = np.clip(((P_n[:, h[0]] - lo[h[0]]) / (span[h[0]] or 1) * G).astype(int), 0, G - 1)
gy = np.clip(((P_n[:, h[1]] - lo[h[1]]) / (span[h[1]] or 1) * G).astype(int), 0, G - 1)
ceiling = np.full((G, G), -1e9)
np.maximum.at(ceiling, (gx, gy), P_n[:, v])
floor = np.full((G, G), 1e9)
np.minimum.at(floor, (gx, gy), P_n[:, v])
# evidence mask: cells where terrain was actually observed, dilated 1 cell
evid = ceiling > -1e8
dil = evid.copy()
dil[1:, :] |= evid[:-1, :]; dil[:-1, :] |= evid[1:, :]
dil[:, 1:] |= evid[:, :-1]; dil[:, :-1] |= evid[:, 1:]
# fill dilated-but-unobserved cells from neighbors' band; truly empty cells stay closed
for arr, red, init in ((ceiling, np.maximum, -1e9), (floor, np.minimum, 1e9)):
    src = np.where(evid, arr, init)
    grown = src.copy()
    grown[1:, :] = red(grown[1:, :], src[:-1, :]); grown[:-1, :] = red(grown[:-1, :], src[1:, :])
    grown[:, 1:] = red(grown[:, 1:], src[:, :-1]); grown[:, :-1] = red(grown[:, :-1], src[:, 1:])
    arr[~evid] = grown[~evid]
limit = ceiling + margin_m * scale
floor_lim = floor - margin_m * scale

# our save_ply layout: all-float32 properties, means first
head = b""
f = open(ply_path, "rb")
while b"end_header" not in head:
    head += f.readline()
lines = head.decode(errors="ignore").split("\n")
props = [l.split()[-1] for l in lines if l.startswith("property")]
nv = int([l for l in lines if l.startswith("element vertex")][0].split()[-1])
body = np.frombuffer(f.read(), dtype=np.float32).reshape(nv, len(props))
mx, my = body[:, props.index("x")], body[:, props.index("y")]
mz3 = body[:, [props.index("x"), props.index("y"), props.index("z")]]
sgx = np.clip(((mz3[:, h[0]] - lo[h[0]]) / (span[h[0]] or 1) * G).astype(int), 0, G - 1)
sgy = np.clip(((mz3[:, h[1]] - lo[h[1]]) / (span[h[1]] or 1) * G).astype(int), 0, G - 1)
in_evid = dil[sgx, sgy]
keep = in_evid & (mz3[:, v] <= limit[sgx, sgy]) & (mz3[:, v] >= floor_lim[sgx, sgy])
print(f"envelope strip: {nv:,} -> {int(keep.sum()):,} "
      f"(cut {int((~in_evid).sum()):,} off-footprint, "
      f"{int((in_evid & ~keep).sum()):,} outside height band ±{margin_m:.0f}m)")
out = body[keep]
with open(ply_path, "wb") as g:
    hdr = head.decode(errors="ignore").replace(f"element vertex {nv}", f"element vertex {len(out)}")
    g.write(hdr.encode())
    g.write(out.tobytes())
print("SKY-STRIP-OK")
