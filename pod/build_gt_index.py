#!/usr/bin/env python3
"""Match dataset passes to GT flight dirs by FIRST-FRAME CAMERA CENTER.
images.txt stores poses in raw ENU metres (the trainer normalizes later), and
GT meta.json stores geodetic poses sharing the base flight's anchor — so the
match is exact geometry, not naming convention.

    build_gt_index.py <sparse0_dir> <gt_root> <anchor_flight_json> -> <gt_root>/gt-index.json
"""
import json, math, os, sys

sparse, gt_root, flight_json = sys.argv[1], sys.argv[2], sys.argv[3]
anchor = json.load(open(flight_json))["anchor"]
M_LAT = 111320.0
m_lon = M_LAT * math.cos(math.radians(anchor["latitude"]))

def enu(p):
    return (
        (p["longitude"] - anchor["longitude"]) * m_lon,
        (p["latitude"] - anchor["latitude"]) * M_LAT,
        p["altitudeAbsoluteMetres"] - anchor["altitudeAbsoluteMetres"],
    )

# first image center per pass from images.txt: C = -R^T t
first = {}
for line in open(f"{sparse}/images.txt"):
    t = line.strip().split()
    if len(t) >= 10 and t[9].endswith((".jpg", ".png")) and "_frame_0001." in t[9]:
        qw, qx, qy, qz, tx, ty, tz = map(float, t[1:8])
        R = [
            [1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
            [2*(qx*qy+qw*qz),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
            [2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx),   1-2*(qx*qx+qy*qy)],
        ]
        C = tuple(-(R[0][i]*tx + R[1][i]*ty + R[2][i]*tz) for i in range(3))
        first[t[9].split("_frame_")[0]] = C

index, worst = {}, 0.0
for d in sorted(os.listdir(gt_root)):
    meta_path = f"{gt_root}/{d}/meta.json"
    if not os.path.isfile(meta_path):
        continue
    p0 = enu(json.load(open(meta_path))["poses"][0])
    best, bd = None, 1e18
    for pass_, C in first.items():
        dist = sum((a-b)**2 for a, b in zip(C, p0)) ** 0.5
        if dist < bd:
            bd, best = dist, pass_
    index[best] = d
    worst = max(worst, bd)
    print(f"{best} -> {d}  ({bd:.3f} m)")
if worst > 2.0:
    sys.exit(f"REFUSING: worst pass match {worst:.2f} m — geometry disagrees")
json.dump(index, open(f"{gt_root}/gt-index.json", "w"), indent=2)
print(f"gt-index.json written ({len(index)} passes, worst {worst*1000:.0f} mm)")
