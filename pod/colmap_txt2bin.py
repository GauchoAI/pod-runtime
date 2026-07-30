"""Convert COLMAP sparse model from text → binary (cameras/images/points3D)."""
import os, sys, struct, numpy as np

def write_cameras_bin(cameras, out_path):
    # cameras: list of (cam_id, model_name, width, height, params_list)
    model_name_to_id = {"SIMPLE_PINHOLE": 0, "PINHOLE": 1, "SIMPLE_RADIAL": 2,
                        "RADIAL": 3, "OPENCV": 4, "OPENCV_FISHEYE": 5}
    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cam_id, model_name, w, h, params in cameras:
            mid = model_name_to_id[model_name]
            f.write(struct.pack("<iiQQ", cam_id, mid, w, h))
            f.write(struct.pack(f"<{len(params)}d", *params))

def write_images_bin(images, out_path):
    # images: list of (img_id, qw, qx, qy, qz, tx, ty, tz, cam_id, name, points2d)
    # points2d: empty list (we don't track 2D observations)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for img_id, qw, qx, qy, qz, tx, ty, tz, cam_id, name, pts2d in images:
            f.write(struct.pack("<idddddddi", img_id, qw, qx, qy, qz, tx, ty, tz, cam_id))
            f.write(name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", len(pts2d)))
            for x, y, pid in pts2d:
                f.write(struct.pack("<ddq", x, y, pid))

def write_points3D_bin(points, out_path):
    # points: list of (pid, x, y, z, r, g, b, error, track)
    # track: list of (image_id, point2d_idx)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for pid, x, y, z, r, g, b, err, track in points:
            f.write(struct.pack("<QdddBBBd", pid, x, y, z, int(r), int(g), int(b), err))
            f.write(struct.pack("<Q", len(track)))
            for iid, pt2d_idx in track:
                f.write(struct.pack("<ii", iid, pt2d_idx))

def main():
    in_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else in_dir
    os.makedirs(out_dir, exist_ok=True)

    # cameras.txt
    cameras = []
    with open(os.path.join(in_dir, "cameras.txt")) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            w, h = int(parts[2]), int(parts[3])
            params = [float(x) for x in parts[4:]]
            cameras.append((cam_id, model, w, h, params))
    write_cameras_bin(cameras, os.path.join(out_dir, "cameras.bin"))
    print(f"wrote cameras.bin ({len(cameras)} cams)")

    # images.txt (alternating pose line + points2D line)
    images = []
    with open(os.path.join(in_dir, "images.txt")) as f:
        lines = [l.rstrip() for l in f if not l.startswith("#")]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        parts = line.split()
        img_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        cam_id = int(parts[8])
        name = parts[9]
        # Next line is points2D (may be empty)
        pts2d = []
        if i + 1 < len(lines):
            parts2 = lines[i+1].strip().split()
            for j in range(0, len(parts2), 3):
                pts2d.append((float(parts2[j]), float(parts2[j+1]), int(parts2[j+2])))
        images.append((img_id, qw, qx, qy, qz, tx, ty, tz, cam_id, name, pts2d))
        i += 2
    write_images_bin(images, os.path.join(out_dir, "images.bin"))
    print(f"wrote images.bin ({len(images)} imgs)")

    # points3D.txt
    points = []
    path = os.path.join(in_dir, "points3D.txt")
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split()
            pid = int(parts[0])
            x, y, z = map(float, parts[1:4])
            r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
            err = float(parts[7])
            # Remaining: track (image_id, point2d_idx) pairs
            track = []
            rest = parts[8:]
            for j in range(0, len(rest), 2):
                track.append((int(rest[j]), int(rest[j+1])))
            points.append((pid, x, y, z, r, g, b, err, track))
    write_points3D_bin(points, os.path.join(out_dir, "points3D.bin"))
    print(f"wrote points3D.bin ({len(points)} pts)")

if __name__ == "__main__":
    main()
