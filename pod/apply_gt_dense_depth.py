#!/usr/bin/env python3
"""Inject dense ground-truth depth supervision into the pinned gsplat trainer.
Idempotent string-surgery against the known pin (608e19ad). Enable by setting
GT_DEPTH_DIR=/workspace/gt at train time; without the env the trainer is
byte-for-byte stock behavior.

Loss design (per Miguel's doctrine):
  terrain (class 1): full dense L1 on expected depth — the surface is exact
  trees/foliage (2/3): ZERO geometric loss — photometrics keep them fat
  sky (0): sentinel 65535 -> excluded here (sparse tracks + opacity reg
           already starve sky floaters; alpha supervision is a later lever)
"""
import sys

GSPLAT = sys.argv[1] if len(sys.argv) > 1 else "/workspace/gsplat"

# ---- datasets/colmap.py: load GT maps beside each image --------------------
p = f"{GSPLAT}/examples/datasets/colmap.py"
s = open(p).read()
MARK = "gt_dense_depth"
if MARK not in s:
    old = """            data["points"] = torch.from_numpy(points).float()
            data["depths"] = torch.from_numpy(depths).float()

        return data"""
    new = """            data["points"] = torch.from_numpy(points).float()
            data["depths"] = torch.from_numpy(depths).float()

        # gt_dense_depth: engine-rendered supervision (GT_DEPTH_DIR set by job)
        import os as _os
        _gt_root = _os.environ.get("GT_DEPTH_DIR")
        if _gt_root:
            import json as _json
            if not hasattr(self, "_gt_index"):
                self._gt_index = _json.load(open(f"{_gt_root}/gt-index.json"))
            _name = self.parser.image_names[index]  # passN_frame_XXXX.jpg
            _pass, _frame = _name.split("_frame_")
            _flight = self._gt_index.get(_pass)
            if _flight is not None:
                _n = _frame.split(".")[0]
                _dp = f"{_gt_root}/{_flight}/depth_{_n}.png"
                _cp = f"{_gt_root}/{_flight}/class_{_n}.png"
                if _os.path.isfile(_dp) and _os.path.isfile(_cp):
                    _d16 = imageio.imread(_dp).astype(np.float32)
                    _cls = imageio.imread(_cp)
                    # metres -> the trainer's normalized world units
                    _scale = float(np.linalg.norm(self.parser.transform[0, :3])) if hasattr(self.parser, "transform") else 1.0
                    _gt = _d16 * 0.032 * _scale
                    _gt[_d16 >= 65535] = np.inf          # sky sentinel
                    data["gt_depth"] = torch.from_numpy(_gt).float()
                    data["gt_weight"] = torch.from_numpy((_cls == 1).astype(np.float32))

        return data"""
    assert old in s, "colmap.py anchor not found — pin drifted"
    s = s.replace(old, new, 1)
    open(p, "w").write(s)
    print("colmap.py: GT loader injected")
else:
    print("colmap.py: already injected")

# ---- simple_trainer.py: dense L1 on rendered expected depth ----------------
p = f"{GSPLAT}/examples/simple_trainer.py"
s = open(p).read()
if MARK not in s:
    old = """                loss += depthloss * cfg.depth_lambda"""
    new = """                loss += depthloss * cfg.depth_lambda
            # gt_dense_depth: engine-exact per-pixel supervision, class-weighted
            if cfg.depth_loss and "gt_depth" in data:
                _gtd = data["gt_depth"].to(device)   # [1, H, W] normalized units
                _gtw = data["gt_weight"].to(device)  # [1, H, W] 1=terrain
                _rd = renders[..., 3]                # [1, H, W] expected depth
                _valid = (_gtw > 0) & torch.isfinite(_gtd)
                if _valid.any():
                    gt_dense_loss = (_rd - _gtd).abs()[_valid].mean() / self.scene_scale
                    loss += gt_dense_loss * cfg.depth_lambda"""
    assert old in s, "simple_trainer.py anchor not found — pin drifted"
    s = s.replace(old, new, 1)
    open(p, "w").write(s)
    print("simple_trainer.py: dense depth loss injected")
else:
    print("simple_trainer.py: already injected")
print("APPLY-GT-DENSE-DEPTH-OK")
