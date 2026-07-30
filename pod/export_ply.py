#!/usr/bin/env python3
"""Export a gsplat checkpoint to PLY on CPU (pods must not need the GPU for this).
    python3 export_ply.py <ckpt.pt> <out.ply>
"""
import sys, torch
from gsplat.utils import save_ply

ck = torch.load(sys.argv[1], map_location="cpu")["splats"]
save_ply({k: v for k, v in ck.items()}, sys.argv[2])
print(f"EXPORT-OK {sys.argv[2]}")
