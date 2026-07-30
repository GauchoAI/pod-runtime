#!/usr/bin/env python3
"""Upload splat artifacts to the HF dataset repo that serves as our S3.

Runs ON THE POD so multi-GB PLYs go datacenter → HF directly, never
through the laptop. Token comes from ~/.cache/huggingface/token (the
fleet convention — see alambique / cloud-rendering-experimentation) or
the HF_TOKEN env var.

Usage:
  python3 hf_upload.py <file> [<file>…] \
      [--repo miguelemosreverte/alambique-datasets] \
      [--prefix results/image-generation/splats/<scene>]

Prints the https URL of each uploaded file.
"""
import argparse
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--repo", default=os.environ.get("HF_REPO_ID", "miguelemosreverte/alambique-datasets"))
    ap.add_argument("--repo-type", default=os.environ.get("HF_REPO_TYPE", "dataset"))
    ap.add_argument("--prefix", default=os.environ.get("HF_RESULT_PREFIX", "results/image-generation"))
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("huggingface_hub not installed — run: uv pip install --system --break-system-packages huggingface_hub")

    api = HfApi()  # token resolved from HF_TOKEN or ~/.cache/huggingface/token
    for f in args.files:
        p = Path(f)
        if not p.is_file():
            sys.exit(f"not a file: {p}")
        dest = f"{args.prefix.rstrip('/')}/{p.name}"
        print(f"uploading {p} ({p.stat().st_size / 1e6:.1f} MB) → {args.repo}/{dest}", flush=True)
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=dest,
            repo_id=args.repo,
            repo_type=args.repo_type,
        )
        print(f"  https://huggingface.co/datasets/{args.repo}/blob/main/{dest}", flush=True)


if __name__ == "__main__":
    main()
