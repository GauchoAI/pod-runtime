# pod-runtime

Everything a GPU pod needs to join the Kazbek render/train farm. This repo is
**public by design**: pods `git pull` it with zero credentials.

## Credential doctrine
- **This repo contains no secrets** — ever. CI checks nothing in because nothing secret belongs here.
- **The one secret is the Hugging Face token**, and it is provided **by hand**,
  manually, at provision time only: written to `~/.cache/huggingface/token`
  (or exported as `HF_TOKEN`). It grants the vault: wheelhouse, source bundles,
  checkpoints, world tiles, render cache.
- Private application repos are never cloned on pods. Their code arrives via the
  HF **sources bundle** (`pod-env/igen-splat/sources/`), gated by the same token.

## CI/CD
Push to `main` → every live pod picks it up within 60 s (`sync.sh` loop).
Stopped pods catch up on wake (`onstart.sh`). Install on a fresh pod:

    git clone https://github.com/GauchoAI/pod-runtime /workspace/pod-runtime
    bash /workspace/pod-runtime/onstart.sh          # starts the sync loop
    # place HF token BY HAND, then:
    bash /workspace/pod-runtime/pod/setup_pod.sh    # 5-minute cold start via HF wheelhouse

## Layout
- `pod/` — provisioning, environment snapshot, pipeline, HF upload tools, host ledger
- `sync.sh` — 60 s self-update loop  ·  `onstart.sh` — wake-up hook
