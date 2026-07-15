#!/bin/bash
# Find and fix the truncated step0 video.
#
# Training dies with:
#   IndexError: Invalid frame index=1723 for streamIndex=0; must be less than 1567
# Episode 83 is 1724 frames long, and on the Hub all three of its videos really do
# have 1724 frames -- so a local copy is short. This job checks EVERY video against
# the episode length meta declares (the earlier diag only decoded frame 0, which a
# truncated file passes), then re-downloads whatever is broken.
#
#SBATCH --job-name=verify_step0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-verify-%j.out

set -euo pipefail

export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export USER="${USER:-$(id -un)}"

IMAGE="docker://hyeonseoke/lerobot:v1"
export APPTAINER_CACHEDIR="/tmp/apptainer-$(whoami)"
mkdir -p "$APPTAINER_CACHEDIR"

export APPTAINERENV_HF_TOKEN="$(tr -d '[:space:]' < ~/.hf_token)"
export APPTAINERENV_HUGGING_FACE_HUB_TOKEN="$APPTAINERENV_HF_TOKEN"
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_DS="$HOME/datasets/towel_fold01_step0"

# 1. Drop the node-local staging copy. It is a derivative of $HOME/datasets and
#    is known to be corrupt (two jobs raced on it); it will be rebuilt.
echo "=== removing staged copy ==="
rm -rf "/tmp/lerobot-ds-$(whoami)/towel_fold01_step0"
rm -rf "/tmp/lerobot-ds-$(whoami)/towel_fold01_step0.partial."* 2>/dev/null || true

echo "=== verifying $HOME/datasets/towel_fold01_step0 against the Hub ==="

apptainer exec "$IMAGE" python - <<'PY'
import json, os, shutil
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from torchcodec.decoders import VideoDecoder

REPO = "lhwdev/towel_fold01_step0"
ds = Path(os.environ["DS"])


def episodes(root):
    rows = []
    for f in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        rows += pq.read_table(f).to_pylist()
    rows.sort(key=lambda r: r["episode_index"])
    return rows


def cameras(root):
    info = json.loads((root / "meta" / "info.json").read_text())
    return [k for k in info["features"] if k.startswith("observation.images")]


def check(root):
    """Return the videos whose real frame count is short of the episode length."""
    bad = []
    cams = cameras(root)
    for r in episodes(root):
        for k in cams:
            v = root / "videos" / k / f"chunk-{r[f'videos/{k}/chunk_index']:03d}" / \
                f"file-{r[f'videos/{k}/file_index']:03d}.mp4"
            try:
                n = VideoDecoder(str(v), seek_mode="approximate").metadata.num_frames
            except Exception as e:
                bad.append((r["episode_index"], k, v, -1, r["length"], repr(e)[:60]))
                continue
            if n < r["length"]:
                bad.append((r["episode_index"], k, v, n, r["length"], ""))
    return bad, len(episodes(root)) * len(cams)


bad, total = check(ds)
print(f"checked {total} videos, {len(bad)} short", flush=True)
for ep, k, v, n, need, err in bad[:20]:
    print(f"  ep{ep:3d} {k.split('.')[-1]:10s} {v.name}: {n} frames, need {need} {err}", flush=True)

if bad:
    print("\n=== re-downloading the whole dataset (local copy is corrupt) ===", flush=True)
    shutil.rmtree(ds)
    snapshot_download(repo_id=REPO, repo_type="dataset", revision="main",
                      local_dir=str(ds), max_workers=4)
    bad, total = check(ds)
    print(f"\nafter re-download: checked {total} videos, {len(bad)} short", flush=True)
    for ep, k, v, n, need, err in bad[:20]:
        print(f"  ep{ep:3d} {k.split('.')[-1]:10s} {v.name}: {n} frames, need {need} {err}", flush=True)
    if bad:
        raise SystemExit("STILL BROKEN after a clean download -- the Hub copy is at fault")

# Load it the way training will, and pull the exact frame that was failing.
from lerobot.datasets.lerobot_dataset import LeRobotDataset

d = LeRobotDataset(REPO, root=str(ds))
rows = episodes(ds)
ep83 = next(r for r in rows if r["episode_index"] == 83)
last = ep83["dataset_to_index"] - 1
item = d[last]
print(f"\nOK: episodes={d.num_episodes} frames={d.num_frames}", flush=True)
print(f"OK: read ep83's last frame (global index {last})", flush=True)
PY

echo "=== done ==="
