#!/bin/bash
# Reproduce the failing item against the LOCAL copies and compare them to the Hub.
#
# Everything on the Hub is self-consistent (meta <-> data <-> videos all agree),
# and every code path in lerobot resolves the video from the same episode row.
# Yet training queried ep83's timestamp (57.4333s) against ep84's video
# (file-085.mp4, which is 52.2s long). That can only happen if a LOCAL file
# disagrees with the Hub. This job finds which one.
#
#SBATCH --job-name=diag2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-diag2-%j.out

set -euo pipefail

export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export USER="${USER:-$(id -un)}"

IMAGE="docker://hyeonseoke/lerobot:v1"
export APPTAINER_CACHEDIR="/tmp/apptainer-$(whoami)"
mkdir -p "$APPTAINER_CACHEDIR"

export APPTAINERENV_HF_TOKEN="$(tr -d '[:space:]' < ~/.hf_token)"
export APPTAINERENV_HUGGING_FACE_HUB_TOKEN="$APPTAINERENV_HF_TOKEN"
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_HOMEDS="$HOME/datasets/towel_fold01_step0"
export APPTAINERENV_TMPDS="/tmp/lerobot-ds-$(whoami)/towel_fold01_step0"

apptainer exec "$IMAGE" python - <<'PY'
import os
from pathlib import Path

import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO = "lhwdev/towel_fold01_step0"
CAMS = ["observation.images.top", "observation.images.left_cam", "observation.images.right_cam"]


def meta_rows(root):
    rows = []
    for f in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        rows += pq.read_table(f).to_pylist()
    return rows  # deliberately in on-disk order -- lerobot indexes this POSITIONALLY


for label, root in [("HOME", Path(os.environ["HOMEDS"])), ("TMP", Path(os.environ["TMPDS"]))]:
    if not (root / "meta" / "info.json").exists():
        print(f"### {label}: {root} MISSING\n", flush=True)
        continue
    print(f"### {label}: {root}", flush=True)
    rows = meta_rows(root)
    print(f"  meta rows: {len(rows)}", flush=True)

    # lerobot does `self._meta.episodes[ep_idx]` -- position, not episode_index.
    misordered = [(p, r["episode_index"]) for p, r in enumerate(rows) if p != r["episode_index"]]
    print(f"  position != episode_index: {len(misordered)}", flush=True)
    for p, e in misordered[:5]:
        print(f"    position {p} holds episode {e}", flush=True)

    for pos in (83, 84):
        r = rows[pos]
        vids = " ".join(
            f"{c.split('.')[-1]}=file-{r[f'videos/{c}/file_index']:03d}" for c in CAMS
        )
        print(f"  position {pos}: ep={r['episode_index']} len={r['length']} "
              f"data=file-{r['data/file_index']:03d} {vids}", flush=True)

    # The row training actually sampled: global frame index 113750 (ep83's last).
    ds = LeRobotDataset(REPO, root=str(root))
    hf = ds.hf_dataset if hasattr(ds, "hf_dataset") else ds._reader.hf_dataset
    print(f"  hf_dataset rows: {len(hf)}  (meta says {ds.num_frames})", flush=True)
    for i in (113750, 113751):
        row = hf[i]
        print(f"  hf_dataset[{i}]: episode_index={int(row['episode_index'])} "
              f"index={int(row['index'])} timestamp={float(row['timestamp']):.4f}", flush=True)
    print(flush=True)
PY
