#!/bin/bash
# Delete the orphan data/video files from the local step0 copy.
#
# ROOT CAUSE of every "IndexError: Invalid frame index" / "Errno 5" failure:
#
#   data/chunk-000/file-022.parquet is an orphan -- 3 rows, episode_index=24,
#   index 15963..15965 -- that NO episode in meta/ points at. A 3-frame scrap
#   left over from recording.
#
#   lerobot globs every parquet under data/ and concatenates them
#   (io_utils.py:75), so those 3 rows land at position ~15963 and shift every
#   later row by +3: the dataset has 131366 rows while meta declares 131363.
#   The reader then looks rows up by treating the absolute `index` as a
#   POSITION (dataset_reader.py:244), so from that point on it reads the wrong
#   row -- which is how ep83's timestamp (57.43s) ended up being queried against
#   ep84's video (file-085.mp4, only 52.2s long).
#
# Deleting the orphan restores position == index. Videos have the same orphan.
#
#SBATCH --job-name=fix_orphans
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-orphans-%j.out

set -euo pipefail

export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export USER="${USER:-$(id -un)}"

IMAGE="docker://hyeonseoke/lerobot:v1"
export APPTAINER_CACHEDIR="/tmp/apptainer-$(whoami)"
mkdir -p "$APPTAINER_CACHEDIR"

export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_DS="$HOME/datasets/towel_fold01_step0"

echo "=== dropping the node-local staging copy (will be rebuilt from HOME) ==="
rm -rf "/tmp/lerobot-ds-$(whoami)/towel_fold01_step0"
rm -rf "/tmp/lerobot-ds-$(whoami)/towel_fold01_step0.partial."* 2>/dev/null || true

apptainer exec "$IMAGE" python - <<'PY'
import json, os
from pathlib import Path

import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO = "lhwdev/towel_fold01_step0"
ds = Path(os.environ["DS"])

rows = []
for f in sorted((ds / "meta" / "episodes").rglob("*.parquet")):
    rows += pq.read_table(f).to_pylist()
info = json.loads((ds / "meta" / "info.json").read_text())
cams = [k for k in info["features"] if k.startswith("observation.images")]

# Every (chunk, file) meta actually points at. Anything else under data/ or
# videos/ is dead weight that lerobot will nonetheless concatenate.
keep_data = {(r["data/chunk_index"], r["data/file_index"]) for r in rows}
keep_vid = {c: {(r[f"videos/{c}/chunk_index"], r[f"videos/{c}/file_index"]) for r in rows} for c in cams}

removed = 0
for f in sorted((ds / "data").rglob("*.parquet")):
    ci = int(f.parent.name.split("-")[1])
    fi = int(f.stem.split("-")[1])
    if (ci, fi) not in keep_data:
        n = pq.read_table(f).num_rows
        eps = sorted(set(pq.read_table(f).to_pydict()["episode_index"]))
        print(f"[orphan] {f.relative_to(ds)}  rows={n} episode_index={eps} -> deleting", flush=True)
        f.unlink()
        removed += 1

for c in cams:
    for f in sorted((ds / "videos" / c).rglob("*.mp4")):
        ci = int(f.parent.name.split("-")[1])
        fi = int(f.stem.split("-")[1])
        if (ci, fi) not in keep_vid[c]:
            print(f"[orphan] {f.relative_to(ds)} -> deleting", flush=True)
            f.unlink()
            removed += 1

print(f"\nremoved {removed} orphan files\n", flush=True)

# Verify: row count must now equal what meta declares, and position == index.
d = LeRobotDataset(REPO, root=str(ds))
hf = d.hf_dataset if hasattr(d, "hf_dataset") else d._reader.hf_dataset
print(f"hf_dataset rows = {len(hf)}   meta total_frames = {d.num_frames}", flush=True)
assert len(hf) == info["total_frames"], "row count still disagrees with meta"

for i in (0, 15963, 113750, len(hf) - 1):
    r = hf[i]
    assert int(r["index"]) == i, f"position {i} holds index {int(r['index'])} -- still shifted"
print("position == index at every probe", flush=True)

# Pull the exact item that was crashing training.
item = d[113750]
print(f"read frame 113750 OK: episode={int(item['episode_index'])} "
      f"cams={sorted(k for k in item if k.startswith('observation.images'))}", flush=True)
print("\nALL OK -- resubmit train.sbatch", flush=True)
PY
