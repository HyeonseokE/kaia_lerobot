#!/bin/bash
# Diagnose the step0 "OSError: [Errno 5] Input/output error" from the video
# decoder. EIO on a local read means the bytes on disk are bad, not that the
# dataset content is wrong -- most likely a truncated download (home filled up)
# or a partially-written file.
#
# Prints disk/quota usage, then opens EVERY video file in the step0 dataset and
# reports the ones that fail.
#
#SBATCH --job-name=diag
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-diag-%j.out

set -euo pipefail

export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export USER="${USER:-$(id -un)}"

IMAGE="docker://hyeonseoke/lerobot:v1"
export APPTAINER_CACHEDIR="/tmp/apptainer-$(whoami)"
mkdir -p "$APPTAINER_CACHEDIR"

echo "############ 1. disk space / quota ############"
df -h "$HOME" || true
echo
quota -s 2>/dev/null || echo "(quota command unavailable)"
echo

echo "############ 2. what is using the home dir ############"
du -sh "$HOME"/* 2>/dev/null | sort -h || true
echo
echo "--- datasets ---"
du -sh "$HOME"/datasets/* 2>/dev/null || true
echo
echo "--- runs ---"
du -sh "$HOME"/runs/* 2>/dev/null || true
echo

echo "############ 3. verify every step0 video decodes ############"
export APPTAINERENV_DS="$HOME/datasets/towel_fold01_step0"
apptainer exec "$IMAGE" python - <<'PY'
import json, os
from pathlib import Path

ds = Path(os.environ["DS"])
videos = sorted(ds.rglob("*.mp4"))
print(f"{len(videos)} video files under {ds}\n", flush=True)

from torchcodec.decoders import VideoDecoder

bad = []
zero = []
for v in videos:
    size = v.stat().st_size
    if size == 0:
        zero.append(v)
        continue
    try:
        d = VideoDecoder(str(v), seek_mode="approximate")
        _ = d[0]                      # actually pull a frame, not just open
    except Exception as e:
        bad.append((v, size, type(e).__name__, str(e)[:120]))

print(f"zero-byte files : {len(zero)}", flush=True)
for v in zero[:20]:
    print("   ", v.relative_to(ds), flush=True)

print(f"undecodable     : {len(bad)}", flush=True)
for v, size, etype, msg in bad[:20]:
    print(f"    {v.relative_to(ds)}  {size} bytes  {etype}: {msg}", flush=True)

if not bad and not zero:
    print("\nAll videos decode fine -- the EIO is NOT truncated video files.", flush=True)
else:
    print("\nCorrupt/truncated files found. Delete the dataset dir and re-run prefetch.", flush=True)

info = json.loads((ds / "meta" / "info.json").read_text())
print(f"\ninfo.json: episodes={info['total_episodes']} frames={info['total_frames']}", flush=True)
PY

echo "############ done ############"
