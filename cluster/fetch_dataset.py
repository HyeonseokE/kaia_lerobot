"""Download one LeRobot dataset into $DATASETS_DIR, or exit 42 if it is not on the Hub.

Lives as a file rather than a heredoc inside the body because the body is itself
sourced from a wrapper that SLURM copies -- nesting a heredoc through that has bitten
this directory before. Exit 42 means "not collected yet", which the caller turns into a
clean skip; any other non-zero is a real failure.
"""
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

repo = os.environ["DS_REPO"]
dst = Path(os.environ["DATASETS_DIR"]) / repo.split("/")[-1]
try:
    snapshot_download(repo_id=repo, repo_type="dataset", revision="main",
                      local_dir=str(dst), max_workers=4)
except Exception as e:  # noqa: BLE001 -- any failure to resolve the repo means "wait"
    print(f"[wait] {repo} is not on the Hub yet ({type(e).__name__}: {e})", flush=True)
    sys.exit(42)
print(f"[done] {repo}", flush=True)
