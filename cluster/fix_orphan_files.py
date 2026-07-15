#!/usr/bin/env python
"""Find and remove orphan data/video files in a LeRobotDataset v3 directory.

LeRobot concatenates every parquet under ``data/`` regardless of whether
``meta/episodes`` references it. An unreferenced ("orphan") file — typically the
remains of an aborted recording — therefore shifts every absolute frame index
after it, and since the dataset looks rows up by array position, all later
episodes silently read frames from the wrong episode until a timestamp finally
runs past the end of a video and it crashes.

This detects orphans by diffing the file indices referenced in meta/episodes
against the files actually on disk, and (with --apply) deletes them.

Note: files under meta/episodes/ use an independent chunking and are never
orphans — only data/ and videos/ are checked.

    python fix_orphan_files.py <dataset_root> [--apply]
"""

import argparse
import pathlib
import sys

import pandas as pd


def find_orphans(root: pathlib.Path) -> list[pathlib.Path]:
    eps = pd.concat([pd.read_parquet(p) for p in sorted((root / "meta/episodes").rglob("*.parquet"))])

    orphans: list[pathlib.Path] = []

    referenced = set(eps["data/file_index"])
    for p in sorted((root / "data").rglob("*.parquet")):
        if int(p.stem.split("-")[1]) not in referenced:
            orphans.append(p)

    videos = root / "videos"
    if videos.is_dir():
        for cam_dir in sorted(d for d in videos.iterdir() if d.is_dir()):
            col = f"videos/{cam_dir.name}/file_index"
            if col not in eps.columns:
                continue
            referenced = set(eps[col])
            for p in sorted(cam_dir.rglob("*.mp4")):
                if int(p.stem.split("-")[1]) not in referenced:
                    orphans.append(p)

    return orphans


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=pathlib.Path)
    ap.add_argument("--apply", action="store_true", help="delete the orphans (default: report only)")
    args = ap.parse_args()

    root = args.root
    orphans = find_orphans(root)

    rows_on_disk = sum(
        len(pd.read_parquet(p, columns=["index"])) for p in sorted((root / "data").rglob("*.parquet"))
    )
    eps = pd.concat([pd.read_parquet(p) for p in sorted((root / "meta/episodes").rglob("*.parquet"))])
    expected = int(eps["dataset_to_index"].max())

    print(f"{root.name}: rows on disk={rows_on_disk} expected={expected} drift={rows_on_disk - expected}")

    if not orphans:
        print("  no orphans")
        return 0

    for p in orphans:
        print(f"  ORPHAN {p.relative_to(root)}")

    if not args.apply:
        print("  (dry run — pass --apply to delete)")
        return 1

    for p in orphans:
        p.unlink()

    rows_after = sum(
        len(pd.read_parquet(p, columns=["index"])) for p in sorted((root / "data").rglob("*.parquet"))
    )
    print(f"  deleted {len(orphans)} file(s) -> rows on disk={rows_after}")
    if rows_after != expected:
        print(f"  ERROR: still {rows_after - expected} rows off", file=sys.stderr)
        return 1
    print("  OK: position == index restored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
