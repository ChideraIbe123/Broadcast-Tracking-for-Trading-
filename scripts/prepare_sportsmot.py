"""Extract the basketball sequences of a SportsMOT split from the Hugging Face tarball.

  uv run python scripts/prepare_sportsmot.py val      # after scripts/download_datasets.py / hf download

Reads data/sportsmot_hf/dataset/<split>.tar and data/sportsmot_hf/splits_txt/*.txt,
writes data/sportsmot/<split>/<seq>/{img1,gt,seqinfo.ini} for basketball only and
copies splits_txt. SportsMOT terms: do not redistribute.
"""
from __future__ import annotations

import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF = ROOT / "data/sportsmot_hf"
OUT = ROOT / "data/sportsmot"


def main(split: str) -> None:
    names = set((HF / "splits_txt/basketball.txt").read_text().split())
    tar_path = HF / "dataset" / f"{split}.tar"
    if not tar_path.exists():
        sys.exit(f"missing {tar_path}")
    (OUT / "splits_txt").mkdir(parents=True, exist_ok=True)
    for f in (HF / "splits_txt").glob("*.txt"):
        shutil.copy(f, OUT / "splits_txt" / f.name)
    done = set()
    with tarfile.open(tar_path) as t:
        members = []
        for m in t:
            parts = Path(m.name).parts
            # layouts seen: <split>/<seq>/... or <seq>/...
            seq = parts[1] if len(parts) > 1 and parts[0] == split else parts[0]
            if seq in names:
                rel = Path(*parts[parts.index(seq):])
                m.name = str(Path(split) / rel)
                members.append(m)
                done.add(seq)
        t.extractall(OUT, members=members, filter="data")
    print(f"{split}: extracted {len(done)} basketball sequences -> {OUT / split}")
    missing = names - done
    if missing:
        print("not in this split:", len(missing))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "val")
