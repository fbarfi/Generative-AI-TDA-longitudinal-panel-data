#!/usr/bin/env python3
"""Restore the raw input data (Data_original) from this repository's data release.

The raw files (31 files, ~670 MB per pipeline version, including dyadic_format_v3.dta
at ~518 MB) exceed GitHub's file-size limit for files kept in the repository and are
third-party datasets (Correlates of War, Polity5, Gleditsch, Gibler). They are therefore
not tracked in git but attached as two zip files to the GitHub release "data-v1":

    Data_original_conda.zip    inputs of the Conda pipeline  (conda/.../Data_original)
    Data_original_docker.zip   inputs of the Docker pipeline (docker/.../Data_original)

This script downloads them, extracts each into its pipeline tree, and verifies every
file against the SHA-256 manifests in this folder.

Usage (from the repository root):
    python data/fetch_data.py                       # download both zips from the release
    python data/fetch_data.py --only conda          # restore one tree only
    python data/fetch_data.py --zip-dir ~/Downloads # use zips downloaded by hand from the release page

Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

RELEASE_TAG = "data-v1"
RELEASE_URL = ("https://github.com/fbarfi/Generative-AI-TDA-longitudinal-panel-data/"
               f"releases/download/{RELEASE_TAG}/")

REPO = Path(__file__).resolve().parent.parent

# tree -> (zip name on the release, target dir, manifest)
TREES = {
    "conda": (
        "Data_original_conda.zip",
        REPO / "conda/project/project_data/Data/Data_original",
        REPO / "data/conda_Data_original.sha256",
    ),
    "docker": (
        "Data_original_docker.zip",
        REPO / "docker/docker_execution/project_data/Data/Data_original",
        REPO / "data/docker_Data_original.sha256",
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_manifest(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            out[name.lstrip("*")] = digest
    return out


def download(name: str, dest: Path) -> None:
    url = RELEASE_URL + name
    print(f"Downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "fetch_data.py"})
    with urllib.request.urlopen(req) as r, dest.open("wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)


def restore(zip_path: Path, key: str) -> bool:
    _, target, manifest_path = TREES[key]
    manifest = read_manifest(manifest_path)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        members = {Path(m).name: m for m in z.namelist()
                   if not m.endswith("/") and "__MACOSX" not in m}
        missing = [n for n in manifest if n not in members]
        if missing:
            print(f"[{key}] not in {zip_path.name}: {', '.join(missing)}")
            return False
        for name in manifest:
            with z.open(members[name]) as src, (target / name).open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)
    bad = [n for n, d in manifest.items() if sha256(target / n) != d]
    rel = target.relative_to(REPO)
    if bad:
        print(f"[{key}] checksum mismatch in {rel}: {', '.join(bad)}")
        return False
    print(f"[{key}] {len(manifest)} files restored and verified in {rel}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--zip-dir", type=Path,
                    help="folder holding Data_original_conda.zip / Data_original_docker.zip "
                         f"downloaded by hand from the release '{RELEASE_TAG}'")
    ap.add_argument("--only", choices=sorted(TREES), help="restore one tree only")
    args = ap.parse_args()

    keys = [args.only] if args.only else list(TREES)
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for key in keys:
            zip_name = TREES[key][0]
            if args.zip_dir is not None:
                zip_path = args.zip_dir.expanduser() / zip_name
                if not zip_path.exists():
                    print(f"[{key}] {zip_path} not found")
                    ok = False
                    continue
            else:
                zip_path = Path(tmp) / zip_name
                download(zip_name, zip_path)
            ok = restore(zip_path, key) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
