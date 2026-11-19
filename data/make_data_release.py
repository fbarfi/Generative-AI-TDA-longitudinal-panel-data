#!/usr/bin/env python3
"""Build the two data zips attached to the GitHub release "data-v1" (author's tool).

Checks every Data_original file against the SHA-256 manifests in this folder and writes

    dist/Data_original_conda.zip
    dist/Data_original_docker.zip

with the 31 files of each tree at the top level of the zip.

Usage (from the repository root):
    python data/make_data_release.py                         # read the trees inside this repository
    python data/make_data_release.py --conda DIR --docker DIR  # read Data_original folders elsewhere

Standard library only.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_data import REPO, TREES, read_manifest, sha256  # noqa: E402


def build(key: str, src: Path, out_dir: Path) -> bool:
    zip_name, _, manifest_path = TREES[key]
    manifest = read_manifest(manifest_path)
    missing = [n for n in manifest if not (src / n).is_file()]
    if missing:
        print(f"[{key}] missing in {src}: {', '.join(missing)}")
        return False
    bad = [n for n, d in manifest.items() if sha256(src / n) != d]
    if bad:
        print(f"[{key}] checksum mismatch in {src}: {', '.join(bad)}")
        return False
    out = out_dir / zip_name
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in sorted(manifest):
            z.write(src / name, arcname=name)
    print(f"[{key}] {len(manifest)} files verified -> {out} ({out.stat().st_size / 1e6:.0f} MB)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--conda", type=Path, default=TREES["conda"][1], help="Conda Data_original folder")
    ap.add_argument("--docker", type=Path, default=TREES["docker"][1], help="Docker Data_original folder")
    ap.add_argument("--out", type=Path, default=REPO / "dist", help="output folder (default: dist/)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ok = build("conda", args.conda.expanduser(), args.out)
    ok = build("docker", args.docker.expanduser(), args.out) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
