#!/usr/bin/env python3
"""Back up a subject's pages directory to pages.zip beside it."""

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def backup(subject_dir: str | Path) -> Path:
    subject_path = Path(subject_dir).resolve()
    pages_dir = subject_path / "pages"
    if not pages_dir.is_dir():
        raise FileNotFoundError(f"Pages directory not found: {pages_dir}")

    archive_path = subject_path / "pages.zip"
    archive_path.unlink(missing_ok=True)
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(pages_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(subject_path))
    return archive_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Back up a subject pages directory")
    parser.add_argument("subject_dir", help="Subject directory containing pages/")
    args = parser.parse_args()
    output = backup(args.subject_dir)
    print(f"[backup_pages] Created {output}")
