#!/usr/bin/env python3
"""Mark chapter headings listed in a bad-topics CSV as ``bad_chapter``.

Usage:
    python scripts/mark_bad_chapters.py qbank/maths/rakesh_unique_bad_topics.csv

The bad-topics CSV must contain a ``topic`` column. Its parent directory is
used to locate the subject's ``pages/`` directory.
"""

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bad_topics_csv", type=Path)
    args = parser.parse_args()

    if not args.bad_topics_csv.is_file():
        parser.error(f"file does not exist: {args.bad_topics_csv}")

    with args.bad_topics_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "topic" not in (reader.fieldnames or []):
            parser.error("CSV must contain a 'topic' column")
        bad_topics = {row["topic"].strip() for row in reader if row.get("topic", "").strip()}

    pages_dir = args.bad_topics_csv.parent / "pages"
    if not pages_dir.is_dir():
        parser.error(f"pages directory does not exist: {pages_dir}")

    changed = 0
    for page_path in sorted(pages_dir.glob("*.json")):
        with page_path.open(encoding="utf-8") as handle:
            page = json.load(handle)
        page_changed = False
        for content in page.get("contents", []):
            data = content.get("data", {})
            if (content.get("type") == "chapter"
                    and data.get("chapter_name", "").strip() in bad_topics):
                content["type"] = "bad_chapter"
                page_changed = True
                changed += 1
        if page_changed:
            with page_path.open("w", encoding="utf-8") as handle:
                json.dump(page, handle, ensure_ascii=False, indent=4)
                handle.write("\n")

    print(f"Bad topics: {len(bad_topics)}")
    print(f"Chapter entries changed: {changed}")


if __name__ == "__main__":
    main()
