#!/usr/bin/env python3
"""Check every per-page JSON file generated for a PDF."""

import argparse
import json
import subprocess
from pathlib import Path
from extract_page_to_json import mark_page_failed


def pdf_page_count(pdf_path: Path) -> int | None:
    try:
        output = subprocess.check_output(["pdfinfo", str(pdf_path)], text=True, stderr=subprocess.STDOUT)
        for line in output.splitlines():
            if line.startswith("Pages:"):
                return int(line.split(":", 1)[1].strip())
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None
    return None


def validate_page(path: Path, expected_page: int) -> str | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return f"json parse failed: {exc}"
    if not isinstance(data, dict):
        return "root is not a JSON object"
    if data.get("page") != expected_page:
        return f"page field is {data.get('page')!r}, expected {expected_page}"
    if not isinstance(data.get("contents"), list):
        return "missing or invalid contents array"
    for index, item in enumerate(data["contents"]):
        if not isinstance(item, dict) or item.get("type") not in {"chapter", "question", "answer"}:
            return f"invalid contents item at index {index}"
        if not isinstance(item.get("data"), dict):
            return f"missing data object at contents index {index}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Find corrupt per-page JSON files for a PDF")
    parser.add_argument("pdf", help="Path to the PDF")
    parser.add_argument("--update-index", action="store_true", help="Record corrupt pages as failed in the PDF index")
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    pages_dir = pdf_path.parent / "pages"
    count = pdf_page_count(pdf_path)
    if count is None:
        page_numbers = sorted(
            int(path.stem.rsplit("_", 1)[1])
            for path in pages_dir.glob("page_*.json")
            if path.stem.rsplit("_", 1)[1].isdigit()
        )
        count = max(page_numbers, default=0)
        print("WARNING: could not read PDF page count; checking through highest page JSON.")

    corrupt = []
    missing = []
    for page in range(1, count + 1):
        path = pages_dir / f"page_{page}.json"
        if not path.exists():
            missing.append(path)
            continue
        error = validate_page(path, page)
        if error:
            corrupt.append((path, error))

    print(f"PDF: {pdf_path}")
    print(f"Expected pages: {count}")
    print(f"Valid: {count - len(corrupt) - len(missing)}")
    print(f"Corrupt: {len(corrupt)}")
    for path, error in corrupt:
        print(f"CORRUPT: {path} — {error}")
        if args.update_index:
            page = int(path.stem.rsplit("_", 1)[1])
            mark_page_failed(
                str(pdf_path.parent / f"{pdf_path.stem}_index.json"),
                page,
                error,
            )
    print(f"Missing: {len(missing)}")
    for path in missing:
        print(f"MISSING: {path}")
    return 1 if corrupt or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
