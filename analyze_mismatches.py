#!/usr/bin/env python3
"""Print bounded manual-fix instructions for critical answer mismatches."""

import argparse
import json
from pathlib import Path


CONTEXT = 50


def matching_lines(path: Path, question_no: object) -> list[int]:
    needle = f'"question_no": {json.dumps(question_no)}'
    lines = path.read_text(encoding="utf-8").splitlines()
    return [number for number, line in enumerate(lines, 1) if needle in line]


def ranges(lines: list[int], total: int) -> str:
    if not lines:
        return "no matching line found"
    windows = []
    for line in lines:
        start = max(1, line - CONTEXT)
        end = min(total, line + CONTEXT)
        if windows and start <= windows[-1][1] + 1:
            windows[-1] = (windows[-1][0], max(windows[-1][1], end))
        else:
            windows.append((start, end))
    return ", ".join(f"{start}-{end}" for start, end in windows)


def page_hint(subject_dir: Path, page: int, question_no: object, label: str) -> str:
    path = subject_dir / "pages" / f"page_{page}.json"
    if not path.exists():
        return f"{label}: {path} (does not exist)"
    lines = path.read_text(encoding="utf-8").splitlines()
    matches = matching_lines(path, question_no)
    return (
        f"{label}: {path}\n"
        f"  inspect lines {ranges(matches, len(lines))} "
        f'(matching `"question_no": {question_no}`)'
    )


def main(pdf: str) -> None:
    pdf_path = Path(pdf).resolve()
    subject_dir = pdf_path.parent
    index_path = subject_dir / f"{pdf_path.stem}_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    mismatches = index.get("critical_mismatched_answers", [])
    if not mismatches:
        print(f"No critical mismatches in {index_path}")
        return

    for mismatch in mismatches:
        qno = mismatch.get("question_no", "")
        qpage = int(mismatch["page_no"])
        apage = int(mismatch.get("answer_page_no", qpage))
        print(f"Mismatch in question_no {qno} of page {qpage}")
        print(page_hint(subject_dir, qpage, qno, "Question source"))
        next_qpage = subject_dir / "pages" / f"page_{qpage + 1}.json"
        if next_qpage.exists():
            print(f"  Also inspect the top 50 lines of {next_qpage}")
        print(
            f"  If the JSON is insufficient, inspect PDF page {qpage} in {pdf_path}"
            + (f" and adjacent page {qpage + 1}" if next_qpage.exists() else "")
            + "."
        )
        print(page_hint(subject_dir, apage, qno, "Answer source"))
        next_apage = subject_dir / "pages" / f"page_{apage + 1}.json"
        if next_apage.exists():
            print(f"  Also inspect the top 50 lines of {next_apage}")
        if apage != qpage or next_apage.exists():
            pdf_pages = [str(apage)]
            if next_apage.exists() and apage + 1 != qpage:
                pdf_pages.append(str(apage + 1))
            print(
                f"  If the JSON is insufficient, inspect PDF page(s) {', '.join(pdf_pages)} in {pdf_path}."
            )
        print(
            "Surgically edit the exact pages/page_<N>.json record listed above where the question "
            "or answer content starts "
            "(the question page for question content, or the answer page for answer content). "
            "If content spills onto the adjacent page, copy the spilled content into the starting "
            "record so that record becomes complete, then set the copied continuation fields on the "
            "adjacent-page record to empty strings (\"\"). Do not delete the continuation record. "
            "Do not edit generated qbank or index files."
        )
        print("-" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze critical qbank mismatches")
    parser.add_argument("pdf", help="Subject PDF path")
    args = parser.parse_args()
    main(args.pdf)
