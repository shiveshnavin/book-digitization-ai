#!/usr/bin/env python3
"""Assemble per-page extraction JSON into a chapter-aware question bank."""

import argparse
import json
from pathlib import Path
from typing import Any

QBANK_SCHEMA = [
    "tenant", "exam", "images", "rating", "subject", "topic", "question",
    "optionA", "optionB", "optionC", "optionD", "correct_option",
    "correct_option_text", "explanation", "plan", "duration", "ext_links",
    "explanation_A", "explanation_B", "explanation_C", "explanation_D",
    "creator_id", "creator_name", "tags",
]
EMPTY_FIELDS = {
    "tenant", "exam", "images", "rating", "subject", "plan", "duration",
    "ext_links", "explanation_A", "explanation_B", "explanation_C",
    "explanation_D", "creator_id", "creator_name",
}


def _page_number(path: Path, data: dict[str, Any]) -> int:
    try:
        return int(data.get("page", path.stem.rsplit("_", 1)[-1]))
    except (TypeError, ValueError):
        return 0


def load_pages(pages_dir: Path) -> list[dict[str, Any]]:
    pages = []
    for path in sorted(pages_dir.glob("page_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("contents"), list):
                raise ValueError("expected an object with a contents array")
            pages.append(data)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Invalid page JSON {path}: {exc}") from exc
    return sorted(pages, key=lambda d: _page_number(Path(), d))


def flatten_contents(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for page in pages:
        page_no = _page_number(Path(), page)
        for position, item in enumerate(page["contents"]):
            if not isinstance(item, dict) or item.get("type") not in {"chapter", "question", "answer"}:
                raise ValueError(f"Invalid content item on page {page_no}, position {position}")
            record = item.get("data", {})
            if not isinstance(record, dict):
                raise ValueError(f"Invalid data on page {page_no}, position {position}")
            result.append({"type": item["type"], "page": page_no, "position": position, **record})
    return result


def assign_chapters(records: list[dict[str, Any]]) -> None:
    # Chapter state is shared in reading order, while each record receives the
    # chapter active at the exact point where it appeared.
    active = ""
    for record in records:
        if record["type"] == "chapter":
            active = str(record.get("chapter_name", "")).strip()
        elif record["type"] in {"question", "answer"}:
            record["chapter_name"] = active


def make_qbank(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    questions = [r for r in records if r["type"] == "question"]
    answers = [r for r in records if r["type"] == "answer"]
    answer_map: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for answer in answers:
        key = (answer.get("chapter_name", "").casefold(), str(answer.get("question_no", "")).strip())
        answer_map.setdefault(key, []).append(answer)

    used: set[int] = set()
    output = []
    for question in questions:
        key = (question.get("chapter_name", "").casefold(), str(question.get("question_no", "")).strip())
        candidates = answer_map.get(key, [])
        answer = next((a for a in candidates if id(a) not in used), None)
        if answer is None:
            # Handles -1/blank spillovers without attaching an answer twice.
            answer = next((a for a in answers if id(a) not in used and a["page"] >= question["page"]), None)
        if answer is not None:
            used.add(id(answer))
        row = {field: "" for field in QBANK_SCHEMA}
        row.update({k: question.get(k, "") for k in ("question", "optionA", "optionB", "optionC", "optionD")})
        row["topic"] = question.get("chapter_name", "")
        row["tags"] = question.get("exam_tags", "")
        if answer:
            row.update({k: answer.get(k, "") for k in ("correct_option", "correct_option_text", "explanation")})
        for field in EMPTY_FIELDS:
            row[field] = ""
        output.append(row)
    return output


def combine(pdf: str | Path) -> tuple[Path, Path]:
    pdf_path = Path(pdf).resolve()
    pages_dir = pdf_path.parent / "pages"
    pages = load_pages(pages_dir)
    records = flatten_contents(pages)
    assign_chapters(records)
    for record in records:
        if record["type"] in {"question", "answer"} and str(record.get("question_no", "")).strip() == "-1":
            record["remarks"] = "splitjoin"
        else:
            record.pop("remarks", None)
    pages_out = pdf_path.parent / f"{pdf_path.stem}_pages.json"
    qbank_out = pdf_path.parent / f"{pdf_path.stem}_qbank.json"
    qbank = make_qbank(records)
    pages_out.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    qbank_out.write_text(json.dumps(qbank, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    validate(pages_out, qbank_out)
    return pages_out, qbank_out


def validate(pages_path: Path, qbank_path: Path) -> None:
    pages = json.loads(pages_path.read_text(encoding="utf-8"))
    qbank = json.loads(qbank_path.read_text(encoding="utf-8"))
    if not isinstance(pages, list) or not all(isinstance(r, dict) for r in pages):
        raise ValueError(f"{pages_path} must contain a JSON array of records")
    if not isinstance(qbank, list) or any(list(row) != QBANK_SCHEMA for row in qbank):
        raise ValueError(f"{qbank_path} has an invalid qbank schema")
    if any(row[field] != "" for row in qbank for field in EMPTY_FIELDS):
        raise ValueError(f"{qbank_path} has non-empty reserved fields")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", help="PDF whose pages/page_*.json files should be assembled")
    args = parser.parse_args()
    pages_file, qbank_file = combine(args.pdf)
    print(f"[combine_json] Pages -> {pages_file}")
    print(f"[combine_json] Qbank -> {qbank_file}")
