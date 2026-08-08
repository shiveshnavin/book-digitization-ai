#!/usr/bin/env python3
"""Assemble per-page extraction JSON into a chapter-aware question bank."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

QBANK_SCHEMA = [
    "tenant", "exam", "images", "rating", "subject", "topic", "question",
    "question_no", "page_no", "answer_page_no", "index",
    "optionA", "optionB", "optionC", "optionD", "optionE", "correct_option",
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


def merge_split_questions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge question continuation records marked with question_no -1."""
    merged: list[dict[str, Any]] = []
    last_question: dict[str, Any] | None = None
    for record in records:
        if record["type"] == "question":
            if str(record.get("question_no", "")).strip() == "-1" and last_question is not None:
                continuation = str(record.get("question", "")).strip()
                if continuation:
                    original = str(last_question.get("question", "")).strip()
                    last_question["question"] = f"{original}\n\n{continuation}" if original else continuation
                for field in ("optionA", "optionB", "optionC", "optionD", "optionE", "exam_tags"):
                    if not str(last_question.get(field, "")).strip() and str(record.get(field, "")).strip():
                        last_question[field] = record[field]
                continue
            last_question = record
        merged.append(record)
    return merged


def make_qbank(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def normalize_chapter(value: Any) -> str:
        return str(value or "").strip().casefold()

    output = []
    used_answers: set[int] = set()
    qbank_index = 0
    for question_index, question in enumerate(records):
        if question["type"] != "question":
            continue
        key = (normalize_chapter(question.get("chapter_name")),
               str(question.get("question_no", "")).strip())
        # Answers are in reading order, but a split question/answer can place
        # continuation records between the question and its answer. Search
        # forward from this question and use the first unused answer with the
        # same chapter and question number. This prevents repeated numbers in
        # later sections from consuming an earlier answer.
        start = question_index + 1
        answer = None
        for candidate in records[start:]:
            if candidate["type"] != "answer":
                continue
            candidate_key = (normalize_chapter(candidate.get("chapter_name")),
                             str(candidate.get("question_no", "")).strip())
            if candidate_key == key and id(candidate) not in used_answers:
                answer = candidate
                used_answers.add(id(candidate))
                break
        row = {field: "" for field in QBANK_SCHEMA}
        row.update({k: question.get(k, "") for k in ("question", "optionA", "optionB", "optionC", "optionD", "optionE")})
        row["question_no"] = question.get("question_no", "")
        row["page_no"] = question.get("page", "")
        row["answer_page_no"] = answer.get("page", "") if answer else ""
        row["index"] = qbank_index
        row["topic"] = question.get("chapter_name", "")
        row["tags"] = question.get("exam_tags", "")
        if answer:
            row.update({k: answer.get(k, "") for k in ("correct_option", "correct_option_text", "explanation")})
        for field in EMPTY_FIELDS:
            row[field] = ""
        output.append(row)
        qbank_index += 1
    return output


def combine(pdf: str | Path) -> tuple[Path, Path]:
    pdf_path = Path(pdf).resolve()
    index_path = pdf_path.parent / f"{pdf_path.stem}_index.json"
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Cannot combine: invalid index JSON {index_path}: {exc}") from exc
        failed = index.get("failed", [])
        if failed:
            pages = ", ".join(str(entry.get("page", "?")) for entry in failed)
            raise ValueError(f"Cannot combine: index contains failed pages ({pages})")
    pages_dir = pdf_path.parent / "pages"
    pages = load_pages(pages_dir)
    records = flatten_contents(pages)
    assign_chapters(records)
    records = merge_split_questions(records)
    for record in records:
        if record["type"] in {"question", "answer"} and str(record.get("question_no", "")).strip() == "-1":
            previous_page = max(1, record["page"] - 1)
            record["splitjoin"] = f"{previous_page},{record['page']}"
        else:
            record.pop("splitjoin", None)
    pages_out = pdf_path.parent / f"{pdf_path.stem}_pages.json"
    qbank_out = pdf_path.parent / f"{pdf_path.stem}_qbank.json"
    qbank_csv_out = pdf_path.parent / f"{pdf_path.stem}_qbank.csv"
    qbank = make_qbank(records)
    postprocess_option_e_answers(qbank)
    pages_out.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    qbank_out.write_text(json.dumps(qbank, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    critical_indices = {
        mismatch["index"]
        for mismatch in validate_answer_options(pdf_path, qbank, records)
    }
    with qbank_csv_out.open("w", encoding="utf-8", newline="") as handle:
        csv_schema = [
            field for field in QBANK_SCHEMA
            if field not in {"question_no", "page_no", "answer_page_no", "index"}
        ]
        writer = csv.DictWriter(
            handle,
            fieldnames=csv_schema,
            quoting=csv.QUOTE_ALL,
            escapechar="\\",
            lineterminator="\n",
        )
        writer.writeheader()
        safe_rows = [
            {
                key: "".join(
                    char for char in str(row.get(key, ""))
                    if char in "\n\r\t" or ord(char) >= 32
                )
                for key in csv_schema
            }
            for row in qbank
            if row["index"] not in critical_indices
        ]
        writer.writerows(safe_rows)
    validate(pages_out, qbank_out)
    return pages_out, qbank_out


def postprocess_option_e_answers(qbank: list[dict[str, Any]]) -> None:
    """Normalize banks where an E/5 answer belongs in option D instead."""
    for row in qbank:
        if not str(row.get("optionE", "")).strip():
            continue
        answer = str(row.get("correct_option", "")).strip().casefold().strip("()[]{}.")
        if answer not in {"e", "5"}:
            continue
        row["optionD"], row["optionE"] = row.get("optionE", ""), row.get("optionD", "")
        row["correct_option"] = "4" if answer == "5" else "d"


def validate_answer_options(pdf_path: Path, qbank: list[dict[str, Any]], records: list[dict[str, Any]]) -> list[int]:
    """Record answer texts that do not occur among their question options."""
    mismatches = []
    seen_indices = set()
    question_records = [record for record in records if record["type"] == "question"]
    for row_index, (row, question_record) in enumerate(zip(qbank, question_records)):
        answer = str(row.get("correct_option_text", "")).strip()
        if not answer:
            continue
        options = {
            str(row.get(field, "")).strip().casefold()
            for field in ("optionA", "optionB", "optionC", "optionD", "optionE")
        }
        correct_option = str(row.get("correct_option", "")).strip().casefold()
        normalized_option = correct_option.strip("()[]{}.")
        option_index = {
            "a": "optionA", "b": "optionB", "c": "optionC",
            "d": "optionD", "e": "optionE",
            "1": "optionA", "2": "optionB", "3": "optionC",
            "4": "optionD", "5": "optionE",
        }.get(normalized_option)
        # In formats such as Spot the Error, correct_option_text is the
        # corrected sentence fragment, while the option itself is only (1),
        # (2), (3), etc. A valid option pointer is sufficient in that case.
        option_pointer_is_valid = bool(option_index and str(row.get(option_index, "")).strip())
        if answer.casefold() not in options and not option_pointer_is_valid:
            if row_index not in seen_indices:
                mismatches.append({
                    "index": row_index,
                    "question_no": question_record.get("question_no", ""),
                    "page_no": question_record.get("page", ""),
                    "answer_page_no": row.get("answer_page_no", ""),
                })
                seen_indices.add(row_index)

    index_path = pdf_path.parent / f"{pdf_path.stem}_index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["critical_mismatched_answers"] = mismatches
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"\n========== CRITICAL ANSWER MISMATCHES | {pdf_path.stem} | "
        f"COUNT: {len(mismatches)} =========="
    )
    return mismatches


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
