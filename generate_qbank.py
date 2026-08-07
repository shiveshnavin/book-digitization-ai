import os
import json
import csv
import argparse
import pathlib
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Generate final question bank CSV from index.json")
    parser.add_argument("index_file", help="Path to the index.json file")
    args = parser.parse_args()

    index_path = pathlib.Path(args.index_file).resolve()
    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    files = index_data.get("files")
    if not files:
        raise ValueError("No 'files' block found in index.json")

    pdf_path = pathlib.Path(files["original_pdf"])
    questions_csv = pathlib.Path(files["questions_csv"])
    answers_csv = pathlib.Path(files["answers_csv"])
    chapters_csv = pathlib.Path(files["chapters_csv"])
    
    out_csv = index_path.parent / f"{pdf_path.stem}_question_bank.csv"

    # 1. Load chapters, sorted by ocr_page
    chapters = []
    if chapters_csv.exists():
        with open(chapters_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("type") != "chapter":
                    continue
                try:
                    page_num = int(row["ocr_page"])
                    chapter_name = row.get("chapter_name", "").strip()
                    if chapter_name:
                        chapters.append((page_num, chapter_name))
                except ValueError:
                    pass
            chapters.sort(key=lambda x: x[0])

    # Helper function to find chapters on a specific page
    def get_chapters_on_page(p):
        return [c for page, c in chapters if page == p]

    # 2. Load questions and track sequential chapters for questions
    questions_list = []
    if questions_csv.exists():
        with open(questions_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("type") != "question":
                    continue
                try:
                    row["ocr_page"] = int(row["ocr_page"])
                    questions_list.append(row)
                except ValueError:
                    pass
        # Sort questions by ocr_page and question_no
        questions_list.sort(key=lambda x: (x["ocr_page"], int(x.get("question_no", 0)) if str(x.get("question_no", "")).isdigit() else 9999))

    # Determine chapter for each question sequentially
    curr_q_chapter = ""
    for q in questions_list:
        p = q["ocr_page"]
        chaps_on_p = get_chapters_on_page(p)
        if chaps_on_p:
            # If multiple chapters on this page, the chapter headers are encountered.
            # We can use the first one as default or handle sequential transition.
            # Simple forward fill: last chapter header found up to page p.
            curr_q_chapter = chaps_on_p[-1]
        else:
            # Find the most recent chapter before page p
            temp_chap = ""
            for cp, cn in chapters:
                if cp <= p:
                    temp_chap = cn
                else:
                    break
            if temp_chap:
                curr_q_chapter = temp_chap
        q["computed_chapter"] = curr_q_chapter

    # 3. Load answers and track sequential chapters for answers separately
    answers_list = []
    if answers_csv.exists():
        with open(answers_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("type") != "answer":
                    continue
                try:
                    row["ocr_page"] = int(row["ocr_page"])
                    answers_list.append(row)
                except ValueError:
                    pass
        # Sort answers by ocr_page and question_no
        answers_list.sort(key=lambda x: (x["ocr_page"], int(x.get("question_no", 0)) if str(x.get("question_no", "")).isdigit() else 9999))

    # Determine chapter for each answer sequentially
    curr_a_chapter = ""
    for ans in answers_list:
        p = ans["ocr_page"]
        chaps_on_p = get_chapters_on_page(p)
        if chaps_on_p:
            curr_a_chapter = chaps_on_p[-1]
        else:
            # Find the most recent chapter before page p
            temp_chap = ""
            for cp, cn in chapters:
                if cp <= p:
                    temp_chap = cn
                else:
                    break
            if temp_chap:
                curr_a_chapter = temp_chap
        ans["computed_chapter"] = curr_a_chapter

    # Create map from answers: (computed_chapter, question_no) -> answer_row
    answers_map = {}
    answers_by_page = defaultdict(list)
    for ans in answers_list:
        ch = ans["computed_chapter"]
        q_no = str(ans.get("question_no", "")).strip()
        p = ans["ocr_page"]
        if ch and q_no:
            answers_map[(ch.lower(), q_no)] = ans
        answers_by_page[p].append(ans)

    # 4. Merge questions with answers
    output_rows = []
    for row in questions_list:
        chapter = row["computed_chapter"]
        q_no = str(row.get("question_no", "")).strip()
        p = row["ocr_page"]

        ans = {}
        if chapter and q_no and (chapter.lower(), q_no) in answers_map:
            ans = answers_map[(chapter.lower(), q_no)]
        elif answers_by_page[p]:
            ans = answers_by_page[p].pop(0)

        # Build the merged row based on target schema
        merged = {
            "tenant": "",
            "exam": "",
            "images": "",
            "rating": "",
            "subject": "",
            "topic": chapter,
            "question": row.get("question", ""),
            "optionA": row.get("optionA", ""),
            "optionB": row.get("optionB", ""),
            "optionC": row.get("optionC", ""),
            "optionD": row.get("optionD", ""),
            "correct_option": ans.get("correct_option", ""),
            "correct_option_text": ans.get("correct_option_text", ""),
            "explanation": ans.get("explanation", ""),
            "plan": "",
            "duration": "",
            "ext_links": "",
            "explanation_A": "",
            "explanation_B": "",
            "explanation_C": "",
            "explanation_D": "",
            "creator_id": "",
            "creator_name": "",
            "tags": row.get("exam_tags", row.get("instruction", ""))
        }
        output_rows.append(merged)

    # 4. Write final CSV
    headers = [
        "tenant","exam","images","rating","subject","topic","question",
        "optionA","optionB","optionC","optionD","correct_option",
        "correct_option_text","explanation","plan","duration","ext_links",
        "explanation_A","explanation_B","explanation_C","explanation_D",
        "creator_id","creator_name","tags"
    ]

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Generated question bank with {len(output_rows)} questions: {out_csv}")

if __name__ == "__main__":
    main()
