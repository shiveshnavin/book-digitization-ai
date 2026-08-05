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

    # 1. Load chapters to build page -> chapter mapping
    # Assuming chapters are ordered, we forward-fill the chapter for pages without one
    page_to_chapter = {}
    if chapters_csv.exists():
        with open(chapters_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Find the max page we might need to forward fill to, or just fill as we go
            # We'll just build a sorted list of (page, chapter_name)
            chapters = []
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
            # We'll write a helper to lookup chapter for a page
            def get_chapter(page):
                current_chapter = ""
                for p, c in chapters:
                    if p <= page:
                        current_chapter = c
                    else:
                        break
                return current_chapter
    else:
        def get_chapter(page):
            return ""

    # 2. Load answers grouped by ocr_page in sequential order
    answers_by_page = defaultdict(list)
    if answers_csv.exists():
        with open(answers_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("type") != "answer":
                    continue
                try:
                    p = int(row["ocr_page"])
                    answers_by_page[p].append(row)
                except ValueError:
                    pass

    # 3. Process questions and merge with answers sequentially per page
    output_rows = []
    if questions_csv.exists():
        with open(questions_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("type") != "question":
                    continue
                try:
                    p = int(row["ocr_page"])
                except ValueError:
                    continue

                ans = answers_by_page[p].pop(0) if answers_by_page[p] else {}
                chapter = get_chapter(p)
                
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
                    "tags": row.get("instruction", "")
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
