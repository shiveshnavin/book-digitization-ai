import os
import pathlib
import argparse
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()
_raw_keys = os.getenv("GEMINI_API_KEY")
if not _raw_keys:
    raise RuntimeError("GEMINI_API_KEY not set — add it to your .env file")
GEMINI_API_KEYS = [k.strip() for k in _raw_keys.split(",") if k.strip()]
if not GEMINI_API_KEYS:
    raise RuntimeError("No valid keys found in GEMINI_API_KEY")

class Question(BaseModel):
    ocr_page: int
    question_no: int
    question: str
    exam_tags: str
    optionA: str
    optionB: str
    optionC: str
    optionD: str
    optionE: str
    has_images: bool

class Answer(BaseModel):
    ocr_page: int
    question_no: int
    has_images: bool
    correct_option: str
    correct_option_text: str
    explanation: str

class Chapter(BaseModel):
    ocr_page: int
    chapter_no: str
    chapter_name: str

class QuestionsList(BaseModel):
    items: list[Question]

class AnswersList(BaseModel):
    items: list[Answer]

class ChaptersList(BaseModel):
    items: list[Chapter]

SYS_QUESTIONS = """You are a precise data-extraction assistant.
You are given an image of a single PDF page from an Indian competitive exam study book.
Extract ALL questions present on the page and return them.
"question" must be the self-contained verbatim question text. Use Markdown for underlines/bold/table etc.
"exam_tags" is the explicit exam metadata tag printed next to or below the question (e.g. "SSC CHSL TIER-I, 29/11/2025 (Shift-01)", "SSC CGL 2023", etc.). Do NOT extract generic instructions like "Select the correct option."
"optionA", "optionB", "optionC", "optionD", "optionE"  are the five answer options (only the text, no numbering).
OptionE may be empty if the question has only four options.
"has_images" is true if the question has any images, diagrams, or figures associated with
"""

SYS_ANSWERS = """You are a precise data-extraction assistant.
You are given an image of a single PDF page from an Indian competitive exam study book.
Extract ALL answers present on the page and return them.
"correct_option" is A/B/C/D/E or 1/2/3/4/5.
"correct_option_text" is the text of the correct option.
"explanation" is the step-by-step solution text verbatim. Use Markdown for underlines/bold/table etc.
"""

SYS_CHAPTERS = """You are a precise data-extraction/OCR assistant.
You are given an image of a single PDF page from an Indian competitive exam study book.
Your ONLY task is to extract a major CHAPTER TITLE or CHAPTER BANNER (e.g., 'CHAPTER 5: ...' or a large standalone topic title like 'SPOT THE ERROR') IF AND ONLY IF it is prominently printed on this page.

CRITICAL RULES:
1. Look for large, bold main chapter headings or major topic titles at the top of the page.
2. If NO explicit main chapter banner/heading is printed on this page, return an EMPTY list: `items: []`.
3. DO NOT invent, guess, hallucinate, or infer chapter names from question text, options, or minor topics.
4. DO NOT extract minor sub-headers (like 'Examination wise Questions', 'Solutions', or 'SSC CGL 2025 Tier - I') as chapters.
5. If no explicit main chapter heading is present on the image, `items` MUST be empty (`[]`).
"""

def extract_pdf_page(pdf_path: str, page_number: int) -> str:
    """Extract a single PDF page (1-indexed) to a temporary PDF file next to the original.
    Returns the path of the created PDF."""
    from pypdf import PdfReader, PdfWriter

    pdf_dir = pathlib.Path(pdf_path).parent
    pdf_stem = pathlib.Path(pdf_path).stem
    out_path = pdf_dir / f"{pdf_stem}_page{page_number}.pdf"

    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    
    # page_number is 1-indexed
    writer.add_page(reader.pages[page_number - 1])
    
    with open(out_path, "wb") as f:
        writer.write(f)
        
    return str(out_path)


def generate_category(model, file_bytes, schema, sys_prompt, category_name, debug=False, progress_cb=None):
    import time
    import random
    
    api_key = random.choice(GEMINI_API_KEYS)
    client = genai.Client(api_key=api_key)
    
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    mime_type="application/pdf",
                    data=file_bytes,
                )
            ],
        ),
    ]
    # Removed GoogleSearch tool as it's not needed for OCR and consumes search quota
    tools = []
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level="MINIMAL",
        ),
        tools=tools,
        system_instruction=[
            types.Part.from_text(text=sys_prompt)
        ],
        response_mime_type="application/json",
        response_schema=schema,
    )

    chunks = []
    total_chars = 0
    t_start = time.perf_counter()

    if debug:
        print(f"\n--- Extracting {category_name} ---")
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if text := chunk.text:
            chunks.append(text)
            total_chars += len(text)
            
            if debug:
                print(text, end="", flush=True)
            elif progress_cb:
                progress_cb(category_name, len(text))

    elapsed = time.perf_counter() - t_start
    cps_final = total_chars / elapsed if elapsed > 0 else 0
    
    if debug:
        print()
        print(f"  [{category_name}] {total_chars} chars | {cps_final:.1f} ch/s | {elapsed:.1f}s")

    return "".join(chunks)

def generate(file_path: str, debug: bool = False, progress_cb=None) -> tuple[str, str, str]:
    import random
    import time
    import threading
    from concurrent.futures import ThreadPoolExecutor

    file_bytes = pathlib.Path(file_path).read_bytes()
    
    # Use gemma model for all queries as requested to support iterative testing
    model_q = "gemma-4-31b-it"
    model_a = "gemma-4-31b-it"
    model_c = "gemma-4-31b-it"

    t_start = time.perf_counter()
    category_chars = {"Questions": 0, "Answers": 0, "Chapters": 0}
    progress_lock = threading.Lock()

    def cat_progress_cb(cat_name, chunk_len):
        if not progress_cb:
            return
        with progress_lock:
            category_chars[cat_name] += chunk_len
            total_chars = sum(category_chars.values())
            elapsed = time.perf_counter() - t_start
            cps = total_chars / elapsed if elapsed > 0 else 0
            progress_cb(total_chars, cps, elapsed, done=False)

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_q = executor.submit(generate_category, model_q, file_bytes, QuestionsList, SYS_QUESTIONS, "Questions", debug, cat_progress_cb if progress_cb else None)
        future_a = executor.submit(generate_category, model_a, file_bytes, AnswersList, SYS_ANSWERS, "Answers", debug, cat_progress_cb if progress_cb else None)
        future_c = executor.submit(generate_category, model_c, file_bytes, ChaptersList, SYS_CHAPTERS, "Chapters", debug, cat_progress_cb if progress_cb else None)

        # Wait for all
        questions_text = future_q.result()
        answers_text = future_a.result()
        chapters_text = future_c.result()
    
    total_time = time.perf_counter() - t_start
    if progress_cb:
        total_chars = len(questions_text) + len(answers_text) + len(chapters_text)
        cps = total_chars / total_time if total_time > 0 else 0
        progress_cb(total_chars, cps, total_time, done=True)
    
    return questions_text, answers_text, chapters_text


DELIMITER = "---++---"


def _append_json_to_csv(json_text: str, output_path: str, injected_type: str, page_num: int | None = None):
    """Parse JSON containing a dict with an 'items' list and append to CSV."""
    import json, csv
    json_text = json_text.strip()
    if json_text.startswith("```json"):
        json_text = json_text[7:]
    if json_text.startswith("```"):
        json_text = json_text[3:]
    if json_text.endswith("```"):
        json_text = json_text[:-3]
    json_text = json_text.strip()

    if not json_text:
        return
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e} \nContent: {json_text}")
        return

    # Extract the items array from the parsed dict
    if isinstance(data, dict) and "items" in data:
        items = data["items"]
    elif isinstance(data, list):
        items = data
    else:
        items = []

    if not items:
        return

    # Inject the type manually at the front of each dictionary
    items_with_type = [{"type": injected_type, "physical_page": page_num, **item} for item in items]

    # Use fixed schemas for each type to guarantee column ordering across pages
    schemas = {
        "question": ["type", "physical_page", "ocr_page", "question_no", "question", "exam_tags", "optionA", "optionB", "optionC", "optionD", "has_images"],
        "answer": ["type", "physical_page", "ocr_page", "question_no", "has_images", "correct_option", "correct_option_text", "explanation"],
        "chapter": ["type", "physical_page", "ocr_page", "chapter_no", "chapter_name"]
    }
    headers = schemas.get(injected_type)
    if not headers:
        headers = list(items_with_type[0].keys())

    rows = [headers]
    for item in items_with_type:
        rows.append([item.get(h, "") for h in headers])

    file_exists = os.path.exists(output_path)
    with open(output_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            # write header + all rows
            writer.writerows(rows)
        else:
            # skip header
            data_rows = rows[1:] if len(rows) > 1 else []
            if data_rows:
                writer.writerows(data_rows)

def _append_raw(raw_text: str, page_num: int, raw_path: str, category: str):
    """Append one row to the audit raw CSV."""
    import csv
    file_exists = os.path.exists(raw_path)
    with open(raw_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["page_num", "category", "raw_output"])
        writer.writerow([page_num, category, raw_text.strip()])


def save_csvs(
    questions_text: str,
    answers_text:   str,
    chapters_text:  str,
    questions_path: str,
    answers_path:   str,
    chapters_path:  str,
    raw_path:      str | None = None,
    page_num:      int | None = None,
):
    """Save the JSON parts to CSV."""
    _append_json_to_csv(questions_text, questions_path, "question", page_num)
    _append_json_to_csv(answers_text,   answers_path, "answer", page_num)
    _append_json_to_csv(chapters_text,  chapters_path, "chapter", page_num)
    
    if raw_path and page_num is not None:
        _append_raw(questions_text, page_num, raw_path, "Questions")
        _append_raw(answers_text, page_num, raw_path, "Answers")
        _append_raw(chapters_text, page_num, raw_path, "Chapters")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract Q&A from a PDF page using Gemini."
    )
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument(
        "--start",
        type=int,
        required=True,
        help="Page number to extract (1-indexed)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output CSV file (default: <pdf_stem>_page<N>.csv next to the PDF)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Stream model output to console (default: silent, stats only)",
    )
    args = parser.parse_args()

    pdf_path = pathlib.Path(args.pdf)
    base          = pdf_path.parent / f"{pdf_path.stem}_page{args.start}"
    questions_csv = str(base) + "_questions.csv"
    answers_csv   = str(base) + "_answers.csv"
    chapters_csv  = str(base) + "_chapters.csv"
    raw_csv       = str(pdf_path.parent / f"{pdf_path.stem}_raw.csv")

    page_pdf_path = extract_pdf_page(args.pdf, args.start)
    print(f"[extract_page] Extracting page {args.start} -> {page_pdf_path}")
    try:
        questions_text, answers_text, chapters_text = generate(page_pdf_path, debug=args.debug)
        save_csvs(questions_text, answers_text, chapters_text, questions_csv, answers_csv, chapters_csv, raw_csv, args.start)
        print(f"[extract_page] Questions -> {questions_csv}")
        print(f"[extract_page] Answers   -> {answers_csv}")
        print(f"[extract_page] Chapters  -> {chapters_csv}")
        print(f"[extract_page] Raw audit -> {raw_csv}")
    finally:
        if os.path.exists(page_pdf_path):
            os.remove(page_pdf_path)