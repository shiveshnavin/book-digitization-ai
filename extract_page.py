import os
import pathlib
import argparse
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
_raw_keys = os.getenv("GEMINI_API_KEY")
if not _raw_keys:
    raise RuntimeError("GEMINI_API_KEY not set — add it to your .env file")
GEMINI_API_KEYS = [k.strip() for k in _raw_keys.split(",") if k.strip()]
if not GEMINI_API_KEYS:
    raise RuntimeError("No valid keys found in GEMINI_API_KEY")

SYSTEM_INSTRUCTIONS="""

You are a precise data-extraction assistant.
You are given an image of a single PDF page from an Indian competitive exam study book.

Extract ALL questions, answers, and chapters present on the page and return them ONLY as JSON arrays. No extra text, no markdown fences (like ```json).

Rules:
You need to output exactly 3 JSON arrays separated by ---++---.
The first array is for questions, followed by ---++---, then an array for answers, followed by ---++---, then an array for chapters.

1. The FIRST array (QUESTIONS) must start with exactly this header array:
   ["type", "ocr_page", "question_no", "question", "instruction", "optionA", "optionB", "optionC", "optionD", "has_images"]
   - Each subsequent element is a data array with exactly 10 values.
   - "type": "question"
   - "ocr_page": the page number as printed in the book
   - "question_no": the question number (1, 2, 3, ...)
   - "question": the self-contained verbatim question text. Use Markdown for underlines/bold.
   - "instruction": any explicit instruction text shown (optional)
   - "optionA", "optionB", "optionC", "optionD": the four answer options (only the text, no numbering)
   - "has_images": true/false
   Leave other fields empty if not present.

2. The SECOND array (ANSWERS) must start with exactly this header array:
   ["type", "ocr_page", "question_no", "has_images", "correct_option", "correct_option_text", "explanation"]
   - "type": "answer"
   - "correct_option": A/B/C/D or 1/2/3/4
   - "correct_option_text": the text of the correct option
   - "explanation": step-by-step solution text verbatim. Use Markdown for underlines/bold.

3. The THIRD array (CHAPTERS) must start with exactly this header array:
   ["type", "ocr_page", "chapter_no", "chapter_name"]
   - "type": "chapter"
   - "chapter_no": the chapter number if available
   - "chapter_name": the name of the chapter (look for large bold font)

If a section has no data, output only the header array inside the main array.
Example structure:
[
  ["type", "ocr_page", "question_no", "question", "instruction", "optionA", "optionB", "optionC", "optionD", "has_images"],
  ["question", 1, 1, "What is X?", "", "A", "B", "C", "D", false]
]
---++---
[
  ["type", "ocr_page", "question_no", "has_images", "correct_option", "correct_option_text", "explanation"],
  ["answer", 1, 1, false, "B", "B", "Explanation here"]
]
---++---
[
  ["type", "ocr_page", "chapter_no", "chapter_name"]
]

MANDATORY INSTRUCTIONS:
- Do NOT output anything outside the JSON arrays and delimiters - no explanations, no prose.
- Do NOT wrap your output in ```json or any other markdown code block format.
- Output MUST be valid JSON (use double quotes for strings, no trailing commas).
"""


def pdf_page_to_jpg(pdf_path: str, page_number: int) -> str:
    """Convert a single PDF page (1-indexed) to a JPG file next to the PDF.
    Returns the path of the created JPG."""
    from pdf2image import convert_from_path

    pdf_dir = pathlib.Path(pdf_path).parent
    pdf_stem = pathlib.Path(pdf_path).stem
    jpg_path = pdf_dir / f"{pdf_stem}_page{page_number}.jpg"

    pages = convert_from_path(
        pdf_path,
        first_page=page_number,
        last_page=page_number,
        dpi=200,
        fmt="jpeg",
    )

    if not pages:
        raise ValueError(f"Could not render page {page_number} from {pdf_path}")

    pages[0].save(str(jpg_path), "JPEG")
    return str(jpg_path)


def generate(image_path: str, debug: bool = False, progress_cb=None) -> str:
    """Run extraction and return the full CSV text.

    Args:
        progress_cb: optional callable(chars, cps, elapsed, done) called on
                     every streamed chunk instead of printing to stdout.
                     When provided, generate() stays completely silent.
    """
    import time
    import random

    api_key = random.choice(GEMINI_API_KEYS)
    client = genai.Client(
        api_key=api_key,
    )

    image_bytes = pathlib.Path(image_path).read_bytes()

    model = "gemma-4-31b-it"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    mime_type="image/jpeg",
                    data=image_bytes,
                )
            ],
        ),
    ]
    tools = [
        types.Tool(googleSearch=types.GoogleSearch()),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level="MINIMAL",
        ),
        tools=tools,
        system_instruction=[
            types.Part.from_text(text=SYSTEM_INSTRUCTIONS)
        ],
    )

    chunks = []
    total_chars = 0
    t_start = time.perf_counter()

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if text := chunk.text:
            chunks.append(text)
            total_chars += len(text)
            elapsed_so_far = time.perf_counter() - t_start
            cps = total_chars / elapsed_so_far if elapsed_so_far > 0 else 0
            if progress_cb:
                progress_cb(total_chars, cps, elapsed_so_far, done=False)
            elif debug:
                print(text, end="", flush=True)
            else:
                print(f"\r  {total_chars} chars | {cps:.1f} ch/s | {elapsed_so_far:.1f}s", end="", flush=True)

    elapsed = time.perf_counter() - t_start
    cps_final = total_chars / elapsed if elapsed > 0 else 0
    if progress_cb:
        progress_cb(total_chars, cps_final, elapsed, done=True)
    else:
        if debug:
            print()  # newline after raw stream
        print(f"\r  {total_chars} chars | {cps_final:.1f} ch/s | {elapsed:.1f}s")

    return "".join(chunks)


DELIMITER = "---++---"


def _append_json_to_csv(json_text: str, output_path: str):
    """Parse JSON array of arrays and append to CSV securely using python's csv module."""
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
        rows = json.loads(json_text)
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e} \nContent: {json_text}")
        return

    if not isinstance(rows, list) or not rows:
        return

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

def _append_raw(raw_text: str, page_num: int, raw_path: str):
    """Append one row (page_num, raw_output) to the audit raw CSV."""
    import csv
    file_exists = os.path.exists(raw_path)
    with open(raw_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["page_num", "raw_output"])
        writer.writerow([page_num, raw_text.strip()])


def split_and_save_csv(
    raw_text:      str,
    questions_path: str,
    answers_path:   str,
    chapters_path:  str,
    raw_path:      str | None = None,
    page_num:      int | None = None,
):
    """Split model output on DELIMITER and save each part to its own CSV.
    Optionally appends the full raw output to an audit file.
    """
    parts = raw_text.split(DELIMITER)
    questions_block = parts[0] if len(parts) > 0 else ""
    answers_block   = parts[1] if len(parts) > 1 else ""
    chapters_block  = parts[2] if len(parts) > 2 else ""
    _append_json_to_csv(questions_block, questions_path)
    _append_json_to_csv(answers_block,   answers_path)
    _append_json_to_csv(chapters_block,  chapters_path)
    if raw_path and page_num is not None:
        _append_raw(raw_text, page_num, raw_path)


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

    jpg_path = pdf_page_to_jpg(args.pdf, args.start)
    print(f"[extract_page] Rendering page {args.start} → {jpg_path}")
    try:
        raw_text = generate(jpg_path, debug=args.debug)
        split_and_save_csv(raw_text, questions_csv, answers_csv, chapters_csv, raw_csv, args.start)
        print(f"[extract_page] Questions → {questions_csv}")
        print(f"[extract_page] Answers   → {answers_csv}")
        print(f"[extract_page] Chapters  → {chapters_csv}")
        print(f"[extract_page] Raw audit → {raw_csv}")
    finally:
        if os.path.exists(jpg_path):
            os.remove(jpg_path)