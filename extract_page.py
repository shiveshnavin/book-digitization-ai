import os
import pathlib
import argparse
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set — add it to your .env file")

SYSTEM_INSTRUCTIONS="""

You are a precise data-extraction assistant.
You are given an image of a single PDF page from an Indian competitive exam study book 

Extract ALL questions and answers present on the page and return them ONLY as raw CSV (comma-separated values). No extra text

Rules:
You need to have 2 sections in the CSV: one for questions and one for answers separated by ---++---.  The question section comes first, followed by the answer section.  Each section has its own header row.

1. The FIRST line must be EXACTLY this header (13 columns):
   type,ocr_page,question_no,question,instruction,optionA,optionB,optionC,optionD,has_images
2. Each subsequent line is one data row with exactly 13 comma-separated values.
3. "type" must be one of:  question | answer
4. For a QUESTION row fill: type, ocr_page, question_no, question,instruction, optionA, optionB, optionC, optionD, has_images.
    type: question
    ocr_page: the page number as printed in the book (may differ from PDF page order)
    question_no: the question number (1, 2, 3, ...)
    question: Must be simple Markdown to account for underlines, bold highlighted words etc. the self-contained standalone verbatim of the question containing the sentence/word/context
    instruction: any explicit instruction text shown on the page (optional)
    optionA/B/C/D: the four answer options, ALL OPTIONS MUST BE ENCLOSED IN DOUBLE-QUOTES.  Escape any internal double-quotes by doubling them. Only option text, dont include numbering 
    has_images: true/false – whether visuals exist on this item
    Leave instruction empty unless explicitly shown. Leave correct_option, correct_option_text, explanation empty.
ONCE ALL QUESTIONS ARE EXTRACTED, THEN EXTRACT ALL ANSWERS
ADD A SEPARATOR ---++--- before the first answer row.  This is to help distinguish between question and answer rows.

5. For an ANSWER section wull have the headers: 
type,ocr_page,question_no,has_images,correct_option,correct_option_text,explanation
each answer row will have the following fields:
    type: answer
    question_no: the question number (1, 2, 3, ...)
    has_images: true/false – whether visuals exist on this item
    correct_option: A/B/C/D or 1/2/3/4 depending on the option numbering
    correct_option_text: the text of the correct option (e.g. \"42\" or \"Agra\") if available, else leave empty
    explanation: the step-by-step solution text verbatim if available, else leave empty.  Must be Markdown to account for underlines, bold highlighted words etc

   Leave question, instruction, optionA, optionB, optionC, optionD ALL empty.
6. ocr_page = the page number as PRINTED/VISIBLE on the image
7. If a field value contains a comma, newline, or double-quote, wrap the entire field in double-quotes and escape internal double-quotes by doubling them (\"\").
8. Do NOT output anything outside the CSV - no fences, no explanations, no prose.
9. A single page may have both question AND answer. question row and answer row separately
10. If the page is blank, a cover, a table of contents, or contains no questions/answers, output only the header line.



MANDATORY INSTRUCTIONS:
- only question and explanation text should be in Markdown. All other fields should be plain text.
- If a field has one or more commas (,) then whole field must be wrapped in double quotes (") and any internal double quotes must be escaped by doubling them ("").
- YOUR OUTPUT MUST BE A CSV IN GIVEN FORMAT
- Donot add extra spaces with separators or commas.  Only the required number of commas for the number of columns.
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

    client = genai.Client(
        api_key=GEMINI_API_KEY,
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


def _append_csv(csv_text: str, output_path: str):
    """Append CSV block to a file, writing the header only on first write."""
    lines = csv_text.strip().splitlines()
    if not lines:
        return
    file_exists = os.path.exists(output_path)
    with open(output_path, "a", encoding="utf-8", newline="") as f:
        if not file_exists:
            f.write("\n".join(lines) + "\n")
        else:
            data_rows = lines[1:]   # skip repeated header
            if data_rows:
                f.write("\n".join(data_rows) + "\n")


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
    raw_path:      str | None = None,
    page_num:      int | None = None,
):
    """Split model output on DELIMITER and save each half to its own CSV.
    Optionally appends the full raw output to an audit file.
    """
    parts = raw_text.split(DELIMITER)
    questions_block = parts[0] if len(parts) > 0 else ""
    answers_block   = parts[1] if len(parts) > 1 else ""
    _append_csv(questions_block, questions_path)
    _append_csv(answers_block,   answers_path)
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
    raw_csv       = str(pdf_path.parent / f"{pdf_path.stem}_raw.csv")

    jpg_path = pdf_page_to_jpg(args.pdf, args.start)
    print(f"[extract_page] Rendering page {args.start} → {jpg_path}")
    try:
        raw_text = generate(jpg_path, debug=args.debug)
        split_and_save_csv(raw_text, questions_csv, answers_csv, raw_csv, args.start)
        print(f"[extract_page] Questions → {questions_csv}")
        print(f"[extract_page] Answers   → {answers_csv}")
        print(f"[extract_page] Raw audit → {raw_csv}")
    finally:
        if os.path.exists(jpg_path):
            os.remove(jpg_path)