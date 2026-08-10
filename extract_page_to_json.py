import os
import json
import pathlib
import argparse
from typing import List, Literal, Union
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai
from google.genai import types

def mark_page_failed(index_path: str, page: int, error: str) -> None:
    path = pathlib.Path(index_path)
    index = {"completed": [], "failed": []}
    if path.exists():
        try:
            index = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    index["completed"] = [p for p in index.get("completed", []) if p != page]
    index["failed"] = [e for e in index.get("failed", []) if e.get("page") != page]
    index["failed"].append({"page": page, "error": error})
    path.write_text(json.dumps(index, indent=2), encoding="utf-8")

load_dotenv()
_raw_keys = os.getenv("GEMINI_API_KEY")
if not _raw_keys:
    raise RuntimeError("GEMINI_API_KEY not set — add it to your .env file")
GEMINI_API_KEYS = [k.strip() for k in _raw_keys.split(",") if k.strip()]
if not GEMINI_API_KEYS:
    raise RuntimeError("No valid keys found in GEMINI_API_KEY")

class ChapterData(BaseModel):
    chapter_no: str
    chapter_name: str
    instructions: str = ""

class QuestionData(BaseModel):
    question_no: int
    question: str
    exam_tags: str
    optionA: str
    optionB: str
    optionC: str
    optionD: str
    optionE: str
    has_images: bool

class AnswerData(BaseModel):
    question_no: int
    has_images: bool
    correct_option: str
    correct_option_text: str
    explanation: str

class ChapterContent(BaseModel):
    type: str = Field(description="Must be exactly 'chapter'")
    data: ChapterData

class QuestionContent(BaseModel):
    type: str = Field(description="Must be exactly 'question'")
    data: QuestionData

class AnswerContent(BaseModel):
    type: str = Field(description="Must be exactly 'answer'")
    data: AnswerData

class PageContent(BaseModel):
    page: int
    contents: list[Union[ChapterContent, QuestionContent, AnswerContent]]

SYS_PROMPT = """You are a precise data-extraction assistant processing a single PDF page from an Indian competitive exam study book.
Your task is to extract all chapters, questions, and answers from the page IN THE EXACT SEQUENCE they appear (normal reading order: top to bottom, left to right).

CRITICAL RULES:
1. Sequence matters: If a chapter heading appears before a question, the chapter must appear before the question in the `contents` list.
2. MISSING FIELDS & SPILLOVERS:
   - If a question or answer is a continuation from the previous page and question number cannot be determined, extract it anyway and set "question_no" to -1.
   - If you cannot determine any string field with certainty (e.g., chapter_no, exam_tags, correct_option, explanation, optionA-E), set it strictly to "" (empty string).
3. For Chapters:
   - Extract major CHAPTER TITLE or CHAPTER BANNER (e.g., 'CHAPTER 5: ...' or a large standalone topic title like 'SPOT THE ERROR') IF AND ONLY IF it is prominently printed.
   - Extract the chapter "instructions" (e.g. 'Direction :- Spot the section with errors...') if printed under or near the chapter heading. If no explicit instructions are present, set it to "".
   - DO NOT extract minor sub-headers (like 'Examination wise Questions', 'Solutions', or 'SSC CGL 2025 Tier - I') as chapters.
   - If no explicit chapter heading is present, simply don't add a chapter item.
4. For Questions:
   - Extract the full verbatim question text. Use Markdown for underlines/bold/tables/etc.
   - "exam_tags" is the explicit exam metadata tag printed next to or below the question. Do NOT extract generic instructions.
   - "optionA", "optionB", "optionC", "optionD", "optionE" are the five answer options (text only, no numbering). OptionE may be empty.
   - "has_images" is true if the question has any images, diagrams, or figures.
5. For Answers:
   - "correct_option" is A/B/C/D/E or 1/2/3/4/5.
   - "correct_option_text" is the text of the correct option.
   - "explanation" is the step-by-step solution text verbatim. Use Markdown for underlines/bold/tables.
6. Content Types:
   - Set the `type` field precisely to "chapter", "question", or "answer".
"""

def extract_pdf_page(pdf_path: str, page_number: int) -> str:
    """Extract a single PDF page (1-indexed) to a temporary PDF file next to the original."""
    from pypdf import PdfReader, PdfWriter

    pdf_dir = pathlib.Path(pdf_path).parent
    pdf_stem = pathlib.Path(pdf_path).stem
    out_path = pdf_dir / f"{pdf_stem}_temp_page{page_number}.pdf"

    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    
    writer.add_page(reader.pages[page_number - 1])
    
    with open(out_path, "wb") as f:
        writer.write(f)
        
    return str(out_path)

def generate_page_json(
    pdf_file_path: str,
    page_number: int,
    debug: bool = False,
    progress_cb=None
) -> str:
    import time
    import base64
    import pathlib
    from openai import OpenAI

    file_bytes = pathlib.Path(pdf_file_path).read_bytes()
    pdf_b64 = base64.b64encode(file_bytes).decode("utf-8")

    client = OpenAI(
        base_url="https://ai.semibit.in/openai/v1",
        api_key="YOUR_BIFROST_API_KEY",
    )

    t_start = time.perf_counter()
    total_chars = 0
    chunks = []

    if debug:
        print(f"\n--- Extracting Page {page_number} ---")

    stream = client.chat.completions.create(
        model="gemini/gemma-4-31b-it",

        messages=[
            {
                "role": "system",
                "content": SYS_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Extract the content for page {page_number}.",
                    },
                    {
                        "type": "file",
                        "file": {
                            "filename": "document.pdf",
                            "file_data": f"data:application/pdf;base64,{pdf_b64}",
                        },
                    },
                ],
            },
        ],

        stream=True,

        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "page_content",
                "schema": PageContent.model_json_schema(),
                "strict": True,
            },
        },
    )

    for chunk in stream:
        if not chunk.choices:
            continue

        text = chunk.choices[0].delta.content

        if text:
            chunks.append(text)
            total_chars += len(text)

            if debug:
                print(text, end="", flush=True)

            elif progress_cb:
                elapsed = time.perf_counter() - t_start
                cps = total_chars / elapsed if elapsed > 0 else 0

                try:
                    progress_cb(
                        total_chars,
                        cps,
                        elapsed,
                        done=False
                    )
                except Exception:
                    pass

    elapsed = time.perf_counter() - t_start

    if progress_cb:
        cps = total_chars / elapsed if elapsed > 0 else 0

        try:
            progress_cb(
                total_chars,
                cps,
                elapsed,
                done=True
            )
        except Exception:
            pass

    if debug:
        print()

    return "".join(chunks)


# Deprecated: Use `generate_page_json` instead, which uses the OpenAI-compatible API.
def generate_page_json_genai(pdf_file_path: str, page_number: int, debug: bool = False, progress_cb=None) -> str:
    import time
    import random
    
    file_bytes = pathlib.Path(pdf_file_path).read_bytes()
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
    
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level="MINIMAL",
        ),
        tools=[],
        system_instruction=[
            types.Part.from_text(text=SYS_PROMPT)
        ],
        response_mime_type="application/json",
        response_schema=PageContent,
    )

    t_start = time.perf_counter()
    total_chars = 0
    if debug:
        print(f"\n--- Extracting Page {page_number} ---")
        
    chunks = []
    for chunk in client.models.generate_content_stream(
        model="gemma-4-31b-it",
        contents=contents,
        config=generate_content_config,
    ):
        if text := chunk.text:
            chunks.append(text)
            total_chars += len(text)
            if debug:
                print(text, end="", flush=True)
            elif progress_cb:
                elapsed = time.perf_counter() - t_start
                cps = total_chars / elapsed if elapsed > 0 else 0
                try:
                    progress_cb(total_chars, cps, elapsed, done=False)
                except Exception:
                    # Don't let progress callback failures break extraction
                    pass

    elapsed = time.perf_counter() - t_start
    if progress_cb:
        cps = total_chars / elapsed if elapsed > 0 else 0
        try:
            progress_cb(total_chars, cps, elapsed, done=True)
        except Exception:
            pass

    if debug:
        print()
        
    return "".join(chunks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract unified JSON from a PDF page.")
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument("--start", type=int, required=True, help="Page number to extract (1-indexed)")
    parser.add_argument("--debug", action="store_true", help="Stream model output to console")
    args = parser.parse_args()

    pdf_path = pathlib.Path(args.pdf)
    pages_dir = pdf_path.parent / "pages"
    pages_dir.mkdir(exist_ok=True)
    
    out_json_path = pages_dir / f"page_{args.start}.json"
    temp_pdf_path = extract_pdf_page(args.pdf, args.start)
    
    print(f"[extract_page_to_json] Extracting page {args.start} -> {temp_pdf_path}")
    try:
        json_text = generate_page_json(temp_pdf_path, args.start, debug=args.debug)
        
        # Clean up JSON if necessary
        json_text = json_text.strip()
        if json_text.startswith("```json"):
            json_text = json_text[7:]
        if json_text.startswith("```"):
            json_text = json_text[3:]
        if json_text.endswith("```"):
            json_text = json_text[:-3]
        
        # Ensure it has the page number injected correctly just in case the LLM messed it up
        try:
            data = json.loads(json_text)
            data["page"] = args.start
            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except json.JSONDecodeError:
            with open(out_json_path, "w", encoding="utf-8") as f:
                f.write(json_text)
            mark_page_failed(
                str(pdf_path.parent / f"{pdf_path.stem}_index.json"),
                args.start,
                "json parse failed: model returned invalid JSON",
            )
            raise ValueError("json parse failed: model returned invalid JSON")
                
        print(f"[extract_page_to_json] Saved unified JSON -> {out_json_path}")
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
