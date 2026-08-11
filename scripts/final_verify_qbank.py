"""Verify and, when necessary, correct every question in a generated QBank."""

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError


CACHE_LOCK = threading.Lock()
REQUEST_TIMEOUT_SECONDS = 180
REQUEST_TIMEOUT_MS = REQUEST_TIMEOUT_SECONDS * 1000
MAX_VERIFICATION_ATTEMPTS = 4


class QbankRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant: str
    exam: str
    images: str
    rating: str
    subject: str
    topic: str
    question: str
    question_no: int
    page_no: int
    answer_page_no: int
    index: int
    optionA: str
    optionB: str
    optionC: str
    optionD: str
    optionE: str
    correct_option: str
    correct_option_text: str
    explanation: str
    plan: str
    duration: str
    ext_links: str
    explanation_A: str
    explanation_B: str
    explanation_C: str
    explanation_D: str
    creator_id: str
    creator_name: str
    tags: str


class VerificationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["ok", "corrected"]
    data: dict[str, Any] | None = None


VERIFICATION_ADAPTER = TypeAdapter(VerificationResponse)
INT_FIELDS = ("question_no", "page_no", "answer_page_no", "index")


SYSTEM_PROMPT = """You verify one question object from an exam question bank.
Check all of these:
1. The question is coherent and independently understandable.
2. correct_option is actually the correct answer among optionA-optionE.
3. correct_option_text exactly matches the selected option's text.

Return status 'ok' only when no change is needed. If anything is wrong, return
status 'corrected' and put the COMPLETE corrected question object in data.
Preserve every original field and value that does not need correction. Never
drop metadata fields. Preserve Markdown in question, options, correct_option_text,
and explanation exactly where it is valid; do not convert Markdown to plain text.
Use the original option labels (A-E, case-insensitive) and make corrections only
when supported by the question and its choices. Do not add commentary outside JSON.
"""


def _gemini_row_schema() -> types.Schema:
    row_properties = {
        "tenant": types.Schema(type=types.Type.STRING),
        "exam": types.Schema(type=types.Type.STRING),
        "images": types.Schema(type=types.Type.STRING),
        "rating": types.Schema(type=types.Type.STRING),
        "subject": types.Schema(type=types.Type.STRING),
        "topic": types.Schema(type=types.Type.STRING),
        "question": types.Schema(type=types.Type.STRING),
        "question_no": types.Schema(type=types.Type.INTEGER),
        "page_no": types.Schema(type=types.Type.INTEGER),
        "answer_page_no": types.Schema(type=types.Type.INTEGER),
        "index": types.Schema(type=types.Type.INTEGER),
        "optionA": types.Schema(type=types.Type.STRING),
        "optionB": types.Schema(type=types.Type.STRING),
        "optionC": types.Schema(type=types.Type.STRING),
        "optionD": types.Schema(type=types.Type.STRING),
        "optionE": types.Schema(type=types.Type.STRING),
        "correct_option": types.Schema(type=types.Type.STRING),
        "correct_option_text": types.Schema(type=types.Type.STRING),
        "explanation": types.Schema(type=types.Type.STRING),
        "plan": types.Schema(type=types.Type.STRING),
        "duration": types.Schema(type=types.Type.STRING),
        "ext_links": types.Schema(type=types.Type.STRING),
        "explanation_A": types.Schema(type=types.Type.STRING),
        "explanation_B": types.Schema(type=types.Type.STRING),
        "explanation_C": types.Schema(type=types.Type.STRING),
        "explanation_D": types.Schema(type=types.Type.STRING),
        "creator_id": types.Schema(type=types.Type.STRING),
        "creator_name": types.Schema(type=types.Type.STRING),
        "tags": types.Schema(type=types.Type.STRING),
    }
    return types.Schema(
        type=types.Type.OBJECT,
        properties=row_properties,
        required=list(row_properties.keys()),
        propertyOrdering=list(row_properties.keys()),
    )


def _gemini_verification_schema() -> types.Schema:
    return types.Schema(
        type=types.Type.OBJECT,
        properties={
            "status": types.Schema(type=types.Type.STRING, enum=["ok", "corrected"]),
            "data": types.Schema(type=types.Type.OBJECT, properties=_gemini_row_schema().properties),
        },
        required=["status"],
        propertyOrdering=["status", "data"],
    )


def qbank_path_for(pdf_path: Path) -> Path:
    return pdf_path.with_name(f"{pdf_path.stem}_qbank.json")


def final_qbank_path_for(pdf_path: Path) -> Path:
    return pdf_path.with_name(f"{pdf_path.stem}_qbank_final.json")


def cache_path_for(pdf_path: Path) -> Path:
    return pdf_path.parent / "corrections" / "cache.json"


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}: {exc}") from exc


def write_cache(path: Path, cache: dict[str, Any]) -> None:
    start_time = time.time()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    end_time = time.time()
    # print(f"Cache written to {path} in {end_time - start_time:.2f} seconds")
    temporary.replace(path)


def write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def cache_key(question: dict[str, Any]) -> str:
    chapter = str(question.get("chaptername") or question.get("chapter_name") or question.get("topic") or "").strip()
    number = str(question.get("question_no", "")).strip()
    return f"{chapter}+{number}"


def _openai_schema() -> dict[str, Any]:
    return VERIFICATION_ADAPTER.json_schema()


def _build_openai_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("BIFROST_API_KEY") or os.getenv("OPENAI_API_KEY") or "YOUR_BIFROST_API_KEY"
    return OpenAI(
        base_url="https://ai.semibit.in/openai/v1",
        api_key=api_key,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def _build_genai_client() -> genai.Client:
    load_dotenv()
    raw_keys = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not raw_keys:
        raise RuntimeError("GEMINI_API_KEY not set — add it to your .env file")
    keys = [key.strip() for key in raw_keys.split(",") if key.strip()]
    if not keys:
        raise RuntimeError("No valid keys found in GEMINI_API_KEY")
    return genai.Client(
        api_key=random.choice(keys),
        # google-genai expects HttpOptions.timeout in milliseconds.
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
    )


def _request_verification_openai(client: OpenAI, question: dict[str, Any], prompt: str) -> str:
    response = client.chat.completions.create(
        model="gemini/gemma-4-31b-it",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt + "\n" + json.dumps(question, ensure_ascii=False)},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "question_verification",
                "schema": _openai_schema(),
                "strict": True,
            },
        },
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("LLM returned an empty response")
    return content.strip()


def _request_verification_genai(client: genai.Client, question: dict[str, Any], prompt: str) -> str:
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=prompt + "\n" + json.dumps(question, ensure_ascii=False)
                )
            ],
        )
    ]
    config = types.GenerateContentConfig(
        tools=[],
        system_instruction=[
            types.Part.from_text(text=SYSTEM_PROMPT)
        ],
        response_mime_type="application/json",
        response_schema=_gemini_verification_schema(),
    )
    chunks: list[str] = []
    print(f"Calling Gemini API for question {cache_key(question)}...")
    for chunk in client.models.generate_content_stream(
        model="gemma-4-31b-it",
        contents=contents,
        config=config,
    ):
        if text := getattr(chunk, "text", None):
            chunks.append(text)
    content = "".join(chunks).strip()
    if not content:
        raise ValueError("LLM returned an empty response")
    return content


def _strip_wrappers(content: str) -> str:
    # Some compatible gateways still wrap schema-constrained JSON in ```json.
    # Remove only an outer fence; the JSON contents (including Markdown strings)
    # remain untouched.
    content = content.strip()
    if content.startswith("```"):
        first_newline = content.find("\n")
        if first_newline != -1:
            content = content[first_newline + 1:]
        if content.rstrip().endswith("```"):
            content = content.rstrip()[:-3].rstrip()
    # Also tolerate a gateway that emits only a trailing fence or a short
    # introductory line around the JSON object.
    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end >= start:
        content = content[start:end + 1]
    return content


def _parse_verification(content: str) -> dict[str, Any]:
    normalized = _strip_wrappers(content)
    result = VERIFICATION_ADAPTER.validate_json(normalized)
    result = result.model_dump(exclude_none=True)
    if result.get("status") == "corrected" and isinstance(result.get("data"), dict):
        result["data"] = dict(result["data"])
    return result


def _normalize_corrected(question: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") != "corrected":
        return result
    merged = dict(question)
    if isinstance(result.get("data"), dict):
        merged.update(result["data"])
    validation_input = dict(merged)
    # Some source rows use blank strings for page bookkeeping fields. Keep those
    # original blanks, but validate the rest of the corrected payload normally.
    for field in INT_FIELDS:
        if validation_input.get(field) == "":
            validation_input[field] = 0
    validated = QbankRow.model_validate(validation_input)
    normalized = validated.model_dump()
    for field in INT_FIELDS:
        if merged.get(field) == "":
            normalized[field] = ""
    result["data"] = normalized
    return result


def _repair_missing_data(provider: str, client: Any, question: dict[str, Any], content: str) -> dict[str, Any]:
    prompt = (
        "You returned a corrected verification result without the required data field.\n"
        "Return only valid JSON matching the schema.\n"
        "Original question object:\n"
        + json.dumps(question, ensure_ascii=False)
        + "\nPrevious response:\n"
        + content
    )
    repaired = _request_verification(provider, client, question, prompt)
    return _parse_verification(repaired)


def _request_verification(provider: str, client: Any, question: dict[str, Any], prompt: str) -> str:
    if provider == "openai":
        return _request_verification_openai(client, question, prompt)
    return _request_verification_genai(client, question, prompt)


def _is_retryable_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {429, 500, 502, 503, 504}:
        return True
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "high demand",
            "rate limit",
            "temporary",
            "unavailable",
            "timeout",
            "timed out",
            "connection",
        )
    )


def verify_with_llm(question: dict[str, Any], use_openai: bool = False) -> dict[str, Any]:
    provider = "openai" if use_openai else "genai"
    client = _build_openai_client() if use_openai else _build_genai_client()
    prompt = "Verify this complete JSON question object:"
    content = ""
    last_exc: Exception | None = None
    for attempt in range(MAX_VERIFICATION_ATTEMPTS):
        attempt_no = attempt + 1
        try:
            print(
                f"[verify:{provider}] Attempt {attempt_no}/{MAX_VERIFICATION_ATTEMPTS} "
                f"for {cache_key(question)}"
            )
            content = _request_verification(provider, client, question, prompt)
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
            print(
                f"[verify:{provider}] Error on attempt {attempt_no}/"
                f"{MAX_VERIFICATION_ATTEMPTS} for {cache_key(question)}: {exc}"
            )
            if attempt == MAX_VERIFICATION_ATTEMPTS - 1 or not _is_retryable_error(exc):
                raise
            delay = 2 ** attempt
            print(
                f"[verify:{provider}] Retrying in {delay}s"
            )
            time.sleep(delay)
    if last_exc is not None:
        raise last_exc
    try:
        parsed = _parse_verification(content)
    except ValidationError:
        try:
            parsed = json.loads(_strip_wrappers(content))
        except json.JSONDecodeError:
            raise
        if isinstance(parsed, dict) and parsed.get("status") == "corrected" and "data" not in parsed:
            return _normalize_corrected(question, _repair_missing_data(provider, client, question, content))
        raise
    if parsed.get("status") == "corrected" and not isinstance(parsed.get("data"), dict):
        return _normalize_corrected(question, _repair_missing_data(provider, client, question, content))
    return _normalize_corrected(question, parsed)


def main(pdf_arg: str, parallel: int, use_openai: bool = False) -> None:
    print(f"Verifying QBank for PDF: {pdf_arg} | parallel: {parallel} | use_openai: {use_openai}")
    pdf_path = Path(pdf_arg).expanduser().resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    qbank_path = qbank_path_for(pdf_path)
    if not qbank_path.is_file():
        raise FileNotFoundError(f"Corresponding QBank not found: {qbank_path}")
    if parallel < 1:
        raise ValueError("--parallel must be at least 1")

    questions = load_json(qbank_path, [])
    if not isinstance(questions, list) or not all(isinstance(q, dict) for q in questions):
        raise ValueError(f"Expected a JSON array of question objects in {qbank_path}")
    cache_path = cache_path_for(pdf_path)
    with CACHE_LOCK:
        cache = load_json(cache_path, {})
        if not isinstance(cache, dict):
            raise ValueError(f"Expected cache object in {cache_path}")

    pending: list[tuple[str, dict[str, Any]]] = []
    results: dict[str, dict[str, Any]] = {}
    for question in questions:
        key = cache_key(question)
        cached = cache.get(key)
        if isinstance(cached, dict) and cached.get("status") in {"ok", "corrected"}:
            results[key] = cached
        else:
            if isinstance(cached, dict) and cached.get("status") == "failed":
                cache.pop(key, None)
                write_cache(cache_path, cache)
            pending.append((key, question))

    def worker(item: tuple[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        key, question = item
        return key, verify_with_llm(question, use_openai=use_openai)

    print(f"Questions: {len(questions)} | cached: {len(results)} | to verify: {len(pending)}")
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        future_to_item = {pool.submit(worker, item): item for item in pending}
        for future in as_completed(future_to_item):
            try:
                key, result = future.result()
            except Exception as exc:
                key, question = future_to_item[future]
                err_result = {"status": "failed", "error": str(exc)}
                results[key] = err_result
                with CACHE_LOCK:
                    cache[key] = err_result
                    write_cache(cache_path, cache)
                print(f"[failed] {key}: {exc}")
                continue
            results[key] = result
            with CACHE_LOCK:
                cache[key] = result
                write_cache(cache_path, cache)
            print(f"[{result['status']}] {key}")
    print(f"Verification complete. Cache: {cache_path}")
    final_questions: list[dict[str, Any]] = []
    for question in questions:
        key = cache_key(question)
        result = results.get(key, {"status": "failed"})
        if result["status"] == "corrected" and isinstance(result.get("data"), dict):
            corrected = dict(question)
            corrected.update(result["data"])
            final_questions.append(corrected)
        else:
            final_questions.append(dict(question))
    final_path = final_qbank_path_for(pdf_path)
    write_json_atomic(final_path, final_questions)
    print(f"Done. Cache: {cache_path}")
    print(f"Final QBank: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify a QBank question-by-question with an LLM.")
    parser.add_argument("pdf_path", help="Path to the source PDF")
    parser.add_argument("--parallel", type=int, default=10, help="Maximum concurrent LLM calls (default: 10)")
    parser.add_argument("--openai", action="store_true", help="Use the OpenAI-compatible endpoint instead of Gemini")
    args = parser.parse_args()
    main(args.pdf_path, args.parallel, use_openai=args.openai)
