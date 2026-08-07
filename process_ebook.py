import os
import sys
import json
import pathlib
import argparse
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

# Allow importing extract_page from the same directory
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from extract_page_to_json import extract_pdf_page, generate_page_json
import time

# ── Locks ─────────────────────────────────────────────────────────────────────
_csv_lock   = threading.Lock()
_index_lock = threading.Lock()


# ── Multi-line slot display ───────────────────────────────────────────────────

class SlotDisplay:
    """Reserves N terminal lines — one per parallel worker — and updates each
    in-place using ANSI escape codes so lines never collide."""

    def __init__(self, n_slots: int):
        self.n = n_slots
        self._lock = threading.Lock()
        # Reserve lines up front
        sys.stdout.write("\n" * n_slots)
        sys.stdout.flush()

    def update(self, slot: int, text: str):
        """Overwrite the line reserved for `slot` (0-indexed from top)."""
        with self._lock:
            up = self.n - slot
            # move up → clear line → write → move back down
            sys.stdout.write(f"\x1b[{up}A\r\x1b[2K{text}\x1b[{up}B\r")
            sys.stdout.flush()

    def println(self, text: str):
        """Print a persistent line ABOVE the slot area (e.g. summary lines)."""
        with self._lock:
            # move above all slots, insert line, return
            sys.stdout.write(f"\x1b[{self.n}A\x1b[L{text}\n\x1b[{self.n}B\r")
            sys.stdout.flush()

    def teardown(self):
        """Move cursor past the reserved area so the shell prompt appears below."""
        sys.stdout.write("\n")
        sys.stdout.flush()


# ── Index helpers ─────────────────────────────────────────────────────────────

def load_index(index_path: str) -> dict:
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": [], "failed": []}


def _write_index(index_path: str, index: dict):
    """Write index to disk — caller must hold _index_lock."""
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


# ── Page worker ───────────────────────────────────────────────────────────────

def process_page(
    pdf_path:      str,
    page_num:      int,
    questions_csv: str,
    answers_csv:   str,
    chapters_csv:  str,
    raw_csv:       str,
    index_path:    str,
    index:         dict,
    debug:         bool,
    parallel:      int,
    slot:          int,
    display:       SlotDisplay,
) -> tuple[int, bool]:

    tag = f"[page {page_num:>4}]"

    def _set(text: str):
        display.update(slot, f"{tag} {text}")

    _elapsed = [0.0]  # mutable container to capture final elapsed

    def progress_cb(chars, cps, elapsed, done=False):
        _elapsed[0] = elapsed
        _set(f"{chars} chars | {cps:.1f} ch/s | {elapsed:.1f}s")

    _set("starting…")
    temp_pdf_path = None
    try:
        temp_pdf_path = extract_pdf_page(pdf_path, page_num)

        use_debug = debug and parallel == 1  # stream only when single

        t0 = time.perf_counter()
        json_text = generate_page_json(temp_pdf_path, page_num, debug=use_debug)
        _elapsed[0] = time.perf_counter() - t0

        # Save unified JSON to pages/page_{n}.json
        pages_dir = pathlib.Path(pdf_path).parent / "pages"
        pages_dir.mkdir(exist_ok=True)
        out_json_path = pages_dir / f"page_{page_num}.json"

        json_text = json_text.strip()
        if json_text.startswith("```json"):
            json_text = json_text[7:]
        if json_text.startswith("```"):
            json_text = json_text[3:]
        if json_text.endswith("```"):
            json_text = json_text[:-3]

        try:
            data = json.loads(json_text)
            data["page"] = page_num
            with _csv_lock:
                with open(out_json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)
        except json.JSONDecodeError:
            with _csv_lock:
                with open(out_json_path, "w", encoding="utf-8") as f:
                    f.write(json_text)

        with _index_lock:
            if page_num not in index["completed"]:
                index["completed"].append(page_num)
            _write_index(index_path, index)

        _set(f"[OK] done in {_elapsed[0]:.1f}s")
        return page_num, True

    except Exception as exc:
        err_msg = str(exc)
        _set(f"[ERR] failed - {err_msg}")

        with _index_lock:
            failed_pages = [e["page"] for e in index.get("failed", [])]
            if page_num not in failed_pages:
                index["failed"].append({"page": page_num, "error": err_msg})
            _write_index(index_path, index)

        return page_num, False

    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)


# ── PDF page count ────────────────────────────────────────────────────────────

def get_total_pages(pdf_path: str) -> int:
    from pdf2image import pdfinfo_from_path
    info = pdfinfo_from_path(pdf_path)
    return info["Pages"]


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a PDF ebook page-by-page using extract_page.py"
    )
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument(
        "--start", type=int, default=1,
        help="First page to process (1-indexed, default: 1)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of pages to process (default: all remaining)",
    )
    parser.add_argument(
        "--parallel", type=int, default=1,
        help="Number of concurrent workers (default: 1)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Stream raw model output — only clean with --parallel 1",
    )
    args = parser.parse_args()

    pdf_path = pathlib.Path(args.pdf).resolve()
    pdf_dir  = pdf_path.parent
    pdf_stem = pdf_path.stem

    questions_csv = str(pdf_dir / f"{pdf_stem}_questions.csv")
    answers_csv   = str(pdf_dir / f"{pdf_stem}_answers.csv")
    chapters_csv  = str(pdf_dir / f"{pdf_stem}_chapters.csv")
    raw_csv       = str(pdf_dir / f"{pdf_stem}_raw.csv")
    index_path    = str(pdf_dir / f"{pdf_stem}_index.json")

    # ── Load index & determine page range ─────────────────────────────────────
    index         = load_index(index_path)
    
    # Ensure index contains file paths
    index["files"] = {
        "original_pdf": str(pdf_path),
        "questions_csv": questions_csv,
        "answers_csv": answers_csv,
        "chapters_csv": chapters_csv,
        "raw_csv": raw_csv,
    }
    _write_index(index_path, index)

    completed_set = set(index.get("completed", []))
    failed_set    = {e["page"] for e in index.get("failed", [])}

    total_pages = get_total_pages(str(pdf_path))
    end_page    = min(
        args.start + args.limit - 1 if args.limit else total_pages,
        total_pages,
    )

    pages_to_process = [
        p for p in range(args.start, end_page + 1)
        if p not in completed_set
    ]
    skipped = (end_page - args.start + 1) - len(pages_to_process)

    print(f"[process_ebook] PDF      : {pdf_path.name}")
    print(f"[process_ebook] Pages    : {args.start}–{end_page}  ({total_pages} total in PDF)")
    print(f"[process_ebook] To do    : {len(pages_to_process)}  |  Skipped (done): {skipped}  |  Previously failed: {len(failed_set)}")
    print(f"[process_ebook] Questions -> {questions_csv}")
    print(f"[process_ebook] Answers   -> {answers_csv}")
    print(f"[process_ebook] Chapters  -> {chapters_csv}")
    print(f"[process_ebook] Raw audit -> {raw_csv}")

    print(f"[process_ebook] Index     : {index_path}")
    print(f"[process_ebook] Workers  : {args.parallel}")

    if not pages_to_process:
        print("\n[process_ebook] Nothing left to do. All pages already completed.")
        sys.exit(0)

    # ── Slot pool — each worker borrows a slot index while it runs ────────────
    slot_pool = Queue()
    for i in range(args.parallel):
        slot_pool.put(i)

    display = SlotDisplay(args.parallel)

    done_count   = 0
    failed_count = 0

    def run_page(page_num: int) -> tuple[int, bool]:
        slot = slot_pool.get()
        try:
            return process_page(
                str(pdf_path), page_num, questions_csv, answers_csv, chapters_csv, raw_csv,
                index_path, index, args.debug, args.parallel, slot, display,
            )
        finally:
            slot_pool.put(slot)

    try:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(run_page, p): p for p in pages_to_process}
            for future in as_completed(futures):
                _, success = future.result()
                if success:
                    done_count += 1
                else:
                    failed_count += 1
    finally:
        display.teardown()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n[process_ebook] Finished - [OK] {done_count} succeeded  [ERR] {failed_count} failed")
    if index["failed"]:
        print(f"[process_ebook] Failed pages :")
        for entry in sorted(index["failed"], key=lambda e: e["page"]):
            print(f"                  page {entry['page']:>4} — {entry['error']}")
        print(f"[process_ebook] Re-run same command to retry failed pages.")
    print(f"[process_ebook] Questions -> {questions_csv}")
    print(f"[process_ebook] Answers   -> {answers_csv}")
    print(f"[process_ebook] Chapters  -> {chapters_csv}")
    print(f"[process_ebook] Raw audit -> {raw_csv}")

    # ── Final Assembly ────────────────────────────────────────────────────────
    print(f"\n[process_ebook] Assigning chapters to CSVs...")
    import subprocess
    try:
        # Run per-page JSON extraction using extract_page_to_json.py for completed pages.
        # Skip pages that already have a JSON file in the pages/ directory.
        completed_pages = sorted(index.get("completed", []))
        pages_dir = pdf_dir / "pages"
        pages_dir.mkdir(exist_ok=True)
        for p in completed_pages:
            out_json = pages_dir / f"page_{p}.json"
            if out_json.exists():
                continue
            subprocess.run(
                ["python", "extract_page_to_json.py", str(pdf_path), "--start", str(p)],
                check=True
            )
    except subprocess.CalledProcessError as e:
        print(f"[process_ebook] ✗ Failed to extract pages to JSON: {e}")
