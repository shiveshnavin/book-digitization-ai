# Manual Fix Prompt: Critical Mismatched Answers

You are correcting extracted question-and-answer data.

Subject directory:
`<PATH_TO_SUBJECT_DIRECTORY>`

PDF:
`<PDF_FILENAME>`

The subject's index file contains:

```json
"critical_mismatched_answers": [
  {
    "index": <zero_based_qbank_index>,
    "question_no": <question_number>,
    "page_no": <source_pdf_page>
  }
]
```

For every object in `critical_mismatched_answers`:

0. Before touching any source file, create a backup of the subject's `pages/` directory:

   ```bash
   python backup_pages.py <PATH_TO_SUBJECT_DIRECTORY>
   ```

   This creates or replaces:

   `<PATH_TO_SUBJECT_DIRECTORY>/pages.zip`

   Confirm that the backup completes successfully before making any edits.

1. Open the corresponding source file:

   `<PATH_TO_SUBJECT_DIRECTORY>/pages/page_<page_no>.json`

2. Locate the question using its `question_no`, page number, and nearby record position.

3. Inspect the original PDF page `<page_no>`, plus the adjacent one or two PDF pages if necessary. The PDF is authoritative.

4. Identify the exact question text, all answer options, correct answer, correct option letter, and explanation if present.

5. Check nearby page JSON records for split content:

   - `question_no: -1` may be a continuation of the question.
   - An answer with `question_no: -1` may be a continuation of the answer.
   - Preserve and merge split text correctly where required.

6. Make surgical edits only to the affected source files:

   `<PATH_TO_SUBJECT_DIRECTORY>/pages/page_<N>.json`

   Do not regenerate the page files. Do not edit:

   - `*_qbank.json`
   - `*_qbank.csv`
   - `*_pages.json`
   - `*_index.json`
   - CSV extraction files

7. Preserve JSON formatting, record order, page and position values, chapter name, exam tags, unrelated questions and answers, and all fields that are already correct.

8. Only change fields proven incorrect from the PDF. Typical fields include:

   - `question`
   - `optionA`–`optionE`
   - `correct_option`
   - `correct_option_text`
   - `explanation`
   - `question_no`, only if the PDF clearly proves it is wrong

9. After each edit, verify:

   - `correct_option` is one of `a`, `b`, `c`, `d`, or `e`.
   - `correct_option_text` matches one of the options, ignoring case and surrounding whitespace.
   - The question and answer have the correct question number and chapter.
   - No unrelated records changed.

10. At the end, report a table containing:

    - index
    - question number
    - page number
    - source file edited
    - fields changed
    - old value
    - new value
    - brief PDF-based justification

Important:

- Work one mismatch at a time.
- Never guess an answer.
- If the PDF is unclear, leave the record unchanged and report it as unresolved.
- Do not remove a mismatch from the index manually. The index and qbank will be regenerated later from the edited page files.
