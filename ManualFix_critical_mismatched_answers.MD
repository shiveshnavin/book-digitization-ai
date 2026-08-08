# Manual Fix: Critical Answer Mismatches

Subject PDF:

`<PATH_TO_SUBJECT_DIRECTORY>/<BOOK>.pdf`

Follow these steps:

1. Create a backup before editing anything:

   ```bash
   python backup_pages.py <PATH_TO_SUBJECT_DIRECTORY>
   ```

2. Generate focused instructions:

   ```bash
   python analyze_mismatches.py <PATH_TO_SUBJECT_DIRECTORY>/<BOOK>.pdf
   ```

3. Process one printed mismatch at a time. Use the exact:

   - `pages/page_<N>.json` file paths
   - line ranges
   - question number
   - question page
   - answer page
   - adjacent-page guidance

4. First inspect the indicated JSON ranges. If the JSON is unclear, inspect the PDF page numbers printed by the analyzer.

5. Edit only the exact `pages/page_<N>.json` record where the content starts:

   - question content: edit the question-page record
   - answer content: edit the answer-page record

6. If content spills onto the adjacent page, copy the continuation into the starting record so it becomes complete. Then set the copied continuation fields on the adjacent-page record to empty strings (`""`). Do not delete the continuation record.

7. Make only surgical corrections to fields proven by the PDF, such as:

   - `question`
   - `optionA`–`optionE`
   - `correct_option`
   - `correct_option_text`
   - `explanation`
   - `question_no`, only when clearly wrong

8. Preserve record order, formatting, page, position, chapter, tags, and unrelated data. Do not edit generated qbank, CSV, combined pages, or index files.

9. Never guess. If the PDF is unclear, leave the record unchanged and report it as unresolved.

10. After all edits, regenerate the qbank and validation files:

    ```bash
    python process_ebook.py "<PATH_TO_SUBJECT_DIRECTORY>/<BOOK>.pdf" --parallel 5
    ```

    Replace the PDF path with the actual subject PDF. For the English book, use:

    ```bash
    python process_ebook.py "/home/ubuntu/python/book-digitization-ai/qbank/english/ebp_p8_to_end.pdf" --parallel 5
    ```

11. Reopen the regenerated `<BOOK>_index.json` and check `critical_mismatched_answers` again. If entries remain, repeat the focused analysis and manual fixes. Do not manually remove entries from the index.

12. Report each changed index, source file, changed fields, old value, new value, and PDF justification.
