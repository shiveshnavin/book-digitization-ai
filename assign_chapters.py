import os
import sys
import json
import csv

def load_chapters(chapters_csv: str) -> dict:
    """Returns a dict mapping physical_page (int) to a list of chapter names."""
    page_to_chapters = {}
    if not os.path.exists(chapters_csv):
        return page_to_chapters
    
    with open(chapters_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                p_page = int(row['physical_page'])
                c_name = row['chapter_name'].strip()
                if c_name:
                    if p_page not in page_to_chapters:
                        page_to_chapters[p_page] = []
                    page_to_chapters[p_page].append(c_name)
            except (ValueError, KeyError):
                continue
    return page_to_chapters

def assign_chapters_to_file(input_csv: str, chapters_map: dict):
    if not os.path.exists(input_csv):
        print(f"File not found: {input_csv}")
        return

    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader, None)
        if not headers:
            return
        
        rows = list(reader)

    # find indexes
    try:
        type_idx = headers.index('type')
        p_page_idx = headers.index('physical_page')
        q_no_idx = headers.index('question_no')
    except ValueError as e:
        print(f"Missing required column in {input_csv}: {e}")
        return

    # Update headers
    new_headers = headers.copy()
    if 'chapter_name' in new_headers:
        # Already has it, remove to insert at correct spot
        new_headers.remove('chapter_name')
    
    insert_idx = type_idx + 1
    new_headers.insert(insert_idx, 'chapter_name')

    active_chapter = ""
    prev_q_no = 0
    current_p_page = -1
    chapter_idx_on_page = 0
    new_rows = []

    for row in rows:
        try:
            p_page = int(row[p_page_idx])
        except (ValueError, IndexError):
            p_page = 0
            
        try:
            q_no = int(row[q_no_idx])
        except (ValueError, IndexError):
            q_no = 0
            
        # Check boundary
        if p_page in chapters_map:
            page_chapters = chapters_map[p_page]
            
            # Reset page chapter index if we moved to a new page
            if p_page != current_p_page:
                current_p_page = p_page
                chapter_idx_on_page = 0
            
            # Boundary detection: exactly 1, or drop in number
            if q_no == 1 or (q_no > 0 and prev_q_no > 0 and q_no < prev_q_no):
                # Consume the next chapter on this page
                if chapter_idx_on_page < len(page_chapters):
                    active_chapter = page_chapters[chapter_idx_on_page]
                    chapter_idx_on_page += 1
        
        # Keep track of previous valid question_no
        prev_q_no = q_no if q_no > 0 else prev_q_no

        # Construct new row
        row_dict = dict(zip(headers, row))
        # Remove old chapter_name if it was there
        if 'chapter_name' in row_dict:
            del row_dict['chapter_name']
            
        row_dict['chapter_name'] = active_chapter
        
        new_row = []
        for h in new_headers:
            new_row.append(row_dict.get(h, ""))
        new_rows.append(new_row)

    with open(input_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(new_headers)
        writer.writerows(new_rows)
        
    print(f"[assign_chapters] Processed {len(new_rows)} rows in {os.path.basename(input_csv)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python assign_chapters.py <index_json_path>")
        sys.exit(1)
        
    index_path = sys.argv[1]
    with open(index_path, 'r', encoding='utf-8') as f:
        index_data = json.load(f)
        
    files = index_data.get("files", {})
    q_csv = files.get("questions_csv")
    a_csv = files.get("answers_csv")
    c_csv = files.get("chapters_csv")
    
    if not (q_csv and a_csv and c_csv):
        print("Could not find CSV paths in index.json")
        sys.exit(1)
        
    print(f"[assign_chapters] Loading chapters from {os.path.basename(c_csv)}")
    chapters_map = load_chapters(c_csv)
    
    print(f"[assign_chapters] Assigning chapters to {os.path.basename(q_csv)}")
    assign_chapters_to_file(q_csv, chapters_map)
    
    print(f"[assign_chapters] Assigning chapters to {os.path.basename(a_csv)}")
    assign_chapters_to_file(a_csv, chapters_map)
    
    print("[assign_chapters] Done.")
