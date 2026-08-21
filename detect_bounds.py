import os
import argparse
import cv2
import json
import fitz  # PyMuPDF
import pytesseract
import concurrent.futures
import uuid
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from pdf2image import convert_from_path

# ============================================================
# CONFIG / FEATURE FLAGS
# ============================================================
DPI = 200
SCALE = DPI / 72.0
CONF_THRESHOLD = 0.2

FORCE_OCR = True            # True = always OCR (use for scanned/vector-glyph PDFs like this one)
VISUALIZE = True            # master switch for drawing boxes/labels
SAVE_ANNOTATED_IMAGES = True

FONT_SCALE = 0.35
LINE_THICKNESS = 1
BOX_COLOR = (0, 0, 255)       # red (BGR)
TEXT_COLOR = (255, 255, 255)

# Point this to your local Tesseract install (Update this if you are on Linux/Mac)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

TYPE_MAP = {
    "title": "title",
    "plain text": "paragraph",
    "figure": "figure",
    "figure_caption": "caption",
    "table": "table",
    "table_caption": "caption",
    "formula": "formula",
    "table_footnote": "footnote",
    "abandon": "abandon",
}

# ============================================================
# Text Extraction Helpers
# ============================================================
def extract_text_from_pdf(pdf_page, bbox_px, scale):
    x1, y1, x2, y2 = bbox_px
    rect = fitz.Rect(x1 / scale, y1 / scale, x2 / scale, y2 / scale)
    text = pdf_page.get_text("text", clip=rect).strip()
    return text.replace("\n", " ")

def extract_text_via_ocr(img_cv, bbox_px, padding=2):
    x1, y1, x2, y2 = map(int, bbox_px)
    h, w = img_cv.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    crop = img_cv[y1:y2, x1:x2]
    if crop.size == 0:
        return ""

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    text = pytesseract.image_to_string(gray, config="--psm 6")
    return text.strip().replace("\n", " ")

def extract_text(pdf_page, img_cv, bbox_px, scale, force_ocr=False):
    if not force_ocr:
        text = extract_text_from_pdf(pdf_page, bbox_px, scale)
        if text:
            return text
    return extract_text_via_ocr(img_cv, bbox_px)

# ============================================================
# Layout Ordering Helpers
# ============================================================
def assign_reading_order(elements, page_width_px):
    mid_x = page_width_px / 2
    left_col, right_col = [], []
    for el in elements:
        x1, y1, x2, y2 = el["_bbox_px"]
        box_center_x = (x1 + x2) / 2
        (left_col if box_center_x < mid_x else right_col).append(el)
    left_col.sort(key=lambda e: e["_bbox_px"][1])
    right_col.sort(key=lambda e: e["_bbox_px"][1])
    return left_col + right_col

def touches_bottom_margin(bbox_px, page_height_px, margin_px=40):
    _, _, _, y2 = bbox_px
    return (page_height_px - y2) < margin_px

def touches_top_margin(bbox_px, margin_px=40):
    _, y1, _, _ = bbox_px
    return y1 < margin_px

def draw_annotations(img_cv, page_elements):
    annotated = img_cv.copy()
    for el in page_elements:
        x1, y1, x2, y2 = map(int, el["_bbox_px"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, LINE_THICKNESS)

        label_text = f"{el['type']} {el.get('confidence', 0):.2f}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, LINE_THICKNESS)
        cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw + 2, y1), BOX_COLOR, -1)
        cv2.putText(annotated, label_text, (x1 + 1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SCALE, TEXT_COLOR, LINE_THICKNESS, cv2.LINE_AA)
    return annotated

# ============================================================
# Overlap Filtering Helpers
# ============================================================
def get_intersection_over_smaller_area(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0, None

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    if area1 < area2:
        return intersection_area / area1, 1
    else:
        return intersection_area / area2, 2

def filter_overlapping_boxes(boxes_data, overlap_threshold=0.8):
    keep_indices = set(range(len(boxes_data)))
    for i in range(len(boxes_data)):
        for j in range(i + 1, len(boxes_data)):
            if i not in keep_indices or j not in keep_indices:
                continue
            box1 = boxes_data[i]['bbox_px']
            box2 = boxes_data[j]['bbox_px']
            overlap_ratio, smaller_idx_indicator = get_intersection_over_smaller_area(box1, box2)
            if overlap_ratio > overlap_threshold:
                if smaller_idx_indicator == 1:
                    keep_indices.discard(i)
                elif smaller_idx_indicator == 2:
                    keep_indices.discard(j)
    return [boxes_data[i] for i in sorted(list(keep_indices))]

# ============================================================
# Worker Function for Parallel Processing
# ============================================================
def process_page_worker(page_num, image, pdf_path, model, output_dir, images_dir):
    """Processes a single page. Designed to run in a thread."""
    local_doc = fitz.open(pdf_path)
    pdf_page = local_doc[page_num]

    temp_raw_path = os.path.join(output_dir, f"temp_raw_page_{page_num}.png")
    image.save(temp_raw_path)
    img_cv = cv2.imread(temp_raw_path)
    page_h, page_w = img_cv.shape[:2]

    results = model.predict(temp_raw_path, imgsz=1024, conf=CONF_THRESHOLD, device="cpu", verbose=False)

    # 1. Gather all raw boxes
    raw_boxes_data = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        raw_label = model.names[cls_id]
        label = TYPE_MAP.get(raw_label, raw_label)
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        raw_boxes_data.append({
            "label": label,
            "conf": conf,
            "bbox_px": (x1, y1, x2, y2)
        })

    # 2. Filter out heavily overlapping smaller boxes
    filtered_boxes = filter_overlapping_boxes(raw_boxes_data, overlap_threshold=0.8)

    # 3. Extract text and crop the image for kept boxes
    page_elements = []
    for b_data in filtered_boxes:
        bbox_px = b_data["bbox_px"]
        label = b_data["label"]
        conf = b_data["conf"]

        text = extract_text(pdf_page, img_cv, bbox_px, SCALE, force_ocr=FORCE_OCR)
        
        # --- NEW CROP LOGIC ---
        cx1, cy1, cx2, cy2 = map(int, bbox_px)
        cx1, cy1 = max(0, cx1), max(0, cy1)
        cx2, cy2 = min(page_w, cx2), min(page_h, cy2)
        
        crop_img = img_cv[cy1:cy2, cx1:cx2]
        
        # Save temporary image slice
        tmp_img_path = None
        if crop_img.size > 0:
            tmp_filename = f"tmp_page{page_num}_{uuid.uuid4().hex}.jpg"
            tmp_img_path = os.path.join(images_dir, tmp_filename)
            cv2.imwrite(tmp_img_path, crop_img)

        page_elements.append({
            "type": label,
            "text": text,
            "page": page_num,
            "parent_chapter": -1,
            "outline": [str(round(v, 1)) for v in bbox_px],
            "is_chapter_title": label == "title",
            "rotation": "0.0",
            "continued": False,
            "page_merged_paragraph": None,
            "confidence": round(conf, 4),
            "_bbox_px": bbox_px,
            "_tmp_img_path": tmp_img_path  # Temporary reference for later renaming
        })

    page_elements = assign_reading_order(page_elements, page_w)

    if VISUALIZE:
        annotated = draw_annotations(img_cv, page_elements)
        if SAVE_ANNOTATED_IMAGES:
            annotated_path = os.path.join(output_dir, f"page_{page_num:03d}.png")
            cv2.imwrite(annotated_path, annotated)

    if os.path.exists(temp_raw_path):
        os.remove(temp_raw_path)

    local_doc.close()
    return page_num, page_elements, page_h

# ============================================================
# Main Execution
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Extract layout and text from a PDF.")
    parser.add_argument("pdf_path", help="Path to the input PDF file")
    parser.add_argument("--parallel", type=int, default=1, 
                        help="Number of pages to process in parallel (e.g., set to your CPU core count. Default is 1).")
    args = parser.parse_args()

    pdf_path = args.pdf_path
    if not os.path.exists(pdf_path):
        print(f"Error: File not found -> {pdf_path}")
        return

    # Generate Output Paths
    pdf_dir = os.path.dirname(os.path.abspath(pdf_path))
    pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
    
    output_dir = os.path.join(pdf_dir, f"{pdf_basename}_page_bounds")
    images_dir = os.path.join(pdf_dir, f"{pdf_basename}_images")
    json_output_path = os.path.join(pdf_dir, f"{pdf_basename}_detected.json")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    print(f"Output directories initialized:\n- {output_dir}\n- {images_dir}")

    # Load Model (Shared across threads)
    print("Loading YOLO model...")
    model_path = hf_hub_download(
        repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
        filename="doclayout_yolo_docstructbench_imgsz1024.pt"
    )
    model = YOLOv10(model_path)

    print(f"Loading PDF images: {pdf_path}")
    pages_img = convert_from_path(pdf_path, dpi=DPI)
    total_pages = len(pages_img)

    results_by_page = {}
    
    # ---------------------------------------------------------
    # Parallel Page Processing Phase
    # ---------------------------------------------------------
    print(f"\nProcessing {total_pages} pages using {args.parallel} thread(s)...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {
            executor.submit(process_page_worker, page_num, image, pdf_path, model, output_dir, images_dir): page_num
            for page_num, image in enumerate(pages_img)
        }

        for future in concurrent.futures.as_completed(futures):
            page_num = futures[future]
            try:
                p_num, page_elements, page_h = future.result()
                results_by_page[p_num] = (page_elements, page_h)
                print(f"Page {p_num:03d}/{total_pages}: Processed {len(page_elements)} elements.")
            except Exception as e:
                print(f"Error processing page {page_num}: {e}")

    # ---------------------------------------------------------
    # Sequential Reassembly Phase
    # ---------------------------------------------------------
    all_elements = []
    global_index = 1
    prev_page_last_paragraph = None
    prev_page_h = None

    for page_num in range(total_pages):
        if page_num not in results_by_page:
            continue
            
        page_elements, page_h = results_by_page[page_num]

        # Apply Cross-Page Paragraph Linking
        if prev_page_last_paragraph is not None and page_elements:
            first_el = page_elements[0]
            if (first_el["type"] == "paragraph"
                    and prev_page_last_paragraph["type"] == "paragraph"
                    and touches_bottom_margin(prev_page_last_paragraph["_bbox_px"], prev_page_h)
                    and touches_top_margin(first_el["_bbox_px"])):
                prev_page_last_paragraph["continued"] = True
                first_el["page_merged_paragraph"] = {
                    "page": prev_page_last_paragraph["page"],
                    "index": prev_page_last_paragraph.get("index"),
                }

        # Apply Global Index and Rename Image files
        for el in page_elements:
            el["index"] = global_index
            
            # Remove spaces in the type name so the filename is clean (e.g. 'plain_text' instead of 'plain text')
            safe_type = el['type'].replace(' ', '_')
            
            # Construct final filename: <page>_<index>_<type>.jpg
            final_img_filename = f"{el['page']}_{global_index}_{safe_type}.jpg"
            final_img_path = os.path.join(images_dir, final_img_filename)
            
            # Check if temp image exists, then rename it to final name
            tmp_path = el.get("_tmp_img_path")
            if tmp_path and os.path.exists(tmp_path):
                os.rename(tmp_path, final_img_path)
                # Store relative path so your JSON stays portable
                el["image"] = f"{pdf_basename}_images/{final_img_filename}"
            else:
                el["image"] = None

            global_index += 1

        paragraphs_on_page = [e for e in page_elements if e["type"] == "paragraph"]
        prev_page_last_paragraph = paragraphs_on_page[-1] if paragraphs_on_page else None
        prev_page_h = page_h

        all_elements.extend(page_elements)

    # Strip internal-only coordinate keys before saving JSON
    for el in all_elements:
        el.pop("_bbox_px", None)
        el.pop("_tmp_img_path", None)

    # Save JSON
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(all_elements, f, indent=4)

    print(f"\nDone! Processed {total_pages} pages.")
    print(f"Layout data saved to: {json_output_path}")
    print(f"Annotated bounds images saved in: {output_dir}")
    print(f"Extracted crop images saved in: {images_dir}")

if __name__ == "__main__":
    main()