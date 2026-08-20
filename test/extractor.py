import os
import re
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

# Allow high-resolution book scans without memory limits
Image.MAX_IMAGE_PIXELS = None

class MathPdfExtractor:
    def __init__(self, dpi=300, top_header_pct=0.035):
        """
        :param dpi: Rendering DPI for high-resolution graphics (300 DPI gives print-quality crops)
        :param top_header_pct: Top page margin fraction to ignore (header titles / page numbers)
        """
        self.dpi = dpi
        self.top_header_pct = top_header_pct

    def find_visual_dividers(self, img_pil):
        """
        Uses OpenCV color thresholding on the actual page image to locate:
        1. The central vertical red/pink column dividing line.
        2. The bottom horizontal red footer banner.
        """
        img_np = np.array(img_pil.convert('RGB'))
        h, w, _ = img_np.shape

        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 40, 50]), np.array([12, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([145, 25, 50]), np.array([180, 255, 255]))
        red_mask = mask1 | mask2

        # 1. Central Vertical Red Line (Search between 35% and 65% of page width)
        mid_start = int(w * 0.35)
        mid_end = int(w * 0.65)
        col_sums = np.sum(red_mask[:, mid_start:mid_end] > 0, axis=0)
        
        if len(col_sums) > 0 and np.max(col_sums) > h * 0.15:
            red_line_x = mid_start + int(np.argmax(col_sums))
        else:
            red_line_x = w // 2

        # 2. Bottom Red Footer Banner (Search in bottom 15% of page height)
        bottom_start = int(h * 0.85)
        row_sums = np.sum(red_mask[bottom_start:, :] > 0, axis=1)
        dense_rows = np.where(row_sums > w * 0.20)[0]
        
        if len(dense_rows) > 0:
            footer_top_y = bottom_start + int(dense_rows[0]) - 8
        else:
            footer_top_y = int(h * 0.95)

        return red_line_x, footer_top_y

    def get_ocr_words(self, img_pil):
        """Extracts OCR text and bounding boxes using Tesseract to find anchor positions."""
        data = pytesseract.image_to_data(img_pil, output_type=pytesseract.Output.DICT)
        words = []
        n = len(data['text'])
        for i in range(n):
            text = data['text'][i].strip()
            if not text:
                continue
            x = data['left'][i]
            y = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]
            try:
                conf = float(data['conf'][i])
            except (ValueError, TypeError):
                conf = 0.0
            if conf > 15:
                words.append({
                    'text': text,
                    'x0': x,
                    'top': y,
                    'x1': x + w,
                    'bottom': y + h,
                    'height': h,
                    'center_x': x + (w / 2.0)
                })
        return words

    def group_into_lines(self, words):
        """Groups words into horizontal lines based on Y coordinate."""
        if not words:
            return []
        avg_h = sum(w['height'] for w in words) / max(1, len(words))
        line_y_tol = max(8, int(avg_h * 0.65))

        words_sorted = sorted(words, key=lambda w: (w['top'], w['x0']))
        lines = []
        curr_line = []
        curr_y = -1

        for w in words_sorted:
            if curr_y == -1 or abs(w['top'] - curr_y) <= line_y_tol:
                curr_line.append(w)
                curr_y = w['top']
            else:
                curr_line.sort(key=lambda item: item['x0'])
                lines.append(curr_line)
                curr_line = [w]
                curr_y = w['top']
        if curr_line:
            curr_line.sort(key=lambda item: item['x0'])
            lines.append(curr_line)
        return lines

    def process_pdf(self, pdf_path, output_dir="extracted_output"):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Flat output directories
        q_dir = os.path.join(output_dir, "questions")
        e_dir = os.path.join(output_dir, "explanations")
        os.makedirs(q_dir, exist_ok=True)
        os.makedirs(e_dir, exist_ok=True)

        print(f"[+] Rendering actual PDF pages at {self.dpi} DPI...")
        page_images = convert_from_path(pdf_path, dpi=self.dpi)
        total_pages = len(page_images)
        print(f"[+] Total pages to process: {total_pages}")

        total_q = 0
        total_e = 0
        in_explanation_mode = False

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            # Real rendered image of the scanned page
            p_img = page_images[page_idx]
            img_w, img_h = p_img.size

            print(f"\n" + "="*55)
            print(f"[*] Processing Page {page_num}/{total_pages} ({img_w}x{img_h} px)")
            print(f"="*55)

            # 1. Detect Visual Boundaries on the actual page
            red_line_x, footer_top_y = self.find_visual_dividers(p_img)
            header_limit = int(img_h * self.top_header_pct)
            print(f"  [+] Landmarks: Red Column Line at x={red_line_x}, Footer Bar at y={footer_top_y}")

            # 2. OCR text to locate anchors
            words = self.get_ocr_words(p_img)
            content_words = [w for w in words if header_limit <= w['top'] <= footer_top_y]

            # 3. Detect Section Breaks (ANSWER-KEY vs EXPLANATION)
            answer_key_top = None
            explanation_header_y = None

            all_lines = self.group_into_lines(content_words)
            for line in all_lines:
                line_str = " ".join([w['text'] for w in line]).strip().upper()
                if "ANSWER-KEY" in line_str or "ANSWER KEY" in line_str:
                    answer_key_top = min(w['top'] for w in line)
                if "EXPLANATION" in line_str or "EXPLANATIONS" in line_str:
                    explanation_header_y = max(w['bottom'] for w in line)
                    in_explanation_mode = True

            # Filter words strictly outside Answer-Key grid
            valid_words = []
            for w in content_words:
                if answer_key_top and not explanation_header_y:
                    if w['bottom'] < answer_key_top - 10:
                        valid_words.append(w)
                elif explanation_header_y:
                    if w['top'] > explanation_header_y + 10:
                        valid_words.append(w)
                elif in_explanation_mode:
                    valid_words.append(w)
                else:
                    valid_words.append(w)

            if not valid_words:
                continue

            # 4. Partition Words into Left and Right Columns based on Red Line
            left_words = [w for w in valid_words if w['x1'] <= red_line_x]
            right_words = [w for w in valid_words if w['x0'] >= red_line_x]

            col_configs = [
                {
                    "name": "Left",
                    "words": left_words,
                    "x0": max(0, int(img_w * 0.02)),
                    "x1": red_line_x - 3
                },
                {
                    "name": "Right",
                    "words": right_words,
                    "x0": red_line_x + 3,
                    "x1": min(img_w, int(img_w * 0.98))
                }
            ]

            for col in col_configs:
                c_words = col["words"]
                if not c_words:
                    continue

                lines = self.group_into_lines(c_words)
                if not lines:
                    continue

                # 5. Locate Anchors in Column
                anchors = []
                for l_idx, line in enumerate(lines):
                    line_str = " ".join([w['text'] for w in line]).strip()

                    if "TYPE-" in line_str.upper() or "WIZARD" in line_str.upper():
                        continue

                    # Explanation Pattern: "1. (c)", "10. (a)", "64. (d)"
                    exp_match = re.match(r'^\s*(\d{1,4})\s*[\.\,\:\-\)]\s*(?:\(?([a-dA-D])\)|\b([a-dA-D])\b)', line_str)
                    # Question Pattern: "351.", "1."
                    q_match = re.match(r'^\s*(?:Q(?:uestion)?[\.\s:]*)?(\d{1,4})\s*[\.\,\:\-\)]\s*', line_str)

                    if in_explanation_mode:
                        match = exp_match if exp_match else q_match
                        if match:
                            anchors.append({
                                'num': match.group(1),
                                'type': 'explanation',
                                'top': min(w['top'] for w in line),
                                'line_idx': l_idx
                            })
                    else:
                        if q_match:
                            anchors.append({
                                'num': q_match.group(1),
                                'type': 'question',
                                'top': min(w['top'] for w in line),
                                'line_idx': l_idx
                            })

                if not anchors:
                    continue

                # 6. Crop the ACTUAL Region from the Real Page Image (p_img)
                for i, anchor in enumerate(anchors):
                    item_num = anchor['num']
                    item_type = anchor['type']

                    start_l = anchor['line_idx']
                    end_l = anchors[i+1]['line_idx'] if i + 1 < len(anchors) else len(lines)
                    item_words = [w for li in range(start_l, end_l) for w in lines[li]]
                    if not item_words:
                        continue

                    item_top = min(w['top'] for w in item_words)
                    item_bottom = max(w['bottom'] for w in item_words)

                    # Top edge = top of question/explanation number
                    y_start = max(header_limit, item_top - 6)

                    # Bottom edge = top of below question number OR item bottom + buffer (capped at footer)
                    if i + 1 < len(anchors):
                        y_end = anchors[i + 1]['top'] - 6
                    else:
                        if answer_key_top and not in_explanation_mode:
                            y_end = min(answer_key_top - 10, item_bottom + 25)
                        else:
                            y_end = min(footer_top_y, item_bottom + 25)

                    if y_end <= y_start + 30:
                        continue

                    block_lines = [" ".join([w['text'] for w in lines[li]]) for li in range(start_l, end_l)]
                    full_text = "\n".join(block_lines)

                    target_dir = e_dir if item_type == 'explanation' else q_dir

                    if item_type == 'explanation':
                        total_e += 1
                    else:
                        total_q += 1

                    # 1. Save Text File
                    txt_path = os.path.join(target_dir, f"q_{item_num}.txt")
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(full_text)

                    # 2. CROP ACTUAL IMAGE PIXELS DIRECTLY FROM THE SCANNED PAGE
                    # This captures the real drawings, geometric figures, and math equations
                    crop_box = (
                        int(col["x0"]),
                        int(y_start),
                        int(col["x1"]),
                        int(y_end)
                    )
                    cropped_actual_image = p_img.crop(crop_box)
                    png_path = os.path.join(target_dir, f"q_{item_num}.png")
                    cropped_actual_image.save(png_path, format="PNG")

                    print(f"  [✓] Cropped {item_type.capitalize()} #{item_num} ({col['name']} Col) -> {item_type}s/q_{item_num}.png [Size: {cropped_actual_image.size}]")

        print(f"\n" + "="*55)
        print(f"[+] EXTRACTION COMPLETE:")
        print(f"    • Questions:    {total_q} actual image crops -> {q_dir}/")
        print(f"    • Explanations: {total_e} actual image crops -> {e_dir}/")
        print(f"="*55)
        return output_dir

if __name__ == "__main__":
    import sys
    pdf_path = sys.argv if len(sys.argv) > 1 else "rakesh_pages_261_to_265.pdf"
    extractor = MathPdfExtractor(dpi=300)
    extractor.process_pdf(pdf_path)