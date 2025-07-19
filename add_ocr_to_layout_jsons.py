#!/usr/bin/env python

import os
import sys
import json
import cv2
import numpy as np
import pytesseract
import uuid
from PIL import Image
from collections import Counter



#TESS_PATH_LOCAL = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
TESS_PATH_LOCAL = r"/usr/bin/tesseract"



def find_labelme_jsons(input_dir):
    json_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def load_image_for_json(json_path):
    base = os.path.splitext(json_path)[0]
    for ext in ['.jpg', '.jpeg', '.png']:
        img_path = base + ext
        if os.path.exists(img_path):
            return img_path
    return None

def remove_vertical_edges(np_img, edge_width=3, black_thresh=50):
    """
    Remove black border pixels from the left and right `edge_width` columns
    if their average intensity is below `black_thresh`.
    """
    h, w = np_img.shape
    # Left edge
    if np.mean(np_img[:, :edge_width]) < black_thresh:
        np_img[:, :edge_width] = 255
    # Right edge
    if np.mean(np_img[:, -edge_width:]) < black_thresh:
        np_img[:, -edge_width:] = 255
    return np_img


def enhance_pil_cell(pil_cell):
    """
    Enhance a PIL image for OCR:
    - Convert to grayscale
    - Remove vertical lines
    - Remove light gray noise (pixels above threshold set to white)
    - (Optional) Apply dilation to reconnect broken digits
    - Upscale by 2x
    Returns a new enhanced PIL image.
    """
    gray = pil_cell.convert("L")
    np_img = np.array(gray)

    # Upscale
    upscaled = cv2.resize(np_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    return Image.fromarray(upscaled)


def extract_ocr_for_shapes(img, shapes, tess_config, temp_dir="temp_cells", fixed_cell_height=28):
    print(f"Extracting OCR for {len(shapes)} shapes.")
    os.makedirs(temp_dir, exist_ok=True)

    for shape in shapes:
        if shape.get("label") != "numerical_cell":
            continue

        if "points" not in shape or len(shape["points"]) < 2:
            shape["tesseract_output"] = {"ocr_text": "", "ocr_score": None}
            continue

        pts = np.array(shape["points"], dtype=np.int32)
        x1, y1 = np.min(pts[:, 0]), np.min(pts[:, 1])
        x2, y2 = np.max(pts[:, 0]), np.max(pts[:, 1])
        roi = img[y1:y2, x1:x2]
        #roi = crop_left_gap_from_roi(roi)
        if roi.size == 0:
            shape["tesseract_output"] = {"ocr_text": "", "ocr_score": None}
            continue

        roi_height = roi.shape[0]
        max_cells = roi_height // fixed_cell_height

        # Preprocessing
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

        # Horizontal projection
        projection = np.sum(binary, axis=1)

        # Score each possible center: sum of projection in the implied cell
        half = fixed_cell_height // 2
        scores = []
        for center in range(half, roi_height - half):
            top = center - half
            bottom = center + half
            score = np.sum(projection[top:bottom])
            scores.append((center, score))

        # Greedy selection of non-overlapping cells with best coverage
        scores.sort(key=lambda x: x[1], reverse=True)
        selected = []
        occupied = np.zeros(roi_height, dtype=bool)

        for center, _ in scores:
            top = max(0, center - half)
            bottom = min(roi_height, top + fixed_cell_height)
            if not occupied[top:bottom].any():
                selected.append((top, bottom))
                occupied[top:bottom] = True
                if len(selected) >= max_cells:
                    break

        selected.sort()

        # Annotate and save
        roi_annotated = roi.copy()
        cell_texts = []
        confs = []

        for top, bottom in selected:
            cell_img = roi[top:bottom, :]

            # Draw rectangle
            cv2.rectangle(roi_annotated, (0, top), (roi.shape[1], bottom), (0, 0, 255), 1)

            # Enhance and OCR
            pil_cell = Image.fromarray(cv2.cvtColor(cell_img, cv2.COLOR_BGR2RGB))
            pil_cell = enhance_pil_cell(pil_cell)
            pil_cell = crop_left_gap_from_pil_cell(pil_cell)

            text = pytesseract.image_to_string(pil_cell, config=tess_config).strip()
            cell_texts.append(text)

            # Confidence
            data = pytesseract.image_to_data(pil_cell, config=tess_config, output_type=pytesseract.Output.DICT)
            conf = [float(str(c)) for c in data['conf'] if str(c).isdigit()]
            if conf:
                confs.extend(conf)

            # Put OCR result as overlay text
            label = text if text else "?"
            font_scale = 0.5
            thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            (w_label, h_label), _ = cv2.getTextSize(label, font, font_scale, thickness)
            y_text = top + h_label + 2
            x_text = 2
            cv2.putText(roi_annotated, label, (x_text, y_text), font, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)

        # Visualize what Tesseract saw
        enhanced_stack = [np.array(enhance_pil_cell(Image.fromarray(cv2.cvtColor(
            roi[top:bottom, :] ,
            cv2.COLOR_BGR2RGB)))) for top, bottom in selected]
        tess_input_view = cv2.vconcat(enhanced_stack)
        if len(tess_input_view.shape) == 2:
            tess_input_view = cv2.cvtColor(tess_input_view, cv2.COLOR_GRAY2BGR)
        if roi_annotated.shape[0] != tess_input_view.shape[0]:
            tess_input_view = cv2.resize(tess_input_view, (tess_input_view.shape[1], roi_annotated.shape[0]))
        side_by_side = cv2.hconcat([roi_annotated, tess_input_view])

        # Save combined image
        annotated_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}_annotated.png")
        Image.fromarray(cv2.cvtColor(side_by_side, cv2.COLOR_BGR2RGB)).save(annotated_path)

        ocr_text = "\n".join(cell_texts)
        ocr_score = float(np.mean(confs)) if confs else None
        shape["tesseract_output"] = {
            "ocr_text": ocr_text,
            "ocr_score": ocr_score,
            "annotated_image": annotated_path
        }
        print(ocr_text)

    return shapes


def main():
    if len(sys.argv) < 2:
        print("Usage: python text_extractor.py <input_dir>")
        sys.exit(1)
    input_dir = sys.argv[1]

    pytesseract.pytesseract.tesseract_cmd = TESS_PATH_LOCAL
    #tess_config = '--psm 6 -l hun '
    #tess_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789-.'
    tess_config = r'--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.—- -c preserve_interword_spaces=1 -c tessedit_do_invert=0'
    #tess_config = r'--psm 6 --oem 3  '

    json_files = find_labelme_jsons(input_dir)
    print(f"Found {len(json_files)} LabelMe JSON files.")

    for json_path in json_files:
        print(f"Processing {json_path}...")
        img_path = load_image_for_json(json_path)
        if not img_path:
            print(f"No image found for {json_path}, skipping.")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        shapes = data.get("shapes", [])
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image {img_path}, skipping.")
            continue
        print(json_path)
        extract_ocr_for_shapes(img, shapes, tess_config)
        # shapes are modified in place

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"OCR results added to {json_path}")

if __name__ == "__main__":
    main()