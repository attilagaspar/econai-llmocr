#!/usr/bin/env python

import os
import sys
import json
import cv2
import numpy as np
import pytesseract
from PIL import Image

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

def extract_ocr_for_shapes(img, shapes, tess_config):
    print(f"Extracting OCR for {len(shapes)} shapes.")
    x = sum(1 for shape in shapes if "tesseract_output" in shape)
    if x > 0:
        # If there are shapes with "tesseract_output" already present, skip OCR for those
       print(f"{x} shapes with 'tesseract_output' already present, skipping OCR for those.")
    for shape in shapes:
        # Only OCR if "tesseract_output" field is missing
        if "tesseract_output" in shape:
            continue
        if "points" not in shape or len(shape["points"]) < 2:
            shape["tesseract_output"] = {"ocr_text": "", "ocr_score": None}
            continue
        pts = np.array(shape["points"], dtype=np.int32)
        x1 = np.min(pts[:, 0])
        y1 = np.min(pts[:, 1])
        x2 = np.max(pts[:, 0])
        y2 = np.max(pts[:, 1])
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            shape["tesseract_output"] = {"ocr_text": "", "ocr_score": None}
            continue
        pil_roi = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        ocr_text = pytesseract.image_to_string(pil_roi, config=tess_config).strip()
        ocr_data = pytesseract.image_to_data(pil_roi, config=tess_config, output_type=pytesseract.Output.DICT)
        confs = [float(str(conf)) for conf in ocr_data['conf'] if str(conf).isdigit()]
        ocr_score = float(np.mean(confs)) if confs else None
        shape["tesseract_output"] = {"ocr_text": ocr_text, "ocr_score": ocr_score}
    return

def main():
    if len(sys.argv) < 2:
        print("Usage: python text_extractor.py <input_dir>")
        sys.exit(1)
    input_dir = sys.argv[1]

    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    tess_config = '--psm 6 -l hun '

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

        extract_ocr_for_shapes(img, shapes, tess_config)
        # shapes are modified in place

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"OCR results added to {json_path}")

if __name__ == "__main__":
    main()