#
# This script is part of the table ocr workflow and applies OCR to the bounding boxes extracted from the PDFs. 
# The script reads the bounding box information from the .bb files generated in the previous step and applies 
# OCR to the extracted cell images. The OCR results are saved in a .bbocr file, and optionally, the script 
# generates a PDF with the OCR text overlaid on the original PDF pages.
#

#!/usr/bin/env python
import os
import json
import cv2
import numpy as np
import img2pdf
from pdf2image import convert_from_path
import pytesseract
import fitz  # PyMuPDF, for optional PDF annotation
from table_cell_extractor import extract_table_cell_bboxes  # our opencv-based cell extractor
from tqdm import tqdm

# --- Configuration ---
input_pdf_dir = "input_pdfs"             # Folder containing input PDFs.
bb_dir = "bounding_boxes"                # Folder with bounding box files (.bb).
bbocr_dir = "bbox_ocrs"                  # Folder to save OCR bounding box JSON files (.bbocr).
pdf_with_ocr_dir = "pdfs_with_ocr"       # Folder to save PDFs with OCR text overlaid.
html_output_dir = "html_output"          # Folder to save HTML files for OCR review.
generate_pdf_with_ocr = True             # Toggle to generate annotated PDFs.
render_dpi = 300                         # DPI for rendering PDFs to images.
# OCR configurations: Hungarian language with separate whitelists for characters and numbers.
char_whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzáéíóöőüűÁÉÍÓÖŐÚÜŰ."
num_whitelist = "0123456789,.-‐—–−"
ocr_config_char = f"--psm 6 -l hun " # -c tessedit_char_whitelist={char_whitelist}
ocr_config_num = f"--psm 6 -l hun -c tessedit_char_whitelist={num_whitelist}"

# You can adjust epsilon (expansion) if desired for cell boxes.
epsilon = 0

# Set pytesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Create output directories if they don't exist.
for d in [bbocr_dir, pdf_with_ocr_dir, html_output_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

# Basic cleaning of page image
def clean_image(image):
    """
    Perform minimal cleaning on the input image to remove noise and make it easier for OCR.
    Args:
        image (numpy.ndarray): The input image to be cleaned.
    Returns:
        numpy.ndarray: The cleaned image.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply a binary threshold to remove non-black and non-white elements
    #_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Use morphological operations to reduce noise
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return cleaned


def remove_lines(gray_image):
    """
    Remove vertical and horizontal lines from a grayscale image and clean up noise.
    
    Args:
    - gray_image: Grayscale image (numpy array)
    
    Returns:
    - cleaned_image: Cleaned image with lines removed and noise filtered (numpy array)
    """
    # Apply adaptive thresholding to get a binary image
    binary = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    vertical_removed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    horizontal_removed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

    # Combine the masks
    lines_removed = cv2.bitwise_or(vertical_removed, horizontal_removed)
    cleaned_image = cv2.bitwise_and(binary, cv2.bitwise_not(lines_removed))

    # Further noise removal using morphological closing and opening
    noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned_image = cv2.morphologyEx(cleaned_image, cv2.MORPH_CLOSE, noise_kernel)
    cleaned_image = cv2.morphologyEx(cleaned_image, cv2.MORPH_OPEN, noise_kernel)

    # Invert the image to get black on white
    cleaned_image = cv2.bitwise_not(cleaned_image)

    return cleaned_image



# Process each PDF in the input directory.
pdf_files = [f for f in os.listdir(input_pdf_dir) if f.lower().endswith(".pdf")]
for pdf_filename in tqdm(pdf_files, desc="Processing PDFs"):
    pdf_path = os.path.join(input_pdf_dir, pdf_filename)
    base_name = os.path.splitext(pdf_filename)[0]
    print(f"\nProcessing PDF: {pdf_filename}")

    # Find corresponding bounding box file (.bb) in bb_dir.
    bb_file = os.path.join(bb_dir, f"{base_name}.bb")
    if not os.path.exists(bb_file):
        print(f"Bounding box file not found for {pdf_filename}. Skipping.")
        continue

    with open(bb_file, "r", encoding="utf-8") as f:
        bb_data = json.load(f)  # Expected to be a list of page dictionaries.

    # Convert the PDF pages to images using pdf2image.
    try:
        pages = convert_from_path(pdf_path, dpi=render_dpi)
    except Exception as e:
        print(f"Error converting PDF {pdf_filename}: {e}")
        continue

    # Optionally, open the PDF with PyMuPDF for annotation.
    try:
        pdf_fitz = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF with PyMuPDF: {e}")
        pdf_fitz = None

    ocr_result_data = []  # To store OCR results per page.
    overlay_pages = []    # To collect pages with overlaid OCR text.
    temp_image_dir = os.path.join("temp_images", base_name)
    os.makedirs(temp_image_dir, exist_ok=True)

    # Process each page (as described in the .bb file).
    for page_info in tqdm(bb_data, desc=f"Processing pages in {pdf_filename}", leave=False):
        page_num = page_info.get("page")
        tables = page_info.get("tables", [])
        print(f"Processing page {page_num}...")

        try:
            page_img = cv2.cvtColor(np.array(pages[page_num - 1]), cv2.COLOR_RGB2BGR)
            # Clean the page image
            page_img = clean_image(page_img)
        except IndexError:
            print(f"Page {page_num} not found in rendered images. Skipping page.")
            continue

        page_result = {"page": page_num, "tables": []}
        page_overlay = page_img.copy()

        # Process each table in the page.
        for table_idx, table in enumerate(tqdm(tables, desc=f"Processing tables on page {page_num}", leave=False)):
            # Expect table has key "table_coordinates": [tx1, ty1, tx2, ty2]
            table_coords = table.get("table_coordinates")
            if not table_coords or len(table_coords) != 4:
                continue
            tx1, ty1, tx2, ty2 = map(int, table_coords)
            table_region = page_img[ty1:ty2, tx1:tx2]

            # Run the cell extractor on the table region.
            # (This version uses only morphological boundaries, no text-based lines.)
            cell_bboxes = table.get("cell_bboxes")

            # Get grid shape
            grid_shape = table.get("grid_shape")

            abs_cells = []
            cells_info = []  # Will store OCR info for each cell.
            # For each cell bounding box (relative to table_region), convert to absolute coordinates.
            for cell_idx, bbox in enumerate(tqdm(cell_bboxes, desc="Processing cells", leave=False)):
                cx1, cy1, cx2, cy2 = bbox.get("bbox")
                cx1 += epsilon
                cy1 += epsilon
                cx2 -= epsilon
                cy2 -= epsilon
                abs_bbox = (cx1, cy1, cx2, cy2)
                #print(abs_bbox)
                abs_cells.append(abs_bbox)
                # Crop the cell region from the page image.
                #cell_img = remove_lines(page_img[abs_bbox[1]:abs_bbox[3], abs_bbox[0]:abs_bbox[2]])
                cell_img = page_img[abs_bbox[1]:abs_bbox[3], abs_bbox[0]:abs_bbox[2]]

                height, width = cell_img.shape[:2]
                if height == 0 or width == 0:
                    print(f"Invalid cell image dimensions for bbox {abs_bbox}. Skipping OCR.")
                    continue

                # Save the cell image to a temporary file.
                cell_img_path = os.path.join(temp_image_dir, f"page_{page_num}_table_{table_idx}_cell_{cell_idx}.png")
                cv2.imwrite(cell_img_path, cell_img)

                # Run OCR on the cell region using pytesseract with character whitelist.
                ocr_data_char = pytesseract.image_to_data(cell_img, config=ocr_config_char, output_type=pytesseract.Output.DICT)
                # Run OCR on the cell region using pytesseract with number whitelist.
                ocr_data_num = pytesseract.image_to_data(cell_img, config=ocr_config_num, output_type=pytesseract.Output.DICT)

                # Combine words and compute average confidence for character whitelist.
                texts_char = []
                confs_char = []
                for i in range(len(ocr_data_char["text"])):
                    word = ocr_data_char["text"][i].strip()
                    try:
                        conf = float(ocr_data_char["conf"][i])
                    except:
                        conf = -1
                    if word != "":
                        texts_char.append(word)
                        if conf != -1:
                            confs_char.append(conf)
                combined_text_char = " ".join(texts_char)
                avg_conf_char = sum(confs_char)/len(confs_char) if confs_char else 0

                # Combine words and compute average confidence for number whitelist.
                texts_num = []
                confs_num = []
                for i in range(len(ocr_data_num["text"])):
                    word = ocr_data_num["text"][i].strip()
                    try:
                        conf = float(ocr_data_num["conf"][i])
                    except:
                        conf = -1
                    if word != "":
                        texts_num.append(word)
                        if conf != -1:
                            confs_num.append(conf)
                combined_text_num = " ".join(texts_num)
                avg_conf_num = sum(confs_num)/len(confs_num) if confs_num else 0

                # Store both OCR results and their confidence scores.
                cells_info.append({
                    "bbox": abs_bbox,
                    "text_char": combined_text_char,
                    "ocr_conf_char": avg_conf_char,
                    "text_num": combined_text_num,
                    "ocr_conf_num": avg_conf_num,
                    "image_path": cell_img_path
                })
                # Optionally, draw the OCR text on the overlay image.
                cv2.putText(page_overlay, combined_text_char, (abs_bbox[0], abs_bbox[1]-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

            # Save table info.
            page_result["tables"].append({
                "table_coordinates": table_coords,
                "grid_shape": grid_shape,
                "cells": cells_info
            })

        ocr_result_data.append(page_result)
        overlay_pages.append(page_overlay)

    # Save the OCR results to a .bbocr file in the bbox_ocrs folder.
    bbocr_output_path = os.path.join(bbocr_dir, f"{base_name}.bbocr")
    with open(bbocr_output_path, "w", encoding="utf-8") as f:
        json.dump(ocr_result_data, f, indent=2)
    print(f"Saved OCR bounding box info to: {bbocr_output_path}")

    # Generate HTML file for OCR review.
    html_output_path = os.path.join(html_output_dir, f"{base_name}.html")
    with open(html_output_path, "w", encoding="utf-8") as f:
        f.write("<html><body>\n")
        f.write(f"<h1>OCR Results for {pdf_filename}</h1>\n")
        for page in ocr_result_data:
            f.write(f"<h2>Page {page['page']}</h2>\n")
            for table in page["tables"]:
                f.write("<table border='1'>\n")
                for cell in table["cells"]:
                    f.write("<tr>\n")
                    f.write(f"<td><img src='../{cell['image_path']}'></td>\n")
                    f.write(f"<td>Text (Char): {cell['text_char']}<br>Confidence (Char): {cell['ocr_conf_char']}<br>"
                            f"Text (Num): {cell['text_num']}<br>Confidence (Num): {cell['ocr_conf_num']}<br>"
                            f"Bounding Box: {cell['bbox']}</td>\n")
                    f.write("</tr>\n")
                f.write("</table>\n")
        f.write("</body></html>\n")
    print(f"Generated HTML review file: {html_output_path}")

    # If the toggle is on, generate a PDF with the OCR annotations overlaid.
    if generate_pdf_with_ocr and overlay_pages:
        temp_overlay_dir = os.path.join("temp_overlays", base_name)
        os.makedirs(temp_overlay_dir, exist_ok=True)
        overlay_paths = []
        for idx, overlay_img in enumerate(overlay_pages, start=1):
            overlay_path = os.path.join(temp_overlay_dir, f"page_{idx}.png")
            cv2.imwrite(overlay_path, overlay_img)
            overlay_paths.append(overlay_path)
        output_pdf_path = os.path.join(pdf_with_ocr_dir, f"{base_name}.pdf")
        try:
            with open(output_pdf_path, "wb") as f_out:
                f_out.write(img2pdf.convert(overlay_paths))
            print(f"Generated annotated PDF with OCR text: {output_pdf_path}")
        except Exception as e:
            print(f"Error generating annotated PDF for {pdf_filename}: {e}")

print("Processing complete.")