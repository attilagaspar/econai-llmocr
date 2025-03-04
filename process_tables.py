# --- Monkey-Patching Section ---
# 1. Ensure Pillow's Image module has the attribute 'LINEAR' (needed by Detectron2)
import PIL.Image as Image
if not hasattr(Image, 'LINEAR'):
    Image.LINEAR = Image.BILINEAR

# 2. Monkey-patch FreeTypeFont to add getsize() if missing (using getbbox)
from PIL import ImageFont
if not hasattr(ImageFont.FreeTypeFont, "getsize"):
    def getsize(self, text):
        bbox = self.getbbox(text)  # returns (left, top, right, bottom)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return (width, height)
    ImageFont.FreeTypeFont.getsize = getsize

# 3. Patch iopath's HTTPURLHandler to remove query parameters from cached filenames.
from iopath.common.file_io import HTTPURLHandler
_original_http_get_local_path = HTTPURLHandler._get_local_path
def patched_http_get_local_path(self, path, force=False, **kwargs):
    if isinstance(path, str) and '?' in path:
        path = path.split('?')[0]
    return _original_http_get_local_path(self, path, force=force, **kwargs)
HTTPURLHandler._get_local_path = patched_http_get_local_path
# --- End of Monkey-Patching Section ---

import layoutparser as lp
from pdf2image import convert_from_path
import numpy as np
import cv2
import pytesseract
import json
import os
import pandas as pd

# Import the cell extraction function
from table_cell_extractor import extract_table_cell_bboxes

# --- Configuration ---
doc_tag = "mg_osszeiras"
pdf_path = 'raw/1935mg_osszeiras_sample-2-9-1.pdf'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Directories for outputs
cell_images_dir = "output/cell_images"
cell_ocr_dir = "output/cell_ocr"
excel_dir = "output/excel_tables"
for d in [cell_images_dir, cell_ocr_dir, excel_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

# --- Convert PDF to Image (first page) ---
pages = convert_from_path(pdf_path, dpi=600)
page_image = np.array(pages[0])
page_image = cv2.cvtColor(page_image, cv2.COLOR_RGB2BGR)
img_height, img_width = page_image.shape[:2]

# --- Initialize Layout Parser Model ---
model = lp.Detectron2LayoutModel(
    'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
)
layout = model.detect(page_image)

# --- Merge Adjacent Table Elements (process only table elements) ---
def merge_boxes(box1, box2):
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])
    return [x1, y1, x2, y2]

def are_adjacent(box1, box2, gap_threshold=10):
    intersect_x1 = max(box1[0], box2[0])
    intersect_y1 = max(box1[1], box2[1])
    intersect_x2 = min(box1[2], box2[2])
    intersect_y2 = min(box1[3], box2[3])
    if intersect_x1 < intersect_x2 and intersect_y1 < intersect_y2:
        return True
    if box1[3] < box2[1]:
        vertical_gap = box2[1] - box1[3]
    elif box2[3] < box1[1]:
        vertical_gap = box1[1] - box2[3]
    else:
        vertical_gap = 0
    horizontal_overlap = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    width1 = box1[2] - box1[0]
    width2 = box2[2] - box2[0]
    if vertical_gap <= gap_threshold and (horizontal_overlap >= 0.5 * width1 or horizontal_overlap >= 0.5 * width2):
        return True
    if box1[2] < box2[0]:
        horizontal_gap = box2[0] - box1[2]
    elif box2[2] < box1[0]:
        horizontal_gap = box1[0] - box2[2]
    else:
        horizontal_gap = 0
    vertical_overlap = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    height1 = box1[3] - box1[1]
    height2 = box2[3] - box2[1]
    if horizontal_gap <= gap_threshold and (vertical_overlap >= 0.5 * height1 or vertical_overlap >= 0.5 * height2):
        return True
    return False

def merge_adjacent_tables(layout_elements, gap_threshold=10):
    table_elements = [elem for elem in layout_elements if elem.type.lower() == "table"]
    non_table_elements = [elem for elem in layout_elements if elem.type.lower() != "table"]
    
    merged_tables = []
    while table_elements:
        current = table_elements.pop(0)
        current_box = current.coordinates
        current_score = current.score
        merged = True
        while merged:
            merged = False
            for other in table_elements[:]:
                if are_adjacent(current_box, other.coordinates, gap_threshold):
                    current_box = merge_boxes(current_box, other.coordinates)
                    current_score = max(current_score, other.score)
                    table_elements.remove(other)
                    merged = True
                    break
        new_block = lp.Rectangle(*current_box)
        new_table = current.__class__(new_block, score=current_score, type=current.type)
        merged_tables.append(new_table)
    
    return non_table_elements + merged_tables

merged_layout = merge_adjacent_tables(layout, gap_threshold=10)

# Save parsed layout details (optional)
output_text_file = f"output/{doc_tag}_parsed_layout.txt"
with open(output_text_file, "w", encoding="utf-8") as f:
    for idx, element in enumerate(merged_layout):
        f.write(f"Element {idx}:\n")
        f.write(f"   Type: {element.type}\n")
        f.write(f"   Score: {element.score:.3f}\n")
        f.write(f"   Coordinates: {element.coordinates}\n\n")
print(f"Layout details saved to {output_text_file}")

# --- Process Each Table Element ---
excel_data = []  # to store info about processed tables
table_count = 0

# This list will collect all expanded OCR rows for each table so that we can later compile into an Excel file.
for element in merged_layout:
    if element.type.lower() != "table":
        continue
    table_count += 1
    # Get absolute table region coordinates from the page.
    tx1, ty1, tx2, ty2 = map(int, element.coordinates)
    table_img = page_image[ty1:ty2, tx1:tx2]
        

    # Optionally, restrict text-based line detection to the leftmost 300 pixels:
    text_roi_x_min = 0
    text_roi_x_max = 1300

    # Use the cell extractor (with default tolerance levels: min_line_length=150, merge_threshold=20)
    #cell_bboxes, (n_rows, n_cols) = extract_table_cell_bboxes(table_img, min_line_length=150, merge_threshold=20)
    cell_bboxes, grid_img, grid_shape = extract_table_cell_bboxes(
        table_img, 
        min_line_length=150, 
        merge_threshold=20, 
        text_roi_x_min=text_roi_x_min, 
        text_roi_x_max=text_roi_x_max,
    )    
    cv2.imwrite("output/grid_visualization.png", grid_img)
    (n_rows, n_cols) = grid_shape
    print(f"Table {table_count}: Detected {len(cell_bboxes)} cells ({n_rows} rows, {n_cols} cols)")
    
# Build a grid (2D list) for OCR results.
    ocr_grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    total_cells = n_rows * n_cols
    
    # Use tqdm to show a horizontal progress bar for cell OCR processing.
    from tqdm import tqdm
    for cell_idx in tqdm(range(total_cells), desc=f"Processing Table {table_count} cells"):
        r = cell_idx // n_cols
        c = cell_idx % n_cols
        bbox = cell_bboxes[cell_idx]
        cx1, cy1, cx2, cy2 = bbox  # relative to table_img
        # Map cell coordinates to absolute page coordinates.
        abs_x1 = tx1 + cx1
        abs_y1 = ty1 + cy1
        abs_x2 = tx1 + cx2
        abs_y2 = ty1 + cy2
        cell_img = page_image[abs_y1:abs_y2, abs_x1:abs_x2]
        
        # Save the cell image.
        cell_img_filename = f"{doc_tag}_table_{table_count}_row_{r+1}_col_{c+1}.png"
        cell_img_path = os.path.join(cell_images_dir, cell_img_filename)
        cv2.imwrite(cell_img_path, cell_img)
        
        # Run OCR on the cell image.
        ocr_text = pytesseract.image_to_string(cell_img, lang='hun').strip()
        # Save OCR output to an individual text file.
        ocr_text_filename = f"{doc_tag}_table_{table_count}_row_{r+1}_col_{c+1}.txt"
        ocr_text_path = os.path.join(cell_ocr_dir, ocr_text_filename)
        with open(ocr_text_path, "w", encoding="utf-8") as f:
            f.write(ocr_text)
        
        ocr_grid[r][c] = ocr_text
    
    # --- Expand OCR Grid Based on Newlines ---
    # For each table row, split each cell's OCR text by newline.
    expanded_grid = []
    for row in ocr_grid:
        # Split each cell's text into lines.
        split_cells = [cell.splitlines() if cell else [""] for cell in row]
        # Determine the maximum number of lines in this row.
        max_lines = max(len(lines) for lines in split_cells)
        # Pad cells with fewer lines with empty strings.
        padded_cells = [lines + [""] * (max_lines - len(lines)) for lines in split_cells]
        # Each line in the cell becomes a row in the expanded grid.
        for i in range(max_lines):
            new_row = [padded_cells[j][i] for j in range(n_cols)]
            expanded_grid.append(new_row)
    
    # Save the expanded grid to an Excel file.
    df = pd.DataFrame(expanded_grid)
    excel_filename = f"{doc_tag}_table_{table_count}_cells.xlsx"
    excel_path = os.path.join(excel_dir, excel_filename)
    df.to_excel(excel_path, index=False, header=False)
    print(f"Saved Excel file for table {table_count}: {excel_path}")

print("Processing complete. Cell images, OCR outputs, and Excel files have been saved.")
