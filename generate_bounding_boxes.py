#!/usr/bin/env python
import os
import json
import cv2
import numpy as np
import img2pdf
from pdf2image import convert_from_path
import layoutparser as lp

# --- Monkey-Patching Section ---
import PIL.Image as Image
if not hasattr(Image, 'LINEAR'):
    Image.LINEAR = Image.BILINEAR

from PIL import ImageFont
if not hasattr(ImageFont.FreeTypeFont, "getsize"):
    def getsize(self, text):
        bbox = self.getbbox(text)  # returns (left, top, right, bottom)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return (width, height)
    ImageFont.FreeTypeFont.getsize = getsize

from iopath.common.file_io import HTTPURLHandler
_original_http_get_local_path = HTTPURLHandler._get_local_path
def patched_http_get_local_path(self, path, force=False, **kwargs):
    if isinstance(path, str) and '?' in path:
        path = path.split('?')[0]
    return _original_http_get_local_path(self, path, force=force, **kwargs)
HTTPURLHandler._get_local_path = patched_http_get_local_path
# --- End of Monkey-Patching Section ---

# Import the cell extractor from our module.
from table_cell_extractor import extract_table_cell_bboxes

# --- Configuration ---
input_pdf_dir = "input_pdfs"           # Folder containing PDFs
parsed_layout_dir = "parsed_layouts"     # Folder with parsed layout files (.lo)
bounding_boxes_dir = "bounding_boxes"    # Folder to save bounding box info (.bb files)
pdfs_with_bb_dir = "pdfs_with_bounding_boxes"  # Folder to save PDFs with overlaid bounding boxes
temp_bbox_dir = "temp_bbox"              # Folder to save table pieces defined by bounding boxes
html_output_bbox_dir = "html_output_bbox"  # Folder to save HTML files for bounding box review

size_tolerance = 10 # if cell width or height is smaller than this, ignore it

# Toggle: if True, generate PDFs with bounding boxes overlaid.
generate_pdf_with_bb = True

# Rendering DPI for PDF conversion.
render_dpi = 300

# Parameter for expanding each cell bounding box.
epsilon = 5

# Create output directories if they don't exist.
for d in [bounding_boxes_dir, pdfs_with_bb_dir, temp_bbox_dir, html_output_bbox_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

# Process each PDF in input_pdf_dir.
for pdf_filename in os.listdir(input_pdf_dir):
    if not pdf_filename.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(input_pdf_dir, pdf_filename)
    base_name = os.path.splitext(pdf_filename)[0]
    print(f"Processing PDF: {pdf_filename}")

    # Load the corresponding parsed layout file (with extension .lo).
    layout_file = os.path.join(parsed_layout_dir, f"{base_name}.lo")
    if not os.path.exists(layout_file):
        print(f"Parsed layout file for {pdf_filename} not found. Skipping.")
        continue

    with open(layout_file, "r", encoding="utf-8") as f:
        parsed_pages = json.load(f)  # Expected to be a list of pages.

    # Convert PDF pages to images.
    try:
        pages = convert_from_path(pdf_path, dpi=render_dpi)
    except Exception as e:
        print(f"Error converting PDF {pdf_filename}: {e}")
        continue

    pdf_bb_info = []  # Will hold bounding boxes info for each page.
    overlay_pages = []  # For optionally generating a PDF with bounding boxes.

    # Process each page as defined in the parsed layout file.
    for page_info in parsed_pages:
        page_num = page_info.get("page")
        layout_elements = page_info.get("layout", [])
        print(f"Processing page {page_num}...")

        try:
            page = pages[page_num - 1]
        except IndexError:
            print(f"Page {page_num} not found in rendered images. Skipping page.")
            continue

        page_img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
        page_overlay = page_img.copy()
        page_tables_info = []  # Will hold table cell bounding boxes for this page.

        # For each layout element in this page, process table elements.
        for elem in layout_elements:
            # Here we expect table elements to be dictionaries with keys: "x_1", "y_1", "x_2", "y_2"
            if elem.get("type", "").lower() != "table":
                continue

            # Extract table coordinates from parsed layout file.
            try:
                tx1 = int(round(elem.get("x_1")))
                ty1 = int(round(elem.get("y_1")))
                tx2 = int(round(elem.get("x_2")))
                ty2 = int(round(elem.get("y_2")))
            except Exception as e:
                print(f"Error reading table coordinates: {e}")
                continue

            # Crop the table region from the page image.
            table_region = page_img[ty1:ty2, tx1:tx2]
            # Run our opencv-based table structure analyzer.
            cell_bboxes, grid_img, grid_shape = extract_table_cell_bboxes(
                table_region,
                min_line_length=150,
                merge_threshold=20,
                text_roi_x_min=None,  # Only morphological lines in this version.
                text_roi_x_max=None,
                epsilon=epsilon
            )
            # Convert cell bounding boxes to absolute coordinates.
            abs_cell_bboxes = []
            for bbox_idx, bbox in enumerate(cell_bboxes):
                cx1, cy1, cx2, cy2 = bbox
                if cx2 - cx1 < size_tolerance or cy2 - cy1 < size_tolerance:
                    continue
                abs_bbox = (tx1 + cx1, ty1 + cy1, tx1 + cx2, ty1 + cy2)
                # Draw bounding box on the page overlay for visualization.
                cv2.rectangle(page_overlay, (abs_bbox[0], abs_bbox[1]), (abs_bbox[2], abs_bbox[3]), (0, 0, 255), 2)
                # Save the table piece defined by the bounding box to a separate file.
                bbox_img = page_img[abs_bbox[1]:abs_bbox[3], abs_bbox[0]:abs_bbox[2]]
                bbox_img_path = os.path.join(temp_bbox_dir, f"{base_name}_page_{page_num}_bbox_{bbox_idx}.png")
                cv2.imwrite(bbox_img_path, bbox_img)
                # Add the image path to the bounding box info.
                abs_bbox_info = {
                    "bbox": abs_bbox,
                    "image_path": bbox_img_path
                }
                abs_cell_bboxes.append(abs_bbox_info)
            # Save table info.
            page_tables_info.append({
                "table_coordinates": [tx1, ty1, tx2, ty2],
                "grid_shape": grid_shape,
                "cell_bboxes": abs_cell_bboxes
            })
        # End for each layout element.
        pdf_bb_info.append({
            "page": page_num,
            "tables": page_tables_info
        })
        overlay_pages.append(page_overlay)
    # End for each page.

    # Save the bounding box information for this PDF as a JSON file with extension ".bb".
    bb_output_path = os.path.join(bounding_boxes_dir, f"{base_name}.bb")
    with open(bb_output_path, "w", encoding="utf-8") as f:
        json.dump(pdf_bb_info, f, indent=2)
    print(f"Saved bounding box info to {bb_output_path}")

    # Generate HTML file for bounding box review.
    html_output_path = os.path.join(html_output_bbox_dir, f"{base_name}.html")
    with open(html_output_path, "w", encoding="utf-8") as f:
        f.write("<html><body>\n")
        f.write(f"<h1>Bounding Box Results for {pdf_filename}</h1>\n")
        for page in pdf_bb_info:
            f.write(f"<h2>Page {page['page']}</h2>\n")
            for table in page["tables"]:
                f.write("<table border='1'>\n")
                for cell in table["cell_bboxes"]:
                    f.write("<tr>\n")
                    f.write(f"<td><img src='../{cell['image_path']}'></td>\n")
                    f.write(f"<td>{cell['bbox']}</td>\n")
                    f.write("</tr>\n")
                f.write("</table>\n")
        f.write("</body></html>\n")
    print(f"Generated HTML review file: {html_output_path}")

    # If toggle is on, generate a PDF with bounding boxes overlaid.
    if generate_pdf_with_bb and overlay_pages:
        temp_overlay_dir = os.path.join("temp_overlays", base_name)
        os.makedirs(temp_overlay_dir, exist_ok=True)
        overlay_paths = []
        for idx, overlay_img in enumerate(overlay_pages, start=1):
            overlay_path = os.path.join(temp_overlay_dir, f"page_{idx}.png")
            cv2.imwrite(overlay_path, overlay_img)
            overlay_paths.append(overlay_path)
        output_pdf_path = os.path.join(pdfs_with_bb_dir, f"{base_name}.pdf")
        try:
            with open(output_pdf_path, "wb") as f_out:
                f_out.write(img2pdf.convert(overlay_paths))
            print(f"Generated PDF with bounding boxes: {output_pdf_path}")
        except Exception as e:
            print(f"Error generating PDF for {pdf_filename}: {e}")

print("Processing complete.")