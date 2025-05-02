#!/usr/bin/env python
# This script processes Table layout elements from parsed layout files.
# This script is part of the table extraction pipeline.
# It finds table cells using an OpenCV-based method and saves the bounding box information.
import os
import json
import img2pdf
from pdf2image import convert_from_path
import layoutparser as lp
import numpy as np
import cv2
import fitz  # PyMuPDF

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

# --- Configuration ---
input_pdf_dir = "../../census/pdf"         # Folder containing PDFs
output_dir = "output"                # Folder where output will be saved
parsed_layout_dir = "parsed_layouts"   # Folder where .lo files will be saved
pdfs_with_layouts_dir = "pdfs_with_layouts"  # (Optional) Folder for PDFs with overlaid layouts

# Toggle: If True, generate new PDFs with the layouts overlaid.
generate_pdf_with_layouts = True

# Create output directories if they don't exist.
for d in [output_dir, parsed_layout_dir, pdfs_with_layouts_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

# estimate dpi of a pdf page
def get_pdf_dpi(pdf_path, page_num):
    """
    Estimate the DPI of a PDF page.
    Args:
        pdf_path (str): Path to the PDF file.
        page_num (int): Page number (0-indexed).
    Returns:
        float: Estimated DPI of the PDF page.
    """
    pdf_doc = fitz.open(pdf_path)
    page = pdf_doc.load_page(page_num)
    
    # Get page dimensions in points (1 point = 1/72 inch)
    rect = page.rect
    width_in_points = rect.width
    height_in_points = rect.height
    
    # Render the page to an image
    pix = page.get_pixmap(dpi=300)  # Render at 300 DPI
    width_in_pixels = pix.width
    height_in_pixels = pix.height
    
    # Calculate DPI
    dpi_x = width_in_pixels / (width_in_points / 72)
    dpi_y = height_in_pixels / (height_in_points / 72)
    
    # Average DPI
    dpi = (dpi_x + dpi_y) / 2
    
    return dpi


# --- Merging Functions for Table Elements ---
def merge_boxes(box1, box2):
    """Given two bounding boxes [x1, y1, x2, y2], return their union box."""
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])
    return [x1, y1, x2, y2]

def are_adjacent(box1, box2, gap_threshold=10):
    """
    Check if two boxes (format [x1, y1, x2, y2]) are overlapping or adjacent.
    The gap_threshold parameter defines the maximum allowed gap (in pixels)
    to consider the boxes as part of the same table.
    """
    # Check if boxes intersect.
    intersect_x1 = max(box1[0], box2[0])
    intersect_y1 = max(box1[1], box2[1])
    intersect_x2 = min(box1[2], box2[2])
    intersect_y2 = min(box1[3], box2[3])
    if intersect_x1 < intersect_x2 and intersect_y1 < intersect_y2:
        return True

    # If not intersecting, check vertical gap.
    if box1[3] < box2[1]:
        vertical_gap = box2[1] - box1[3]
    elif box2[3] < box1[1]:
        vertical_gap = box1[1] - box2[3]
    else:
        vertical_gap = 0

    # Calculate horizontal overlap.
    horizontal_overlap = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    width1 = box1[2] - box1[0]
    width2 = box2[2] - box2[0]
    
    if vertical_gap <= gap_threshold and (horizontal_overlap >= 0.5 * width1 or horizontal_overlap >= 0.5 * width2):
        return True

    # Similarly, check horizontal gap for side-by-side boxes.
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
    """
    Given a list of layout elements, merge adjacent elements of type 'Table'.
    Returns a new list of layout elements where adjacent table elements have been merged.
    """
    # Separate table elements from others.
    table_elements = [elem for elem in layout_elements if elem.type.lower() == "table"]
    non_table_elements = [elem for elem in layout_elements if elem.type.lower() != "table"]
    
    merged_tables = []
    while table_elements:
        current = table_elements.pop(0)
        current_box = current.coordinates  # current_box is a list [x1, y1, x2, y2]
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

# --- End of Merging Functions ---

# --- Initialize Layout Parser Model ---
try:
    model = lp.Detectron2LayoutModel(
        config_path = "../layout-model-training/outputs/census/fast_rcnn_R_50_FPN_3x/config.yaml",
        model_path = "../layout-model-training/outputs/census/fast_rcnn_R_50_FPN_3x/model_0034999.pth",
        extra_config = ["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8] # <-- Only output high accuracy preds
    )
except FileNotFoundError as e:
    print(f"Error: Configuration or model file not found. {e}")
    raise
except Exception as e:
    print(f"Error initializing Detectron2LayoutModel: {e}")
    raise



annotation_id = 1

# For each PDF in input_pdf_dir, process it.
for pdf_filename in os.listdir(input_pdf_dir):
    if not pdf_filename.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(input_pdf_dir, pdf_filename)
    print(pdf_path, get_pdf_dpi(pdf_path, 1))
    base_name = os.path.splitext(pdf_filename)[0]
    print(f"Processing PDF: {pdf_filename}")
    
    # Create output subfolder for the current PDF
    pdf_output_dir = os.path.join(output_dir, base_name)
    images_dir = os.path.join(pdf_output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    # Convert all pages of the PDF to images.
    try:
        pages = convert_from_path(pdf_path, dpi=300)
    except Exception as e:
        print(f"Error converting PDF {pdf_filename}: {e}")
        continue

    parsed_pages = []  # To collect layout information for each page.
    layout_images = [] # If generating PDFs with layouts, collect overlay images.
    

    # COCO JSON structure
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "Figure"},
            {"id": 1, "name": "List"},
            {"id": 2, "name": "Table"},
            {"id": 3, "name": "Text"},
            {"id": 4, "name": "Title"}
        ]
    }

    # Process each page.
    for page_num, page in enumerate(pages, start=1):
        # Convert PIL image to numpy array (BGR)
        page_img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
        
        # Save the page image as JPG
        image_path = os.path.join(images_dir, f"page_{page_num}.jpg")
        cv2.imwrite(image_path, page_img)

        # Run layout parser on the page image.
        layout = model.detect(page_img)
        
        # Merge adjacent table elements.
        # merged_layout = merge_adjacent_tables(layout, gap_threshold=10)
        # new model does not have table layouts
        merged_layout = layout
        
        # Add image info to COCO JSON
        image_info = {
            "id": page_num,
            "file_name": f"page_{page_num}.jpg",
            "width": page_img.shape[1],
            "height": page_img.shape[0]
        }
        coco_output["images"].append(image_info)
        
        # For storage, convert each layout element to a dict (if available)
        for elem in merged_layout:
            x1, y1, x2, y2 = elem.coordinates
            width = x2 - x1
            height = y2 - y1
            # Map element type to category_id
            category_id = next((cat["id"] for cat in coco_output["categories"] if cat["id"] == elem.type), None)
            if category_id is None:
                print(f"Warning: Unknown element type '{elem.type}'")
                continue
            annotation = {
                "id": annotation_id,
                "image_id": page_num,
                "category_id": category_id,
                "bbox": [x1, y1, width, height],
                "area": width * height,
                "iscrowd": 0
            }
            coco_output["annotations"].append(annotation)
            annotation_id += 1
        
        # Optionally, generate an overlay image with the layout drawn.
        overlay = lp.draw_box(page_img, merged_layout, box_width=3, show_element_type=True)
        # Ensure overlay is a NumPy array.
        if not isinstance(overlay, np.ndarray):
            overlay = np.array(overlay)
        layout_images.append(overlay)
    
    # Save the parsed layout as a COCO JSON file.
    output_coco_path = os.path.join(pdf_output_dir, f"{base_name}.json")
    with open(output_coco_path, "w", encoding="utf-8") as f:
        json.dump(coco_output, f, indent=2)
    print(f"Saved parsed layout to: {output_coco_path}")
    
    # If the toggle is on, generate a new PDF with the layouts overlaid.
    if generate_pdf_with_layouts and layout_images:
        temp_overlay_dir = os.path.join("temp_overlays", base_name)
        os.makedirs(temp_overlay_dir, exist_ok=True)
        overlay_paths = []
        for idx, overlay_img in enumerate(layout_images, start=1):
            overlay_path = os.path.join(temp_overlay_dir, f"page_{idx}.png")
            cv2.imwrite(overlay_path, overlay_img)
            overlay_paths.append(overlay_path)
        
        output_pdf_path = os.path.join(pdfs_with_layouts_dir, f"{base_name}.pdf")
        try:
            with open(output_pdf_path, "wb") as f_out:
                f_out.write(img2pdf.convert(overlay_paths))
            print(f"Generated PDF with layouts: {output_pdf_path}")
        except Exception as e:
            print(f"Error generating PDF for {pdf_filename}: {e}")
    
print("Processing complete.")