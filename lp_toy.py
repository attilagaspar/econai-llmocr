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
import matplotlib.pyplot as plt
import json
import os
import pytesseract
# Define the path to your PDF file.
pdf_path = 'raw/1935mg_osszeiras_sample-2-9-1.pdf'
#pdf_path = "raw/MagyarCompass_1936_2__pages354-354.pdf"

doc_tag = "mg_osszeiras"
#doc_tag = "compass"
# Locate tesseract manually
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# need ta add Hungarian mode to tessdata dir

# Convert PDF pages to images (using 300 DPI for clarity).
pages = convert_from_path(pdf_path, dpi=300)

# For this example, use the first page.
page_image = np.array(pages[0])
# Convert from RGB (PIL default) to BGR (OpenCV default).
page_image = cv2.cvtColor(page_image, cv2.COLOR_RGB2BGR)

# Get image dimensions.
img_height, img_width = page_image.shape[:2]

# Initialize Layout Parser's Detectron2-based layout model using the PubLayNet configuration.
model = lp.Detectron2LayoutModel(
    'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
)

# Detect the layout on the image.
layout = model.detect(page_image)

# --- Functions to Merge Adjacent Table Elements ---

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
        # Create a new coordinate object using lp.Rectangle with unpacked coordinates.
        new_block = lp.Rectangle(*current_box)
        # Create a new layout element with the merged coordinates.
        new_table = current.__class__(new_block, score=current_score, type=current.type)
        merged_tables.append(new_table)
    
    return non_table_elements + merged_tables

# --- End of Merge Functions ---

# Merge adjacent table elements.
merged_layout = merge_adjacent_tables(layout, gap_threshold=10)

# --- Save Parsed Layout Details to a Text File ---
output_text_file = f"output/{doc_tag}_parsed_layout.txt"
with open(output_text_file, "w", encoding="utf-8") as f:
    for idx, element in enumerate(merged_layout):
        f.write(f"Element {idx}:\n")
        f.write(f"   Type: {element.type}\n")
        f.write(f"   Score: {element.score:.3f}\n")
        f.write(f"   Coordinates: {element.coordinates}\n")
        f.write("\n")
print(f"Layout details saved to {output_text_file}")

# --- Generate COCO-Formatted JSON Output ---

# Define image information.
# We'll use the first page's image information.
image_info = {
    "id": 1,
    "file_name": "page_1.png",  # you can change this to the actual file name if saving the image
    "width": img_width,
    "height": img_height
}

# Define a mapping from layout element types to COCO category IDs.
# Adjust these as needed.
category_map = {
    "text": 1,
    "title": 2,
    "list": 3,
    "table": 4,
    "figure": 5
}

# Create the categories list for COCO.
categories = [{"id": cid, "name": name} for name, cid in category_map.items()]

# Prepare the annotations list.
annotations = []
ann_id = 1
for element in merged_layout:
    # Assume element.coordinates is a list: [x1, y1, x2, y2]
    # x1, y1, x2, y2 = element.coordinates
    x1, y1, x2, y2 = map(int, element.coordinates)
    bbox = [x1, y1, x2 - x1, y2 - y1]  # COCO expects [x, y, width, height]
    # Use the element type (converted to lower case) to map to a category_id.
    cat_id = category_map.get(element.type.lower(), 1)
    annotation = {
        "id": ann_id,
        "image_id": image_info["id"],
        "category_id": cat_id,
        "bbox": bbox,
        "score": element.score,  # optional: include confidence score
        # Optional: include recognized text if available, e.g., "text": element.text
        "text": ""  # if not available, leave empty or remove this key
    }
    annotations.append(annotation)
    ann_id += 1

    # Save cropped image of layout element to a file.
    cropped_region = page_image[y1:y2, x1:x2]
    output_file = f"output/cropped_images/{doc_tag}_element_{ann_id}.png"

    cv2.imwrite(output_file, cropped_region)
    img = Image.open(output_file)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # OCR using pytesseract
    cropped_region = page_image[y1:y2, x1:x2]
    print(f"Recognizing Text for Element {ann_id}")
    ocr_text = pytesseract.image_to_string(cropped_region, lang='hun')
    
    #print("Recognized Text:", ocr_text)
    with open(f"output/{doc_tag}_ocroutput_{ann_id}.txt", "w", encoding="utf-8") as f:
        f.write(ocr_text)


# Assemble the final COCO-style dictionary.
coco_dict = {
    "images": [image_info],
    "annotations": annotations,
    "categories": categories
}

# Save the COCO annotations to a JSON file.
coco_output_file = f"output/{doc_tag}_coco_annotations.json"
with open(coco_output_file, "w", encoding="utf-8") as f:
    json.dump(coco_dict, f, indent=2)
print(f"COCO annotation file saved to {coco_output_file}")

# --- Visualization ---

# Visualize the (merged) detected layout using Layout Parser's drawing function.
viz_image = lp.draw_box(page_image, merged_layout, box_width=3, show_element_type=True)

# Ensure viz_image is a NumPy array.
if not isinstance(viz_image, np.ndarray):
    viz_image = np.array(viz_image)

# (Optional) Convert the visualization image from BGR to RGB.
viz_image_rgb = cv2.cvtColor(viz_image, cv2.COLOR_BGR2RGB)

# Save the visualization to an image file.
output_image_file = f"output/{doc_tag}_detected_layout.png"
cv2.imwrite(output_image_file, cv2.cvtColor(viz_image_rgb, cv2.COLOR_RGB2BGR))
print(f"Layout visualization saved to {output_image_file}")
