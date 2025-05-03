import os
import json
import img2pdf
from pdf2image import convert_from_path
import layoutparser as lp
import numpy as np
import cv2
import fitz  # PyMuPDF


# --- Configuration ---
input_pdf_dir = "censuspdf"         # Folder containing PDFs
output_dir = "output"                # Folder where output will be saved
parsed_layout_dir = "parsed_layouts"   # Folder where .lo files will be saved
pdfs_with_layouts_dir = "pdfs_with_layouts"  # (Optional) Folder for PDFs with overlaid layouts

# Toggle: If True, generate new PDFs with the layouts overlaid.
generate_pdf_with_layouts = True

# Create output directories if they don't exist.
for d in [output_dir, parsed_layout_dir, pdfs_with_layouts_dir]:
    if not os.path.exists(d):
        os.makedirs(d)


# --- Helper Function for Image Compression ---
def save_compressed_image(image, output_path, quality=75):
    """
    Save an image with JPEG compression.

    Args:
        image (numpy.ndarray): The image to save.
        output_path (str): The path to save the compressed image.
        quality (int): JPEG quality (1-100, higher is better quality).
    """
    cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])


# --- Main Processing ---
annotation_id = 1

# For each PDF in input_pdf_dir, process it.
for pdf_filename in os.listdir(input_pdf_dir):
    if not pdf_filename.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(input_pdf_dir, pdf_filename)
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
            {"id": 0, "name": "column_header"},
            {"id": 1, "name": "numerical_cell"},
            {"id": 2, "name": "text_cell"}
        ]
    }

    # Process each page.
    for page_num, page in enumerate(pages, start=1):
        # Convert PIL image to numpy array (BGR)
        page_img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
        
        # Save the page image as a compressed JPG
        image_path = os.path.join(images_dir, f"page_{page_num}.jpg")
        save_compressed_image(page_img, image_path, quality=75)

        # Run layout parser on the page image.
        layout = model.detect(page_img)
        
        # Merge adjacent table elements.
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
            category_id = elem.type  # Assuming elem.type is already mapped to category_id
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
        color_map = {
            0: (255, 0, 0),  # Red for column headers
            1: (0, 255, 0),  # Green for numerical cells
            2: (0, 0, 255)   # Blue for text cells
        }
        overlay = page_img.copy()

        # Draw bounding boxes and write scores for each element in the merged layout
        for elem in merged_layout:
            x1, y1, x2, y2 = map(int, elem.coordinates)  # Ensure coordinates are integers
            box_color = color_map.get(elem.type, (0, 0, 0))  # Default to black if type is unknown
            cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, thickness=3)

        # Add the overlay to the layout images
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
            overlay_path = os.path.join(temp_overlay_dir, f"page_{idx}.jpg")
            save_compressed_image(overlay_img, overlay_path, quality=75)
            overlay_paths.append(overlay_path)
        
        output_pdf_path = os.path.join(pdfs_with_layouts_dir, f"{base_name}.pdf")
        try:
            with open(output_pdf_path, "wb") as f_out:
                f_out.write(img2pdf.convert(overlay_paths))
            print(f"Generated PDF with layouts: {output_pdf_path}")
        except Exception as e:
            print(f"Error generating PDF for {pdf_filename}: {e}")
    
print("Processing complete.")