# --- Monkey-Patching Section ---

# 1. Ensure Pillow's Image module has the attribute 'LINEAR' (needed by Detectron2)
import PIL.Image as Image
if not hasattr(Image, 'LINEAR'):
    Image.LINEAR = Image.BILINEAR

# 2. Monkey-patch FreeTypeFont to add getsize() if missing (using getbbox)
from PIL import ImageFont
if not hasattr(ImageFont.FreeTypeFont, "getsize"):
    def getsize(self, text):
        # getbbox returns (left, top, right, bottom)
        bbox = self.getbbox(text)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return (width, height)
    ImageFont.FreeTypeFont.getsize = getsize

# 3. (Optional) Patch iopath's HTTPURLHandler to remove query parameters from cached filenames.
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

# Define the path to your PDF file.
pdf_path = 'raw/1935mg_osszeiras_sample-2-9-1.pdf'

# Convert PDF pages to images (using 300 DPI for clarity).
pages = convert_from_path(pdf_path, dpi=300)

# For this example, we work with the first page.
page_image = np.array(pages[0])
# Convert from RGB (PIL default) to BGR (OpenCV default)
page_image = cv2.cvtColor(page_image, cv2.COLOR_RGB2BGR)

# Initialize Layout Parser's Detectron2-based layout model using the PubLayNet configuration.
model = lp.Detectron2LayoutModel(
    'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
)

# Detect the layout on the image.
layout = model.detect(page_image)

# --- Save Parsed Layout Details to a Text File ---
output_text_file = "parsed_layout.txt"
with open(output_text_file, "w", encoding="utf-8") as f:
    for idx, element in enumerate(layout):
        # Each 'element' is typically a LayoutElement with attributes like:
        #   - type: the detected label (e.g., "Text", "Title", etc.)
        #   - score: the detection confidence score
        #   - coordinates: the bounding box coordinates (usually [x1, y1, x2, y2])
        f.write(f"Element {idx}:\n")
        f.write(f"   Type: {element.type}\n")
        f.write(f"   Score: {element.score:.3f}\n")
        f.write(f"   Coordinates: {element.coordinates}\n")
        f.write("\n")
print(f"Layout details saved to {output_text_file}")


# Visualize the detected layout using Layout Parser's built-in drawing function.
viz_image = lp.draw_box(page_image, layout, box_width=3, show_element_type=True)

# Ensure viz_image is a NumPy array.
if not isinstance(viz_image, np.ndarray):
    viz_image = np.array(viz_image)

# If needed, convert the image from BGR to RGB.
# (If you prefer to save in the standard OpenCV BGR color space, you can omit this step.)
viz_image_rgb = cv2.cvtColor(viz_image, cv2.COLOR_BGR2RGB)

# Save the visualization to an image file.
output_filename = "detected_layout.png"
# cv2.imwrite expects a BGR image, so we convert the RGB image back to BGR.
cv2.imwrite(output_filename, cv2.cvtColor(viz_image_rgb, cv2.COLOR_RGB2BGR))

print(f"Layout visualization saved to {output_filename}")
