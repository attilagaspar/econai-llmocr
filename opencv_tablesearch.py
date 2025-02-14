import cv2
import numpy as np
import os

# Directories
input_dir = "output/cropped_images"
output_dir = "output/detected_table_lines"

# Create output directory if it doesn't exist.
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Parameter: minimum line length (in pixels) required to keep a detected line.
min_line_length = 200  # Adjust this value as needed.

# Supported image extensions.
img_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

# Loop through each image file in the input directory.
for filename in os.listdir(input_dir):
    if filename.lower().endswith(img_extensions):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not load image: {filename}")
            continue
        
        # Convert the image to grayscale.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding (invert the image so that lines become white).
        binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, blockSize=15, C=-2)
        
        # ---------------------------
        # Detect Horizontal Lines
        # ---------------------------
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.erode(binary, horizontal_kernel, iterations=1)
        horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=1)
        
        # Filter horizontal lines by minimum length.
        filtered_horizontal = np.zeros_like(horizontal_lines)
        contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w >= min_line_length:
                cv2.drawContours(filtered_horizontal, [cnt], -1, 255, thickness=cv2.FILLED)
        
        # ---------------------------
        # Detect Vertical Lines
        # ---------------------------
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        vertical_lines = cv2.erode(binary, vertical_kernel, iterations=1)
        vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=1)
        
        # Filter vertical lines by minimum length.
        filtered_vertical = np.zeros_like(vertical_lines)
        contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h >= min_line_length:
                cv2.drawContours(filtered_vertical, [cnt], -1, 255, thickness=cv2.FILLED)
        
        # ---------------------------
        # Combine Filtered Lines
        # ---------------------------
        table_lines = cv2.add(filtered_horizontal, filtered_vertical)
        
        # Save the output image with the detected table lines.
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, table_lines)
        print(f"Processed {filename} with min_line_length={min_line_length} -> {output_path}")
