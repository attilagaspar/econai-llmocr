import cv2
import numpy as np
import os

def nms_line_coords(coords, threshold=10):
    """
    Given a list of coordinates (ints), sort and merge those that are within 'threshold' pixels.
    Returns a list of averaged coordinates.
    """
    if not coords:
        return []
    coords = sorted(coords)
    merged = []
    group = [coords[0]]
    for c in coords[1:]:
        if abs(c - group[-1]) < threshold:
            group.append(c)
        else:
            merged.append(int(round(sum(group)/len(group))))
            group = [c]
    merged.append(int(round(sum(group)/len(group))))
    return merged

# Directories for input and outputs
input_dir = "output/cropped_images"
interim_dir = "output/detected_table_lines"    # Interim version: detected lines (filtered, not extended)
final_dir   = "output/extended_table_lines"      # Final version: extended and merged lines

# Create output directories if they don't exist.
for directory in [interim_dir, final_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Parameter: minimum line length (in pixels) required to keep a detected line.
min_line_length = 150  # adjust as needed
# Parameter for merging lines: if two lines are within this many pixels, merge them.
merge_threshold = 20  # adjust as needed

# Supported image extensions.
img_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

for filename in os.listdir(input_dir):
    if filename.lower().endswith(img_extensions):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not load image: {filename}")
            continue

        # Convert image to grayscale.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding (invert image so lines become white).
        binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, blockSize=15, C=-2)
        
        # ---------------------------
        # Detect Horizontal Lines
        # ---------------------------
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        hor_lines = cv2.erode(binary, horizontal_kernel, iterations=1)
        hor_lines = cv2.dilate(hor_lines, horizontal_kernel, iterations=1)
        filtered_horizontal = np.zeros_like(hor_lines)
        hor_y_coords = []
        contours, _ = cv2.findContours(hor_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w >= min_line_length:
                hor_y_coords.append(int(y + h/2))
                cv2.drawContours(filtered_horizontal, [cnt], -1, 255, thickness=cv2.FILLED)
        
        # ---------------------------
        # Detect Vertical Lines
        # ---------------------------
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        ver_lines = cv2.erode(binary, vertical_kernel, iterations=1)
        ver_lines = cv2.dilate(ver_lines, vertical_kernel, iterations=1)
        filtered_vertical = np.zeros_like(ver_lines)
        ver_x_coords = []
        contours, _ = cv2.findContours(ver_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h >= min_line_length:
                ver_x_coords.append(int(x + w/2))
                cv2.drawContours(filtered_vertical, [cnt], -1, 255, thickness=cv2.FILLED)
        
        # ---------------------------
        # Save the Interim Detected Lines
        # ---------------------------
        # Combine the filtered horizontal and vertical lines to form the initial detected table grid.
        interim_mask = cv2.add(filtered_horizontal, filtered_vertical)
        interim_output_path = os.path.join(interim_dir, filename)
        cv2.imwrite(interim_output_path, interim_mask)
        print(f"Interim detected lines saved: {interim_output_path}")
        
        # ---------------------------
        # Determine Table Boundaries
        # ---------------------------
        union = cv2.add(filtered_horizontal, filtered_vertical)
        nonzero = cv2.findNonZero(union)
        if nonzero is not None:
            table_x, table_y, table_w, table_h = cv2.boundingRect(nonzero)
        else:
            table_x, table_y, table_w, table_h = 0, 0, img.shape[1], img.shape[0]
        
        # ---------------------------
        # Merge overlapping line coordinates using NMS
        # ---------------------------
        merged_hor = nms_line_coords(hor_y_coords, threshold=merge_threshold)
        merged_ver = nms_line_coords(ver_x_coords, threshold=merge_threshold)
        
        # ---------------------------
        # Draw Extended Lines
        # ---------------------------
        extended_mask = np.zeros_like(gray)
        # Extend horizontal lines: draw a line from the left to right edge of the table.
        for y in merged_hor:
            cv2.line(extended_mask, (table_x, y), (table_x + table_w, y), 255, thickness=2)
        # Extend vertical lines: draw a line from the top to bottom edge of the table.
        for x in merged_ver:
            cv2.line(extended_mask, (x, table_y), (x, table_y + table_h), 255, thickness=2)
        
        # ---------------------------
        # Optional: Remove duplicates via connected components if needed.
        # ---------------------------
        # For now, we assume that the NMS step is sufficient.
        
        final_output_path = os.path.join(final_dir, filename)
        cv2.imwrite(final_output_path, extended_mask)
        print(f"Final extended lines saved: {final_output_path}")
