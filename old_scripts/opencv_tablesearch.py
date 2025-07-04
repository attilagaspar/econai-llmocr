# This script processes looks processes Table layout elements from parsed layout files.
# It finds table cells using an OpenCV-based method and saves the bounding box information.
# It also optionally generates PDFs with bounding boxes overlaid for review.
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
            merged.append(int(round(sum(group) / len(group))))
            group = [c]
    merged.append(int(round(sum(group) / len(group))))
    return merged

# Directories
input_dir = "output/cropped_images"
cells_dir = "output/cells_bboxes"

if not os.path.exists(cells_dir):
    os.makedirs(cells_dir)

# Parameters
min_line_length = 150   # minimum length (in pixels) to keep a line
merge_threshold  = 20  # threshold for merging similar line coordinates

# Supported image extensions
img_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

for filename in os.listdir(input_dir):
    if filename.lower().endswith(img_extensions):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not load image: {filename}")
            continue

        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Adaptive thresholding (invert so that lines are white)
        binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, blockSize=15, C=-2)

        # ----------------------------------
        # Detect Horizontal Lines
        # ----------------------------------
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        hor_lines = cv2.erode(binary, horizontal_kernel, iterations=1)
        hor_lines = cv2.dilate(hor_lines, horizontal_kernel, iterations=1)
        filtered_horizontal = np.zeros_like(hor_lines)
        hor_y_coords = []
        contours, _ = cv2.findContours(hor_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w >= min_line_length:
                # Record the center y-coordinate of the horizontal line
                hor_y_coords.append(int(y + h/2))
                cv2.drawContours(filtered_horizontal, [cnt], -1, 255, thickness=cv2.FILLED)

        # ----------------------------------
        # Detect Vertical Lines
        # ----------------------------------
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

        # ----------------------------------
        # Determine Table Boundaries
        # ----------------------------------
        union = cv2.add(filtered_horizontal, filtered_vertical)
        nonzero = cv2.findNonZero(union)
        if nonzero is not None:
            table_x, table_y, table_w, table_h = cv2.boundingRect(nonzero)
        else:
            table_x, table_y, table_w, table_h = 0, 0, img.shape[1], img.shape[0]

        # ----------------------------------
        # Merge overlapping line coordinates (NMS)
        # ----------------------------------
        merged_hor = nms_line_coords(hor_y_coords, threshold=merge_threshold)
        merged_ver = nms_line_coords(ver_x_coords, threshold=merge_threshold)

        # To compute cell boundaries, include the table edges.
        final_hor = sorted([table_y] + merged_hor + [table_y + table_h])
        final_ver = sorted([table_x] + merged_ver + [table_x + table_w])

        # Compute cell bounding boxes from the grid defined by final_hor and final_ver.
        cell_bboxes = []
        for i in range(len(final_ver) - 1):
            for j in range(len(final_hor) - 1):
                x1 = final_ver[i]
                y1 = final_hor[j]
                x2 = final_ver[i+1]
                y2 = final_hor[j+1]
                cell_bboxes.append((x1, y1, x2, y2))

        # Visualize the cells on a copy of the original image.
        img_cells = img.copy()
        for (x1, y1, x2, y2) in cell_bboxes:
            cv2.rectangle(img_cells, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

        # Optionally, print out cell bounding boxes.
        print(f"{filename} - Number of detected cells: {len(cell_bboxes)}")
        for bbox in cell_bboxes:
            print(bbox)

        # Save the final visualization image.
        output_path = os.path.join(cells_dir, filename)
        cv2.imwrite(output_path, img_cells)
        print(f"Saved cell bounding box visualization for {filename} -> {output_path}")
