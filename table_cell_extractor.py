import cv2
import numpy as np
import pytesseract
import layoutparser as lp

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

def extract_table_cell_bboxes(img, min_line_length=150, merge_threshold=20, 
                                text_roi_x_min=None, text_roi_x_max=None, epsilon=0):
    """
    Given a table image (cropped table region as a BGR image), this function:
      1. Detects horizontal and vertical lines using morphological operations.
      2. (For now, it ignores text-based horizontal boundaries.)
      3. Uses only the morphological horizontal boundaries and vertical boundaries (plus the table edges)
         to produce the final grid.
      4. Computes cell bounding boxes from the grid.
      5. Expands each bounding box by 'epsilon' pixels in all directions.
      6. Draws the grid on a copy of the input image for visualization.
    
    Parameters:
      - img: Input table image (BGR, numpy array).
      - min_line_length: Minimum pixel length to keep a detected line.
      - merge_threshold: Threshold (in pixels) for merging nearby boundaries.
      - text_roi_x_min, text_roi_x_max: (Unused in this version) Provided for compatibility.
      - epsilon: Number of pixels to expand each cell's bounding box in all directions.
    
    Returns:
      - cell_bboxes: List of tuples (x1, y1, x2, y2) for each cell (coordinates relative to img, expanded by epsilon).
      - grid_img: BGR image with the grid overlaid. Horizontal lines are drawn in blue; vertical lines in green.
      - grid_shape: Tuple (n_rows, n_cols) representing the grid dimensions.
    """
    # Convert to grayscale and apply adaptive thresholding.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, blockSize=15, C=-2)
    
    # ---------------------------
    # Morphological Horizontal Lines
    # ---------------------------
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    hor_lines = cv2.erode(binary, horizontal_kernel, iterations=1)
    hor_lines = cv2.dilate(hor_lines, horizontal_kernel, iterations=1)
    hor_y = []
    contours, _ = cv2.findContours(hor_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= min_line_length:
            hor_y.append(int(y + h/2))
    
    # ---------------------------
    # Morphological Vertical Lines
    # ---------------------------
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    ver_lines = cv2.erode(binary, vertical_kernel, iterations=1)
    ver_lines = cv2.dilate(ver_lines, vertical_kernel, iterations=1)
    ver_x = []
    contours, _ = cv2.findContours(ver_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h >= min_line_length:
            ver_x.append(int(x + w/2))
    
    # ---------------------------
    # Define Table Boundaries
    # ---------------------------
    table_x, table_y_edge, table_w, table_h = 0, 0, img.shape[1], img.shape[0]
    
    # ---------------------------
    # Combine Horizontal Boundaries (morphological only)
    # ---------------------------
    candidate_hor = nms_line_coords(hor_y, threshold=merge_threshold) + [table_y_edge, table_y_edge + table_h]
    final_hor = sorted(list(set(candidate_hor)))
    
    # ---------------------------
    # Combine Vertical Boundaries (morphological only)
    # ---------------------------
    candidate_ver = ver_x + [table_x, table_x + table_w]
    final_ver = sorted(list(set(nms_line_coords(candidate_ver, threshold=merge_threshold))))
    
    n_rows = len(final_hor) - 1
    n_cols = len(final_ver) - 1
    
    # Compute cell bounding boxes (row-major order), then expand each by epsilon.
    cell_bboxes = []
    for j in range(n_rows):
        for i in range(n_cols):
            x1 = final_ver[i]
            y1 = final_hor[j]
            x2 = final_ver[i+1]
            y2 = final_hor[j+1]
            exp_x1 = max(0, x1 - epsilon)
            exp_y1 = max(0, y1 - epsilon)
            exp_x2 = min(img.shape[1], x2 + epsilon)
            exp_y2 = min(img.shape[0], y2 + epsilon)
            cell_bboxes.append((exp_x1, exp_y1, exp_x2, exp_y2))
    
    # ---------------------------
    # Create Grid Visualization Image
    # ---------------------------
    grid_img = img.copy()
    # Draw vertical lines in green.
    for x in final_ver:
        cv2.line(grid_img, (x, table_y_edge), (x, table_y_edge + table_h), (0, 255, 0), 2)
    # Draw horizontal lines in blue.
    for y in final_hor:
        cv2.line(grid_img, (table_x, y), (table_x + table_w, y), (255, 0, 0), 2)
    
    return cell_bboxes, grid_img, (n_rows, n_cols)
