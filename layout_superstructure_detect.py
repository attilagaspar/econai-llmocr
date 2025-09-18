import sys
import os
import json
import numpy as np
import shutil

def assign_super_columns_and_rows(labelme_json, start_tol=10):
    shapes = [s for s in labelme_json["shapes"] if s.get("label") in ("numerical_cell", "column_header")]

    # Save original coordinates before smoothing
    for s in shapes:
        s["raw"] = [list(map(int, p)) for p in s["points"]]

    # Assign super_column numbers
    remaining_shapes = shapes[:]
    current_column = 1
    while remaining_shapes:
        leftmost_x = min(min(p[0] for p in shape["points"]) for shape in remaining_shapes)
        column_shapes = []
        for shape in remaining_shapes:
            x_min = min(p[0] for p in shape["points"])
            x_max = max(p[0] for p in shape["points"])
            if x_min <= leftmost_x + start_tol and x_max >= leftmost_x:
                shape["super_column"] = current_column
                column_shapes.append(shape)
        avg_x1 = sum(min(p[0] for p in shape["points"]) for shape in column_shapes) / len(column_shapes)
        avg_x2 = sum(max(p[0] for p in shape["points"]) for shape in column_shapes) / len(column_shapes)
        additional_shapes = []
        for shape in remaining_shapes:
            if shape in column_shapes:
                continue
            x_min = min(p[0] for p in shape["points"])
            x_max = max(p[0] for p in shape["points"])
            centroid_x = (x_min + x_max) / 2
            if avg_x1 <= centroid_x <= avg_x2:
                shape["super_column"] = current_column
                additional_shapes.append(shape)
        column_shapes.extend(additional_shapes)
        remaining_shapes = [shape for shape in remaining_shapes if shape not in column_shapes]
        current_column += 1

    # Assign super_row numbers
    remaining_shapes = shapes[:]
    current_row = 1
    while remaining_shapes:
        topmost_y = min(min(p[1] for p in shape["points"]) for shape in remaining_shapes)
        row_shapes = []
        for shape in remaining_shapes:
            y_min = min(p[1] for p in shape["points"])
            y_max = max(p[1] for p in shape["points"])
            if y_min <= topmost_y + start_tol and y_max >= topmost_y:
                shape["super_row"] = current_row
                row_shapes.append(shape)
        avg_y1 = sum(min(p[1] for p in shape["points"]) for shape in row_shapes) / len(row_shapes)
        avg_y2 = sum(max(p[1] for p in shape["points"]) for shape in row_shapes) / len(row_shapes)
        additional_shapes = []
        for shape in remaining_shapes:
            if shape in row_shapes:
                continue
            y_min = min(p[1] for p in shape["points"])
            y_max = max(p[1] for p in shape["points"])
            centroid_y = (y_min + y_max) / 2
            if avg_y1 <= centroid_y <= avg_y2:
                shape["super_row"] = current_row
                additional_shapes.append(shape)
        row_shapes.extend(additional_shapes)
        remaining_shapes = [shape for shape in remaining_shapes if shape not in row_shapes]
        current_row += 1

    # Update the original JSON shapes with super_row and super_column
    for s in labelme_json["shapes"]:
        for ss in shapes:
            if s is ss:
                s["super_row"] = ss.get("super_row")
                s["super_column"] = ss.get("super_column")
                s["raw"] = ss.get("raw")

    # --- Smoothing step ---
    # Smooth vertical coordinates for each super_row
    for row in set(s.get("super_row") for s in shapes if "super_row" in s):
        row_shapes = [s for s in shapes if s.get("super_row") == row]
        upper_ys = [min(p[1] for p in s["points"]) for s in row_shapes]
        lower_ys = [max(p[1] for p in s["points"]) for s in row_shapes]
        median_upper = int(np.median(upper_ys))
        median_lower = int(np.median(lower_ys))
        for s in row_shapes:
            p1, p2 = s["points"]
            if p1[1] < p2[1]:
                s["points"][0][1] = median_upper
                s["points"][1][1] = median_lower
            else:
                s["points"][1][1] = median_upper
                s["points"][0][1] = median_lower

    # Smooth horizontal coordinates for each super_column
    for col in set(s.get("super_column") for s in shapes if "super_column" in s):
        col_shapes = [s for s in shapes if s.get("super_column") == col]
        left_xs = [min(p[0] for p in s["points"]) for s in col_shapes]
        right_xs = [max(p[0] for p in s["points"]) for s in col_shapes]
        median_left = int(np.median(left_xs))
        median_right = int(np.median(right_xs))
        for s in col_shapes:
            p1, p2 = s["points"]
            if p1[0] < p2[0]:
                s["points"][0][0] = median_left
                s["points"][1][0] = median_right
            else:
                s["points"][1][0] = median_left
                s["points"][0][0] = median_right

    # --- Missing cell prediction ---
    predict_missing_cells(labelme_json, shapes)


def predict_missing_cells(labelme_json, shapes):
    """Predict and add missing cells based on superstructure analysis"""
    superstructure_cell_types = ["numerical_cell", "column_header"]
    
    print("Starting missing cell prediction...")
    
    for cell_type in superstructure_cell_types:
        print(f"\nAnalyzing cell type: {cell_type}")
        
        # Get all shapes of this type with superstructure coordinates
        type_shapes = [s for s in shapes if s.get("label") == cell_type and 
                      "super_row" in s and "super_column" in s]
        
        if not type_shapes:
            print(f"No {cell_type} shapes found with superstructure coordinates")
            continue
            
        # Find all contiguous blocks of this cell type
        blocks = find_contiguous_blocks(type_shapes, cell_type)
        
        # Sort blocks by size (largest first)
        blocks.sort(key=lambda b: len(b), reverse=True)
        
        for i, block in enumerate(blocks):
            print(f"Processing block {i+1} of {cell_type} with {len(block)} cells")
            predicted_cells = predict_missing_in_block(block, cell_type, labelme_json)
            
            # Add predicted cells to the JSON and update the JSON for future overlap checks
            for predicted_cell in predicted_cells:
                labelme_json["shapes"].append(predicted_cell)
                print(f"Added predicted cell: {predicted_cell['label']} at row {predicted_cell.get('super_row')}, col {predicted_cell.get('super_column')}")
                # The predicted cell is now part of labelme_json, so future predictions will avoid it


def find_contiguous_blocks(shapes, cell_type):
    """Find all contiguous rectangular blocks of the same cell type"""
    if not shapes:
        return []
    
    # Create a simpler approach: find the overall bounding box and treat it as one block
    # This is more aggressive but will catch more missing cells
    all_rows = [s.get("super_row") for s in shapes]
    all_cols = [s.get("super_column") for s in shapes]
    
    min_row, max_row = min(all_rows), max(all_rows)
    min_col, max_col = min(all_cols), max(all_cols)
    
    # Create blocks based on contiguous regions
    blocks = []
    processed = set()
    
    # Use flood-fill approach to find connected components
    for shape in shapes:
        if id(shape) in processed:
            continue
            
        # Start a new block with flood-fill
        block = []
        queue = [shape]
        
        while queue:
            current = queue.pop(0)
            if id(current) in processed:
                continue
                
            processed.add(id(current))
            block.append(current)
            
            current_row = current.get("super_row")
            current_col = current.get("super_column")
            
            # Find adjacent cells (allowing for some gaps)
            for candidate in shapes:
                if id(candidate) in processed:
                    continue
                    
                candidate_row = candidate.get("super_row")
                candidate_col = candidate.get("super_column")
                
                # More permissive adjacency - allow cells within 2 positions
                row_distance = abs(candidate_row - current_row)
                col_distance = abs(candidate_col - current_col)
                
                if (row_distance <= 2 and col_distance <= 2) and (row_distance + col_distance <= 3):
                    queue.append(candidate)
        
        if block:
            blocks.append(block)
    
    # Also create one large block encompassing all cells to ensure comprehensive coverage
    if len(shapes) > 1:
        blocks.append(shapes)  # Add all shapes as one big block
    
    return blocks


def predict_missing_in_block(block, cell_type, labelme_json):
    """Predict missing cells within a contiguous block"""
    predicted_cells = []
    
    if not block:
        return predicted_cells
    
    # Get all cells of any type to understand the full table structure
    all_cells = [s for s in labelme_json["shapes"] if 
                "super_row" in s and "super_column" in s and 
                s.get("label") in ["numerical_cell", "column_header", "numerical_cell_predicted", "column_header_predicted"]]
    
    if all_cells:
        # Use the full table boundaries for comprehensive prediction
        all_rows = [s.get("super_row") for s in all_cells]
        all_cols = [s.get("super_column") for s in all_cells]
        global_min_row, global_max_row = min(all_rows), max(all_rows)
        global_min_col, global_max_col = min(all_cols), max(all_cols)
    else:
        # Fall back to block boundaries
        rows = [s.get("super_row") for s in block]
        cols = [s.get("super_column") for s in block]
        global_min_row, global_max_row = min(rows), max(rows)
        global_min_col, global_max_col = min(cols), max(cols)
    
    # Get the bounding rectangle of the current block
    rows = [s.get("super_row") for s in block]
    cols = [s.get("super_column") for s in block]
    
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    
    print(f"Block spans rows {min_row}-{max_row}, columns {min_col}-{max_col}")
    print(f"Global table spans rows {global_min_row}-{global_max_row}, columns {global_min_col}-{global_max_col}")
    
    # Check each position in the block's bounding rectangle AND extend to global boundaries
    # if the block is near the edges
    extend_min_row = min(min_row, global_min_row) if abs(min_row - global_min_row) <= 2 else min_row
    extend_max_row = max(max_row, global_max_row) if abs(max_row - global_max_row) <= 2 else max_row
    extend_min_col = min(min_col, global_min_col) if abs(min_col - global_min_col) <= 2 else min_col
    extend_max_col = max(max_col, global_max_col) if abs(max_col - global_max_col) <= 2 else max_col
    
    existing_positions = set((s.get("super_row"), s.get("super_column")) for s in block)
    
    for row in range(extend_min_row, extend_max_row + 1):
        for col in range(extend_min_col, extend_max_col + 1):
            if (row, col) not in existing_positions:
                # Check if any shape exists at this position (including predicted cells)
                position_occupied = any(
                    s.get("super_row") == row and s.get("super_column") == col 
                    for s in labelme_json["shapes"] 
                    if "super_row" in s and "super_column" in s
                )
                
                if not position_occupied:
                    # Additional check: is this position reasonable to predict?
                    # Only predict if there are nearby cells of the same type
                    nearby_same_type = any(
                        s.get("label") == cell_type and
                        abs(s.get("super_row") - row) <= 1 and 
                        abs(s.get("super_column") - col) <= 1
                        for s in block
                    )
                    
                    if nearby_same_type or len(block) == 1:  # Always predict for single cells
                        print(f"Predicting missing {cell_type} at row {row}, column {col}")
                        predicted_cell = create_predicted_cell(row, col, block, cell_type, labelme_json)
                        if predicted_cell is not None:  # Only add if not completely overlapped
                            predicted_cells.append(predicted_cell)
    
    return predicted_cells


def create_predicted_cell(target_row, target_col, block, cell_type, labelme_json):
    """Create a predicted cell based on coordinates from existing cells"""
    
    # Find cells in the same row and column for coordinate reference
    same_row_cells = [s for s in block if s.get("super_row") == target_row]
    same_col_cells = [s for s in block if s.get("super_column") == target_col]
    
    # If no cells in same row/column, use nearby cells
    if not same_row_cells:
        same_row_cells = [s for s in block if abs(s.get("super_row") - target_row) <= 1]
    if not same_col_cells:
        same_col_cells = [s for s in block if abs(s.get("super_column") - target_col) <= 1]
    
    # Calculate predicted coordinates
    if same_row_cells:
        # Use Y coordinates from same row
        y_coords = []
        for s in same_row_cells:
            y_coords.extend([p[1] for p in s["points"]])
        pred_y_min = min(y_coords)
        pred_y_max = max(y_coords)
    else:
        # Estimate from all cells
        all_heights = []
        for s in block:
            y_coords = [p[1] for p in s["points"]]
            all_heights.append(max(y_coords) - min(y_coords))
        avg_height = sum(all_heights) / len(all_heights)
        
        # Estimate Y position
        row_diff = target_row - block[0].get("super_row")
        ref_y = min(p[1] for p in block[0]["points"])
        pred_y_min = int(ref_y + row_diff * avg_height)
        pred_y_max = int(pred_y_min + avg_height)
    
    if same_col_cells:
        # Use X coordinates from same column
        x_coords = []
        for s in same_col_cells:
            x_coords.extend([p[0] for p in s["points"]])
        pred_x_min = min(x_coords)
        pred_x_max = max(x_coords)
    else:
        # Estimate from all cells
        all_widths = []
        for s in block:
            x_coords = [p[0] for p in s["points"]]
            all_widths.append(max(x_coords) - min(x_coords))
        avg_width = sum(all_widths) / len(all_widths)
        
        # Estimate X position
        col_diff = target_col - block[0].get("super_column")
        ref_x = min(p[0] for p in block[0]["points"])
        pred_x_min = int(ref_x + col_diff * avg_width)
        pred_x_max = int(pred_x_min + avg_width)
    
    # Create initial predicted rectangle
    predicted_rect = [pred_x_min, pred_y_min, pred_x_max, pred_y_max]
    
    # Check for overlaps with existing cells and trim if necessary
    trimmed_rect = trim_overlapping_areas(predicted_rect, labelme_json, target_row, target_col)
    
    if trimmed_rect is None:
        # The predicted cell would be completely overlapped or too small
        return None
    
    # Create the predicted cell structure
    predicted_cell = {
        "label": f"{cell_type}_predicted",
        "points": [
            [trimmed_rect[0], trimmed_rect[1]],
            [trimmed_rect[2], trimmed_rect[3]]
        ],
        "group_id": None,
        "shape_type": "rectangle",
        "flags": {},
        "super_row": target_row,
        "super_column": target_col
    }
    
    return predicted_cell


def trim_overlapping_areas(predicted_rect, labelme_json, target_row, target_col):
    """Trim predicted rectangle to avoid overlaps with existing cells"""
    x_min, y_min, x_max, y_max = predicted_rect
    
    # Get all existing cells that have coordinates
    existing_cells = [s for s in labelme_json["shapes"] if "points" in s and len(s["points"]) >= 2]
    
    # Find overlapping cells
    overlapping_cells = []
    for cell in existing_cells:
        cell_x_min = min(p[0] for p in cell["points"])
        cell_y_min = min(p[1] for p in cell["points"])
        cell_x_max = max(p[0] for p in cell["points"])
        cell_y_max = max(p[1] for p in cell["points"])
        
        # Check if rectangles overlap
        if not (x_max <= cell_x_min or x_min >= cell_x_max or y_max <= cell_y_min or y_min >= cell_y_max):
            overlapping_cells.append((cell_x_min, cell_y_min, cell_x_max, cell_y_max))
    
    if not overlapping_cells:
        return predicted_rect  # No overlaps, return original
    
    # Trim the predicted rectangle to avoid overlaps
    trimmed_rect = [x_min, y_min, x_max, y_max]
    
    for overlap_x_min, overlap_y_min, overlap_x_max, overlap_y_max in overlapping_cells:
        # Calculate overlap area
        overlap_left = max(trimmed_rect[0], overlap_x_min)
        overlap_top = max(trimmed_rect[1], overlap_y_min)
        overlap_right = min(trimmed_rect[2], overlap_x_max)
        overlap_bottom = min(trimmed_rect[3], overlap_y_max)
        
        if overlap_left < overlap_right and overlap_top < overlap_bottom:
            # There is an overlap, we need to trim
            print(f"  Overlap detected at row {target_row}, col {target_col}. Trimming...")
            
            # Try different trimming strategies and keep the one that preserves the most area
            original_area = (trimmed_rect[2] - trimmed_rect[0]) * (trimmed_rect[3] - trimmed_rect[1])
            best_rect = None
            best_area = 0
            
            # Strategy 1: Trim from left
            if overlap_left > trimmed_rect[0]:
                left_trimmed = [overlap_right, trimmed_rect[1], trimmed_rect[2], trimmed_rect[3]]
                if left_trimmed[0] < left_trimmed[2]:  # Valid rectangle
                    area = (left_trimmed[2] - left_trimmed[0]) * (left_trimmed[3] - left_trimmed[1])
                    if area > best_area:
                        best_area = area
                        best_rect = left_trimmed
            
            # Strategy 2: Trim from right
            if overlap_right < trimmed_rect[2]:
                right_trimmed = [trimmed_rect[0], trimmed_rect[1], overlap_left, trimmed_rect[3]]
                if right_trimmed[0] < right_trimmed[2]:  # Valid rectangle
                    area = (right_trimmed[2] - right_trimmed[0]) * (right_trimmed[3] - right_trimmed[1])
                    if area > best_area:
                        best_area = area
                        best_rect = right_trimmed
            
            # Strategy 3: Trim from top
            if overlap_top > trimmed_rect[1]:
                top_trimmed = [trimmed_rect[0], overlap_bottom, trimmed_rect[2], trimmed_rect[3]]
                if top_trimmed[1] < top_trimmed[3]:  # Valid rectangle
                    area = (top_trimmed[2] - top_trimmed[0]) * (top_trimmed[3] - top_trimmed[1])
                    if area > best_area:
                        best_area = area
                        best_rect = top_trimmed
            
            # Strategy 4: Trim from bottom
            if overlap_bottom < trimmed_rect[3]:
                bottom_trimmed = [trimmed_rect[0], trimmed_rect[1], trimmed_rect[2], overlap_top]
                if bottom_trimmed[1] < bottom_trimmed[3]:  # Valid rectangle
                    area = (bottom_trimmed[2] - bottom_trimmed[0]) * (bottom_trimmed[3] - bottom_trimmed[1])
                    if area > best_area:
                        best_area = area
                        best_rect = bottom_trimmed
            
            if best_rect and best_area > original_area * 0.3:  # Keep if at least 30% of original area
                trimmed_rect = best_rect
            else:
                print(f"  Predicted cell at row {target_row}, col {target_col} would be too small after trimming. Skipping.")
                return None
    
    # Final validation: ensure the trimmed rectangle is reasonable
    final_width = trimmed_rect[2] - trimmed_rect[0]
    final_height = trimmed_rect[3] - trimmed_rect[1]
    
    if final_width < 10 or final_height < 10:  # Minimum size threshold
        print(f"  Predicted cell at row {target_row}, col {target_col} too small after trimming ({final_width}x{final_height}). Skipping.")
        return None
    
    return trimmed_rect


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python add_super_rowcol.py input_folder output_folder")
        sys.exit(1)
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    os.makedirs(output_folder, exist_ok=True)
    for root, _, files in os.walk(input_folder):
        # Compute relative path to preserve folder structure
        rel_dir = os.path.relpath(root, input_folder)
        out_dir = os.path.join(output_folder, rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        for fname in files:
            if fname.lower().endswith(".json"):
                in_path = os.path.join(root, fname)
                out_path = os.path.join(out_dir, fname)
                with open(in_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "shapes" not in data:
                    print(f"Skipping {in_path}: no 'shapes' element.")
                    continue
                assign_super_columns_and_rows(data, start_tol=10)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                print(f"Saved with super_row and super_column (smoothed):{out_path}")
                # Also copy the corresponding JPG (or JPEG/PNG) to the output folder
                base = os.path.splitext(fname)[0]
                for ext in [".jpg", ".jpeg", ".png"]:
                    img_in_path = os.path.join(root, base + ext)
                    if os.path.exists(img_in_path):
                        img_out_path = os.path.join(out_dir, base + ext)
                        shutil.copy2(img_in_path, img_out_path)
                        print(f"Copied image: {img_out_path}")
                        break