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
    
    # --- Additional comprehensive gap filling ---
    comprehensive_gap_filling(labelme_json)


def predict_missing_cells(labelme_json, shapes):
    """Predict ALL missing cells in the superstructure lattice"""
    superstructure_cell_types = ["numerical_cell", "column_header"]
    
    print("Starting comprehensive lattice-based prediction...")
    
    # Get all cells with superstructure coordinates to determine lattice bounds
    all_super_cells = [s for s in labelme_json["shapes"] if 
                      "super_row" in s and "super_column" in s and 
                      s.get("label") in superstructure_cell_types]
    
    if not all_super_cells:
        print("No cells with superstructure found")
        return
    
    # Determine the complete lattice bounds
    all_rows = [s.get("super_row") for s in all_super_cells]
    all_cols = [s.get("super_column") for s in all_super_cells]
    min_row, max_row = min(all_rows), max(all_rows)
    min_col, max_col = min(all_cols), max(all_cols)
    
    print(f"Complete superstructure lattice: rows {min_row}-{max_row}, columns {min_col}-{max_col}")
    print(f"Total lattice positions: {(max_row - min_row + 1) * (max_col - min_col + 1)}")
    
    # Create a map of existing positions
    position_to_cell = {}
    for cell in all_super_cells:
        pos = (cell.get("super_row"), cell.get("super_column"))
        position_to_cell[pos] = cell
    
    print(f"Existing cells: {len(position_to_cell)}")
    
    # Predict every missing position in the lattice
    total_predicted = 0
    
    for row in range(min_row, max_row + 1):
        for col in range(min_col, max_col + 1):
            pos = (row, col)
            
            if pos not in position_to_cell:
                # This position is missing - predict it!
                predicted_type = determine_cell_type_for_position(row, col, position_to_cell, min_row, max_row, min_col, max_col)
                
                print(f"Predicting {predicted_type} at lattice position ({row}, {col})")
                
                # Find reference cells for coordinate calculation
                reference_cells = find_reference_cells_for_position(row, col, all_super_cells, predicted_type)
                
                if reference_cells:
                    predicted_cell = create_predicted_cell(row, col, reference_cells, predicted_type, labelme_json)
                    
                    if predicted_cell is not None:
                        # Try to fit the cell by finding maximum non-overlapping rectangle
                        final_cell = find_maximum_non_overlapping_rectangle(predicted_cell, labelme_json)
                        
                        if final_cell is not None:
                            labelme_json["shapes"].append(final_cell)
                            all_super_cells.append(final_cell)  # Add to working list for future reference
                            position_to_cell[pos] = final_cell
                            total_predicted += 1
                            print(f"✓ Added {predicted_type}_predicted at ({row}, {col})")
                        else:
                            print(f"✗ Could not fit non-overlapping rectangle at ({row}, {col})")
                    else:
                        print(f"✗ Could not create cell at ({row}, {col})")
                else:
                    print(f"✗ No reference cells found for ({row}, {col})")
    
    print(f"\nLattice prediction completed. Added {total_predicted} predicted cells.")
    print(f"Final coverage: {len(position_to_cell)} / {(max_row - min_row + 1) * (max_col - min_col + 1)} positions")


def determine_cell_type_for_position(row, col, position_to_cell, min_row, max_row, min_col, max_col):
    """Determine what type of cell should be at this position"""
    
    # Strategy 1: Check same row for type patterns
    row_types = []
    for c in range(min_col, max_col + 1):
        if (row, c) in position_to_cell:
            cell = position_to_cell[(row, c)]
            cell_type = cell.get("label", "").replace("_predicted", "")
            if cell_type in ["numerical_cell", "column_header"]:
                row_types.append(cell_type)
    
    # Strategy 2: Check same column for type patterns
    col_types = []
    for r in range(min_row, max_row + 1):
        if (r, col) in position_to_cell:
            cell = position_to_cell[(r, col)]
            cell_type = cell.get("label", "").replace("_predicted", "")
            if cell_type in ["numerical_cell", "column_header"]:
                col_types.append(cell_type)
    
    # Strategy 3: Use surrounding cells
    surrounding_types = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            check_pos = (row + dr, col + dc)
            if check_pos in position_to_cell:
                cell = position_to_cell[check_pos]
                cell_type = cell.get("label", "").replace("_predicted", "")
                if cell_type in ["numerical_cell", "column_header"]:
                    surrounding_types.append(cell_type)
    
    # Decision logic: prefer majority vote, with preferences
    all_types = row_types + col_types + surrounding_types
    
    if all_types:
        type_counts = {}
        for t in all_types:
            type_counts[t] = type_counts.get(t, 0) + 1
        
        # Use most common type, prefer numerical_cell if tied
        predicted_type = max(type_counts.keys(), 
                           key=lambda x: (type_counts[x], x == "numerical_cell"))
    else:
        # Default to numerical_cell if no context
        predicted_type = "numerical_cell"
    
    return predicted_type


def find_reference_cells_for_position(row, col, all_cells, predicted_type):
    """Find the best reference cells for coordinate calculation"""
    
    # Prioritize cells of the same type
    same_type_cells = [s for s in all_cells if 
                      s.get("label", "").replace("_predicted", "") == predicted_type]
    
    if same_type_cells:
        # Find cells in same row and column
        same_row = [s for s in same_type_cells if s.get("super_row") == row]
        same_col = [s for s in same_type_cells if s.get("super_column") == col]
        
        if same_row or same_col:
            return same_row + same_col
    
    # Fallback: use nearby cells of any type
    nearby_cells = [s for s in all_cells if 
                   abs(s.get("super_row") - row) <= 2 and 
                   abs(s.get("super_column") - col) <= 2]
    
    return nearby_cells if nearby_cells else all_cells[:5]  # Last resort: any cells


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
                    # More aggressive check: predict if there are any cells of the same type nearby
                    # or if this position is surrounded by cells (indicating it's inside the table)
                    nearby_same_type = any(
                        (s.get("label") == cell_type or s.get("label") == f"{cell_type}_predicted") and
                        abs(s.get("super_row") - row) <= 2 and 
                        abs(s.get("super_column") - col) <= 2
                        for s in labelme_json["shapes"] 
                        if "super_row" in s and "super_column" in s
                    )
                    
                    # Check if surrounded by any cells (indicating it's inside a table structure)
                    surrounding_cells = 0
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            surrounding_occupied = any(
                                s.get("super_row") == row + dr and s.get("super_column") == col + dc
                                for s in labelme_json["shapes"] 
                                if "super_row" in s and "super_column" in s
                            )
                            if surrounding_occupied:
                                surrounding_cells += 1
                    
                    # More aggressive prediction criteria
                    should_predict = (
                        nearby_same_type or 
                        surrounding_cells >= 2 or  # Reduced from 3 to 2
                        len(block) == 1 or  # Always predict for single cells
                        # Additional criterion: if we're inside the table boundaries
                        (extend_min_row < row < extend_max_row and extend_min_col < col < extend_max_col)
                    )
                    
                    if should_predict:
                        print(f"Predicting missing {cell_type} at row {row}, column {col}")
                        predicted_cell = create_predicted_cell(row, col, block, cell_type, labelme_json)
                        if predicted_cell is not None:
                            # Try to fit the cell by finding maximum non-overlapping rectangle
                            final_cell = find_maximum_non_overlapping_rectangle(predicted_cell, labelme_json)
                            if final_cell is not None:
                                predicted_cells.append(final_cell)
                                print(f"  ✓ Added prediction at row {row}, col {col}")
                            else:
                                print(f"  ✗ Could not fit non-overlapping rectangle at row {row}, col {col}")
                        # Only add if not completely overlapped
    
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
    
    # Create the predicted cell structure WITHOUT trimming overlaps
    # Let the spatial overlap check handle this instead
    predicted_cell = {
        "label": f"{cell_type}_predicted",
        "points": [
            [predicted_rect[0], predicted_rect[1]],
            [predicted_rect[2], predicted_rect[3]]
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


def comprehensive_gap_filling(labelme_json):
    """Comprehensive gap filling with strict overlap prevention"""
    print("\nComprehensive gap-filling pass...")
    
    # Get all cells with superstructure coordinates
    all_cells = [s for s in labelme_json["shapes"] if 
                "super_row" in s and "super_column" in s and 
                s.get("label") in ["numerical_cell", "column_header", "numerical_cell_predicted", "column_header_predicted"]]
    
    if not all_cells:
        return
    
    # Get the full table boundaries
    all_rows = [s.get("super_row") for s in all_cells]
    all_cols = [s.get("super_column") for s in all_cells]
    min_row, max_row = min(all_rows), max(all_rows)
    min_col, max_col = min(all_cols), max(all_cols)
    
    print(f"Table grid spans rows {min_row}-{max_row}, columns {min_col}-{max_col}")
    
    # Create a comprehensive grid analysis
    occupied_positions = set()
    position_to_cell = {}
    
    for cell in all_cells:
        pos = (cell.get("super_row"), cell.get("super_column"))
        occupied_positions.add(pos)
        position_to_cell[pos] = cell
    
    gaps_filled = 0
    max_gap_fill_iterations = 2
    
    for iteration in range(max_gap_fill_iterations):
        print(f"\nGap-filling iteration {iteration + 1}")
        cells_added_this_round = 0
        
        # Find all empty positions in the grid
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                if (row, col) not in occupied_positions:
                    # Analyze this empty position
                    should_predict, predicted_type = analyze_empty_position(
                        row, col, occupied_positions, position_to_cell, 
                        min_row, max_row, min_col, max_col
                    )
                    
                    if should_predict and predicted_type:
                        # Create the prediction with strict overlap checking
                        predicted_cell = create_gap_fill_prediction(
                            row, col, predicted_type, all_cells, labelme_json
                        )
                        
                        if predicted_cell is not None:
                            # Try to fit the cell by finding maximum non-overlapping rectangle
                            final_cell = find_maximum_non_overlapping_rectangle(predicted_cell, labelme_json)
                            if final_cell is not None:
                                labelme_json["shapes"].append(final_cell)
                                all_cells.append(final_cell)  # Update working list
                                occupied_positions.add((row, col))
                                position_to_cell[(row, col)] = final_cell
                                cells_added_this_round += 1
                                print(f"Gap-filled: {predicted_type}_predicted at row {row}, col {col}")
                            else:
                                print(f"Could not fit gap-fill rectangle at row {row}, col {col}")
        
        print(f"Gap-filling iteration {iteration + 1} added {cells_added_this_round} cells")
        if cells_added_this_round == 0:
            break
        gaps_filled += cells_added_this_round
    
    print(f"Comprehensive gap-filling completed. Total filled: {gaps_filled} gaps.")


def analyze_empty_position(row, col, occupied_positions, position_to_cell, 
                          min_row, max_row, min_col, max_col):
    """Analyze if an empty position should be predicted and what type"""
    
    # Don't predict on the absolute edges unless there's strong evidence
    is_edge = (row == min_row or row == max_row or col == min_col or col == max_col)
    
    # Count surrounding cells and their types
    surrounding_positions = [
        (row-1, col), (row+1, col), (row, col-1), (row, col+1),  # Adjacent
        (row-1, col-1), (row-1, col+1), (row+1, col-1), (row+1, col+1)  # Diagonal
    ]
    
    adjacent_cells = 0
    diagonal_cells = 0
    cell_types = []
    
    for i, (sr, sc) in enumerate(surrounding_positions):
        if (sr, sc) in occupied_positions:
            cell = position_to_cell[(sr, sc)]
            label = cell.get("label", "").replace("_predicted", "")
            if label in ["numerical_cell", "column_header"]:
                cell_types.append(label)
                if i < 4:  # Adjacent positions
                    adjacent_cells += 1
                else:  # Diagonal positions
                    diagonal_cells += 1
    
    total_surrounding = adjacent_cells + diagonal_cells
    
    # Decision logic
    should_predict = False
    predicted_type = None
    
    if not is_edge:
        # Interior position - more lenient
        if adjacent_cells >= 2 or total_surrounding >= 4:
            should_predict = True
    else:
        # Edge position - require stronger evidence
        if adjacent_cells >= 3 or total_surrounding >= 6:
            should_predict = True
    
    # Determine cell type
    if should_predict and cell_types:
        # Use most common type, prefer numerical_cell
        type_counts = {}
        for t in cell_types:
            type_counts[t] = type_counts.get(t, 0) + 1
        
        if type_counts:
            predicted_type = max(type_counts.keys(), 
                               key=lambda x: (type_counts[x], x == "numerical_cell"))
    
    return should_predict, predicted_type


def create_gap_fill_prediction(row, col, cell_type, all_cells, labelme_json):
    """Create a predicted cell for gap filling"""
    
    # Find reference cells for coordinate calculation
    reference_cells = [s for s in all_cells if 
                      abs(s.get("super_row") - row) <= 2 and 
                      abs(s.get("super_column") - col) <= 2 and
                      s.get("label", "").replace("_predicted", "") == cell_type]
    
    if not reference_cells:
        # Fallback to any nearby cells
        reference_cells = [s for s in all_cells if 
                          abs(s.get("super_row") - row) <= 2 and 
                          abs(s.get("super_column") - col) <= 2]
    
    if not reference_cells:
        return None
    
    # Use create_predicted_cell but with stricter overlap checking
    predicted_cell = create_predicted_cell(row, col, reference_cells, cell_type, labelme_json)
    
    return predicted_cell


def find_maximum_non_overlapping_rectangle(predicted_cell, labelme_json):
    """Find the maximum rectangle that doesn't overlap with existing cells"""
    if not predicted_cell or "points" not in predicted_cell:
        return None
    
    pred_points = predicted_cell["points"]
    orig_x1, orig_y1 = min(p[0] for p in pred_points), min(p[1] for p in pred_points)
    orig_x2, orig_y2 = max(p[0] for p in pred_points), max(p[1] for p in pred_points)
    
    pred_row = predicted_cell.get("super_row")
    pred_col = predicted_cell.get("super_column")
    
    print(f"    Finding max non-overlapping rectangle for ({pred_row}, {pred_col}): original bbox ({orig_x1}, {orig_y1}) to ({orig_x2}, {orig_y2})")
    
    # Get all existing shapes that could potentially overlap
    existing_rects = []
    for shape in labelme_json["shapes"]:
        if "points" not in shape or len(shape["points"]) < 2:
            continue
        if shape is predicted_cell:  # Don't check against itself
            continue
            
        shape_points = shape["points"]
        shape_x1, shape_y1 = min(p[0] for p in shape_points), min(p[1] for p in shape_points)
        shape_x2, shape_y2 = max(p[0] for p in shape_points), max(p[1] for p in shape_points)
        
        # Only consider shapes that actually overlap with our original rectangle
        if not (orig_x2 <= shape_x1 or orig_x1 >= shape_x2 or orig_y2 <= shape_y1 or orig_y1 >= shape_y2):
            existing_rects.append((shape_x1, shape_y1, shape_x2, shape_y2))
            shape_label = shape.get("label", "?")
            shape_row = shape.get("super_row", "?")
            shape_col = shape.get("super_column", "?")
            print(f"      Overlapping with {shape_label} at ({shape_row}, {shape_col}): bbox ({shape_x1}, {shape_y1}) to ({shape_x2}, {shape_y2})")
    
    if not existing_rects:
        print(f"      No overlaps found - returning original rectangle")
        return predicted_cell
    
    # Try different strategies to find the maximum non-overlapping rectangle
    best_rect = None
    best_area = 0
    
    # Strategy 1: Trim from each side and find the best result
    strategies = [
        ("left", lambda x1, y1, x2, y2, ox1, oy1, ox2, oy2: (max(x1, ox2), y1, x2, y2)),
        ("right", lambda x1, y1, x2, y2, ox1, oy1, ox2, oy2: (x1, y1, min(x2, ox1), y2)),
        ("top", lambda x1, y1, x2, y2, ox1, oy1, ox2, oy2: (x1, max(y1, oy2), x2, y2)),
        ("bottom", lambda x1, y1, x2, y2, ox1, oy1, ox2, oy2: (x1, y1, x2, min(y2, oy1))),
    ]
    
    for strategy_name, trim_func in strategies:
        test_rect = [orig_x1, orig_y1, orig_x2, orig_y2]
        
        # Apply this trimming strategy against all overlapping rectangles
        for ox1, oy1, ox2, oy2 in existing_rects:
            new_x1, new_y1, new_x2, new_y2 = trim_func(test_rect[0], test_rect[1], test_rect[2], test_rect[3], ox1, oy1, ox2, oy2)
            
            # Ensure valid rectangle
            if new_x1 < new_x2 and new_y1 < new_y2:
                test_rect = [new_x1, new_y1, new_x2, new_y2]
            else:
                # This strategy failed
                test_rect = None
                break
        
        if test_rect:
            # Check that this rectangle doesn't overlap with any existing shapes
            rect_valid = True
            for ox1, oy1, ox2, oy2 in existing_rects:
                if not (test_rect[2] <= ox1 or test_rect[0] >= ox2 or test_rect[3] <= oy1 or test_rect[1] >= oy2):
                    rect_valid = False
                    break
            
            if rect_valid:
                area = (test_rect[2] - test_rect[0]) * (test_rect[3] - test_rect[1])
                print(f"      Strategy '{strategy_name}': valid rectangle ({test_rect[0]}, {test_rect[1]}) to ({test_rect[2]}, {test_rect[3]}), area = {area}")
                
                if area > best_area:
                    best_area = area
                    best_rect = test_rect
            else:
                print(f"      Strategy '{strategy_name}': still overlapping")
        else:
            print(f"      Strategy '{strategy_name}': invalid rectangle")
    
    # Check if the best rectangle is worth keeping (at least 30% of original area)
    original_area = (orig_x2 - orig_x1) * (orig_y2 - orig_y1)
    min_area_threshold = original_area * 0.3
    min_dimension_threshold = 10  # Minimum width/height
    
    if best_rect and best_area >= min_area_threshold:
        width = best_rect[2] - best_rect[0]
        height = best_rect[3] - best_rect[1]
        
        if width >= min_dimension_threshold and height >= min_dimension_threshold:
            print(f"      Selected rectangle: ({best_rect[0]}, {best_rect[1]}) to ({best_rect[2]}, {best_rect[3]}), area = {best_area} ({best_area/original_area*100:.1f}% of original)")
            
            # Create the trimmed cell
            trimmed_cell = predicted_cell.copy()
            trimmed_cell["points"] = [
                [best_rect[0], best_rect[1]],
                [best_rect[2], best_rect[3]]
            ]
            return trimmed_cell
        else:
            print(f"      Best rectangle too small: {width}x{height} (min {min_dimension_threshold}x{min_dimension_threshold})")
    else:
        if best_rect:
            print(f"      Best rectangle area {best_area} < threshold {min_area_threshold} ({min_area_threshold/original_area*100:.1f}% of original)")
        else:
            print(f"      No valid rectangle found")
    
    return None


def check_spatial_overlap(predicted_cell, labelme_json):
    """Check if predicted cell spatially overlaps with any existing cell"""
    if not predicted_cell or "points" not in predicted_cell:
        return True  # Assume overlap if invalid
    
    pred_points = predicted_cell["points"]
    pred_x1, pred_y1 = min(p[0] for p in pred_points), min(p[1] for p in pred_points)
    pred_x2, pred_y2 = max(p[0] for p in pred_points), max(p[1] for p in pred_points)
    
    pred_row = predicted_cell.get("super_row")
    pred_col = predicted_cell.get("super_column")
    
    print(f"    Checking overlap for predicted cell at ({pred_row}, {pred_col}): bbox ({pred_x1}, {pred_y1}) to ({pred_x2}, {pred_y2})")
    
    # Check against all existing shapes with coordinates
    overlaps_found = 0
    for shape in labelme_json["shapes"]:
        if "points" not in shape or len(shape["points"]) < 2:
            continue
        if shape is predicted_cell:  # Don't check against itself
            continue
            
        shape_points = shape["points"]
        shape_x1, shape_y1 = min(p[0] for p in shape_points), min(p[1] for p in shape_points)
        shape_x2, shape_y2 = max(p[0] for p in shape_points), max(p[1] for p in shape_points)
        
        # Check for overlap
        if not (pred_x2 <= shape_x1 or pred_x1 >= shape_x2 or 
                pred_y2 <= shape_y1 or pred_y1 >= shape_y2):
            shape_row = shape.get("super_row", "?")
            shape_col = shape.get("super_column", "?")
            shape_label = shape.get("label", "?")
            print(f"    OVERLAP DETECTED with {shape_label} at ({shape_row}, {shape_col}): bbox ({shape_x1}, {shape_y1}) to ({shape_x2}, {shape_y2})")
            overlaps_found += 1
    
    if overlaps_found > 0:
        print(f"    Total overlaps found: {overlaps_found}")
        return True
    else:
        print(f"    No overlaps found - cell is safe to add")
        return False


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