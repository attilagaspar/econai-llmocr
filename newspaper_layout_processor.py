#!/usr/bin/env python3
"""
Newspaper Layout Column Detection and Correction Script

This script processes newspaper layout JSON files to:
1. Detect column structure (3 columns)
2. Assign column_number and row_number to layout elements
3. Correct "szoveg" element coordinates to extend to bottom of page
4. Copy processed files to output directory

Usage: python newspaper_layout_processor.py <input_folder> <output_folder>
"""

import json
import os
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def find_json_jpg_pairs(input_folder: str) -> List[Tuple[str, str]]:
    """Find all JSON files that have corresponding JPG files."""
    pairs = []
    
    print(f"🔍 Scanning directory: {input_folder}")
    
    for root, dirs, files in os.walk(input_folder):
        json_files = [f for f in files if f.lower().endswith('.json')]
        
        for json_file in json_files:
            json_path = os.path.join(root, json_file)
            base_name = os.path.splitext(json_file)[0]
            
            # Look for matching JPG
            jpg_path = None
            for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
                potential_jpg = os.path.join(root, base_name + ext)
                if os.path.exists(potential_jpg):
                    jpg_path = potential_jpg
                    break
            
            if jpg_path:
                pairs.append((json_path, jpg_path))
    
    print(f"✓ Found {len(pairs)} JSON-JPG pairs")
    return pairs


def get_element_center_x(shape: Dict) -> float:
    """Get the horizontal center coordinate of a shape."""
    points = shape.get("points", [])
    if len(points) >= 2:
        x_coords = [p[0] for p in points]
        return (min(x_coords) + max(x_coords)) / 2
    return 0.0


def get_element_bounds(shape: Dict) -> Tuple[float, float, float, float]:
    """Get bounding box coordinates (x1, y1, x2, y2) of a shape."""
    points = shape.get("points", [])
    if len(points) >= 2:
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        return min(x_coords), min(y_coords), max(x_coords), max(y_coords)
    return 0.0, 0.0, 0.0, 0.0


def detect_column_boundaries(shapes: List[Dict], image_width: float) -> Tuple[float, float]:
    """
    Detect column boundaries by analyzing the distribution of single-column elements.
    Returns the x-coordinates of the two column separators.
    """
    # Get center x-coordinates of single-column elements (szoveg and hasabkozi_cim)
    single_col_elements = [s for s in shapes if s.get("label") in ["szoveg", "hasabkozi_cim"]]
    
    if not single_col_elements:
        # Fallback: divide page into thirds
        return image_width / 3, 2 * image_width / 3
    
    centers = [get_element_center_x(shape) for shape in single_col_elements]
    centers.sort()
    
    if len(centers) < 3:
        # Fallback: divide page into thirds
        return image_width / 3, 2 * image_width / 3
    
    # Find the largest gaps in the sorted centers to identify column separators
    gaps = []
    for i in range(1, len(centers)):
        gap_size = centers[i] - centers[i-1]
        gap_mid = (centers[i] + centers[i-1]) / 2
        gaps.append((gap_size, gap_mid))
    
    # Sort gaps by size and take the two largest
    gaps.sort(reverse=True)
    
    if len(gaps) >= 2:
        # The midpoints of the two largest gaps are our column boundaries
        boundary1 = gaps[0][1]
        boundary2 = gaps[1][1]
        
        # Ensure proper order (left to right)
        boundary1, boundary2 = sorted([boundary1, boundary2])
        return boundary1, boundary2
    
    # Fallback: divide page into thirds
    return image_width / 3, 2 * image_width / 3


def assign_column_number(center_x: float, boundary1: float, boundary2: float) -> int:
    """Assign column number (1, 2, or 3) based on x-coordinate."""
    if center_x < boundary1:
        return 1
    elif center_x < boundary2:
        return 2
    else:
        return 3


def assign_row_numbers(shapes_by_column: Dict[int, List[Dict]]) -> None:
    """Assign row numbers to elements in each column, sorted by y-coordinate."""
    for column_num, shapes in shapes_by_column.items():
        # Sort by top y-coordinate (ascending - top to bottom)
        shapes.sort(key=lambda s: get_element_bounds(s)[1])
        
        # Assign row numbers starting from 1
        for i, shape in enumerate(shapes, 1):
            shape["row_number"] = i


def find_elements_below(shape: Dict, all_shapes: List[Dict], tolerance: float = 50) -> List[Dict]:
    """Find elements that are directly below the given shape."""
    x1, y1, x2, y2 = get_element_bounds(shape)
    shape_bottom = y2
    
    below_elements = []
    
    for other_shape in all_shapes:
        if other_shape is shape:
            continue
            
        other_x1, other_y1, other_x2, other_y2 = get_element_bounds(other_shape)
        
        # Check if the other element is below and overlaps horizontally
        if (other_y1 > shape_bottom - tolerance and 
            other_y1 < shape_bottom + tolerance * 3 and  # Allow more tolerance for detection
            not (other_x2 < x1 or other_x1 > x2)):  # Horizontal overlap check
            below_elements.append(other_shape)
    
    return below_elements


def correct_szoveg_coordinates(data: Dict) -> None:
    """
    Correct coordinates of 'szoveg' elements to extend to bottom of page.
    Only extends elements that are at the bottom of their column.
    """
    shapes = data.get("shapes", [])
    image_width = data.get("imageWidth", 3000)  # Default fallback
    
    # Detect column boundaries
    boundary1, boundary2 = detect_column_boundaries(shapes, image_width)
    
    # Group szoveg elements by column
    szoveg_by_column = {1: [], 2: [], 3: []}
    
    for shape in shapes:
        if shape.get("label") == "szoveg":
            center_x = get_element_center_x(shape)
            column = assign_column_number(center_x, boundary1, boundary2)
            szoveg_by_column[column].append(shape)
    
    # Find the bottommost szoveg in each column that has no elements below it
    bottommost_candidates = []
    
    for column_num, column_shapes in szoveg_by_column.items():
        if not column_shapes:
            continue
            
        # Sort by bottom coordinate (descending - bottommost first)
        column_shapes.sort(key=lambda s: get_element_bounds(s)[3], reverse=True)
        
        # Check the bottommost shape to see if it has anything below it
        bottommost_shape = column_shapes[0]
        elements_below = find_elements_below(bottommost_shape, shapes)
        
        if not elements_below:
            # This is the bottommost element in the column with nothing below
            bottommost_candidates.append(bottommost_shape)
            print(f"    📍 Column {column_num}: bottommost szoveg found (no elements below)")
        else:
            print(f"    📍 Column {column_num}: bottommost szoveg has {len(elements_below)} element(s) below it")
    
    if not bottommost_candidates:
        print("    ⚠️  No bottommost szoveg elements found for extension")
        return
    
    # Find the absolute bottom coordinate among all candidates
    bottom_coords = [get_element_bounds(shape)[3] for shape in bottommost_candidates]
    target_bottom = max(bottom_coords)
    
    print(f"    📐 Extending {len(bottommost_candidates)} szoveg elements to y={target_bottom:.1f}")
    
    # Extend the bottom coordinate of all candidates to the target bottom
    for shape in bottommost_candidates:
        points = shape.get("points", [])
        if len(points) >= 2:
            x1, y1, x2, y2 = get_element_bounds(shape)
            # Update the points to extend to target_bottom
            shape["points"] = [[x1, y1], [x2, target_bottom]]
            print(f"    ✅ Extended szoveg from y={y2:.1f} to y={target_bottom:.1f}")


def process_page_layout(data: Dict) -> Dict:
    """Process a single page layout JSON."""
    shapes = data.get("shapes", [])
    image_width = data.get("imageWidth", 3000)  # Default fallback
    
    if not shapes:
        return data
    
    print(f"    📋 Processing {len(shapes)} layout elements")
    
    # Count elements by type
    label_counts = {}
    for shape in shapes:
        label = shape.get("label", "unknown")
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"    🏷️  Element types: {dict(sorted(label_counts.items()))}")
    
    # Detect column boundaries
    boundary1, boundary2 = detect_column_boundaries(shapes, image_width)
    print(f"    🏛️  Column boundaries at x={boundary1:.1f} and x={boundary2:.1f}")
    
    # Group elements by column for single-column elements
    shapes_by_column = {1: [], 2: [], 3: []}
    
    # Assign column numbers to appropriate elements
    for shape in shapes:
        label = shape.get("label", "")
        
        if label in ["szoveg", "hasabkozi_cim"]:
            # Single-column elements
            center_x = get_element_center_x(shape)
            column = assign_column_number(center_x, boundary1, boundary2)
            shape["column_number"] = column
            shapes_by_column[column].append(shape)
            
        elif label in ["oldalfejlec", "szeles_cim"]:
            # Multi-column elements (span all 3 columns)
            shape["column_number"] = 0  # Special value for multi-column
            
        elif label == "hirdetes":
            # Variable width elements - determine based on position and width
            x1, y1, x2, y2 = get_element_bounds(shape)
            center_x = (x1 + x2) / 2
            width = x2 - x1
            
            # If width is more than 1.5 times average column width, consider it multi-column
            avg_column_width = image_width / 3
            if width > 1.5 * avg_column_width:
                shape["column_number"] = 0  # Multi-column
            else:
                column = assign_column_number(center_x, boundary1, boundary2)
                shape["column_number"] = column
                shapes_by_column[column].append(shape)
        else:
            # Unknown element type - assign based on position
            center_x = get_element_center_x(shape)
            column = assign_column_number(center_x, boundary1, boundary2)
            shape["column_number"] = column
            shapes_by_column[column].append(shape)
    
    # Assign row numbers within each column
    assign_row_numbers(shapes_by_column)
    
    # Count elements by column
    column_counts = {i: len(shapes_by_column[i]) for i in [1, 2, 3]}
    multi_col_count = len([s for s in shapes if s.get("column_number") == 0])
    
    print(f"    📊 Column distribution: Col1={column_counts[1]}, Col2={column_counts[2]}, Col3={column_counts[3]}, Multi-col={multi_col_count}")
    
    # Correct szoveg coordinates
    correct_szoveg_coordinates(data)
    
    return data


def process_json_file(json_path: str, jpg_path: str, output_folder: str, input_folder: str) -> bool:
    """Process a single JSON-JPG pair."""
    try:
        # Load JSON data
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"  🔄 Processing: {os.path.relpath(json_path, input_folder)}")
        
        # Process the layout
        processed_data = process_page_layout(data)
        
        # Create output directory structure
        rel_path = os.path.relpath(json_path, input_folder)
        output_json_path = os.path.join(output_folder, rel_path)
        output_dir = os.path.dirname(output_json_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save processed JSON
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        # Copy corresponding JPG
        rel_jpg_path = os.path.relpath(jpg_path, input_folder)
        output_jpg_path = os.path.join(output_folder, rel_jpg_path)
        shutil.copy2(jpg_path, output_jpg_path)
        
        print(f"    ✅ Saved to: {os.path.relpath(output_json_path, output_folder)}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error processing {json_path}: {e}")
        return False


def main():
    if len(sys.argv) != 3:
        print("Usage: python newspaper_layout_processor.py <input_folder> <output_folder>")
        print("\nThis script processes newspaper layout JSON files to:")
        print("- Detect 3-column structure")
        print("- Assign column_number and row_number to layout elements")
        print("- Correct 'szoveg' coordinates to extend to page bottom")
        print("- Copy processed files to output directory")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    
    # Validate input
    if not os.path.exists(input_folder):
        print(f"❌ Input folder not found: {input_folder}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    print("🚀 Starting newspaper layout processing...")
    print(f"📁 Input folder: {input_folder}")
    print(f"📁 Output folder: {output_folder}")
    
    # Find all JSON-JPG pairs
    pairs = find_json_jpg_pairs(input_folder)
    
    if not pairs:
        print("❌ No JSON-JPG pairs found!")
        sys.exit(1)
    
    # Process all pairs
    print(f"\n📋 Processing {len(pairs)} files...")
    successful = 0
    failed = 0
    
    for i, (json_path, jpg_path) in enumerate(pairs):
        print(f"\n[{i+1}/{len(pairs)}] " + "="*60)
        if process_json_file(json_path, jpg_path, output_folder, input_folder):
            successful += 1
        else:
            failed += 1
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"🎉 Processing complete!")
    print(f"✅ Successfully processed: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success rate: {successful/(successful+failed)*100:.1f}%" if (successful+failed) > 0 else "N/A")


if __name__ == "__main__":
    main()