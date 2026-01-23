import os
import json
import argparse
from pathlib import Path
import openpyxl
from openpyxl.styles import Alignment

# Type useful labels (same as in layout_superstructure_detect.py)
TYPE_USEFUL = ["tablazatelem", "tablazatfejlec"]


def identify_table_groups(shapes):
    """Group shapes into separate tables based on super_row/super_column sequences.
    
    If shapes have non-contiguous super_row or super_column sequences, or if there's
    a significant gap in the numbering, they belong to different tables (different regions).
    """
    # Filter for TYPE_USEFUL shapes that have super_row and super_column
    useful_shapes = [
        s for s in shapes 
        if s.get("label") in TYPE_USEFUL 
        and "super_row" in s 
        and "super_column" in s
    ]
    
    if not useful_shapes:
        return []
    
    # Group by detecting regions - shapes with similar coordinate ranges belong together
    # This handles the case where one page has multiple tables (regions)
    tables = []
    processed = set()
    
    for shape in useful_shapes:
        if id(shape) in processed:
            continue
        
        # Start a new table group
        table_group = [shape]
        processed.add(id(shape))
        
        # Get the coordinate range for this shape
        points = shape.get("points", [])
        if len(points) < 2:
            continue
        
        y_coords = [p[1] for p in points]
        group_y_min = min(y_coords)
        group_y_max = max(y_coords)
        
        # Find all shapes that are vertically nearby (within 500 pixels)
        proximity_threshold = 500
        changed = True
        while changed:
            changed = False
            for other_shape in useful_shapes:
                if id(other_shape) in processed:
                    continue
                
                other_points = other_shape.get("points", [])
                if len(other_points) < 2:
                    continue
                
                other_y = [p[1] for p in other_points]
                other_y_min = min(other_y)
                other_y_max = max(other_y)
                
                # Check if this shape is close to any shape in the current group
                if not (other_y_max < group_y_min - proximity_threshold or 
                        other_y_min > group_y_max + proximity_threshold):
                    table_group.append(other_shape)
                    processed.add(id(other_shape))
                    # Update group boundaries
                    group_y_min = min(group_y_min, other_y_min)
                    group_y_max = max(group_y_max, other_y_max)
                    changed = True
        
        if table_group:
            tables.append(table_group)
    
    # Sort tables by their vertical position (top to bottom)
    tables.sort(key=lambda t: min(min(p[1] for p in s["points"]) for s in t if len(s.get("points", [])) >= 2))
    
    return tables


def export_table_to_xlsx(shapes, output_path, data_field):
    """Export a table to an XLSX file.
    
    Args:
        shapes: List of shape dictionaries with super_row and super_column
        output_path: Path to save the XLSX file
        data_field: Name of the field to extract from shapes (e.g., 'original_pdf_text_layer')
    """
    if not shapes:
        return False
    
    # Find the range of rows and columns
    rows = set()
    cols = set()
    for shape in shapes:
        if "super_row" in shape and "super_column" in shape:
            rows.add(shape["super_row"])
            cols.add(shape["super_column"])
    
    if not rows or not cols:
        return False
    
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    
    # Create a workbook
    wb = openpyxl.Workbook()
    ws = wb.active
    
    # Create a mapping from (row, col) to data
    table_data = {}
    for shape in shapes:
        if "super_row" in shape and "super_column" in shape:
            row = shape["super_row"]
            col = shape["super_column"]
            data = shape.get(data_field, "")
            table_data[(row, col)] = data
    
    # Write to Excel - map super_row/super_column to Excel coordinates
    for row_idx in range(min_row, max_row + 1):
        for col_idx in range(min_col, max_col + 1):
            data = table_data.get((row_idx, col_idx), "")
            # Excel uses 1-based indexing, adjust for super_row/super_column
            excel_row = row_idx - min_row + 1
            excel_col = col_idx - min_col + 1
            cell = ws.cell(row=excel_row, column=excel_col, value=data)
            # Set text wrapping and top alignment
            cell.alignment = Alignment(wrap_text=True, vertical='top')
    
    # Auto-adjust column widths (set to reasonable default)
    for col_idx in range(1, max_col - min_col + 2):
        ws.column_dimensions[openpyxl.utils.get_column_letter(col_idx)].width = 20
    
    # Save
    wb.save(output_path)
    return True


def process_json_file(json_path, output_dir, data_field, input_root):
    """Process a single JSON file and export tables.
    
    Returns:
        Number of tables exported
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        shapes = data.get("shapes", [])
        
        # Identify separate tables (regions)
        table_groups = identify_table_groups(shapes)
        
        if not table_groups:
            return 0
        
        # Generate output filenames based on relative path
        rel_path = os.path.relpath(json_path, input_root)
        # Remove .json extension and replace path separators with underscores
        base_name = rel_path.replace('.json', '').replace(os.sep, '_').replace('/', '_')
        
        exported_count = 0
        for table_idx, table_shapes in enumerate(table_groups, start=1):
            output_filename = f"{base_name}_table{table_idx}.xlsx"
            output_path = os.path.join(output_dir, output_filename)
            
            if export_table_to_xlsx(table_shapes, output_path, data_field):
                print(f"Exported: {output_filename} ({len(table_shapes)} shapes)")
                exported_count += 1
        
        return exported_count
        
    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description='Export tables from labelme JSONs with super_row/super_column to XLSX files'
    )
    parser.add_argument('input_folder', help='Input folder to search recursively for JSON files')
    parser.add_argument('output_folder', help='Output folder for XLSX files')
    parser.add_argument('--data-field', default='original_pdf_text_layer',
                        help='JSON field to export as cell content (default: original_pdf_text_layer)')
    
    args = parser.parse_args()
    
    input_folder = args.input_folder
    output_folder = args.output_folder
    data_field = args.data_field
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all JSON files recursively
    json_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    print(f"Found {len(json_files)} JSON files")
    print(f"Exporting '{data_field}' field to XLSX files...")
    print()
    
    total_tables = 0
    processed_files = 0
    
    for json_file in json_files:
        tables_exported = process_json_file(json_file, output_folder, data_field, input_folder)
        if tables_exported > 0:
            processed_files += 1
            total_tables += tables_exported
    
    print()
    print(f"=== EXPORT COMPLETE ===")
    print(f"Processed {processed_files} JSON files with tables")
    print(f"Exported {total_tables} XLSX files to {output_folder}")


if __name__ == "__main__":
    main()
