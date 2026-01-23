#!/usr/bin/env python3
"""
Merge multiple Excel files into a single master Excel file.
Appends all tables one after another with a source filename column.
"""

import argparse
import os
import re
from pathlib import Path
from openpyxl import Workbook, load_workbook


def natural_sort_key(filename):
    """
    Natural sort key for filenames like page_1, page_2, page_10, page_15_table2, etc.
    Converts numeric parts to integers for proper sorting.
    """
    def atoi(text):
        return int(text) if text.isdigit() else text.lower()
    
    return [atoi(c) for c in re.split(r'(\d+)', filename)]


def merge_excel_files(input_folder, output_file):
    """
    Merge all XLSX files from input_folder into a single output_file.
    Adds a 'Source File' column to track which file each row came from.
    """
    input_path = Path(input_folder)
    
    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: Input folder '{input_folder}' does not exist or is not a directory.")
        return
    
    # Find all XLSX files
    xlsx_files = sorted(input_path.glob("*.xlsx"), key=lambda p: natural_sort_key(p.stem))
    
    if not xlsx_files:
        print(f"No XLSX files found in '{input_folder}'")
        return
    
    print(f"Found {len(xlsx_files)} XLSX files to merge")
    
    # Create master workbook
    master_wb = Workbook()
    master_ws = master_wb.active
    master_ws.title = "Merged Tables"
    
    current_row = 1
    first_file = True
    
    for xlsx_file in xlsx_files:
        print(f"Processing: {xlsx_file.name}")
        
        try:
            wb = load_workbook(xlsx_file, data_only=True)
            ws = wb.active
            
            # Get max row and column
            max_row = ws.max_row
            max_col = ws.max_column
            
            if max_row == 0 or max_col == 0:
                print(f"  Skipping empty file: {xlsx_file.name}")
                continue
            
            # Add header with "Source File" column on first file only
            if first_file:
                # Copy header row
                for col_idx in range(1, max_col + 1):
                    cell_value = ws.cell(row=1, column=col_idx).value
                    master_ws.cell(row=current_row, column=col_idx, value=cell_value)
                
                # Add "Source File" header
                master_ws.cell(row=current_row, column=max_col + 1, value="Source File")
                current_row += 1
                first_file = False
            
            # Copy data rows (skip header row for all files)
            for row_idx in range(2, max_row + 1):
                for col_idx in range(1, max_col + 1):
                    cell_value = ws.cell(row=row_idx, column=col_idx).value
                    master_ws.cell(row=current_row, column=col_idx, value=cell_value)
                
                # Add source filename
                master_ws.cell(row=current_row, column=max_col + 1, value=xlsx_file.name)
                current_row += 1
            
            wb.close()
            print(f"  Added {max_row - 1} rows")
            
        except Exception as e:
            print(f"  Error processing {xlsx_file.name}: {e}")
            continue
    
    # Save master workbook
    try:
        master_wb.save(output_file)
        print(f"\n=== MERGE COMPLETE ===")
        print(f"Merged {len(xlsx_files)} files")
        print(f"Total rows (including header): {current_row - 1}")
        print(f"Output saved to: {output_file}")
    except Exception as e:
        print(f"Error saving output file: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple Excel tables into a single master Excel file"
    )
    parser.add_argument(
        "input_folder",
        help="Folder containing XLSX files to merge"
    )
    parser.add_argument(
        "output_file",
        help="Output master Excel file path"
    )
    
    args = parser.parse_args()
    
    merge_excel_files(args.input_folder, args.output_file)


if __name__ == "__main__":
    main()
