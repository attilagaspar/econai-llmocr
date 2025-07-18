import sys
import os
import json
import pandas as pd

def get_final_value(shape):
    # Priority: human_output.human_corrected_text > openai_output.response > tesseract_output.ocr_text (lines)
    if "human_output" in shape and "human_corrected_text" in shape["human_output"]:
        return shape["human_output"]["human_corrected_text"]
    elif shape.get("label") in ("text_cell", "column_header") and "openai_output" in shape and "response" in shape["openai_output"]:
        return shape["openai_output"]["response"]
    elif shape.get("label") == "numerical_cell" and "tesseract_output" in shape and "ocr_text" in shape["tesseract_output"]:
        return shape["tesseract_output"]["ocr_text"]
    else:
        return ""

def get_shape_width(shape):
    points = shape.get("points", [])
    if len(points) == 2:
        x1, _ = points[0]
        x2, _ = points[1]
        return abs(x2 - x1)
    return float("inf")

def process_json(json_path):
    print(f"Processing JSON: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    shapes = [s for s in data.get("shapes", []) if get_shape_width(s) < 500]
    print(f"  Found {len(shapes)} shapes narrower than 500px")
    # Group shapes by super_row
    rows = {}
    for shape in shapes:
        row = shape.get("super_row")
        col = shape.get("super_column")
        if row is None or col is None:
            print(f"    Skipping shape without super_row or super_column")
            continue
        val = get_final_value(shape)
        if row not in rows:
            rows[row] = {}
        rows[row][col] = val

    # For each super_row, split each cell into lines and pad to max block height
    row_blocks = []
    for row in sorted(rows.keys()):
        cols = rows[row]
        # Split each cell into lines
        cell_lines = {col: str(cols[col]).split('\n') for col in cols}
        
        max_lines = max(len(lines) for lines in cell_lines.values()) if cell_lines else 1
        print(f"  Row {row}: max block height = {max_lines}")
        # Pad each cell to max_lines
        for col in cell_lines:
            if len(cell_lines[col]) < max_lines:
                print(f"    Padding column {col} in row {row} from {len(cell_lines[col])} to {max_lines} lines")
            cell_lines[col] += [""] * (max_lines - len(cell_lines[col]))
        # For each line, build a row for excel (each line is a new row)
        for i in range(max_lines):
            excel_row = {"_super_row": row}
            for col in cell_lines:
                excel_row[col] = cell_lines[col][i]
            row_blocks.append(excel_row)
    print(f"  Generated {len(row_blocks)} excel rows for this JSON")
    return row_blocks

def main(input_folder, output_excel):
    print(f"Scanning folder: {input_folder}")
    all_rows = []
    json_files = [fname for fname in sorted(os.listdir(input_folder)) if fname.lower().endswith(".json")]
    print(f"Found {len(json_files)} JSON files.")
    for fname in json_files:
        json_path = os.path.join(input_folder, fname)
        row_blocks = process_json(json_path)
        all_rows.extend(row_blocks)
        print(f"Appended {len(row_blocks)} rows from {fname}")
    # Find all super_columns used
    all_cols = set()
    for row in all_rows:
        all_cols.update([c for c in row if c != "_super_row"])
    sorted_cols = sorted(all_cols, key=lambda x: int(x))
    print(f"Detected columns: {sorted_cols}")
    df = pd.DataFrame(all_rows)
    df = df[["_super_row"] + sorted_cols]
    print(f"Writing {len(df)} rows to Excel: {output_excel}")
    df.to_excel(output_excel, index=False)
    print(f"Exported to {output_excel}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python json_join_excel_export.py input_json_folder output_excel_file.xlsx")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
    print("Done processing all JSON files.")