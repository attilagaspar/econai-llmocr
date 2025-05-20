import json
import os
import openpyxl
from openpyxl import Workbook

def json_to_excel(json_folder, output_excel_path):
    """
    Convert JSON files with LLM-cleaned results into a single Excel sheet.

    Args:
        json_folder (str): Path to the folder containing JSON files.
        output_excel_path (str): Path to save the resulting Excel file.
    """
    # Create a new Excel workbook and sheet
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "LLM Cleaned Results"

    # Add header row
    sheet.append(["Row", "Column", "Corrected Text"])

    # Process each JSON file in the folder
    for filename in os.listdir(json_folder):
        if filename.endswith(".json"):
            json_path = os.path.join(json_folder, filename)

            # Load the JSON file
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Group entries by row
            rows = {}
            for entry in data["shapes"]:
                row = entry["row"]
                column = entry["column"]
                corrected_text = entry.get("corrected_text", "")

                if row not in rows:
                    rows[row] = {}
                rows[row][column] = corrected_text.split("\n")  # Split text into internal rows

            # Process each row in sorted order
            for row in sorted(rows.keys()):  # Sort rows by their row number
                columns = rows[row]

                # Find the maximum number of internal rows in this row
                max_internal_rows = max(len(texts) for texts in columns.values())

                # Write each column's data to the Excel sheet
                for i in range(max_internal_rows):
                    excel_row = [row]  # Start with the row number
                    for column in sorted(columns.keys()):  # Sort columns for consistent order
                        texts = columns[column]
                        excel_row.append(texts[i] if i < len(texts) else "")  # Add text or empty cell
                    sheet.append(excel_row)

    # Save the Excel file
    workbook.save(output_excel_path)
    print(f"Excel file saved to {output_excel_path}")


# Main function
def main():
    # Define input and output paths
    json_folder = "metatable_llm"  # Folder containing JSON files
    output_excel_path = "LLM_Cleaned_Results.xlsx"  # Output Excel file

    # Convert JSON to Excel
    json_to_excel(json_folder, output_excel_path)


if __name__ == "__main__":
    main()