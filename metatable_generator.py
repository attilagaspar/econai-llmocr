# This script combines annotated tables from JSON files and images, 
# Generates a combined meta-table,
# assigns columns and rows to each box, runs OCR on the boxes,
# processes them, and generates an HTML table with OCR data.
import os
import json
from PIL import Image, ImageDraw, ImageFont
import pytesseract

# Set up PyTesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
tess_config = '--psm 6 -l hun'


def process_json_and_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    combined_elements = []
    current_y_offset = 0
    combined_image_width = 0
    combined_image_height = 0
    images_to_append = []
    unique_id_counter = 1  # Initialize a counter for unique IDs

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".json"):
            json_path = os.path.join(input_dir, file_name)
            image_path = os.path.join(input_dir, file_name.replace(".json", ".jpg"))

            if not os.path.exists(image_path):
                print(f"Image file {image_path} not found. Skipping.")
                continue

            with open(json_path, 'r') as f:
                data = json.load(f)

            # Filter out elements of type "column_header"
            elements = [el for el in data['shapes'] if el['label'] != "column_header"]

            if not elements:
                continue

            # Find the smallest bounding box containing all elements
            x_min = min(el['points'][0][0] for el in elements)
            y_min = min(el['points'][0][1] for el in elements)
            x_max = max(el['points'][1][0] for el in elements)
            y_max = max(el['points'][1][1] for el in elements)

            # Open the corresponding image and crop the bounding box
            image = Image.open(image_path)
            cropped_image = image.crop((x_min, y_min, x_max, y_max))
            images_to_append.append(cropped_image)

            # Update combined image dimensions
            combined_image_width = max(combined_image_width, cropped_image.width)
            combined_image_height += cropped_image.height

            # Update element coordinates for the combined JSON
            for el in elements:
                # Adjust coordinates relative to the cropped image
                adjusted_points = [
                    [el['points'][0][0] - x_min, el['points'][0][1] - y_min + current_y_offset],
                    [el['points'][1][0] - x_min, el['points'][1][1] - y_min + current_y_offset]
                ]

                # Drop the element if it has negative coordinates
                if any(coord < 0 for point in adjusted_points for coord in point):
                    continue

                el['points'] = adjusted_points
                el['id'] = unique_id_counter  # Assign a unique ID to the element
                unique_id_counter += 1  # Increment the counter
                combined_elements.append(el)

            current_y_offset += cropped_image.height

    # Combine all cropped images vertically
    combined_image = Image.new('RGB', (combined_image_width, combined_image_height))
    y_offset = 0
    for img in images_to_append:
        combined_image.paste(img, (0, y_offset))
        y_offset += img.height

    # Save the combined image and JSON
    combined_image_path = os.path.join(output_dir, "combined.jpg")
    combined_json_path = os.path.join(output_dir, "combined.json")

    combined_image.save(combined_image_path)

    combined_json = {
        "shapes": combined_elements
    }
    with open(combined_json_path, 'w') as f:
        json.dump(combined_json, f, indent=4)

    # Assign column and row numbers
    assign_columns_and_rows(combined_json, 10)

    # Run OCR and update JSON
    run_ocr_on_boxes(combined_image, combined_json)

    # Save the updated JSON with rows, columns, and OCR data
    with open(combined_json_path, 'w') as f:
        json.dump(combined_json, f, indent=4)

    # Draw bounding boxes on the combined image
    draw_boxes_on_image(combined_image_path, combined_json_path)

    # Generate HTML table
    generate_html_table(combined_json, os.path.join(output_dir, "output_table.html"))

def assign_columns_and_rows(combined_json, start_tol=10):
    """
    Assign column and row numbers to each box in the combined JSON.

    Args:
        combined_json (dict): The combined JSON data.
        start_tol (int): The tolerance for grouping boxes into the same column or row.
    """
    print("Assigning columns and rows...")
    shapes = combined_json["shapes"]

    # Assign column numbers
    remaining_shapes = shapes[:]
    current_column = 1

    while remaining_shapes:
        # Find the left-most x-coordinate of the remaining boxes
        leftmost_x = min(min(p[0] for p in shape["points"]) for shape in remaining_shapes)

        # Assign column number to all boxes that intersect the vertical line at (leftmost_x + start_tol)
        column_shapes = []
        for shape in remaining_shapes:
            x_min = min(p[0] for p in shape["points"])
            x_max = max(p[0] for p in shape["points"])
            if x_min <= leftmost_x + start_tol and x_max >= leftmost_x:
                shape["column"] = current_column
                column_shapes.append(shape)

        # Calculate the average left and right edge of the current column
        avg_x1 = sum(min(p[0] for p in shape["points"]) for shape in column_shapes) / len(column_shapes)
        avg_x2 = sum(max(p[0] for p in shape["points"]) for shape in column_shapes) / len(column_shapes)

        # Include remaining boxes whose centroid falls within the average bounds of the current column
        additional_shapes = []
        for shape in remaining_shapes:
            if shape in column_shapes:
                continue
            x_min = min(p[0] for p in shape["points"])
            x_max = max(p[0] for p in shape["points"])
            centroid_x = (x_min + x_max) / 2
            if avg_x1 <= centroid_x <= avg_x2:
                shape["column"] = current_column
                additional_shapes.append(shape)

        # Add the additional shapes to the current column
        column_shapes.extend(additional_shapes)

        # Remove assigned shapes from the remaining list
        remaining_shapes = [shape for shape in remaining_shapes if shape not in column_shapes]

        # Increment column number
        current_column += 1

    print(f"Assigned columns: {[shape['column'] for shape in shapes]}")

    # Assign row numbers
    remaining_shapes = shapes[:]
    current_row = 1

    while remaining_shapes:
        # Find the top-most y-coordinate of the remaining boxes
        topmost_y = min(min(p[1] for p in shape["points"]) for shape in remaining_shapes)

        # Assign row number to all boxes that intersect the horizontal line at (0, topmost_y + start_tol)
        row_shapes = []
        for shape in remaining_shapes:
            y_min = min(p[1] for p in shape["points"])
            y_max = max(p[1] for p in shape["points"])
            if y_min <= topmost_y + start_tol and y_max >= topmost_y:
                shape["row"] = current_row
                row_shapes.append(shape)

        # Calculate the average top and bottom edge of the current row
        avg_y1 = sum(min(p[1] for p in shape["points"]) for shape in row_shapes) / len(row_shapes)
        avg_y2 = sum(max(p[1] for p in shape["points"]) for shape in row_shapes) / len(row_shapes)

        # Include remaining boxes whose centroid falls within the average bounds of the current row
        additional_shapes = []
        for shape in remaining_shapes:
            if shape in row_shapes:
                continue
            y_min = min(p[1] for p in shape["points"])
            y_max = max(p[1] for p in shape["points"])
            centroid_y = (y_min + y_max) / 2
            if avg_y1 <= centroid_y <= avg_y2:
                shape["row"] = current_row
                additional_shapes.append(shape)

        # Add the additional shapes to the current row
        row_shapes.extend(additional_shapes)

        # Remove assigned shapes from the remaining list
        remaining_shapes = [shape for shape in remaining_shapes if shape not in row_shapes]

        # Increment row number
        current_row += 1

    print(f"Assigned rows: {[shape['row'] for shape in shapes]}")

    # Update the combined JSON with the assigned rows and columns
    combined_json["shapes"] = shapes


def draw_boxes_on_image(image_path, json_path):
    """
    Draw bounding boxes on the combined image based on the JSON file.

    Args:
        image_path (str): Path to the combined image.
        json_path (str): Path to the combined JSON file.
    """
    # Open the image and load the JSON
    image = Image.open(image_path)
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Define colors for labels
    label_colors = {
        "numerical_cell": "red",
        "text_cell": "green"
    }

    # Draw each shape
    for shape in data["shapes"]:
        label = shape["label"]
        points = shape["points"]
        color = label_colors.get(label, "blue")  # Default to blue if label is unknown

        # Ensure coordinates are ordered correctly
        x1, y1 = points[0]
        x2, y2 = points[1]
        x1, x2 = sorted([x1, x2])  # Ensure x1 <= x2
        y1, y2 = sorted([y1, y2])  # Ensure y1 <= y2

        # Draw the rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Write column and row numbers
        column = shape.get("column", "?")
        row = shape.get("row", "?")
        text = f"({column},{row})"
        draw.text((x1, y1 - 10), text, fill=color)

    # Save the image with bounding boxes
    boxed_image_path = image_path.replace(".jpg", "_boxed.jpg")
    image.save(boxed_image_path)
    print(f"Image with bounding boxes saved to {boxed_image_path}")

def run_ocr_on_boxes(image, combined_json):
    """
    Run OCR on each box and update the JSON with the extracted text and OCR goodness.

    Args:
        image (PIL.Image): The combined image.
        combined_json (dict): The combined JSON data.
    """
    print("Running OCR on boxes...")

    print(f"Number of elements in combined_json['shapes']: {len(combined_json['shapes'])}")
    current_element = 0
    for shape in combined_json["shapes"]:
        current_element += 1
        print(f"Processing element {current_element}/{len(combined_json['shapes'])}")
        x1, y1 = shape["points"][0]
        x2, y2 = shape["points"][1]

        # Ensure coordinates are ordered correctly
        x1, x2 = sorted([x1, x2])  # Ensure x1 <= x2
        y1, y2 = sorted([y1, y2])  # Ensure y1 <= y2

        # Crop the box from the image
        cropped_box = image.crop((x1, y1, x2, y2))

        # Run OCR
        ocr_text = pytesseract.image_to_string(cropped_box, config=tess_config).strip()
        ocr_goodness = pytesseract.image_to_data(cropped_box, config=tess_config, output_type=pytesseract.Output.DICT)["conf"]

        # Add OCR data to the JSON
        shape["ocr_text"] = ocr_text
        shape["ocr_goodness"] = ocr_goodness


def generate_html_table(combined_json, output_html_path):
    """
    Generate an HTML table from the JSON data.

    Args:
        combined_json (dict): The combined JSON data.
        output_html_path (str): Path to save the HTML file.
    """
    # Find the maximum row and column numbers
    max_row = max(shape["row"] for shape in combined_json["shapes"])
    max_column = max(shape["column"] for shape in combined_json["shapes"])

    # Create a 2D list to represent the table
    table = [["" for _ in range(max_column)] for _ in range(max_row)]

    # Populate the table with OCR text
    for shape in combined_json["shapes"]:
        row = shape["row"] - 1
        column = shape["column"] - 1
        ocr_text = shape.get("ocr_text", "")
        table[row][column] = ocr_text

    # Generate HTML
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write("<html><body><table border='1'>\n")
        f.write("<tr><th></th>")  # Empty corner cell
        for col in range(1, max_column + 1):
            f.write(f"<th>Col {col}</th>")
        f.write("</tr>\n")
        for row_idx, row in enumerate(table, start=1):
            f.write(f"<tr><th>Row {row_idx}</th>")
            for cell in row:
                # Create an inner table for each cell
                f.write("<td><table border='0'>")
                for line in cell.split("\n"):  # Split OCR text by new lines
                    f.write(f"<tr><td>{line}</td></tr>")
                f.write("</table></td>")
            f.write("</tr>\n")
        f.write("</table></body></html>")

    print(f"HTML table saved to {output_html_path}")



# Define input and output directories
input_directory = "annotated_tables"
output_directory = "metatable_raw"

# Process the JSON files and images
process_json_and_images(input_directory, output_directory)