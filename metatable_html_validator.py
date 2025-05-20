from PIL import Image

# This script loads the LLM-corrected JSON files and creates a 
# HTML file which 1) has the number of columns and rows as the
# entries in the JSON file, so each cell corresponds to an 
# entry in the JSON. Within that cell, the script puts next to
# one another the original image, the raw OCR text, and the 
# LLM-corrected text.
# The original image is the metatable JPG, from which the 
# bounding boxes are used to generate the cutout.

import os
import json

raw_folder = "metatable_raw" # here is the corresponding JPG image
output_folder = "metatable_llm" # here is the JSON file
html_output_folder = "metatable_html" # here is the HTML file
def generate_html_with_images_and_text(combined_json, raw_folder, output_html_path):
    """
    Generate an HTML table from the JSON data, including images, OCR text, and LLM-corrected text.

    Args:
        combined_json (dict): The combined JSON data.
        raw_folder (str): Folder containing the raw images.
        output_html_path (str): Path to save the HTML file.
    """
    # Find the maximum row and column numbers
    max_row = max(shape["row"] for shape in combined_json["shapes"])
    max_column = max(shape["column"] for shape in combined_json["shapes"])

    # Create a 2D list to represent the table
    table = [["" for _ in range(max_column)] for _ in range(max_row)]

    # Populate the table with image paths, OCR text, and LLM-corrected text
    for shape in combined_json["shapes"]:
        row = shape["row"] - 1
        column = shape["column"] - 1
        ocr_text = shape.get("ocr_text", "")
        llm_text = shape.get("corrected_text", "")
        # Load the corresponding image
        image_path = os.path.join(raw_folder, os.path.splitext(json_file)[0] + ".jpg")
        if os.path.exists(image_path):
            with Image.open(image_path) as img:
                # Extract bounding box points
                points = shape["points"]
                # Ensure points are in the correct order (top-left and bottom-right)
                left = min(points[0][0], points[1][0])
                top = min(points[0][1], points[1][1])
                right = max(points[0][0], points[1][0])
                bottom = max(points[0][1], points[1][1])

                # Crop the image using the bounding box
                cropped_image = img.crop((left, top, right, bottom))

                # Save the cropped image to the same folder as the HTML
                cropped_image_filename = f"crop_{shape['row']}_{shape['column']}.jpg"
                cropped_image_path = os.path.join(html_output_folder, cropped_image_filename)
                cropped_image.save(cropped_image_path)
                table[row][column] = (cropped_image_filename, ocr_text, llm_text)

    # Generate HTML
    with open(output_html_path, "w", encoding="utf-8") as f:
        # Add CSS for styling the table
        f.write("""
        <html>
        <head>
            <style>
                table {
                    font-size: 20px; /* Increase font size for the entire table */
                    border-collapse: collapse;
                }
                th, td {
                    padding: 8px;
                    text-align: left;
                    border: 1px solid black;
                }
                input[type="text"] {
                    font-size: 14px; /* Adjust font size for text inputs */
                }
            </style>
        </head>
        <body>
        <form id='correctionForm'>
        <table border='1'>
        """)    
        f.write("<tr><th></th>")  # Empty corner cell
        for col in range(1, max_column + 1):
            f.write(f"<th>Col {col}</th>")
        f.write("</tr>\n")
        for row_idx, row in enumerate(table, start=1):
            f.write(f"<tr><th>Row {row_idx}</th>")
            for cell_idx, cell in enumerate(row):
                # Handle empty cells with default values
                if not cell:
                    image_path = ""
                    ocr_text = "No OCR text"
                    llm_text = "No LLM text"
                else:
                    image_path, ocr_text, llm_text = cell

                f.write("<td><table border='0'><tr>")
                # Add the image
                if image_path and os.path.exists(os.path.join(html_output_folder, image_path)):
                    with Image.open(os.path.join(html_output_folder, image_path)) as img:
                        original_width, original_height = img.size
                        scaled_width = int(original_width * 1.25)  # Scale width by 50%
                        scaled_height = int(original_height * 1.25)  # Scale height by 50%
                    f.write(f"<td><img src='{image_path}' width='{scaled_width}' height='{scaled_height}'></td>")
                else:
                    print(f"Image not found: {image_path}")  # Debugging: Check missing images
                    
                # Combine OCR and LLM text into a single table
                f.write("<td><table border='1'>")
                ocr_lines = ocr_text.split("\n")
                llm_lines = llm_text.split("\n")
                max_lines = max(len(ocr_lines), len(llm_lines))
                for i in range(max_lines):
                    ocr_line = ocr_lines[i] if i < len(ocr_lines) else ""
                    llm_line = llm_lines[i] if i < len(llm_lines) else ""
                    f.write("<tr>")
                    #f.write(f"<td>{ocr_line}</td>")
                    #f.write(f"<td>{llm_line}</td>")
                    f.write(f"<td>  1: {ocr_line} <br> 2: {llm_line}</td>")
                    # Add radio buttons for each row, aligned horizontally
                    f.write("<td>")
                    f.write(f"<input type='radio' name='choice_{row_idx}_{cell_idx}_{i}' value='ocr'> 1 ")
                    f.write(f"<input type='radio' name='choice_{row_idx}_{cell_idx}_{i}' value='llm'> 2 ")
                    f.write(f"<input type='radio' name='choice_{row_idx}_{cell_idx}_{i}' value='both' checked> B ")
                    f.write(f"<input type='radio' name='choice_{row_idx}_{cell_idx}_{i}' value='neither'> N ")
                    f.write(f"<input type='text' name='correction_{row_idx}_{cell_idx}_{i}' placeholder='Correct text' style='width: 150px;'>")
                    f.write("</td>")
                    f.write("</tr>")
                f.write("</table></td>")

                f.write("</tr></table></td>")
            f.write("</tr>\n")
        f.write("</table>\n")
        f.write("<button type='button' onclick='submitCorrections()'>Submit</button>\n")

        # Add JavaScript for exporting corrections and calculating percentages
        f.write("""
        <script>
        function submitCorrections() {
            const formData = new FormData(document.getElementById('correctionForm'));
            const corrections = {};
            const counts = { ocr: 0, llm: 0, both: 0, neither: 0 };
            let total = 0;

            // Process form data
            for (const [key, value] of formData.entries()) {
                corrections[key] = value;

                // Count the selected options
                if (key.startsWith('choice_')) {
                    counts[value] = (counts[value] || 0) + 1;
                    total++;
                }
            }

            // Calculate percentages
            const percentages = {
                ocr: ((counts.ocr || 0) / total * 100).toFixed(2),
                llm: ((counts.llm || 0) / total * 100).toFixed(2),
                both: ((counts.both || 0) / total * 100).toFixed(2),
                neither: ((counts.neither || 0) / total * 100).toFixed(2)
            };

            // Display percentages on the page
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = `
                <p>OCR: ${percentages.ocr}%</p>
                <p>LLM: ${percentages.llm}%</p>
                <p>Both: ${percentages.both}%</p>
                <p>Neither: ${percentages.neither}%</p>
            `;

            // Add percentages to the output file
            corrections['percentages'] = percentages;

            // Export corrections as JSON
            const blob = new Blob([JSON.stringify(corrections, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'corrections.json';
            a.click();
            URL.revokeObjectURL(url);
        }
        </script>
        """)
        f.write("<div id='results'></div>")  # Add a div to display the percentages
        f.write("</form></body></html>")

    print(f"HTML table with images and text saved to {output_html_path}")
# Main script
if not os.path.exists(html_output_folder):
    os.makedirs(html_output_folder)

for json_file in os.listdir(output_folder):
    if json_file.endswith(".json"):
        json_path = os.path.join(output_folder, json_file)
        with open(json_path, "r", encoding="utf-8") as f:
            combined_json = json.load(f)

        output_html_path = os.path.join(html_output_folder, f"{os.path.splitext(json_file)[0]}.html")
        generate_html_with_images_and_text(combined_json, raw_folder, output_html_path)
