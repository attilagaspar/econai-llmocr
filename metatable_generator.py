import os
import json
from PIL import Image, ImageDraw


def process_json_and_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    combined_elements = []
    current_y_offset = 0
    combined_image_width = 0
    combined_image_height = 0
    images_to_append = []

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
                el['points'] = [
                    [el['points'][0][0] - x_min, el['points'][0][1] - y_min + current_y_offset],
                    [el['points'][1][0] - x_min, el['points'][1][1] - y_min + current_y_offset]
                ]
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

    print(f"Combined image saved to {combined_image_path}")
    print(f"Combined JSON saved to {combined_json_path}")

    # Draw bounding boxes on the combined image
    draw_boxes_on_image(combined_image_path, combined_json_path)


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

    # Save the image with bounding boxes
    boxed_image_path = image_path.replace(".jpg", "_boxed.jpg")
    image.save(boxed_image_path)
    print(f"Image with bounding boxes saved to {boxed_image_path}")


# Define input and output directories
input_directory = "annotated_tables"
output_directory = "metatable_raw"

# Process the JSON files and images
process_json_and_images(input_directory, output_directory)