#this script generates LabelMe compatible input data from COCO jsons
import json
import os
import cv2
from collections import defaultdict

def coco_to_labelme(coco_json_path, images_dir, output_dir):
    # Load COCO JSON
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Map image IDs to filenames
    image_id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}

    # Map category_id to category name (if available)
    category_id_to_name = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    # Group annotations by image ID
    image_annotations = defaultdict(list)
    for annotation in coco_data["annotations"]:
        image_id = annotation["image_id"]
        image_annotations[image_id].append(annotation)

    # Process each image
    for image_id, annotations in image_annotations.items():
        image_filename = image_id_to_filename.get(image_id)
        if not image_filename:
            print(f"Warning: No image found for image_id {image_id}")
            continue

        image_path = os.path.join(images_dir, "images", image_filename)

        # Read image to get dimensions
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            continue
        height, width, _ = image.shape

        # Convert all COCO bounding boxes for this image
        shapes = []
        for annotation in annotations:
            x, y, bbox_width, bbox_height = annotation["bbox"]
            category_id = annotation["category_id"]
            label = category_id_to_name.get(category_id, str(category_id))  # Use name if available, else ID

            shape = {
                "label": label,  # Now inherits from COCO category
                "points": [[x, y], [x + bbox_width, y + bbox_height]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
            shapes.append(shape)

        # Create LabelMe annotation structure
        labelme_annotation = {
            "version": "4.5.9",
            "flags": {},
            "shapes": shapes,
            "imagePath": image_filename,  # <-- Only filename, no folder
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width
        }

        # Save LabelMe JSON
        output_json_path = os.path.join(output_dir, "images", f"{os.path.splitext(image_filename)[0]}.json")
        with open(output_json_path, "w", encoding="utf-8") as out_f:
            json.dump(labelme_annotation, out_f, indent=4)

        print(f"Saved: {output_json_path}")

# Iterate over immediate subfolders in the output base directory
output_base_dir = "../../census/arcanum1layout"
for subfolder in os.listdir(output_base_dir):
    subfolder_path = os.path.join(output_base_dir, subfolder)
    if os.path.isdir(subfolder_path):
        # Find the JSON file in the subfolder
        coco_json_path = None
        for file in os.listdir(subfolder_path):
            if file.endswith(".json"):
                coco_json_path = os.path.join(subfolder_path, file)
                break
        
        if coco_json_path:
            images_dir = subfolder_path
            output_dir = subfolder_path
            print(f"Processing: {coco_json_path}")
            coco_to_labelme(coco_json_path, images_dir, output_dir)
        else:
            print(f"No JSON file found in {subfolder_path}")