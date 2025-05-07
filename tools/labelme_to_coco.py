import os
import json
import cv2


def labelme_to_coco(labelme_dir, output_coco_path):
    """
    Converts LabelMe JSON files and images in a directory to a single COCO JSON file.

    Args:
        labelme_dir (str): Path to the directory containing LabelMe JSON files and images.
        output_coco_path (str): Path to save the output COCO JSON file.
    """
    # Initialize COCO JSON structure
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Category mapping (LabelMe labels to COCO category IDs)
    label_to_category_id = {}
    next_category_id = 1

    # Annotation ID counter
    annotation_id = 1

    # Iterate over JSON files in the directory
    for file in os.listdir(labelme_dir):
        if not file.endswith(".json"):
            continue

        json_path = os.path.join(labelme_dir, file)
        with open(json_path, "r", encoding="utf-8") as f:
            labelme_data = json.load(f)

        # Get image information
        image_filename = labelme_data["imagePath"]
        image_path = os.path.join(labelme_dir, image_filename)

        # Read the image to get dimensions
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            continue
        height, width, _ = image.shape

        # Add image entry to COCO JSON
        image_id = len(coco_output["images"]) + 1
        coco_output["images"].append({
            "id": image_id,
            "file_name": image_filename,
            "width": width,
            "height": height
        })

        # Process annotations
        for shape in labelme_data["shapes"]:
            label = shape["label"]
            points = shape["points"]

            # Convert points to COCO bbox format: [x, y, width, height]
            x1, y1 = points[0]
            x2, y2 = points[1]
            x, y = min(x1, x2), min(y1, y2)
            bbox_width, bbox_height = abs(x2 - x1), abs(y2 - y1)

            # Get or create category ID
            if label not in label_to_category_id:
                label_to_category_id[label] = next_category_id
                coco_output["categories"].append({
                    "id": next_category_id,
                    "name": label
                })
                next_category_id += 1
            category_id = label_to_category_id[label]

            # Add annotation entry to COCO JSON
            coco_output["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x, y, bbox_width, bbox_height],
                "area": bbox_width * bbox_height,
                "iscrowd": 0
            })
            annotation_id += 1

    # Save the COCO JSON file
    with open(output_coco_path, "w", encoding="utf-8") as f:
        json.dump(coco_output, f, indent=4)
    print(f"Saved COCO JSON: {output_coco_path}")


# Iterate over immediate subfolders in ../output
output_base_dir = "../output"
for subfolder in os.listdir(output_base_dir):
    subfolder_path = os.path.join(output_base_dir, subfolder)
    if os.path.isdir(subfolder_path):
        # Define the output COCO JSON path
        output_coco_path = os.path.join(subfolder_path, f"{subfolder}_coco.json")

        # Convert LabelMe JSONs to COCO JSON
        print(f"Processing subfolder: {subfolder_path}")
        labelme_to_coco(subfolder_path, output_coco_path)