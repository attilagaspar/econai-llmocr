import os
import json
import cv2
import shutil

def sanitize_filename(path):
    # Replace os separators with underscores for unique filenames
    return path.replace(os.sep, "_").replace("/", "_")

def labelme_to_coco(input_base_dir, output_base_dir):
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    label_to_category_id = {}
    next_category_id = 1
    annotation_id = 1
    image_id = 1

    images_out_dir = os.path.join(output_base_dir, "images")
    os.makedirs(images_out_dir, exist_ok=True)

    for root, _, files in os.walk(input_base_dir):
        for file in files:
            if not file.endswith(".json"):
                continue
            json_path = os.path.join(root, file)
            image_name = os.path.splitext(file)[0] + ".jpg"
            image_path = os.path.join(root, image_name)
            if not os.path.isfile(image_path):
                continue  # skip if no matching image

            # Create a unique image filename based on subfolder structure
            rel_dir = os.path.relpath(root, input_base_dir)
            unique_img_name = sanitize_filename(os.path.join(rel_dir, image_name)) if rel_dir != "." else image_name
            output_img_path = os.path.join(images_out_dir, unique_img_name)

            # Copy image to output images folder
            shutil.copy(image_path, output_img_path)

            # Read image for dimensions
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not read image {image_path}")
                continue
            height, width, _ = image.shape

            # Add image entry to COCO JSON
            coco_output["images"].append({
                "id": image_id,
                "file_name": os.path.join("images", unique_img_name),
                "width": width,
                "height": height
            })

            # Process annotations
            with open(json_path, "r", encoding="utf-8") as f:
                labelme_data = json.load(f)
            for shape in labelme_data.get("shapes", []):
                label = shape["label"]
                points = shape["points"]
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                x, y = min(x_coords), min(y_coords)
                bbox_width, bbox_height = max(x_coords) - x, max(y_coords) - y

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
            image_id += 1

    # Save the COCO JSON file
    output_coco_path = os.path.join(output_base_dir, "all_annotations_coco.json")
    with open(output_coco_path, "w", encoding="utf-8") as f:
        json.dump(coco_output, f, indent=4)
    print(f"Saved COCO JSON: {output_coco_path}")
    print(f"Images are in {images_out_dir}")

# ---- CONFIG ----
#input_base_dir = "C:/Users/agaspar/Dropbox/research/leporolt_adatok/compass/annotations"
input_base_dir = "C:/Users/agaspar/Dropbox/research/leporolt_adatok/econai/census/census2_deskewed_output_check/hand_corrected_batches_1_9"
#output_base_dir = "C:/Users/agaspar/Dropbox/research/leporolt_adatok/compass/coco_annotations"
output_base_dir = "C:/Users/agaspar/Dropbox/research/leporolt_adatok/econai/census/census2_deskewed_output_check/retrain_input_1_16"
if __name__ == "__main__":
    labelme_to_coco(input_base_dir, output_base_dir)