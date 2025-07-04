

# This script looks at the subdirectories in "model_training_input",
# and merges the COCO JSON files and images into a single COCO JSON file
# and a single images directory.
# The output is saved in "model_training_input/merged".

import os
import shutil
import json
from pathlib import Path
import math

input_dir = Path("model_training_input")
output_dir = input_dir / "merged"
output_images_dir = output_dir / "images"
output_json_path = output_dir / "merged.json"

output_dir.mkdir(exist_ok=True)
output_images_dir.mkdir(exist_ok=True)

image_id = 1
annotation_id = 1
merged_images = []
merged_annotations = []
merged_categories = []
category_mapping = {}
category_id_counter = 1

def get_subdirs(path):
    return [d for d in path.iterdir() if d.is_dir() and d.name != "merged"]
def is_valid_bbox(bbox):
    # bbox should be [x, y, w, h] with w > 0 and h > 0 and no NaNs
    if len(bbox) != 4:
        return False
    if any(math.isnan(x) for x in bbox):
        return False
    if bbox[2] <= 0 or bbox[3] <= 0:
        return False
    return True

for subdir in get_subdirs(input_dir):
    coco_jsons = list(subdir.glob("*.json"))
    images_dir = subdir / "images"
    if not coco_jsons or not images_dir.exists():
        continue
    with open(coco_jsons[0], "r", encoding="utf-8") as f:
        data = json.load(f)
    # Handle categories
    for cat in data.get("categories", []):
        cat_name = cat["name"]
        if cat_name not in category_mapping:
            cat_copy = cat.copy()
            cat_copy["id"] = category_id_counter
            category_mapping[cat_name] = category_id_counter
            merged_categories.append(cat_copy)
            category_id_counter += 1
    # Handle images and annotations
    img_id_map = {}
    for img in data.get("images", []):
        old_id = img["id"]
        img_filename = img["file_name"]
        new_filename = f"{subdir.name}_{img_filename}"
        src_img_path = images_dir / img_filename
        dst_img_path = output_images_dir / new_filename
        shutil.copy2(src_img_path, dst_img_path)
        img_copy = img.copy()
        img_copy["id"] = image_id
        img_copy["file_name"] = new_filename
        merged_images.append(img_copy)
        img_id_map[old_id] = image_id
        image_id += 1
    for ann in data.get("annotations", []):
        bbox = ann.get("bbox", [])
        if not is_valid_bbox(bbox):
            print(f"Warning: Parsing a badly shaped bounding box: {bbox}")
        ann_copy = ann.copy()
        ann_copy["id"] = annotation_id
        ann_copy["image_id"] = img_id_map[ann["image_id"]]
        # Remap category_id
        for cat in data.get("categories", []):
            if cat["id"] == ann["category_id"]:
                ann_copy["category_id"] = category_mapping[cat["name"]]
                break
        merged_annotations.append(ann_copy)
        annotation_id += 1

merged_coco = {
    "images": merged_images,
    "annotations": merged_annotations,
    "categories": merged_categories
}

with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(merged_coco, f, ensure_ascii=False, indent=2)