import os
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Input and output paths
INPUT_DIR = "../census/agricultural_census1935_layout"  # <-- Replace this
OUTPUT_IMAGES_DIR = "effocr_traindata/images"
OUTPUT_LABELS_PATH = "effocr_traindata/labels.json"

# Ensure output dirs exist
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)

def extract_cells(json_path, image_path, output_image_dir, coco_images, coco_annotations, counter, ann_id):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img = Image.open(image_path)

    for shape in data.get("shapes", []):
        if "human_output" not in shape or shape["label"]!= "numerical_cell":
            continue

        text = shape["human_output"].get("human_corrected_text", "").strip()
        if not text:
            continue

        # Bounding box
        points = shape["points"]
        (x1, y1), (x2, y2) = points
        left, upper = int(min(x1, x2)), int(min(y1, y2))
        right, lower = int(max(x1, x2)), int(max(y1, y2))
        width, height = right - left, lower - upper

        # Crop and save image
        filename = f"{counter:06d}.jpg"
        cropped = img.crop((left, upper, right, lower))
        cropped.save(os.path.join(output_image_dir, filename))

        # Add COCO-style entries
        coco_images.append({
            "id": counter,
            "file_name": filename,
            "width": width,
            "height": height
        })

        coco_annotations.append({
            "id": ann_id,
            "image_id": counter,
            "category_id": 0,
            "bbox": [0, 0, width, height],  # Full image
            "text": text,
            "iscrowd": 0,
        })

        counter += 1
        ann_id += 1

    return counter, ann_id

def main():
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "text"}]
    }

    counter = 0
    ann_id = 0

    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(root, file)
                jpg_path = json_path.replace(".json", ".jpg")
                if not os.path.exists(jpg_path):
                    continue

                counter, ann_id = extract_cells(
                    json_path, jpg_path,
                    OUTPUT_IMAGES_DIR,
                    coco_output["images"],
                    coco_output["annotations"],
                    counter, ann_id
                )

    # Save labels.json in COCO format
    with open(OUTPUT_LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(coco_output, f, ensure_ascii=False, indent=2)

    print(f"Processed {counter} cropped cells into COCO JSON format.")

if __name__ == "__main__":
    main()