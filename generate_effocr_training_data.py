import os
import json
from pathlib import Path
from PIL import Image

from tqdm import tqdm

# Input and output paths
INPUT_DIR = "../census/agricultural_census1935_layout"  # <-- Replace this
OUTPUT_IMAGES_DIR = "effocr_traindata/images"
OUTPUT_LABELS_PATH = "effocr_traindata/labels.tsv"

# Ensure output dirs exist
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)

def extract_cells(json_path, image_path, output_image_dir, label_list, counter):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img = Image.open(image_path)

    for shape in data.get("shapes", []):
        if "human_output" not in shape:
            continue

        text = shape["human_output"].get("human_corrected_text", "").strip()
        if not text:
            continue

        # Bounding box
        points = shape["points"]
        (x1, y1), (x2, y2) = points
        left, upper = int(min(x1, x2)), int(min(y1, y2))
        right, lower = int(max(x1, x2)), int(max(y1, y2))

        # Crop and save image
        cropped = img.crop((left, upper, right, lower))
        filename = f"{counter:06d}.jpg"
        cropped.save(os.path.join(output_image_dir, filename))

        # Add to label list
        label_list.append((filename, text))
        counter += 1

    return counter

def main():
    label_entries = []
    counter = 0

    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(root, file)
                jpg_path = json_path.replace(".json", ".jpg")

                if not os.path.exists(jpg_path):
                    continue

                counter = extract_cells(
                    json_path, jpg_path,
                    OUTPUT_IMAGES_DIR, label_entries,
                    counter
                )

    # Write labels.tsv
    with open(OUTPUT_LABELS_PATH, "w", encoding="utf-8") as f:
        for filename, text in label_entries:
            f.write(f"{filename}\t{text}\n")

    print(f"Processed {counter} cells.")

if __name__ == "__main__":
    main()
