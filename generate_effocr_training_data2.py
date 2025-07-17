import os
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Set input/output paths
INPUT_DIR = "../census/agricultural_census1935_layout"  # <-- update if needed
OUTPUT_IMAGES_DIR = "effocr_traindata2/images"
OUTPUT_LABELS_PATH = "effocr_traindata2/labels.json"

# Ensure output dir exists
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)

def extract_text_snippets(json_path, image_path, output_image_dir, output_list, counter):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img = Image.open(image_path)

    for shape in data.get("shapes", []):
        if "human_output" not in shape or shape["label"] != "numerical_cell":
            continue

        text = shape["human_output"].get("human_corrected_text", "").strip()
        if not text:
            continue

        points = shape["points"]
        (x1, y1), (x2, y2) = points
        left, upper = int(min(x1, x2)), int(min(y1, y2))
        right, lower = int(max(x1, x2)), int(max(y1, y2))
        width, height = right - left, lower - upper

        filename = f"{counter:06d}.png"
        cropped = img.crop((left, upper, right, lower))
        cropped.save(os.path.join(output_image_dir, filename))

        output_list.append({
            "id": counter,
            "file_name": filename,
            "width": width,
            "height": height,
            "text": text
        })

        counter += 1

    return counter

def main():
    output = {"images": []}
    counter = 0

    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if not file.endswith(".json"):
                continue

            json_path = os.path.join(root, file)
            image_path = json_path.replace(".json", ".jpg")
            if not os.path.exists(image_path):
                continue

            counter = extract_text_snippets(
                json_path,
                image_path,
                OUTPUT_IMAGES_DIR,
                output["images"],
                counter
            )

    with open(OUTPUT_LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ Exported {counter} cropped images and label entries.")

if __name__ == "__main__":
    main()
