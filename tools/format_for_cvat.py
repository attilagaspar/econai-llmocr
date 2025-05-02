import os
import json
import shutil
import zipfile

def prepare_cvat_zip(coco_json_path, image_dir, output_zip_path):
    temp_dir = "cvat_dataset"
    annotations_dir = os.path.join(temp_dir, "annotations")
    images_dir = os.path.join(temp_dir, "images")

    # Ensure clean directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(annotations_dir)
    os.makedirs(images_dir)

    # Copy and rename COCO JSON to CVAT-compatible format
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    coco_json_output = os.path.join(annotations_dir, "instances_default.json")
    with open(coco_json_output, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=4)

    # Copy images
    image_filenames = {img["file_name"] for img in coco_data["images"]}
    for img_file in os.listdir(image_dir):
        if img_file in image_filenames:
            shutil.copy(os.path.join(image_dir, img_file), images_dir)

    # Create ZIP
    zip_path = output_zip_path if output_zip_path.endswith(".zip") else output_zip_path + ".zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(temp_dir):
            print(temp_dir)
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, temp_dir)
                zipf.write(file_path, arcname)

    print(f"✅ CVAT-compatible ZIP created: {zip_path}")


coco_json_path = "../output/MagyarCompass_1933_2__pages111-160/MagyarCompass_1933_2__pages111-160.json"  # Replace with your COCO JSON file
image_dir = "../output/MagyarCompass_1933_2__pages111-160"  # Replace with the folder containing JPGs
output_zip_path = "../output/MagyarCompass_1933_2__pages111-160/output.zip"  # The final ZIP file for CVAT import

prepare_cvat_zip(coco_json_path, image_dir, output_zip_path)