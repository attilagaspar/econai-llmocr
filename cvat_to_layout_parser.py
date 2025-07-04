# this script converts CVAT annotations to LayoutParser COCO format
# if 2 arguments are provided, it downloads the zip from GCS and processes it
# if 1 argument is provided, it processes a local zip file
import os
import json
import zipfile
import shutil
import xml.etree.ElementTree as ET
from google.cloud import storage

def download_from_gcs(bucket_name, blob_name, dest_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(dest_path)
    print(f"Downloaded {blob_name} from bucket {bucket_name} to {dest_path}")

def parse_cvat_xml(xml_path, images_dir):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Parse images and annotations
    images = []
    annotations = []
    categories = []
    label_name_to_id = {}
    ann_id = 1

    # Parse categories
    #for label in root.findall("./meta/task/labels/label"):
    for label in root.findall("./meta/job/labels/label"):
        name = label.find("name").text
        cat_id = len(categories) + 1
        label_name_to_id[name] = cat_id
        categories.append({"id": cat_id, "name": name})
    # After parsing categories
    print("Parsed labels:", list(label_name_to_id.keys()))
    # Parse images and boxes
    for img_el in root.findall("image"):
        img_id = int(img_el.attrib["id"])
        file_name = img_el.attrib["name"]
        width = int(img_el.attrib["width"])
        height = int(img_el.attrib["height"])
        images.append({
            "id": img_id,
            "file_name": file_name,
            "width": width,
            "height": height
        })
        for box in img_el.findall("box"):
            label = box.attrib["label"]
            cat_id = label_name_to_id[label]
            x = float(box.attrib["xtl"])
            y = float(box.attrib["ytl"])
            x2 = float(box.attrib["xbr"])
            y2 = float(box.attrib["ybr"])
            w = x2 - x
            h = y2 - y
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            ann_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

def main(bucket_name, zip_blob_name):
    batch_name = os.path.splitext(os.path.basename(zip_blob_name))[0]
    out_dir = os.path.join("model_training_input", batch_name)
    os.makedirs(out_dir, exist_ok=True)
    images_out = os.path.join(out_dir, "images")
    os.makedirs(images_out, exist_ok=True)

    # Download zip from GCS
    local_zip = os.path.join(out_dir, f"{batch_name}.zip")
    download_from_gcs(bucket_name, zip_blob_name, local_zip)

    # Extract zip
    with zipfile.ZipFile(local_zip, "r") as zf:
        zf.extractall(out_dir)

    # Parse CVAT XML
    xml_path = os.path.join(out_dir, "annotations.xml")
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"annotations.xml not found in {out_dir}")

    coco_json = parse_cvat_xml(xml_path, images_out)

    # Move images to images_out
    images_folder = os.path.join(out_dir, "images")
    if not os.path.isdir(images_folder):
        # Sometimes images are extracted to a different path
        images_folder = os.path.join(out_dir, "images")
    for img in coco_json["images"]:
        src = os.path.join(out_dir, "images", img["file_name"])
        dst = os.path.join(images_out, img["file_name"])
        if os.path.isfile(src):
            shutil.move(src, dst)

    # Save LayoutParser COCO JSON
    json_path = os.path.join(out_dir, f"{batch_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(coco_json, f, indent=2)
    print(f"Saved LayoutParser COCO JSON to {json_path}")
    print(f"Images are in {images_out}")

if __name__ == "__main__":
    # Example usage:
    # python cvat_to_layoutparser.py <bucket_name> <zip_blob_name>
    # or
    # python cvat_to_layoutparser.py <local_zip_path>
    import sys
    if len(sys.argv) == 2:
        local_zip = sys.argv[1]
        batch_name = os.path.splitext(os.path.basename(local_zip))[0]
        out_dir = os.path.join("model_training_input", batch_name)
        os.makedirs(out_dir, exist_ok=True)
        images_out = os.path.join(out_dir, "images")
        os.makedirs(images_out, exist_ok=True)

        # Copy zip to out_dir if not already there
        if os.path.abspath(local_zip) != os.path.abspath(os.path.join(out_dir, f"{batch_name}.zip")):
            shutil.copy(local_zip, os.path.join(out_dir, f"{batch_name}.zip"))
        local_zip = os.path.join(out_dir, f"{batch_name}.zip")

        # Extract zip
        with zipfile.ZipFile(local_zip, "r") as zf:
            zf.extractall(out_dir)

        # Parse CVAT XML
        xml_path = os.path.join(out_dir, "annotations.xml")
        if not os.path.isfile(xml_path):
            raise FileNotFoundError(f"annotations.xml not found in {out_dir}")

        coco_json = parse_cvat_xml(xml_path, images_out)

        # Move images to images_out
        images_folder = os.path.join(out_dir, "images")
        for img in coco_json["images"]:
            src = os.path.join(out_dir, "images", img["file_name"])
            dst = os.path.join(images_out, img["file_name"])
            if os.path.isfile(src):
                shutil.move(src, dst)

        # Save LayoutParser COCO JSON
        json_path = os.path.join(out_dir, f"{batch_name}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(coco_json, f, indent=2)
        print(f"Saved LayoutParser COCO JSON to {json_path}")
        print(f"Images are in {images_out}")

    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python cvat_to_layoutparser.py <bucket_name> <zip_blob_name>")
        print("   or: python cvat_to_layoutparser.py <local_zip_path>")
        exit(1)