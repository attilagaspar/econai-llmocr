# this script converts layoutparser annotations to CVAT XML format
# and uploads the resulting zip to Google Cloud Storage
import os
import json
import shutil
import xml.etree.ElementTree as ET
from glob import glob
import zipfile
import time
from google.cloud import storage


bucket_name = "layout_parsing_output"  # Replace with your GCS bucket name


def safe_rmtree(path, tries=10, delay=0.5):
    """Retry rmtree on Windows file-locking."""
    for i in range(tries):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            if i == tries - 1:
                print(f"Could not delete {path}; please remove manually.")
                return
            time.sleep(delay)

def convert_layoutparser_to_cvat_xml(json_path, images_dir, output_zip):
    # --- Load LayoutParser JSON ---
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Map image_id → info, and collect annotations per image
    img_id2info = {img['id']: img for img in data['images']}
    img_id2anns = {}
    label_set = set()
    for ann in data['annotations']:
        img_id2anns.setdefault(ann['image_id'], []).append(ann)
        label_set.add(str(ann['category_id']))

    # Prepare a map category_id → label name (if categories section exists)
    category_id2label = {}
    for cat in data.get('categories', []):
        category_id2label[str(cat['id'])] = cat.get('name', str(cat['id']))
    # Fallback: use the raw category_id string
    for lbl in list(label_set):
        category_id2label.setdefault(lbl, lbl)

    # --- Create temp folder structure ---
    tmpdir = "_tmpcvat"
    if os.path.exists(tmpdir):
        safe_rmtree(tmpdir)
    os.makedirs(tmpdir)
    images_out = os.path.join(tmpdir, "images")
    os.makedirs(images_out)

    # Copy images over
    for img in data['images']:
        src = os.path.join(images_dir, img['file_name'])
        if not os.path.isfile(src):
            print(f"Warning: image {src} not found, skipping.")
            continue
        shutil.copy(src, os.path.join(images_out, img['file_name']))

    # --- Build CVAT XML tree ---
    root = ET.Element("annotations")
    # version
    version = ET.SubElement(root, "version")
    version.text = "1.1"

    # meta → task → labels
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    labels_el = ET.SubElement(task, "labels")
    for lbl in sorted(label_set):
        label_el = ET.SubElement(labels_el, "label")
        name_el = ET.SubElement(label_el, "name")
        name_el.text = category_id2label[lbl]
        ET.SubElement(label_el, "attributes")  # empty attributes

    # image elements
    for img in data['images']:
        # skip missing
        if not os.path.isfile(os.path.join(images_out, img['file_name'])):
            continue
        img_attrs = {
            "id": str(img['id']),
            "name": img['file_name'],
            "width": str(img['width']),
            "height": str(img['height']),
        }
        img_el = ET.SubElement(root, "image", img_attrs)
        for ann in img_id2anns.get(img['id'], []):
            lbl = str(ann['category_id'])
            x, y, w, h = ann['bbox']
            box_attrs = {
                "label": category_id2label[lbl],
                "occluded": "0",
                "source": "manual",
                "xtl": str(x),
                "ytl": str(y),
                "xbr": str(x + w),
                "ybr": str(y + h),
            }
            ET.SubElement(img_el, "box", box_attrs)

    # Write single annotations.xml
    xml_path = os.path.join(tmpdir, "annotations.xml")
    tree = ET.ElementTree(root)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    # --- Zip up ---
    with zipfile.ZipFile(output_zip, "w") as zf:
        # annotations.xml at root
        zf.write(xml_path, "annotations.xml")
        # images folder
        for imgfile in glob(os.path.join(images_out, "*")):
            zf.write(imgfile, os.path.join("images", os.path.basename(imgfile)))

    # Clean up
    safe_rmtree(tmpdir)
    print(f"Created CVAT 1.1 XML ZIP: {output_zip}")

def upload_to_gcs(bucket_name, source_file_path, destination_blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_path)
    print(f"File {source_file_path} uploaded to gs://{bucket_name}/{destination_blob_name}")

# Usage example:
# convert_layoutparser_to_cvat_voc("batch_1.json", "images", "cvat_voc.zip")
if __name__ == "__main__":
    # Example usage
    output_root = "output_compass"
    for batch_name in os.listdir(output_root):
        batch_path = os.path.join(output_root, batch_name)
        if not os.path.isdir(batch_path):
            continue
        json_path = os.path.join(batch_path, f"{batch_name}.json")
        images_dir = os.path.join(batch_path, "images")
        output_zip = os.path.join("cvat_upload", f"{batch_name}.zip")
        destination_blob_name = output_zip

        if not os.path.isfile(json_path):
            print(f"JSON file not found for batch {batch_name}, skipping.")
            continue
        if not os.path.isdir(images_dir):
            print(f"Images directory not found for batch {batch_name}, skipping.")
            continue
        os.makedirs("cvat_upload", exist_ok=True)

        # Skip creation and upload if output_zip already exists
        if os.path.exists(output_zip):
            print(f"{output_zip} already exists, skipping creation and upload.")
            continue
        convert_layoutparser_to_cvat_xml(json_path, images_dir, output_zip)
        upload_to_gcs(bucket_name, output_zip, destination_blob_name)
    #json_path = f"output/{batch_name}/{batch_name}.json"
    #images_dir = f"output/{batch_name}/images"
    #output_zip =f"cvat_upload/{batch_name}.zip"
    #destination_blob_name = f"{output_zip}"

    #convert_layoutparser_to_cvat_xml(json_path, images_dir, output_zip)
    #upload_to_gcs(bucket_name, output_zip, destination_blob_name)