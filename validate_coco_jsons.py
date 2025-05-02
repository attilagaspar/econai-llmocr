# this script checks for bad bounding boxes in the COCO format annotations
import json 

with open("../koren/census/train.json") as f:
    data = json.load(f)

bad_boxes = []

for ann in data["annotations"]:
    x, y, w, h = ann["bbox"]
    if w <= 0 or h <= 0:
        bad_boxes.append(ann["id"])
    elif any(map(lambda v: v is None or isinstance(v, float) and (v != v), [x, y, w, h])):  # checks for NaN
        bad_boxes.append(ann["id"])

print(f"Found {len(bad_boxes)} bad boxes.")
if bad_boxes:
    print("Bad annotation IDs:", bad_boxes)
