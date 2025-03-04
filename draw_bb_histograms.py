import os
import json
import matplotlib.pyplot as plt

# Directory containing the .bb files.
bb_dir = "bounding_boxes"
# Directory to save histogram images.
hist_dir = "bb_histograms"
if not os.path.exists(hist_dir):
    os.makedirs(hist_dir)

# Iterate over each .bb file in the bounding_boxes folder.
for filename in os.listdir(bb_dir):
    if filename.lower().endswith(".bb"):
        bb_file_path = os.path.join(bb_dir, filename)
        with open(bb_file_path, "r", encoding="utf-8") as f:
            bb_data = json.load(f)
        
        # The .bb file is assumed to be a list of pages,
        # each page is a dict with a "tables" key.
        areas = []
        for page in bb_data:
            tables = page.get("tables", [])
            for table in tables:
                # Each table should have a "cells" key, which is a list of cell dictionaries.
                cells = table.get("cell_bboxes", [])
                for cell in cells:
                    bbox = cell
                    if bbox and len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        # Compute the area.
                        area = (x2 - x1) * (y2 - y1)
                        areas.append(area)
        
        if areas:
            # Create a histogram of cell areas.
            plt.figure(figsize=(8, 6))
            plt.hist(areas, bins=50, color="blue", alpha=0.7)
            plt.xlabel("Cell Area (pixels²)")
            plt.ylabel("Frequency")
            plt.title(f"Histogram of Cell Areas for {filename}")
            
            # Save the histogram image.
            hist_file = os.path.join(hist_dir, f"{os.path.splitext(filename)[0]}.png")
            plt.savefig(hist_file)
            plt.close()
            print(f"Saved histogram for {filename} to {hist_file}")
        else:
            print(f"No cell areas found in {filename}.")
