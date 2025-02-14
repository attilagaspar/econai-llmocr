import os
import cv2
import json

# Directories
img_dir = "output/cropped_images"
json_dir = "output/json_dir"
output_dir = "output/image_bbox_dir"

# Create the output directory if it doesn't exist.
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set the minimum score tolerance.
tolerance = 0.8

# Loop over every file in the cropped_images directory.
for filename in os.listdir(img_dir):
    # Process files with .png extension (since both images and JSON have .png extension).
    if filename.lower().endswith(".png"):
        img_path = os.path.join(img_dir, filename)
        json_path = os.path.join(json_dir, filename)  # Same filename in json_dir
        
        # Check if the corresponding JSON file exists.
        if not os.path.exists(json_path):
            print(f"JSON file not found for {filename}. Skipping...")
            continue
        
        # Load the image.
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not load image {img_path}. Skipping...")
            continue

        # Load the bounding box data from the JSON file.
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                annotations = json.load(f)
            except Exception as e:
                print(f"Error reading JSON file {json_path}: {e}")
                continue

        # Loop over each annotation and draw the rectangle if score >= tolerance.
        for ann in annotations:
            score = ann.get("score", 0)
            if score < tolerance:
                continue  # Skip if below tolerance

            label = ann.get("label", "").lower()  # Normalize label for comparison.
            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
                continue  # Skip invalid bbox.
            
            # Convert bbox coordinates to integers.
            x1, y1, x2, y2 = map(int, bbox)
            
            # Choose the color based on the label.
            if label == "table column":
                color = (0, 0, 255)      # Red for table columns.
            elif label == "table row":
                color = (0, 255, 255)    # Yellow for table rows.
            else:
                color = (255, 0, 0)      # Blue for everything else.
            
            # Draw the rectangle with a thickness of 2.
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)
        
        # Save the output image to the output directory with the same filename.
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, img)
        print(f"Processed {filename} -> {output_path}")
