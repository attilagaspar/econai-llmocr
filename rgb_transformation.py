import os
from PIL import Image

# Define the directory containing the images.
input_dir = "output/cropped_images"

# Loop through all files in the directory.
for filename in os.listdir(input_dir):
    # Process common image file types.
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        file_path = os.path.join(input_dir, filename)
        try:
            # Open the image.
            with Image.open(file_path) as img:
                # Convert the image to RGB.
                img_rgb = img.convert("RGB")
                # Save the converted image (overwrites the original).
                img_rgb.save(file_path)
                print(f"Converted {filename} to RGB.")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
