import os
# This script removes watermarks from images in a specified folder and saves the processed images to another folder.

# It uses the Python Imaging Library (PIL) to manipulate images.
# The script checks each pixel in the image and if it is a shade of gray (but not black), it sets it to black.
# This is a simple approach and may not work for all types of watermarks.
# It is recommended to use more advanced techniques for watermark removal, such as inpainting or deep learning methods.
#   However, this script serves as a basic example of how to process images in Python.
#
# Usage: python watermark_remover.py <input_folder> <output_folder>


import os
import sys
import cv2
import numpy as np

def remove_watermark(input_folder, output_folder, threshold):
    """
    Removes watermarks by setting every pixel brighter than the threshold to white.
    
    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder where processed images will be saved.
        threshold (int): Brightness threshold (0-255). Pixels brighter than this will be set to white.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".jpg"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Read the image using OpenCV
            img = cv2.imread(input_path)
            if img is None:
                print(f"Error: Could not read image {input_path}")
                continue

            # Convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Create a mask for pixels brighter than the threshold
            mask = gray > threshold

            # Set pixels brighter than the threshold to white
            img[mask] = [255, 255, 255]

            # Save the processed image
            cv2.imwrite(output_path, img)
            print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python watermark_remover.py <input_folder> <output_folder> <threshold>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    try:
        threshold = int(sys.argv[3])
        if not (0 <= threshold <= 255):
            raise ValueError
    except ValueError:
        print("Error: Threshold must be an integer between 0 and 255.")
        sys.exit(1)

    remove_watermark(input_folder, output_folder, threshold)