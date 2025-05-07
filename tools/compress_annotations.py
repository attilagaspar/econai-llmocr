import os
import zipfile


def compress_images_subfolders(output_dir, compressed_dir):
    """
    Compress the "images" subfolder of each immediate subfolder in `output_dir` into a zip file
    and save it in `compressed_dir`.

    Args:
        output_dir (str): Path to the output directory containing subfolders.
        compressed_dir (str): Path to the directory where compressed zip files will be saved.
    """
    # Ensure the compressed output directory exists
    os.makedirs(compressed_dir, exist_ok=True)

    # Iterate over immediate subfolders in the output directory
    for subfolder in os.listdir(output_dir):
        subfolder_path = os.path.join(output_dir, subfolder)

        # Skip if it's not a directory
        if not os.path.isdir(subfolder_path):
            continue

        # Check if the "images" subfolder exists
        images_folder = os.path.join(subfolder_path, "images")
        if not os.path.exists(images_folder) or not os.path.isdir(images_folder):
            print(f"Skipping {subfolder}: 'images' subfolder not found.")
            continue

        # Check if the "images" folder contains at least one image and a JSON file
        contains_images = any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(images_folder))
        contains_json = any(f.lower().endswith('.json') for f in os.listdir(images_folder))
        if not (contains_images and contains_json):
            print(f"Skipping {subfolder}: 'images' folder does not contain both images and a JSON file.")
            continue

        # Create the zip file
        zip_filename = f"{subfolder}_annotations.zip"
        zip_path = os.path.join(compressed_dir, zip_filename)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(images_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, images_folder)  # Relative path inside the zip
                    zipf.write(file_path, arcname)

        print(f"Compressed {subfolder} -> {zip_path}")


if __name__ == "__main__":
    # Define paths
    output_dir = "../output"  # Replace with the path to your /output directory
    compressed_dir = "../compressed_outputs"  # Replace with the path to save compressed zips

    # Run the compression
    compress_images_subfolders(output_dir, compressed_dir)