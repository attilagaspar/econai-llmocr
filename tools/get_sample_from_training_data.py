import os
import sys
import random
import shutil

def select_and_copy_files(input_folder, output_folder, num_files):
    # Ensure the input folder exists
    if not os.path.isdir(input_folder):
        print(f"Error: The folder {input_folder} does not exist.")
        return

    # Ensure the output folder exists
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # Collect all files from the input folder and its subfolders
    all_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            all_files.append(os.path.join(root, file))

    # Check if there are enough files to select
    if len(all_files) < num_files:
        print(f"Error: The input folder contains only {len(all_files)} files, but {num_files} files were requested.")
        return

    # Randomly select X files
    selected_files = random.sample(all_files, num_files)

    # Copy selected files to the output folder
    for file_path in selected_files:
        shutil.copy(file_path, output_folder)
        print(f"Copied {file_path} to {output_folder}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python select_and_copy_files.py <input_folder> <output_folder> <num_files>")
    else:
        input_folder = sys.argv[1]
        output_folder = sys.argv[2]
        num_files = int(sys.argv[3])
        select_and_copy_files(input_folder, output_folder, num_files)