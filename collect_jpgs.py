import os
import sys
import shutil
from pathlib import Path

def sanitize_filename(filename):
    """Replace characters that are invalid in filenames with underscores"""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def collect_jpgs(input_folder, output_folder=None):
    """
    Recursively collect all JPG files and copy them to a 'collected' subfolder
    with names augmented by their relative paths.
    """
    input_path = Path(input_folder)
    
    if not input_path.exists():
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return
    
    if not input_path.is_dir():
        print(f"Error: '{input_folder}' is not a directory.")
        return
    
    # Set output folder - default to 'collected' subfolder in input folder
    if output_folder is None:
        output_path = input_path / "collected"
    else:
        output_path = Path(output_folder)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(exist_ok=True)
    
    print(f"Searching for JPG files in: {input_path}")
    print(f"Output folder: {output_path}")
    print("-" * 50)
    
    jpg_files = []
    
    # Find all JPG files recursively
    for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG']:
        jpg_files.extend(input_path.rglob(ext))
    
    if not jpg_files:
        print("No JPG files found.")
        return
    
    print(f"Found {len(jpg_files)} JPG files.")
    
    copied_count = 0
    skipped_count = 0
    
    for jpg_file in jpg_files:
        try:
            # Get relative path from input folder
            relative_path = jpg_file.relative_to(input_path)
            
            # Create new filename by replacing path separators with underscores
            # and keeping the original extension
            path_parts = list(relative_path.parts[:-1])  # All parts except filename
            original_name = relative_path.name
            name_without_ext = relative_path.stem
            extension = relative_path.suffix
            
            if path_parts:
                # Join path parts with underscores and add to filename
                path_prefix = "_".join(path_parts) + "_"
                new_name = sanitize_filename(path_prefix + original_name)
            else:
                # File is in root directory
                new_name = original_name
            
            # Ensure unique filename in case of duplicates
            output_file_path = output_path / new_name
            counter = 1
            base_name = sanitize_filename(path_prefix + name_without_ext) if path_parts else name_without_ext
            
            while output_file_path.exists():
                new_name = f"{base_name}_{counter:03d}{extension}"
                output_file_path = output_path / new_name
                counter += 1
            
            # Copy the file
            shutil.copy2(jpg_file, output_file_path)
            print(f"Copied: {relative_path} -> {new_name}")
            copied_count += 1
            
        except Exception as e:
            print(f"Error processing {jpg_file}: {e}")
            skipped_count += 1
    
    print("-" * 50)
    print(f"Processing complete!")
    print(f"Files copied: {copied_count}")
    print(f"Files skipped (errors): {skipped_count}")
    print(f"Output directory: {output_path.absolute()}")

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python collect_jpgs.py <input_folder> [output_folder]")
        print("If output_folder is not specified, files will be copied to <input_folder>/collected")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) == 3 else None
    
    collect_jpgs(input_folder, output_folder)

if __name__ == "__main__":
    main()
