# collect_jpgs.py

A Python script that recursively collects all JPG files from an input folder and copies them to a "collected" subfolder with augmented filenames based on their original paths.

## Features:
1. **Recursive search**: Finds all JPG files (including .jpg, .jpeg, .JPG, .JPEG) in the input folder and all subfolders
2. **Path augmentation**: Adds the relative path to each filename using underscores as separators
3. **Safe filename handling**: Replaces invalid filename characters with underscores
4. **Duplicate handling**: If files with the same augmented name exist, adds a counter suffix
5. **Flexible output**: Can specify custom output folder or defaults to "collected" subfolder

## Usage:
```bash
python collect_jpgs.py <input_folder> [output_folder]
```

## Examples:
- `python collect_jpgs.py C:\MyPhotos` - Creates `C:\MyPhotos\collected\` 
- `python collect_jpgs.py C:\MyPhotos D:\AllPhotos` - Copies to `D:\AllPhotos\`

## File naming example:
If you have a file at `input_folder\subfolder1\subfolder2\photo.jpg`, it will be copied as:
`subfolder1_subfolder2_photo.jpg`

The script provides detailed progress output showing which files are being processed and gives a summary at the end.
