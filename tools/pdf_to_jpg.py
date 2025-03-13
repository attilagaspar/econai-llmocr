import os
import sys
from pdf2image import convert_from_path

def pdf_to_jpg(pdf_folder, output_folder):
    # Ensure the PDF folder exists
    if not os.path.isdir(pdf_folder):
        print(f"Error: The folder {pdf_folder} does not exist.")
        return

    # Ensure the output folder exists
    if not os.path.isdir(output_folder):
        print(f"Error: The folder {output_folder} does not exist.")
        return

    # Iterate through all PDF files in the input folder
    for pdf_filename in os.listdir(pdf_folder):
        if pdf_filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_filename)
            
            # Extract the PDF file name without extension
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

            # Create a subfolder with the same name as the PDF
            subfolder_path = os.path.join(output_folder, pdf_name)
            os.makedirs(subfolder_path, exist_ok=True)

            # Convert PDF to JPG
            images = convert_from_path(pdf_path, dpi=300)

            # Save each page as a JPG file
            for i, image in enumerate(images):
                image_path = os.path.join(subfolder_path, f"{pdf_name}_page_{i + 1}.jpg")
                image.save(image_path, 'JPEG')
                print(f"Saved {image_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pdf_to_jpg.py <pdf_folder> <output_folder>")
    else:
        pdf_folder = sys.argv[1]
        output_folder = sys.argv[2]
        pdf_to_jpg(pdf_folder, output_folder)