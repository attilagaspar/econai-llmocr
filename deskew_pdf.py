import os
import cv2
import fitz  # PyMuPDF
import tempfile
from pdf2image import convert_from_path
import img2pdf
from PIL import Image


def detect_skew_and_rotate(image):
    # Convert to grayscale
    gray = image.copy()  # if you need a separate copy

    # Invert and threshold
    _, binary = cv2.threshold(cv2.bitwise_not(gray), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Morphological closing to connect text lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Get text/table pixel positions
    coords = np.column_stack(np.where(morph > 0))
    if len(coords) < 100:
        return image  # skip mostly empty pages

    # Get skew angle
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Reject near-90 degree angles — those are false positives
    if abs(abs(angle) - 90) < 2:
        return image

    # Reject tiny skew
    if abs(angle) < 0.5:
        return image

    # Rotate image
    h_orig, w_orig = image.shape[:2]
    center = (w_orig // 2, h_orig // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    print("Before:", image.shape)
    rotated = cv2.warpAffine(image, M, (w_orig, h_orig), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    print("After:", rotated.shape)
    h, w = rotated.shape[:2]
    if h * w > 178956970:
        print("Warning: Image size exceeds 178956970 pixels")
    # Double-check for flipped page shape
    h_rot, w_rot = rotated.shape[:2]
    if abs(h_rot - w_orig) < 5 and abs(w_rot - h_orig) < 5:
        # Probably a wrong 90° rotation → revert
        return image

    return rotated






def deskew_pdf(input_pdf_path, output_pdf_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        print("Converting PDF to images...")
        images = convert_from_path(input_pdf_path, dpi=300)

        corrected_image_paths = []

        for i, image in enumerate(images):
            print(f"Processing page {i+1}/{len(images)}...")
            #image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            image_cv = np.array(image.convert("L"))  # grayscale mode

            corrected = detect_skew_and_rotate(image_cv)

            output_img_path = os.path.join(temp_dir, f"page_{i+1}.jpg")
            #cv2.imwrite(output_img_path, corrected)
            #cv2.imwrite(output_img_path, corrected, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            #cv2.imwrite(output_img_path, corrected, [cv2.IMWRITE_JPEG_QUALITY, 75])
            im = Image.fromarray(corrected)
            im.save(output_img_path, format="JPEG", quality=40,  dpi=(300, 300), optimize=True)
            corrected_image_paths.append(output_img_path)

        print("Creating deskewed PDF...")
        with open(output_pdf_path, "wb") as f_out:
            f_out.write(img2pdf.convert(corrected_image_paths))

        print(f"Output saved to {output_pdf_path}")


if __name__ == "__main__":
    import argparse
    import numpy as np
    import glob
    import os

    parser = argparse.ArgumentParser(description="Deskew tables in a PDF.")
    parser.add_argument("input_pdf", help="Path to the input PDF, can include wildcard '*'")
    parser.add_argument("output_pdf", help="Path for the output PDF, must include '*' if using wildcard input")
    args = parser.parse_args()

    input_pattern = args.input_pdf
    output_pattern = args.output_pdf

    if "*" in input_pattern:
        input_files = glob.glob(input_pattern)
        if "*" not in output_pattern:
            raise ValueError("If using wildcard input, output pattern must also include '*'.")

        for in_file in input_files:
            base = os.path.splitext(os.path.basename(in_file))[0]
            out_file = output_pattern.replace("*", base)
            print(f"Processing {in_file} → {out_file}")
            deskew_pdf(in_file, out_file)
    else:
        deskew_pdf(input_pattern, output_pattern)

