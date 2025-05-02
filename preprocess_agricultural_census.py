import fitz  # PyMuPDF


def split_double_pages(input_pdf_path, output_pdf_path, start_page, end_page, split_ratio):
    """
    Splits double-page horizontal scans into single pages at a fixed vertical ratio
    and saves them to a new PDF with compression.

    Args:
        input_pdf_path (str): Path to the input PDF file.
        output_pdf_path (str): Path to save the output PDF file.
        start_page (int): Starting page (1-based index).
        end_page (int): Ending page (1-based index).
        split_ratio (float): Vertical split ratio as a percentage (e.g., 51.2 for 51.2%).
    """
    # Open the input PDF
    pdf_document = fitz.open(input_pdf_path)
    output_pdf = fitz.open()  # Create an empty PDF for output

    # Ensure the page range is valid
    if start_page < 1 or end_page > len(pdf_document) or start_page > end_page:
        raise ValueError("Invalid page range specified.")

    # Iterate through the specified page range
    for page_num in range(start_page - 1, end_page):  # Convert to 0-based index
        print(f"Processing page {page_num + 1}")
        # Get the page
        page = pdf_document[page_num]
        width, height = page.rect.width, page.rect.height

        # Calculate the split position based on the split ratio
        split_position = width * (split_ratio / 100.0)

        # Define the rectangles for the left and right pages
        left_rect = fitz.Rect(0, 0, split_position, height)
        right_rect = fitz.Rect(split_position, 0, width, height)

        # Extract the left and right pages as new pages
        left_page = page.get_pixmap(clip=left_rect, dpi=150)  # Lower DPI
        right_page = page.get_pixmap(clip=right_rect, dpi=150)  # Lower DPI

        # Add the left and right pages to the output PDF
        output_pdf.new_page(width=left_page.width, height=left_page.height).insert_image(
            fitz.Rect(0, 0, left_page.width, left_page.height), stream=left_page.tobytes("jpeg")
        )
        output_pdf.new_page(width=right_page.width, height=right_page.height).insert_image(
            fitz.Rect(0, 0, right_page.width, right_page.height), stream=right_page.tobytes("jpeg")
        )

    # Save the output PDF
    output_pdf.save(output_pdf_path)
    print(f"Output PDF saved to: {output_pdf_path}")


if __name__ == "__main__":
    # Input parameters
    input_pdf_path = "raw/MSK_105_1935os_mg_osszeiras_telepulesenkent.pdf"  # Replace with the path to your input PDF
    output_pdf_path = "raw/cut_census_output.pdf"  # Replace with the desired output PDF path
    start_page = 5  # Replace with the starting page (1-based index)
    end_page = 272  # Replace with the ending page (1-based index)
    split_ratio = 54  # Replace with the desired split ratio (e.g., 51.2 for 51.2%)

    # Run the function
    split_double_pages(input_pdf_path, output_pdf_path, start_page, end_page, split_ratio)