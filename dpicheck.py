import fitz  # PyMuPDF

# Open your PDF
file_path = "raw/1935mg_osszeiras_sample-2-9-1.pdf"

doc = fitz.open(file_path)
page = doc[0]  # work with the first page

# Get the page size in points (width, height)
page_rect = page.mediabox  # or page.rect
width_points = page_rect.width  # in points
height_points = page_rect.height  # in points

# Convert to inches (1 inch = 72 points)
width_in = width_points / 72
height_in = height_points / 72

# Render the page as a pixmap at default resolution (72 DPI)
pix = page.get_pixmap()  
pixel_width = pix.width
pixel_height = pix.height

# Calculate DPI based on rendered pixel dimensions.
dpi_x = pixel_width / width_in
dpi_y = pixel_height / height_in

print("Page width:", width_points, "points =", width_in, "inches")
print("Rendered width:", pixel_width, "pixels")
print("Calculated DPI (x):", dpi_x)

print("Page height:", height_points, "points =", height_in, "inches")
print("Rendered height:", pixel_height, "pixels")
print("Calculated DPI (y):", dpi_y)
