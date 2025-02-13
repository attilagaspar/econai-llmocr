import camelot

# Extract tables from a PDF file.
pdf_path = 'raw/1935mg_osszeiras_sample-2-9-1.pdf'

tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream', table_areas= ["0.78558349609375, 249.41888427734375, 4342.583984375, 2881.107177734375"])

# Print the first table as a DataFrame.
if tables:
    tables[0].to_csv("extracted_table.csv")  # Save as CSV to review
    tables[0].df.to_excel("extracted_table.xlsx")  # Optionally save as Excel
else:
    print("No tables detected in the PDF.")
