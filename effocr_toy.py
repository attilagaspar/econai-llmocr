from efficient_ocr import EffOCR
from IPython.display import Image, display
from pdf2image import convert_from_path

# Convert PDF pages to images (300 DPI is a common setting)
pdf_path='raw/1935mg_osszeiras_sample-2-9-1.pdf'
pages = convert_from_path(pdf_path, dpi=300)

# Initialize EffOCR as usual
effocr = EffOCR(
  config={
      'Recognizer': {
          'char': {
              'model_backend': 'onnx',
              'model_dir': './models',
              'hf_repo_id': 'dell-research-harvard/effocr_en/char_recognizer',
          },
          'word': {
              'model_backend': 'onnx',
              'model_dir': './models',
              'hf_repo_id': 'dell-research-harvard/effocr_en/word_recognizer',
          },
      },
      'Localizer': {
          'model_dir': './models',
          'hf_repo_id': 'dell-research-harvard/effocr_en',
          'model_backend': 'onnx'
      },
      'Line': {
          'model_dir': './models',
          'hf_repo_id': 'dell-research-harvard/effocr_en',
          'model_backend': 'onnx',
      },
  },
  data_json="coco_annotations.json"
)

# Process each page (or a specific page)
for i, page in enumerate(pages):
    page.save(f"page_{i}.jpg")  # optionally save the image
    result = effocr.infer(f"page_{i}.jpg", visualize = f'sample_viz_{i}.jpg')
    display(Image(f'sample_viz_{i}.jpg'))
    # Process the result as needed




# English


#results = effocr.infer("raw/test_table_piece.JPG")


#print(type(results))
#print(results)
#result_item = results[0]
#print(result_item.__dict__)