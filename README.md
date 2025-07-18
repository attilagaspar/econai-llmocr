# EconAI LLM-OCR Pipeline

This project provides a complete pipeline for extracting, cleaning, and exporting tabular data from scanned documents using OCR and LLM-based post-processing. The workflow is designed for LabelMe-annotated layouts and produces a final Excel file with structured data.

## Workflow Overview

1. **Detect Table Superstructure**
   - `layout_superstructure_detect.py`
   - Assigns row and column indices (`super_row`, `super_column`) to each cell in LabelMe JSONs.
   - Optionally smooths cell coordinates for better alignment.

2. **Run OCR on Cells**
   - `add_ocr_to_layout_jsons.py`
   - Applies OCR to each **numerical** cell region and stores the result in the JSON.

3. **LLM Cleaning Phase**
   - `add_llm_cleaning_to_layout_jsons.py`
   - Uses a language model (OpenAI API) to clean and correct OCR output for text cells and column headers. 

4. **Export to Excel**
   - `json_join_excel_export.py`
   - Joins all processed JSONs and exports the final table to an Excel file, preserving the logical row/column structure.

+1 **Data Visualizing and Cleaning GUI (Windows)**
   - `ocr_llm_validator_human_gui.py`
   - Projects bounding boxes on pages and shows OCR/LLM results
   - Lets user to "hand-clean" the OCR/LLM results
   - Results (if exist) supersede LLM/OCR results in Step 4 



## Usage

You can run the entire pipeline with the provided shell script:

```bash
bash do_everything.sh
```

This will:
- Process all JSONs in the `input` folder,
- Write intermediate results to the `intermediate` folder,
- Export the final Excel file to `output/machines1935.xlsx`.

## Script Details

### `layout_superstructure_detect.py`
- Recursively processes LabelMe JSONs in the input folder.
- Assigns and smooths `super_row` and `super_column` for each cell.

### `add_ocr_to_layout_jsons.py`
- Runs OCR (Tesseract) on each cell region.
- Stores OCR results in the JSON.

### `add_llm_cleaning_to_layout_jsons.py`
- Cleans OCR output for text cells and column headers using an LLM (e.g., OpenAI API).
- Only processes relevant cell types.

### `json_join_excel_export.py`
- Recursively collects all processed JSONs.
- Exports the final table to Excel, aligning rows and columns according to the detected structure.

## Requirements

- Python 3.8+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- Python packages: `pandas`, `openai`, `Pillow`, `cv2` (OpenCV), etc.
- (Optional) OpenAI API key for LLM cleaning

## Folder Structure

- `input/` — Original LabelMe JSONs and images
- `intermediate/` — Intermediate JSONs after each processing step
- `output/` — Final Excel file

## Notes

- All scripts process folders recursively.
- The pipeline is modular; you can run each step independently if needed.
- The Excel export script ensures logical ordering of pages and proper alignment of multiline cell contents.

## License

MIT License

---

For questions or issues, please contact the repository
