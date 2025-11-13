#!/usr/bin/env python3
"""
Raw Strike Description Collector

This script searches for the "TŐKE ÉS MUNKA" column in Népszava newspaper OCR results
and extracts the complete column content following the document structure.

Usage: python raw_strike_description_collector.py <input_folder> <output_folder>
"""

import json
import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import unicodedata


def remove_accents(text: str) -> str:
    """Remove Hungarian accents from text for matching."""
    return ''.join(c for c in unicodedata.normalize('NFD', text) 
                   if unicodedata.category(c) != 'Mn')


def normalize_text_for_search(text: str) -> str:
    """Normalize text for searching: lowercase, remove accents, clean whitespace."""
    if not text:
        return ""
    # Convert to lowercase and remove accents
    normalized = remove_accents(text.lower())
    # Clean up whitespace and special characters
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized


def natural_sort_key(text: str) -> List:
    """Generate a natural sorting key that handles numbers correctly."""
    def convert(text_part):
        return int(text_part) if text_part.isdigit() else text_part.lower()
    
    return [convert(c) for c in re.split('([0-9]+)', text)]


def find_json_jpeg_pairs_ordered(input_folder: str) -> List[Tuple[str, str]]:
    """Find all JSON-JPEG pairs and return them in natural sorted order."""
    pairs = []
    
    print(f"🔍 Scanning directory: {input_folder}")
    
    for root, dirs, files in os.walk(input_folder):
        json_files = [f for f in files if f.lower().endswith('.json')]
        
        for json_file in json_files:
            json_path = os.path.join(root, json_file)
            base_name = os.path.splitext(json_file)[0]
            
            # Look for matching JPEG
            jpeg_path = None
            for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
                potential_jpeg = os.path.join(root, base_name + ext)
                if os.path.exists(potential_jpeg):
                    jpeg_path = potential_jpeg
                    break
            
            if jpeg_path:
                pairs.append((json_path, jpeg_path))
    
    # Sort pairs by their full paths using natural sorting
    pairs.sort(key=lambda x: natural_sort_key(x[0]))
    
    print(f"✓ Found {len(pairs)} JSON-JPEG pairs in natural order")
    return pairs


def extract_text_from_shape(shape: Dict) -> Optional[str]:
    """Extract text from a shape element."""
    # Check for tesseract_output (OCR results)
    tesseract_output = shape.get("tesseract_output", {})
    if tesseract_output and "ocr_text" in tesseract_output:
        ocr_text = tesseract_output["ocr_text"]
        if ocr_text and ocr_text.strip():
            return ocr_text.strip()
    
    # Check for other possible text fields
    for field in ["text", "description", "content", "value"]:
        if field in shape and shape[field] and str(shape[field]).strip():
            return str(shape[field]).strip()
    
    return None


def contains_toke_munka(text: str) -> bool:
    """Check if text contains 'tőke' and 'munka' (case insensitive, with/without accents)."""
    if not text:
        return False
    
    normalized = normalize_text_for_search(text)
    
    # Check for various forms of the words
    toke_variants = ["tőke", "toke"]
    munka_variants = ["munka"]
    
    has_toke = any(variant in normalized for variant in toke_variants)
    has_munka = any(variant in normalized for variant in munka_variants)
    
    return has_toke and has_munka


def find_toke_munka_subtitle(shapes: List[Dict]) -> Optional[Dict]:
    """Find the hasabkozi_cim element containing 'tőke' and 'munka'."""
    for shape in shapes:
        if shape.get("label") == "hasabkozi_cim":
            text = extract_text_from_shape(shape)
            if text and contains_toke_munka(text):
                print(f"    🎯 Found 'TŐKE ÉS MUNKA' subtitle: {text[:50]}...")
                return shape
    return None


def find_szeles_cim(shapes: List[Dict]) -> Optional[str]:
    """Find and extract text from szeles_cim element (newspaper header)."""
    for shape in shapes:
        if shape.get("label") == "szeles_cim":
            text = extract_text_from_shape(shape)
            if text:
                print(f"    📰 Found newspaper header: {text[:50]}...")
                return text
    return None


def get_next_stopping_point(current_file_index: int, current_column: int, current_row: int, 
                           all_pairs: List[Tuple[str, str]]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Find the next hasabkozi_cim or szeles_cim that should stop the collection.
    Returns (file_index, column, row) or (None, None, None) if not found.
    """
    # Start from current position
    for file_idx in range(current_file_index, len(all_pairs)):
        json_path, _ = all_pairs[file_idx]
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            shapes = data.get("shapes", [])
            
            # Get elements with column and row info, sorted by reading order
            elements = []
            for shape in shapes:
                if (shape.get("label") in ["hasabkozi_cim", "szeles_cim", "szoveg"] and 
                    "column_number" in shape and "row_number" in shape):
                    elements.append(shape)
            
            # Sort by column, then row
            elements.sort(key=lambda s: (s.get("column_number", 999), s.get("row_number", 999)))
            
            for element in elements:
                elem_col = element.get("column_number")
                elem_row = element.get("row_number")
                elem_label = element.get("label")
                
                # Skip if we're still before our starting position
                if (file_idx == current_file_index and 
                    (elem_col < current_column or 
                     (elem_col == current_column and elem_row <= current_row))):
                    continue
                
                # Check if this is a stopping point
                if elem_label == "szeles_cim":
                    print(f"    🛑 Found stopping point: szeles_cim at file {file_idx}, col {elem_col}, row {elem_row}")
                    return file_idx, elem_col, elem_row
                elif elem_label == "hasabkozi_cim":
                    text = extract_text_from_shape(element)
                    if text and not contains_toke_munka(text):  # Different hasabkozi_cim
                        print(f"    🛑 Found stopping point: different hasabkozi_cim at file {file_idx}, col {elem_col}, row {elem_row}")
                        return file_idx, elem_col, elem_row
        
        except Exception as e:
            print(f"    ⚠️  Error reading {json_path}: {e}")
            continue
    
    return None, None, None


def collect_column_content(start_file_index: int, start_shape: Dict, all_pairs: List[Tuple[str, str]]) -> str:
    """
    Collect all content starting from the toke_munka subtitle until the next stopping point.
    """
    content_parts = []
    start_column = start_shape.get("column_number")
    start_row = start_shape.get("row_number")
    
    print(f"    📖 Starting collection from file {start_file_index}, column {start_column}, row {start_row}")
    
    # Find the stopping point
    stop_file, stop_col, stop_row = get_next_stopping_point(
        start_file_index, start_column, start_row, all_pairs
    )
    
    if stop_file is None:
        print(f"    📚 No stopping point found, will collect to end of available files")
    else:
        print(f"    📚 Will collect until file {stop_file}, column {stop_col}, row {stop_row}")
    
    # Collect content
    for file_idx in range(start_file_index, len(all_pairs)):
        json_path, _ = all_pairs[file_idx]
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            shapes = data.get("shapes", [])
            
            # Get elements with column and row info
            elements = []
            for shape in shapes:
                if (shape.get("label") in ["hasabkozi_cim", "szoveg"] and 
                    "column_number" in shape and "row_number" in shape):
                    elements.append(shape)
            
            # Sort by column, then row
            elements.sort(key=lambda s: (s.get("column_number", 999), s.get("row_number", 999)))
            
            page_content = []
            for element in elements:
                elem_col = element.get("column_number")
                elem_row = element.get("row_number")
                elem_label = element.get("label")
                
                # Skip if we're still before our starting position
                if (file_idx == start_file_index and 
                    (elem_col < start_column or 
                     (elem_col == start_column and elem_row <= start_row))):
                    continue
                
                # Stop if we've reached the stopping point
                if (stop_file is not None and file_idx == stop_file and 
                    elem_col == stop_col and elem_row == stop_row):
                    print(f"    🏁 Reached stopping point, ending collection")
                    break
                
                # Extract text content
                text = extract_text_from_shape(element)
                if text:
                    if elem_label == "hasabkozi_cim":
                        page_content.append(f"[SUBTITLE] {text}")
                    else:  # szoveg
                        page_content.append(text)
                    
                    print(f"    📝 Collected {elem_label} (Col{elem_col}, Row{elem_row}): {text[:50]}...")
            
            if page_content:
                content_parts.extend(page_content)
            
            # Stop if we've reached the stopping point
            if (stop_file is not None and file_idx >= stop_file):
                break
                
        except Exception as e:
            print(f"    ⚠️  Error processing {json_path}: {e}")
            continue
    
    return "\n\n".join(content_parts)


def generate_output_filename(json_path: str, input_folder: str) -> str:
    """Generate output filename based on input folder structure."""
    rel_path = os.path.relpath(json_path, input_folder)
    # Remove file extension and replace path separators with underscores
    filename_base = rel_path.replace('.json', '').replace(os.sep, '_').replace('/', '_')
    return f"toke_munka_{filename_base}.json"


def process_files(all_pairs: List[Tuple[str, str]], input_folder: str, output_folder: str) -> int:
    """Process all files looking for TŐKE ÉS MUNKA columns."""
    found_columns = 0
    
    for i, (json_path, jpeg_path) in enumerate(all_pairs):
        print(f"\n[{i+1}/{len(all_pairs)}] " + "="*60)
        print(f"  📄 Processing: {os.path.relpath(json_path, input_folder)}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            shapes = data.get("shapes", [])
            if not shapes:
                print(f"    ⚠️  No shapes found")
                continue
            
            # Look for TŐKE ÉS MUNKA subtitle
            toke_munka_shape = find_toke_munka_subtitle(shapes)
            if not toke_munka_shape:
                print(f"    ❌ No 'TŐKE ÉS MUNKA' subtitle found")
                continue
            
            # Find newspaper header
            newspaper_header = find_szeles_cim(shapes)
            if not newspaper_header:
                print(f"    ⚠️  No newspaper header (szeles_cim) found")
                newspaper_header = "Unknown Issue"
            
            # Collect column content
            print(f"    🔄 Collecting column content...")
            column_content = collect_column_content(i, toke_munka_shape, all_pairs)
            
            if not column_content.strip():
                print(f"    ⚠️  No content collected")
                continue
            
            # Generate output
            output_filename = generate_output_filename(json_path, input_folder)
            output_path = os.path.join(output_folder, output_filename)
            
            output_data = {
                "newspaper_header": newspaper_header,
                "column_content": column_content,
                "source_file": os.path.relpath(json_path, input_folder),
                "subtitle_text": extract_text_from_shape(toke_munka_shape),
                "content_length": len(column_content),
                "extraction_info": {
                    "start_column": toke_munka_shape.get("column_number"),
                    "start_row": toke_munka_shape.get("row_number"),
                    "total_characters": len(column_content)
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            found_columns += 1
            print(f"    ✅ Saved column to: {output_filename}")
            print(f"    📊 Content length: {len(column_content):,} characters")
            
        except Exception as e:
            print(f"    ❌ Error processing {json_path}: {e}")
            continue
    
    return found_columns


def main():
    if len(sys.argv) != 3:
        print("Usage: python raw_strike_description_collector.py <input_folder> <output_folder>")
        print("\nThis script searches for 'TŐKE ÉS MUNKA' columns in Népszava newspaper")
        print("OCR results and extracts the complete column content.")
        print("\nFeatures:")
        print("- Searches for hasabkozi_cim containing 'tőke' and 'munka' (case insensitive)")
        print("- Extracts newspaper header (szeles_cim) from the same page")
        print("- Collects content following document structure until next subtitle/header")
        print("- Outputs individual JSON files for each found column")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    
    # Validate input
    if not os.path.exists(input_folder):
        print(f"❌ Input folder not found: {input_folder}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    print("🚀 Starting TŐKE ÉS MUNKA column extraction...")
    print(f"📁 Input folder: {input_folder}")
    print(f"📁 Output folder: {output_folder}")
    
    # Find all JSON-JPEG pairs in natural order
    pairs = find_json_jpeg_pairs_ordered(input_folder)
    
    if not pairs:
        print("❌ No JSON-JPEG pairs found!")
        sys.exit(1)
    
    # Process all files
    print(f"\n📋 Processing {len(pairs)} files...")
    found_columns = process_files(pairs, input_folder, output_folder)
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"🎉 Extraction complete!")
    print(f"✅ Found 'TŐKE ÉS MUNKA' columns: {found_columns}")
    print(f"📄 Output files saved to: {output_folder}")
    
    if found_columns == 0:
        print("⚠️  No 'TŐKE ÉS MUNKA' columns were found in the processed files.")
        print("   Make sure the input files contain OCR results with the expected subtitle.")


if __name__ == "__main__":
    main()