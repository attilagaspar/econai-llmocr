#!/usr/bin/env python3
"""
Newspaper Text Extraction Script

This script extracts text content from processed newspaper layout JSON files
and concatenates them into a single text file following the natural reading order.

Usage: python extract_newspaper_text.py <input_folder> <output_text_file>
"""

import json
import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def natural_sort_key(text: str) -> List:
    """
    Generate a natural sorting key that handles numbers correctly.
    E.g., "page10" comes after "page2" instead of before it.
    """
    def convert(text_part):
        return int(text_part) if text_part.isdigit() else text_part.lower()
    
    return [convert(c) for c in re.split('([0-9]+)', text)]


def find_json_jpeg_pairs_ordered(input_folder: str) -> List[Tuple[str, str]]:
    """
    Find all JSON-JPEG pairs and return them in natural sorted order.
    """
    pairs = []
    
    print(f"🔍 Scanning directory: {input_folder}")
    
    # Collect all pairs first
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
    """
    Extract text from a shape element. Looks for OCR results or other text content.
    """
    # Check for tesseract_output (OCR results)
    tesseract_output = shape.get("tesseract_output", {})
    if tesseract_output and "ocr_text" in tesseract_output:
        ocr_text = tesseract_output["ocr_text"]
        if ocr_text and ocr_text.strip():
            return ocr_text.strip()
    
    # Check for other possible text fields
    if "text" in shape and shape["text"].strip():
        return shape["text"].strip()
    
    # Check for description or content fields
    for field in ["description", "content", "value"]:
        if field in shape and shape[field] and str(shape[field]).strip():
            return str(shape[field]).strip()
    
    return None


def process_json_file(json_path: str) -> str:
    """
    Process a single JSON file and extract text in reading order.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        shapes = data.get("shapes", [])
        if not shapes:
            return ""
        
        print(f"  📄 Processing: {os.path.basename(json_path)} ({len(shapes)} shapes)")
        
        # Separate elements by type
        szeles_cim_elements = []
        column_elements = []
        
        for shape in shapes:
            label = shape.get("label", "")
            
            if label == "szeles_cim":
                szeles_cim_elements.append(shape)
            elif label in ["szoveg", "hasabkozi_cim"]:
                # Only include elements that have column_number and row_number
                if "column_number" in shape and "row_number" in shape:
                    column_elements.append(shape)
        
        # Sort szeles_cim elements by their vertical position (top to bottom)
        szeles_cim_elements.sort(key=lambda s: min(p[1] for p in s.get("points", [[0, 0]])))
        
        # Sort column elements by column first, then by row within each column
        column_elements.sort(key=lambda s: (s.get("column_number", 999), s.get("row_number", 999)))
        
        # Extract text content
        text_parts = []
        
        # First add szeles_cim elements
        for shape in szeles_cim_elements:
            text = extract_text_from_shape(shape)
            if text:
                text_parts.append(f"[TITLE] {text}")
                print(f"    📰 szeles_cim: {text[:50]}...")
        
        # Then add column elements in reading order
        current_column = None
        for shape in column_elements:
            column_num = shape.get("column_number")
            row_num = shape.get("row_number")
            label = shape.get("label")
            
            # Add column separator when moving to a new column
            if current_column is not None and column_num != current_column:
                text_parts.append(f"\n[COLUMN {column_num}]\n")
            elif current_column is None:
                text_parts.append(f"[COLUMN {column_num}]")
            
            current_column = column_num
            
            text = extract_text_from_shape(shape)
            if text:
                if label == "hasabkozi_cim":
                    text_parts.append(f"[SUBTITLE] {text}")
                    print(f"    📝 hasabkozi_cim (Col{column_num}, Row{row_num}): {text[:50]}...")
                else:  # szoveg
                    text_parts.append(text)
                    print(f"    📖 szoveg (Col{column_num}, Row{row_num}): {text[:50]}...")
        
        # Join all text parts
        if text_parts:
            page_text = "\n\n".join(text_parts)
            return page_text
        else:
            print(f"    ⚠️  No text content found in {os.path.basename(json_path)}")
            return ""
            
    except Exception as e:
        print(f"  ❌ Error processing {json_path}: {e}")
        return ""


def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_newspaper_text.py <input_folder> <output_text_file>")
        print("\nThis script extracts text from processed newspaper layout JSON files")
        print("and concatenates them into a single text file in reading order:")
        print("- Folders and files are processed in natural numerical order")
        print("- Within each page: szeles_cim first, then column-by-column, top-to-bottom")
        print("- Only 'szeles_cim', 'szoveg', and 'hasabkozi_cim' elements are included")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_text_file = sys.argv[2]
    
    # Validate input
    if not os.path.exists(input_folder):
        print(f"❌ Input folder not found: {input_folder}")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_text_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 Starting newspaper text extraction...")
    print(f"📁 Input folder: {input_folder}")
    print(f"📄 Output text file: {output_text_file}")
    
    # Find all JSON-JPEG pairs in natural order
    pairs = find_json_jpeg_pairs_ordered(input_folder)
    
    if not pairs:
        print("❌ No JSON-JPEG pairs found!")
        sys.exit(1)
    
    # Process all files and collect text
    print(f"\n📋 Processing {len(pairs)} files...")
    all_text_parts = []
    processed = 0
    
    for i, (json_path, jpeg_path) in enumerate(pairs):
        print(f"\n[{i+1}/{len(pairs)}] " + "="*60)
        
        # Add file separator
        rel_path = os.path.relpath(json_path, input_folder)
        all_text_parts.append(f"\n{'='*80}")
        all_text_parts.append(f"FILE: {rel_path}")
        all_text_parts.append(f"{'='*80}\n")
        
        # Process the JSON file
        page_text = process_json_file(json_path)
        
        if page_text.strip():
            all_text_parts.append(page_text)
            processed += 1
            print(f"    ✅ Extracted {len(page_text)} characters")
        else:
            all_text_parts.append("[No text content found in this page]")
            print(f"    ⚠️  No text content extracted")
    
    # Write all text to output file
    try:
        final_text = "\n".join(all_text_parts)
        
        with open(output_text_file, 'w', encoding='utf-8') as f:
            # Add header
            f.write("NEWSPAPER TEXT EXTRACTION RESULTS\n")
            f.write(f"Generated from: {input_folder}\n")
            f.write(f"Total files processed: {len(pairs)}\n")
            f.write(f"Files with text content: {processed}\n")
            f.write(f"Total characters: {len(final_text)}\n")
            f.write("="*80 + "\n\n")
            
            # Add extracted text
            f.write(final_text)
        
        print(f"\n{'='*80}")
        print(f"🎉 Text extraction complete!")
        print(f"✅ Successfully processed: {processed}/{len(pairs)} files")
        print(f"📄 Output saved to: {output_text_file}")
        print(f"📊 Total characters extracted: {len(final_text):,}")
        
    except Exception as e:
        print(f"\n❌ Error writing output file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()