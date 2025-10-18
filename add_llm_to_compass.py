#!/usr/bin/env python

import json
import openai
import os
import sys
import base64
import platform
from PIL import Image
import cv2
import numpy as np


def load_config(config_path):
    """Load LLM configuration from JSON file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        print(f"✓ Loaded config from {config_path}")
        return config
    except Exception as e:
        print(f"✗ Error loading config file {config_path}: {e}")
        sys.exit(1)


def validate_config(config):
    """Validate configuration and determine processing mode."""
    required_keys = ["OCR", "Image", "ocronly", "imageonly", "both", "model", "label_types", "llm_overwrite", "context"]
    for key in required_keys:
        if key not in config:
            print(f"✗ Missing required config key: {key}")
            sys.exit(1)
    
    ocr_enabled = config["OCR"]
    image_enabled = config["Image"]
    
    if not ocr_enabled and not image_enabled:
        print("✗ Error: Both OCR and Image are set to false. At least one must be enabled.")
        sys.exit(1)
    
    if ocr_enabled and image_enabled:
        mode = "both"
        prompt = config["both"]
    elif ocr_enabled and not image_enabled:
        mode = "ocronly"
        prompt = config["ocronly"]
    elif not ocr_enabled and image_enabled:
        mode = "imageonly"
        prompt = config["imageonly"]
    
    return mode, prompt



def find_json_image_pairs(input_dir):
    """Find all matching JSON-image pairs in the directory."""
    pairs = []
    
    print(f"🔍 Scanning directory: {input_dir}")
    
    total_files = 0
    total_json_files = 0
    
    for root, dirs, files in os.walk(input_dir):
        total_files += len(files)
        json_files = [f for f in files if f.lower().endswith('.json')]
        total_json_files += len(json_files)
        
        for json_file in json_files:
            json_path = os.path.join(root, json_file)
            base_name = os.path.splitext(json_file)[0]
            
            # Look for matching image
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                potential_img = os.path.join(root, base_name + ext)
                if os.path.exists(potential_img):
                    img_path = potential_img
                    break
            
            if img_path:
                pairs.append((json_path, img_path))
    
    print(f"✓ Found {len(pairs)} JSON-image pairs (from {total_json_files} JSON files)")
    
    if len(pairs) == 0:
        print("⚠️  No pairs found. Sample files:")
        try:
            root_files = os.listdir(input_dir)[:5]
            for f in root_files:
                print(f"    {f}")
        except Exception as e:
            print(f"    Error: {e}")
    
    return pairs


def should_process_shape(shape, label_types, model, mode, llm_overwrite, run_name=None):
    """Determine if a shape should be processed based on config."""
    # Check if label is in target types
    if shape.get("label") not in label_types:
        return False
    
    # Check if LLM output already exists based on run_name logic
    openai_outputs = shape.get("openai_outputs", [])
    
    if run_name is not None:
        # If run_name is specified, look for existing output with same run_name
        for output in openai_outputs:
            if output.get("run_name") == run_name:
                # Found existing output with same run_name
                if llm_overwrite:
                    return True  # Will overwrite this specific run_name
                else:
                    return False  # Skip processing as run_name already exists
        # No existing output with this run_name found, proceed with processing
        return True
    else:
        # Original logic for when run_name is not specified
        if not llm_overwrite:
            for output in openai_outputs:
                if output.get("model") == model and output.get("mode") == mode:
                    return False
        return True



def extract_roi_from_shape(img, shape):
    """Extract region of interest from image based on shape points."""
    if "points" not in shape or len(shape["points"]) < 2:
        return None
    
    pts = np.array(shape["points"], dtype=np.int32)
    x1, y1 = np.min(pts[:, 0]), np.min(pts[:, 1])
    x2, y2 = np.max(pts[:, 0]), np.max(pts[:, 1])
    
    # Ensure coordinates are within image bounds
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    roi = img[y1:y2, x1:x2]
    return roi if roi.size > 0 else None


def encode_image_to_base64(roi):
    """Convert image ROI to base64 string."""
    try:
        # Convert BGR to RGB if needed
        if len(roi.shape) == 3:
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(roi_rgb)
        else:
            pil_img = Image.fromarray(roi)
        
        # Save to bytes
        import io
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        
        # Encode to base64
        base64_string = base64.b64encode(image_bytes).decode('utf-8')
        return base64_string
    except Exception as e:
        print(f"    ⚠️  Error encoding image: {e}")
        return None


def call_openai_api(context, prompt, model, mode, roi=None, ocr_text=None):
    """Call OpenAI API with appropriate content."""
    try:
        messages = [
            {"role": "system", "content": "You are an expert assistant for processing historical document content."}
        ]
        
        if mode == "ocronly":
            content = f"{context}\n\n{prompt}\n\nOCR Text: {ocr_text}"
            messages.append({"role": "user", "content": content})
        
        elif mode == "imageonly":
            if roi is None:
                return None
            base64_image = encode_image_to_base64(roi)
            if base64_image is None:
                return None
            
            full_prompt = f"{context}\n\n{prompt}"
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            })
        
        elif mode == "both":
            if roi is None:
                return None
            base64_image = encode_image_to_base64(roi)
            if base64_image is None:
                return None
            
            content_text = f"{context}\n\n{prompt}\n\nOCR Text: {ocr_text}"
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": content_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            })
        
        # Set up OpenAI client
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            timeout=30
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"    🚨 Error during API call: {e}")
        return None



def update_openai_outputs(shape, response, model, mode, llm_overwrite, run_name=None):
    """Update or add to openai_outputs in the shape."""
    if "openai_outputs" not in shape:
        shape["openai_outputs"] = []
    
    new_output = {
        "response": response,
        "model": model,
        "mode": mode
    }
    
    # Add run_name if provided
    if run_name is not None:
        new_output["run_name"] = run_name
    
    # Find existing output based on run_name logic
    existing_index = None
    
    if run_name is not None:
        # Look for existing output with same run_name
        for i, output in enumerate(shape["openai_outputs"]):
            if output.get("run_name") == run_name:
                existing_index = i
                break
    else:
        # Original logic: look for same model and mode
        for i, output in enumerate(shape["openai_outputs"]):
            if output.get("model") == model and output.get("mode") == mode:
                existing_index = i
                break
    
    if existing_index is not None and llm_overwrite:
        # Replace existing output
        shape["openai_outputs"][existing_index] = new_output
    elif existing_index is None:
        # Add new output
        shape["openai_outputs"].append(new_output)

def process_json_image_pair(json_path, img_path, config, mode, prompt):
    """Process a single JSON-image pair."""
    print(f"    🔄 Processing: {os.path.basename(json_path)}")
    
    # Load JSON
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"    ✗ Error loading JSON: {e}")
        return False
    
    # Load image if needed
    img = None
    if config["Image"]:
        img = cv2.imread(img_path)
        if img is None:
            print(f"    ✗ Error loading image: {img_path}")
            return False
    
    shapes = data.get("shapes", [])
    label_types = config["label_types"]
    model = config["model"]
    llm_overwrite = config["llm_overwrite"]
    context = config["context"]
    ocr_enabled = config["OCR"]
    
    print(f"    📋 Found {len(shapes)} shapes in {os.path.basename(json_path)}")
    if shapes:
        found_labels = set(shape.get("label", "unknown") for shape in shapes)
        print(f"    🏷️  Labels found: {sorted(found_labels)}")
        print(f"    🎯 Target labels: {label_types}")
    
    # Filter shapes to process
    run_name = config.get("run_name")
    shapes_to_process = []
    for i, shape in enumerate(shapes):
        # First check basic criteria
        if not should_process_shape(shape, label_types, model, mode, llm_overwrite, run_name):
            continue
        
        # If OCR is enabled, only process shapes that have OCR text
        if ocr_enabled:
            tesseract_output = shape.get("tesseract_output", {})
            ocr_text = tesseract_output.get("ocr_text", "")
            if not ocr_text or ocr_text.strip() == "":
                continue  # Skip shapes without OCR text
        
        shapes_to_process.append((i, shape))
    
    if not shapes_to_process:
        print(f"    ⚠️  No shapes to process in {os.path.basename(json_path)} (found {len(shapes)} total shapes)")
        return True
    
    print(f"    🎯 Processing {len(shapes_to_process)} shapes in {os.path.basename(json_path)}")
    
    # Process each shape
    processed_count = 0
    for i, (shape_idx, shape) in enumerate(shapes_to_process):
        # Get OCR text if needed
        ocr_text = None
        if config["OCR"]:
            tesseract_output = shape.get("tesseract_output", {})
            ocr_text = tesseract_output.get("ocr_text", "")
        
        # Extract ROI if needed
        roi = None
        if config["Image"]:
            roi = extract_roi_from_shape(img, shape)
            if roi is None:
                continue
        
        # Call OpenAI API with context
        response = call_openai_api(context, prompt, model, mode, roi, ocr_text)
        
        if response is None:
            continue
        else:
            print(f"response: {response}...")  # Print first 100 chars of response
        
        # Update shape with API response
        update_openai_outputs(shape, response, model, mode, llm_overwrite, run_name)
        processed_count += 1
    
    # Save updated JSON
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        if processed_count > 0:
            print(f"    ✅ Processed {processed_count} shapes")
        return True
    except Exception as e:
        print(f"    ✗ Error saving JSON: {e}")
        return False


def main():
    print(len(sys.argv))
    if len(sys.argv) < 3:
        print("Usage: python llm_processor.py <config_file> <input_directory>")
        print("\nExample config file (llm_config.json):")
        print(json.dumps({
            "OCR": True,
            "Image": False,
            "model": "gpt-4-turbo",
            "label_types": ["text_cell", "List"],
            "llm_overwrite": False,
            "context": "This is from 'Magyar Compass', a historical publication...",
            "ocronly": "Please correct and clean up the following OCR text:",
            "imageonly": "Please describe what you see in this image:",
            "both": "Please analyze both the OCR text and image to provide the best interpretation:"
        }, indent=2))
        sys.exit(1)
    
    config_file = sys.argv[1]
    input_dir = sys.argv[2]
    
    print(f"📋 Arguments received:")
    print(f"    Config file: {config_file}")
    print(f"    Input directory: {input_dir}")
    print(f"    Arguments count: {len(sys.argv)}")
       
    if not os.path.exists(config_file):
        print(f"✗ Config file not found: {config_file}")
        sys.exit(1)
    
    if not os.path.exists(input_dir):
        print(f"✗ Input directory not found: {input_dir}")
        sys.exit(1)
    
    print("🚀 Starting LLM batch processing...")
    print(f"📁 Input directory: {input_dir}")
    print(f"⚙️  Config file: {config_file}")
    
    # Load and validate configuration
    config = load_config(config_file)
    mode, prompt = validate_config(config)
    
    print(f"🤖 Model: {config['model']}")
    if 'run_name' in config:
        print(f"🏷️  Run name: {config['run_name']}")
    print(f"🎯 Target labels: {config['label_types']}")
    print(f"🔄 Mode: {mode}")
    print(f"♻️  Overwrite existing: {config['llm_overwrite']}")
    print(f"📝 Context: {config['context'][:100]}..." if len(config['context']) > 100 else f"📝 Context: {config['context']}")
    
    # Find all JSON-image pairs
    pairs = find_json_image_pairs(input_dir)
    
    if not pairs:
        print("✗ No JSON-image pairs found!")
        sys.exit(1)
    
    # Process all pairs
    print(f"\n📋 Processing {len(pairs)} pairs...")
    successful = 0
    failed = 0
    
    for i, (json_path, img_path) in enumerate(pairs):
        print(f"\n[{i+1}/{len(pairs)}] " + "="*60)
        if process_json_image_pair(json_path, img_path, config, mode, prompt):
            successful += 1
        else:
            failed += 1
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"🎉 Batch processing complete!")
    print(f"✅ Successfully processed: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success rate: {successful/(successful+failed)*100:.1f}%" if (successful+failed) > 0 else "N/A")

if __name__ == "__main__":
    main()