import os
import sys
import json
import csv

def find_jsons_with_images(folder):
    print(f"Searching for JSON files in: {folder}")
    if not os.path.exists(folder):
        print(f"ERROR: Folder does not exist: {folder}")
        return []
    
    jsons = []
    total_json_files = 0
    total_jpg_files = 0
    
    for root, _, files in os.walk(folder):
        json_files = [f for f in files if f.lower().endswith('.json')]
        jpg_files = set(f.lower() for f in files if f.lower().endswith('.jpg'))
        
        total_json_files += len(json_files)
        total_jpg_files += len(jpg_files)
        
        if json_files or jpg_files:
            print(f"In folder {root}: {len(json_files)} JSON files, {len(jpg_files)} JPG files")
        
        for jf in json_files:
            img_name = os.path.splitext(jf)[0] + '.jpg'
            if img_name in jpg_files:
                full_path = os.path.join(root, jf)
                jsons.append(full_path)
                print(f"  Found pair: {jf} + {img_name}")
            else:
                print(f"  No matching JPG for: {jf} (looking for {img_name})")
    
    print(f"\nSummary: Found {len(jsons)} JSON-JPG pairs out of {total_json_files} JSON files and {total_jpg_files} JPG files")
    return jsons

def process_json(json_path, base_folder):
    rows = []
    rel_path = os.path.relpath(json_path, base_folder)
    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except Exception:
            return rows
    shapes = data.get('shapes', [])
    for shape in shapes:
        openai_outputs = shape.get('openai_outputs', [])
        if not openai_outputs:
            continue
        
        # Merge all outputs, with later ones overriding earlier ones
        merged_response = {}
        for output in openai_outputs:
            response_str = output.get('response', '')
            if not response_str:
                continue
            
            # Parse the JSON response string
            try:
                response = json.loads(response_str)
                # Merge this response into the accumulated response
                # Later values override earlier values for the same keys
                merged_response.update(response)
            except (json.JSONDecodeError, TypeError):
                continue
        
        if not merged_response:
            continue
            
        firm_name = merged_response.get('firm_name', '')
        industry = merged_response.get('industry', '')
        share_capital = merged_response.get('share_capital', '')
        personal_names = merged_response.get('personal_names', [])
        for person in personal_names:
            name = person.get('name', '')
            title = person.get('title', '')
            rows.append([name, title, firm_name, industry, share_capital, rel_path])
    return rows

def main():
    if len(sys.argv) != 3:
        print("Usage: python collapse_firm_jsons_to_csv.py <input_folder> <output_csv>")
        sys.exit(1)
    input_folder = sys.argv[1]
    output_csv = sys.argv[2]
    
    print(f"Input folder: {input_folder}")
    print(f"Output CSV: {output_csv}")

    all_rows = []
    json_files = find_jsons_with_images(input_folder)
    
    if not json_files:
        print("\nNo JSON-JPG pairs found! Exiting.")
        return
    
    print(f"\nProcessing {len(json_files)} JSON files...")
    for i, json_path in enumerate(json_files):
        print(f"[{i+1}/{len(json_files)}] Processing: {json_path}")
        rows = process_json(json_path, input_folder)
        all_rows.extend(rows)
        print(f"  -> Found {len(rows)} person records")

    print(f"\nTotal records collected: {len(all_rows)}")
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['name', 'title', 'firm', 'industry', 'share_capital', 'path'])
        writer.writerows(all_rows)
    
    print(f"CSV file written: {output_csv}")

if __name__ == "__main__":
    main()