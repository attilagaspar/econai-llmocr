import os
import sys
import json
import csv

def find_jsons_with_images(folder):
    jsons = []
    for root, _, files in os.walk(folder):
        json_files = [f for f in files if f.lower().endswith('.json')]
        jpg_files = set(f.lower() for f in files if f.lower().endswith('.jpg'))
        for jf in json_files:
            img_name = os.path.splitext(jf)[0] + '.jpg'
            if img_name in jpg_files:
                jsons.append(os.path.join(root, jf))
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

    all_rows = []
    json_files = find_jsons_with_images(input_folder)
    for json_path in json_files:
        print(json_path)
        all_rows.extend(process_json(json_path, input_folder))

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['name', 'title', 'firm', 'industry', 'share_capital', 'path'])
        writer.writerows(all_rows)

if __name__ == "__main__":
    main()