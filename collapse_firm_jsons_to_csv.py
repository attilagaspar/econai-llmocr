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
        response = shape.get('openai_output', {}).get('response')
        if not response:
            continue
        firm_name = response.get('firm_name', '')
        industry = response.get('industry', '')
        personal_names = response.get('personal_names', [])
        for person in personal_names:
            name = person.get('name', '')
            title = person.get('title', '')
            rows.append([name, title, firm_name, industry, rel_path])
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
        writer.writerow(['name', 'title', 'firm', 'industry', 'path'])
        writer.writerows(all_rows)

if __name__ == "__main__":
    main()