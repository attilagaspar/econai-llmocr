import os
import json

def process_json_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return

    def swap_labels(obj):
        if isinstance(obj, dict):
            if "label" in obj:
                if obj["label"] == "text_cell":
                    obj["label"] = "column_header"
                elif obj["label"] == "column_header":
                    obj["label"] = "text_cell"
            for v in obj.values():
                swap_labels(v)
        elif isinstance(obj, list):
            for item in obj:
                swap_labels(item)

    swap_labels(data)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def process_folder(folder):
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith('.json'):
                print(f"Processing file: {file}")
                filepath = os.path.join(root, file)
                process_json_file(filepath)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python change_labels_py.py <folder_path>")
        exit(1)
    process_folder(sys.argv[1])