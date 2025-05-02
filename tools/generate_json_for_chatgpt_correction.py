import os
import json

input_dir = '../ocr_results'
output_dir = '../llm_input'
output_file = os.path.join(output_dir, 'combined_output.json')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

combined_data = []
file_counts = {}

for filename in os.listdir(input_dir):
    if filename.endswith('.bbocr'):
        file_path = os.path.join(input_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            filtered_data = [item for item in data if item.get('type') == 3]
            combined_data.extend(filtered_data)
            for item in filtered_data:
                item['unique_id'] = f"{filename}_{filtered_data.index(item)}"
            file_counts[filename] = len(filtered_data)

with open(output_file, 'w', encoding='utf-8') as outfile:
    json.dump(combined_data, outfile, ensure_ascii=False, indent=4)

total_count = sum(file_counts.values())

print("Counts by input file:")
for file, count in file_counts.items():
    print(f"{file}: {count}")

print(f"Total count of elements with 'type' = 3: {total_count}")