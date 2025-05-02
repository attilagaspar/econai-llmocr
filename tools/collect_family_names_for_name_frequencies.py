# Description: This script reads the JSON output of the OCR-LLM pipeline and extracts the personal names from the ChatGPT results. It then writes the unique ID and personal names to a CSV file.

import json
import csv

# Define the input and output file paths
input_file_path = '../ocr_llm_results/combined_output.json'
output_file_path = '../ocr_llm_results/personal_names_rt.csv'

# Open and read the JSON file
with open(input_file_path, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# Open the CSV file for writing
with open(output_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Write the header row
    csv_writer.writerow(['unique_id', 'personal_name'])
    
    # Iterate over the JSON elements
    for element in data:
        if "korlátolt fel" not in element.get("ocr_text"):
            unique_id = element.get('unique_id')
            personal_names = element.get('chatgpt_result', {}).get('PERSONAL_NAMES', [])
            
            # Write each personal name to the CSV file
            for name in personal_names:
                csv_writer.writerow([unique_id, name])