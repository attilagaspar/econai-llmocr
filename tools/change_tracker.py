import os
import csv
from datetime import datetime

def collect_json_file_info(folder_path, output_csv):
    # List to store file information
    file_info_list = []

    # Walk through the folder structure
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                last_modified_time = os.path.getmtime(file_path)
                formatted_time = datetime.fromtimestamp(last_modified_time).strftime('%Y-%m-%d %H:%M:%S')
                file_info_list.append([file, file_path, formatted_time])

    # Write the collected information to a CSV file
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Filename', 'Path', 'Last Modified Date'])
        writer.writerows(file_info_list)

if __name__ == "__main__":
    # Define folder path and output CSV file path here
    folder_to_scan = "../../../compass/annotations"
    output_csv_file = "../../../compass/progress.csv"
    
    collect_json_file_info(folder_to_scan, output_csv_file)
    print(f"JSON file information has been saved to {output_csv_file}")