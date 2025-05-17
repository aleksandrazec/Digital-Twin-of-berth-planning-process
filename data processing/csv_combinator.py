import csv
import os
from collections import defaultdict
import pandas as pd


def combine_two_csv_files(file_path1, file_path2, output_file, column_name):

    file1_df = pd.read_csv(file_path1)
    file2_df = pd.read_csv(file_path2)
    
    merged_df = pd.merge(file1_df, file2_df, on=column_name, how="inner")
    merged_df.to_csv(output_file, index=False)

def combine_csv_files_from_folder(input_folder, output_file, id_column=0):

    input_files = [os.path.join(input_folder, f) for f in sorted(os.listdir(input_folder)) 
                  if f.lower().endswith('.csv')]
    
    if not input_files:
        print(f"Error: No CSV files in {input_folder}")
        return

    combined_data = {}  
    all_headers = set()
    file_headers = []  

    for file_index, file_path in enumerate(input_files):
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            try:
                headers = next(reader)
                file_headers.append(headers)
                all_headers.update(headers)
                
                for row in reader:
                    if len(row) <= id_column:
                        continue
                    id_val = row[id_column]
                    if id_val not in combined_data:
                        combined_data[id_val] = {}
                    for i, val in enumerate(row):
                        if i < len(headers):
                            combined_data[id_val][headers[i]] = val
            except StopIteration:
                print(f"Warning: Empty file skipped: {os.path.basename(file_path)}")
                continue

    all_headers = list(all_headers)
    if file_headers and id_column < len(file_headers[0]):
        id_header = file_headers[0][id_column]
        all_headers.remove(id_header)
        all_headers.insert(0, id_header)
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(all_headers)
        
        for id_val, data in combined_data.items():
            row = [data.get(h, '') for h in all_headers]
            writer.writerow(row)
    