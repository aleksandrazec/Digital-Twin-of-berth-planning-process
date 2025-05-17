import csv
import os
from collections import defaultdict

def filter_nonempty_rows(input_file, output_file, id_column=0):
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
        except StopIteration:
            print(f"Error: Empty input file: {input_file}")
            return
            
        rows_to_keep = []
        num_columns = len(headers)
        
        for row in reader:
            if len(row) != num_columns:
                continue
                
            all_filled = all(val.strip() for i, val in enumerate(row) if i != id_column)
            if all_filled:
                rows_to_keep.append(row)
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows_to_keep)
    

if __name__ == "__main__":
    input_file = "./combined_times.csv"  
    output_file = "vessels_with_full_info.csv"
    filter_nonempty_rows(input_file, output_file)