import csv
import os
from collections import defaultdict

def seperate_actual_and_estimated(input_file, output_file1, output_file2, id_column=0):
    column1="ATA_TIME"
    column2="ATD_TIME"
    column3="ETA_TIME"
    column4="ETD_TIME"

    with open(input_file, mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        
        fieldnames = reader.fieldnames

        
        fields1 = [f for f in fieldnames if f != column3 and f!=column4]  
        fields2 = [f for f in fieldnames if f != column1 and f!=column2]

        with open(output_file1, mode='w', newline='') as outfile1, \
             open(output_file2, mode='w', newline='') as outfile2:
            
            writer1 = csv.DictWriter(outfile1, fieldnames=fields1)
            writer2 = csv.DictWriter(outfile2, fieldnames=fields2)
            
            writer1.writeheader()
            writer2.writeheader()
            
            for row in reader:
                row1 = {k: v for k, v in row.items() if k != column3 and k!=column4}
                writer1.writerow(row1)
                
                row2 = {k: v for k, v in row.items() if k != column1 and k!=column2}
                writer2.writerow(row2)

    