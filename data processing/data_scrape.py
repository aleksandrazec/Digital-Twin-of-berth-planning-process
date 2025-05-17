import pandas as pd
import xml.etree.ElementTree as ET
import requests
import os

def extract_data():
    output_dir = "vessel_info"
    os.makedirs(output_dir, exist_ok=True)

    csv_file = "xml_links.csv"
    df_links = pd.read_csv(csv_file)

    urls = df_links['xml_link'].dropna().tolist()

    counter=0

    for i, url in enumerate(urls):
        try:
            if i%4==0:
                counter=counter+1

            response = requests.get(url)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)

            data = []
            for vessel in root.iter('G_SQL1'):
                record = {child.tag: child.text for child in vessel}
                data.append(record)
        
            df = pd.DataFrame(data)

            if "ATA_TIME" in df.columns:
                filename = f"ATA{counter}.csv"
            elif "ATD_TIME" in df.columns:
                filename = f"ATD{counter}.csv"
            elif "ETA_TIME" in df.columns:
                filename = f"ETA{counter}.csv"
            elif "ETD_TIME" in df.columns:
                filename = f"ETD{counter}.csv"

            output_path = os.path.join(output_dir, filename)

            df.to_csv(output_path, index=False)


        except Exception as e:
            print(f"Failed to process {url}: {e}")