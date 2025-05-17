import url_extractor
import data_scrape
import csv_combinator
import csv_nonempty
import seperator
import system_schedule
import weather_congestion

url_extractor.extract_urls
data_scrape.extract_data

input_folder = "./vessel_info"  
output_file = "combined_times.csv"
csv_combinator.combine_csv_files_from_folder(input_folder, output_file)

input_file = "./combined_times.csv"  
output_file = "vessels_with_full_info.csv"
csv_nonempty.filter_nonempty_rows(input_file, output_file)

input_file = "./vessels_with_full_info.csv"  
output_file1 = "actual_times.csv"
output_file2= "estimated_times.csv"
seperator.seperate_actual_and_estimated(input_file, output_file1, output_file2)

system_schedule.main()

weather_congestion.main()

input_file1 = "./estimated_times_with_impacts.csv"  
input_file2 = "./operator_parameters.csv"
output_file = "estimated_final.csv"
csv_combinator.combine_two_csv_files(input_file1, input_file2, output_file, "AGENT_NAME")