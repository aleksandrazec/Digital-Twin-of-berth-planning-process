import pandas as pd
from datetime import datetime, timedelta

def load_data():
    berth_df = pd.read_csv('./berth/simplified_berth_details.csv')
    
    vessels_df = pd.read_csv('./estimated_times.csv', parse_dates=['ETA_TIME', 'ETD_TIME'])
    
    return berth_df, vessels_df

def preprocess_data(berth_df, vessels_df):
    vessels_df = vessels_df.sort_values('ETA_TIME')
    
    vessels_df['VESSEL_MAX_DRAFT'] = pd.to_numeric(vessels_df['VESSEL_MAX_DRAFT'], errors='coerce')
    
    return berth_df, vessels_df

def assign_berths(berth_df, vessels_df):
    berth_availability = {berth: [] for berth in berth_df['BERTH'].unique()}
    
    assignments = []
    
    for _, vessel in vessels_df.iterrows():
        vessel_name = vessel['VESSEL_NAME']
        eta = vessel['ETA_TIME']
        etd = vessel['ETD_TIME']
        vessel_draft = vessel['VESSEL_MAX_DRAFT']
        vessel_size = vessel.get('MAX_SIZE', 0)  

        
        compatible_berths = berth_df[
            (berth_df['MAX_DRAFT'] >= vessel_draft) & 
            (berth_df['MAX_SIZE'] >= vessel_size) 
        ]
        
        if len(compatible_berths) == 0:
            print(f"No compatible berths found for {vessel_name}")
            continue
        
        best_berth = None
        best_start_time = None
        min_wait_time = timedelta.max
        
        for _, berth in compatible_berths.iterrows():
            berth_name = berth['BERTH']
            berth_schedule = berth_availability[berth_name]
            
            stay_duration = etd - eta if pd.notna(etd) else timedelta(hours=12) 

            available_start = find_available_slot(eta, stay_duration, berth_schedule)
            
            wait_time = available_start - eta if available_start > eta else timedelta(0)
            
            if wait_time < min_wait_time:
                best_berth = berth_name
                best_start_time = available_start
                min_wait_time = wait_time
        
        if best_berth:
            assignment = {
                'VESSEL_NAME': vessel_name,
                'BERTH': best_berth,
                'ETA': eta,
                'ASSIGNED_START': best_start_time,
                'ESTIMATED_END': best_start_time + stay_duration,
                'WAIT_TIME': min_wait_time
            }
            assignments.append(assignment)
            
            berth_availability[best_berth].append((best_start_time, best_start_time + stay_duration))
            berth_availability[best_berth].sort() 
    
    return pd.DataFrame(assignments)

def find_available_slot(requested_start, duration, schedule):

    if not schedule:
        return requested_start
    
    first_booking_start, first_booking_end = schedule[0]
    if requested_start + duration <= first_booking_start:
        return requested_start
    
    for i in range(len(schedule) - 1):
        current_end = schedule[i][1]
        next_start = schedule[i+1][0]
        
        available_start = max(requested_start, current_end)
        if available_start + duration <= next_start:
            return available_start
    
    last_booking_end = schedule[-1][1]
    return max(requested_start, last_booking_end)

def main():
    berth_df, vessels_df = load_data()
    berth_df, vessels_df = preprocess_data(berth_df, vessels_df)
    
    assignments = assign_berths(berth_df, vessels_df)
    assignments.to_csv('berth_assignments.csv', index=False)

