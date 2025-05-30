import pandas as pd
import numpy as np
import os

def weather_impact_by_hour(hour):
    if 0 <= hour < 6:
        return np.random.uniform(0.3, 0.5)
    elif 6 <= hour < 18:
        return np.random.uniform(0.05, 0.2)
    else:
        return np.random.uniform(0.2, 0.35)

def congestion_impact_by_hour(hour):
    if 8 <= hour < 18:
        return np.random.uniform(0.3, 0.6)
    else:
        return np.random.uniform(0.05, 0.2)

def aggregate_impacts(start_time, end_time, impact_func):
    hours_range = pd.date_range(start=start_time.floor('h'), end=end_time.floor('h'), freq='h')
    if hours_range.empty:
        return None
    hours = hours_range.hour
    impacts = [impact_func(h) for h in hours]
    return round(np.mean(impacts), 2)

def main():
    
    input_filename = "./estimated_times.csv"
    output_filename = "estimated_times_with_impacts.csv"

    df = pd.read_csv(input_filename)

    eta_parsed = pd.to_datetime(df["ETA_TIME"], errors='coerce')
    etd_parsed = pd.to_datetime(df["ETD_TIME"], errors='coerce')

    weather_impacts = []
    congestion_impacts = []

    for eta, etd in zip(eta_parsed, etd_parsed):
        if pd.isna(eta) or pd.isna(etd) or eta >= etd:
            weather_impacts.append("0")
            congestion_impacts.append("0")
        else:
            weather_impacts.append(aggregate_impacts(eta, etd, weather_impact_by_hour))
            congestion_impacts.append(aggregate_impacts(eta, etd, congestion_impact_by_hour))

    df["WEATHER_IMPACT_PCT"] = weather_impacts
    df["CONGESTION_IMPACT_PCT"] = congestion_impacts

    df.to_csv(output_filename, index=False)

