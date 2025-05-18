import pandas as pd
from datetime import timedelta

def main(
    estimated_file='estimated_final.csv',
    assignments_file='berth_assignments.csv',
    actual_file='actual_times.csv',
    output_file='augmented_berth_assignments.csv'
):
    estimated_df = pd.read_csv(estimated_file, parse_dates=['ETA_TIME', 'ETD_TIME'])
    assignments_df = pd.read_csv(assignments_file, parse_dates=['ASSIGNED_START', 'ESTIMATED_END'])
    actual_df = pd.read_csv(actual_file, parse_dates=['ATA_TIME', 'ATD_TIME'])

    merged_df = assignments_df.merge(actual_df[['CALL_SIGN', 'ATA_TIME', 'ATD_TIME']], on='CALL_SIGN', how='left')
    augmented_df = merged_df.merge(
        estimated_df[['CALL_SIGN', 'WEATHER_IMPACT_PCT', 'CONGESTION_IMPACT_PCT',
                      'EFFECTIVENESS_SCORE', 'RELIABILITY_SCORE', 'WORK_ENV_SCORE']],
        on='CALL_SIGN',
        how='left'
    )

    augmented_df['EFFECTIVENESS_SCORE'] = augmented_df['EFFECTIVENESS_SCORE'].fillna(0.5)
    augmented_df['RELIABILITY_SCORE'] = augmented_df['RELIABILITY_SCORE'].fillna(0.5)
    augmented_df['WORK_ENV_SCORE'] = augmented_df['WORK_ENV_SCORE'].fillna(0.5)
    augmented_df['WEATHER_IMPACT_PCT'] = augmented_df['WEATHER_IMPACT_PCT'].fillna(0.5)
    augmented_df['CONGESTION_IMPACT_PCT'] = augmented_df['CONGESTION_IMPACT_PCT'].fillna(0.5)

    augmented_df['PRIORITY_SCORE'] = (
        augmented_df['EFFECTIVENESS_SCORE'] +
        augmented_df['RELIABILITY_SCORE'] +
        augmented_df['WORK_ENV_SCORE'] -
        (augmented_df['WEATHER_IMPACT_PCT'] + augmented_df['CONGESTION_IMPACT_PCT'])
    )

    augmented_df = augmented_df.sort_values(by='PRIORITY_SCORE', ascending=False)

    berth_schedule = {berth: [] for berth in augmented_df['BERTH'].unique()}
    new_assignments = []

    for _, vessel in augmented_df.iterrows():
        vessel_id = vessel['CALL_SIGN']
        berth = vessel['BERTH']
        ata = vessel['ATA_TIME']
        atd = vessel['ATD_TIME']
        duration = vessel['ESTIMATED_END'] - vessel['ASSIGNED_START']
        latest_start = min(atd - duration, ata)
        earliest_start = vessel['ASSIGNED_START']
        schedule = berth_schedule[berth]

        assigned_start = None
        candidate_start = earliest_start
        while candidate_start <= latest_start:
            candidate_end = candidate_start + duration
            overlap = any(not (candidate_end <= s[0] or candidate_start >= s[1]) for s in schedule)
            if not overlap:
                assigned_start = candidate_start
                break
            candidate_start += timedelta(minutes=30)

        if assigned_start is None:
            assigned_start = min(vessel['ASSIGNED_START'], ata)
        assigned_end = assigned_start + duration

        berth_schedule[berth].append((assigned_start, assigned_end))
        berth_schedule[berth].sort()

        new_assignments.append({
            'CALL_SIGN': vessel_id,
            'BERTH': berth,
            'ADJUSTED_START': assigned_start,
            'ADJUSTED_END': assigned_end
        })

    adjusted_df = pd.DataFrame(new_assignments)
    adjusted_df.to_csv(output_file, index=False)
