from typing import List, Optional

from fitparse import FitFile
import numpy as np
import pandas as pd


def merge_dataframes(dfs: List[pd.DataFrame], tolerance_ms: int = 500) -> pd.DataFrame:
    """
    Merges multiple dataframes using 'asof' join for non-exact timestamps.
    Assumes the first DF is the 'anchor' (base) timeline.
    """
    if not dfs:
        return pd.DataFrame()

    # Filter out NaNs from the merge key for every dataframe in the list
    cleaned_dfs = [df.dropna(subset=['Time (s)']) for df in dfs]

    # Start with the first dataframe and sort timestamps
    base_df = cleaned_dfs[0].sort_values('Time (s)')

    # Iteratively merge the rest
    for next_df in cleaned_dfs[1:]:
        next_df = next_df.sort_values('Time (s)')

        base_df = pd.merge_asof(
            base_df,
            next_df,
            on='Time (s)',
            direction='nearest',  # Finds the closest timestamp in either direction
            tolerance=tolerance_ms / 1000.0,  # e.g., don't match if > 0.5s apart
            suffixes=('', '_extra')
        )

    # Make sure the 'Time (s)' column is the first column
    columns = base_df.columns.tolist()
    columns.remove('Time (s)')
    base_df = base_df[['Time (s)'] + columns]

    return base_df


def df_from_movesense_json(df: pd.DataFrame):
    """
    Extract heart rate data from Movesense JSON format and convert to DataFrame.
    """
    # Extract heart rate data and transform to DataFrame
    heart_rate_data = df['data']
    flattened_data = []
    current_time_ms = 0

    for entry in heart_rate_data:
        # Use the average HR provided by the sensor
        hr_val = entry['heartRate']['average']

        # Calculate how much time this packet actually covers by summing all RR intervals
        packet_duration_ms = sum(entry['heartRate']['rrData'])

        # Update the timeline
        current_time_ms += packet_duration_ms

        flattened_data.append({
            'Heart Rate (BPM) Movesense': hr_val,
            'Time (s)': current_time_ms / 1000.0
        })
    df = pd.DataFrame(flattened_data)
    return df


def df_from_garmin_csv(df: pd.DataFrame):
    """
    Extract heart rate data from Garmin CSV format and convert to DataFrame.
    """
    df = df[df['Message'] == 'record'][['Value 1', 'Value 4']].iloc[1:].reset_index(drop=True)
    df.columns = ['Time (s)', 'Heart Rate (BPM) Garmin']
    starttime = int(df.loc[0, 'Time (s)'])
    df['Time (s)'] = df['Time (s)'].astype(float) - starttime
    return df


def df_from_garmin_fit(file_path):
    """
    Extract heart rate data from Garmin FIT format and convert to DataFrame.
    """
    # Read fit file into dataframe
    fitfile = FitFile(file_path)
    data = []
    for record in fitfile.get_messages('record'):
        data.append(record.get_values())
    
    if not data:
        return pd.DataFrame(columns=['Heart Rate (BPM) Garmin', 'Time (s)'])
        
    df = pd.DataFrame(data)

    # Check if required columns exist
    if 'timestamp' not in df.columns or 'heart_rate' not in df.columns:
        return pd.DataFrame(columns=['Heart Rate (BPM) Garmin', 'Time (s)'])

    # Extract relevant timestamp and heart rate columns
    df = df[['timestamp', 'heart_rate']].rename(columns={'timestamp': 'Timestamp',
                                                         'heart_rate': 'Heart Rate (BPM) Garmin'})
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Create time column from timestamp
    df['Time (s)'] = df['Timestamp']
    df['Time (s)'] = df['Time (s)'] - df.loc[0, 'Timestamp']
    df['Time (s)'] = (df['Time (s)'].dt.total_seconds()).astype('float64')

    # Drop the timestamp column
    df.drop(columns=['Timestamp'], inplace=True)

    return df


def clip_df_on_time(df: pd.DataFrame, start_seconds: Optional[int], end_seconds: Optional[int]):
    if start_seconds:
        df = df.loc[(df['Time (s)'] >= start_seconds)]
    if end_seconds:
        df = df.loc[(df['Time (s)'] <= end_seconds)]
    return df