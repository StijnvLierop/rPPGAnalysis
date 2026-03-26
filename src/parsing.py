from typing import List

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

    # Start with the first dataframe and sort timestamps
    base_df = dfs[0].sort_values('Time (s)')

    # Iteratively merge the rest
    for next_df in dfs[1:]:
        next_df = next_df.sort_values('Time (s)')

        base_df = pd.merge_asof(
            base_df,
            next_df,
            on='Time (s)',
            direction='nearest',  # Finds the closest timestamp in either direction
            tolerance=tolerance_ms / 1000.0,  # e.g., don't match if > 0.5s apart
            suffixes=('', '_extra')
        )

    return base_df


def df_from_movesense_json(df: pd.DataFrame):
    """
    Extract heart rate data from Movesense JSON format and convert to DataFrame.
    """
    # Extract heart rate data and transform to DataFrame
    heart_rate_data = df['data']
    flattened_data = [{
        'Heart Rate (BPM) Movesense': entry['heartRate']['average'],
        'rrData': entry['heartRate']['rrData'][0],
    } for entry in heart_rate_data]
    df = pd.DataFrame(flattened_data)

    # Calculate time from rrData (time since last RR interval in ms)
    df['Time (s)'] = df['rrData'].cumsum() / 1000
    df = df.drop(columns=['rrData'])

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


def df_from_garmin_fit(file_path: str):
    """
    Extract heart rate data from Garmin FIT format and convert to DataFrame.
    """
    # Read fit file into dataframe
    fitfile = FitFile(file_path)
    data = []
    for record in fitfile.get_messages('record'):
        data.append(record.get_values())
    df = pd.DataFrame(data)

    # Extract relevant timestamp and heart rate columns
    df = df[['timestamp', 'heart_rate']].rename(columns={'timestamp': 'Timestamp',
                                                         'heart_rate': 'Heart Rate (BPM) Garmin'})
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Create time column from timestamp
    df['Time (s)'] = df['Timestamp']
    df['Time (s)'] = df['Time (s)'] - df.loc[0, 'Timestamp']
    df['Time (s)'] = (df['Time (s)'].astype('int64') // 10**9).astype('float64')

    # Drop the timestamp column
    df.drop(columns=['Timestamp'], inplace=True)

    return df


def calculate_mae_robust(y_true, y_pred):
    # Convert to numpy and find where both have valid data
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)

    if not np.any(mask):
        return None  # No overlapping valid data

    return np.mean(np.abs(y_true[mask] - y_pred[mask]))
