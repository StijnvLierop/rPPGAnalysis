from typing import List, Optional

from fitparse import FitFile
import pandas as pd


def merge_dataframes(dfs: List[pd.DataFrame], target_freq: str = '33ms') -> pd.DataFrame:
    """
    Merges dataframes based on timestamp and interpolates missing values.
    """
    processed_dfs = []

    for i, df in enumerate(dfs):
        # Prepare copy and convert float seconds to Timedelta
        temp_df = df.copy()
        temp_df['Time (s)'] = pd.to_timedelta(temp_df['Time (s)'], unit='s')

        # Set index (Required for resample/interpolate)
        temp_df = temp_df.set_index('Time (s)')

        # Average any duplicate timestamps (common in rPPG)
        temp_df = temp_df.groupby(level=0).mean()
        processed_dfs.append(temp_df)

    # Outer Join: Keep every single timestamp from every source
    combined = pd.concat(processed_dfs, axis=1).sort_index()

    # Linear Interpolation: Fill gaps between sensors
    combined = combined.interpolate(method='linear')

    # Resample to a uniform grid (e.g., 33ms)
    result = combined.resample(target_freq).mean()

    # Fill any remaining NaNs at the very start/end created by resampling
    result = result.interpolate(method='linear').dropna()

    # Convert index back to float seconds for readability
    result.index = result.index.total_seconds()
    result.index.name = 'Time (s)'

    return result.reset_index()


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