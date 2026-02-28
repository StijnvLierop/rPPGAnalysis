from typing import List

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
    heart_rate_data = df['data']
    flattened_data = [{
        'Heart Rate (BPM) Movesense': entry['heartRate']['average'],
        'rrData': entry['heartRate']['rrData'][0],
    } for entry in heart_rate_data]
    df = pd.DataFrame(flattened_data)
    df['Time (s)'] = df['rrData'].cumsum() / 1000
    return df.drop(columns=['rrData'])


def df_from_garmin_csv(df: pd.DataFrame):
    df = df[df['Message'] == 'record'][['Value 1', 'Value 4']].iloc[1:].reset_index(drop=True)
    df.columns = ['Time (s)', 'Heart Rate (BPM) Garmin']
    starttime = int(df.loc[0, 'Time (s)'])
    df['Time (s)'] = df['Time (s)'].astype(float) - starttime
    return df


def calculate_mae_robust(y_true, y_pred):
    # Convert to numpy and find where both have valid data
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)

    if not np.any(mask):
        return None  # No overlapping valid data

    return np.mean(np.abs(y_true[mask] - y_pred[mask]))
