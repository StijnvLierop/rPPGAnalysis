import pandas as pd


def df_from_movesense_json(df: pd.DataFrame):
    heart_rate_data = df['data']
    flattened_data = [{
        'BPM': entry['heartRate']['average'],
        'rrData': entry['heartRate']['rrData'][0]
    } for entry in heart_rate_data]

    df = pd.DataFrame(flattened_data)
    df['Timestamp'] = df['rrData'].cumsum() / 1000
    return df.drop(columns=['rrData'])


def df_from_garmin_csv(df: pd.DataFrame):
    df = df[df['Message'] == 'record'][['Value 1', 'Value 4']].iloc[1:].reset_index(drop=True)
    df.columns = ['Timestamp', 'BPM']
    starttime = int(df.loc[0, 'Timestamp'])
    df['Timestamp'] = df['Timestamp'].astype(float) - starttime
    return df
