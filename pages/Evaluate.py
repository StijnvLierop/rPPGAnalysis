import pandas as pd
import streamlit as st

from src.parsing import df_from_movesense_json, df_from_garmin_csv, merge_dataframes, calculate_mae_robust

# Page title
st.title("Evaluate rPPG Signal")

st.sidebar.title("Upload Data")
with st.sidebar:
    # Upload reference data files
    garmin_file = st.file_uploader("Upload Garmin Reference Data", type=["csv"])
    movesense_file = st.file_uploader("Upload Movesense Reference Data", type=["json"])

    # Upload predicted data file
    predicted_file = st.file_uploader("Upload Predicted Data", type=["csv"])

# If Garmin file
if garmin_file:
    garmin_df = df_from_garmin_csv(pd.read_csv(garmin_file))
else:
    garmin_df = None

# If Movesense file
if movesense_file:
    movesense_df = df_from_movesense_json(pd.read_json(movesense_file))
else:
    movesense_df = None

# If predicted file
if predicted_file:
    predicted_df = pd.read_csv(predicted_file)
    predicted_df.rename(columns={'Heart Rate (BPM)': 'Heart Rate (BPM) Predicted'}, inplace=True)
else:
    predicted_df = None

# Merge files
if garmin_df is not None or predicted_df is not None or movesense_df is not None:
    # Merge dataframes
    df = merge_dataframes([df for df in [garmin_df, movesense_df, predicted_df] if df is not None])
    if 'Heart Rate (BPM) Garmin' in df.columns and 'Heart Rate (BPM) Movesense' in df.columns:
        df['Heart Rate (BPM) Avg'] = df[['Heart Rate (BPM) Garmin', 'Heart Rate (BPM) Movesense']].mean(axis=1)

    # Plot
    st.line_chart(df, x='Time (s)')

    # Calculate MAE
    if 'Heart Rate (BPM) Avg' in df.columns and 'Heart Rate (BPM) Predicted' in df.columns:
        st.write("MAE Avg-Predicted:", calculate_mae_robust(df['Heart Rate (BPM) Avg'], df['Heart Rate (BPM) Predicted']))
    if 'Heart Rate (BPM) Garmin' in df.columns and 'Heart Rate (BPM) Predicted' in df.columns:
        st.write("MAE Garmin-Predicted:", calculate_mae_robust(df['Heart Rate (BPM) Garmin'], df['Heart Rate (BPM) Predicted']))
    if 'Heart Rate (BPM) Movesense' in df.columns and 'Heart Rate (BPM) Predicted' in df.columns:
        st.write("MAE Movesense-Predicted:", calculate_mae_robust(df['Heart Rate (BPM) Movesense'], df['Heart Rate (BPM) Predicted']))

    # Write dataframe
    st.write(df)
else:
    st.write("Not enough data available. Please upload both reference and predicted data.")