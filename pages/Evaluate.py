import pandas as pd
import streamlit as st

from src.parsing import df_from_movesense_json, df_from_garmin_csv

# Page title
st.title("Extract rPPG Signal")
st.set_page_config(
    page_title="Extract",
)

st.sidebar.title("Tool")
with st.sidebar:
    # Upload reference data files
    garmin_file = st.file_uploader("Upload Garmin Reference Data", type=["csv"])
    movesense_file = st.file_uploader("Upload Movesense Reference Data", type=["json"])

    # Upload predicted data file
    predicted_file = st.file_uploader("Upload Predicted Data", type=["csv"])

# If Garmin file
if garmin_file:
    garmin_df = df_from_garmin_csv(pd.read_csv(garmin_file))

# If Movesense file
if movesense_file:
    movesense_df = df_from_movesense_json(pd.read_json(movesense_file))

# Merge files
reference_df = merge_reference_data(garmin_df, movesense_df)
df = merge_predicted_data(reference_df, predicted_df)

# Plot
st.line_chart(df, x='Time (s)', y='BPM')