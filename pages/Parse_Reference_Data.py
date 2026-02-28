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
    # Upload reference data file
    file = st.file_uploader("Upload Reference Data", type=["csv", "json"])

# If file
if file:

    # If data is csv / Garmin
    if file.name.endswith(".csv"):
        df = df_from_garmin_csv(pd.read_csv(file))
    elif file.name.endswith(".json"):
        df = df_from_movesense_json(pd.read_json(file))
    else:
        df = None
        st.error("Unsupported file type. Only csv and json are supported.")

    if df is not None:
        st.line_chart(df, x='Timestamp', y='BPM')