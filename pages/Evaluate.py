import pandas as pd
import streamlit as st

from src.parsing import df_from_movesense_json, merge_dataframes, \
    df_from_garmin_fit, remove_start_buffer
from src.metrics import calculate_mae_robust, calculate_SNR

# Page title
st.title("Evaluate rPPG Signal")

st.sidebar.title("Upload Data")
with st.sidebar:
    # Upload reference data files
    garmin_file = st.file_uploader("Upload Garmin Reference Data", type=["fit"])
    movesense_file = st.file_uploader("Upload Movesense Reference Data", type=["json"])

    # Upload predicted data file
    predicted_bpm_file = st.file_uploader("Upload Predicted BPM Data", type=["csv"])
    predicted_bvp_file = st.file_uploader("Upload Predicted BVP Data", type=["csv"])

    # Start buffer size
    buffer_seconds = st.sidebar.slider("Buffer (s)", 0, 10, 5)

# --- Data Loading ---
garmin_df = df_from_garmin_fit(garmin_file) if garmin_file else None
movesense_df = df_from_movesense_json(pd.read_json(movesense_file)) if movesense_file else None

if predicted_bpm_file:
    predicted_bpm_df = pd.read_csv(predicted_bpm_file)
    if 'Heart Rate (BPM)' not in predicted_bpm_df.columns:
        st.error("Uploaded file does not contain 'Heart Rate (BPM)' column.")
    predicted_bpm_df.rename(columns={'Heart Rate (BPM)': 'Heart Rate (BPM) Predicted'}, inplace=True)
else:
    predicted_bpm_df = None

if predicted_bvp_file:
    predicted_bvp_df = pd.read_csv(predicted_bvp_file)
    if 'BVP' not in predicted_bvp_df.columns:
        st.error("Uploaded file does not contain 'BVP' column.")
else:
    predicted_bvp_df = None

# --- Merging & Processing ---
dataframes_to_merge = [df for df in [movesense_df, garmin_df, predicted_bpm_df, predicted_bvp_df] if df is not None]

if len(dataframes_to_merge) > 0:
    # Merge dataframes
    df = merge_dataframes(dataframes_to_merge)

    if 'Heart Rate (BPM) Garmin' in df.columns and 'Heart Rate (BPM) Movesense' in df.columns:
        df['Heart Rate (BPM) Avg'] = df[['Heart Rate (BPM) Garmin', 'Heart Rate (BPM) Movesense']].mean(axis=1)

    # Remove start buffer
    df = remove_start_buffer(df, seconds=buffer_seconds)

    # Plot
    st.subheader("Heart Rate Comparison")
    # Filter columns to only plot BPM related data for clarity
    plot_cols = [c for c in df.columns if 'Heart Rate' in c or c == 'Time (s)']
    st.line_chart(df[plot_cols], x='Time (s)')

    # --- Metrics Section ---
    st.subheader("Performance Metrics")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Mean Absolute Error (MAE)**")
        if 'Heart Rate (BPM) Movesense' in df.columns and 'Heart Rate (BPM) Garmin' in df.columns:
            st.metric(
                f"Movesense vs Garmin:", round(calculate_mae_robust(df['Heart Rate (BPM) Movesense'], df['Heart Rate (BPM) Garmin']), 2))

        if 'Heart Rate (BPM) Predicted' in df.columns:
            if 'Heart Rate (BPM) Avg' in df.columns:
                st.metric(
                    f"Avg vs Predicted:", round(calculate_mae_robust(df['Heart Rate (BPM) Avg'], df['Heart Rate (BPM) Predicted']), 2))
            if 'Heart Rate (BPM) Garmin' in df.columns:
                st.metric(
                    f"Garmin vs Predicted:", round(calculate_mae_robust(df['Heart Rate (BPM) Garmin'], df['Heart Rate (BPM) Predicted']), 2))
            if 'Heart Rate (BPM) Movesense' in df.columns:
                st.metric(
                    f"Movesense vs Predicted:", round(calculate_mae_robust(df['Heart Rate (BPM) Movesense'], df['Heart Rate (BPM) Predicted']), 2))

    with col2:
        st.markdown("**Signal to Noise Ratio (SNR)**")
        # Calculate SNR only if BVP and a ground truth BPM exist
        if 'BVP' in df.columns:
            # Use Movesense as ground truth for SNR if available, otherwise Garmin
            reference_col = None
            if 'Heart Rate (BPM) Movesense' in df.columns:
                reference_col = 'Heart Rate (BPM) Movesense'
            elif 'Heart Rate (BPM) Garmin' in df.columns:
                reference_col = 'Heart Rate (BPM) Garmin'

            if reference_col:
                snr_data = df[['BVP', reference_col]].dropna()
                snr_val = calculate_SNR(snr_data['BVP'], snr_data[reference_col], fs=100)
                st.metric("SNR", f"{snr_val:.2f} dB")
                st.write(f"Ground Truth: {reference_col}")
            else:
                st.warning("Upload Reference Data to calculate SNR.")
        else:
            st.info("Upload Predicted BVP Data to calculate SNR.")

    # Write raw dataframe
    with st.expander("View Raw Data"):
        st.write(df)
else:
    st.info("Please upload data files to begin evaluation.")