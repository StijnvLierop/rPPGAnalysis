import tempfile

import streamlit as st
import pandas as pd

from src.extract import extract_signal_from_video
from src.landmarks import LANDMARK_REGIONS

# Page title
st.set_page_config(
    page_title="rPPG Analysis",
)
st.title("Extract rPPG Signal")

results, landmarks_video, smooth_signal = None, None, None

with st.sidebar:
    # Create file uploader
    file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

    # If file
    if file:

        # Pick analysis algorithm
        analysis_algorithm = st.selectbox('Select rPPG Extraction Method',
                                          options=['GREEN', 'POS', 'RhythmMamba'])

        with st.expander("Advanced Options"):
            # Whether to smooth the signal
            st.write("Signal Smoothing")
            smooth_signal = st.checkbox("Smooth Signal", value=True)

            # Landmark regions to take into account
            st.write("Landmark Regions")
            for region in LANDMARK_REGIONS:
                if region in ['high_prio_forehead', 'high_prio_left_cheek', 'high_prio_right_cheek']:
                    st.checkbox(region, key=region, value=True)
                else:
                    st.checkbox(region, key=region, value=False)

        # Show an extract signal button
        extract_signal_button = st.button("Extract Signal")

# If the button is pressed
if file and extract_signal_button:

    # Create a temporary file to save the uploaded bytes
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(file.read())
        temp_path = tfile.name

    # Extract signal
    with st.spinner('Extracting signal...'):
        # Get selected landmark regions
        selected_landmark_regions = [region for region in LANDMARK_REGIONS if st.session_state.get(region)]

        # Run signal extraction
        results, landmarks_video = extract_signal_from_video(temp_path,
                                                             analysis_algorithm,
                                                             selected_landmark_regions,
                                                             smooth_signal)

        if results is None:
            st.error("No face landmarks found. Could not extract signal.")

# If results are available
if results is not None:
    results_tab, video_tab = st.tabs(["Results", "Landmarks Video"])

    # If landmarks video, show
    with video_tab:
        if landmarks_video is not None:
            # Display video
            st.video(landmarks_video, format="video/mp4")
        else:
            st.write("No landmarks video found.")

    # Display results tab
    with results_tab:
        # Only get BPM information
        bpm_results = {'Time (s)': results['Timesteps BPM (s)'],
                       'Heart Rate (BPM)': results['Heart Rate (BPM)']}

        # Transform result to a dataframe
        result_df = pd.DataFrame(bpm_results)

        # Plot signal
        st.write("Estimated Heart Rate")
        st.line_chart(result_df, x="Time (s)", y="Heart Rate (BPM)")

        # Create a download button for a csv file
        st.download_button("Download CSV",
                           data=result_df.to_csv(index=False).encode('utf-8'),
                           file_name=f"{analysis_algorithm}_{file.name}.csv",
                           mime="text/csv",
                           key="download_result"
                           )
else:
    st.write("No results yet. Upload a video and run an extraction method to visualize results.")