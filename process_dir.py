import argparse
import os
from tqdm import tqdm
import pandas as pd

from src.extract import extract_signal_from_video, read_video_rgb
from src.metrics import calculate_mae_robust, calculate_SNR
from src.parsing import df_from_movesense_json, df_from_garmin_fit, clip_df_on_time, merge_dataframes


def process_dir(input_dir: str, output_file: str, algorithm: str, smooth_signal: bool, start_buffer: int, measurement_window: int):
    # Store results
    result_df = pd.DataFrame(columns=['Participant', 'Condition', 'Camera', 'Compression', 'mae_movesense', 'snr_movesense', 'mae_garmin', 'snr_garmin', 'mae_movesense_garmin'])

    # Loop over participants
    for participant in tqdm(os.listdir(input_dir)):

        # Get participant directory
        participant_dir = os.path.join(input_dir, participant)

        # Read start times file
        start_time_df = pd.read_csv(os.path.join(participant_dir, 'start_times.csv'), sep=';')
        start_time_df['start_time'] = start_time_df['start_time'] / 100
        start_time_df['end_time'] = start_time_df['start_time'] + start_buffer + measurement_window

        # Read participant reference files
        participant_reference_data = {}
        mae_movesense_garmin = {}
        for condition in ['s1', 's2', 's3', 's4', 's5']:
            garmin_file = os.path.join(participant_dir, f'{participant}_{condition}.fit')
            movesense_file = os.path.join(participant_dir, f'{participant}_{condition}.json')
            df = merge_dataframes([df_from_garmin_fit(garmin_file), df_from_movesense_json(pd.read_json(movesense_file))])
            participant_reference_data[condition] = df
            mae_movesense_garmin[condition] = calculate_mae_robust(df['Heart Rate (BPM) Garmin'], df['Heart Rate (BPM) Movesense'])

        # Loop over video's
        for video in os.listdir(participant_dir):

            # Get file path
            file_path = os.path.join(participant_dir, video)

            # If file is a video
            if file_path.endswith('.mp4') or file_path.endswith('.MOV'):

                # Get condition information
                video_metadata = video.split('.')[0].split('_')
                if len(video_metadata) == 4:
                    v_participant, condition, camera, compression = video_metadata
                elif len(video_metadata) == 3:
                    v_participant, condition, camera = video_metadata
                    compression = 'None'
                else:
                    raise ValueError(f'Invalid video metadata: {video_metadata}')

                # Process video
                print(f'Processing video: {file_path}')

                # Get selected landmark regions
                selected_landmark_regions = ['high_prio_forehead', 'high_prio_left_cheek', 'high_prio_right_cheek']

                # Run signal extraction
                results, landmarks_video = extract_signal_from_video(file_path,
                                                                     algorithm,
                                                                     selected_landmark_regions,
                                                                     smooth_signal)

                # Get FPS from video
                _, fps = read_video_rgb(file_path)

                if results is None:
                    print(f'No landmarks found for video: {video}')
                    continue

                # Get BPM information
                bpm_results = {'Time (s)': results['Timesteps BPM (s)'],
                               'Heart Rate (BPM) Predicted': results['Heart Rate (BPM)']}

                # Transform result to a dataframe
                video_bpm_result_df = pd.DataFrame(bpm_results).dropna(subset=['Time (s)'])

                # Get BVP information
                bvp_results = {'Time (s)': results['Timesteps BVP (s)'],
                               'BVP': results['BVP']}

                # Transform result to a dataframe
                video_bvp_result_df = pd.DataFrame(bvp_results).dropna(subset=['Time (s)'])

                # Create row to add
                row = {'Participant': v_participant,
                 'Condition': condition,
                 'Camera': camera,
                 'Compression': compression,
                 'mae_movesense_garmin': mae_movesense_garmin[condition],
                 }

                # Get start and end times
                try:
                    start_time = int(start_time_df[start_time_df['filename'].str.contains(f"{v_participant}_{condition}_{camera}", na=False, regex=True)]['start_time'].iloc[0])
                    end_time = int(start_time_df[start_time_df['filename'].str.contains(f"{v_participant}_{condition}_{camera}", na=False, regex=True)]['end_time'].iloc[0])
                except:
                    start_time = 0
                    end_time = None
                    print(f'Start time not found for video: {video}')

                # Prepare Reference Data (Shift sensor 0 to match Video 'start_time' which represents the time at which the recording started in the video)
                ref_df = participant_reference_data[condition].copy()
                ref_df['Time (s)'] += start_time

                # Merge dataframes once per video
                df_merged = merge_dataframes([video_bpm_result_df, video_bvp_result_df, ref_df])

                # Remove start buffer
                df_merged = clip_df_on_time(df_merged, start_seconds=start_time + start_buffer, end_seconds=end_time)

                if len(df_merged) == 0:
                    print("No union between reference and predicted data found. Skipping video...")
                    continue

                for key in ref_df.columns:

                    # Skip this step for the time column
                    if key == 'Time (s)':
                        continue

                    # Calculate MAE
                    mae = calculate_mae_robust(df_merged[key], df_merged['Heart Rate (BPM) Predicted'])
                    row['mae_' + key.split(' ')[-1].lower()] = mae

                    # Calculate SNR
                    snr_input = df_merged[[key, 'BVP']].dropna()
                    snr = calculate_SNR(snr_input['BVP'], snr_input[key], fs=fps)
                    row['snr_' + key.split(' ')[-1].lower()] = snr

                # Store results
                result_df.loc[len(result_df)] = row

    # Write dataframe to csv
    result_df.to_csv(output_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', '-i', type=str, required=True, help='Directory to process.')
    parser.add_argument('--output_file', '-o', type=str, required=True, help='Output file.')
    parser.add_argument('--algorithm', '-a', type=str, default='POS', help='Analysis algorithm to use.')
    parser.add_argument('--smooth_signal', '-s', type=bool, default=True, help='Smooth signal.')
    parser.add_argument('--start_buffer', '-b', type=int, default=5, help='Start buffer (s).')
    parser.add_argument('--measurement_window', '-m', type=int, default=30, help='Measurement window (s).')

    process_dir(**vars(parser.parse_args()))
