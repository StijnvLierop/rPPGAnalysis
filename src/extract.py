from typing import Mapping, Optional, Tuple, List

import cv2
import numpy as np
from scipy.signal import butter, filtfilt, welch, medfilt

from src.landmarks import draw_landmark_video, extract_landmarks
from src.methods.green import green
from src.methods.pos import pos


def read_video_rgb(video_path: str):
    """Reads a video file and returns a list of frames in BGR format."""
    # Read video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    # Get FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0  # fallback

    # Read and store frames
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(np.ascontiguousarray(frame))
    cap.release()

    if not frames:
        raise ValueError("Video does not contain any frames.")

    return frames, float(fps)


def calculate_bpm_from_bvp(bvp_signal: np.ndarray,
                           fps: float,
                           winsize: int,
                           step: int,
                           smooth_signal: bool = True) -> List[float]:
    """
    Calculates the Heart Rate (BPM) signal from a BVP signal.

    :param bvp_signal: 1D numpy array of signal values.
    :param fps: Sampling rate of the video (frames per second).
    :param winsize: Length of the sliding window to calculate BPM from in seconds.
    :param step: How much to slide the window.
    :param smooth_signal: Whether to smooth the BPM signal using a median filter.
    :return: Estimated BPM (float).
    """
    # Bandpass Filter (Removes high-frequency noise and low-frequency drift)
    # Human heart rate is typically between 40 and 200 BPM
    low_cut = 0.7  # 42 BPM
    high_cut = 3.5  # 210 BPM
    nyquist = 0.5 * fps
    low = low_cut / nyquist
    high = high_cut / nyquist
    b, a = butter(2, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, bvp_signal)

    # Windowing
    window_samples = int(winsize * fps)
    step_samples = int(step * fps)

    # Calculate BPM over windows
    bpm_signal = []
    for i in range(0, len(filtered_signal) - window_samples + 1, step_samples):
        window = filtered_signal[i: i + window_samples]

        # Normalize window
        window = (window - np.mean(window)) / np.std(window)

        # Apply Welch's method to find the Power Spectrum
        freqs, psd = welch(window, fs=fps, nperseg=len(window), nfft=2048)

        # Constrain to human heart rate range (0.7 - 3.5 Hz)
        valid_idx = np.where((freqs >= 0.7) & (freqs <= 3.5))[0]
        psd = psd[valid_idx]
        freqs = freqs[valid_idx]

        # Find the frequency with the highest power
        if len(psd) > 0:
            max_freq = freqs[np.argmax(psd)]
            bpm = max_freq * 60.0
            bpm_signal.append(bpm)
        else:
            bpm_signal.append(0.0)

    # Smooth BPM signal
    if smooth_signal:
        bpm_signal = medfilt(np.array(bpm_signal), kernel_size=5)

    return bpm_signal


def extract_signal_from_video(video_path: str,
                              analysis_method: str,
                              selected_landmark_regions: List[str],
                              smooth_signal: bool = True) -> Tuple[Optional[Mapping], Optional[str]]:
    """
    Run rPPG analysis on a video and return the extracted signal.

    :param video_path: Path to the input video file.
    :param analysis_method: rPPG method to use. Should be either 'green' or 'pos'.
    :param selected_landmark_regions: List of landmark regions to extract. Should be a list of
                                      strings corresponding to Mediapipe landmark regions.
    :param smooth_signal: Whether to smooth the BPM signal using a median filter.
    :return: - A dictionary with the following keys:
                 - 'Time BVP (s)': A list of timestep values matching the BVP signal (in seconds).
                 - 'Time BPM (s)': A list of timestep values matching the BPM signal (in seconds).
                 - 'BVP': A list of BVP values.
                 - 'Heart Rate (BPM)': A list of estimated BPM values.
             - A path to the rendered landmark video.
    """
    # Read video frames
    frames_rgb, fps = read_video_rgb(video_path)

    # Extract roi landmarks
    rgb_signal, predicted_landmarks = extract_landmarks(frames_rgb, fps, selected_landmark_regions)

    # If the length of landmarks is 0, return
    if len(rgb_signal) == 0:
        print("No landmarks found in video. Returning empty result.")
        return None, None

    # Select method
    if analysis_method == 'green':
        bvp_signal = green(rgb_signal)
    elif analysis_method == 'pos':
        bvp_signal = pos(rgb_signal, fps)
    else:
        raise ValueError(f"Unknown analysis method: {analysis_method}")

    # Generate landmark vide
    landmarks_video = draw_landmark_video(frames_rgb, predicted_landmarks, selected_landmark_regions)

    # Define window size and stepsize for calculating BPM
    winsize = 6 # Length of the sliding window to calculate BPM from in seconds
    step = 1 # How much to slide the window

    # Calculate BPM
    bpm_signal = calculate_bpm_from_bvp(bvp_signal,
                                        fps,
                                        winsize=winsize,
                                        step=step,
                                        smooth_signal=smooth_signal)

    # Calculate timestamps for the raw BVP
    timestamps_bvp = np.arange(0, len(bvp_signal)) / fps

    # Calculate timestamps for the BPM (centered on windows)
    num_windows = len(bpm_signal)
    timestamps_bpm = np.arange(0, num_windows) * step + (winsize / 2.0)

    # Format result
    result = {
        "Timesteps BVP (s)": timestamps_bvp,
        "Timesteps BPM (s)": timestamps_bpm,
        "BVP": bvp_signal,
        "Heart Rate (BPM)": bpm_signal,
    }
    return result, landmarks_video
