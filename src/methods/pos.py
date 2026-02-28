import math

import numpy as np
from scipy import signal
from src.methods import utils


def pos(frames: np.ndarray, fs):
    """
    Extracts the BVP signal using the POS method.

    Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017).
    Algorithmic principles of remote PPG.
    IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491.

    :param frames: A list of frames containing the average RGB pixel value of the relevant
                   landmark locations per frame. Should be of shape (num_frames, 3).
    :param fs: The sampling frequency of the video.
    :return: A numpy array of shape (n_frames) containing the BVP signal.
    """
    N = frames.shape[0]
    l = int(1.6 * fs)
    H = np.zeros(N)

    # Projection matrix (Plane Orthogonal to Skin)
    # This remains constant
    P = np.array([[0, 1, -1],
                  [-2, 1, 1]])

    for i in range(N - l):
        # 1. Select window
        window = frames[i:i + l, :]

        # 2. Step 1: Temporal normalization
        # C_n = C(t) / E[C(t)]
        C_n = window / np.mean(window, axis=0)
        C_n = C_n.T  # Shape (3, l)

        # 3. Step 2: Projection
        # S = P * C_n
        S = P @ C_n  # Shape (2, l)

        # 4. Step 3: Tuning (Alpha calculation)
        alpha = np.std(S[0, :]) / np.std(S[1, :])

        # 5. Step 4: Overlap-Add
        h = S[0, :] + alpha * S[1, :]

        # Normalize the windowed segment (zero mean)
        H[i:i + l] += (h - np.mean(h))

    # Post-processing: Bandpass filter (typical heart rate range 45-180 BPM)
    # Note: Using 0.7 Hz to 3.0 Hz
    nyquist = fs / 2
    low = 0.7 / nyquist
    high = 3.0 / nyquist
    b, a = signal.butter(2, [low, high], btype='bandpass')

    # Filter the result
    bvp = signal.filtfilt(b, a, H)
    return bvp