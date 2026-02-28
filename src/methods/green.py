import numpy as np


def green(frames: np.ndarray):
    """
    Extracts the BVP signal using the Green method.

    GREEN
    Verkruysse, W., Svaasand, L. O. & Nelson, J. S.
    Remote plethysmographic imaging using ambient light.
    Optical. Express 16, 21434–21445 (2008).

    :param frames: A list of frames containing the average RGB pixel value of the relevant
                   landmark locations per frame. Should be of shape (num_frames, 3).
    :return: A numpy array of shape (n_frames) containing the BVP signal.
    """
    bvp_signal = frames[:, 1]
    bvp_signal = bvp_signal.reshape(-1)
    return bvp_signal