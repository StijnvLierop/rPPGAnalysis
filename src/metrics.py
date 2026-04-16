import numpy as np
from scipy import signal


def calculate_windowed_snr(pred_ppg, label_hr, fs, window_sec=10, step_sec=1):
    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)
    snrs = []

    for start in range(0, len(pred_ppg) - window_samples, step_samples):
        end = start + window_samples

        # Slice the window
        window_pred = pred_ppg[start:end]
        window_label = label_hr[start:end]

        # Use your existing calculate_SNR function on the slice
        score = calculate_SNR(window_pred, window_label, fs=fs)
        snrs.append(score)

    return np.mean(snrs) if snrs else 0.0


def calculate_SNR(pred_ppg_signal: np.ndarray,
                  label_hr_signal: np.ndarray,
                  fs: int = 100,
                  low_pass: float = 0.75,
                  high_pass: float = 4.0):
    pred_ppg_signal = np.array(pred_ppg_signal).flatten()
    label_hr_signal = np.array(label_hr_signal).flatten()

    # Reference Frequencies
    avg_hr_bpm = np.nanmean(label_hr_signal)
    f0 = avg_hr_bpm / 60.0
    f1 = 2 * f0
    deviation = 6.0 / 60.0

    # PSD Calculation
    # Increase N to improve frequency resolution (helps if signal is short)
    sig = signal.detrend(pred_ppg_signal, type='linear')
    nperseg = min(len(sig), 512)
    # Ensure nperseg is not 0
    if nperseg == 0:
        return -20.0
    f, pxx = signal.welch(sig, fs=fs, nperseg=nperseg, nfft=2048)

    # Masks
    h1_mask = (f >= (f0 - deviation)) & (f <= (f0 + deviation))
    h2_mask = (f >= (f1 - deviation)) & (f <= (f1 + deviation))

    # Range of interest (Standard rPPG window)
    total_range_mask = (f >= low_pass) & (f <= high_pass)

    # Noise is the range of interest MINUS the harmonics
    remainder_mask = total_range_mask & ~(h1_mask | h2_mask)

    # Sum Power
    sig_pwr = np.sum(pxx[h1_mask]) + np.sum(pxx[h2_mask])
    noise_pwr = np.sum(pxx[remainder_mask])

    # If noise_pwr is 0, it means the harmonic masks 'swallowed' all bins in the 0.75-2.5Hz range.
    # We fallback to using the whole spectrum as noise to avoid division by zero.
    if noise_pwr == 0:
        # Use a tiny epsilon or fallback to total power outside harmonics
        noise_pwr = np.sum(pxx[total_range_mask]) - sig_pwr
        # If still 0, use the whole pxx sum as a last resort
        if noise_pwr <= 0:
            noise_pwr = 1e-12

    # Calculation
    if sig_pwr > 0:
        snr = 10 * np.log10(sig_pwr / noise_pwr)
    else:
        snr = -20.0  # Signal not found in the expected HR bins

    return snr


def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def power2db(mag):
    """Convert power to db."""
    return 10 * np.log10(mag)


def calculate_mae_robust(y_true, y_pred):
    # Convert to numpy and find where both have valid data
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)

    if not np.any(mask):
        return None  # No overlapping valid data

    return np.mean(np.abs(y_true[mask] - y_pred[mask]))