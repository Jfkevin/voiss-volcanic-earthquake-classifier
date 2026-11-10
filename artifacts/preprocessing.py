import numpy as np
from scipy.signal import stft
from scipy.signal.windows import tukey

SAMPLING_RATE = 100
EXPECTED_SAMPLES = 6000
EXPECTED_SPECTROGRAM_SHAPE = (96, 60)


def _validate_waveform(signal):
    """Validate a single waveform while preserving numeric precision."""
    signal = np.asarray(signal).squeeze()
    if signal.ndim != 1:
        raise ValueError("Input harus berupa satu waveform 1D.")
    if signal.size != EXPECTED_SAMPLES:
        raise ValueError(
            f"Waveform harus memiliki {EXPECTED_SAMPLES} titik; diterima {signal.size}."
        )
    if not np.issubdtype(signal.dtype, np.number):
        raise TypeError("Waveform harus berisi nilai numerik.")
    if not np.all(np.isfinite(signal)):
        raise ValueError("Waveform mengandung NaN atau nilai tak hingga.")
    return signal


def preprocess_waveform(signal):
    """Prepare SeismicNet input with batch shape (1, 6000, 1)."""
    signal = _validate_waveform(signal)
    return signal.astype(np.float32).reshape(1, EXPECTED_SAMPLES, 1)


def preprocess_spectrogram(signal):
    """Reproduce notebook STFT preprocessing."""
    signal = _validate_waveform(signal)
    cleaned_signal = signal - np.mean(signal)
    tapered_signal = cleaned_signal * tukey(len(cleaned_signal), alpha=0.1)

    frequencies, times, zxx = stft(
        tapered_signal,
        fs=SAMPLING_RATE,
        window="hann",
        nperseg=1000,
        noverlap=900,
        boundary="zeros",
        padded=True,
    )

    magnitude = np.abs(zxx)
    spectrogram_db = 20.0 * np.log10(magnitude + 1e-10)
    frequency_mask = (frequencies >= 0.5) & (frequencies <= 10.0)
    cropped = spectrogram_db[frequency_mask, :60]

    if cropped.shape != EXPECTED_SPECTROGRAM_SHAPE:
        raise ValueError(
            f"Shape spektrogram tidak sesuai: {cropped.shape}; "
            f"seharusnya {EXPECTED_SPECTROGRAM_SHAPE}."
        )

    spec_min = np.min(cropped)
    spec_max = np.max(cropped)
    normalized = (cropped - spec_min) / (spec_max - spec_min + 1e-10)
    return normalized.astype(np.float32)[None, :, :, None]


def prepare_multimodal_inputs(signal):
    """Return [waveform_batch, spectrogram_batch] from the same event."""
    return [preprocess_waveform(signal), preprocess_spectrogram(signal)]
