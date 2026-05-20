from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt


@dataclass
class GammatoneConfig:
    work_fs: int = 16000
    out_fs: int = 64
    n_bands: int = 8
    fmin: float = 150.0
    fmax: float = 8000.0
    compression: float = 0.3
    lowpass_hz: float = 30.0


def hz_to_erb_rate(f_hz: np.ndarray | float) -> np.ndarray:
    return 21.4 * np.log10(4.37e-3 * np.asarray(f_hz) + 1.0)


def erb_rate_to_hz(erb: np.ndarray | float) -> np.ndarray:
    return (10 ** (np.asarray(erb) / 21.4) - 1.0) / 4.37e-3


def erb_center_frequencies(n_bands: int, fmin: float, fmax: float) -> np.ndarray:
    erb_min = hz_to_erb_rate(fmin)
    erb_max = hz_to_erb_rate(fmax)
    return erb_rate_to_hz(np.linspace(erb_min, erb_max, n_bands))


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=1)


def _resample(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    if fs_in == fs_out:
        return x.astype(np.float64)

    gcd = np.gcd(fs_in, fs_out)
    up = fs_out // gcd
    down = fs_in // gcd
    return resample_poly(x, up, down).astype(np.float64)


def _bandpass(x: np.ndarray, fs: int, low: float, high: float) -> np.ndarray:
    nyq = fs / 2
    high = min(high, nyq * 0.95)

    sos = butter(
        N=4,
        Wn=[low / nyq, high / nyq],
        btype="bandpass",
        output="sos",
    )
    return sosfiltfilt(sos, x)


def _lowpass(x: np.ndarray, fs: int, cutoff: float) -> np.ndarray:
    nyq = fs / 2
    cutoff = min(cutoff, nyq * 0.95)

    sos = butter(
        N=4,
        Wn=cutoff / nyq,
        btype="lowpass",
        output="sos",
    )
    return sosfiltfilt(sos, x, axis=0)


def compute_gammatone_spectrogram(
    wav_path: str | Path,
    config: GammatoneConfig = GammatoneConfig(),
) -> dict[str, np.ndarray | int | float]:
    """
    Compute an auditory-inspired gammatone-like spectrogram.

    Output shape:
        spectrogram: (n_times, n_bands)

    This is meant as an mTRF predictor:
        EEG ~ gammatone_spectrogram
    """

    wav_path = Path(wav_path)

    audio, fs = sf.read(wav_path)
    audio = _to_mono(audio).astype(np.float64)

    # Normalize safely
    audio = audio - np.mean(audio)
    max_abs = np.max(np.abs(audio))
    if max_abs > 0:
        audio = audio / max_abs

    # Work at fixed audio rate
    audio = _resample(audio, fs, config.work_fs)
    fs_work = config.work_fs

    centers = erb_center_frequencies(
        config.n_bands,
        config.fmin,
        min(config.fmax, fs_work / 2 * 0.95),
    )

    spectrogram = []

    for cf in centers:
        # Approximate auditory band around each center frequency
        low = max(config.fmin, cf / np.sqrt(2))
        high = min(fs_work / 2 * 0.95, cf * np.sqrt(2))

        band = _bandpass(audio, fs_work, low, high)

        # Hilbert envelope
        env = np.abs(hilbert(band))

        # Cochlear-like compression
        env = env ** config.compression

        spectrogram.append(env)

    spectrogram = np.stack(spectrogram, axis=1)

    # Smooth modulation frequencies before downsampling
    spectrogram = _lowpass(spectrogram, fs_work, config.lowpass_hz)

    # Resample to EEG/TRF sampling rate
    spectrogram = _resample(spectrogram, fs_work, config.out_fs)

    # Standardize each band for model stability
    spectrogram = spectrogram - spectrogram.mean(axis=0, keepdims=True)
    std = spectrogram.std(axis=0, keepdims=True)
    spectrogram = spectrogram / np.where(std == 0, 1.0, std)

    envelope = spectrogram.mean(axis=1)

    return {
        "spectrogram": spectrogram.astype(np.float32),
        "envelope": envelope.astype(np.float32),
        "center_frequencies": centers.astype(np.float32),
        "fs": config.out_fs,
        "n_bands": config.n_bands,
    }


def save_gammatone_features(
    wav_path: str | Path,
    out_path: str | Path,
    config: GammatoneConfig = GammatoneConfig(),
) -> None:
    features = compute_gammatone_spectrogram(wav_path, config)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(out_path, **features)