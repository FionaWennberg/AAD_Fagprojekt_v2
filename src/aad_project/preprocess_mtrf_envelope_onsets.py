from __future__ import annotations

"""
Preprocess Fuglsang selective-attention EEG data for an mTRF / forward Eelbrain model.

This pipeline extends the existing backward-envelope preprocessing by extracting
both acoustic envelope and acoustic onset predictors for each speech stream.

Output keys
-----------
Single-talker:
    pred_st         object array; each trial shape (time, 2), columns [envelope, onset]
    stim_st_env     object array; each trial shape (time,)
    stim_st_onset   object array; each trial shape (time,)
    resp_st         object array; each trial shape (time, channels)

Two-talker:
    pred_att        object array; each trial shape (time, 2), attended [envelope, onset]
    pred_ign        object array; each trial shape (time, 2), ignored [envelope, onset]
    stim_att_env    object array; each trial shape (time,)
    stim_att_onset  object array; each trial shape (time,)
    stim_ign_env    object array; each trial shape (time,)
    stim_ign_onset  object array; each trial shape (time,)
    resp_tt         object array; each trial shape (time, channels)

Backward-compatible aliases:
    stim_att        same as stim_att_env
    stim_ign        same as stim_ign_env
    stim_st         same as stim_st_env

Recommended forward mTRF comparison
-----------------------------------
Baseline:
    EEG ~ attended envelope

Experiment:
    EEG ~ attended envelope + attended onset

Example
-------
python -m src.aad_project.preprocess_mtrf_env_onset \
  --bidsdir data/raw/ds-eeg-snhl \
  --subject 4 \
  --out data/processed/sub-004_mtrf_env_onset.npz
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt

from src.aad_project.data import ensure_dirs, subject_id


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def _as_float64(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _zscore_safe(x: np.ndarray, axis: int = 0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = np.nanmean(x, axis=axis, keepdims=True)
    sd = np.nanstd(x, axis=axis, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (x - mu) / sd


def _sos_filter(
    x: np.ndarray,
    fs: float,
    *,
    low: float | None = None,
    high: float | None = None,
    order: int = 4,
    axis: int = 0,
) -> np.ndarray:
    nyq = fs / 2.0

    if low is not None and high is not None:
        wn = [low / nyq, high / nyq]
        btype = "bandpass"
    elif low is not None:
        wn = low / nyq
        btype = "highpass"
    elif high is not None:
        wn = high / nyq
        btype = "lowpass"
    else:
        raise ValueError("Need low and/or high cutoff")

    sos = butter(order, wn, btype=btype, output="sos")
    return sosfiltfilt(sos, x, axis=axis)


def _resample(x: np.ndarray, fs_in: float, fs_out: float, axis: int = 0) -> np.ndarray:
    if fs_in == fs_out:
        return np.asarray(x)

    from fractions import Fraction

    ratio = Fraction(fs_out / fs_in).limit_denominator(1000)
    return resample_poly(x, ratio.numerator, ratio.denominator, axis=axis)


def crop_toi_1d(x: np.ndarray, fs: float, tmin: float, tmax: float) -> np.ndarray:
    start = max(0, int(round(tmin * fs)))
    stop = min(len(x), int(round(tmax * fs)))
    return np.asarray(x[start:stop])


def crop_toi_2d_time(x: np.ndarray, fs: float, tmin: float, tmax: float) -> np.ndarray:
    start = max(0, int(round(tmin * fs)))
    stop = min(x.shape[0], int(round(tmax * fs)))
    return np.asarray(x[start:stop, :])


def _to_object_trials(trials: list[np.ndarray], dtype=np.float64) -> np.ndarray:
    out = np.empty((len(trials),), dtype=object)
    for i, trial in enumerate(trials):
        out[i] = np.asarray(trial, dtype=dtype)
    return out


# -----------------------------------------------------------------------------
# Auditory filterbank helpers
# -----------------------------------------------------------------------------


def hz_to_erb_rate(f_hz: np.ndarray | float) -> np.ndarray:
    f = np.asarray(f_hz, dtype=float)
    return 21.4 * np.log10(4.37e-3 * f + 1.0)


def erb_rate_to_hz(erb: np.ndarray | float) -> np.ndarray:
    e = np.asarray(erb, dtype=float)
    return (10 ** (e / 21.4) - 1.0) / 4.37e-3


def erb_bandwidth(f_hz: np.ndarray | float) -> np.ndarray:
    f = np.asarray(f_hz, dtype=float)
    return 24.7 * (4.37e-3 * f + 1.0)


def erb_center_frequencies(fmin: float, fmax: float, n_bands: int) -> np.ndarray:
    erb_min = hz_to_erb_rate(fmin)
    erb_max = hz_to_erb_rate(fmax)
    erb_points = np.linspace(erb_min, erb_max, n_bands)
    return erb_rate_to_hz(erb_points)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class AcousticFeatureConfig:
    fs_out: float = 64.0
    n_bands: int = 16
    fmin_hz: float = 150.0
    fmax_hz: float = 8000.0
    filter_order: int = 2
    audio_work_fs: float = 16000.0
    band_env_lp: float = 30.0
    compression: float = 0.3
    hp_out: float = 1.0
    lp_out: float = 9.0
    final_order: int = 4
    onset_smooth_hz: float = 10.0
    normalize_features: bool = True


@dataclass
class EEGConfig:
    mastoids: tuple[str, str] = ("TP7", "TP8")
    fs_lowpass: float = 30.0
    fs_out: float = 64.0
    hp_out: float = 1.0
    lp_out: float = 9.0
    crop_tmin: float = 6.0
    crop_tmax: float = 43.0
    scalp_only: bool = True
    eog_regression: bool = True
    filt_order: int = 4


# -----------------------------------------------------------------------------
# Stimulus preprocessing
# -----------------------------------------------------------------------------


def load_wav_mono(path: Path) -> tuple[np.ndarray, float]:
    x, fs = sf.read(path)
    x = np.asarray(x, dtype=float)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x, float(fs)


def _prepare_audio_workrate(x: np.ndarray, fs: float, cfg: AcousticFeatureConfig) -> tuple[np.ndarray, float]:
    x = _as_float64(x).ravel()
    if cfg.audio_work_fs is None:
        return x, fs

    fs_target = float(cfg.audio_work_fs)
    if cfg.fmax_hz >= 0.95 * (fs_target / 2.0):
        fs_target = max(fs_target, 2.2 * cfg.fmax_hz)

    if fs > fs_target:
        x = _sos_filter(x, fs, high=0.45 * fs_target, order=cfg.final_order)
        x = _resample(x, fs, fs_target)
        fs = fs_target

    return x, fs


def extract_auditory_envelope(x: np.ndarray, fs: float, cfg: AcousticFeatureConfig) -> np.ndarray:
    """Extract one final auditory-inspired broadband envelope."""
    x, fs = _prepare_audio_workrate(x, fs, cfg)
    nyq = fs / 2.0
    fmax = min(cfg.fmax_hz, 0.95 * nyq)
    centers = erb_center_frequencies(cfg.fmin_hz, fmax, cfg.n_bands)

    band_envs: list[np.ndarray] = []
    for cf in centers:
        bw = float(erb_bandwidth(cf))
        low = max(20.0, cf - bw / 2.0)
        high = min(0.95 * nyq, cf + bw / 2.0)
        if not (0 < low < high < nyq):
            continue

        band = _sos_filter(x, fs, low=low, high=high, order=cfg.filter_order)
        env = np.abs(hilbert(band))
        env = np.maximum(env, 0.0) ** cfg.compression
        band_envs.append(env)

    if not band_envs:
        raise RuntimeError("No valid auditory filterbank bands were created")

    envs = np.stack(band_envs, axis=1)
    envs = _sos_filter(envs, fs, high=cfg.band_env_lp, order=cfg.final_order, axis=0)
    env = envs.mean(axis=1)
    env = _resample(env, fs, cfg.fs_out)
    env = _sos_filter(env, cfg.fs_out, low=cfg.hp_out, order=cfg.final_order)
    env = _sos_filter(env, cfg.fs_out, high=cfg.lp_out, order=cfg.final_order)
    return np.asarray(env, dtype=np.float64)


def extract_acoustic_onset_from_envelope(env: np.ndarray, fs: float, cfg: AcousticFeatureConfig) -> np.ndarray:
    """Extract one-dimensional acoustic onset as half-wave rectified envelope derivative."""
    env = np.asarray(env, dtype=np.float64).ravel()
    derivative = np.diff(env, prepend=env[0]) * fs
    onset = np.maximum(derivative, 0.0)

    if cfg.onset_smooth_hz is not None and cfg.onset_smooth_hz > 0:
        onset = _sos_filter(onset, fs, high=cfg.onset_smooth_hz, order=cfg.final_order)

    onset = np.maximum(onset, 0.0)
    onset = _sos_filter(onset, fs, low=cfg.hp_out, order=cfg.final_order)
    onset = _sos_filter(onset, fs, high=cfg.lp_out, order=cfg.final_order)
    return np.asarray(onset, dtype=np.float64)


def extract_envelope_and_onset(x: np.ndarray, fs: float, cfg: AcousticFeatureConfig) -> dict[str, np.ndarray]:
    """Extract envelope, onset, and a two-column predictor matrix [envelope, onset]."""
    env = extract_auditory_envelope(x, fs, cfg)
    onset = extract_acoustic_onset_from_envelope(env, cfg.fs_out, cfg)

    n = min(len(env), len(onset))
    env = env[:n]
    onset = onset[:n]

    if cfg.normalize_features:
        env = _zscore_safe(env, axis=0)
        onset = _zscore_safe(onset, axis=0)

    pred = np.column_stack([env, onset])
    return {"env": env, "onset": onset, "pred": pred}


# -----------------------------------------------------------------------------
# EEG preprocessing
# -----------------------------------------------------------------------------


def preprocess_eeg_bdf(bdf_path: Path, cfg: EEGConfig) -> tuple[mne.io.BaseRaw, dict[str, Any]]:
    raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose="ERROR")
    ch_names = set(raw.ch_names)

    if all(ch in ch_names for ch in cfg.mastoids):
        raw.set_eeg_reference(ref_channels=list(cfg.mastoids), verbose="ERROR")

    raw.filter(l_freq=None, h_freq=cfg.fs_lowpass, verbose="ERROR")
    raw.resample(cfg.fs_out, verbose="ERROR")

    if cfg.eog_regression:
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
        eog_picks = mne.pick_types(raw.info, eog=True, exclude="bads")
        if len(eeg_picks) and len(eog_picks):
            x_eog = raw.get_data(picks=eog_picks).T
            x_eog = np.c_[x_eog, np.ones(x_eog.shape[0])]
            y_eeg = raw.get_data(picks=eeg_picks).T
            beta, *_ = np.linalg.lstsq(x_eog, y_eeg, rcond=None)
            y_hat = x_eog @ beta
            data = raw.get_data()
            data[eeg_picks, :] = (y_eeg - y_hat).T
            raw._data = data

    raw.filter(l_freq=cfg.hp_out, h_freq=None, verbose="ERROR")
    raw.filter(l_freq=None, h_freq=cfg.lp_out, verbose="ERROR")

    if cfg.scalp_only:
        keep = [
            ch
            for ch in raw.ch_names
            if ch.upper().startswith(
                ("FP", "AF", "F", "FC", "C", "CP", "P", "PO", "O", "FT", "TP", "T", "CZ", "PZ", "OZ", "FZ")
            )
            and "EXG" not in ch.upper()
        ]
        if keep:
            raw.pick(keep)

    return raw, {"n_channels": len(raw.ch_names), "fs": float(raw.info["sfreq"]), "channels": raw.ch_names}


def extract_target_events(ev: pd.DataFrame, events_path: Path) -> pd.DataFrame:
    if "trigger_type" not in ev.columns:
        raise ValueError(f"No trigger_type column in {events_path}")
    if "sample" not in ev.columns:
        raise ValueError(f"No sample column in {events_path}")
    return ev.loc[ev["trigger_type"].astype(str) == "targetonset"].copy()


def extract_eeg_trials(raw: mne.io.BaseRaw, target_events: pd.DataFrame, cfg: EEGConfig) -> list[np.ndarray]:
    fs = float(raw.info["sfreq"])
    data = raw.get_data().T
    trials: list[np.ndarray] = []

    for _, row in target_events.iterrows():
        sample = int(round(float(row["sample"]) * (fs / 512.0))) if fs != 512.0 else int(row["sample"])
        start = sample - int(round(5 * fs))
        stop = sample + int(round(50 * fs))
        if start < 0 or stop > len(data):
            continue
        trial = data[start:stop]
        trial = crop_toi_2d_time(trial, fs, cfg.crop_tmin, cfg.crop_tmax)
        trials.append(np.asarray(trial, dtype=np.float64))

    return trials


# -----------------------------------------------------------------------------
# Trial matching
# -----------------------------------------------------------------------------


def _subject_variants(subject: int | str) -> list[str]:
    sid = subject_id(subject)
    digits = "".join(ch for ch in sid if ch.isdigit())
    n = int(digits)
    return [f"sub-{n:03d}", f"sub{n:03d}", f"sub-{n}", f"sub{n}"]


def resolve_audio_path(
    bidsdir: Path,
    subject: int | str,
    stim_value: str,
    trigger_type: str,
    variant: str = "plain",
) -> Path:
    stem = Path(str(stim_value)).stem
    folder = "target" if trigger_type == "targetonset" else "masker"

    if variant == "plain":
        candidate_names = [stem]
    elif variant == "woa":
        candidate_names = [f"{stem}woa"]
    elif variant == "woacontrol":
        candidate_names = [f"{stem}woacontrol"]
    else:
        raise ValueError(f"Unknown audio variant: {variant}")

    subject_dirs = [bidsdir / "stimuli" / v / folder for v in _subject_variants(subject)]

    for sdir in subject_dirs:
        for name in candidate_names:
            candidate = sdir / f"{name}.wav"
            if candidate.exists():
                return candidate

    raise FileNotFoundError(f"Could not resolve WAV for {stim_value} ({trigger_type}) with variant={variant}")


def collect_trial_specs_from_events(
    bidsdir: Path,
    subject: int | str,
    ev: pd.DataFrame,
    audio_variant: str,
) -> list[dict[str, Any]]:
    rows = ev.loc[ev["trigger_type"].astype(str).isin(["targetonset", "maskeronset"])].copy()
    specs: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for _, row in rows.iterrows():
        trig = str(row["trigger_type"])

        if trig == "targetonset":
            if current is not None:
                specs.append(current)
            current = {
                "target_path": resolve_audio_path(bidsdir, subject, row["stim_file"], trig, variant=audio_variant),
                "masker_path": None,
                "masker_delay_sec": None,
                "single_talker_two_talker": row.get("single_talker_two_talker", None),
                "target_onset_sec": float(row.get("onset", np.nan)),
                "target_row": row.to_dict(),
            }

        elif trig == "maskeronset" and current is not None:
            current["masker_path"] = resolve_audio_path(bidsdir, subject, row["stim_file"], trig, variant=audio_variant)
            try:
                current["masker_delay_sec"] = float(row["onset"]) - float(current["target_onset_sec"])
            except Exception:
                current["masker_delay_sec"] = 0.0
            current["masker_row"] = row.to_dict()

    if current is not None:
        specs.append(current)

    return specs


# -----------------------------------------------------------------------------
# Main preprocessing
# -----------------------------------------------------------------------------


def process_subject_mtrf_env_onset(
    bidsdir: Path,
    subject: int | str,
    feature_cfg: AcousticFeatureConfig,
    eeg_cfg: EEGConfig,
    audio_variant: str = "plain",
) -> dict[str, Any]:
    sub = subject_id(subject)
    pfile = bidsdir / "participants.tsv"
    hearing_status = None

    if pfile.exists():
        participants = pd.read_csv(pfile, sep="\t")
        if "participant_id" in participants.columns and "hearing_status" in participants.columns:
            row = participants.loc[participants["participant_id"] == sub]
            if len(row):
                hearing_status = row.iloc[0]["hearing_status"]

    print(f"\nProcessing subject: {sub}")
    print(f"hearing_status = {hearing_status}")
    print(f"audio_variant = {audio_variant}")

    eeg_dir = bidsdir / sub / "eeg"
    run_specs = [(eeg_dir / f"{sub}_task-selectiveattention_eeg.bdf", eeg_dir / f"{sub}_task-selectiveattention_events.tsv")]

    run2_bdf = eeg_dir / f"{sub}_task-selectiveattention_run-2_eeg.bdf"
    run2_events = eeg_dir / f"{sub}_task-selectiveattention_run-2_events.tsv"
    if run2_bdf.exists() and run2_events.exists():
        run_specs.append((run2_bdf, run2_events))

    pred_st_trials: list[np.ndarray] = []
    stim_st_env_trials: list[np.ndarray] = []
    stim_st_onset_trials: list[np.ndarray] = []
    resp_st_trials: list[np.ndarray] = []

    pred_att_trials: list[np.ndarray] = []
    pred_ign_trials: list[np.ndarray] = []
    stim_att_env_trials: list[np.ndarray] = []
    stim_att_onset_trials: list[np.ndarray] = []
    stim_ign_env_trials: list[np.ndarray] = []
    stim_ign_onset_trials: list[np.ndarray] = []
    resp_tt_trials: list[np.ndarray] = []
    trial_meta: list[dict[str, Any]] = []

    feature_cache: dict[str, dict[str, np.ndarray]] = {}

    def get_features_cached(path: Path) -> dict[str, np.ndarray]:
        key = str(path.resolve())
        if key not in feature_cache:
            x, fs = load_wav_mono(path)
            feature_cache[key] = extract_envelope_and_onset(x, fs, feature_cfg)
        return feature_cache[key]

    for bdf_path, events_path in run_specs:
        if not bdf_path.exists():
            raise FileNotFoundError(f"Missing EEG file: {bdf_path}")
        if not events_path.exists():
            raise FileNotFoundError(f"Missing events file: {events_path}")

        raw, eeg_meta = preprocess_eeg_bdf(bdf_path, eeg_cfg)
        ev = pd.read_csv(events_path, sep="\t")
        target_events = extract_target_events(ev, events_path)
        eeg_trials = extract_eeg_trials(raw, target_events, eeg_cfg)
        stim_specs = collect_trial_specs_from_events(bidsdir, subject, ev, audio_variant)

        print(f"\nRun: {bdf_path.name}")
        print(f"events file: {events_path.name}")
        print(f"len(eeg_trials) = {len(eeg_trials)}")
        print(f"len(stim_specs) = {len(stim_specs)}")

        if len(eeg_trials) != len(stim_specs):
            raise RuntimeError(
                "Mismatch between EEG trials and stimulus specs: "
                f"{len(eeg_trials)=}, {len(stim_specs)=} for {events_path}"
            )

        for trial_idx, (eeg_trial, spec) in enumerate(zip(eeg_trials, stim_specs), start=1):
            print(
                f"Trial {trial_idx:02d}: "
                f"kind={spec['single_talker_two_talker']}, "
                f"target={Path(spec['target_path']).name}, "
                f"masker={None if spec['masker_path'] is None else Path(spec['masker_path']).name}, "
                f"delay={spec['masker_delay_sec']}"
            )

            tgt_feat = get_features_cached(spec["target_path"])
            tgt_env = crop_toi_1d(tgt_feat["env"], feature_cfg.fs_out, eeg_cfg.crop_tmin, eeg_cfg.crop_tmax)
            tgt_onset = crop_toi_1d(tgt_feat["onset"], feature_cfg.fs_out, eeg_cfg.crop_tmin, eeg_cfg.crop_tmax)
            tgt_pred = crop_toi_2d_time(tgt_feat["pred"], feature_cfg.fs_out, eeg_cfg.crop_tmin, eeg_cfg.crop_tmax)

            msk_env = None
            msk_onset = None
            msk_pred = None

            if spec["masker_path"] is not None:
                msk_feat = get_features_cached(spec["masker_path"])
                raw_msk_env = msk_feat["env"]
                raw_msk_onset = msk_feat["onset"]
                raw_msk_pred = msk_feat["pred"]

                if spec["masker_delay_sec"]:
                    pad = int(round(float(spec["masker_delay_sec"]) * feature_cfg.fs_out))
                    if pad > 0:
                        raw_msk_env = np.concatenate([np.zeros(pad, dtype=np.float64), raw_msk_env])
                        raw_msk_onset = np.concatenate([np.zeros(pad, dtype=np.float64), raw_msk_onset])
                        raw_msk_pred = np.vstack([np.zeros((pad, 2), dtype=np.float64), raw_msk_pred])

                msk_env = crop_toi_1d(raw_msk_env, feature_cfg.fs_out, eeg_cfg.crop_tmin, eeg_cfg.crop_tmax)
                msk_onset = crop_toi_1d(raw_msk_onset, feature_cfg.fs_out, eeg_cfg.crop_tmin, eeg_cfg.crop_tmax)
                msk_pred = crop_toi_2d_time(raw_msk_pred, feature_cfg.fs_out, eeg_cfg.crop_tmin, eeg_cfg.crop_tmax)

            lengths = [len(tgt_env), len(tgt_onset), tgt_pred.shape[0], eeg_trial.shape[0]]
            if msk_env is not None and msk_onset is not None and msk_pred is not None:
                lengths.extend([len(msk_env), len(msk_onset), msk_pred.shape[0]])

            n_samples = min(lengths)
            tgt_env = np.asarray(tgt_env[:n_samples], dtype=np.float64)
            tgt_onset = np.asarray(tgt_onset[:n_samples], dtype=np.float64)
            tgt_pred = np.asarray(tgt_pred[:n_samples, :], dtype=np.float64)
            eeg_trial = np.asarray(eeg_trial[:n_samples, :], dtype=np.float64)

            if msk_env is None:
                pred_st_trials.append(tgt_pred)
                stim_st_env_trials.append(tgt_env)
                stim_st_onset_trials.append(tgt_onset)
                resp_st_trials.append(eeg_trial)
                trial_kind = "singletalker"
            else:
                assert msk_onset is not None and msk_pred is not None
                msk_env = np.asarray(msk_env[:n_samples], dtype=np.float64)
                msk_onset = np.asarray(msk_onset[:n_samples], dtype=np.float64)
                msk_pred = np.asarray(msk_pred[:n_samples, :], dtype=np.float64)

                pred_att_trials.append(tgt_pred)
                pred_ign_trials.append(msk_pred)
                stim_att_env_trials.append(tgt_env)
                stim_att_onset_trials.append(tgt_onset)
                stim_ign_env_trials.append(msk_env)
                stim_ign_onset_trials.append(msk_onset)
                resp_tt_trials.append(eeg_trial)
                trial_kind = "twotalker"

            trial_meta.append(
                {
                    "run_bdf": str(bdf_path),
                    "events_path": str(events_path),
                    "trial_kind": trial_kind,
                    "target_path": str(spec["target_path"]),
                    "masker_path": None if spec["masker_path"] is None else str(spec["masker_path"]),
                    "masker_delay_sec": spec["masker_delay_sec"],
                    "aligned_samples": int(n_samples),
                    "aligned_seconds": float(n_samples / eeg_cfg.fs_out),
                    "attend_left_right": spec["target_row"].get("attend_left_right"),
                    "single_talker_two_talker": spec["target_row"].get("single_talker_two_talker"),
                    "stim_file_target_event": spec["target_row"].get("stim_file"),
                    "stim_file_masker_event": None if "masker_row" not in spec else spec["masker_row"].get("stim_file"),
                    "predictor_columns": ["envelope", "onset"],
                    "eeg_meta": eeg_meta,
                }
            )

    meta = {
        "subject": sub,
        "hearing_status": hearing_status,
        "feature_config": asdict(feature_cfg),
        "eeg_config": asdict(eeg_cfg),
        "audio_variant": audio_variant,
        "predictor_columns": ["envelope", "onset"],
        "recommended_models": {
            "baseline_forward_mtrf": {
                "description": "Predict EEG from attended envelope only",
                "two_talker_predictor_key": "stim_att_env",
                "response_key": "resp_tt",
                "lag_window_seconds": [0.0, 0.5],
            },
            "experiment_forward_mtrf": {
                "description": "Predict EEG from attended envelope + attended onset",
                "two_talker_predictor_key": "pred_att",
                "response_key": "resp_tt",
                "predictor_columns": ["envelope", "onset"],
                "lag_window_seconds": [0.0, 0.5],
            },
        },
        "notes": [
            "WAV-based preprocessing only.",
            "Envelope is computed from ERB-spaced filterbank envelopes collapsed by averaging.",
            "Onset is computed as half-wave rectified temporal derivative of the final envelope.",
            "Stimulus arrays are saved as object-array trial lists because trial lengths can differ slightly.",
        ],
    }

    return {
        "meta": np.array(json.dumps(_jsonable(meta)), dtype=object),
        "pred_st": _to_object_trials(pred_st_trials),
        "stim_st_env": _to_object_trials(stim_st_env_trials),
        "stim_st_onset": _to_object_trials(stim_st_onset_trials),
        "resp_st": _to_object_trials(resp_st_trials),
        "pred_att": _to_object_trials(pred_att_trials),
        "pred_ign": _to_object_trials(pred_ign_trials),
        "stim_att_env": _to_object_trials(stim_att_env_trials),
        "stim_att_onset": _to_object_trials(stim_att_onset_trials),
        "stim_ign_env": _to_object_trials(stim_ign_env_trials),
        "stim_ign_onset": _to_object_trials(stim_ign_onset_trials),
        "resp_tt": _to_object_trials(resp_tt_trials),
        "stim_att": _to_object_trials(stim_att_env_trials),
        "stim_ign": _to_object_trials(stim_ign_env_trials),
        "stim_st": _to_object_trials(stim_st_env_trials),
        "trial_meta": np.array(json.dumps(_jsonable(trial_meta)), dtype=object),
        "fs_stim": np.array(feature_cfg.fs_out, dtype=np.float64),
        "fs_resp": np.array(eeg_cfg.fs_out, dtype=np.float64),
        "predictor_names": np.array(["envelope", "onset"], dtype=object),
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Fuglsang WAV stimuli into mTRF-ready envelope + onset trial lists")
    parser.add_argument("--bidsdir", type=Path, required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audio-variant", choices=["plain", "woa", "woacontrol"], default="plain")
    parser.add_argument("--n-bands", type=int, default=16)
    parser.add_argument("--audio-work-fs", type=float, default=16000.0)
    parser.add_argument("--no-normalize-features", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_cfg = AcousticFeatureConfig(
        n_bands=args.n_bands,
        audio_work_fs=args.audio_work_fs,
        normalize_features=not args.no_normalize_features,
    )
    payload = process_subject_mtrf_env_onset(
        bidsdir=args.bidsdir,
        subject=args.subject,
        feature_cfg=feature_cfg,
        eeg_cfg=EEGConfig(),
        audio_variant=args.audio_variant,
    )
    ensure_dirs(args.out.parent)
    np.savez(args.out, **payload)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
