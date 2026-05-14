from __future__ import annotations

"""
Train a backward Eelbrain decoder on gamma-spectrogram/EEG data.

Input
-----
A .npz file created by a gamma preprocessing script.

Required keys:
    stim_att_gamma      : attended spectrogram trials, each (time, bands)
    stim_ign_gamma      : ignored spectrogram trials, each (time, bands)
    resp_tt             : EEG trials, each (time, channels)
    fs_stim
    fs_resp
    center_frequencies

Output
------
1. JSON summary with decoding accuracy and correlations
2. NPZ file with predictions and per-trial metrics
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.aad_project.data import ensure_dirs, load_npz_json_field


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class DecoderConfig:
    tstart: float = -0.5
    tstop: float = 0.0
    basis: float = 0.05
    basis_window: str = "hamming"
    partitions: int | None = None
    error: str = "l2"
    selective_stopping: int = 1
    scale_data: bool = True
    debug: bool = False


@dataclass
class TrialResult:
    trial_index: int
    r_att: float
    r_ign: float
    correct: int


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backward Eelbrain decoding from gamma-spectrogram features"
    )

    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)

    parser.add_argument("--tstart", type=float, default=-0.5)
    parser.add_argument("--tstop", type=float, default=0.0)
    parser.add_argument("--basis", type=float, default=0.05)
    parser.add_argument("--basis-window", default="hamming")
    parser.add_argument("--partitions", type=int, default=None)
    parser.add_argument("--error", choices=["l1", "l2"], default="l2")
    parser.add_argument("--selective-stopping", type=int, default=1)
    parser.add_argument("--no-scale-data", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save-resp", action="store_true")
    parser.add_argument("--timing", action="store_true")

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _timer() -> float:
    return time.perf_counter()


def _elapsed(t0: float) -> float:
    return time.perf_counter() - t0


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if x.size != y.size:
        raise ValueError(f"Correlation inputs must have same length: {x.size} vs {y.size}")

    x = x - x.mean()
    y = y - y.mean()

    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom == 0:
        return float("nan")

    return float(np.sum(x * y) / denom)


def _mean_band_correlation(pred: np.ndarray, true: np.ndarray) -> float:
    """
    pred/true shape:
        (bands, time)
    """
    if pred.shape != true.shape:
        raise ValueError(f"Band correlation shape mismatch: {pred.shape} vs {true.shape}")

    rs = [_pearsonr(pred[b], true[b]) for b in range(pred.shape[0])]
    return float(np.nanmean(rs))


def _safe_float(x: Any) -> float | None:
    try:
        xf = float(x)
    except Exception:
        return None

    if np.isnan(xf) or np.isinf(xf):
        return None

    return xf


def _require_keys(obj: np.lib.npyio.NpzFile, keys: list[str]) -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        raise KeyError(f"Missing required keys: {missing}. Available keys: {list(obj.keys())}")


def _stack_trials_2d(trials: np.ndarray, name: str) -> np.ndarray:
    if len(trials) == 0:
        raise ValueError(f"{name} contains no trials")

    arrs = [np.asarray(t, dtype=np.float64) for t in trials]
    shapes = [a.shape for a in arrs]

    if len(set(shapes)) != 1:
        raise ValueError(f"{name} trials do not all have same shape: {sorted(set(shapes))}")

    return np.stack(arrs, axis=0)


def _stack_trials_3d(trials: np.ndarray, name: str) -> np.ndarray:
    """
    Expected output:
        trials x time x bands
    """
    arr = _stack_trials_2d(trials, name)

    if arr.ndim != 3:
        raise ValueError(f"{name} should be 3D after stacking, got shape {arr.shape}")

    return arr


def _import_eelbrain():
    try:
        import eelbrain as eb
    except Exception as exc:
        raise RuntimeError(
            "Could not import eelbrain. Activate your eelbrain environment first."
        ) from exc

    return eb


# -----------------------------------------------------------------------------
# Eelbrain conversion and model fitting
# -----------------------------------------------------------------------------

def _make_sensor_dim(eb, n_channels: int):
    return eb.Categorial("sensor", [f"ch{i + 1:02d}" for i in range(n_channels)])


def _make_gamma_ndvars(
    eb,
    stim_att_gamma: np.ndarray,
    resp_tt: np.ndarray,
    fs: float,
    center_frequencies: np.ndarray,
):
    """
    Input:
        stim_att_gamma : trials x time x bands
        resp_tt        : trials x time x channels

    Eelbrain y:
        Case x frequency x time
    """

    n_trials, n_time, n_bands = stim_att_gamma.shape
    n_channels = resp_tt.shape[2]

    if len(center_frequencies) != n_bands:
        raise ValueError(
            f"center_frequencies has length {len(center_frequencies)}, "
            f"but spectrogram has {n_bands} bands"
        )

    time_dim = eb.UTS(0.0, 1.0 / fs, n_time)
    freq_dim = eb.Scalar("frequency", center_frequencies, unit="Hz")
    sensor_dim = _make_sensor_dim(eb, n_channels)

    # Convert trials x time x bands -> trials x bands x time
    stim_att_bt = np.transpose(stim_att_gamma, (0, 2, 1))

    y_att = eb.NDVar(
        stim_att_bt,
        (eb.Case, freq_dim, time_dim),
        name="stim_att_gamma",
    )

    x_eeg = eb.NDVar(
        resp_tt,
        (eb.Case, time_dim, sensor_dim),
        name="eeg",
    )

    return y_att, x_eeg


def _train_backward_cv(eb, y_att, x_eeg, cfg: DecoderConfig):
    n_trials = y_att.x.shape[0]
    partitions = cfg.partitions or n_trials

    if partitions < 2:
        raise ValueError("Need at least 2 partitions/trials for cross-validation")
    if partitions > n_trials:
        raise ValueError(f"partitions={partitions} exceeds n_trials={n_trials}")

    return eb.boosting(
        y_att,
        x_eeg,
        cfg.tstart,
        cfg.tstop,
        basis=cfg.basis,
        basis_window=cfg.basis_window,
        partitions=partitions,
        test=1,
        error=cfg.error,
        selective_stopping=cfg.selective_stopping,
        scale_data=cfg.scale_data,
        partition_results=True,
        debug=cfg.debug,
    )


def _get_y_pred(result, x_ndvar) -> np.ndarray:
    y_pred_obj = getattr(result, "y_pred", None)

    if y_pred_obj is not None and getattr(y_pred_obj, "x", None) is not None:
        return np.asarray(y_pred_obj.x, dtype=np.float64)

    try:
        y_pred_obj = result.cross_predict(x_ndvar, scale="original")
    except TypeError:
        y_pred_obj = result.cross_predict(x_ndvar)

    return np.asarray(y_pred_obj.x, dtype=np.float64)


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def _summarize_gamma_decoder(
    result,
    x_eeg,
    stim_att_gamma: np.ndarray,
    stim_ign_gamma: np.ndarray,
):
    """
    stim_att_gamma/stim_ign_gamma:
        trials x time x bands

    y_pred from Eelbrain:
        trials x bands x time
    """

    y_pred = _get_y_pred(result, x_eeg)

    stim_att_bt = np.transpose(stim_att_gamma, (0, 2, 1))
    stim_ign_bt = np.transpose(stim_ign_gamma, (0, 2, 1))

    if y_pred.shape != stim_att_bt.shape:
        raise RuntimeError(
            f"Prediction shape mismatch: y_pred={y_pred.shape}, "
            f"stim_att={stim_att_bt.shape}"
        )

    trial_results: list[TrialResult] = []

    for i in range(y_pred.shape[0]):
        r_att = _mean_band_correlation(y_pred[i], stim_att_bt[i])
        r_ign = _mean_band_correlation(y_pred[i], stim_ign_bt[i])
        correct = int(r_att > r_ign)
        trial_results.append(TrialResult(i, r_att, r_ign, correct))

    r_att_vals = np.array([tr.r_att for tr in trial_results], dtype=float)
    r_ign_vals = np.array([tr.r_ign for tr in trial_results], dtype=float)
    correct_vals = np.array([tr.correct for tr in trial_results], dtype=float)
    diff_vals = r_att_vals - r_ign_vals

    summary = {
        "feature_type": "gamma_spectrogram",
        "n_trials": int(len(trial_results)),
        "mean_r_att": float(np.nanmean(r_att_vals)),
        "mean_r_ign": float(np.nanmean(r_ign_vals)),
        "median_r_att": float(np.nanmedian(r_att_vals)),
        "median_r_ign": float(np.nanmedian(r_ign_vals)),
        "mean_r_att_minus_ign": float(np.nanmean(diff_vals)),
        "median_r_att_minus_ign": float(np.nanmedian(diff_vals)),
        "decoding_accuracy": float(np.nanmean(correct_vals)),
        "n_correct": int(np.nansum(correct_vals)),
        "eelbrain_r": _safe_float(getattr(result, "r", np.nan)),
        "eelbrain_r_rank": _safe_float(getattr(result, "r_rank", np.nan)),
        "proportion_explained": _safe_float(getattr(result, "proportion_explained", np.nan)),
        "l1_residual": _safe_float(getattr(result, "l1_residual", np.nan)),
        "l1_total": _safe_float(getattr(result, "l1_total", np.nan)),
        "l2_residual": _safe_float(getattr(result, "l2_residual", np.nan)),
        "l2_total": _safe_float(getattr(result, "l2_total", np.nan)),
        "correlation_method": "Pearson r per frequency band, then averaged across bands",
    }

    return trial_results, summary, y_pred, stim_att_bt, stim_ign_bt


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    timings: dict[str, float] = {}
    t_all = _timer()

    cfg = DecoderConfig(
        tstart=args.tstart,
        tstop=args.tstop,
        basis=args.basis,
        basis_window=args.basis_window,
        partitions=args.partitions,
        error=args.error,
        selective_stopping=args.selective_stopping,
        scale_data=not args.no_scale_data,
        debug=args.debug,
    )

    t0 = _timer()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file does not exist: {args.input}")

    obj = np.load(args.input, allow_pickle=True)

    _require_keys(
        obj,
        [
            "stim_att_gamma",
            "stim_ign_gamma",
            "resp_tt",
            "fs_stim",
            "fs_resp",
            "center_frequencies",
        ],
    )

    meta = load_npz_json_field(obj, "meta")
    trial_meta = load_npz_json_field(obj, "trial_meta")

    fs_stim = float(np.asarray(obj["fs_stim"]).item())
    fs_resp = float(np.asarray(obj["fs_resp"]).item())

    if not np.isclose(fs_stim, fs_resp):
        raise ValueError(f"Stimulus and response sampling rates differ: {fs_stim} vs {fs_resp}")

    center_frequencies = np.asarray(obj["center_frequencies"], dtype=np.float64).ravel()

    stim_att_gamma = _stack_trials_3d(obj["stim_att_gamma"], "stim_att_gamma")
    stim_ign_gamma = _stack_trials_3d(obj["stim_ign_gamma"], "stim_ign_gamma")
    resp_tt = _stack_trials_2d(obj["resp_tt"], "resp_tt")

    timings["load_and_stack_sec"] = _elapsed(t0)

    if not (
        stim_att_gamma.shape == stim_ign_gamma.shape
        and stim_att_gamma.shape[0] == resp_tt.shape[0]
        and stim_att_gamma.shape[1] == resp_tt.shape[1]
    ):
        raise ValueError(
            "Trial arrays do not align: "
            f"stim_att_gamma={stim_att_gamma.shape}, "
            f"stim_ign_gamma={stim_ign_gamma.shape}, "
            f"resp_tt={resp_tt.shape}"
        )

    t0 = _timer()
    eb = _import_eelbrain()
    y_att, x_eeg = _make_gamma_ndvars(
        eb,
        stim_att_gamma,
        resp_tt,
        fs_stim,
        center_frequencies,
    )
    timings["make_ndvars_sec"] = _elapsed(t0)

    t0 = _timer()
    result = _train_backward_cv(eb, y_att, x_eeg, cfg)
    timings["fit_two_talker_sec"] = _elapsed(t0)

    t0 = _timer()
    trial_results, summary, y_pred, stim_att_bt, stim_ign_bt = _summarize_gamma_decoder(
        result,
        x_eeg,
        stim_att_gamma,
        stim_ign_gamma,
    )
    timings["predict_and_summarize_sec"] = _elapsed(t0)

    timings["total_before_save_sec"] = _elapsed(t_all)

    payload = {
        "input_file": str(args.input),
        "fs_stim": fs_stim,
        "fs_resp": fs_resp,
        "decoder_config": asdict(cfg),
        "meta": meta,
        "summary_two_talker": summary,
        "trial_results_two_talker": [asdict(tr) for tr in trial_results],
        "n_channels": int(resp_tt.shape[2]),
        "n_timepoints": int(resp_tt.shape[1]),
        "n_bands": int(stim_att_gamma.shape[2]),
        "center_frequencies": center_frequencies.tolist(),
        "trial_meta": trial_meta,
        "timings_sec": timings,
        "notes": [
            "This model reconstructs a multi-band gamma/gammatone spectrogram from EEG.",
            "Correlations are computed per frequency band and then averaged across bands.",
            "The final decoding accuracy is directly comparable to the envelope baseline because it is still based on r_att > r_ign.",
        ],
    }

    ensure_dirs(args.outdir)

    stem = args.input.stem
    json_path = args.outdir / f"{stem}_gamma_backward_summary.json"
    npz_path = args.outdir / f"{stem}_gamma_backward_predictions.npz"

    t0 = _timer()

    save_payload = {
        "y_pred_att_gamma": y_pred,
        "stim_att_gamma": stim_att_bt,
        "stim_ign_gamma": stim_ign_bt,
        "center_frequencies": center_frequencies,
        "trial_r_att": np.array([tr.r_att for tr in trial_results], dtype=np.float64),
        "trial_r_ign": np.array([tr.r_ign for tr in trial_results], dtype=np.float64),
        "trial_correct": np.array([tr.correct for tr in trial_results], dtype=np.int64),
        "decoder_h": getattr(getattr(result, "h_scaled", None), "x", np.asarray([])),
    }

    if args.save_resp:
        save_payload["resp_tt"] = resp_tt

    np.savez_compressed(npz_path, **save_payload)

    timings["save_sec"] = _elapsed(t0)
    timings["total_sec"] = _elapsed(t_all)
    payload["timings_sec"] = timings

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Saved summary:", json_path)
    print("Saved predictions:", npz_path)
    print()
    print("Gamma two-talker summary")
    print("------------------------")
    for key, value in summary.items():
        print(f"{key}: {value}")

    if args.timing:
        print()
        print("Timing")
        print("------")
        for key, value in timings.items():
            print(f"{key}: {value:.2f} sec")


if __name__ == "__main__":
    main()