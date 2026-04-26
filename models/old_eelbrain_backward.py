from __future__ import annotations

"""
Train a backward Eelbrain decoder on preprocessed Fuglsang-style envelope/EEG data.

Expected input
--------------
An .npz file created by the preprocessing pipeline with keys:
    stim_att : object array of shape (n_trials,), each entry (time,)
    stim_ign : object array of shape (n_trials,), each entry (time,)
    resp_tt  : object array of shape (n_trials,), each entry (time, channels)
Optional:
    stim_st, resp_st for single-talker sanity-check reconstruction

Main output
-----------
1. JSON summary with decoding accuracy and correlation metrics
2. NPZ file with per-trial predictions and metrics

Notes
-----
- Uses Eelbrain's backward model: stimulus envelope is y, EEG is x.
- Uses test=1 cross-validation so every trial is predicted from complementary
  training data.
- Uses partitions = n_trials by default, which becomes leave-one-trial-out.
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


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

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backward Eelbrain decoding from preprocessed .npz data")
    p.add_argument("--input", type=Path, required=True, help="Path to preprocessed .npz file")
    p.add_argument("--outdir", type=Path, required=True, help="Directory for result files")

    p.add_argument("--tstart", type=float, default=-0.5, help="Decoder lag-window start in seconds")
    p.add_argument("--tstop", type=float, default=0.0, help="Decoder lag-window stop in seconds")
    p.add_argument("--basis", type=float, default=0.05, help="Basis window length in seconds")
    p.add_argument("--basis-window", default="hamming", help="Basis window type for eelbrain.boosting")
    p.add_argument("--partitions", type=int, default=None, help="CV partitions; default is n_trials")
    p.add_argument("--error", choices=["l1", "l2"], default="l2")
    p.add_argument("--selective-stopping", type=int, default=1)
    p.add_argument("--no-scale-data", action="store_true", help="Disable Eelbrain input normalization")
    p.add_argument("--debug", action="store_true")

    # Efficiency controls
    p.add_argument(
        "--run-single-talker",
        action="store_true",
        help="Also run single-talker sanity model. This can add substantial runtime.",
    )
    p.add_argument(
        "--save-resp",
        action="store_true",
        help="Also save resp_tt in predictions NPZ. Usually unnecessary because it is already in input NPZ.",
    )
    p.add_argument(
        "--timing",
        action="store_true",
        help="Print timing information for loading, fitting, prediction, and saving.",
    )

    return p.parse_args()


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
        raise ValueError(f"Correlation inputs must have same length, got {x.size} and {y.size}")

    x = x - x.mean()
    y = y - y.mean()

    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom == 0:
        return float("nan")

    return float(np.sum(x * y) / denom)


def _safe_float(x: Any) -> float | None:
    try:
        xf = float(x)
    except Exception:
        return None
    if np.isnan(xf) or np.isinf(xf):
        return None
    return xf


def _load_json_scalar(arr: np.ndarray | Any) -> Any:
    if isinstance(arr, np.ndarray) and arr.shape == ():
        arr = arr.item()
    if isinstance(arr, bytes):
        arr = arr.decode("utf-8")
    if isinstance(arr, str):
        try:
            return json.loads(arr)
        except json.JSONDecodeError:
            return arr
    return arr


def _require_keys(obj: np.lib.npyio.NpzFile, keys: list[str]) -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        raise KeyError(f"Missing required keys in input NPZ: {missing}. Available keys: {list(obj.keys())}")


def _stack_trials_1d(trials: np.ndarray, name: str) -> np.ndarray:
    if len(trials) == 0:
        raise ValueError(f"{name} contains no trials")

    arrs = [np.asarray(t, dtype=np.float64).ravel() for t in trials]
    lengths = [a.shape[0] for a in arrs]
    if len(set(lengths)) != 1:
        raise ValueError(f"{name} trials do not all have same length: {sorted(set(lengths))}")

    return np.stack(arrs, axis=0)


def _stack_trials_2d(trials: np.ndarray, name: str) -> np.ndarray:
    if len(trials) == 0:
        raise ValueError(f"{name} contains no trials")

    arrs = [np.asarray(t, dtype=np.float64) for t in trials]
    shapes = [a.shape for a in arrs]
    if len(set(shapes)) != 1:
        raise ValueError(f"{name} trials do not all have same shape: {sorted(set(shapes))}")

    return np.stack(arrs, axis=0)


def _import_eelbrain():
    try:
        import eelbrain as eb
    except Exception as exc:
        raise RuntimeError(
            "Could not import eelbrain. Activate your eelbrain environment first, "
            "for example: conda activate eelbrain"
        ) from exc
    return eb


# -----------------------------------------------------------------------------
# Eelbrain conversion and model fitting
# -----------------------------------------------------------------------------

def _make_sensor_dim(eb, n_channels: int):
    return eb.Categorial("sensor", [f"ch{i + 1:02d}" for i in range(n_channels)])


def _make_ndvars(eb, stim_att: np.ndarray, resp_tt: np.ndarray, fs: float):
    _, n_time = stim_att.shape
    n_channels = resp_tt.shape[2]

    time_dim = eb.UTS(0.0, 1.0 / fs, n_time)
    sensor_dim = _make_sensor_dim(eb, n_channels)

    y_att = eb.NDVar(stim_att, (eb.Case, time_dim), name="stim_att")
    x_eeg = eb.NDVar(resp_tt, (eb.Case, time_dim, sensor_dim), name="eeg")

    return y_att, x_eeg


def _train_backward_cv(eb, y_att, x_eeg, cfg: DecoderConfig):
    n_trials = y_att.x.shape[0]
    partitions = cfg.partitions or n_trials

    if partitions < 2:
        raise ValueError("Need at least 2 partitions/trials for cross-validation")
    if partitions > n_trials:
        raise ValueError(f"partitions={partitions} exceeds number of trials={n_trials}")

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
    """
    Get cross-validated predictions robustly across Eelbrain versions.
    """
    y_pred_obj = getattr(result, "y_pred", None)
    if y_pred_obj is not None and getattr(y_pred_obj, "x", None) is not None:
        return np.asarray(y_pred_obj.x, dtype=np.float64)

    try:
        y_pred_obj = result.cross_predict(x_ndvar, scale="original")
    except TypeError:
        y_pred_obj = result.cross_predict(x_ndvar)

    return np.asarray(y_pred_obj.x, dtype=np.float64)


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

def _summarize_decoder(result, x_eeg, stim_att: np.ndarray, stim_ign: np.ndarray):
    y_pred = _get_y_pred(result, x_eeg)

    if y_pred.shape != stim_att.shape:
        raise RuntimeError(f"Prediction shape mismatch: {y_pred.shape=} but {stim_att.shape=}")

    trial_results: list[TrialResult] = []

    for i in range(y_pred.shape[0]):
        r_att = _pearsonr(y_pred[i], stim_att[i])
        r_ign = _pearsonr(y_pred[i], stim_ign[i])
        correct = int(r_att > r_ign)
        trial_results.append(TrialResult(i, r_att, r_ign, correct))

    r_att_vals = np.array([tr.r_att for tr in trial_results], dtype=float)
    r_ign_vals = np.array([tr.r_ign for tr in trial_results], dtype=float)
    correct_vals = np.array([tr.correct for tr in trial_results], dtype=float)

    proportion_explained = getattr(result, "proportion_explained", None)
    if proportion_explained is None:
        l2_total = _safe_float(getattr(result, "l2_total", np.nan))
        l2_residual = _safe_float(getattr(result, "l2_residual", np.nan))
        if l2_total in (None, 0.0) or l2_residual is None:
            prop_explained = None
        else:
            prop_explained = 1.0 - (l2_residual / l2_total)
    else:
        prop_explained = _safe_float(proportion_explained)

    summary = {
        "n_trials": int(len(trial_results)),
        "mean_r_att": float(np.nanmean(r_att_vals)),
        "mean_r_ign": float(np.nanmean(r_ign_vals)),
        "median_r_att": float(np.nanmedian(r_att_vals)),
        "median_r_ign": float(np.nanmedian(r_ign_vals)),
        "mean_r_att_minus_ign": float(np.nanmean(r_att_vals - r_ign_vals)),
        "median_r_att_minus_ign": float(np.nanmedian(r_att_vals - r_ign_vals)),
        "decoding_accuracy": float(np.nanmean(correct_vals)),
        "n_correct": int(np.nansum(correct_vals)),
        "eelbrain_r": _safe_float(getattr(result, "r", np.nan)),
        "eelbrain_r_rank": _safe_float(getattr(result, "r_rank", np.nan)),
        "proportion_explained": prop_explained,
        "l1_residual": _safe_float(getattr(result, "l1_residual", np.nan)),
        "l1_total": _safe_float(getattr(result, "l1_total", np.nan)),
        "l2_residual": _safe_float(getattr(result, "l2_residual", np.nan)),
        "l2_total": _safe_float(getattr(result, "l2_total", np.nan)),
    }

    return trial_results, summary, y_pred


def _single_talker_summary(
    eb,
    stim_st: np.ndarray,
    resp_st: np.ndarray,
    fs: float,
    cfg: DecoderConfig,
) -> dict[str, Any] | None:
    if stim_st.size == 0 or resp_st.size == 0:
        return None
    if stim_st.shape[0] < 2:
        return None
    if not (stim_st.shape[0] == resp_st.shape[0] and stim_st.shape[1] == resp_st.shape[1]):
        return None

    time_dim = eb.UTS(0.0, 1.0 / fs, stim_st.shape[1])
    sensor_dim = _make_sensor_dim(eb, resp_st.shape[2])

    y_st = eb.NDVar(stim_st, (eb.Case, time_dim), name="stim_st")
    x_st = eb.NDVar(resp_st, (eb.Case, time_dim, sensor_dim), name="resp_st")

    st_partitions = cfg.partitions or stim_st.shape[0]
    st_partitions = min(max(2, st_partitions), stim_st.shape[0])

    st_result = eb.boosting(
        y_st,
        x_st,
        cfg.tstart,
        cfg.tstop,
        basis=cfg.basis,
        basis_window=cfg.basis_window,
        partitions=st_partitions,
        test=1,
        error=cfg.error,
        selective_stopping=cfg.selective_stopping,
        scale_data=cfg.scale_data,
        partition_results=True,
        debug=cfg.debug,
    )

    y_st_pred = _get_y_pred(st_result, x_st)
    r_st = np.array([_pearsonr(y_st_pred[i], stim_st[i]) for i in range(stim_st.shape[0])], dtype=float)

    return {
        "n_trials": int(stim_st.shape[0]),
        "mean_r": float(np.nanmean(r_st)),
        "median_r": float(np.nanmedian(r_st)),
        "eelbrain_r": _safe_float(getattr(st_result, "r", np.nan)),
        "eelbrain_r_rank": _safe_float(getattr(st_result, "r_rank", np.nan)),
        "l2_residual": _safe_float(getattr(st_result, "l2_residual", np.nan)),
        "l2_total": _safe_float(getattr(st_result, "l2_total", np.nan)),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

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
    obj = np.load(args.input, allow_pickle=True)
    _require_keys(obj, ["stim_att", "stim_ign", "resp_tt", "fs_stim", "fs_resp"])

    meta = _load_json_scalar(obj["meta"]) if "meta" in obj else None
    trial_meta = _load_json_scalar(obj["trial_meta"]) if "trial_meta" in obj else None

    fs_stim = float(np.asarray(obj["fs_stim"]).item())
    fs_resp = float(np.asarray(obj["fs_resp"]).item())

    if not np.isclose(fs_stim, fs_resp):
        raise ValueError(f"Stimulus and response sampling rates differ: {fs_stim} vs {fs_resp}")

    stim_att = _stack_trials_1d(obj["stim_att"], "stim_att")
    stim_ign = _stack_trials_1d(obj["stim_ign"], "stim_ign")
    resp_tt = _stack_trials_2d(obj["resp_tt"], "resp_tt")

    timings["load_and_stack_sec"] = _elapsed(t0)

    if not (
        stim_att.shape == stim_ign.shape
        and stim_att.shape[0] == resp_tt.shape[0]
        and stim_att.shape[1] == resp_tt.shape[1]
    ):
        raise ValueError(
            "Two-talker trial arrays do not align: "
            f"stim_att {stim_att.shape}, stim_ign {stim_ign.shape}, resp_tt {resp_tt.shape}"
        )

    t0 = _timer()
    eb = _import_eelbrain()
    y_att, x_eeg = _make_ndvars(eb, stim_att, resp_tt, fs_stim)
    timings["make_ndvars_sec"] = _elapsed(t0)

    t0 = _timer()
    result = _train_backward_cv(eb, y_att, x_eeg, cfg)
    timings["fit_two_talker_sec"] = _elapsed(t0)

    t0 = _timer()
    trial_results, summary, y_pred = _summarize_decoder(result, x_eeg, stim_att, stim_ign)
    timings["predict_and_summarize_sec"] = _elapsed(t0)

    single_talker_summary = None
    if args.run_single_talker and "stim_st" in obj and "resp_st" in obj:
        t0 = _timer()
        stim_st = _stack_trials_1d(obj["stim_st"], "stim_st")
        resp_st = _stack_trials_2d(obj["resp_st"], "resp_st")
        single_talker_summary = _single_talker_summary(eb, stim_st, resp_st, fs_stim, cfg)
        timings["single_talker_sec"] = _elapsed(t0)

    timings["total_sec"] = _elapsed(t_all)

    payload = {
        "input_file": str(args.input),
        "fs_stim": fs_stim,
        "fs_resp": fs_resp,
        "decoder_config": asdict(cfg),
        "meta": meta,
        "summary_two_talker": summary,
        "summary_single_talker": single_talker_summary,
        "trial_results_two_talker": [asdict(tr) for tr in trial_results],
        "n_channels": int(resp_tt.shape[2]),
        "n_timepoints": int(resp_tt.shape[1]),
        "trial_meta": trial_meta,
        "timings_sec": timings,
        "notes": [
            "Single-talker model is optional because it can add substantial runtime.",
            "resp_tt is not saved in predictions NPZ unless --save-resp is used, because it already exists in the input NPZ.",
        ],
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem
    json_path = args.outdir / f"{stem}_backward_summary.json"
    npz_path = args.outdir / f"{stem}_backward_predictions.npz"

    t0 = _timer()
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    save_payload = {
        "y_pred_att": y_pred,
        "stim_att": stim_att,
        "stim_ign": stim_ign,
        "trial_r_att": np.array([tr.r_att for tr in trial_results], dtype=np.float64),
        "trial_r_ign": np.array([tr.r_ign for tr in trial_results], dtype=np.float64),
        "trial_correct": np.array([tr.correct for tr in trial_results], dtype=np.int64),
        "decoder_h": getattr(getattr(result, "h_scaled", None), "x", np.asarray([])),
    }

    if args.save_resp:
        save_payload["resp_tt"] = resp_tt

    np.savez(npz_path, **save_payload)
    timings["save_sec"] = _elapsed(t0)

    print("Saved summary:", json_path)
    print("Saved predictions:", npz_path)
    print()
    print("Two-talker summary")
    print("------------------")
    for key, value in summary.items():
        print(f"{key}: {value}")

    if single_talker_summary is not None:
        print()
        print("Single-talker sanity summary")
        print("----------------------------")
        for key, value in single_talker_summary.items():
            print(f"{key}: {value}")

    if args.timing:
        print()
        print("Timing")
        print("------")
        for key, value in timings.items():
            print(f"{key}: {value:.2f} sec")


if __name__ == "__main__":
    main()