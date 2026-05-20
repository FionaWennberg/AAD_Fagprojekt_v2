from __future__ import annotations

"""
Backward mTRF / Eelbrain decoder for multiple acoustic features.

Train a backward decoder:

    EEG -> [envelope, onset]

and use the reconstructed features to classify attended vs ignored speech.

Place this file at:

    src/aad_project/model_backward_mtrf_env_onset.py

Expected preprocessing input keys:

    pred_att        object array, each trial shape (time, 2)
    pred_ign        object array, each trial shape (time, 2)
    resp_tt         object array, each trial shape (time, channels)
    predictor_names usually ["envelope", "onset"]

Example:

python -m src.aad_project.model_backward_mtrf_env_onset \
  --input data/processed/sub-004_mtrf_env_onset.npz \
  --out-dir results_backward_mtrf \
  --subject 4
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import eelbrain as eb
except ImportError as exc:
    raise ImportError("This model requires eelbrain. Activate your eelbrain environment first.") from exc

from src.aad_project.data import ensure_dirs, subject_id


@dataclass
class BackwardMTRFConfig:
    tstart: float = -0.5
    tstop: float = 0.0
    basis: float = 0.05
    basis_window: str = "hamming"
    test: int = 1
    partitions: int | None = None
    error: str = "l2"
    selective_stopping: int = 1
    scale_data: bool = True
    score_mode: str = "mean"  # mean, weighted, env_only
    feature_weights: tuple[float, float] = (1.0, 1.0)


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


def _load_json_scalar(x: Any) -> dict[str, Any]:
    if isinstance(x, np.ndarray):
        x = x.item()
    if isinstance(x, bytes):
        x = x.decode("utf-8")
    if isinstance(x, str):
        return json.loads(x)
    return {}


def _as_trial_list(x: np.ndarray) -> list[np.ndarray]:
    return [np.asarray(v, dtype=np.float64) for v in x]


def _to_object_trials(trials: list[np.ndarray], dtype=np.float64) -> np.ndarray:
    out = np.empty((len(trials),), dtype=object)
    for i, trial in enumerate(trials):
        out[i] = np.asarray(trial, dtype=dtype)
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = min(len(a), len(b))
    if n < 3:
        return np.nan
    a = a[:n]
    b = b[:n]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _make_case_time_ndvar_1d(trials: list[np.ndarray], fs: float, name: str) -> eb.NDVar:
    lengths = [len(t) for t in trials]
    if len(set(lengths)) != 1:
        raise ValueError(f"{name} trials have unequal lengths after truncation.")
    data = np.stack([np.asarray(t, dtype=np.float64) for t in trials], axis=0)
    case = eb.Case(len(trials))
    time = eb.UTS(0.0, 1.0 / fs, data.shape[1])
    return eb.NDVar(data, (case, time), name=name)


# def _make_case_time_sensor_ndvar(trials: list[np.ndarray], fs: float, name: str) -> eb.NDVar:
#     shapes = [t.shape for t in trials]
#     if len(set(shapes)) != 1:
#         raise ValueError(f"{name} trials have unequal shapes after truncation.")
#     data = np.stack([np.asarray(t, dtype=np.float64) for t in trials], axis=0)
#     case = eb.Case(len(trials))
#     time = eb.UTS(0.0, 1.0 / fs, data.shape[1])
#     sensor = eb.Sensor.from_names([f"EEG{i:03d}" for i in range(data.shape[2])])
#     return eb.NDVar(data, (case, time, sensor), name=name)

def _make_case_time_sensor_ndvar(data: np.ndarray, fs: float, name: str = "eeg"):
    """
    Convert EEG trials to an Eelbrain NDVar with dimensions:
        case x time x sensor

    data shape must be:
        (n_trials, n_times, n_channels)
    """
    data = np.asarray(data, dtype=float)

    if data.ndim != 3:
        raise ValueError(
            f"Expected EEG data with shape (trials, time, channels), got {data.shape}"
        )

    n_trials, n_times, n_channels = data.shape

    case = eb.Case(n_trials)
    time = eb.UTS(0, 1 / fs, n_times)

    sensor_names = [f"EEG{i:03d}" for i in range(n_channels)]

    # Compatibility fix:
    # Some Eelbrain versions do not have eb.Sensor.from_names().
    # A categorical sensor dimension is enough for boosting/decoding.
    try:
        sensor = eb.Sensor.from_names(sensor_names)
    except AttributeError:
        sensor = eb.Categorial("sensor", sensor_names)

    return eb.NDVar(data, (case, time, sensor), name=name)


def _truncate_all_trials(
    pred_att: list[np.ndarray],
    pred_ign: list[np.ndarray],
    resp_tt: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    min_global = min(
        min(x.shape[0] for x in pred_att),
        min(x.shape[0] for x in pred_ign),
        min(x.shape[0] for x in resp_tt),
    )
    out_att, out_ign, out_resp = [], [], []
    for a, i, r in zip(pred_att, pred_ign, resp_tt):
        n = min(min_global, a.shape[0], i.shape[0], r.shape[0])
        out_att.append(np.asarray(a[:n, :], dtype=np.float64))
        out_ign.append(np.asarray(i[:n, :], dtype=np.float64))
        out_resp.append(np.asarray(r[:n, :], dtype=np.float64))
    return out_att, out_ign, out_resp


def _get_cross_prediction(result: Any, x: eb.NDVar) -> eb.NDVar:
    if hasattr(result, "cross_predict"):
        try:
            return result.cross_predict(x, scale="original")
        except TypeError:
            return result.cross_predict(x)

    if hasattr(result, "y_pred"):
        return result.y_pred

    if hasattr(result, "prediction"):
        return result.prediction

    raise AttributeError("Could not find cross-validated predictions on Eelbrain result.")

def _feature_score(r_att: np.ndarray, r_ign: np.ndarray, mode: str, weights: tuple[float, float]) -> float:
    diff = np.asarray(r_att, dtype=np.float64) - np.asarray(r_ign, dtype=np.float64)
    if mode == "env_only":
        return float(diff[0])
    if mode == "mean":
        return float(np.nanmean(diff))
    if mode == "weighted":
        w = np.asarray(weights, dtype=np.float64)[: len(diff)]
        if np.sum(np.abs(w)) < 1e-12:
            raise ValueError("feature_weights cannot all be zero")
        w = w / np.sum(np.abs(w))
        return float(np.nansum(w * diff))
    raise ValueError(f"Unknown score_mode: {mode}")


def fit_one_feature_backward(
    eeg_trials: list[np.ndarray],
    target_trials: list[np.ndarray],
    fs: float,
    cfg: BackwardMTRFConfig,
    feature_name: str,
) -> tuple[Any, np.ndarray]:
    eeg = _make_case_time_sensor_ndvar(eeg_trials, fs, name="eeg")
    y = _make_case_time_ndvar_1d(target_trials, fs, name=feature_name)
    partitions = cfg.partitions if cfg.partitions is not None else len(eeg_trials)

    result = eb.boosting(
    y,
    eeg,
    cfg.tstart,
    cfg.tstop,
    basis=cfg.basis,
    basis_window=cfg.basis_window,
    test=cfg.test,
    partitions=partitions,
    partition_results=True,
    error=cfg.error,
    selective_stopping=cfg.selective_stopping,
    scale_data=cfg.scale_data,
)

    y_hat = np.asarray(_get_cross_prediction(result, eeg).x, dtype=np.float64)
    return result, y_hat


def run_backward_mtrf_env_onset(
    input_path: Path,
    out_dir: Path,
    subject: int | str | None = None,
    cfg: BackwardMTRFConfig | None = None,
) -> dict[str, Any]:
    if cfg is None:
        cfg = BackwardMTRFConfig()

    ensure_dirs(out_dir)
    data = np.load(input_path, allow_pickle=True)

    pred_att = _as_trial_list(data["pred_att"])
    pred_ign = _as_trial_list(data["pred_ign"])
    resp_tt = _as_trial_list(data["resp_tt"])

    if "predictor_names" in data:
        predictor_names = [str(x) for x in data["predictor_names"]]
    else:
        predictor_names = ["envelope", "onset"]

    fs = float(data["fs_resp"])
    if "fs_stim" in data:
        fs_stim = float(data["fs_stim"])
        if not np.isclose(fs, fs_stim):
            raise ValueError(f"fs_resp={fs} and fs_stim={fs_stim} do not match")

    if len(pred_att) == 0:
        raise RuntimeError("No two-talker trials found in pred_att")

    pred_att, pred_ign, resp_tt = _truncate_all_trials(pred_att, pred_ign, resp_tt)
    n_features = pred_att[0].shape[1]
    if n_features < 2:
        raise ValueError(f"Expected at least two features [envelope, onset], got {pred_att[0].shape}")

    if subject is None:
        try:
            meta = _load_json_scalar(data["meta"])
            subject_label = str(meta.get("subject", "unknown"))
        except Exception:
            subject_label = "unknown"
    else:
        subject_label = subject_id(subject)

    print(f"\nRunning backward mTRF for subject {subject_label}")
    print(f"input: {input_path}")
    print(f"n_trials = {len(resp_tt)}")
    print(f"fs = {fs}")
    print(f"predictors = {predictor_names}")
    print(f"score_mode = {cfg.score_mode}")

    reconstructed_features: list[np.ndarray] = []
    feature_results: dict[str, Any] = {}

    for j in range(n_features):
        feature_name = predictor_names[j] if j < len(predictor_names) else f"feature_{j}"
        print(f"\nFitting backward decoder for feature {j}: {feature_name}")
        target_trials = [trial[:, j] for trial in pred_att]
        result, y_hat = fit_one_feature_backward(resp_tt, target_trials, fs, cfg, feature_name)
        reconstructed_features.append(y_hat)
        feature_results[feature_name] = result

    y_hat_multi = np.stack(reconstructed_features, axis=-1)  # trials x time x features

    rows: list[dict[str, Any]] = []
    for trial_idx in range(len(resp_tt)):
        r_att, r_ign = [], []
        for j in range(n_features):
            r_att.append(_corr(y_hat_multi[trial_idx, :, j], pred_att[trial_idx][:, j]))
            r_ign.append(_corr(y_hat_multi[trial_idx, :, j], pred_ign[trial_idx][:, j]))

        r_att_arr = np.asarray(r_att, dtype=np.float64)
        r_ign_arr = np.asarray(r_ign, dtype=np.float64)
        score = _feature_score(r_att_arr, r_ign_arr, cfg.score_mode, cfg.feature_weights)

        row: dict[str, Any] = {
            "subject": subject_label,
            "trial": trial_idx,
            "score": score,
            "correct": bool(score > 0),
        }
        for j in range(n_features):
            feature_name = predictor_names[j] if j < len(predictor_names) else f"feature_{j}"
            row[f"r_att_{feature_name}"] = r_att_arr[j]
            row[f"r_ign_{feature_name}"] = r_ign_arr[j]
            row[f"diff_{feature_name}"] = r_att_arr[j] - r_ign_arr[j]
        rows.append(row)

    trial_df = pd.DataFrame(rows)
    accuracy = float(trial_df["correct"].mean())

    summary = {
        "subject": subject_label,
        "input_path": str(input_path),
        "n_trials": int(len(resp_tt)),
        "n_features": int(n_features),
        "predictor_names": predictor_names,
        "accuracy": accuracy,
        "config": asdict(cfg),
    }

    print("\nBackward mTRF env+onset results")
    print(f"accuracy = {accuracy:.4f}")
    print(f"n_correct = {int(trial_df['correct'].sum())} / {len(trial_df)}")

    out_prefix = out_dir / f"{subject_label}_backward_mtrf_env_onset"
    trial_csv = out_prefix.with_suffix(".trial_results.csv")
    summary_json = out_prefix.with_suffix(".summary.json")
    pred_npz = out_prefix.with_suffix(".predictions.npz")

    trial_df.to_csv(trial_csv, index=False)
    summary_json.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    np.savez_compressed(
        pred_npz,
        y_hat=y_hat_multi.astype(np.float32),
        pred_att=_to_object_trials(pred_att, dtype=np.float32),
        pred_ign=_to_object_trials(pred_ign, dtype=np.float32),
        predictor_names=np.array(predictor_names, dtype=object),
        fs=np.array(fs, dtype=np.float64),
        summary=np.array(json.dumps(_jsonable(summary)), dtype=object),
    )

    print(f"\nSaved trial results: {trial_csv}")
    print(f"Saved summary:       {summary_json}")
    print(f"Saved predictions:   {pred_npz}")

    return {
        "summary": summary,
        "trial_results": trial_df,
        "predictions": y_hat_multi,
        "feature_results": feature_results,
        "paths": {"trial_csv": trial_csv, "summary_json": summary_json, "pred_npz": pred_npz},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train backward Eelbrain mTRF decoder: EEG -> [envelope, onset]")
    parser.add_argument("--input", type=Path, required=True, help="Preprocessed mTRF env+onset .npz")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for model results")
    parser.add_argument("--subject", type=int, default=None)
    parser.add_argument("--tstart", type=float, default=-0.5)
    parser.add_argument("--tstop", type=float, default=0.0)
    parser.add_argument("--basis", type=float, default=0.05)
    parser.add_argument("--basis-window", type=str, default="hamming")
    parser.add_argument("--error", choices=["l1", "l2"], default="l2")
    parser.add_argument("--selective-stopping", type=int, default=1)
    parser.add_argument("--no-scale-data", action="store_true")
    parser.add_argument("--score-mode", choices=["mean", "weighted", "env_only"], default="mean")
    parser.add_argument("--env-weight", type=float, default=1.0)
    parser.add_argument("--onset-weight", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BackwardMTRFConfig(
        tstart=args.tstart,
        tstop=args.tstop,
        basis=args.basis,
        basis_window=args.basis_window,
        error=args.error,
        selective_stopping=args.selective_stopping,
        scale_data=not args.no_scale_data,
        score_mode=args.score_mode,
        feature_weights=(args.env_weight, args.onset_weight),
    )
    run_backward_mtrf_env_onset(args.input, args.out_dir, args.subject, cfg)


if __name__ == "__main__":
    main()
