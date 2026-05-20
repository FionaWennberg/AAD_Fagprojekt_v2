from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.aad_project.data import ensure_dirs, load_npz_json_field


def parse_args():
    parser = argparse.ArgumentParser(description="Create gamma Eelbrain input NPZ")

    parser.add_argument("--input", type=Path, required=True,
                        help="Existing envelope preprocessed NPZ")
    parser.add_argument("--spectrogram-dir", type=Path, required=True,
                        help="Directory with *_spectrogram.npz files")
    parser.add_argument("--outdir", type=Path, required=True)

    parser.add_argument("--crop-start-sec", type=float, default=6.0)

    return parser.parse_args()


def load_spectrogram(path: Path):
    obj = np.load(path)
    return (
        np.asarray(obj["spectrogram"], dtype=np.float64),  # time x bands
        np.asarray(obj["center_frequencies"], dtype=np.float64),
        float(np.asarray(obj["fs"]).item()),
    )


def feature_path(spectrogram_dir: Path, stim_file_event: str) -> Path:
    rel = Path(stim_file_event)
    return (spectrogram_dir / rel).with_name(rel.stem + "_spectrogram.npz")


def crop_target(spec: np.ndarray, fs: float, crop_start_sec: float, n_time: int) -> np.ndarray:
    start = int(round(crop_start_sec * fs))
    end = start + n_time
    return crop_or_pad(spec[start:end], n_time)


def crop_masker(
    spec: np.ndarray,
    fs: float,
    crop_start_sec: float,
    n_time: int,
    masker_delay_sec: float,
) -> np.ndarray:
    delay = int(round(masker_delay_sec * fs))
    padded = np.vstack([
        np.zeros((delay, spec.shape[1]), dtype=spec.dtype),
        spec,
    ])

    start = int(round(crop_start_sec * fs))
    end = start + n_time
    return crop_or_pad(padded[start:end], n_time)


def crop_or_pad(x: np.ndarray, n_time: int) -> np.ndarray:
    if x.shape[0] == n_time:
        return x
    if x.shape[0] > n_time:
        return x[:n_time]

    pad = np.zeros((n_time - x.shape[0], x.shape[1]), dtype=x.dtype)
    return np.vstack([x, pad])


def main():
    args = parse_args()

    obj = np.load(args.input, allow_pickle=True)

    meta = load_npz_json_field(obj, "meta")
    trial_meta_all = load_npz_json_field(obj, "trial_meta")

    resp_tt = obj["resp_tt"]
    fs_resp = float(np.asarray(obj["fs_resp"]).item())

    two_talker_meta = [
        tm for tm in trial_meta_all
        if tm.get("trial_kind") == "twotalker"
    ]

    if len(two_talker_meta) != len(resp_tt):
        raise ValueError(
            f"Expected trial_meta twotalker count to match resp_tt. "
            f"Got {len(two_talker_meta)} metadata trials and {len(resp_tt)} EEG trials."
        )

    stim_att_gamma = []
    stim_ign_gamma = []

    center_frequencies = None
    fs_stim = None

    for i, tm in enumerate(two_talker_meta):
        target_event = tm["stim_file_target_event"]
        masker_event = tm["stim_file_masker_event"]
        masker_delay_sec = float(tm["masker_delay_sec"])

        target_path = feature_path(args.spectrogram_dir, target_event)
        masker_path = feature_path(args.spectrogram_dir, masker_event)

        if not target_path.exists():
            raise FileNotFoundError(f"Missing target spectrogram: {target_path}")
        if not masker_path.exists():
            raise FileNotFoundError(f"Missing masker spectrogram: {masker_path}")

        target_spec, centers, fs = load_spectrogram(target_path)
        masker_spec, centers_m, fs_m = load_spectrogram(masker_path)

        if center_frequencies is None:
            center_frequencies = centers
            fs_stim = fs
        else:
            if not np.allclose(center_frequencies, centers):
                raise ValueError("Center frequencies differ across target files")
            if not np.isclose(fs_stim, fs):
                raise ValueError("Sampling rates differ across target files")

        if not np.allclose(centers, centers_m):
            raise ValueError(f"Target/masker center frequencies differ in trial {i}")
        if not np.isclose(fs, fs_m):
            raise ValueError(f"Target/masker sampling rates differ in trial {i}")

        n_time = np.asarray(resp_tt[i]).shape[0]

        target_crop = crop_target(
            target_spec,
            fs,
            args.crop_start_sec,
            n_time,
        )

        masker_crop = crop_masker(
            masker_spec,
            fs,
            args.crop_start_sec,
            n_time,
            masker_delay_sec,
        )

        stim_att_gamma.append(target_crop)
        stim_ign_gamma.append(masker_crop)

    stim_att_gamma = np.stack(stim_att_gamma, axis=0)
    stim_ign_gamma = np.stack(stim_ign_gamma, axis=0)

    if not np.isclose(fs_stim, fs_resp):
        raise ValueError(f"Stimulus and EEG fs differ: {fs_stim} vs {fs_resp}")

    ensure_dirs(args.outdir)

    out_path = args.outdir / f"{args.input.stem}_gamma_eelbrain.npz"

    np.savez_compressed(
        out_path,
        stim_att_gamma=stim_att_gamma.astype(np.float32),
        stim_ign_gamma=stim_ign_gamma.astype(np.float32),
        resp_tt=resp_tt,
        fs_stim=np.array(fs_stim, dtype=np.float64),
        fs_resp=np.array(fs_resp, dtype=np.float64),
        center_frequencies=center_frequencies.astype(np.float32),
        meta=json.dumps(meta),
        trial_meta=json.dumps(two_talker_meta),
    )

    print("Saved:", out_path)
    print("stim_att_gamma:", stim_att_gamma.shape)
    print("stim_ign_gamma:", stim_ign_gamma.shape)
    print("resp_tt:", np.asarray(resp_tt).shape)


if __name__ == "__main__":
    main()