from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd


def subject_id(subject: int | str) -> str:
    """Return BIDS-style subject ID, e.g. 4 -> sub-004."""
    digits = "".join(ch for ch in str(subject) if ch.isdigit())
    return f"sub-{int(digits):03d}"


def processed_path(processed_dir: Path, subject: int | str) -> Path:
    """Path to one subject's preprocessed NPZ."""
    sid = subject_id(subject)
    return processed_dir / f"{sid}_backward_eelbrain.npz"


def subject_results_dir(results_dir: Path, subject: int | str) -> Path:
    """Path to one subject's model-result folder."""
    return results_dir / subject_id(subject)


def find_subjects(bidsdir: Path) -> list[int]:
    """
    Find available subjects from participants.tsv if possible,
    otherwise from sub-* folders.
    """
    participants_path = bidsdir / "participants.tsv"

    if participants_path.exists():
        participants = pd.read_csv(participants_path, sep="\t")
        if "participant_id" in participants.columns:
            subjects = []
            for pid in participants["participant_id"]:
                digits = "".join(ch for ch in str(pid) if ch.isdigit())
                if digits:
                    subjects.append(int(digits))
            return sorted(subjects)

    subjects = []
    for path in bidsdir.glob("sub-*"):
        if path.is_dir():
            digits = "".join(ch for ch in path.name if ch.isdigit())
            if digits:
                subjects.append(int(digits))

    return sorted(subjects)


def load_npz_json_field(obj: np.lib.npyio.NpzFile, key: str):
    """Load JSON field saved as scalar object array/string."""
    if key not in obj:
        return None

    value = obj[key]

    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()

    if isinstance(value, bytes):
        value = value.decode("utf-8")

    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    return value


def load_processed_subject(path: Path) -> dict:
    """
    Load one preprocessed subject NPZ into a simple dictionary.

    This does not stack trials or build Eelbrain objects.
    That should happen in model.py.
    """
    if not path.exists():
        raise FileNotFoundError(f"Processed file does not exist: {path}")

    obj = np.load(path, allow_pickle=True)

    return {
        "path": path,
        "meta": load_npz_json_field(obj, "meta"),
        "trial_meta": load_npz_json_field(obj, "trial_meta"),
        "fs_stim": float(np.asarray(obj["fs_stim"]).item()) if "fs_stim" in obj else None,
        "fs_resp": float(np.asarray(obj["fs_resp"]).item()) if "fs_resp" in obj else None,
        "keys": list(obj.keys()),
        "npz": obj,
    }


def ensure_dirs(*dirs: Path) -> None:
    """Create folders if they do not already exist."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)