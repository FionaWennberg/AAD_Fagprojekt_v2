from pathlib import Path
import argparse
import json
import pandas as pd


def normalize_subject_id(x):
    """Return subject id in sub-XXX format when possible."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.startswith("sub-"):
        return s
    if s.isdigit():
        return f"sub-{int(s):03d}"
    return s


def load_participants(participants_tsv):
    """Load participant metadata if a participants.tsv path is provided."""
    if participants_tsv is None:
        return None

    participants_tsv = Path(participants_tsv)
    if not participants_tsv.exists():
        raise FileNotFoundError(f"participants.tsv not found: {participants_tsv}")

    participants = pd.read_csv(participants_tsv, sep="\t")
    if "participant_id" not in participants.columns:
        raise ValueError("participants.tsv must contain a 'participant_id' column.")

    participants["subject"] = participants["participant_id"].apply(normalize_subject_id)
    return participants


def get_hearing_status(subject, summary_data, participants):
    """Try to obtain hearing status from summary JSON, then participants.tsv."""
    # Some pipeline versions may eventually store it directly
    for key in ["hearing_status", "group", "listener_group"]:
        if key in summary_data and summary_data[key] is not None:
            return str(summary_data[key]).strip()

    meta = summary_data.get("meta", {})
    if isinstance(meta, dict):
        for key in ["hearing_status", "group", "listener_group"]:
            if key in meta and meta[key] is not None:
                return str(meta[key]).strip()

    if participants is not None:
        row = participants.loc[participants["subject"] == subject]
        if len(row) == 1 and "hearing_status" in row.columns:
            return str(row.iloc[0]["hearing_status"]).strip()

    return None


def group_hi_from_status(hearing_status):
    """Encode hearing group as 0 = NH, 1 = HI, if known."""
    if hearing_status is None or pd.isna(hearing_status):
        return None

    s = str(hearing_status).strip().upper()

    if s in {"HI", "HEARING IMPAIRED", "HEARING-IMPAIRED", "IMPAIRED"}:
        return 1
    if s in {"NH", "NORMAL HEARING", "NORMAL-HEARING", "NORMAL"}:
        return 0

    return None


def combine_env_onset_results(results_dir, out_path, participants_tsv=None):
    results_dir = Path(results_dir)
    out_path = Path(out_path)

    participants = load_participants(participants_tsv)

    summary_paths = sorted(
        results_dir.glob("sub-*/sub-*_backward_mtrf_env_onset.summary.json")
    )

    if len(summary_paths) == 0:
        raise FileNotFoundError(
            f"No env+onset summary files found in {results_dir}. "
            "Expected pattern: sub-*/sub-*_backward_mtrf_env_onset.summary.json"
        )

    rows = []

    for summary_path in summary_paths:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        subject = normalize_subject_id(summary_data.get("subject"))
        if subject is None:
            # Fallback from filename
            subject = summary_path.parent.name

        accuracy = summary_data.get("accuracy")
        n_trials_summary = summary_data.get("n_trials")
        n_features = summary_data.get("n_features")
        predictor_names = summary_data.get("predictor_names")
        input_path = summary_data.get("input_path")
        config = summary_data.get("config", {})

        hearing_status = get_hearing_status(subject, summary_data, participants)
        group_HI = group_hi_from_status(hearing_status)

        trial_csv = summary_path.with_name(
            summary_path.name.replace(".summary.json", ".trial_results.csv")
        )

        if not trial_csv.exists():
            print(f"Warning: missing trial CSV for {subject}: {trial_csv}")
            continue

        trial_df = pd.read_csv(trial_csv)

        required_cols = {
            "subject",
            "trial",
            "score",
            "correct",
            "r_att_envelope",
            "r_ign_envelope",
            "diff_envelope",
            "r_att_onset",
            "r_ign_onset",
            "diff_onset",
        }

        missing = required_cols - set(trial_df.columns)
        if missing:
            raise ValueError(
                f"Missing columns in {trial_csv}: {sorted(missing)}"
            )

        for _, trial in trial_df.iterrows():
            # score appears to be the combined decision score.
            # From the available columns, it matches approximately:
            # diff_envelope + diff_onset.
            rows.append({
                "subject": normalize_subject_id(trial["subject"]),
                "hearing_status": hearing_status,
                "group_HI": group_HI,

                "trial_index": int(trial["trial"]),
                "correct": int(trial["correct"]),

                # Combined model score
                "score": float(trial["score"]),

                # Feature-specific correlations
                "r_att_envelope": float(trial["r_att_envelope"]),
                "r_ign_envelope": float(trial["r_ign_envelope"]),
                "diff_envelope": float(trial["diff_envelope"]),
                "r_att_onset": float(trial["r_att_onset"]),
                "r_ign_onset": float(trial["r_ign_onset"]),
                "diff_onset": float(trial["diff_onset"]),

                # Combined difference, useful for direct plotting/statistics
                "r_diff_combined": float(trial["score"]),

                # Subject-level summary repeated on each trial row
                "subject_decoding_accuracy": accuracy,
                "subject_n_trials": n_trials_summary,
                "subject_n_features": n_features,
                "predictor_names": ",".join(predictor_names) if isinstance(predictor_names, list) else predictor_names,
                "input_path": input_path,

                # Config
                "tstart": config.get("tstart"),
                "tstop": config.get("tstop"),
                "basis": config.get("basis"),
                "basis_window": config.get("basis_window"),
                "test": config.get("test"),
                "partitions": config.get("partitions"),
                "error": config.get("error"),
                "selective_stopping": config.get("selective_stopping"),
                "scale_data": config.get("scale_data"),
                "score_mode": config.get("score_mode"),
                "feature_weights": ",".join(map(str, config.get("feature_weights", [])))
                    if isinstance(config.get("feature_weights"), list)
                    else config.get("feature_weights"),

                "summary_path": str(summary_path),
                "trial_csv_path": str(trial_csv),
            })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError("No rows were combined. Check whether trial CSV files exist.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print()
    print(df.head())
    print()
    print("Number of subjects:", df["subject"].nunique())
    print("Number of rows/trials:", len(df))
    print()
    print("Subjects by hearing status:")
    if "hearing_status" in df.columns:
        print(df.groupby("hearing_status", dropna=False)["subject"].nunique())

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Combine envelope+onset mTRF results into one trial-level CSV."
    )
    parser.add_argument(
        "--results-dir",
        default="results_env_onset",
        help="Folder containing sub-XXX result folders.",
    )
    parser.add_argument(
        "--out-path",
        default="statistics1/envelope_onsets/aad_trial_level_results_env_onset.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--participants-tsv",
        default=None,
        help="Optional path to participants.tsv for hearing_status.",
    )

    args = parser.parse_args()

    combine_env_onset_results(
        results_dir=args.results_dir,
        out_path=args.out_path,
        participants_tsv=args.participants_tsv,
    )


if __name__ == "__main__":
    main()
