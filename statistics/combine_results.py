from pathlib import Path
import json
import pandas as pd

# Folder containing one sub-xxx folder per subject
results_dir = Path("results_baseline_all")

# Output file
out_path = Path("statistics") / "aad_trial_level_results.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)

rows = []

summary_paths = sorted(
    results_dir.glob("sub-*/sub-*_backward_eelbrain_backward_summary.json")
)

if len(summary_paths) == 0:
    raise FileNotFoundError(
        f"No summary files found in {results_dir}. "
        "Check that the folder name and file pattern are correct."
    )

for summary_path in summary_paths:
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("meta", {})
    eeg_config = meta.get("eeg_config", {})
    summary = data.get("summary_two_talker", {})

    subject = meta.get("subject")
    hearing_status = meta.get("hearing_status")

    if subject is None:
        raise ValueError(f"Missing subject in {summary_path}")

    if hearing_status is None:
        raise ValueError(f"Missing hearing_status in {summary_path}")

    # Numeric group variable for statistics:
    # 0 = normal hearing, 1 = hearing impaired
    group_HI = 1 if hearing_status.lower() == "hi" else 0

    # This describes which EEG channels were used in this model output.
    # If True, the current result is based on scalp EEG only.
    scalp_only = eeg_config.get("scalp_only", None)

    decoding_accuracy = summary.get("decoding_accuracy")
    n_trials = summary.get("n_trials")
    n_correct = summary.get("n_correct")

    trial_results = data.get("trial_results_two_talker", [])

    if len(trial_results) == 0:
        print(f"Warning: no two-talker trial results found for {subject}")

    for trial in trial_results:
        r_att = trial["r_att"]
        r_ign = trial["r_ign"]

        rows.append({
            "subject": subject,
            "hearing_status": hearing_status,
            "group_HI": group_HI,
            "scalp_only": scalp_only,
            "trial_index": trial["trial_index"],
            "r_att": r_att,
            "r_ign": r_ign,
            "r_diff": r_att - r_ign,
            "correct": trial["correct"],
            "subject_decoding_accuracy": decoding_accuracy,
            "subject_n_trials": n_trials,
            "subject_n_correct": n_correct,
            "summary_path": str(summary_path),
        })

df = pd.DataFrame(rows)

df.to_csv(out_path, index=False)

print("Saved:", out_path)
print()
print(df.head())
print()
print("Number of subjects:", df["subject"].nunique())
print()
print("Subjects by hearing status:")
print(df.groupby("hearing_status")["subject"].nunique())
print()
print("Subjects by scalp_only setting:")
print(df.groupby("scalp_only")["subject"].nunique())
print()
print("Number of rows/trials:", len(df))