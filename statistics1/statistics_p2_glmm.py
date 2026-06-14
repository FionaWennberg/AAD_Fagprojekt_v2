import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from pymer4.models import glmer

# --------------------------------------------------
# 1. Path setup
# --------------------------------------------------

PROJECT_ROOT = Path.cwd()

# If you run this from statistics1/, project root is one folder up
if PROJECT_ROOT.name == "statistics1":
    PROJECT_ROOT = PROJECT_ROOT.parent

RESULTS_PATH = PROJECT_ROOT / "statistics1" / "envelope_onsets" / "aad_trial_level_results_env_onset.csv"

# Change this if your participants.tsv is somewhere else
PARTICIPANTS_PATH = PROJECT_ROOT / "participants.tsv"

OUT_DIR = PROJECT_ROOT / "statistics1" / "envelope_onsets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Project root:", PROJECT_ROOT)
print("Results path:", RESULTS_PATH)
print("Participants path:", PARTICIPANTS_PATH)
print("Output folder:", OUT_DIR)


# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

df = pd.read_csv(RESULTS_PATH)

print("\nLoaded data:")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())


# --------------------------------------------------
# 3. Check required columns
# --------------------------------------------------

required_cols = {"subject", "correct"}
missing = required_cols - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.copy()

df["subject"] = df["subject"].astype(str)
df["correct"] = pd.to_numeric(df["correct"], errors="coerce")


# --------------------------------------------------
# 4. Add or repair group_HI
# --------------------------------------------------

has_valid_group_HI = (
    "group_HI" in df.columns
    and df["group_HI"].notna().any()
)

has_valid_hearing_status = (
    "hearing_status" in df.columns
    and df["hearing_status"].notna().any()
)

if has_valid_group_HI:
    print("\nUsing existing group_HI column.")

    df["group_HI"] = pd.to_numeric(df["group_HI"], errors="coerce")

elif has_valid_hearing_status:
    print("\nCreating group_HI from existing hearing_status column.")

    hearing_clean = (
        df["hearing_status"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["group_HI"] = hearing_clean.map({
        "NH": 0,
        "NORMAL": 0,
        "NORMAL HEARING": 0,
        "NORMAL-HEARING": 0,
        "HI": 1,
        "IMPAIRED": 1,
        "HEARING IMPAIRED": 1,
        "HEARING-IMPAIRED": 1,
    })

else:
    print("\nNo usable group_HI/hearing_status found in results CSV.")
    print("Merging hearing status from participants.tsv...")

    if not PARTICIPANTS_PATH.exists():
        raise FileNotFoundError(
            f"Could not find participants.tsv at: {PARTICIPANTS_PATH}\n"
            "Update PARTICIPANTS_PATH to the correct location."
        )

    participants_df = pd.read_csv(PARTICIPANTS_PATH, sep="\t")

    if "participant_id" not in participants_df.columns:
        raise ValueError("participants.tsv must contain a 'participant_id' column.")

    if "hearing_status" not in participants_df.columns:
        raise ValueError("participants.tsv must contain a 'hearing_status' column.")

    participants_df = participants_df[["participant_id", "hearing_status"]].copy()
    participants_df = participants_df.rename(columns={"participant_id": "subject"})

    participants_df["subject"] = participants_df["subject"].astype(str)

    df = df.drop(columns=["hearing_status", "group_HI"], errors="ignore")
    df = df.merge(participants_df, on="subject", how="left")

    hearing_clean = (
        df["hearing_status"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["group_HI"] = hearing_clean.map({
        "NH": 0,
        "NORMAL": 0,
        "NORMAL HEARING": 0,
        "NORMAL-HEARING": 0,
        "HI": 1,
        "IMPAIRED": 1,
        "HEARING IMPAIRED": 1,
        "HEARING-IMPAIRED": 1,
    })


# --------------------------------------------------
# 5. Final cleaning / sanity checks
# --------------------------------------------------

df = df.dropna(subset=["subject", "group_HI", "correct"]).copy()

df["group_HI"] = df["group_HI"].astype(int)
df["correct"] = df["correct"].astype(int)

if df.empty:
    raise ValueError(
        "DataFrame is empty after cleaning. "
        "Most likely group_HI/hearing_status could not be matched correctly."
    )

if not set(df["group_HI"].unique()).issubset({0, 1}):
    raise ValueError("group_HI must contain only 0 (NH) and 1 (HI).")

if not set(df["correct"].unique()).issubset({0, 1}):
    raise ValueError("correct must contain only 0 and 1.")

print("\nDATA OVERVIEW")
print("-------------")
print(f"Total rows/trials: {len(df)}")
print(f"Total subjects: {df['subject'].nunique()}")
print(f"NH subjects: {df.loc[df['group_HI'] == 0, 'subject'].nunique()}")
print(f"HI subjects: {df.loc[df['group_HI'] == 1, 'subject'].nunique()}")

print("\nMean raw accuracy by group:")
print(df.groupby("group_HI")["correct"].mean().rename({0: "NH", 1: "HI"}))

# --------------------------------------------------
# 8. Save outputs for reporting
# --------------------------------------------------
fixed_effects.to_csv("statistics1/envelope_onsets/glmm_fixed_effects.csv", index=False)
pred_df.to_csv("statistics1/envelope_onsets/glmm_predicted_probabilities.csv", index=False)

# Optional: subject-level raw accuracies for descriptive plotting
subject_acc = (
    df.groupby(["subject", "group_HI"], as_index=False)["correct"]
      .mean()
      .rename(columns={"correct": "mean_accuracy"})
)
subject_acc.to_csv("statistics1/envelope_onsets/envons_subject_level_mean_accuracy.csv", index=False)

print("\nSaved:")
print("  glmm_fixed_effects.csv")
print("  glmm_predicted_probabilities.csv")
print("  subject_level_mean_accuracy.csv")