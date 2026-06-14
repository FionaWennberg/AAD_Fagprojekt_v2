import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from pymer4.models import glmer

# Envelope + Onsets pipeline

# --------------------------------------------------
# 1. Path setup
# --------------------------------------------------

PROJECT_ROOT = Path.cwd()

# If you run this from statistics1/, project root is one folder up
if PROJECT_ROOT.name == "statistics1":
    PROJECT_ROOT = PROJECT_ROOT.parent

RESULTS_PATH = (
    PROJECT_ROOT
    / "statistics1"
    / "envelope_onsets"
    / "aad_trial_level_results_env_onset.csv"
)

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
# 6. Convert pandas DataFrame to Polars for pymer4
# --------------------------------------------------

df_r = pl.from_pandas(df)


# --------------------------------------------------
# 7. Fit mixed-effects logistic regression
# --------------------------------------------------

print("\nFitting GLMM")
print("------------")

model = glmer(
    "correct ~ group_HI + (1 | subject)",
    data=df_r,
    family="binomial",
)

model.fit(exponentiate=False, summary=True)

print("\nMODEL SUMMARY")
print("-------------")
print(model.result_fit)


# --------------------------------------------------
# 8. Extract fixed effects table
# --------------------------------------------------

fixed_effects = model.result_fit

if hasattr(fixed_effects, "to_pandas"):
    fixed_effects = fixed_effects.to_pandas()
else:
    fixed_effects = pd.DataFrame(fixed_effects)

print("\nFIXED EFFECTS TABLE")
print("-------------------")
print(fixed_effects)


# --------------------------------------------------
# 9. Compute odds ratio for the group effect
# --------------------------------------------------

term_col = None
for c in fixed_effects.columns:
    if str(c).lower() in {"term", "effect", "name", "predictor"}:
        term_col = c
        break

if term_col is None:
    raise ValueError(
        "Could not find the term/effect column in model.result_fit. "
        f"Available columns are: {list(fixed_effects.columns)}"
    )

group_row = fixed_effects[fixed_effects[term_col].astype(str) == "group_HI"]

if len(group_row) != 1:
    raise ValueError(
        "Could not uniquely identify the group_HI row in fixed effects table. "
        f"Available terms are: {fixed_effects[term_col].astype(str).tolist()}"
    )

group_row = group_row.iloc[0]

estimate_col = next(
    (c for c in fixed_effects.columns if str(c).lower() in {"estimate", "b", "beta"}),
    None,
)

se_col = next(
    (c for c in fixed_effects.columns if str(c).lower() in {"se", "std.error", "std_error"}),
    None,
)

z_col = next(
    (c for c in fixed_effects.columns if "z" in str(c).lower()),
    None,
)

p_col = next(
    (c for c in fixed_effects.columns if str(c).lower() in {"p", "p-val", "pvalue", "p_value", "p.value"}),
    None,
)

ci_low_col = next(
    (c for c in fixed_effects.columns if "ci-low" in str(c).lower() or "2.5" in str(c).lower()),
    None,
)

ci_high_col = next(
    (c for c in fixed_effects.columns if "ci-high" in str(c).lower() or "97.5" in str(c).lower()),
    None,
)

if estimate_col is None:
    raise ValueError(
        "Could not find estimate column in fixed effects table. "
        f"Available columns are: {list(fixed_effects.columns)}"
    )

beta = float(group_row[estimate_col])
odds_ratio = np.exp(beta)

print("\nPRIMARY GROUP EFFECT")
print("--------------------")
print(f"log-odds coefficient for HI vs NH: {beta:.4f}")
print(f"odds ratio for HI vs NH: {odds_ratio:.4f}")

if se_col is not None:
    print(f"SE: {float(group_row[se_col]):.4f}")

if z_col is not None:
    print(f"z-statistic: {float(group_row[z_col]):.4f}")

if p_col is not None:
    print(f"p-value: {group_row[p_col]}")

if ci_low_col is not None and ci_high_col is not None:
    ci_low = float(group_row[ci_low_col])
    ci_high = float(group_row[ci_high_col])
    print(f"95% CI (log-odds): [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"95% CI (odds ratio): [{np.exp(ci_low):.4f}, {np.exp(ci_high):.4f}]")


# --------------------------------------------------
# 10. Predicted probabilities for NH and HI
# --------------------------------------------------

pred_df = model.empredict({"group_HI": [0, 1]})

if hasattr(pred_df, "to_pandas"):
    pred_df = pred_df.to_pandas()
else:
    pred_df = pd.DataFrame(pred_df)

print("\nPREDICTED PROBABILITIES")
print("-----------------------")
print(pred_df)


# --------------------------------------------------
# 11. Save outputs for reporting
# --------------------------------------------------

fixed_effects.to_csv(
    OUT_DIR / "envons_glmm_fixed_effects.csv",
    index=False,
)

pred_df.to_csv(
    OUT_DIR / "envons_glmm_predicted_probabilities.csv",
    index=False,
)

subject_acc = (
    df.groupby(["subject", "group_HI"], as_index=False)["correct"]
      .mean()
      .rename(columns={"correct": "mean_accuracy"})
)

subject_acc.to_csv(
    OUT_DIR / "envons_subject_level_mean_accuracy.csv",
    index=False,
)

print("\nSaved:")
print(OUT_DIR / "envons_glmm_fixed_effects.csv")
print(OUT_DIR / "envons_glmm_predicted_probabilities.csv")
print(OUT_DIR / "envons_subject_level_mean_accuracy.csv")