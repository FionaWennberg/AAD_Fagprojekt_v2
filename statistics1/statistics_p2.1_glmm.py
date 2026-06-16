import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from pymer4.models import glmer


# Envelope + Onsets pipeline
# Robust GLMM version:
#   1. Full model: HI vs NH
#   2. NH-only intercept model: NH vs chance
#   3. HI-only intercept model: HI vs chance


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
    raise ValueError("group_HI must contain only 0 for NH and 1 for HI.")

if not set(df["correct"].unique()).issubset({0, 1}):
    raise ValueError("correct must contain only 0 and 1.")

print("\nDATA OVERVIEW")
print("-------------")
print(f"Total rows/trials: {len(df)}")
print(f"Total subjects: {df['subject'].nunique()}")
print(f"NH subjects: {df.loc[df['group_HI'] == 0, 'subject'].nunique()}")
print(f"HI subjects: {df.loc[df['group_HI'] == 1, 'subject'].nunique()}")

print("\nMean raw accuracy by group:")
raw_group_acc = (
    df.groupby("group_HI")["correct"]
      .mean()
      .rename(index={0: "NH", 1: "HI"})
)
print(raw_group_acc)

print("\nRaw counts by group:")
raw_counts = (
    df.groupby("group_HI")["correct"]
      .agg(["sum", "count", "mean"])
      .rename(index={0: "NH", 1: "HI"})
      .rename(columns={"sum": "n_correct", "count": "n_total", "mean": "raw_accuracy"})
)
print(raw_counts)


# --------------------------------------------------
# 6. Helper functions
# --------------------------------------------------

def to_pandas_table(table):
    """Convert pymer4 output to pandas if needed."""
    if hasattr(table, "to_pandas"):
        return table.to_pandas()
    return pd.DataFrame(table)


def find_col(df_table, possible_names=None, contains=None):
    """Find a column robustly across slightly different pymer4 versions."""
    possible_names = possible_names or set()

    for c in df_table.columns:
        c_lower = str(c).lower()

        if c_lower in possible_names:
            return c

        if contains is not None and contains in c_lower:
            return c

    return None


def extract_model_columns(fixed_effects):
    """Identify important columns in the pymer4 fixed-effects table."""
    term_col = find_col(
        fixed_effects,
        possible_names={"term", "effect", "name", "predictor"},
    )

    estimate_col = find_col(
        fixed_effects,
        possible_names={"estimate", "b", "beta"},
    )

    se_col = find_col(
        fixed_effects,
        possible_names={"se", "std.error", "std_error"},
    )

    z_col = find_col(
        fixed_effects,
        possible_names={"z", "z-stat", "z_stat", "z.value", "z-value"},
    )
    if z_col is None:
        z_col = find_col(fixed_effects, contains="z")

    p_col = find_col(
        fixed_effects,
        possible_names={"p", "p-val", "pvalue", "p_value", "p.value"},
    )

    ci_low_col = find_col(fixed_effects, contains="ci-low")
    if ci_low_col is None:
        ci_low_col = find_col(fixed_effects, contains="2.5")

    ci_high_col = find_col(fixed_effects, contains="ci-high")
    if ci_high_col is None:
        ci_high_col = find_col(fixed_effects, contains="97.5")

    if term_col is None:
        raise ValueError(
            "Could not find term/effect column in fixed-effects table. "
            f"Available columns are: {list(fixed_effects.columns)}"
        )

    if estimate_col is None:
        raise ValueError(
            "Could not find estimate column in fixed-effects table. "
            f"Available columns are: {list(fixed_effects.columns)}"
        )

    return term_col, estimate_col, se_col, z_col, p_col, ci_low_col, ci_high_col


def extract_term(fixed_effects, possible_term_names):
    """
    Extract estimate, SE, z, p, odds ratio, probability and CIs
    for one model term.
    """
    fixed_effects = to_pandas_table(fixed_effects)

    (
        term_col,
        estimate_col,
        se_col,
        z_col,
        p_col,
        ci_low_col,
        ci_high_col,
    ) = extract_model_columns(fixed_effects)

    if isinstance(possible_term_names, str):
        possible_term_names = [possible_term_names]

    term_values = fixed_effects[term_col].astype(str).tolist()

    row_df = fixed_effects[
        fixed_effects[term_col].astype(str).isin(possible_term_names)
    ]

    if len(row_df) != 1:
        raise ValueError(
            f"Could not uniquely identify term from {possible_term_names}. "
            f"Available terms are: {term_values}"
        )

    row = row_df.iloc[0]

    estimate = float(row[estimate_col])
    odds_ratio = np.exp(estimate)
    probability = odds_ratio / (1 + odds_ratio)

    output = {
        "estimate_log_odds": estimate,
        "odds_ratio": odds_ratio,
        "predicted_probability": probability,
    }

    output["SE"] = float(row[se_col]) if se_col is not None else np.nan
    output["z"] = float(row[z_col]) if z_col is not None else np.nan
    output["p_value"] = row[p_col] if p_col is not None else np.nan

    if ci_low_col is not None and ci_high_col is not None:
        ci_low = float(row[ci_low_col])
        ci_high = float(row[ci_high_col])

        output["ci_low_log_odds"] = ci_low
        output["ci_high_log_odds"] = ci_high
        output["ci_low_odds_ratio"] = np.exp(ci_low)
        output["ci_high_odds_ratio"] = np.exp(ci_high)
    else:
        output["ci_low_log_odds"] = np.nan
        output["ci_high_log_odds"] = np.nan
        output["ci_low_odds_ratio"] = np.nan
        output["ci_high_odds_ratio"] = np.nan

    return output


def add_wald_ci(row):
    """
    If pymer4 does not return confidence intervals, compute approximate
    Wald intervals using estimate +/- 1.96 * SE.
    """
    estimate = row.get("estimate_log_odds", np.nan)
    se = row.get("SE", np.nan)

    if pd.isna(row.get("ci_low_log_odds", np.nan)) and pd.notna(estimate) and pd.notna(se):
        row["ci_low_log_odds"] = estimate - 1.96 * se
        row["ci_high_log_odds"] = estimate + 1.96 * se
        row["ci_low_odds_ratio"] = np.exp(row["ci_low_log_odds"])
        row["ci_high_odds_ratio"] = np.exp(row["ci_high_log_odds"])

    return row


def format_p_value(p):
    """Pretty formatting for p-values."""
    try:
        p_float = float(p)
    except Exception:
        return str(p)

    if p_float < 0.001:
        return "< .001"

    return f"{p_float:.3f}"


def add_diagnostics(row):
    """
    Add flags for suspicious values.
    This does not change the results, but helps catch unstable model output.
    """
    warnings = []

    se = row.get("SE", np.nan)
    z = row.get("z", np.nan)
    estimate = row.get("estimate_log_odds", np.nan)

    if pd.notna(se) and se < 0.01:
        warnings.append("Very small SE; check model stability")

    if pd.notna(z) and abs(z) > 50:
        warnings.append("Very large z; check model stability")

    if pd.notna(estimate) and abs(estimate) > 10:
        warnings.append("Very large log-odds estimate; possible separation")

    if not warnings:
        return "OK"

    return "; ".join(warnings)


def fit_glmm(formula, data, model_name):
    """Fit a binomial GLMM and return model plus fixed-effects table."""
    print(f"\nFitting {model_name}")
    print("-" * (8 + len(model_name)))

    model = glmer(
        formula,
        data=pl.from_pandas(data),
        family="binomial",
    )

    model.fit(exponentiate=False, summary=True)

    fixed_effects = to_pandas_table(model.result_fit)

    print(f"\nMODEL SUMMARY: {model_name}")
    print("-" * (15 + len(model_name)))
    print(fixed_effects)

    print("\nColumns returned by pymer4:")
    print(list(fixed_effects.columns))

    return model, fixed_effects


# --------------------------------------------------
# 7. Create group-specific datasets
# --------------------------------------------------

df_nh = df[df["group_HI"] == 0].copy()
df_hi = df[df["group_HI"] == 1].copy()

if df_nh["subject"].nunique() < 2:
    raise ValueError("NH group has fewer than 2 subjects. GLMM may not be meaningful.")

if df_hi["subject"].nunique() < 2:
    raise ValueError("HI group has fewer than 2 subjects. GLMM may not be meaningful.")


# --------------------------------------------------
# 8. Fit full model for HI vs NH
# --------------------------------------------------
# This tests whether decoding accuracy differs between groups.

model_full, fixed_effects_full = fit_glmm(
    formula="correct ~ group_HI + (1 | subject)",
    data=df,
    model_name="full model: HI vs NH",
)


# --------------------------------------------------
# 9. Fit NH-only model for NH vs chance
# --------------------------------------------------
# This tests whether NH decoding accuracy is above 50% chance.
# In a logistic model, chance corresponds to log-odds = 0.

model_nh_chance, fixed_effects_nh_chance = fit_glmm(
    formula="correct ~ 1 + (1 | subject)",
    data=df_nh,
    model_name="NH-only model: NH vs chance",
)


# --------------------------------------------------
# 10. Fit HI-only model for HI vs chance
# --------------------------------------------------
# This tests whether HI decoding accuracy is above 50% chance.
# In a logistic model, chance corresponds to log-odds = 0.

model_hi_chance, fixed_effects_hi_chance = fit_glmm(
    formula="correct ~ 1 + (1 | subject)",
    data=df_hi,
    model_name="HI-only model: HI vs chance",
)


# --------------------------------------------------
# 11. Extract report rows
# --------------------------------------------------

intercept_names = ["(Intercept)", "Intercept"]
group_names = ["group_HI"]

nh_vs_chance = extract_term(fixed_effects_nh_chance, intercept_names)
hi_vs_chance = extract_term(fixed_effects_hi_chance, intercept_names)
hi_vs_nh = extract_term(fixed_effects_full, group_names)

report_rows = [
    {
        "comparison": "NH vs chance",
        "model_used": "NH-only GLMM",
        "term_used": "Intercept",
        "interpretation": "NH decoding accuracy compared with 50% chance",
        **nh_vs_chance,
    },
    {
        "comparison": "HI vs chance",
        "model_used": "HI-only GLMM",
        "term_used": "Intercept",
        "interpretation": "HI decoding accuracy compared with 50% chance",
        **hi_vs_chance,
    },
    {
        "comparison": "HI vs NH",
        "model_used": "Full GLMM",
        "term_used": "group_HI",
        "interpretation": "Difference in decoding accuracy between HI and NH",
        **hi_vs_nh,
    },
]

report_table = pd.DataFrame(report_rows)

# For a group contrast, logistic(estimate) is not a group probability.
# Therefore, remove predicted_probability for HI vs NH.
report_table.loc[
    report_table["comparison"] == "HI vs NH",
    "predicted_probability"
] = np.nan

# Add approximate Wald confidence intervals if pymer4 did not return CIs
report_table = report_table.apply(add_wald_ci, axis=1)

report_table["p_value_formatted"] = report_table["p_value"].apply(format_p_value)
report_table["diagnostic"] = report_table.apply(add_diagnostics, axis=1)

report_table_rounded = report_table.copy()
numeric_cols = report_table_rounded.select_dtypes(include=[np.number]).columns
report_table_rounded[numeric_cols] = report_table_rounded[numeric_cols].round(4)

print("\nGLMM REPORT TABLE")
print("-----------------")
print(report_table_rounded)


# --------------------------------------------------
# 12. Predicted probabilities from full model
# --------------------------------------------------

pred_df = model_full.empredict({"group_HI": [0, 1]})
pred_df = to_pandas_table(pred_df)

if "group_HI" in pred_df.columns:
    pred_df["group"] = pred_df["group_HI"].map({0: "NH", 1: "HI"})

print("\nPREDICTED PROBABILITIES FROM FULL MODEL")
print("---------------------------------------")
print(pred_df)


# --------------------------------------------------
# 13. Subject-level raw accuracies
# --------------------------------------------------

subject_acc = (
    df.groupby(["subject", "group_HI"], as_index=False)["correct"]
      .mean()
      .rename(columns={"correct": "mean_accuracy"})
)

subject_acc["group"] = subject_acc["group_HI"].map({0: "NH", 1: "HI"})

print("\nSUBJECT-LEVEL MEAN ACCURACY")
print("---------------------------")
print(subject_acc.head())


# --------------------------------------------------
# 14. Save outputs for reporting
# --------------------------------------------------

fixed_effects_full.to_csv(
    OUT_DIR / "envons_glmm_fixed_effects_full_HI_vs_NH.csv",
    index=False,
)

fixed_effects_nh_chance.to_csv(
    OUT_DIR / "envons_glmm_fixed_effects_NH_vs_chance.csv",
    index=False,
)

fixed_effects_hi_chance.to_csv(
    OUT_DIR / "envons_glmm_fixed_effects_HI_vs_chance.csv",
    index=False,
)

report_table.to_csv(
    OUT_DIR / "envons_glmm_report_table.csv",
    index=False,
)

report_table_rounded.to_csv(
    OUT_DIR / "envons_glmm_report_table_rounded.csv",
    index=False,
)

pred_df.to_csv(
    OUT_DIR / "envons_glmm_predicted_probabilities_full_model.csv",
    index=False,
)

subject_acc.to_csv(
    OUT_DIR / "envons_subject_level_mean_accuracy.csv",
    index=False,
)

raw_group_acc.to_csv(
    OUT_DIR / "envons_raw_group_accuracy.csv",
    header=["mean_accuracy"],
)

raw_counts.to_csv(
    OUT_DIR / "envons_raw_group_counts.csv",
)

print("\nSaved:")
print(OUT_DIR / "envons_glmm_fixed_effects_full_HI_vs_NH.csv")
print(OUT_DIR / "envons_glmm_fixed_effects_NH_vs_chance.csv")
print(OUT_DIR / "envons_glmm_fixed_effects_HI_vs_chance.csv")
print(OUT_DIR / "envons_glmm_report_table.csv")
print(OUT_DIR / "envons_glmm_report_table_rounded.csv")
print(OUT_DIR / "envons_glmm_predicted_probabilities_full_model.csv")
print(OUT_DIR / "envons_subject_level_mean_accuracy.csv")
print(OUT_DIR / "envons_raw_group_accuracy.csv")
print(OUT_DIR / "envons_raw_group_counts.csv")