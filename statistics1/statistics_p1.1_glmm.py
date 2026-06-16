import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from pymer4.models import glmer


# --------------------------------------------------
# 1. Path setup
# --------------------------------------------------

INPUT_FILE = Path("statistics1") / "aad_trial_level_results.csv"

OUTPUT_DIR = Path("statistics1") / "envelope_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

# Expected columns:
#   subject   : subject ID
#   group_HI  : 0 = NH, 1 = HI
#   correct   : 0 = wrong classification, 1 = correct classification

required_cols = {"subject", "group_HI", "correct"}
missing = required_cols - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {missing}")


# --------------------------------------------------
# 3. Basic cleaning / type checks
# --------------------------------------------------

df = df.copy()

df = df.dropna(subset=["subject", "group_HI", "correct"])

df["subject"] = df["subject"].astype(str)
df["group_HI"] = df["group_HI"].astype(int)
df["correct"] = df["correct"].astype(int)

if not set(df["group_HI"].unique()).issubset({0, 1}):
    raise ValueError("group_HI must contain only 0 for NH and 1 for HI.")

if not set(df["correct"].unique()).issubset({0, 1}):
    raise ValueError("correct must contain only 0 and 1.")


# --------------------------------------------------
# 4. Quick data overview
# --------------------------------------------------

n_subjects_total = df["subject"].nunique()
n_subjects_nh = df.loc[df["group_HI"] == 0, "subject"].nunique()
n_subjects_hi = df.loc[df["group_HI"] == 1, "subject"].nunique()

print("\nDATA OVERVIEW")
print("-------------")
print(f"Input file: {INPUT_FILE}")
print(f"Output folder: {OUTPUT_DIR}")
print(f"Total rows: {len(df)}")
print(f"Total subjects: {n_subjects_total}")
print(f"NH subjects: {n_subjects_nh}")
print(f"HI subjects: {n_subjects_hi}")

print("\nMean raw accuracy by group:")
raw_group_acc = (
    df.groupby("group_HI")["correct"]
      .mean()
      .rename(index={0: "NH", 1: "HI"})
)
print(raw_group_acc)


# --------------------------------------------------
# 5. Helper functions
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
        contains="z",
    )

    p_col = find_col(
        fixed_effects,
        possible_names={"p", "p-val", "pvalue", "p_value", "p.value"},
    )

    ci_low_col = find_col(
        fixed_effects,
        contains="ci-low",
    )
    if ci_low_col is None:
        ci_low_col = find_col(fixed_effects, contains="2.5")

    ci_high_col = find_col(
        fixed_effects,
        contains="ci-high",
    )
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


def extract_term(fixed_effects, term_name):
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

    row_df = fixed_effects[fixed_effects[term_col].astype(str) == term_name]

    if len(row_df) != 1:
        raise ValueError(
            f"Could not uniquely identify term '{term_name}'. "
            f"Available terms are: {fixed_effects[term_col].astype(str).tolist()}"
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

    if se_col is not None:
        output["SE"] = float(row[se_col])
    else:
        output["SE"] = np.nan

    if z_col is not None:
        output["z"] = float(row[z_col])
    else:
        output["z"] = np.nan

    if p_col is not None:
        output["p_value"] = row[p_col]
    else:
        output["p_value"] = np.nan

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


def format_p_value(p):
    """Pretty formatting for p-values."""
    try:
        p_float = float(p)
    except Exception:
        return str(p)

    if p_float < 0.001:
        return "< .001"

    return f"{p_float:.3f}"


# --------------------------------------------------
# 6. Fit GLMM with NH as reference
# --------------------------------------------------
# group_HI = 0 means NH
# group_HI = 1 means HI
#
# Model:
#   correct ~ group_HI + (1 | subject)
#
# Interpretation:
#   Intercept = log-odds of correct decoding for NH
#   group_HI  = difference in log-odds between HI and NH

print("\nFitting GLMM with NH as reference")
print("---------------------------------")

df_r = pl.from_pandas(df)

model_nh_ref = glmer(
    "correct ~ group_HI + (1 | subject)",
    data=df_r,
    family="binomial",
)

model_nh_ref.fit(exponentiate=False, summary=True)

fixed_effects_nh_ref = to_pandas_table(model_nh_ref.result_fit)

print("\nMODEL SUMMARY: NH REFERENCE")
print("---------------------------")
print(fixed_effects_nh_ref)


# --------------------------------------------------
# 7. Fit GLMM with HI as reference
# --------------------------------------------------
# Create group_NH:
#   group_NH = 0 means HI
#   group_NH = 1 means NH
#
# Model:
#   correct ~ group_NH + (1 | subject)
#
# Interpretation:
#   Intercept = log-odds of correct decoding for HI
#   group_NH  = difference in log-odds between NH and HI

df_hi_ref = df.copy()
df_hi_ref["group_NH"] = 1 - df_hi_ref["group_HI"]

df_hi_ref_r = pl.from_pandas(df_hi_ref)

print("\nFitting GLMM with HI as reference")
print("---------------------------------")

model_hi_ref = glmer(
    "correct ~ group_NH + (1 | subject)",
    data=df_hi_ref_r,
    family="binomial",
)

model_hi_ref.fit(exponentiate=False, summary=True)

fixed_effects_hi_ref = to_pandas_table(model_hi_ref.result_fit)

print("\nMODEL SUMMARY: HI REFERENCE")
print("---------------------------")
print(fixed_effects_hi_ref)


# --------------------------------------------------
# 8. Extract rows for report table
# --------------------------------------------------

nh_vs_chance = extract_term(fixed_effects_nh_ref, "(Intercept)")
hi_vs_chance = extract_term(fixed_effects_hi_ref, "(Intercept)")
hi_vs_nh = extract_term(fixed_effects_nh_ref, "group_HI")

report_rows = [
    {
        "comparison": "NH vs chance",
        "term_used": "Intercept from NH-reference model",
        "interpretation": "NH decoding accuracy compared with 50% chance",
        **nh_vs_chance,
    },
    {
        "comparison": "HI vs chance",
        "term_used": "Intercept from HI-reference model",
        "interpretation": "HI decoding accuracy compared with 50% chance",
        **hi_vs_chance,
    },
    {
        "comparison": "HI vs NH",
        "term_used": "group_HI from NH-reference model",
        "interpretation": "Difference in decoding accuracy between HI and NH",
        **hi_vs_nh,
    },
]

report_table = pd.DataFrame(report_rows)

report_table["p_value_formatted"] = report_table["p_value"].apply(format_p_value)

report_table_rounded = report_table.copy()

numeric_cols = report_table_rounded.select_dtypes(include=[np.number]).columns
report_table_rounded[numeric_cols] = report_table_rounded[numeric_cols].round(4)

print("\nGLMM REPORT TABLE")
print("-----------------")
print(report_table_rounded)


# --------------------------------------------------
# 9. Predicted probabilities for NH and HI
# --------------------------------------------------

pred_df = model_nh_ref.empredict({"group_HI": [0, 1]})
pred_df = to_pandas_table(pred_df)

if "group_HI" in pred_df.columns:
    pred_df["group"] = pred_df["group_HI"].map({0: "NH", 1: "HI"})

print("\nPREDICTED PROBABILITIES")
print("-----------------------")
print(pred_df)


# --------------------------------------------------
# 10. Subject-level raw accuracies
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
# 11. Save outputs
# --------------------------------------------------

fixed_effects_nh_ref.to_csv(
    OUTPUT_DIR / "glmm_fixed_effects_NH_reference.csv",
    index=False,
)

fixed_effects_hi_ref.to_csv(
    OUTPUT_DIR / "glmm_fixed_effects_HI_reference.csv",
    index=False,
)

report_table.to_csv(
    OUTPUT_DIR / "glmm_report_table.csv",
    index=False,
)

report_table_rounded.to_csv(
    OUTPUT_DIR / "glmm_report_table_rounded.csv",
    index=False,
)

pred_df.to_csv(
    OUTPUT_DIR / "glmm_predicted_probabilities.csv",
    index=False,
)

subject_acc.to_csv(
    OUTPUT_DIR / "subject_level_mean_accuracy.csv",
    index=False,
)

raw_group_acc.to_csv(
    OUTPUT_DIR / "raw_group_accuracy.csv",
    header=["mean_accuracy"],
)

print("\nSaved:")
print(f"  {OUTPUT_DIR / 'glmm_fixed_effects_NH_reference.csv'}")
print(f"  {OUTPUT_DIR / 'glmm_fixed_effects_HI_reference.csv'}")
print(f"  {OUTPUT_DIR / 'glmm_report_table.csv'}")
print(f"  {OUTPUT_DIR / 'glmm_report_table_rounded.csv'}")
print(f"  {OUTPUT_DIR / 'glmm_predicted_probabilities.csv'}")
print(f"  {OUTPUT_DIR / 'subject_level_mean_accuracy.csv'}")
print(f"  {OUTPUT_DIR / 'raw_group_accuracy.csv'}")