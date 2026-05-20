from pathlib import Path
import json
import pandas as pd

results_dir = Path("results_backward")

rows = []

for summary_path in sorted(results_dir.glob("sub-*/sub-*_backward_eelbrain_backward_summary.json")):
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    subject = data["meta"]["subject"]
    hearing_status = data["meta"]["hearing_status"]

    # Make numeric group variable for statistics
    group_HI = 1 if hearing_status.lower() == "hi" else 0

    summary = data.get("summary_two_talker", {})
    decoding_accuracy = summary.get("decoding_accuracy")

    for trial in data["trial_results_two_talker"]:
        rows.append({
            "subject": subject,
            "hearing_status": hearing_status,
            "group_HI": group_HI,
            "trial_index": trial["trial_index"],
            "r_att": trial["r_att"],
            "r_ign": trial["r_ign"],
            "r_diff": trial["r_att"] - trial["r_ign"],
            "correct": trial["correct"],
            "subject_decoding_accuracy": decoding_accuracy,
        })

df = pd.DataFrame(rows)

df.to_csv("statistics/aad_trial_level_results.csv", index=False)

print(df.head())
print(df["subject"].nunique(), "subjects")
print(df.groupby("hearing_status")["subject"].nunique())