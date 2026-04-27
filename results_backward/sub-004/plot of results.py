import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def corr(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")

    x = x - x.mean()
    y = y - y.mean()

    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom == 0:
        return np.nan

    return np.sum(x * y) / denom


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

processed_path = Path("data/processed/sub-004_backward_eelbrain.npz")

# Change this filename if yours is slightly different
predictions_path = Path(
    "results_backward/sub-004/sub-004_backward_eelbrain_backward_predictions.npz"
)

# ---------------------------------------------------------------------
# Load preprocessing output
# ---------------------------------------------------------------------

d = np.load(processed_path, allow_pickle=True)

print("Preprocessed keys:")
print(d.files)
print()

stim_att = np.stack([np.asarray(x, dtype=float).ravel() for x in d["stim_att"]])
stim_ign = np.stack([np.asarray(x, dtype=float).ravel() for x in d["stim_ign"]])
resp_tt = np.stack([np.asarray(x, dtype=float) for x in d["resp_tt"]])

print("Preprocessed shapes")
print("stim_att:", stim_att.shape)
print("stim_ign:", stim_ign.shape)
print("resp_tt:", resp_tt.shape)
print("fs_stim:", d["fs_stim"])
print("fs_resp:", d["fs_resp"])
print()

# ---------------------------------------------------------------------
# Basic preprocessing sanity plot
# ---------------------------------------------------------------------

i = 0

plt.figure(figsize=(12, 4))
plt.plot(stim_att[i][:1000])
plt.title("Real attended envelope trial 0")
plt.xlabel("Samples")
plt.ylabel("Envelope")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(stim_ign[i][:1000])
plt.title("Real ignored envelope trial 0")
plt.xlabel("Samples")
plt.ylabel("Envelope")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(resp_tt[i][:1000, 0])
plt.title("EEG channel 0 trial 0")
plt.xlabel("Samples")
plt.ylabel("EEG")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------
# Load model output
# ---------------------------------------------------------------------

p = np.load(predictions_path, allow_pickle=True)

print("Prediction keys:")
print(p.files)
print()

y_pred = np.asarray(p["y_pred_att"], dtype=float)

print("Prediction shape:", y_pred.shape)
print()

# ---------------------------------------------------------------------
# Evaluation check
# ---------------------------------------------------------------------

r_att = np.array([corr(y_pred[j], stim_att[j]) for j in range(len(y_pred))])
r_ign = np.array([corr(y_pred[j], stim_ign[j]) for j in range(len(y_pred))])

correct = r_att > r_ign

print("Evaluation check")
print("----------------")
print("mean r_att:", np.nanmean(r_att))
print("mean r_ign:", np.nanmean(r_ign))
print("mean r_att - r_ign:", np.nanmean(r_att - r_ign))
print("accuracy:", np.nanmean(correct))
print("n_correct:", np.sum(correct), "/", len(correct))
print()

# ---------------------------------------------------------------------
# Label flip diagnostic
# ---------------------------------------------------------------------

flipped_correct = r_ign > r_att

print("If labels were flipped")
print("----------------------")
print("flipped accuracy:", np.nanmean(flipped_correct))
print("flipped n_correct:", np.sum(flipped_correct), "/", len(flipped_correct))
print()

# ---------------------------------------------------------------------
# Per-trial inspection
# ---------------------------------------------------------------------

for j in range(min(10, len(y_pred))):
    print(
        f"trial {j:02d}: "
        f"r_att={r_att[j]: .4f}, "
        f"r_ign={r_ign[j]: .4f}, "
        f"correct={bool(correct[j])}"
    )

# ---------------------------------------------------------------------
# Plot predicted vs real envelopes for one trial
# ---------------------------------------------------------------------

trial = 0
fs = float(np.asarray(d["fs_stim"]).item())
t = np.arange(len(y_pred[trial])) / fs

att_c = stim_att[0] - stim_att[0].mean()
pred_c = y_pred[0] - y_pred[0].mean()

pred_c_scaled = pred_c / pred_c.std() * att_c.std()

plt.figure(figsize=(14, 5))
plt.plot(att_c, label="attended centered")
plt.plot(pred_c_scaled, label="predicted centered + scaled")
plt.legend()
plt.show()

print("corr:", corr(pred_c, att_c))
print("att std:", att_c.std())
print("pred std:", pred_c.std())
import numpy as np, json

d = np.load("data/processed/sub-004_backward_eelbrain.npz", allow_pickle=True)
trial_meta = json.loads(d["trial_meta"].item())

for i, tr in enumerate(trial_meta[:10]):
    if tr["trial_kind"] == "twotalker":
        print(i)
        print("target event:", tr["stim_file_target_event"])
        print("masker event:", tr["stim_file_masker_event"])
        print("attend_left_right:", tr["attend_left_right"])
        print("target path:", tr["target_path"])
        print("masker path:", tr["masker_path"])
        print()

plt.figure(figsize=(14, 5))
plt.plot(t, stim_att[trial], label="Real attended envelope")
plt.plot(t, stim_ign[trial], label="Real ignored envelope", alpha=0.7)
plt.plot(t, y_pred[trial], label="TRF reconstructed envelope", linewidth=2)

plt.title(
    f"Trial {trial}: "
    f"r_att={r_att[trial]:.3f}, "
    f"r_ign={r_ign[trial]:.3f}, "
    f"correct={bool(correct[trial])}"
)
plt.xlabel("Time (s)")
plt.ylabel("Envelope")
plt.legend()
plt.tight_layout()
plt.show()
