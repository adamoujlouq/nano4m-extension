import json
import matplotlib.pyplot as plt
import os

def load_metrics(path):
    steps, train_loss, val_loss = [], [], []
    val_rgb, val_depth, val_normal, val_caption = [], [], [], []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            steps.append(d["step"])
            train_loss.append(d["train_loss"])
            val_loss.append(d["val_loss"])
            val_rgb.append(d["val_loss_tok_rgb@256"])
            val_depth.append(d["val_loss_tok_depth@256"])
            val_normal.append(d["val_loss_tok_normal@256"])
            val_caption.append(d["val_loss_scene_desc"])
    return steps, train_loss, val_loss, val_rgb, val_depth, val_normal, val_caption

runs = {
    "baseline_seed42":  "results/results/baseline_seed42/metrics.jsonl",
    "baseline_seed123": "results/results/baseline_seed123/metrics.jsonl",
    "swiglu_seed42":    "results/results/swiglu_seed42/metrics.jsonl",
}

colors = {
    "baseline_seed42":  "steelblue",
    "baseline_seed123": "cornflowerblue",
    "swiglu_seed42":    "tomato",
}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("nano4M — Baseline vs SwiGLU", fontsize=14, fontweight="bold")

for run_name, path in runs.items():
    if not os.path.exists(path):
        print(f"Skipping {run_name} — fichier introuvable")
        continue
    steps, train_loss, val_loss, val_rgb, val_depth, val_normal, val_caption = load_metrics(path)
    c = colors[run_name]

    # Plot 1 — Val loss globale
    axes[0].plot(steps, val_loss, label=run_name, color=c)

    # Plot 2 — Val loss par modalité (seulement pour le premier run disponible)
    if run_name == "baseline_seed123":
        axes[1].plot(steps, val_rgb,     label="RGB",     color="steelblue")
        axes[1].plot(steps, val_depth,   label="Depth",   color="tomato")
        axes[1].plot(steps, val_normal,  label="Normals", color="seagreen")
        axes[1].plot(steps, val_caption, label="Captions",color="darkorange")

axes[0].set_title("Val Loss globale")
axes[0].set_xlabel("Steps")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_title("Val Loss par modalité (baseline_seed123)")
axes[1].set_xlabel("Steps")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/metrics_comparison.png", dpi=150)
plt.show()
print("Saved: results/metrics_comparison.png")