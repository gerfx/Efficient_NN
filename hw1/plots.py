import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hw1.equations import energy, latency, memory


RESULTS = Path(__file__).resolve().parent / "results"


def plot_grid(data, column, predict, unit):
    name = column.removesuffix("_ms")
    sizes = sorted(data.S.unique())
    batches = np.geomspace(data.B.min(), data.B.max(), 200)
    predictions = predict(np.array(sizes)[:, None], batches[None, :])
    fig, axes = plt.subplots(3, 4, figsize=(15, 9), constrained_layout=True)
    for ax, s, predicted in zip(axes.flat, sizes, predictions):
        subset = data[data.S == s]
        ax.plot(batches, predicted, color="black", label="Prediction")
        for validation, marker, label in ((False, "o", "Calibration"),
                                           (True, "x", "Validation")):
            points = subset[(subset.is_validation == validation) & (subset[column] > 0)]
            ax.scatter(points.B, points[column], marker=marker, s=25, label=label)
        oom = subset[subset.memory.isna()]
        if not oom.empty:
            ax.scatter(oom.B, predict(oom.S.to_numpy(), oom.B.to_numpy()),
                       marker="^", facecolors="none", edgecolors="red", s=60,
                       label="OOM (at prediction)")
        ax.set(title=f"S = {s}", xlabel="Batch size B", ylabel=f"{name.capitalize()} ({unit})",
               xscale="log", yscale="log")
        ax.grid(alpha=0.25, which="both")
    axes.flat[-1].axis("off")
    handles = {}
    for ax in axes.flat:
        artists, labels = ax.get_legend_handles_labels()
        handles.update(zip(labels, artists))
    axes.flat[-1].legend(handles.values(), handles.keys(), loc="center", frameon=False)
    fig.savefig(RESULTS / "figures" / f"{name}.png", dpi=150)
    plt.close(fig)


def main():
    data = pd.read_csv(RESULTS / "measurements.csv")
    data["memory"] = pd.to_numeric(data.memory, errors="coerce") / 1024**2
    theta = json.loads((RESULTS / "theta.json").read_text())
    latency_theta = [theta["latency"][key] for key in ("t0", "P", "BW")]
    (RESULTS / "figures").mkdir(exist_ok=True)
    plot_grid(data, "latency_ms", lambda s, b: latency(s, b, latency_theta) * 1000, "ms")
    plot_grid(data, "memory", lambda s, b: memory(s, b) / 1024**2, "MiB")
    energy_theta = [theta["energy"][key] for key in ("E0", "e_flops", "e_bytes")]
    plot_grid(data, "energy", lambda s, b: energy(s, b, energy_theta), "J")
    print(f"Figures saved to {RESULTS / 'figures'}")


if __name__ == "__main__":
    main()
