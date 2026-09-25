import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, nnls

from equations import bytes_moved, energy, flops, latency, memory


RESULTS = Path(__file__).resolve().parent / "results"


def errors(measured, predicted):
    delta = np.asarray(predicted) - np.asarray(measured)
    return {
        "MAE": float(np.mean(np.abs(delta))),
        "RMSE": float(np.sqrt(np.mean(delta**2))),
        "MAPE_percent": float(np.mean(np.abs(delta) / measured) * 100),
    }


def main():
    data = pd.read_csv(RESULTS / "measurements.csv")
    data["memory"] = pd.to_numeric(data["memory"], errors="coerce")
    calibration = data[~data.is_validation & (data.latency_ms > 0)]
    s, b = calibration.S.to_numpy(), calibration.B.to_numpy()
    measured = calibration.latency_ms.to_numpy()

    def residuals(theta):
        return (latency(s, b, theta) * 1000 - measured) / measured

    fits = [
        least_squares(residuals, start, bounds=([0, 1e-6, 1e-6], np.inf))
        for start in ([0.1, 1, 20], [0.3, 4, 100], [0.5, 10, 300])
    ]
    fit = min(fits, key=lambda result: result.cost)
    theta = {"latency": dict(zip(("t0", "P", "BW"), fit.x.tolist()))}

    energy_data = calibration[calibration.energy > 0]
    s, b = energy_data.S.to_numpy(), energy_data.B.to_numpy()
    measured = energy_data.energy.to_numpy()
    features = np.column_stack((np.ones(len(s)), flops(s, b) / 1e12,
                                bytes_moved(s, b) / 1e9))
    energy_theta, _ = nnls(features / measured[:, None], np.ones(len(s)))
    theta["energy"] = dict(zip(("E0", "e_flops", "e_bytes"), energy_theta.tolist()))

    predictors = {
        "latency_ms": (lambda s, b: latency(s, b, fit.x) * 1000, "ms"),
        "memory": (lambda s, b: memory(s, b) / 1024**2, "MiB"),
        "energy": (lambda s, b: energy(s, b, energy_theta), "J"),
    }

    metrics = {}
    for split, is_validation in (("calibration", False), ("validation", True)):
        metrics[split] = {}
        for name, (predict, unit) in predictors.items():
            subset = data[(data.is_validation == is_validation) & (data[name] > 0)]
            measured = subset[name].to_numpy()
            if name == "memory":
                measured = measured / 1024**2
            predicted = predict(subset.S.to_numpy(), subset.B.to_numpy())
            metrics[split][name] = {"n": len(subset), "unit": unit,
                                    **errors(measured, predicted)}

    oom = data[data.memory.isna()][["S", "B"]].copy()
    oom["predicted_memory_GiB"] = memory(oom.S.to_numpy(), oom.B.to_numpy()) / 1024**3
    metrics["oom"] = oom.to_dict(orient="records")
    for name, value in (("theta.json", theta), ("metrics.json", metrics)):
        (RESULTS / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(theta, indent=2))
    print("Validation:", json.dumps(metrics["validation"], indent=2))
    print("OOM:\n", oom.to_string(index=False))


if __name__ == "__main__":
    main()
