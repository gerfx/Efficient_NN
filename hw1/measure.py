import math
import random
import time
from pathlib import Path
from statistics import median

import pandas as pd
import pynvml
import torch

from hw1.models import CustomCNN


SEED = 42
BASE_S = [32, 64, 128, 224, 256, 384, 512]
BASE_B = [1, 2, 4, 8, 16, 32, 64, 128, 256]
WARMUP = 10
REPEATS = 20
ENERGY_WINDOW_MS = 1000
RESULTS = Path(__file__).resolve().parent / "results"


def measurement_grid(seed=SEED):
    rng = random.Random(seed)
    extra_s = sorted(rng.sample([s for s in range(32, 513, 16) if s not in BASE_S], 4))
    extra_b = sorted(rng.sample([b for b in range(1, 257) if b not in BASE_B], 3))
    return BASE_S + extra_s, BASE_B + extra_b


@torch.inference_mode()
def measure_one(model, s, b, handle):
    x = torch.randn(b, 3, s, s, device="cuda", dtype=torch.float32)
    for _ in range(WARMUP):
        model(x)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    output = model(x)
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated()
    del output

    times = []
    for _ in range(REPEATS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = model(x)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
        del output
    latency_ms = median(times)

    count = max(10, math.ceil(ENERGY_WINDOW_MS / latency_ms))
    torch.cuda.synchronize()
    before = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    for _ in range(count):
        model(x)
    torch.cuda.synchronize()
    after = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    energy = (after - before) / 1000 / count
    return latency_ms, peak_memory, energy


def main():
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = CustomCNN().cuda().float().eval()
    sizes, batches = measurement_grid()
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(f"GPU: {torch.cuda.get_device_name()}; PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}")

    pynvml.nvmlInit()
    rows = []
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        for s in sizes:
            for b in batches:
                row = {"S": s, "B": b, "latency_ms": None, "memory": None,
                       "energy": None, "is_validation": s not in BASE_S or b not in BASE_B}
                try:
                    latency_ms, memory, energy = measure_one(model, s, b, handle)
                    row.update(latency_ms=latency_ms, memory=memory, energy=energy)
                    detail = f"{latency_ms:.3f} ms, {memory / 1024**2:.1f} MiB, {energy:.6f} J"
                except torch.cuda.OutOfMemoryError:
                    row["memory"] = "OOM"
                    detail = "OOM"
                rows.append(row)
                pd.DataFrame(rows).to_csv(RESULTS / "measurements.csv", index=False)
                torch.cuda.empty_cache()
                print(f"[{len(rows):3}/132] S={s}, B={b}: {detail}", flush=True)
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
