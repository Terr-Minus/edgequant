"""Confirm the INT8-is-slower result with repeated trials.

The single run in results/quant-int8.json showed INT8 inference p50 at 1.58x
FP32. One measurement is not evidence for a claim that goes into a document, so
this does three things the single run did not:

  1. 3 trials per precision, not 1.
  2. Interleaved order (fp32, fp16, int8, fp32, fp16, int8, ...). Running all of
     one precision then all of the next lets thermal drift and background load
     correlate with precision, which is how a fake effect appears.
  3. Longer warmup and more iterations (50 / 500 vs 20 / 200), because p95/p50
     came out above the harness warning threshold last time.

Reports per-precision p50 and p95 as mean and spread across trials, and the
ratio of INT8 to FP32 with its range, so the claim is stated with its noise.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np

from quantize import bench_onnx_latency, register_torch_cuda_dlls

register_torch_cuda_dlls()

PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
WARMUP = 50
ITERS = 500
TRIALS = 3

MODELS = {
    "fp32": "models/resnet18-dermamnist-fp32.onnx",
    "fp16": "models/resnet18-dermamnist-fp16.onnx",
    "int8-qdq": "models/resnet18-dermamnist-int8-qdq.onnx",
}

for name, path in MODELS.items():
    if not Path(path).is_file():
        raise SystemExit(f"missing {path}; run quantize.py first")

results: dict[str, dict[str, list[float]]] = {n: {"inference": [], "end_to_end": []} for n in MODELS}

print("=" * 72)
print(f"interleaved trials: {TRIALS} per precision, warmup={WARMUP}, iters={ITERS}")
print("=" * 72)

for trial in range(1, TRIALS + 1):
    print(f"\n--- trial {trial}/{TRIALS} ---")
    for name, path in MODELS.items():
        r = bench_onnx_latency(path, PROVIDERS, WARMUP, ITERS)
        inf = r["inference"]
        e2e = r["end_to_end"]
        results[name]["inference"].append(inf["p50_ms"])
        results[name]["inference_p95"] = results[name].get("inference_p95", [])
        results[name]["inference_p95"].append(inf["p95_ms"])
        results[name]["end_to_end"].append(e2e["p50_ms"])
        active = "CUDA" if "CUDAExecutionProvider" in r["providers"] else "CPU!"
        print(f"  {name:<10} [{active}] inference p50={inf['p50_ms']:>7.3f} "
              f"p95={inf['p95_ms']:>7.3f} (p95/p50={inf['p95_over_p50']:>5.2f})  "
              f"end-to-end p50={e2e['p50_ms']:>7.3f}")

print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)


def summarize(vals):
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.fmean(vals), statistics.pstdev(vals)


print(f"{'precision':<10} {'inf p50 mean':>13} {'sd':>7} {'inf p95 mean':>13} "
      f"{'e2e p50 mean':>13} {'sd':>7}")
inf_p50 = {}
inf_p95 = {}
e2e_p50 = {}
for name in MODELS:
    m, s = summarize(results[name]["inference"])
    m95, _ = summarize(results[name]["inference_p95"])
    me, se = summarize(results[name]["end_to_end"])
    inf_p50[name] = (m, s)
    inf_p95[name] = m95
    e2e_p50[name] = (me, se)
    print(f"{name:<10} {m:>13.3f} {s:>7.3f} {m95:>13.3f} {me:>13.3f} {se:>7.3f}")

base_i = inf_p50["fp32"][0]
base_e = e2e_p50["fp32"][0]
print()
for name in ("fp16", "int8-qdq"):
    ri = inf_p50[name][0] / base_i
    re_ = e2e_p50[name][0] / base_e
    print(f"  {name:<10} inference {ri:>5.2f}x FP32   end-to-end {re_:>5.2f}x FP32")

# Does the sign of the effect hold in every trial?
print("\nper-trial INT8/FP32 inference ratio (does the direction hold every time?):")
ratios = [i / f for i, f in zip(results["int8-qdq"]["inference"], results["fp32"]["inference"])]
for t, r in enumerate(ratios, 1):
    print(f"  trial {t}: {r:.2f}x")
print(f"  --> all trials above 1.0: {all(r > 1.0 for r in ratios)}")

fp16_ratios = [i / f for i, f in zip(results["fp16"]["inference"], results["fp32"]["inference"])]
print("per-trial FP16/FP32 inference ratio:")
for t, r in enumerate(fp16_ratios, 1):
    print(f"  trial {t}: {r:.2f}x")

out = {
    "trials": TRIALS,
    "warmup": WARMUP,
    "iters": ITERS,
    "order": "interleaved",
    "raw": results,
    "int8_over_fp32_inference": {
        "mean": statistics.fmean(ratios),
        "min": min(ratios),
        "max": max(ratios),
        "all_above_1": all(r > 1.0 for r in ratios),
    },
    "fp16_over_fp32_inference": {
        "mean": statistics.fmean(fp16_ratios),
        "min": min(fp16_ratios),
        "max": max(fp16_ratios),
    },
}
Path("results/quant-repeatability.json").write_text(
    json.dumps(out, indent=2), encoding="utf-8"
)
print("\nwrote results/quant-repeatability.json")
