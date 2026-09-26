#!/usr/bin/env python
"""Compare ONNX models on equal footing: paired accuracy, provider placement, latency.

Why this file exists
--------------------
`quantize.py` measures one precision/format per run and writes one JSON file.
Reading two of those files side by side is **not** a comparison, for three
separate reasons, and each one has burned this project or its neighbours before:

1. **The runs happen minutes apart.** Thermal state, GPU clocks and background
   load drift between them, so a latency difference can be manufactured by the
   clock rather than by the format. Confirm_quant.py established the fix for the
   latency half of this: interleave the models within each trial. This script
   applies the same rule to an arbitrary model set.

2. **An accuracy gap needs a paired test, not two proportions.** 2005 test
   samples at ~0.76 accuracy give a standard error of ~0.95 points, so a
   2-point gap is roughly two standard errors -- "probably real" is not a
   result. But the models were evaluated on the *same* samples, which is far
   more information than two independent proportions use. Comparing them as a
   paired difference (McNemar's exact test on the discordant pairs, plus a
   bootstrap CI on the accuracy difference) is what turns the gap into either
   evidence or noise.

3. **`session.get_providers()` does not say where the nodes ran.** It lists the
   execution providers that were *registered*. A graph can register the CUDA EP
   and still execute most of its nodes on the CPU -- this project already paid
   for that lesson with the silent CPU fallback (see ENVIRONMENT.md). The
   profiler reports the provider of every individual node, which is the only
   claim worth making.

Relationship to confirm_quant.py
--------------------------------
`confirm_quant.py` is the frozen first-generation check for one specific
question (is INT8-QDQ really slower than FP32?) over a hardcoded 3-model set,
and `results/quant-repeatability.json` is the record it produced. This script
generalises that trial loop to N models and adds the paired accuracy test and
the placement probe. The duplication is deliberate: the published record keeps
the script that produced it.

Usage
-----
    python compare_models.py                          # every models/resnet18-dermamnist-*.onnx
    python compare_models.py --trials 5 --iters 500
    python compare_models.py --models fp32=models/a.onnx myfmt=models/b.onnx
    python compare_models.py --skip-latency           # accuracy + placement only
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEFAULT_GLOB = "models/resnet18-dermamnist-*.onnx"
DEFAULT_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


# ---------------------------------------------------------------------------
# Model set
# ---------------------------------------------------------------------------


def discover_models(pattern: str) -> dict[str, str]:
    """Name -> path for every exported model, with the reference first.

    Ordered, because the interleaved trial loop iterates in this order and the
    FP32 export is what every other model is compared against.
    """
    found = {}
    for path in sorted(Path().glob(pattern)):
        name = path.stem.replace("resnet18-dermamnist-", "")
        found[name] = str(path)
    if not found:
        raise SystemExit(f"no models matched {pattern}; run quantize.py first")
    return {k: found[k] for k in sorted(found, key=lambda k: (not k.startswith("fp32"), k))}


def parse_model_args(values: list[str]) -> dict[str, str]:
    """Parse `name=path` pairs. Refuses a bare path rather than guessing a name."""
    models: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"--models entries must be name=path, got {item!r}")
        name, path = item.split("=", 1)
        if not Path(path).is_file():
            raise SystemExit(f"missing model file: {path}")
        models[name] = path
    return models


# ---------------------------------------------------------------------------
# Accuracy, kept per-sample so the comparison can be paired
# ---------------------------------------------------------------------------


def evaluate(path: str, loader, providers: list[str]) -> dict:
    """Run one model over the test split, keeping the per-sample predictions.

    The test loader is not shuffled (see train_dermamnist.loaders), so index i
    refers to the same sample for every model. That is what makes the paired
    test below valid -- without it the predictions could not be matched up.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(path, providers=providers)
    active = session.get_providers()
    input_name = session.get_inputs()[0].name

    preds: list[np.ndarray] = []
    truth: list[np.ndarray] = []
    for images, targets in loader:
        logits = session.run(None, {input_name: images.numpy().astype(np.float32)})[0]
        preds.append(np.asarray(logits.argmax(axis=1)))
        # medmnist hands back an (N, 1) label array whose dtype is not guaranteed
        # to be integer; flatten and cast so the paired comparison is on ints.
        truth.append(np.asarray(targets).reshape(-1).astype(np.int64))

    return {
        "predictions": np.concatenate(preds),
        "targets": np.concatenate(truth),
        "active_providers": list(active),
    }


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact McNemar test on the discordant pairs of two paired classifiers.

    Only the samples where the two models disagree carry information about which
    one is better; the ones both get right (or both get wrong) are identical
    information. Under H0 (no difference) each discordant sample is a fair coin,
    so the count is Binomial(n, 0.5) and the two-sided exact p-value is twice the
    lower tail.
    """
    b = int(np.sum(correct_a & ~correct_b))   # a right, b wrong
    c = int(np.sum(~correct_a & correct_b))   # a wrong, b right
    n = b + c
    if n == 0:
        p = 1.0
    else:
        tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2.0**n
        p = min(1.0, 2.0 * tail)
    return {
        "reference_right_other_wrong": b,
        "reference_wrong_other_right": c,
        "discordant": n,
        "exact_p_value": round(p, 6),
    }


def bootstrap_diff_ci(correct_a: np.ndarray, correct_b: np.ndarray,
                      iters: int = 4000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for (accuracy_a - accuracy_b) on paired samples.

    Resampling *samples* (not the two models separately) preserves the pairing,
    which is the whole point: the interval is for the difference, not for two
    accuracies that happen to be close.

    Convention: the returned interval belongs to `a - b`, in accuracy units
    (0.01 = 1 point). Callers must pass the arguments in the same order as the
    difference they print, or the interval comes out with the wrong sign.
    """
    diff = correct_a.astype(np.int8) - correct_b.astype(np.int8)
    n = diff.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(iters, n))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Where did the nodes actually run?
# ---------------------------------------------------------------------------


def profile_placement(path: str, providers: list[str], runs: int = 5,
                      outdir: str = "results/.ort_profile") -> dict:
    """Profile a session and count nodes per execution provider.

    The profiler is enabled on a *separate* session from the timed one: enabling
    profiling adds overhead to every node and would poison the latency figures.

    Shape-inference and other compile-time events are not `cat == "Node"` and are
    skipped, so what is counted is the executed graph.
    """
    import onnxruntime as ort

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(out / Path(path).stem)

    session = ort.InferenceSession(path, sess_options=options, providers=providers)
    input_name = session.get_inputs()[0].name
    x = np.random.randn(1, 3, 28, 28).astype(np.float32)
    for _ in range(runs):
        session.run(None, {input_name: x})
    profile_path = session.end_profiling()

    data = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    by_provider: dict[str, int] = {}
    ops_by_provider: dict[str, dict[str, int]] = {}
    for event in data:
        if event.get("cat") != "Node":
            continue
        args = event.get("args") or {}
        provider = args.get("provider") or "unknown"
        op = args.get("op_name") or event.get("name", "?").rsplit("_", 1)[0]
        by_provider[provider] = by_provider.get(provider, 0) + 1
        ops_by_provider.setdefault(provider, {})
        ops_by_provider[provider][op] = ops_by_provider[provider].get(op, 0) + 1

    total = sum(by_provider.values())
    cpu = by_provider.get("CPUExecutionProvider", 0)
    # The profiler records one event per node *execution*, so N inferences give N
    # copies of the graph. Dividing by the run count makes the number readable as
    # "nodes in this graph" instead of "nodes executed during this probe".
    per_run = {p: round(n / runs) for p, n in by_provider.items()}
    return {
        "registered_providers": list(session.get_providers()),
        "probe_runs": runs,
        "node_executions_total": total,
        "node_executions_by_provider": by_provider,
        "graph_nodes_by_provider": per_run,
        "graph_nodes_total": sum(per_run.values()),
        "ops_by_provider": ops_by_provider,
        "cpu_node_fraction": round(cpu / total, 4) if total else None,
        "profile_file": str(profile_path),
    }


# ---------------------------------------------------------------------------
# Interleaved latency
# ---------------------------------------------------------------------------


def interleaved_latency(models: dict[str, str], providers: list[str], trials: int,
                        warmup: int, iters: int, verbose: bool = True) -> tuple[dict, dict]:
    """Time every model once per trial, rotating through the whole set each round.

    Rotating means slow drift (clocks settling, a background process ramping)
    correlates with *trial index*, not with model -- which is the difference
    between measuring a format and measuring the machine.
    """
    from quantize import bench_onnx_latency

    raw = {n: {"inference": [], "inference_p95": [], "p95_over_p50": [],
               "end_to_end": []} for n in models}
    for trial in range(1, trials + 1):
        if verbose:
            print(f"\n--- trial {trial}/{trials} ---")
        for name, path in models.items():
            r = bench_onnx_latency(path, providers, warmup, iters)
            raw[name]["inference"].append(r["inference"]["p50_ms"])
            raw[name]["inference_p95"].append(r["inference"]["p95_ms"])
            raw[name]["p95_over_p50"].append(r["inference"]["p95_over_p50"])
            raw[name]["end_to_end"].append(r["end_to_end"]["p50_ms"])
            if verbose:
                active = "CUDA" if "CUDAExecutionProvider" in r["providers"] else "CPU!"
                print(f"  {name:<18} [{active}] inference p50={r['inference']['p50_ms']:>7.3f} "
                      f"p95={r['inference']['p95_ms']:>7.3f} "
                      f"(p95/p50={r['inference']['p95_over_p50']:>5.2f})  "
                      f"end-to-end p50={r['end_to_end']['p50_ms']:>7.3f}")

    def spread(values: list[float]) -> tuple[float, float]:
        if len(values) == 1:
            return values[0], 0.0
        return statistics.fmean(values), statistics.pstdev(values)

    reference = next(iter(models))
    summary = {}
    for name in models:
        inf_mean, inf_sd = spread(raw[name]["inference"])
        e2e_mean, e2e_sd = spread(raw[name]["end_to_end"])
        summary[name] = {
            "inference_p50_mean_ms": round(inf_mean, 3),
            "inference_p50_sd_ms": round(inf_sd, 3),
            "inference_p50_range_ms": [round(min(raw[name]["inference"]), 3),
                                       round(max(raw[name]["inference"]), 3)],
            "inference_p95_mean_ms": round(statistics.fmean(raw[name]["inference_p95"]), 3),
            "end_to_end_p50_mean_ms": round(e2e_mean, 3),
            "end_to_end_p50_sd_ms": round(e2e_sd, 3),
        }
    for name in models:
        summary[name]["inference_ratio_vs_reference"] = round(
            summary[name]["inference_p50_mean_ms"] / summary[reference]["inference_p50_mean_ms"], 3
        )
        summary[name]["end_to_end_ratio_vs_reference"] = round(
            summary[name]["end_to_end_p50_mean_ms"] / summary[reference]["end_to_end_p50_mean_ms"], 3
        )
    return raw, summary


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", nargs="*", default=None,
                        help="name=path pairs; default is every exported model")
    parser.add_argument("--glob", default=DEFAULT_GLOB)
    parser.add_argument("--reference", default=None,
                        help="model every comparison is against (default: the first one)")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--skip-placement", action="store_true")
    parser.add_argument("--output", default="results/model-comparison.json")
    args = parser.parse_args()

    # Must precede any onnxruntime import (see ENVIRONMENT.md, trap #1).
    from quantize import register_torch_cuda_dlls

    register_torch_cuda_dlls()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available to torch. Refusing to produce latency numbers "
            "that would silently be CPU numbers."
        )

    models = parse_model_args(args.models) if args.models else discover_models(args.glob)
    reference = args.reference or next(iter(models))
    if reference not in models:
        raise SystemExit(f"reference {reference!r} is not in the model set")

    # The machine state is part of the measurement, not context around it: the
    # QDQ graph crosses the PCIe bus 21 times per inference, so what else is
    # resident changes its p50 (measured: 1.48x vs 1.72x FP32). Recorded here
    # with the same helper benchmark.py uses, for the same reason.
    from benchmark import device_memory_snapshot

    device_before = device_memory_snapshot()

    print("=" * 74)
    print("model comparison")
    print("=" * 74)
    print(f"  machine: {device_before.get('device_used_mb')} MB used of "
          f"{device_before.get('device_total_mb')} MB, "
          f"{device_before.get('device_utilization_pct')}% util before the run")
    for name, path in models.items():
        size_mb = Path(path).stat().st_size / 1024**2
        marker = "  <- reference" if name == reference else ""
        print(f"  {name:<18} {size_mb:>7.2f} MB  {path}{marker}")

    print("\n--- accuracy on the test split (paired, per-sample) ---")
    from train_dermamnist import loaders

    _, _, test_loader, test_ds = loaders(
        batch_size=args.batch_size, num_workers=0, root=args.data_root
    )
    # len(test_loader.dataset), NOT len(test_ds): loaders() returns
    # (train_loader, val_loader, test_loader, train_ds) -- the fourth element is
    # the TRAIN dataset despite every caller naming it test_ds. Using it here
    # printed 7007 while the evaluation ran on 2005 samples.
    n_samples = len(test_loader.dataset)
    print(f"  test samples: {n_samples}  (train dataset object also returned: "
          f"{len(test_ds)}, unused)")

    accuracy: dict = {}
    correct: dict[str, np.ndarray] = {}
    for name, path in models.items():
        r = evaluate(path, test_loader, DEFAULT_PROVIDERS)
        hit = r["predictions"] == r["targets"]
        correct[name] = hit
        accuracy[name] = {
            "accuracy": round(float(hit.mean()), 4),
            "n_correct": int(hit.sum()),
            "n_samples": int(hit.size),
            "active_providers": r["active_providers"],
        }
        print(f"  {name:<18} acc={hit.mean():.4f}  ({int(hit.sum())}/{hit.size})")

    paired: dict = {}
    ref_correct = correct[reference]
    print(f"\n--- paired difference vs {reference} (McNemar exact + bootstrap 95% CI) ---")
    for name in models:
        if name == reference:
            continue
        hit = correct[name]
        diff_pts = (accuracy[name]["accuracy"] - accuracy[reference]["accuracy"]) * 100
        # Argument order follows the printed difference: other minus reference.
        lo, hi = bootstrap_diff_ci(hit, ref_correct, iters=args.bootstrap, seed=args.seed)
        stats = mcnemar_exact(ref_correct, hit)
        # Two tests, deliberately both required. The bootstrap CI is on the
        # difference in accuracy (a mean over all samples); McNemar uses only the
        # discordant pairs and is the appropriate exact test for paired binary
        # outcomes. Near the boundary they disagree -- FP16 lands at CI [+0.05,
        # +0.40] with p=0.13, i.e. a CI that just excludes zero while the exact
        # test does not. Calling that "real" would be a false positive, so the
        # verdict needs both, and the interval is quoted either way.
        significant = bool((lo > 0 or hi < 0) and stats["exact_p_value"] < 0.05)
        paired[name] = {
            "accuracy_diff_pts_vs_reference": round(diff_pts, 2),
            "diff_ci95_pts": [round(lo * 100, 2), round(hi * 100, 2)],
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "mcnemar_significant_at_5pct": stats["exact_p_value"] < 0.05,
            "significant": significant,
            "n_samples_differing_predictions": int(np.sum(ref_correct != hit)),
            "mcnemar": stats,
        }
        verdict = "REAL" if significant else "inside noise"
        print(f"  {name:<18} {diff_pts:>+6.2f} pts  CI [{lo * 100:+.2f}, {hi * 100:+.2f}]  "
              f"p={stats['exact_p_value']:.4f}  "
              f"samples differing={paired[name]['n_samples_differing_predictions']:>3}  "
              f"-> {verdict}")

    placement: dict = {}
    if not args.skip_placement:
        print("\n--- where the nodes actually ran (profiler, not get_providers) ---")
        for name, path in models.items():
            r = profile_placement(path, DEFAULT_PROVIDERS)
            placement[name] = r
            by = r["graph_nodes_by_provider"]
            cpu_pct = (r["cpu_node_fraction"] or 0) * 100
            print(f"  {name:<18} {r['graph_nodes_total']:>4} graph nodes  {by}  "
                  f"(CPU {cpu_pct:.0f}% of executions)")

    latency_raw, latency_summary = {}, {}
    if not args.skip_latency:
        print(f"\n--- interleaved latency: {args.trials} trials, "
              f"warmup={args.warmup}, iters={args.iters} ---")
        latency_raw, latency_summary = interleaved_latency(
            models, DEFAULT_PROVIDERS, args.trials, args.warmup, args.iters
        )
        print(f"\n{'model':<18} {'inf p50':>9} {'sd':>7} {'e2e p50':>9} {'vs ref':>8}")
        for name in models:
            s = latency_summary[name]
            print(f"{name:<18} {s['inference_p50_mean_ms']:>9.3f} "
                  f"{s['inference_p50_sd_ms']:>7.3f} {s['end_to_end_p50_mean_ms']:>9.3f} "
                  f"{s['inference_ratio_vs_reference']:>7.2f}x")

    report = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "reference": reference,
        "accuracy_split": "test",
        "calibration_split": "train",
        "n_test_samples": n_samples,
        "machine_state_before": device_before,
        "machine_state_after": device_memory_snapshot(),
        "models": models,
        "sizes_mb": {n: round(Path(p).stat().st_size / 1024**2, 2) for n, p in models.items()},
        "accuracy": accuracy,
        "paired_vs_reference": paired,
        "provider_placement": placement,
        "latency": {
            "config": {"trials": args.trials, "warmup": args.warmup,
                       "iters": args.iters, "order": "interleaved"},
            "raw": latency_raw,
            "summary": latency_summary,
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")

    print("\nConclusion discipline: a difference is only claimed when the CI "
          "excludes zero AND the sign holds in every trial. Otherwise say 'inside "
          "noise' and quote the interval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
