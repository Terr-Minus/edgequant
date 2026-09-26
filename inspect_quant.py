#!/usr/bin/env python
"""Read the quantisation parameters back out of an ONNX file.

Why this exists
---------------
Two questions came up in the learning notes that a latency table cannot answer:

1. **Was the quantisation actually per-channel?** `quantize.py` passes
   `per_channel=True` and then never verifies it. "We asked for it" is not
   evidence -- the same mistake as trusting `get_available_providers()` instead
   of `session.get_providers()`. The answer is in the file: a per-channel weight
   scale is a 1-D initializer of length `out_channels`, a per-tensor one is a
   scalar.

2. **How much does per-channel actually buy?** The per-channel scales *are* the
   per-channel dynamic ranges (for symmetric INT8, `scale_i = max|w_i| / 127`).
   So the spread of those scales measures directly how wrong a single shared
   scale would be: the narrowest channel keeps only `min(scale)/max(scale)` of
   the 255 available levels. If every channel had the same range, per-channel
   quantisation would be pure overhead and would change nothing.

It also compares two quantised files' scales, which is how the QDQ vs QOperator
accuracy gap gets attributed: identical scales mean the difference is in the
kernel arithmetic (fixed-point requantisation in the integer ops vs a float
convolution between quantised boundaries), not in the calibration.

Static analysis only -- no runtime, no GPU, no calibration data needed.

Usage
-----
    python inspect_quant.py                                   # every quantised model
    python inspect_quant.py --models models/a.onnx models/b.onnx
    python inspect_quant.py --reference models/resnet18-dermamnist-fp32.onnx
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# node type -> (index of the quantised weight input, index of its scale input)
WEIGHT_SLOTS = {
    "DequantizeLinear": (0, 1),
    "QLinearConv": (3, 4),
    "QLinearMatMul": (3, 4),
    "QGemm": (3, 4),
}

QUANT_OPS = (
    "QuantizeLinear", "DequantizeLinear", "QLinearConv", "QLinearMatMul",
    "QGemm", "QLinearAdd", "QLinearGlobalAveragePool", "ConvInteger",
    "MatMulInteger", "DynamicQuantizeLinear",
)


def load_graph(path: str):
    import onnx

    model = onnx.load(path)
    initializers = {init.name: init for init in model.graph.initializer}
    return model.graph, initializers


def initializer_array(initializers: dict, name: str):
    import onnx
    from onnx import numpy_helper

    if name not in initializers:
        return None
    return numpy_helper.to_array(initializers[name])


def op_inventory(graph) -> dict:
    counts: dict[str, int] = {}
    for node in graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return {k: v for k, v in sorted(counts.items()) if k in QUANT_OPS}


def granularity(scale: np.ndarray) -> str:
    """per-tensor if the scale has no channel axis, per-channel if it has one."""
    if scale.size == 1:
        return "per-tensor"
    return "per-channel"


def weight_quantizers(graph, initializers: dict) -> list[dict]:
    """Every (weight tensor, its scale) pair the graph actually contains."""
    found = []
    for node in graph.node:
        slots = WEIGHT_SLOTS.get(node.op_type)
        if not slots:
            continue
        weight_idx, scale_idx = slots
        if len(node.input) <= max(weight_idx, scale_idx):
            continue
        weight_name = node.input[weight_idx]
        scale_name = node.input[scale_idx]
        weight = initializer_array(initializers, weight_name)
        scale = initializer_array(initializers, scale_name)
        if weight is None or scale is None:
            continue
        found.append({
            "op": node.op_type,
            "weight": weight_name,
            "weight_dtype": str(weight.dtype),
            "weight_shape": list(weight.shape),
            "scale": scale_name,
            "scale_shape": list(scale.shape),
            "granularity": granularity(np.atleast_1d(scale)),
            "n_channels": int(np.atleast_1d(scale).size),
        })
    return found


def activation_quantizers(graph, initializers: dict) -> list[dict]:
    """QuantizeLinear nodes over tensors that are not initializers."""
    found = []
    for node in graph.node:
        if node.op_type != "QuantizeLinear" or len(node.input) < 2:
            continue
        tensor = node.input[0]
        if tensor in initializers:
            continue  # a weight, counted above
        scale = initializer_array(initializers, node.input[1])
        if scale is None:
            continue
        found.append({
            "tensor": tensor,
            "scale_shape": list(scale.shape),
            "granularity": granularity(np.atleast_1d(scale)),
        })
    return found


def channel_spread(scale: np.ndarray) -> dict:
    """What a single shared scale would cost, straight from the scales.

    Assumes symmetric weight quantisation (scale_i = max|w_i| / 127), which is
    what `quantize_static(weight_type=QInt8)` produces. Under one per-tensor
    scale the shared step must be `S = max(scale)`, so a channel whose own step
    is `s_i` can only land on values spaced `S` apart: it keeps
    `2 * 127 * (s_i / S) + 1` distinct integer levels out of the 255 that
    per-channel quantisation gives it.
    """
    s = np.atleast_1d(np.asarray(scale, dtype=np.float64))
    if s.size <= 1:
        return {}
    widest, narrowest = float(s.max()), float(s.min())
    ratio = widest / narrowest if narrowest > 0 else float("inf")
    return {
        "n_channels": int(s.size),
        "widest_channel_scale": widest,
        "narrowest_channel_scale": narrowest,
        "widest_over_narrowest": round(ratio, 2),
        # distinct levels the worst-served channel retains under one shared step
        "worst_channel_levels": round(254.0 * narrowest / widest + 1.0, 1),
    }


def weight_scale_map(graph, initializers: dict) -> dict:
    """Logical weight name -> its quantisation scale, for any quantised format.

    The key strips the `_quantized` suffix so the QDQ and QOperator graphs can be
    compared tensor by tensor: the two formats name their scales after the same
    logical weight even though the surrounding graph looks nothing alike.
    """
    out = {}
    for node in graph.node:
        slots = WEIGHT_SLOTS.get(node.op_type)
        if not slots:
            continue
        weight_idx, scale_idx = slots
        if len(node.input) <= max(weight_idx, scale_idx):
            continue
        # Require the tensor to be an initializer. Without this check a
        # DequantizeLinear on an *activation* (input[0] = a graph tensor) is
        # counted as a weight, which inflated QDQ's count from 42 to 74.
        if initializer_array(initializers, node.input[weight_idx]) is None:
            continue
        scale = initializer_array(initializers, node.input[scale_idx])
        if scale is None:
            continue
        out[node.input[weight_idx].replace("_quantized", "")] = np.atleast_1d(scale)
    return out


def compare_scales(baseline_path: str, other_path: str) -> dict:
    """Are two quantised models' quantisation parameters the same numbers?

    If they are, an accuracy difference between the two formats cannot come from
    calibration -- only from what the kernels do with those parameters.

    The baseline must be a *quantised* model: an FP32 export has no scales at all,
    so comparing against it yields an empty intersection and proves nothing.
    """
    _, base_init = load_graph(baseline_path)
    base_graph, _ = load_graph(baseline_path)
    oth_graph, oth_init = load_graph(other_path)
    base = weight_scale_map(base_graph, base_init)
    other = weight_scale_map(oth_graph, oth_init)
    shared = sorted(set(base) & set(other))
    if not shared:
        return {
            "shared_weights": 0,
            "note": "no weight tensors in common; scale values cannot be compared",
            "baseline_weights": len(base),
            "other_weights": len(other),
        }

    max_rel = 0.0
    worst = None
    shape_mismatches = 0
    compared = 0
    for name in shared:
        a, b = base[name], other[name]
        if a.shape != b.shape:
            # Different granularity: a per-tensor scale is a scalar, a
            # per-channel one is a vector. These are not "equal", they are not
            # comparable at all -- and silently skipping them while still
            # reporting identical_within_1e-6=True is how this function lied the
            # first time it was pointed at a per-tensor model.
            shape_mismatches += 1
            continue
        compared += 1
        denom = np.maximum(np.abs(a), 1e-12)
        rel = float(np.max(np.abs(a - b) / denom))
        if rel > max_rel:
            max_rel, worst = rel, name
    return {
        "baseline": Path(baseline_path).stem,
        "shared_weights": len(shared),
        "tensors_compared": compared,
        "baseline_only": len(set(base) - set(other)),
        "other_only": len(set(other) - set(base)),
        "shape_mismatches": shape_mismatches,
        "max_relative_scale_difference": max_rel if compared else None,
        "worst_tensor": worst,
        "identical_within_1e-6": bool(compared and max_rel < 1e-6),
    }


def inspect(path: str) -> dict:
    graph, initializers = load_graph(path)
    weights = weight_quantizers(graph, initializers)
    activations = activation_quantizers(graph, initializers)

    spreads = [channel_spread(initializer_array(initializers, w["scale"])) for w in weights]
    spreads = [s for s in spreads if s]

    per_channel = [w for w in weights if w["granularity"] == "per-channel"]
    return {
        "path": path,
        "size_mb": round(Path(path).stat().st_size / 1024**2, 2),
        "op_inventory": op_inventory(graph),
        "weight_quantizers": len(weights),
        "weight_granularity": {
            "per-channel": len(per_channel),
            "per-tensor": len(weights) - len(per_channel),
        },
        "activation_quantizers": len(activations),
        "activation_granularity": {
            "per-channel": sum(1 for a in activations if a["granularity"] == "per-channel"),
            "per-tensor": sum(1 for a in activations if a["granularity"] == "per-tensor"),
        },
        "weight_scale_examples": weights[:3],
        "channel_spread": {
            "tensors_measured": len(spreads),
            "max_widest_over_narrowest": round(
                max(s["widest_over_narrowest"] for s in spreads), 2) if spreads else None,
            "median_widest_over_narrowest": round(float(np.median(
                [s["widest_over_narrowest"] for s in spreads])), 2) if spreads else None,
            "worst_channel_levels": min(
                (s["worst_channel_levels"] for s in spreads), default=None),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", nargs="*", default=None,
                        help="quantised ONNX files; default is every int8 export")
    parser.add_argument("--scale-baseline", default=None,
                        help="quantised ONNX whose scales the others are compared "
                             "against (default: the first model that has any)")
    parser.add_argument("--output", default="results/quant-graph-facts.json")
    args = parser.parse_args()

    models = args.models or sorted(
        str(p) for p in Path("models").glob("resnet18-dermamnist-*.onnx")
        if any(tag in p.name for tag in ("int8", "fp16"))
    )
    if not models:
        raise SystemExit("no quantised models found; run quantize.py first")

    report = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "note": "static graph inspection; no runtime, no GPU",
        "models": {},
    }

    for path in models:
        facts = inspect(path)
        report["models"][Path(path).stem] = facts
        print("=" * 74)
        print(f"{Path(path).name}   {facts['size_mb']} MB")
        print("=" * 74)
        if not facts["weight_quantizers"] and not facts["op_inventory"]:
            print("  no quantisation parameters at all -- FP16 is a dtype cast")
            print("  (scale/zero-point are fixed by IEEE 754, nothing is calibrated),")
            print("  so there is no per-tensor/per-channel question to answer here.\n")
            continue
        print(f"  quantised ops : {facts['op_inventory']}")
        print(f"  weights       : {facts['weight_quantizers']} quantised "
              f"({facts['weight_granularity']['per-channel']} per-channel, "
              f"{facts['weight_granularity']['per-tensor']} per-tensor)")
        print(f"  activations   : {facts['activation_quantizers']} quantised "
              f"({facts['activation_granularity']['per-channel']} per-channel, "
              f"{facts['activation_granularity']['per-tensor']} per-tensor)")
        for ex in facts["weight_scale_examples"]:
            print(f"    e.g. {ex['op']:<18} weight {ex['weight_shape']} "
                  f"dtype={ex['weight_dtype']:<6} scale {ex['scale_shape']} "
                  f"-> {ex['granularity']}")
        cs = facts["channel_spread"]
        if cs["tensors_measured"]:
            print(f"  channel spread: widest/narrowest scale per tensor — "
                  f"median {cs['median_widest_over_narrowest']}x, "
                  f"max {cs['max_widest_over_narrowest']}x "
                  f"(the worst channel keeps {cs['worst_channel_levels']} of 255 levels "
                  f"under a single shared scale)")
        print()

    # The scale comparison needs a QUANTISED baseline. An FP32 or FP16 export has
    # no scales, so comparing against one intersects to the empty set and proves
    # nothing -- which is exactly what an earlier version of this script did.
    baseline = args.scale_baseline
    if baseline is None:
        for path in models:
            graph, initializers = load_graph(path)
            if weight_scale_map(graph, initializers):
                baseline = path
                break
    if baseline:
        print("=" * 74)
        print(f"scale comparison, baseline = {Path(baseline).name}")
        print("=" * 74)
        report["scale_comparison"] = {"baseline": Path(baseline).stem, "vs": {}}
        for path in models:
            cmp = compare_scales(baseline, path)
            report["scale_comparison"]["vs"][Path(path).stem] = cmp
            if cmp.get("tensors_compared"):
                print(f"  {Path(path).stem:<34} compared={cmp['tensors_compared']} "
                      f"max_rel_diff={cmp['max_relative_scale_difference']:.3e}  "
                      f"identical={cmp['identical_within_1e-6']}")
            elif cmp.get("shape_mismatches"):
                print(f"  {Path(path).stem:<34} NOT COMPARABLE: "
                      f"{cmp['shape_mismatches']} tensors differ in granularity "
                      f"(scalar vs vector scale)")
            else:
                print(f"  {Path(path).stem:<34} {cmp.get('note')}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
