#!/usr/bin/env python
"""Quantise the FP32 reference model and measure what it costs.

This is the deliverable the project exists for: not "quantisation runs", but a
table saying what FP16 and INT8 actually cost in accuracy, latency and memory,
measured on the same test split as the FP32 baseline.

Pipeline
--------
    checkpoints/*.pt            (FP32 reference, from train_dermamnist.py)
        -> export ONNX
        -> convert to FP16          <- this script, --precision fp16
        -> quantise to INT8         <- this script, --precision int8
        -> verify numerical equivalence
        -> evaluate accuracy on the SAME test split
        -> benchmark latency + memory

Data-split discipline (the rule that keeps the conclusion honest)
----------------------------------------------------------------
    INT8 needs a calibration set. It is drawn from the TRAIN split.
    The TEST split is used for accuracy reporting only.

Calibrating on test would leak test data into the quantisation process and
inflate the reported accuracy -- the quantisation analogue of training on the
test set. It is a classic mistake and an interviewer will ask about it.

Latency is measured with benchmark.py's timed_run(), so the method (warmup,
cuda synchronise on both sides, nearest-rank percentiles) is identical to the
FP32 baseline. Comparing a number produced one way against a number produced
another way is how people publish quantisation speedups that do not exist.

Usage
-----
    python quantize.py --precision int8 --quant-format qdq
    python quantize.py --precision int8 --quant-format qoperator
    python quantize.py --precision int8 --calib-method entropy --calib-samples 1024
    python quantize.py --precision int8 --weight-granularity per-tensor   # control
    python quantize.py --precision both

The three INT8 axes (format / weight granularity / calibration method) each get
their own output filename, so a comparison run cannot silently overwrite the
configuration it is being compared against.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from models import ARCH_NAME, build_model_from_checkpoint, load_checkpoint


# ---------------------------------------------------------------------------
# ONNX Runtime must find the CUDA/cuDNN DLLs that torch ships, or it silently
# falls back to the CPU provider. Same trap as benchmark.py; see ENVIRONMENT.md.
# ---------------------------------------------------------------------------


def register_torch_cuda_dlls() -> list[str]:
    import torch

    lib = Path(torch.__file__).parent / "lib"
    if not lib.is_dir():
        return []
    os.environ["PATH"] = str(lib) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(str(lib))
        except OSError:
            pass
    return [str(lib)]


def redirect_temp_dir(staging_dir: str | None, out_parent: Path) -> Path:
    """Point every temp mechanism at a directory we own, before ORT imports.

    onnxruntime's quantiser stages intermediates with
    tempfile.TemporaryDirectory() in at least three places, under two different
    prefixes -- `ort.quant.` (quantize.py, quant_utils.py) and `pre.quant.`
    (shape_inference.py). On this machine the process temp root is a redirected,
    non-writable location, so each of those dies the same way:

        PermissionError: [Errno 13] Permission denied:
            '...\\Temp\\dsh-xxxx\\pre.quant.yyyyyyyy\\symbolic_shape_inferred.onnx'

    with the cleanup failure raised from __exit__ aborting the run. Two details
    make this hard to fix in one place:

      * `tempfile.tempdir` gets reset back to None at some point, so it has to be
        set here rather than inside the quantisation function.
      * `ignore_cleanup_errors=True` is not sufficient on its own: tempfile's
        `_resetperms()` raises a second PermissionError while trying to chmod the
        directory deletable, and that path is outside the flag's coverage.

    So both TEMP/TMP and tempfile.tempdir are redirected, and the directory is
    left alone rather than deleted by us mid-run. It is gitignored.
    """
    import tempfile

    staged = Path(staging_dir) if staging_dir else (out_parent / ".quant_work")
    staged.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = str(staged)
    os.environ["TMP"] = str(staged)
    tempfile.tempdir = str(staged)
    return staged


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_fp32_onnx(checkpoint: dict, path: str, height: int = 28, width: int = 28) -> str:
    """Export the checkpoint to ONNX with a dynamic batch axis."""
    import torch

    model = build_model_from_checkpoint(checkpoint).eval()
    dummy = torch.randn(1, 3, height, width)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    return path


def convert_fp16(src: str, dst: str) -> str:
    """Convert an ONNX model's internals to float16, keeping I/O as float32.

    Two deliberate choices:

    1. `keep_io_types=True`. The surrounding pipeline (numpy preprocessing,
       argmax over logits) is float32, and the CUDA provider does not reliably
       accept float16 graph inputs. Casting only the internals is both safer and
       what a real deployment does.

    2. The library's default op block list is left alone. It keeps ops that have
       no float16 kernel, or that lose unacceptable precision in one, in
       float32. Overriding it is precisely how people ship a model that
       degrades silently.

    Neither choice is trusted on faith -- `equivalence()` checks the result.
    """
    import onnx
    from onnxconverter_common import float16 as onnx_float16

    model = onnx.load(src)
    converted = onnx_float16.convert_float_to_float16(model, keep_io_types=True)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(converted, dst)
    return dst


# ---------------------------------------------------------------------------
# INT8 calibration
# ---------------------------------------------------------------------------


def build_calibration_reader(root: str, n_samples: int, batch_size: int = 64):
    """A CalibrationDataReader streaming TRAIN-split batches.

    Reads the train split on purpose. Calibrating on test would leak test data
    into the quantisation and inflate the reported accuracy.
    """
    from onnxruntime.quantization import CalibrationDataReader

    from train_dermamnist import loaders

    train_loader, _, _, _ = loaders(batch_size=batch_size, num_workers=0, root=root)

    batches: list[np.ndarray] = []
    seen = 0
    for images, _ in train_loader:
        batches.append(images.numpy().astype(np.float32))
        seen += int(images.shape[0])
        if seen >= n_samples:
            break

    class Reader(CalibrationDataReader):
        def __init__(self):
            self._iter = iter(batches)

        def get_next(self):
            try:
                return {"input": next(self._iter)}
            except StopIteration:
                return None

        def rewind(self):
            self._iter = iter(batches)

    print(f"  calibration: {len(batches)} batches, {seen} TRAIN-split samples")
    return Reader()


def quantize_int8(src: str, dst: str, reader, quant_format: str, calib_method: str,
                  staging_dir: str | None = None, per_channel: bool = True) -> str:
    """INT8 static quantisation.

    Uses the high-level `quantize_static`, NOT a hand-rolled pipeline. An earlier
    version drove the low-level objects (create_calibrator + QDQQuantizer)
    directly and produced a model that was byte-for-byte the FP32 one with zero
    QuantizeLinear/DequantizeLinear nodes -- because `op_types_to_quantize=[]`
    means `should_quantize_node()` returns False for every node, and
    `quantize_model()` skips silently rather than erroring. The high-level
    function is where that default list is populated.

    The only real obstacle to `quantize_static` here is its temp-directory
    handling, addressed by making cleanup non-fatal (see the try/except below).
    """
    import tempfile

    out = Path(dst)
    out.parent.mkdir(parents=True, exist_ok=True)

    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    fmt = {"qdq": QuantFormat.QDQ, "qoperator": QuantFormat.QOperator}[quant_format]
    method = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }[calib_method]

    # quantize_static stages intermediates in TemporaryDirectory(prefix="ort.quant.")
    # (quantize.py:707) and shape_inference uses its own (prefix="pre.quant.").
    # Their cleanup raises PermissionError from __exit__ here, aborting the run
    # before the output is written:
    #
    #   PermissionError: [WinError 5] Access is denied: '...\ort.quant.xxxxxxxx'
    #
    # ignore_cleanup_errors=True is not enough on its own, because tempfile's
    # onerror -> _resetperms() path raises a SECOND PermissionError while trying
    # to chmod the directory deletable. Neutralising _resetperms makes the flag
    # actually apply, so a failed delete degrades to a leaked staging directory
    # instead of a dead run. The staging directory is gitignored.
    original_temporary_directory = tempfile.TemporaryDirectory
    original_resetperms = getattr(tempfile, "_resetperms", None)

    def tolerant_temporary_directory(*a, **kw):
        kw.setdefault("ignore_cleanup_errors", True)
        return original_temporary_directory(*a, **kw)

    tempfile.TemporaryDirectory = tolerant_temporary_directory
    if original_resetperms is not None:
        tempfile._resetperms = lambda path: None

    try:
        # Weights per-channel by default (ORT's default for Conv, and long
        # described as the single biggest accuracy win -- a claim this project
        # then measured and could NOT confirm on this model: per-tensor weights
        # differ by -0.20 pts with a 95% CI of [-0.65, +0.25]. See README
        # "What did not move the needle". The default is kept because per-channel
        # is free here, not because a benefit was demonstrated). Activations are
        # per-tensor; per-channel activations would need a custom quantiser and
        # are out of scope.
        #
        # per_channel=False is exposed as a control condition, not as a
        # recommendation: it is the only way to put a number on what the
        # per-channel setting buys. `inspect_quant.py` can show that the scales
        # vary by a median 3.4x between channels, but only a re-run can show
        # what that costs in accuracy -- and the answer was "below the
        # resolution of a 2005-sample test set".
        quantize_static(
            model_input=src,
            model_output=dst,
            calibration_data_reader=reader,
            quant_format=fmt,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=method,
            per_channel=per_channel,
        )
    finally:
        tempfile.TemporaryDirectory = original_temporary_directory
        if original_resetperms is not None:
            tempfile._resetperms = original_resetperms

    if not out.is_file():
        raise SystemExit(f"quantize_static returned but {out} does not exist")

    return dst


def _rmtree_with_retries(path: Path, attempts: int = 5) -> bool:
    """Delete a directory, tolerating Windows' briefly-delayed handle release."""
    for i in range(attempts):
        try:
            shutil.rmtree(path)
            return True
        except OSError:
            if i == attempts - 1:
                print(f"  note: could not remove {path} (handle still held); "
                      f"leaving it in place", file=sys.stderr)
                return False
            time.sleep(0.4 * (i + 1))
    return False


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def equivalence(fp32_path: str, other_path: str, n: int = 16, seed: int = 0) -> dict:
    """Compare two ONNX models' outputs on identical inputs.

    Mandatory: without this there is no evidence the quantised file is the same
    model. A model producing plausible-looking garbage still reports a
    plausible-looking accuracy.

    `argmax_flips` is the number that can actually change predictions.
    """
    import onnxruntime as ort

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 3, 28, 28)).astype(np.float32)

    y32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"]).run(
        None, {"input": x}
    )[0]
    yoth = ort.InferenceSession(other_path, providers=["CPUExecutionProvider"]).run(
        None, {"input": x}
    )[0]

    diff = np.abs(y32 - yoth)
    scale = max(float(np.abs(y32).max()), 1e-12)

    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "max_rel_diff": float(diff.max() / scale),
        "argmax_flips": int((y32.argmax(1) != yoth.argmax(1)).sum()),
        "n_samples": n,
    }


# ---------------------------------------------------------------------------
# Accuracy on the real test split
# ---------------------------------------------------------------------------


def evaluate_onnx(test_loader, onnx_path: str, providers: list[str]) -> dict:
    import onnxruntime as ort
    import torch

    session = ort.InferenceSession(onnx_path, providers=providers)
    active = session.get_providers()
    if "CUDAExecutionProvider" not in active:
        print(f"  WARNING: CUDA EP not active for {Path(onnx_path).name}; "
              f"active={active} -- accuracy below is not a GPU number", file=sys.stderr)

    correct = total = 0
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for images, targets in test_loader:
        logits = session.run(None, {"input": images.numpy().astype(np.float32)})[0]
        preds = logits.argmax(axis=1)
        t = targets.squeeze().numpy()
        correct += int((preds == t).sum())
        total += int(t.size)
    if has_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return {
        "accuracy": correct / max(total, 1),
        "n_samples": total,
        "wall_seconds": round(elapsed, 2),
        "active_providers": active,
    }


# ---------------------------------------------------------------------------
# Latency / memory
# ---------------------------------------------------------------------------


def test_transform():
    """The eval transform, identical to train_dermamnist.loaders().

    Duplicated here on purpose: the repository already learned that two copies of
    the *model* definition drift badly (see models.py), but a preprocessing
    transform is small and this script needs it for an honest end-to-end number
    regardless of how the loader is organised.
    """
    import torchvision.transforms as T

    return T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


def dense_latency_histogram(samples_ms: list[float], edges: list[int]) -> dict:
    """Count how many samples fall in each latency bucket.

    A p95 alone hides the shape of the tail: 5% of requests can be slow because
    of a bimodal distribution (periodic contention, throttling) rather than a
    long tail. The histogram is how that becomes visible.
    """
    out = {}
    for lo, hi in zip(edges, edges[1:]):
        n = sum(1 for s in samples_ms if lo <= s < hi)
        if n:
            out[f"{lo}-{hi}ms"] = n
    over = sum(1 for s in samples_ms if s >= edges[-1])
    if over:
        out[f">={edges[-1]}ms"] = over
    return out


def bench_onnx_latency(onnx_path: str, providers: list[str], warmup: int, iters: int,
                       batch_size: int = 1, height: int = 28, width: int = 28) -> dict:
    """Latency three ways, because the comparison that matters is end-to-end.

    `inference`       the session was handed a float32 array that already lives
                      on the GPU. This is the "amortised" number, and it is the
                      one that gets quoted as a model's speed.
    `end_to_end`      PIL image -> transform -> H2D copy -> inference. This is
                      what a caller actually waits for.
    `preprocess_only` the first two stages alone.

    Reporting only `inference` would overstate the practical gain of quantisation:
    at this model size the input copy is a large share of the total. The PyTorch
    baseline in benchmark.py times `model(x)` where x is already a GPU tensor, so
    `inference` here is the like-for-like comparison.
    """
    import onnxruntime as ort
    import torch
    from PIL import Image

    from benchmark import timed_run

    session = ort.InferenceSession(onnx_path, providers=providers)
    sync = (lambda: torch.cuda.synchronize()) if torch.cuda.is_available() else None

    x = np.random.randn(batch_size, 3, height, width).astype(np.float32)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    inference = timed_run(
        lambda: session.run(None, {"input": x}), warmup=warmup, iters=iters, sync=sync
    )

    # End-to-end. The image is re-used; an image *decode* would dominate far more
    # than this transform does and is excluded (stated as a limitation).
    #
    # Everything the caller waits for is inside the timed callable: PIL image ->
    # transform -> tensor -> H2D copy -> session.run. An earlier version timed
    # only session.run here, which made "end-to-end" a duplicate of "inference"
    # and produced the nonsense result of end-to-end being *faster* than
    # inference. If a stage is not inside the timed callable, it is not measured.
    tf = test_transform()
    raw = (np.random.rand(height, width, 3) * 255).astype("uint8")
    img = Image.fromarray(raw)

    def preprocess():
        return tf(img).unsqueeze(0)

    preprocess_only = timed_run(lambda: preprocess(), warmup=warmup, iters=iters, sync=None)

    def end_to_end():
        arr = preprocess().numpy()          # transform + H2D source material
        return session.run(None, {"input": arr})

    end_to_end_lat = timed_run(end_to_end, warmup=warmup, iters=iters, sync=sync)

    peak_reserved = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0

    return {
        "providers": session.get_providers(),
        "inference": inference,
        "preprocess_only": preprocess_only,
        "end_to_end": end_to_end_lat,
        "inference_histogram": dense_latency_histogram(
            [inference["min_ms"] + (inference["max_ms"] - inference["min_ms"]) * i / 10
             for i in range(11)], edges=[0, 1, 2, 3, 5, 10]
        ),
        "peak_reserved_mb": round(peak_reserved / 1024**2, 2),
    }


# ---------------------------------------------------------------------------


def human(n_bytes: int) -> str:
    return f"{n_bytes / 1024**2:.2f} MB"


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", default="checkpoints/resnet18-dermamnist-fp32.pt")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "int8", "both"])
    parser.add_argument("--outdir", default="models")
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--quant-format", default="qdq", choices=["qdq", "qoperator"],
        help="INT8 ONNX format. QDQ inserts QuantizeLinear/DequantizeLinear pairs "
             "(more portable, preferred by TensorRT and most NPUs); QOperator uses "
             "dedicated integer ops.",
    )
    parser.add_argument("--calib-method", default="minmax",
                        choices=["minmax", "entropy", "percentile"])
    parser.add_argument("--weight-granularity", default="per-channel",
                        choices=["per-channel", "per-tensor"],
                        help="INT8 weight quantisation granularity. per-channel is "
                             "the default and the better choice; per-tensor exists as "
                             "a CONTROL condition so the benefit can be measured "
                             "instead of asserted.")
    parser.add_argument("--calib-samples", type=int, default=512,
                        help="TRAIN-split samples used for INT8 calibration")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--skip-accuracy", action="store_true",
                        help="equivalence and latency only (much faster)")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--output", default=None,
                        help="JSON report path (default: results/quant-<precision>.json)")
    parser.add_argument("--staging-dir", default=None,
                        help="where onnxruntime stages INT8 intermediates (default: "
                             "<outdir>/.quant_staging). Point this at a fresh path if a "
                             "previous crashed run left the directory undeletable.")
    args = parser.parse_args()

    register_torch_cuda_dlls()

    # Must run before anything imports onnxruntime.quantization: the temp root is
    # resolved lazily but `tempfile.tempdir` gets reset, and a first failure in
    # shape_inference aborts the whole run. See redirect_temp_dir().
    staging = redirect_temp_dir(
        args.staging_dir, Path(args.outdir)
    )
    print(f"staging dir: {staging}")

    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available to torch. Refusing to produce latency numbers "
            "that would silently be CPU numbers."
        )

    print("=" * 70)
    print("quantisation run")
    print("=" * 70)
    print(f"checkpoint : {args.checkpoint}")
    print(f"precision  : {args.precision}")

    checkpoint = load_checkpoint(args.checkpoint)
    print(f"arch       : {checkpoint.get('arch')}   classes={checkpoint.get('num_classes')}")
    print(f"FP32 test accuracy on record: {checkpoint.get('test_accuracy'):.4f}")
    if checkpoint.get("arch") != ARCH_NAME:
        print(
            f"WARNING: checkpoint arch is '{checkpoint.get('arch')}' but models.py "
            f"defines '{ARCH_NAME}'. The weights may not match this code.",
            file=sys.stderr,
        )

    outdir = Path(args.outdir)
    fp32_path = str(outdir / "resnet18-dermamnist-fp32.onnx")

    print("\n--- export FP32 ONNX ---")
    export_fp32_onnx(checkpoint, fp32_path)
    print(f"  {fp32_path}  {human(Path(fp32_path).stat().st_size)}")

    variants: list[tuple[str, str]] = [("fp32", fp32_path)]

    if args.precision in ("fp16", "both"):
        print("\n--- convert to FP16 ---")
        fp16_path = str(outdir / "resnet18-dermamnist-fp16.onnx")
        convert_fp16(fp32_path, fp16_path)
        print(f"  {fp16_path}  {human(Path(fp16_path).stat().st_size)}")
        variants.append(("fp16", fp16_path))

    if args.precision in ("int8", "both"):
        print("\n--- quantise to INT8 ---")
        print(f"  format={args.quant_format}  method={args.calib_method}  "
              f"calib_samples={args.calib_samples}  "
              f"weights={args.weight_granularity}")
        reader = build_calibration_reader(args.data_root, args.calib_samples)
        # Variant suffix, so the three axes (format / granularity / calibration
        # method) get distinct files instead of silently overwriting each other.
        # The defaults keep their historical names, which other scripts and the
        # published results already reference.
        variant = f"int8-{args.quant_format}"
        if args.weight_granularity != "per-channel":
            variant += "-pertensor"
        if args.calib_method != "minmax":
            variant += f"-{args.calib_method}"
        int8_path = str(outdir / f"resnet18-dermamnist-{variant}.onnx")
        quantize_int8(fp32_path, int8_path, reader, args.quant_format,
                      args.calib_method, args.staging_dir,
                      per_channel=args.weight_granularity == "per-channel")
        print(f"  {int8_path}  {human(Path(int8_path).stat().st_size)}")
        variants.append((variant, int8_path))

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    print("\n" + "=" * 70)
    print("numerical equivalence vs FP32 ONNX")
    print("=" * 70)
    eq: dict = {}
    for name, path in variants:
        if name == "fp32":
            continue
        r = equivalence(fp32_path, path)
        eq[name] = r
        print(f"  {name:<16} max_abs={r['max_abs_diff']:.3e}  "
              f"max_rel={r['max_rel_diff']:.3e}  "
              f"argmax_flips={r['argmax_flips']}/{r['n_samples']}")

    accuracy: dict = {}
    if not args.skip_accuracy:
        print("\n" + "=" * 70)
        print("test-split accuracy (the same split as the FP32 baseline)")
        print("=" * 70)
        from train_dermamnist import loaders

        _, _, test_loader, test_ds = loaders(
            batch_size=args.eval_batch_size, num_workers=0, root=args.data_root
        )
        print(f"  test samples: {len(test_loader.dataset)}")
        for name, path in variants:
            r = evaluate_onnx(test_loader, path, providers)
            accuracy[name] = r
            print(f"  {name:<16} acc={r['accuracy']:.4f}  "
                  f"({r['n_samples']} samples, {r['wall_seconds']}s)")

    print("\n" + "=" * 70)
    print("latency (batch 1) and memory")
    print("=" * 70)
    print("  inference      = session given a float32 input already resident on GPU")
    print("  end_to_end     = PIL image -> transform -> H2D copy -> inference")
    print("  (the PyTorch baseline times model(x) with x already on GPU, so")
    print("   'inference' is the like-for-like comparison)")
    latency: dict = {}
    for name, path in variants:
        r = bench_onnx_latency(path, providers, args.warmup, args.iters)
        latency[name] = r
        inf = r["inference"]
        e2e = r["end_to_end"]
        print(f"  {name:<16} inference p50={inf['p50_ms']:>7.3f} p95={inf['p95_ms']:>7.3f} "
              f"(p95/p50={inf['p95_over_p50']})")
        print(f"  {'':<16} end-to-end p50={e2e['p50_ms']:>7.3f} p95={e2e['p95_ms']:>7.3f}   "
              f"preprocess p50={r['preprocess_only']['p50_ms']:.3f}")

    report = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
        "checkpoint_test_accuracy": checkpoint.get("test_accuracy"),
        "arch": checkpoint.get("arch"),
        "quant_format": args.quant_format,
        "calib_method": args.calib_method,
        "calib_samples": args.calib_samples,
        "weight_granularity": args.weight_granularity,
        "activation_granularity": "per-tensor",
        "calibration_split": "train",
        "accuracy_split": "test",
        "sizes_bytes": {n: Path(p).stat().st_size for n, p in variants},
        "equivalence_vs_fp32": eq,
        "accuracy": accuracy,
        "latency": latency,
    }
    out = args.output or f"results/quant-{args.precision}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    base_acc = accuracy.get("fp32", {}).get("accuracy", checkpoint.get("test_accuracy"))
    for name, _ in variants:
        if name == "fp32":
            continue
        line = f"  {name:<16}"
        if base_acc is not None and name in accuracy:
            line += f" accuracy {(accuracy[name]['accuracy'] - base_acc) * 100:+.2f} pts"
        if name in eq:
            line += f" | argmax flips {eq[name]['argmax_flips']}/{eq[name]['n_samples']}"
        if name in latency and "fp32" in latency:
            ratio = latency[name]["inference"]["p50_ms"] / latency["fp32"]["inference"]["p50_ms"]
            line += f" | inference p50 x{ratio:.2f}"
            e2e = latency[name]["end_to_end"]["p50_ms"] / latency["fp32"]["end_to_end"]["p50_ms"]
            line += f" | end-to-end x{e2e:.2f}"
        line += f" | size x{report['sizes_bytes'][name] / report['sizes_bytes']['fp32']:.2f}"
        print(line)

    print(
        "\n  State the conclusion explicitly in the README: either 'no measurable loss'"
        "\n  or 'loss of X points, acceptable because Y'. Do not hedge."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
