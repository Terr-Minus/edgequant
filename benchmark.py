#!/usr/bin/env python
"""Inference benchmark harness: measures accuracy-agnostic runtime metrics.

Day-2 deliverable of the quantized-inference project. Measures the three
numbers every later comparison depends on:

  * latency      -- mean / p50 / p95 / min / max over N timed iterations
  * memory       -- peak GPU allocated + reserved bytes
  * environment  -- torch / CUDA / GPU / driver versions (without these the
                    numbers cannot be reproduced later, so the file is useless)

Results are written as structured JSON so that later FP16/INT8 runs can be
tabulated automatically instead of copied by hand.

Usage
-----
    python benchmark.py --warmup 10 --iters 100

    python benchmark.py --model yolov8n.pt --input samples/ --iters 200 \
        --output results/baseline-fp32.json

    python benchmark.py --onnx model.onnx --input samples/      # ONNX Runtime path

Notes
-----
This file deliberately avoids the two most common methodology mistakes:

  1. Timing a single pass, or reporting only the mean. p95 is what a reviewer
     will ask about, and it is routinely 2-3x the mean.
  2. Forgetting torch.cuda.synchronize(). CUDA kernels are asynchronous, so
     timing without a synchronize measures queue-submission time, not compute.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ----------------------------------------------------------------------------
# Environment capture
# ----------------------------------------------------------------------------


def device_memory_snapshot() -> dict:
    """Whole-device VRAM usage, read from nvidia-smi.

    This is deliberately kept separate from the per-process numbers reported by
    torch. Rationale, because the distinction matters when reading the output:

      * torch.cuda.max_memory_allocated() counts only what THIS process asked
        PyTorch for. Other applications living on the same GPU do not inflate it.
      * nvidia-smi 'memory.used' is whole-device, so browser and desktop
        compositor VRAM IS included.

    Recording the whole-device figure at start and end lets a reader confirm two
    things: that this process's own allocations were the dominant consumer, and
    that the device never got close to OOM (a near-full device can change
    allocation behaviour and thus latency).

    Parsed from nvidia-smi text rather than through pynvml/torch, because those
    helpers are inconsistently present and torch may or may not be installed.
    """
    snapshot: dict = {}
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if out.returncode == 0 and out.stdout.strip():
            parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
            if len(parts) >= 4:
                snapshot["device_total_mb"] = int(parts[0])
                snapshot["device_used_mb"] = int(parts[1])
                snapshot["device_free_mb"] = int(parts[2])
                snapshot["device_utilization_pct"] = int(parts[3])
    except (OSError, subprocess.SubprocessError, ValueError):
        pass

    if not snapshot:
        # Fallback: free/total is still available through torch.
        try:
            import torch

            if torch.cuda.is_available():
                free_b, total_b = torch.cuda.mem_get_info()
                snapshot["device_total_mb"] = round(total_b / 1024**2)
                snapshot["device_free_mb"] = round(free_b / 1024**2)
                snapshot["device_used_mb"] = round((total_b - free_b) / 1024**2)
        except ImportError:
            pass

    return snapshot


def collect_environment() -> dict:
    """Record everything needed to reproduce these numbers later."""
    env: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }

    try:
        import torch

        env["torch"] = torch.__version__
        env["torch_cuda_build"] = torch.version.cuda
        env["cudnn"] = torch.backends.cudnn.version()
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            env["gpu_name"] = props.name
            env["gpu_total_memory_bytes"] = props.total_memory
            env["gpu_capability"] = f"{props.major}.{props.minor}"
            env["gpu_count"] = torch.cuda.device_count()
    except ImportError:
        env["torch"] = None

    try:
        import onnxruntime as ort

        env["onnxruntime"] = ort.__version__
        env["onnxruntime_providers"] = ort.get_available_providers()
    except ImportError:
        env["onnxruntime"] = None

    return env


def _register_torch_cuda_dlls() -> list[str]:
    """Make the CUDA/cuDNN DLLs bundled inside torch visible to ONNX Runtime.

    Why this exists (it is not cosmetic -- it silently corrupts measurements):

    On Windows, onnxruntime's onnxruntime_providers_cuda.dll needs cublasLt64_12.dll,
    cudnn64_9.dll and friends. torch ships all of them inside `<site-packages>/torch/lib`,
    but that directory is NOT on the process DLL search path unless something adds it.
    torch itself finds its own DLLs; onnxruntime does not.

    When the load fails, onnxruntime does NOT raise. It logs an error and quietly
    falls back to the CPU provider. `ort.get_available_providers()` still lists
    CUDAExecutionProvider, so the failure is invisible unless you check
    `session.get_providers()` -- which is exactly how you end up publishing a
    "GPU" latency number that was measured on the CPU.

    Returns the directory that was registered, or an empty list if not found.
    """
    import os
    import sys
    from pathlib import Path

    candidates = []
    try:
        import torch

        candidates.append(Path(torch.__file__).parent / "lib")
    except ImportError:
        pass

    registered: list[str] = []
    for directory in candidates:
        if not directory.is_dir():
            continue
        os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
        if sys.version_info >= (3, 8) and hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(directory))
            except OSError:
                pass
        registered.append(str(directory))

    return registered


# ----------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------


def timed_run(fn, warmup: int, iters: int, sync=None) -> dict:
    """Time `fn` with a proper warmup and per-iteration synchronization.

    Returns millisecond statistics over the `iters` measured runs.
    """
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()

    samples_ms: list[float] = []
    for _ in range(iters):
        if sync is not None:
            sync()
        start = time.perf_counter()
        fn()
        if sync is not None:
            sync()  # without this, we would time only kernel dispatch
        samples_ms.append((time.perf_counter() - start) * 1000.0)

    ordered = sorted(samples_ms)

    def pct(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        # nearest-rank percentile, adequate for the sample sizes used here
        idx = min(len(ordered) - 1, int(round(p / 100.0 * (len(ordered) - 1))))
        return ordered[idx]

    return {
        "iters": len(samples_ms),
        "warmup": warmup,
        "mean_ms": round(statistics.fmean(samples_ms), 3),
        "p50_ms": round(pct(50), 3),
        "p95_ms": round(pct(95), 3),
        "min_ms": round(ordered[0], 3),
        "max_ms": round(ordered[-1], 3),
        "stdev_ms": round(statistics.pstdev(samples_ms), 3) if len(samples_ms) > 1 else 0.0,
        # p95/p50 ratio > ~1.5 usually means warmup was insufficient,
        # another process shares the GPU, or thermal throttling is kicking in
        "p95_over_p50": round(pct(95) / pct(50), 3) if pct(50) > 0 else None,
    }


# ----------------------------------------------------------------------------
# PyTorch path
# ----------------------------------------------------------------------------


def load_torch_model(path: str, device):
    """Load a TorchScript archive OR a train_dermamnist.py state_dict checkpoint.

    Both formats end in `.pt`, so the extension cannot be trusted to tell them
    apart. TorchScript is attempted first; on failure we fall back to rebuilding
    the ResNet-18 28x28 architecture and loading the state dict.

    Returns (model, checkpoint_metadata). Metadata is empty for TorchScript.
    """
    import torch

    try:
        return torch.jit.load(path, map_location=device), {}
    except (RuntimeError, ValueError):
        # Not TorchScript. Older torch raises RuntimeError; some versions raise
        # ValueError for a non-archive file.
        pass

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # noqa: BLE001 - surfaced with context below
        raise SystemExit(
            f"Could not load {path} as TorchScript or as a checkpoint: {error}"
        )

    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise SystemExit(
            f"{path} is neither a TorchScript module nor a checkpoint with a "
            "'state_dict' key. Refusing to benchmark an unidentified file."
        )

    import torch.nn as nn
    from torchvision.models import resnet18

    # Architecture must match train_dermamnist.py EXACTLY, or the state dict
    # will not load. Keep these two in sync if either script changes.
    num_classes = int(checkpoint.get("num_classes", 7))
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint


def bench_torch(args) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available to torch. Either install a CUDA build of "
            "PyTorch or pass --device cpu (but then peak GPU memory is "
            "meaningless)."
        )

    device = torch.device(args.device)
    model, checkpoint = load_torch_model(args.model, device)

    if checkpoint:
        # Report the checkpoint's own provenance rather than the filename, so a
        # reader can tell which training run produced these numbers.
        print("loaded checkpoint:")
        for key in ("arch", "dataset", "precision", "split_protocol",
                    "test_accuracy", "num_classes"):
            if key in checkpoint:
                print(f"    {key:<15} {checkpoint[key]}")
    else:
        print("loaded TorchScript module")

    model = model.to(device).eval()
    x = torch.randn(args.batch_size, 3, args.height, args.width, device=device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        latency = timed_run(
            lambda: model(x),
            warmup=args.warmup,
            iters=args.iters,
            sync=lambda: torch.cuda.synchronize(),
        )

    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    # --precision is an explicit user choice; fall back to what the checkpoint
    # recorded (train_dermamnist.py writes "fp32") when the user did not say.
    precision = args.precision or checkpoint.get("precision") or "fp32"

    return {
        "backend": "pytorch",
        "precision": precision,
        "checkpoint_metadata": {
            k: v for k, v in checkpoint.items() if k != "state_dict"
        },
        "input_shape": [args.batch_size, 3, args.height, args.width],
        "latency": latency,
        "memory": {
            "peak_allocated_bytes": peak_alloc,
            "peak_allocated_mb": round(peak_alloc / 1024**2, 2),
            "peak_reserved_bytes": peak_reserved,
            "peak_reserved_mb": round(peak_reserved / 1024**2, 2),
        },
    }


# ----------------------------------------------------------------------------
# ONNX Runtime path
# ----------------------------------------------------------------------------


def bench_onnx(args) -> dict:
    import numpy as np

    # Must happen BEFORE onnxruntime is imported, so the provider DLL can
    # resolve cublas/cudnn at load time. See _register_torch_cuda_dlls().
    dll_dirs = _register_torch_cuda_dlls()

    import onnxruntime as ort

    providers = args.providers or (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(args.onnx, providers=providers)
    input_meta = session.get_inputs()[0]
    name = input_meta.name

    active = session.get_providers()
    if dll_dirs:
        print(f"registered DLL dir(s): {', '.join(dll_dirs)}")
    print(f"requested providers : {providers}")
    print(f"ACTIVE providers    : {active}")
    if "CUDAExecutionProvider" not in active:
        print(
            "\nWARNING: CUDAExecutionProvider is NOT active -- this run is being\n"
            "measured on the CPU. Any latency number here is NOT a GPU number.\n"
            "Check that torch's lib directory holds cublasLt64_12.dll and\n"
            "cudnn64_9.dll, and that this script ran _register_torch_cuda_dlls()\n"
            "before importing onnxruntime.",
            file=sys.stderr,
        )

    shape = [
        args.batch_size if isinstance(d, str) else d for d in input_meta.shape
    ]
    if len(shape) != 4:
        raise SystemExit(
            f"Expected a 4-D NCHW input for this harness, got shape {input_meta.shape}"
        )
    shape[1], shape[2], shape[3] = 3, args.height, args.width

    x = np.random.randn(*shape).astype(np.float32)

    import torch  # only for the synchronize primitive; optional

    latency = timed_run(
        lambda: session.run(None, {name: x}),
        warmup=args.warmup,
        iters=args.iters,
        sync=(lambda: torch.cuda.synchronize()) if torch.cuda.is_available() else None,
    )

    return {
        "backend": "onnxruntime",
        "requested_providers": providers,
        "providers": session.get_providers(),
        "cuda_ep_active": "CUDAExecutionProvider" in session.get_providers(),
        "input_name": name,
        "input_shape": shape,
        "latency": latency,
    }


# ----------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        help="path to a TorchScript .pt module OR a train_dermamnist.py checkpoint "
             "(both use .pt; the loader sniffs which one it is)",
    )
    parser.add_argument("--onnx", help="ONNX model path (onnxruntime path)")
    parser.add_argument("--input", help="input image or folder (metadata only for now)")
    parser.add_argument("--output", default="results/benchmark.json", help="JSON output path")
    parser.add_argument("--iters", type=int, default=100, help="measured iterations")
    parser.add_argument("--warmup", type=int, default=10, help="warmup iterations, untimed")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--height", type=int, default=28,
        help="input height; 28 is the DermaMNIST default (override for other models)",
    )
    parser.add_argument(
        "--width", type=int, default=28,
        help="input width; 28 is the DermaMNIST default (override for other models)",
    )
    parser.add_argument(
        "--precision",
        default=None,
        choices=[None, "fp32", "fp16", "int8"],
        help="override the precision label; by default it is read from the "
             "checkpoint metadata. Label only -- the loaded model decides the truth",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--providers", nargs="*", help="ONNX Runtime providers in priority order")
    parser.add_argument("--label", default=None, help="tag for this run, e.g. baseline-fp32")
    args = parser.parse_args()

    if not args.model and not args.onnx:
        parser.error("pass either --model (torch) or --onnx (onnxruntime)")

    result: dict = {
        "label": args.label or args.precision or "fp32",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "environment": collect_environment(),
        # captured before any model allocation, so a reader can verify that
        # unrelated desktop/browser VRAM did not dominate the device
        "device_memory_before_mb": device_memory_snapshot(),
    }

    try:
        if args.onnx:
            result.update(bench_onnx(args))
        else:
            result.update(bench_torch(args))
    except Exception as error:  # noqa: BLE001 - report and persist partial info
        result["error"] = f"{type(error).__name__}: {error}"
        _write(args.output, result)
        print(f"FAILED: {result['error']}", file=sys.stderr)
        return 1

    result["device_memory_after_mb"] = device_memory_snapshot()
    _write(args.output, result)
    _report(result)
    return 0


def _write(path_str: str, payload: dict) -> None:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {path}")


def _report(result: dict) -> None:
    env = result.get("environment", {})
    lat = result.get("latency", {})
    mem = result.get("memory", {})

    print("\n--- environment ---")
    for key in ("python", "torch", "torch_cuda_build", "cudnn", "gpu_name", "onnxruntime"):
        if env.get(key) is not None:
            print(f"  {key:<18} {env[key]}")

    print("\n--- latency (ms) ---")
    for key in ("mean_ms", "p50_ms", "p95_ms", "min_ms", "max_ms", "stdev_ms", "p95_over_p50"):
        if lat.get(key) is not None:
            print(f"  {key:<18} {lat[key]}")

    if mem:
        print("\n--- memory (this process, via torch) ---")
        for key in ("peak_allocated_mb", "peak_reserved_mb"):
            if mem.get(key) is not None:
                print(f"  {key:<18} {mem[key]}")

    before = result.get("device_memory_before_mb") or {}
    after = result.get("device_memory_after_mb") or {}
    if before.get("device_used_mb") is not None:
        print("\n--- whole-device VRAM (via nvidia-smi, includes desktop/browser) ---")
        print(f"  before             {before.get('device_used_mb')} / {before.get('device_total_mb')} MB used")
        if after.get("device_used_mb") is not None:
            print(f"  after              {after.get('device_used_mb')} / {after.get('device_total_mb')} MB used")
        if before.get("device_total_mb") and before.get("device_free_mb") is not None:
            free_pct = 100.0 * before["device_free_mb"] / before["device_total_mb"]
            print(f"  free before        {before['device_free_mb']} MB  ({free_pct:.0f}%)")
            if free_pct < 25:
                print(
                    "  NOTE: less than 25% of device VRAM was free before the run. "
                    "Peak figures may be affected by allocation pressure; close "
                    "other GPU consumers and re-measure if these numbers matter."
                )

    ratio = lat.get("p95_over_p50")
    if ratio is not None and ratio > 1.5:
        print(
            f"\nWARNING: p95/p50 = {ratio}. That is high. Common causes: warmup too "
            "short, another process using the GPU, or thermal throttling. "
            "Increase --warmup before trusting these numbers."
        )


if __name__ == "__main__":
    sys.exit(main())
