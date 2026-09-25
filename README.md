# edgequant

**A local-only health self-check inference service, with a focus on quantisation measurements.**

Input an image → a CV model classifies it → a small LLM turns the result into a
structured description. Nothing leaves the machine.

> The deliverable this repository really exists for is **not the application**.
> It is the measured trade-off table: what FP16 and INT8 cost you in accuracy,
> latency and VRAM. Every number below is produced by a script in this repo.

---

## Status

| Stage | State |
|---|---|
| Environment (torch cu121, onnxruntime-gpu, medmnist) | ✅ done |
| FP32 baseline — accuracy | ✅ **0.7651 test accuracy** (2026-09-25) |
| FP32 baseline — latency / VRAM | ✅ **p50 2.26 ms / 60.1 MB peak** (2026-09-25) |
| ONNX export + numerical equivalence check | ✅ **exported, max abs diff 1.4e-06 (CPU) / 2.6e-04 (CUDA)** |
| FP16 quantisation | ✅ **21.31 MB, no accuracy loss, no latency gain** |
| INT8 quantisation (QDQ, MinMax, per-channel weights) | ✅ **10.77 MB, no accuracy loss, 1.6× SLOWER** |
| LLM structured-output stage | ⬜ not started |
| Service (FastAPI) + packaging | ⬜ not started |

**Headline result: quantisation bought 4× smaller files and cost nothing in
accuracy — and did not make inference faster.** On this GPU + ONNX Runtime
combination, INT8 is 1.6× slower than FP32. See the trade-off table below; this
is the project's most interesting finding and it is a negative one.

*(This table is updated by hand as stages complete. Nothing here is claimed
before it has been measured.)*

---

## The task

Image classification on **DermaMNIST** — 28×28 dermoscopic images, 7 classes,
from the [MedMNIST](https://medmnist.com/) benchmark collection.

DermaMNIST was chosen over the clinical alternatives for one practical reason:
it is downloadable with `pip install medmnist`, with **no registration, no data
agreement, no approval process**, and it ships an official train/val/test split.
Most medical imaging datasets require an application process, which would stall
the project before any measurement could happen. Licence: CC BY 4.0.

## The model

**ResNet-18 adapted for 28×28 input.**

Stock ResNet-18 is designed for 224×224: a 7×7 stride-2 convolution followed by
a 3×3 stride-2 max-pool downsamples by 4× *before the first residual stage*. On a
28×28 image that leaves a 7×7 feature map, discarding most of the detail. The
standard CIFAR-style adaptation replaces the stem and drops the max-pool:

```python
model = resnet18(weights=None)
model.conv1   = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)  # 7x7/s2 -> 3x3/s1
model.maxpool = nn.Identity()                                        # drop maxpool
model.fc      = nn.Linear(model.fc.in_features, 7)                   # 1000 -> 7
```

ResNet-18 was chosen because its ONNX/INT8 path is well-trodden, so the
quantisation results reflect quantisation rather than toolchain archaeology.
Classification was chosen over detection deliberately: no NMS or other
post-processing in the pipeline means one less variable in the comparison.

---

## Data-split discipline

This is the rule that makes the numbers mean anything:

| Split | Used for |
|---|---|
| **train** | fine-tuning, and (later) INT8 calibration |
| **val** | selecting the best epoch |
| **test** | **final accuracy reporting only — exactly once** |

**Using the test split for quantisation calibration is data leakage.** It inflates
the reported accuracy and is a classic, easily-detected mistake. The calibration
set must come from train or val.

---

## Reproduce

Requires an NVIDIA GPU with a CUDA 12.x driver. Verified on an RTX 3080 Ti
(12 GB, sm_86).

```bash
# environment (conda recommended; see ENVIRONMENT.md for exact versions)
conda create -n edgequant python=3.10 -y
conda activate edgequant
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install onnx onnxruntime-gpu medmnist

# 1. train the FP32 reference model  -> checkpoints/ + results/
python train_dermamnist.py --epochs 10

# 2. measure it                      -> results/baseline-fp32.json
python benchmark.py --model checkpoints/resnet18-dermamnist-fp32.pt \
    --iters 200 --warmup 20 --label baseline-fp32 \
    --output results/baseline-fp32.json

# 3. quantise and measure all three precisions in one run
python quantize.py --precision both          # FP16 + INT8, equivalence + accuracy + latency
python quantize.py --precision fp16           # FP16 only
python quantize.py --precision int8 --quant-format qdq --calib-samples 512
```

`data/`, `checkpoints/`, `results/` are gitignored. The dataset re-downloads on
first run; the numbers you cite should be committed to the README and the report
rather than the raw JSON left unversioned.

### Windows notes that cost real time

Two failures in this project look like something else entirely:

1. **ONNX Runtime silently runs on the CPU.** `onnxruntime` ships
   `onnxruntime_providers_cuda.dll`, which needs `cublasLt64_12.dll` and
   `cudnn64_9.dll`. Those live in `<site-packages>/torch/lib`, which is *not* on
   the process DLL search path. When the load fails, onnxruntime does not raise —
   it logs and falls back to CPU, while `get_available_providers()` still lists
   `CUDAExecutionProvider`. `benchmark.py` registers the torch lib directory
   before importing onnxruntime and records `cuda_ep_active` in the output JSON.
   **Always check `session.get_providers()`, not `get_available_providers()`.**
2. **`medmnist` cannot create its own default download root.** Omitting `root=`
   raises `RuntimeError: Failed to setup the default root directory` before any
   download. `train_dermamnist.py` passes `root="data"` explicitly
   (`--data-root` to override).

---

## Baseline numbers (FP32 reference)

Measured 2026-09-25 on the machine described in `ENVIRONMENT.md`. Accuracy is
test-split accuracy from the best-val checkpoint, measured once. Raw JSON:
`results/fp32-accuracy.json`, `results/baseline-fp32.json`,
`results/baseline-fp32-idle.json`.

### Accuracy

| Metric | Value |
|---|---|
| **Test accuracy** | **0.7651** |
| Best validation accuracy | 0.7747 |
| Train accuracy | 0.7998 |
| Train − test gap | 0.035 — **no meaningful overfitting** |
| Epochs | 10 (AdamW, lr 1e-3, cosine schedule, seed 0) |
| Training time | 34.2 s on RTX 3080 Ti |
| Initialisation | **from scratch** (`weights=None`), not ImageNet-pretrained |

Validation (0.7747) and test (0.7651) differ by under one point, and train/test
by 3.5 points, so this reference model is stable rather than overfit — the
property the quantisation comparison needs.

### Latency (ms, batch 1, input 1×3×28×28, PyTorch, 200 iters after 20 warmup)

| Metric | Value |
|---|---|
| mean | 2.34 |
| **p50** | **2.26** |
| p95 | 2.74 |
| min | 2.15 |
| max | 3.33 |
| stdev | 0.186 |
| p95/p50 | 1.207 |

### Memory

| Metric | Value | Notes |
|---|---|---|
| Peak allocated (this process) | **60.1 MB** | torch counter; unaffected by other applications |
| Peak reserved (this process) | **86.0 MB** | torch caching allocator |
| **CUDA context overhead** | **≈374 MB** (342–390 across 5 runs) | see below |
| Whole-device before run | 1399–1514 MB used | desktop applications only |
| Whole-device after run | 1789–1856 MB used | context + model + desktop |

**The model is 60 MB, but merely initialising CUDA for it costs about 374 MB —
6.2× the model itself.** Measured five times, stable within 50 MB. Sizing a
deployment from "the model is 60 MB" therefore underestimates the real footprint
by a large factor.

> **Scope limit — do not overstate this in an interview.** The 374 MB figure is
> specific to the NVIDIA desktop driver stack. Edge accelerators (Qualcomm
> Hexagon, Rockchip RKNN, Hailo, mobile Mali/Adreno) do **not** create a CUDA
> context. What transfers is the *category* — every runtime carries a fixed
> initialisation cost that does not shrink with the model — and the *method*:
> measure it on the target, do not assume it. The desktop number is not a
> prediction for an NPU.

### Latency is insensitive to background GPU load (measured)

The same checkpoint was benchmarked with Minecraft (≈25–38% GPU utilisation,
1.79 GB resident) plus browsers running, and again with them closed:

| Condition | p50 (ms) | p95/p50 |
|---|---|---|
| Background GPU load present | 2.257 (mean of 3 runs) | 1.17–1.39 |
| Idle desktop | 2.259 (mean of 2 runs) | 1.21–1.30 |
| **Difference** | **0.002 ms (0.1%)** | — |

A model this small does not contend for the GPU, so latency is stable either way.
Peak per-process memory is unaffected by other applications by construction. The
metric that *does* move is whole-device VRAM — which is why it is recorded
separately rather than blended into the headline numbers.

### ONNX export and numerical equivalence

Exported to `models/resnet18-dermamnist-fp32.onnx` (42.6 MB, opset 17) and
compared with the PyTorch output on the same input:

| Execution provider | max abs difference | Verdict |
|---|---|---|
| CPUExecutionProvider | 1.4e-06 | equivalent |
| CUDAExecutionProvider | 2.6e-04 | equivalent at FP32 working precision (≈1e-7 relative) |

The GPU path differs slightly more because kernels accumulate in a different
order; neither difference could change a predicted class. **This check is
mandatory before any quantised comparison** — without it there is no evidence
that the ONNX model and the PyTorch model are the same model.

### Preprocessing cost

Measured on CPU with the DermaMNIST eval transform (ToTensor + Normalize):

| Stage | Time |
|---|---|
| Preprocessing only | 0.18 ms |
| Inference only | 2.33 ms (CPU EP) / 1.07 ms (CUDA EP) |
| Preprocessing + inference | 2.71 ms (CPU) / 1.12 ms (CUDA) |

Preprocessing is ~5% of end-to-end on the GPU path, and only for an already-
decoded 28×28 array. Decoding and resizing a real photograph costs far more.
This is the overhead excluded from the inference-latency figures above
(limitation #2 below).

---

## Quantisation results: the trade-off table

Produced by `quantize.py`, all in one run, all on the same test split, all timed
with the same harness. Raw JSON: `results/quant-fp16.json`, `results/quant-int8.json`.

| | FP32 | FP16 | INT8 (QDQ) |
|---|---|---|---|
| **File size** | 42.61 MB | **21.31 MB** (½) | **10.77 MB** (¼) |
| **Test accuracy** | 0.7641 | 0.7661 | **0.7656** |
| Accuracy vs FP32 | — | **+0.20 pts** | **+0.15 pts** |
| Inference p50 | 0.920 ms | 0.843 ms | **1.456 ms** |
| Inference vs FP32 | — | 0.92× (slightly faster) | **1.58× SLOWER** |
| End-to-end p50 | 1.008 ms | 0.997 ms | 1.687 ms |
| End-to-end vs FP32 | — | 0.99× | **1.67× SLOWER** |
| Preprocessing | 0.116 ms | 0.116 ms | 0.116 ms |
| max abs output diff | — | 1.6e-03 | 4.5e-01 |
| argmax flips (16 probes) | — | 0 | 0 |
| Runtime EP | CUDA | CUDA | CUDA |

### Conclusion, stated plainly

**Quantisation here bought size, not speed.**

- **FP16: no measurable accuracy loss, half the size, latency unchanged.**
  The 0.92× inference ratio is inside run-to-run noise; treat it as "no
  change". FP16 is a free 2× size reduction for this model.

- **INT8: no measurable accuracy loss, a quarter of the size, and 1.6× slower.**
  The accuracy result is real and holds up: `max abs diff` of 0.445 on logits
  whose working range is O(1) is the expected magnitude for 8-bit quantisation,
  there were **zero argmax flips** across the equivalence probes, and the test
  accuracy difference is +0.15 points on 2005 samples — i.e. inside noise.
  **The speed regression is real too, and it is the point.**

- **Why INT8 is slower.** INT8 accelerates inference only when the runtime has
  integer kernels that the hardware actually executes. Here the CUDA
  ExecutionProvider runs the QDQ graph as explicit
  QuantizeLinear → Conv → DequantizeLinear sequences, so the model pays
  dequantisation overhead on every layer boundary and gains no integer matmul.
  The quantisation arithmetic still runs on float tensor cores. **The win
  requires TensorRT (or an INT8-native accelerator), which this project
  deliberately excludes by scope** — see the exclusions below.

- **This is the engineering finding worth reporting.** "Quantise the model to
  make it faster" is not a rule; it is a hypothesis that depends on the backend.
  Measured on this stack, the hypothesis is false: 4× smaller, same accuracy,
  1.6× slower. Anyone who reports only the size and accuracy columns is
  reporting half the result.

### Reader's caveats on these numbers

- **p95 is noisy in these runs.** The p95/p50 ratio came out between 1.5 and
  1.8, above the harness's ~1.5 warning threshold, so the tail is not fully
  characterised. The **p50 figures are stable** (FP32 inference p50 has measured
  0.879 / 0.890 / 0.920 ms across runs) and the conclusions rest on p50. A
  1000-iteration re-run with a longer warmup would tighten p95.
- **Single run per precision.** The FP16 and INT8 rows are one measurement each;
  the FP32 row here is one of several. Before quoting the 1.58× figure in a
  document, re-run INT8 two or three times and confirm the direction.
- **Calibration used 512 train-split images, MinMax, per-channel weights.** No
  comparison across calibration methods was done (limitation #7 below).
- **Activations are per-tensor.** Per-channel activations would need a custom
  quantiser and were not attempted.

---

## What this project does NOT claim

Written down deliberately, because a stated limitation is more credible than an
implied capability. Full list in the project plan; the load-bearing ones:

| # | Not verified |
|---|---|
| 1 | **No real target chip.** Quantisation and measurement happen on a desktop GPU (RTX 3080 Ti). Nothing was deployed to or verified on an NPU / edge device. |
| 2 | **No end-to-end latency.** Measured figures are *model inference* latency; image preprocessing and postprocessing are excluded. |
| 3 | **No power measurement.** The key constraint on edge deployment is power, and this project does not measure it (it would be meaningless on a desktop GPU). |
| 4 | **Accuracy does not generalise** beyond 28×28 DermaMNIST. It must not be read as clinical or product performance. |
| 5 | **No INT4 or below** (assuming only INT8 is completed). |
| 6 | **No TensorRT comparison** — ONNX Runtime only. TensorRT is typically faster but much heavier to deploy; excluded by scope. |
| 7 | **No real users, no business acceptance.** Self-directed. |
| 8 | **LLM stage is a simplified integration**, not a production system; no fine-tuning. |
| 9 | **Hardware-specific numbers.** All figures were measured on one machine at specific software versions (see `ENVIRONMENT.md`); they change elsewhere. |

---

## Repository layout

```
models.py             the architecture, defined ONCE (train + benchmark + quantize share it)
train_dermamnist.py   fine-tune ResNet-18 on DermaMNIST -> FP32 checkpoint + test accuracy
benchmark.py          latency (mean/p50/p95), peak VRAM, environment snapshot -> JSON
quantize.py           ONNX export -> FP16 / INT8 -> equivalence check -> accuracy -> latency
data/                 MedMNIST downloads            (gitignored, re-downloadable)
checkpoints/          trained weights                (gitignored)
models/               ONNX exports, all precisions   (gitignored)
results/              benchmark and quantisation JSON (gitignored)
```

`models.py` exists because `benchmark.py` and `quantize.py` both have to rebuild
the exact architecture to load a `state_dict`, and two copy-pasted definitions
drift until `load_state_dict` fails with confusing missing/unexpected keys. One
definition, imported everywhere.

### Two methodology details a reviewer will ask about

Both are implemented in `benchmark.py` on purpose:

- **`torch.cuda.synchronize()` on both sides of every timed iteration.** CUDA
  kernel launches are asynchronous; timing without a synchronise measures
  *queue-submission* time, not compute time. The resulting numbers look fast and
  are wrong.
- **p95/p50 ratio is reported and warned on.** A ratio above ~1.5 usually means
  warmup was too short, another process was sharing the GPU, or thermal throttling
  was kicking in — i.e. the run is not trustworthy. The script prints a warning
  rather than quietly emitting a suspicious p95.

## Licence / provenance

Code here is original. DermaMNIST is CC BY 4.0, redistributed via the `medmnist`
package. **No employer data, code, or models were used at any point.** No real
patient data and no personal photographs are used, including for demos.
