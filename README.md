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
| FP32 baseline — accuracy | ⬜ not run yet |
| FP32 baseline — latency / VRAM | ⬜ not run yet |
| ONNX export + numerical equivalence check | ⬜ not started |
| FP16 / INT8 quantisation | ⬜ not started |
| LLM structured-output stage | ⬜ not started |
| Service (FastAPI) + packaging | ⬜ not started |

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

# 3. (later) the ONNX path
python benchmark.py --onnx models/resnet18-dermamnist-fp32.onnx \
    --iters 200 --warmup 20 --label onnx-fp32 \
    --output results/onnx-fp32.json
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

> ⬜ **Not measured yet.** Run step 1 and 2 above, then fill this in.
> Reported accuracy is test-split accuracy from the best-val checkpoint, measured
> once. Do not fill this table from memory or from a paper.

| Metric | Value |
|---|---|
| Test accuracy | _pending_ |
| Latency p50 (ms) | _pending_ |
| Latency p95 (ms) | _pending_ |
| Peak VRAM — allocated (MB) | _pending_ |
| Peak VRAM — reserved (MB) | _pending_ |
| Input shape | 1×3×28×28 |
| Precision | FP32 |
| Trained from scratch or ImageNet-pretrained? | **from scratch** (`weights=None`) |

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
train_dermamnist.py   fine-tune ResNet-18 on DermaMNIST -> FP32 checkpoint + test accuracy
benchmark.py          latency (mean/p50/p95), peak VRAM, environment snapshot -> JSON
data/                 MedMNIST downloads            (gitignored, re-downloadable)
checkpoints/          trained weights                (gitignored)
models/               ONNX exports                   (gitignored)
results/              benchmark JSON output          (gitignored)
```

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
