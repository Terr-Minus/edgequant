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
| INT8 quantisation (QDQ, MinMax, per-channel weights) | ✅ **10.77 MB, no accuracy loss, slower than FP32 in 9/9 trials** |
| INT8 quantisation (QOperator format) | ✅ **10.71 MB, −2.05 pts (statistically real), 78% of nodes execute on the CPU** |
| INT8 format choice (QDQ vs QOperator) | ✅ **settled by measurement** — see "QDQ vs QOperator" below |
| LLM structured-output stage | ⬜ not started |
| Service (FastAPI) + packaging | ⬜ not started |

**Headline result: quantisation bought 4× smaller files and cost nothing in
accuracy — and did not make inference faster.** INT8 (QDQ) was slower than FP32
in every one of nine interleaved trials across three sessions; the ratio ranged
1.48× (quiet machine) to 1.72× (browser and desktop GPU load present). See the
trade-off table below. This is the project's most interesting finding and it is
a negative one.

**The format matters twice over.** The same quantisation parameters, emitted as
QDQ and as QOperator, differ by 2 accuracy points and by 78% of the graph's
placement. ONNX Runtime's CUDA provider implements **no integer kernels at
all**: "INT8 on the GPU" here means float convolution between quantised
boundaries, and the QOperator graph — which does use integer ops — has nowhere
to run them but the CPU. That is measured, not inferred; see the placement probe
below.

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
python quantize.py --precision int8 --quant-format qoperator   # the other INT8 format

# 4. compare everything at once: paired accuracy, node placement, interleaved latency
python compare_models.py --trials 3 --warmup 50 --iters 500

# 5. read the quantisation parameters back out of the files (no GPU needed)
python inspect_quant.py
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

Produced by `quantize.py`, on the same test split, timed with the same harness.
Raw JSON: `results/quant-fp16.json`, `results/quant-int8.json`,
`results/quant-repeatability.json`.

**Latency figures below are from `confirm_quant.py`: 3 trials per precision, run
interleaved (fp32, fp16, int8, fp32, …), 50 warmup / 500 iterations each.** The
single-run numbers from `quantize.py` are in the JSON but are not quoted here —
one measurement is not evidence. Interleaving matters: running all of one
precision and then all of the next lets thermal drift correlate with precision,
which manufactures effects that are not there.

| | FP32 | FP16 | INT8 (QDQ) |
|---|---|---|---|
| **File size** | 42.61 MB | **21.31 MB** (½) | **10.77 MB** (¼) |
| **Test accuracy** | 0.7641 | 0.7661 | **0.7656** |
| Accuracy vs FP32 | — | +0.20 pts | +0.15 pts |
| **Inference p50** (3-trial mean ± sd) | 0.835 ± 0.007 ms | 0.826 ± 0.008 ms | **1.239 ± 0.006 ms** |
| **Inference vs FP32** | — | **0.99×** (no change) | **1.48× SLOWER** (range 1.47–1.50) |
| End-to-end p50 (3-trial mean ± sd) | 1.000 ± 0.005 ms | 0.996 ± 0.005 ms | 1.406 ± 0.007 ms |
| End-to-end vs FP32 | — | 1.00× | 1.41× slower |
| Preprocessing | 0.116 ms | 0.116 ms | 0.116 ms |
| max abs output diff | — | 1.6e-03 | 4.5e-01 |
| argmax flips (16 probes) | — | 0 / 16 | 0 / 16 |
| Runtime EP | CUDA | CUDA | CUDA |

> These three columns are **session 1**, on a quiet desktop. The table further
> down adds the QOperator format, a paired significance test on every accuracy
> number, and the measured node placement — and shows that the INT8 penalty grows
> to ~1.7× once browsers and the desktop compositor are competing for the CPU.
> Read the two tables together; the absolute milliseconds are not comparable
> between sessions, only the ratios within a session are.

### Conclusion, stated plainly

**Quantisation here bought size, not speed.**

- **FP16: half the size, no measurable accuracy or latency change.**
  0.99× across three trials (range 0.97–1.01) is "no change", not "faster".
  FP16 is a free 2× size reduction for this model.

- **INT8: a quarter of the size, no measurable accuracy loss, and slower.**
  Both halves are solid:

  *Accuracy.* `max abs diff` of 0.445 on logits whose working range is O(1) is
  the expected magnitude for 8-bit quantisation. The 16-random-probe
  equivalence check saw **zero argmax flips**, and on the real 2005-sample test
  split only **19 samples change prediction** — a net +3 correct, i.e. +0.15
  points. Tested as a paired difference (McNemar exact p = 0.65, bootstrap 95%
  CI on the difference [−0.25, +0.55] points) that is **inside noise**, which is
  the claim: no measurable loss. The quantised model is structurally genuine:
  42 quantised weight tensors, every one with a per-channel scale, plus 32
  QuantizeLinear / 74 DequantizeLinear nodes.

  *Speed.* INT8 (QDQ) ran slower in **every one of nine interleaved trials
  across three sessions**. The instruction is in the direction, not in a single
  number: session means were 1.48×, 1.59× and 1.72× FP32, and the per-trial
  extremes were 1.47× and 1.91×. See "Why the ratio moves between sessions".

- **Why INT8 is slower — measured, not guessed.** INT8 accelerates inference only
  when the runtime has integer kernels the hardware actually executes. Nothing in
  `session.get_providers()` says whether that is true, so the graph was profiled
  node by node (`compare_models.py`):

  | Model | Graph nodes on CUDA | on CPU | What the GPU is actually running |
  |---|---|---|---|
  | FP32 | 48 / 48 | 0 | everything |
  | FP16 | 50 / 50 | 0 | everything |
  | INT8 QDQ | 145 / 166 | 21 (13%) | **float `Conv`**, on tensors dequantised at every boundary |
  | INT8 QOperator | 9 / 41 | **32 (78%)** | only `QuantizeLinear`/`DequantizeLinear`/`Memcpy` — every `QLinearConv`, `QLinearAdd`, `QGemm` and `QLinearGlobalAveragePool` runs on the **CPU** |

  So the CUDA ExecutionProvider in this build has **no integer kernels**: it
  executes the QDQ graph as float convolution between quantised boundaries,
  which pays 21 host↔device memcpys per inference and gains no integer matmul.
  The integer ops it *cannot* run are exactly the ones QOperator is made of.
  **The win requires TensorRT, or an INT8-native accelerator, which this project
  excludes by scope.**

### Why the ratio moves between sessions

The same models were measured three times on the same machine, hours apart:

| Session | Background | FP32 p50 | INT8-QDQ p50 | QDQ / FP32 |
|---|---|---|---|---|
| 1 (2026-09-25, `confirm_quant.py`) | quiet desktop | 0.835 ms | 1.239 ms | **1.48×** (1.47–1.50) |
| 2 (2026-09-26) | browsers, VS Code, Wallpaper Engine | 0.936 ms | 1.491 ms | **1.59×** (1.47–1.72) |
| 3 (2026-09-26) | same | 1.019 ms | 1.754 ms | **1.72×** (1.59–1.91) |

The signed effect never changed; the magnitude did, by ~16%. The QDQ graph is the
sensitive one because it crosses the PCIe bus **21 times per inference** — CPU
scheduling pressure and memcpy latency show up directly in its p50, while FP32
never leaves the device. The p95/p50 warning threshold (~1.5) was also exceeded
in sessions 2–3 and not in session 1, which is the harness reporting the same
thing.

**So the honest statement is: INT8 was slower in 9/9 trials, by 1.5×–1.7×
depending on what else is running.** Quoting a single figure for this machine
would be quoting the machine state, not the model.

- **This is the engineering finding worth reporting.** "Quantise the model to
  make it faster" is not a rule; it is a hypothesis about the backend. Measured
  on this stack the hypothesis is false: **4× smaller, same accuracy, 1.5×
  slower.** Anyone reporting only the size and accuracy columns is reporting
  half the result — and the missing half is the one that decides whether the
  change is worth making.

### Reader's caveats on these numbers

- **p95 is still noisy.** Even at 500 iterations the p95/p50 ratio sits at
  ~2.0 (fp32/fp16) and ~1.8 (int8), above the harness's ~1.5 warning threshold.
  The tail is therefore **not characterised**, and no conclusion here uses p95.
  Everything rests on p50. A much longer run on an idle machine would be needed
  to say anything about the tail, and this desktop has background GPU load.
- **Repeatability was checked across three separate sessions** (`confirm_quant.py`
  for the first, then `compare_models.py` with the full four-model set twice
  more, all interleaved at 50 warmup / 500 iters). INT8 was slower in all nine
  trials, so the direction is established. The magnitude is stated as a range
  (1.48–1.72× by session) rather than a point estimate — see "Why the ratio moves
  between sessions".
- **Calibration used 512 train-split images, MinMax, per-channel weights.** No
  comparison across calibration methods was done (limitation #7 below), and no
  per-tensor-weight control run was done either.
- **Activations are per-tensor.** Per-channel activations would need a custom
  quantiser and were not attempted. Per-channel *weights* are verified from the
  file rather than assumed — see `inspect_quant.py`.
- **onnxruntime warns that it inserted 21 Memcpy nodes for the CUDA EP**, which
  it says may hurt performance. This warning appears for the FP32 model. The
  profiler shows what they cost: the QDQ model crosses the bus 21 times per
  inference (`.` → `MemcpyFromHost` in the placement table).

---

## QDQ vs QOperator: the same parameters, two different models

Everything above quantises with `QuantFormat.QDQ`. ONNX Runtime's other option
is `QuantFormat.QOperator`, which **replaces** the float operators with integer
ones (`QLinearConv`, `QGemm`, …) instead of inserting `QuantizeLinear` /
`DequantizeLinear` pairs around them. It was never tested in this project until
now — and it turned out not to be an implementation detail.

Reproduce: `python quantize.py --precision int8 --quant-format qoperator`,
then `python compare_models.py`, then `python inspect_quant.py`.

| | FP32 | FP16 | INT8 QDQ | INT8 QOperator |
|---|---|---|---|---|
| File size | 42.61 MB | 21.31 MB | 10.77 MB | **10.71 MB** |
| Test accuracy | 0.7641 | 0.7661 | 0.7656 | **0.7436** |
| Difference vs FP32 | — | +0.20 pts | +0.15 pts | **−2.05 pts** |
| Paired bootstrap 95% CI | — | [+0.05, +0.40] | [−0.25, +0.55] | **[−3.29, −0.75]** |
| McNemar exact p | — | 0.125 | 0.648 | **0.0019** |
| Test samples changing prediction | — | 4 / 2005 | 19 / 2005 | **167 / 2005** |
| max abs logit diff | — | 1.6e-03 | 0.445 | 0.614 |
| Graph nodes on CPU | 0% | 0% | 13% | **78%** |
| inference p50, session 3 (3 trials) | 1.019 ms | 0.932 ms | 1.754 ms | 1.837 ms |
| vs FP32 | — | 0.92× | 1.72× | **1.80×** |

Four things follow, and the first is the one worth remembering:

1. **A 2-point accuracy gap had to be tested, not eyeballed.** 2005 samples at
   ~0.76 put one standard error at ~0.95 points, so −2.05 points is about two of
   them — suggestive, not established. Because both models were evaluated on the
   *same* samples, the comparison is paired: 167 samples change prediction, the
   bootstrap CI on the difference excludes zero, and exact McNemar gives
   p = 0.0019. The FP16 (+0.20 pts) and QDQ (+0.15 pts) columns are the useful
   control: a CI that just excludes zero with p = 0.13 is **not** a result, and
   `compare_models.py` requires both tests before printing "REAL".

2. **The accuracy loss is not a calibration difference.** `inspect_quant.py`
   reads the quantisation parameters back out of both files and compares them
   tensor by tensor: the 21 weight tensors present in both have a **maximum
   relative scale difference of 0.000e+00** — bit-identical scales and
   zero-points from the same MinMax calibration over the same 512 train images.
   Identical parameters, different accuracy: the difference is in what the
   kernels *do* with them. QDQ dequantises to float and accumulates the
   convolution in float; QOperator accumulates in int32 and requantises through
   a fixed-point multiplier at every layer. That requantisation rounds, and 20
   convolutions compound it. *(The mechanism is an inference from the structure
   and the error magnitudes; per-layer error propagation was not measured.)*

3. **QOperator cannot run on this GPU.** The profiler shows 32 of its 41 nodes on
   the CPU — every integer operator. The CUDA EP in onnxruntime 1.23.2 supports
   none of them. So the format choice is not just accuracy-vs-portability: here
   **the more portable format (QDQ) is also the accurate one**, and the "INT8
   accelerator" format is the one that falls off the GPU.

4. **QOperator is the slowest of the four**, slower than QDQ in 6/6 interleaved
   trials, consistent with 78% of its graph crossing back to the CPU. It is still
   only ~4% slower than QDQ on p50, which is itself informative: the bottleneck
   is the quantise/dequantise and memcpy traffic, not the arithmetic that got
   moved to the CPU.

**Interview-safe summary:** *"I measured both INT8 formats rather than assuming
they were equivalent. They produced identical quantisation parameters and a
2-point accuracy difference, and the profiler showed the integer format had 78%
of its nodes on the CPU because the CUDA execution provider has no integer
kernels. So the portable format won on both counts, and 'we quantised to INT8'
is not a statement about speed until you know where the nodes ran."*

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
| 9 | **Hardware-specific numbers.** All figures were measured on one machine at specific software versions (see `ENVIRONMENT.md`); they change elsewhere. See "Why the ratio moves between sessions" — the INT8 penalty in particular depends on what else is running. |
| 10 | **The QOperator accuracy loss is attributed, not isolated.** Identical quantisation parameters rule out calibration as the cause, but per-layer error propagation through integer requantisation was not measured. |
| 11 | **No per-tensor-weight control.** `per_channel=True` is verified in the emitted files (42/42 weight tensors carry per-channel scales, and the channel ranges span a median 3.4×), but the accuracy cost of turning it off was not measured. |

---

## Repository layout

```
models.py             the architecture, defined ONCE (train + benchmark + quantize share it)
train_dermamnist.py   fine-tune ResNet-18 on DermaMNIST -> FP32 checkpoint + test accuracy
benchmark.py          latency (mean/p50/p95), peak VRAM, environment snapshot -> JSON
quantize.py           ONNX export -> FP16 / INT8 -> equivalence check -> accuracy -> latency
confirm_quant.py      repeats the quantisation latency comparison, interleaved, 3 trials
compare_models.py     N models: paired accuracy significance, per-node provider placement, interleaved latency
inspect_quant.py      reads quantisation parameters back out of the ONNX file (per-tensor vs per-channel, scale identity)
data/                 MedMNIST downloads            (gitignored, re-downloadable)
checkpoints/          trained weights                (gitignored)
models/               ONNX exports, all precisions   (gitignored)
results/              benchmark and quantisation JSON (gitignored)
```

`models.py` exists because `benchmark.py` and `quantize.py` both have to rebuild
the exact architecture to load a `state_dict`, and two copy-pasted definitions
drift until `load_state_dict` fails with confusing missing/unexpected keys. One
definition, imported everywhere.

`compare_models.py` deliberately does *not* replace `confirm_quant.py`: the
latter is the script that produced `results/quant-repeatability.json`, and a
published number should keep pointing at the script that measured it. The newer
file generalises the trial loop to an arbitrary model set and adds the two things
a latency table cannot show — paired significance and node placement.

### Five methodology details a reviewer will ask about

- **`torch.cuda.synchronize()` on both sides of every timed iteration**
  (`benchmark.py`). CUDA kernel launches are asynchronous; timing without a
  synchronise measures *queue-submission* time, not compute time. The resulting
  numbers look fast and are wrong.
- **p95/p50 ratio is reported and warned on.** A ratio above ~1.5 usually means
  warmup was too short, another process was sharing the GPU, or thermal throttling
  was kicking in — i.e. the run is not trustworthy. The script prints a warning
  rather than quietly emitting a suspicious p95.
- **Models are interleaved within each trial** (`compare_models.py`). Running all
  of one precision and then all of the next lets thermal drift and background load
  correlate with precision, which manufactures effects that are not there.
- **Accuracy differences are tested as paired differences** (`compare_models.py`).
  Two accuracies on the same test samples are not two independent proportions;
  the per-sample predictions are kept, and McNemar's exact test plus a bootstrap
  CI on the difference are required to agree before a gap is called real.
- **Node placement comes from the profiler, not from `session.get_providers()`**
  (`compare_models.py`). The latter lists the execution providers that were
  *registered*; a graph can register CUDA and execute 78% of its nodes on the CPU.

## Licence / provenance

Code here is original. DermaMNIST is CC BY 4.0, redistributed via the `medmnist`
package. **No employer data, code, or models were used at any point.** No real
patient data and no personal photographs are used, including for demos.
