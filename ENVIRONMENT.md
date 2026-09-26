# Environment snapshot

Recorded 2026-09-25. Every number this repository publishes was measured on this
machine with these versions. **Change any of them and the latency and VRAM
figures are no longer comparable** — re-measure the baseline before comparing.

## Hardware

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3080 Ti |
| VRAM | 12288 MiB (12 GB) |
| Compute capability | sm_86 |
| Driver | 576.40 (supports CUDA 12.9) |
| OS | Windows 10 (10.0.19044) |

## Software

| Package | Version |
|---|---|
| Python | 3.10.21 (Anaconda build, 64-bit) |
| torch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| CUDA (torch build) | 12.1 |
| cuDNN | 90100 (9.1.0) |
| onnx | 1.23.0 |
| onnxruntime-gpu | 1.23.2 |
| medmnist | 3.0.2 |
| numpy | 2.2.6 |
| pillow | 12.3.0 |

Full transitive pin list: `requirements.txt`.

## How this environment was created

```bash
conda create -n edgequant python=3.10 -y
conda activate edgequant
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install onnx onnxruntime-gpu medmnist
```

## Windows-specific facts that are easy to get wrong

### 1. ONNX Runtime falls back to the CPU silently

`onnxruntime-gpu`'s `onnxruntime_providers_cuda.dll` needs `cublasLt64_12.dll`,
`cudnn64_9.dll` and friends. torch bundles all of them in
`<site-packages>/torch/lib`, but that directory is **not** on the process DLL
search path unless something adds it.

When the load fails, onnxruntime logs an error and **quietly uses the CPU
provider** — while `get_available_providers()` still reports
`CUDAExecutionProvider`. Observed symptom:

```
requested providers : ['CUDAExecutionProvider', 'CPUExecutionProvider']
ACTIVE providers    : ['CPUExecutionProvider']      # <-- measured on CPU
```

`benchmark.py` calls `_register_torch_cuda_dlls()` before importing onnxruntime,
which prepends the torch lib directory to `PATH` and registers it with
`os.add_dll_directory()`. The output JSON records `cuda_ep_active` so a run can
be audited afterwards.

**Always assert on `session.get_providers()`, never `get_available_providers()`.**

### 2. `medmnist` cannot create its own download root

Omitting `root=` makes medmnist target `~/.medmnist`, fail to create it, and raise
`RuntimeError: Failed to setup the default root directory` — **before any download
starts**. Both scripts pass `root="data"` explicitly
(`--data-root` overrides it).

### 3. `conda activate` needs a clean PATH

`conda activate` on PowerShell builds a command string and runs it through
`Invoke-Expression`. Unpaired double quotes anywhere in `PATH` therefore break
activation with a misleading `Unexpected token '<path>'` error, even though CMD
is unaffected. Keep PATH entries quote-free and free of trailing backslashes.

### 4. VRAM baseline is polluted by desktop applications

The measured whole-device VRAM baseline on this machine is roughly **1.2–1.7 GB**
in use by desktop applications (browser, compositor, chat clients) before any
model is loaded. `benchmark.py` records whole-device usage before and after each
run via `nvidia-smi` so a reader can confirm the process's own allocations were
the dominant consumer. **Close GPU-consuming applications before measuring peak
VRAM**, and record what was running.

## Model-download routes measured from this machine (2026-09-26)

Measured *before* choosing anything, because the LLM stage needs a multi-GB
download and this machine's proxy setup has already cost days. The trap is worth
recording because it is invisible: the registry proxy is stored as a **bare
`127.0.0.1:10809`**, so a client that reads it infers the scheme from the target
URL and gets `https://127.0.0.1:10809` -- an `https` proxy pointing at a
plaintext HTTP port, which fails as a TLS handshake while reporting a
certificate problem. Two rules follow: a genuine direct test needs an explicitly
empty `ProxyHandler`, since everything else silently inherits `HTTP_PROXY`; and
both routes get measured, because the proxy is unstable rather than uniformly
slow.

| Host | direct | via `127.0.0.1:10809` | note |
|---|---|---|---|
| `pypi.org` | **6.20 MB/s** | 0.72 MB/s | direct is 8.6× faster |
| `huggingface.co` | **FAILS** (`WinError 10060`) | 2.14 MB/s | **only reachable through the proxy** |
| `hf-mirror.com` | 2.41 MB/s | HTTP 308 | the proxy redirects; use direct |
| `modelscope.cn` | 3.38 MB/s | **4.73 MB/s** | fastest of the four, and fast on **both** routes |

Measured with an 8 MB cap on a `tokenizer.json` from each host, so these are
**burst** figures, not sustained throughput. The practical consequence: a 6 GB
model is roughly 21–30 minutes from ModelScope versus 42–47 minutes from either
Hugging Face route — and ModelScope is the only host that is fast on both routes,
which matters because the proxy is the unstable half of the link.

Disk at the same moment: `C:` 50 GB free, `G:` 650 GB free, `I:` 191 GB free.
**Point the model cache (`HF_HOME` or equivalent) at a drive other than `C:`** —
50 GB is not a comfortable margin for repeated multi-GB pulls.

`transformers`, `accelerate`, `safetensors`, `tokenizers`, `bitsandbytes` and
`onnxruntime-genai` are **all absent** from the `edgequant` environment; every LLM
option needs a new install, so none of them is "free" on dependency risk.
