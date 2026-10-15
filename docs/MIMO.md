# Experimental MiMo-V2.6-Flash-RL

This fork adds a native `mimo2` text backend to `strata-poly`, enabled only by
`-DSTRATA_EXPERIMENTAL_MIMO=ON`. The first target is a dedicated RTX 5090 with 32 GB VRAM and 96 GB system RAM,
leaving 5–10 GiB of physical RAM available during testing. That is a target, not a measured fit.

The CUDA kernels and a four-layer synthetic model have run on an RTX 5090 under Windows with CUDA 13.2.51,
MSVC 19.38 and driver 617.14. The full checkpoint has **not** been downloaded or run. There are no full-model
quality, throughput, startup-time or resident-memory results yet. HIP and SYCL are not validated by this change;
the experimental option rejects them. Existing model families have not received a new full-model regression run.

## Target checkpoint

[pfeifferj/MiMo-V2.6-Flash-RL-GSQ-RCO-GGUF](https://huggingface.co/pfeifferj/MiMo-V2.6-Flash-RL-GSQ-RCO-GGUF)
is a third-party GSQ/RCO quantization of Xiaomi's RL checkpoint. Its card describes a text-only export without
the vision/audio components or MTP blocks. The inspected GGUF header agrees: 48 trunk blocks, no MTP layers.

| File | Bytes | GiB | Decision |
| --- | ---: | ---: | --- |
| `MiMo-V2.6-Flash-RL-GSQ-RCO-3bit.gguf` | 115,678,863,104 | 107.73 | Initial target |
| `MiMo-V2.6-Flash-RL-GSQ-RCO-3.5bit.gguf` | about 134.98 billion | about 125.71 | Too little room for runtime buffers and the requested headroom |

The exact revision, size and SHA-256 are in [mimo-target.json](mimo-target.json). Revision:
`a37b69042dbe3e375790d10ffefda6333365c718`.

The 3-bit file uses Q8_0 dense matrices and Q2_K/Q3_K/MXFP4 experts. It is not a uniform three-bit tensor format.
Sixteen layers have different gate and up types. This implementation runs those layers' experts on the CPU,
using each matrix's own type and stride; same-format layers can use the existing GPU expert cache and staging.
Mixed gate/up GPU kernels are future optimization work. The manifest lists the affected zero-based layers.

Three down-expert tensors use MXFP4. The existing exact MXFP4-to-Q5_0 conversion uses **4.125 GiB of host heap**
for these tensors. This fork fixes conversion running twice during model loading: the sizing pass no longer
allocates an unused second copy. Converted weights remain in host memory even when some experts are cached
on the GPU. The source GGUF is mapped, so committed heap alone is not the model's physical working set.

## What is implemented

- Fused QKV projections, 192-wide keys, 128-wide values, and partial 64-dimensional NeoX RoPE.
- Per-layer full/sliding-window GQA, the 128-token sliding window, different RoPE bases, attention sinks,
  and the model's attention value scale.
- Dense first FFN, then sigmoid routing with correction bias, top-k selection and normalized unbiassed weights.
- Q8_0 dense matrices and the target's mixed expert formats through existing CPU/GPU product implementations.
- Prompt batches, sequential decode, staged expert loading, and saving/restoring KV rows.
- `qwen2` tokenization and MiMo-specific NFC normalization, retained when extracting the tokenizer directory.
  This follows Xiaomi's tokenizer JSON; NFC is not inferred for other `qwen2` models.

The baseline keeps FP16 keys and values for every position, including sliding layers. For this checkpoint,
KV payload is **0.85 GiB at 4K** or **1.70 GiB at 8K**, plus allocation granularity and runtime buffers.
Context is capped at 8192 tokens. The attention kernel is a correctness baseline, not an optimized flash-attention
implementation. MTP/DFlash is disabled; an explicit draft model is rejected. Multimodal input is outside this port.

## Build and prepare

Use an x64 Visual Studio developer terminal on Windows, with CMake, Ninja and CUDA available. The commands below
target a 5090 (`sm_120`). Use the repository's pinned ggml source, not an arbitrary installed ggml version.
The tested checkout was llama.cpp `3cf03257f219afbe7334045ff7c6a06ac68c627d`.

```powershell
cmake -S . -B build-mimo -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_MIMO=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_BUILD_TESTS=OFF
cmake --build build-mimo --target strata-poly poly_mimo_test poly_mimo_engine_test --parallel 4
```

If the pinned llama.cpp source is already present, also pass `-DSTRATA_GGML_DIR=PATH/TO/llama.cpp` when configuring.
The ordinary setup installer has no MiMo family selection yet; use this manual build and configuration path.

Download the **one pinned 3-bit file** when ready. It needs 115.7 GB on disk; the build and source need additional space.
Install `huggingface_hub`, `numpy`, `regex` and `jinja2` in the Python environment if absent, plus the server's usual
dependencies. `tokenizers` is needed only for the parity check below.

```powershell
hf download pfeifferj/MiMo-V2.6-Flash-RL-GSQ-RCO-GGUF MiMo-V2.6-Flash-RL-GSQ-RCO-3bit.gguf --revision a37b69042dbe3e375790d10ffefda6333365c718 --local-dir F:/Models/MiMo
python tools/prepare_mimo.py --gguf F:/Models/MiMo/MiMo-V2.6-Flash-RL-GSQ-RCO-3bit.gguf
```

`prepare_mimo.py` checks size and SHA-256, extracts the tokenizer/template and writes `build-mimo/run/mimo.json`.
It does not start inference. `--skip-hash` is available only for a file already checked independently. The
generated configuration starts with 4K context, prompt batch 64, a 512 MiB staging buffer, 1024 MiB VRAM reserve,
no draft/lookup speculation, no expert swaps, and no prompt or parked conversation cache. These are initial
acceptance settings; cache tuning comes after a successful load and reference comparison.
`--context 8192` selects the current maximum.

With other applications closed or idle, start on loopback:

```powershell
python serve/server.py --engine strata --config build-mimo/run/mimo.json --host 127.0.0.1 --port 8096
```

The web app is at `http://127.0.0.1:8096`. Start with a short prompt and thinking disabled
(`chat_template_kwargs: {"enable_thinking": false}`) to stay within the initial context. Generated answers and
tool-call behavior still require real checkpoint acceptance. The tokenizer/template tests alone do not establish them.

## Reproduce validation without the large model

Substitute the same pinned llama.cpp path used for the build:

```powershell
$env:OPENBLAS_NUM_THREADS = '2'
python tools/make_tiny_mimo.py --gguf-py PATH/TO/llama.cpp/gguf-py --out build-mimo/tiny-mimo
python tools/make_tiny_mimo.py --gguf-py PATH/TO/llama.cpp/gguf-py --out build-mimo/tiny-gsq --gsq
build-mimo/poly_mimo_test.exe
build-mimo/poly_mimo_engine_test.exe build-mimo/tiny-mimo
build-mimo/poly_mimo_engine_test.exe build-mimo/tiny-gsq
python -m unittest tools.test_strata_tokenizer tools.test_tokenizer_plain_spans
```

The four-layer fixtures use 137 tokens to cross the sliding-window boundary. The GSQ fixture includes both
Q2_K-gate/Q3_K-up and Q3_K-gate/Q2_K-up, and an MXFP4 down matrix. NumPy dequantizes the packed weights and
computes independent reference logits. CPU and GPU paths pass the 1.5% relative-L2 and elementwise error bounds
against that reference (the engine additionally quantizes activations). Checkpoint restoration reproduced the
saved logits exactly in the recorded runs. Cache shrinking forces the staged GPU path. The Q8 fixture also
tests skipping appended MTP metadata. Invalid context and explicit draft options are rejected.

Primitive tests cover the real attention head widths/counts, sinks, partial RoPE and the 8K limit. Existing
`poly_route_test`, `poly_workers_test` and `poly_mxfp4_test` pass on this machine. The 11 existing Python
tokenizer tests pass. A build with the MiMo option off also succeeds and rejects a `mimo2` fixture as unsupported;
the option was then restored and the enabled CLI generated four tokens from the synthetic GSQ fixture.
To compare MiMo token IDs and extracted-directory loading against Xiaomi's reference:

```powershell
hf download XiaomiMiMo/MiMo-V2.6-Flash-RL tokenizer.json --revision 3b38d063180c3e4aed9691fdc735f3d10b266ee4 --local-dir build-mimo/xiaomi-tokenizer
python tools/test_mimo_tokenizer.py --gguf F:/Models/MiMo/MiMo-V2.6-Flash-RL-GSQ-RCO-3bit.gguf --reference build-mimo/xiaomi-tokenizer/tokenizer.json
```

All 520 tested strings/prompts match reference token IDs, including composed/decomposed Unicode, digits,
whitespace, emoji, Chinese and rendered prompts with thinking on/off. For this check only, a complete GGUF
header suffices; the recorded run read the target's header without downloading its tensor payload.

## Remaining acceptance work

1. Download and hash the pinned checkpoint; load it on the dedicated machine at 4K.
2. Measure physical available RAM, GPU allocation, committed heap and page reads during startup and generation.
   Aim for 5–10 GiB available RAM after the active experts have been read. Persistent paging means the fit failed,
   even if the process started. Do not count reclaimable GPU-backed file pages twice.
3. Compare real checkpoint logits/greedy output against a current llama.cpp MiMo2 build; then verify API chat,
   thinking, stopping and tool calls. Synthetic parity cannot detect a missed full-checkpoint convention.
4. Measure prefill, decode and time to first token separately, with model, context and cache state recorded.
5. Optimize mixed gate/up GPU execution and attention only after those checks. Larger context needs a different
   attention implementation and revised cache budgeting.

## Related upstream work

PolyStrata's [MiMo issue #4](https://github.com/VecSzn/PolyStrata/issues/4) tracks the new architecture request.
[lighttransport/Strata's glm53f branch](https://github.com/lighttransport/Strata/tree/glm53f) contains a MiMo Q2
packing experiment and `docs/MIMO_V26_TR16_B550.md`; its runtime work was incomplete when inspected, so its
estimated speeds are not evidence of a running backend.
[llama.cpp's MiMo2 implementation](https://github.com/ggml-org/llama.cpp/blob/cd0a5b6a180dff0048e5fd993098b21c0b2c91b8/src/models/mimo2.cpp)
provides the architecture reference. This port implements the text trunk directly rather than depending on
those experimental packing tools.
