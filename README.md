# OpenNeedle

**English** | [中文](README_zh.md)

<picture>
  <source media="(prefers-reduced-motion: reduce) and (prefers-color-scheme: dark)" srcset="docs/assets/openeedle-hero-static-dark.svg">
  <source media="(prefers-reduced-motion: reduce)" srcset="docs/assets/openeedle-hero-static.svg">
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/openeedle-hero-dark.svg">
  <img src="docs/assets/openeedle-hero-light.svg" width="1200" alt="OpenNeedle compressed inference pipeline and CPU speed comparison: Official 488.90, OpenNeedle SDOT 150.59, Native FP32 120.77, PyTorch 9.23 token/s. SDOT is approximate; see the performance notes for full conditions.">
</picture>

**Open-source CPU inference engine and PyTorch toolchain for Needle 2.**

Supports direct inference on CQ2/CQ4 compressed weights, bidirectional PyTorch conversion, fine-tuning, and QAT. The standalone C++ engine provides FP32 and ARM SDOT execution paths, with prefix caching and grammar-constrained decoding for tool calls.

[Quickstart](#quickstart) · [Model Conversion](#model-conversion) · [Build Guide](docs/build.md) · [Agent Skills](#agent-skills) · [Documentation](#documentation)

## Current Performance

Measured on a 4-core ARM Neoverse-N1 with the same release model. Three tool requests are each repeated five times after warmup, serially interleaving separate persistent backend processes. Warm requests use the [prefix-cache API](docs/native-engine.md#固定-tools-前缀复用); values are medians of 15 measurements.

| Backend | Threads | Decode Speed ↑ | Warm Request Compute Time ↓ |
|---|---:|---:|---:|
| Official Closed-Source Engine | Auto | 488.90 token/s | 47.9 ms |
| OpenNeedle FP32 | 4 | **120.77 token/s** | **244.0 ms** |
| OpenNeedle SDOT | 4 | **150.59 token/s** | **167.7 ms** |
| PyTorch FP32 | 1 | 9.23 token/s | 1825.1 ms |

FP32 / SDOT decode throughputs are **13.1× / 16.3×** the displayed PyTorch result; SDOT reaches **30.8%** of the official engine. PyTorch uses CPU eager without `torch.compile`; all 1/2/4-thread results are in [Benchmark Methodology & Reproduction](docs/backend-comparison.md). Measurements explicitly set `OMP_WAIT_POLICY=PASSIVE`; the library does not change the global wait policy. Timing workloads differ across the independent and official interfaces, so this is an application-level comparison.

Revision comparison: [16f2bd3f → a65a9a11](docs/backend-comparison.md#revision-comparison).

**FP32 is the default; SDOT is an optional approximate mode that adds quantization error.** The official engine and both native modes pass 13 of 15 tool-call quality cases; full BFCL has not been evaluated. Four-row SDOT is tested for exact equality with single-row arithmetic across 24 parameter combinations. See the [Validation Report](docs/results.md) and [Grammar Documentation](docs/grammar.md).

<a id="快速开始"></a>
## Quickstart

Requires **Python ≥ 3.10, a C++17 compiler, and OpenMP**; tested on Linux ARM64. SDOT additionally requires CPU DotProd support.

```bash
git clone https://github.com/chamsechan/OpenNeedle.git
cd OpenNeedle
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

# Download the pinned official model
python scripts/download_official.py

# Run tool-calling inference
python -m needle2 run artifacts/official/needle2.cact \
  --tools examples/tools.json \
  --prompt 'Turn on the kitchen light.' \
  --backend native --threads 4
```

The resulting `function_calls`:

```json
[{"name": "set_light", "arguments": {"room": "kitchen", "on": true}}]
```

The initial native invocation automatically compiles and caches the C++ kernel without requiring official closed-source libraries; you can also compile ahead of time with `python -m needle2 build-native`. See the [Build Guide](docs/build.md) for CMake builds and precompiled library deployment. Tool execution is handled by your application.

Add `--matmul sdot` to enable SDOT; switch to `--backend torch` to run the PyTorch reference backend. Adjust thread count to fit the target CPU.

<a id="模型转换"></a>
## Model Conversion

```bash
# Official deployment weights → PyTorch FP32
python -m needle2 to-torch artifacts/official/needle2.cact artifacts/pytorch

# PyTorch → official-compatible CQ2/CQ4 deployment archive
python -m needle2 quantize artifacts/pytorch artifacts/roundtrip.cact

# Inspect structure, bitwidth, and hashes
python -m needle2 inspect artifacts/roundtrip.cact
```

The conversion directory preserves `weights.safetensors`, `config.json`, and `source.cact`. Unmodified tensors retain original compressed bytes, while modified tensors are re-quantized. The converter supports matching Needle 2 architectures.

See the [Usage Guide](docs/usage.md) for FP16 master conversion, Python API, and fine-tuning/QAT.

## Implementation & References

The C++ engine shares one Hadamard input transform across Q/K/V/gate projections, reads packed CQ weights directly, and uses NEON FMA or optional SDOT integer dot products. SDOT computes four output rows together to reuse activation loads; ordinary matrix operations with fewer than 128 output rows run serially. Weights retain their packed row layout, and KV state remains FP32.

Fixed tool prefixes can be reused through the [NativeEngine prefix-cache API](docs/native-engine.md#固定-tools-前缀复用). Decoding computes full-vocabulary logits and applies schema constraints to select tool-call tokens. The four-row kernel is checked against single-row SDOT arithmetic across CQ2/CQ4, padding, tail rows, and 1/2/4 threads.

Model architecture and quantization follow pinned [Needle source](https://github.com/cactus-compute/needle/tree/53df049c4a1a82fca1027b81f9ff21336dfb0861) and [release weights](https://huggingface.co/Cactus-Compute/needle2/tree/32e9e3a93b205f786929697446ae669cf0a84579). See the [Technical Reference](docs/research.md) for the architecture, CQ format, Arm intrinsics and Cactus kernel references; see the [Native Engine](docs/native-engine.md) for execution details and numerical limits.

## Agent Skills

Provides four skills: [Inference](skills/openeedle-inference/SKILL.md), [Model Conversion](skills/openeedle-convert/SKILL.md), [Fine-tuning/QAT](skills/openeedle-finetune/SKILL.md), and [Benchmarking](skills/openeedle-benchmark/SKILL.md). See the [Skills Guide](skills/README.md) for installation and usage.

You can also prompt an assistant directly: "Read `skills/openeedle-inference/SKILL.md` and run inference with my tool schema."

<a id="文档"></a>
## Documentation

| Document | Description |
|---|---|
| [Usage Guide](docs/usage.md) | Conversion, Python API, training/QAT, and prefix caching |
| [Build Guide](docs/build.md) | Automatic compilation, CMake, and precompiled deployment |
| [Native Engine](docs/native-engine.md) | Kernel layout, optimization strategies, and platform capabilities |
| [Backend Comparison](docs/backend-comparison.md) · [Validation Report](docs/results.md) | Measurement methodology, benchmark results, and reproduction commands |
| [Technical Reference](docs/research.md) | Model architecture, quantization format, pinned revisions, and references |

## License

Source code is licensed under [Apache-2.0](LICENSE); see [NOTICE](NOTICE) for upstream attributions. Official models and baseline libraries are downloaded separately and subject to their respective upstream licenses.
