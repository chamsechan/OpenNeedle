# OpenNeedle

**English** | [中文](README_zh.md)

<picture>
  <source media="(prefers-reduced-motion: reduce) and (prefers-color-scheme: dark)" srcset="docs/assets/openeedle-hero-static-dark.svg">
  <source media="(prefers-reduced-motion: reduce)" srcset="docs/assets/openeedle-hero-static.svg">
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/openeedle-hero-dark.svg">
  <img src="docs/assets/openeedle-hero-light.svg" width="1200" alt="OpenNeedle compressed inference pipeline and CPU speed comparison: Official 407.80, OpenNeedle SDOT 168.21, Native FP32 133.24, PyTorch 9.35 token/s. SDOT is an approximate mode; see performance notes below for full conditions.">
</picture>

**Open-source CPU inference engine and PyTorch toolchain for Needle 2.**

Supports direct inference on CQ2/CQ4 compressed weights, bidirectional PyTorch conversion, fine-tuning, and QAT. The standalone C++ engine provides FP32 and ARM SDOT execution paths, with prefix caching and grammar-constrained decoding for tool calls.

[Quickstart](#quickstart) · [Model Conversion](#model-conversion) · [Build Guide](docs/build.md) · [Agent Skills](#agent-skills) · [Documentation](#documentation)

## Current Performance

Benchmarked on a 4-core ARM Neoverse-N1 using the identical release model. 3 tool request suites were each repeated 5 times after warmup. The table reports median values; warm requests reuse the tool prefix.

| Backend | Threads | Decode Speed ↑ | Warm Request Latency ↓ |
|---|---:|---:|---:|
| Official Closed-Source Engine | Auto | 407.80 token/s | 57.7 ms |
| OpenNeedle FP32 | 4 | **133.24 token/s** | **236.7 ms** |
| OpenNeedle SDOT | 4 | **168.21 token/s** | **159.9 ms** |
| PyTorch FP32 | 2 | 9.35 token/s | 1803.3 ms |

FP32 / SDOT throughputs are **14.2× / 18.0×** that of PyTorch; SDOT reaches **41.2%** of the official engine. PyTorch runs in CPU eager mode without `torch.compile`. Workload timing boundaries vary slightly across backends; the above represents an application-level comparison. See [Benchmark Methodology & Reproduction](docs/backend-comparison.md) for details.

**FP32 is the default; SDOT is an optional approximate mode that introduces additional quantization error.** Out of the 15 quality regression tests, the official engine and both native modes pass 13 cases; full BFCL has not yet been evaluated. Precision results and supported scope can be found in the [Validation Report](docs/results.md) and [Grammar Documentation](docs/grammar.md).

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

The engine computes Hadamard transforms on input activations, reads compressed codewords directly, and accumulates results via table lookups; decoding overhead is reduced through shared projection transforms, NEON/SDOT kernels, and prefix caching. Model architecture and quantization logic follow pinned upstream [Needle source](https://github.com/cactus-compute/needle/tree/53df049c4a1a82fca1027b81f9ff21336dfb0861) and [release weights](https://huggingface.co/Cactus-Compute/needle2/tree/32e9e3a93b205f786929697446ae669cf0a84579).

For architecture, CQ format, and reference papers such as SAN, mHC, Engram, QuIP#, and LUT-GEMM, see [Research Notes](docs/research.md); for kernel implementation and optimization details, see [Native Engine](docs/native-engine.md).

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
| [Research Notes](docs/research.md) | Model architecture, quantization format, pinned revisions, and references |

## License

Source code is licensed under [Apache-2.0](LICENSE); see [NOTICE](NOTICE) for upstream attributions. Official models and baseline libraries are downloaded separately and subject to their respective upstream licenses.
