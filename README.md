# OpenNeedle

**English** | [中文](README_zh.md)

<picture>
  <source media="(prefers-reduced-motion: reduce) and (prefers-color-scheme: dark)" srcset="docs/assets/openeedle-hero-static-dark.svg">
  <source media="(prefers-reduced-motion: reduce)" srcset="docs/assets/openeedle-hero-static.svg">
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/openeedle-hero-dark.svg">
  <img src="docs/assets/openeedle-hero-light.svg" width="1200" alt="OpenNeedle packed CPU inference. Expanded decode throughput: official 493.70 token/s (earlier run), OpenNeedle SDOT + INT8 KV 327.40 token/s. Timing definitions differ.">
</picture>

**Open-source CPU inference engine and PyTorch toolchain for Needle 2.**

Supports direct inference on CQ2/CQ4 compressed weights, bidirectional PyTorch conversion, fine-tuning, and QAT. The standalone C++ engine provides FP32 and ARM SDOT execution paths, with prefix caching and grammar-constrained decoding for tool calls.

[Quickstart](#quickstart) · [Model Conversion](#model-conversion) · [Build Guide](docs/build.md) · [Agent Skills](#agent-skills) · [Documentation](#documentation)

## Current Performance

4-core ARM Neoverse-N1, 4 threads, **SDOT + INT8 KV**, same official release weights. Values are medians; basic and expanded suites contain 3 and 16 distinct requests.

| Metric | Official 2.0.4¹ | OpenNeedle | OpenNeedle relative to official |
|---|---:|---:|---|
| Basic warm request compute | 45.1 ms | **60.3 ms** | 33.6% more time |
| Expanded warm request compute | 439.1 ms | **87.4 ms** | 80.1% less time |
| Expanded query prefill | Not exposed | **32.3 ms** | Not directly comparable |
| Expanded decode throughput | 493.7 token/s | **327.4 token/s** | 33.7% lower throughput |
| Tool-call quality | 13/15 (86.7%) | 13/15 (86.7%) | Same score on this sample |

¹ Official measurements are from an earlier run on the same host (5 repeats/case); OpenNeedle measurements use 9 repeats/case. **They are not from a single interleaved official/native run.** Official TPS is self-reported, with different internal work and timing boundaries. Lower expanded request time does not establish a faster decoder kernel. Warm compute excludes model/prefix/DFA initialization; native query tokenization and final parsing are also outside the timer. See [Methodology, raw samples and reproduction](docs/backend-comparison.md).

**FP32 remains the default. SDOT and INT8 KV are optional approximations.** All four native configurations retain expected quality outputs and pass the test suite (**194 tests + 4 subtests**). This does not mean the four configurations, or the official logits, are identical. Full BFCL has not been evaluated. See [validation](docs/results.md) and [grammar coverage](docs/grammar.md).

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

Add `--matmul sdot --kv-cache int8` to select the approximate mode used in the performance table; omit these options for the FP32 defaults. Use `--matmul sdot` alone to retain FP32 KV. Native-only installation needs NumPy; install `pip install -e '.[torch]'` for the PyTorch backend, conversion or training. Adjust thread count to fit the target CPU.

For repeated requests, create one `needle2.inference.InferenceSession` and call its `generate()` method. It reuses the model, tokenizer, tool grammar and native tool-prefix cache. Tokenizer BPE, tool-schema compilation and constrained decoding run in C++; Python grammar is only a reference/PyTorch path. The module-level `generate()` remains a one-shot convenience API. See [session usage and timing](docs/python-runtime.md) and [native frontend / official compatibility boundaries](docs/native-frontend.md).

<a id="模型转换"></a>
## Model Conversion

```bash
# Install optional conversion/training dependencies
python -m pip install -e '.[torch]'

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

The C++ engine shares one Hadamard input transform across Q/K/V/gate projections, reads packed CQ weights directly, and uses NEON FMA or optional SDOT integer dot products. SDOT computes four output rows together to reuse activation loads; ordinary matrix operations with fewer than 128 output rows run serially. Weights retain their packed row layout. KV defaults to FP32, with an optional INT8 cache that stores per-head scales and adds quantization error.

Fixed tool prefixes can be reused through the [NativeEngine prefix-cache API](docs/native-engine.md#固定-tools-前缀复用). Native tool decoding compiles schema and UTF-8 constraints into a token DFA, projects only candidate rows, and skips the LM head for a single candidate. Large grammars fall back to the Python schema gate. The four-row kernel is checked against single-row SDOT arithmetic across CQ2/CQ4, padding, tail rows, and 1/2/4 threads.

The C++ engine validates an immutable DFA once and reuses it across requests, while retaining vocabulary and raw C ABI checks. Attention reuses GQA work lists and KV slot mappings, accumulating V in 32-dimension NEON register tiles. Batched SDOT prefill shares packed-weight decoding between adjacent tokens and computes RoPE trigonometry once per chunk for all layers. These implementations preserve existing quantization and per-dimension accumulation order; [implementation details](docs/native-engine.md#prefill-与-attention-数据复用).

Dense mHC decode projections now share input loads between pairs of rows and use one thread-pool dispatch. ARM64 INT8-KV attention specializes the 64-dimensional paired QK path while keeping FP32 queries and the original reduction order. Separate request benchmarks and numerical validation are documented in the [mHC report](docs/mhc-optimization.md) and [attention report](docs/attention-optimization.md); these measurements do not update the historical official-engine comparison above.

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
