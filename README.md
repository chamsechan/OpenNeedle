# OpenNeedle

**English** | [中文](README_zh.md)

**An open-source C++ CPU inference engine for packed Needle 2 weights.**

The core lives in [`needle2/csrc/`](needle2/csrc/). It computes directly on CQ2/CQ4 packed weights and supports FP32, optional ARM SDOT, prefix caching, and schema-constrained tool calls. Tokenization, schema compilation, and the decode loop run in C++; Python provides model loading, sessions, the CLI, and optional conversion/training tools.

## Quick start

Requires Python ≥ 3.10, a C++17 compiler, and OpenMP. The default installation depends only on NumPy; Linux ARM64 has been tested. SDOT additionally requires DotProd support. See the [build guide](docs/build.md) for platform details.

```bash
git clone https://github.com/chamsechan/OpenNeedle.git
cd OpenNeedle
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/download_official.py

python -m needle2 run artifacts/official/needle2.cact \
  --tools examples/tools.json \
  --prompt 'Turn on the kitchen light.' --threads 4
```

The response includes `function_calls`:

```json
[{"name": "set_light", "arguments": {"room": "kitchen", "on": true}}]
```

The first call compiles and caches the native library. Run `python -m needle2 build-native` to build it ahead of time. For repeated requests, reuse `needle2.inference.InferenceSession`; see the [usage guide](docs/usage.md#持续服务入口). Your application executes the returned tool calls.

FP32 is the default. Add `--matmul sdot --kv-cache int8` to enable both approximate modes; they introduce additional quantization error. Unsupported schema constraints or compilation budgets produce explicit errors. See [supported schemas](docs/architecture.md#grammar-编译和执行).

## C++ build and integration

Build the shared library independently on Linux:

```bash
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --parallel 4
export NEEDLE2_NATIVE_LIBRARY="$PWD/build/native/libneedle2_native.so"
```

Subsequent Python calls load this library. See the [build guide](docs/build.md), [`frontend.h`](needle2/csrc/frontend.h), and the [C example](examples/native_frontend.c). The current public C header exposes tokenizer/grammar components; full model loading and requests are still coordinated through Python sessions.

## Performance and validation

Published measurements on a 4-core ARM Neoverse-N1: 4 threads, SDOT + INT8 KV, warm persistent sessions, five measurements per case.

| Median | Basic (3 requests) | Expanded (16 requests) |
|---|---:|---:|
| Full warm request | 62.86 ms | 76.70 ms |
| Decode throughput | 488.16 token/s | 402.38 token/s |

Warm requests exclude model initialization and initial grammar/prefix construction; the rows measure different scopes. All 19 cases produced the expected calls, with identical tokens before and after the native frontend migration. This is a small regression set; full BFCL has not been evaluated. Numerical error, historical official comparisons, and reproduction commands are in [benchmarks and validation](docs/benchmark.md).

## Optional Python tools

```bash
python -m pip install -e '.[torch]'
python -m needle2 to-torch artifacts/official/needle2.cact artifacts/pytorch
python -m needle2 quantize artifacts/pytorch artifacts/roundtrip.cact
python -m needle2 inspect artifacts/roundtrip.cact
```

Keep `weights.safetensors`, `config.json`, and `source.cact` together. Unchanged tensors retain their original packed bytes; changed tensors are requantized. See the [usage guide](docs/usage.md) for PyTorch inference, fine-tuning, QAT, and retrieval APIs.

## Repository

| Path | Contents |
|---|---|
| [`needle2/csrc/`](needle2/csrc/) | C++ compute, tokenizer, and grammar |
| [`needle2/`](needle2/) | Python bindings, sessions, and optional model tools |
| [`tests/`](tests/) · [`benchmarks/`](benchmarks/) | Regression tests and reproducible cases |
| [`examples/`](examples/) · [`scripts/`](scripts/README.md) | Integration examples, downloads, validation, and benchmarks |
| [`docs/`](docs/) | [Build](docs/build.md), [usage](docs/usage.md), [architecture](docs/architecture.md), [benchmarks](docs/benchmark.md), and [technical references](docs/reference.md) (Chinese) |

Install `.[test]` and run `python -m pytest -q`. Models, build caches, and raw measurements go into Git-ignored directories and are not included in the source delivery.

## License

Source code is licensed under [Apache-2.0](LICENSE); see [NOTICE](NOTICE) for attribution. Model weights and the official comparison library are downloaded separately under their own licenses. Independent inference does not call the official closed-source library.
