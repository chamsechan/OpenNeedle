# OpenNeedle

[English](README.md) | **中文**

**面向 Needle 2 压缩权重的开源 C++ CPU 推理引擎。**

核心位于 [`needle2/csrc/`](needle2/csrc/)，直接对 CQ2/CQ4 packed 权重计算，支持 FP32、可选 ARM SDOT、前缀缓存和工具调用约束解码。Tokenizer、schema 编译和解码循环在 C++ 中执行；Python 提供模型加载、会话、CLI 和可选的转换/训练工具。

## 快速开始

需要 Python ≥ 3.10、C++17 编译器和 OpenMP。默认安装仅依赖 NumPy；已实测 Linux ARM64。SDOT 额外要求 CPU 支持 DotProd，平台说明见[编译指南](docs/build.md)。

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

输出中的 `function_calls`：

```json
[{"name": "set_light", "arguments": {"room": "kitchen", "on": true}}]
```

首次调用自动编译并缓存原生库，也可用 `python -m needle2 build-native` 提前编译。重复请求复用 `needle2.inference.InferenceSession`，用法见[使用指南](docs/usage.md#持续服务入口)。应用负责执行返回的工具调用。

默认使用 FP32。添加 `--matmul sdot --kv-cache int8` 可启用两项近似模式；它们会增加量化误差。原生 schema 编译不支持的约束或超出预算时明确报错，支持范围见[引擎架构](docs/architecture.md#grammar-编译和执行)。

## C++ 构建与接入

Linux 下独立构建共享库：

```bash
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --parallel 4
export NEEDLE2_NATIVE_LIBRARY="$PWD/build/native/libneedle2_native.so"
```

Python 随后的调用加载该库。完整部署步骤见[编译指南](docs/build.md)，组件接口见 [`frontend.h`](needle2/csrc/frontend.h) 和 [C 示例](examples/native_frontend.c)。当前 C ABI 提供 tokenizer/grammar 组件，完整模型加载和请求仍通过 Python 会话协调。

## 性能与验证

以下为已发布测量：4 核 ARM Neoverse-N1、4 线程、SDOT＋INT8 KV；常驻会话预热后，每例测量 5 次。

| 中位数 | Basic（3 个请求） | Expanded（16 个请求） |
|---|---:|---:|
| 完整热请求耗时 | 62.86 ms | 76.70 ms |
| Decode 吞吐 | 488.16 token/s | 402.38 token/s |

热请求不含模型初始化及首次 grammar/前缀构建；两行是不同计时范围。19 个用例调用正确，前端迁移前后完整 token 一致，不代表广泛真实请求准确率。完整 BFCL 尚未评估。数值误差、历史官方对照和复现方法集中在[基准与验证](docs/benchmark.md)。

## 可选 Python 工具

```bash
python -m pip install -e '.[torch]'
python -m needle2 to-torch artifacts/official/needle2.cact artifacts/pytorch
python -m needle2 quantize artifacts/pytorch artifacts/roundtrip.cact
python -m needle2 inspect artifacts/roundtrip.cact
```

保留转换目录中的 `weights.safetensors`、`config.json` 和 `source.cact`；未修改张量保留原始压缩字节，修改后的张量重新量化。PyTorch 推理、微调、QAT 和检索接口见[使用指南](docs/usage.md)。

## 项目结构

| 路径 | 内容 |
|---|---|
| [`needle2/csrc/`](needle2/csrc/) | C++ 计算、tokenizer 和 grammar |
| [`needle2/`](needle2/) | Python 绑定、会话与可选模型工具 |
| [`tests/`](tests/) · [`benchmarks/`](benchmarks/) | 回归测试与可复现用例 |
| [`examples/`](examples/) · [`scripts/`](scripts/README.md) | 接入示例、下载、验证与测速 |
| [`docs/`](docs/) | [编译](docs/build.md)、[使用](docs/usage.md)、[架构](docs/architecture.md)、[基准](docs/benchmark.md)、[技术参考](docs/reference.md) |

安装 `.[test]` 后运行 `python -m pytest -q`。模型、构建缓存和原始测量写入被 Git 忽略的目录，不随源码交付。

## 许可证

源码采用 [Apache-2.0](LICENSE)，上游归因见 [NOTICE](NOTICE)。模型与官方比较库单独下载，遵循各自许可。独立推理不调用官方闭源库。
