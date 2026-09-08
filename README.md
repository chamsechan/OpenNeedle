# OpenNeedle

<picture>
  <source media="(prefers-reduced-motion: reduce) and (prefers-color-scheme: dark)" srcset="docs/assets/openeedle-hero-static-dark.svg">
  <source media="(prefers-reduced-motion: reduce)" srcset="docs/assets/openeedle-hero-static.svg">
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/openeedle-hero-dark.svg">
  <img src="docs/assets/openeedle-hero-light.svg" width="1200" alt="OpenNeedle 压缩推理流程与 CPU 速度对比：官方 407.80、OpenNeedle SDOT 168.21、原生 FP32 133.24、PyTorch 9.35 token/s。SDOT 为近似模式；完整条件见下方性能说明。">
</picture>

**面向 Needle 2 的开源 CPU 推理引擎与 PyTorch 工具链。**

支持 CQ2/CQ4 压缩权重直接推理、PyTorch 双向转换、微调与 QAT。独立 C++ 引擎提供 FP32 和 ARM SDOT 路径，支持前缀缓存与工具调用约束解码。

[快速开始](#快速开始) · [模型转换](#模型转换) · [编译指南](docs/build.md) · [Agent Skills](#agent-skills) · [文档](#文档)

## 当前性能

4 核 ARM Neoverse-N1，相同发布模型，3 组工具请求各预热后重复 5 次。下表为中位数，热请求复用工具前缀。

| 后端 | 线程 | 解码速度 ↑ | 热请求计算耗时 ↓ |
|---|---:|---:|---:|
| 官方闭源引擎 | 自动 | 407.80 token/s | 57.7 ms |
| OpenNeedle FP32 | 4 | **133.24 token/s** | **236.7 ms** |
| OpenNeedle SDOT | 4 | **168.21 token/s** | **159.9 ms** |
| PyTorch FP32 | 2 | 9.35 token/s | 1803.3 ms |

FP32 / SDOT 吞吐分别为 PyTorch 的 **14.2× / 18.0×**；SDOT 达到官方的 **41.2%**。PyTorch 使用 CPU eager，未启用 `torch.compile`。各后端计时工作量有所不同，以上为应用层比较，详见 [测速方法与复现](docs/backend-comparison.md)。

**默认 FP32；SDOT 为可选近似模式，会增加量化误差。** 当前 15 项质量回归中，官方与两种原生模式均通过 13 项；完整 BFCL 尚未评估。精度结果和支持范围见 [验证报告](docs/results.md) 与 [grammar 文档](docs/grammar.md)。

## 快速开始

需要 **Python ≥ 3.10、C++17 编译器及 OpenMP**；已实测 Linux ARM64。SDOT 额外要求 CPU 支持 DotProd。

```bash
git clone https://github.com/chamsechan/OpenNeedle.git
cd OpenNeedle
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

# 下载固定版本的官方模型
python scripts/download_official.py

# 运行工具调用推理
python -m needle2 run artifacts/official/needle2.cact \
  --tools examples/tools.json \
  --prompt 'Turn on the kitchen light.' \
  --backend native --threads 4
```

输出中的 `function_calls`：

```json
[{"name": "set_light", "arguments": {"room": "kitchen", "on": true}}]
```

首次原生调用自动编译并缓存 C++ 内核，无需官方闭源库；可用 `python -m needle2 build-native` 提前编译。CMake 和预编译部署见 [编译指南](docs/build.md)。工具执行由应用接入。

添加 `--matmul sdot` 启用 SDOT；改用 `--backend torch` 运行 PyTorch 参考后端。线程数按目标 CPU 调整。

## 模型转换

```bash
# 官方部署权重 → PyTorch FP32
python -m needle2 to-torch artifacts/official/needle2.cact artifacts/pytorch

# PyTorch → 官方兼容的 CQ2/CQ4 部署文件
python -m needle2 quantize artifacts/pytorch artifacts/roundtrip.cact

# 检查结构、位宽与哈希
python -m needle2 inspect artifacts/roundtrip.cact
```

转换目录中的 `weights.safetensors`、`config.json`、`source.cact` 应完整保留。未修改张量保留原始压缩字节，修改后的张量重新量化。转换器支持相同 Needle 2 架构。

FP16 master 转换、Python API 与微调/QAT 见 [使用指南](docs/usage.md)。

## 实现与参考

引擎在输入侧计算 Hadamard 变换，直接读取压缩码字并查表累加；通过共享投影变换、NEON/SDOT 和前缀缓存降低解码成本。模型与量化实现依据固定版本的 [Needle 源码](https://github.com/cactus-compute/needle/tree/53df049c4a1a82fca1027b81f9ff21336dfb0861) 和 [发布权重](https://huggingface.co/Cactus-Compute/needle2/tree/32e9e3a93b205f786929697446ae669cf0a84579)。

架构、CQ 格式及 SAN、mHC、Engram、QuIP#、LUT-GEMM 等参考文献见 [研究记录](docs/research.md)；内核实现与优化细节见 [原生引擎](docs/native-engine.md)。

## Agent Skills

提供 [安装推理](skills/openeedle-inference/SKILL.md)、[模型转换](skills/openeedle-convert/SKILL.md)、[微调/QAT](skills/openeedle-finetune/SKILL.md)、[性能对比](skills/openeedle-benchmark/SKILL.md) 四个 skills，安装与调用方式见 [Skills 指南](skills/README.md)。

也可直接让助手执行：“读取 `skills/openeedle-inference/SKILL.md`，用我的工具 schema 跑通推理。”

## 文档

| 文档 | 内容 |
|---|---|
| [使用指南](docs/usage.md) | 转换、Python API、训练/QAT 与前缀缓存 |
| [编译指南](docs/build.md) | 自动编译、CMake 与预编译部署 |
| [原生引擎](docs/native-engine.md) | 内核布局、优化策略与平台能力 |
| [性能对比](docs/backend-comparison.md) · [精度验证](docs/results.md) | 测量方法、结果与复现命令 |
| [研究记录](docs/research.md) | 模型架构、量化格式、版本与参考文献 |

## 许可证

源码采用 [Apache-2.0](LICENSE)，上游归因见 [NOTICE](NOTICE)。官方模型与基线库单独下载，遵循各自的上游许可。
