# OpenNeedle

[English](README.md) | **中文**

<picture>
  <source media="(prefers-reduced-motion: reduce) and (prefers-color-scheme: dark)" srcset="docs/assets/openeedle-hero-static-dark.svg">
  <source media="(prefers-reduced-motion: reduce)" srcset="docs/assets/openeedle-hero-static.svg">
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/openeedle-hero-dark.svg">
  <img src="docs/assets/openeedle-hero-light.svg" width="1200" alt="OpenNeedle 压缩推理流程与 CPU 速度对比：官方 488.90、OpenNeedle SDOT 150.59、原生 FP32 120.77、PyTorch 9.23 token/s。SDOT 为近似模式；完整条件见下方性能说明。">
</picture>

**面向 Needle 2 的开源 CPU 推理引擎与 PyTorch 工具链。**

支持 CQ2/CQ4 压缩权重直接推理、PyTorch 双向转换、微调与 QAT。独立 C++ 引擎提供 FP32 和 ARM SDOT 路径，支持前缀缓存与工具调用约束解码。

[快速开始](#快速开始) · [模型转换](#模型转换) · [编译指南](docs/build.md) · [Agent Skills](#agent-skills) · [文档](#文档)

## 当前性能

4 核 ARM Neoverse-N1，相同发布模型，3 组工具请求各预热后重复 5 次；各后端在独立持久进程中串行交错运行。热请求使用 [前缀缓存 API](docs/native-engine.md#固定-tools-前缀复用)，下表为 15 次测量的中位数。

| 后端 | 线程 | 解码速度 ↑ | 热请求计算耗时 ↓ |
|---|---:|---:|---:|
| 官方闭源引擎 | 自动 | 488.90 token/s | 47.9 ms |
| OpenNeedle FP32 | 4 | **120.77 token/s** | **244.0 ms** |
| OpenNeedle SDOT | 4 | **150.59 token/s** | **167.7 ms** |
| PyTorch FP32 | 1 | 9.23 token/s | 1825.1 ms |

FP32 / SDOT 解码吞吐分别为表中 PyTorch 的 **13.1× / 16.3×**；SDOT 为官方的 **30.8%**。PyTorch 使用 CPU eager，未启用 `torch.compile`；1/2/4 线程完整结果见 [测速方法与复现](docs/backend-comparison.md)。测试显式设置 `OMP_WAIT_POLICY=PASSIVE`；库不修改全局等待策略。独立引擎与官方接口的计时工作量存在差异，以上为应用层比较。

版本速度对比：[16f2bd3f → a65a9a11](docs/backend-comparison.md#revision-comparison)。

**默认 FP32；SDOT 为可选近似模式，会增加量化误差。** 15 项工具调用质量回归中，官方与两种原生模式均通过 13 项；完整 BFCL 尚未评估。四行 SDOT 与单行算术的逐元素一致测试覆盖 24 种参数组合。结果和支持范围见 [验证报告](docs/results.md) 与 [grammar 文档](docs/grammar.md)。

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

C++ 引擎在 Q/K/V/gate 投影间共享一次 Hadamard 输入变换，直接读取 packed CQ 权重，使用 NEON FMA 或可选 SDOT 整数点积。SDOT 同时计算四个输出行以复用激活加载；普通矩阵运算的输出行数少于 128 时串行执行。权重保持原有压缩行布局，KV 默认使用 FP32，可选按 head 保存尺度的 INT8 缓存；INT8 会引入额外量化误差。

固定工具前缀通过 [NativeEngine 前缀缓存 API](docs/native-engine.md#固定-tools-前缀复用) 复用。native 工具解码将 schema 与 UTF-8 约束编译成 token DFA，只投影合法候选行，单候选时跳过 LM head；大型 grammar 回退到 Python schema 检查。四行内核与单行 SDOT 的逐元素一致测试覆盖 CQ2/CQ4、padding、尾行及 1/2/4 线程。

模型架构与量化依据固定版本的 [Needle 源码](https://github.com/cactus-compute/needle/tree/53df049c4a1a82fca1027b81f9ff21336dfb0861) 和 [发布权重](https://huggingface.co/Cactus-Compute/needle2/tree/32e9e3a93b205f786929697446ae669cf0a84579)。架构、CQ 格式、Arm 指令与 Cactus 内核参考见 [技术参考](docs/research.md)；执行细节和数值边界见 [原生引擎](docs/native-engine.md)。

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
| [技术参考](docs/research.md) | 模型架构、量化格式、版本与参考文献 |

## 许可证

源码采用 [Apache-2.0](LICENSE)，上游归因见 [NOTICE](NOTICE)。官方模型与基线库单独下载，遵循各自的上游许可。
