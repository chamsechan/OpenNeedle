---
name: openeedle-inference
description: 安装 OpenNeedle、运行 Needle 2 工具调用推理，或接入原生 CPU/PyTorch 后端与前缀缓存。用于 OpenNeedle 的环境准备、推理排障和应用集成。
---

# OpenNeedle 推理

完成从环境准备到有效工具调用输出的流程，沿用用户指定的模型、工具 schema 和后端。

## 定位项目

先定位 OpenNeedle checkout：根目录包含 `pyproject.toml`（项目名 `needle2-open`）、
`needle2/cli.py` 和 `scripts/download_official.py`。以下路径均相对项目根目录，
与 skill 安装目录无关。没有 checkout 时，在用户的工作目录克隆
`https://github.com/chamsechan/OpenNeedle.git`。复用已有 Python 环境和模型文件。

## 安装与最小推理

需要 Python ≥ 3.10；原生编译需要 C++17 与 OpenMP。Linux ARM64 是已验证平台。
依赖或模型缺失时执行相应步骤：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/download_official.py
```

下载器获取固定版本的部署模型、FP16 master 与 tokenizer；已有用户模型时直接使用。
在同一环境中运行：

```bash
python -m needle2 run artifacts/official/needle2.cact \
  --tools examples/tools.json \
  --prompt 'Turn on the kitchen light.' \
  --backend native --threads 4
```

检查退出状态、输出 JSON 中的 `function_calls` 和参数。此示例期望
`set_light`，参数为 `{"room":"kitchen","on":true}`。命令只生成调用，
应用负责执行工具。交付实际命令、环境和输出，不把成功加载等同于任务成功。

## 后端与集成

- 默认原生 FP32；按目标 CPU 调整 `--threads`。首次运行会自动编译内核，
  缓存位于 `~/.cache/needle2`，无需官方库。
- `--matmul sdot` 是额外量化激活与码本的近似模式，要求 ARM DotProd。
  先检查平台与能力，比较目标案例输出后再采用；初始化失败时报告原因，
  不把回退到 FP32 的结果标成 SDOT。
- `--backend torch` 支持 `.cact` 和转换后的模型目录。原生后端使用 `.cact`。
  `--prefill-backend torch` 的混合路径额外保留约 175 MB FP32 权重。
- 工具 schema 支持范围见项目的 `docs/grammar.md`；保持默认 grammar，
  超出支持范围时先说明限制，再按用户用途调整 schema。
- 接入 Python 或复用工具前缀时，读取 `docs/usage.md` 的对应示例。
  前缀快照必须与相同模型和工具 token 序列配套；CLI 的每次新进程不共享缓存。

编译和能力检测细节见 `docs/native-engine.md`。排障先保留实际错误、平台和
编译器信息，再修复对应依赖或配置。
