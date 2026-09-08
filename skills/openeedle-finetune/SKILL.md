---
name: openeedle-finetune
description: 使用 OpenNeedle 对 Needle 2 做监督微调或 CQ 量化感知训练，准备工具调用 JSONL 数据并导出部署权重。用于已有训练目标与数据的微调、QAT 和训练后验证。
---

# OpenNeedle 微调与 QAT

定位包含 `scripts/finetune.py` 和项目名 `needle2-open` 的 OpenNeedle checkout，
从项目根目录工作。文档和数据路径相对 checkout，不相对 skill 安装目录。
确认用户的训练目标、数据路径、设备与训练规模；缺失的真实标签或保留集需向用户获取。
单步示例仅用于验证训练流程。

## 输入与数据

优先从官方 FP16 master 转换的 canonical checkpoint 开始。目录需要
`weights.safetensors`、`config.json`、`source.cact` 三个文件。
具体转换入口见项目 `README.md` 的“模型转换”。

训练数据为 JSONL，每行包含：

- `tools`：实际工具 schema 列表，格式参考 `examples/tools.json`。
- `query`：用户请求字符串。
- `answers`：`[{"name":"set_timer","arguments":{"minutes":5}}]` 形式的目标调用。
- 可选 `system`、`reasoning`。

检查调用名称、必需参数和类型与 schema 一致，并检查序列长度。
脚本使用内嵌 tokenizer 与 `render_prompt`，只监督回答部分；超过模型
`max_seq_len` 的样本会报错，需要明确缩短。数据准备示例见 `docs/usage.md`
的“CQ QAT 与监督微调”。

## 训练与导出

替换示例路径与参数为用户的实验配置；输出必须是新目录：

```bash
python scripts/finetune.py \
  artifacts/pytorch_master artifacts/training/train.jsonl \
  artifacts/experiments/qat_run \
  --qat --steps 20 --lr 1e-5 --device cpu --threads 1 --seed 0

python -m needle2 quantize \
  artifacts/experiments/qat_run artifacts/experiments/qat_run.cact
```

普通监督微调省略 `--qat`。该 CLI 的 `--qat` 同时启用 CQ 权重 QAT 与
公开 A8 fake quant；仅做权重 QAT 时按 `docs/usage.md` 使用 Python API。
CPU 全模型 QAT 成本较高；步骤数、设备和学习率是实验参数，不能作为质量承诺。

自定义训练代码时，先移动模型到设备，再 `enable_qat()`，之后创建 optimizer。
保存前调用默认的 `disable_qat()`，恢复更新后的 master 权重。
`save_torch_weights()` 写入已有 canonical 目录，创建新产物时先复制源目录，
保留 config 和 source；脚本输出不含 optimizer 或 scheduler 恢复状态。

## 验证与交付

检查 loss 有限、输出目录完整及导出文件可加载。用独立保留集对比训练前模型、
训练后 PyTorch 和量化后 `.cact` 的工具调用正确率；报告各自样本数和错误案例。
缺少保留集时明确仅完成流程验证。
交付数据来源、训练配置、`training.json`、checkpoint 和部署文件路径。
