---
name: openeedle-convert
description: 将 Needle 2 官方 CACT 或 FP16 master 转为 PyTorch，并将相同架构的 PyTorch checkpoint 导出为 CQ2/CQ4 CACT。用于 OpenNeedle 双向转换、量化导出与往返一致性检查。
---

# OpenNeedle 模型转换

定位包含 `needle2/cli.py` 和项目名 `needle2-open` 的 OpenNeedle checkout，
在根目录使用已安装项目依赖的 Python。本文中的文档和产物路径均相对该根目录，
不相对 skill 安装目录。输入文件、输出位置和是否重新量化按用户要求选择。

## 选择输入路径

先查看输入类型；转换器面向 Needle 2 架构。

```bash
# 部署权重 → 可训练的 PyTorch FP32
python -m needle2 to-torch \
  artifacts/official/needle2.cact artifacts/pytorch

# FP16 master → PyTorch，保留模板中的部署元数据
python -m needle2 to-torch \
  artifacts/official/checkpoints/needle2.pkl artifacts/pytorch_master \
  --template artifacts/official/needle2.cact
```

以上是两种可选入口，只执行任务需要的一种。已有输出时复用经检查的结果，
或选择新目录，避免覆盖训练权重。部署模型的反量化值保留原有量化误差；
需要浮点 master 精度时使用第二种入口。

转换目录必须完整保留 `weights.safetensors`、`config.json`、`source.cact`。
后两者包含模型结构、tokenizer、码本与来源信息；单独的 safetensors 不足以
完成本项目的量化导出。缺少匹配元数据时先找回原目录或对应模板，不能自行猜测。

## 量化与验证

```bash
python -m needle2 quantize artifacts/pytorch artifacts/roundtrip.cact
python -m needle2 inspect artifacts/roundtrip.cact
```

默认恢复未修改张量的原始压缩字节，仅重新量化修改过的张量。
`--force-requantize` 会重算全部 CQ 权重，适用于明确要求独立重新量化的实验。
沿用模板的混合位宽、内嵌码本和 FP16 norms。

- 对未修改部署模型的往返转换，比较输入输出的 SHA-256 或完整字节；应一致。
- 对 master 或训练权重重新量化，检查文件可读取，并用导出的 `.cact`
  执行用户的工具调用样例。量化后质量应在保留样本上评估。
- 用户要求官方兼容验证时，通过项目已有官方比较脚本加载导出文件；
  报告实际完成的加载和调用结果，区分文件一致、logits 误差与任务正确率。

交付输出路径、输入来源、量化方式、哈希及验证结果。
Python 保存接口与训练后导出见 `docs/usage.md`；逐张量格式和舍入规则见
`docs/research.md`；CLI 参数以 `python -m needle2 <command> --help` 为准。
