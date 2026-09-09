# 维护脚本

从仓库根目录运行，用 `python scripts/<文件名> --help` 查看参数。推理和转换统一通过 `python -m needle2`；下列脚本负责下载、开发验证和性能分析。

| 用途 | 入口 |
|---|---|
| 下载固定版本模型与可选官方平台文件 | `download_official.py` |
| 微调与 QAT | `finetune.py` |
| 权重与数值验证 | `validate.py`、`validate_jax.py`、`validate_sdot.py` |
| 官方引擎加载转换结果的兼容性检查 | `compare_official_exports.py` |
| 工具调用质量评估 | `evaluate_quality.py` |
| 多后端真实请求比较 | `benchmark_backends.py` |
| 压缩矩阵和固定 token 引擎微基准 | `benchmark_native.py` |
| 官方基线单独测速及比较库准备 | `benchmark_official.py` |
| 原生阶段耗时分析 | `profile_native.py` |

这三个测速入口测量不同范围，不能直接比较其吞吐。测试依赖、历史结果与计时方法见[基准与验证](../docs/benchmark.md)，转换和训练见[使用指南](../docs/usage.md)。输出写入被 Git 忽略的 `artifacts/`；大型模型、原始报告及一次性实验不随源码交付。
