# 项目维护脚本

从仓库根目录运行。使用 `python scripts/<文件名> --help` 查看参数；验证与测速通常需要先下载模型，部分路径需要官方库或 ARM DotProd 硬件。原始输出默认写入被 Git 忽略的 `artifacts/`。

| 用途 | 脚本 |
|---|---|
| 下载模型与公开参考资料 | `download_official.py` |
| 转换格式的兼容入口 | `convert_to_pytorch.py`、`quantize_to_cact.py`；新用法优先采用 `python -m needle2 to-torch` / `quantize` |
| 微调和 QAT | `finetune.py` |
| 权重、转换和数值验证 | `validate.py`、`validate_jax.py`、`validate_sdot.py`、`compare_official_exports.py` |
| 工具调用质量评估 | `evaluate_quality.py` |
| 通用测速 | `benchmark_backends.py`、`benchmark_native.py`、`benchmark_official.py` |
| 原生引擎阶段耗时分析 | `profile_native.py` |
| 从测量总结生成首页图片 | `render_readme_assets.py`（直接运行，无命令行参数） |

绑定旧提交的优化对照、一次性插桩、专用查表微基准和早期重复测速入口已移除。历史实验的脚本和原始数据仍可在 Git 历史中查阅；相关技术文档提供固定提交链接。正式推理实现及 `tests/` 回归测试继续保留。
