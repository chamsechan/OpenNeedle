---
name: openeedle-benchmark
description: 复现 OpenNeedle、官方 Needle 2 引擎与 PyTorch 的 CPU 性能对比，分析已有测速报告及精度代价。用于吞吐、延迟、线程配置与后端选择的实测评估。
---

# OpenNeedle 性能对比

定位包含 `scripts/benchmark_backends.py` 和项目名 `needle2-open` 的 checkout，
从根目录工作。以下路径均相对 checkout，不相对 skill 安装目录。
用户只要求解释已有结果时，读取 `reports/backend_comparison.json` 和
`docs/backend-comparison.md`；需要目标机器数据时才运行实测。

## 检查平台与测量条件

先读取 `docs/backend-comparison.md` 的计时边界，并检查脚本的 `--help`。
当前三方比较脚本使用 Linux CPU affinity，固定包含 SDOT 后端，因此要求
Linux ARM64、DotProd 和可用的官方动态库。其他平台可解释已有报告，或使用
适用的单后端入口；三方脚本不支持用参数跳过 SDOT，不要虚构该选项。

获取 `os.sched_getaffinity(0)` 中实际允许的 CPU IDs；不能假定容器可用核为 0–3。
按选定核数设置原生及 PyTorch 线程。各后端使用同一部署 `.cact`、工具 schema、
请求集、生成上限与 CPU 集合；PyTorch 基线由部署模型反量化得到。
保持请求串行，测量期间避免同时编译、训练或修改引擎源码。

## 执行

官方库缺失时，用固定版本下载入口准备基线：

```bash
python scripts/benchmark_official.py --fetch-library --repeat 1 \
  --output artifacts/benchmarks/official_setup.json
```

下面示例适用于允许使用 CPU 0–3 的机器；根据实际 affinity 调整：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/benchmark_backends.py \
  --model artifacts/official/needle2.cact \
  --tools examples/tools.json --cases benchmarks/cases.jsonl \
  --affinity 0,1,2,3 --native-threads 4 --torch-threads 1,2,4 --repeat 5 \
  --output artifacts/benchmarks/backend_comparison.json
```

将新实验写入单独的输出路径；只有用户要更新项目基准时才替换 `reports/` 中的发布数据。
案例 JSONL 格式读取 `benchmarks/cases.jsonl`；请求必须共享同一工具前缀。
脚本已经实现独立持久进程、预热和串行交错采样，不需要外部并行启动多个副本。

## 解读与交付

报告硬件、CPU 集合、线程、模型哈希、重复次数，并检查报告中的源码变更状态、
错误、调用正确率和生成 token 一致性。表格列出 decode token/s 中位数、
热请求计算耗时中位数、首次工具前缀成本；首次前缀只有每进程一次观测。

速度比只能从同一轮可比数据计算。官方 TPS 为内部自报值，独立路径包含 grammar，
计时工作量存在差异；标为应用层比较。PyTorch 是 CPU eager 基线。
SDOT 额外量化激活和码本，调用一致不能证明 logits 无误差。
测速案例与质量保留集分开报告；需要数值诊断时检查 `scripts/validate_sdot.py --help`。

用户要求更新首页时，先核对新的原始报告，再同步 README 表格与说明；
运行 `python scripts/render_readme_assets.py` 更新 SVG。
生成器中的硬件、模型标签及 PyTorch 基线 key 含固定值，换机器或配置时需同步调整，
并遵循 `docs/assets/README.md` 检查图中标注。
