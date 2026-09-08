# CPU 性能与测量方法

测试设备为 4 核 ARM Neoverse-N1，所有后端使用相同的官方 `needle2.cact`。PyTorch 2.11.0+cu130 实际运行于 CPU，权重从发布文件反量化为 FP32，使用 eager、eval、inference_mode；未启用 torch.compile。

各配置使用独立持久进程、相同 affinity `0,1,2,3`。3 组工具请求各预热一次，再重复 5 次；请求串行交错。原生使用 4 线程，官方自动选择线程；PyTorch intra-op 分别为 1/2/4，inter-op 为 1。进程启动前统一设置 `OMP_WAIT_POLICY=PASSIVE`、`OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`；各后端再按上述线程配置执行。

## 测量结果

| 后端 | 解码 token/s | 请求计算耗时 ms | 首次工具前缀 ms | 正确调用 |
|---|---:|---:|---:|---:|
| 官方闭源库 | 488.90 | 47.9 | 194.9 | 15/15 |
| OpenNeedle FP32 / 4 线程 | 120.77 | 244.0 | 942.0 | 15/15 |
| OpenNeedle SDOT / 4 线程 | 150.59 | 167.7 | 694.9 | 15/15 |
| PyTorch eager FP32 / 1 线程 | 9.23 | 1825.1 | 772.1 | 15/15 |
| PyTorch eager FP32 / 2 线程 | 8.57 | 1950.8 | 810.7 | 15/15 |
| PyTorch eager FP32 / 4 线程 | 8.36 | 1993.6 | 791.1 | 15/15 |

解码速度与请求耗时为 15 次测量的中位数；首次前缀成本为每进程一次观测。独立路径固定工具前缀为 177 tokens。所有独立模式与 PyTorch 在全部请求中生成相同 token 序列，六种配置均为 15/15 正确调用。这是固定工具请求的性能测试，质量回归见 [验证结果](results.md)。

README 展示 PyTorch 1 线程配置，它是这组测试中吞吐最高的 PyTorch 配置。以它为参照，FP32 和 SDOT 解码吞吐分别为 13.08 倍和 16.31 倍；SDOT 为官方的 30.80%。这些比值描述给定应用和接口，不是相同算子工作量下的内核加速比。

## 执行方式

- Native 在 C++ 内完成逐 token 模型前向，直接读取压缩 CQ 权重；Q/K/V/gate 共享输入变换与并行区域。
- SDOT 同时计算四行以共享激活加载；普通矩阵运算少于 128 个输出行时串行执行。KV 与 mHC dense routing 使用 FP32。
- 工具前缀快照保存各层 KV、token 历史和 Engram 环；热请求恢复快照后消费 query。该复用通过 NativeEngine API 显式执行。
- 解码计算完整词表 logits，再应用 grammar；prefill 仅在需要输出的位置计算 LM head。
- PyTorch 参考路径包含逐 token eager 小算子、20 轮 Sinkhorn、Hadamard 和 KV 操作。它的结果不代表编译或融合优化后的 PyTorch 上限。

SDOT 对旋转激活和码本引入 INT8 舍入。四行与单行 SDOT 的逐元素一致性不消除这种 FP32→SDOT 误差，详见 [数值验证](results.md) 和 [内核说明](native-engine.md)。

## 计时边界

- 热请求包括前缀恢复、query prefill、grammar 和逐 token decode；不包括模型加载、首次编译及工具初始化。
- 独立路径 query 预分词和最终字符串解析位于计时外；官方 wall time 包含其公开接口内部输入处理与 JSON 解析。因此表中是请求计算耗时，不是完整服务端到端延迟。
- 独立 decode TPS = 实际 forward 次数 N−1 / 含 grammar 的 decode 时间。首 token 来自 prefill，最后选出的 token 不额外 forward。官方 TPS 由其响应自报，内部 prompt、grammar、验证与置信度流程不同。
- 各请求间隔 20 ms，帮助空闲运行时线程退出忙等待；间隔和 IPC 不计入 worker 时间。报告保留全部样本与 process CPU time，云主机调度仍可能影响结果。
- FP32 前缀快照和可选 PyTorch dense prefill 各有额外内存成本；本表不宣称与官方 INT8 引擎具有相同 RSS。

## 复现

```bash
OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/benchmark_backends.py \
  --native-threads 4 --torch-threads 1,2,4 --repeat 5 \
  --output reports/backend_comparison.json
```

完整数据：[backend_comparison.json](../reports/backend_comparison.json)。报告记录模型、源码和官方库哈希，线程环境、启动成本、全部请求及 token 一致性；`source_changed_during_run=false`。运行性能测试时避免同时进行编译或其他 CPU 密集任务。

<a id="revision-comparison"></a>

## 版本速度对比：16f2bd3f → a65a9a11

本次在同一台主机按上述配置重新测量两个提交，先运行 16f2bd3f，再运行 a65a9a11；表内均为 15 次测量的中位数。当前页面及 README 已同步为 a65a9a11 的本次数据。

| 后端 | 16f2bd3f token/s | a65a9a11 token/s | 吞吐变化 | 16f2bd3f 请求 ms | a65a9a11 请求 ms | 耗时降低 |
|---|---:|---:|---:|---:|---:|---:|
| official | 500.90 | 488.90 | -2.40% | 59.0 | 47.9 | +18.87% |
| native_fp32 | 116.51 | 120.77 | +3.66% | 225.8 | 244.0 | -8.04% |
| native_sdot | 149.75 | 150.59 | +0.56% | 175.1 | 167.7 | +4.19% |
| pytorch_t1 | 9.39 | 9.23 | -1.69% | 1817.1 | 1825.1 | -0.44% |
| pytorch_t2 | 8.42 | 8.57 | +1.79% | 2044.1 | 1950.8 | +4.56% |
| pytorch_t4 | 8.58 | 8.36 | -2.57% | 1958.6 | 1993.6 | -1.79% |

两个版本的模型、官方库、测试脚本、工具、案例与线程设置一致，源码哈希均未在运行中变化。各版本六个后端均正确调用 15/15；独立后端的完整生成 token 序列在版本内及跨版本均一致。

代码变化集中于 SDOT 四行激活复用和独立 SDOT 矩阵乘的小矩阵并行阈值，FP32 计算路径未修改。两个版本不是逐请求交错测量；官方与 PyTorch 对照也可能出现漂移，因此以上变化包含主机调度噪声，不代表统计显著性或所有负载的固定加速比。

原始数据：[16f2bd3f](../reports/backend_comparison_16f2bd3f.json)、[a65a9a11](../reports/backend_comparison.json)、[版本对比汇总](../reports/revision_comparison.json)。复现时在两个提交的独立 worktree 中串行执行上方命令，以 `--model`、`--library` 指向相同文件，并为 `--output` 指定不同路径。
