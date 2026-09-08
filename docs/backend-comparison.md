# CPU 性能与测量方法

测试设备为 4 核 ARM Neoverse-N1，所有后端使用相同的官方 `needle2.cact`。PyTorch 2.11.0+cu130 实际运行于 CPU，权重从发布文件反量化为 FP32，使用 eager、eval、inference_mode；未启用 torch.compile。

各配置使用独立持久进程、相同 affinity `0,1,2,3`。3 组工具请求各预热一次，再重复 5 次；请求串行交错。原生使用 4 线程，官方自动选择线程；PyTorch intra-op 分别为 1/2/4，inter-op 为 1。进程启动前统一设置 `OMP_WAIT_POLICY=PASSIVE`、`OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`；各后端再按上述线程配置执行。

## 测量结果

| 后端 | 解码 token/s | 请求计算耗时 ms | 首次工具前缀 ms | 正确调用 |
|---|---:|---:|---:|---:|
| 官方闭源库 | 504.70 | 51.7 | 194.7 | 15/15 |
| OpenNeedle FP32 / 4 线程 | 124.40 | 210.1 | 1064.7 | 15/15 |
| OpenNeedle SDOT / 4 线程 | 152.30 | 168.4 | 793.5 | 15/15 |
| PyTorch eager FP32 / 1 线程 | 9.43 | 1792.7 | 775.5 | 15/15 |
| PyTorch eager FP32 / 2 线程 | 8.63 | 1945.9 | 812.6 | 15/15 |
| PyTorch eager FP32 / 4 线程 | 8.35 | 1978.7 | 818.6 | 15/15 |

解码速度与请求耗时为 15 次测量的中位数；首次前缀成本为每进程一次观测。独立路径固定工具前缀为 177 tokens。所有独立模式与 PyTorch 在全部请求中生成相同 token 序列，六种配置均为 15/15 正确调用。这是固定工具请求的性能测试，质量回归见 [验证结果](results.md)。

README 展示 PyTorch 1 线程配置，它是这组测试中吞吐最高的 PyTorch 配置。以它为参照，FP32 和 SDOT 解码吞吐分别为 13.20 倍和 16.16 倍；SDOT 为官方的 30.18%。这些比值描述给定应用和接口，不是相同算子工作量下的内核加速比。

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
