# 官方、独立实现与 PyTorch 的 CPU 速度对比

本轮在 4 核 ARM Neoverse-N1 上重新测量全部后端，未拼接上一轮数据。模型都是同一份官方 `needle2.cact`；PyTorch 从该发布文件反量化到 FP32，不使用未量化 master。PyTorch 版本为 2.11.0+cu130，但实际设备是 CPU，使用 eager、eval 和 inference_mode，没有启用 torch.compile。

每个配置在独立持久进程中运行，共享相同 CPU 允许集合 0,1,2,3；请求串行交错，没有并行进行模型推理。固定 3 组工具请求，每组先预热，再重复 5 次。原生使用 4 线程，官方自动选择线程；PyTorch 测了 1、2、4 个 intra-op 线程，inter-op 为 1。

| 后端 | 解码 token/s | 请求计算耗时 ms | 首次工具前缀 ms | 正确调用 |
|---|---:|---:|---:|---:|
| 官方闭源库 | 407.80 | 57.7 | 190.3 | 15/15 |
| 独立原生 FP32 | 133.24 | 236.7 | 825.7 | 15/15 |
| 独立原生 SDOT | 168.21 | 159.9 | 657.4 | 15/15 |
| PyTorch eager FP32 / 1线程 | 9.23 | 1822.3 | 805.8 | 15/15 |
| PyTorch eager FP32 / 2线程 | 9.35 | 1803.3 | 672.4 | 15/15 |
| PyTorch eager FP32 / 4线程 | 9.18 | 1871.6 | 780.6 | 15/15 |

解码与请求计算耗时是 15 次测量的中位数；首次前缀成本每进程只观测一次，不是重复测量中位数。独立路径的固定工具前缀为 177 tokens。所有独立模式及 PyTorch 在全部请求中生成相同的 token 序列。

以本轮 PyTorch 2线程的中位数为参照，独立原生 FP32 的解码吞吐为 14.25 倍，SDOT 为 17.98 倍，官方为 43.60 倍。SDOT 达到同轮官方吞吐的 41.25%，请求计算耗时约为官方的 2.77 倍。PyTorch 1/2/4线程吞吐差异约 2%以内，没有明显的多线程收益，不能把该小差距当作稳定的线程排名。

PyTorch 比较慢的是当前参考实现的逐 token 执行：每层有大量 eager 小算子，包含 20轮 Sinkhorn、Hadamard 变换及 KV 拼接/筛选；大矩阵使用解压后的 FP32 权重。原生实现融合了多项计算并使用 packed CQ/NEON，SDOT 另外采用整数 dot-product。这个结果不代表经过 torch.compile、专用融合算子或 GPU 优化后的 PyTorch 上限。SDOT 的额外量化误差仍存在，本轮调用一致不能将其视为无损加速。

## 计时边界

- 工具前缀预计算一次并复用。热请求包括前缀状态恢复、query prefill、grammar 选择及逐 token decode；不包括模型加载、首次编译和首次工具初始化。PyTorch 复用只读 prefix cache，不需要人为复制 KV。
- 为避免无用计算，原生与 PyTorch 都不计算前缀词表输出；query prefill 只投影最后位置，decode 每步计算一个位置。PyTorch 运算来自原有模型的 _run 与 LM head，没有改变模型数学定义。
- 独立路径的 query 预分词和输出字符串解析在计时外；官方公开接口的外部 wall time 包含其内部输入处理以及 JSON 结果解析。因此此处是接口计算成本的应用比较，并非逐项完全相同的端到端服务测试。
- 独立 decode TPS 的分子是实际 forward 步数 N−1，计时包含 grammar；官方 decode/prefill TPS 是其响应内自报值，内部 prompt、grammar、验证和置信度计算仍有差异，不能把 TPS 比值称作同算子加速比。
- 每请求间隔20 ms缓解空闲线程池竞争，该间隔和IPC不计入被测进程的时间。这个措施不能保证完全消除云主机调度或第三方运行时的影响；报告保留全部样本、process CPU时间及workload。
- 本轮进程隔离方法与之前 comparison*.json 不同，官方实测速率也会波动。41% 与上一轮约35%的差别不代表又完成了一次代码提速，应以同轮结果比较。

## 复现

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/benchmark_backends.py \
  --native-threads 4 --torch-threads 1,2,4 --repeat 5 \
  --output reports/backend_comparison.json
```

脚本：[benchmark_backends.py](../scripts/benchmark_backends.py)。完整数据：[backend_comparison.json](../reports/backend_comparison.json)，包含初始化、全部请求、线程配置、模型/源码哈希、token一致性及运行负载。此次 `source_changed_during_run=false`，六个后端均15/15正确。
