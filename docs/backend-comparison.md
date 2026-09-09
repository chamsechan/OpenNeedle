# CPU 性能与测量方法

当前实测实现为 `f4f9b38`（2026-09-09），同轮原生基线为 `e809ffc`。硬件为 4 核 ARM Neoverse-N1，使用同一份官方 CACT。原始样本、配置、官方库/模型哈希和数值验证汇总保存在 [performance_f4f9b38.json](../reports/performance_f4f9b38.json)。

## 官方与优化后 OpenNeedle

OpenNeedle 使用 4 线程、SDOT＋INT8 KV；这是可选近似配置，默认仍为 FP32＋FP32 KV。

| 指标 | 官方 2.0.4 | OpenNeedle f4f9b38 | OpenNeedle 相对官方 |
|---|---:|---:|---|
| 基础热请求计算耗时 | 45.1 ms | 60.3 ms | 耗时高 33.6% |
| 扩展热请求计算耗时 | 439.1 ms | 87.4 ms | 耗时低 80.1% |
| 扩展 query prefill 耗时 | 未公开 | 32.3 ms | 无法直接比较 |
| 扩展解码吞吐 | 493.7 token/s | 327.4 token/s | 吞吐低 33.7% |
| 质量集正确调用 | 13/15 | 13/15 | 该样本得分相同 |

基础与扩展集分别包含 3、16 个独立请求；每例预热一次。官方沿用同机较早一轮、每例 5 次测量的数据；新版每例测量 9 次。**官方与新版并非同轮交错运行。** 所有时间/吞吐为中位数。官方 TPS 为自报值，内部 token 数、prompt、validation 及计时范围不公开；不能将吞吐比值当作相同算子的加速比，也不能从扩展请求耗时推断解码内核比官方更快。

<a id="revision-comparison"></a>

## 同轮版本对比：e809ffc → f4f9b38

两个原生版本使用相同的 SDOT＋INT8 KV、4 线程配置，在独立常驻进程中串行交错测量，每例预热 1 次、测量 9 次。基础集各 27 个计时样本，扩展集各 144 个。

| 用例集 | 指标 | e809ffc | f4f9b38 | 变化 |
|---|---|---:|---:|---:|
| 基础 | 请求计算 ms | 82.80 | 60.29 | −27.2% |
| 基础 | Query prefill ms | 28.87 | 22.54 | −21.9% |
| 基础 | Decode token/s | 329.68 | 412.30 | +25.1% |
| 扩展 | 请求计算 ms | 132.77 | 87.40 | −34.2% |
| 扩展 | Query prefill ms | 41.23 | 32.33 | −21.6% |
| 扩展 | Decode ms | 90.08 | 54.94 | −39.0% |
| 扩展 | Decode token/s | 201.31 | 327.40 | +62.6% |

每列独立取中位数，不能相加还原请求中位数。19 个独立用例的全部测量均调用正确，优化前后生成 token 完全相同。云主机有明显长尾，这些比例不代表所有输入和硬件的固定收益。

首次工具前缀准备含 native prefill、快照和 DFA 编译，不在热请求计时中。扩展集单次观测：基线 1565 ms、新版 1360 ms；不是冷启动中位数。更早版本和 PyTorch 的测量保留在 [历史文档](https://github.com/chamsechan/OpenNeedle/blob/0d7c15d/docs/backend-comparison.md)，未重新测量的数据不用于宣称当前性能。

## 实现变化与计时边界

- **DFA 验证复用**：构造时验证完整图，冻结字段、使用不可变 bytes 存储；热请求仅检查当前词表上界。原始 C ABI 的结构检查保留。
- **Attention**：复用 GQA 工作列表和 KV 槽位映射，NEON V 累加使用 32 维寄存器分块。固定前缀 sink 与滑窗语义不变。
- **Prefill**：SDOT 相邻 token 共享 packed 权重解包，Prepared 缓冲区在投影间复用；RoPE 三角函数每个 prompt 块只计算一次，供所有层读取。Decode 的四行 SDOT 保留。
- C++ 常驻线程池执行模型算子与 DFA 解码；只计算合法候选的 LM head，单候选跳过投影。大型 DFA 回退到 Python schema 检查。
- 热请求包括 prefix restore、query prefill、grammar 和 decode，不含模型加载、首次编译与工具初始化。Native 查询分词和最终解析在计时外，官方 wall 包括其 API 内部处理；这不是完全相同处理链的服务端到端延迟。
- Native decode TPS = 实际模型 forward 次数 N−1 / 含 grammar 的 decode 时间；首 token 来自 prefill。请求前间隔 20 ms 和 IPC 不计时。
- CPU affinity 为 `0,1,2,3`；进程环境为 `OMP_WAIT_POLICY=PASSIVE`、`OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`，native 引擎显式使用 4 线程。库不修改全局等待策略。

实现细节见 [原生引擎](native-engine.md)、[grammar](grammar.md)。四种计算模式在本轮优化前后的全词表 logits 均逐位一致；这不消除 SDOT/INT8 原有量化误差，详见 [精度验证](results.md)。

## 复现

下载相同模型后，在该主机上复现版本对照的脚本如下。基线目录应为空；脚本使用固定的 `e809ffc` 源码目录和当前 checkout，避免在不同提交间切换运行中的进程。

```bash
mkdir -p /tmp/needle2-e809ffc
git archive e809ffc | tar -x -C /tmp/needle2-e809ffc
OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 reports/published_f4f9b38/benchmark.py
```

此文档提交的运行时代码与 `f4f9b38` 相同；脚本使用当前 checkout，以后在新运行时提交运行得到的是新提交的性能。脚本保存全部生成 token 和逐请求计时，并检查两个版本的输出相等、调用正确以及测量期间源码未变化。

如需重新进行官方、native 和 PyTorch 的同轮扩展集比较，可运行现有通用脚本。它会产生新的测量，不能把其结果当作上表历史记录：

```bash
OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 scripts/benchmark_backends.py \
  --tools benchmarks/expanded_tools.json --cases benchmarks/expanded_cases.jsonl \
  --native-threads 4 --torch-threads 1 --kv-cache int8 --repeat 9 \
  --max-new-tokens 192 --output reports/backend_comparison_current.json
```

先按 [官方下载脚本](../scripts/download_official.py) 准备权重和比较库。测试期间避免同时编译或运行其他 CPU 密集任务。首页 SVG 从已保存的数据生成，不会触发测速。
