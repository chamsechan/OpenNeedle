# 解码优化逐项实验（2026-09-09）

基于 `32005c7818aa44bdd9a3f6750e9f92214c1b5931`，按大候选投影、attention 并行策略、唯一候选批量推进的顺序验证。这批候选最终没有修改默认运行时：投影的局部收益未转化为稳定的请求级收益，attention 串行化明显退化，当前用例没有唯一候选链。保留独立实验代码、原始数据与回归测试。没有证据支持此前估计的 Expanded TPS 提升 20%～30%。

后续已采用的 [mHC](mhc-optimization.md) 和 [QK](attention-optimization.md) 优化另有报告。

## 测量边界

硬件为 ARM Neoverse-N1，固定 CPU 0–3；请求对照使用 4 个 native 线程、SDOT、INT8 KV、相同模型及工具 schema。每个版本在独立常驻进程中运行，先完成工具前缀缓存及 DFA 编译；每个用例预热一次，再执行 5 次轮换顺序的串行交错测量，请求前间隔 20 ms。包含 basic 3 例、expanded 16 例。计时包含 prefix restore、query prefill、grammar 和 decode，不含模型加载和首次工具初始化。

所有请求均检查完整生成 token 序列与同轮基线一致，以及工具调用与预期一致。云主机仍有噪声，几百分点以内的差别不作为稳定吞吐提升的证明。各列中位数独立计算；decode TPS 是生成 token 数减一除以 decode 时间，不是单个算子的速度。

## 1. 大候选投影

对照版本：

- `baseline`：原逐候选 `row`。
- `dense512`：候选数达到 512 时，使用全量四行投影再 gather；用于验证原建议。
- `batch4`：四个任意候选共同投影；只存在于生成的实验源码中。
- `dense6144`：提高全量切换阈值到 6144，用于后续对照。
- `dense7680`：最终保守候选方案，同时要求候选数至少 7680 且覆盖词表的 15/16；最终未采用。

投影微基准独立于模型层前向，测试 128–8192 个候选、1/2/4 线程、排序及随机排列。每种情况预热 5 次、测量 31 次。所有被比较的候选 logits 逐位一致。原始数据见 [微基准](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports/decode_projection_micro.json)。该微基准的输入来自一个四 token 前缀，不能代表所有 hidden 状态或所有硬件。

4 线程、8037 个候选的中位数：

| 候选排列 | 原逐行投影 | 全量投影＋gather | 任意四候选投影 |
|---|---:|---:|---:|
| 按 ID 排序 | 约 0.081 ms | 约 0.075 ms | 约 0.070 ms |
| 随机顺序 | 约 0.189 ms | 约 0.079 ms | 约 0.129 ms |

排序对原逐行路径的缓存局部性影响很大，不能将随机候选的收益直接外推到真实 DFA。6144 个排序候选时，全量路径在 1/2/4 线程均略慢，所以最终阈值进一步收紧。512 阈值在较小候选集上做了不必要的全量计算，未采用。

[首轮请求对照](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports/decode_projection_experiment.json)中，Expanded decode TPS 中位数分别为 331.4（原路径）、337.9（dense512）、336.0（batch4）；不足以支持 20%～30% 的收益预期。没有把任意四行内核加入生产代码，避免为当前工作负载的微小收益扩充内核实现。

保守候选方案仅影响 SDOT 大候选分支；保留小候选路径、候选输出顺序和并列分数的处理。首次需要时分配、随后复用一个全词表 float 缓冲区，在当前 8192 词表下增加 32 KiB。

[最终同轮请求对照](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports/decode_optimization_final.json)：

| 用例集 | 基线请求 ms | dense7680 请求 ms | 基线 decode TPS | dense7680 decode TPS |
|---|---:|---:|---:|---:|
| Basic | 59.15 | 59.57 | 407.35 | 405.25 |
| Expanded | 86.31 | 88.30 | 335.58 | 332.32 |

这轮差异方向与此前部分实验不同，不能认定有稳定请求级收益。因此撤回默认运行时修改，保留 `dense7680` 为独立实验变体。没有将这些小幅波动解释为确定的加速或确定的回退。

## 2. Attention 小范围对照

在 `dense6144` 基础上比较：原并行策略、仅 decode 全串行、仅 decode 有效 attention 长度不超过 64 时串行。prefill 不变。原始数据见 [attention 对照](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports/decode_attention_experiment.json)。

| 用例集 | 原并行 decode ms | decode 全串行 ms |
|---|---:|---:|
| Basic | 36.65 | 52.61 |
| Expanded | 53.67 | 96.16 |

逐样本配对耗时比的中位数分别为 1.467 和 1.849，串行明显退化。四个工作项各自包含整个上下文的计算；任务数少不足以证明线程开销主导。没有保留串行化，也没有据此推断所有短上下文都应并行。64 阈值版本在当前工具前缀较长的工作负载上不能证明短上下文收益。

本次小范围对照未调整矩阵并行阈值或实现二线程子调度；没有测得这些策略的收益。

## 3. 唯一候选连续批量推进

将同轮生成的完整 token 序列逐个重放到实际 DFA，统计首 token 之后每个 decode 步骤的候选数量，以及连续唯一候选链。见 [机会统计](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports/decode_chunk_opportunities.json)。

19 个用例没有任何单候选步骤，因此也没有可合并的唯一候选链。JSON 字节上的固定骨架不等于 tokenizer/DFA 下只有一个合法 token。当前基准无法从这项优化获益，故没有加入 chunking 运行时代码。该结论只适用于这组工具与请求，不否定其他 schema 上的潜力。

## 验证与复现

新增边界测试覆盖 7679/7680、8037、8192 及小候选集，包含乱序、重复候选，1/2/4 线程，FP32/INT8 KV，多步 hidden 和后续完整 logits 一致性。候选投影、既有 row4 和 native grammar 测试在候选改动上共 33 项通过，撤回后默认路径同样验证通过。

这组一次性实验脚本已从当前版本移除。原脚本及完整复现步骤见[清理前的历史版本](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/docs/decode-experiments.md)；需要复现时，请在该提交的独立 checkout 中按历史说明运行。

当前默认实现仍可运行：

```bash
python -m pytest -q tests/test_candidate_projection_dense.py tests/test_sdot_row4.py tests/test_native_grammar.py
```
