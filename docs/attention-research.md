# Attention 优化调查（2026-09-09）

后续已实现固定 64 维 QK 路径，并完成真实请求对照；见 [实现与验证报告](attention-optimization.md)。以下保留优化前的调查数据。

## 结论

存在值得验证的空间，重点是 QK 打分与 V 加权。当前 `matmul="sdot"` 加速 CQ 投影，但 attention 的 QK 仍是 FP32 Q 与转为 FP32 的 INT8 K 做浮点乘加。官方发行库存在由 `forward_range` 调用的 64 维双路整数点积路径，这是具体的实现差异；不过把 Q 量化为 INT8 会增加近似，不能视为逐位无损优化。

本轮仅增加独立插桩和调查记录，没有修改生产 attention 实现，保留上一轮 mHC 优化。没有测得或承诺新的性能提升。

## 真实请求阶段测量

使用当前 mHC 优化后的实现、4 个 native 线程、CPU 0–3、SDOT 投影、INT8 KV；19 个真实工具调用用例，每例预热一次、测量三次，检查生成 token 与基线一致及预期工具调用。原始结果和源码/库哈希见 [profile 报告](../reports/real_decode_profile_attention_detail.json)。

下表是跨层累计、按输出 token 归一化的主线程阶段耗时，不是单层耗时。

| 阶段 | Basic ms/token | Expanded ms/token |
|---|---:|---:|
| Q/K norm | 0.027 | 0.028 |
| RoPE | 0.015 | 0.015 |
| KV 写入及量化 | 0.033 | 0.032 |
| attention 调度与计算 | 0.522 | 0.960 |
| sigmoid gate | 0.040 | 0.039 |

Expanded 的工作线程累计时间分布：QK 打分约 55.2%，softmax 约 6.4%，V 加权约 38.4%。这些是重叠的线程时间之和，不能与主线程时间相加，也不能相减推算线程等待。插桩会扰动执行，适合定位优先级；正式加速结论必须用无插桩版本交错对照。

Basic 固定前缀 177 token，Expanded 418 token，前缀作为 sink 永久保留，另外还有近期滑窗。因此 Expanded 不是仅扫描 256 个 token 的场景。上轮直接串行化 attention 没有收益，不建议重复采用。

## 当前实现与候选优化

当前已经把共享一个 KV head 的两个 Q head 配对，QK 复用 K 转换；V 加权按 32 维分块，同样复用一对 head 的 V。不能把“增加 GQA 配对”当成尚未实现的收益。

| 优先级 | 实验 | 收益依据与约束 |
|---|---|---|
| 1 | 固定 64 维 FP32-Q/INT8-K QK 内核，尝试 Q 驻留寄存器、交错处理 2/4 个 key、预取 | 减少重复 Q 加载并隐藏转换/乘加延迟；必须检查寄存器溢出和汇编。保留各输出的乘加、归约顺序，以现有实现逐位一致为验收目标。实际收益未知。 |
| 2 | V 加权的地址、scale 与分块复用 | 当前 64 维分两次扫描上下文，可尝试复用索引或预先计算概率乘 scale；需要计入额外临时存储成本。不能盲目展开为 64 维双 head：仅累加器就占满 32 个向量寄存器。 |
| 3 | 独立开关的 INT8-Q SDOT QK 路径 | 与官方二进制线索一致，可能省去 K 到 FP32 的转换并提高点积吞吐；但新增 Q 量化误差，须同时评估 logits、token 一致性和任务质量。不能承诺无损。 |

norm、RoPE、KV 写入、softmax 的优先级低于 QK/V。KV absmax 和量化有依赖：最终 scale 确定前不能直接完成所有元素的最终量化，所谓“单趟”需要暂存、寄存器驻留或改变算法，不是直接合并循环即可。

每个候选先做覆盖实际 head=64、GQA=2、固定前缀及环形回绕的局部对照，再做真实请求的无插桩交错测试。保留 sink、绝对位置和窗口语义。精确路径检查 logits/hidden 逐位一致；近似路径单独报告误差，不能用少量工具调用 token 一致代替精度证明。

## 官方公开资料与证据边界

### Needle 产品说明、参考实现与 SAN 论文

- [Needle 官方技术页](https://cactuscompute.com/needle)说明 256 token 滑窗、永久固定的 system/tools sinks、INT8 激活/KV、整数点积和线程池等设计，但没有完整公布生产 attention 内核及量化舍入规则。
- [官方 Needle 仓库](https://github.com/cactus-compute/needle)提供模型、训练/导出与接口相关代码；[固定版本 decode.py](https://github.com/cactus-compute/needle/blob/53df049c4a1a82fca1027b81f9ff21336dfb0861/needle/model/decode.py)是模型计算参考，不能当作发行二进制的 CPU 内核实现。
- [A Controlled Study of Attention-Only Transformers](https://arxiv.org/html/2607.18363v1)讨论 SAN 架构与消融，包括 GQA、QK norm 和 RoPE。它主要研究去除 FFN 的模型效果，不是 Needle2 生产内核优化论文，也不支持直接删除 QK norm。

### 官方通用 Cactus 引擎：可借鉴，但不等同于 Needle2

[公开 kernel 文档](https://docs.cactuscompute.com/latest/docs/cactus_kernels/)包含 hybrid INT8/FP16 attention 接口、GQA、窗口及 KV scale。其分组量化与当前项目按 token/head 的量化不完全相同。

[PR #625](https://github.com/cactus-compute/cactus/pull/625?plain=1)公开了具体方案：Q 按 head/group 量化后用 SDOT 打分、交错处理四个 KV 位置、延后 scale 修正和横向归约、轻量预取，以及 FP16 V 分块累加后定期扩展为 FP32。滑窗快速路径有适用条件，回绕等情况需要保留正确回退。对应[固定版本源码](https://github.com/cactus-compute/cactus/blob/45753df42e662f11ce43df63f3ab9392dcf1c9df/cactus-kernels/src/attention.cpp)。

该 PR 报告的约 1.9–2.4 倍是特定 head=128 等形状下的内核测试；Gemma 4 E2B 整体解码约从 29.2 提升到 32.5 TPS，约 10%。这些不是 Needle2 的测试，不能直接套用。报告的相对 FP32 NRMSE 约 0.0040–0.0043 也明确不是逐位一致。

[PR #485](https://github.com/cactus-compute/cactus/pull/485)优化 online softmax 的 scale 修正，只在需要时缩放历史累加值。当前实现使用完整 score 向量后做 softmax，不能直接移植这一局部改动；改为 online 算法也会改变浮点归约顺序。

### 官方 Needle2 发行二进制

检查固定发行版本的 [libneedle.a](https://huggingface.co/Cactus-Compute/needle2/blob/32e9e3a93b205f786929697446ae669cf0a84579/linux-arm64/libneedle.a)，SHA-256 为 `daea5a6610ec5872c9e5c4b4751f4edf00bf8774493e34b7726e5cf2f9be6285`。

[证据记录](../reports/official_attention_evidence.json)保存符号、调用点和 SDOT 指令：

- `forward_range` 的内部 lambda 有对 `dot_i8_pair64_dp` 的实际调用重定位。
- 该函数包含 8 条 SDOT 指令，随后做整数归约、浮点转换与 scale 乘法。
- 调试信息中存在 `q_i8`、`qscale0/1`、`scores0/1`。

因此不是仅凭函数名推测“官方可能使用整数点积”。但本轮未完整恢复其量化规则、分支条件和实际请求中调用频率，也未证明其与通用 Cactus PR 内核完全相同。公开材料已有技术说明；本次未找到完整公开的 Needle2 生产 attention 源码或专门内核论文。

## 复现

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/profile_attention.py build
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/profile_attention.py run
```

插桩脚本从当前工作区生成独立库，生产源码不插入计时代码。改变工作区后重跑属于新的实现，应记录报告哈希并重新建立对照。
