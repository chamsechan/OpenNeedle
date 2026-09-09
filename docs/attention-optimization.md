# Attention 固定 64 维 QK 优化

本轮在已有 mHC 优化上，保留一项小范围生产改动：ARM64、INT8 KV、64 维且双 Q head 共享 KV 时，使用编译期常量 64 调用现有 `dot_pair_i8_f32`。编译器可以展开原循环并调度加载、INT8 转 FP32 和 FMA，减少动态维度循环与尾部判断的开销。该路径同时用于 decode 与 prefill 中的 attention。

仍使用 FP32 Q，没有增加 Q 量化。每个 head 的两个累加器、乘加与归约顺序、K scale 和 attention scale 的两次乘法保持一致。其它维度、未配对 head、FP32 KV 及非 ARM64 继续使用通用路径。固定前缀 sink、滑窗映射及线程策略不变。

## 实验选择

先实现四个 key 共享 Q 加载，再比较两个 key 和固定 64 维单 key。四 key 的局部测试更慢，真实请求没有稳定收益；双 key 也未胜过固定维度单 key。因此生产代码只保留原点积函数的固定维度调用，无新增批处理模板。汇编检查确认固定路径展开了维度循环；不是用额外整数量化换取速度。

初轮报告：[四 key](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports/attention_qk_experiment.json)、[单/双 key](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports/attention_qk_sizes.json)。这些是探索结果，不作为最终实现的性能数字。

## 无插桩请求性能

[确认报告](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports/attention_fixed64_confirmation.json)使用同机 CPU 0–3、4 个 native 线程、SDOT 投影、INT8 KV；每例预热一次，测量九次，常驻进程交错执行并轮换顺序。基线 `attention_before` 是本轮开始时已经包含 mHC 优化的版本；生产实现是 `attention_fixed64`。`attention_qk1` 为保留在报告中的中间实现。

| 场景 | 原 decode TPS | 优化后 decode TPS | TPS 提升 | 原请求 ms | 优化后请求 ms |
|---|---:|---:|---:|---:|---:|
| Basic | 455.36 | 473.15 | 3.9% | 56.78 | 53.93 |
| Expanded | 366.73 | 393.63 | 7.3% | 82.04 | 76.39 |

请求耗时中位数分别减少 5.0% 和 6.9%。各列独立取中位数，不应将它们相加或相除还原某次请求。19 个用例各自的 decode TPS 中位数均有所提高；云主机数据不保证适用于其它硬件。所有请求均检查生成 token 一致及预期工具调用。本轮没有重测官方引擎，不能据此宣称达到官方的某个百分比。

## 数值和回归验证

完整模型的 logits/hidden 使用修改前后独立库逐字节比较，覆盖 FP32/SDOT 投影、FP32/INT8 KV、1/4 线程、280 步生成、固定前缀、滑窗回绕和前缀恢复。结果见 [数值报告](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports/attention_numerical_validation.json)。这是相对本轮基线的一致性验证，不消除已有 INT8 KV 和 SDOT 投影的近似误差。

216 组数组、1,828,304 个元素逐字节一致；31 项回归测试和 4 个 subtest 通过。

扩展 `test_optimized_attention_prefill.py` 到 64 维，结合既有 32/38 维覆盖，验证 GQA 比例 3 的配对及未配对 head、不同 KV 格式、prefill/逐步解码、短上下文、回绕和前缀恢复。FP32 路径还对照独立 PyTorch 模型。完整验证状态及源码/库哈希见 [实验清单](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports/attention_optimization_manifest.json)。

## 历史复现

这组一次性实验脚本已从当前版本移除。原脚本及完整复现步骤见[清理前的历史版本](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/docs/attention-optimization.md)；需要复现时，请在该提交的独立 checkout 中按历史说明运行。

当前实现的回归检查见 `tests/test_optimized_attention_prefill.py` 和 `tests/test_mhc_projection.py`。

[最终微基准结果](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports/attention_qk_micro.csv)中，孤立通用点积与固定 64 维点积耗时基本相同。请求收益出现在集成引擎中，包含固定维度及配对条件外提后的编译效果；本轮没有单独证明其中某项是唯一原因，也不宣称 QK 内核本身提升了 7.3%。
