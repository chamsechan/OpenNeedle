# mHC 投影优化与真实请求 profile

本轮基线为 `32005c7818aa44bdd9a3f6750e9f92214c1b5931`。改动是解码阶段 mHC 三个 FP32 投影成对读取同一输入，并把三组工作合为一次线程池任务。总矩阵元素少于 32768 时保持串行；CQ 描述符回退到原有三个 `linear` 调用。prefill、attention、Sinkhorn、量化格式与采样规则不变。

## 先纠正执行路径判断

`NativeEngine` 加载时会将 `mhc_*` 张量反量化为 FP32，即使启用 SDOT，也不会让这些矩阵走 SDOT `prepare`。先前根据 C++ 三次 `linear` 调用推断存在三次量化预处理，遗漏了 Python 装载层，推断不成立。实际 profile 中每个 decode token 的 SDOT prepare 次数为 59，并非假设 mHC 参与后的 113。试验过的 SDOT 共享预处理分支不会生效，已撤除。

保留的优化复用现有 `dot_pair`：两行权重共享输入加载，各自仍使用与原 `dot_f32` 对应的累加器和归约顺序。发布模型每层有 4＋4＋16 个输出、输入宽度 2048，原本是串行的 24 个逐行点积；现在为 12 对行，由常驻线程池一次分发。没有把 mHC 改成 INT8。

## 真实请求的耗时分布

新 profiler 使用实际 basic/expanded 工具 schema 和生成路径，4 线程、SDOT、INT8 KV。先初始化工具前缀及 DFA，再对每个请求预热一次、测量三次，只在 decode 范围开启计数。每个生成序列与已记录的基线完全一致，工具调用正确。阶段耗时互斥，按实际 token 总数加权求平均。

Expanded 的基线分布约为：

| 阶段 | 占 decode 时间 |
|---|---:|
| Q/K norm、RoPE、KV 存储、attention、gate | 37.8% |
| Q/K/V/G 投影 | 17.1% |
| mHC 前投影及相关输入归一化 | 15.1% |
| attention 输出投影及 Hadamard MLP | 14.3% |
| Sinkhorn、残差混合及 hidden 汇总 | 11.5% |
| 候选投影 | 0.6% |

Basic 固定前缀为 177 token，Expanded 为 418 token。固定前缀作为 sink 保留，不能把这些请求等同于只关注 256 个 token 的无前缀微基准。这个分布也解释了为何上轮 LM head 局部优化无法带来显著请求级收益。

[基线 profile](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports/real_decode_profile_baseline.json)和[最终 profile](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports/real_decode_profile_mhc_final.json)保留逐请求阶段计时及计数。mHC 前半段在 Expanded 中约从 0.495 降至 0.255 ms/token。profile 版本带计时开销，且两个版本分开运行，阶段变化只用于定位；性能结论使用无插桩库的交错请求对照。

## 无插桩性能对照

同机 CPU 0–3、4 个 native 线程，独立常驻进程，轮换请求执行顺序。每例预热一次；首轮测量 5 次，最终确认测量 9 次。使用同一模型、工具、query、KV 配置和长度上限，检查所有 token 及预期工具调用。

[首轮实验](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports/mhc_decode_experiment.json)比较逐行基线、仅成对投影、成对投影加合并并行。Expanded decode TPS 中位数分别为 335.8、348.2、372.4。因此选择合并并行，并为较小矩阵保留串行路径。

最终确认数据见 [原始报告](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports/mhc_decode_final.json)。

| 用例集 | 基线请求 ms | 优化后请求 ms | 基线 decode TPS | 优化后 decode TPS | TPS 提升 |
|---|---:|---:|---:|---:|---:|
| basic | 61.21 | 55.54 | 422.07 | 470.34 | 11.4% |
| expanded | 86.83 | 82.24 | 334.24 | 365.97 | 9.5% |

云主机有长尾，报告各列独立取中位数，不能直接相加。收益不代表所有硬件和输入。

## 精度与边界验证

- [数值对照](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports/mhc_numerical_validation.json)：每个变体 216 个数组、1,828,304 个 logits/hidden 元素，与基线逐位一致。
- 覆盖 FP32/SDOT、FP32/INT8 KV、1/4 线程，带固定前缀的 280 步生成、滑窗回绕及前缀恢复。
- 小模型包含多种 mHC 归档存储格式；装载后仍是 FP32，不将它们误记为运行时 SDOT mHC 覆盖。
- 新回归测试覆盖 1/2/3/4 lane、奇数行尾部、滑窗和两种 KV 格式，对照未修改的单 token prefill 路径。
- 相关 pytest 共 55 项、4 个 subtest 通过。逐位一致性是本轮样本验证结果，不消除已有 SDOT/INT8 KV 相对参考模型的近似误差。

## 官方比较口径

官方公开响应包含 `decode_tps` 和 `prefill_tps`，但文档没有提供对应内部 token 计数、阶段耗时和所有计时边界。因此不能用本轮 native TPS 直接计算“已达到官方百分之多少”。官方还包含工具检索、置信度和验证处理，完整请求与单个解码阶段必须分别比较。本轮未重新运行官方 benchmark，也未宣称消除了官方性能差距。

后续已完成 [attention 细分调查](attention-research.md)和[固定 64 维 QK 优化](attention-optimization.md)。本报告保留 mHC 实验当时的数据；其中 attention 大类包含 norm/RoPE/gate，不能把整个 37.8% 都算作 attention 核心。

## 历史复现

这组一次性实验脚本已从当前版本移除。原脚本及完整复现步骤见[清理前的历史版本](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/docs/mhc-optimization.md)；需要复现时，请在该提交的独立 checkout 中按历史说明运行。

当前实现的数值回归仍由 `tests/test_mhc_projection.py` 和 `tests/test_native.py` 等测试覆盖。
