# 基准与验证

本文汇总历史测量；各结果的版本和原始依据见对应链接。平台为 Linux ARM64、4 核 Neoverse-N1；模型 SHA256 为 `b43aabfcaf1a6db6acf488076eab71d823c08697c7af4521fc1d174b60ede5ba`。官方 2.0.4 库仅用于显式对照，独立运行时不调用它。

## 已发布的原生前端测量

4 个 native 线程，SDOT＋INT8 KV，常驻 `InferenceSession`；Basic 3 个、Expanded 16 个独立请求，每例热请求测量 5 次。默认运行模式仍为 FP32，SDOT 与 INT8 KV 是可选近似模式。

| 中位数 | Basic | Expanded |
|---|---:|---:|
| 完整热请求墙钟 ms | 62.86 | 76.70 |
| Query prefill ms | 23.61 | 29.92 |
| Decode token/s | 488.16 | 402.38 |
| Prepare ms | 0.21 | 0.26 |

完整请求包含准备、前缀恢复、query prefill、decode 和解析，不含模型初始化及首次 grammar/前缀构建。各列独立取中位数，不能相加。首次请求准备分别为 21.50 ms、97.80 ms，均为已初始化会话中的单次观测，不是模型冷启动。

原生前端迁移前后 19 个用例的完整 token 与预期调用一致。前端历史验证记录为 248 项测试、4 个 subtest 通过；它不代表当前 checkout 的新测试结果。

原始数据：[前端测量](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports/frontend_benchmark.json)、[前端验证](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports/frontend_validation.json)。

官方较早一轮 Expanded decode 自报 493.70 token/s，来自[历史官方对照](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports/performance_f4f9b38.json)。其内部 token 数、prompt、validation 与计时边界未完全公开，且与上述原生结果不是同轮交错测量，不能用二者比值宣称内核相对速度。

## 测量方法

- 比较版本时固定模型、工具 schema、请求、数值模式、线程、CPU affinity 和生成长度；预热后轮换执行顺序，保存逐请求输出和全部计时样本。
- 独立报告模型初始化、首次工具准备和热请求。`decode_tokens_per_second` 使用生成 token 数减一除以 decode 时间，首 token 来自 prefill；它不是纯矩阵内核速度。
- 当前会话字段的精确范围见[使用指南](usage.md#计时含义)。微基准、插桩 profile、完整工具请求分别解释，不能相互替代。
- 云主机可能有长尾；公开配置、模型哈希、源码版本和重复次数。FP32 与 SDOT/INT8 KV 的误差分别验证。

## 历史正确性验证

以下表格来自已保存的转换和数值实验，各自范围与来源见对应链接；完整 BFCL 尚未评估。

## 模型转换

| 检验 | 结果 |
|---|---|
| 发布 `.cact` → PyTorch → `.cact` | 原始 13,737,807 字节全部一致 |
| 官方 FP16 master → PyTorch → CQ2.2 | 仅 4 个字节不同：1 个 packed 字节、3 个 FP16 norm 元素 |
| 原版、无损往返版、master 重新量化版由同一官方库加载 | 3 组工具调用的语义输出均一致 |
| PyTorch 参数文件 | 两种来源各 174,571,900 字节，FP32 safetensors |

无损往返依赖保存原始码本、codes/norms 及张量指纹；修改的张量才重新量化。master 路径实际重新执行 Hadamard、归一化、码本最近邻和 norm 舍入，微小差异来自浮点归约和量化边界。这不能证明修改模型或重新训练后的质量不变。

数据：[转换后的官方引擎回归](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports/conversion_parity.json)、[权重与数值验证](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports/validation.json)。格式、真实位宽及原始资料见 [技术参考](reference.md)。

## 数值对齐

| 比较对象 | 覆盖 | 结果 |
|---|---|---|
| 公开 JAX master FP32 → PyTorch master FP32 | 16 × 8192 logits | 最大绝对误差 0.00342，RMSE 0.000123，top-1 16/16 |
| PyTorch 发布权重 FP32 → 独立原生 FP32 | 192 × 8192 logits | 最大绝对误差 0.000275，RMSE 0.0000100，top-1 192/192 |
| 独立 FP32 → SDOT 近似模式 | 64 × 8192 logits | 相对 L2 误差 0.335%，RMSE 0.0804，top-1 62/64 |

前两项分别隔离架构迁移和原生实现的误差，没有把已经量化的权重与未量化 master 混为同一参考。JAX 报告的 cosine=1 是 FP32 统计舍入，不代表逐位相同。原生验证的 cosine 用 FP64 统计，为 0.999999999999868。

SDOT 会额外舍入旋转后的激活与码本；64-token 诊断中最大 logits 误差约 1.45，位置 3、63 的 top-1 改变。因此它是可选速度/精度取舍，不能套用 FP32 的验收结论。上述历史诊断的两种原生模式均使用 FP32 KV；当前引擎还支持可选 INT8 KV，没有声称复刻官方所有 A8/KV8 舍入。官方公开 C ABI 没有 logits 导出接口，不能直接测闭源 logits 的逐元素相等。

数据：[JAX 对照](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports/model_jax_parity.json)、[FP32 对照](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports/validation.json)、[SDOT 误差诊断](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports/sdot_model_error.json)。

### 四行 SDOT 算术一致性

[test_sdot_row4.py](../tests/test_sdot_row4.py) 将四行内核输出与单行 SDOT 逐元素比较，覆盖 CQ2/CQ4、group64/128、9/129 行、129 列 padding，以及 1/2/4 线程，共 24 种参数组合。每种组合测试随机、全零和 one-hot 输入，输出完全一致。129 行覆盖并行阈值和尾行。

四行复用不改变每行点积、量化或逐 group 累加公式；它保持 SDOT 自身的数值语义，FP32→SDOT 的近似误差仍按上表单独报告。

## f4f9b38 优化前后数值验证

基线为 `e809ffc`。固定同一 CACT、prompt、prefix sink 和逐步历史 token，对扩展 16 例＋质量 15 例进行 teacher forcing；所有位置投影完整的 8192 词表，不使用单候选 `[0.0]` 占位值。首个位置来自 query prefill，后续来自逐 token decode。

| 每种配置相对自身优化前版本 | 解码决策位置 | 比较 logits 数 | 逐位不同元素 | 最大绝对误差 |
|---|---:|---:|---:|---:|
| FP32＋FP32 KV | 1026 | 8,404,992 | 0 | 0 |
| FP32＋INT8 KV | 1026 | 8,404,992 | 0 | 0 |
| SDOT＋FP32 KV | 1026 | 8,404,992 | 0 | 0 |
| SDOT＋INT8 KV | 1026 | 8,404,992 | 0 | 0 |

Top-1 与 grammar 筛选后的选择均无分歧。共比较 33,619,968 个 logits；结论限定于这些样例和 Linux ARM64 的 4 线程配置，不是不同量化模式彼此相同，也不是与官方闭源 logits 相同。官方接口没有 logits 导出能力。

对应回归测试覆盖不可变 DFA 的数据/形状/字段保护、跨词表验证，GQA 比例 3 的配对和单 head、32/38 维 SIMD/尾部、CQ2/CQ4、奇数 batch、滑窗回绕、prefix 恢复及 1/4 线程。[新内核测试](../tests/test_optimized_attention_prefill.py) 对照逐 token 路径与独立 PyTorch FP32 参考。

## 工具调用质量

固定 [15 个案例](../benchmarks/quality_cases.jsonl) 和 [工具定义](../benchmarks/quality_tools.json)，使用区分布尔/数字类型及调用顺序的严格 JSON 比较。

| 模式 | 正确调用 | 优化前后完整生成 token |
|---|---:|---|
| 官方 2.0.4（此前实测） | 13/15 | 不适用 |
| FP32＋FP32 KV | 13/15 | 15/15 一致 |
| FP32＋INT8 KV | 13/15 | 15/15 一致 |
| SDOT＋FP32 KV | 13/15 | 15/15 一致 |
| SDOT＋INT8 KV | 13/15 | 15/15 一致 |

未答对的两例仍为 `negation_no_action` 与 `off_topic`；官方和原生实现的具体错误 calls 并不相同。表中的 token 一致性是各模式优化前后比较，不是四种模式之间比较。该轮性能优化没有改变这组案例的得分。官方提供的 negation/grounding 标记未用于过滤本表输出。

这些是诊断小样本，不代表 BFCL 或广泛真实请求准确率。可复核的官方输出、该轮各模式 token、逐位置误差摘要见 [发布测量记录](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports/performance_f4f9b38.json)。


## 复现与开发检查

```bash
python -m pip install -e '.[test]'
python scripts/download_official.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q
python scripts/validate.py --output artifacts/reports/validation.json
python scripts/validate_sdot.py --output artifacts/reports/sdot_model_error.json
```

SDOT 需要支持 DotProd 的 ARM64。JAX 对照的上游源码准备见[使用指南](usage.md#可选-jax-源码准备与验证)。与官方库比较前需显式下载对应平台的比较库，可运行 `python scripts/benchmark_official.py --fetch-library` 下载固定版本并生成官方基线报告；详细参数见 `--help`。

```bash
OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python scripts/benchmark_backends.py \
  --tools benchmarks/expanded_tools.json --cases benchmarks/expanded_cases.jsonl \
  --native-threads 4 --torch-threads 1 --kv-cache int8 --repeat 9 \
  --max-new-tokens 192 --output artifacts/reports/backend_comparison_current.json
```

按目标机器调整 affinity 和线程参数。此命令产生新结果，不复刻历史数字。端到端比较、内核微基准、官方单独测速与插桩分析的入口见[维护脚本](../scripts/README.md)。新报告统一写入被 Git 忽略的 `artifacts/reports/`。

## 历史实验

当前采用的优化与未采用方案索引见[技术参考](reference.md#优化记录)。原始报告、一次性脚本及旧 Skills 可从[精简前版本](https://github.com/chamsechan/OpenNeedle/tree/ab18e89)查阅；更早原始测量保存在[报告归档](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports)。重跑历史实验应使用对应固定提交的独立 checkout，不将当前源码结果标为旧版本结果。
