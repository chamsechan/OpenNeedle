# 验证结果

测试平台：Linux ARM64，4 核 Neoverse-N1，CPU 推理。模型为官方 `needle2.cact`，SHA256：`b43aabfcaf1a6db6acf488076eab71d823c08697c7af4521fc1d174b60ede5ba`。官方 2.0.4 库仅用于显式比较，独立引擎不调用它。

验证覆盖权重转换、模型数值、缓存状态、四行 SDOT 与工具调用。工具质量采用固定小样本回归；完整 BFCL 未评估。

## 模型转换

| 检验 | 结果 |
|---|---|
| 发布 `.cact` → PyTorch → `.cact` | 原始 13,737,807 字节全部一致 |
| 官方 FP16 master → PyTorch → CQ2.2 | 仅 4 个字节不同：1 个 packed 字节、3 个 FP16 norm 元素 |
| 原版、无损往返版、master 重新量化版由同一官方库加载 | 3 组工具调用的语义输出均一致 |
| PyTorch 参数文件 | 两种来源各 174,571,900 字节，FP32 safetensors |

无损往返依赖保存原始码本、codes/norms 及张量指纹；修改的张量才重新量化。master 路径实际重新执行 Hadamard、归一化、码本最近邻和 norm 舍入，微小差异来自浮点归约和量化边界。这不能证明修改模型或重新训练后的质量不变。

数据：[转换后的官方引擎回归](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports/conversion_parity.json)、[权重与数值验证](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports/validation.json)。格式、真实位宽及原始资料见 [技术参考](research.md)。

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

新增回归覆盖不可变 DFA 的数据/形状/字段保护、跨词表验证，GQA 比例 3 的配对和单 head、32/38 维 SIMD/尾部、CQ2/CQ4、奇数 batch、滑窗回绕、prefix 恢复及 1/4 线程。[新内核测试](../tests/test_optimized_attention_prefill.py) 对照逐 token 路径与独立 PyTorch FP32 参考。

## 工具调用质量

固定 [15 个案例](../benchmarks/quality_cases.jsonl) 和 [工具定义](../benchmarks/quality_tools.json)，使用区分布尔/数字类型及调用顺序的严格 JSON 比较。

| 模式 | 正确调用 | 优化前后完整生成 token |
|---|---:|---|
| 官方 2.0.4（此前实测） | 13/15 | 不适用 |
| FP32＋FP32 KV | 13/15 | 15/15 一致 |
| FP32＋INT8 KV | 13/15 | 15/15 一致 |
| SDOT＋FP32 KV | 13/15 | 15/15 一致 |
| SDOT＋INT8 KV | 13/15 | 15/15 一致 |

未答对的两例仍为 `negation_no_action` 与 `off_topic`；官方和原生实现的具体错误 calls 并不相同。表中的 token 一致性是各模式优化前后比较，不是四种模式之间比较。本轮性能优化没有修复已有的质量错误，也没有新增错误。官方提供的 negation/grounding 标记未用于过滤本表输出。

这些是诊断小样本，不代表 BFCL 或广泛真实请求准确率。可复核的官方输出、当前各模式 token、逐位置误差摘要见 [发布测量记录](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports/performance_f4f9b38.json)。

## 性能

`e809ffc` 与 `f4f9b38` 同机交错测量，4 线程 SDOT＋INT8 KV：扩展请求中位耗时 **132.8 → 87.4 ms**，query prefill **41.2 → 32.3 ms**，decode 吞吐 **201.3 → 327.4 token/s**。每例预热 1 次、测量 9 次。基础 3 例和扩展 16 例的全部生成 token 均保持相同，调用全部正确。

官方对照、计时边界和复现见 [性能说明](backend-comparison.md)。

## 复现

准备依赖、模型和官方比较库后，从项目根目录串行执行：

```bash
OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q
OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/validate.py
OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/validate_sdot.py
OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/evaluate_quality.py \
  --threads 1 --matmul fp32 --output artifacts/reports/quality.json
OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/evaluate_quality.py \
  --threads 1 --matmul sdot --output artifacts/reports/quality_sdot.json
OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/benchmark_backends.py \
  --native-threads 4 --torch-threads 1,2,4 --repeat 5 \
  --output artifacts/reports/backend_comparison_current.json
```

当前优化测试结果：**194 passed，4 subtests passed**，日志见 [pytest.txt](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports/published_f4f9b38/pytest.txt)。覆盖格式校验、CQ 数值 oracle、转换、模型 cache、probe、QAT 梯度、tokenizer、grammar、native 构建与 SDOT 路径。
