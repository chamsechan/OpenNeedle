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

数据：[转换后的官方引擎回归](../reports/conversion_parity.json)、[权重与数值验证](../reports/validation.json)。格式、真实位宽及原始资料见 [技术参考](research.md)。

## 数值对齐

| 比较对象 | 覆盖 | 结果 |
|---|---|---|
| 公开 JAX master FP32 → PyTorch master FP32 | 16 × 8192 logits | 最大绝对误差 0.00342，RMSE 0.000123，top-1 16/16 |
| PyTorch 发布权重 FP32 → 独立原生 FP32 | 192 × 8192 logits | 最大绝对误差 0.000275，RMSE 0.0000100，top-1 192/192 |
| 独立 FP32 → SDOT 近似模式 | 64 × 8192 logits | 相对 L2 误差 0.335%，RMSE 0.0804，top-1 62/64 |

前两项分别隔离架构迁移和原生实现的误差，没有把已经量化的权重与未量化 master 混为同一参考。JAX 报告的 cosine=1 是 FP32 统计舍入，不代表逐位相同。原生验证的 cosine 用 FP64 统计，为 0.999999999999868。

SDOT 会额外舍入旋转后的激活与码本；64-token 诊断中最大 logits 误差约 1.45，位置 3、63 的 top-1 改变。因此它是可选速度/精度取舍，不能套用 FP32 的验收结论。两种原生模式均保留 FP32 KV，没有声称复刻官方所有 A8/KV8 舍入。官方公开 C ABI 没有 logits 导出接口，不能直接测闭源 logits 的逐元素相等。

数据：[JAX 对照](../reports/model_jax_parity.json)、[FP32 对照](../reports/validation.json)、[SDOT 误差诊断](../reports/sdot_model_error.json)。

### 四行 SDOT 算术一致性

[test_sdot_row4.py](../tests/test_sdot_row4.py) 将四行内核输出与单行 SDOT 逐元素比较，覆盖 CQ2/CQ4、group64/128、9/129 行、129 列 padding，以及 1/2/4 线程，共 24 种参数组合。每种组合测试随机、全零和 one-hot 输入，输出完全一致。129 行覆盖并行阈值和尾行。

四行复用不改变每行点积、量化或逐 group 累加公式；它保持 SDOT 自身的数值语义，FP32→SDOT 的近似误差仍按上表单独报告。

## 工具调用质量

固定的 [15 个案例](../benchmarks/quality_cases.jsonl) 和 [工具定义](../benchmarks/quality_tools.json) 覆盖数字改写、布尔值、零值、optional/enum、多调用、重复调用、自我更正、否定和无关请求。预期答案在对比前固定，采用区分布尔/数字类型的 JSON 严格比较。

| 模式 | 正确调用 | 可解析输出 | 与官方原始 calls 一致 |
|---|---:|---:|---:|
| 官方 2.0.4 | 13/15 | 15/15 | — |
| 独立 FP32 | 13/15 | 15/15 | 13/15 |
| 独立 SDOT | 13/15 | 15/15 | 13/15 |

FP32 与 SDOT 的 15/15 完整生成 token 序列一致。双方未答对的两例是 `negation_no_action` 与 `off_topic`，且各自生成的错误 calls 并不相同。官方接口还提供置信度和 validation 字段；独立 CLI 的输出契约见 [grammar 文档](grammar.md)。因而“calls 正确数相同”不等于产品整体行为相同。

这些是手工小样本回归，不能当作 BFCL、泛化精度或长上下文准确率。质量脚本每例重新建立独立模型，包含完整前缀的耗时只作诊断，不用于宣称热请求性能。

数据：[FP32 质量报告](../reports/quality.json)、[SDOT 质量报告](../reports/quality_sdot.json)。两份报告均记录固定模型、源码和实际动态库哈希，`source_changed_during_run=false`。

## 性能

4 核 ARM Neoverse-N1、原生 4 线程、固定工具前缀复用：FP32 为 **120.77 token/s、244.0 ms/请求**，SDOT 为 **150.59 token/s、167.7 ms/请求**。3 组请求各预热后重复 5 次，均为中位数；各后端 15/15 次调用正确。

官方及 PyTorch 1/2/4 线程完整数据、等待策略、计时边界和首次前缀成本见 [CPU 性能与测量方法](backend-comparison.md)。原始数据为 [backend_comparison.json](../reports/backend_comparison.json)。

## 复现

准备依赖、模型和官方比较库后，从项目根目录串行执行：

```bash
OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q
OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/validate.py
OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/validate_sdot.py
OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/evaluate_quality.py \
  --threads 1 --matmul fp32 --output reports/quality.json
OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/evaluate_quality.py \
  --threads 1 --matmul sdot --output reports/quality_sdot.json
OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/benchmark_backends.py \
  --native-threads 4 --torch-threads 1,2,4 --repeat 5 \
  --output reports/backend_comparison.json
```

完整测试结果：**90 passed，4 subtests passed**，日志见 [pytest.txt](../reports/pytest.txt)。覆盖格式校验、CQ 数值 oracle、转换、模型 cache、probe、QAT 梯度、tokenizer、grammar、native 构建与 SDOT 路径。
