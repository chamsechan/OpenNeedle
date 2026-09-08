# 实测结果与可复现边界

测试日期：2026-09-08。机器为 Linux ARM64、4 核 Neoverse-N1，无 GPU。模型为锁定版本的官方 `needle2.cact`，SHA256 为 `b43aabfcaf1a6db6acf488076eab71d823c08697c7af4521fc1d174b60ede5ba`；闭源比较基线来自官方 2.0.4 wheel。独立推理不调用该库。

目前已完成相同架构的双向转换、可训练 PyTorch 模型、直接读取 CQ 压缩权重的 C++ 引擎和显式 SDOT 加速。**尚未达到官方闭源引擎等速，也没有完成足以宣称完整任务精度相同的 BFCL 评测。** 以下分别说明实际证据。

## 模型转换

| 检验 | 结果 |
|---|---|
| 发布 `.cact` → PyTorch → `.cact` | 原始 13,737,807 字节全部一致 |
| 官方 FP16 master → PyTorch → CQ2.2 | 仅 4 个字节不同：1 个 packed 字节、3 个 FP16 norm 元素 |
| 原版、无损往返版、master 重新量化版由同一官方库加载 | 3 组工具调用的语义输出均一致 |
| PyTorch 参数文件 | 两种来源各 174,571,900 字节，FP32 safetensors |

无损往返依赖保存原始码本、codes/norms 及张量指纹；修改的张量才重新量化。master 路径实际重新执行 Hadamard、归一化、码本最近邻和 norm 舍入，微小差异来自浮点归约和量化边界。这不能证明修改模型或重新训练后的质量不变。

数据：[转换后的官方引擎回归](../reports/conversion_parity.json)、[权重与数值验证](../reports/validation.json)。格式、真实位宽及原始资料见 [研究记录](research.md)。

## 数值对齐

| 比较对象 | 覆盖 | 结果 |
|---|---|---|
| 公开 JAX master FP32 → PyTorch master FP32 | 16 × 8192 logits | 最大绝对误差 0.00342，RMSE 0.000123，top-1 16/16 |
| PyTorch 发布权重 FP32 → 独立原生 FP32 | 192 × 8192 logits | 最大绝对误差 0.000275，RMSE 0.0000100，top-1 192/192 |
| 独立 FP32 → SDOT 近似模式 | 64 × 8192 logits | 相对 L2 误差 0.335%，RMSE 0.0804，top-1 62/64 |

前两项分别隔离架构迁移和原生实现的误差，没有把已经量化的权重与未量化 master 混为同一参考。JAX 报告的 cosine=1 是 FP32 统计舍入，不代表逐位相同。原生验证的 cosine 用 FP64 统计，为 0.999999999999868。

SDOT 会额外舍入旋转后的激活与码本；64-token 诊断中最大 logits 误差约 1.45，位置 3、63 的 top-1 改变。因此它是可选速度/精度取舍，不能套用 FP32 的验收结论。两种原生模式均保留 FP32 KV，没有声称复刻官方所有 A8/KV8 舍入。官方公开 C ABI 没有 logits 导出接口，不能直接测闭源 logits 的逐元素相等。

数据：[JAX 对照](../reports/model_jax_parity.json)、[FP32 对照](../reports/validation.json)、[SDOT 误差诊断](../reports/sdot_model_error.json)。

## 工具调用质量

固定的 [15 个案例](../benchmarks/quality_cases.jsonl) 和 [工具定义](../benchmarks/quality_tools.json) 覆盖数字改写、布尔值、零值、optional/enum、多调用、重复调用、自我更正、否定和无关请求。预期答案在对比前固定，采用区分布尔/数字类型的 JSON 严格比较。

| 模式 | 正确调用 | 可解析输出 | 与官方原始 calls 一致 |
|---|---:|---:|---:|
| 官方 2.0.4 | 13/15 | 15/15 | — |
| 独立 FP32 | 13/15 | 15/15 | 13/15 |
| 独立 SDOT | 13/15 | 15/15 | 13/15 |

FP32 与 SDOT 的 15/15 完整生成 token 序列一致。双方未答对的两例是 `negation_no_action` 与 `off_topic`，且各自生成的错误 calls 并不相同。官方同时返回 negation/ungrounded 验证标记以及约 0.4511/0.033 的低置信度，可供产品层处理；独立 CLI 没有复刻这些最终校准和验证逻辑。因而“calls 正确数相同”不等于产品整体行为相同。

这些是手工小样本回归，不能当作 BFCL、泛化精度或长上下文准确率。质量脚本每例重新建立独立模型，包含完整前缀的耗时只作诊断，不用于宣称热请求性能。

数据：[FP32 质量报告](../reports/quality.json)、[SDOT 质量报告](../reports/quality_sdot.json)。FP32 报告保留 `source_changed_during_run=true`：运行时 `native.py` 加入了 SDOT 输出数量边界检查和 ctypes void 返回类型，不涉及 FP32 运算；该轮 C++ 源码和实际动态库哈希没有改变。SDOT 轮源码未改变。随后新增前缀快照接口，只显式用于重复请求的性能评测，质量脚本仍采用从头计算路径。

## 性能

最终应用对比使用相同模型、3 组 [固定请求](../benchmarks/cases.jsonl)、相同 CPU 允许集合 `0,1,2,3`，每组预热后各重复 5 次。独立模式显式选择线程数，官方保留其自动线程调度。两个引擎交错执行，工具前缀只预计算一次，实际复用；请求 wall time 包含前缀状态恢复，但不包括首次加载、编译或工具初始化。

| 模式 / 线程 | 独立 decode token/s | 同轮官方 token/s | 独立请求 ms | 同轮官方请求 ms | 原始报告 |
|---|---:|---:|---:|---:|---|
| FP32 / 2 | 112.2 | 517.7 | 230.5 | 45.5 | [comparison](../reports/comparison.json) |
| FP32 / 4 | 139.7 | 504.2 | 211.0 | 48.5 | [comparison_fp32_threads4](../reports/comparison_fp32_threads4.json) |
| SDOT / 2 | 162.0 | 487.9 | 158.0 | 52.9 | [comparison_sdot](../reports/comparison_sdot.json) |
| SDOT / 4 | 172.9 | 488.0 | 149.2 | 48.4 | [comparison_sdot_threads4](../reports/comparison_sdot_threads4.json) |

表内均为 15 次测量的中位数，双方在每轮 15/15 次请求中都生成正确调用。当前机器最佳 FP32 配置为 4 线程，吞吐约为同轮官方的 27.7%；最佳 SDOT 配置为 4 线程，约为同轮官方的 35.4%，请求延迟约为官方的 3.08 倍。速度目标仍存在明显差距。

SDOT / 4 的 177-token 工具前缀首次预计算约 581.6 ms，同轮官方初始化约 193.9 ms；独立前缀恢复中位数约 0.89 ms，已计入请求 wall time。这些数值不包括模型加载和首次 C++ 编译。

独立 decode TPS 包含 grammar 选择和 token forward，第一 token 来自 prefill，最后 token 不额外 forward。官方 TPS 为其响应内报告的值，其内部 prompt、token 数、验证和置信度计算并非全部与独立实现相同。**TPS 比值是应用层近似比较，不能称为同算子加速比。** 请求 wall time 是外部时钟测量，更接近固定工具已初始化后的实际延迟。

云主机存在调度波动，不同轮官方中位数也会变化。保留每次请求的 wall/process CPU 时间、环境、源码哈希及全部响应，不选取最短一次代替中位数。早期单独官方测试曾受到明显 CPU 竞争，`official_benchmark.json` 不是官方速度上限；历史未复用前缀的轮次保存在 `comparison*_before_prefix_cache.json`，不作为最终热请求延迟。

主要优化包括共享 q/k/v/gate 输入变换和线程区、ARM NEON、按形状启用 FP32 activation 查表、mHC 小矩阵加载时解压、正数域 SIMD Sinkhorn、共享 GQA 的 KV 读取、缓存 RoPE，以及显式 SDOT。剩余差距包括 FP32 KV/attention、线程协调和未实现的官方整数执行策略。前缀快照额外保存 KV 与 Engram 状态；混合 PyTorch prefill 还需要约 175 MB dense 权重，均不能宣称与官方的运行内存相同。

单矩阵实验仅用于解释优化方向：[SDOT 单核实验](../reports/sdot_experiment.json) 在真实 q_proj 和 LM head 上分别约快 2.30×、4.88×，但这并没有转化为同等的整网提速；[查表整网实验](../reports/lookup_engine_benchmark.json) 单独记录了约 3% 的增益和浮点误差。

## 复现

从项目根目录运行，先按 [README](../README.md) 安装和准备锁定模型。官方动态库只供显式基线脚本使用；新机器可先运行 `scripts/benchmark_official.py --fetch-library` 获取固定版本。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/validate.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/validate_sdot.py
python scripts/evaluate_quality.py --threads 1 --matmul fp32 --output reports/quality.json
python scripts/evaluate_quality.py --threads 1 --matmul sdot --output reports/quality_sdot.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/benchmark_compare.py \
  --threads 2 --repeat 5 --matmul fp32 --output reports/comparison.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/benchmark_compare.py \
  --threads 2 --repeat 5 --matmul sdot --output reports/comparison_sdot.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/benchmark_compare.py \
  --threads 4 --repeat 5 --matmul fp32 --output reports/comparison_fp32_threads4.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/benchmark_compare.py \
  --threads 4 --repeat 5 --matmul sdot --output reports/comparison_sdot_threads4.json
```

性能脚本应串行运行，避免与测试、编译或其他 CPU 密集任务重叠。单元测试覆盖格式校验、CQ 独立数值 oracle、转换、全模型 cache、probe、QAT 梯度、tokenizer、grammar 和 SDOT 近似路径；最终测试输出保存在 [pytest.txt](../reports/pytest.txt)。

最终完整测试为 **61 passed，4 subtests passed**。wheel 另在新的 venv 中安装，从项目目录之外导入并重新编译随包携带的 C++ 源码；安装验收记录见 [wheel_install.json](../reports/wheel_install.json)。
