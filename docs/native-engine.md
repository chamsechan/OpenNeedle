# Native CPU 引擎

`needle2/csrc/cq.cpp`、`engine.cpp` 与 `sdot.cpp` 是基于公开 CACT 格式和公开模型计算图独立实现的 C++17 代码，不链接或执行官方 `libneedle.a` / `libneedle.so`。首次使用由 `CXX` 指定的编译器编译（默认 `c++`，需支持 OpenMP），共享库按源码哈希缓存到 `~/.cache/needle2`；可用 `NEEDLE2_NATIVE_CACHE` 指定缓存目录。安装 wheel 会包含源码。

可用 `python -m needle2 build-native` 提前构建并检查能力，或使用 CMake 生成共享库，
通过 `NEEDLE2_NATIVE_LIBRARY` 指定预编译产物。完整步骤见[编译指南](build.md)。

## 数值与布局

- CQ 权重保留 `[out, in]` 压缩布局，每组保存 LSB-first 2/3/4 bit 索引及 FP16 L2 norm。也支持 marker 5 的 ternary crumbs。
- GEMV 先计算每个输入组的归一化 Hadamard 变换，再直接对打包索引对应的 Lloyd-Max 码本值点积。权重无需完整展开；不需要训练时的原始 FP32 权重。
- ARM64 使用 NEON FMA 与字节查表；其他平台有标量回退。2 bit 的单字节映射为四个 FP32 值；4 bit 的单字节映射为两个值。
- 默认模式使用 FP32 计算，包含 GQA/RoPE、mHC、Hadamard MLP、Engram 哈希与因果卷积。mHC 先减最大值再取指数，在正数域进行 20 次先行后列归一化；四 lane 使用 NEON SIMD。当 routing logits 的极差大于 60 或最大值非有限时，退回 log-space 路径。它与公开 JAX 的 log-domain Sinkhorn 在实数代数上对应，但浮点舍入不同。
- ARM64 的 attention softmax、sigmoid 和 SiLU 使用向量指数近似；FP32 不代表与逐项 `expf` 或 JAX 逐位相等。验证需要同时检查 logits 误差与 token 一致率。
- 大型 attention、embedding 和 engram 矩阵保持 CQ；小型 mHC routing 矩阵加载后展开为 FP32。KV 缓存是 FP32 的滑动环，可保留固定前缀 sink，因此内存占用不等同于官方 INT8 KV 引擎。
- 默认数值目标是公开 JAX `decode.py` 的 FP32 路径。`activation_bits=8` 开启公开 A8 fake quant；这仍不等价于闭源库的全部 INT8 内部舍入。当前整模型引擎仅接收 KV8 模型，CQ KV2/3/4 请使用 PyTorch 参考模型。

## 使用

```python
from needle2.native import NativeEngine

engine = NativeEngine("artifacts/official/needle2.cact", threads=2)
engine.reset(prefix_len=16)
logits = engine.prefill(token_ids, last_only=True)
next_id = int(logits.argmax())
logits = engine.step(next_id)
```

每个实例服务一条不带 padding 的序列，调用会更新状态，不能并发调用同一个实例。`step(..., compute_logits=False)` 和 `prefill(..., last_only=True)` 跳过不需要的词表投影。`reset()` 清空逻辑状态；`reset(prefix_len=k)` 将前 k 个位置保留为 sink。直接处理长序列时，KV 内存由窗口与 prefix 长度决定。

压缩矩阵也可以单独使用：

```python
from needle2.archive import Archive
from needle2.native import NativeCQ

archive = Archive.load("artifacts/official/needle2.cact")
weight = NativeCQ.from_record(archive.tensors["layer00.q_proj"])
y = weight.linear(x, threads=2)       # x: [..., in], y: [..., out]
rows = weight.rows([0, 7])             # 只反量化所选行
```

## 固定 tools 前缀复用

固定前缀可显式预计算并保存一次，随后恢复状态处理不同请求：

```python
engine.reset(prefix_len=len(tool_prefix_ids))
engine.prefill(tool_prefix_ids, last_only=True)
engine.cache_prefix()                 # 只创建一次快照

for suffix_ids in (request_a_ids, request_b_ids):
    engine.reset_to_prefix()           # 恢复状态，不重新计算 tools 前缀
    logits = engine.prefill(suffix_ids, last_only=True)
    next_id = int(logits.argmax())
    logits = engine.step(next_id)
```

`cache_prefix()` 仅允许在 `position == prefix_len > 0` 时调用，保存当前实例的各层 prefix KV、prefix token 历史和 Engram raw-v 环。`reset_to_prefix()` 复制回这些状态；不改变模型、量化模式或计算方法。普通 `step()` 不创建快照。FP32 和 SDOT 均已验证：同一快照后的两个不同 suffix，与按相同模式从零重算逐元素完全一致。

任何 `reset()` 都会删除快照，包括再次设置相同的 `prefix_len`；此后调用 `reset_to_prefix()` 会报错，必须重新消费前缀并调用 `cache_prefix()`。当前 `generate()` 内部也调用 `reset()`；复用快照时应按示例组合 `prefill()` 和 `step()`。快照只属于创建它的实例，不能跨模型或实例使用。

快照需要额外内存：KV 部分约为 `2 × num_layers × prefix_len × num_kv_heads × head_dim × 4` 字节，另加 Engram 环和 token 历史。当前模型约为每个 prefix token **54 KiB**；190-token 前缀的 KV 快照约 **10.0 MiB**，另加约 40 KiB 的 Engram 环和少量历史数据。这是在现有运行时缓存之外的开销。恢复包含内存复制，其时间应计入每个请求的 wall time；前缀前向计算只在快照创建前执行一次。

## 可选 PyTorch 批量 prefill

```python
engine.reset(prefix_len=16)
logits = engine.prefill(token_ids, last_only=True, backend="torch")
engine.release_prefill_model()       # 释放可选 dense 模型；不影响 native decode
logits = engine.step(int(logits.argmax()))
```

此路径用 PyTorch FP32 批量计算初始 prompt，再导入绝对位置对应的 KV 状态并重建 Engram 的短历史。之后解码继续使用 native packed 权重。它只支持 `reset()` 后的初始 prompt，当前会保留一份 dense PyTorch 模型供多次请求复用；43.6M 参数对应约 175 MB FP32 权重，首次转换还有临时内存与加载时间。释放后再次使用会重新创建。这是以内存换 prefill 吞吐的显式选项，默认仍是完全 native 的逐 token prefill。

## 验证与测量

`tests/test_native.py` 将四种 packed 格式对照独立显式 Hadamard 矩阵，并验证真实结构的小模型在滑动窗口、固定前缀、非零卷积历史、A8/FP32，以及 PyTorch prefill 后 native 续算的一致性。这里的“一致”采用数值容差和 token 比较，不是承诺逐位相同。

```sh
python3 -m pytest tests/test_native.py -q
python3 scripts/benchmark_native.py --output reports/native_kernel_benchmark.json
```

性能脚本测真实模型矩阵、固定输入 token 解码以及初次/复用 prefill；报告包含模型 SHA256、环境、均值及分位数。运行时应避免其他 CPU 密集任务。更多线程未必更快；短矩阵的线程协调和云主机调度会带来长尾。与官方库比较时须统一 token、上下文、prefix、计算输出、线程和采样约束，不能将只返回工具 JSON 的总时间直接当作固定 token 解码吞吐。

## FP32 投影优化与实验开关

q/k/v/gate 四个投影共享一次输入 Hadamard 变换和一个 OpenMP 区域。`projection_lookup=-1`（默认）在 CQ2、group=128、合并输出至少 1024 行且线程数不超过 2 时，进一步使用每线程 32 KB 的 activation lookup table；其余形状保留 NEON FMA。表内仍是 FP32 码本与 FP32 旋转输入的乘积和，没有增加 INT8 舍入，但求和顺序变化会产生小的浮点差异。`projection_lookup=0` 可固定使用 NEON FMA 供基线比较。

`projection_lookup=1/2/3/4` 和 `NativeCQ.linear(..., lookup=1/2/3)` 是显式实验模式，不保证每种形状都更快。真实 190-token 上下文后的固定 32-token 对照中，私有查表相对融合 NEON 约提升 3%，logits 最大差约 `6.87e-5`，argmax 全部相同。原始数据和复现实验分别在 `reports/lookup_engine_benchmark.json`、`scripts/benchmark_lookup_engine.py`；单矩阵和融合投影数据在 `reports/cq_lookup_benchmark.json`。这些优化没有证明与官方闭源库等速。

## 显式 SDOT 近似后端

```python
from needle2.native import NativeEngine, sdot_available

if sdot_available():
    engine = NativeEngine("artifacts/official/needle2.cact", threads=2,
                          matmul="sdot", activation_bits=0)
```

`matmul="sdot"` 需要 Linux ARM64 的 DotProd 指令。运行时先检查 HWCAP，只在检查通过后调用单独标注 DotProd target 的函数；默认 FP32 不依赖该指令。该模式对 CQ2/CQ4 投影和 LM head 使用直接 packed 的 INT8 dot product，mHC 的 dense 投影与 KV 缓存继续使用 FP32。其它 CQ 位宽回退到 FP32。

此模式**增加量化误差**：分别将 Hadamard 后的输入按组缩放到 INT8，并将 Lloyd-Max centroid 缩放到 INT8；与公开 `activation_bits=8` 的量化位置不同，因此两者组合会明确报错。默认仍为 `matmul="fp32"`。可单独调用 `NativeCQ.linear_sdot(x, threads=2)` 检查某个矩阵的速度及误差。

独立整数 oracle 已覆盖 CQ2/CQ4、64/128 分组、非整组输入、零输入、one-hot 和多 batch；小型完整模型验证使用独立的近似误差预算，不能套用 FP32 的容差。原始单核实验的 q_proj 相对 L2 误差约 0.57%、LM head 约 0.25%，这不是整模型质量承诺。完整模型的质量和吞吐应以 `reports` 下对应 `matmul` 的最终报告为准。

`prefill(..., backend="torch")` 在 SDOT 模式下仍使用 FP32 PyTorch 完成初始 prompt，之后切换到 SDOT decode。此路径和完全 native SDOT prefill 的舍入过程不同。

## 四行 SDOT 与线程调度

`SdotCQ::row4` 同时计算四个 CQ2/CQ4 输出行，在每个 group 内复用激活向量加载，各行独立完成 INT32 点积和 FP32 缩放累加。权重保持行主序 packed 布局；不满四行的尾部使用单行内核。融合 QKVG 也按四行工作单元分配输出区间。

普通 `linear` 与单矩阵 `multiply` 在输出行数少于 128 时串行执行，避免小任务的 OpenMP 协调成本。线程数通过 `threads` 指定；库不设置全局 `OMP_WAIT_POLICY`。性能报告使用的等待策略和线程环境见 [测量条件](backend-comparison.md)。

四行实现与单行 SDOT 使用相同量化规则及逐 group 累加公式。[test_sdot_row4.py](../tests/test_sdot_row4.py) 对 CQ2/CQ4、group64/128、9/129 行、129 列 padding、1/2/4 线程，以及随机、零和 one-hot 输入做逐元素完全一致检查。这说明四行计算不额外改变 SDOT 数值，不代表 SDOT 与 FP32 等价。

指令定义见 [Arm NEON Intrinsics Reference](https://arm-software.github.io/acle/neon_intrinsics/advsimd.html)；公开 Cactus 的交错 CQ GEMV 与本实现的布局区别见 [技术参考](research.md#62-sdot-四行计算与公开内核参考)。

## 分阶段 profiling

在支持 DotProd 的 ARM64 上，可构建独立插桩库测量输入/Engram、QKVG、attention、MLP、mHC 和 LM head 的耗时：

```bash
OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1 python scripts/profile_native.py \
  --threads 1 --matmul sdot --tokens 64 --output reports/native_profile.json
```

插桩构建不覆盖生产库。报告对嵌套 Engram 投影做扣除，输出互斥阶段、剩余开销、head 投影行数、step 次数和最终保留的 KV 长度。插桩本身会增加计时成本，应用性能使用 [backend_comparison.json](../reports/backend_comparison.json)。
