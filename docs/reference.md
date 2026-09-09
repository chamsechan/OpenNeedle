# Needle 2 / CQ2.2 技术参考

本文说明发布模型的架构、CQ 文件格式、独立 CPU 引擎的执行方式及其公开依据。性能、质量与复现见[基准与验证](benchmark.md)。

## 1. 研究对象与版本

研究对象为 Needle 2 发布模型的 CQ2/CQ4 混合量化格式、PyTorch 转换与压缩权重推理。部署文件包含 43,634,423 个数值参数，逐张量统计见下文。

实现依据的固定版本资料：

| 来源 | 版本 / 地址 | 用途 |
|---|---|---|
| Needle Python/JAX 官方源码 | [`53df049c4a1a82fca1027b81f9ff21336dfb0861`](https://github.com/cactus-compute/needle/tree/53df049c4a1a82fca1027b81f9ff21336dfb0861) | 发布架构、导出格式、量化、tokenizer、参考解码 |
| 官方模型仓库 | [`32e9e3a93b205f786929697446ae669cf0a84579`](https://huggingface.co/Cactus-Compute/needle2/tree/32e9e3a93b205f786929697446ae669cf0a84579) | `.cact`、原始 checkpoint、平台引擎、配置 |
| SAN 论文 | [A Controlled Study of Attention-Only Transformers, arXiv:2607.18363v1](https://arxiv.org/html/2607.18363v1) | 受控 attention-only 架构消融，2026-07-20 |
| Cactus 通用引擎 | [`09cb35ab29aaad66189615192d27d84bebbc0522`](https://github.com/cactus-compute/cactus/tree/09cb35ab29aaad66189615192d27d84bebbc0522) | 公开 CQ 实现与优化 kernel 参考 |
| Needle 产品技术说明 | [cactuscompute.com/needle](https://cactuscompute.com/needle) | 部署意图及官方性能声明；网页可变 |

本地官方文件 `artifacts/official/needle2.cact` 的 SHA-256：

```text
b43aabfcaf1a6db6acf488076eab71d823c08697c7af4521fc1d174b60ede5ba
```

配置文件的 `engine_version` 是 `2.0.4`。这是配置字段，不能替代对实际下载平台库进行版本/文件哈希登记。

## 2. 论文与发布模型不能混为一谈

SAN 论文的核心实验删除了 FFN：pre-norm → GQA → QK norm → RoPE → attention → 标量门控残差；没有学习的逐位置特征映射。论文主要比较参数、训练 FLOPs 和深度分别匹配的模型，主 SAN 为 20 层、宽 512、约 24M 参数。参数匹配时增加 attention 深度能大幅缩小损失差距；QK norm 是深层 SAN 稳定性的重要因素。这提供设计动机，**不构成发布版 Needle 2 的完整实现规格**。[论文正文 Experimental Setup、Component Ablations](https://arxiv.org/html/2607.18363v1)

发布版源码另外具有 Hadamard MLP、engram、四 lane mHC、attention output gate、检索与置信度 probe heads。它与论文里的纯 attention 实验模型具有实质差异。本文实现规格以锁定版本的 [`architecture.py`](https://github.com/cactus-compute/needle/blob/53df049c4a1a82fca1027b81f9ff21336dfb0861/needle/model/architecture.py) 和 [`export.py`](https://github.com/cactus-compute/needle/blob/53df049c4a1a82fca1027b81f9ff21336dfb0861/needle/model/export.py) 为准。

官方把量化质量归因于训练阶段适应部署数值，而非任意模型事后压到 2 bit。该声明不能推广为“把任何 PyTorch 模型压成 2.2 bit 都无损”。其生产引擎还包含 grammar 候选裁剪、工具检索、滑动窗口与 int8 kernel；论文复现、模型前向复现和完整产品复现是不同验收层次。[官方技术说明](https://cactuscompute.com/needle)

## 3. 发布版精确架构

以下几何来自官方 [`config.json`](https://huggingface.co/Cactus-Compute/needle2/blob/32e9e3a93b205f786929697446ae669cf0a84579/config.json)，并由 `.cact` header 独立核对。

| 项目 | 发布值 |
|---|---:|
| 词表 / hidden size / 层数 | 8192 / 512 / 27 |
| Q heads / KV heads / head dim | 8 / 4 / 64 |
| RoPE theta / 最大位置 | 100000 / 2048 |
| mHC lanes | 4 |
| engram 层索引，零起始 | 2、15 |
| engram orders / heads per order | 2、3 / 2 |
| 每 site 表数 / slots / sub-dim | 4 / 8192 / 128 |
| engram convolution | 4 taps，dilation 3 |
| Hadamard MLP 长度 | 512 |
| KV window / KV bits / activation bits | 256 / 8 / 8 |
| 默认 CQ scheme | `embedding=4,mhc=4,default=2` |

以下公式直接按源码重述，采用行向量约定。`ZCN(x, γ) = (1+γ) x / sqrt(mean(x²)+1e-6)`，不是把 `γ` 直接作为 RMSNorm gain。所有 attention linear 无 bias。Flax kernel 存为 `[in,out]`；`.cact` 和 PyTorch Linear 存为 `[out,in]`，转换时不能遗漏转置。

### 3.1 embedding、attention 和 Hadamard MLP

输入为 `sqrt(512) * embedding[token]`；四 lane 初始都复制该输入，最后对 lane 求平均，再做 final ZCN。输出投影与输入 embedding 共享权重。

attention 先对 block 输入做 ZCN，随后 Q/K/V 投影；Q/K 分别在 64 维 head 内做 ZCN，再做 RoPE。RoPE 将 head 前半与后半配对，**不是相邻偶奇维配对**。GQA 的两个 Q head 对应同一个 KV head。attention 结果还乘输入相关门 `sigmoid(x @ W_gate)`，之后做 output projection。projection 结果经过 post-attention ZCN，再乘 `sigmoid(attn_gate)` 加回残差。

Hadamard 子层为：

```text
z = ZCN(x, pre_hada)
z = (d1 * pad(z)) H
z = SiLU(d2 * z) H
y = x + crop(d3 * z)
```

`H` 是正交 Sylvester/Walsh-Hadamard 矩阵，长度为不小于 hidden size 的最小 2 次幂。`d1,d2,d3` 是学习对角向量；`d3` 初始为 0.02。这个锁定源码版本的 Hadamard 残差没有单独 post norm 或标量 sigmoid gate，不能根据概览图自行添加。快速实现可以用蝶形 FWHT 替代显式矩阵乘，但改变求和次序会产生浮点舍入差异。

### 3.2 engram

每个 `(order, head)` 用 uint32 溢出算术计算散列。初始化 `0x9E3779B9 * (table_index+1)`；按当前 token、前 1 token、前 2 token 的顺序，对该 order 所需 token 执行 `(acc XOR token) * 0x01000193`；最后 `acc XOR (acc >> 15)`，取模 8192。序列开头右移补零，但无足够 n-gram 历史的 gather 结果还须清零。不能用 Python 无界整数而不做 uint32 截断。

四个表各取 128 维拼为 512 维，得到 `e`；`k=eW_key`，`v=eW_value`。value 再应用深度方向的逐通道因果卷积，取位置 `t,t-3,t-6,t-9`。在指定 block 前，用 `sigmoid(dot(rms_unit(u),rms_unit(k))/sqrt(512))` 门控 value 并加到 `u`。后续 mHC 的 branch delta 必须减去**engram 注入前的 u**。mask 除 attention 外还影响 n-gram 和卷积是否跨文档边界。

### 3.3 mHC

令 `X` 为 `[4,512]` 的 lane 状态，`nx=rms_unit(flatten(X))`。第 `l` 层选中 lane `l mod 4`。

```text
h_pre  = sigmoid(a_pre  * (nx Phi_pre)  + b_pre  + pre_off)
u      = sum_lanes(h_pre * X)
delta  = Block(u) - u
h_post = 2 * sigmoid(a_post * (nx Phi_post) + b_post + post_off)
A      = a_res * reshape(nx Phi_res, [4,4]) + b_res
P      = sinkhorn_logspace(A, 20)
X_new  = P @ X + h_post[:,None] * delta[None,:]
```

`pre_off` 为选中 lane +4，其余 -4；`post_off` 为选中 lane 0，其余 -4。**锁定的官方 JAX 源码**中，Sinkhorn 每次先行归一化再列归一化，通过减 `logsumexp` 执行 20 次，最后 exp。每层不是简单 softmax routing，亦不能少算一次迭代而称精确等价。本项目 native 优化版保留 20 次行列归一化，通常在正数域用 SIMD 执行，遇到高动态范围则退回 log-space；浮点求值次序不同，具体见 [原生引擎说明](architecture.md)。

### 3.4 训练附加分支与 probe heads

源码包含 MTP combine/block/norm 训练分支，但标准 `.cact` 不导出这些参数。对比参数数量时须说明是否包含 MTP。检索 head 使用 4 probes，置信度 head 使用 8 probes；它们在 embedding 与每层平均 hidden cells 的集合上做 attention pooling，flatten 后投影。最终 layer hidden 或最后 token 不能替代该 pooling 而宣称检索、confidence 等价。标准 LM forward 不依赖这两个头。

## 4. CQ 量化的精确数学与格式

依据官方 [`quantize.py`](https://github.com/cactus-compute/needle/blob/53df049c4a1a82fca1027b81f9ff21336dfb0861/needle/model/quantize.py) 与 [`export.py`](https://github.com/cactus-compute/needle/blob/53df049c4a1a82fca1027b81f9ff21336dfb0861/needle/model/export.py)。CQ2 是 **2-bit 码本索引**，不是均匀 signed int2。

### 4.1 分组、旋转、码本、舍入

对 `[out,in]` 的每行，沿输入轴每 128 元素分组；不足补零。令 `w` 是一组、`H=Hadamard(128)/sqrt(128)`：

```text
r       = w H
n       = sqrt(sum(r²))                 # FP32
u       = r / max(n, 1e-12)            # 先用未舍入的 norm 归一化
q_i     = argmin_j |u_i - codebook[j]|  # 距离相等选较小下标
n_store = float16(n)
w_hat   = (codebook[q] * float32(n_store)) H
```

码本生成：`RandomState(0).randn(400000)` 的样本上运行 200 次 Lloyd-Max，初始中心来自分位点，最终码本除 `sqrt(group_size)` 转成 FP32。经验样本码本略不对称，不能替换成理论对称中心。当前官方 CQ2 header 值为：

```text
[-0.1331721395254135, -0.03990209102630615,
  0.04003528133034706, 0.13346044719219208]
```

实现解析时应以文件内嵌 codebook 为准。官方 `read_export()` 虽读出 header codebook，内部 `_cq_unpack()` 又通过算法重建码本；这在已发布文件上吻合，但对自定义码本不构成通用保证。

### 4.2 文件布局

全文件 little-endian。固定 header 为 `struct '<29If'`，120 bytes；tag `0x05E12A83`。随后是 CQ2/CQ3/CQ4 共享 FP32 码本，共 4+8+16=28 元素。张量目录每条 `'<BBHIIIIQQII'`，44 bytes；包含 dtype、ndim、4 维 shape、offset、nbytes、group、bits。tensor blob 按 64-byte 对齐，**目录不存名字**，必须按架构规定的顺序命名。

dtype 1=FP16，2=FP32，3=CQ，4=RAW。CQ blob 先放全部行的 packed indices，再放全部 FP16 norms。每 8 个 code 按 `code[i] << (i*bits)` 拼 word，按 little-endian 发出 `bits` 个字节；可视作每行连续 LSB-first bitstream。group scale 是 L2 norm，不是 absmax scale。

官方还定义 `record bits=5` 表示三值量化，实际存 signed 2-bit crumbs，不是五位权重，更不是理想熵编码的 1.58 bit。当前发布模型只有 CQ2/CQ4，不需要把三值变体误当默认。

norm、Hadamard diagonals、gate 和 probe heads 保留 FP16；`mhc_phi*` 使用 CQ4。embedding CQ4 与 attention/engram CQ2 共同组成默认 scheme。

### 4.3 “2.2 bit”到底统计了什么

**本地测量**，按目录和 shape 计算：

| 类别 | 张量数 | 参数元素数 | blob bytes |
|---|---:|---:|---:|
| CQ2 | 141 | 37,748,736 | 10,027,008 |
| CQ4 | 4 | 5,521,408 | 2,846,976 |
| FP16 | 259 | 364,279 | 728,558 |
| RAW tokenizer | 1 | 不计为参数 | 115,215 |

合计导出数值参数 43,634,423。量化参数码字加权位宽为 **2.255206** bit；FP16 norms 676,096 bytes。包含 norms 和 FP16 参数的数值 payload 为 **2.493910 bpw**；含 tokenizer、目录、对齐的全文件 13,737,807 bytes，相当于 **2.518710 bpw**。因此配置里的 `effective_bits: 2.2` 是近似口径，不是所有文件字节除参数数的结果。

一般公式为 `bits + 16/group`，所以纯 CQ2/group128 的无 padding 矩阵已经是 2.125 bpw；混合位宽、未量化参数和目录另算。不能只用 `parameter_count * 2.2 / 8` 作为 `.cact` 文件大小验收。

### 4.4 为什么往返必须保留 packed 元数据

**工程推导**：`w_hat` 的旋转方向经过离散化后，其长度通常不再为 1。因此再次对 `w_hat` 执行上述“重新计算 norm、重新选 code”流程，可能改变 norm 甚至 code。量化映射不保证幂等。

`.cact → PyTorch → .cact` 若要求未编辑时逐字节等价，应保留原始 packed codes、norms、header、RAW tokenizer；检测某张量是否改变，仅对改过的张量重新量化。反量化所得 FP32 不能恢复原始训练前 FP16 权重；从官方原始 checkpoint 转换与从部署 blob 反量化是两种不同来源。

## 5. PyTorch 数值复现应采用哪些基线

**源码事实**：官方公开路径存在不同的数值行为。

| 路径 | 权重与计算行为 | 适合验证 |
|---|---|---|
| JAX `architecture.py`, `quant=False` | 原始或已反量化权重，计算 dtype 来自 config | 原始 checkpoint 架构移植 |
| JAX `architecture.py`, `quant=True` | 激活若干位置做 A8 fake quant；权重量化须调用者准备 | 训练/量化感知前向 |
| JAX `decode.py` cached path | 显式 FP32 cache；无 A8 fake quant | FP32 cached forward、mask、位置、engram 历史 |
| 发布的 native engine | 平台库实际行为 | 生产 token、JSON、延迟、RSS |

`fake_quant_act` 沿最后一维用 absmax/127 计算 scale，`round` 后 clamp 到 [-128,127]；发生在 QKV 输入、attention gate 后的 output projection 输入、engram gather 后以及最终词表投影前。这不等价于所有中间值始终 int8。

另一个细节是 `configure_deploy(kv_bits>=8)` 将内部 `KV_BITS` 置为 0，不模拟 KV8 舍入；只有低于 8 bit 时才通过 CQ fake quant 模拟 KV。故公开 JAX 参考无法单独证明生产 int8 KV 和全部舍入行为完全相同。相关代码见 [`decode.py`](https://github.com/cactus-compute/needle/blob/53df049c4a1a82fca1027b81f9ff21336dfb0861/needle/model/decode.py) 和 [`quantize.py`](https://github.com/cactus-compute/needle/blob/53df049c4a1a82fca1027b81f9ff21336dfb0861/needle/model/quantize.py)。

验证覆盖文件往返、独立 Hadamard/CQ 数值 oracle、逐层 hidden/logits、滑动 KV 与固定 sinks、Engram 历史，以及工具调用 JSON。四行 SDOT 另外与单行内核做逐元素完全一致检查。具体数值、数据范围和复现命令见 [验证结果](benchmark.md)。

## 6. CPU 推理执行方式

### 6.1 不展开权重的核心恒等式

由 `H H^T=I`，对每个 group：

```text
dot(w_hat, x) = n_store * dot(codebook[q], H x)
```

所以每个输入 group 的 FWHT 可以在 activation 侧只计算一次，再对所有 output rows 重用。推理时只读 packed codes 和 norms、查小码本累加，无需生成完整 dense weight。对于共享同一输入的 Q/K/V/gate，可以复用 activation 变换；out_proj 则有不同输入，不能复用错缓存。这是由格式推导的精确实数恒等式，不是速度测量。

独立 C++ 引擎将单 token 的整层循环放在 native 内执行。embedding/Engram 只解码被索引的行；Hadamard MLP 使用 FWHT；四 lane mHC 使用 SIMD Sinkhorn；GQA 中共享 KV 的两个 query head 复用 K/V 读取。native prefill 逐 token 执行，另提供显式 PyTorch 批量 prefill。

### 6.2 SDOT 四行计算与公开内核参考

OpenNeedle 的 `SdotCQ::row4` 在现有行主序 packed CQ2/CQ4 权重上同时计算四个输出行。每个 group 的激活向量只加载一次，供四组独立 INT32 累加器使用；group 点积再按各行 norm 与激活 scale 转回 FP32 并累加。尾行使用单行内核。该实现不重排磁盘或内存中的权重，也不创建整模型 INT8/dense 副本。

`row4` 与单行 SDOT 保持相同的码本量化、激活量化和逐 group 求和公式。CQ2/CQ4、group64/128、padding、尾行及不同线程数的逐元素对照见 [四行内核测试](../tests/test_sdot_row4.py)。相对于 FP32 的 SDOT 舍入误差仍存在，见 [精度结果](benchmark.md)。

矩阵 `multiply` 与普通引擎 `linear` 仅在输出行数至少 128 且线程数大于 1 时进入并行区域；融合 QKVG 按四行工作单元划分任务。OpenMP 等待策略由调用方和运行时决定，库导入不修改进程环境。

[Arm NEON Intrinsics Reference](https://arm-software.github.io/acle/neon_intrinsics/advsimd.html) 定义了此实现使用的 `vdotq_s32`、`vqtbl1q_s8` 与横向归约指令。运行时通过 Linux HWCAP 检查 DotProd，指令增强函数单独标记 target，默认 FP32 路径可使用普通 NEON。

通用 Cactus 的 [`cactus-kernels/src/matmul.cpp`](https://github.com/cactus-compute/cactus/blob/09cb35ab29aaad66189615192d27d84bebbc0522/cactus-kernels/src/matmul.cpp) 有 `cactus_quant_2bit_gemv_interleaved`：

- 4 output rows 一组，按 group 交错存储，减少解包与横向规约。
- activation 先 Hadamard，再每组 absmax 量化到 int8；码本也按 maxabs/127 映射 int8。
- NEON table lookup 展开 2-bit code，SDOT 做 int8 dot，再乘 activation scale、codebook scale 和 FP16 norm。
- transform 与 GEMV 两阶段使用持久线程池；共享 input 的多个矩阵有专门融合路径。

这是工程参考，不是 Needle 闭源库的源码。它的接口使用 FP16 activation，中途有 int8 近似和不同布局；移植到 AVX2 或直接调用也需重新验证精度。Cactus 的 [CQ 文档](https://github.com/cactus-compute/cactus/blob/09cb35ab29aaad66189615192d27d84bebbc0522/docs/cactus_quants.md) 介绍更广泛 PTQ 应用，与 Needle 的量化感知训练不可互换。

### 6.3 性能比较协议

至少记录 CPU 型号、ISA、物理核心、线程数与 affinity、编译器/flags、binary hash、模型 hash、CPU 频率策略、暖机数、重复次数、prompt token 数、输出 token 数、grammar 和 retrieval 设置。分别记录模型加载、tokenization、prefill、首 token、decode、完整 JSON 延迟和峰值 RSS。

原始词表全部 logits 与 grammar 候选投影的工作量不同，不能把前者的 token/s 与后者比较后直接声称 kernel 更慢多少。官方产品页声称约 500 tok/s Raspberry Pi 5 decode；这只是官方目标量级，不是当前机器上的复现结果。是否接近官方，必须在相同机器、输入和生产设置下测量。[官方部署说明](https://cactuscompute.com/needle)

报告应分开：独立引擎 / 官方引擎、FP32 reference / packed kernel、全词表 / grammar、单 token / 完整工具调用。最终性能结论必须来自本项目实际生成的报告文件。

## 7. 相关文献与适用边界

| 文献 | 对本项目的作用 | 不能据此假定 |
|---|---|---|
| [QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks, ICML 2024](https://arxiv.org/abs/2402.04396) | 理解 Hadamard 使分布更适于低位码本量化、量化后微调的重要性 | CQ 就是 QuIP#；QuIP# 使用随机 Hadamard 和 E8 向量码本，Needle 是固定组 Hadamard + 标量 Lloyd-Max |
| [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880) | 多 stream residual、双随机 routing 的设计背景 | 可以替换 Needle 的 lane offsets、迭代次序或参数化 |
| [Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](https://arxiv.org/abs/2601.07372) | n-gram 条件存储、以 gather 分离容量与算量的背景 | 大规模 Engram 论文的 hash/config 就是 Needle 的发布配置 |
| [LUT-GEMM: Quantized Matrix Multiplication based on LUTs for Efficient Inference in Large-Scale Generative Language Models](https://arxiv.org/abs/2206.09557) | 不完整反量化、查表累加和内存带宽的优化思路 | 其 GPU benchmark 能代表此处小模型 CPU 的速度 |

公开 JAX 架构和导出格式定义模型与权重布局；生产库的全部整数舍入、grammar 候选投影和校准行为未由这些参考完整定义。独立实现的验证范围以源码对照、数值测试和工具调用结果为准。

## 优化记录

当前有效实现的计算路径与数值边界集中在[引擎架构](architecture.md)。以下链接固定到实验记录，保留依据而不在主分支重复维护逐轮报告。

| 工作 | 结论 | 历史记录 |
|---|---|---|
| mHC 解码投影 | FP32 成对点积共享输入，三个投影合并调度；小矩阵串行 | [mHC 实验](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/docs/mhc-optimization.md) |
| 固定 64 维 QK | 固定维度帮助编译器展开原 FP32 Q × INT8 K 路径，没有新增 Q 量化 | [QK 实验](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/docs/attention-optimization.md) |
| Attention 调查 | 区分 norm/RoPE/gate 与 attention 核心，公开参考不等于闭源实现 | [调查](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/docs/attention-research.md) |
| 大候选投影、attention 串行化、唯一候选链 | 未确认稳定请求收益或当前工作负载缺少机会，未进入默认实现 | [解码实验](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/docs/decode-experiments.md) |
| 常驻 Python 会话 | 复用模型、工具分词、grammar 与前缀，减少完整请求重复工作 | [会话实验](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/docs/python-runtime.md) |
| C++ tokenizer 与 grammar | 原生编译和解码；保留 Python 元数据视图与独立参考测试 | [前端实验](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/docs/native-frontend.md) |

各轮有不同基线与计时范围，收益不能叠加。数值逐位一致指相对各自实验基线，不消除 SDOT/INT8 KV 原有误差。历史复现须在对应提交的独立 checkout 中运行；当前回归测试继续保留在 `tests/`。
