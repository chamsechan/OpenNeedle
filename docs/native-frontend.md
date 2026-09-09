# C++ tokenizer 与工具 grammar 编译器

native 推理的 tokenizer、工具 JSON 解析、schema 编译、首 token 选择和 DFA 解码均在 C++。Python 的 `frontend.py` 只负责句柄生命周期、UTF-8/JSON 字符串传入和数组适配。`InferenceSession` 默认使用这个前端；不再调用 Python grammar 编译器，也不再在 DFA 超限时自动进入 Python 正则循环。

## 与官方的关系

官方发布的 `needle.h` 暴露 `needle_load(cact, n)`、`needle_init(system, tools_json, index)`、`needle_complete(input, limit, out, capacity)` 和 `needle_reset()`，其 tokenizer/grammar 编译属于原生库内部职责。[官方技术说明](https://cactuscompute.com/needle)同样明确列出这一设计。

本次对齐的是**原生职责与数据边界**：tokenizer 从发布模型内嵌数据构建，工具 schema 以 JSON 传入 C++，约束状态由原生引擎执行，不要求 Python 编译 DFA。OpenNeedle 继续采用实例句柄，公开符号使用 `needle2_*`；没有冒充官方全局 ABI 的可直接替换实现，也没有复制其闭源 Matcher。

官方二进制包含 `Matcher::feed(char)`、`Frame`、`Branch`、`feed_schema_all` 等符号；本实现使用独立的 NFA → UTF-8 字节 DFA → token DFA 编译流程，不能声称内部算法与官方完全相同。

[官方初始化探测](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports/official_grammar_probe.json)显示官方会接受 range、pattern、type union、anyOf 等 schema。[生成探测](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports/official_grammar_behavior.json)中 union 和 anyOf 成功生成，已补入本原生编译器；范围、pattern、数组边界冲突样例出现截断，不能仅凭这些失败反推完整约束语义。探测固定使用本项目已下载的官方 2.0.4 库，不代表所有版本。

**尚未完全兼容的边界：** 当前不实现 pattern、数值 minimum/maximum、$ref 等全部 JSON Schema 关键字，遇到未实现字段明确报错；type union/anyOf 的组合也受本实现的关键字校验约束。对象键依照 schema 声明顺序，工具调用外层依照 name、arguments 顺序。不能据此宣称支持官方全部 schema 或逐 token 产品输出一致。

## Tokenizer

`Tokenizer` 在 C++ 中读取 `.cact` 内嵌 tokenizer blob，包括词片、score、类型、字节回退和 dummy-prefix 标志。UTF-8 字符边界、`▁` 空格替换、用户定义特殊 token 及字节回退沿用公开参考规则。

BPE 使用带版本标记的相邻链表与优先队列。合并优先级为 score，平分时选原始位置最靠左的一对；过期队列项丢弃，每次只更新相邻候选，避免 Python 全段反复扫描。decode 在 C++ 中处理字节 token、跳过控制/unknown token、替换非法 UTF-8 并恢复空格。长度显式传递，支持嵌入 NUL。

`NativeTokenizer` 保留 Python metadata 视图以兼容现有 `p2id/pieces/types` 调用，实际 encode/decode 不执行 Python BPE。`RefTokenizer` 留作独立参考，PyTorch 后端仍可使用。

## Grammar 编译和执行

1. 原生 JSON parser 保留 object 属性顺序，处理字符串转义和 Unicode surrogate pair，拒绝非法 UTF-8、重复键及尾随垃圾。
2. schema 编译为 Thompson NFA，支持闭合对象、required/optional 字段、嵌套数组及 minItems/maxItems、字符串、整数、number、boolean、null、标量 enum、type union、anyOf。
3. 合成 UTF-8 验证状态，拒绝 overlong 编码、surrogate code point 和超出 U+10FFFF 的字节序列。
4. 通过 tokenizer 字节 trie 生成 token 转移；反向可达性剔除无法完成工具调用的状态，支持没有完整字节回退的词表。
5. C++ 拥有连续 DFA 表，复用既有候选投影与解码循环。首 token 直接从 prefill logits 在 C++ 选择。

保留确定的编译预算：schema 深度 32、JSON 深度 64、NFA/字节 DFA 各 20,000 状态、token DFA 4,096 状态、4,000,000 条 token 转移；数组显式长度界限最多 1,024。超过预算明确报错。这些是本实现的资源边界，不是已确认的官方限制。

Python 的 `grammar.py` 和 `_grammar_dfa.py` 保留用于参考、测试以及 PyTorch 路径。native 不导入它们；因此默认 Python 安装仅需 NumPy，regex 与 PyTorch/safetensors 一起放入可选依赖。

## C ABI 与所有权

接口声明见 [frontend.h](../needle2/csrc/frontend.h)，实现见 [frontend.cpp](../needle2/csrc/frontend.cpp)。通过 CMake 安装时头文件位于 `include/needle2/`。

| 接口 | 职责 |
|---|---|
| `needle2_tokenizer_create/free` | 从模型 tokenizer blob 构造/释放 tokenizer |
| `needle2_tokenizer_encode/decode` | UTF-8 文本与 token 转换 |
| `needle2_grammar_compile/free` | 从工具 JSON 构造/释放原生 DFA |
| `needle2_engine_decode_compiled` | 首 token 选择与后续约束解码，复用已有 engine |
| `needle2_frontend_error` | 获取本线程最近一次错误 |

输入缓冲区仅在调用期间借用，tokenizer 和 grammar 复制持久状态。编译后的 grammar 不依赖 tokenizer 的生命周期。encode/decode 返回所需长度；容量不足时不写入、不截断，调用方扩容后重试。decode 输出以返回长度界定，不保证末尾 NUL。相同 engine 的解码必须串行；不同实例互不共享 grammar 状态。

该 ABI 是前端组件接口。模型加载、prompt 渲染、前缀复用和最终业务响应组装仍由现有 `InferenceSession` 协调，尚未提供官方 `needle_load/init/complete` 的二进制兼容整包替代品。

## 验证

- 原生/Python DFA 按可达状态对逐个比较候选 token 集合及终止状态，覆盖标量、enum、对象、数组、可选字段及空工具列表；union/anyOf 对照等价的标量语言。
- tokenizer 与 Python 参考及发布的 SentencePiece 模型对照；覆盖平分合并、中文、emoji、特殊 token、dummy prefix、NUL 和非法字节 decode。
- C ABI 检查截断 blob、非法 JSON、无效 UTF-8、容量不足及越界 token。
- 独立 C 程序链接共享库完成 tokenizer + grammar，完全不依赖 Python；全新 venv 仅安装 NumPy 也完成真实工具调用。
- [历史脚本 benchmark_frontend.py](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/scripts/benchmark_frontend.py) 对照提交 `0988b81` 的 Python 前端，比较首次工具/schema 请求和后续热请求，检查 19 个真实用例的完整 token 与预期调用。

性能原始数据见 [前端对照报告](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports/frontend_benchmark.json)，验证清单见 [frontend_validation.json](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports/frontend_validation.json)。首次请求包含分词/grammar 编译及前缀 prefill；这不是全新进程冷启动，不能与纯 decode TPS 或官方内部 TPS 混用。


## 本机性能观察

ARM64、4 线程、SDOT、INT8 KV，同一当前神经网络引擎，对照 `0988b81` 的 Python 前端。每套工具的首次准备只测一次；热请求每例预热后交错测量五次。

| 场景 | Python 首次准备 ms | C++ 首次准备 ms | Python 热请求 ms | C++ 热请求 ms |
|---|---:|---:|---:|---:|
| Basic | 240.06 | 21.50 | 64.44 | 62.86 |
| Expanded | 980.52 | 97.80 | 77.84 | 76.70 |

首次准备包括提示词分词和 grammar 编译，不含模型初始化或 prefill；这些是本次单次观测，不能当作跨设备保证。此前常驻会话已经缓存两者，所以热请求只小幅变化，尚不宣称稳定的显著吞吐提升。两种实现对 19 个真实用例的 token 和预期工具调用一致。

## 构建和复现

```bash
cmake -S . -B artifacts/frontend_cmake -DCMAKE_BUILD_TYPE=Release
cmake --build artifacts/frontend_cmake -j2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/test_native_frontend.py tests/test_inference_session.py
```

[独立 C 示例](../examples/native_frontend.c)接收从模型归档导出的 tokenizer blob 路径。它演示组件 ABI，不加载神经网络模型，也不依赖 Python 运行时。编译时包含 `needle2/csrc`，并链接构建产物 `libneedle2_native.so`。

这组一次性实验脚本已从当前版本移除。原脚本及完整复现步骤见[清理前的历史版本](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/docs/native-frontend.md)；需要复现时，请在该提交的独立 checkout 中按历史说明运行。
