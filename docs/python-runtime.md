# Python 运行时与常驻会话

native 模式的矩阵投影、attention、mHC、KV 管理和完整 DFA 解码循环都在 C++。Python 保留部署文件加载、tokenizer、提示词、schema 编译、调用接口及结果解析。文件数量不能代表解码开销。

## 持续服务入口

```python
import json
from pathlib import Path
from needle2.inference import InferenceSession

session = InferenceSession(
    'artifacts/official/needle2.cact', threads=4,
    matmul='sdot', kv_cache='int8',
)
tools = json.loads(Path('examples/tools.json').read_text())
first = session.generate('Turn on the kitchen light.', tools=tools)
second = session.generate('Set a timer for 5 minutes.', tools=tools)
print(second['function_calls'], second['prefix_cache_hit'])
print(session.setup_seconds, second['request_seconds'])
```

模型、tokenizer 和引擎在构造时建立。会话只保留最近一组工具的 grammar/DFA 和一个原生前缀快照，避免随请求数增长的会话级缓存。grammar 的状态每次重置，但复用编译结果和词表索引。底层 DFA 编译器另外保留既有、最多 8 项的全局 LRU。

当 system/tools 的前缀 token 相同，恢复快照后只 prefill query 后缀；变化时重建前缀。不带 tools 的请求重置完整状态，不复用之前的工具前缀。每个请求都是独立的单轮输入，不自动积累对话历史。固定工具前缀的分词结果也复用：仅在 `</tools>` 是独立特殊 token、且不被其它特殊 token 包含时分段；该边界会结束 BPE 段，后缀编码显式关闭额外 dummy prefix。其它 tokenizer 形状回退为整段分词。KV 缓存匹配仍以实际前缀 token 为准。模型 SHA-256 在会话初始化时计算一次，不再为每个响应重新哈希模型。

同一会话的请求用锁串行执行；并发流应使用独立会话。不要在请求执行期间直接修改其底层 engine 或配置属性。异常不会将失败请求的生成状态带入下一次请求，下一次仍先重置/恢复缓存。会话保留模型及一个前缀快照占用的内存，使用结束后释放会话引用即可。

模块级 `generate(model_path, prompt, ...)` 保留原参数，创建临时会话并调用相同实现；CLI 继续使用它。常驻服务应显式保存会话，而不是重复调用这个单次入口。PyTorch 后端也可复用模型，但目前每次建立自己的请求 cache，不复用工具前缀。native 引擎选择 `prefill_backend="torch"` 时同样每次从空 cache 做完整 prefill，再导入 native cache；只复用模型、分词和 grammar，避免破坏 Torch prefill 的位置要求。检索功能仍走原来的工具排序接口，检索 encoder/未提供的工具 embedding 暂未增加跨请求缓存。

## 计时含义

| 字段 | 范围 |
|---|---|
| `session.setup_seconds` | 归档、tokenizer、引擎/模型初始化，可能包含首次原生库编译 |
| `request_seconds` | 会话取得锁后到响应构造完成，含 Python 准备及解析，不含排队等待或会话初始化 |
| `prepare_seconds` | 提示词、分词、检索、grammar 准备与重置 |
| `prefix_restore_seconds` | native 重置或恢复前缀 |
| `prefill_seconds` | 本次实际前向的 token；首次包括工具前缀，命中时仅后缀 |
| `native_decode_call_seconds` | `engine.decode()` 的调用墙钟时间，包含 Python 包装、FFI 与输出复制，不是纯 C++ 内核计时；回退/PyTorch 路径为 null |
| `decode_seconds` | 包含首 token 选择的整个解码阶段 |
| `parse_seconds` | token 解码、响应解析及结果字段构造 |
| 单次入口的 `setup_seconds` / `total_seconds` | 临时会话初始化 / 整个单次函数耗时 |

`prefill_tokens` 是实际处理数，`prompt_tokens` 是完整输入数；命中缓存后不能用完整 prompt token 数计算 prefill TPS。已有 `decode_tokens_per_second` 仍以生成 token 数减一除以 decode 时间。零生成长度返回空 token 列表，不调用解码循环。

## Python 清理范围

- 单次入口与常驻入口共用生成实现，移除原来 native 分支中的重复引擎构造。
- grammar 增加显式 reset，清除重复的逐请求词表索引构建。
- 缓存特殊 token 边界之前的工具前缀分词结果，移除每次重复运行的固定前缀 BPE；模型哈希移到初始化阶段。
- 默认依赖只保留 NumPy 和 regex；PyTorch、safetensors 通过 `pip install -e '.[torch]'` 安装，测试依赖包含它们。
- `model.py`、`qat.py`、`convert.py` 等保留为训练、转换和参考验证功能，native 推理不会导入它们。研究/benchmark 脚本不属于运行时包。
- Python grammar 循环保留为 DFA 超限时的精确回退，不能直接删除或换成无约束生成。
- 底层 `NativeEngine.generate()` 保留自定义 EOS/不指定 EOS 的原有行为，区别于默认在 EOS/stop 结束的 `decode()`；没有强行合并而改变停止语义。

本次没有把分词器、加载器全部翻译成 C++。需要继续迁移时，应根据完整请求阶段计时选择热点。

## 验证与复现

`tests/test_inference_session.py` 覆盖重复请求的隔离、工具/system 切换、无工具请求、零长度、grammar 回退重置、异常恢复、PyTorch 后端和 Torch prefill，以及禁止导入 torch/safetensors 时运行 native 推理。分段分词测试覆盖有/无 dummy prefix、特殊 token、中文和 Unicode。

完整套件 225 项测试、4 个 subtest 通过；随后补充 Torch prefill 兼容分支及对应测试，相关 15 项测试再次通过。在全新 venv 中执行默认安装，仅安装 NumPy、regex 和本包，未安装 torch/safetensors，重复真实工具调用及前缀复用验证通过。源码哈希、安装依赖与验证状态见 [验证清单](../reports/session_validation.json)。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/test_inference_session.py tests/test_runtime_regressions.py tests/test_grammar.py tests/test_native_grammar.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/benchmark_session.py --repeat 3
```

基准比较提交 `2664107` 的单次入口与当前常驻会话，包含分词、初始化、解析；19 个真实工具用例检查 token 一致与预期调用。单次入口在同一进程中重复执行，因此模块导入、库和文件页缓存已热，不应标注为全新进程冷启动。结果保存到 `reports/session_benchmark.json`。

## 本机完整请求结果

[最终报告](../reports/session_benchmark.json)使用 ARM64、CPU 0–3、native 4 线程、SDOT 和 INT8 KV。每例三次测量、交错调用旧单次入口和常驻入口，另检查新单次包装的 token 一致性。

| 场景 | 旧单次入口 ms | 常驻会话 ms |
|---|---:|---:|
| Basic | 467.35 | 53.35 |
| Expanded | 1,222.27 | 76.89 |

Expanded 的准备阶段中位数为 0.38 ms，prefill 29.63 ms，decode 46.98 ms，解析/结果构造 0.06 ms。首次会话初始化另计，本轮约 79 ms；首次工具 schema 分词、DFA 编译和前缀 prefill 发生在首个请求，未计入热请求表格。

[中间版本](../reports/session_benchmark_initial.json)仅复用引擎、grammar 和 KV 前缀，Expanded 仍约 412 ms，其中重复工具前缀分词约 328 ms。最终版本缓存分词结果，并将模型哈希移至初始化。两版阶段计时支持这些工作的优化优先级，但不同测量轮次不能直接当作严格配对实验。

完整入口收益主要来自消除重复 Python 工作和前缀 prefill，不是 C++ 解码内核获得同倍数提升。云主机测量存在长尾，各列中位数不能直接相加，也不保证其它机器获得同样的比例。19 个用例所有被比较结果的 token 和预期工具调用一致；不能把本结果与此前只测常驻引擎的 TPS 表混用，也没有重新测量官方完整接口。
