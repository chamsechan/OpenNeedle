# 使用指南

安装和最小推理示例见[项目首页](../README_zh.md)。所有命令从仓库根目录运行，Python 包及命令仍名为 `needle2`。默认 native 依赖仅为 NumPy；转换、PyTorch 推理、微调和 QAT 需要 `python -m pip install -e '.[torch]'`。

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

模型、tokenizer 和引擎在构造时建立。会话只保留最近一组工具的 grammar/DFA 和一个原生前缀快照，避免随请求数增长的会话级缓存。原生 grammar 是可复用的不可变状态表，每次 decode 使用独立状态游标；Python 参考编译器的最多 8 项全局 LRU 不参与 native 路径。

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
| `native_decode_call_seconds` | `engine.decode_from_logits()` 的调用墙钟时间，包含 Python 包装、FFI 与输出复制，不是纯 C++ 内核计时；PyTorch 路径为 null |
| `decode_seconds` | 包含首 token 选择的整个解码阶段（native 中首 token 也由 C++ 选择） |
| `parse_seconds` | token 解码、响应解析及结果字段构造 |
| 单次入口的 `setup_seconds` / `total_seconds` | 临时会话初始化 / 整个单次函数耗时 |

`prefill_tokens` 是实际处理数，`prompt_tokens` 是完整输入数；命中缓存后不能用完整 prompt token 数计算 prefill TPS。已有 `decode_tokens_per_second` 仍以生成 token 数减一除以 decode 时间。零生成长度返回空 token 列表，不调用解码循环。

## 模型转换

```bash
python -m needle2 to-torch artifacts/official/needle2.cact artifacts/pytorch
python -m needle2 quantize artifacts/pytorch artifacts/roundtrip.cact
python -m needle2 inspect artifacts/roundtrip.cact
```

保留转换目录中的 `weights.safetensors`、`config.json` 和 `source.cact`。未修改张量保留原始压缩字节，修改后的张量重新量化；支持相同 Needle 2 架构。

## FP16 master 转换与量化导出

微调可从官方 FP16 master 开始，使用匹配的部署文件提供模型结构与量化元数据：

```bash
python -m needle2 to-torch artifacts/official/checkpoints/needle2.pkl \
  artifacts/pytorch_master --template artifacts/official/needle2.cact
python -m needle2 quantize artifacts/pytorch_master artifacts/from_master.cact
```

转换目录中的 `weights.safetensors`、`config.json` 和 `source.cact` 应一并保留。
部署 `.cact` 反量化后的数值保留原有量化误差；FP16 master 入口用于保留发布的
浮点权重精度。转换统一使用 `python -m needle2 to-torch` 和 `quantize`。

## 加载、增量推理与保存 PyTorch 模型

`load_torch_model()` 接收转换后的 checkpoint 目录，也可直接读取 `.cact` 并反量化为可训练的 `NeedleModel`。部署权重的反量化值与量化前的 master 参数不同；继续训练通常应从 `artifacts/pytorch_master` 开始。

下面的示例独立执行，包含 tokenizer、prompt 和输入张量的构造：

```python
import json
from pathlib import Path
import torch

from needle2.convert import load_torch_model
from needle2.prompt import render_prompt
from needle2.tokenizer import RefTokenizer

checkpoint = Path("artifacts/pytorch_master")
tokenizer = RefTokenizer.from_cact(checkpoint / "source.cact")
tools = json.loads(Path("examples/tools.json").read_text())
text = render_prompt("Set a timer for 5 minutes.", tools)
ids = [2] + tokenizer.encode(text)  # 当前发布模型的 BOS ID 为 2
inputs = torch.tensor([ids], dtype=torch.long)

model = load_torch_model(checkpoint).eval()
with torch.inference_mode():
    logits = model(inputs)                  # [batch, time, vocab_size]
    split = len(ids) // 2
    first_logits, cache = model(inputs[:, :split], use_cache=True)
    rest_logits, cache = model(inputs[:, split:], cache, use_cache=True)

print(logits.shape, cache.position)
```

`NeedleCache` 保存各层 KV、绝对位置、滑动窗口及 Engram 短历史。批量输入可传 `attention_mask`；固定前缀可传 `sink_mask`，形状为 `[batch, current_time]` 或 `[batch, total_time]`。底层 `model.generate()` 是 greedy token 生成，工具 schema 约束由上层推理接口提供，详见 [grammar 说明](architecture.md)。

`save_torch_weights()` 更新一个**已有 canonical checkpoint 目录**中的权重，保留其 `config.json` 和 `source.cact`。保存到新目录前应先复制 checkpoint；它不是创建任意空目录的通用 `save_pretrained()`。以下片段沿用上面的 `model`、`checkpoint`：

```python
import shutil
from pathlib import Path
from needle2.convert import save_torch_weights

output = Path("artifacts/experiments/python_weights")
shutil.copytree(checkpoint, output)  # output 必须是新目录，保护输入 checkpoint
save_torch_weights(model, output)
```

`source.cact` 保留码本、tokenizer 和部署格式；`config.json` 保留几何信息与来源指纹。重新导出时需要两者。训练修改后的权重会按其原有精度配置重新量化，导出方法见[项目首页](../README.md)。该接口不保存 optimizer 或训练调度器状态。

## CQ QAT 与监督微调

`enable_qat()` 按模板 `.cact` 中每个张量的位宽和码本注册 CQ straight-through parametrization。应在模型迁移到训练设备之后、创建 optimizer 之前调用。`disable_qat()` 默认恢复更新后的 master 参数，随后才能保存和重新导出。

下面是一个完整的单样本训练步骤。它单独定义全部输入，并将结果写入新目录：

```python
import json
import shutil
from pathlib import Path
import torch
import torch.nn.functional as F

from needle2.convert import load_torch_model, save_torch_weights
from needle2.prompt import render_prompt
from needle2.qat import enable_qat, disable_qat
from needle2.tokenizer import RefTokenizer

source = Path("artifacts/pytorch_master")
output = Path("artifacts/experiments/qat_python_step")
shutil.copytree(source, output)  # 已存在时直接报错，不覆盖原有训练结果

device = torch.device("cpu")
torch.set_num_threads(1)
tokenizer = RefTokenizer.from_cact(source / "source.cact")
tools = json.loads(Path("examples/tools.json").read_text())
prompt_ids = [2] + tokenizer.encode(render_prompt("Set a timer for 5 minutes.", tools))
answers = [{"name": "set_timer", "arguments": {"minutes": 5}}]
target = ("<tool_call>" + json.dumps(answers, separators=(",", ":"))
          + "</tool_call><|im_end|>")
ids = prompt_ids + tokenizer.encode(target) + [1]  # 发布模型 EOS ID 为 1

model = load_torch_model(source, quant_activations=True).to(device).train()
if len(ids) > model.config.max_seq_len:
    raise ValueError("训练样本超过 max_seq_len")
enable_qat(model, source / "source.cact")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

inputs = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
labels = torch.tensor([ids[1:]], dtype=torch.long, device=device)
labels[:, :len(prompt_ids) - 1] = -100  # 只监督回答部分
optimizer.zero_grad(set_to_none=True)
logits = model(inputs)
loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()

disable_qat(model)
save_torch_weights(model, output)
print(float(loss.detach()))
```

此例同时启用权重 CQ QAT 和公开 A8 fake quant；仅需权重量化时可省略 `quant_activations=True`。`disable_qat(model, keep_quantized=True)` 会保留当前量化值，通常不用于恢复 master 后再次导出的流程。CPU 全模型 QAT 的时间和内存成本较高；普通 PyTorch GPU 设备迁移可用，但本次工作区没有 GPU 实测结果。

批量管理样本可使用 [scripts/finetune.py](../scripts/finetune.py)。JSONL 每行包含 `tools`、`query`、`answers`，可选 `reasoning` 和 `system`。以下代码创建一个格式示例：

```python
import json
from pathlib import Path

path = Path("artifacts/training/timer.jsonl")
path.parent.mkdir(parents=True, exist_ok=True)
row = {
    "tools": json.loads(Path("examples/tools.json").read_text()),
    "query": "Set a timer for 5 minutes.",
    "answers": [{"name": "set_timer", "arguments": {"minutes": 5}}],
}
path.write_text(json.dumps(row, ensure_ascii=False) + "\n")
```

运行一次训练步骤以检查流程；输出目录必须尚不存在：

```bash
python scripts/finetune.py \
  artifacts/pytorch_master artifacts/training/timer.jsonl \
  artifacts/experiments/qat_timer \
  --qat --steps 1 --lr 1e-5 --device cpu --threads 1
```

脚本会复制输入 checkpoint，保存新权重和 `training.json`。单样本、单步骤用于演示接口；训练与量化后的质量需要在独立保留集上重新评估。

## 固定 tools 前缀只计算一次

`NativeEngine` 支持同一实例内的显式 prefix 快照。应先完整编码每个请求，再按 `</tools>` token 切分，这样前缀和 suffix 拼接后仍与原始请求 token 完全一致。

以下示例使用同一工具目录处理两个请求，并启用上层工具约束：

```python
import json
from pathlib import Path

from needle2.grammar import ToolGrammar
from needle2.native import NativeEngine
from needle2.prompt import render_prompt, parse_response
from needle2.tokenizer import RefTokenizer

model_path = Path("artifacts/official/needle2.cact")
tokenizer = RefTokenizer.from_cact(model_path)
tools = json.loads(Path("examples/tools.json").read_text())
queries = ["Set a timer for 5 minutes.", "Turn on the kitchen light."]
requests = [[2] + tokenizer.encode(render_prompt(query, tools)) for query in queries]
prefix_len = requests[0].index(tokenizer.p2id["</tools>"]) + 1
prefix_ids = requests[0][:prefix_len]
assert all(ids[:prefix_len] == prefix_ids for ids in requests)

engine = NativeEngine(model_path, threads=2)
engine.reset(prefix_len=prefix_len)
engine.prefill(prefix_ids, last_only=True)
engine.cache_prefix()  # 仅在 position == prefix_len > 0 时允许创建

for ids in requests:
    engine.reset_to_prefix()
    grammar = ToolGrammar(tools, tokenizer)
    logits = engine.prefill(ids[prefix_len:], last_only=True)
    generated = []
    for index in range(96):
        token = grammar.select(logits)
        grammar.accept(token)
        generated.append(token)
        if token in (1, 5) or grammar.finished or index == 95:
            break
        logits = engine.step(token)
    print(parse_response(tokenizer.decode(generated))["function_calls"])
```

快照保存 prefix KV、token 历史和 Engram raw-v 环，恢复不重新执行前缀模型计算。任何 `reset()` 都会使快照失效，包括设置相同 `prefix_len`；当前 `generate()` 内部也调用 `reset()`，因此复用快照时使用上述 `prefill()` / `step()` 组合。同一个实例不能并发处理多个请求。

快照在运行时缓存之外增加约 `2 × layers × prefix_len × kv_heads × head_dim × 4` 字节的 KV 内存。当前模型约为 54 KiB/token；190-token 前缀约增加 10 MiB，再加约 40 KiB Engram 环和少量历史数据。恢复包含内存复制，应用测速应计入这一耗时。完整状态规则见[原生引擎说明](architecture.md)。

## PyTorch 批量 prefill 与 native decode

`backend="torch"` 将初始 prompt 交给 PyTorch 批量计算，再把 KV 和历史状态导入原生引擎。下面的示例可独立执行：

```python
import json
from pathlib import Path
import torch

from needle2.native import NativeEngine
from needle2.prompt import render_prompt
from needle2.tokenizer import RefTokenizer

model_path = Path("artifacts/official/needle2.cact")
tokenizer = RefTokenizer.from_cact(model_path)
tools = json.loads(Path("examples/tools.json").read_text())
ids = [2] + tokenizer.encode(render_prompt("Set the volume to 30 percent.", tools))
prefix_len = ids.index(tokenizer.p2id["</tools>"]) + 1

# NativeEngine 不隐式修改 PyTorch 的全局线程设置，由调用方显式选择。
torch.set_num_threads(2)
engine = NativeEngine(model_path, threads=2)
engine.reset(prefix_len=prefix_len)
logits = engine.prefill(ids, last_only=True, backend="torch")
next_id = int(logits.argmax())
logits = engine.step(next_id)  # 从这里继续执行 native decode
engine.release_prefill_model()
```

此例展示 prefill 和 decode 的衔接；完整工具生成可接入上一节的 `ToolGrammar` 循环。PyTorch prefill 只接受 `reset()` 后的初始 prompt，并保留一份约 175 MB 的 FP32 权重供后续请求复用。`release_prefill_model()` 释放这份引用，已导入的 native 状态仍可继续使用；下次调用 PyTorch prefill 会重新加载。

可显式设置 `NativeEngine(..., matmul="sdot")` 使用 ARM DotProd 近似 decode，但其 PyTorch prefill 仍为 FP32。SDOT 会额外舍入旋转后的激活及 centroid，不能与 `activation_bits=8` 组合。能力检测、精度边界及内存说明见[原生引擎说明](architecture.md)，线程选择见[同轮后端对比](benchmark.md)。

## Retrieval 与 confidence probe heads

发布模型包含检索和置信度 probe heads。PyTorch API 与普通模型参数一起加载：

```python
import torch
from needle2.convert import load_torch_model
from needle2.tokenizer import RefTokenizer

model_path = "artifacts/official/needle2.cact"
tokenizer = RefTokenizer.from_cact(model_path)
ids = [2] + tokenizer.encode("Start a five-minute timer.")
tokens = torch.tensor([ids], dtype=torch.long)
model = load_torch_model(model_path).eval()
with torch.inference_mode():
    embedding = model.encode_contrastive(tokens)   # [batch, 128]，单位长度
    confidence = model.forward_confidence(tokens) # [batch]，原始 logit
print(embedding.shape, confidence.shape)
```

训练 probe 时，两个 API 默认把 backbone 的 hidden cells detach；`train_backbone=True` 可让梯度继续回传到 backbone。上例使用 `inference_mode()`，不计算梯度。其它 checkpoint 可能未导出这些可选 heads。

原生流式接口可避免保存完整序列的 hidden cells，同时保留大矩阵压缩权重：

```python
from needle2.heads import NativeProbeEncoder
from needle2.tokenizer import RefTokenizer

model_path = "artifacts/official/needle2.cact"
tokenizer = RefTokenizer.from_cact(model_path)
ids = [2] + tokenizer.encode("Start a five-minute timer.")
encoder = NativeProbeEncoder(model_path, threads=2)
result = encoder.both(ids)  # 一次 backbone 计算共享给两个 heads
embedding = result["embedding"]
confidence_logit = result["confidence_logit"]
print(embedding.shape, confidence_logit)

# 单独需要一个 head 时：encoder.encode(ids) / encoder.confidence_logit(ids)
```

`NativeProbeEncoder` 每次调用重置内部引擎，要求一条不带 padding 的非空序列；可传 `prefix_len` 固定 attention 前缀。confidence 输出是公开 head 的原始 logit，不等于官方产品最终的校准分数。CLI 尚未自动实现大型工具目录的完整检索流程。

## 可选 JAX 源码准备与验证

常规使用不需要 JAX、Flax 或官方 Python 包。独立参考验证可安装可选依赖：

```bash
python -m pip install -e '.[reference,test]'
```

上游源码不随仓库分发。在新目录检出固定 commit，再通过 `--upstream` 指定路径：

```bash
git clone --filter=blob:none https://github.com/cactus-compute/needle.git \
  artifacts/reference/needle-jax
git -C artifacts/reference/needle-jax checkout --detach \
  53df049c4a1a82fca1027b81f9ff21336dfb0861

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/validate_jax.py \
  --checkpoint artifacts/official/checkpoints/needle2.pkl \
  --upstream artifacts/reference/needle-jax/needle/model \
  --output artifacts/reports/reproduced/model_jax_parity.json
```

已有 `third_party/needle` 时可省略 `--upstream`。此脚本使用官方 master checkpoint，对照公开 `architecture.SimpleAttentionNetwork` 的 FP32 路径；默认 16 个 token，序列不能超过 checkpoint 的 `kv_window`。它不加载闭源运行库，也不检验闭源引擎的逐位 INT8 运算。

其它验证入口可写入新的报告目录，保留现有实测证据：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q
python scripts/validate.py --output artifacts/reports/reproduced/validation.json
python scripts/validate_sdot.py --output artifacts/reports/reproduced/sdot_model_error.json
```

SDOT 验证需要支持 DotProd 的 Linux ARM64 CPU，输出是近似误差诊断。与官方库的转换回归、工具质量和性能比较涉及显式加载官方二进制，应按各脚本 `--help` 选择模型、库路径和输出文件。协议与既有结果见[实测报告](benchmark.md)和[同轮后端对比](benchmark.md)；技术依据见[技术参考](reference.md)。
