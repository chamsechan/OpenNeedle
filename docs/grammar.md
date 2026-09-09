# 独立工具调用 grammar

`needle2.grammar.ToolGrammar` 使用 `regex` 的 partial matching，在 `<tool_call>` 之后按照工具 schema 筛选每一个候选 token。候选按 logits 从大到小尝试；只有能延伸为合法 JSON 前缀的 token 可以输出。`<tool_call>` 之前保持原始贪心选择，完整 JSON 数组之后允许 `</tool_call>`。

```python
grammar = ToolGrammar(tools, tokenizer)
token = grammar.select(logits)
grammar.accept(token)
if grammar.finished:
    # 当前调用的 JSON 已完成。
    pass
```

支持工具数组、空调用 `[]`、多次调用，以及 `string`、`integer`、`number`、`boolean`、`null`，嵌套 closed object 和同质 array。object 支持 `required` 与可选字段；array 支持 `minItems`、`maxItems`；标量支持 `enum`。工具声明可以是直接 `name/description/parameters`，也可以是 OpenAI 风格的 `type=function` 外层包装。

所有 object 字段必须按 schema 的 `properties` 声明顺序输出，工具调用对象必须先 `name` 再 `arguments`。可选字段可以省略；模式大小随字段数线性增长，没有枚举所有可选字段组合。无 `additionalProperties` 的工具 schema 被解释为闭合参数集合；显式 `additionalProperties: true` 会报错。

这是受限 schema 编译器。`$ref`、`oneOf/anyOf/allOf`、类型联合、字符串 `pattern/format/minLength/maxLength`、数值范围、动态属性、tuple array 等未实现验证关键词会在初始化时报错。不会忽略这些限制然后把输出宣称为 schema 合法。默认值、description 等纯注释字段不参与验证。嵌套深度上限 32，数组显式计数上限 1024；标量 enum 使用 JSON 的规范字面值表示，不接受其他等价转义写法。

正常 token 将 SentencePiece 的 `▁` 替换为 ASCII 空格；BYTE token `<0xHH>` 使用原始字节。匹配期间允许合法 UTF-8 的未完成尾部，拒绝非法 UTF-8；只有完整 UTF-8 与完整 JSON 才能关闭调用。JSON 字符串支持转义。控制 token 不能闯入 JSON。

native 默认通过 `compile_tool_dfa(tools, tokenizer)` 将同一 schema 编译为字节 NFA，再与 UTF-8 状态机组合，并沿 tokenizer 字节 trie 构造 token DFA。工具名、参数名、分隔符均由实际 token 字节决定。仅 `<tool_call>` 之前不受约束；字符串和数字内部仍严格筛选。完整 `</tool_call>` 后立即停止，与 Python grammar 相同。

C++ `NativeEngine.decode(..., grammar_dfa=dfa)` 根据候选集合裁剪 LM head；只有一个合法候选时跳过投影，但仍更新隐藏状态和 KV。公开 `step_candidates` 的单候选分数是 `[0.0]` 占位值，不是真实 logit；多个候选返回实际分数，`return_hidden=True` 始终返回 `(logits, hidden_array)`。候选 ID 必须为词表内整数，非法输入在推进模型前拒绝。

编译结果按 schema 与 tokenizer 缓存。NFA/字节 DFA 各限制 20000 状态，token DFA 限制 4096 个正文状态和 4000000 条转移；超过限制抛出 `GrammarTooLarge`。`generate` 和 benchmark 会回退到原来的 Python `ToolGrammar`，保留约束及候选投影优化；不支持的 schema 仍明确报错。生成结果的 `grammar_backend`、benchmark 的逐结果元数据记录实际路径。首次编译属于初始化开销，不能当作热请求速度的一部分忽略不报。

该实现保证其支持子集内的结构与枚举约束，不能保证数值来自用户原文、工具选择正确，或拒绝无关请求。它没有复制官方工具检索、negation/grounding validator、官方内部提示词；独立候选投影和 DFA 是本项目的实现，并非官方内部算法的复刻。比较结果须保留这些差异。
