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

该实现保证其支持子集内的结构与枚举约束，不能保证数值来自用户原文、工具选择正确，或拒绝无关请求。它没有复制官方工具检索、negation/grounding validator、官方内部提示词，也没有官方按候选行裁剪词表投影的优化；独立 runtime 仍先计算词表 logits。比较结果须保留这些差异。
