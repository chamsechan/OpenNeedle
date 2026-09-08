# OpenNeedle Skills

这些 skills 为 AI 编程助手提供 OpenNeedle 专用的操作流程，可按需使用。
每个目录包含标准 `SKILL.md` 和 Codex 界面元数据；实际命令调用仓库现有工具。

| Skill | 用途 | 请求示例 |
|---|---|---|
| [openeedle-inference](openeedle-inference/SKILL.md) | 安装、工具调用推理、缓存与应用接入 | “用我的 tools.json 跑通一次原生推理。” |
| [openeedle-convert](openeedle-convert/SKILL.md) | CACT / PyTorch 双向转换与一致性检查 | “把官方模型转为 PyTorch，再验证未修改权重的往返一致性。” |
| [openeedle-finetune](openeedle-finetune/SKILL.md) | 监督微调、CQ QAT 与部署导出 | “用我的 JSONL 数据做 QAT，导出模型并在保留集上评估。” |
| [openeedle-benchmark](openeedle-benchmark/SKILL.md) | 官方、原生与 PyTorch 的性能对比 | “在当前机器测三种后端的速度，记录线程与精度差异。” |

## 直接使用

在 OpenNeedle 仓库中，让助手读取所需 skill 并执行任务：

```text
请读取 skills/openeedle-convert/SKILL.md，将 artifacts/official/needle2.cact
转换到 artifacts/pytorch，并验证往返导出的一致性。
```

## 安装到 Codex

从 OpenNeedle 仓库根目录执行，安装全部四个 skills：

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
cp -Rn skills/openeedle-* "${CODEX_HOME:-$HOME/.codex}/skills/"
```

`-n` 保留已有文件；更新已安装版本时，应先检查本地修改再替换对应目录。
只安装一个时，将 `skills/openeedle-*` 换为对应的 skill 目录。
重新启动 Codex 后，可显式调用：

```text
$openeedle-inference 使用 examples/tools.json 运行原生 FP32 推理。
```

在其他工作目录调用时，请同时提供 OpenNeedle checkout 路径。Skills 不包含
模型权重，也不安装 Python 依赖；助手按任务需要使用项目的安装和下载入口。
