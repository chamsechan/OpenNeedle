# 性能与验证总结

本目录只保留本总结。原始测速、实验中间结果及测试日志已从主分支工作树移除，
可在[清理前的固定提交](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports)中查阅；下述数字是既有测量记录，不是本次清理后重新运行的结果。

## 最新前端测量

4 核 ARM Neoverse-N1，native 4 线程，SDOT＋INT8 KV；Basic 3 个、Expanded 16 个请求，每例测量 5 次。

| 中位数 | Basic | Expanded |
|---|---:|---:|
| 完整热请求耗时 ms | 62.86 | 76.70 |
| Query prefill ms | 23.61 | 29.92 |
| Decode token/s | 488.16 | 402.38 |
| Prepare ms | 0.21 | 0.26 |

首次请求准备分别为 21.50 ms、97.80 ms，属于已初始化会话中的单次观测。
热请求不含模型初始化及首次 grammar/前缀构建。各列独立取中位数，不能直接相加。
历史官方 2.0.4 的 Expanded decode 为 493.70 token/s，属于另一轮内部自报计时；
与本项目的比值不代表同口径加速比。默认仍为 FP32，SDOT 和 INT8 KV 为可选近似模式。

## 验证结论

- 前端迁移的 19 个请求均生成预期调用，迁移前后完整 token 一致。
- 前端记录：248 项测试、4 个 subtest 通过；后续前端/会话相关 33 项测试通过。
- 历史质量诊断集：官方与原生均为 13/15；小样本不代表 BFCL 或广泛真实请求准确率。
- mHC 和固定 64 维 attention 优化分别完成相对各自基线的数值一致性验证；这不消除量化误差。

计时边界及复现见[性能说明](../docs/backend-comparison.md)、
[前端说明](../docs/native-frontend.md)和[精度验证](../docs/results.md)。
文档中的原始报告链接指向 Git 历史，各报告对应的提交及环境以原记录为准。

## 维护与复现

测速和验证脚本默认将新结果写入被 Git 忽略的 `artifacts/reports/`，支持 `--output` 的脚本可另选路径。
一次性实验脚本也已移除；历史实验请在上述固定提交的独立 checkout 中复现。
本目录不再收录原始结果。更新公开数字时，同步修改本总结及相关文档。

首页图表通过 `python3 scripts/render_readme_assets.py` 从以下两项中位数生成；原始依据为历史中的
[前端记录](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports/frontend_benchmark.json)与
[官方对照](https://github.com/chamsechan/OpenNeedle/blob/e096870b4b45b979b9714a172666233ffd99ab48/reports/performance_f4f9b38.json)。

```json
{
  "official": {
    "decode_tps_median": 493.7
  },
  "optimized": {
    "decode_tps_median": 402.37850683384045
  }
}
```
