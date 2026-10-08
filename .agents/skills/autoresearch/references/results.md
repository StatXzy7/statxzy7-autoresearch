# 实验结果与复验

## 单次运行

每次实验使用新的 run ID。保留实际命令、配置和源码身份、协议、split、seed、时间、状态、原始指标及足以重算结果的产物。源码身份可用固定提交加 diff，或内容哈希明确的小型源码快照；指标记录工具不会自动替你保存源码。

```sh
python /path/to/autoresearch/scripts/research_run.py --root . --id baseline-s0 --kind baseline --timeout 600 --metric accuracy --direction max --protocol dev-fixed --split dev --seed 0 -- python src/experiment.py --config configs/baseline.json
```

替换脚本和实验入口路径。类型为 `baseline`、`reference`、`upper_bound`、`candidate` 或 `final`。命令 cwd 为 `--root`，参数按 argv 执行，不自动解释 shell 语法。命令行会记录到日志，凭据不应作为参数传入。

子进程从环境变量读取 `AUTORESEARCH_RUN_DIR` 和 `AUTORESEARCH_RESULT`。前者是本次专属输出目录，后者为评测器应写入的 `metrics.json` 路径。评测器从实际产物计算原始指标，并用 UTF-8 写入以下结构；示例数值仅说明格式：

```json
{
  "metric": "accuracy",
  "value": 0.75,
  "protocol": "dev-fixed",
  "split": "dev",
  "seed": 0,
  "artifacts": ["predictions.json", "config.json", "source-identity.json"]
}
```

产物路径相对本次 run，必须真实、非空且位于 run 内。记录工具检查退出码、有限指标、协议/seed/split 一致性以及产物路径，绑定 metrics 和产物哈希。成功为 `SUCCESS`，失败为 `FAILED`，超时为 `TIMED_OUT`；失败指标为 null。保留 `command.json`、`status.json`、stdout/stderr 与产物。

至少一个文件满足工具合同，不等于它足以支持科学结论。训练实验要保存真实 checkpoint 及独立重载日志，不能用配置或空模型文件替代。使用数据集时记录版本、划分和评价器身份。

## 汇总与选择

```sh
python /path/to/autoresearch/scripts/research_run.py --root . --summary
```

汇总会重查成功结果的指标与产物哈希，按 metric、direction、protocol、split、seed 分组，列出同组 Baseline 和最佳 candidate。Reference、U 与 final 不进入 candidate 排序；final 独立列出。缺失和失败记录仍在 runs 中，需在总报告中纳入实验次数与成本。

按 seed 分组得到的最优运行不等于跨 seed 的最佳方法。用冻结的聚合规则比较同一方法的全部正式 seeds；不能把不同方法各自最好的 seed 拼成一个方案。重复读取同一 checkpoint 不是独立训练。

## 方法轨迹与最终验证

按 [轨迹模板](../assets/trajectory.json) 记录方法假设、实际运行、原始指标、分析、保留/回退决定及下一步。时间使用带时区的 ISO 8601，失败保留原因和空指标。运行时长、方法分析与产物证据应能相互定位，不要求保存私有推理链。

冻结最佳候选后，恢复完整方法、配置及权重，在新进程和新 run ID 下复验。独立最终评价使用 `--kind final` 和对应 split/protocol，评价结果不用于再次选择候选。若评价口径变化，需要匹配该口径的对照。

报告原始分数、差值、波动、样本或 seed 数和实际成本。单次差值不证明统计显著或泛化改善；联合改动的效果不能自动归因到每个组件。分别记录环境就绪、Agent 执行、论文复现、方法改进和最终复验的结论与限制。
