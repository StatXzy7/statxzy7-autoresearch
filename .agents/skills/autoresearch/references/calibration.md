# 基线、参考方案与目标标定

用于需要归一化评分、评估可改进空间或构建算法挑战的任务。仅比较原始指标时可以不使用此工具。

- **B**：健康且可执行的 Baseline，聚合全部正式运行。
- **R**：独立实现并实跑的参考改进，与 B 使用相同数据、协议、seeds 和可比预算。
- **U**：同协议和资源约束下有证据支持的目标。可为经验估计、实跑可达值或有明确解释的指标边界。

在正式比较前冻结 B/R 的协议、U 依据及分辨规则。不能事后移动 U、删除失败 seed 或削弱 B 来达到预定结论。构建受保护任务时，R 的解法和目标推导中的私有内容留在可信侧。

## 归一化

```text
越大越好：(x - B) / (U - B)，要求 U > B
越小越好：(B - x) / (B - U)，要求 U < B
```

B 映射到 0、U 映射到 1。不裁剪，允许负分及超过 1；R 不改变公式或斜率。U 不是全局最优证明。

参考方案的允许归一化区间由 `reference_score_range` 声明，默认 `[0, 1]`，可按任务设计设置更窄区间。`sigma_required` 默认 3，表示改善和剩余 R→U 空间相对 B 样本标准差的分辨要求；允许预先声明其他非负值。这是任务设计启发式，不是统计显著性检验。设置 0 不构成抗噪声证据。

随机协议至少需要三个配对 seeds 才能通过此工具的噪声检查，高波动时增加重复。B 的标准差使用 n−1 分母。确定性协议应声明 `stochastic=false`，不虚构随机显著性；实际误差来源仍需说明。

## 证据强度

| 状态 | 含义 |
|---|---|
| `DEMONSTRATED` | `empirical` 模式下每个声明 seed 都有独立 upper_bound 运行，聚合值与 U 一致 |
| `ESTIMATED` | `estimate` 模式下有方法、噪声和资源依据，尚未证明实际达到 |
| `NOT_PROVEN` | 理论边界或缺乏实际可达证据，不能称作已实现 |

`metric_ceiling` 必须说明为什么缺少可靠的可达估计。准确率 1 或非负误差 0 不自动代表预算内可实现。若任务要求 U 必须实证可达，声明 `require_demonstrated_u=true`。

## 运行

从 [标定配置](../assets/calibration-spec.json) 创建任务文件，填写真实 `metric/direction/protocol/split/stochastic/seeds`、B/R run ID、U 和证据路径。示例空值是待填写项，不能用于正式标定。

```sh
python /path/to/autoresearch/scripts/calibrate.py --root /workspace/research --spec /workspace/research/calibration-spec.json --out /workspace/reports/calibration.json
```

全部 run 来自 `research_run.py` 的收据。工具重新检查身份、结果和产物哈希，缺少或失败的正式 seed 不以部分均值替代。每次重试使用新 ID，并在报告中解释哪些运行按预先声明的规则进入正式聚合，所有失败记录继续保留。

输出 `CALIBRATED`、`INCOMPLETE` 或 `FAIL`，同时保留 U 证据强度。数值通过不判断算法合理性、预算公平性、权限隔离或资料可信性。最终评价若与 Dev 口径不同，必须单独建立对照和标定。
