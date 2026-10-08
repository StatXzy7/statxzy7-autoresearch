# AutoResearch

一个独立的研究实验 Skill：从论文、代码或研究问题出发，在本地或 SSH Linux GPU 环境中建立基线、迭代方法、管理预算与实验记录，并恢复和复验最佳方案。

## 安装

将 [.agents/skills/autoresearch](.agents/skills/autoresearch) 整个目录复制到目标项目的 `.agents/skills/autoresearch`，即可通过 `$autoresearch` 使用。只需这一个目录；其中包含全部流程说明、模板和工具。

辅助工具需要 Python 3.9 或更高版本，无额外 Python 包依赖。GPU 算法所需的框架、数据和设备由实际研究项目提供。

## 使用

> 使用 $autoresearch，根据这篇论文和代码建立可复现的基线，在指定 SSH GPU 环境中迭代方法。先确认评价协议和预算，保留每次运行的证据，最后复验最佳方案。

入口：[SKILL.md](.agents/skills/autoresearch/SKILL.md)。根据任务需要填写研究计划与任务配置；模型、算力、持续时间和实验规模均由当前任务确定。

| 工具 | 用途 |
|---|---|
| `probe_environment.py` | 解释器、磁盘、命令和 GPU 探测 |
| `research_run.py` | 实验执行、超时、日志及结果绑定 |
| `calibrate.py` | 可选的基线、参考方案和目标标定 |
| `agent_loop.py` | Agent CLI 的有界监督与恢复 |

```sh
python .agents/skills/autoresearch/scripts/probe_environment.py --out environment.json
python -m unittest discover -s tests -v
```

环境探测不会安装依赖或调用模型。单元测试使用合成数据验证工具行为；真实研究是否完成，需要相应实验、配对比较和干净复验的证据。运行工具不提供权限沙箱或费用隔离。

## 许可证

[MIT](LICENSE)。
