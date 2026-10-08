# 环境、持续运行与恢复

## 环境探测

先核验目标主机、用户、目录、解释器、数据位置和已分配设备。使用已有 SSH 配置及正常主机密钥验证；复用兼容环境，避免覆盖共享驱动或全局依赖。

从安装 Skill 的项目根执行，`research-host` 为要替换的 SSH alias：

```sh
python .agents/skills/autoresearch/scripts/probe_environment.py --host research-host --workdir /workspace/research --out environment.json
```

省略 `--host` 探测本地；用 `--python` 指定目标解释器。`--gpu-smoke --device 0` 会实际分配显存并执行一个很小的 CUDA 矩阵运算，需要目标已有可用的 PyTorch/CUDA。它只检验相应设备链路，不证明训练、数据管线或长时间稳定性。

探测后运行真实项目的最小输入，确认数据读取、方法调用、产物写入和独立算分均成功。较长任务还应验证 checkpoint、重载和恢复流程，并按计划观察环境稳定性。安装、等待资源、环境故障与有效实验时间分别统计。

## 模型与资源

模型、推理设置、CLI 参数及账户权限以当前任务的实际配置为准。先读取已安装 CLI 的帮助，确认输入方式、工作目录、输出和退出行为，再做一次已授权的短模型运行。程序存在或合成命令成功都不证明模型通道可用。

多 Agent 比较时，分别建立干净工作区和独立输出，保持相同起点、题面、评分协议及预算；禁止读取其他运行的答案与轨迹。私有资产、凭据和资源限额由执行环境控制，需实际验证。

## CLI 监督器

在目标主机上运行监督器。示例中的路径、预算和 `agent-cli` 都需替换为已验证的任务值；示例参数不代表获准使用的资源。

```sh
python /path/to/autoresearch/scripts/agent_loop.py --root /workspace/research --prompt /workspace/research/agent-prompt.md --max-rounds 10 --max-seconds 3600 --round-timeout 600 --max-failures 3 -- agent-cli run
```

默认通过 stdin 传入提示词；CLI 若接受提示文件，在其 argv 中使用 `{prompt_file}`，监督器会替换成本轮文件路径。argv 不自动解释 shell 管道。

可从 [Agent 提示模板](../assets/agent-prompt.md) 创建任务提示，并补入真实目标、允许范围、协议和预算。

监督器在研究根目录创建 `agent-loops/<id>/`，保留总体状态、每轮命令、日志、时间和退出状态。它向子进程传入 `AUTORESEARCH_LOOP_DIR` 和 `AUTORESEARCH_TURN_DIR`。Agent 判断任务完成或没有有用进展时，在前者下写入 `STOP`，并在研究根的 `resume.md` 中说明证据和原因。操作者也可写入该文件；停止标记在轮次边界检查，不会立即打断当前调用。

SSH 长任务可在已有持久终端会话中运行监督器，并保存会话和进程身份。连接断开后先检查远端状态，再连接原会话；终端会话不能防止实例销毁，关键产物应存到持久磁盘。

## 中断恢复

```sh
python /path/to/autoresearch/scripts/agent_loop.py --root /workspace/research --resume /workspace/research/agent-loops/ACTUAL_ID
```

恢复使用原命令、提示词、预算和截止时间，不会重新赠送一份预算。检查旧进程及所有子进程确实结束后，才可对中断启动使用 `--acknowledge-stopped-children`。不要在无法确定归属时清理锁或杀进程；超时或中断的记录仍需保留。

监督器只约束墙钟、轮数、单轮超时与累计失败次数。GPU 配额、费用、网络、工具权限及逃逸进程必须由实际环境限制。进程树清理不能保证捕获已经脱离父进程的后台服务。退出码 0 只表示一次程序调用正常退出。
