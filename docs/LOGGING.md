# 全局日志

平台所有进程（Web、项目调度器、作业 worker、CLI）共用一套日志设施：
**按组件分目录 → 每个组件目录里再分高频/低频 → 一个进程一个文件 → 级别可配 + 自动轮转**。

实现：`src/mlplatform/logs.py`　配置：`configs/platform.yaml` 的 `logging` 段。

---

## 1. 为什么不直接 `logging.basicConfig()`

| 约束 | 做法 |
|---|---|
| 四个 conda 环境都要能用 | 只依赖标准库（worker 在 `label/train/test_system` 里跑） |
| 多进程同时写日志 | 轮转文件按 **pid 命名**，进程之间不共享文件（`RotatingFileHandler` 不是多进程安全的） |
| CLI 的 stdout 要给管道用 | 日志一律走 **stderr**，`bin/tp ... --json \| jq` 不会被日志污染 |
| 访问日志会刷屏，告警不能淹没 | **高频/低频分目录**，各自轮转、各自保留期 |
| 作业日志要给 Web 增量读 | `jobs/<job-id>/log.txt` 协议不变（只追加、不轮转） |

## 2. 目录结构

```
running_time/log/                                    # 日志总目录（唯一入口）
├── startup.log                                      # start.sh 启动输出（logger 就绪前的报错）
├── web/                                             # 组件 = 哪一类进程
│   ├── high/web-2577480.log                         #   高频：HTTP 访问日志（逐请求）
│   ├── high/web.log -> web-2577480.log              #   软链：永远指向当前进程（tail -f 用）
│   ├── low/web-2577480.log                          #   低频：启动/关闭/错误
│   └── low/web.log -> web-2577480.log
├── runner/
│   ├── high/runner-demo-3697791.log                 #   高频：逐条轮询明细（默认不打）
│   ├── low/runner-demo-3697791.log                  #   低频：启停、作业开始/结束
│   └── demo.stdio.log                               #   stdio 兜底（import 期崩溃才用）
├── job/
│   ├── high/job-20261112-170711-a78f-3698480.log    #   高频：逐张图片 / 逐轮 epoch
│   └── low/job-20261112-170711-a78f-3698480.log     #   低频：环境、数据、指标、告警
└── cli/
    ├── high/cli-3698448.log
    └── low/cli-3698448.log
```

命名规则：`<组件>/<通道>/<组件>[-<scope>][-<pid>].log`

| 组件 | scope | 谁在写 |
|---|---|---|
| `web` | 无 | FastAPI + uvicorn（`main_system`） |
| `runner` | 项目 id | 项目调度器（每个项目一个进程） |
| `job` | 作业 id | 阶段 worker（label / tile / relation / train / evaluate） |
| `cli` | 无 | `bin/tp` 每次调用 |

## 3. 高频 vs 低频

分的是**"会不会刷屏"**，不是级别。

| 通道 | 典型来源 | 轮转默认 | 保留默认 |
|---|---|---|---|
| `high` 高频 | HTTP 访问日志（每请求一行）、逐张图片标注结果、逐轮 epoch 指标 | 5MB × 3 份 | 3 天 |
| `low` 低频 | 进程启停、作业生命周期、参数快照、数据/指标摘要、**所有告警与异常** | 10MB × 10 份 | 30 天 |

两条保证：

1. **WARNING 及以上永远同时进 `low`**——即使这条日志来自高频 logger。
   所以"错误只出现在马上要被轮转掉的高频文件里"不会发生。
2. **高频明细一律用 `DEBUG` 级别**——控制台（默认 INFO）看不到，只有 `high/*.log` 有。

哪些 logger 算高频（`logs.HIGH_LOGGERS`，前缀匹配）：

| logger | 用途 |
|---|---|
| `uvicorn.access` | HTTP 访问日志 |
| `job.detail` | 作业 worker 的逐张图片 / 逐轮 epoch（`ctx.detail(...)`） |
| `web.access` / `runner.detail` | 预留 |

要再加：配置里 `logging.high_loggers: [my.noisy.logger]`，或代码里 `logs.mark_high("x.y")`。

## 4. 一行日志长什么样

```
2026-11-12 17:07:12 INFO    [runner][project=demo] 启动作业 job-...（evaluate）
2026-11-12 17:07:16 DEBUG   [job.detail][project=demo job=... stage=evaluate] 图片 3/20 000012.jpg: 4 个目标
```

| 字段 | 含义 |
|---|---|
| `%(asctime)s` | 时间（`logging.date_format`） |
| `%(levelname)s` | 级别 |
| `%(component)s` | 组件短名（`web` / `runner` / `job.detail` / `commands` / `uvicorn.access` …） |
| `%(ctx)s` | 进程绑定的上下文（`[project=… job=… stage=…]`，没有就是空） |
| `%(message)s` | 正文 |

## 5. 级别

```yaml
logging:
  level: INFO            # 全局默认（根级别取各 handler 最低值）
  console_level: INFO    # 控制台（stderr）
  file_level: DEBUG      # 两个文件通道的默认级别
  high: {level: DEBUG}   # 通道级覆盖
  low:  {level: INFO}
  levels:                # 按 logger 覆盖
    job: INFO
    uvicorn.access: WARNING
```

优先级：**参数 > 环境变量 > `levels` 覆盖 > 通道 > `logging.level`**。

```bash
export MLPLATFORM_LOG_LEVEL=DEBUG        # 全局
bin/tp --log-level DEBUG train run ...   # 只这一次
bin/tp train run ... -v                  # 等价
MLPLATFORM_LOG_SPLIT=0 bin/tp ...        # 高频/低频合并成一个文件（临时排障）
```

`bin/tp` 的控制台默认只有 `WARNING` 以上：命令的正常输出在 stdout，日志不该抢戏。

## 6. 轮转与清理

```yaml
logging:
  rotation: size         # size | daily | none（两个通道默认）
  max_bytes: 5242880
  backup_count: 3
  when: midnight         # rotation=daily 的切分点
  retention_days: 7
  high: {max_bytes: 5242880, backup_count: 3, retention_days: 3}
  low:  {max_bytes: 10485760, backup_count: 10, retention_days: 30}
```

- `rotation: size` → `RotatingFileHandler`：`x.log` → `.log.1` → `.log.2` …
- `rotation: daily` → `TimedRotatingFileHandler`：`x.log.2026-11-12`
- `rotation: none` → 纯追加（交给系统 logrotate）
- **清理按通道各自的 `retention_days`**：高频文件 3 天、低频文件 30 天，不会一刀切。
  每个进程启动时顺手清一次（最多每小时一次，标记文件 `running_time/log/.cleanup`），
  不在写日志的路径上做 IO。

```bash
bin/tp logs clean --dry-run      # 按通道保留期算，看会删什么
bin/tp logs clean --days 7       # 强制统一 7 天
```

### 多进程安全

`RotatingFileHandler` **不是**多进程安全的：两个进程同时轮转会互相改名/截断。
所以默认 `per_process: true`（pid 进文件名）。确实要单文件时：

```yaml
logging:
  per_process: false
  rotation: none        # 必须
```

`per_process: false` + 轮转 是非法组合，`bin/tp doctor` 会直接报错。

## 7. 代码里怎么用

```python
from . import logs

logs.configure(component="runner", scope=project.id, extra={"project": project.id})
logs.install_excepthook()                       # 未捕获异常也写进日志

log = logs.get_logger("runner")                 # 低频
log.info("启动作业 %s", job_id)                  # 用 %-占位符，别用 f-string

detail = logs.get_logger("job.detail")          # 高频（内置在 HIGH_LOGGERS）
detail.debug("图片 %s: %d 个目标", name, n)      # 只进 high/，控制台不显示

logs.mark_high("my.noisy.logger")               # 运行时把某个 logger 划到高频
logs.set_context(job=job_id)                    # 后续日志自动带 job=xxx
logs.describe()                                 # 当前进程的日志配置
logs.current_files()                            # {"high": Path, "low": Path}
```

作业 worker 里更简单：`StageContext` 已经接好。

```python
ctx.log("标注完成：…")        # 低频（含 [warn]/[error] 前缀自动映射级别）
ctx.detail("图片 3/20 …")     # 高频明细（DEBUG）
ctx.warn("…") / ctx.error("…")
```

**模块导入期**要拿 logger 用 `logs.module_logger("commands")`：不触发自动配置，
等进程入口 `configure()` 时统一挂 handler。

### 接第三方库

```python
logs.attach_stdlib("uvicorn", "uvicorn.error", "uvicorn.access")
```

这些 logger 的 handler 换成平台这一套（`propagate=False`）。频率按名字判定，
所以 `uvicorn.access` 自动进 high 通道。`bin/tp web` 已经这么做了。

## 8. 查日志

```bash
bin/tp logs show                        # 目录结构 + 通道参数 + 每个组件两个通道的文件
bin/tp logs list                        # 所有文件（组件 / 通道 / 大小 / 时间）
bin/tp logs list --channel high         # 只看高频
bin/tp logs list -c job --limit 10
bin/tp logs path [-c web]               # 目录路径（可 cd 过去）
bin/tp logs tail --lines 100            # 最新写入的那个文件
bin/tp logs tail -c job --channel low   # 只看某个组件的低频（告警都在这）
bin/tp logs tail -c runner --follow     # tail -f（新文件出现会自动跟着切）
bin/tp logs show --json                 # 给脚本/看板用
```

系统工具同样可以：

```bash
tail -f running_time/log/web/low/web.log       # 软链指向当前进程
grep -r WARNING running_time/log/*/low/        # 所有告警（高频通道的告警也在 low 里）
du -sh running_time/log/*/                     # 谁在吃磁盘
```

## 8.1 CPU 降级一定会留告警（不允许静默变慢）

`compat.resolve_device()` 决定真正用哪个设备。只要有降级，就记一条 **WARNING**
（`compat` 组件），同时出现在：

- `running_time/log/job/low/<作业 id>-<pid>.log`（带 `project=` / `job=` / `stage=` 上下文）
- `projects/<p>/jobs/<id>/log.txt` → Web「作业日志」面板里直接能看到

```
WARNING [compat][project=demo job=job-... stage=evaluate] device=auto 不可用
（Error 304: OS call failed or operation not supported on this OS），本次已降级为 cpu 运行，
速度会明显变慢。本机有 NVIDIA 设备，但 CUDA 初始化失败（驱动 / NVML 异常，或被其它进程占用）。
排查：nvidia-smi；python -c "import torch;torch.cuda.is_available()"
```

| 情况 | 日志 |
|---|---|
| `device: auto` 但没有可用 CUDA | WARNING（说明是"没设备"还是"有设备但起不来"） |
| `device: 0` / `0,1` 等显式 GPU 但不可用 | WARNING |
| 探测 GPU 算子失败（arch 不匹配等） | WARNING（带 torch 的原始错误） |
| `device: cpu`（显式要求 CPU） | 不告警，这是正常配置 |

同一个进程里同一种降级只打一次。平台级别还有 `bin/tp doctor` 的 `⚠ GPU:` 一行。

## 9. 与 `jobs/<job-id>/log.txt` 的关系

两者**不是同一份东西**：

| | `jobs/<id>/log.txt` | `running_time/log/job/*/<id>-<pid>.log` |
|---|---|---|
| 给谁看 | Web「作业日志」面板（按 offset 增量读） | 人 / `grep` / 归档 |
| 内容 | worker 的 stdout+stderr（含 ultralytics 原始输出） | 结构化日志（级别 + 项目/作业上下文） |
| 轮转 | **不轮转**（offset 协议要求文件只追加） | 按通道配置轮转 |
| 生命周期 | 随作业目录（删作业就删） | 随通道保留期（high 3 天 / low 30 天） |

worker 的 `print` 已经不在了：`ctx.log()` / `ctx.detail()` 走 logger → 控制台 handler（stderr）
被 runner 重定向进 `log.txt`，同时文件 handler 写 `running_time/log/job/{high,low}/`。
所以**一次调用，两处落地**，`log.txt` 里也能看到高频明细。

## 10. 新增一个组件（例如后台轮询任务）

```python
logs.configure(component="watcher", scope=project_id, extra={"project": project_id})
log = logs.get_logger("watcher")
try:
    ...
except Exception:
    log.exception("轮询失败")       # 栈也会进 low/
```

不用改配置：`running_time/log/watcher/{high,low}/watcher-<pid>.log` 自动出现，
`bin/tp logs list` 就能看到。

## 11. 排障清单

| 现象 | 检查 |
|---|---|
| 日志文件没出现 | `logging.file: true`？进程写过日志吗（`delay=True`：没写就不建空文件） |
| 只有控制台没有文件 | `MLPLATFORM_LOG_FILE=0`？目录不可写？`bin/tp doctor` 会报 |
| 想找某个请求/某张图 | 它在 **high**：`bin/tp logs tail -c web --channel high` |
| 想找告警/失败原因 | 它在 **low**：`bin/tp logs tail -c job --channel low`，或 `grep -r WARNING running_time/log/*/low/` |
| high 里没有某个 logger 的输出 | 该 logger 没被标成高频；加 `logging.high_loggers` 或 `logs.mark_high(...)` |
| 文件里没有 DEBUG | `logging.file_level` / 通道 `level`，或该 logger 被 `levels` 覆盖成 INFO |
| 日志文件很多 | 通道 `retention_days`（高频 3 天）；`bin/tp logs clean` 手动清 |
| 轮转丢失历史 | `backup_count` 决定保留几份；需要更久就调大或改 `rotation: daily` |
| 日志里出现密钥 | 不应发生：作业 spec 落日志前会脱敏（`commands._redact`），VLM Key 从不进 `job.json`/日志 |
| 看到 `已降级为 cpu 运行` | GPU 不可用（见 8.1）；`bin/tp doctor` 看 `⚠ GPU` |
