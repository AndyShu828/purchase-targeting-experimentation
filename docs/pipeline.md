# 每日打分管道（P5）

> Claude 2026-10-12 起草，决定归 Andy。记录 P5 的范围、环境和 DAG 的设计决定；改动任何一条，在原处注明日期和理由。

## 1. 范围

- [x] **决定（Andy，2026-10-12）：P5 只做一个 DAG `daily_score`**：每个 cutoff 打分、覆盖写入当天分区、算 PSI、超阈值告警；
  回放 10 月中到 12-01，README 放回放成功的网格视图截图和 CI 徽章（计划的两条验收）。
  - 理由：计划原文的「进度落后」一条写了「P5 可以先只做一个 DAG」。`daily_score` 用到前面三样东西：
    P2 特征函数的 `cutoff` 参数、P3 的批量评分（`scripts/score_users.py`）、特征分布，产出的分数表和 PSI 表给 P6 Dashboard 的「模型监控」页用。
  - 决定不做，记为以后的扩展：DAG `daily_events`（按天抽取、质量检查、写 Parquet）、DAG `weekly_train`（达到阈值才注册新模型）、
    写回 BigQuery（沙盒的表 60 天过期）、`ENV=dev/prod` 切换 BigQuery dataset、合并到 `main` 打 tag 视为发布到 prod。

## 2. 环境

- [x] **决定（Andy，2026-10-12）：Airflow 3.3.2 装在 WSL 里单独的 venv（`~/.venvs/airflow`），用 `airflow standalone` 起，不用 Docker**。
  - 理由：Spark 和数据都已经在 WSL 里，任务直接调现有脚本；Docker 要先装 Docker Engine，还要把 Java、Spark、数据目录挂进容器。
    计划原文写的是 docker-compose，验收条件不要求 Docker。
  - venv 和 `pte` 的 venv 分开，Airflow 的依赖用官方 constraints 文件锁版本
    （`pip install "apache-airflow==3.3.2" --constraint .../constraints-3.3.2/constraints-3.12.txt`）。
  - 本机开发设置（不入库，写在 WSL 的 `~/airflow/start.sh`）：DAG 目录指向仓库的 `dags/`；不加载示例；不登录（全员 admin）；
    新文件扫描间隔 30 秒（默认 300 秒）。UI 在 http://localhost:8080，语言按浏览器存（左下 User → Select language）。

## 3. 一次运行对应哪个 cutoff

- [x] **决定（Andy，2026-10-12）：cutoff = `data_interval_end` 的日期**，不用 `logical_date`。

Airflow 有两种按 cron 排运行的模式。2026-10-12 在 3.3.2 上实测（`hello_dates` 练习，跑完已删）：

| 模式 | 11-17 早上这一次什么时候开始跑 | `logical_date` | `data_interval_start` | `data_interval_end` |
| --- | --- | --- | --- | --- |
| A：按触发时刻（Airflow 3 默认，`create_cron_data_intervals = False`） | 11-17 00:00 | 11-17 | 11-17 | 11-17 |
| B：按区间（Airflow 2 默认；3 里设 `create_cron_data_intervals = True`） | 11-17 00:00 | **11-16** | 11-16 | 11-17 |

- A 是 `schedule="@daily"`、11-28 到 11-30 回放三次的实测：每次三列相等，都是那天的 00:00 UTC。
- B 是同一个 DAG 打开 `create_cron_data_intervals` 后，调度器给第一格排的：`data_interval = [11-28, 11-29)`，`run_after = 11-29 00:00`。
  区间过完才开始跑，`logical_date` 记区间起点。

理由：

- `data_interval_end` 在两种模式下都等于「哪天早上跑」，`logical_date` 换到模式 B 会早一天：不报错，打的分整体晚一天。
- 意思也对得上：cutoff 是特征的不含上界（`event_date < cutoff`），`data_interval_end` 是「这一刻之前的数据齐了」，也是不含上界。
- Paidy 用哪个编排工具、哪个版本查不到（JD 只写「like Prefect, Airflow, Argo」），取 `data_interval_end` 在 2 和 3 上意思不变。

## 4. `daily_score` 的 task

| task | 输入 | 输出 |
| --- | --- | --- |
| `score` | cutoff、事件 Parquet、冻结的模型 | `scores/<cutoff>/scores.parquet` 和 `features/` |
| `psi` | 当天的分数和特征、参照分布 | 每个变量一行 PSI |
| `alert` | PSI 表 | 超阈值则 task 失败（网格里变红） |

- [x] **决定（Andy，2026-10-12）：`score` 用 `BashOperator` 调现有的 `scripts/score_users.py`**，
  命令 `PTE_ENV=wsl ~/.venvs/pte/bin/python scripts/score_users.py --cutoff {{ data_interval_end | ds }}`。
  - 理由：脚本已测过，覆盖写入（特征 `mode("overwrite")`、分数直接覆盖），同一 cutoff 重跑结果相同；
    `pte` 的 venv 有 Spark 和 LightGBM，Airflow 的 venv 没有，分开装是为了避开依赖冲突。
  - 2026-10-12 实测：`BashOperator` 里 `{{ data_interval_end | ds }}` 渲染成 `2019-11-28`，`~/.venvs/pte/bin/python` 能 `import pyspark, lightgbm, pte`。
  - 回放范围：`check_cutoff` 接受 10-15 到 12-01（数据 10-01 起、窗口 14 天；12-01 是数据结束的次日早上），共 48 次。
