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

## 5. PSI

`PSI = sum over bins of (a - e) * ln(a / e)`，e 是参照里这一箱的比例，a 是当天落进这一箱的比例。

- **比的是比例，不是人数**：两边各自加起来等于 1；人数整体翻倍、各箱比例不变时 PSI = 0。总人数的变化要另做检查。
- **箱的边界用参照切、冻结后沿用**：当天若按自己的分数重切十分位，每箱都是 10%，PSI 恒为 0。
- **每一项都不小于 0**（a − e 与 ln(a / e) 同号），所以一箱的大偏差不会被别的箱抵消；报警看一个变量所有箱的总和，各箱的项用来诊断人往哪里移。
- **对变少的箱更敏感**：同样差 0.15，0.25 → 0.10 的项（0.1374）约是 0.25 → 0.40（0.0705）的两倍；某箱当天为 0 时 ln(0) 发散，要给比例加下限。
- **和 SRM 的关系**：差别小时 `(a - e) * ln(a / e) ≈ (a - e)^2 / e`，所以 PSI ≈ 卡方统计量 / N。
  SRM 用卡方的 p 值（乘了 N，人越多越敏感，要的就是「机制坏一点也报」）；PSI 除掉 N，只量差多大，用固定阈值，否则几百万人下每天都显著。
  玩具例子（4 箱，e 各 0.25，a = 0.40 / 0.30 / 0.20 / 0.10）：PSI 0.2282，近似式 0.20。

- [x] **决定（Andy，2026-10-12）：参照 = Valid（cutoff 11-01）**，切点从 Valid 的分数和特征上切，冻结存档。
  - 理由：模型没在 Valid 上训练（只用于 early stopping 和校准），是上线前最后看到的一批人，同一个冻结模型、同一套校准；分数已在 `scores/2019-11-01`。
  - 不用 Train（10-15）：模型在上面训练，分数是样本内的，人群不变也会和新数据不同。不用前一天：慢漂移每天只差一点，永远不报警。

- [x] **决定（Andy，2026-10-12）：算 PSI 的变量 = 分数 `p` + SHAP 前 5 个特征**：`n_purchases_per_day`、`avg_session_seconds`、
  `days_since_last_event`、`n_categories`、`active_days`（`model_report.md` 第 4 节，合计约占 SHAP 的 64%）。
  - 理由：只看 `p` 时分数漂了说不出是哪个特征；24 个全看则检查次数多、误报多。
  - 待处理：`n_purchases_per_day` 大多数人是 0，十分位切点会重复，写 PSI 函数时处理。
  - 猜测（未验证）：11-14 到 11-16 的购买记录缺失落在特征窗口里时，`n_purchases_per_day` 的 PSI 可能跳起来。

- [x] **决定（Andy，2026-10-12，回放之前）：6 个变量中任何一个 PSI ≥ 0.25 → `alert` 失败（网格变红）；在 0.1–0.25 之间只写日志**。
  - 阈值是信用评分行业的经验值（< 0.1 没什么变化、0.1–0.25 值得看、≥ 0.25 要查原因），没有统计推导，表达的是「变化多大才值得管」。
  - 抽样波动比阈值小四个数量级：没有漂移时 PSI ≈ (箱数 − 1) × (1/n1 + 1/n2)，10 箱、179 万对 200 万人约 9.5e-06；
    模拟 200 次平均 9.4e-06、99% 分位 2.1e-05（2026-10-12）。
  - 特征也能变红：特征漂了、分数没漂，也可能是数据管道的问题（如购买记录缺失）。
  - 在回放之前写定，和 P4 一样，不看结果再定判据。

### 计算（`make_cuts` / `bin_shares` / `psi`，Andy 在 `~/p5_psi.ipynb` 写，2026-10-12）

- `make_cuts(reference, n_bins=10)`：参照的 n_bins 等分点（`np.quantile`），去重。只对参照算一次，存档（相当于 fit）。
  Valid 上去重后的箱数：`p` 10、`n_purchases_per_day` **2**（90.1% 是 0，切点全是 0）、`avg_session_seconds` 8、
  `days_since_last_event` 10、`n_categories` 5、`active_days` 4。
- `bin_shares(values, cuts)`：等于切点的值归左边那箱（`np.searchsorted(..., side="left")`），`np.bincount(..., minlength=len(cuts) + 1)` 保证空箱也有 0
  （`groupby` / `value_counts` 只给出现过的箱，会少行）。对参照算出 e，对当天算出 a，用同一组切点（相当于 transform）。e 用实际数出来的比例，不假设每箱 10%。
- `psi(expected, actual, floor=1e-4)`：两组比例先 `np.maximum(..., floor)` 托底，再 `sum (a - e) * ln(a / e)`。

- [x] **决定（Andy，2026-10-12）：floor = 1e-4**。值是 Claude 提的，没有标准出处（业界有 1e-4、1e-6，也有每箱人数加 0.5）。
  - 只在当天某箱几乎没人时起作用：参照每箱最少约 9.9%，碰不到；1e-4 在 200 万人里约 200 人。
  - 一箱从 e = 0.1 变成 0 时，这一项 = `(floor - 0.1) * ln(floor / 0.1)`：floor 1e-2 → 0.207（**不到 0.25，漏报**）、1e-3 → 0.456、1e-4 → 0.690、1e-6 → 1.151。
    要让空掉一箱单独就报警：`0.1 * ln(0.1 / floor) ≥ 0.25`，即 **floor ≤ 0.1 / e^2.5 ≈ 0.0082**；低于这个数，取多少只改变 PSI 的大小，不改变报不报警。
  - 不托底时 numpy 只给 `RuntimeWarning`，PSI = inf；托底后 e = [0.5, 0.5]、a = [1.0, 0.0] 的 PSI = 4.6043。
