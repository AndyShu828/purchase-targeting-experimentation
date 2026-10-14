# Dashboard 与收尾（P6）

> Claude 2026-10-14 起草，决定归 Andy。记录 P6 的范围和设计决定；改动任何一条，在原处注明日期和理由。

## 1. 范围

- [x] **决定（Andy，2026-10-14）：P6 做下面六项，Dashboard 之外按这个顺序；时间不够先砍最后一项**。
  1. **Looker Studio Dashboard 三页**：① 业务漏斗与留存 ② 模型监控 ③ 实验报告（含利润曲线）。
     计划原文：「最小可行版本是 P1、P3、P4，加上 P6 的 Dashboard」，不砍。
  2. **`docs/monitoring.md`：两类监控的设计**（PSI 当天可算；PR-AUC / lift 要等 14 天标签）。
     这是 P5 验收的第 3 条，P5 收尾时漏了（P5 review 之后对照计划原文发现）；写成第 ② 页的规格说明，和第 ② 页一起做。
  3. **README 定稿**：问题 → 方法 → 结果 → Mermaid 架构图 → 复现，英文为主，附日文概要和「数据与模拟说明」。
  4. **LookML 示例**（计划里是可选）：`view` + `explore`，注明未部署。
     理由：JD「Looker dashboards are one of the main tools we use to deliver data science solutions」；Looker Studio 没有语义层，光做 Dashboard 说明不了 Looker。
  5. **日文决策备忘录** `docs/decision_memo_ja.md`：一页，结论 / 建议 / 风险。
     依赖 Paidy 业务研究：券的设定 2026-10-14 已按 Paidy 的实际产品更正（`experiment_design.md` 第 7 节）。
  6. **`docs/event_mapping.md`**（可选，最先砍）：换成 BNPL App 的事件时，特征和标签怎么对应。理由：JD「clickstream data, app activity data」。
- 决定不做：`docs/` 全量英文化。计划原文只要求 README 英文为主。
- 决定不做（Andy，2026-10-14）：Tableau 版 Dashboard。JD 的「a BI tool like Looker, Tableau」紧接着说 Looker 是主要工具；概念与 Looker Studio 相同，
  加的主要是界面操作。Tableau Public 只读抽取文件 / Google Sheets，官方 FAQ 的连接器里没有 BigQuery，要多搭一条导出路径。以后投 Tableau / Power BI 多的公司时再拿第 ① 页做一版（约 2–3h）。
- 仓库之外（计划同周）：Paidy 业务研究的其余部分（日本 BNPL 市场）、自我介绍中英日、5 分钟项目讲解、至少两轮模拟面试、简历条目。

## 2. 约束

- **BigQuery 沙盒的表 60 天过期**：`pte.events` 2026-09-22 建，**2026-11-21 过期**（`bq show` 实测）；
  数据集默认表过期也是 60 天，Dashboard 用的新表从建表起再算 60 天。⇒ 建表 / 灌数据的脚本进仓库，README 放截图兜底。

## 3. 数据怎么进 Looker Studio

- [x] **决定（Andy，2026-10-14）：三页都走同一条路 —— 本地 DuckDB 跑全量（10–11 月）→ 结果表存 Parquet → `bq load` 脚本上传到 BigQuery → Looker Studio 连 BigQuery**。
  - 理由：第 ② 页的 PSI 和第 ③ 页的利润曲线本来就只在本地，上传脚本无论如何要写；第 ① 页也走它，数字和文档（全量）一致。
  - 不选「在 BigQuery 上跑 SQL 建结果表」：沙盒只放得下 10 月（10 GB 上限，10 月占 5.6 GB），Dashboard 的数字会和 README、文档对不上。
    两条 SQL 的 dry run 实测通过（`session_funnel` 2.21 GB、`weekly_cohort_retention` 0.24 GB），但除零是运行期错误，dry run 查不出。
  - 不用 Looker Studio 的「上传文件」入口：仓库里没有记录，表过期后无法重建。上传 Parquet 而不是 CSV，保住日期和数字列的类型。
  - Looker Studio 只连小的结果表，不对原始表写自定义查询（每次刷新都要重扫）—— 和 Looker 的 PDT（持久派生表）是同一个道理。

## 4. 第 ① 页：业务漏斗与留存

- [x] **决定（Andy，2026-10-14）：漏斗的头号数字 = loose 购买率 5.98%（`session_funnel` 的 `loose_pct`），回答「有多少会话最后买了」**。
  - 不用 strict 的 3.99%：它丢掉 467,503 个购买会话（1,405,954 − 938,451，占 33.25%），其中「有浏览、无加购」占 98%（`session_funnel_purchase_decomposition.sql`）。
    strict 回答的是「卡在哪一步」（加购 → 购买 40.45%），不是购买率。
- [x] **决定（Andy，2026-10-14）：留存放两样 —— k = 1 的一个数 + 整张 cohort × k 热力图**。
  - k = 1：9 个 cohort 里 8 个有这一格（k = 8 只有 1 个）；1–2 周和项目「14 天内购买」同一尺度。
  - 热力图：单个数藏住两件事 —— 11-11 那周的对角线（日历效应，11-04 cohort 的 k = 1 是 14.67，被抬高）、11-11 cohort 的 k = 1 只有 7.17（全表最低）。
    `weekly_cohort_retention.sql` 是长表，透视交给 Looker Studio（文件头原本就这么写）。
- [x] **决定（Andy，2026-10-14）：k = 1 的数字 = 合并算 12.78%**（8 个 cohort 的 k = 1 回购人数 ÷ cohort 人数 = 82,548 / 645,800），理由：各 cohort 人数不同。
  - 不用简单平均 13.29%（每个 cohort 一票，146,154 人的 11-11 和 57,783 人的 11-18 同权；11-25 没有 k = 1，不在这 8 个里）；不用最新 cohort 11-18 的 11.36%（被 11-14 起 38.4 小时的埋点故障抬高）。
  - 副作用：人群结构会动它 —— 11-11 cohort 人多、k = 1 只有 7.17，把合并数往下拉；所以旁边放热力图。
  - k 是日历周的差，不是天数：k = 1 的实际间隔是 1–13 天，周末首购的人窗口贴着首购（回购最密的几天），定义上偏高。
    11-11 cohort 76.5% 是周六、周日首购，定义在帮他们，k = 1 仍最低（`weekly_cohort_first_purchase_day.sql`：周一到周四 8.16–9.59，周六 7.12、周日 6.43）。
    热力图加注：「k = calendar weeks since the first-purchase week (Mon–Sun); k = 1 means bought again the following week, 1–13 days later.」
