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
- 仓库之外（计划同周）：Paidy 业务研究的其余部分（日本 BNPL 市场）、自我介绍中英日、5 分钟项目讲解、至少两轮模拟面试、简历条目。

## 2. 约束

- **BigQuery 沙盒的表 60 天过期**：`pte.events` 2026-09-22 建，**2026-11-21 过期**（`bq show` 实测）；
  数据集默认表过期也是 60 天，Dashboard 用的新表从建表起再算 60 天。⇒ 建表 / 灌数据的脚本进仓库，README 放截图兜底。
