# SQL 方言对照（DuckDB ↔ BigQuery）

本项目在本地 DuckDB 上开发，目标方言是 BigQuery：`sql/analysis/` 下的文件约定写 BigQuery 语法，
由 `scripts/run_sql.py` 在 DuckDB 上**原样**执行，不打补丁。本文记录两边**实测**出来的差异，
以及它们在本项目里落在哪。

> **实测环境**：DuckDB 1.5.5（Windows，仓库 `.venv`）；BigQuery 沙盒（`bq` 2.1.38，2026-09-22）。
> 每条都是字面量查询，BigQuery 侧扫描 0 字节。**除「未实测」一节外，本文每一行都在两边跑过。**

---

## 核心：宽严是有方向的，本地全绿只挡得住一个方向

| 方向 | 本地会怎样 | 什么时候发现 |
| --- | --- | --- |
| **BigQuery 宽、DuckDB 严** | DuckDB 报错（例：`SAFE_DIVIDE` 不存在） | **本地就发现** —— 这正是先本地跑的意义 |
| **DuckDB 宽、BigQuery 严** | DuckDB 返回一个像样的结果 | **只有搬上 BigQuery 才发现**，而且 sqlfluff 也拦不住（它查语法，不做名称解析和类型检查） |

第二个方向是本文的重点。下表里「会咬」一列标的都是它。

---

## 两边行为不同（全部实测）

| 写法 | DuckDB 1.5.5 | BigQuery | 会咬 |
| --- | --- | --- | --- |
| `1 / 0` | `inf` | **报错** `division by zero: 1 / 0` | 🔴 BigQuery |
| `0 * 100.0 / 0`（空组的条件计数相除） | **`NaN`**（显示为 `-nan`；`IS NULL` 为 false） | **报错** `division by zero` | 🔴 BigQuery |
| `SAFE_DIVIDE(1, 0)` | **报错**：函数不存在 | `NULL` | DuckDB（本地可见） |
| 同一个 `SELECT` 里引用前面的别名：`SELECT 1 AS a, a + 1 AS b` | `(1, 2)` | **报错** `Unrecognized name: a` | 🔴 BigQuery |
| 同上，但引用发生在窗口里：`x * 2 AS y, SUM(y) OVER ()` | 正常 | **报错** `Unrecognized name: y` | 🔴 BigQuery |
| 同一 CTE 里同名别名定义两次：`WITH t AS (SELECT 1 AS a, 2 AS a) SELECT a FROM t` | **静默取第一个**，返回 `1` | **报错** `Column name a is ambiguous` | 🔴 BigQuery |
| `DATE_DIFF('second', a, b)`（DuckDB 形式，`part` 在前） | `60` | **报错** `A valid date part name is required` | 🔴 BigQuery |
| `TIMESTAMP_DIFF(b, a, SECOND)` / `DATE_DIFF(b, a, SECOND)` / `UNIX_SECONDS(b) - UNIX_SECONDS(a)` | **报错**：函数不存在 / 找不到列 `SECOND` | `60` | DuckDB（本地可见） |
| `ROUND(7, 4)`（整数参数） | `7`，**仍是整数** | `7.0`，FLOAT64 | 类型不同，值相同 |
| 未起别名的 `COUNT(*)` | 列名 `count_star()` | 列名 `f0_` | 两边都难读 —— **所有聚合都起别名** |

⚠️ **时间差没有两边通用的写法。** 上表两行合起来：DuckDB 认的形式 BigQuery 不认，BigQuery 认的三种形式
DuckDB 一个都没有。这是本文唯一一个「写对了也跑不了两边」的差异，见下一节。

⚠️ **零分母在 DuckDB 上不是 `NULL`。** 默认开着 `ieee_floating_point_ops`，所以 `x / 0` 是 `inf`、`0 / 0` 是 `NaN`。
`NaN` 不会被 `IS NULL`、`COUNT(col)` 或 `COALESCE` 当成缺失，排序时排在所有数值之后。
如果要的是 `NULL`，分母写 `NULLIF(b, 0)` —— 两边都返回 `NULL`，这是零分母唯一的通用写法。

---

## 以为不同、实测相同

这几条此前是凭记忆或推断写的，实测后更正：

| 写法 | 两边结果 | 此前的说法 |
| --- | --- | --- |
| `SELECT a, b, FROM ...`（`SELECT` 尾逗号） | 都允许 | BigQuery 侧「未实测」 |
| `COUNT(DISTINCT x) OVER ()` | **都支持** | 「BigQuery 判断不支持」—— **错** |
| 不带 `WHERE` 的 `QUALIFY` | 都允许 | —— |
| `5 / 2` | 都是 `2.5`（整数除法截断是 PostgreSQL 的行为） | —— |
| `MAX(x) OVER (ORDER BY ts)` 遇到同一 `ts` 的并列行 | 都把并列行算进窗口（默认 frame 是 `RANGE`） | —— |
| `SUM(COUNT(*)) OVER ()`（在分组层拿全局总数） | 都支持 | —— |
| `UNPIVOT (v FOR k IN (c1, c2))` | **语法相同**，都能跑；输出的列顺序不同（BigQuery 把值列放前面） | 「语法不同，没核对」 |

---

## 对本项目的影响

### 🔴 7 个文件在 BigQuery 上跑不了（第 4 步的阻塞项）

会话切分那一层用的是 DuckDB 形式的 `DATE_DIFF('second', previous_event_time, event_time)`，
而 BigQuery 不接受字符串形式的 `part`。用到它的文件：

- `cart_abandonment_by_category.sql`
- `category_conversion_rate.sql`
- `data_quality_session_gaps.sql`
- `repurchase_interval_distribution.sql`
- `sessionization_event_count_distribution.sql`
- `sessionization_purchase_rate.sql`
- `sessionization_purchase_segment.sql`

⑤ 的两个新文件也会用到。**CI 是绿的**：sqlfluff（`dialect = bigquery`）只查语法，不查函数签名。

由于没有两边通用的写法，第 4 步之前要选一个处理方式（**未决定**）：

1. `run_sql.py` 在 DuckDB 侧注册一个 `TIMESTAMP_DIFF` 宏，文件统一写 BigQuery 形式 —— 和「不打补丁」的原则冲突，但只补这一个函数
2. 用转译工具（如 sqlglot）在 DuckDB 上执行前把 BigQuery 方言转成 DuckDB 方言
3. 文件保持 DuckDB 形式，上 BigQuery 时再改写 —— 最省事，但违反「`sql/analysis/` 写 BigQuery 语法」的约定

### ④ 的过滤理由成立，但文件头有一句不对

`cart_abandonment_by_category.sql` 在「对」这一层过滤掉没有加购的对，理由是 BigQuery 遇到零分母会报错 ——
**实测成立**。但文件头说 DuckDB 在零分母时「returns NULL」，实测是 `NaN`。结论不变，措辞待改。

### 搬 BigQuery 前的检查清单

以下几类 DuckDB 放行、BigQuery 报错，本地全绿不代表没有：

- [ ] 所有除法：分母能不能是 0？能就用 `NULLIF(b, 0)`，或者在更早的层把零分母过滤掉
- [ ] 同一个 `SELECT` 里有没有引用本层定义的别名（包括窗口函数里面）
- [ ] 同一层有没有两个同名列
- [ ] 时间差的写法（见上一节）

后三类是**绑定期**错误，第 4 步用 `bq query --dry_run` 就能一次性暴露（只做解析和绑定、扫描 0 字节、不耗配额）。
**第一类不行**：零分母是**运行期**错误，实测 `SELECT 1 / 0` 的 dry run 显示「successfully validated」，
只有真跑、而且数据里真出现 0 才会报 —— 所以它只能靠读代码查。

---

## 未实测（PostgreSQL，来自刷 LeetCode 时的报错，本次没有重跑）

- 两个整数相除会**截断**：`15331243 / 109950743` 得 `0`。转换要写在除法之前：`COUNT(...) * 100.0 / COUNT(*)`
- `SELECT` 尾逗号报错
- 没有 `QUALIFY`，要用子查询包一层再 `WHERE`
- 未起别名的 `COUNT(*)` 列名是 `count`
