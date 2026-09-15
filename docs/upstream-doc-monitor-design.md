# upstream-doc-monitor 设计文档

面向维护者的实现说明。用户视角的安装与接入见 [upstream-doc-monitor.md](upstream-doc-monitor.md)。

## 组件

- 检测引擎 `scripts/upstream_doc_monitor.py`：读配置、采集、分类、同步 Issue、产出报告。
- 清单 `.github/upstream-doc-monitor.yaml`：唯一人工维护的输入。
- 触发器 `.github/workflows/upstream-doc-monitor.yml`：定时/手动跑一轮，渲染 Step Summary，上传 artifact，持久化基线。
- 校验器 `.github/workflows/validate-upstream-doc-monitor-config.yml`：PR 改清单时触发的合法性门禁。
- 契约 `schemas/upstream_doc_monitor_result.schema.json`：`result.json` 的结构定义。

## 数据流

```
yaml 清单 ──load_config──▶ 归一化条目[] ──采集器──▶ 每项 sha/状态
                                    │
                    基线 state.json ┤ (actions/cache 跨轮)
                                    ▼
                       事件序列 → Issue 同步 → report.json / result.json
```

## 采集与判变

两类源，判定都是「当前指纹 vs 基线指纹」：

- **repo_file**：GitHub Contents API 取该文件的 git blob SHA（`fetch_repo_file_sha`）。内容不变则 SHA 不变，与提交历史解耦。
- **web_page**：HTTP 条件请求（`fetch_web_page`）。带 `If-None-Match`/`If-Modified-Since`，命中 304 直接复用基线 SHA、不下载正文；200 则对响应体算 SHA-256，并把新的 `etag`/`last_modified` 写回基线供下轮用。

URL 归一（`_parse_source_url`）：先剥 `#fragment`/`?query` 再判类型，`README.md#L1-L5`、`?plain=1` 不会污染 API 路径。条目身份 key 纳入 `repo/branch/path`，避免同名文档跨仓库/分支撞车。

## 错误分类与退出码

核心区分：**上游侧观测失败不影响 job 绿**（本来就是在替用户发现上游把文档改坏了）；**本仓库侧运行故障才标红**。

| 异常 | 归属 | 计入「完成观测」 | job |
| --- | --- | --- | --- |
| `DocNotFound` (404) | 上游 | 是（有效观测） | 绿 |
| `RepoError` / `FetchError` | 上游 | 否 | 绿 |
| `RateLimitError` | 本仓库 | 否 | 红 (exit 2) |
| 配置/基线 `FatalError` | 本仓库 | — | 红 (exit 1) |

- repo_file 收到 404 会再探一次仓库本体：仓库可达判 `DocNotFound`（文档真没了），仓库也不可达判 `RepoError`（多半 url 分支/路径写错）。
- `observed == 0`（无一条目完成观测）→ exit 2，视作环境问题而非「全 404」。限流后剩余 repo_file 直接记 `rate_limited` 跳过，不再打 API。

## 事件同步（Issue 交付）

每条目每轮算出事件序列 `events`（`recovery` / `change`，可含 error 路径单事件），交给 `_sync_events`：

- 无 open Issue → 新建 Issue，首事件全并入正文（处理人只收 1 封通知）；带 `upstream-doc-monitor` 标签，标题 `[<project>] <标签> — <目标>`。
- 已有 open Issue → 按事件追加评论。
- **`synced=False`（create/comment 实际失败）→ 调用方回退 `baseline[key]=prior`**，本轮不推进基线，下一轮重新检测重新同步，宁可重复评论也不静默吞掉变化。这是消除「建 Issue 失败 = 永久丢事件」的关键不变式。
- 单 `recovery` 且无 open Issue → 不为「恢复」单独开 Issue（噪音）。
- 同型错误连续轮次走 `last_error_type` 频控，不重复刷屏。
- 未填 owner 的条目兜底 @ `maintainer`。

## 版本号解析

`fetch_upstream_version` 三级降级，全部失败返回 `null` 且只告警不阻断：`releases/latest` → `releases?per_page=20`（取首条非 draft）→ `tags?per_page=1`。每轮重新查询、只在 run 内按 repo 去重（同仓库多条目只查一次，`version_cache`），不落基线缓存——前端用 `version` 校验信息时效，所以它必须每轮与上游最新对齐。

## result.json 生成

Issue 同步之后才扫 open Issue 关联（`fetch_open_ticket_map`），本轮新建的 Issue 即计入 `pending`。状态优先级：本轮 `error` 或 `sync_failed` > 有未关闭 Issue (`pending`) > 其余 `unchanged`。扫描按 label 过滤收窄，失败/截断时回退基线 `issue_url` 保底，绝不误降级为 `unchanged`。字段契约见用户文档「结果投递到看板」。

## 配置校验：单一规则来源

`--validate-only` 复用运行期同一个 `load_config`，在取 token 前短路退出。PR 门禁跑的正是这条命令，因此不存在「本地合法、跑起来才报错」的口径漂移；`owners` 语义也不会从任何一处复活。

## HTTP 层

`_http_once` 对响应头做小写归一，`x-ratelimit-remaining`/`etag`/`last-modified` 取值不受服务端大小写影响。`http_with_retry` 对 5xx/网络/限流做指数退避（`(2,4,8)`s），耗尽后按限流或瞬断分类抛出。
