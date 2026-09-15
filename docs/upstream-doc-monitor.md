# upstream-doc-monitor 使用文档

## 背景

本仓库的快速入门文档跟随上游仓库发布。上游改了文档不会主动通知任何人，没人跟进的话，用户就会照着已经失效的安装步骤踩坑。这个 workflow 用来自动检测上游文档变化，在仓库内创建 Issue 并 @ 处理人，确保文档跟上游保持同步。

## 解决方式

每 6 小时自动检测一次（cron `0 */6 * * *`），也可在 Actions 页面手动触发。检测到内容变化或检测异常时，创建/更新 Issue 并 @ 处理人——Issue 即待办，处理人核对上游变更、更新本仓库文档后关闭 Issue。

## 接入配置

监控清单 `.github/upstream-doc-monitor.yaml` 是唯一需要人工维护的文件。

**完整示例**

```yaml
maintainer: wangwu
projects:
  - project: transformers
    owner: zhangsan
    url: https://github.com/huggingface/transformers/blob/main/README.md

  - project: lm-eval-ascend-doc
    owner: lisi
    url: https://ascend.github.io/docs/sources/lm_evaluation/quick_start.html
```

**配置项**

- `maintainer`：本 workflow 管理人，条目未填 `owner` 时的默认 @ 对象。必填，单个 GitHub 用户名。
- `project`：展示标签，用于 Issue 标题与结果分组。同一项目可重复，对应多个监控条目。
- `owner`：处理人 GitHub 用户名，用于 Issue 中 @ 提及。可选，不填则兜底 @ `maintainer`。
- `url`：监控地址，自动识别类型。仓库内文档填 `https://github.com/用户或组织/仓库/blob/分支/路径`，分支用该仓库的默认分支；其他 http(s) 地址按网页处理。链接尾部的 `#L1-L5`、`?plain=1` 会被自动剥离。

**新增监控条目**

在 `projects:` 下加一条，合入 main 后下一轮检测即生效。新增条目首轮只登记基线、不创建 Issue，从第二轮起正常检测变化。

**更换处理人**

改对应条目的 `owner` 字段，留空即回落到 @ `maintainer`。已 open 的 Issue 不受影响，新事件的新评论会 @ 新处理人。

**配置校验（PR 触发）**

改动清单文件的 PR 会触发一个独立的校验 workflow，检查项通过（绿）才能合入。校验规则与运行时同一套，不会出现本地合法、跑起来才报错的情况。合入前可以在本地先自检这条命令：

```bash
python scripts/upstream_doc_monitor.py --validate-only --config .github/upstream-doc-monitor.yaml
```

配置无误时输出 `config: OK` 并给出条目数。

**配置报错怎么定位**

配置类问题（缺字段、url 非法、条目重复、用户名格式非法）在 PR 阶段就被拦下，校验日志含条目索引 `projects[i]` 与字段名，改到绿为止。url 写对了但指不到内容（上游改名、仓库删除、页面 404）属于运行期问题：该条目记为检测异常，在 Issue 里可见，不影响其余监控项。

## 结果投递到看板

每轮检测把 `result.json` 上传为 Actions artifact，名如 `upstream-doc-monitor-result-<run_id>`。看板用 Artifacts API 列出本 workflow 最新 run 的 artifacts，下载该 zip 读其中的 `result.json`，整表替换本地视图即可，无需增量合并。

**完整示例**

```json
[
  {
    "project": "transformers",
    "doc": "https://github.com/huggingface/transformers/blob/main/README.md",
    "version": "v4.56.0",
    "status": "pending",
    "ticket": "https://github.com/your-org/workflows/issues/12"
  },
  {
    "project": "lm-eval-ascend-doc",
    "doc": "https://ascend.github.io/docs/sources/lm_evaluation/quick_start.html",
    "version": null,
    "status": "unchanged",
    "ticket": null
  }
]
```

**字段**

- `project`：监控条目的展示标签，与配置里的 `project` 一致，同一项目可出现多条。
- `doc`：被监控文档的完整地址。仓库内文档为 blob 链接，外部文档为网页地址。
- `version`：上游仓库的最新版本号。网页类条目与没有版本标记的仓库为 `null`。
- `status`：当前状态，三值。`error` 检测异常（文档 404、仓库不可达、抓取失败）；`pending` 已变化待确认（存在未关闭 Issue）；`unchanged` 未变化（无异常且无未关闭 Issue，含新增条目首轮基线登记）。
- `ticket`：未关闭 Issue 的页面地址，没有则为 `null`。
