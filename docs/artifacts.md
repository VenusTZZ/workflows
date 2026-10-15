# Artifact 输出规范

每个看护 job 无论成败都要留下机器可读的产物，让外部机器不必打开 GitHub UI 就能判断这次跑了什么、结果如何。设计依据见 [guarding-examples.md](guarding-examples.md) 的「要求 4」。

## 命名

所有 artifact 名称必须以 `<project>-` 开头。`<project>` 是 [projects.yaml](../projects.yaml) 里的 `name`，全小写。

| 种类 | 名称 | 上传时机 |
| --- | --- | --- |
| example 结果 | `<project>-examples-<run_id>-<job_index>` | 每个 `publish-result` matrix job（对应一个 `run-example`），`if: always()` |
| quick-start 结果 | `<project>-quick-start-<run_id>`，一轮测多棵树时用 `<project>-quick-start-<run_id>-<job_index>` | 托管 runner 上的 `publish-result`（或仍在同一 job 里写结果的旧流水线）。NPU 上的测试 job 不上传。无变化时整个 run 在 monitor 后结束，不产生此 artifact |

`run_id` 是 GitHub Actions 的 `github.run_id`。`job_index` 是 matrix 的 `strategy.job-index`（从 0 起）。失败也必须上传，不能只在绿的时候留文件。

## 内容与 schema

### `<project>-examples-<run_id>-<job_index>`

单个文件 `result.json`，符合 [schemas/result.schema.json](../schemas/result.schema.json)。

artifact 刻意只装 result.json，不装运行日志和训练产物，且不由自托管 NPU runner 上传：NPU runner 到 GitHub artifact 存储的链路不可靠（曾出现过 example 跑通、仅因上传 stall 而全线判红的 run），所以 result.json 由跑在 GitHub 托管 runner 上的 `publish-result` job 生成——它通过 Jobs API 读取对应 `run-example` 的 conclusion 再写文件上传。完整训练输出在 Actions 页面 `Run example` 步骤的控制台日志里。清单里还没有 supported 条目时，`run-example` 和 `publish-result` 整体跳过，不产生此 artifact、也不判红。同样，`schedule` 触发但 monitor 未检测到上游变化时，整个 run 在 monitor job 后结束，本 artifact 也不产生。

`result.json` 必填字段：`trigger`、`target_repo`、`target_ref`、`path`、`image`、`job_status`。`job_status` 只能是 `success`、`failure` 或 `cancelled`。允许附加字段。

### `<project>-quick-start-<run_id>` 或 `<project>-quick-start-<run_id>-<job_index>`

至少包含 `result.json`（同一份 [result.schema.json](../schemas/result.schema.json)）。还可以附带 unittest 日志、release API 响应等调试文件。一轮只测一棵树时用前者。一轮测多棵树时用后者，避免两个 job 重名。

llama.cpp 的 quick-start 结果由跑在 GitHub 托管 runner 上的 `publish-result` 生成，不由 NPU runner 上传。它通过 Jobs API 读取对应 `run-quick-start` 的 conclusion 再写文件。`schedule` 触发但 monitor 未检测到变化时，后面的 job 跳过，不产生 artifact。

example 流水线里，每个 `publish-result` job 在上传前用 `check-jsonschema --schemafile schemas/result.schema.json` 校验自己生成的 `result.json`；quick-start 流水线同样在上传前校验。缺文件或不合规即红。

## 外部机器如何读取

两条路径都要能用。把 `{run_id}` 换成一次 Actions run 的数字 id。

### Jobs API

看每个 job 的 `conclusion`（success / failure / cancelled）：

```bash
gh api repos/cosdt-ci-test/workflows/actions/runs/{run_id}/jobs
```

### Artifacts API

列出这次 run 的 artifact，再下载 zip，从中读 `result.json`：

```bash
gh api repos/cosdt-ci-test/workflows/actions/runs/{run_id}/artifacts
gh run download {run_id} --repo cosdt-ci-test/workflows --name <project>-examples-{run_id}-0
```

下载后打开 `result.json`，用 `job_status` 判断该 example 是否跑通，用 `path` / `target_ref` 判断测的是哪条、哪个提交。
