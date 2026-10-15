# cosdt-ci-test/workflows

workflows的开发、测试仓。被看护的项目各自有一条 example 流水线和 一条 quick-start 流水线；项目专属的清单、fixture 放在 `projects/<项目名>/`。

已接入 [ms-swift](projects/ms-swift/)（样板）、[llama.cpp](projects/llama.cpp/)、[whisper.cpp](projects/whisper.cpp/) 和 [mooncake](projects/mooncake/)（example 看护跑 Ascend Direct 双卡传输例程，Quick Start 看护跑本仓文档；两条流水线的轮询都还未开）。设计说明见 [docs/guarding-examples.md](docs/guarding-examples.md)。

## 目录

```
projects.yaml                        # 项目注册表
schemas/                             # result.json 等产物的 JSON Schema
scripts/                             # 全项目共用：初始化清单、supported 校验
templates/                           # quick-start 等可复制的 workflow 骨架
docs/guarding-examples.md            # 看护 Examples 设计文档
docs/artifacts.md                    # artifact 命名与读取约定
.github/workflows/<project>-examples.yml
.github/workflows/<project>-quick-start.yml
projects/<project>/                  # 该项目的清单、fixture、专用脚本
```

`projects.yaml` 里的 `name`、`category`（分类）、`support_level`（支持程度）取自[项目表](https://docs.google.com/spreadsheets/d/1GtLB4Zvi_rzGqsH6dWShebk3DvRg6b9NNBeNeeDuNUM/edit?gid=753359062#gid=753359062)的「项目名称」「分类」「支持程度」列。`name` 全小写；`category` 和 `support_level` 保留表里的中文值。

## 接入新项目

1. 读 [docs/guarding-examples.md](docs/guarding-examples.md)，判断该项目处于阶段 A 还是阶段 B。
2. examples 看护写一个调用共享引擎 [examples-template.yml](.github/workflows/examples-template.yml) 的薄触发器（参考 [ray-examples.yml](.github/workflows/ray-examples.yml)，引擎设计见 [docs/examples-guard-engine.md](docs/examples-guard-engine.md)）。触发靠流水线里的 monitor job 定时轮询上游（见设计文档「要求 2」），不需要在上游或其他仓库部署任何东西。
3. 需要看护 Quick Start 文档时，请参考 [templates/project-quick-start.yml](templates/project-quick-start.yml)。
4. 在 `projects.yaml` 注册该项目。
5. 建 `projects/<project>/` 目录。用 `scripts/bootstrap_manifest.py --target-root <checkout> --output projects/<project>/examples_manifest.yaml --supported <path>` 生成清单（扫描目录、扩展名可用 `--scan-root` / `--include-extension` 调整；C++ 目录型 example 加 `--unit directories`），再手工补全 supported 条目的 profile / runner / npu_devices / image / timeout_minutes（必填：example 流水线按这些字段调度 setup 例程、runner、卡、容器镜像和超时；overlay_args 可选；path 不是启动文件时再写 exec）。supported 为空时流水线只做清单比对，不跑 example。
6. 按 [docs/guarding-examples.md](docs/guarding-examples.md)「项目运行脚本契约」提供 `projects/<project>/scripts/` 下的 `setup_example.sh` 和 `run_example.sh`。
7. 按 [docs/artifacts.md](docs/artifacts.md) 确认 artifact 名称和 `result.json` 字段。

`scripts/check_supported_entries.py` 由共享引擎的 `manifest-check` job 调用：校验 supported 条目（path 在目标树上存在、调度字段齐全）并把 supported 列表写成 matrix，缺条目立即判红。它不扫描目标树；「上游新增 example、清单未登记」的发现由每日对账 workflow 承担（设计见 [docs/examples-check-design.md](docs/examples-check-design.md)）。`scripts/bootstrap_manifest.py` 只在接入时离线使用，CI 不调用。

## Artifact

每个 job 无论成败都上传产物。外部机器用 Jobs API 看 conclusion，或下载 artifact 读 `result.json`。字段与命名见 [docs/artifacts.md](docs/artifacts.md)，schema 在 [schemas/](schemas/)。
