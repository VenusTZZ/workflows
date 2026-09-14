# 看护 Examples

## 总之

一般每个项目都存在 examples。我们的工作是让这些 Examples 被 github workflow 看护。我们开发的workflow 的理想情况是进入上游社区，随上游 CI 触发；在此之前，由本仓（workflows 仓）的流水线**定时轮询上游社区的改动**来触发。核心思路是，在下游仓库准备好 workflow 和 examples 并跑通，然后尽可能地往上游社区推送。

## 阶段划分

1. 阶段 A：在 [https://github.com/cosdt-ci-test/workflows/](https://github.com/cosdt-ci-test/workflows/) 仓内，开发看护 example 用的 workflow。流水线通过轮询监听上游仓库（schedule 定时 + monitor 步骤对比状态，见「要求 2」），`target_repo` 默认直接对准上游仓库；开发调试时用 `workflow_dispatch` 手动指定任意 `target_repo` / `target_ref`。
2. 阶段 B：按上游的接受度递进，一步一步来，每一步都可以停在原地继续用阶段 A 的轮询能力：
  1. 先推 example 本身（如果上游还缺昇腾 example）。
  2. 最理想是看护 workflow 直接合入上游、随上游 CI 触发，workflow 的成功、失败在上游社区可见。**这一步通常需要我们向上游提供 self-hosted runner 和容器镜像**。为上游社区接入 self-hosted runner 参考：[https://ascend-gha-runners.github.io/docs/user-manual-gha-zh/#github-app-runner_1](https://ascend-gha-runners.github.io/docs/user-manual-gha-zh/#github-app-runner_1)。注意这些 runner 全部位于中国大陆、无 docker hub 代理，这点需要和社区沟通。



## 零、前提

上游社区至少存在昇腾支持的一条 example。如果不存在，需要新增一条。example 需要展示软件的基本功能。

动手前先判断你负责的项目属于哪种情况：

- 上游已有健康的昇腾 CI（如 LlamaFactory、veRL、VeOmni）：先确认 example 是否已被该 CI 覆盖。若已覆盖，直接进入阶段 B（在上游补 example 看护）。
- 上游有昇腾 CI 但停摆或不健康（如 DeepSpeed、ONNXRuntime、llama.cpp、SGLang）：进入阶段 A，先把 example 在本仓流水线跑通。
- 上游无昇腾 CI（如 transformers、PEFT、diffusers）：按本文完整走。example 若不存在，先写好并用 `workflow_dispatch` 把流水线指向放着草稿的任意仓库分支调通，作为将来推向上游的内容。未来需要和上游社区沟通接入 self-hosted runner 的事宜。



## 一、workflow 要求

推荐在 [https://github.com/cosdt-ci-test/workflows](https://github.com/cosdt-ci-test/workflows) 下新增目标软件的 github workflow，workflow 的核心要求是：

- 对用昇腾能跑通的 examples，必须在合适的 runner 机器上跑通。
- 触发条件：轮询发现上游 examples 变动、Release、main 有新 commit；也可手动触发。
- 能感知新增的 example。
- Job 的结果能被外部机器通过 job api 或者读 Artifact 文件而获取。

以下是为了满足各个要求的详细步骤。

### 要求 1

1. 选择合适的 runner。在 workflows 仓 Settings → Actions → Runners 查看可用 runner。标签形如 linux-aarch64-a2-N，后缀 N 是该机器可用的昇腾卡数。按 example 需要的卡数选：例如 ms-swift 那条 example 用 tensor_model_parallel_size 2（2 卡张量并行），所以选 linux-aarch64-a2-2、npu_devices 用 "0,1"。选好后写进清单 supported 条目的 runner / npu_devices / timeout_minutes 字段，example 流水线按条目调度，不在 workflow 里硬编码。`npu_devices` 必须是 `0` 或 `0,1` 这种逗号分隔的卡号；manifest-check 会据此派生容器 `--device=/dev/davinciN` 挂载（`device_options`），workflow 不再写死卡数。
2. 选择合适的镜像。注意，**cosdt-ci-test 下的 runners 全部位于中国大陆**，代理没有配置 docker hub，所以无法从 docker hub 拉取镜像。推荐从 Ascend 官方镜像仓拉取，在 [https://www.hiascend.com/developer/ascendhub](https://www.hiascend.com/developer/ascendhub) 中，选择合适的镜像版本，点下载时，即可看到 SWR 地址。选好后写进清单 supported 条目的 image 字段（GitHub 不允许 container.image 引用 env，workflow 从 matrix 取镜像，result.json 同源，两处不会漂移）。
3. 配置安装依赖命令。注意，由于 runners 位于国内，安装时可能存在网络问题，所以需要配置镜像。昇腾包用镜像：[https://repo.huaweicloud.com/ascend/repos/pypi](https://repo.huaweicloud.com/ascend/repos/pypi)。
4. 运行 example。有些 example 很大，例如训练加速的有些项目的 example 会真实训练模型，需要通过参数覆盖的方式控制  training steps，减少 CI 资源占用。



#### 项目运行脚本契约

每个项目在 `projects/<project>/scripts/` 下提供自己的脚本，由 example 流水线的 `run-example` job 调用。仓库不提供通用实现——每个项目的 example 形态不同——但调用契约固定。

`setup_example.sh` 按清单条目的 `profile` 准备环境（安装项目本身和该族 example 的依赖）：

- 位置参数：`$1` 是清单 supported 条目的 `profile`（例如 ms-swift 的 `megatron`）。
- 环境变量（workflow 已设好，脚本直接用）：`TARGET_ROOT`、`GITHUB_WORKSPACE`、`GITHUB_ENV`。
- 未知 `profile` 必须在安装任何东西之前以非 0 退出，并打印已支持的 profile 列表。
- 每个 profile 对应一个函数或子脚本，不要在 workflow YAML 里写项目私货。

`run_example.sh` 跑一条 example：

- 位置参数：`$1` 是 example 相对目标仓根的路径（即清单 supported 条目的 `path`）。
- 环境变量（workflow 已设好，脚本直接用）：
  - `PROJECT_ROOT`：`projects/<project>/` 的绝对路径；
  - `TARGET_ROOT`：目标仓 checkout 的绝对路径；
  - `FIXTURE_DIR`：`projects/<project>/fixtures/` 的绝对路径；
  - `CI_OUTPUT_DIR`：训练/运行输出必须写到这个目录；
  - `ASCEND_RT_VISIBLE_DEVICES`：清单条目的 `npu_devices`；
  - `OVERLAY_ARGS`：清单条目 `overlay_args` 的 JSON 数组（条目没写时为 `[]`）。脚本必须能处理空数组。
  - `EXEC`：清单条目的 `exec`（条目没写时为空）。写了就启动这个相对目标仓根的文件，再拼 `overlay_args`；没写就把 `path` 当脚本跑。
- 退出码即结果：非 0 判红。不比对 loss 等数值。
- 红线：只允许修改 CI 工作区里的目标仓副本（例如给 example 追加 `"$@"` 透传参数），绝不 `git add` / `commit` / `push`，绝不向目标仓远端发起任何写操作。

`overlay_args` 是把 example 压到 CI 规模的参数覆盖，写在清单 supported 条目上，可选。每一项是一个或多个命令行参数，支持 shell 引号、`$VAR` / `${VAR}` 环境变量展开（常用 `${FIXTURE_DIR}`、`${CI_OUTPUT_DIR}`）。把 flag 和值写在同一项里（例如 `--max_length 512`），不要把裸数字单独成项，否则 YAML 会收成整数、schema 校验会失败。参考 `projects/ms-swift/examples_manifest.yaml` 里那条 supported。`path` 可以是文件或目录；目录本身通常不能执行，这时用可选字段 `exec` 写出编译产物（例如 `build/bin/llama-simple`），`overlay_args` 拼在这次启动后面。

注意：example 脚本内部命令级内联的环境变量（如 ms-swift 那条 example 里的 `ASCEND_RT_VISIBLE_DEVICES=0,1`）优先级高于 workflow 导出的值，`overlay_args` 也覆盖不了它。清单 `npu_devices` 必须与 example 内联值一致，改卡时要连 example 一起改。

### 要求 2

触发靠**轮询**，不要求上游（或任何别的仓库）部署任何东西。参考实现是 `.github/workflows/ms-swift-examples.yml` 的 `monitor` / `record-outcome` job，机制与同仓 quick-start 流水线的 monitor 一致：

1. `schedule` 定时唤醒 `monitor` job（跑在免费的 ubuntu-latest 上，几秒即完成）。它轮询上游的三个信号，逐个与上次记录的值比较：
  - **examples**：上游 main 上最后一次触碰 `examples/`（清单 `scan.root`）的 commit SHA；
  - **release**：上游 latest release 的 tag；
  - **commit**：上游 main HEAD 的 SHA。
2. 任一信号变化即触发本次看护。被测 ref 按 examples > release > commit 的优先级取自第一个变化的信号：examples / commit 用对应的 commit SHA，release 用 tag。全部无变化则本次 run 在 monitor 后直接结束，不占 NPU runner。
3. 看护失败会重试：run 结束后 `record-outcome` job 把本次成败回写进 monitor 状态；上次失败的信号即使没有新变化，下个周期也会以 `<信号>-retry` 为由再跑一次，直到成功。
4. `workflow_dispatch` 手动触发不经过 monitor 门（changed 恒为 true），`target_repo` / `target_ref` 从输入取，默认上游仓库的 main。

monitor 状态存在 `.monitor-state/` 目录，用 actions/cache 持久化（条目约 7 天未访问会被清除，monitor 每周期的 restore 会保活；状态丢失只会让下次 run 多跑一轮，无害）。注意：同一项目各流水线的 cache key 前缀必须互不为对方的前缀，因为 restore-keys 按前缀匹配，前缀重叠会串状态。当前约定：legacy examples workflow 用 `<project>-examples-monitor-state-`；引擎化的流水线用严格格式 `<引擎>-monitor-state-<project>_<run_id>`（examples 引擎前缀 `examples-monitor-state-`，quick-start 引擎前缀 `quick-start-monitor-state-`）——`_` 作项目名终止分隔符、项目名禁 `_`，构造性无跨项目碰撞，restore 后另有 matched-key 断言与 `.project` 属主标签校验，校验失败按无缓存处理（告警 + 冷启动，不判红）。

### 要求 3

机制是「清单 + 差集」：

1. 初始化清单。用 `scripts/bootstrap_manifest.py` 扫描目标仓 examples/。默认按文件扫 `.sh` / `.py` / `.yaml`（`--scan-root` / `--include-extension` 可调）。C++ 这类「一条 example 是一个子目录」的项目用 `--unit directories`（可再加 `--marker` / `--max-depth`，默认只要目录里有 `CMakeLists.txt`、只认一层子目录）。扫描规则记在清单的 scan 段，检查脚本按它执行。你确认能在 CI 跑通的 example 列入 supported——supported 条目必须写全 path、profile、runner、npu_devices、image、timeout_minutes（overlay_args 可选；path 不是启动文件时再写 exec）；其余全部自动写入 unsupported。`profile` 命名该 example 的环境准备例程，由项目的 `setup_example.sh` 解释；容器卡挂载由 `npu_devices` 派生，加一条 supported 只改清单（和必要时的 overlay_args / exec / setup 分支），不改 workflow。
2. 每次 CI 比对。创建一个跑在免费的 ubuntu-latest 上的 job，这个 job 把目标仓磁盘上实际存在的 example 与清单求差集：新增的路径（磁盘有、清单无）和失效的路径（清单有、磁盘无）打印到日志，并写进 manifest_check_result.json 随 artifact 上传（外部机器读 `new_paths` / `stale_paths` 字段），一般不使 job 失败——新 example 出现是「提示有待办」。唯一例外：supported 条目的 path 在磁盘上已不存在时立即判红，避免 run-example 在 NPU runner 上装完依赖才发现 example 没了。



### 要求 4

外部获取有两条路径：github Job API 和 Artifact 文件。两条都要满足，各自对 workflow 写法有前提约定。

每个 example job 无论成败（if: always()）写一个 result.json 上传为 Artifact，字段如下：

```json
{
  "trigger": "workflow_dispatch | schedule",
  "target_repo": "目标仓库（例如 modelscope/ms-swift）",
  "target_ref": "被测的分支 / tag / sha",
  "path": "被测example路径，例如 examples/ascend/train/qwen3/qwen3_lora_megatron.sh",
  "image": "所用容器镜像",
  "job_status": "success | failure | cancelled"
}
```



## 契约

项目名称从 [https://docs.google.com/spreadsheets/d/1GtLB4Zvi_rzGqsH6dWShebk3DvRg6b9NNBeNeeDuNUM/edit?gid=753359062#gid=753359062](https://docs.google.com/spreadsheets/d/1GtLB4Zvi_rzGqsH6dWShebk3DvRg6b9NNBeNeeDuNUM/edit?gid=753359062#gid=753359062) 表格的「项目名称」获取，全小写。

workflows 仓目录结构

```
workflows/
├── projects.yaml                     # 项目注册表
├── schemas/
│   └── result.schema.json            # result.json 的 JSON Schema
├── scripts/                          # 全项目共用脚本
├── templates/                        # 可复制的 workflow 骨架
├── .github/workflows/
│   ├── <project>-examples.yml        # example 看护流水线
│   └── <project>-quick-start.yml     # quick start 看护流水线
└── projects/
    └── <project>/                    # 项目专属数据
```

