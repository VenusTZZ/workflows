# mooncake

本目录是 [Mooncake](https://github.com/kvcache-ai/Mooncake) 的看护配套数据，不是 Mooncake 源码。example 流水线在 [.github/workflows/mooncake-examples.yml](../../.github/workflows/mooncake-examples.yml)。Quick Start 流水线在 [.github/workflows/mooncake-quick-start.yml](../../.github/workflows/mooncake-quick-start.yml)。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类：推理加速；支持程度：新兴适配；阶段 A）。

上游默认分支是 `main`。上游有 `ci_ascend.yml`（nightly / 带 `run-e2e-ci` 标签的 E2E）：用私有镜像编 `-DUSE_ASCEND_DIRECT=ON`，`BUILD_EXAMPLES=OFF`，跑的是 HIXL 仓里的 Mooncake Store Python 样例，不跑 Transfer Engine 的 C++ example。本仓阶段 A 看护 Transfer Engine 的 Ascend Direct 例程、需要单独构建树的 HCCL Ascend Transport 例程，以及 Direct 那棵树上能编出来的主机侧 TCP / 拓扑程序。主机侧绿灯不是昇腾传输绿。HCCL 绿灯也不是 Ascend Direct 绿。Quick Start 是另一条线。

## 清单

- `examples_manifest.yaml` 由仓库根目录 `scripts/bootstrap_manifest.py` 扫描目标仓 `mooncake-transfer-engine/example/` 下的 `.cpp` / `.py` / `.cu` 文件。`http-metadata-server/` 是 Go，不在这个扩展名集合里。
- `supported` 有 8 条。`unsupported` 只表示本看护体系当前不跑它们，不是社区支不支持。
  - `transfer_engine_heterogeneous_ascend_perf_initiator.cpp` 要 GPU 对端。
  - `transfer_engine_bench_with_retry.cpp` 在 `CMakeLists.txt` 里没有 target，任何配置都不会被编译。
  - `memory_pool.cpp` 只有 target，传输写死 `installTransport("rdma")`，`while (true) sleep` 不退出，没有 initiator，也没有成功判据。
  - `device_transport_example.cu`、`nccl_device_transport_example.cu`、`nccl_host_transport_example.cpp` 是 CUDA / NCCL。
  - `efa_first_submit_probe.cpp`、`efa_per_transfer_latency_bench.py` 要 AWS EFA 硬件，或两台 AWS 主机做 SSH 编排。
  - `batch_register_bench.py`、`kvcache_prefix_bench.py` 的 `--protocol` 默认是 `efa`，但可以改。真正挡住的是：Python 绑定要另编 mooncake wheel，当前 setup 关掉了这份构建；Python 侧也不设 ACL 设备，`protocol=ascend` 会在昇腾传输初始化时失败。剩下的只有 TCP 主机路径，不单开 job。
- 用卡的条目挂 `linux-aarch64-a2-2`、`npu_devices: '0,1'`。主机型条目同样挂这台 runner 和同一镜像，`npu_devices: '0'` 只是调度约束：这套 CANN 环境在托管 `ubuntu-latest` 上没有。镜像都是 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。
- `profile` 由 `setup_example.sh` 解释。未知 profile 在 apt / cmake 之前非 0 退出，并打印 `ascend-direct ascend-direct-http host-tcp host-oneshot ascend-hccl`。`ascend-direct` / `ascend-direct-http` / `host-tcp` / `host-oneshot` 共用 `build/`，cmake 开 `-DUSE_ASCEND_DIRECT=ON`。`ascend-hccl` 用 `build-ascend-hccl/`，cmake 开 `-DUSE_ASCEND=ON`。上游 `src/transport/CMakeLists.txt` 里这两个开关是 if/else，同一棵树编不出两套传输。清单 `exec` 必须以对应构建目录开头，对不上 setup 立即失败。只编 `basename($EXEC)`。主机型不靠第二套 cmake，靠运行时 `MC_FORCE_TCP`。HCCL 这条线**不要**设 `MC_FORCE_TCP`，否则 init 只装 TCP。
  - `ascend-direct`：`run_example.sh` 先起 target（NPU 0），从日志解析 `listening on <IP>:<port>`，再起 initiator（NPU 1），`--metadata_server=P2PHANDSHAKE`。P2P 会把段名改成这个动态端口，所以 `--segment_id` 用解析结果。initiator 退出后再杀掉 target。HIXL 需要 `/etc/hccn.conf`（setup 缺文件即失败；workflow 从宿主机只读挂入）。日志必须有 `Success to initialize adxl engine`、`Test completed:`，以及 `npu:<logicid>` 或 `mem type:device`（device buffer 登记）。`Failed to install Ascend transport`、`getTransferStatus FAILED` 或 `Sync data transfer timeout` 判红。上游 initiator 在 FAILED/TIMEOUT 时仍会打印 `Test completed:` 并 `return 0`，所以必须扫这些失败串。**绿灯 = 这次 Ascend Direct 写传输在两张 NPU 之间跑完，不是二进制编过了。** `--protocol` 在该 `.cpp` 里声明了但运行时不用，传输后端是编译期 `USE_ASCEND_DIRECT`。不要把 `--mode` / `--segment_id` 写进 `overlay_args`。overlay 只压规模：`--block_iteration=1 --batch_size=2 --block_size=16384`。
  - `ascend-direct-http`：同一个二进制、同一套昇腾断言。先起 `bootstrap_server.py`，两个进程的 `--metadata_server` 改为 `http://127.0.0.1:8080/metadata`。这条不是 P2P，段名保持 target 的 `--local_server_name`（`127.0.0.1:12345`），不要改成日志里的动态监听端口。上游 HTTP 插件和这个 Python 服务在请求成功时都不打日志，运行脚本会打开 `aiohttp.access`，访问日志里必须有 `PUT /metadata`。缺 `aiohttp` 时用 `PIP_EXTRA_INDEX_URL` 安装，装不上是 setup 失败，不是 example 失败。**绿灯 = HTTP metadata 服务和两张 NPU 之间的 Ascend Direct 写传输都发生了。**
  - `host-tcp`：覆盖 `transfer_engine_validator`、`transfer_engine_bench`、`transfer_engine_bench_with_notify`。运行前导出 `MC_FORCE_TCP=1`。原因：`USE_ASCEND_DIRECT` 构建里，`init` 在所有自动发现之前如果看到这个变量，就只装 TCP 并返回；不设的话，后面那段不受 `auto_discovery` 控制的代码会强制 `installTransport("ascend")`，装不上就 init 失败。设了之后，example 里再调一次 `installTransport("tcp")` 会拿到已经装好的对象。metadata 仍用 `P2PHANDSHAKE`，`--segment_id` 用 target 日志里的动态端口。validator 要有 `Data validation passed` 和 `Test completed:`。bench 与 bench_with_notify 要有 `Test completed:`，且日志里不能有独立单词 `FAILED`。当前上游 `transfer_engine_bench_with_notify.cpp` 把 `initiatorWorker` 放在主线程里同步调用，`running` 要等这个函数返回才变成 false，进程不会自己结束。运行脚本最多等 180 秒，超时判红，不改上游源码。**这三条的绿灯不是昇腾传输绿。** 它们证明 Transfer Engine 内核和 TCP 传输能在 aarch64 + CANN 镜像上编过、跑通。不要给它们加 `npu:` 一类设备锚点。
  - `host-oneshot`：单进程跑 `show_link --discover_only=true --json=true`。退出码 0，且 stdout 能被 `json.loads` 解析、顶层有 `local_nics`。必须带 `--discover_only`，否则它会装 rdma 并连 etcd。**绿灯不是昇腾传输绿**，只证明本机拓扑发现在这棵构建上能跑完。
  - `ascend-hccl`：覆盖 `transfer_engine_ascend_one_sided` 和 `transfer_engine_ascend_perf`。构建树与 Direct 互斥。CANN 9.1 公开的 `include/hccl` 不够编这条线，setup 把内部头目录加进 `CPATH`（`aicpu_kfc/pub_inc` 和 `pkg_inc`），并加 `-fpermissive`：上游把 `SalGetBareTgid` 的参数写成 `uint32_t*`，当前 toolkit 头文件是 `s32*`。缺 `adapter_hccp_common.h` 或装不上 `mpich` / `libmpich-dev` 是 setup 失败，不是 example 失败。装 mpich 前若已有 openmpi 会打印警告，上游文档点名这两套会冲突。运行时要求 `ASCEND_RT_VISIBLE_DEVICES` 至少两张卡。`aclrtSetDevice` 用逻辑号 0 和 1。物理号优先读 `npu-smi info -m` 里 Chip Logic ID 0/1 对应的 NPU ID，读不到才退回清单里的 `npu_devices` 值。容器只挂部分卡时，逻辑号和物理号会不一样，不能把 `npu_devices` 同时当成两种 ID。导出 `ASCEND_TRANSPORT_PRINT=1`。metadata 用 `P2PHANDSHAKE`，`--segment_id` 用 target 日志里的动态端口。不要把 `--mode` / `--segment_id` / `--metadata_server` / `--device_logicid` / `--device_phyid` 写进 overlay。overlay 只压规模：`--batch_size=2 --block_size=2097152`，perf 再加 `--block_iteration=1`。`block_size` 用 2 MiB 起步，因为上游文档要求注册内存 2 MB 对齐。initiator 退出码非 0 透传；扫 `Failed to install Ascend transport`、`getTransferStatus FAILED`、`Sync data transfer timeout`、`Hccl transport failed`、`nicServerSocket_ Listen failed`。必须有 batch 锚点行 `local devicePhyId` / `target devicePhyId` 且两端不同。one_sided 还要 `The First Time Send OK`、`The Second Time Send OK`、`Test completed:`；perf 要 `Test completed:`。上游 initiator 在 FAILED/TIMEOUT 时仍会打印 `Test completed:` 并 `return 0`，所以必须扫失败串。**绿灯 = 这次 HCCL Ascend Transport 写传输在两张 NPU 之间跑完，不是二进制编过了。** 上游文档已把该后端标为计划废弃并建议改用 Ascend Direct。上游哪天删掉这两个文件，清单差集检查会直接红，这是预期中的诚实红，不是看护噪音。HCCL 还要能在设备网卡上 listen；`hccn.conf` 里的 `address_<phyid>` 对不上真实 NPU NIC 时，会出现 `nicServerSocket_ Listen failed` 并 abort，同样记诚实红，不改回 unsupported。
  - 二进制默认 `local_server_name` 是实验室 IP 或 etcd 地址。脚本强制 `127.0.0.1` 和 `P2PHANDSHAKE` / HTTP URL。不要把 `--mode` / `--segment_id` / `--metadata_server` 写进 `overlay_args`。
- 编译依赖用 `apt-get` 装 glog / gflags / ibverbs 等；镜像已有 cmake / g++，不再为编译器 `apt-get`。不改 `/etc/apt/sources.list`。`extern/pybind11` 在 setup 里 `submodule update`，失败则按目标仓 gitlink SHA 从 `ghfast.top` 拉同一份 commit，对不上就失败。只有 `ascend-hccl` 额外装 `mpich` 和 `libmpich-dev`。

重新生成清单会**整文件覆盖** `--output` 指向的 yaml。生成器不会合并已填好的 `profile` / `exec` / `overlay_args`。

```bash
python3 scripts/bootstrap_manifest.py \
  --target-root /path/to/Mooncake \
  --output projects/mooncake/examples_manifest.yaml \
  --scan-root mooncake-transfer-engine/example \
  --include-extension .cpp \
  --include-extension .py \
  --include-extension .cu \
  --supported mooncake-transfer-engine/example/transfer_engine_ascend_direct_perf.cpp \
  --supported mooncake-transfer-engine/example/http-metadata-server-python/bootstrap_server.py \
  --supported mooncake-transfer-engine/example/transfer_engine_validator.cpp \
  --supported mooncake-transfer-engine/example/transfer_engine_bench.cpp \
  --supported mooncake-transfer-engine/example/transfer_engine_bench_with_notify.cpp \
  --supported mooncake-transfer-engine/example/show_link.cpp \
  --supported mooncake-transfer-engine/example/transfer_engine_ascend_one_sided.cpp \
  --supported mooncake-transfer-engine/example/transfer_engine_ascend_perf.cpp
```

## 触发

`mooncake-examples.yml` 有两种入口。`monitor` job 跑在 `ubuntu-latest`，不占 NPU。

- `schedule`：cron 写在文件里但是注释掉的。接入阶段保持注释，不要打开。
- `workflow_dispatch`：手动触发。默认 `force=false`，和定时走同一套监控、同一份 cache。只有 `force=true` 才跳过监控门、必跑，并且不读不写 monitor cache。`target_repo` / `target_ref` 只在 `force=true` 时有意义。

两个监控信号都跑，是「或」，互不跳过、没有优先级：

1. 清单 `supported` 各 `path` 在上游 `main` 上的文件内容哈希（Contents API 的 blob SHA；目录会递归到文件。404 记成 `MISSING`）。本信号亮了，测的是这一轮解析到的 `main` commit SHA。
2. `/releases/latest` 的 release **id**（数字，不是 tag 字符串）。本信号亮了，测的是该 release tag 当前指到的 commit SHA。

`force=true` 时 `targets` 只有一项，`reason=manual`。

不做失败重试。NPU job 不上传 artifact。`result.json` 由托管 runner 上的 `publish-result` 按 job 名回看 conclusion 后上传。

## Quick Start 看护

文档在 `docs/Quick-start-Ascend.md`。流水线是 `.github/workflows/mooncake-quick-start.yml`。文档方言见 [docs/markdown_doc_test_label.md](../../docs/markdown_doc_test_label.md)。无标签的 `shell` 块给用户复制，看护跳过。

### 看护范围

- **看护**：第 3 节安装编译依赖；第 4 节克隆并按 `-DUSE_ASCEND_DIRECT=ON` 编译 `transfer_engine_ascend_direct_perf`；第 5 节双进程写传输（target NPU 0 + initiator NPU 1，`P2PHANDSHAKE`）。
- **不看护**（无标签块）：第 1 节加载 CANN 环境（由测试 `prepare_environment` 合并进 `os.environ`，与文档同一份 `set_env.sh`）；第 2 节 `npu-smi info`（设备表数值每次不同）；`/etc/hccn.conf` 由驱动/宿主机提供，文档只说明需要它。
- **本文也不覆盖**：`pip install mooncake-transfer-engine-npu`、vLLM / SGLang 接入、Mooncake Store、已弃用的 HCCL Ascend Transport、异构 GPU+NPU、UBSHMEM。

`schedule` 保持注释。薄触发器 `doc_url` 走 Contents API + `ref=${{ github.sha }}`。
