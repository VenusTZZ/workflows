# whisper.cpp

本目录是 [whisper.cpp](https://github.com/ggml-org/whisper.cpp) 的看护配套数据，不是 whisper.cpp 源码。example 流水线在 [.github/workflows/whisper.cpp-examples.yml](../../.github/workflows/whisper.cpp-examples.yml)。Quick Start 流水线在 [.github/workflows/whisper.cpp-quick-start.yml](../../.github/workflows/whisper.cpp-quick-start.yml)。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类：推理加速；支持程度：新兴适配；阶段 A）。

上游默认分支是 `master`（不是 `main`）。上游有 CANN 后端（`cmake -DGGML_CANN=on`），但没有健康的昇腾 CI。本仓先走阶段 A：在本仓流水线把 example 跑通。

## 清单

- `examples_manifest.yaml` 由仓库根目录 `scripts/bootstrap_manifest.py` 以 files 单位扫描目标仓 `examples/` 下的 `.cpp` / `.sh` / `.py`：对账单位 = 每个源文件（递归不限层），目录条目（`cli`、`bench`…）作为前缀毯子覆盖其内部源文件。顶层共享 helper（`common*.cpp`、`ffmpeg-transcode.cpp`、`grammar-parser.cpp`）不是 example，登记在 `scan.exclude`。`whisper.android` / `whisper.android.java` / `whisper.nvim` / `whisper.objc` / `whisper.swiftui` 是 Java/Lua/ObjC/Swift 移植，无 `.cpp/.sh/.py` 源，在对账中恒为 stale 提示（扫描视野外，不判红）。
- `supported` 覆盖用户在昇腾服务器上会用到的非交互工具：`cli`、`bench`、`quantize`、`parakeet-quantize`、`parakeet-cli`、`test-cmake`、`vad-speech-segments`、`addon.node`、`server`。`unsupported` 只表示本看护体系当前不跑它们，不是社区支不支持。
  - `examples/bench.wasm`、`examples/command.wasm`、`examples/stream.wasm`、`examples/whisper.wasm` 是浏览器 / WebAssembly 移植。
  - `examples/server.py` 是给 WASM 静态页用的 `serve_forever` HTTP 服务器。真正的转写 HTTP 服务是 `examples/server`。
  - `examples/python` 调用已废弃的 `./main`，仓内测试音频也不在。
  - `examples/generate-karaoke.sh` 要麦克风、sox 和 ffplay。
  - `examples/livestream.sh`、`examples/twitch.sh` 拉外部直播并且不退出。
  - `examples/yt-wsp.sh` 依赖 YouTube / yt-dlp / ffmpeg，下载体积和时间无上界。
  - `examples/command`、`examples/stream` 要麦克风和 SDL2，交互式。
  - `examples/deprecation-warning` 是改名垫片，总会失败退出。
  - `examples/lsp` 是长驻 language server，还依赖 SDL2。
  - `examples/sycl` 要 Intel oneAPI。
  - `examples/talk-llama` 还要 llama.cpp、SDL2 和麦克风。
  - `examples/wchess` 是语音下棋，SDL2 / 交互。
  - `examples/whisper.android`、`examples/whisper.android.java` 是 Android SDK / NDK 应用。
  - `examples/whisper.nvim` 要 Neovim 和麦克风。
  - `examples/whisper.objc`、`examples/whisper.swiftui` 是 Xcode / iOS / macOS 应用。
- 每条 `supported` 都挂 `linux-aarch64-a2-1`、`npu_devices: '0'`、镜像 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。
- `profile` 由 `setup_example.sh` 的 `setup_*` 解释。未知 profile 在 cmake / 下载 / 装 Node 之前非 0 退出，并打印已支持列表：`cann`、`host`、`cmake-pkg`、`vad`、`node`、`parakeet`。
  - `cann`（`cli` / `bench` / `server`）：`cmake -DGGML_CANN=on`，只编 `basename($EXEC)`。跑完日志必须有 `CANN[0-9]`。**绿灯 = 这条工具在 NPU 上做了转写或 bench，不是 CPU 跑通。** `server` 还要成功打完一次 `POST /inference`，响应里必须有 JFK 样例的已知句子（`ask not what your country`）。只认 `"text"` 会把 `"text":""` 的 HTTP 200 判绿。只打 `/health` 或启动后立刻杀也会假绿。`whisper-server` 绑定后不退出；`run_example.sh` 后台拉起、curl、再 `SIGTERM`。`EXIT` trap 保证 curl 失败也不会留下后台进程。不要传 `--convert`（要 ffmpeg）。
  - `parakeet`（`parakeet-cli`）：同样 `cmake -DGGML_CANN=on`，只编 `parakeet-cli`。输入是已为 `parakeet-quantize` 缓存的同一份 `ggml-parakeet-tdt-0.6b-v3-f16.bin`，不另下权重，也不先转 q8_0。跑完日志必须有 `CANN[0-9]`，并且必须出现 JFK 样例的已知句子（`ask not what your country` 或 `fellow Americans`）。只认 CANN 会把「模型加载成功、音频没读到、进程仍 exit 0」判绿。**绿灯 = Parakeet 在 NPU 上完成非交互转写。**
  - `host`（`quantize` / `parakeet-quantize`）：同样按 CANN 编出转换器，但转换本身是 CPU。**绿灯 = 写出完整 ggml 权重，不是昇腾推理绿。** `quantize` 输入是已缓存的 `ggml-tiny.en.bin`（F16 可以，`ggml_common_quantize_0` 会先转 F32）。`parakeet-quantize` 输入是 `ggml-parakeet-tdt-0.6b-v3-f16.bin`（约 1.17 GiB），禁止用仓内 `for-tests-ggml-parakeet-*.bin` 桩。三个位置参数，没有 `--device`。
  - `cmake-pkg`（`test-cmake`）：按用户脚本形状 `install` + `find_package(whisper)`，安装带 `-DGGML_CANN=on`。二进制只打印 `whisper_version()`。**绿灯 = 安装包能被下游 cmake 找到并链上，不是推理绿。**
  - `vad`（`vad-speech-segments`）：编 `whisper-vad-speech-segments`，用官方 `ggml-silero-v6.2.0.bin` 切 `samples/jfk.wav`。库把 VAD GPU **强制关掉**（`src/whisper.cpp` 里 `use_gpu = false`，`-ug` 也改不了算力）。**绿灯 = CPU VAD 切出至少一段，不是昇腾推理绿。** 不要传 `-ug` / `--no-prints`。未知 flag 会 `exit(0)`。
  - `node`（`addon.node`）：默认 cmake **不编**这个目录，只有 `cmake-js` 才编。镜像没有 Node 时从华为云镜像拉官方 `linux-arm64` tarball，`npm` 走华为云 registry。cmake-js 7.x 会无视 `--dist-url`，头文件下载读 `NVM_NODEJS_ORG_MIRROR`（设成 `https://mirrors.huaweicloud.com/nodejs`），不要打 `nodejs.org`。`~/.cmake-js/node-arm64/v20.18.2/include/node/node.h` 存在时 cmake-js 完全不联网；setup 把 `$NODE_DEST/include` 同步到该路径，并持久化到 `/root/.cache/cosdt-ci-test/whisper.cpp/cmake-js/`。Node 目录写进 `$GITHUB_PATH`（Actions 扩展后续 step 的 PATH 的文档方式）；coder 没有 Actions，所以同时写 `WHISPER_CI_NODE_BIN` 到 `$GITHUB_ENV`，`run_example.sh` 会 prepend。不要把整行 `PATH=...` 写进 `$GITHUB_ENV`。工作副本把 `index.js` 的 `no_prints: true` 改成 `false`，否则 CANN 日志被掐掉。overlay 只用 JS 认识的 `--key=value`：`--model` / `--fname_inp` / `--language`。不要传 `--device`（会被静默忽略）。不要传 `--use_gpu=true` 这种字符串（C++ `IsBoolean()` 失败，解析不到）。Worker 丢掉推理返回值，失败也常 `exit 0`，所以必须看到 JFK 转写文本，并且日志有 `CANN[0-9]`。**绿灯 = Node 绑定在 NPU 上转写出文本。** coder 上当前是诚实红：`run_with_progress` 一开始就 `free(): invalid pointer` abort（`use_gpu` true / false 都会）。已经能加载到 CANN0，但转写完不成。不修上游、不把这条藏进 `unsupported`。
  - 模型缓存：`/root/.cache/cosdt-ci-test/whisper.cpp/`。`tiny.en` / Silero / Parakeet 都放这里（`tiny.en` 被 5 个 job 共用，不要下到 per-job workspace）。`flock` 包住整个 fetch，先写 `.part` 再 `mv`。复用条件是**精确字节数 + SHA-256 + 小端 ggml 头**（磁盘前 4 字节 `lmgg`）。只看「≥ 下限 + magic」会把截断文件当成命中，后面模型加载失败会伪装成 example 红。`curl -C -` 续传失败，或续传「成功」但校验对不上（包括 HTTP 416 错误页被写成 `.part`），都会删掉 `.part` 并在同一次 setup 里从零再下；失败不留下确定不可用的 `.part`。禁止 `models/for-tests-ggml-*.bin`（约 587 KiB 无权重桩）。
  - Node 缓存同一目录。tarball 钉死官方 `v20.18.2` linux-arm64 的字节数和 SHA-256；`flock` 包住下载和解压。校验或解压失败会删坏包，下次可以自愈。
  - 钉死的对象（coder 上对过转写 / 切段 / 量化）：`ggml-tiny.en.bin` 77704715 字节 `921e4cf8…20b1f`；`ggml-silero-v6.2.0.bin` 885098 字节 `2aa269b7…fb6987`；`ggml-parakeet-tdt-0.6b-v3-f16.bin` 1255897319 字节 `833bffc9…2b426f`；`node-v20.18.2-linux-arm64.tar.xz` 24896668 字节 `5c1437aa…efb44`。
  - 三个模型的主源是 ModelScope（字节与 Hugging Face 一致，sha256 钉死值不变），失败再试 `hf-mirror.com`。coder 上 `huggingface.co` 会 443 超时；`hf-mirror.com` 的 `/resolve/` 会 308 到 huggingface.co 再 302 到境外 CDN，时通时断，所以不能当主源。下载失败算 setup 失败（看护噪音），不会换备用模型。
  - ModelScope 源：`ggml-tiny.en.bin` → `cjc1887415157/whisper.cpp`；`ggml-silero-v6.2.0.bin` → `ggml-org/whisper-vad`；`ggml-parakeet-tdt-0.6b-v3-f16.bin` → `ggml-org/parakeet-GGUF`。路径都是 `resolve/master/<filename>`。
- `overlay_args` 必须是该二进制 / JS 认识的真实 flag。`whisper-cli` / `whisper-bench` / `whisper-vad-speech-segments` 遇到未知 flag 会打印 usage 然后 `exit(0)`。不要抄 llama.cpp 的 `-ngl`。
- `run_example.sh` 把 overlay JSON 展开后按 `path` 跑。`cann` / `parakeet` / `node` 在日志里找 `whisper_backend_init_gpu: using CANN[0-9] backend` 或 `CANN[0-9]`；`parakeet-cli` 和 `addon.node` / `server` 还要看到 JFK 转写文本。`host` / `cmake-pkg` / `vad` 跳过 CANN 这条，改验输出文件 / 版本行 / 语音段。ggml 在卡没挂上时会静默退回 CPU 且进程仍退出 0。
- 编译工具用镜像自带的 `cmake` / `g++` / `make` / `git` / `curl`，不 `apt-get`，不改 `/etc/apt/sources.list`。Node 只放缓存目录，不污染镜像全局。

重新生成清单会**整文件覆盖** `--output` 指向的 yaml。生成器不会合并已填好的 `profile` / `exec` / `overlay_args`，也不会保留 `scan.exclude` 段与 unsupported 注释；不要用这条命令直接覆盖手头已经调度好的清单，除非你准备把它们手工填回去。

```bash
python3 scripts/bootstrap_manifest.py \
  --target-root /path/to/whisper.cpp \
  --output projects/whisper.cpp/examples_manifest.yaml \
  --include-extension .cpp \
  --include-extension .sh \
  --include-extension .py
```

## 触发

`whisper.cpp-examples.yml` 有两种入口。`monitor` job 跑在 `ubuntu-latest`，不占 NPU。

- `schedule`：cron 写在文件里但是注释掉的。接入阶段保持注释，不要打开。
- `workflow_dispatch`：手动触发。默认 `force=false`，和定时走同一套监控、同一份 cache。只有 `force=true` 才跳过监控门、必跑，并且不读不写 monitor cache。`target_repo` / `target_ref` 只在 `force=true` 时有意义。

两个监控信号都跑，是「或」，互不跳过、没有优先级：

1. 清单 `supported` 各 `path` 在上游 `master` 上的文件内容哈希（Contents API 的 blob SHA；目录会递归到文件。404 记成 `MISSING`，哈希会变）。本信号亮了，测的是这一轮解析到的 `master` commit SHA。
2. `/releases/latest` 的 release **id**（数字，不是 tag 字符串）。本信号亮了，测的是该 release tag 当前指到的 commit SHA。

谁亮了就测谁的树：

- 只有 supported 亮：测 `MASTER_SHA`，`reason=supported`。
- 只有 release 亮：测 `RELEASE_SHA`，`reason=release`。
- 两个都亮且两个 SHA 不同：跑两份，`reason` 分别是 `supported` 和 `release`。
- 两个都亮且 SHA 相同：只跑一份，`reason=supported,release`。
- 都没亮：`targets` 为空，后面的 job 跳过，不写 `result.json`，不产包。

`force=true` 时 `targets` 只有一项，`reason=manual`，repo/ref 用输入解析出的 commit SHA。

不做失败重试。两个 monitor 步骤只读旧 cache、算出当前值和 `changed`，不在中途改写 `.monitor-state`。`Decide targets` 成功之后才一次性写入本轮的 hash 和 release id，然后 `cache/save`。任一步失败都不保存候选状态，避免「信号变了、decide 没跑、下次却当成已看过」。后面 NPU 红了也不为同一对值再跑。要再跑：等信号再变，或 `force=true`。

## Quick Start 看护

文档在 `docs/Quick-start-Ascend.md`。流水线是 `.github/workflows/whisper.cpp-quick-start.yml`。文档方言与 ms-swift 相同：围栏 info 行用 `#test` / `#test-setup` / `#test-result`（契约见 [docs/markdown_doc_test_label.md](../../docs/markdown_doc_test_label.md)）。无标签的 `shell` 块给用户复制，看护跳过。

触发契约和上面的 example 看护相同，只是信号 A 换成这篇文档的 sha256。

- `schedule`：cron 是 `45 */6 * * *`（避开 ms-swift 的 `0 */6`、llama.cpp Quick Start 的 `15 */6`、whisper.cpp examples 的 `30 */6`）。**接入阶段保持注释**，第一次 `force=true` 跑绿后再打开。
- `workflow_dispatch`：手动触发。默认 `force=false`，和定时走同一套监控、同一份 cache。只有 `force=true` 才跳过监控门、必跑，并且不读不写 monitor cache。`target_repo` / `target_ref` 只在 `force=true` 时有意义。

两个监控信号都跑，是「或」，互不跳过、没有优先级：

1. 本仓这篇 Quick Start 文档的 sha256。`MONITORED_DOC_URL` 走 GitHub Contents API（`ref` 是这次 run 的 `github.sha`），不用 `raw.githubusercontent.com`：NPU runner 的出口到不了那个域名。monitor 和测试拉同一 URL。本信号亮了，测的是这一轮解析到的上游 `master` commit SHA。
2. `/releases/latest` 的 release **id**（数字，不是 tag 字符串）。本信号亮了，测的是该 release tag 当前指到的 commit SHA。

谁亮了就测谁的树：

- 只有 doc 亮：测 `MASTER_SHA`，`reason=doc`。
- 只有 release 亮：测 `RELEASE_SHA`，`reason=release`。
- 两个都亮且两个 SHA 不同：跑两份，`reason` 分别是 `doc` 和 `release`。
- 两个都亮且 SHA 相同：只跑一份，`reason=doc,release`。
- 都没亮：`targets` 为空，后面的 job 跳过，不写 `result.json`。

`force=true` 时 `targets` 只有一项，`reason=manual`，repo/ref 用输入解析出的 commit SHA。

不做失败重试。两个 monitor 步骤只读旧 cache、算出当前值和 `changed`，不在中途改写 `.monitor-state`。`Decide targets` 成功之后才一次性写入本轮的 hash 和 release id，然后 `cache/save`。任一步失败都不保存候选状态。后面 NPU 红了也不为同一对值再跑。要再跑：等信号再变，或 `force=true`。

NPU job 不上传 artifact。`result.json` 由托管 runner 上的 `publish-result` 按 job 名回看 conclusion 后上传。
