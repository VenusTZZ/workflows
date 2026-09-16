# cache-seed：CI runner 共享缓存的 fallback 投递目录

这个目录是 CI runner **找不到替代源时的兜底**。优先级：

```
ModelScope / HuggingFace 直接下  >  本目录 cache-seed/<project>/
```

- **优先**：ModelScope 在 CI 集群可达，比 hf-mirror 更稳。新增模型/数据集时
  先看 ModelScope 是否有镜像（绝大多数主流 repo 都有，如
  `AI-ModelScope/roberta-base`），有就直接 `from modelscope import
  snapshot_download` 在 setup / 运行时拉，不要塞进本目录
- **兜底**：只有 ModelScope 也没有的（如 gated repo、私有模型、上游刚发布
  ModelScope 还没同步的），才走本目录流程：本机代理下 → 分片 → push → CI
  runner 拷贝组装

**不要无脑把所有 hub-id 内容都 push 到本目录**——能 ModelScope 就 ModelScope。
本目录的存在只是为了解决"ModelScope 没有 + hf-mirror 不稳"的小集合。

## 整体流程（仅兜底场景）

```
本机（直连 HF）                       GitHub 仓库                CI runner
─────────────────                     ──────────                ──────────
huggingface_hub 下内容       ──────>  cache-seed/<project>/    cache-seed.yml
（ModelScope 没有的）                 manifest.yaml          ─→ scripts/cache_seed.py
scripts/bundle_cache.py       ──────>  <prefix>/<file>        ─→ 拷贝 → SHARED_CACHE_ROOT
                                     <file>.part-aa/ab/...    → sha256 校验
```

## 单校验

| 时机 | 校验内容 |
|---|---|
| 投递前（staging 阶段） | bundle_cache.py 流式算 sha256，写入 manifest.yaml |
| 投递后（CI 阶段）     | cache_seed.py 拷完后重算 sha256，对照 manifest |

`bundle_cache.py` 流式处理（1MB chunk / 文件），单文件内存峰值 ≈ 95MB。

## 目录约定

| 内容 | runner 共享缓存目标 | 备注 |
|---|---|---|
| `<prefix>/<file>` | `<SHARED_CACHE_ROOT>/<prefix>/<file>` | 直接 cp，拷完校验 |
| `<prefix>/<file>.part-aa/ab/...` | `<SHARED_CACHE_ROOT>/<prefix>/<file>` | cat 拼回，拷完校验 |

`<prefix>` 由 staging 时 `--prefix` 指定；CI 不另设 `extract_to`，路径里直接编码。

## 幂等

- target 已存在且 sha256 匹配 → 跳过
- target 存在但 sha 不匹配 → 删掉重试

## 什么时候 dispatch

- 新增了 ModelScope 也没有的内容
- 上游文件改动（sha 漂移 → 重新 stage 并 seed）
- runner 池扩容（matrix 4 路撒点，未覆盖的机器再 dispatch 一次即可）

## 加速本机的 staging

```bash
# 1. 下载（huggingface_hub；ModelScope 拉得到的话根本走不到这一步）
export HF_HOME=/tmp/hf
huggingface-cli download <repo-id>     # ModelScope 没有的 repo

# 2. 打包（流式算 sha256；> 95MB 自动切分）
python scripts/bundle_cache.py \
    --project peft \
    --src /tmp/hf/hub/models--<repo-id> \
    --prefix hub/models--<repo-id>

# 3. commit & push
git add cache-seed/peft/
git commit -m "peft: seed <repo-id> (ModelScope fallback)"
git push
```

> ⚠️ HF cache 结构里 `snapshots/<sha>/` 是 symlink → `blobs/<sha>`；bundle_cache.py
> 默认跳过 `blobs/` 目录（HF 内部 dedup 用），symlink 自动跟随 → 内容落到
> `snapshots/<sha>/<file>`（普通文件），符合 `from_pretrained` 期望。

## manifest.yaml 格式

```yaml
files:
  - path: hub/models--roberta-base/refs/main          # 相对 SHARED_CACHE_ROOT
    sha256: <hex>
  - path: hub/models--roberta-base/snapshots/<sha>/config.json
    sha256: <hex>
  - path: hub/models--roberta-base/snapshots/<sha>/model.safetensors
    sha256: <hex>
    size: 999999999    # 可选；切分文件的实际大小，仅供人查阅
```

不要手填 sha256 —— 必须由 `bundle_cache.py` 流式算才能与拷完后一致。

## peft 的现状（实测 ModelScope API 后，2026-09-16）

peft examples 所需的 5 个模型 + 4 个数据集里：

- **5 个模型（roberta-base / bert-base-uncased / mt0-small / dinov2-base /
  Qwen2.5-0.5B）+ 2 个数据集（nyu-mll/glue、modelscope/imdb）**：
  ModelScope 都有镜像，setup_example.sh 的 `modelscope.snapshot_download`
  一次性拉到本机再 symlink plant 到 `~/.cache/huggingface/hub/`，runner
  上 `from_pretrained(<hf_id>)` / `load_dataset(<hf_id>, ...)` 命中本地
  缓存，不走网络（注意 imdb 的命名是 `modelscope/imdb` 不是 `imdb`）
- **beans、gtfintechlab/financial_phrasebank_sentences_allagree**：
  ModelScope 没有，**只能走本目录**（cache-seed/peft/manifest.yaml
  已列）——本机代理下 → bundle_cache.py stage → push → runner 解出