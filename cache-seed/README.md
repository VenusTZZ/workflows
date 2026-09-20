# cache-seed：CI runner 共享缓存的投递目录 + seed spec

这个目录（+ cache-seed workflow）承担 CI runner 共享缓存
（SHARED_CACHE_ROOT，默认 `~/.cache/huggingface`）的**全部**投递，两半都幂等：

1. **ModelScope plant（`ms_seeds.yaml`）**：例程硬编码的主流模型/数据集。
   spec 声明 `ms_id / hf_id / kind / allow_patterns`，workflow 的
   `scripts/ms_seed.py` 从 ModelScope 下载并按 HF hub cache 布局落到共享
   缓存（`refs/main` = 真实 HF sha，例程 `from_pretrained` / `load_dataset`
   全程命中本地，不打 xet）。**各项目 setup 不再下载任何资产**，只装栈 +
   校验在位（peft 的 overlay env 路径从 `refs/main` 解析）。
2. **repo bundle（`manifest.yaml` + 分片文件）**：ModelScope 也没有的
   （gated repo、小众数据集），本机代理下 → 分片 → push → workflow 拷贝组装。

优先级不变：能在 ModelScope 找到镜像的资产进 `ms_seeds.yaml`，不要打成
bundle（绝大多数主流 repo 都有，如 `AI-ModelScope/roberta-base`）。

## 整体流程

```
ModelScope 侧（ms_seeds.yaml，主路径）      GitHub 仓库                CI runner
────────────────────────────              ──────────                ──────────
                                           cache-seed/<project>/    cache-seed.yml
                                           ms_seeds.yaml          ─→ scripts/ms_seed.py
ModelScope（CI 集群直连）    <─────────────────────────────────────  下载 → plant →
                                                                     <root>/hub/…

repo bundle 侧（仅 ModelScope 没有的）      GitHub 仓库                CI runner
────────────────────────────              ──────────                ──────────
本机（代理直连 HF）                        cache-seed/<project>/    cache-seed.yml
huggingface_hub 下内容       ──────>       manifest.yaml          ─→ scripts/cache_seed.py
scripts/bundle_cache.py       ──────>       <prefix>/<file>        ─→ 拷贝 → SHARED_CACHE_ROOT
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

- 新增了 ms_seeds.yaml 条目（新项目接入 / 新资产）
- 新增了 ModelScope 也没有的内容（bundle）
- 上游文件改动（sha 漂移 → 重新 stage 并 seed；ms_seed 侧自动跟随）
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

## peft 的现状（实测 ModelScope API 后，2026-09-16；plant 迁入 workflow 2026-09-17；
## +2026-09-18 adamss_image 三资产）

peft supported 例的 model/dataset 来源，按例分别走哪条路：

| 例 | model 路径来源 | dataset 路径来源 |
|---|---|---|
| sft | overlay `--model_name_or_path ${SFT_MODEL_PATH}`（Qwen0.5B 本地） | overlay fixture `ci_sft_8.jsonl` |
| miss / mica | overlay `--base_model_name_or_path ${SFT_MODEL_PATH}` | 硬编码 `imdb`（**cache-seed**）|
| supertuning | overlay `--base_model ${SFT_MODEL_PATH}` | overlay fixture `ci_supertuning_8.jsonl` |
| beft | 硬编码 `bigscience/mt0-small`（ms_seeds plant） | 硬编码 `gtfintechlab/financial_phrasebank...` → **cache-seed** |
| pvera | 硬编码 `facebook/dinov2-base`（ms_seeds plant） | 硬编码 `beans` → **cache-seed** |
| sequence_classification | overlay `--model_name_or_path ${BERT_BASE_UNCASED_PATH}` | `glue/mrpc`（ms_seeds plant）|
| adamss ×2 | overlay `--model_name_or_path ${ROBERTA_BASE_PATH}` | `glue/mrpc` 或 `glue/cola`（ms_seeds plant）|
| adamss_image | 硬编码 `google/vit-base-patch16-224-in21k` → **cache-seed** | 硬编码 `Multimodal-Fatima/CIFAR10_train` + `CIFAR10_test` → **cache-seed** |
| dreambooth ×5 | overlay `--pretrained_model_name_or_path ${SD_MODEL_PATH}`（peft_dreambooth profile 从 refs/main 解析） | overlay fixture `ci_dummy_image`（2 张 256×256 PNG）|

**ms_seeds.yaml（`cache-seed/peft/ms_seeds.yaml`）装 ModelScope 有的**：
- model ×6：Qwen2.5-0.5B / roberta-base（裸 id，adamss_manual 硬编码）/
  bert-base-uncased / mt0-small / dinov2-base /
  stable-diffusion-v1-5（dreambooth ×5；与 accelerate 的 seed 条目同资产，
  共享缓存卷已 plant，本条目冷缓存兜底）
- dataset ×1：glue 只取 mrpc/cola 子树

2026-09-17 前 plant 活在 peft 的 setup_example.sh（TO_ENV/TO_PLANT_MODEL/
TO_PLANT_DATASET），现统一迁入本 workflow；peft setup 只剩从 `refs/main`
解析 `${SFT_MODEL_PATH}` / `${ROBERTA_BASE_PATH}` / `${BERT_BASE_UNCASED_PATH}`
三个 overlay 路径。

**ModelScope 没有 parquet 数据的 6 个（均走 repo bundle，投递后已移除）**：
- `stanfordnlp/imdb`（~80MB plain_text parquet ×3）— miss / mica 用 `train[:1%]`
- `AI-Lab-Makerere/beans`（~137MB）— pvera 用
- `gtfintechlab/financial_phrasebank_sentences_allagree` / 5768（~196KB）— beft 用
- `google/vit-base-patch16-224-in21k`（safetensors ~330MB）— adamss_image 默认
  `--model_name_or_path`；ModelScope 搜不到同名镜像（`AI-ModelScope/` 命名空间
  无此 repo）
- `Multimodal-Fatima/CIFAR10_train`（~114MB）/ `CIFAR10_test`（~23MB）—
  adamss_image 的 `DATASET_CONFIGS["cifar10"]` 硬编码这两个用户 repo（不是
  `uoft-cs/cifar10`！）；ModelScope 的 `huizyuan/cifar10` 是空 repo、`star07/cifar10`
  是原始 python tar 包，均不可用（run 35222723526 实测 302 cas-bridge 超时）

前 3 个（imdb / beans / financial_phrasebank）恢复入口 `git checkout b260ad8 --
cache-seed/peft`；后 3 个（vit + CIFAR10 ×2，run 35298832674 投递、peft-examples
35299048160 全绿佐证）恢复入口 `git checkout 6985592 -- cache-seed/peft`。删除
bundle 时同步删 manifest.yaml（无 bundle 条目时整文件删除，同 accelerate 做法；
**不要只删文件留 manifest 条目** —— 9cc8282 rebase 曾这样留下 12 条 stale 条目，
让 cache-seed dispatch 对 peft 必然 exit 1，2026-09-18 才清理掉）。

`modelscope/imdb` 有 imdb.py script 但**没 parquet 数据**，所以 imdb 不走 modelscope。

## accelerate 的现状（实测 ModelScope API 后，2026-09-17）

accelerate 的资产源整体切换（解决 2026-09-16 hf-mirror Xet 302 事故）：
9 个可 ModelScope 的资产全部进 `cache-seed/accelerate/ms_seeds.yaml`，
由 workflow 的 `scripts/ms_seed.py` 统一 plant（spec 里有每条的用途注释）：

| 资产 | ModelScope id | 备注 |
|---|---|---|
| bert-base-cased | `AI-ModelScope/bert-base-cased` | 例里硬编码裸 id，plant 到 `models--bert-base-cased` |
| glue mrpc | `nyu-mll/glue` | 同 peft |
| oxford-iiit-pet | `timm/oxford-iiit-pet` | ~790MB parquet，cv 两例 |
| SmolLM-360M | `HuggingFaceTB/SmolLM-360M` | 同名镜像，allow_patterns 跳过 3.9G onnx/ |
| wikitext-2-v1 | `Salesforce/wikitext` | 只取 wikitext-2-v1/* ~7.4MB |
| phi-2 | `microsoft/phi-2` | 同名镜像 |
| SD v1.5 | `AI-ModelScope/stable-diffusion-v1-5` | 例用 fp32+torch_dtype=fp16（无 variant），plant 只取 safetensors ~5.5G |
| mms-tts-eng | `facebook/mms-tts-eng` | 同名镜像 |
| LLaVA-NeXT-Video-7B-hf | `llava-hf/LLaVA-NeXT-Video-7B-hf` | 同名镜像，14G |

infer 组按例拆 profile（`accelerate-infer-phi2/sd/tts/llava`），各例 setup
只校验本例硬编码的模型与数据集在位，不下载。

**已投递后移除（2026-09-17）**：accelerate 的两个 ModelScope 缺口数据集
（均为例里硬编码 `load_dataset` / `snapshot_download`，无 MS 镜像或等价物）
曾打包在本目录并通过 cache-seed workflow（run 35177063140）投递到 runner
共享缓存，投递完成后 bundle 已从仓库移除以减轻每次 CI checkout：

- `svjack/pokemon-blip-captions-en-zh`（~100MB）—
  distributed_speech_generation 用其 `en_text` 列；MS 只有原版
  lambdalabs/pokemon-blip-captions（无 en/zh 列），不可替代
- `malterei/LLaVA-Video-small-swift`（~505MB，204 视频）— llava_next_video
  运行时 `snapshot_download(repo_type="dataset")`，os.walk 全量使用

**新 runner 缺数据时的重建流程**（本机代理下 → push → dispatch）：

```bash
export HF_HOME=/tmp/hf HTTPS_PROXY=http://127.0.0.1:7890
python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
for repo in ("svjack/pokemon-blip-captions-en-zh", "malterei/LLaVA-Video-small-swift"):
    snapshot_download(repo, repo_type="dataset")
PY
python scripts/bundle_cache.py --project accelerate \
  --src /tmp/hf/hub/datasets--svjack--pokemon-blip-captions-en-zh \
  --prefix hub/datasets--svjack--pokemon-blip-captions-en-zh \
  --src /tmp/hf/hub/datasets--malterei--LLaVA-Video-small-swift \
  --prefix hub/datasets--malterei--LLaVA-Video-small-swift
git add cache-seed/accelerate && git commit -m "accelerate: re-seed datasets" && git push
# 再 dispatch cache-seed workflow（projects=accelerate），完成后同样可删
```

历史 bundle 也可从 git 直接恢复：`git checkout 0242ba8 -- cache-seed/accelerate`。

## xtuner 的现状（2026-09-20 数据集迁入）

之前 train_hf.py 的 `--dataset_name_or_path` 用仓内 8 行 fixture + sitecustomize
shim（run_example.sh 里 monkey-patch `datasets.load_dataset` 把单文件路径改成
`load_dataset("json", data_files=...)`）。现改为 seed `tatsu-lab/alpaca`
（HF，~52k 行），load_dataset 原生命中本地缓存，fixture 与 shim 均删。

| 资产 | `ms_id`（ModelScope） | `hf_id`（例里硬编码/overlay 传） | 备注 |
|---|---|---|---|
| alpaca 数据集 | `OmniData/alpaca` | `tatsu-lab/alpaca` | 原 id 镜像 `angelala00/tatsu-lab-alpaca` 把 parquet 放仓库根、缺 `data/` 前缀；OmniData 布局与 HF 一致。其陈旧 `dataset_infos.json`（features 缺 `dtype`，datasets 3.x 解析崩）由 ms_seed.py 统一丢弃 |

注意：xtuner 的模型（Qwen2.5-0.5B）暂仍由 setup 直接 ModelScope 下载（老模式），
未走 seed——与 peft/accelerate 的 `models--Qwen--Qwen2.5-0.5B` 是同一资产，共享
缓存卷里已 plant，若后续要统一可再加条目并让 setup 改从 refs/main 解析
`${LLM_MODEL_PATH}`。