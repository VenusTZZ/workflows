# 快速开始：在昇腾 NPU 上运行 MiniOneRec 生成式推荐模型

> **阅读本文前**，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。本文聚焦**第一次跑通**：在单卡 NPU 上完成 MiniOneRec 的 SID 监督微调（SFT）冒烟训练。

[MiniOneRec](https://github.com/AkaliKong/MiniOneRec) 是一个全开源的**生成式推荐**框架，覆盖 SID 构建、SFT 与推荐导向 RL 全流程。本文只验证 SFT 主路径（RL 阶段的 GRPO 依赖 vLLM/CUDA 或低吞吐的 transformers 原生生成，暂不纳入冒烟范围）。

---

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**，需支持 bf16；910A 不支持 bf16 不在验证范围）。本文示例为**单卡**。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit + 驱动固件已安装并可 `source set_env.sh`（验证基线：CANN 9.1.0） |
| Python | 3.10+（验证基线：3.12） |
| 编译工具 | git |

---

## 1. 加载 CANN 环境

新开终端后 CANN 变量不会自动生效。

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
```

---

## 2. 检查环境是否就绪

```shell #test-setup
source /usr/local/Ascend/ascend-toolkit/set_env.sh
test -n "$ASCEND_HOME_PATH"
command -v npu-smi
npu-smi info
python --version
```

**预期**：`ASCEND_HOME_PATH` 非空；`npu-smi` 打印设备列表且 Health 为 OK。

---

## 3. 克隆 MiniOneRec

将 `<UPSTREAM_REF>` 换成目标**分支、tag 或 commit**（上游默认分支为 `main`）。

```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```

```shell #test-setup id="clone" load="upstream_ref>>UPSTREAM_REF"
git clone https://github.com/AkaliKong/MiniOneRec.git
cd MiniOneRec
git checkout <UPSTREAM_REF>
```

---

## 4. 安装 NPU 依赖

**不要**执行 `pip install -r requirements.txt`：其中 `torchrec/fbgemm_gpu/torchsnapshot` 是 CUDA 专属包、`bitsandbytes` 在 aarch64 上 wheel 可用性不确定（本仓库 xtuner 守门已验证 0.45.0 无 aarch64 wheel）、全部 `nvidia-*-cu11/12` 为 CUDA 运行库。SFT 冒烟只需要下列包。

torch 栈与 CANN 9.1.0 配套（torch 2.9.0 + torch_npu 2.9.0.post2）：

```shell #test-setup id="install"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pip install torch==2.9.0 torch-npu==2.9.0.post2
pip install modelscope transformers==4.57.1 accelerate==1.10.1 datasets==4.2.0 \
    tokenizers==0.22.1 fire==0.7.1 pandas==2.2.2 tqdm
```

---

## 5. 打 NPU 适配补丁

两个文件（`sft.py` 训练入口、`evaluate.py` 评估入口）做同样的最小改动：(1) 顶部 `import bitsandbytes as bnb` 从未实际使用，剥掉以消除 aarch64 wheel 风险；(2) 注入 `torch_npu` 的 `transfer_to_npu` 转换层，把 `torch.cuda.*`、`device_map` 等透明映射到 NPU。

```shell #test id="patch"
cd MiniOneRec
python - <<'EOF'
patch = "import torch_npu\nfrom torch_npu.contrib import transfer_to_npu\n"
for name in ("sft.py", "evaluate.py"):
    s = open(name).read()
    if "transfer_to_npu" not in s:
        s = s.replace("import bitsandbytes as bnb\n", "")
        s = s.replace("import torch\n", "import torch\n" + patch, 1)
        open(name, "w").write(s)
print("patched")
EOF
grep -c "transfer_to_npu" sft.py evaluate.py
# grep -c exits 1 on zero matches; the pipe keeps the block green while
# still printing the count for the result assertion below.
grep -c "bitsandbytes" sft.py evaluate.py || true
```

输出结果如下（多文件 `grep -c` 逐文件输出 `文件名:匹配行数`；`transfer_to_npu` 只出现在 `from torch_npu.contrib import transfer_to_npu` 一行——`import torch_npu` 不含该子串，每个文件匹配 1 行；`bitsandbytes` 出现 0 次）：

```shell #test-result id="patch"
patched
sft.py:1
evaluate.py:1
sft.py:0
evaluate.py:0
```

---

## 6. 下载基座模型与预处理数据

基座模型走 ModelScope（Qwen2.5-**base**，规避 README 公告的 Instruct 模型约束解码失效问题）；官方预处理的 Amazon Industrial_and_Scientific 数据只在 HuggingFace 侧，经 hf-mirror 拉取（约 27 MB，排除 .npy 嵌入文件——仅 RQ/SID 构建阶段需要）。`info/` 下的 SID-条目清单 txt 是评估阶段构建约束 trie 与对齐评测目标的输入，一并拉取。

```shell #test-setup id="download"
cd MiniOneRec
modelscope download --model Qwen/Qwen2.5-0.5B --local_dir ./Qwen2.5-0.5B
export HF_ENDPOINT=https://hf-mirror.com
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='kkknight/MiniOneRec', allow_patterns=['Amazon/index/Industrial_and_Scientific.index.json','Amazon/index/Industrial_and_Scientific.item.json','Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv','Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv','Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv','Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt'], local_dir='data')"
```

校验资产落位：

```shell #test id="verify-assets"
cd MiniOneRec
test -f Qwen2.5-0.5B/config.json && echo "base model ok"
test -f data/Amazon/index/Industrial_and_Scientific.index.json && echo "sid index ok"
test -f data/Amazon/index/Industrial_and_Scientific.item.json && echo "item meta ok"
test -f data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv && echo "train csv ok"
test -f data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv && echo "valid csv ok"
test -f data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt && echo "info txt ok"
```

输出结果如下：

```shell #test-result id="verify-assets"
base model ok
sid index ok
item meta ok
train csv ok
valid csv ok
info txt ok
```

---

## 7. 验证 torch_npu 与 CUDA-API 映射

```shell #test id="check-npu"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -c "import torch, torch_npu; from torch_npu.contrib import transfer_to_npu; print('npu available:', torch.npu.is_available()); print('npu count:', torch.npu.device_count()); print('cuda-mapped available:', torch.cuda.is_available())"
```

输出结果如下：

```shell #test-result id="check-npu"
npu available: True
npu count: 1
cuda-mapped available: True
```

---

## 8. SFT 冒烟训练

单卡 1 epoch 冒烟：`sample=600` 限制每个子数据集（SID 序列 / SID-条目对齐 / 融合序列推荐）各 600 条，共约 1800 样本；`cutoff_len=256` 截断序列。完整训练（8 卡 A100、全量数据）见上游 `sft.sh`。

```shell #test-setup id="sft-run"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
cd MiniOneRec
python sft.py \
    --base_model ./Qwen2.5-0.5B \
    --train_file ./data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
    --eval_file ./data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
    --output_dir ./smoke_out \
    --batch_size 8 \
    --micro_batch_size 2 \
    --num_epochs 1 \
    --cutoff_len 256 \
    --sample 600 \
    --category Industrial_and_Scientific \
    --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
    --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json \
    --freeze_LLM False \
    --train_from_scratch False
```

训练完成后校验产物（`model.safetensors` + `config.json` + tokenizer 配置）：

```shell #test id="verify-sft"
cd MiniOneRec
test -f smoke_out/final_checkpoint/model.safetensors && echo "model saved"
test -f smoke_out/final_checkpoint/config.json && echo "config saved"
test -f smoke_out/final_checkpoint/tokenizer_config.json && echo "tokenizer saved"
grep -o '"model_type": "[a-z0-9_.-]*"' smoke_out/final_checkpoint/config.json
```

输出结果如下：

```shell #test-result id="verify-sft"
model saved
config saved
tokenizer saved
"model_type": "qwen2"
```

---

## 9. 约束解码评估（冒烟）

用冒烟训练的 `final_checkpoint` 在 test 切片上做 beam search 评估，再算 HR/NDCG/CC。**CC（无效 SID 生成数）必须为 0**——`ConstrainedLogitsProcessor` 用 trie 约束每步只允许合法 SID 前缀 token，这是约束解码在 NPU 上生效的关键信号。冒烟用 `num_beams=20`（官方 50）+ 200 行 test 切片 + `batch_size=2`，单卡约 1 分钟。

切出 test 小样本（200 行 = 1 行表头 + 199 条样本）：

```shell #test-setup id="prep-eval"
cd MiniOneRec
head -n 200 data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv > test_small.csv
wc -l test_small.csv
```

运行评估（进度条与 transformers 警告走 stderr，stdout 只有 category 名）：

```shell #test id="run-eval"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
cd MiniOneRec
python evaluate.py \
    --base_model ./smoke_out/final_checkpoint \
    --info_file ./data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt \
    --test_data_path test_small.csv \
    --result_json_data ./eval_result.json \
    --category Industrial_and_Scientific \
    --num_beams 20 --batch_size 2
```

输出结果如下（尾部进度条信息在 stderr，不参与比对）：

```shell #test-result id="run-eval"
industrial and scientific items
...
```

校验预测产物（199 条预测，每条带 `predict` 字段）：

```shell #test id="verify-eval"
cd MiniOneRec
python -c "import json; d=json.load(open('eval_result.json')); print(len(d)); print('predict' in d[0])"
```

输出结果如下：

```shell #test-result id="verify-eval"
199
True
```

计算 HR/NDCG/CC（HR/NDCG 数值随训练状态浮动，用 `...` 占位匹配；末行 CC 必须精确为 0）：

```shell #test id="calc-metrics"
cd MiniOneRec
python calc.py \
    --path ./eval_result.json \
    --item_path ./data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt
```

输出结果如下（前两行为 beam 数与生效的 top-k 列表，确定值）：

```shell #test-result id="calc-metrics"
20
[1, 3, 5, 10, 20]
NDCG:...
HR...
0
```

---

## 故障排查

| 现象 | 可能原因 | 建议 |
| --- | --- | --- |
| `npu-smi` 找不到 | 未 `source set_env.sh`，或 `npu-smi` 不在 `PATH` | 重做第 1-2 节 |
| `import torch_npu` 失败 | torch/torch_npu 版本不匹配 | 检查[兼容矩阵](https://gitcode.com/Ascend/pytorch)，确认 torch 2.9.0 ↔ torch_npu 2.9.0.post2 |
| `npu available: False` | NPU 设备未挂载或驱动问题 | 检查 `/dev/davinci0` 是否存在 |
| DataLoader worker 段错误 | torch_npu/CANN 与 fork 的兼容性问题 | 在 `TrainingArguments` 加 `dataloader_num_workers=0`（默认即为 0；仅在改过该参数后出现时回退） |
| bf16 相关报错 | 硬件为 910A（不支持 bf16） | 更换 910B 系列硬件 |
| ModelScope 下载慢/失败 | 网络 | 重试（支持断点续传）；数据侧确认 `HF_ENDPOINT=https://hf-mirror.com` 已导出 |
| `model_type` grep 为空 | `final_checkpoint/config.json` 缺失 | 训练未完成，检查上一块退出码与日志 |
| 评估报 `FileNotFoundError: [Errno 2] No such file or directory: ''` | 未传 `--result_json_data`，结果文件路径为空 | 补上 `--result_json_data ./eval_result.json` |
| CC 大于 0 | 约束解码未生效 | 确认第 5 节补丁已注入 `evaluate.py`（`grep -c transfer_to_npu evaluate.py` 应为 1）；基座模型须为 Qwen2.5-**base**（Instruct 版约束解码失效） |
| HR/NDCG 与文中示例差异大 | 冒烟训练仅 1 epoch/600 样本，`num_beams` 也降为 20 | 数值仅供参考；CI 只断言 CC=0，HR/NDCG 用占位符匹配 |
