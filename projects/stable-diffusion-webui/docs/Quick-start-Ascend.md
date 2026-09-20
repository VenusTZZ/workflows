# Quick Start (Ascend NPU)

在单卡昇腾 NPU 上以无头 API 模式运行 stable-diffusion-webui，通过 REST API 完成一次文生图推理。

## 前置条件

Atlas 900 A2 单卡，已装好 CANN 以及配套的 torch、torch_npu，`torch.npu.is_available()` 为 True。设置 CANN 环境变量：
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

容器需已安装 opencv 运行库 `libgl1` 与 `libglib2.0-0`。

| 组件 | 版本 | 来源 |
| --- | --- | --- |
| Python | 3.12 | 昇腾 CANN 镜像 |
| CANN | 9.1.0 | 昇腾 CANN 镜像 |
| torch | 2.9.0+cpu | 昇腾 PyPI 源 |
| torch_npu | 2.9.0.post2 | 昇腾 PyPI 源 |
| transformers | 4.44.2 | PyPI |
| blendmodes | 2023 | PyPI |
| scikit-image | 0.25.2 | PyPI |
| Pillow | 10.4.0 | PyPI |
| stable-diffusion-webui | v1.10.1 | GitHub |
| 模型 | `AI-ModelScope/sd-turbo` | ModelScope，约 3.4 GB，自动下载 |

## 环境检查

检查 Python 版本：
```shell #test id="check-py"
python --version
```
```shell #test-result id="check-py" fuzzy='xxx'
Python 3.12.xxx
```

检查 torch / torch_npu 是否装好且 NPU 设备可用：
```shell #test id="check-torch"
python -c "import torch, torch_npu; print('torch=', torch.__version__); print('torch_npu=', torch_npu.__version__); print('is_available:', torch.npu.is_available()); print('count:', torch.npu.device_count()); print('device:', torch_npu.npu.get_device_name(0))"
```
```shell #test-result id="check-torch" fuzzy='xxx'
torch= xxx
torch_npu= xxx
is_available: True
count: 1
device: xxx
```

## 获取代码

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

克隆上游仓库并 checkout 到最新 release：
```shell #test id="clone-repo" load="upstream_ref>>ref"
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
cd stable-diffusion-webui
git checkout <ref>
echo "HEAD $(git log -1 --format=%h)"
```
```shell #test-result id="clone-repo" fuzzy='xxx'
HEAD xxx
```

## 安装依赖

按上游锁定清单安装依赖，先解除 Python 3.12 与 NPU 无法安装的钉死版本；CLIP 无预编译包从 GitHub 源码安装，modelscope 用于下载模型：
```shell #test-setup
cd stable-diffusion-webui
sed -i -e 's/transformers==4.30.2/transformers==4.44.2/' -e 's/blendmodes==2022/blendmodes==2023/' -e 's/scikit-image==0.21.0/scikit-image==0.25.2/' -e 's/Pillow==9.5.0/Pillow==10.4.0/' requirements_versions.txt
pip install -r requirements_versions.txt
pip install modelscope
pip install torch==2.9.0 torchvision==0.24.0 torch_npu==2.9.0.post2
sed -i -e 's/transformers==4.30.2/transformers==4.44.2/' -e 's/blendmodes$/blendmodes==2023/' -e 's/scikit-image>=0.19/scikit-image==0.25.2/' requirements.txt
pip install -r requirements.txt
pip install 'setuptools<70' wheel
pip install --no-build-isolation "https://github.com/openai/CLIP/archive/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1.zip"
```

验证依赖可用，并复核各包版本：
```shell #test id="install-webui"
python -c "import torch, torch_npu, modelscope, gradio, fastapi, transformers, tokenizers, skimage, PIL; print('deps ok', torch.__version__, torch_npu.__version__, transformers.__version__, skimage.__version__, PIL.__version__)"
```
```shell #test-result id="install-webui" fuzzy='xxx'
deps ok xxx xxx xxx xxx xxx
```

## 无头文生图（单卡 NPU）

sd-turbo 约 3.4 GB，首次运行时自动下载到默认缓存：
<!--
```shell #test-setup store="model_dir"
python -c "from modelscope import snapshot_download; print(snapshot_download('AI-ModelScope/sd-turbo'))" > /tmp/sd-turbo-model-dir.txt && tail -n 1 /tmp/sd-turbo-model-dir.txt
```
-->

注入 autocast 补丁并指向 stablediffusion 的社区 fork，以 API 模式启动；`<ckpt>` 为模型下载目录：
```shell #test-setup load="model_dir>>ckpt"
cd stable-diffusion-webui
export STABLE_DIFFUSION_REPO=https://github.com/w-e-w/stablediffusion.git
mkdir -p db
python -c "p='modules/devices.py'; t=open(p).read(); t=t.replace('if has_xpu() or has_mps() or cuda_no_autocast():','if npu_specific.has_npu or has_xpu() or has_mps() or cuda_no_autocast():'); open(p,'w').write(t)"
nohup python launch.py --nowebui --skip-torch-cuda-test --ckpt <ckpt>/sd_turbo.safetensors --port 7861 &
```

校验 API 已就绪：
```shell #test id="wait-ready"
curl -sf http://127.0.0.1:7861/docs > /dev/null && echo "api ready" || echo "api not ready"
```
```shell #test-result id="wait-ready"
api ready
```

发起文生图推理：
```shell #test id="txt2img"
curl -s -X POST http://127.0.0.1:7861/sdapi/v1/txt2img -H 'Content-Type: application/json' -d '{"prompt": "a cute cat", "steps": 1, "cfg_scale": 1.0, "width": 512, "height": 512}' > /tmp/sd-turbo-resp.json
python -c "import json, base64; r=json.load(open('/tmp/sd-turbo-resp.json')); print('txt2img images:', len(r['images'])); open('/tmp/sd-turbo-out.png','wb').write(base64.b64decode(r['images'][0]))"
```
```shell #test-result id="txt2img"
txt2img images: 1
```

校验 PNG 文件头与大小下限，防止空图坏图：
```shell #test id="verify-png"
python -c "import os; p='/tmp/sd-turbo-out.png'; s=os.path.getsize(p); assert s>10000; assert open(p,'rb').read(8)==b'\\x89PNG\\r\\n\\x1a\\n'; print(s,'bytes')"
```
```shell #test-result id="verify-png" fuzzy='xxx'
xxx bytes
```

更多用法见 [stable-diffusion-webui wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki)。
