#!/usr/bin/env bash
# Build Dream-v0-Instruct-7B.Q8_0.gguf on the runner NFS cache.
# ModelScope has no GGUF of this model; hf-mirror 302s to us.aws.cdn.hf.co
# which the NPU runners cannot reach. Official safetensors are on
# ModelScope and stay in cache so a failed convert does not re-download.
set -euo pipefail

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
# Shared persistent runner cache (pool volume at ~/.cache/huggingface);
# the legacy default /root/.cache/cosdt-ci-test/llama.cpp existed only
# via a container bind-mount the engine does not provide. Keep the
# DREAM_CACHE_ROOT expression in sync with setup_example.sh.
CACHE_ROOT="${DREAM_CACHE_ROOT:-${HF_HOME:-${HOME}/.cache/huggingface}/llama.cpp}"
HF_DIR="$CACHE_ROOT/Dream-v0-Instruct-7B-hf"
GGUF_PATH="$CACHE_ROOT/Dream-v0-Instruct-7B.Q8_0.gguf"
SITE_DIR="$CACHE_ROOT/convert-site"
PIP_CACHE_DIR="$CACHE_ROOT/pip-cache"
MS_BASE=https://www.modelscope.cn/models/Dream-org/Dream-v0-Instruct-7B/resolve/master
# Sizes from the ModelScope repo tree (Revision=master). A short file
# after a dropped connection must not count as complete.
HF_FILES=(
  'added_tokens.json:656'
  'config.json:927'
  'configuration.json:76'
  'configuration_dream.py:3197'
  'generation_config.json:326'
  'generation_utils.py:22332'
  'merges.txt:1671853'
  'model-00001-of-00004.safetensors:4877660776'
  'model-00002-of-00004.safetensors:4932751008'
  'model-00003-of-00004.safetensors:4330865200'
  'model-00004-of-00004.safetensors:1089994880'
  'model.safetensors.index.json:27752'
  'modeling_dream.py:36763'
  'special_tokens_map.json:661'
  'tokenization_dream.py:14028'
  'tokenizer_config.json:7486'
  'vocab.json:3383407'
)

file_size() {
  wc -c < "$1" | tr -d '[:space:]'
}

is_gguf() {
  [[ -f "$1" && "$(head -c 4 "$1")" == "GGUF" ]]
}

mkdir -p "$CACHE_ROOT"
exec 9>"$CACHE_ROOT/prepare.lock"
flock 9

if is_gguf "$GGUF_PATH"; then
  echo "reusing $GGUF_PATH"
  exit 0
fi
if [[ -e "$GGUF_PATH" ]]; then
  echo "removing incomplete GGUF at $GGUF_PATH" >&2
  rm -f "$GGUF_PATH"
fi

mkdir -p "$HF_DIR"
for spec in "${HF_FILES[@]}"; do
  name="${spec%%:*}"
  want="${spec##*:}"
  dest="$HF_DIR/$name"
  if [[ -f "$dest" && "$(file_size "$dest")" == "$want" ]]; then
    echo "reusing $dest"
    continue
  fi
  echo "downloading $name ($want bytes)"
  part="$dest.part"
  rm -f "$dest"
  if ! curl -fL --retry 5 --retry-delay 5 --connect-timeout 30 -C - \
    "$MS_BASE/$name" -o "$part"; then
    echo "Dream HF download failed: $MS_BASE/$name" >&2
    exit 1
  fi
  got="$(file_size "$part")"
  if [[ "$got" != "$want" ]]; then
    echo "Dream HF size mismatch: $name got $got want $want" >&2
    exit 1
  fi
  mv -f "$part" "$dest"
done

CONVERT_PY="$TARGET_ROOT/convert_hf_to_gguf.py"
if [[ ! -f "$CONVERT_PY" ]]; then
  echo "convert_hf_to_gguf.py not found in $TARGET_ROOT" >&2
  exit 1
fi

export PYTHONPATH="$SITE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PIP_CACHE_DIR
if ! python3 -c 'import numpy, torch, transformers' >/dev/null 2>&1; then
  echo "installing convert deps into $SITE_DIR"
  mkdir -p "$SITE_DIR" "$PIP_CACHE_DIR"
  # Huawei first (runners are in mainland China). CPU extra-index is last
  # so a CUDA wheel from PyPI is not preferred when a +cpu wheel exists.
  python3 -m pip install --target "$SITE_DIR" --cache-dir "$PIP_CACHE_DIR" \
    -i https://repo.huaweicloud.com/ascend/repos/pypi \
    --extra-index-url https://mirrors.huaweicloud.com/repository/pypi/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    'numpy>=1.26,<2' \
    'transformers==4.57.6' \
    'sentencepiece>=0.1.98,<0.3.0' \
    'protobuf>=4.21.0,<5.0.0' \
    torch
  python3 -c 'import numpy, torch, transformers'
fi

echo "converting $HF_DIR -> $GGUF_PATH"
part="$GGUF_PATH.part"
rm -f "$part"
python3 "$CONVERT_PY" "$HF_DIR" --outtype q8_0 --outfile "$part"
if ! is_gguf "$part"; then
  echo "convert produced no GGUF file: $part" >&2
  exit 1
fi
mv -f "$part" "$GGUF_PATH"
echo "Dream GGUF ready: $GGUF_PATH"
