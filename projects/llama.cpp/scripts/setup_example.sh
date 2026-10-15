#!/usr/bin/env bash
# Prepare the CI environment for one supported example.
# $1 is the manifest profile. Unknown profiles fail before any install.
# EXEC and EXAMPLE_PATH come from the workflow as environment variables.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"
EXEC_REL="${EXEC:-}"
EXAMPLE_REL="${EXAMPLE_PATH:-}"

# 910B DataType table lists FP16 / Q8_0 / Q4_0 / BF16 (Q4_K_M etc. not in table); upstream docs/backend/CANN.md.
QWEN_MODEL_URL=https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/master/qwen2.5-0.5b-instruct-q4_0.gguf
QWEN_MODEL_FILE=qwen2.5-0.5b-instruct-q4_0.gguf
# Dream GGUF is produced by prepare_dream_gguf.sh (ModelScope HF →
# Q8_0) on the shared persistent runner cache (the pool mounts one
# volume at ~/.cache/huggingface - the legacy /root/.cache/cosdt-ci-
# test/llama.cpp existed only via a container bind-mount the engine
# does not provide).
DREAM_MODEL_FILE=Dream-v0-Instruct-7B.Q8_0.gguf
DREAM_CACHE_ROOT="${DREAM_CACHE_ROOT:-${HF_HOME:-${HOME}/.cache/huggingface}/llama.cpp}"
DREAM_MODEL_DEST="$DREAM_CACHE_ROOT/$DREAM_MODEL_FILE"

require_exec() {
  if [[ -z "$EXEC_REL" ]]; then
    echo "EXEC is required for profile ${PROFILE}" >&2
    exit 1
  fi
}

source_cann() {
  export PATH="/usr/local/sbin:$PATH"
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
}

assert_cann_lib() {
  if ! compgen -G "$1/libggml-cann.*" >/dev/null; then
    echo "build produced no CANN backend library under $1" >&2
    exit 1
  fi
}

cmake_llama() {
  local -a extra=("$@")
  cmake -S "$TARGET_ROOT" -B "$TARGET_ROOT/build" \
    -DCMAKE_BUILD_TYPE=Release \
    "${extra[@]}"
}

build_target() {
  require_exec
  cmake --build "$TARGET_ROOT/build" --target "$(basename "$EXEC_REL")" -j "$(nproc)"
}

fetch_gguf() {
  local url="$1"
  local dest="$2"
  local fail_msg="$3"
  local part="$dest.part"
  mkdir -p "$(dirname "$dest")"
  if [[ -f "$dest" && "$(head -c 4 "$dest")" == "GGUF" ]]; then
    echo "reusing $dest"
  else
    # Write to .part and resume. A truncated dest still starts with GGUF,
    # so never treat an in-progress file as complete.
    if ! curl -fL --retry 3 --retry-delay 5 --connect-timeout 30 -C - "$url" -o "$part"; then
      echo "$fail_msg" >&2
      exit 1
    fi
    if [[ "$(head -c 4 "$part")" != "GGUF" ]]; then
      echo "$fail_msg" >&2
      exit 1
    fi
    mv -f "$part" "$dest"
  fi
  echo "LLAMA_CI_MODEL=$dest" >> "$GITHUB_ENV"
  echo "model ready: $dest"
}

patch_working_copy() {
  case "$EXAMPLE_REL" in
    examples/simple-chat)
      python3 - "$TARGET_ROOT/examples/simple-chat/simple-chat.cpp" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
old = """    // only print errors
    llama_log_set([](enum ggml_log_level level, const char * text, void * /* user_data */) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            fprintf(stderr, "%s", text);
        }
    }, nullptr);
"""
if old not in text:
    raise SystemExit(f"simple-chat log filter not found in {path}")
path.write_text(text.replace(old, "", 1), encoding="utf-8")
PY
      ;;
    examples/retrieval)
      python3 - "$TARGET_ROOT/examples/retrieval/retrieval.cpp" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
old = """        std::getline(std::cin, query);
        std::vector<int32_t> query_tokens = common_tokenize(ctx, query, true);
"""
new = """        std::getline(std::cin, query);
        if (query.empty() || std::cin.eof()) {
            break;
        }
        std::vector<int32_t> query_tokens = common_tokenize(ctx, query, true);
"""
if old not in text:
    raise SystemExit(f"retrieval getline loop not found in {path}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
PY
      ;;
  esac
}

setup_cann() {
  source_cann
  patch_working_copy
  cmake_llama -DGGML_CANN=on
  build_target
  assert_cann_lib "$TARGET_ROOT/build/bin"
  fetch_gguf "$QWEN_MODEL_URL" "$GITHUB_WORKSPACE/models/$QWEN_MODEL_FILE" \
    "downloaded file is not a GGUF model: $GITHUB_WORKSPACE/models/$QWEN_MODEL_FILE"
}

setup_cpu() {
  cmake_llama
  build_target
}

setup_cann-diffusion() {
  source_cann
  cmake_llama -DGGML_CANN=on
  build_target
  assert_cann_lib "$TARGET_ROOT/build/bin"
  # The legacy pipeline staged the Dream GGUF via a separate
  # prepare-dream-gguf job + an NFS bind-mount shared with this job;
  # the engine has no extra-job hooks, so this leg's setup prepares it
  # itself on the shared persistent cache. Idempotent and flock-
  # protected: warm caches return in seconds, and only this leg pays
  # the cold cost (manifest timeout covers the full prepare + build).
  bash "${PROJECT_ROOT:?PROJECT_ROOT is required}/scripts/prepare_dream_gguf.sh"
  if [[ ! -f "$DREAM_MODEL_DEST" || "$(head -c 4 "$DREAM_MODEL_DEST")" != "GGUF" ]]; then
    echo "Dream GGUF missing or incomplete after prepare: $DREAM_MODEL_DEST" >&2
    exit 1
  fi
  echo "LLAMA_CI_MODEL=$DREAM_MODEL_DEST" >> "$GITHUB_ENV"
  echo "model ready: $DREAM_MODEL_DEST"
}

setup_cmake-pkg() {
  source_cann
  cmake -S "$TARGET_ROOT" -B "$TARGET_ROOT/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CANN=on \
    -DCMAKE_INSTALL_PREFIX="$TARGET_ROOT/inst"
  cmake --build "$TARGET_ROOT/build" -j "$(nproc)"
  cmake --install "$TARGET_ROOT/build" --prefix "$TARGET_ROOT/inst"
  cmake -S "$TARGET_ROOT/examples/simple-cmake-pkg" \
    -B "$TARGET_ROOT/examples/simple-cmake-pkg/build" \
    -DCMAKE_PREFIX_PATH="$TARGET_ROOT/inst/lib/cmake"
  cmake --build "$TARGET_ROOT/examples/simple-cmake-pkg/build" -j "$(nproc)"
  if ! compgen -G "$TARGET_ROOT/inst/lib/libggml-cann.*" >/dev/null \
    && ! compgen -G "$TARGET_ROOT/build/bin/libggml-cann.*" >/dev/null; then
    echo "build produced no CANN backend library under inst/lib or build/bin" >&2
    exit 1
  fi
  fetch_gguf "$QWEN_MODEL_URL" "$GITHUB_WORKSPACE/models/$QWEN_MODEL_FILE" \
    "downloaded file is not a GGUF model: $GITHUB_WORKSPACE/models/$QWEN_MODEL_FILE"
  local bin="$TARGET_ROOT/examples/simple-cmake-pkg/build/llama-simple-cmake-pkg"
  if [[ ! -f "$bin" ]]; then
    echo "cmake-pkg binary not found: $bin" >&2
    exit 1
  fi
  # RUNPATH on the consumer binary does not apply to libggml's NEEDED
  # backends (libggml-cpu / libggml-cann). Always export inst/lib.
  echo "LD_LIBRARY_PATH=$TARGET_ROOT/inst/lib" >> "$GITHUB_ENV"
}

supported_profiles() {
  declare -F | awk '/^declare -f setup_/ { sub(/^declare -f setup_/, ""); print }' | paste -sd' ' -
}

if ! declare -F "setup_${PROFILE}" >/dev/null 2>&1; then
  echo "unknown profile: ${PROFILE} (supported: $(supported_profiles))" >&2
  exit 1
fi

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
GITHUB_WORKSPACE="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
GITHUB_ENV="${GITHUB_ENV:?GITHUB_ENV is required}"

"setup_${PROFILE}"
