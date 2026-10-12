"""用 manifest 中的 Hydra 覆盖参数生成配置，再运行原始 ROLL launcher。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import shlex
import sys
import tempfile

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def expand_overlay(raw: str) -> list[str]:
    if not raw.strip():
        return []
    items = json.loads(raw)
    if items in (None, ''):
        return []
    if not isinstance(items, list):
        raise ValueError('OVERLAY_ARGS must be a JSON array')
    tokens = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise ValueError('OVERLAY_ARGS items must be non-empty strings')
        # 保留 Hydra 引号、列表和插值；普通 --参数允许写成一个 manifest 条目。
        if not item.startswith('--') and (item.startswith('~') or '=' in item):
            tokens.append(item)
        else:
            tokens.extend(shlex.split(os.path.expandvars(item), posix=True))
    return tokens


def compose_config(target_root: Path, config_path: str, config_name: str,
                   overrides: list[str]):
    examples = (target_root / 'examples').resolve()
    directory = (examples / config_path).resolve()
    directory.relative_to(examples)
    if not directory.is_dir():
        raise ValueError(f'upstream config directory not found: {directory}')
    if '/' in config_name or '\\' in config_name:
        raise ValueError('config_name must be a file name without a directory')
    with initialize_config_dir(config_dir=str(directory), version_base='1.1'):
        return compose(config_name=config_name, overrides=overrides)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launcher', required=True)
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--config_name', required=True)
    args, overrides = parser.parse_known_args(argv)
    for item in overrides:
        if item.startswith('-') or not (item.startswith('~') or '=' in item):
            parser.error(f'expected a Hydra key=value override, got: {item}')
    target = Path(os.environ['TARGET_ROOT']).resolve()
    launcher = (target / args.launcher).resolve()
    launcher.relative_to(target / 'examples')
    if not launcher.is_file():
        parser.error(f'launcher not found: {launcher}')
    config = compose_config(target, args.config_path, args.config_name, overrides)
    # 先解析环境变量和上游 defaults 引用，原始配置与源码均保持不变。
    text = OmegaConf.to_yaml(config, resolve=True)
    output = Path(os.environ['CI_OUTPUT_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'resolved_config.yaml').write_text(text, encoding='utf-8')
    # release launcher 的 initialize() 只接受相对自身的配置目录。
    with tempfile.TemporaryDirectory(prefix='.ci-roll-', dir=target / 'examples') as tmp:
        config_file = Path(tmp) / 'ci.yaml'
        config_file.write_text(text, encoding='utf-8')
        print(f'upstream config: {args.config_path}/{args.config_name}.yaml', flush=True)
        print(f'resolved CI config: {output / "resolved_config.yaml"}', flush=True)
        return subprocess.run(
            [sys.executable, str(launcher), '--config_path', Path(tmp).name,
             '--config_name', 'ci'], cwd=target, check=False).returncode


if __name__ == '__main__':
    if sys.argv[1:] == ['--print-overlay']:
        print(shlex.join(expand_overlay(os.environ.get('OVERLAY_ARGS', ''))))
        raise SystemExit(0)
    raise SystemExit(main())
