#!/usr/bin/env python3
"""Scan a target tree's examples and write examples_manifest.yaml.

Paths passed with --supported are written to the supported section. Every
other scanned example is written as unsupported. That classification is a
task rule, not a community judgment.

--scan-root, --include-extension, --unit, --marker, and --max-depth
control what is scanned and are recorded in the manifest's scan
section. The scan implementation lives in this file; the scan-section
semantics that the examples-check daily audit replays are specified in
docs/examples-check-design.md (§2.6) - keep the two aligned when
changing either. Note that regenerating a manifest overwrites the file:
hand-maintained sections (unsupported comments, scan.exclude) must be
restored by hand.
Default unit is files (.sh / .py / .yaml). unit=directories treats
each child directory as one example. unit=mixed unions depth-limited
directories with depth-limited files. --runner / --npu-devices /
--image / --timeout-minutes / --profile apply to every supported
entry; entries that need different values must be edited by hand
afterwards. overlay_args and exec are optional and left as comments
for hand editing.

CI does not call this script. Use it once when onboarding a project, then
fill in the scheduling fields on each supported entry.
"""
from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_SCAN_ROOT = 'examples'
DEFAULT_INCLUDE_EXTENSIONS = ('.sh', '.py', '.yaml')
DEFAULT_MIXED_INCLUDE_EXTENSIONS = ('.sh', '.py')
DEFAULT_SCAN_UNIT = 'files'
DEFAULT_DIR_MARKER = 'CMakeLists.txt'
DEFAULT_DIR_MAX_DEPTH = 1
SCAN_UNITS = ('files', 'directories', 'mixed')


def normalize_extension(ext: str) -> str:
    if not isinstance(ext, str):
        raise SystemExit(
            f'scan.include_extensions items must be strings, got {type(ext).__name__}')
    ext = ext.strip()
    if not ext:
        raise SystemExit('empty value in scan.include_extensions')
    return ext if ext.startswith('.') else f'.{ext}'


def normalize_extensions(value, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or not value:
        raise SystemExit('scan.include_extensions must be a non-empty list')
    return tuple(normalize_extension(item) for item in value)


def _normalize_marker(unit: str, marker):
    if unit == 'directories':
        if marker is None:
            return DEFAULT_DIR_MARKER
        if not isinstance(marker, str) or not marker.strip():
            raise SystemExit('scan.marker must be a non-empty string')
        return marker.strip()
    if unit == 'mixed':
        if marker is None or marker == '':
            return ''
        if not isinstance(marker, str):
            raise SystemExit(
                f'scan.marker must be a string or empty, got {type(marker).__name__}')
        if not marker.strip():
            raise SystemExit(
                'scan.marker must be a non-empty string or explicitly empty')
        return marker.strip()
    return None


def load_scan(scan: dict) -> dict:
    unit = scan.get('unit') or DEFAULT_SCAN_UNIT
    if unit not in SCAN_UNITS:
        raise SystemExit(
            f'scan.unit must be one of {SCAN_UNITS}, got {unit!r}')
    max_depth = scan.get('max_depth', DEFAULT_DIR_MAX_DEPTH)
    if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 1:
        raise SystemExit(
            f'scan.max_depth must be an integer >= 1, got {max_depth!r}')
    if unit == 'mixed':
        default_extensions = DEFAULT_MIXED_INCLUDE_EXTENSIONS
    else:
        default_extensions = DEFAULT_INCLUDE_EXTENSIONS
    # An explicitly empty root means the repo root itself (deepspeed /
    # torch.vision style); only an absent root falls back to examples/.
    root = scan.get('root')
    return {
        'scan_root': DEFAULT_SCAN_ROOT if root is None else root,
        'unit': unit,
        'marker': _normalize_marker(unit, scan.get('marker')),
        'max_depth': max_depth,
        'include_extensions': normalize_extensions(
            scan.get('include_extensions'), default_extensions),
    }


def scan_file_units(
        examples_root: Path, target_root: Path,
        include_extensions: tuple[str, ...],
        max_depth: int | None = None) -> list[str]:
    found: list[str] = []
    for path in sorted(examples_root.rglob('*')):
        if not path.is_file() or path.suffix not in include_extensions:
            continue
        if (max_depth is not None
                and len(path.relative_to(examples_root).parts) > max_depth):
            continue
        found.append(path.relative_to(target_root).as_posix())
    return found


def scan_directory_units(examples_root: Path, target_root: Path,
                         marker: str | None, max_depth: int) -> list[str]:
    found: list[str] = []

    def visit(directory: Path, depth: int) -> None:
        if depth >= max_depth:
            return
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            if not child.is_dir():
                continue
            if not marker or (child / marker).is_file():
                found.append(child.relative_to(target_root).as_posix())
            visit(child, depth + 1)

    visit(examples_root, 0)
    return found


def scan_examples(target_root: Path, scan: dict) -> list[str]:
    examples_root = target_root / scan['scan_root']
    if not examples_root.is_dir():
        raise SystemExit(f"{scan['scan_root']}/ not found under {target_root}")
    unit = scan['unit']
    if unit == 'directories':
        found = scan_directory_units(
            examples_root, target_root, scan['marker'], scan['max_depth'])
    elif unit == 'mixed':
        found = scan_directory_units(
            examples_root, target_root, scan['marker'], scan['max_depth'])
        found.extend(scan_file_units(
            examples_root, target_root, scan['include_extensions'],
            scan['max_depth']))
    else:
        found = scan_file_units(
            examples_root, target_root, scan['include_extensions'])
    return sorted(set(found))


def render_supported_entry(
    path: str,
    runner: str | None,
    npu_devices: str | None,
    image: str | None,
    timeout_minutes: int | None,
    profile: str | None,
    unit: str,
) -> list[str]:
    lines = [f'  - path: {path}']
    if profile is not None:
        lines.append(f'    profile: {profile}')
    else:
        lines.append('    # profile: <setup-profile>')
    if runner is not None:
        lines.append(f'    runner: {runner}')
    else:
        lines.append('    # runner: <runner-label>')
    if npu_devices is not None:
        lines.append(f"    npu_devices: '{npu_devices}'")
    else:
        lines.append("    # npu_devices: '0,1'")
    if image is not None:
        lines.append(f'    image: {image}')
    else:
        lines.append('    # image: <swr-image>')
    if unit in ('directories', 'mixed'):
        lines.append('    # exec: build/bin/<binary>')
    lines.append('    # overlay_args: []')
    if timeout_minutes is not None:
        lines.append(f'    timeout_minutes: {timeout_minutes}')
    else:
        lines.append('    # timeout_minutes: 180')
    return lines


def render_scan_section(scan: dict) -> list[str]:
    lines = [
        'scan:',
        f"  root: {scan['scan_root']}",
    ]
    unit = scan['unit']
    if unit == 'directories':
        lines.append('  unit: directories')
        lines.append(f"  marker: {scan['marker']}")
        lines.append(f"  max_depth: {scan['max_depth']}")
        return lines
    rendered_extensions = ', '.join(
        f"'{ext}'" for ext in scan['include_extensions'])
    if unit == 'mixed':
        lines.append('  unit: mixed')
        lines.append(f"  max_depth: {scan['max_depth']}")
        if scan['marker']:
            lines.append(f"  marker: {scan['marker']}")
        lines.append(f'  include_extensions: [{rendered_extensions}]')
        return lines
    lines.append(f'  include_extensions: [{rendered_extensions}]')
    return lines


def render_manifest(
    paths: list[str],
    supported_paths: list[str],
    scan: dict,
    runner: str | None,
    npu_devices: str | None,
    image: str | None,
    timeout_minutes: int | None,
    profile: str | None,
) -> str:
    missing = [p for p in supported_paths if p not in paths]
    if missing:
        raise SystemExit(f'supported example missing from scan: {missing[0]}')
    supported_set = set(supported_paths)
    unsupported = [p for p in paths if p not in supported_set]
    lines = [
        'version: 1',
        *render_scan_section(scan),
        'supported:',
    ]
    if supported_paths:
        for path in supported_paths:
            lines.extend(render_supported_entry(
                path, runner, npu_devices, image, timeout_minutes,
                profile, scan['unit'],
            ))
    else:
        lines.append('  []')
    lines.append('unsupported:')
    for path in unsupported:
        lines.append(f'  - {path}')
    lines.append('')
    return '\n'.join(lines)


def scan_from_args(args: argparse.Namespace) -> dict:
    raw: dict = {
        'root': args.scan_root,
        'unit': args.unit,
        'max_depth': args.max_depth,
    }
    if args.include_extension is not None:
        raw['include_extensions'] = args.include_extension
    if args.marker is not None:
        raw['marker'] = args.marker
    return load_scan(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-root', required=True, help='Checkout of the target project tree')
    parser.add_argument('--output', required=True, help='Path to write examples_manifest.yaml')
    parser.add_argument(
        '--supported',
        action='append',
        default=[],
        help='Example path (relative to target root) to mark supported. Repeatable.',
    )
    parser.add_argument(
        '--scan-root',
        default=DEFAULT_SCAN_ROOT,
        help='Directory under the target root to scan (default: examples; '
             'pass an empty string for the repo root itself)',
    )
    parser.add_argument(
        '--include-extension',
        action='append',
        default=None,
        help="File extension to scan, with or without the leading dot "
             "(default: .sh .py .yaml for files, .sh .py for mixed). "
             "Repeatable. Ignored when --unit directories.",
    )
    parser.add_argument(
        '--unit',
        choices=SCAN_UNITS,
        default=DEFAULT_SCAN_UNIT,
        help='Example unit to scan (default: files)',
    )
    parser.add_argument(
        '--marker',
        default=None,
        help='File that must exist in a directory unit '
             f'(default: {DEFAULT_DIR_MARKER} when --unit directories; '
             'omit or pass empty for mixed with no marker)',
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=DEFAULT_DIR_MAX_DEPTH,
        help='How many directory levels under --scan-root to treat as '
             'example units (default: 1)',
    )
    parser.add_argument('--runner', default=None, help='Runner label written on every supported entry')
    parser.add_argument('--npu-devices', default=None, help="Value for npu_devices, e.g. 0,1")
    parser.add_argument('--image', default=None, help='Container image written on every supported entry')
    parser.add_argument('--timeout-minutes', type=int, default=None, help='Timeout written on every supported entry')
    parser.add_argument('--profile', default=None, help='Setup profile written on every supported entry')
    args = parser.parse_args()
    target_root = Path(args.target_root).resolve()
    output = Path(args.output)
    scan = scan_from_args(args)
    text = render_manifest(
        scan_examples(target_root, scan),
        args.supported,
        scan,
        args.runner,
        args.npu_devices,
        args.image,
        args.timeout_minutes,
        args.profile,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding='utf-8')
    print(f'wrote {output} ({text.count(chr(10))} lines)')


if __name__ == '__main__':
    main()
