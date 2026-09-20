from __future__ import annotations

import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "examples_manifest.yaml"

EXPECTED_SUPPORTED = {
    "examples/fully_async/run-qwen2.5-0.5B-fully_async.sh": {
        "profile": "slime_fully_async",
        "runner": "linux-aarch64-a2-4",
        "image": (
            "swr.cn-south-1.myhuaweicloud.com/ascendhub/"
            "cann:9.1.0-910b-ubuntu22.04-py3.12"
        ),
        "timeout_minutes": 300,
    },
}

# Entries whose execution recipe needs a minimum visible-device count.
ENTRY_DEVICE_REQUIREMENTS = {
    "examples/fully_async/run-qwen2.5-0.5B-fully_async.sh": 4,
}

# Pin table copied from the fork's Dockerfile ARGs and quick_install.sh;
# tests assert our setup pins match the fork recipe.
FORK_PINS = {
    "sglang_ref": "v0.5.13",
    "megatron_commit_prefix": "1dcf0dafa",
    "mbridge_commit_prefix": "89eb1088",
    "megatron_adaptor_commit": "f707a3b6",
    "transformer_engine_npu_commit": "47d60449",
    "torch": "2.10.0",
    "torch_npu": "2.10.0",
    "torchvision": "0.25.0",
    "triton_ascend": "3.2.1",
    "python_tag": "cp312",
    "cann_tag": "cann9.1.0",
    "board_tag": "910b",
}


def load_manifest() -> dict:
    return yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))


def unsupported_paths(manifest: dict) -> list[str]:
    return [entry for entry in manifest.get("unsupported") or []]


def test_supported_entries_match_plan() -> None:
    manifest = load_manifest()
    supported = {entry["path"]: entry for entry in manifest["supported"]}
    assert set(supported) == set(EXPECTED_SUPPORTED), (
        f"phase-1 supported set drifted: {sorted(supported)}")
    for path, expectations in EXPECTED_SUPPORTED.items():
        entry = supported[path]
        for field, expected in expectations.items():
            assert entry[field] == expected, (
                f"{path}: {field}={entry[field]!r} != {expected!r}")


def test_supported_and_unsupported_do_not_overlap() -> None:
    manifest = load_manifest()
    supported_paths = {entry["path"] for entry in manifest["supported"]}
    unsupported = unsupported_paths(manifest)
    overlap = supported_paths & set(unsupported)
    assert not overlap, f"entries listed in both sections: {overlap}"
    assert len(unsupported) == len(set(unsupported)), (
        "duplicate unsupported entries")


def test_unsupported_entries_carry_inline_comments() -> None:
    manifest = load_manifest()
    raw_lines = MANIFEST_PATH.read_text(encoding="utf-8").splitlines()

    def commented_mentions(path: str) -> bool:
        """A comment line references the path (full or basename form)."""
        for line in raw_lines:
            stripped = line.lstrip()
            if not stripped.startswith("#"):
                continue
            if path in stripped:
                return True
            # Group comments may reference only the basename (library
            # notes mention e.g. base_env.py without its directory).
            if Path(path).name in stripped:
                return True
        return False

    for path in unsupported_paths(manifest):
        assert commented_mentions(path), f"no ledger comment mentions {path}"


def test_scan_config_targets_fork_examples_tree() -> None:
    manifest = load_manifest()
    scan = manifest["scan"]
    assert scan["root"] == "examples"
    assert ".sh" in scan["include_extensions"]
    assert ".py" in scan["include_extensions"]
    # scan.exclude was retired 2026-09-20; the package marker is now a
    # ledger entry in `unsupported`.
    assert "exclude" not in scan
    unsupported = unsupported_paths(manifest)
    assert "examples/__init__.py" in unsupported
    assert "examples/tau-bench/token_delta.py" in unsupported


def test_fixture_rows_match_dapo_schema() -> None:
    fixture = PROJECT_ROOT / "fixtures" / "ci_dapo_16.jsonl"
    rows = [json.loads(line) for line in fixture.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 16
    for row in rows:
        assert set(row) == {"prompt", "label"}
        assert isinstance(row["prompt"], list) and row["prompt"]
        assert all(set(message) == {"content", "role"} for message in row["prompt"])
        assert isinstance(row["label"], str)


def test_setup_pins_match_fork_recipe() -> None:
    setup = (PROJECT_ROOT / "scripts" / "setup_example.sh").read_text(encoding="utf-8")
    for needle in (
        f"{FORK_PINS['megatron_commit_prefix']}",
        f"{FORK_PINS['mbridge_commit_prefix']}",
        FORK_PINS["megatron_adaptor_commit"],
        FORK_PINS["transformer_engine_npu_commit"],
        f"torch=={FORK_PINS['torch']}",
        f"torch_npu=={FORK_PINS['torch_npu']}",
        f"torchvision=={FORK_PINS['torchvision']}",
        f"triton-ascend=={FORK_PINS['triton_ascend']}",
        f"{FORK_PINS['sglang_ref']}",
        FORK_PINS["cann_tag"],
        FORK_PINS["python_tag"],
        FORK_PINS["board_tag"],
    ):
        assert needle in setup, f"setup_example.sh missing fork pin: {needle}"


def test_run_example_exports_npu_contract() -> None:
    run_script = (PROJECT_ROOT / "scripts" / "run_example.sh").read_text(encoding="utf-8")
    for needle in (
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1",
        "CUDA_DEVICE_MAX_CONNECTIONS=1",
        "HCCL_HOST_SOCKET_PORT_RANGE",
        "HCCL_NPU_SOCKET_PORT_RANGE",
        "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "WANDB_MODE=offline",
        "require_visible_devices 4",
    ):
        assert needle in run_script, f"run_example.sh missing: {needle}"


def test_workflow_registers_engine_call() -> None:
    workflow = (
        PROJECT_ROOT.parents[1] / ".github" / "workflows" / "slime-examples.yml"
    ).read_text(encoding="utf-8")
    assert "name: slime-examples" in workflow
    assert "uses: ./.github/workflows/examples-template.yml" in workflow
    assert "upstream_repo: THUDM/slime" in workflow
    assert "max_parallel: 1" in workflow
    # Bring-up phase: the cron line stays commented out.
    assert "  schedule:" not in workflow
    assert "#   - cron:" in workflow


def test_projects_yaml_registers_examples_workflow() -> None:
    registry = (
        PROJECT_ROOT.parents[1] / "projects.yaml"
    ).read_text(encoding="utf-8")
    assert "examples: .github/workflows/slime-examples.yml" in registry


def test_device_requirements_cover_all_supported_entries() -> None:
    manifest = load_manifest()
    for entry in manifest["supported"]:
        assert entry["path"] in ENTRY_DEVICE_REQUIREMENTS, (
            f"missing device requirement for {entry['path']}")
