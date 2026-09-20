# Ray

This directory contains Ascend guard data for
[ray-project/ray](https://github.com/ray-project/ray), not Ray source code.
Ray is registered as a basic-support inference-acceleration project in phase A.

## Examples

The examples workflow uses the shared `examples-template.yml` engine. A manual
run with an empty `target_ref` selects the latest Ray release; an explicit tag,
branch, or SHA can still be supplied. The schedule remains disabled until the
expanded matrix is validated on NPU runners.

One manifest holds both sources. `source: upstream` paths resolve in the Ray
checkout; `source: project` paths resolve in this directory's `example/` tree.
The matrix job name is the full relative test path without its extension.
The shared checker verifies both path origins before a self-hosted NPU runner
is allocated.

The two upstream Ray tests remain unchanged:

- `python/ray/tests/accelerators/test_npu.py` exercises Ray Core's native
  `NPUAcceleratorManager` contract;
- `python/ray/train/tests/test_torch_device_manager.py` exercises Ray Train's
  `NPUTorchDeviceManager`; the CI overlay selects `test_npu_device_manager`
  so unrelated CUDA and TPU tests in the same upstream file are not run.

The manifest has 16 jobs: two upstream tests and 14 project-owned test files.
The project-owned cases cover real-device discovery; Task/Actor assignment and
isolation; resource queuing, release, and actor process recovery; fractional
scheduling capacity; single-worker training; same-node two-worker HCCL and DDP;
checkpoint restoration; Ray Data task and actor inference; Ray Serve inference
and HTTP batching; and Ray Tune learning-rate search. See
[the application coverage notes](example/README.md) for the upstream document
patterns and exact assertions. A passing test must prove the stated behavior,
not just import Ray or return exit code zero. Cross-machine tests
are intentionally excluded: the available two-card runner is one host, not a
multi-node Ray cluster. The `scan.paths` list is limited to the two upstream
paths and does not claim to discover every new upstream NPU test.

`setup_example.sh` installs a released target directly from its matching PyPI
version. Development targets such as Ray master use the official Linux aarch64
wheel built from the exact target commit.
Ray wheels intentionally exclude test packages. After the matching runtime is
installed, the setup uses the target checkout's own `python/ray/setup-dev.py`
to link only `ray/tests` into the wheel. A project-local verifier rejects a
runtime version mismatch or a test package that resolves outside the target
checkout.

## Quick Start

`docs/Quick-start-Ascend.md` is based on Ray's upstream Accelerator Support
document. It validates two-device discovery, Task/Actor allocation,
`ASCEND_RT_VISIBLE_DEVICES`, and a real `torch_npu` operation. The workflow uses
the repository's shared Quick Start engine; scheduled polling stays disabled
until the first manual run is green.
