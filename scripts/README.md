# Build and verification tooling

[`check.py`](check.py) is the current source-build and verification entry point.
It uses Python's standard library and keeps host checks, CUDA compilation,
checkpoint integration and GPU execution separate:

```sh
python3 scripts/check.py host
python3 scripts/check.py archive
python3 scripts/check.py cuda-build
python3 scripts/check.py gpu --fixtures .cache/forward-fixtures
python3 scripts/check.py gpu --fixtures .cache/forward-fixtures --force-ptx
```

`host` needs CMake 3.24+, a C++20 compiler and OpenSSL 3 development files. Initial
CMake configuration fetches hash-pinned nlohmann/json 3.12.0 unless already cached.
It builds the host targets and runs isolated C++ units. `archive` runs the real
file-integrity/binding scenario from that build; run `host` first.

`cuda-build` builds a Linux/amd64 CUDA 13.2.0 development image pinned by manifest
digest in the root [Dockerfile](../Dockerfile), then compiles the CUDA targets.
The Docker context contains only that Dockerfile: source and build output are
mounted for compilation, while weights remain outside the image. CMake, Ninja
and OpenSSL development package versions are explicit in the image. Native
CUDA 13.2+ builds are also available with `cuda-build --native`.

`gpu` executes the device, tiny forward and pinned 8B forward scenarios on a
compatible Linux NVIDIA host. It writes per-scenario JSON reports under
`.cache/checks` by default. `--force-ptx` sets `CUDA_FORCE_PTX_JIT=1`; reports from
that run are separate from native-code reports. Use `--container` only with
working Docker GPU access. Container execution requires the NVIDIA Container Toolkit; native execution
requires compatible host CUDA libraries. See [verification commands](../tests/README.md)
for the reference policy and scenario coverage. Keep reports in ignored storage.

## Independent reference fixtures

[`validation/forward_reference.py`](validation/forward_reference.py) generates
fixtures without calling Dolittle, using the pinned verification environment and
CUDA. It verifies existing 8B weights against `models-8b.json`; it does not fetch
missing weights. The generated files record provenance, dtype, consumed-token
positions and numerical tolerances. For example:

```sh
python3 scripts/validation/forward_reference.py \
  --output .cache/forward-fixtures --kind all \
  --actual-checkpoint /path/to/verified/Qwen3-8B
```

The generator documents its FP32 attention-intermediate policy and the separate
stock-eager sensitivity check. Those details matter when interpreting reduced-
precision comparisons. Historical checkpoint-selection scripts remain in
[`validation/`](validation/README.md); they are reference tools, not the engine.

## Later application launch

The planned `run.py` entry point will check Docker/GPU prerequisites, acquire
and verify declared model files in a persistent cache, and launch the application.
It is not implemented in this forward-pass slice. Keep downloads out of image
builds and never reset a user's checkout or existing model cache.

## Project site

`python3 scripts/site.py serve` builds and serves the public project site on
loopback, independently of the GPU application. It requires Python 3.11+ and
Hugo 0.154.2. See [site setup](../web/README.md) for publication rules and URLs.
The same tooling validates generated files through a pure in-memory snapshot
checker, with a filesystem wrapper used after actual Hugo builds.
