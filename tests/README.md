# Verification

Follow the [test scope and stability rules](../docs/process/workflow.md#test-scope-and-stability).
Unit tests call isolated code in process and check stable contracts/invariants;
they do not launch systems or depend on private implementation details.
Integration tests cover real component interactions and are capped at five
across the repository. A test runner's name does not determine test scope.

The forward-pass sources and comparison harness are implemented. Builds and
unit checks alone do not establish transformer equivalence: use the independent
GPU scenarios below. NLA computation and the browser/API remain later work.

The suite contains **eleven C++ unit executables, twelve Python unit tests and
five integration scenarios**. Fixtures and full weights stay in ignored caches
with recorded revisions, hashes and reference environment details.

[gpu/forward_reference.h](gpu/forward_reference.h) compares completed results:
exact input provenance, selected logits and configurable observation sites.
[support/numerical.h](support/numerical.h) computes array error metrics. Both
accept ordinary in-memory inputs and launch no model. The separate GPU driver
obtains completed results through the production loading/execution path.

## VS Code and compiler settings

Open the repository root as a VS Code folder, or add it as its own folder in a
multi-root workspace. Install the recommended C/C++ and CMake Tools extensions.
Choose **CMake: Select Configure Preset**, then `host-debug` on macOS/Linux or
`cuda-debug` in a Linux environment with CUDA installed. Configuration runs on
open; **CMake: Configure** reruns it after dependency or preset changes.

The [shared presets](../CMakePresets.json) also work from the terminal:

```sh
cmake --preset host-debug
cmake --build --preset host-debug --parallel
ctest --preset host-units
```

CMake exports `compile_commands.json` into each build directory. The VS Code
C/C++ extension uses CMake Tools as its configuration provider, so compiler,
language standard, defines and includes come from the selected targets. This
includes fetched JSON headers, OpenSSL and CUDA where enabled. No manually
maintained include-path list or machine-specific SDK path is checked in.
See [Microsoft's configuration-provider documentation](https://code.visualstudio.com/docs/cpp/configure-intellisense).

Use ignored `CMakeUserPresets.json` for local dependency locations, for example
a preset inheriting `host-debug` with `OPENSSL_ROOT_DIR` set to your OpenSSL
installation. CUDA editing requires the Linux CUDA environment (for example,
VS Code Remote SSH); the Mac host preset cannot provide an absent CUDA SDK.
Container-generated database paths must be consumed inside the corresponding
container environment, not copied to another host's editor.

Project C++ targets use C++20 without GNU language extensions and
`-Wall -Wextra -Wpedantic -Werror` with GCC/Clang. CUDA translation units use
`--Werror=all-warnings` for NVCC and forward `-Wall -Wextra -Werror` to the host
compiler. `-Wpedantic` is CXX-only because NVCC's generated host stubs contain
GNU line markers. These settings apply to tests as well as engine libraries.

## Host unit tests

The root CMake build requires C++20, CMake 3.24+ and OpenSSL 3 development files.
It compiles with strict warnings and fetches SHA-pinned nlohmann/json 3.12.0 on
initial configuration. These support hashing and metadata parsing; model
computation is implemented directly. CUDA and model downloads are unnecessary
for host unit checks. The standard entry point is:

```sh
python3 scripts/check.py host
```

Equivalent direct build commands are:

```sh
cmake -S . -B build/host -DCMAKE_BUILD_TYPE=Debug
cmake --build build/host --parallel
ctest --test-dir build/host --output-on-failure -L unit
```

The runner starts test executables; each test itself calls isolated units
directly in process, with no filesystem, subprocess, service or GPU dependency.
The executable names group assertions for one unit; they are not integration
workflows hidden behind a unit label.

| Unit | Contracts exercised |
| --- | --- |
| Tensor layouts | Scalar/empty/strided extents, byte containment, invalid metadata and arithmetic overflow |
| BF16 | All 65,536 stored encodings, exact widening/serialization, nearest-even boundaries and canonical NaNs |
| Observation preparation | Configurable sites/positions, owned metadata, moves and independent byte/row budgets |
| Scratch planning | Alignment, containment, inclusive lifetime safety, identity and budget rejection |
| Numerical comparison | Explicit tolerances, zero norms, nonfinite inputs and widened arithmetic |
| Operation validation | Mathematical shape/dtype/constant requirements and distinct operand access roles |
| Recipe validation | Single producers, read-before-write rejection, immutable inputs and complete output/site metadata |
| Qwen3 model | Configuration bounds, parameter shapes, advertised observations and requested-logit metadata |
| Safetensors parser | Unique keys, typed extents, complete payload coverage and bounded metadata |
| Qwen3 checkpoint binding | Supported config semantics, exact names/layouts, owned bindings and failure cleanup |
| Completed-result comparison | Provenance, shape, position and numerical acceptance contracts |

Recipe tests do not establish numerical equivalence to Qwen3. No CUDA kernel
or complete forward pass is executed here. Tests avoid private slot numbering,
operation counts, allocator offsets and diagnostic wording.

To check host memory and undefined behavior with Clang or GCC, use a separate
build directory:

```sh
cmake -S . -B build/host-sanitize -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined'
cmake --build build/host-sanitize --parallel
ctest --test-dir build/host-sanitize --output-on-failure -L unit --timeout 30
```

For UBSan-only checking, replace `-fsanitize=address,undefined` with
`-fsanitize=undefined` in both flags. Host sanitizer checks are separate from
CUDA numerical verification. Keep run logs and implementation reports local.

## Integration inventory

The repository uses **all five integration slots**:

| Scenario | Interaction |
| --- | --- |
| Site publication | Real Hugo build and publication boundary under a deployment subpath |
| `integration.checkpoint_archive` | Real file integrity, root containment, mapped storage and Qwen3 binding |
| `integration.device` | CUDA resource ownership, queue completion and buffer transfers |
| `integration.forward_tiny` | Tiny independent FP32/BF16 forward fixtures through the CUDA path |
| `integration.forward_8b` | Pinned Qwen3-8B forward fixture through the CUDA path |

The device scenario requires a CUDA build and NVIDIA GPU. The two forward
scenarios additionally require generated fixtures; CTest registers them when
`DOLITTLE_FORWARD_FIXTURES` is configured. `scripts/check.py gpu` accepts the
fixture root directly and runs the three GPU scenarios. This inventory records
scope, not a claim that every scenario has passed. Replace existing coverage
before adding another integration workflow.

The archive scenario creates only owned temporary files, verifies a complete
tiny safetensors/config/index set, binds the model and checks mapped-data
lifetime. Size/hash/index/path failures must prevent publication. It runs
without CUDA after a host build:

```sh
python3 scripts/check.py archive
# Or select its registered CTest entry.
ctest --test-dir build/host --output-on-failure -R integration.checkpoint_archive
```

## CUDA build and numerical verification

On Linux, build from the digest-pinned CUDA 13.2.0/Ubuntu 24.04 environment:

```sh
python3 scripts/check.py cuda-build
```

The image pins CMake, Ninja and OpenSSL development packages. Weights are never
part of the image. The CUDA build emits sm86 native code and retained PTX; no
successful execution on later NVIDIA architectures is implied. An installed
CUDA 13.2+ toolchain can build directly with `cuda-build --native`.

The Docker image builds successfully. The development host currently lacks
NVIDIA Container Toolkit configuration, so GPU verification uses native Linux
execution. Container GPU execution is unverified. With independent fixtures
prepared in the pinned reference environment, run both paths:

```sh
python3 scripts/check.py gpu --fixtures .cache/forward-fixtures
python3 scripts/check.py gpu --fixtures .cache/forward-fixtures --force-ptx
```

These commands write separate per-scenario JSON reports under `.cache/checks`.
Use `--container` only after Docker GPU access works. Fixture generation is
provided by [forward_reference.py](../scripts/validation/forward_reference.py);
see [script usage](../scripts/README.md#independent-reference-fixtures).

The numerical scenarios cover selected logits and block observations, causal
prefix behavior, capture-on/off equivalence and owned result lifetime. Exact
consumed token IDs avoid mixing tokenizer correctness into this slice. The
reference records environment, revisions, weights and numerical tolerances
before acceptance. Its dense FP32 attention-intermediate policy and stock-eager
sensitivity comparison are explicit; reduced-precision equations must not be
assumed identical merely because their algebra matches.

The 8B fixtures exercise full five-token, three-token prefix and alternate
five-token inputs, observing block input/output at 0, 18, 24 and 35. These
scenarios establish bounded numerical checks, not long-context coverage or
performance on every supported GPU. Keep generated reports in ignored caches.

CUDA memory checking can reuse the tiny forward scenario:

```sh
compute-sanitizer --tool memcheck --error-exitcode 99 \
  build/cuda/test_forward .cache/forward-fixtures/tiny-fp32/fixture.json \
  .cache/forward-fixtures/tiny-bf16/fixture.json
```

## Site verification

Site coverage consists of **one integration test and twelve unit tests**. The
integration builds a disposable site under a deployment subpath and checks its
publication boundary: allowlisted articles, draft/future exclusion, milestone
rendering and private/unlisted content isolation. It requires Python 3.11+ and
Hugo 0.154.2.

The unit tests call `check_site_snapshot` with small byte mappings. They preserve
the former negative guard coverage for broken anchors, responsive assets,
stale share identity, missing images, accidental documents and fonts, and also
check path boundaries, RSS membership and valid-snapshot immutability. They do
not create files, invoke Hugo or share the integration build fixture.

```sh
# All site coverage.
python3 -m unittest discover -s tests -p 'test_site.py' -v
# Isolated publication guard units only; Hugo is not needed.
python3 -m unittest discover -s tests -p 'test_site.py' -k SiteValidationUnitTests -v
# The one real site publication integration scenario.
python3 -m unittest discover -s tests -p 'test_site.py' -k PublicationIntegrationTests -v
```

`check_site` remains the filesystem-reading wrapper used by real builds; it
reads a snapshot and delegates to the same pure validator. Moving these guard
checks into unit tests does not remove their use in actual publication builds.
Each retained integration scenario covers one component interaction; isolated
negative guards do not consume integration slots.
