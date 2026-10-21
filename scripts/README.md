# Build and verification tooling

[`check.py`](check.py) is the current source-build and verification entry point.
It uses Python's standard library and keeps host checks, CUDA compilation,
application integration and GPU execution separate:

```sh
python3 scripts/check.py host
python3 scripts/check.py cuda-build
python3 scripts/check.py gpu --fixtures .cache/generation-fixtures
python3 scripts/check.py gpu --fixtures .cache/generation-fixtures --force-ptx
```

`host` needs CMake 3.24+, a C++20 compiler and OpenSSL 3 development files. Initial
CMake configuration fetches hash-pinned nlohmann/json 3.12.0, utf8proc 2.10.0
Oniguruma 6.9.10 and cpp-httplib 0.62.0 unless already cached; a C compiler builds the Unicode
helpers. BPE and model computation remain direct implementations.
It builds the host targets and runs isolated C++ units. The live application
scenario replaces the former standalone archive integration; see tests/README.md.

`cuda-build` builds a Linux/amd64 CUDA 13.2.0 development image pinned by manifest
digest in the root [Dockerfile](../Dockerfile), then compiles the CUDA targets.
The default build target supplies a toolchain; source and output are mounted
for development compilation. The explicit runtime target copies source and
assets and builds the service inside the image. Weights remain outside both. CMake, Ninja
and OpenSSL development package versions are explicit in the image. Native
CUDA 13.2+ builds are also available with `cuda-build --native`.

`gpu` executes the device, tiny target-to-AV and pinned 8B target-to-AV scenarios on a
compatible Linux NVIDIA host. It writes per-scenario JSON reports under
`.cache/checks` by default. `--force-ptx` sets `CUDA_FORCE_PTX_JIT=1`; reports from
that run are separate from native-code reports. Use `--container` only with
working Docker GPU access. Container execution requires the NVIDIA Container Toolkit; native execution
requires compatible host CUDA libraries. See [verification commands](../tests/README.md)
for the reference policy and scenario coverage. Keep reports in ignored storage.

## Formatting

C, C++ and CUDA use the unmodified Google preset in the root `.clang-format`,
matching Serenity. clang-format **21.1.8** is pinned across editor saves, local
checks and CI; this version is independent of Linux's clang-tidy requirement.
Install the small development-only tool environment from the repository root:

```sh
python3 -m venv .venv-tools
.venv-tools/bin/python -m pip install -r scripts/requirements-tools.txt
python3 scripts/check.py format        # Read-only; fails on formatting drift.
python3 scripts/check.py format --fix  # Apply Google formatting to project sources.
```

The checker includes C/C++/CUDA sources and headers under `src/` and `tests/`,
including new untracked files. Build output and dependencies are excluded.
The **C++ formatting** GitHub workflow runs the read-only check on PRs and pushes
to `main`. It does not require a compiler, GPU, model weights or a CMake database.

Open Dolittle itself as a VS Code workspace folder and install the recommended
Microsoft C/C++ extension. Workspace settings select that extension as formatter
and enable whole-file format-on-save for C, C++ and CUDA (`.cu` and `.cuh`).
Reload the window if it was already open. Editor saves call `scripts/clang-format`,
which resolves the same pinned binary as the checker. It preserves the extension's
stdin and arguments, including the file path needed to find `.clang-format`.

Discovery checks `CLANG_FORMAT` when explicitly set; otherwise it checks the
local tool environment, PATH and Homebrew's `llvm@21`. A missing or mismatched
formatter produces an installation message; it never installs tools on save or
silently switches to the extension's bundled version. No machine-specific paths
are committed. The existing CMake include/completion setup is unchanged.

VS Code tasks **Dolittle: check formatting** and **Dolittle: format sources**
run the same commands. clang-tidy remains the separate static-analysis gate below.

## Static analysis

The [clang-tidy policy](../.clang-tidy) follows Serenity: Google, C++ Core
Guidelines, bug-prone, performance and Clang analyzer checks, with every finding
an error. Test fixtures may use literal expected values through
[`tests/.clang-tidy`](../tests/.clang-tidy). No findings baseline is suppressed.
This is a separate analysis command; it does not rewrite source or change the
compiler used for normal builds. Host span access uses the checked helper in
`src/bounds.h` because C++20 spans lack `at()`. Local annotations explain the
remaining native CUDA indexing, C API byte views, destructor cleanup and deliberate
negative-test exceptions; they do not disable those checks elsewhere.

On macOS, use LLVM **21.1.8** (matching Serenity), configure the host preset, then:

```sh
cmake --preset host-debug
python3 scripts/check.py tidy --build-dir build/host-debug
```

Set `CLANG_TIDY` or pass `--clang-tidy /path/to/clang-tidy` if necessary. The runner
checks the version and discovers the selected Xcode SDK/libc++ headers; no local
SDK paths are tracked. A host database covers only the configured host targets.

On Linux, use LLVM **22.1.8** for CUDA 13.2 header compatibility. The optional
analysis Docker stage provides pinned Clang packages without changing the
default build image. Build the CUDA targets first so NVCC response files exist:

```sh
python3 scripts/check.py cuda-build
python3 scripts/check.py tidy --container --build-dir build/cuda
```

GPU access and model weights are unnecessary for static analysis. Native Linux
LLVM 22.1.8 is also supported by omitting `--container`. Consume a compilation
database in the environment where its absolute paths resolve; reconfigure and
build after moving a checkout. Do not copy databases between machines. Container
build and report directories must be inside the mounted checkout.

The runner checks project C++ and CUDA compilation entries and their included
headers, excluding dependency translation units. CUDA gets separate host and
device AST passes. It expands NVCC response files and translates driver options
in temporary databases, preserving includes, defines, C++20, sm86 and disabled
implicit FMA. CMake's database stays untouched. Unknown options fail visibly.

LLVM 22.1.8 contains the CUDA 13 texture-header and CUDA 13.2 rsqrt declaration
fixes in its [CUDA wrapper](https://github.com/llvm/llvm-project/blob/llvmorg-22.1.8/clang/lib/Headers/__clang_cuda_runtime_wrapper.h),
but still reports 12.9 as its latest partially supported SDK. The runner checks
for toolkit 13.2 and suppresses **only** that SDK-version notice. All parsing
errors and code findings remain fatal. This establishes analysis coverage, not
Clang code-generation support or GPU numerical correctness; production uses NVCC.

Diagnostics print to the terminal and are saved with `summary.json` under
`.local/clang-tidy/`. `--report-dir` changes that ignored destination; `--jobs`
controls concurrency. A nonzero exit means findings or a tooling failure; a
successful invocation alone is not a clean result. New code and cleanup must
pass the applicable scope before being described as clang-tidy clean.

VS Code's **Tasks: Run Task → Dolittle: clang-tidy** uses the selected CMake build
directory and sends diagnostics to the Problems panel. It requires the pinned
native analyzer. Use the container command above when analyzing a Docker build.
The CMake Tools/C++ completion configuration is unchanged.

The command-adaptation unit tests are pure in-memory checks:

```sh
python3 -m unittest discover -s tests -p test_tidy.py
```

## Independent reference fixtures

For native Metal comparisons, run the independent oracle on the same Apple GPU.
[`validation/mps_reference.py`](validation/mps_reference.py) imports the existing
frozen reference functions unchanged. It verifies and reuses checkpoint files,
token traces and numeric limits from existing fixtures, then freezes separately
computed MPS outputs. Its output root must be new; it refuses to overwrite an
existing fixture directory.

```sh
python3.12 -m venv .local/mps-oracle-venv
.local/mps-oracle-venv/bin/python -m pip install -r scripts/validation/requirements-mps.lock
.local/mps-oracle-venv/bin/python scripts/validation/mps_reference.py \
  --output .local/mps-fixtures \
  .cache/generation-fixtures/tiny-fp32/fixture.json \
  .cache/generation-fixtures/tiny-bf16/fixture.json \
  .cache/generation-fixtures/qwen3-8b-bf16/fixture.json
```

This requires native MPS and the pinned PyTorch 2.7.1 macOS wheel. CPU fallback,
MPS fast math and forced alternate matmul kernels must be disabled. The wrapper
unsets `PYTORCH_MPS_PREFER_METAL`: PyTorch 2.7.1 enables that option when the
variable exists, including when its value is `0`. It records the literal
environment values and PyTorch git revision. `PYTORCH_MPS_FAST_MATH=0` controls
PyTorch-built Metal shaders; it does not expose Apple's MPSGraph compiler policy.
The wrapper checks parameter placement/dtype, reference source hashes, exact single-sequence
replay/prefix invariance and the original greedy trace before publishing each
fixture. Peer-batch differences are recorded as diagnostics: Dolittle's execution
contract covers one sequence, and MPS does not provide peer-batch identity here. A
`frozen.json` manifest records fixture/source hashes, device, OS and package
versions. Configure the Metal test build with this new fixture root. Preserve
the CUDA root for CUDA acceptance and cross-backend diagnostic comparisons.

CUDA fixture generation continues to use the original script below.

[`validation/forward_reference.py`](validation/forward_reference.py) generates
fixtures without calling Dolittle, using the pinned verification environment and
CUDA by default (`--device cpu` also generates independent reference values
without occupying the GPU). It verifies existing 8B weights against `models-8b.json`; it does not fetch
missing weights. The generated files record provenance, dtype, consumed-token
positions and numerical tolerances. For example:

```sh
python3 scripts/validation/forward_reference.py \
  --output .cache/generation-fixtures --kind all \
  --actual-checkpoint /path/to/verified/Qwen3-8B
```

The generator uses independent fixed-shape FP32 library projections and dense
causal attention, with BF16 storage for the actual model. It requires exact
stored-byte invariance across prefixes and peer batches. Keep this acceptance
reference and its error-budget rules fixed while correcting the engine.
FP64 is not an acceptance oracle. See `validation/README.md`.

The two model workflows now also require sibling `.cache/av-fixtures` files
created by [av_reference.py](validation/av_reference.py). Follow the
[AV fixture commands](validation/README.md#independent-av-fixtures), including
the supplemental pinned target metadata. The runner verifies the target fixture,
AV generator and original oracle hashes recorded by each AV fixture before GPU
execution. It never regenerates or replaces a reference to clear a failure.
Container mode mounts the AV fixture/model roots read-only in addition to the
target fixtures; native and container paths must resolve to the recorded files.

## Project site

`python3 scripts/site.py serve` builds and serves the public project site on
loopback, independently of the GPU application. It requires Python 3.11+ and
Hugo 0.154.2. See [site setup](../web/README.md) for publication rules and URLs.
The same tooling validates generated files through a pure in-memory snapshot
checker, with a filesystem wrapper used after actual Hugo builds.

## Application launch

`python3 scripts/run.py` builds from source: Docker Compose on Linux/NVIDIA,
or the native `metal-release` CMake preset on Apple Silicon. It prepares pinned
artifacts once, starts the service and opens its private browser link. Native
Metal requires macOS 26+, Xcode 26 with the Metal toolchain, CMake and OpenSSL 3. Subsequent launches reuse the setup receipt when the
manifest and file metadata match, without hashing weight payloads.
`--verify-models` forces full verification/repair before launch.
`scripts/models.py --offline` fully verifies an existing cache
without downloads. Python 3.11+ is required. See [application setup](../app/README.md)
for options and the explicit native execution path. Model acquisition uses bounded
streaming downloads, declared sizes/SHA256 and atomic publication. It never
executes downloaded model code or fetches an application binary.
