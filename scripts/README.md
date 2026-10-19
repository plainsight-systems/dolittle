# Build and verification tooling

[`check.py`](check.py) is the current source-build and verification entry point.
It uses Python's standard library and keeps host checks, CUDA compilation,
checkpoint integration and GPU execution separate:

```sh
python3 scripts/check.py host
python3 scripts/check.py archive
python3 scripts/check.py cuda-build
python3 scripts/check.py gpu --fixtures .cache/generation-fixtures
python3 scripts/check.py gpu --fixtures .cache/generation-fixtures --force-ptx
```

`host` needs CMake 3.24+, a C++20 compiler and OpenSSL 3 development files. Initial
CMake configuration fetches hash-pinned nlohmann/json 3.12.0, utf8proc 2.10.0
and Oniguruma 6.9.10 unless already cached; a C compiler builds the Unicode
helpers. BPE and model computation remain direct implementations.
It builds the host targets and runs isolated C++ units. `archive` runs the real
file-integrity/binding scenario from that build; run `host` first.

`cuda-build` builds a Linux/amd64 CUDA 13.2.0 development image pinned by manifest
digest in the root [Dockerfile](../Dockerfile), then compiles the CUDA targets.
The Docker context contains only that Dockerfile: source and build output are
mounted for compilation, while weights remain outside the image. CMake, Ninja
and OpenSSL development package versions are explicit in the image. Native
CUDA 13.2+ builds are also available with `cuda-build --native`.

`gpu` executes the device, tiny target-to-AV and pinned 8B target-to-AV scenarios on a
compatible Linux NVIDIA host. It writes per-scenario JSON reports under
`.cache/checks` by default. `--force-ptx` sets `CUDA_FORCE_PTX_JIT=1`; reports from
that run are separate from native-code reports. Use `--container` only with
working Docker GPU access. Container execution requires the NVIDIA Container Toolkit; native execution
requires compatible host CUDA libraries. See [verification commands](../tests/README.md)
for the reference policy and scenario coverage. Keep reports in ignored storage.

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

## Later application launch

The planned `run.py` entry point will check Docker/GPU prerequisites, acquire
and verify declared model files in a persistent cache, and launch the application.
It is not implemented in this interpretation slice. Keep downloads out of image
builds and never reset a user's checkout or existing model cache.

## Project site

`python3 scripts/site.py serve` builds and serves the public project site on
loopback, independently of the GPU application. It requires Python 3.11+ and
Hugo 0.154.2. See [site setup](../web/README.md) for publication rules and URLs.
The same tooling validates generated files through a pure in-memory snapshot
checker, with a filesystem wrapper used after actual Hugo builds.
