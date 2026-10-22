# Verification

The target numerical workflow includes isolated operation checks against
independent FP32 equations with explicit BF16 rounding. Their published
operation/dtype elementwise tolerances and scope are defined in the
[numerical portability contract](../docs/architecture/numerical-portability.md).
Reports keep these separate from the unchanged frozen full-model absolute and
relative L2 checks. Metal reports whole-model differences without failing on that budget; CUDA
continues enforcing it. Operation accuracy, finite/structurally valid outputs and
exact same-backend full/cached comparisons remain required. No additional integration scenario
is registered.

Follow the [test scope and stability rules](../docs/process/workflow.md#test-scope-and-stability).
Unit tests call isolated code in process and check stable contracts/invariants;
they do not launch systems or depend on private implementation details.
Integration tests cover real component interactions and are capped at five
across the repository. A test runner's name does not determine test scope.

The full-forward, cached generation and target-to-AV interpretation paths are implemented. Builds and
unit checks alone do not establish transformer equivalence: use the independent
GPU scenarios below. The browser/API use the live application integration below; the same workflow now includes AR reconstruction and matched steering replay.

The suite contains **twenty-two C++ unit executables, twenty Python unit tests, eighteen browser
unit tests and five integration scenarios**. Fixtures and full weights stay in ignored caches
with recorded revisions, hashes and reference environment details.

[gpu/forward_reference.h](gpu/forward_reference.h) compares completed results:
exact input provenance, selected logits and configurable observation sites.
[support/numerical.h](support/numerical.h) computes array error metrics. Both
accept ordinary in-memory inputs and launch no model. The separate GPU driver
obtains completed results through the production loading/execution path.

## VS Code and compiler settings

Open the repository root as a VS Code folder, or add it as its own folder in a
multi-root workspace. Install the recommended C/C++ and CMake Tools extensions.
Choose **CMake: Select Configure Preset**, then `host-debug` for host-only work, `metal-debug` on Apple Silicon or
`cuda-debug` in a Linux environment with CUDA installed. Configuration runs on
open; **CMake: Configure** reruns it after dependency or preset changes.

The [shared presets](../CMakePresets.json) also work from the terminal:

```sh
cmake --preset host-debug
cmake --build --preset host-debug --parallel
ctest --preset host-units
```

C/C++/CUDA use Google formatting with whole-file format-on-save through the
recommended C/C++ extension. Install the pinned formatter using the
[formatting setup](../scripts/README.md#formatting); `python3 scripts/check.py format`
and the formatting CI workflow enforce the same version and configuration.

CMake exports `compile_commands.json` into each build directory. The VS Code
C/C++ extension uses CMake Tools as its configuration provider, so compiler,
language standard, defines and includes come from the selected targets. This
includes fetched JSON/Unicode headers, OpenSSL and CUDA where enabled. No manually
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

Clang-tidy is a separate static-analysis gate with the same check families as
Serenity and findings promoted to errors. Use `python3 scripts/check.py tidy`
after configuring the host preset, or the Linux analysis container for the
CUDA build. See [static-analysis setup and scope](../scripts/README.md#static-analysis)
for pinned versions, editor tasks, CUDA compatibility and reports. Its four
command-adaptation unit tests use in-memory inputs, with no compiler or GPU.

## Checked generation allocation contract

Configure the CUDA verification build with `-DDOLITTLE_ALLOCATION_CHECKS=ON`.
The existing device, generation and application scenarios then enforce zero C++
heap and explicit GPU/staging allocations on successful device-generation
iterations, including prompt publication, token decoding and event publication.
A violation aborts the check. Prepared storage, final records, exceptional failure
reporting and HTTP reader threads are outside this scope; arbitrary C allocation
inside shared libraries is not intercepted. Normal builds leave this diagnostic
off. No extra integration scenario is introduced.

The device scenario also completes a prepared sequence before collecting any
output, checks cancellation/EOS/final consumption, and verifies that invalid GPU
normalization prevents inference. The existing tiny/8B target-to-AV scenarios
compare GPU normalization on frozen source rows, export completed captures only
for numerical diagnostics, and verify retained device ownership across model
switches. Native and forced-PTX runs use the unchanged fixture hashes and limits.

## Host unit tests

The root CMake build requires C++20, CMake 3.24+ and OpenSSL 3 development files.
It compiles with strict warnings and fetches SHA-pinned nlohmann/json 3.12.0,
utf8proc 2.10.0 and Oniguruma 6.9.10 on initial configuration. A C compiler is
also required for the Unicode libraries. These supply metadata parsing, NFC and
Unicode regex primitives; hashing uses OpenSSL. BPE and model computation are
implemented directly. Dependency licenses are recorded in `NOTICE`. CUDA and model downloads are unnecessary
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
| NLA conversion | FP32 normalization, immutable input, BF16 rounding and explanation delimiter ranges |
| Experiment record | Retained row ownership, stable variants, atomic rejection and count/byte budgets |
| Categorical sampling | Greedy ties, top-k/top-p boundaries, masked inputs, one-draw RNG semantics and independent streams |
| Sequence progression | Emitted versus consumed counts, stop precedence and final-token consumption |
| Byte BPE | Ranked merges, normalization, literal added tokens, raw-byte provenance and bounds |
| Qwen3 chat | Exact prompt bytes, optional system messages, thinking modes and delimiter classification |
| Checked host bounds | Empty/end/oversized rejection, borrowed reference identity and constness |
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
| `integration.application` | Source launcher, verified checkpoints, live token/capture stream, AV selection, idempotent retries and reconnect |
| `integration.device` | CUDA resource ownership, queue completion, transfers, device selection and autonomous decode control |
| `integration.generation_tiny` | Tiny FP32/BF16 target generation, retained captures, AV replacement and fresh-cache interpretations |
| `integration.generation_8b` | Pinned tokenizer/protocol, Qwen3-8B target generation and matched AV interpretation |

The device scenario also starts a bounded decode graph and drains it before any
CPU collection. Exact seeded IDs, positions, budget, EOS and cancellation verify
that output observation cannot be required for device progress. This remains part
of the device resource/dispatch scenario; transformer numerics use the independent
fixtures below.

The device scenario requires a CUDA build and NVIDIA GPU. The two generation
scenarios additionally require generated fixtures; CTest registers them when
`DOLITTLE_GENERATION_FIXTURES` is configured. `scripts/check.py gpu` accepts the
fixture root directly and runs the three GPU scenarios. This inventory records
scope, not a claim that every scenario has passed. Replace existing coverage
before adding another integration workflow.

The application scenario replaces the standalone archive integration. It verifies
setup receipt publication/reuse, invalidation for changed or missing files,
forced verification, successful production checkpoint acquisition/binding, and
one actual prompt through AV interpretation, retry/reconnect and orderly shutdown. It does not preserve all
adversarial filesystem cases from the retired executable. Parser/bounds units remain.
Use an existing pinned revision cache with either the source-built Linux/CUDA
executable or Docker Compose (which builds the application from source):

```sh
python3 tests/integration/test_application.py --binary build/cuda/dolittle --cache models
python3 tests/integration/test_application.py --container --cache models
node --test tests/app/*.test.mjs
```

CTest registers the application scenario when `DOLITTLE_APPLICATION_CACHE` is
configured. CTest uses the native developer path. The script uses `--offline`
in both modes and writes logs only under `.local/`; its report identifies the
execution mode. Only a successful `--container` run establishes Docker application
GPU execution; it requires host NVIDIA Container Toolkit configuration.
Browser visual/interaction inspection follows this same application flow; no
separate mock-backend integration suite is introduced.

## CUDA build and numerical verification

### Target-to-AV verification

The two generation scenarios retain independent full-forward comparisons and
extend them through uneven cached prefill and one-token decode. The generator
records a four-token greedy trace, then recomputes its complete consumed prefix
without caching to produce logits and raw observations at every position.
Oracle arithmetic is explicit in fixture provenance; numerical budgets do
not adapt to custom-engine results. The harness compares chunk-final logits and every captured row,
including the final emitted token, against those independent values.
It also executes the same complete trace without caching and requires each
cached chunk's logits and captures to match that path byte for byte. This
separates cache/positioning regressions from differences against the independent
reference. Numerical failures retain their output/position, measured error and
fixed limit in diagnostics and cause a nonzero exit.

The acceptance oracle uses BF16/FP32 storage and fixed-shape independent FP32
library projections and dense causal attention. It requires stored-byte
invariance across prefix lengths and peer-batch positions. Engine changes
must pass that reference and the original budgets. Agreement with a rewritten reference that follows the engine's
arithmetic cannot substitute for this gate.

It also checks capture-on/off and one-row/chunked equivalence. Tiny workloads
exercise stop IDs, zero generation budget, cooperative cancellation and a
selector failure, preserving completed captures. Exact greedy IDs are checked
for these bounded fixtures; this is not a claim of identical sampled prose
across arbitrary floating-point implementations. Categorical RNG semantics are
covered by isolated unit contracts, independently of Transformers' RNG.

The 8B scenario verifies 405 exact reference tokenizer ID vectors and 12 prompt
templates (thinking enabled/disabled, absent/empty/nonempty system message).
Vectors include NFC, Unicode scripts, whitespace, embedded NUL, contractions,
added tokens, overlapping byte merges and seeded Unicode strings. The fixture
records tokenizers 0.22.2; the native Unicode dependencies are source-pinned.
Passing these vectors establishes bounded compatibility evidence, not an
exhaustive proof for every Unicode string or future tokenizer version.

Reports separate preparation/upload, host-ready prefill/decode calls, complete
generation and diagnostic comparison time. They include admitted device bytes
and actual capture/logit readback byte counts. A host-ready call includes host
allocation, submission, transfer and completion wait; it is not isolated kernel
time. Isolating transfer/wait costs and representative long-context profiling
remain performance work. Reports and measurements stay in ignored storage.

The same two workflows continue through the AV. A separate frozen reference
supplies independently normalized target rows, exact injection bytes, AV logits
and selected block inputs/outputs. Full replay and cached prefill/decode receive
identical reference inputs; the injection occurs in a later prefill chunk.
The production Run also generates the target, retains its capture, releases
target GPU resources, and produces two greedy AV variants with fresh caches
and retained AV weights. It checks cancellation before execution, rejected
capture keys and closed-run calls. Raw captures survive model release.

These are four-token numerical fixtures, not tests of explanation quality.
Incomplete explanation tags remain explicit in the record. Reference prose
agreement is not an accuracy gate. See the [AV fixture setup](../scripts/validation/README.md#independent-av-fixtures).

### Building and running the scenarios

On Linux, build from the digest-pinned CUDA 13.2.0/Ubuntu 24.04 environment:

```sh
python3 scripts/check.py cuda-build
```

The image pins CMake, Ninja and OpenSSL development packages. Weights are never
part of the image. The CUDA build emits sm86 native code and retained PTX; no
successful execution on later NVIDIA architectures is implied. An installed
CUDA 13.2+ toolchain can build directly with `cuda-build --native`.

The Docker image builds successfully. On the 3090 Ti, NVIDIA Container Toolkit
1.20.1 enables the verified Docker application flow: target generation, capture
selection, AV interpretation, retry/reconnect and clean shutdown. Independent
numerical comparisons have passed through native Linux execution and forced PTX;
those results are distinct from the container application check. With independent
fixtures prepared in the pinned reference environment, run both paths:

```sh
python3 scripts/check.py gpu --fixtures .cache/generation-fixtures
python3 scripts/check.py gpu --fixtures .cache/generation-fixtures --force-ptx
```

These commands write separate per-scenario JSON reports under `.cache/checks`.
Use `--container` only after Docker GPU access works. Fixture generation is
provided by [forward_reference.py](../scripts/validation/forward_reference.py);
see [script usage](../scripts/README.md#independent-reference-fixtures).

The numerical scenarios cover selected logits and block observations, causal
prefix behavior, cached generation, capture-on/off equivalence and owned result
lifetime. Tokenizer parity is checked before device execution. The
reference records environment, revisions, weights and numerical tolerances
before acceptance. Its accumulation and rounding policies and stock-eager
sensitivity comparison are explicit; floating-point implementations must not
be assumed identical merely because their algebra matches.

The 8B fixtures retain five-token, three-token prefix and alternate-token
comparisons, then generate from the real thinking-enabled chat prompt
“What is 2 + 2?”. They observe block input/output at 0, 18, 24 and 35. These
scenarios establish bounded numerical checks, not long-context coverage or
performance on every supported GPU. Keep generated reports in ignored caches.

CUDA memory checking can reuse the tiny generation scenario:

```sh
compute-sanitizer --tool memcheck --error-exitcode 99 \
  build/cuda/test_generation .cache/generation-fixtures/tiny-fp32/fixture.json \
  .cache/generation-fixtures/tiny-bf16/fixture.json
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

## Native Metal application

The `metal-debug` and `metal-release` presets build the application and its
verification drivers on Apple Silicon with macOS 26+, Xcode's Metal compiler
and OpenSSL 3. CMake fetches pinned `metal-cpp` headers.

```sh
cmake --preset metal-debug -DDOLITTLE_GENERATION_FIXTURES=/path/to/frozen-fixtures
cmake --build --preset metal-debug --parallel
ctest --test-dir build/metal-debug --output-on-failure
python3 scripts/check.py tidy --build-dir build/metal-debug
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 \
  build/metal-debug/test_device
```

Metal reuses the existing device, tiny, 8B and application workflow slots. The
model workflows include AV interpretation and AR steering. Use `--target-only`
only for a focused target diagnostic. Run the complete source launcher flow with:

```sh
python3 tests/integration/test_application.py --native-source --cache models
```

For Metal target comparisons, use fixtures from the independent oracle running through
MPS on the same Apple GPU; see [reference setup](../scripts/README.md#independent-reference-fixtures).
Their weights, tokens, selections and numeric limits match the original CUDA
fixtures, while expected values come from MPS. The frozen CUDA comparison
remains a cross-backend diagnostic. The MPS oracle is frozen at batch size one:
exact replay and prefix identity are required; peer-batch variation is reported
separately. Run the default native tests as well as the validation-layer
invocation above. Current validation status is recorded in the work queue.

Row reports retain their frozen absolute/relative-L2 verdicts and include
additional `diagnostics`: cosine similarity, actual/reference norm ratio and
NMSE. Logit rows also report temperature-one KL(reference || actual), total
variation, argmax IDs and their reciprocal ranks. These reuse completed test
readbacks outside inference timing. They do not change acceptance limits or
oracle precision. See [whole-model comparisons](../docs/architecture/numerical-portability.md#whole-model-comparisons)
for the accepted policy. Whole-model limit results remain visible; operation
accuracy, finite outputs and exact same-backend replay determine acceptance.
NLA fixtures retain their independent CUDA provenance; no MPS NLA baseline is
claimed. When relocating fixtures, change only their model-root paths and keep
all expected values, weights, source identities and limits unchanged.

The device workflow checks native
resource/view transfers, prepared autonomous progress with collection delayed
until completion, capture positions, final-token consumption, EOS, forward-only
execution and cancellation, full-vocabulary GPU sampling, RNG continuation
and independent scalar-math vectors. Three pure units cover model retention,
view placement and bounded sequence planning. The integration inventory stays
at five; backend selection changes the device implementation under test.

API validation covers the whole Metal workflow. Shader instrumentation covers
direct control, publication and cancellation dispatches. Apple requires
[pipeline and buffer inheritance for instrumented ICBs](https://developer.apple.com/documentation/xcode/validating-your-apps-metal-shader-usage).
Dolittle's reusable bodies own those bindings, so their pipelines explicitly
disable unsupported shader instrumentation. Their results remain covered by
device assertions; this is not a claim of instrumented ICB-kernel coverage.
