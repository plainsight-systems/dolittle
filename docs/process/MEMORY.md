# Project memory

## Identity

- **Project:** Dolittle, named after Doctor Dolittle.
- **Parent entity:** Plainsight Systems LLC; no separate operating brand.
- **Status:** project site, CUDA forward/generation, target-to-AV interpretation
  and the live browser/API are implemented. Bounded independent numerical checks
  and the Docker application flow run on the 3090 Ti. Reconstruction remains later work.
- **Remote:** `origin` is https://github.com/plainsight-systems/dolittle.git.
  The site’s Code link points to this repository. Andrew authorized the first
  push and Pages deployment on 2026-10-16.
- **Production site:** https://dolittle.plainsight-systems.com/ (deployed; HTTPS certificate pending at first verification).

## Decisions made 2026-10-16

- Motivation, clarified by Andrew: he read Anthropic’s NLA paper and thought
  it would be a fun experiment to build. He already understood thinking tokens;
  discovering them in Charlotte did not motivate Dolittle. Keep his broader
  position on whether those tokens matter separate from the project’s origin.

- Demonstrate understanding through a direct implementation of inference and
  natural language autoencoders, with an approachable interface. Novel research
  is not the project's objective.
- Show target-model thinking text beside NLA interpretations aligned with
  capture positions in the same generation. Interpretations are imperfect;
  they are separate from the target's generated reasoning.
- C++20 and CUDA on Linux, built with CMake. Docker will provide the pinned
  build and runtime environments. Exact versions remain to be selected.
- Develop and verify on a 24 GB Ampere GPU. The actual machine reports an
  RTX 3090 Ti (verified 2026-10-16), correcting the initial 3090 shorthand.
  Other hardware claims distinguish tested configurations from expected
  compatibility.
- Retain baseline PTX for every kernel alongside native 3090 code. Exercise
  `CUDA_FORCE_PTX_JIT=1`. Newer NVIDIA GPUs, including RTX 40- and 50-series,
  are intended to use the compatible code path, subject to driver and memory
  requirements. Architecture-specific optimization must keep that baseline.
- Distribute source builds, not prebuilt application executables. Published
  model weights remain inputs. A single entry point will build, fetch/cache
  weights, and launch the application.
- Use Python for the setup/launch entry point, following Cradle. Pin the build
  environment and dependencies; the host supplies Docker, the NVIDIA driver,
  and container GPU access.
- Keep model downloads separate from image builds. Track model revisions,
  required files, hashes, and provenance in a manifest; keep weights in a
  persistent ignored cache.
- Give ordinary build/non-GPU checks and 3090 execution checks stable,
  separate commands. Hosted build success does not establish GPU correctness.
- Serve `app/` through the engine deployment. A browser on another machine,
  including a Mac, can access the Linux deployment. `web/` is the separate
  public project site, now implemented as a local static build. The live
  inference interface is implemented under the accepted live-application design.
- Prioritize the project site before engine architecture. Follow Cradle’s
  explanatory format and family styling, with explanations, explorations, and
  Serenity-style milestones sharing a Writing index. Milestones support the
  writing rather than defining its sequence. Use Hugo 0.154.2, self-hosted Plex
  fonts, explicit article publication, and a separate loopback preview command.
- Platform plan updated by Andrew: Linux/NVIDIA first, Metal on macOS afterward.
  AMD support is undecided. Metal build/runtime details and hardware requirements
  remain future design work. Browser-only inference stays out of scope.
  The audience can reasonably be expected to have capable GPU hardware.
- Andrew selected **Qwen3 8B with marco-bazzani/Qwen3-8B-nla** after the
  comparison. Target revision: `b968826d9c46dd6066d109eabc6255188de91218`;
  NLA revision: `12c6ed1b1e684bdb8a788c989e2f3d15f9790db3`. File hashes are
  pinned in [models-8b.json](../../scripts/validation/models-8b.json).
  Capture the output of zero-based block 24. The full verbalizer and
  28-block reconstructor with a separate value head have distinct weights.
  Sequential BF16 reference execution fits the 3090 Ti; design scheduling
  around these actual weights and measured costs. The earlier shared-base
  4B experiment remains comparison evidence. Semantic faithfulness is
  unverified; agreement with the reasoning trace is not a gate.
- Follow Serenity's modified engineering workflow, written locally in
  `workflow.md`. No imported governance files, submodule, or symlinks.
- Architecture comes first, then file-header contracts and design. No packets.
  Design optimizations in, derive data-path costs first, verify on the GPU,
  and use independent review on request.
- Preserve the source brief locally as ignored `brief.md`, following Serenity.
- Code/configuration: Apache-2.0. Prose/original artwork: CC BY 4.0.
- Create project art before the first commit. The initial banner is
  `docs/assets/dolittle-readme.png`; its prompt is recorded alongside it.
- Repository setup recommendations accepted, with `web/` explicit and all
  workflow guidance local. Toolchain versions and architecture remain upcoming
  design work; the 8B model pairing is selected.

## Validation findings, 2026-10-16

- The actual host is an RTX 3090 Ti (24,564 MiB, sm86), driver 595.91.07.
- One BF16 Qwen3 4B base with both FP32 adapters holds 7.739 GiB of parameters.
  This is reference evidence for the layout, not a custom-engine measurement.
- The initial interpretation verdict was too strong. Descriptions contain
  transcript errors and unverified claims about internal representations.
  Disagreement alone establishes neither NLA failure nor hidden computation.
  Neither the reasoning trace nor NLA output is ground truth for the other.
- Andrew confirmed the framing: Dolittle is an application demo that makes
  both views accessible, including their agreement, divergence, and uncertainty.
  Validate computation and alignment; do not require matching narratives or
  turn the project into a novel interpretability research effort.
- The reference uses PyTorch/Transformers/PEFT only for verification. The direct
  C++/CUDA implementation decision is unchanged.
- Docker initially lacked NVIDIA Container Toolkit/GPU access. Host setup was
  completed with Toolkit 1.20.1; the Docker application flow now passes on the
  3090 Ti, including target generation, AV interpretation, reconnect and shutdown.
- Source revisions, manifests, full outputs, costs, and caveats live in the
  [feasibility note](../research/2026-10-16-nla-validation/README.md).

## Decisions made 2026-10-17

- Andrew agreed to an experiment overlay that generates and captures first,
  then interprets selected positions after switching from target to AV.
  Reconstruction is optional and explicit. Start with one active experiment
  on one GPU; pause–interpret–resume remains later work.
- Retain exact tokens and raw block-24 captures. Align each capture with the
  token consumed, not the token predicted. Make sampled AV variants and
  neighboring-position comparison available; neither recurrence nor vector
  reconstruction is semantic confidence.
- Keep the experiment flow and transformer execution in separate algorithmic
  overlay documents. The latter covers block mathematics, prefill/decode,
  the different target/AV/AR paths, and observation boundaries across backends.
  Derive change axes after reviewing both, then map files. Backend variation
  must preserve capture semantics while remaining distinct from model changes.
- The paper comparison supports frozen-target capture, embedding injection,
  sampled verbalization, and optional AR use. Its causal interventions are
  beyond the initial scope; the community 8B pair does not inherit the paper's
  empirical results. Layer 24 is a checkpoint contract, not a universal best
  layer for interpreting every behavior.

The [experiment overlay](../architecture/logical-overview.md) and
[transformer overlay](../architecture/transformer-execution.md) were reviewed
and accepted by Andrew on 2026-10-17. They establish the design baseline for
change-axis analysis, followed by file mapping. No engine implementation or
new GPU validation accompanies them.

A [change-axis design](../architecture/change-axes.md) was drafted at Andrew's
request after overlay acceptance. It separates model/input semantics,
generation and experiment behavior, execution planning, KV storage, numerical
representation, device runtime, and operation implementations per backend.
Application, build/launch, verification and measurement concerns are included.
Andrew liked the design and authorized file/folder mapping, while expressing
concern that overly specific axes could produce unnatural groupings. Sampling
was his example; the subsequent mapping review tested and revised the boundaries.

The revised [file and folder mapping](../architecture/file-mapping.md) and
corresponding change-axis clarifications were accepted by Andrew on
2026-10-17. The review corrected two structural issues:

- Keep computation under `kernels/cuda/` and device resources under
  `gpu/cuda/`, with corresponding Metal locations deferred.
- The earlier generic tokenizer/checkpoint/cache layout overfit the selected
  model. Families and their kinds must be visible in the folder structure;
  supporting only one initially is not a reason to hide the extension points.

The accepted mapping separates `models/qwen3/`, `tokenizers/`, tensor formats and
representations, `inference/cache/`, `inference/sampling/`, NLA pairs and metric
implementations. Shared contracts are distinct from named concrete kinds.
This supersedes the earlier single sampler pair directly under inference and
generic model/tokenizer/cache implementations. No alternate kinds, source
stubs or engine code were built. File-header design and numerical verification
planning are next. Cache and sampling baseline choices still require the
cost/verification work stated in the mapping; layout acceptance is not a
runtime or performance result.

## Current implementation

The engine implements verified Qwen3 checkpoint loading, full-forward CUDA
execution and target generation: byte BPE, Qwen3 single-turn protocol,
categorical/greedy sampling, contiguous persistent KV, chunked prefill/decode,
stop handling and final consumed-token captures. Observation sites remain
configurable; shared execution does not hardcode block 24. The AV continuation
adds one-row injection, retained records and sequential model scheduling.
Reconstruction and the application remain future work.

The first forward slice is accepted and pushed as `77f038c`. Target generation
and subsequent kernel corrections are verified and pushed as `20d8740`.
The restored independent
BF16/FP32 oracle now passes: tiny FP32, tiny BF16 and pinned Qwen3-8B pass native
and forced-PTX execution on the 3090 Ti. All 485 measured 8B numerical comparisons
have zero absolute/relative error; cached/full-trace bytes and expected generated
IDs also agree. Strict builds, 15 host units and tiny CUDA memcheck pass.

The oracle generator, original 8B fixture and original limits stayed unchanged
through the fix, verified by checksum. The engine corrections preserve projection
partitions by feature widths, distinct head/residual RMSNorm reduction orders,
and FP32 softmax probabilities before value accumulation. Attention recomputes
scores in three passes with O(d) shared state. The numerical contract and kernel
headers describe these choices; long contexts and other hardware remain unmeasured.

The earlier exact-match claim from rewriting the reference to follow engine
arithmetic remains withdrawn. The accepted evidence above comes from the restored
independent reference: fixed one-row library projections, dense per-query causal
FP32 attention, and stock Transformers RMSNorm/RoPE/SiLU.

Keep the independent reference and original error-budget rules fixed while
correcting the engine. A separately written reference deliberately aligned to
engine arithmetic checks conformance, not independent fidelity, and cannot
replace this acceptance gate. FP64 and loosened limits remain rejected as
workarounds. Shape invariance and exact cache/full-trace equality remain
required but do not substitute for independent numerical acceptance.

Detailed diagnostics and the withdrawn conformance reference remain ignored.
Andrew authorized committing and pushing this slice, followed by a separate
cleanup commit before further implementation.

Unicode helpers are utf8proc 2.10.0 and Oniguruma 6.9.10, source-hash pinned.
They supply NFC and regex matching; BPE and inference remain direct code.
The pure progression rule is linked in `inference/progression.cpp` so its
invariants can be tested without CUDA. The integration inventory stays at five:
tiny/8B generation scenarios replace the former forward scenarios.

The initial cache lifecycle is an accepted simple baseline: allocate contiguous
capacity once per prompt/response turn, reuse it during generation, then release
after final captures and pending GPU work complete. It is not optimized for
reuse across turns; per-turn allocation cost and unused reserved capacity are
known tradeoffs. Evaluate representative runs before introducing allocation
pooling, prefix reuse or another cache strategy; no optimality is claimed.

Implementation records and validation reports belong in gitignored `.local/`.
Keep this file for settled project decisions and the queue for concise work
state; do not publish the development cycle as research or articles.

C++ builds use C++20 without GNU extensions and warnings as errors. CUDA builds
also promote NVCC diagnostics to errors. CMake owns editor include paths and
compiler settings through the shared presets and VS Code configuration provider.
Local machine overrides belong in ignored `CMakeUserPresets.json`.

Andrew clarified that the requested tooling is clang-tidy static analysis like
Serenity, rather than clangd. The policy, check entry point and VS Code task are
implemented. Keep the same five check families, warnings as errors, and the
test-literal exception. macOS host analysis uses LLVM 21.1.8; Linux uses 22.1.8
because CUDA 13.2 headers do not parse with LLVM 21. The optional analysis image
does not replace the production NVCC image. CUDA analysis checks both host and
device ASTs with the narrowly documented SDK-version compatibility exception.
The source cleanup uses checked host span/container access, explicit resource
state checks and named operand/source fields. Native CUDA indexing and C API
byte boundaries have narrow, documented exceptions; no check family or findings
baseline is disabled. The independent oracle, fixtures and acceptance limits
remain fixed. Diagnostics and detailed validation evidence stay ignored.

See [QUEUE.md](QUEUE.md), [architecture](../architecture/README.md) and
[build and verification commands](../../tests/README.md).

## Target-to-AV interpretation

The approved header design was committed as `996aac1`. Its complete implementation
covers target generation through selectable AV interpretations. Persistent model
weights are separate from each sequence cache: repeated AV requests reuse weights
while receiving fresh contiguous KV. Target GPU resources are released before AV
loading; owned raw captures and exact token IDs survive in the record. An explicit
embedding boundary places a one-row replacement after lookup and before consumers,
including when the marker falls in a later prefill chunk. The pairing owns block
24, scale 150, pinned sources, trained prompt and explanation delimiters.

Both C++ corpora were consulted for implementation; opening comments cite the
retrieved rules and describe numerical steps, ownership, failures and costs.
`BoundCheckpoint::source_identity` supplies verified structured provenance to
pair validation. Run owns bounded attempts and phase measurements; estimated
allocations remain distinct from measured process/device peaks. Record preserves
raw AV output even when explanation tags are missing or malformed. Cancellation,
load failure and execution failure do not silently regenerate the target.

The AV reference was frozen before implementation and imports the unchanged
independent operator policy. Norm, divide and scale remain FP32, followed by one
BF16 cast. Engine normalization uses eight interleaved FP32 sums with separate
multiply/add rounding; reference arithmetic, original target fixtures and limits
remain unchanged. Comparison uses identical frozen injection inputs, per-row
activation/logit bounds, exact cached/full final logits and bounded greedy IDs.
The workflow also checks retained captures and repeated fresh-cache variants.
These short numerical fixtures do not establish semantic fidelity or long-context
behavior. Reconstruction and the browser/API remain subsequent work. Verification
status is recorded in the queue; detailed reports remain ignored.

## Formatting

Andrew selected Google's formatting preset and format-on-save. The repository
uses `.clang-format` with `BasedOnStyle: Google`, clang-format 21.1.8, and the same
resolver for VS Code and `scripts/check.py format`. C/C++/CUDA format on save via
the existing Microsoft C/C++ extension. CI checks formatting on PRs and `main`
pushes. Tool installation is explicit; no machine-specific path or automatic
download on save. clang-tidy remains a separate static-analysis gate.

## Live application

Andrew accepted the complete source-launch/browser slice. Session owns one
persistent worker, Run and device; HTTP threads only submit typed commands and
read bounded immutable pages. Incremental classification stays with the model
text protocol. Backend-neutral execution now compiles under the host preset;
CUDA links only at executable composition. A Metal port supplies its runtime,
kernels, build/composition and numerical evidence without replacing session,
transport or presentation contracts. No alternate backend is implemented yet.

The implementation uses source-pinned cpp-httplib for HTTP and the existing JSON
library for wire translation. The launcher builds locally, verifies revision-aware
model caches and starts the service. It never downloads an application executable.
Loopback is the default; LAN requires explicit configuration. API access tokens,
Host/Origin checks and server epochs protect the single-user boundary. Retries keep
both request identity and the originating epoch. Records remain in memory only.

The browser preserves exact IDs/raw byte ranges, supports selection/windows and
separate AV variants, and reconnects by cursor without regenerating. Publication
failures are explicit and stop gracefully. Each AV sequence gets fresh KV while
resident AV weights are reused. Setup hashes every model artifact and saves a
receipt; unchanged installations launch without repeating that scan. The service
retains shared const archive mappings across turns and target/AV switches;
metadata/binding and GPU upload use the prepared bytes. Direct service calls
still default to full verification. Files must remain unchanged while the service
runs. Only the active model occupies VRAM. Mapped pages remain OS-reclaimable,
and both archives' metadata allowances are included in host admission.

The application integration explicitly replaces standalone archive integration,
keeping five scenarios. Successful real-file verification/binding remains covered;
the retired adversarial filesystem cases are not claimed as preserved. Pure
parser and byte/state contracts remain units. Frozen numerical oracles, fixtures
and limits are unchanged. Reconstruction remains optional later work. Verification
status is recorded in the queue. The existing application integration supports
native and Docker execution modes; container verification uses the default source
launcher, read-only model mounts and a localhost-published service. Host NVIDIA
Container Toolkit remains a prerequisite for users' own Docker deployments.

## Model setup and platform decisions, 2026-10-19

- Full model SHA-256 belongs to setup/download/repair, not every application
  launch. Persist a versioned receipt bound to the manifest and file metadata;
  unchanged installations launch without scanning weight payloads. Changed or
  missing artifacts require setup again. Explicit verification remains available.
  Prepared mode is a trusted launcher precondition; standalone archive/service
  calls default to SHA-256. Keep cache files unchanged during service lifetime.
- Linux/NVIDIA builds and runs inside Docker. Future macOS/Metal uses the native
  Apple toolchain; Windows/NVIDIA uses native MSVC/CUDA without requiring WSL2.
  Only Linux/NVIDIA is currently implemented and validated.
- Exploration budgets: target output defaults to 4,096 tokens shared by thinking
  and answer; total target context is 8,192 including formatted prompt. Keep AV
  settings independent (256 output default, 512 total context). Admission rejects
  oversize requests rather than truncating them. Early model completion remains
  normal. This is application configuration, not a paper-derived model limit.
