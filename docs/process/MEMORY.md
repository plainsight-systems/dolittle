# Project memory

## Identity

- **Project:** Dolittle, named after Doctor Dolittle.
- **Parent entity:** Plainsight Systems LLC; no separate operating brand.
- **Status:** local project site with explanatory writing, original artwork,
  and reference GPU feasibility checks. No engine or live application yet.
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
  inference interface remains unimplemented.
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
- Docker is installed but NVIDIA Container Toolkit/GPU access is missing.
  Host administrator setup is required before container execution checks.
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

## Current work

Andrew approved a commit point for the first-slice designs on 2026-10-18.
The layout audit accounts for all 39 C++/CUDA design files in the documented
tree. The forward-reference pair was present under the right folder but was
not individually listed; the tree now names it explicitly. Axis ownership
also names observation vocabulary, observation preparation and scratch planning.
Include paths and model/format dependency boundaries check out. Only surplus
EOF blank lines changed in source files during staging; declarations/contracts
are unchanged, so the prior 18-header C++20 checks still apply.
The private brief and cache remain ignored. This checkpoint is committed
locally; no push or runtime/GPU validation was requested for this step.

The remaining first-slice header-design pass requested on 2026-10-18 is ready.
It adds BF16 host conversion, pure scratch-lifetime placement and array error
metrics; existing owners now expose in-memory safetensors/config parsing and
operation/recipe validation. These are contract-focused unit boundaries without
file access, builds, processes or device launches. Completed-result comparison
records reference provenance and delegates numerical metrics to the pure helper.

Fifteen C++ and six CUDA source files now contain their opening local contract,
algorithm, cost and corpus comments plus includes only. There are no function
bodies, kernel bodies or test cases in those files. CUDA designs cover tiled
projection, row/head RMSNorm, split-half RoPE, streaming causal GQA, embedding/
row copies and elementwise residual/SiLU work. BF16 rounding boundaries were
checked against the pinned Transformers v4.57.3 Qwen3 source; no numerical
agreement or GPU performance is claimed. The original reference probe uses
SDPA, so future fixtures must record its actual backend and numerical policy.

Both local corpora were queried for this phase. Full rules read/applied: Core
I.1, I.7, C.183, R.1; performance GPU.1, GPU.2, GPU.7, GPU.9 and GDSA.16.
All 18 headers pass standalone repeated-inclusion checks and two combined
include orders under Apple Clang 17/C++20 with strict warnings as errors.
These are declaration checks, not unit tests or CUDA compilation. No integration
scenarios were added; the five site tests still occupy the integration budget.
The first-slice file-design scope is complete through result comparison;
toolchain/support-library pinning, independent fixtures/tolerances and actual
implementation remain next. Later cache/generation/NLA/app files stay deferred.

Andrew accepted the remaining first-slice headers on 2026-10-18 and required
a strict testing policy. Unit tests execute isolated code directly, test stable
contracts and invariants, and do not launch builds, processes or system flows.
Behavior-preserving refactors and unrelated features should not force assertion
changes. Integration tests must remain a small handful; the workflow records
that as at most five repository-wide, without bundling or relabeling loopholes.
The existing five site tests are integration tests, despite using unittest;
rebalance that coverage before adding engine integration tests. This change
documents the policy and classification; it does not refactor the test suite.

On 2026-10-18 Andrew accepted the initial five interface headers and authorized
continuing. The remaining first-slice boundary designs now cover tensor layouts,
verified safetensors loading, model recipes and bound checkpoints, the Qwen3
target recipe/binder, device resources with a CUDA factory, and kernel operations.
There are 15 headers total; all passed standalone repeated-inclusion checks and
two combined include orders with Apple Clang 17, C++20 and strict warnings.
These are header designs only; no runtime implementations or GPU checks ran.

The model owns an ordered recipe of a bounded operation catalog. Executor
planning validates dependencies, chooses kernels and reuses storage after last
use, including capture copies. Checkpoints own verified host backing and model
bindings; forward plans own resident weights, one ordered queue, scratch and
bounded staging. Operations report workspace and operand alignment before
allocation. Qwen3 equations and checkpoint names stay in their concrete family;
the initial binder supports target-style output, not AV injection or AR output.
Artifact loading checks supplied hashes and extents and retains backing files;
files must remain unchanged while open. Device completion establishes readback
readiness. A uint32 representation was added for index buffers only.

Both corpora were queried again for this header-design phase. Full Core I.4,
I.7 and R.1 and performance GPU.1, GPU.7, GPU.9, COPY.5 and CDSA.32 were read;
headers cite the rules applied. No implementation-phase consultation is implied.
Next: review the remaining boundary designs, select/pin build dependencies,
establish independent fixtures and numerical bounds, then implement the slice
with fresh consultation of both corpora. Header checks do not validate those
runtime contracts. This stage preceded the local design checkpoint above.

Andrew confirmed sliced implementation and configurable observation hooks on
2026-10-17. The first numerical slice is a complete causal forward pass without
persistent KV caching; current-pass K/V are still computed. Model kinds
advertise semantic sites, callers select sites/positions before execution,
and the planner validates and resolves requests. Block 24 belongs to the NLA
pair's compatibility contract, never a shared-executor default. Initial sites
are block input and output; no arbitrary activation-mutating callbacks.

Initial header designs now exist for representation, observation vocabulary,
observation preparation, forward execution and the reference-comparison
boundary. Both local corpora were queried and full applicable rules read:
Core I.4, I.7, R.1, E.5; performance GPU.1, GPU.7, CDSA.32, COPY.5. Opening
comments cite the rules applied to each header and specify contracts and
algorithm steps. Forward results own host-ready outputs, exact token IDs and
checkpoint/execution/numerical provenance. Five standalone header-inclusion
checks passed with Apple Clang 17, C++20, strict warnings treated as errors.
These are declaration checks only: no linked engine, CUDA kernels, independent
fixtures or numerical GPU comparisons exist yet. Review these interfaces,
then design the remaining first-slice model/loading/device/kernel boundaries
before their implementation; requery both corpora in the implementation phase.

Workflow requirement strengthened by Andrew on 2026-10-17: both C++ header
design and implementation must query the locally running C++ best-practices
and performance-corpus MCP servers. Opening file comments must include the
contract and the specific algorithm responsibility/steps, and cite retrieved
rules with their application. Both corpora are mandatory in both phases;
implementation consultation is not satisfied by earlier header citations.
The detailed process and unavailable-server handling are in
[workflow.md](workflow.md). This documentation update introduces no C++ code.

Andrew authorized committing and pushing this architecture/workflow checkpoint
on 2026-10-17. Pre-push validation passed: local documentation links and
structure, axis/contract coverage, derived storage calculations, whitespace,
and all five site publication tests. The private brief and research cache
remain ignored. These are documentation/publication checks, not engine or GPU
validation.

The requested [4B/8B comparison](../research/2026-10-16-nla-comparison/README.md)
is complete. Qwen3 8B plus `marco-bazzani/Qwen3-8B-nla` runs sequentially in
BF16 on the 3090 Ti: peak allocated memory 15.378 GiB; median AV time 4.73 s
versus 10.28 s in the 4B reference run. All eight thinking descriptions
completed within the cap, but they still substituted unsupported transcript
details. Neither pair has established semantic faithfulness. The 8B pair
uses about 45 GB of downloads, separate weights and one capture layer (24);
4B remains smaller and supports multiple depths. Andrew selected 8B after
reviewing these tradeoffs. Andrew then prioritized standing up the explanatory
project site. No product inference code exists. Raw reconstruction scores across models are not a common accuracy
scale.

The project site is built and running locally: Home, Writing, About, two
opening articles, RSS, and 404. Its architecture and publication rules are in
[project-site.md](../architecture/project-site.md) and
[web/README.md](../../web/README.md). Five publication tests pass, including a
real subpath build, draft/future exclusion, milestone rendering, and rejection
of broken assets/anchors, stale share metadata, missing share images, or
unintended documents. The GitHub Pages workflow now validates PRs and deploys
main with a pinned Hugo installer. The first push (`47f560a`) and deployment succeeded on 2026-10-16:
https://github.com/plainsight-systems/dolittle/actions/runs/38967736528.
The custom domain returned HTTP 200 with the correct GitHub menu link and
production share metadata. HTTPS certificate provisioning was still pending;
the default github.io project URL redirects to the custom domain.

See [QUEUE.md](QUEUE.md). Architecture starts at
[`../architecture/README.md`](../architecture/README.md). Investigations are
indexed in [`../research/README.md`](../research/README.md).
