# Change axes

**Reviewed and accepted with the revised file mapping — 2026-10-17.** These
are the independent reasons Dolittle's implementation will change, derived from the accepted
[NLA experiment overlay](logical-overview.md) and
[transformer execution overlay](transformer-execution.md). The accepted
[file and folder mapping](file-mapping.md) makes implementation families
explicit, following review of granularity and overfitting to the first model.

One implementation unit should have one reason to change. The test for a split
is concrete: name a change that affects one responsibility without changing
the other. If two pieces must always change together, keep them together
unless a language or build boundary requires separate files.

An overlay phase can cross several axes. Conversely, one axis may serve the
target, verbalizer, and reconstructor. Neither a phase nor a model role is
automatically a module. The tables name responsibilities and permitted changes;
they do not prescribe classes, plugins, or one file per row. An implementation
unit can be a cohesive group of small operations within a file. Folders group
related work and expose where implementations can vary. A split needs more
justification than a distinct function name, but a selected implementation
must not masquerade as the entire family. The revised mapping makes both
the family contract and its first concrete kind explicit.

## Kinds are open sets

Andrew's mapping review identified an overfit to the selected Qwen3 pair:
generic tokenizer, checkpoint and cache files hid independently replaceable
implementations. The architecture must permit other kinds even though the
first release implements only one of each.

| Axes | Family boundary | What a new kind adds |
| --- | --- | --- |
| M2 | Tensor container formats | A reader for another serialized format |
| M3, M6 | Model architectures and their checkpoint bindings | Model mathematics and parameter/configuration interpretation, in separate files within that family |
| M4 | Tokenizer algorithms | Another exact encode/decode mechanism, independently of the model architecture |
| M5 | Text protocols | Another message/template and output-segmentation convention |
| M7 | Trained NLA pair conventions | Another declared capture, injection and reconstruction protocol |
| R1, R5 | Sampling mechanisms and diagnostic metrics, respectively | A sibling implementation of the relevant contract |
| X2 | KV storage strategies | Another allocation/addressing strategy and declared layout |
| X3 | Numerical representations | Another encoding/conversion convention and required computation modes |
| X4, X5 | Device backends and operation implementations | Another backend or another operation implementation on an existing backend |

Model kind, tokenizer kind, checkpoint data, cache strategy and backend are
separate choices. Their compatibility is checked explicitly. Sharing an
architecture does not prove two checkpoints share tokenizer configuration;
supporting a file format does not prove support for every model it can store.

Adding a kind should leave existing kinds untouched. The composition point,
capability selection, tests and genuinely extended contracts may change. In
particular, a new cache layout can require compatible attention operations;
that does not justify embedding all cache implementations in one file.

The folder structure must make these families and concrete members visible.
This does not require empty alternative implementations, virtual classes,
dynamic plugins, or a new directory for every operation. Values such as
vocabularies and dimensions remain data when the existing mechanism supports
them. Novel behavior gets its own implementation rather than an unrelated
model-specific branch in shared generation code.

## Model inputs and meaning

| Axis | Triggered by | May change |
| --- | --- | --- |
| **M1 Artifact acquisition** | A different source, download/cache strategy, or integrity mechanism | Fetching and locating pinned files, verifying content, reporting acquisition progress |
| **M2 Tensor container decoding** | A different serialized tensor format or shard-reading strategy | Reading tensor names, shapes, dtypes, offsets and payloads from files |
| **M3 Checkpoint interpretation** | A different checkpoint schema or naming convention | Translating configuration and named tensors into validated model roles, dimensions and parameter bindings |
| **M4 Token encoding** | A different tokenizer algorithm or tokenization format | Text-to-token and token-to-byte conversion, special-token handling and incremental decoding |
| **M5 Text protocol** | A different conversation template or output delimiter convention | Formatting messages, thinking-mode controls, and recognizing thinking/answer or explanation boundaries |
| **M6 Transformer mathematics** | A different model computation | Block equations, causal visibility, position rules, normalization placement and supported input/output operations |
| **M7 NLA conversion** | A different trained activation-to-text or text-to-activation convention | Pair compatibility, capture specification, vector normalization/injection, trained prompt use, reconstruction endpoint and head selection |

Artifact acquisition establishes that the requested bytes are present. Tensor
decoding establishes what tensors those bytes contain. Checkpoint interpretation
establishes what those tensors mean to the supported architecture. None chooses
which GPU is resident or how matrix multiplication executes.

Text protocol uses token encoding; it does not implement the tokenizer. NLA
conversion supplies its trained prompt and validates marker placement through
those mechanisms. It owns the requirement that AV input comes from an
activation and that AR output comes from the prescribed residual/head path.
Transformer mathematics supplies the operations needed to execute that path.

The first implementation supports only the selected Qwen3-8B pair. A future
pair is a useful boundary test, not a commitment to a general model framework.
Unsupported metadata or operations fail explicitly rather than being guessed.

## Generation and the experiment

| Axis | Triggered by | May change |
| --- | --- | --- |
| **R1 Sampling** | A new token-selection or random-number rule | Logit filtering, greedy/stochastic selection, RNG progression and seed behavior |
| **R2 Sequence progression** | A different rule for advancing or terminating one sequence | Prefill/decode progression, emitted versus consumed positions, stop handling, and the final capture-only consumption step |
| **R3 Experiment scheduling** | A different order or residency policy for target, AV and AR work | Run admission, phase transitions, request queuing, cancellation routing, model loading/release decisions and future concurrency |
| **R4 Observation records** | A different requirement for retaining and identifying results | Exact tokens, capture identity/availability, immutable vector ownership, interpretation variants, provenance and completion status |
| **R5 Reconstruction diagnostics** | A different numerical comparison | Metric formulas, normalization and required baselines, with results tied to the exact vectors/descriptions compared |

Sequence progression knows whether an emitted token has been consumed. It
requests the final consumption needed for capture and performs no extra
sampling during that step. Experiment scheduling knows when the whole target
phase can end and AV work can start. It does not reproduce the token loop.

Observation records preserve the association between a position, its capture,
and each interpretation. They own retained results after transient execution
state is released. They do not decide how long a model stays resident or how a
GPU copy is performed. The initial records are session state; an export format
would introduce a separate serialization responsibility when requested.

Sampling is shared by target and AV generation with independent state and
settings. Changing an AV seed must not advance the target's random stream.
AR execution does not use sampling. Diagnostics remain independent of both
description generation and how a score is displayed.

## Execution and hardware

| Axis | Triggered by | May change |
| --- | --- | --- |
| **X1 Execution planning** | A different way to schedule the same model computation | Prefill chunking, operation selection, compatible fusion, scratch lifetimes, capture copy placement and resource estimates |
| **X2 KV storage** | A different cache organization or growth strategy | Cache addressing, capacity, append/read rules, prefix ownership and layout descriptions consumed by attention |
| **X3 Numerical representation** | A different supported storage or computation precision | Tensor encoding, conversion, packing, accumulation requirements and explicit numerical-mode descriptions |
| **X4 Device runtime** | A different GPU API or device resource mechanism | Device discovery, allocation, transfer, command submission, completion events and resource destruction |
| **X5 Operation implementation** | A different algorithm for an operation on a backend | The operation's kernel, local tuning, workspace requirements and launch configuration |

The distinction between mathematics and execution is essential. Model
mathematics requires a particular Q/K normalization, positional transform and
causal attention result. Planning chooses an available execution path for it.
Operation implementations compute that result; the device runtime supplies
resources and ordering primitives.

Planning reports the resources needed by a supported numerical/execution path.
Experiment scheduling admits or rejects the requested workload using that
estimate and the available budget. Device allocation can still fail; failure
must leave completed observations intact and end or reject the affected work
explicitly. An allocation failure is not permission to lower precision or drop
captures silently.

### Operations form a family over backends

X5 is parameterized by operation and backend. It is not a single collection
of all CUDA code or all Metal code. The initial operation families are:

| Family | Example of a change confined to that family |
| --- | --- |
| Dense projection | Retile a matrix multiplication for single-token decoding |
| Normalization and reductions | Change a reduction strategy for RMSNorm or vector norms |
| Positional transform | Change how rotary values are generated or applied |
| Attention | Replace the attention algorithm while preserving masking, grouping and cache semantics |
| Elementwise and indexing operations | Fuse a gated activation or optimize embedding lookup/copy |

Within a family, independently changing operations remain separate units when
the split test warrants it. For example, embedding lookup and gated activation
do not belong in one implementation merely because both appear in the last
row. Target, AV, and AR use operations with different weights and shapes;
those roles do not require three copies of a matrix multiplication kernel.

A CUDA kernel and the host code defining its exact launch geometry change
together and belong to the same operation implementation. Generic device
allocation and submission stay in X4. A Metal version is another backend
implementation of the same operation contract. Optimizing one backend must not
require edits to the other when the shared contract is unchanged.

This design does not choose between virtual dispatch, static dispatch or
another interface mechanism. It also does not require a universal graph
compiler. Those are implementation choices to justify during file mapping and
header design, using the concrete supported computation.

### Cache layout and fusion expose real coupling

An attention kernel must understand the cache view it receives. X2 owns that
view's layout definition; X5 owns access within the attention algorithm. Adding
a new layout can legitimately require a compatible kernel and X1 selection.
Such a change crosses explicit contracts; hiding layout assumptions in several
unrelated kernels would erase the boundary.

A fused operation owns its internal algorithm and launch. X1 decides where
that operation can replace separate steps. The required block-24 output remains
observable: a fusion that would discard it needs a supported capture path or
must not be selected there. The capture specification belongs to M7, its
placement to X1, copy completion to X4, and retained identity/data to R4.

X3 defines representation rather than granting each kernel permission to pick
its own precision. Changing representation can require new operation variants,
cache layouts and resource estimates. Quantization is therefore a coordinated
future feature, not a one-line setting whose correctness is assumed.

For ownership, X3 defines how numerical elements and any quantization metadata
are encoded. X2 defines how sequence positions and heads occupy cache storage.
X5 owns an operation's private tiling and packed working layout; X1 accounts
for any conversion or workspace that layout requires. A shared packed format
must have one declared owner when file mapping makes those contracts concrete.

## Application and development surfaces

| Axis | Triggered by | May change |
| --- | --- | --- |
| **A1 Application transport** | A different command/event encoding or connection behavior | Engine/browser request delivery, response serialization, ordering, disconnect handling and backpressure |
| **A2 Interaction behavior** | A different way to explore an experiment | Token/window selection, requesting variants or reconstruction, navigation and pending/error interaction states |
| **A3 Presentation** | A different visual or accessibility treatment | Transcript layout, controls, typography, responsive behavior and accessible rendering |
| **D1 Build environment** | A different compiler, SDK or dependency/toolchain version | Build configuration, reproducible environments and native/PTX build targets |
| **D2 Launch orchestration** | A different installation or startup path | Preflight checks and coordinating build, model acquisition and service startup |
| **D3 Numerical verification** | A new correctness claim, reference case or supported precision | Independent fixtures, comparison procedures, tolerances and alignment cases |
| **D4 Measurement** | A new cost question or instrumentation method | Timing boundaries, memory/transfer/dispatch counters, workload measurement and reports |

Transport carries authoritative engine status; disconnect behavior must not
invent a generation stop that R3 has not applied. Interaction maps user intent
to commands. Presentation maps retained and pending states to a readable view.
All three preserve exact token/capture identity rather than deriving it from
rendered character offsets alone.

Launch orchestration calls acquisition and build mechanisms; it does not copy
their logic. The initial source-build path remains Linux/Docker/NVIDIA. A
future native Metal launch path must not force NLA conversion or token
progression to know which operating system launched them.

Measurement defines what is timed or counted. X4 supplies GPU timing/completion
primitives, while X1 and R3 supply operation/phase boundaries. Verification
defines what evidence establishes correctness. Neither becomes a separate
production inference implementation.

The existing public project site has its own
[architecture and publication rules](project-site.md). Its content and build
remain independent of the live inference interface. This document introduces
no new site architecture or publication changes.

## Values are data and capabilities are explicit

A new value within an existing mechanism should not require a new algorithm.
Keep these distinctions when mapping the axes:

| Values or declarations | Mechanism that interprets them |
| --- | --- |
| Repository revision, file names, sizes and hashes | M1 acquisition and M2 decoding |
| Tensor names, dimensions, block count and role | M3 checkpoint interpretation and M6 mathematics |
| Trained prompt, marker IDs, injection scale and supported capture location | M5 text protocol and M7 NLA conversion |
| Temperature, top-p/top-k, seed and token limit | R1 sampling and R2 progression |
| Selected positions and number of requested variants | A2 interaction and R3 scheduling |
| Admitted context limit, capture budget and numerical mode | R3 admission using X1 estimates and declared execution capabilities |

Metadata does not authorize arbitrary execution. A dimension change can be
data when supported by the existing operations; a new normalization rule needs
new mathematics. A different capture index is valid only for a compatible NLA
pair. Enabling thinking through a supported template changes input data, not
the block equations. Tuning a kernel tile is local operation tuning, not a
change to the model's declared dimensions.

## Test the boundaries with concrete changes

| Proposed change | Expected implementation impact | Responsibilities that should stay unchanged |
| --- | --- | --- |
| Add Metal support for the existing pair | X4 Metal runtime, X5 Metal operations, X1 capability selection, D1/D2 setup, D3/D4 evidence; X3 if a new precision is needed | M6 mathematics, M7 trained NLA convention, R2 token alignment, R4 result meaning |
| Optimize CUDA attention with the existing cache view | X5 CUDA attention and its launch; X1 only if selection/workspace needs change; D3/D4 checks | Other operation implementations, Metal, NLA conversion, browser behavior |
| Increase the exploration window from three to five positions | A2/R3 request values within the admitted budget | Captured vectors, AV prompt, transformer operations |
| Change AV temperature or request another seed | Recorded R1 settings and a new R4 interpretation variant | Target transcript, target RNG state, capture position and contents |
| Add a new reconstruction metric | R5 formula and required inputs, D3 checks, and A3 display if needed | AR forward computation and the description being measured |
| Load renamed tensors with unchanged shapes and equations | M3 binding convention, manifest values and D3 loading checks | M6 mathematics and X5 kernels |
| Change download hosting with identical verified files | M1 location/transport configuration | Tensor decoding, model computation and experiment state |
| Add pause–interpret–resume later | R3 scheduling plus preservation/resumption contracts in R2/X1/X2 and records as needed | Existing block equations and trained AV injection convention |

A feature may span several axes. Independence means each edit has a named
reason and an explicit boundary, not that every feature changes exactly one
unit. A first new capability may extend a contract; later implementations of
that capability should not repeatedly require unrelated edits.

## Coverage of the accepted overlays

| Overlay responsibility | Axes involved |
| --- | --- |
| Validate pairing and prepare exact input | M1–M5, M7 |
| Generate, stop and finalize consumed-token captures | M6, R1/R2, X1–X5 |
| Preserve observations and change model phases | R3/R4, X1/X4 |
| Select positions and produce AV variants | A1/A2, M5/M7, R1–R4, X1–X5 |
| Reconstruct and compare descriptions | M3/M6/M7, R3–R5, X1–X5 as applicable; no persistent AR decoding cache |
| Present thinking, answer and interpretations | M4/M5, R4, A1–A3 |
| Build, launch, verify and measure | D1–D4 using the same declared computation and capabilities |

## Review and mapping criteria

Before mapping files, check that a proposed unit can name one axis or one
operation/backend member of X5. Distinct files that always change together
need a reason to remain separate. Shared values and contracts need one owner;
consumers must not maintain competing copies of model semantics, capture
alignment, cache layout or numerical-mode definitions.

The accepted file mapping assigns those owners, dependencies and header contracts.
It must preserve the accepted one-run, sequential-residency scope and leave
Metal, quantization, additional pairs and resumable runs as extensions. This
design creates no interfaces, source stubs or implementation requirements for
those deferred features.
