# File and folder mapping

**Reviewed and accepted — 2026-10-17.** This maps the
[change axes](change-axes.md) into a source layout for the accepted
[experiment](logical-overview.md) and [transformer](transformer-execution.md)
overlays. The selected model is the first implementation within this design.
It does not define the limits of the model, tokenizer or cache contracts.

The layout names families of interchangeable implementations. Each family
owns its contract; each kind has an explicit location beneath that family.
Only the selected kinds are planned for initial implementation. Alternative
kinds have documented destinations, without empty source files or claims that
they work already.

## Planned layout

The tree covers the full planned engine/application layout. The first forward-
pass sources are implemented; later-slice files remain planned. Existing site,
documentation and reference files remain in place. `{h,cpp}` means a
header/implementation pair; a contract-only header does not require an empty
implementation file.

```text
dolittle/
├── CMakeLists.txt
├── CMakePresets.json                # Portable host/CUDA build and editor configuration
├── .vscode/                        # CMake-backed IntelliSense and extension recommendations
├── Dockerfile
├── compose.yaml
├── config/
│   └── models.json                  # Pinned artifacts and explicit kind selections
├── scripts/
│   ├── run.py                       # Source-build and launch entry point
│   ├── models.py                    # Acquire and verify declared artifacts
│   ├── check.py                     # Host/archive/CUDA-build/GPU/static-analysis commands
│   ├── tidy.py                      # Pinned analysis of CMake host/CUDA commands
│   ├── site.py                      # Existing independent site tooling
│   └── validation/
│       ├── forward_reference.py     # Independent tiny/8B target fixtures
│       ├── av_reference.py          # Frozen target-row-to-AV numerical fixtures
│       └── ...                      # Existing checkpoint-selection reference tools
├── src/
│   ├── bounds.h                     # Checked host span access shared across metadata families
│   ├── main.cpp                     # Composition of selected implementations
│   ├── measurement.{h,cpp}
│   ├── tensors/
│   │   ├── tensor.{h,cpp}           # Shape, strides, extent and borrowed views
│   │   ├── formats/
│   │   │   ├── format.{h,cpp}       # Named tensor payload/lifetime contract
│   │   │   ├── safetensors.{h,cpp}   # Initial serialized format
│   │   │   └── json_metadata.h      # Internal bounded duplicate-key-safe metadata parser
│   │   └── representations/
│   │       ├── representation.h     # Encoding and computation-mode declarations
│   │       └── bf16.{h,cpp}         # Initial storage representation and conversion
│   ├── models/
│   │   ├── model.{h,cpp}            # Forward-description contract and pure validation
│   │   ├── observation.h            # Semantic sites advertised by a model
│   │   ├── checkpoint.{h,cpp}       # Validated parameter-binding/lifetime contract
│   │   └── qwen3/
│   │       ├── model.{h,cpp}        # Qwen3 mathematics and observation points
│   │       └── checkpoint.{h,cpp}   # Qwen3 tensor names, shapes and config binding
│   ├── tokenizers/
│   │   ├── tokenizer.h              # Exact encode/decode and token identity contract
│   │   └── byte_bpe.{h,cpp}          # Initial tokenizer kind, with checkpoint data
│   ├── text/
│   │   ├── protocol.h               # Message formatting and segmentation contract
│   │   └── qwen3_chat.{h,cpp}        # Initial chat/thinking protocol
│   ├── inference/
│   │   ├── generation.{h,cpp}       # One sequence's generation and termination
│   │   ├── progression.cpp          # Pure next-action rule; linked without CUDA
│   │   ├── executor.{h,cpp}         # Forward execution and resource planning
│   │   ├── observations.{h,cpp}     # Resolve requests, sizes and immutable capture views
│   │   ├── storage_plan.{h,cpp}     # Pure scratch lifetime/alignment placement
│   │   ├── sampling/
│   │   │   ├── sampler.h            # Token selection and independent RNG contract
│   │   │   └── categorical.{h,cpp}  # Initial filtering/draw mechanism; greedy mode
│   │   └── cache/
│   │       ├── cache.h              # Logical KV access, identity and capacity contract
│   │       └── contiguous.{h,cpp}   # Initial preallocated contiguous strategy
│   ├── nla/
│   │   ├── protocol.h               # Target/AV/AR pairing and conversion contract
│   │   ├── pairs/
│   │   │   └── qwen3_8b.{h,cpp}     # Selected community pair's trained conventions
│   │   └── metrics/
│   │       ├── metric.h             # Named reconstruction diagnostic contract
│   │       └── cosine.{h,cpp}       # Initial raw-cosine diagnostic
│   ├── experiment/
│   │   ├── run.{h,cpp}              # Admission, phase scheduling and residency
│   │   └── record.{h,cpp}           # Owned tokens, captures and variants
│   ├── gpu/
│   │   ├── device.h                 # Device resources and completion contract
│   │   └── cuda/
│   │       ├── device.{h,cpp}       # CUDA composition entry and device backend
│   │       └── backend.h            # Private resource/stream bridge for dispatch
│   ├── kernels/
│   │   ├── operations.{h,cpp}       # Pure validation and operation preparation
│   │   └── cuda/
│   │       ├── operations.cpp        # Prepared CUDA bindings and dispatch
│   │       ├── launch.h              # Private validated kernel-launch boundary
│   │       ├── common.cuh            # Private conversion/reduction helpers
│   │       ├── projection.cu
│   │       ├── normalization.cu
│   │       ├── rotary.cu
│   │       ├── attention.cu
│   │       ├── embedding.cu
│   │       └── elementwise.cu
│   └── server/
│       └── http.{h,cpp}              # Assets, command delivery and event transport
├── app/
│   ├── index.html
│   ├── app.css
│   ├── client.js                    # Transport and connection handling
│   ├── controller.js                # Selection, requests and interaction state
│   └── view.js                      # Presentation
├── tests/
│   ├── core/                        # Isolated in-memory contract tests
│   │   ├── test_sampling.cpp
│   │   ├── test_progression.cpp
│   │   ├── test_byte_bpe.cpp
│   │   ├── test_qwen3_chat.cpp
│   │   ├── test_tensor.cpp
│   │   ├── test_bf16.cpp
│   │   ├── test_observations.cpp
│   │   ├── test_storage_plan.cpp
│   │   ├── test_numerical.cpp
│   │   ├── test_nla.cpp             # Pure conversion and output delimiter contracts
│   │   ├── test_record.cpp          # In-memory ownership and append contracts
│   │   ├── test_operations.cpp
│   │   ├── test_model_recipe.cpp
│   │   ├── test_qwen3_model.cpp
│   │   ├── test_safetensors.cpp
│   │   ├── test_qwen3_checkpoint.cpp
│   │   └── test_forward_reference.cpp
│   ├── integration/
│   │   └── test_checkpoint_archive.cpp # Real file integrity and binding workflow
│   ├── gpu/                         # Numerical checks through production paths
│   │   ├── forward_reference.{h,cpp} # Completed-result comparison contract
│   │   ├── test_generation.cpp       # Tiny and pinned 8B target-to-AV scenarios
│   │   ├── interpretation.{h,cpp}    # AV continuation of those same workflows
│   │   └── test_device.cpp           # CUDA resources/transfers/completion scenario
│   ├── app/
│   ├── fixtures/
│   ├── support/
│   │   ├── text_fixture.h           # In-memory tokenizer/protocol contract data
│   │   ├── check.h                  # Minimal in-process assertion helpers
│   │   └── numerical.{h,cpp}        # Pure array error metrics; launches nothing
│   └── test_site.py                 # Existing
├── web/                             # Existing public explanatory site
└── docs/                            # Existing documentation and writing
```

Existing README files are omitted. Source folders are not created by this
document. `config/models.json` stays outside the ignored root `models/`
download directory and starts from the pinned 8B reference manifest. Historical
reference manifests remain unchanged.

## Families and where another kind belongs

The extension rule is structural: adding a kind should not require editing
the implementation of an existing kind. Selection/composition, new evidence,
and genuine contract extensions may change. The following destinations are
examples of future work, not implementation commitments.

| Family | Initial member | Example future destination | Consumers remain dependent on |
| --- | --- | --- | --- |
| Tensor container | `tensors/formats/safetensors.*` | `tensors/formats/gguf.*` | `tensors/formats/format.h` |
| Model architecture and checkpoint binding | `models/qwen3/` | `models/<architecture>/` | `models/model.h` and `models/checkpoint.h` |
| Tokenizer algorithm | `tokenizers/byte_bpe.*` | `tokenizers/unigram.*` or `tokenizers/wordpiece.*` | `tokenizers/tokenizer.h` |
| Text protocol | `text/qwen3_chat.*` | `text/<protocol>.*` | `text/protocol.h` |
| Numerical representation | `tensors/representations/bf16.h` | `tensors/representations/<encoding>.*` | `representation.h` and tensor views |
| Sampling mechanism | `inference/sampling/categorical.*` | `inference/sampling/<mechanism>.*` | `inference/sampling/sampler.h` |
| KV storage strategy | `inference/cache/contiguous.*` | `inference/cache/paged.*` or `growing.*` | `inference/cache/cache.h`, plus declared layout capabilities |
| Trained NLA convention | `nla/pairs/qwen3_8b.*` | `nla/pairs/<pair>.*` | `nla/protocol.h` |
| Reconstruction metric | `nla/metrics/cosine.*` | `nla/metrics/<metric>.*` | `nla/metrics/metric.h` |
| Device backend | `gpu/cuda/` | `gpu/metal/` | `gpu/device.h` |
| Operation implementation | `kernels/cuda/` | `kernels/metal/` | `kernels/operations.h` |

Models and tokenizers vary independently. A second architecture may use the
same tokenizer implementation with different vocabulary/merge data. A new
tokenizer algorithm belongs under `tokenizers/`, not inside Qwen3 code or a
growing switch in one supposedly universal tokenizer implementation.

Model-specific prompt conventions have their own `text/` implementations.
Different trained AV/AR prompts belong to the selected NLA pairing. Neither
the tokenizer contract nor ordinary generation assumes Qwen's thinking
delimiters or this pair's explanation tags.

## Model data, tensor storage and tokenization

The shared tensor view describes shape, strides, element representation and
extent. It does not declare a universal set of transformer weights.
`models/qwen3/checkpoint.cpp` assigns Qwen3 tensor names to parameter roles and
checks dimensions. Another model family owns its own binding. The tensor
container reader only exposes named payloads and their metadata.

Representations describe how values are stored and converted. Kernel-private
packing stays with the operation that uses it; shared packed encodings need
an explicit representation definition. Format support alone does not imply
support for every architecture or numerical encoding found in that format.
Loading a future container may require separate model-binding and kernel work.

The byte-BPE implementation consumes the selected checkpoint's vocabulary,
merges and supported tokenizer pipeline configuration. Normalization,
pre-tokenization, added tokens and decoding behavior must match the checkpoint,
not merely the BPE algorithm's name. Unsupported pipeline behavior fails
explicitly. If another tokenizer needs different code, give that kind its own
implementation rather than silently approximating it with byte BPE.

Keep tokenizer identity/configuration per model role. The target, AV and AR
may share verified artifacts or mechanisms, but sharing is established from
their metadata; it is not assumed from a common architecture name.

## Cache strategies and compatible attention

`cache.h` describes the owning model/sequence, valid logical positions,
capacity, layout identity and readiness requirements. It must not require all
caches to expose one contiguous pointer with universal strides.
`contiguous.h` owns the initial concrete view and addressing rules. A future
paged implementation owns its block table and page addressing separately.
An attention operation declares which concrete views it can consume.

The initial proposal is preallocated contiguous BF16 storage sized to the
admitted run, with attention reading only the valid prefix. Physical allocation
uses the device backend. The strategy is not tied to Qwen3: dimensions and
attention requirements come from the model, and the plan checks compatibility.

This is a deliberately basic baseline, not an optimized cache lifecycle across
turns or a claim of an optimal storage strategy. One run means one prompt and
its response through stopping and final-token capture completion. Allocate
once for that turn, reuse the allocation throughout prefill/decode, and release
it after pending GPU work completes. A follow-up turn starts with a fresh
allocation and prefix; no allocation pooling or prefix reuse is implemented.

Known tradeoffs are allocation/release overhead per turn and unused capacity
when generation stops before the admitted limit. Evaluate representative runs
before changing this policy: measure allocation/release time, peak and unused
capacity, prefill/decode latency and memory pressure when switching to the NLA.
Allocation reuse, pooling or another storage kind are candidates if the measured
cost warrants them. Keeping physical storage would not itself authorize reusing
its logical contents for different tokens or weights.

Alternative allocation strategies, host/device placement and numerical
representation are distinct choices. Offloading is not another model family;
quantization is not merely a cache-addressing option. Supporting either can
require changes across planning, representation, device operations and cache
implementation. Sliding-window eviction also requires compatible attention
semantics; it cannot silently replace full-history attention.

Adding paged storage need not edit the contiguous implementation. It may need
new attention kernels, capability selection and resource estimates. The cache
contract permits different layouts; it does not promise every existing kernel
can consume a new layout or hide its cost behind per-element callbacks.

## Sampling without a class hierarchy

The family is now explicit under `inference/sampling/`. The initial
`categorical.*` implementation owns temperature, top-k/top-p filtering,
categorical draws and its greedy mode. Those steps are not separate strategy
objects. `sampler.h` owns the boundary between logits/settings/RNG state and
the selected token. Another selection mechanism gets a sibling implementation.

Generation owns stopping and consumed-token alignment; the sampler owns token
selection. Target and AV have independent sampling state. A new AV seed never
advances the target RNG stream, and AR execution has no sampler.

Host sampling remains a proposed first baseline, not a measured decision.
Transferring 151936 logits costs 303872 bytes in BF16 or 607744 in FP32 per
generated token, plus completion overhead. Account for it before choosing the
initial execution path. GPU acceleration may add backend operations without
changing the selected sampling semantics. No runtime plugin system is required.

## Axis ownership

Paths are relative to the repository root. Contract and concrete kind belong
to the same axis, but different axes within a model family remain separate
files. Qwen3 checkpoint naming and Qwen3 mathematics are an example.

| Axis | Contract or coordinating owner | Initial implementation or data |
| --- | --- | --- |
| M1 Artifact acquisition | `scripts/models.py` | `config/models.json` supplies pinned artifacts |
| M2 Tensor container decoding | `src/tensors/formats/format.h` | `src/tensors/formats/safetensors.*` |
| M3 Checkpoint interpretation | `src/models/checkpoint.h` | `src/models/qwen3/checkpoint.*` |
| M4 Token encoding | `src/tokenizers/tokenizer.h` | `src/tokenizers/byte_bpe.*` |
| M5 Text protocol | `src/text/protocol.h` | `src/text/qwen3_chat.*` |
| M6 Transformer mathematics | `src/models/model.h`, `src/models/observation.h` | `src/models/qwen3/model.*` |
| M7 NLA conversion | `src/nla/protocol.h` | `src/nla/pairs/qwen3_8b.*` |
| R1 Sampling | `src/inference/sampling/sampler.h` | `src/inference/sampling/categorical.*` |
| R2 Sequence progression | `src/inference/generation.*` | One sequence loop using selected contracts |
| R3 Experiment scheduling | `src/experiment/run.*` | One-run sequential-residency policy |
| R4 Observation records | `src/experiment/record.*` | Owned observations and variants |
| R5 Reconstruction diagnostics | `src/nla/metrics/metric.h` | `src/nla/metrics/cosine.*` |
| X1 Execution planning | `src/inference/executor.*` | `src/inference/observations.*`, `src/inference/storage_plan.*`; forward preparation and capability selection |
| X2 KV storage | `src/inference/cache/cache.h` | `src/inference/cache/contiguous.*` |
| X3 Numerical representation | `src/tensors/tensor.h`, `src/tensors/representations/representation.h` | `src/tensors/representations/bf16.*` |
| X4 Device runtime | `src/gpu/device.h` | `src/gpu/cuda/device.*` |
| X5 Operation implementation | `src/kernels/operations.h` | `src/kernels/cuda/*.cu` |
| A1 Application transport | `src/server/http.*` | `app/client.js` implements the browser side |
| A2 Interaction behavior | `app/controller.js` | User selection, requests and pending/error states |
| A3 Presentation | `app/view.js` | `app/index.html`, `app/app.css` |
| D1 Build environment | `CMakeLists.txt` | `Dockerfile`, explicit backend build targets |
| D2 Launch orchestration | `scripts/run.py`, `src/main.cpp` | `compose.yaml`, concrete composition at startup |
| D3 Numerical verification | `scripts/check.py`, `tests/` | Independent fixtures and existing reference tooling |
| D4 Measurement | `src/measurement.*` | Phase/operation timing and cost reports |

`operations.h` initially declares a bounded catalog of operations. Their
implementations and launch geometry live in backend kernel files. A few small
related kernels can share a file; this does not allow model formulas or device
management to accumulate there. Split operation headers when independent
consumers or size justify it.

The M6 observation vocabulary lives in `models/observation.h`; model kinds
advertise their supported sites using that vocabulary. X1 request resolution
and capture views live in `inference/observations.*`. NLA pairing chooses the
required site through these contracts. Neither the contract nor resolver is
specific to block 24. First-slice header design starts with these boundaries,
numerical representation, forward execution and the numerical harness; it does
not create every file in the planned tree.

The second header-design group added tensor extents and verified safetensors
loading, model recipes and checkpoint ownership, Qwen3 mathematics/binding,
device resources and the operation catalog. The model produces an ordered
recipe using a bounded set of operations; the executor chooses implementations
and storage. `gpu/cuda/device.h` is the explicit CUDA factory used at composition.
That group established declarations/contracts before implementation.

The final first-slice design pass added BF16 conversion, pure scratch placement
and numerical comparison helpers, plus in-memory parsing and validation entry
points in the existing owners. At that checkpoint, the `.cpp` and six CUDA
`.cu` files contained their opening contract/algorithm/cost/corpus comments
and includes only. They did not return placeholder success or claim completed
kernels. Current implementation status follows the checkpoint audit below.

`storage_plan.*` is an X1 helper, not a new cache strategy or change axis. It
places resources whose last uses include capture copies. Mathematical operation
validation stays in `operations.*`; device capabilities remain a preparation
concern. Safetensors and Qwen3 parsers accept in-memory inputs so unit tests do
not have to launch a loader. Numerical array comparison is likewise separate
from the integration harness that obtains completed model outputs.

The forward-pass boundary covers loading through complete forward-result
comparison. Target generation and AV interpretation extend it as described below.
AR conventions, serving and application behavior remain subsequent slices.
Do not generate those files ahead of their design.

The first forward-pass source scope is now implemented: tensor/BF16 handling,
observation and scratch planning, operation/recipe validation, Qwen3 recipes and
checkpoint binding, verified safetensors loading, CUDA resources and kernels,
execution, and completed-result comparison. See the
[verification commands](../../tests/README.md) for scenario coverage.

The forward, generation, AV pairing and experiment source families have implementation
bodies; reconstruction and application destinations remain planned, with no generated stubs.
The concrete additions stay within their existing axes:

- `tensors/formats/json_metadata.h` shares bounded metadata parsing between
  the format reader and model binder; it introduces no model dependency into
  the safetensors implementation.
- `gpu/cuda/backend.h` exposes native resources only within CUDA preparation.
- `kernels/cuda/operations.cpp`, `launch.h` and `common.cuh` keep prepared
  dispatch and shared kernel primitives within the CUDA operation family.
- Eleven isolated C++ unit executables cover preparation and comparison;
  archive/device/forward drivers remain distinct integration workflows.

The host build requires C++20, CMake 3.24+ and OpenSSL 3. nlohmann/json 3.12.0 is
hash-pinned for metadata; OpenSSL supplies SHA-256. CUDA 13.2 builds use a
manifest-digest-pinned Docker image with explicit support package versions.
The image builds successfully, while current GPU execution uses the native
Linux host because NVIDIA Container Toolkit access is not configured. Native
and forced-PTX checks pass for the recorded short-sequence workloads and explicit
reference policy. Newer GPUs, long contexts and container GPU execution remain
unverified. These changes introduce no alternative backend or new architecture axis.

## Composition and dependency direction

`main.cpp` composes the supported model, tokenizer, text protocol, NLA pair,
cache, sampler and backend from explicit validated selections. Construction can
be compiled and straightforward; neither a plugin loader nor a universal model
registry is required. Unknown kinds and incompatible combinations fail before
generation. Adding a supported kind may change this composition point, without
adding model-name branches to the shared generation loop.

The dependency rules are:

- Generation consumes the tokenizer, text-protocol, sampler, model and forward
  contracts. It does not include Qwen3, byte-BPE or contiguous-cache headers.
- The executor consumes a model's forward description, cache/device contracts
  and declared operation capabilities. It does not infer a model's equations
  from its name or own a second copy of the Qwen3 block algorithm.
- Each model implementation owns its equations and supported input/output and
  observation points. Model code has no dependency on server, experiment or
  vendor GPU headers.
- Each checkpoint implementation uses a tensor-format contract and its model
  description. A safetensors reader does not depend on Qwen3 checkpoint code.
- NLA pair implementations use model, tokenizer and text-protocol contracts
  plus explicit compatibility checks. Their constants do not appear in shared
  inference or generic NLA contract headers.
- Cache implementations depend on tensor/device contracts as needed. The
  cache contract and layout descriptions do not include executor or generation
  code. Kernels may read the concrete layout declarations they support.
- Backend kernels and device implementations do not depend on experiment or
  application code. Device code supplies resource/completion primitives;
  kernel code owns computation and exact launch geometry.
- Experiment scheduling composes loading, generation, NLA work and records.
  Records do not call scheduling. HTTP translates requests/events without
  redefining experiment transitions. The reporting layer consumes measurement
  evidence; backend code does not depend on report formatting.

The model's forward description remains bounded to supported operations; no
general graph compiler is implied. Dispatch choices and preparation costs must
be resolved in header design. Family boundaries must not introduce virtual
calls per tensor element or require rebuilding the plan per generated token.

## Contracts for header design

These name owners and guarantees, not final C++ signatures. Paths in this
table are relative to `src/`.

| Contract | Owner | Required guarantee |
| --- | --- | --- |
| C1 Tensor view | `tensors/tensor.h` | Shape, strides, representation and extent are explicit; borrowed views cannot outlive storage |
| C2 Model | `models/model.h` | A selected architecture supplies the mathematical recipe, supported outputs and observation points |
| C3 Checkpoint | `models/checkpoint.h` | Selected binding validates required parameters; no missing-weight initialization fallback |
| C4 Tokenizer | `tokenizers/tokenizer.h` | Exact encoding/decoding and token/byte identity; no fixed vocabulary or assumed tokenizer algorithm |
| C5 NLA pairing | `nla/protocol.h` | Declared target/AV/AR compatibility, capture location, injection and reconstruction semantics |
| C6 Sequence progress | `inference/generation.h` | Emitted and consumed positions remain distinct; capture finalization performs no extra sampling |
| C7 Sampling | `inference/sampling/sampler.h` | Explicit settings and independent RNG state; returned IDs lie within the supplied vocabulary |
| C8 Forward result | `inference/executor.h` | Output location, numerical mode and readiness are explicit; requested captures are raw observation values |
| C9 KV cache | `inference/cache/cache.h` | Model/sequence ownership, logical prefix, capacity and concrete layout are explicit without assuming contiguous storage |
| C10 Device | `gpu/device.h` | Allocation/copy/dispatch dependencies are observable; in-use storage cannot be recycled |
| C11 Operation | `kernels/operations.h` | Supported shapes, numerical modes, cache layouts, workspace and observation capabilities are declared |
| C12 Experiment | `experiment/run.h` | One active run, defined phase/cancellation outcomes, and completed dependent work before switching |
| C13 Record | `experiment/record.h` | Stable run/position/variant identities and owned capture bytes survive model release |
| C14 Diagnostic | `nla/metrics/metric.h` | Formula, normalization, inputs and any baseline accompany a score; no semantic-confidence claim |
| C15 Wire message | `server/http.h` | Commands/events retain experiment identity and authoritative status |
| C16 Measurement | `measurement.h` | Workload/backend context and named loading/compute/copy/switching boundaries |
| C17 Text protocol | `text/protocol.h` | Selected formatting and delimiters preserve original token positions; no assumed thinking tags |
| C18 Tensor format | `tensors/formats/format.h` | Named tensor metadata and readable payloads have validated extents and explicit lifetimes |
| C19 Representation | `tensors/representations/representation.h` | Storage encoding and computation precision are declared; unsupported modes cannot be substituted silently |
| C20 Observation site | `models/observation.h` | Semantic boundary and block identify a model-advertised site with explicit width and dtype |
| C21 Observation plan | `inference/observations.h` | Requested sites/positions are validated and byte extents checked before execution; no arbitrary callbacks or silent dropped captures |

Capture ownership is unchanged by the new layout. The executor arranges a
copy before scratch is reused; generation associates it with the consumed
position; the experiment commits completed bytes into the record. The record
never retains a borrowed pointer into a released target allocation. Copies may
be batched. A failure preserves completed observations and marks unfinished
captures unavailable.

## Verify that the structure permits change

| Change | Expected edits | Existing kind that must not be rewritten |
| --- | --- | --- |
| Add a unigram tokenizer | New tokenizer implementation, explicit selection and exact-token tests | Byte BPE and the shared generation loop |
| Add an architecture using existing tokenization | New `models/<architecture>/`, checkpoint binding, composition and model tests; text protocol only if different | Existing tokenizer mechanism and Qwen3 mathematics |
| Add paged caching | New cache kind, compatible attention operations, plan selection and tests | Contiguous cache implementation, model equations and tokenization |
| Add a new trained NLA convention | New pair implementation and compatibility/injection checks | Existing pair's constants and shared inference loop |
| Add Metal for supported operations | New `gpu/metal/` and `kernels/metal/`, build/composition and numerical evidence | CUDA implementations and tokenizer kinds |
| Change an existing checkpoint's vocabulary or dimensions | Validated data/configuration and reference checks, within supported behavior | Unaffected algorithms; unsupported behavior must be rejected |

Introduce a new contract capability only when a new kind actually requires
it. Different algorithms need not be forced through a contract that cannot
represent their semantics. Conversely, the first implementation must not leak
its private assumptions into every consumer.

## Implementation sequence

### Target generation

The target-generation implementation takes a formatted text prompt through
cached target generation and returns exact token IDs and captures for every
consumed position. Its contracts are in:

| Responsibility | Contract entry point |
| --- | --- |
| Token encoding and byte provenance | `src/tokenizers/tokenizer.h`, `byte_bpe.h` |
| Single-turn prompt and thinking segmentation | `src/text/protocol.h`, `qwen3_chat.h` |
| Selection law and independent random stream | `src/inference/sampling/sampler.h`, `categorical.h` |
| Logical cache ownership and initial storage kind | `src/inference/cache/cache.h`, `contiguous.h` |
| Incremental execution and completed chunk ownership | `src/inference/executor.h`, `observations.h` |
| Sequence progression and final-token consumption | `src/inference/generation.h` |

Incremental recipe, operation and CUDA attention extensions are described in
the existing model/kernel file openings. The no-cache implementation remains
the comparison baseline. The executor must receive explicit cache producers
from the model; it must not infer them from tensor names or operation order.

The baseline uses preallocated contiguous cache capacity and host
sampling. Prefill is chunked; decode consumes one new token. Rotated K and
unrotated V are appended once per block. Absolute positions continue across
chunks, and attention reads only the valid causal prefix. One shared prefix
length commits after the entire call succeeds.

Captures join each chunk's logit readback at its existing completion boundary.
Completed host chunks survive a later device failure. This costs capture bytes
per step but adds no per-block host wait. A final emitted token is consumed
even when no further token will be sampled. Stop IDs remain in provenance.

For the selected BF16 model, KV capacity costs 144 KiB per position; one
captured site costs 8 KiB per retained host position. At 4096 positions these
are 576 MiB and 32 MiB respectively. Weights, bounded chunk scratch, aligned
arenas, transfer staging, per-call copies and metadata are additional costs.
Each host sampling draw requires a 303872-byte BF16 vocabulary row and queue
completion. These are derived costs, not measured generation performance.

The tokenizer design uses the existing pinned tokenizer artifacts, including
NFC normalization, exact Unicode splitting, ranked byte merges and added-token
matching. NFC uses source-pinned utf8proc 2.10.0 and the declared regex uses Oniguruma
6.9.10. The generation scenario checks exact IDs against tokenizers 0.22.2;
these helpers supply Unicode primitives, not tokenization or inference.
The protocol initially supports an optional system message and one user prompt,
with thinking enabled or disabled. History, tools and multimodal inputs are
outside this slice. Sampling uses explicit greedy or temperature/top-k/top-p
selection; its tie and RNG semantics are independent of reference-library RNGs.

`progression.cpp` implements the pure next-action rule separately from the GPU
loop so contract tests can link it without CUDA. This is an implementation split
within the same generation responsibility, not another architectural axis.
Verification requirements and commands are in `tests/README.md`; current
verification state is in the work queue. AV execution continues below;
reconstruction and the browser application remain subsequent complete slices.

### Target-to-AV interpretation header design

This complete slice implements target generation through retained, selectable
AV interpretations. Its approved header contracts are now backed by implementation
in the existing owner families below.

| Owner | Responsibility in this slice |
|---|---|
| `nla/protocol.h` | Pair compatibility, one-row conversion, prepared AV prompt and explanation range contract |
| `nla/pairs/qwen3_8b.h` | Pinned sources, block-24 selection, trained actor prompt, marker checks and FP32 normalization followed by BF16 conversion |
| `models/model.h`, `models/qwen3/model.h` | Explicit embedding output/producer boundary before block 0, for full and incremental recipes |
| `models/qwen3/checkpoint.h` | Role-neutral full causal-LM binding, sharing the existing strict target schema; AR remains excluded |
| `inference/executor.h` | Resident weight owner, fresh per-sequence KV/scratch, ordered single-row embedding replacement |
| `inference/generation.h` | Reuse the existing generation loop; require injection inside the prompt and preserve final-token consumption |
| `experiment/record.h` | Owned target IDs/raw captures, absolute capture keys and immutable interpretation variants |
| `experiment/run.h` | Admission, target release before AV loading, cancellation/failure outcomes and sequential requests |
| `measurement.h` | Phase durations and allocation estimates, explicitly distinct from measured peaks |

These remain the accepted axes and folders. Resident weights belong to execution
planning, rather than a new model-family or cache-strategy directory. Embedding
replacement uses the existing ordered queue upload capability; no NLA-specific
CUDA kernel or duplicate transformer is needed. The selected AV uses the same
verified Qwen3 config and chat template, so its pair composes the existing
tokenizer/text implementations with its own actor prompt and source identity.
The generic interfaces remain capable of other pairs and observation sites.

Each interpretation starts a new contiguous KV cache while the AV weights stay
resident. Cache pooling remains deferred. Target tokens and raw captures live in
the record across both phases; interpretation never regenerates the target.
Reconstruction, server and browser work remain subsequent slices. No placeholder
AR/backend/application files are added for this slice.

Implementation acceptance replaces the tiny and pinned-8B generation
integration scenarios with their corresponding **target → retained capture → AV**
workflows, rather than adding a sixth scenario. They retain their existing target
acceptance gates as the first phase of that same experiment. Archive, device and
site scenarios remain; the total stays five. Pure units cover embedding boundaries,
normalization, delimiter extraction, record alignment/append invariants and
record count/byte limits without filesystem, subprocess or GPU work. Phase
switching and cache isolation belong to the complete integration workflows.

A separate AV reference fixture generator, precision, source hashes, input
vectors and numerical limits were frozen before implementation.
It reuses the established independent operator policy without editing the existing
oracle, target fixtures or limits. The AV reference adds only the independently
specified input transformation: FP32 norm, divide, scale, BF16 cast, replacement
of the embedding lookup row, then ordinary causal inference. Reference fixture
generation must not import engine code or use engine outputs as expected values.
For each comparison, both implementations receive identical raw capture bytes;
also check the complete engine target-to-AV path with the frozen target capture
acceptance gate. Do not compare two AV outputs fed different vectors and attribute
their difference solely to AV arithmetic.

Compare exact formatted IDs/marker alignment, normalized FP32 inputs, BF16
replacement bytes, block-0 input, selected downstream activations/logits and
greedy output IDs. Exercise full and chunked prefill with the injection in a
later chunk, then decode without reinjection. A second interpretation in the
same workflow checks fresh-cache isolation and retained capture ownership across
weight switching. Native/PTX execution and memory checking remain required.
Sampled prose agreement with target thinking is not a correctness gate. Freeze
the AV baseline before debugging and locate the engine's first divergence;
never change reference arithmetic or precision to match the engine.

### Subsequent work

The layout now exposes variation beyond the initial model selection. It does
not expand the first release to multiple models, tokenizers, cache strategies
or platforms. Add only the files needed by the current working slice, with
reviewed headers and relevant checks; do not generate the tree as empty stubs.

For each subsequent slice, select toolchain and support dependencies explicitly,
define numerical comparison workloads/tolerances and derive resource costs.
The current first-slice toolchain and helpers are implemented as described above.
Preserve the independent reference experiments and separate build/non-GPU
checks from device execution. The existing public site remains independent
of model downloads and the live inference application.
