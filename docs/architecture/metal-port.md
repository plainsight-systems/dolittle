# Native Metal port

Architecture accepted 2026-10-21; execution foundation implemented. Transformer
kernels and the native application remain the next two phases.
Port the complete experiment to Apple Silicon:
target generation, retained activations, verbalization, reconstruction and
matched steering. Keep the existing browser interface and source launcher.
Use the same Qwen3-8B/NLA pair, BF16 storage and defined FP32 arithmetic and
rounding. Implement computation directly in C++ and Metal shaders.

## Reference implementations

These source comparisons inform the design; they do not replace the numerical
oracle or establish performance on Dolittle's workload.

| Inspected source | What carries into this port |
|---|---|
| Vigil, `866e65572a61a300ebb620382c279380ce42063a`: `backends/mlx/generate.cpp`, `sampler.cpp`, `kv_cache.cpp` | A deferred sampler returns a device array. The next forward consumes it before the current token is extracted for the host. Asynchronous evaluation overlaps execution and collection. |
| Serenity, `35e20d7`: `src/{metal,cuda}/device/submission.*`, `src/headless/run.cpp`, `docs/research/2026-10-19-cuda-backend.md` | Preserve shared meaning and application behavior; each backend owns submission, resources, synchronization and hardware-specific execution. Reuse submission resources, retire them after completion, and select the backend at composition/build time. |
| [llama.cpp Metal operations][llama-ops] and [device storage][llama-device] | Separate matrix-vector/small-batch work from matrix-matrix prefill. Select kernels by shape and device capability. Use multiple Metal buffer views when a mapped allocation exceeds `maxBufferLength`, keeping each tensor inside a view. |
| [vLLM-Metal decode pipeline][vllm-decode], [attention][vllm-attention] and [memory policy][vllm-memory] | Device tokens feed the next step before output is resolved. Tiled attention keeps query/output state in registers and reuses KV tiles. Memory admission accounts for the recommended working set and other reservations. |

Serenity supplies the porting pattern. Vigil and the open-source engines supply
inference mechanisms. Their inspected generation paths still use host scheduling;
Dolittle additionally requires generation to continue without the collector
advancing each token. Preserve that accepted execution boundary.

Use these references without importing their frameworks, quantization paths,
automatic FP16 conversions or serving-specific requirements. vLLM-Metal uses MLX;
it is a source reference here, not a proposed dependency. Likewise, Serenity's
frame pacing and rendering dependencies do not belong in the inference loop.

## Algorithm and execution boundary

The shared recipe describes the transformer. Metal decides how to execute it.
CPU code prepares requests and collects completed output. Generated tokens,
probabilities, random draws and stop decisions stay on the GPU data path.

```mermaid
flowchart TD
  Host[CPU: tokenize, admit memory, prepare request] --> Start
  subgraph GPU[GPU-owned sequence]
    Start[Prompt IDs or previous sampled ID] --> Embed[Embedding lookup; optional AV injection H3]
    Embed --> Norm1[RMS normalization]
    Norm1 --> QKV[Q, K, V projections]
    QKV --> Rope[Q/K normalization and RoPE]
    Rope --> Cache[Write K/V at device-owned position]
    Cache --> Attention[Causal grouped-query attention]
    Attention --> Residual1[Output projection and first residual addition]
    Residual1 --> Norm2[RMS normalization]
    Norm2 --> Gate[Gate and up projections; SiLU gate times up]
    Gate --> Residual2[Down projection and second residual addition]
    Residual2 --> Hook[Advertised block-output hook H1]
    Hook --> Edit[Optional selected-position steering H4]
    Edit --> Blocks{More blocks?}
    Blocks -->|yes| Norm1
    Blocks -->|target or AV finished| Head[Final RMS and vocabulary projection]
    Blocks -->|AR finished| Vector[AR value head; reconstructed vector]
    Head --> Sample[GPU sampling and RNG]
    Sample --> State[GPU position, EOS, budget and final-token-consumption state]
    State -->|continue| Start
    State --> Journal[Immutable token and completion records]
    Hook --> Captures[Retained device capture arena]
    Captures --> Normalize[GPU normalization for AV]
    Normalize --> Embed
    Vector --> Direction[GPU edited-minus-original direction and normalization]
    Direction --> Edit
  end
  Journal --> Collect[CPU: collect completed records, decode text, publish UI]
```

This diagram combines separate target, AV and AR requests, not simultaneous
execution. Each request's role selects its head and hooks. The experiment
scheduler moves between requests after their completion.

H1 captures the residual after zero-based block 24's second addition, before
block 25's normalization: 4096 BF16 values per position, copied into independently
owned device storage. H3 replaces the trained marker embedding with a normalized
capture. AR runs its 28 blocks and value head. H4 applies the reconstructed
direction to the selected target residual during replay. These are this pair's
conventions; the hook catalog remains dynamic.

The [execution boundary](execution-boundary.md) and
[steering design](steering.md) define the full contracts, including consumption
of the final selected token and matched branch seeds.

## Metal scheduling

CUDA's conditional graphs implement `DeviceSequence`; they are not its portable
interface. Neither Vigil's inspected loop nor vLLM-Metal's one-step-ahead
pipeline provides Dolittle's complete autonomous sequence.

The candidate uses reusable indirect command buffers for prepared forward
bodies. Metal supports an [execution range supplied by a GPU buffer][apple-indirect].
A control dispatch updates position, phase and active ranges from device state.
After termination, later scheduled forward bodies have empty ranges.

Prepare a bounded command timeline from the admitted prompt and generation
budget before starting. It references reusable bodies rather than rebuilding
every layer's bindings for every token. Split the timeline into ordered command
buffers for completion reporting and submit the prepared work without token
readback. GPU decisions control whether a body runs. The collector observes
completed output windows; it never grants permission for another token.

Establish this scheduling mechanism on the device before designing the full
kernel headers. Check indirect execution dependencies, empty ranges,
cancellation, output visibility and prepared-timeline cost. Use supported Metal
completion/events for CPU visibility; do not transplant CUDA's mapped atomic
journal protocol unchanged. Cancellation needs an explicit supported
host-to-GPU publication mechanism independent of collection.

Command metadata scales with the admitted token budget; reusable bodies scale
with the model recipe. Measure preparation latency, metadata bytes and drain
time after early EOS. These costs differ from CUDA's conditional loop and must
appear in the sequence contract. Do not use cross-threadgroup spin barriers in
a persistent kernel without a forward-progress guarantee.

Follow Serenity's Metal 4 command-allocator and resource ownership pattern,
using Apple's pinned `metal-cpp` bindings and a compiled shader library. Initial
deployment proposal: Apple Silicon on macOS 26+, matching Serenity's Metal 4
baseline; first validation on M3 Max. Check capabilities at startup. M5-specific
tensor kernels in newer references are not the M3 implementation.

## Memory and model residency

`Run` owns a cache per trained model role under **R3 experiment scheduling**,
shared by CUDA and Metal. Each request uses fresh sequence/KV storage. The
retention decision follows this order:

1. Reserve the active request's KV, scratch, kernel workspace, capture leases,
   command storage and output records.
2. Require its weights to fit alongside that reservation.
3. Retain other useful models within the remaining budget. Evict idle models
   before refusing a request that can run sequentially.

Target and AV each occupy about 15.26 GiB in BF16 weights, about 30.51 GiB
together. AR and its head add about 11.25 GiB. The current 8192-position target
cache adds 1.125 GiB. Thus weight size alone cannot decide residency. Retaining
models does not imply running them concurrently.

On Metal, budget against the recommended working set, current allocations and
an explicit application limit. Unified memory is one physical pool. Expose
admitted and resident bytes so the scheduling decision is visible.

The executor uses one logical weight arena. Metal checks `maxBufferLength`;
arenas exceeding it use no-copy page-aligned views over one backing allocation.
Crossing operands receive overlapping views that contain the entire operand;
large transfers split at view boundaries. Aliases do not duplicate backing
charges. Mappings/staging survive all queued readers.

Admission conservatively reserves all non-parameter recipe tensors before
lifetime reuse, plus KV, captures, sampling, staging and command headroom.
After preparation the executor checks actual owned sequence buffers and native
command allocator bytes before start. Opaque driver/pipeline allocations still
need headroom; CUDA has no per-executable graph-memory query. Unified accounting
also charges mapped checkpoint payloads because sharing with uploaded weights
has not been established. `Run::retained_model_bytes()` exposes cached ownership;
`release_models()` releases idle weights while leaving records/captures usable.

Keep the existing preallocated contiguous KV strategy for the port. Paged and
rotating caches remain separate cache-family implementations. vLLM's attention
tiles and reductions are useful without importing its multi-request page pool.

## Kernel strategy

Design efficient prefill and decode together:

| Work | Metal implementation direction |
|---|---|
| Decode projections | Cooperative vector loads, SIMD-group reductions and shape-specialized matrix-vector kernels; reusable pipelines and bindings. |
| Prefill projections | Tiled matrix-matrix work, reusing weights across prompt rows. Evaluate M3-supported SIMD-group matrix operations against required BF16/FP32 arithmetic before selecting them. |
| Attention | Separate prefill and single-query paths. Register-resident online-softmax state and output; KV tile reuse; separate reduction dispatch where long-context partitioning needs it. Preserve GQA indexing and causal bounds. |
| RMS, RoPE, SwiGLU | Cooperative reductions and vector operations. Fuse only when specified BF16 rounding and hook boundaries remain intact. |
| Sampling and state | GPU RNG, temperature/top-p/greedy selection, EOS and position updates. No host random draw or sampled-ID feedback. |
| NLA operations | Device capture copies, normalization, embedding replacement, AR value head and steering; independent capture ownership across model changes. |

Some reference kernels use half-valued intermediates or different softmax
details. Do not adopt those silently. Preserve explicit conversion points and
the existing numerical policy. Freeze independent oracle inputs and limits
before evaluating alternate reductions/matrix instructions. Find the first
divergent operation when a candidate fails; do not change the oracle to admit it.

## Change axes and files

Keep the accepted [change axes](change-axes.md) and
[file mapping](file-mapping.md). Add implementations inside existing families.

| Ownership | Files and seams |
|---|---|
| X4 Metal resources, queues and capabilities | `src/gpu/metal/{device.h,device.cpp,backend.h,metal_cpp.cpp}`; explicit allocation and working-budget capabilities in `src/gpu/device.h` |
| X5 Metal operations and prepared sequence | `src/kernels/metal/operations.cpp`, `launch.h`, shaders for embedding, projection, normalization, rotary, attention, elementwise, sampling, decode, injection and steering |
| X1 preparation/placement | `src/inference/executor.*`, `storage_plan.*`; stable multi-buffer bindings, admitted workspaces and command costs |
| R3 residency | `src/experiment/run.*`; extract a pure budget/eviction-policy unit if its responsibility warrants one |
| Composition/build | `src/main.cpp`, CMake files and a Metal library helper; select the backend once, remove unconditional CUDA startup labels/includes |
| Launcher/tooling | `scripts/run.py` and tooling docs; native macOS build/run, Linux Docker/NVIDIA, existing verified-weight reuse and application entry point |

Tokenizer, chat formatting, Qwen3 recipe, NLA conventions, records, HTTP and UI
remain shared. Tensor representations and cache strategies remain explicit
families. Hardware resource handles and shader-specific layouts stay behind
backend boundaries, as Serenity's port demonstrates.

## Delivery

Deliver three complete phases, reviewing their file headers before implementation:

1. **Execution foundation and memory policy.** Native build, resources,
   prepared scheduling, completion/publication, cancellation, placement and
   budget-aware residency. Establish autonomous progress with collection
   deliberately delayed and account for prepared command storage.
2. **Complete target generation and capture.** Efficient prefill/decode,
   preallocated KV, GPU sampling/state and dynamic hooks. Compare logits,
   caches and captures with frozen independent fixtures. Profile GPU execution;
   report TTFS and steady-state tokens/s separately from loading/preparation.
3. **Complete local experiment.** AV interpretation, AR reconstruction,
   matched steering, native source launch and UI streaming. Verify sequential
   and retained-model operation, repeated requests and cancellation.

Before C++ header design and again before implementation, query both local
corpora and cite retrieved rules in file comments. This proposal consulted
Core Guidelines **R.1** (resource lifetime), **C.122** (backend separation), and
performance rules **GPU.1** (device data/control), **GPU.6** (batch/reuse work),
**GPU.7** (overlap and resource lifetime), **GDSA.4** (cross-workgroup progress).

Keep strict warnings, formatting and clang-tidy. Use Metal API/shader validation
and GPU profiling alongside numerical comparisons. Unit tests cover pure
admission, placement and state contracts. Revise the existing five-scenario
integration inventory for backend coverage of the same interactions; do not
hide unrelated workflows under parameterization or exceed five. Preserve CUDA
coverage when shared interfaces change.

The execution foundation is implemented. Full transformer inference and its
TTFS/token-rate measurements belong to the next phase.

## Foundation contracts

The accepted design maps into these concrete units:

| Header | Contract |
|---|---|
| `src/gpu/memory.h` | Backend capabilities/usage snapshots and owned memory footprints, with unified physical bytes counted once. |
| `src/experiment/model_retention.h` | Scheduler policy for retaining/releasing models after reserving the next request; no device operations or loading. Native GPU residency belongs to X4. |
| `src/gpu/metal/device.h` | Native Metal device ownership and required capabilities. |
| `src/gpu/metal/backend.h` | Private checked native bindings; stable owners through completion. |
| `src/gpu/metal/buffer_views.h` | Pure large-allocation view placement with whole-operand containment. |
| `src/gpu/metal/submission.h` | Complete prepared timeline, independent publication windows, command-allocator lifetime and completion. |
| `src/kernels/metal/sequence_plan.h` | Bounded prefill/head/decode schedule and output capacity, with an immediate first-selection publication boundary. |
| `src/kernels/metal/decode.h` | GPU control flow, pending-token consumption, cancellation and ICB ranges. |

`Run` owns the role-scoped model cache. `DeviceSequence` implements the shared
interface and reports prepared resources before execution admission. CUDA also
uses the new retention policy; its sequence arithmetic is unchanged.

Immutable control arguments and tables are reused across matching steps. Hook
constants share one bounded arena. Each publication window gets separate shared backing. A completion event makes
that window readable; subsequent GPU work cannot modify its resource. This
avoids concurrent CPU/GPU access to the live journal. Window preparation and
submission still occur independently of collection.

Cancellation uses a separate GPU control dispatch writing one
device-coherent atomic flag. It has no payload and no waiting kernel; initialization
precedes both queues. This uses the device coherence and atomic rules in the
[Metal Shading Language specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf),
sections 4.8 and 6.16. The M3 device integration exercises cancellation during
autonomous execution and confirms consumption of the final selected token.

Hardware discovery confirms that the current M3 Max's maximum allocation
exceeds a complete model arena. Multi-view placement remains a bounded fallback
for smaller limits, not an excuse to split every tensor or add a paging layer.

[llama-ops]: https://github.com/ggml-org/llama.cpp/blob/0a1ef309cdbb9f2024f58a38fb4c42352c5db569/ggml/src/ggml-metal/ggml-metal-ops.cpp
[llama-device]: https://github.com/ggml-org/llama.cpp/blob/0a1ef309cdbb9f2024f58a38fb4c42352c5db569/ggml/src/ggml-metal/ggml-metal-device.m
[vllm-decode]: https://github.com/vllm-project/vllm-metal/blob/55a3352cd7a16ce19825173e5252ca37a26e740a/vllm_metal/v1/decode_pipeline.py
[vllm-attention]: https://github.com/vllm-project/vllm-metal/blob/55a3352cd7a16ce19825173e5252ca37a26e740a/vllm_metal/metal/kernels_v2/pagedattention_tiled.metal
[vllm-memory]: https://github.com/vllm-project/vllm-metal/blob/55a3352cd7a16ce19825173e5252ca37a26e740a/vllm_metal/v1/cache_policy.py
[apple-indirect]: https://developer.apple.com/documentation/metal/mtl4computecommandencoder/executecommands(buffer:indirectbuffer:)
