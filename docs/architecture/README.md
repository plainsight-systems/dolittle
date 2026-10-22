# Architecture

Dolittle implements the complete Qwen3-8B/NLA experiment on CUDA and Metal:
target generation, block-24 activation capture, verbalization, reconstruction
and matched steering. Both backends use BF16 tensors and FP32 reductions and
serve the same browser application.

The CUDA workflow runs on Linux through Docker or a source-built native binary.
The Metal workflow runs natively on Apple Silicon. The application has
been tested on an RTX 3090 Ti with 24 GB VRAM and an M3 Max with 128 GB unified
memory. Native Windows/NVIDIA remains planned; AMD support is undecided.

## Computation and execution

1. [NLA experiment](logical-overview.md): how a target generation becomes
   retained captures, descriptions and steering trials.
2. [Transformer execution](transformer-execution.md): the equations, tensor
   shapes, rounding boundaries, cache lifetime and observation sites.
3. [GPU execution boundary](execution-boundary.md): GPU-owned prefill, sampling,
   decoding and capture; CPU setup, text encoding and completed-output collection.
4. [Steering](steering.md): reconstruct the original and edited descriptions,
   form their vector difference, and apply it at the selected residual row.
5. [Metal implementation](metal-port.md): native resources, indirect execution,
   kernels and unified-memory admission, informed by Vigil, Serenity, llama.cpp
   and vLLM-Metal.

The selected pair is **Qwen/Qwen3-8B** with **marco-bazzani/Qwen3-8B-nla**.
Pinned revisions and file hashes live in [`config/models.json`](../../config/models.json).
The verbalizer reads zero-based block 24 output. Shared hooks support the model's
advertised block input/output sites; the pair chooses which site its trained NLA
expects. The reconstructor uses 28 blocks and a separate value head.

## Ownership and application

The [change axes](change-axes.md) and [file mapping](file-mapping.md) separate
model mathematics, tokenizer kinds, tensor formats, cache strategies, NLA pairs
and backend implementations. The Metal port uses these existing boundaries.

`Run` schedules one model role at a time. It keeps idle weights loaded when there
is room for the next model, its KV cache and working buffers, and evicts them
when needed. On unified
memory, admission also accounts for mapped checkpoints. Each sequence gets fresh
KV and scratch storage. Captures and interpretation records outlive those
sequence resources and model switches.

The [live application](live-application.md) uses one source launcher:
`python3 scripts/run.py` selects Linux Docker or native macOS Metal. `Session`
owns the worker and bounded event journal, HTTP owns transport, and the browser
owns presentation. [Setup](../../app/README.md) explains requirements and access.
The [project site](project-site.md) is a separate static Hugo site.

## Verification

The [numerical checks](numerical-portability.md) compare GPU operations with
independent reference calculations, reject non-finite values and malformed
results, and require exact replay and full/cached agreement on each backend.
Metal reports whole-model differences against the original fixture limits;
CUDA enforces those limits. Both use the original oracle calculations.

The existing five integration workflows cover site publication, the live
application, device resources, tiny models and the pinned 8B pair. Both model
workflows include interpretation and steering. [Verification commands](../../tests/README.md)
cover those workflows, host units, strict builds, formatting and static analysis.
