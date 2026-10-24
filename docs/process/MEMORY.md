# Engineering decisions

## Scope and platforms

Dolittle implements target generation, activation interpretation, reconstruction
and steering for Qwen3 8B and its trained NLA. The C++20 engine runs on CUDA
and Metal. Linux/NVIDIA uses a source-built Docker image; Apple Silicon uses
the native Xcode/Metal toolchain. Both launch with `python3 scripts/run.py`.
Native Windows/NVIDIA is planned; AMD support is undecided.

`app/` is the browser interface served by the engine. `web/` is the separate
Hugo project site. The public site builds independently of the engine and model
downloads. Code and configuration use Apache-2.0; documentation and original
artwork use CC BY 4.0.

## Model and computation

- The target is Qwen/Qwen3-8B, paired with marco-bazzani/Qwen3-8B-nla.
  Revisions, files and hashes are pinned in
  [`config/models.json`](../../config/models.json).
- The pair reads the output of zero-based block 24. Shared observation hooks
  support the model's advertised block input/output sites; the pair selects
  its trained site. The verbalizer uses the pair's trained prompt, embedding
  replacement and normalization scale of 150.
- The reconstructor uses 28 transformer blocks and a separate value head.
  Steering subtracts the original description's reconstruction from the edited
  description's reconstruction, normalizes the difference and scales it by the
  captured activation norm and requested strength, then applies it once at the
  selected residual row.
- Tensors use BF16 storage with FP32 reductions. Model computation and sampling
  are implemented directly; PyTorch supplies independent numerical references.
  Quantization is outside the current implementation.

The [experiment overlay](../architecture/logical-overview.md) and
[transformer overlay](../architecture/transformer-execution.md) define the
algorithm. The [change axes](../architecture/change-axes.md) and
[file mapping](../architecture/file-mapping.md) separate model families,
tokenizers, tensor formats, cache strategies, NLA pairs and backends.

## Execution and lifetime

Prefill, decoding, sampling, RNG state, stopping, capture, NLA injection and
steering stay on the GPU. CPU work prepares requests, encodes and decodes text,
collects output and handles cancellation. Output collection does not control
token progression. CUDA uses conditional graphs; Metal uses indirect command
buffers. See the [execution boundary](../architecture/execution-boundary.md).

Each generation allocates a contiguous KV cache and working buffers before
execution, reuses them through the sequence and releases them after completion.
The live generation path preallocates storage instead of growing it per token.
Cross-turn cache pooling and prefix reuse are not implemented.

`experiment/model_retention` keeps idle weights loaded when the next model,
cache and working buffers fit; otherwise it evicts idle models. Unified-memory
accounting includes mapped checkpoints. GPU captures outlive model switches and
sequence resources. GPU residency mechanisms belong to the backend.

One session worker owns the device and experiment. HTTP threads submit commands
and read immutable events. Experiments remain in memory; browser refresh
reconnects, while a new generation or service restart clears the experiment.
Interpretation samples and steering comparisons retain their inputs and settings.
See [application ownership](../architecture/live-application.md).

SSH access supports loopback hostnames and forwarded ports. The connection
panel extracts the key from a launcher link even when its address differs;
requests continue through the current browser address. API authentication and
Origin checks remain enabled. See [connection setup](../../app/README.md).

## Setup and verification

Setup downloads model files and verifies SHA-256 hashes, then saves a receipt
bound to the manifest and file metadata. Subsequent launches reuse that receipt.
Changed files require setup again; `--verify-models` forces a full check and
repair. Models remain in an ignored cache and are not included in Docker images.

The [numerical contract](../architecture/numerical-portability.md) requires
independent operation checks and exact same-backend replay/full-versus-cached
agreement. CUDA enforces the frozen whole-model limits. Metal reports errors
against those limits and checks finite, structurally valid outputs. Preserve
independent oracle calculations, fixtures and reported differences.

CUDA builds retain native Ampere code and PTX; verification exercises both
native and forced-PTX execution. The tested systems are an RTX 3090 Ti with
24 GB VRAM and an M3 Max with 128 GB unified memory.

Builds use warnings as errors, Google formatting and clang-tidy. CMake presets
supply editor configuration. C++ header design and implementation both consult
the local best-practices and performance corpora. Units test isolated contracts;
the repository has at most five integration workflows. Commands and requirements
live in [verification](../../tests/README.md) and [workflow](workflow.md).
