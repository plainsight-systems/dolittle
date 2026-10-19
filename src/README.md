# Engine source

The first forward-pass slice is implemented in C++20/CUDA, following the accepted
[file mapping](../docs/architecture/file-mapping.md). It loads a verified Qwen3
target checkpoint and evaluates a complete causal sequence, returning selected
logits and configurable block observations. K/V are computed within that pass;
there is no persistent decoding cache. See the [verification commands](../tests/README.md)
for workloads, reference policy and limits.

The host library owns checked tensor layouts, BF16 conversion, observation and
scratch planning, operation/recipe validation, Qwen3 model recipes, safetensors
loading and checkpoint binding. Safetensors payloads use read-only mappings;
size/hash/index/path checks precede publication. Shared bounded JSON parsing
rejects duplicate keys. Metadata parsing uses nlohmann/json 3.12.0, and OpenSSL 3
supplies SHA-256. Neither library implements the model computation.

CUDA device code owns buffers, staging and ordered queues. Prepared operations
validate bindings and dispatch directly implemented embedding, projection,
normalization, rotary, attention and elementwise kernels. The executor uploads
bound weights, plans scratch lifetimes and copies selected observations before
reuse. It returns owned host-ready results after queue completion. Model equations
live in the model recipe; shared execution has no hardcoded capture block.

Private CUDA bridges and launch/conversion helpers stay under their concrete
backend folders. File opening comments state contracts, algorithm steps, costs
and the rules consulted from both local C++ corpora. These are working source
files, not placeholder success bodies. [Tests](../tests/README.md) distinguish
in-memory contracts, file integration, device behavior and numerical forwards.

The current executable entry points are verification drivers. Tokenizers,
text protocols, sampled generation, persistent KV strategies, NLA verbalizer/
reconstructor execution, serving and the browser application remain later slices.
Metal is planned after NVIDIA; AMD support is undecided.
