# Engine source

Full-forward and target-generation execution are implemented in C++20/CUDA,
following the accepted
[file mapping](../docs/architecture/file-mapping.md). It loads a verified Qwen3
target checkpoint and evaluates a complete causal sequence, returning selected
logits and configurable block observations. The cached path reuses rotated K
and projected V across chunked prefill and one-token decode within one turn. See the [verification commands](../tests/README.md)
for workloads, reference policy and limits.

The host library owns checked tensor layouts, BF16 conversion, observation and
scratch planning, operation/recipe validation, Qwen3 model recipes, safetensors
loading and checkpoint binding. Safetensors payloads use read-only mappings;
size/index/path checks precede publication. SHA-256 is checked at open by
default; the application explicitly admits an unchanged installation verified
during setup, avoiding full weight scans at each launch. Shared bounded JSON parsing
rejects duplicate keys. Metadata parsing uses nlohmann/json 3.12.0, and OpenSSL 3
supplies SHA-256. Byte BPE is implemented directly, using source-pinned utf8proc
2.10.0 for NFC and Oniguruma 6.9.10 for Unicode regex matching. These helpers
do not implement tokenization or model computation.

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
in-memory contracts, file integration, device behavior and numerical generation.

To compose a target run, construct the tokenizer and text protocol from verified
artifact bytes, prepare the complete prompt, then prepare a cached plan for the
prompt plus maximum emitted-token count. Bind a categorical/greedy selector and
call `inference::generate`. The generation loop preserves emitted and consumed
positions separately, consumes the final emitted token even on stop, and returns
owned IDs/captures after completion. Thinking segmentation is a separate text
protocol operation; it never changes the exact trace.

The contiguous cache is the deliberately simple initial strategy: allocate
once per turn, reuse within that turn, and release after completion. Pooling,
cross-turn prefix reuse and alternative cache strategies are later work.

The `dolittle` executable composes the single-worker experiment session,
HTTP boundary and CUDA backend. `scripts/run.py` builds and launches it from
source. Session and transport use shared execution contracts; CUDA selection
lives in `main.cpp` and the backend build target. The browser lives in `app/`.
Target-to-AV interpretation is implemented; reconstruction remains later work.
Metal is planned after NVIDIA, and AMD support is undecided.
