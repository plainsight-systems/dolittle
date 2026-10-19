# Architecture

The [project-site architecture](project-site.md) describes the implemented
static site, publication boundary, and editorial structure.

The engine architecture starts with two algorithmic overlays, reviewed and
accepted on 2026-10-17:

1. [NLA experiment](logical-overview.md): generation, capture alignment,
   interpretation, optional reconstruction, and model/state lifetimes.
2. [Transformer execution](transformer-execution.md): block computation,
   prefill and decoding, NLA entry/exit paths, observation boundaries, and
   backend variation to examine next.

The [change-axis design](change-axes.md) and revised
[file and folder mapping](file-mapping.md) were accepted on 2026-10-17. Following
Andrew's feedback, the mapping exposes model families, tokenizer kinds, tensor formats
and representations, cache strategies, sampling mechanisms, NLA pairs and
backends as explicit families with named initial implementations. It assigns
ownership, dependencies and header contracts without treating the selected
model as the entire architecture. The first forward-pass file-header designs
are complete. The first full forward-pass sources now implement loading and
binding, host preparation, CUDA resources and kernels, execution and completed-
result comparison. See [build commands](../../tests/README.md) for the separate
host, archive, device and forward scenarios; build success alone does not
establish equivalence.

CUDA 13.2 compilation uses a digest-pinned Docker environment, with nlohmann/json
3.12.0 and OpenSSL 3 supporting metadata and hashing. The build retains native
Ampere code plus PTX. The image builds; GPU execution currently uses the native
Linux host because NVIDIA Container Toolkit access is not configured. Native
and forced-PTX results cover the tested short-sequence/reference policy only;
long-context behavior, newer GPUs and container GPU execution remain unverified.

The selected target is **Qwen/Qwen3-8B**, paired with
**marco-bazzani/Qwen3-8B-nla**. The tested revisions and file hashes are in
[`models-8b.json`](../../scripts/validation/models-8b.json); the numerical
contracts and measurements are in the
[comparison](../research/2026-10-16-nla-comparison/README.md).

The selected pairing requires zero-based block-24 output, but shared observation
planning accepts any model-advertised block input/output. The separate full AV,
28-block AR and AR value head remain later implementation work. Sequential BF16
execution fits the 24 GB 3090 Ti in the historical reference workload.

The implemented scope includes full target forward execution and cached target
generation, with tokenizer/text protocols, sampling and completed raw captures.
The next complete target-to-AV slice has
[header designs for review](file-mapping.md#target-to-av-interpretation-header-design):
pairing and injection, resident weights with fresh sequence caches, retained
records and phase scheduling. Those new declarations are not an implemented NLA
path. Reconstruction, serving and the browser application remain subsequent work.
