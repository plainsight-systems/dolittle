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
model as the entire architecture. No source stubs or engine implementation
have been introduced by these documents. Next, design file headers and
numerical checks for the first implementation slice.

The selected target is **Qwen/Qwen3-8B**, paired with
**marco-bazzani/Qwen3-8B-nla**. The tested revisions and file hashes are in
[`models-8b.json`](../../scripts/validation/models-8b.json); the numerical
contracts and measurements are in the
[comparison](../research/2026-10-16-nla-comparison/README.md).

Begin with zero-based block 24 captures, the separate full AV and 28-block AR
weights, and the AR's separate value head. Sequential BF16 execution fits the
24 GB 3090 Ti in the reference workload. Use the accepted overlays and mapping
to design loading, cache ownership and capture/explanation scheduling in the
file headers; add source files as their implementation slices require them.
