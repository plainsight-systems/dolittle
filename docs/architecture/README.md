# Architecture

The [project-site architecture](project-site.md) describes the implemented
static site, publication boundary, and editorial structure.

The engine architecture remains a separate design task. Write these documents in order:

1. `logical-overview.md`: responsibilities, data flow, and system contracts.
2. `change-axes.md`: independent reasons the implementation will change.
3. `file-mapping.md`: ownership, dependencies, and mapping those axes to files.

The selected target is **Qwen/Qwen3-8B**, paired with
**marco-bazzani/Qwen3-8B-nla**. The tested revisions and file hashes are in
[`models-8b.json`](../../scripts/validation/models-8b.json); the numerical
contracts and measurements are in the
[comparison](../research/2026-10-16-nla-comparison/README.md).

Begin with zero-based block 24 captures, the separate full AV and 28-block AR
weights, and the AR's separate value head. Sequential BF16 execution fits the
24 GB 3090 Ti in the reference workload; weight loading, cache ownership, and
capture/explanation scheduling need an application design. Decide source
subdivisions from that design rather than creating speculative modules now.
