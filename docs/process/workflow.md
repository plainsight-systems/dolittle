# Workflow

Dolittle follows Serenity's modified engineering workflow. This document is
self-contained; the repository does not import governance files.

## Design and implementation

1. Establish the logical architecture, its change axes, and the file mapping
   before writing engine code.
2. Write each new or changed file's contract, design, and relevant guideline
   citations in its header. Review those headers before implementation.
3. Implement the change and meaningful verification together.
4. Check correctness and cost, then update project memory and the work queue.

There are no implementation packets. Architecture documents describe the
system; file headers carry the design of the code they accompany.

## C++ and performance

- Consult the `cpp-guidelines` corpus before writing C++; consult
  `cpp-perf-guidelines` for performance-sensitive work. Cite the relevant
  rule IDs in the design headers. If either required server is unavailable,
  report that and obtain explicit agreement before writing C++.
- Design optimizations in. Mark each with `Optimization:` and its concrete
  reason. Measure the effect where possible.
- Derive data-path costs before implementation: weight and cache bytes,
  temporary buffers, copies, allocations, inner-loop work, dispatches,
  synchronization, and readback. Express costs in model dimensions, context
  length, capture count, and generated explanation length.
- Avoid multiplying work per token or element when it can be performed per
  batch or phase. Measure the running path against the derived costs.
- Treat loading, inference, adapter application, activation capture, and GPU
  scheduling as performance-sensitive by default.
- Python tooling and browser code are reviewed for their own correctness and
  performance; C++-specific guidelines do not apply to them.

## Verification

Verify GPU computation on the 3090 against appropriate independent reference
results. Define numerical comparisons for logits, captured activations,
injected inputs, and reconstructed vectors as those paths are designed.
Generated prose alone is insufficient evidence of numerical correctness.

Keep build/non-GPU checks separate from GPU execution checks. Record the device,
driver, toolchain, precision, model revision, and workload for measurements.
Exercise the retained PTX path with `CUDA_FORCE_PTX_JIT=1`. Compatibility on
untested hardware is an expectation, not a measured result.

Independent review is on request, not automatic per commit. Findings and their
resolutions should be recorded with the work they concern.

## Milestones and durable records

Close a milestone with the relevant correctness and performance checks, a
documentation pass, and a write-up in `docs/writeups/`. Integrate completed work
into `main` by fast-forward. Publication follows the user's instructions.

Record settled decisions in `MEMORY.md`, current and completed work in
`QUEUE.md`, and investigations in dated files under `docs/research/`. Preserve
the distinction between planned and implemented behavior in public documents.

Reduce scope instead of weakening quality within the chosen scope. Explain
and record intentional changes to the workflow and project contracts.
