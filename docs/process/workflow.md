# Workflow

Dolittle follows Serenity's modified engineering workflow. This document is
self-contained; the repository does not import governance files.

## Design and implementation

1. Establish the logical architecture, its change axes, and the file mapping
   before writing engine code.
2. For C++ header design, query both local corpus MCP servers as specified
   below. Write each new or changed file's contract, algorithm responsibility,
   design and retrieved guideline citations in its opening comments. Review
   those headers before implementation.
3. For C++ implementation, query both servers again for the concrete work.
   Implement the change and meaningful verification together, applying the
   retrieved guidance and keeping the header comments current.
4. Check correctness and cost, then update project memory and the work queue.

There are no implementation packets. Architecture documents describe the
system; file headers carry the design of the code they accompany.

## Required C++ corpus consultation

Both **header design and implementation work must query both locally running
MCP servers**:

- `cpp-guidelines`: C++ best practices / Core Guidelines, exposed through
  the `cpp_guidelines` MCP tools.
- `cpp-perf-guidelines`: the C++ performance corpus, exposed through the
  `cpp_performance` MCP tools.

This requirement applies to new and changed C++ code, including CUDA C++ and
its host dispatch code. Performance-corpus consultation is mandatory in both
phases, not conditional on first labeling a change performance-sensitive.

Use each server's `search_guidelines` for the actual design or implementation
concerns, then `get_guideline` to read the specific rules used. During header
design, investigate contracts, ownership, lifetimes, interfaces and expected
costs. During implementation, query for the concrete algorithms, data layout,
allocation/copy behavior, dispatch and synchronization involved. Earlier
header citations do not replace implementation-phase consultation. Reuse
retrieved guidance within a coherent work phase; do not make redundant calls
for every function.

Cite the returned corpus name and exact rule ID in the file's opening
comments, along with how the rule applies to that file. Include the returned
source reference when available. Do not invent IDs, cite unread rules, or use
memory, generic advice or web searches as a substitute for these MCP queries.
If a search yields no applicable rule for a corpus, record the queried topic
and that result instead of attaching an unrelated citation.

If either server is unavailable, report which one and stop the affected C++
header-design or implementation work. Obtain explicit user agreement before
proceeding without it; do not silently fall back to another source. This does
not prevent unrelated documentation or other independent work.

## Required file header comments

Each new or changed C++ file, including CUDA C++, must begin with comments
describing its own responsibility. This includes interface headers and implementation files;
an implementation may refer to its interface's public contract, but must
still explain its local contract and algorithm. Keep these comments in sync
with the body as the implementation changes.

Include:

1. **Axis and responsibility:** the change axis, concrete implementation kind
   where applicable, and the work owned by this file.
2. **Contract:** inputs and outputs, preconditions/postconditions, invariants,
   ownership and lifetime, failure behavior, and relevant numerical or
   synchronization requirements.
3. **Algorithm:** the part of the accepted algorithmic overlay performed by
   this file, with a reference to its document and section. Describe the local
   steps and data/state transitions, including what must happen before and
   after them. A label such as “implements attention” is insufficient.
4. **Design and cost:** consequential choices, allocations/copies, work
   scaling and ordering requirements. Identify planned versus measured costs.
5. **Corpus guidance:** applicable rules retrieved from both local MCP
   corpora, with exact IDs and a short explanation of each rule's application,
   or the explicit no-applicable-rule result described above.

For example, a capture-related header should state which block output it
observes, how its position relates to the consumed token, who owns the copied
bytes, and what completion must precede buffer reuse. A file outside the
transformer path should identify its corresponding experiment/setup step
instead of inventing a transformer responsibility. If its algorithm is absent
from the design, clarify that design before implementation.

## C++ performance design

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
