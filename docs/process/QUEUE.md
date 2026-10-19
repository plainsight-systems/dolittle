# Work queue

## Active

- Architecture review is complete: use the accepted
  [file and folder mapping](../architecture/file-mapping.md) and change axes
  as the baseline for file-header design. Preserve explicit implementation
  families and keep the selected model as their first concrete implementation.

- The complete first no-cache C++/CUDA forward slice is implemented and verified.
  Native and forced-PTX tiny/8B numerical scenarios pass on the 3090 Ti, as do
  device/loader integration, host units/sanitizers and CUDA memcheck. Exactly
  five integration scenarios remain. Detailed development records stay local.
  Accepted and pushed to origin/main as `77f038c`. Deliver complete accepted
  slices, not partial batches presented as completed slices.

- Target generation is implemented and verified across the accepted tokenizer,
  text, sampling, cache and execution families. Strict host/CUDA builds, 15 host
  unit tests, Linux ASan/UBSan, tiny native/PTX generation, CUDA memcheck, archive,
  device and site checks pass. Pinned tokenizer parity passes 405 ID vectors and
  12 prompt cases. The integration inventory remains exactly five.
  Independent numerical acceptance now passes against the restored oracle with
  unchanged generator, original 8B fixture and limits (checksum verified).
  Native and forced-PTX tiny FP32/BF16 and 8B pass; all 485 measured 8B numerical
  comparisons have zero error. Exact cache/full-trace comparisons and tiny CUDA
  memcheck pass. Engine fixes cover projection partitions, RMSNorm reduction
  layouts and attention probability/value rounding. The earlier engine-aligned
  reference claim remains withdrawn; only independent evidence closes this gate.
  Accepted and pushed to origin/main as `20d8740`.

- Clang-tidy tooling and source cleanup follow Serenity's check families, with
  a pinned optional analysis image, host/device CUDA parsing and a VS Code task.
  Checked host bounds, explicit resource states and named argument roles replace
  fixable findings; necessary native API exceptions are local and explained.
  Strict host/CUDA builds, 16 host units, ASan/UBSan, archive/site checks and
  native/PTX tiny/8B generation pass. The frozen oracle and limits remain fixed;
  8B numerical comparisons retain zero error and tiny CUDA memcheck is clean.
  Analysis is clean on macOS and Linux/CUDA. Accepted and pushed as `b2be166`.
  Diagnostic inventories and detailed reports remain in ignored storage.

- The approved target-to-AV header design is committed as `996aac1`. The complete
  implementation now covers pinned pairing/normalization, explicit embedding
  replacement, resident weights with fresh sequence caches, retained records,
  scheduling and phase measurements. Both C++ corpora were queried and cited.
  Strict host/CUDA builds, 18 host units, Linux ASan/UBSan, archive/site checks
  and host/CUDA clang-tidy pass. Native and forced-PTX tiny/8B target-to-AV
  numerical workflows pass with unchanged target oracle and separately frozen
  AV fixtures/limits; CUDA memcheck reports zero errors. Normalization/injection
  bytes and final cached/full AV logits match exactly. Independent AV output
  comparisons pass the fixed per-row limits, without a bit-exactness claim.
  The integration inventory remains five; former generation workflows include AV.
  Implementation is accepted for commit; publication remains pending. See the
  [slice mapping](../architecture/file-mapping.md#target-to-av-interpretation-header-design).

- The project site is deployed at `dolittle.plainsight-systems.com`. The first
  build and deployment passed; HTTP content, Code link, and share metadata are
  verified. Check certificate provisioning before enabling enforced HTTPS.
  Engine architecture follows this site work; preserve the selected 8B pair
  and block 24 reference contract.

## Next

1. Configure host NVIDIA Container Toolkit access before claiming Docker GPU
   execution. The digest-pinned source build and native Linux execution work;
   this setup limitation does not invalidate the recorded native checks.
2. Continue toward the server and synchronized browser interface, with complete slice boundaries and at most five integration
   scenarios across the repository. Long-context and additional GPU validation
   remain explicit future evidence rather than inferred compatibility.

## Later platform work

- Implement Metal on macOS after Linux/NVIDIA. Design its native build path
  and verify hardware requirements in that phase.
- Decide whether to support AMD.

## Deferred
- Browser-only inference.
- Explanation editing and causal steering.
- Training an NLA, unless the selected model pairing requires it.
