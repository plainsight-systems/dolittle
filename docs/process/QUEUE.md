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


- The project site is deployed at `dolittle.plainsight-systems.com`. The first
  build and deployment passed; HTTP content, Code link, and share metadata are
  verified. Check certificate provisioning before enabling enforced HTTPS.
  Engine architecture follows this site work; preserve the selected 8B pair
  and block 24 reference contract.

## Next

1. Commit and push the accepted target-generation slice, as authorized by Andrew.
   Keep the independent reference and original limits fixed.
2. Make cleanup a separate commit before further implementation. Development
   diagnostics and reports remain in ignored local storage.
3. Configure host NVIDIA Container Toolkit access before claiming Docker GPU
   execution. The digest-pinned source build and native Linux execution work;
   this setup limitation does not invalidate the recorded native checks.
4. Continue toward selected target/NLA phase scheduling and the synchronized
   browser interface, with complete slice boundaries and at most five integration
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
