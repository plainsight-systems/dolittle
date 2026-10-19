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
  Andrew accepted the slice and authorized commit/push on 2026-10-18. Deliver
  complete accepted slices, not partial batches presented as completed slices.

- The project site is deployed at `dolittle.plainsight-systems.com`. The first
  build and deployment passed; HTTP content, Code link, and share metadata are
  verified. Check certificate provisioning before enabling enforced HTTPS.
  Engine architecture follows this site work; preserve the selected 8B pair
  and block 24 reference contract.

## Next

1. Commit and push the accepted first-forward implementation, including strict
   builds, editor configuration and readable algorithm/implementation comments.
2. Design the next complete target-generation slice: exact tokenizer/text
   protocol, persistent KV storage and token selection/termination. Keep model,
   tokenizer, cache and backend variation in their existing families. Query both
   C++ corpora for header design and again for implementation; retain contracts,
   algorithm responsibilities and exact applied citations in file openings.
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
