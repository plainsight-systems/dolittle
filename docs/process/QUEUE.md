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
  Implementation is committed as `8c1fc19` and pushed to origin/main. See the
  [slice mapping](../architecture/file-mapping.md#target-to-av-interpretation-header-design).

- Google formatting now has a repository config, pinned editor/CLI formatter,
  C/C++/CUDA format-on-save, VS Code tasks and a PR/push CI check. The existing
  sources have been formatted with the Google preset. Formatting checks, strict
  host/CUDA builds, 18 host units and host/CUDA clang-tidy pass (39/61 commands).
  A mechanical comparison confirms source changes are formatting and comment
  layout only, including three relocated tokenizer lint annotations. Formatting
  changes are committed as `fa2668d` and pushed to origin/main.

- The project site is deployed at `dolittle.plainsight-systems.com`. The first
  build and deployment passed; HTTP content, Code link, and share metadata are
  verified. Check certificate provisioning before enabling enforced HTTPS.
  Engine architecture follows this site work; preserve the selected 8B pair
  and block 24 reference contract.

- The accepted [live application slice](../architecture/live-application.md)
  is implemented: source launcher, pinned cache acquisition, single-worker session,
  strict HTTP boundary, live byte/token progress, capture selection, AV variants,
  scoped cancellation and cursor reconnect. Shared execution compiles without
  CUDA under the host preset. The real pinned 8B application flow passes.
  Strict host/CUDA builds, Google formatting, host/CUDA clang-tidy (46/65
  entries), 19 C++ units and Linux ASan/UBSan pass. Twenty Python units, five
  browser-controller units, site publication and the real application flow pass.
  Native/PTX tiny and 8B numerical checks pass with frozen references unchanged;
  tiny CUDA memcheck reports zero errors. Desktop/mobile selection, AV variants,
  scoped stop and refresh recovery are verified in the real browser. The source
  runtime image builds; Docker GPU execution passes on the 3090 Ti with Toolkit
  1.20.1. The container application flow verifies target generation, capture
  selection, AV interpretation, idempotent retries, exact reconnect and shutdown.
  The integration inventory remains five, replacing standalone archive coverage
  explicitly. Accepted for commit and push.

- The load-cost fix retains shared verified host archives across turns and
  model switches. Setup now persists verification receipts; unchanged launches
  check file metadata without full hashing. Explicit repair forces SHA-256. Strict builds,
  host/CUDA static analysis, ownership units and sanitizers pass. Repeated real
  8B requests are substantially faster with exact pre-change target/AV token
  IDs. Container retry/reconnect/shutdown and corrupt-cache rejection pass.
  Device, tiny and pinned 8B numerical comparisons also pass inside Docker
  using native GPU code, with frozen references and limits unchanged.
  Prepared-cache admission and archive opening now take milliseconds; full
  SHA-256 runs during initial setup or forced repair. Receipt contract units
  and the Docker application flow cover reuse, invalidation and verification.
  Detailed timings remain in ignored local validation records.

- The live interface now separates thinking/answer completion states, exposes
  their shared budget, streams NLA output, and keeps selection excerpts beside
  interpretations. Desktop/mobile review and seven controller units pass.
  Detailed UI validation remains in ignored local records; accepted for commit and push.

- Exploration budgets now default to 4,096 target output tokens with an 8,192
  total context ceiling; AV remains independently adjustable. Strict CUDA build,
  19 C++ units, seven browser units, Google formatting and all 65 CUDA/host
  clang-tidy entries pass. The Docker application flow passes. A longer live
  generation crossed the former context limit; full 8,192-token capacity was
  allocated successfully on the 3090 Ti. These runtime checks do not extend
  the frozen short-sequence numerical oracle evidence. Changes are uncommitted.

## Next

1. Measure time to first generated token and generation throughput after the
   live-application push. Keep performance records local.
2. Add the public workflow walkthrough when requested. Long-context numerical
   comparison and additional GPU validation remain future evidence.

## Later platform work

- Implement Metal on macOS after Linux/NVIDIA. Design its native build path
  and verify hardware requirements in that phase.
- Add native Windows/NVIDIA builds with MSVC/CUDA, without requiring WSL2.
- Decide whether to support AMD.

## Deferred
- Browser-only inference.
- Explanation editing and causal steering.
- Training an NLA, unless the selected model pairing requires it.
