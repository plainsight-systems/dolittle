/*
 * Sequential target/AV orchestration contract.
 * Axis R3: admission, one active experiment, model residency and cancellation.
 * Contract: Run owns its pair, model providers, text helpers and retained
 * record; it borrows one device that must outlive it. Public calls are
 * synchronous and serialized, not reentrant or concurrently polled. The device
 * is exclusively leased to this Run by composition; multiple Run objects are
 * not GPU scheduling. Providers open verified host checkpoints on demand, never
 * upload or cache GPU weights. LoadedModel source identity must describe the
 * returned checkpoint's verified backing archive; validate it against Pair
 * before device preparation. A pair's source metadata may be prepared
 * separately without retaining all model payloads. The host budget includes the
 * active loader's reported working allowance; mapped file address space and OS
 * page cache are reported separately.
 *
 * Algorithm: docs/architecture/logical-overview.md, "1 Establish the run",
 * "2 Generate and capture", "3 Retain the experiment and change phases", and
 * "4 Interpret selected positions". Let P be prompt length, G target cap,
 * Q the fixed AV prompt length and A an interpretation's output cap.
 * 1. Before target upload, format/encode P tokens, validate settings and admit
 *    C=P+G capacity (including final emitted-token consumption), all capture
 *    bytes/indexes, decoded text bounds and transient/readback budgets. Load
 * and validate target provenance and the pair's capture site.
 * Overflow/unsupported inputs fail before generation; never shorten a request
 * to fit silently.
 * 2. Load one resident target weight owner, prepare one sequence plan, and call
 *    generate with the pair's dynamic observation site. Retain only completed
 *    captures on cancellation/failure; decode/classify and move into Record.
 * 3. Finish/drain all dependent work; destroy sequence KV/scratch/staging, then
 *    resident target weights and host checkpoint. Publish retained state only
 *    after this release. An allocation failure cannot erase an already
 * committed record; before one can be materialized, the call may throw without
 * a result.
 * 4. interpret(key,settings) validates the retained row and reserves an
 * attempt's record/temporary space. Normalize only that row through Pair. On
 * the first request, open/validate/upload AV after target release; keep its
 * weights resident between requests. No target rerun or target+AV GPU overlap
 * occurs.
 * 5. Admit Q+A capacity and a fresh cache/sequence ID. Install the owned
 * injection during preparation. Generate using the exact AV prompt, selected
 * sampler and seed; consume the final emitted token before sealing as in target
 * mode. No target observation hooks are requested on the AV. Decode output,
 * extract tags, append a variant, and release sequence resources while
 * retaining AV.
 * 6. A canceled request records partial output and returns to AV-ready after
 *    cleanup. A load failure returns to retained so a later request may retry.
 *    A device failure destroys the poisoned resident owner and closes this Run;
 *    in-place device recovery is outside this slice. Record data remains
 * available, with no automatic target rerun. close releases AV resources but
 * keeps the record and forbids further generation/interpretation. Destructor
 * drains before release. Invalid calls (second target generation, unknown
 * capture, closed Run, rejected admission) throw before an attempt is accepted.
 * Recoverable execution errors append a failed attempt. bad_alloc may prevent
 * recording an error and propagates; existing record entries remain intact. A
 * stop requested before interpret starts does no GPU work and creates a
 * canceled attempt if its record slot is admitted. Preflight rejection leaves
 * empty state retryable. Once target execution starts, failure to materialize
 * any record closes the Run; it cannot silently rerun. Execution/sequence IDs
 * are monotonic within Run, separate from RunId and VariantId. Reject counter
 * overflow; never reuse an ID after a failed attempt.
 *
 * Baseline: one target generation per Run; one row per interpretation. Callers
 * explore neighboring rows by successive calls, obtaining separate variants.
 * Explicit settings/seed are always recorded; initial exploration uses
 * temperature 1, while deterministic numerical fixtures use greedy selection.
 * No confidence score, AR, HTTP server or browser behavior belongs to this
 * slice.
 *
 * Cost: peak device arena bytes = max(target weights+target sequence,
 * AV weights+AV sequence), not their sum. Per-model KV is 2*L*C*Hkv*d*b;
 * L=36,Hkv=8,d=128,b=2 gives 147456*C bytes. Account separately for scratch,
 * weight upload staging, injection staging, readback and driver headroom.
 * Host peak includes retained captures 8192*(P+G), record/index/text
 * allowances, one pending variant, active loader, tokenizers, normalized row
 * and staging. Host estimates cover admitted payload/staging and supplied
 * loader/text metadata; recipe/kernel metadata, allocator overhead and mapped
 * address space are excluded. These are allocation estimates, not measured
 * process/device peaks. Optimization: retain AV weights once; fresh contiguous
 * KV per variant keeps isolation simple. Allocation pooling remains deferred,
 * as in target generation. Timings separate loading, target generation,
 * release/switch, AV generation and decoding; GPU completion bounds compute
 * time. No performance claim is made. Corpus consultation, header design and
 * implementation, 2026-10-19 (full rules read): cpp-guidelines R.1 (#rr-raii):
 * sequence dies before resident owner/device. cpp-guidelines I.7 (#ri-post):
 * retained state means target GPU resources released. cpp-perf-guidelines
 * GPU.9: separate persistent weights from transient sequence arenas; source
 * guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 * cpp-perf-guidelines GPU.1: upload each model per phase and one row per
 * attempt; source guidelines/gpu/GPU.1-keep-data-on-device.md.
 */
#pragma once

#include <functional>

#include "experiment/record.h"

namespace dolittle::experiment {
struct LoadedModel {
  tensors::formats::SourceIdentity source;
  std::unique_ptr<models::BoundCheckpoint> checkpoint;
  std::size_t
      host_working_bytes;  // Loader allowance, excludes mapped address space.
};
// The provider must enforce the supplied active-loader host working allowance.
using ModelProvider =
    std::function<LoadedModel(std::size_t max_host_working_bytes)>;
struct RunInputs {
  std::unique_ptr<nla::Pair> pair;
  ModelProvider target;
  ModelProvider verbalizer;
  tensors::ComputationMode target_mode{};
  tensors::ComputationMode verbalizer_mode{};
  std::unique_ptr<tokenizers::Tokenizer> target_tokenizer;
  std::unique_ptr<tokenizers::Tokenizer> verbalizer_tokenizer;
  std::unique_ptr<text::Protocol> target_protocol;
  inference::sampling::TokenSelector selector;
  std::string backend_identity;
  std::size_t metadata_host_bytes{};  // Admitted owned tokenizers, pair and
                                      // provider metadata.
};
struct RunLimits {
  std::size_t max_device_bytes;
  std::size_t max_host_bytes;
  std::size_t max_prefill_chunk;
  std::size_t
      max_target_tokens;  // Prompt + generated, not a truncation policy.
  std::size_t max_verbalizer_tokens;
  std::size_t max_decoded_bytes;
  RecordLimits record;
};
enum class Phase : std::uint8_t { empty, retained, verbalizer_ready, closed };
class Run {
 public:
  Run(gpu::Device&, RunId, RunInputs, RunLimits);
  ~Run();
  Run(const Run&) = delete;
  Run& operator=(const Run&) = delete;
  Run(Run&&) = delete;
  Run& operator=(Run&&) = delete;
  void generate(const text::Prompt&, inference::GenerationSettings,
                const std::stop_token& = {});
  [[nodiscard]] VariantId interpret(CaptureKey, inference::GenerationSettings,
                                    const std::stop_token& = {});
  [[nodiscard]] Phase phase() const noexcept;
  [[nodiscard]] const Record* record()
      const noexcept;  // Null until target result retained.
  [[nodiscard]] const measurement::PhaseMeasurements& measurements()
      const noexcept;
  void close();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace dolittle::experiment
