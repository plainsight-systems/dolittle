# Live experiment application

The application connects the [experiment](logical-overview.md) to a source-built
service and browser interface. Linux/CUDA and native macOS/Metal run generation,
interpretation, reconstruction and steering through the same application.
[Setup](../../app/README.md) lists current platform prerequisites.

## Outcome and scope

From one command, build the service from source, acquire and verify the pinned
target/AV/AR artifacts, and open a browser experiment. Enter a prompt, watch target
output, stop gracefully if desired, then select consumed tokens and compare
separate AV variants. Keep the reasoning, answer and interpretations distinct.
The browser can use a local GPU service or connect to one on another machine.

One service owns one GPU and one retained experiment. Starting another experiment
requires an explicit replacement action while no work is active; old links become
stale. A browser refresh reconnects to the same in-memory experiment. Server
restart loses it. Edited explanations feed the reconstructor and matched target
continuations through the [steering flow](steering.md). Disk session export and
multi-user scheduling remain unimplemented. The public `web/` site is independent.

## Algorithm and ownership

```mermaid
flowchart TD
    A[Source launcher: check host, build, verify artifacts] --> B[Compose device and selected model providers]
    C[Browser commands] --> D[HTTP: bounded parsing and typed requests]
    B --> E[Session: one joined worker owns Run]
    D --> E
    E --> F[GPU target generation and device-owned captures]
    F --> G[Publish exact token deltas and consumed counts]
    F --> H[Finalize final token; retain Record and captures; release sequence storage]
    H --> I[Process selected positions sequentially with resident AV]
    I --> J[Retain each interpretation variant]
    J --> N[AR reconstructs original and edited descriptions on GPU]
    N --> O[Matched baseline and steered target continuations]
    O --> K
    G --> K[Bounded immutable presentation journal]
    H --> K
    J --> K
    K --> L[Cursor pages over HTTP]
    L --> M[Browser: reasoning, answer, position selection and variants]
```

1. `scripts/run.py` selects Docker/CUDA on Linux or the native Metal CMake build
   on Apple Silicon. It checks platform prerequisites and builds from source.
   `scripts/models.py` prepares the pinned cache and reuses its verification
   receipt on subsequent launches.
2. Composition supplies verified target/AV/AR providers, separate tokenizer/protocol
   identities, pair, sampler and explicit memory/context budgets. A persistent
   worker creates and owns the device and every `Run`. HTTP threads never call a
   `Run`, read its `Record`, or release GPU objects.
3. Admission reserves a bounded command slot and request identity. Acceptance
   means queued, not that prompt/model admission has succeeded. The worker runs
   ordinary `Run::generate`; preflight failures become explicit request outcomes.
4. Generation publishes newly appended IDs and completed consumed length after
   existing completion boundaries. An emitted token can precede its own capture.
   The observer adds no kernel, readback, RNG draw or synchronization. It runs
   on the generation thread, borrows only call-scoped data, and never does I/O.
5. Run decodes just the new IDs and advances a protocol-owned segmentation
   cursor. It reports model role and operation identity to the session. The
   session appends owned, immutable presentation events. Raw capture vectors stay
   in `Record`; neither the journal nor browser receives them.
6. After generation returns, finalize Record and release sequence storage.
   Keep idle weights when the next complete request fits, otherwise evict them.
   Publish `target_retained` and make the device-owned captures interpretable.
   A partial failure publishes the actual retained prefix and explicit absent
   captures. A failure with no materialized Record cannot offer interpretation.
7. An accepted bounded list of `(capture, settings, seed)` jobs runs sequentially.
   Requests made during target generation wait. Resolve each capture against the
   retained Record, never a browser character offset. An invalid/unavailable row
   yields a rejected job, not a substituted neighbor. AV weights stay resident;
   each attempt receives a fresh KV cache. One failed job does not silently erase
   the remaining jobs; stop the batch on a fatal device/closed-run condition.
8. Publish each attempt's exact output, tag status, outcome, settings and variant
   ID. Request/job IDs exist before VariantId does: preflight rejection may have
   no variant. Failed or canceled accepted attempts keep their recorded status.
9. HTTP readers borrow immutable event owners under a short lock, then serialize
   and send outside that lock. Slow clients do not hold the GPU worker. Shutdown
   stops admission, signals cooperative stop, drains/joins the worker, releases
   Run and device on their owner thread, then destroys publication storage.

`Session` adds concurrency around `Run`; it does not change Run into a concurrent
object. A single persistent worker avoids thread creation per token/request.
Cancellation uses a per-active-operation stop source. Copy that source under the
session mutex and request stop after unlocking; never run stop callbacks while
holding the mutex. No detached workers or callbacks that retain borrowed spans.

## Progress, segmentation and exact positions

Extend `inference/generation.h` with a borrowed observer accepting appended token
IDs, their first absolute position, prompt length and completed consumed length.
Report initial prompt IDs once, emitted IDs once, and subsequent consumed-count
advances without resending the whole prefix. A missing observer preserves current
behavior. Observer failure requests graceful cancellation and marks publication
failure; it does not masquerade as CUDA arithmetic failure or discard the Record.
Preallocate publication metadata when possible; reserve a terminal error slot.

Add a protocol-owned segmentation cursor to `text/protocol.h`. The Qwen3 kind
uses the same control-ID transitions as `segment`, carrying its state between
chunks. Chunked classification must concatenate to the existing whole-sequence
result, including malformed/nested delimiters. Batch `segment` and the cursor
share a transition rule. No thinking IDs move into HTTP or JavaScript.

A presented token carries its exact ID, absolute position, raw byte contribution,
mapped/unmapped status and segment kind. JSON uses base64 for raw token bytes,
which need not be valid UTF-8. Control and prompt tokens remain inspectable even
when hidden in the default transcript. Positions, IDs and capture keys remain
authoritative; text is a view. Browser decoding buffers split UTF-8 characters
and records every contributing token. A multi-token character selects an explicit
position range; the inspector lets the user choose an individual token. A
terminal incomplete sequence displays replacement text without changing bytes.

`consumed_tokens` is a progress watermark, not permission to interpret during
target execution. Availability is established from the retained Record when the
phase finishes. A selected window expands into independent jobs per position;
it never averages vectors or silently assigns one character to one token.

## Transport contract

Use same-origin HTTP JSON with cursor polling. While work is active, the browser
requests updates at most ten times per second, never overlapping requests. Idle
polling backs off; a disconnection uses bounded retry. This introduces up to one
poll interval of presentation delay. It is a tunable starting value, not a
throughput or latency claim. SSE/WebSocket remain possible A1 changes.

Use [cpp-httplib](https://github.com/yhirose/cpp-httplib) for HTTP framing,
connection handling and a fixed worker pool. Its documented blocking I/O makes
short bounded requests a suitable initial use. Version 0.62.0 and its source archive SHA256 are pinned in CMake; its MIT
license is retained in `docs/licenses/`. It supplies transport
only. Use the existing JSON dependency for strict wire encoding, with duplicate
keys, unknown fields, invalid enums, non-finite values and oversized inputs
rejected. No new inference framework or frontend build framework is required.

| Endpoint | Contract |
| --- | --- |
| `GET /api/v1/status` | Server epoch, current run, phase, supported pair/backend, configured limits, queued/active request IDs |
| `POST /api/v1/runs` | Client request ID, prompt, target settings and expected prior RunId for explicit replacement; 202 queued or 409 busy/stale |
| `POST /api/v1/runs/{run}/interpretations` | Request ID and bounded ordered list of captures/settings/seeds; 202 queued, with per-job eventual outcome |
| `POST /api/v1/runs/{run}/stop` | Request ID plus active operation/request identity; signals only that operation and cancels its remaining jobs; never a later operation |
| `GET /api/v1/runs/{run}/events?after=…` | Bounded ordered immutable events, next cursor and whether more are available; no long-held request |
| `GET /` and fixed asset paths | Packaged `app/` resources only, with correct MIME types; no model/cache directory serving |

Represent 64-bit IDs, event cursors and seeds as decimal strings on the wire;
JavaScript must not round them through Number. Include a random server epoch so
a restart cannot reuse an old browser request/cursor accidentally. Capture keys
include run, semantic site and absolute consumed-token position. The pair supplies
the site's block and boundary; the server never hardcodes 24.

Mutations carry stable client-generated request IDs. Under one admission lock,
check an existing receipt before stale-run validation, then validate and reserve
command plus receipt capacity before acknowledgement. The
same ID and canonical typed payload returns the original receipt; a changed
payload returns conflict. Retain receipts for the current run, including failed
and canceled jobs. Bound the ledger and reject further commands when full;
never evict an ID that could be replayed while its run is current. Retrying a
lost POST response cannot generate twice or create another variant. On explicit
replacement, retain the replacement receipt for the current run; stale requests
carrying the old expected RunId fail rather than replacing again.

An event page contains the current phase/status watermark and events after the
cursor, bounded by both count and encoded bytes. Keep the admitted journal for
the lifetime of the current run: reconnect starts from its last applied cursor,
and refresh can replay from zero. Deduplicate by sequence; reject gaps, future
cursors and epoch mismatches. Never advance a cursor past an event not included
in that page. Run replacement invalidates the old journal with an explicit stale
response; an in-flight response remains valid only for its labeled run/epoch.

Use 400 for malformed input, 404 for unknown resource, 409 for stale/busy/conflict,
413 for request-size limits, 429 for bounded capacity, and 503 for unavailable
service/device. Numerical/preflight errors after acceptance are request events,
not successful descriptions. Errors exposed to clients omit local filesystem
paths. Model bytes/metadata are acquired only through the trusted launcher.

Default published port binds host loopback. Docker may listen on all container
interfaces while Compose publishes only `127.0.0.1`. Explicit LAN binding is an
operator choice. Accept configured Host addresses and the exact loopback names
`localhost`, `127.0.0.1` and `[::1]`, with an optional port from 1 through 65535.
Loopback ports may differ from the service port for SSH forwarding. A supplied
Origin must match a configured origin or the request's loopback HTTP address.
Reject duplicate Host/Origin headers. API requests still require the startup
access token; no wildcard CORS. The pure address predicates live with wire
validation in `server/wire.cpp` and are declared in `server/http.h`.
Hand the token to the local browser in a URL fragment,
remove it after reading and retain it in origin-scoped local storage so the plain
URL works across tabs and browser restarts. Migrate existing session credentials;
fall back to memory if storage is blocked. Discard rejected tokens and show a
connection panel instead of repeatedly sending unauthorized requests. The panel
accepts an HTTP(S) launcher link from another hostname or port and extracts only
its token fragment. It neither navigates to nor fetches the pasted address;
all requests continue to the current page's origin. This lets a forwarded page
use the remote launcher's link directly. A new launcher session requires its new link. Keep tokens
out of application logs and query strings; the launcher's private access link is
the explicit handoff. Serve generated text as text nodes, never HTML. This is a local
experiment service, not an internet-facing multi-user deployment.

## Bounds and cost

Let N be admitted target tokens, A the cap per AV output, V the maximum variants,
B the admitted decoded bytes across target and all attempts, E the event count,
W HTTP worker count and R the maximum response bytes.

- Existing retained capture payload is `8192*N` bytes for this BF16 pair, owned
  once by Record. Publication adds no second vector copy or device readback.
- Presentation payload is `O(N + V*A + B + E)`; base64 wire bytes cost
  `4*ceil(raw_bytes/3)` plus bounded JSON metadata. Account for this separately
  from Record and tokenizer/provider allowances, including actual capacities.
- HTTP response working space is bounded by `W*R` plus bounded request buffers,
  serialization scratch and pinned event references. Explicit replacement cannot
  accumulate old journals: outstanding pages are bounded by W and page size.
- Command queue, total jobs, receipts, event count, event bytes and per-token bytes
  have explicit limits. Admission uses checked sums/products and reserves terminal
  outcome capacity. If progress exceeds its admitted budget, request graceful stop
  and expose publication failure; never silently truncate a successful result.
- Append/decode each delta once. No full-history decode, classification or JSON
  serialization on every token. Poll pages serialize only the requested events.
  Terminal metadata links prior output by range instead of duplicating all text.
- HTTP is off the generation thread. Small publication locks transfer owned
  event pointers; serialization, callbacks, filesystem and device calls stay
  outside them. Measure callback, lock and presentation overhead independently
  of clean inference timing. Do not introduce a lock-free queue without evidence.

The target context ceiling is 8,192 tokens, with a browser output default of
4,096 shared by thinking and answer. Admission checks prompt plus requested
output, including the chat template. AV keeps an independent 512-token context
and 256-token output default. Per-run allocation uses admitted capacity, not the
global ceiling: at 8,192 target tokens, KV is 1.125 GiB and raw retained captures
are 64 MiB, in addition to weights, scratch and metadata. A larger budget does
not force the model to generate to the cap. The frozen numerical cases remain
short; runtime capacity measurements do not establish full-context numerical
agreement with an independent oracle.

## Browser responsibilities

The accepted [investigation interface](investigation-interface.md) separates
investigated-point navigation, the Prompt/Thinking/Answer reader, and samples of
the selected activation. Comparison opens two unedited samples with their own
source identity. Streaming updates preserve focus, sample choice and scroll.

`client.js` owns same-origin transport, remembered access and idempotent command
retries. `controller.js` reduces ordered events and projects exact byte/token
ranges, search matches and sample identities. `view.js` owns accessible DOM
presentation, navigation and optional submission metadata. Model-specific
thinking classification remains in the server protocol. Missing original-prompt
provenance falls back to labelled full input; no delimiter stripping is used.

Every provisional sample is keyed by epoch/run/request/job and its submitted
capture, never by current selection. Terminal journal events supply authoritative
outcomes. Work from another browser has global progress until its capture is
published. Refresh replays the journal without rerunning inference. Selection is
allowed while the target streams; interpretation is enabled after capture
finalization. Stop is not pause/resume. A new experiment explicitly replaces
retained state, and changing epoch/run clears old selection and comparison.

## Files and implementation boundary

| Owner | Responsibility |
| --- | --- |
| `experiment/session.{h,cpp}` | Persistent worker, admission, operation-scoped stop, receipts, bounded immutable event journal |
| `experiment/run.{h,cpp}` | Borrowed progress adapter, role/operation reporting, release-complete phase publication |
| `inference/generation.{h,cpp}` | Optional borrowed token/consumption observer at existing completion boundaries |
| `text/protocol.h`, `text/qwen3_chat.*` | Incremental segmentation cursor sharing the existing batch transition rule |
| `server/http.{h,cpp}`, `server/wire.cpp` | HTTP lifetime, strict JSON translation, IDs, authentication/origin checks, fixed assets and event pages |
| `main.cpp` | Parse local configuration, compose CUDA or Metal, model providers, session and server, orderly shutdown |
| `config/models.json`, `scripts/models.py` | Production target/AV/AR role manifest, bounded acquisition, integrity and persistent revision-aware cache |
| `scripts/run.py`, `compose.yaml`, `Dockerfile`, `CMakeLists.txt` | One source-build launch command; runtime image/asset packaging; native macOS path and optional source-built binary |
| `app/{index.html,app.css,client.js,controller.js,view.js}` | Live experiment interaction described above |

`session` belongs under experiment because scheduling/cancellation outlive HTTP
requests. `http` owns wire DTO encoding rather than leaking JSON into inference.
No per-event folder, generic messaging framework or alternative backend stub is
introduced. The existing change axes remain intact.

Production manifest entries pin target, AV and AR repository revisions,
file sizes and hashes. Setup downloads the weights and metadata needed by all
three roles, plus licenses. The reference manifest remains independent. Cache paths
include revision, reject traversal/symlinks outside the cache, stream to partial
files and atomically publish verified files. Do not execute model repository code.
Setup downloads missing files and verifies every artifact against pinned SHA-256
before atomically publishing a versioned receipt in the persistent cache. The
receipt binds the manifest to each file's size, modification/change times and
filesystem identity. A normal launch checks that small receipt and current file
metadata. Matching files reuse the saved verification result. Changed/missing files or a changed manifest trigger
setup again; offline setup fails if any artifact fails its hash. Explicit
`--verify-models` forces full verification and repairs files when online.
`scripts/models.py` also always performs full verification/repair. Failed setup
leaves no valid receipt. Source rebuilds alone do not invalidate model setup.

Before HTTP readiness the service maps the target, AV and AR archives and
validates sizes, bounded headers, indexes and extents. It retains the read-only
mappings and shares ownership with model bindings across turns. One role runs
at a time; idle weights remain on the GPU when there is room for the next model,
its cache and working buffers. The OS may reclaim mapped pages. Backing files
must stay unchanged until
shutdown; changing model artifacts requires restarting. Direct service/archive
calls default to full SHA-256 verification; prepared mode is an explicit trusted
launcher precondition. Canonical model identities and numerical behavior remain
unchanged. Normal launch and model switching do no full weight hashing.

Linux Docker and native macOS CMake build locally from source. Docker mounts
weights read-only at runtime, after the launcher prepares them. Running the
container with GPU access requires NVIDIA Container Toolkit on the host.

## Verification and review boundary

Both corpora were queried for header design and again for implementation.
Implementation uses CP.22, CP.23, CP.41, CP.43 and CP.50 from cpp-guidelines,
and COPY.7/GPU.1 from the performance corpus. File headers describe their
application and the concrete algorithm.

Unit checks cover strict wire parsing, response bounds, segmentation chunk
equivalence, exact token-byte mapping and browser reducer invariants. They start
no threads, servers or GPU. Admission/retry and journal reconnect are exercised
through the real application integration; no claim of isolated session-policy
unit coverage is made. Do not test incidental chunk sizes or private queue layout.

The integration inventory is **five**: site, device resources, tiny numerical
target-to-AV, pinned 8B numerical target-to-AV, and the live application.
The application scenario replaces the standalone archive scenario. Actual startup
covers successful real-file verification/binding; adversarial filesystem cases
from the retired scenario are not claimed as preserved. Pure safetensors/parser
contracts remain in isolated units. Browser inspection uses the same application
flow, rather than adding another integration scenario.

The application scenario follows one real prompt through streaming, final capture,
selection, interpretation and reconnect using the production service. Numerical
cases retain the independent frozen target/AV fixtures, limits and original oracle.
Native/PTX execution and memory checks remain configurations of those cases.
Builds, formatting and clang-tidy remain separate checks. Verify Docker GPU access,
shutdown ownership and host/thread sanitizers as appropriate; report actual
hardware/workload evidence. Header syntax checking alone establishes none of this.
