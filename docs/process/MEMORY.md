# Project memory

## Identity

- **Project:** Dolittle, named after Doctor Dolittle.
- **Parent entity:** Plainsight Systems LLC; no separate operating brand.
- **Status:** local project site with explanatory writing, original artwork,
  and reference GPU feasibility checks. No engine or live application yet.
- **Remote:** `origin` is https://github.com/plainsight-systems/dolittle.git.
  The site’s Code link points to this repository. Andrew authorized the first
  push and Pages deployment on 2026-10-16.
- **Production site:** https://dolittle.plainsight-systems.com/ (deployed; HTTPS certificate pending at first verification).

## Decisions made 2026-10-16

- Motivation, clarified by Andrew: he read Anthropic’s NLA paper and thought
  it would be a fun experiment to build. He already understood thinking tokens;
  discovering them in Charlotte did not motivate Dolittle. Keep his broader
  position on whether those tokens matter separate from the project’s origin.

- Demonstrate understanding through a direct implementation of inference and
  natural language autoencoders, with an approachable interface. Novel research
  is not the project's objective.
- Show target-model thinking text beside NLA interpretations aligned with
  capture positions in the same generation. Interpretations are imperfect;
  they are separate from the target's generated reasoning.
- C++20 and CUDA on Linux, built with CMake. Docker will provide the pinned
  build and runtime environments. Exact versions remain to be selected.
- Develop and verify on a 24 GB Ampere GPU. The actual machine reports an
  RTX 3090 Ti (verified 2026-10-16), correcting the initial 3090 shorthand.
  Other hardware claims distinguish tested configurations from expected
  compatibility.
- Retain baseline PTX for every kernel alongside native 3090 code. Exercise
  `CUDA_FORCE_PTX_JIT=1`. Newer NVIDIA GPUs, including RTX 40- and 50-series,
  are intended to use the compatible code path, subject to driver and memory
  requirements. Architecture-specific optimization must keep that baseline.
- Distribute source builds, not prebuilt application executables. Published
  model weights remain inputs. A single entry point will build, fetch/cache
  weights, and launch the application.
- Use Python for the setup/launch entry point, following Cradle. Pin the build
  environment and dependencies; the host supplies Docker, the NVIDIA driver,
  and container GPU access.
- Keep model downloads separate from image builds. Track model revisions,
  required files, hashes, and provenance in a manifest; keep weights in a
  persistent ignored cache.
- Give ordinary build/non-GPU checks and 3090 execution checks stable,
  separate commands. Hosted build success does not establish GPU correctness.
- Serve `app/` through the engine deployment. A browser on another machine,
  including a Mac, can access the Linux deployment. `web/` is the separate
  public project site, now implemented as a local static build. The live
  inference interface remains unimplemented.
- Prioritize the project site before engine architecture. Follow Cradle’s
  explanatory format and family styling, with explanations, explorations, and
  Serenity-style milestones sharing a Writing index. Milestones support the
  writing rather than defining its sequence. Use Hugo 0.154.2, self-hosted Plex
  fonts, explicit article publication, and a separate loopback preview command.
- Platform plan updated by Andrew: Linux/NVIDIA first, Metal on macOS afterward.
  AMD support is undecided. Metal build/runtime details and hardware requirements
  remain future design work. Browser-only inference stays out of scope.
  The audience can reasonably be expected to have capable GPU hardware.
- Andrew selected **Qwen3 8B with marco-bazzani/Qwen3-8B-nla** after the
  comparison. Target revision: `b968826d9c46dd6066d109eabc6255188de91218`;
  NLA revision: `12c6ed1b1e684bdb8a788c989e2f3d15f9790db3`. File hashes are
  pinned in [models-8b.json](../../scripts/validation/models-8b.json).
  Capture the output of zero-based block 24. The full verbalizer and
  28-block reconstructor with a separate value head have distinct weights.
  Sequential BF16 reference execution fits the 3090 Ti; design scheduling
  around these actual weights and measured costs. The earlier shared-base
  4B experiment remains comparison evidence. Semantic faithfulness is
  unverified; agreement with the reasoning trace is not a gate.
- Follow Serenity's modified engineering workflow, written locally in
  `workflow.md`. No imported governance files, submodule, or symlinks.
- Architecture comes first, then file-header contracts and design. No packets.
  Design optimizations in, derive data-path costs first, verify on the GPU,
  and use independent review on request.
- Preserve the source brief locally as ignored `brief.md`, following Serenity.
- Code/configuration: Apache-2.0. Prose/original artwork: CC BY 4.0.
- Create project art before the first commit. The initial banner is
  `docs/assets/dolittle-readme.png`; its prompt is recorded alongside it.
- Repository setup recommendations accepted, with `web/` explicit and all
  workflow guidance local. Toolchain versions and architecture remain upcoming
  design work; the 8B model pairing is selected.

## Validation findings, 2026-10-16

- The actual host is an RTX 3090 Ti (24,564 MiB, sm86), driver 595.91.07.
- One BF16 Qwen3 4B base with both FP32 adapters holds 7.739 GiB of parameters.
  This is reference evidence for the layout, not a custom-engine measurement.
- The initial interpretation verdict was too strong. Descriptions contain
  transcript errors and unverified claims about internal representations.
  Disagreement alone establishes neither NLA failure nor hidden computation.
  Neither the reasoning trace nor NLA output is ground truth for the other.
- Andrew confirmed the framing: Dolittle is an application demo that makes
  both views accessible, including their agreement, divergence, and uncertainty.
  Validate computation and alignment; do not require matching narratives or
  turn the project into a novel interpretability research effort.
- The reference uses PyTorch/Transformers/PEFT only for verification. The direct
  C++/CUDA implementation decision is unchanged.
- Docker is installed but NVIDIA Container Toolkit/GPU access is missing.
  Host administrator setup is required before container execution checks.
- Source revisions, manifests, full outputs, costs, and caveats live in the
  [feasibility note](../research/2026-10-16-nla-validation/README.md).

## Current work

The requested [4B/8B comparison](../research/2026-10-16-nla-comparison/README.md)
is complete. Qwen3 8B plus `marco-bazzani/Qwen3-8B-nla` runs sequentially in
BF16 on the 3090 Ti: peak allocated memory 15.378 GiB; median AV time 4.73 s
versus 10.28 s in the 4B reference run. All eight thinking descriptions
completed within the cap, but they still substituted unsupported transcript
details. Neither pair has established semantic faithfulness. The 8B pair
uses about 45 GB of downloads, separate weights and one capture layer (24);
4B remains smaller and supports multiple depths. Andrew selected 8B after
reviewing these tradeoffs. Andrew then prioritized standing up the explanatory
project site. No product inference code exists. Raw reconstruction scores across models are not a common accuracy
scale.

The project site is built and running locally: Home, Writing, About, two
opening articles, RSS, and 404. Its architecture and publication rules are in
[project-site.md](../architecture/project-site.md) and
[web/README.md](../../web/README.md). Five publication tests pass, including a
real subpath build, draft/future exclusion, milestone rendering, and rejection
of broken assets/anchors, stale share metadata, missing share images, or
unintended documents. The GitHub Pages workflow now validates PRs and deploys
main with a pinned Hugo installer. The first push (`47f560a`) and deployment succeeded on 2026-10-16:
https://github.com/plainsight-systems/dolittle/actions/runs/38967736528.
The custom domain returned HTTP 200 with the correct GitHub menu link and
production share metadata. HTTPS certificate provisioning was still pending;
the default github.io project URL redirects to the custom domain.

See [QUEUE.md](QUEUE.md). Architecture starts at
[`../architecture/README.md`](../architecture/README.md). Investigations are
indexed in [`../research/README.md`](../research/README.md).
