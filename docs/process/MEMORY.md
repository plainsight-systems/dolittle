# Project memory

## Identity

- **Project:** Dolittle, named after Doctor Dolittle.
- **Parent entity:** Plainsight Systems LLC; no separate operating brand.
- **Status:** local repository scaffold and original banner artwork; no engine
  or frontend implementation yet.
- **Remote:** not created or configured.

## Decisions made 2026-10-16

- Demonstrate understanding through a direct implementation of inference and
  natural language autoencoders, with an approachable interface. Novel research
  is not the project's objective.
- Show target-model thinking text beside NLA interpretations aligned with
  capture positions in the same generation. Interpretations are imperfect;
  they are separate from the target's generated reasoning.
- C++20 and CUDA on Linux, built with CMake. Docker will provide the pinned
  build and runtime environments. Exact versions remain to be selected.
- Develop and verify on an RTX 3090 with 24 GB VRAM. Other hardware claims
  distinguish tested configurations from expected compatibility.
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
  public project site. Neither frontend is implemented yet.
- Defer native Metal and browser-only inference. The audience can reasonably
  be expected to have capable GPU hardware.
- Examine Qwen3 4B and its community NLA adapters first. Model choice is not
  validated; check reasoning-stream interpretations and the actual footprint.
  Shared base-weight residency across adapters is an option to evaluate.
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
  workflow guidance local. Exact versions, model selection, and architecture
  remain upcoming design work.

## Current work

See [QUEUE.md](QUEUE.md). Architecture starts at
[`../architecture/README.md`](../architecture/README.md). Investigations are
indexed in [`../research/README.md`](../research/README.md).
