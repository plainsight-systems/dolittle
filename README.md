<p align="center">
  <img src="docs/assets/dolittle-readme.png" alt="An engraved illustration of a green parrot speaking into a brass listening horn, with glowing marks becoming handwriting above an open notebook in a lamplit study." width="100%" />
</p>

# Dolittle

Dolittle puts Qwen3’s generated thinking beside natural language explanations
of its internal activations. Enter a prompt, select a token and read what the
NLA describes. Edit that explanation, then compare the model’s continuation
with and without the change.

The C++20 engine runs **Qwen3 8B with marco-bazzani’s natural language
autoencoder**, capturing the output of block 24 (zero-based). Generation,
interpretation and steering run on CUDA and Metal through the same browser
interface. Each experiment keeps the original response, interpretation samples
and steering comparisons available to explore.

## Run Dolittle

Use Python 3.11+ and Git, then run the same command on either supported host:

```sh
git clone https://github.com/plainsight-systems/dolittle.git
cd dolittle
python3 scripts/run.py
```

| Platform | Status and launch path | Tested hardware |
| --- | --- | --- |
| Linux/NVIDIA | Available. Builds and runs in Docker with NVIDIA Container Toolkit. | RTX 3090 Ti, 24 GB VRAM |
| macOS/Apple Silicon | Available. Builds and runs natively with Metal. Requires macOS 26+, Xcode 26 with Metal toolchain, CMake 3.24+ and OpenSSL 3. | M3 Max, 128 GB unified memory |
| Windows/NVIDIA | Native build planned. | — |
| AMD | Undecided. | — |

The launcher builds from source, downloads about 45 GB of model files and checks
their hashes. Later launches reuse those files and the saved verification result.
Use `--verify-models` to check every file again and repair the cache.

Open the full **Open Dolittle** link printed by the launcher and keep that
terminal running. The public project website explains the experiment; the local
service runs it. A browser on another machine can connect to that service.
See [application setup](app/README.md) for prerequisites, cache options and remote
access, then follow the [rhyming walkthrough](https://dolittle.plainsight-systems.com/writing/running-an-investigation/).

The engine uses BF16 tensors and FP32 reductions. It keeps model weights on the
GPU between requests when there is room for the next model, its cache and working
buffers. Each generation gets a fresh KV cache. CUDA builds include native
Ampere code and PTX that a compatible NVIDIA driver can compile for newer cards.

## Development

Development checks:

```sh
python3 scripts/check.py host
# On Linux: build CUDA sources in the pinned Docker environment.
python3 scripts/check.py cuda-build
```

See [build and verification commands](tests/README.md) and
[VS Code setup](tests/README.md#vs-code-and-compiler-settings). The historical
[4B/8B reference comparison](docs/research/2026-10-16-nla-comparison/README.md)
used PyTorch to compare the model pairs before building this engine. The 8B
run used 15.4 GiB peak allocation with the models running in separate BF16 phases.

The project site can be previewed independently:

```sh
python3 scripts/site.py serve
```

Open <http://127.0.0.1:1315/>. Python 3.11+ and Hugo 0.154.2 are required; see
[site setup](web/README.md). The production URL is https://dolittle.plainsight-systems.com/. GitHub Pages
builds and deploys the site on pushes to `main`.

## Why Dolittle

Doctor Dolittle learned to communicate with animals. This project takes its
name from that idea: making an unfamiliar intelligence understandable.

A natural language autoencoder translates an internal activation vector into
text, then reconstructs a vector from that text. Its descriptions are
interpretations that can be wrong. Dolittle distinguishes those
interpretations from the reasoning text the target model generates itself.

The purpose is to understand the machinery by implementing it and making it
easy to interact with. Dolittle is R&D under Plainsight Systems LLC.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/` | C++20, CUDA and Metal inference engine |
| `app/` | Browser interface served with the engine |
| `web/` | Public project website |
| `tests/` | Numerical, GPU, and application verification |
| `scripts/` | Source-build, setup, launch, and check tooling |
| `docs/architecture/` | Logical design, change axes, and file mapping |
| `docs/process/` | Project memory, work queue, and local workflow |
| `docs/research/` | Source investigations and measurements |
| `docs/writeups/` | Explanations, explorations, and milestone articles |
| `docs/assets/` | Project artwork and its provenance |

Downloaded weights live in ignored `models/`; build output and local
experiments also stay out of Git.

## Working on Dolittle

Start with [project memory](docs/process/MEMORY.md), the
[work queue](docs/process/QUEUE.md), and the
[local workflow](docs/process/workflow.md). Agent entry points are
[`AGENTS.md`](AGENTS.md) and [`CLAUDE.md`](CLAUDE.md).

The [architecture overview](docs/architecture/README.md) explains the computation
and how the CUDA and Metal backends share it. [Verification](tests/README.md)
covers numerical comparisons, tests and build checks.

## Background

- [Anthropic's natural language autoencoder article](https://www.anthropic.com/research/natural-language-autoencoders)
- [Natural Language Autoencoders paper](https://transformer-circuits.pub/2026/nla/index.html)
- [Original NLA implementation and checkpoints](https://github.com/kitft/natural_language_autoencoders)
- [Selected Qwen3 8B NLA pair](https://huggingface.co/marco-bazzani/Qwen3-8B-nla)

## License

Code and configuration use Apache-2.0; documentation and original artwork use
CC BY 4.0. See [LICENSE](LICENSE) for scope and [NOTICE](NOTICE) for attribution.
Third-party weights and components retain their own licenses.
