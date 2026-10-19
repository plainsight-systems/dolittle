<p align="center">
  <img src="docs/assets/dolittle-readme.png" alt="An engraved illustration of a green parrot speaking into a brass listening horn, with glowing marks becoming handwriting above an open notebook in a lamplit study." width="100%" />
</p>

# Dolittle

Dolittle is an inference engine being built to show a language model's
generated reasoning beside natural language interpretations of its internal
activations. One prompt, one generation, two views of the model working
through it.

The first engine targets Linux with CUDA and source builds in a pinned Docker
environment. Metal on macOS is planned next; AMD support is undecided. A future
browser interface will let you follow the thinking text and inspect activations
captured at selected positions. The first hardware target is a
24 GB Ampere GPU, with reference measurements on an RTX 3090 Ti. The selected
pairing is **Qwen3 8B with marco-bazzani's natural language autoencoder**,
interpreting activations at block 24 (zero-based).

**Status: target generation is implemented; full-model generation verification
is still in progress.** The C++20/CUDA engine covers verified checkpoint loading,
byte-BPE tokenization, Qwen3 chat formatting, cached prefill/decode, sampling,
stopping and configurable captures. It retains the verified no-cache forward
path. The NLA paths and live browser application remain later slices.

CUDA 13.2 builds retain native Ampere code and PTX. Use the
[build and verification commands](tests/README.md) to check your environment.
Container GPU execution requires the NVIDIA Container Toolkit. The current
numerical scenarios cover short sequences; they do not establish full-context
behavior or performance on other NVIDIA hardware.

```sh
python3 scripts/check.py host
python3 scripts/check.py archive
# On Linux: build CUDA sources in the pinned Docker environment.
python3 scripts/check.py cuda-build
```

See [build and verification commands](tests/README.md) and
[VS Code setup](tests/README.md#vs-code-and-compiler-settings). The historical
[4B/8B reference comparison](docs/research/2026-10-16-nla-comparison/README.md)
used PyTorch, with 15.4 GiB peak allocation for separate BF16 phases and about
45 GB of checkpoint downloads. Those results describe feasibility of the
selected model pair; they are separate from verification of this implementation.

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
interpretations that can be wrong. Dolittle will distinguish those
interpretations from the reasoning text the target model generates itself.

The purpose is to understand the machinery by implementing it and making it
easy to interact with. Dolittle is R&D under Plainsight Systems LLC.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/` | C++20 and CUDA inference engine |
| `app/` | Browser interface served with the engine |
| `web/` | Public project website |
| `tests/` | Numerical, GPU, and application verification |
| `scripts/` | Source-build, setup, launch, and check tooling |
| `docs/architecture/` | Logical design, change axes, and file mapping |
| `docs/process/` | Project memory, work queue, and local workflow |
| `docs/research/` | Source investigations and measurements |
| `docs/writeups/` | Explanations, explorations, and milestone articles |
| `docs/assets/` | Project artwork and its provenance |

Downloaded weights will live in ignored `models/`; build output and local
experiments also stay out of Git.

## Working on Dolittle

Start with [project memory](docs/process/MEMORY.md), the
[work queue](docs/process/QUEUE.md), and the
[local workflow](docs/process/workflow.md). Agent entry points are
[`AGENTS.md`](AGENTS.md) and [`CLAUDE.md`](CLAUDE.md).

The reference checkpoint paths run on the 3090 Ti. Dolittle will let visitors
explore agreement and disagreement between the two views. Neither view is
ground truth for the other; interpretation faithfulness remains unverified.
Newer NVIDIA cards are intended to have a PTX fallback path, subject to driver
and memory requirements; hardware support and performance will be reported
against actual verification.

## Background

- [Anthropic's natural language autoencoder article](https://www.anthropic.com/research/natural-language-autoencoders)
- [Natural Language Autoencoders paper](https://transformer-circuits.pub/2026/nla/index.html)
- [Original NLA implementation and checkpoints](https://github.com/kitft/natural_language_autoencoders)
- [Selected Qwen3 8B NLA pair](https://huggingface.co/marco-bazzani/Qwen3-8B-nla)

## License

Code and configuration use Apache-2.0; documentation and original artwork use
CC BY 4.0. See [LICENSE](LICENSE) for scope and [NOTICE](NOTICE) for attribution.
Third-party weights and components retain their own licenses.
