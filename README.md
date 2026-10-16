<p align="center">
  <img src="docs/assets/dolittle-readme.png" alt="An engraved illustration of a green parrot speaking into a brass listening horn, with glowing marks becoming handwriting above an open notebook in a lamplit study." width="100%" />
</p>

# Dolittle

Dolittle is an inference engine being built to show a language model's
generated reasoning beside natural language interpretations of its internal
activations. One prompt, one generation, two views of the model working
through it.

The engine will run on Linux with CUDA, built from source and run through
Docker. A browser interface will let you follow the thinking text and inspect
activations captured at selected positions. The first hardware target is an
RTX 3090 with 24 GB of VRAM. Qwen3 4B and its community natural language
autoencoder adapters are the first model pairing to evaluate.

**Status: repository setup.** The project structure and artwork are in place.
The engine, application interface, website, and Docker build are not yet
implemented. There is no launch command yet.

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
| `docs/writeups/` | Milestone articles |
| `docs/assets/` | Project artwork and its provenance |

Downloaded weights will live in ignored `models/`; build output and local
experiments also stay out of Git.

## Working on Dolittle

Start with [project memory](docs/process/MEMORY.md), the
[work queue](docs/process/QUEUE.md), and the
[local workflow](docs/process/workflow.md). Agent entry points are
[`AGENTS.md`](AGENTS.md) and [`CLAUDE.md`](CLAUDE.md).

The first engineering task is to establish the model pairing, numerical
requirements, and memory budget on the 3090, then document the architecture.
Newer NVIDIA cards are intended to have a PTX fallback path, subject to driver
and memory requirements; hardware support and performance will be reported
against actual verification.

## Background

- [Anthropic's natural language autoencoder article](https://www.anthropic.com/research/natural-language-autoencoders)
- [Natural Language Autoencoders paper](https://transformer-circuits.pub/2026/nla/index.html)
- [Original NLA implementation and checkpoints](https://github.com/kitft/natural_language_autoencoders)
- [Community Qwen3 4B NLA adapters](https://huggingface.co/collections/anicka/nla-qwen3-4b-universal)

## License

Code and configuration use Apache-2.0; documentation and original artwork use
CC BY 4.0. See [LICENSE](LICENSE) for scope and [NOTICE](NOTICE) for attribution.
Third-party weights and components retain their own licenses.
