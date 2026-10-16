# Qwen3 4B and 8B NLA comparison — 2026-10-16

**Decision after comparison:** Andrew selected Qwen3 8B with
`marco-bazzani/Qwen3-8B-nla`, using the pinned revisions tested below.

## Purpose

Compare two available pairs for Dolittle's native-thinking-versus-activation
demo before selecting the model and designing the engine. This is an
application feasibility comparison, not an attempt to prove either generated
thinking or NLA text is a faithful account of computation.

## Published contracts

| Property | 4B baseline | Selected 8B pair |
| --- | --- | --- |
| Target | Qwen/Qwen3-4B | Qwen/Qwen3-8B |
| Native thinking | Yes | Yes |
| NLA publisher | anicka / nla-at-home | marco-bazzani |
| Verbalizer | Rank-16 LoRA over a shared base | Full fine-tune |
| Reconstructor | LoRA; requested block's last-token residual | 28-block model; final RMS norm bypassed; separate 4096×4096 linear head |
| Capture depth | Depth-conditioned, multiple layers | Block 24 only, zero-based |
| Training distinction | AV refined against frozen AR, centered-cosine and specificity reward | AV and AR jointly trained with reconstruction reward |
| Corpus coverage | SFT descriptions of last-prompt-token activations from about 5.2k texts | About 500k activations, five positions per document, about 100k FineWeb documents |
| Published evaluation | Centered reconstruction results and limitations documented | No comparable held-out results in the inspected model card |

Sources: [4B AV](https://huggingface.co/anicka/nla-qwen3-4b-universal-av-grpo),
[4B SFT provenance](https://huggingface.co/anicka/nla-qwen3-4b-universal-av),
[4B AR](https://huggingface.co/anicka/nla-qwen3-4b-universal-ar),
[8B pair](https://huggingface.co/marco-bazzani/Qwen3-8B-nla).
The original paper's authors did not publish either of these pairs. Larger
training coverage and full fine-tuning motivate comparison; they do not
establish better interpretations. Neither inspected release establishes
fidelity on this generated-thinking workload.

## Reproducibility and comparison limits

The existing [4B experiment](../2026-10-16-nla-validation/README.md) is the
baseline. The new [8B manifest](../../../scripts/validation/models-8b.json)
pins target revision `b968826d9c46dd6066d109eabc6255188de91218` and NLA revision
`12c6ed1b1e684bdb8a788c989e2f3d15f9790db3`, including byte counts and SHA-256
hashes. The [comparison script](../../../scripts/validation/compare_8b.py)
uses the same reference dependency environment and four prompts. The 8B
injection and AR conventions follow
[nla-inference at 38b802a](https://github.com/kitft/nla-inference/blob/38b802a33d1d317f21b6825a9116f388c2141f86/nla_inference.py).

- Seeds 100–103; target temperature 0.6, top-p 0.95, top-k 20; 512-token cap.
- Two replay captures at 35% and 75% of each available thinking segment;
  8B block 24 versus baseline 4B blocks 18, 22, and 26.
- Four thinking-disabled prompt-end controls for 8B; the existing 4B controls
  use block 22. They are controls, not held-out training-distribution tests.
- Greedy AV generation capped at 160 tokens. Each pair uses its own documented
  template and reconstruction protocol, rather than forcing identical prompts.
- Target, AV, and AR run sequentially in BF16, without quantization. The 4B
  reference retains FP32 adapters. No Docker or custom-engine result is implied.

Each target generates different text; captures do not represent identical
reasoning events. Token caps can truncate reasoning or explanations. These are
four cases, not independent samples at every capture. Raw reconstruction
cosines occupy different activation spaces and are not a cross-model quality
score. Matched-versus-swapped descriptions are only within-model diagnostics.
No causal intervention establishes the meaning of either model's activations.

## Storage and scheduling

The 8B target and AV each contain 8,190,735,360 parameters, requiring
16,381,470,720 bytes (15.256 GiB) apiece in BF16. The AR checkpoint stores
12,049,659,904 tensor bytes plus its 33,554,432-byte head. Combined weights
are about 41.766 GiB, before caches and runtime overhead. The complete pinned
download, including metadata and tokenizers, is 44,894,068,535 bytes.

That combined size does not require a larger GPU: the reference experiment
releases one model before loading the next. The AR's unused language-model
head is also removed before GPU transfer. Whether switching fits the desired
interaction latency is a separate product decision. The 4B pair has already
demonstrated 7.739 GiB shared-base parameter residency on the 3090 Ti.

Both target configurations have 36 blocks, eight KV heads, and head dimension
128. Their BF16 KV caches therefore have the same derived cost: 147,456 bytes
per token, or 1.125 GiB at 8,192 tokens. That calculation excludes workspaces
and is not an 8K runtime measurement. The 8B model's larger weight footprint
does not double its KV-cache requirement.

## GPU comparison

Both pairs ran on the RTX 3090 Ti, driver 595.91.07, using Python 3.12.3,
PyTorch 2.7.1+cu128, Transformers 4.57.3, and the same pinned reference
environment. [Full 8B outputs](results-8b.json) include the manifest, package
versions, exact prefixes, descriptions, and timings.

| Measurement | 4B baseline | 8B comparison |
| --- | --- | --- |
| Peak PyTorch allocated memory | 7.915 GiB | 15.378 GiB |
| Peak PyTorch reserved memory | 8.055 GiB | 15.479 GiB |
| Target generation | 27.4–29.7 tokens/s | 29.9–30.4 tokens/s |
| Median AV time, thinking captures | 10.28 s | 4.73 s |
| AV time range, thinking captures | 5.83–10.42 s | 3.65–5.35 s |
| Median AV generated tokens | 160 | 136.5 |
| Median AV tokens/s | 15.54 | 29.37 |
| Thinking-capture AV outputs hitting 160-token cap | 20/24 | 0/8 |
| Finished target answers within 512-token cap | 1/4 | 0/4 |
| AR time per thinking description | 69–71 ms | 32–47 ms |
| Prefix/full replay relative L2 | 0.0092–0.0147 | 0.0072–0.0090 |

The 8B AV completed all twelve descriptions, including four controls, with
proper explanation tags. This is a practical output-completion advantage on
the sampled workload, not a semantic-accuracy result. AV lengths, templates,
and full-fine-tune versus LoRA execution differ, so the observed speed advantage
does not establish that larger models are intrinsically faster or predict
Dolittle's future C++/CUDA performance. The target cap prevented an answer
quality comparison; it did not prevent sampling generated thinking.

Sequential BF16 execution fits this GPU. Target/AV parameter residency was
15.256 GiB each; the AR used 11.222 GiB plus a 0.03125 GiB value head.
Recorded weight-loading times were 2.40 s for the target, 2.57 s for the AV,
and 15.57 s for the AR. These use different loading paths and recently read
files; they are not cold-start or repeated-switch benchmarks. The AV timings
in the table exclude loading. This run batches each phase; it does not test
interleaving explanations with live target decoding or retaining a paused KV
cache during a switch.

The AR checkpoint omits the language-model output head and final RMS norm.
Transformers warned that these were initialized; both were replaced by
identity modules before GPU execution, as required by the reference protocol.
No missing initialized parameter participates in the measured reconstruction.

## What the descriptions show

The 8B descriptions often identify the broad task and the local continuation
better than a generic topic label: the coat-pocket capture ending in “which”
is described as a relative clause about location; a poem-editing capture
ending in “that's not” is described as rejecting a phrase. They also supply
unsupported specific settings, names, quantities, and languages.

| Case | 4B examples from the existing run | 8B examples from this run |
| --- | --- | --- |
| Marble arithmetic | Lottery combinations and numbered-list formatting | Marble subtraction, but invents Anna, 18 marbles and removal of 2; another capture becomes Sally's apples |
| Key in coat pocket | A woman in a man's pocket; unrelated character dialogue | Recognizes location reasoning and “the key,” but substitutes a suitcase/backpack and a Spanish hotel |
| Rabbit poem | Sentence completion or unrelated storm imagery | Recognizes iterative line editing, but substitutes character names, other lines, and countries |
| Cedar syllogism | Broad logical structure or unrelated taxonomy | Recognizes categorical reasoning and example construction, but substitutes whales, red trees, and different premises |

Prompt-end controls also contain substitutions, including a bee/chocolate
story in the key case and a ship/distance problem in the logic case. The
8B pair therefore does not eliminate the issue seen in the 4B descriptions.
These observations concern what the output asserts relative to the visible
context. They do not prove that the model lacks the named internal
associations, and unexpected content is not proof of hidden computation.

Within the 8B run, matched descriptions had higher raw reconstruction cosine
than the fixed different-case description in 8/8 thinking captures and 4/4
prompt-end controls. Mean thinking scores were 0.881 matched and 0.601 swapped;
prompt-end scores were 0.726 and 0.634. This supports limited activation-specific
information in this sample. It does not validate the prose, remove the
layer-mean confound, or establish superiority over the 4B model's scores.

## Implication for Dolittle

The 8B pair is a viable BF16 option on the existing GPU, with faster and
consistently completed descriptions in this reference run and training closer
to the original joint-reconstruction recipe. It costs a much larger download,
separate weight loading, and a single supported capture layer. The 4B pair
offers lower memory/storage needs and multiple capture depths.

There is no established semantic-faithfulness winner. Andrew selected 8B
after reviewing the comparison. The decision adopts the tested pair and its
block 24 capture contract; it does not establish that doubling the target
model solved interpretation quality. Architecture and product implementation
follow this selection. Quantization has not been tested.

## Verification

All 43 pinned files passed byte-count and SHA-256 checks at download and again
before the run. Marker/neighbour IDs and the AR suffix matched metadata;
captured and reconstructed vectors were finite. Python syntax and manifest
consistency checks, local documentation links, and `git diff --check` passed.
No code was committed or pushed. No system packages or GPU drivers changed.
