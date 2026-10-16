# Qwen3 4B / NLA feasibility — 2026-10-16

**Historical baseline:** Andrew subsequently selected the 8B pair after the
[comparison](../2026-10-16-nla-comparison/README.md). This note preserves the
earlier 4B experiment and its findings.

## Scope and reproducibility

This is a reference execution of published weights before implementing
Dolittle. It does not validate an engine, frontend, Docker image, live capture,
or PTX fallback. Reproduction scripts and the full dependency lock are in
[`scripts/validation`](../../../scripts/validation/README.md).

The test machine reports **NVIDIA GeForce RTX 3090 Ti**, 24,564 MiB, compute
capability 8.6; earlier planning called it a 3090. Linux, Python 3.12.3,
NVIDIA driver 595.91.07, PyTorch 2.7.1+cu128, Transformers 4.57.3, PEFT 0.18.0.
The intended baseline remains a 24 GB Ampere card; performance on a non-Ti
3090 has not been measured.

## Checkpoint contract

Every downloaded file passed byte-count and SHA-256 checks against
[`models.json`](../../../scripts/validation/models.json). All 504 tensor
shapes in each LoRA adapter match the pinned base projection shapes.

| Input | Pinned revision |
| --- | --- |
| [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B/tree/1cfa9a7208912126459214e8b04321603b3df60c) | `1cfa9a7208912126459214e8b04321603b3df60c` |
| [Universal AV GRPO](https://huggingface.co/anicka/nla-qwen3-4b-universal-av-grpo/tree/d5f6ecaee901484afca0e79bd9c066bff4a69f6f) | `d5f6ecaee901484afca0e79bd9c066bff4a69f6f` |
| [Universal AR](https://huggingface.co/anicka/nla-qwen3-4b-universal-ar/tree/89835dc3f0ce044913c96ba23d6ab971cc15bcd5) | `89835dc3f0ce044913c96ba23d6ab971cc15bcd5` |

Model cards declare Apache-2.0. Reference conventions were inspected at
[nla-at-home revision 6b404d2](https://github.com/anicka-net/nla-at-home/tree/6b404d2af864569d0cf32a0ab6967b458b0931d0).
The community models are separate from Anthropic's original release.

- Base: 36 blocks, residual width 2,560, 32 query heads, 8 KV heads,
  head dimension 128, MLP width 9,728, tied vocabulary embedding of 151,936.
- Both adapters: ordinary rank-16 LoRA on all seven attention/MLP projections.
  Alpha is 64 for AV and 32 for AR. The reference keeps the base in BF16 and
  the adapters in FP32, with no quantization and only one resident base.
- Capture **block output**, with a zero-based layer index. The state after
  consuming token t predicts token t+1. This pass tested layers 18, 22, 26,
  all among the AV's training layers.
- Map `L * 100 / 36` to the nearest metadata depth percentage, giving
  47%, 63%, 71% for these layers. Do not use `(L + 1)` in that mapping.
- Normalize each activation in FP32 to L2 norm 150 and replace the embedding
  at marker `㈎`, token ID 149705. Preserve the exact metadata AV prompt;
  apply the chat template with `enable_thinking=False`, as the model card
  instructs. Target generation uses `enable_thinking=True`.
- AR uses its raw metadata prompt, without chat formatting, and reads the
  requested block at the final marker. There is no separate value head in
  this checkpoint. The bundle has no training-corpus layer means; raw cosine
  cannot reproduce the published centered-cosine evaluation.

The reference extraction script captures the last formatted prompt position
with thinking disabled. Generated thinking tokens therefore pose a distribution
change. Successful loading does not establish useful interpretations there.

## Memory derivation

The base contains 4,022,468,096 parameters: **8,044,936,192 bytes (7.492 GiB)**
at BF16. Both adapters together add 264,241,152 parameter bytes at FP32,
for **8,309,177,344 bytes (7.739 GiB)** of resident parameters, measured by
summing loaded tensors. File headers and tokenizer files are download costs,
not GPU weight storage.

BF16 KV cache per token, batch size one:

`36 layers × 2 (K,V) × 8 KV heads × 128 dimensions × 2 bytes = 147,456 bytes`

An 8,192-token cache therefore costs **1.125 GiB**. This is a calculation,
not an 8K-context runtime measurement. Capture storage for three FP32 layers
is 30,720 bytes per selected token; retaining all 8,192 positions would cost
240 MiB. Keeping only selected positions avoids that cost.

AV and AR run sequentially with the target. Switching adapters must keep
separate cache ownership: target KV cannot be reused as AV or AR KV. A future
scheduler that preserves a paused target's cache while explaining must budget
both caches, capture buffers, workspace, and allocator/runtime overhead.

## Measured thinking-token pass

The complete [raw results](results.json) preserve all four prompts, generated
traces, 24 descriptions, reconstruction diagnostics, and the input manifest.
The default thinking mode was used; the later optional control mode does not
change that path. Seeds are 100–103, target sampling temperature 0.6, top-p
0.95, top-k 20, maximum 512 new tokens. AV is greedy, maximum 160 new tokens.

| Measurement | Observed |
| --- | --- |
| Loaded parameter storage | 7.739 GiB |
| Peak PyTorch allocated memory | 7.915 GiB |
| Peak PyTorch reserved memory | 8.055 GiB |
| CUDA-reported free memory at end | 15.137 GiB |
| Target generation | 27.4–29.7 tokens/s |
| AV per captured vector | 5.83–10.42 s; median 10.28 s |
| AR per description | 69–71 ms |
| Prefix versus full replay relative L2 | 0.0092–0.0147 |

These are batch-one, short-context, sequential reference measurements. Peak
allocated memory excludes allocations outside PyTorch; reserved memory and
CUDA-reported free memory give additional context. They are neither an 8K
stress test nor predictions of the future direct engine's speed.

Three of four target runs reached the 512-token limit while still thinking.
The poetry run finished at 299 tokens with a final answer. Twenty of 24 AV
runs reached the 160-token cap. Bound generation explicitly in the application
and identify unfinished output. Do not imply every example is a complete
answer or complete interpretation.

Captures replay the generated sequence through the base with adapters disabled.
The prefix-only comparison at one position per case passed the exploratory
2% relative-L2 check; it does not establish bit-exact equivalence to cached
decode states. Direct capture and independent numerical verification remain
engine work.

**Hardware fit passes; semantic faithfulness remains undetermined.** The
initial report treated transcript disagreement too broadly as interpretation
failure. These are 32 outputs across four prompts, not 32 independent cases.
The observed descriptions often identify a broad topic but include specifics
absent from the transcript:

| Actual context | AV behavior |
| --- | --- |
| Counting red and blue marbles | Describes coin distributions, combinations, or subtraction involving apples |
| Key moved to a coat pocket | Mentions a pocket alongside a notebook, a dog, or a bear story absent from the transcript |
| Rabbit poem ending in habit/rabbit | Discusses sentence structure or poetry alongside imagery absent from the transcript |
| Cedar-tree syllogism | Sometimes identifies predicate logic, but also asserts the unsupported conclusion or switches to unrelated classification |

Matched raw AR cosine averaged 0.444, versus 0.409 for a description from a
different prompt at the same layer; matched won 17 of 24 comparisons. Each
swap uses the first different case in fixed prompt order, not all pairwise
controls. This small, non-independent diagnostic is not a semantic fidelity
score or a statistical claim. High or improved raw cosine does not establish
the truth of an interpretation.

The demo remained technically feasible on this hardware, so this pairing
was retained as a candidate pending comparison; 8B was subsequently selected.
Semantic faithfulness remains unverified.
Neither generated reasoning nor NLA output is ground truth for the other.
A claim that the user mentioned apples is false when the transcript contains
only marbles; a claim that an activation represents an apple-counting pattern
could be an inaccurate interpretation or a genuine association. Transcript
comparison alone cannot decide that. An unexpected description is also not
proof of hidden computation.

Andrew confirmed that exploring agreement, divergence, and uncertainty is
part of the demo. Matching narratives is not an implementation gate. Validate
numerical execution and capture alignment, and present both views with their
limits. Independent interventions would be needed to substantiate causal
interpretations; this pass did not perform them. Training a new NLA remains
a separate scope decision. Latency favors selected, on-demand captures;
explaining every generated token is not supported by these measurements.

## Prompt-end control

The [prompt-end control output](prompt-end-results.json) uses the same four
user prompts, thinking-disabled formatting, and the last prompt position at
layer 22. This is closer to the reference extraction convention. It still
produces descriptions that diverge from the transcript: marbles become a
cups-to-liters conversion, a key-location
question becomes the location of a girl, and the rabbit poem becomes unrelated
peaceful imagery. Raw reconstruction cosine is 0.782–0.960 despite those
discrepancies, reinforcing that raw cosine must not be shown as semantic confidence.

This control does not isolate the cause of the discrepancies. It shows that
they are not confined to generated thinking positions. There
is also a source/documentation discrepancy to check: the card explicitly says
`enable_thinking=False`, while the pinned
[universal AV training script](https://github.com/anicka-net/nla-at-home/blob/6b404d2af864569d0cf32a0ab6967b458b0931d0/scripts/train_universal_av.py)
and round-trip evaluation use the tokenizer default when formatting AV
prompts. The [alternate-template control](training-template-results.json)
uses that default on the same inputs. It changes some wording (including
"No, because" in the logic example), but still describes a feet-to-inches
problem absent from the transcript, relocates a girl rather than a key, and describes unrelated poetry.
It does not establish semantic faithfulness. Neither template comparison establishes
the historical training configuration or rules out other numerical/protocol
issues; it prevents attributing the observed discrepancies solely to this one flag.

## Deployment check

Host GPU execution works. Docker 29.9.1 is installed, but NVIDIA Container
Toolkit is absent and GPU container access fails with:

```text
failed to discover GPU vendor from CDI: no known GPU vendor found
```

The configured runtimes are runc and io.containerd.runc.v2. Installing and
configuring the toolkit requires host administrator access, unavailable to
this unattended session. No host driver or system package changes were made.
Follow the [NVIDIA installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
when preparing the Docker host, then repeat the GPU container check.

A source build, custom CUDA kernel correctness, forced PTX JIT, and execution
on RTX 40/50-series hardware remain future checks. No remote was configured
and nothing was pushed during this validation.

## Repository checks

Python syntax checks passed for both reference scripts. The downloader's
cache-verification path ran successfully against every downloaded file.
Markdown local links resolve, the banner is a valid 2,172 × 724 PNG, and
`git diff --check` passed. The brief and caches remain ignored. There is no
Git remote; validation changes remain local for review.
