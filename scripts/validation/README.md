# Reference checks

These scripts evaluate published checkpoints and generate independent fixtures
for Dolittle's C++/CUDA engine. Transformers and PEFT are reference tools, not application
dependencies. Run on Linux x86-64 with Python 3.12 and a BF16-capable NVIDIA GPU.

**Selected pairing:** Qwen3 8B with `marco-bazzani/Qwen3-8B-nla`; use the
[8B commands below](#qwen3-8b-comparison). The original 4B commands and
`models.json` remain available to reproduce the historical baseline.

From the repository root, in an isolated environment:

```sh
python3.12 -m venv .venv-validation
.venv-validation/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r scripts/validation/requirements.lock
.venv-validation/bin/python scripts/validation/fetch_models.py --root .cache/validation-run
.venv-validation/bin/python scripts/validation/reference_probe.py --root .cache/validation-run --output .cache/validation-run/results.json
```

Allow about 8.4 GB for the checkpoint download, plus the Python/CUDA packages
and temporary storage. The manifest pins immutable model revisions, file sizes,
and SHA-256 hashes. Both scripts verify the files. The lock records the exact
installed reference environment; it does not pin the future Docker toolchain.

The probe checks all adapter tensor shapes, uses one BF16 base with two FP32
LoRA adapters, generates four thinking traces, and interprets two positions
at three layers per trace. It captures residuals by causal replay of the actual
generated tokens, compares one position per trace with prefix-only replay,
then runs AV and AR sequentially. The 512-token generation cap can leave
thinking unfinished; this is an activation feasibility check, not an answer
quality benchmark. It does not test live capture inside cached decoding.

AR cosine comparisons against descriptions from other prompts are diagnostic
only. Training-corpus layer means are absent, so these raw scores are not the
published centered-cosine metric and must not become UI confidence values.
The raw output preserves every prompt, prefix, description, and timing.

For the matched prompt-end control, run the same probe with
`--mode prompt-end` and a different `--output` path. This uses thinking-disabled
prompt formatting, captures the last formatted prompt token at layer 22, and
generates four descriptions without a target continuation. It checks a
training-like extraction position, not a held-out training benchmark. Add
`--av-template training` to compare the tokenizer-default AV template used
in the published training/evaluation source with the card's explicit
`enable_thinking=False` instruction. Save each variant to its own output file.

The pinned PEFT version warns about `lora_ga_config` and `use_bdlora` in the
adapter configurations. Both are null in both pinned adapters; no enabled
adapter behavior is omitted. The adapters use ordinary rank-16 LoRA.

Published checkpoint licenses are recorded in the manifest. Consult each model
card before redistributing weights. These scripts download them separately.

## Qwen3 8B comparison

`models-8b.json` pins the separate 8B target, full verbalizer, and truncated
reconstructor with its value head. Allow about 45 GB for this additional
download. Run in the same reference environment, keeping its cache separate:

```sh
.venv-validation/bin/python scripts/validation/fetch_models.py --manifest scripts/validation/models-8b.json --root .cache/validation-8b
.venv-validation/bin/python scripts/validation/compare_8b.py --root .cache/validation-8b --output .cache/validation-8b/results.json
```

The comparison loads each model separately in BF16. It uses the same four
prompts, seeds, sampling settings, 512-token target cap, and 160-token AV cap
as the existing 4B pass. The 8B NLA reads block 24 only, at two positions per
thinking trace plus one thinking-disabled prompt-end control per case.
Each target generates its own trajectory; neither token positions nor raw
reconstruction scores are directly comparable across models. Both workloads
are short feasibility samples, not a semantic-fidelity benchmark.

The 8B AV follows its own default chat template. Its AR uses a raw prompt,
removes the final RMS normalization, and applies the separately saved linear
head to the last block's final-token state. These conventions differ from the
4B adapters and are checked against the checkpoint metadata and cited
reference source. Full outputs and per-phase memory/timings are saved.

## Independent full-forward and generation fixtures

`forward_reference.py` generates expected values without importing or executing
Dolittle. Use the same pinned environment above (PyTorch 2.7.1+cu128,
Transformers 4.57.3). From the repository root:

```sh
.venv-validation/bin/python scripts/validation/forward_reference.py --kind tiny --output .cache/generation-fixtures
.venv-validation/bin/python scripts/validation/forward_reference.py --kind actual --output .cache/generation-fixtures --actual-checkpoint .cache/validation-8b/models/Qwen--Qwen3-8B
```

The tiny FP32 and BF16 bundles use seed 20261018, three blocks, residual width
24, four query heads, two KV heads, head width 8, MLP width 40 and vocabulary
37. These dimensions exercise grouped attention, matrix tails and a head
projection width different from the residual width. Cases cover a 17-token
sequence, its five-token prefix, and different tokens at the original length.
The actual 8B BF16 bundle uses five-token, three-token-prefix and alternate-token
cases. Each records selected logits and block input/output rows at the first,
middle and last blocks, plus block 24 where present. The generation case adds
a real thinking-enabled 8B prompt (a small ID prompt for tiny models), four
greedy emissions and a complete no-cache recomputation of that consumed trace.
Every position has reference logits/captures. The actual bundle also carries
405 reference tokenizer ID vectors and 12 exact prompt templates. No Dolittle
code, cache or NLA execution is used to produce reference values.

CUDA is the default reference device; `--device cpu` supports reference
generation without GPU residency. Device provenance remains in the fixture;
CPU reference generation is not evidence that the CUDA engine executed.

The acceptance oracle retains BF16 storage for the actual model and FP32
accumulation. Each projection uses a fixed one-row FP32 library call; attention
uses independent dense FP32 scores, softmax and weighted sums for each query's
causal prefix. RMSNorm, RoPE and SiLU remain the pinned Transformers operations.
This reference must remain independent of the engine's selected reduction
order. Changing it to match engine arithmetic does not resolve a failed
independent comparison. TF32 and FP64 accumulation are not used.

Before writing any fixture, the generator requires original storage bytes for
all shared logits and selected captures to match at prefix lengths 1, prompt
length and complete-trace length minus one. It also evaluates the trace in both
slots of a two-sequence batch with an unrelated peer. Per-row byte hashes include
signed zero. These checks establish invariance for the fixture/environment;
they do not establish cross-GPU reproducibility. All arithmetic override sources and every
invariance result are recorded in provenance.

Stock Transformers BF16 eager attention also rounds scores and probabilities to
BF16. Its different policy is measured separately, never used to select limits.
FP64 experiments remain diagnostic evidence and do not generate acceptance fixtures.

Acceptance budgets were fixed before custom-engine results. For each output,
let `peak` be its largest absolute reference value; both error bounds must pass:

| Storage / model | Maximum absolute error | Relative L2 error |
| --- | --- | --- |
| FP32 | `max(1e-5, peak * 2e-4)` | `2e-4` |
| BF16, tiny three-block model | `max(1e-4, peak * 0.03125)` | `0.02` |
| BF16, 36-block target | `max(1e-4, peak * 0.0625)` | `0.03` |

The BF16 peak factors allow eight and sixteen unit roundoffs (`u = 2^-8`),
respectively, with a larger budget for the deeper model. These are explicit
numerical acceptance budgets, not a formal error proof or semantic guarantee.
Use each output's recorded bounds; bundle-wide maxima are only conservative
summaries. Large target residual outliers make those summaries especially loose.
Do not change bounds after observing an engine failure merely to admit it.

The 3% target limit remains a required acceptance gate, not an empirically
established bound on variation between BF16 implementations. No automatic
tolerance fitting occurs in this generator. Generation comparisons require each individual row to pass,
as well as retaining aggregate diagnostics and exact cache/full-trace equality.

Each bundle contains standard configuration/weight artifacts and `fixture.json`
with artifact sizes/SHA-256 hashes, exact tokens/selections, FP32-decoded values,
original storage types, per-output tolerances, generator/source hashes, package
versions, device/driver details and attention policy. The actual model uses
revision `b968826d9c46dd6066d109eabc6255188de91218`; its config, tokenizer/protocol artifacts, index and five
shards are verified against `models-8b.json`, then hardlinked into the output
bundle. Keep that output on the checkpoint's filesystem and keep linked artifacts
immutable. Fixture outputs stay in the ignored cache; the actual JSON is about
100 MiB including complete generation rows. GPU execution and comparison belong to the generation integration
scenario, not to this generator or the isolated unit tests.


## Independent AV fixtures

`av_reference.py` freezes the next stage of each existing numerical workflow.
It imports the unchanged forward reference (checking its source SHA-256), never
Dolittle. The captured target row is immutable input: CPU FP32 norm, divide,
scale, then one cast to AV storage and replacement of the stock embedding row.
Ordinary causal inference uses the existing independent operator policy. It
records full replay at each of four greedy output steps, selected intermediate
activations, exact prompt/marker alignment and environment/source hashes.
The normalized-vector limits are fixed before execution; AV output limits come
from the unchanged forward policy. Existing output files are rejected.

After generating the target fixtures above, run from the repository root:

```sh
.venv-validation/bin/python scripts/validation/av_reference.py --target-fixture .cache/generation-fixtures/tiny-fp32/fixture.json --output .cache/av-fixtures/tiny-fp32.json
.venv-validation/bin/python scripts/validation/av_reference.py --target-fixture .cache/generation-fixtures/tiny-bf16/fixture.json --output .cache/av-fixtures/tiny-bf16.json
.venv-validation/bin/python scripts/validation/av_reference.py --target-fixture .cache/generation-fixtures/qwen3-8b-bf16/fixture.json --av .cache/validation-8b/models/marco-bazzani--Qwen3-8B-nla --output .cache/av-fixtures/qwen3-8b-bf16.json
```

The production pair verifies the full target manifest, so its fixture directory
also needs `LICENSE`, `README.md`, `merges.txt` and `vocab.json` from the same pinned
target download. Copy or hardlink these additional immutable artifacts into
`.cache/generation-fixtures/qwen3-8b-bf16/`; do not replace existing files or the
frozen target fixture. AV verification includes all pinned repository artifacts
outside `ar/`; the reconstructor is not loaded. Each AV fixture records its model
root as an absolute path, so generate it where the engine checks will run.

Then run `python3 scripts/check.py gpu --fixtures .cache/generation-fixtures`
and repeat with `--force-ptx`. CTest uses the same sibling `av-fixtures` layout
when `DOLITTLE_GENERATION_FIXTURES` is configured; its working directory is the
repository root so the pinned model manifest can be located. Direct executable
runs also require that working directory. Keep fixture hashes and reports in
ignored storage. The generated four-token output is numerical coverage, not an
interpretation-quality benchmark or proof for arbitrary contexts.
