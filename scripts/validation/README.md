# Reference feasibility check

These scripts evaluate published checkpoints before building Dolittle's own
C++/CUDA engine. Transformers and PEFT are reference tools, not application
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
