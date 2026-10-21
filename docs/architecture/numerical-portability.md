# Numerical portability

CUDA and Metal implement the same transformer equations, storage precision,
rounding boundaries and observation points. Each backend uses its native
FP32 elementary functions. Matching a particular GPU's approximation of
`exp`, `rsqrt`, `sin` or `cos` is not a portability requirement.

BF16 tensors remain BF16, reductions remain FP32, and probabilities are FP32
before multiplication by values. RMSNorm rounds the normalized activation
before multiplying its weight. SwiGLU rounds SiLU before multiplying the up
projection. RoPE retains the documented product and trigonometric rounding.
These are model computation requirements, separate from numerical tolerances.

## Repeatability and accuracy

| Comparison | Requirement |
|---|---|
| Same backend, hardware, build and input | Exact repeatability. |
| Full forward versus cached chunks on that backend | Exact stored logits and captures for identical consumed tokens. |
| An isolated operation versus independent arithmetic | Explicit operation/dtype tolerance, with identical stored inputs. |
| Complete Metal model versus its frozen independent reference | Report the original per-row limits, errors and token agreement; require finite, structurally valid outputs. |
| Complete CUDA model versus its frozen CUDA reference | Continue enforcing the existing fixture limits. |

Cross-backend token equality is reported separately. Small logit differences
can change a close argmax or a sampled token. Same-backend replay still requires
the same RNG sequence, selection policy and results. Shape, causal visibility,
positions, hook identity, copied bytes and integer control have no tolerance.

## Operation checks

The existing target numerical integration workflow also isolates projection,
RMSNorm, RoPE, attention, residual addition and SwiGLU. It supplies deterministic
stored inputs to both the production GPU operation and a small independent
host implementation of the equation. The reference uses FP32 arithmetic and
the specified BF16 conversion points; it does not reproduce GPU reduction
trees or call production mathematical kernels. Host reference work and readback
are diagnostic work, outside inference and its timing.

For each element, require

`abs(actual - reference) <= atol + rtol * abs(reference)`.

Reject nonfinite results. Report the number of failing elements, maximum
absolute error and whole-array relative L2 as separate metrics. The latter is
diagnostic here, not another name for `rtol`.

| Operation | FP32 atol / rtol | BF16 atol / rtol | Source |
|---|---|---|---|
| Projection, RoPE, SwiGLU | 1e-5 / 1.3e-6 | 1e-3 / 1.6e-2 | vLLM IR defaults |
| Weighted RMSNorm | 1e-2 / 1e-2 | 1e-2 / 1e-2 | vLLM layernorm test |
| Causal attention | 1e-3 / 1e-3 | 3e-2 / 2e-2 | vLLM-Metal attention checks |
| Residual addition | 0 / 0 | 0 / 0 | One addition and one storage conversion |

The tolerances are adopted for these operation checks before running them.
They are not whole-transformer error allowances. Dolittle's attention reference
retains FP32 probabilities as required by its computation contract; the upstream
Metal reference casts probabilities to the value dtype. We borrow its tolerance
policy, not that different arithmetic.

## Existing references

The frozen CUDA/PyTorch fixtures, generator and limits remain intact. Their
reports identify the reference device and numerical policy. CUDA acceptance
uses the independent CUDA reference; M3 acceptance uses that oracle running
through PyTorch MPS on the M3. Cross-backend comparisons remain portability
diagnostics with their actual errors and pass/fail results preserved.
A failing same-backend full-model comparison remains a failure even when isolated operations pass. Operator
tolerances cannot be substituted for an activation-row relative L2 limit.
Changing a complete-model criterion requires its own explicit justification;
there is no threshold fitted to the current Metal output.

The MPS wrapper imports the frozen reference's projection/attention functions
unchanged and pins the same Transformers Qwen3 source. It retains the original
serialized weights, token traces, observation identities and numeric limits.
Stock RMSNorm, RoPE and SiLU execute through MPS, with CPU fallback and fast math
disabled. Exact reference replay/prefix invariance and the greedy trace are checked
at batch size one before writing a fresh fixture and its SHA-256 manifest.
That is Dolittle's supported execution shape. MPS peer-batch variation is
reported separately; the baseline makes no multi-sequence invariance claim.
No Dolittle output is
used to construct expected values. The MPS dependency lock and OS/device
provenance identify the independent baseline.

No separate cross-vendor calibration study is required. Use established
verification methods and test our implementation against them.

## Whole-model comparisons

Same hardware does not imply the same accumulation order across independent
implementations. A first differing projection value locates divergence; it does
not establish an incorrect projection. Keep Dolittle's fixed reductions and
exact full/cached agreement. Investigate a kernel when identical-input operation
checks or its computation contract fail.

The upstream checks serve different purposes:

| Check | What the source actually does | Use in Dolittle |
|---|---|---|
| vLLM generation against Hugging Face | The common generation test includes Qwen3-8B, generates 32 tokens and requests five logprobs. At the first token mismatch, the helper requires each selected token to appear in the other's top five, then stops comparing the diverged continuations. | Report reciprocal argmax ranks on identical consumed prefixes. This measures token-choice agreement, not numerical closeness of logprob values. |
| llama.cpp architecture test | Compares CPU/device logits with NMSE <= 1e-4 on small models with generated weights; the default is two layers and width 256. | Use its NMSE definition, but do not transfer the small-model threshold to trained 36-layer Qwen3-8B captures. |
| llama.cpp perplexity tool | Reports KL divergence, likelihood changes and top-token agreement over matched evaluation tokens. | Report distribution differences on matched prefixes. The tool supplies metrics, not a universal pass threshold. |
| vLLM embedding comparison | Requires cosine similarity >= 0.999 for returned embedding vectors by default. | Report capture cosine alongside magnitude. Embedding similarity does not set a residual-stream acceptance limit. |

The existing target comparison reports these additional diagnostics for each
consumed row, reusing completed test readbacks:

- **All rows:** NMSE, cosine similarity and actual/reference norm ratio.
- **Logits:** KL(reference || actual), total variation, each argmax token ID,
  and the rank of each argmax in the other distribution.

Logit distributions use temperature one with no sampling filters. For reference
logits `r` and actual logits `a`, subtract each row's maximum before exponentiating,
then normalize to `p` and `q`. Report `sum(p * (log(p) - log(q)))` in nats and
`0.5 * sum(abs(p - q))`. Ranks are one-based; ties use the smaller token ID.
Calculations use double host intermediates outside inference timing. This changes
neither the model's precision nor reference arithmetic.

For capture vectors, let `s = ||a|| / ||r||` and `c` be their cosine similarity.
Their squared relative L2 error is `(s - 1)^2 + 2*s*(1 - c)`. Reporting both
terms distinguishes changed magnitude from changed direction. Undefined
zero-norm metrics are null; nonfinite actual rows retain their failed numerical
verdict and have unavailable diagnostics.

For Metal, the original 3% whole-model budget is diagnostic. It is not a
release gate. Independent operation accuracy, model precision and rounding
boundaries, finite outputs, exact same-backend replay and full/cached agreement
remain required. This applies to target captures/logits, verbalizer outputs and
reconstructed vectors. Differences in independently generated NLA continuations
are reported; repeated execution on the same backend must agree exactly.

Reports retain each original numerical `passed` result and add `accepted` for
the selected model-comparison policy. No fixture limit or reference arithmetic
changes. Shape and provenance mismatches and nonfinite values always fail.
CUDA retains its established strict model comparisons.

This is sufficient for the experiment: check the operations, preserve repeatable
execution, and exercise interpretation and steering through the real application.
The native-source investigation is complete; matching MPSGraph's private
reduction tree is not a requirement.


## Source basis

- [vLLM IR tolerances](https://github.com/vllm-project/vllm/blob/4a3ee9a0f6a2958f2712ceb3c9f60fe48cd764b3/vllm/ir/tolerances.py)
  specify dtype defaults and operation overrides.
- [vLLM RMSNorm tests](https://github.com/vllm-project/vllm/blob/4a3ee9a0f6a2958f2712ceb3c9f60fe48cd764b3/tests/kernels/core/test_layernorm.py)
  compare weighted normalization with the native reference at 1e-2 absolute and
  relative tolerance.
- [vLLM-Metal attention checks](https://github.com/vllm-project/vllm-metal/blob/55a3352cd7a16ce19825173e5252ca37a26e740a/tools/attention_bench_utils.py)
  distinguish elementwise closeness from exact equality.
- [llama.cpp backend tests](https://github.com/ggml-org/llama.cpp/blob/0a1ef309cdbb9f2024f58a38fb4c42352c5db569/tests/test-backend-ops.cpp)
  use operation/backend overrides and normalized mean squared error. That
  metric is squared relative L2 when the reference is the denominator; it is
  not elementwise relative tolerance.
- [vLLM reproducibility](https://docs.vllm.ai/en/latest/usage/reproducibility/)
  scopes reproducibility to the same hardware and vLLM version.
- [vLLM model comparison helpers](https://github.com/vllm-project/vllm/blob/4a3ee9a0f6a2958f2712ceb3c9f60fe48cd764b3/tests/models/utils.py)
  implement reciprocal top-k token checks and embedding cosine comparison;
  [the common generation test](https://github.com/vllm-project/vllm/blob/4a3ee9a0f6a2958f2712ceb3c9f60fe48cd764b3/tests/models/language/generation/test_common.py)
  includes Qwen3-8B and selects five logprobs.
- [llama.cpp architecture comparison](https://github.com/ggml-org/llama.cpp/blob/0a1ef309cdbb9f2024f58a38fb4c42352c5db569/tests/test-llama-archs.cpp)
  uses generated small models and a 1e-4 NMSE limit.
- [llama.cpp perplexity and KL diagnostics](https://github.com/ggml-org/llama.cpp/blob/0a1ef309cdbb9f2024f58a38fb4c42352c5db569/tools/perplexity/perplexity.cpp)
  report distribution and likelihood differences.
- [PyTorch 2.7.1 MPS linear](https://github.com/pytorch/pytorch/blob/e2d141dbde55c2a4370fac5165b0561b6af4798b/aten/src/ATen/native/mps/operations/Linear.mm)
  builds MPSGraph matrix multiplication directly;
  [MPS linear algebra dispatch](https://github.com/pytorch/pytorch/blob/e2d141dbde55c2a4370fac5165b0561b6af4798b/aten/src/ATen/native/mps/operations/LinearAlgebra.mm)
  distinguishes floating bmm from the optional Metal mm kernel.

These projects establish the approach. Their tests do not execute Dolittle's
custom kernels.
