# Verification

Follow the [test scope and stability rules](../docs/process/workflow.md#test-scope-and-stability).
Unit tests call isolated code in process and check stable contracts/invariants;
they do not launch systems or depend on private implementation details.
Integration tests cover real component interactions and are capped at five
across the repository. A test runner's name does not determine test scope.

Verification will cover independent numerical comparisons of the target and
NLA computation, GPU execution on the 3090, model loading, and the browser/API
interaction. GPU tests and ordinary build checks will have separate entry
points. The GPU path must also run with forced PTX compilation.

Fixtures should be small and attributable. Full model weights belong in the
ignored cache, with pinned revisions and checksums. Tests arrive with the
implementation they verify. Engine verification remains upcoming.

[gpu/forward_reference.h](gpu/forward_reference.h) designs the first harness
boundary: exact input provenance, selected logits and configurable observation
sites compared with independent reference values. It declares comparisons;
no numerical test implementation or reference fixtures have been added yet.

[support/numerical.h](support/numerical.h) separates pure array error metrics
from completed-model-result comparison. Unit tests can cover zero reference
norms, nonfinite actual values and explicit tolerance contracts using ordinary
arrays. Neither comparison helper launches inference. In-memory container and
configuration parsers, operation/recipe validators, BF16 conversion and scratch
placement are also direct unit boundaries. Their tests must check contracts,
not exact private schedules, launch geometry or allocator offsets.

The `.cpp`/`.cu` files added in the header-design pass contain design comments
and includes, with no function bodies or test cases. No integration scenarios
were added; the existing five site tests still occupy the entire budget.

First-slice verification will include:

- Malformed/overflowing tensor extents, shard/index disagreement, artifact
  identity mismatch, and missing or incompatible model parameters.
- Recipe dependencies, Q/K normalization before rotation, causal GQA and both
  residual boundaries, checked against the independently pinned reference.
- Single-token and multi-token prefixes; first, middle and final observation
  blocks; selected/all/no capture rows; empty and selected logit rows.
- Capture-on/off equivalence, prefix causality and output ownership after
  releasing plans/checkpoints; failures must not publish partial success.
- Queue completion, scratch reuse after copies and forced PTX execution on
  the NVIDIA machine, separately from host-only header/build checks.

Numerical thresholds must be established from reference evidence before
accepting the implementation; no thresholds or successful GPU results are
claimed by these interface designs.

The five existing site integration tests run with Python 3.11+ and Hugo 0.154.2:

```sh
python3 -m unittest discover -s tests -p 'test_site.py' -v
```

They build a disposable site under a deployment subpath, check draft/future
exclusion and milestone rendering, and inject broken links, assets, and an
unintended document to verify the publication guards.

These five tests currently fill the integration budget. Before adding engine
integration tests, move isolated publication validation contracts to unit tests
and retain only essential site integration scenarios. The first-slice cases
listed above are verification requirements, not a separate integration test for
every bullet or input variation.
