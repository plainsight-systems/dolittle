# Verification

Verification will cover independent numerical comparisons of the target and
NLA computation, GPU execution on the 3090, model loading, and the browser/API
interaction. GPU tests and ordinary build checks will have separate entry
points. The GPU path must also run with forced PTX compilation.

Fixtures should be small and attributable. Full model weights belong in the
ignored cache, with pinned revisions and checksums. Tests arrive with the
implementation they verify; none exist at this stage.
