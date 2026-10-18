# Engine source

The C++20/CUDA implementation belongs here, following the accepted
[file mapping](../docs/architecture/file-mapping.md). Initial header designs
cover numerical representation, model observation sites, capture preparation,
and a complete causal forward pass without a persistent KV cache. The remaining
boundary designs cover tensor extents, verified safetensors loading, model
recipes, Qwen3 checkpoint binding, device resources and kernel operations.
The headers contain contracts and corpus citations; runtime behavior is not
implemented. Model equations live in the model recipe; the executor resolves
kernel capabilities, memory placement and observation copies.

First-slice `.cpp` and CUDA `.cu` files currently contain opening design comments
and includes only. They specify local algorithms, ownership, numerical behavior,
costs and corpus citations before implementation. No function or kernel bodies
have been written. BF16 conversion, in-memory parsers/validators and scratch
placement provide direct unit-test boundaries without files or device execution.

The planned engine runs target inference, captures activations, executes the
separate NLA verbalizer and reconstructor models, and serves the application
interface. There is no executable yet.
