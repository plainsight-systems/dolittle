# Remaining work

Generation, activation interpretation, reconstruction, steering and the browser
application are implemented on CUDA and Metal. The common source launcher and
illustrated walkthrough cover both platforms.

## Platform work

- Add native Windows/NVIDIA builds with MSVC/CUDA, without requiring WSL2.
- Decide whether to support AMD.
- Extend runtime and numerical coverage to additional GPUs and longer contexts.

## Future extensions

- Export retained experiments to files.
- Expose standalone reconstruction metrics in the interface.
- Evaluate cache pooling or prefix reuse if representative runs justify them.

Browser-only inference, NLA training, additional model pairs and resumable
generation are outside the current scope. The
[architecture overview](../architecture/README.md) describes the implemented
system; [verification](../../tests/README.md) lists the available checks.
