# Setup and launch tooling

The planned Python entry point will check host Docker and NVIDIA GPU access,
build the application from source in a pinned toolchain container, download
and verify model files in a persistent cache, and start the application.

Keep model downloads out of image builds. Record exact revisions, required
files, hashes, and provenance in a model manifest. Preserve existing cache files and
never reset a user's checkout. Build tools and CUDA compilation dependencies
belong inside the container; document host driver and container GPU
prerequisites separately. The root Dockerfile, Compose configuration, CMake
files, and executable commands will be added with the first working build.

The existing [`validation/`](validation/README.md) scripts are isolated reference
experiments for checkpoint selection. They are not the application launcher.

## Project site

`python3 scripts/site.py serve` builds and serves the public project site on
loopback, independently of the GPU application. It requires Python 3.11+ and
Hugo 0.154.2. See [`../web/README.md`](../web/README.md) for publication rules,
base URLs, and verification.
