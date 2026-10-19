# Source-build environment. Weights are supplied at runtime, never in the image.
# CUDA 13.2.0 devel / Ubuntu 24.04, verified linux/amd64 manifest.
FROM nvidia/cuda@sha256:f9492f2eea77fbc3d0c14fa8738f35946b42da72917bf5959d284ca39b4f209a
RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake=3.28.3-1build7 ninja-build=1.11.1-2 \
      libssl-dev=3.0.13-0ubuntu3.16 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
ENTRYPOINT []
CMD ["bash"]
