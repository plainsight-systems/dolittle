# Source-build environment. Weights are supplied at runtime, never in the image.
# CUDA 13.2.0 devel / Ubuntu 24.04, verified linux/amd64 manifest.
FROM nvidia/cuda@sha256:f9492f2eea77fbc3d0c14fa8738f35946b42da72917bf5959d284ca39b4f209a AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake=3.28.3-1build7 ninja-build=1.11.1-2 \
      libssl-dev=3.0.13-0ubuntu3.16 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
ENTRYPOINT []
CMD ["bash"]

# Analysis is opt-in and does not change the production compiler or base image.
FROM build AS analysis
RUN apt-get update && apt-get install -y --no-install-recommends curl gnupg python3 \
    && curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key -o /tmp/llvm.asc \
    && gpg --batch --dearmor -o /usr/share/keyrings/llvm.gpg /tmp/llvm.asc \
    && echo 'deb [signed-by=/usr/share/keyrings/llvm.gpg] https://apt.llvm.org/noble/ llvm-toolchain-noble-22 main' > /etc/apt/sources.list.d/llvm.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       clang-22=1:22.1.8~++20260714014902+ca7933e47d3a-1~exp1~20260714135019.80 \
       clang-tidy-22=1:22.1.8~++20260714014902+ca7933e47d3a-1~exp1~20260714135019.80 \
    && rm -f /tmp/llvm.asc && rm -rf /var/lib/apt/lists/*

# Preserve the original default build environment.
FROM build AS default

# Local source build: this image is produced by the launcher, not distributed
# as an application executable. Model cache mounts are supplied only at runtime.
FROM build AS runtime
WORKDIR /opt/dolittle
COPY CMakeLists.txt ./
COPY src ./src
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DDOLITTLE_CUDA=ON -DBUILD_TESTING=OFF \
    && cmake --build build -j 4
COPY app ./app
COPY config ./config
COPY LICENSE LICENSE-CODE NOTICE ./
COPY docs/licenses ./docs/licenses
CMD ["/opt/dolittle/build/dolittle"]

# Keep ordinary `docker build` compatible with the established build tooling.
FROM build AS final
