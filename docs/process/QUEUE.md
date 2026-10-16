# Work queue

## Active

None.

## Next

1. Inspect the exact Qwen3 4B/NLA checkpoints, activation conventions, precision
   needs, and memory layout. Pin sources and record the budget for the 3090.
2. Write the logical overview, change axes, and file mapping.
3. Select and pin the Docker/CUDA/CMake toolchain, then implement the source
   build and GPU verification entry point on the Linux machine.
4. Implement the target-model and NLA paths and the synchronized browser
   interface, with concrete milestones defined by the architecture.

## Deferred

- Native macOS/Metal and other GPU platforms.
- Browser-only inference.
- Explanation editing and causal steering.
- Training an NLA, unless the selected model pairing requires it.

## Accepted

- **Repository bootstrap, 2026-10-16:** source layout, separate `app/` and
  `web/`, local workflow, license files, original README banner, and project
  memory established. The brief was moved to ignored `brief.md`. Setup
  recommendations accepted with no imported governance. Engine implementation
  and GPU verification remain next work.
