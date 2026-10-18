# Work queue

## Active

- Architecture review is complete: use the accepted
  [file and folder mapping](../architecture/file-mapping.md) and change axes
  as the baseline for file-header design. Preserve explicit implementation
  families and keep the selected model as their first concrete implementation.

- The project site is deployed at `dolittle.plainsight-systems.com`. The first
  build and deployment passed; HTTP content, Code link, and share metadata are
  verified. Check certificate provisioning before enabling enforced HTTPS.
  Engine architecture follows this site work; preserve the selected 8B pair
  and block 24 reference contract.

## Next

1. Design file headers for the first implementation slice using the accepted
   mapping. Query both local C++ corpus MCP servers during header design and
   again during implementation. Headers must state contracts, describe their
   algorithm steps, and cite the retrieved rules and their application.
2. Define numerical checks for the direct implementation and capture alignment.
   Preserve reference outputs; distinguish transcript claims from activation
   hypotheses and do not present raw reconstruction cosine as confidence.
3. Prepare host NVIDIA Container Toolkit access, then select and pin the
   Docker/CUDA/CMake toolchain and implement source-build/GPU checks.
4. Implement the target-model and NLA paths and the synchronized browser
   interface, with concrete milestones defined by the architecture.

## Later platform work

- Implement Metal on macOS after Linux/NVIDIA. Design its native build path
  and verify hardware requirements in that phase.
- Decide whether to support AMD.

## Deferred
- Browser-only inference.
- Explanation editing and causal steering.
- Training an NLA, unless the selected model pairing requires it.

## Accepted

- **File mapping and revised axes, 2026-10-17:** Andrew accepted the revised
  layout with explicit model, tokenizer, tensor-format/representation, cache,
  sampling, NLA-pair, metric and backend families. Only initial kinds are to be
  implemented. File-header design follows; no source scaffolding was created.

- **Change-axis direction, 2026-10-17:** Andrew liked the axes and authorized
  file/folder mapping. Granularity remains subject to evaluation against the
  actual layout; an axis does not mandate a directory or a class hierarchy.

- **Algorithmic overlays, 2026-10-17:** Andrew reviewed and accepted the
  separate NLA experiment and transformer execution documents. They establish
  the computation, capture semantics, state lifetimes, and backend variation
  for the next change-axis design step. Engine implementation remains pending.

- **First publication, 2026-10-16:** pushed main and deployed the project site
  through GitHub Pages. Five local and CI publication tests passed, as did the
  production build and deploy job. Custom-domain HTTP is live; HTTPS certificate
  provisioning remained pending at verification. The private brief/cache were
  excluded. Workflow: https://github.com/plainsight-systems/dolittle/actions/runs/38967736528.

- **Project site, 2026-10-16:** Andrew requested the site before engine
  architecture, closer to Cradle's explanatory writing than a milestone-led
  structure. Built the local Hugo site with the original artwork, two opening
  articles, an About page, and RSS. Explanations, explorations, and milestones
  share one Writing index. Five publication checks pass, including a subpath
  build, draft/future exclusions, broken-link/asset rejection, and share metadata. Local preview
  only; public copy and design are ready for review.

- **Model selection, 2026-10-16:** Andrew selected Qwen3 8B with
  `marco-bazzani/Qwen3-8B-nla`, using the revisions exercised in the comparison.
  Initial capture layer is zero-based block 24. Preserve 4B as historical
  comparison evidence; proceed to architecture for the selected 8B pair.
- **4B/8B comparison, 2026-10-16:** pinned and verified the full 8B pair;
  sequential BF16 reference run completed on the 3090 Ti. Peak allocated
  memory 15.378 GiB, median thinking AV time 4.73 s, all twelve descriptions
  complete. Full results and comparison limitations recorded; Andrew
  subsequently selected 8B. No commit or push made.
- **Reference feasibility pass, 2026-10-16:** pinned input manifests and all
  adapter shapes checked; actual Linux/CUDA execution and shared-base memory
  measured on an RTX 3090 Ti. Semantic faithfulness remains unverified.
  The initial narrative-agreement gate was withdrawn after discussion.
  Docker GPU access is unavailable until the host toolkit is configured.
- **Repository bootstrap, 2026-10-16:** source layout, separate `app/` and
  `web/`, local workflow, license files, original README banner, and project
  memory established. The brief was moved to ignored `brief.md`. Setup
  recommendations accepted with no imported governance. Engine implementation
  and GPU verification remain next work.
