# Working on Dolittle

## Session entry

Read `docs/process/MEMORY.md`, `docs/process/QUEUE.md`, and
`docs/process/workflow.md`, then the relevant architecture documents and
source. `CLAUDE.md` points to the same entry path.

## Local rules

- Follow the self-contained workflow in `docs/process/workflow.md`.
- Do not import governance documents or add a governance submodule or symlinks.
- Design the architecture before the engine: logical overview, change axes,
  then file mapping. Design each implementation change in its file headers.
- Treat inference, loading, memory residency, activation capture, and GPU
  dispatch as performance-sensitive.
- Implement the core computation directly. Reference implementations may
  supply verification evidence; dependency choices must remain explicit.
- Keep decisions and work state in memory and the queue. Keep investigations
  and measurements in `docs/research/`.
- A tech demo is a complete implementation of a narrow scope. Reduce scope
  rather than lowering correctness or quality within it.
- Keep machine access details and local paths out of tracked project files.
  `brief.md` and `CLAUDE.local.md` are local, ignored planning/context files.
- Implement Linux/NVIDIA first. Metal on macOS is planned afterward; AMD
  support is undecided. Browser-only inference remains out of scope.
- Keep `app/` (the inference interface) separate from `web/` (the project site).
- Report checks actually performed. A build is not evidence of GPU correctness
  or runtime performance.
