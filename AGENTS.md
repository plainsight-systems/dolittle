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
- Both C++ header design and implementation must query the local
  `cpp-guidelines` and `cpp-perf-guidelines` MCP servers. Opening file comments
  must cite the retrieved rules and explain their application, state the
  file's contract, and describe its part of the algorithmic overlay. Follow
  the detailed consultation and unavailable-server rules in
  `docs/process/workflow.md`.
- Treat inference, loading, memory residency, activation capture, and GPU
  dispatch as performance-sensitive.
- Implement the core computation directly. Reference implementations may
  supply verification evidence; dependency choices must remain explicit.
- Keep settled decisions and concise work state in memory and the queue.
  Implementation diaries, build logs, validation reports and development-cycle
  notes belong in gitignored `.local/`, not `docs/research/` or public writing.
  Reserve `docs/research/` for substantive research and model investigations.
- A tech demo is a complete implementation of a narrow scope. Reduce scope
  rather than lowering correctness or quality within it.
- Keep machine access details and local paths out of tracked project files.
  `brief.md` and `CLAUDE.local.md` are local, ignored planning/context files.
- Linux/NVIDIA and native macOS/Metal are implemented. Native Windows/NVIDIA
  remains planned; AMD support is undecided. Browser-only inference is out of scope.
- Keep `app/` (the inference interface) separate from `web/` (the project site).
- Report checks actually performed. A build is not evidence of GPU correctness
  or runtime performance.
- Unit tests exercise isolated code contracts and invariants in process. They
  must not launch builds, subprocesses, servers, GPU workloads or application
  flows. Avoid assertions about private implementation or incidental structure.
- Keep integration tests to a small handful: at most five across the repository.
  Do not evade the limit by hiding separate workflows in one test or relabeling
  them. Follow the test rules in `docs/process/workflow.md`.
