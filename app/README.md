# Live experiment interface

`app/` is served by the C++ inference service. `web/` is the independent public
project site. The browser has no inference dependencies or build step.

On Linux with an NVIDIA GPU, a compatible driver, Docker Compose and NVIDIA
Container Toolkit configured:

```sh
python3 scripts/run.py
```

For host GPU access, follow NVIDIA's [Container Toolkit installation and Docker
configuration guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
This is a one-time administrator setup on the Linux host. Installing CUDA inside
the image does not configure Docker's host GPU runtime. The launcher checks that
runtime and GPU visibility before loading models.

Use Python 3.11+. The launcher builds the application from this checkout and
prepares the revision-pinned target/AV cache once, downloading missing files and
verifying SHA-256 before saving a setup receipt. Later launches check the receipt
and file metadata without rehashing the weights. Changed files or a new manifest
trigger setup again. Rebuilding application code does not invalidate model setup.
The service retains read-only mappings across experiments and uploads only the
selected model. Keep cache files unchanged while the service runs.

Use `python3 scripts/run.py --verify-models` to force full verification and repair,
or `python3 scripts/models.py --cache /path/to/cache` to do that without launching.
Add `--offline` to verify without downloading replacements. The receipt remembers
completed verification; normal launch does not perform a fresh content hash.

Keep that process running. Model files live in `models/` by default; use
`--cache /path/to/cache` to choose persistent storage and `--offline` to require
all verified files to exist. The cache needs 32.8 GB (30.5 GiB) for the selected
weights, plus space for metadata and a partial download. No application
executable is downloaded. The initial hardware target is a 24 GB NVIDIA GPU.

The source-built native developer path is explicit:

```sh
python3 scripts/check.py cuda-build
python3 scripts/run.py --native-binary build/cuda/dolittle
```

The build runs in Docker; the second command executes on the Linux host and
requires compatible CUDA runtime libraries. This path does not establish Docker
GPU support. Metal execution is planned, and AMD remains undecided.

The default listener is loopback. To use another machine's browser, either
forward the same port over SSH or explicitly opt into a trusted LAN:

```sh
python3 scripts/run.py --bind 0.0.0.0 --host my-linux-host --no-browser
```

Open the printed access link on that browser. The token is carried in its URL
fragment, removed from the address bar and retained in that tab's session
storage. The service checks Host/Origin and authenticates API requests. The
current transport is HTTP; use an SSH tunnel for connections outside a trusted
LAN. Do not expose this single-user service directly to the internet.

Enter a prompt and choose thinking/sampling settings. Target output streams into
thinking and answer sections. The prompt folds away during generation, and
thinking can be collapsed independently. A visible token budget covers both
thinking and the answer; reaching it before an answer is reported explicitly.
The default output budget is 4,096 tokens. Prompt (including the chat template)
plus requested output must fit within the 8,192-token target context; requests
that exceed it are rejected rather than silently shortened. Generation stops
at the model's end marker even if budget remains. Interpretation settings have
an independent 256-token output default within a 512-token AV context.
Click a text fragment to inspect its exact token positions; Shift-click extends
a range and arrow keys move between fragments. Numeric positions and variants
remain available under details. The selected excerpt stays beside the NLA
output, which streams while it is generated.
“Show prompt & control tokens” reveals the full input and delimiters. A window becomes separate
position jobs, never an averaged activation. The interface enables interpretation
after target generation finishes. Each interpretation has its own seed and cache; AV
weights remain resident between requests. Raw output remains visible when the
trained explanation tags are absent or incomplete.

Stop signals the active request and its remaining jobs. Browser disconnects do
not cancel work. Refresh replays the bounded journal without rerunning inference.
Starting a new experiment explicitly replaces the retained transcript and
captures. Restarting the service loses them; disk export is not implemented.

`client.js` owns transport, `controller.js` owns exact byte/token mapping and
state reconciliation, and `view.js` renders using text nodes. Model control-token
classification belongs to the C++ text protocol, not JavaScript. See the
[application architecture](../docs/architecture/live-application.md) for ownership,
limits, failure behavior and the future backend boundary.
