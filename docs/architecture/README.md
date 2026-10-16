# Architecture

The engine architecture is the next design task. Write these documents in order:

1. `logical-overview.md`: responsibilities, data flow, and system contracts.
2. `change-axes.md`: independent reasons the implementation will change.
3. `file-mapping.md`: ownership, dependencies, and mapping those axes to files.

Begin with the chosen model and adapter layouts, the 3090 memory budget, and
the capture/explanation scheduling requirements. Decide source subdivisions
from that design rather than creating speculative modules now.
