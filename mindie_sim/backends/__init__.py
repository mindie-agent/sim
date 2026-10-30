"""[Role] Translate native hardware metadata into neutral resource declarations.
[Invariants] Importing this package performs no simulation or device work.
[Timeouts & Locks] No waits or locks.
[Side-Effect Policy] No I/O.
"""
from mindie_sim.schema.common import ModelingError
from .ascend import materialize as ascend

ADAPTERS = {"ascend": ascend}


def materialize(raw):
    backend = raw.get("backend")
    if not backend:
        return raw
    try:
        adapter = ADAPTERS[backend]
    except KeyError as exc:
        raise ModelingError(f"hardware_backend_unknown:{backend}:declare_neutral_resources") from exc
    return adapter(raw)
