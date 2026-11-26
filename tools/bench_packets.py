"""Zero-logic CLI/import shim delegating to ``mindie_sim.observations.bench_packets``.

[Role] Preserve backward compatibility for external callers during Wave T2 cutover.
[Invariants] Zero duplicated packet logic; delegates 100% to packaged observations.
[Timeouts & Locks] None.
[Side-Effect Policy] Delegates to ``mindie_sim.observations.bench_packets``.
"""
from __future__ import annotations

from mindie_sim.observations.bench_packets import (  # noqa: F401
    ACCURACY_NAME, CALIBRATE_NAME, FABRIC_SAMPLE_SCHEMA, FABRIC_SUMMARY_SCHEMA,
    FIT_RATE_NAME, FIT_SERVICE_NAME, MANIFEST_NAME, OBSERVATION_V1_SCHEMA,
    OBSERVATION_V2_SCHEMA, SCHEMA, PacketError, assert_public_provenance,
    build_packets, is_public_provenance, iter_receipts, main, write_packets,
)

if __name__ == "__main__":
    raise SystemExit(main())
