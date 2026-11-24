"""Physical cache extents for the P/D connector.

[Role] Project cache owners onto global component rows and packed local storage.
[Invariants] Every delivered row maps to one published source row.
[Timeouts & Locks] Finite interval sweeps, no locks.
[Side-Effect Policy] Address reservations live only for the current handoff.
"""
from mindie_sim.catalog.cache import index_k_block_rows
from mindie_sim.catalog.context_parallel import dcp_indexer_sharded
from mindie_sim.schema.common import ModelingError


def address(bases, extents, storage, plane, size, displacement, count):
    for label, val in (("size", size), ("displacement", displacement), ("count", count)):
        if isinstance(val, bool) or not isinstance(val, int):
            raise ModelingError(f"pd_transfer_address_requires_int:{label}:{val}")
    key = (storage, plane)
    if displacement < 0 or count < 0 or displacement + count > size:
        raise ModelingError("pd_transfer_outside_source_plane")
    if key not in bases:
        bases[key] = extents[storage]
        extents[storage] += size
    return bases[key] + displacement


def owned_runs(component, layer, model, parallel, rank, span, tokens, window_start,
               *, runtime=None):
    del component, layer, model, parallel, rank, span, tokens, window_start, runtime
    raise NotImplementedError(
        "LEGACY_PURGED: deleted truncated lo//ratio slicing; implement exact integer "
        "half-open [start, end) byte-interval partition (S6-W5-13)"
    )


def transfers_for_runs(sources, wanted):
    """Stream adjacent pieces. Replicas tie-break in placement order."""
    del sources, wanted
    raise NotImplementedError(
        "LEGACY_PURGED: deleted float size/rows row splitting; implement exact integer "
        "half-open [start, end) byte-interval partition (S6-W5-13)"
    )
