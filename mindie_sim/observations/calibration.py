"""Explicit bindings from historical profiler variants to primitive parameters.

[Role] Reuse retained observations through the current primitive identity.
[Invariants] A human-supplied binding declares hardware, implementation, shape
  and split. Filename resemblance and an old eligibility bit never bind a row.
[Timeouts & Locks] Bounded reads through ProfilerCatalogue; transactional store.
[Side-Effect Policy] Writes only the explicitly writable PrimitiveStore.
"""
from mindie_sim.prediction.primitives import PrimitiveKey
from .store import ProfilerCatalogue


def import_profiler(catalogue, store, bindings, *, max_samples=1000):
    imported = 0
    with ProfilerCatalogue(catalogue) as source:
        for binding in bindings:
            key = PrimitiveKey.from_mapping(binding["key"])
            rows = list(source.samples(binding["variant"], limit=max_samples + 1))
            if len(rows) > max_samples:
                raise ValueError("profiler_binding_exceeds_sample_budget:select_a_capture_subset")
            if not rows:
                raise ValueError("profiler_binding_has_no_raw_events")
            for row in rows:
                identity = row["source"] + ("::" + row["member"] if row["member"] else "")
                store.record(
                    key, duration_ms=row["duration_ms"],
                    run_id=binding["run_id"] + ":" + identity, repeat_index=row["row"],
                    source=f"{identity}#row={row['row']}", split=binding["split"])
                imported += 1
    return {"imported_samples": imported, "inventory": store.inventory()}
