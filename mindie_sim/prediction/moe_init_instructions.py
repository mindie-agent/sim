"""Source macro and logical SIMT operation inventory for MoE initialization.

[Role] Separate known source counts from unobserved SDK and compiler expansion.
[Invariants] Sort blocks normalize one whole macro, never independent sorts.
  SIMT loads/stores count valid thread elements, not warps or cache misses.
[Timeouts & Locks] O(active AIVs) arithmetic; no I/O, locks or device work.
[Side-Effect Policy] Pure counts; only independently declared service is priced.
"""
from .instructions import PrimitiveInventory


def source_operations(assignments, per_owner, active, experts):
    blocks = (assignments + 31) // 32
    sort = {name: PrimitiveInventory({f"source_{name}32_blocks": blocks})
            for name in ("concat", "sort", "extract")}
    count, inverse = [], []
    for owner in range(active):
        rows = min(per_owner, assignments - owner * per_owner)
        # Full-range GATHER: both passes visit every valid element; each
        # adjacent read skips one boundary. No data-dependent warp is inferred.
        count.append(PrimitiveInventory({"source_simt_gm_load32": 4 * rows - 2}))
        inverse.append(PrimitiveInventory({"source_simt_gm_load32": rows,
                                           "source_simt_gm_store32": rows}))
    return sort, tuple(count), tuple(inverse)


def diagnostics(sort, count, inverse, *, experts, declared_roles):
    return {
        "sort": {key: item.to_dict(declared_roles) for key, item in sort.items()},
        "count_by_owner": [item.to_dict(declared_roles) for item in count],
        "inverse_by_owner": [item.to_dict(declared_roles) for item in inverse],
        "sort_domain": "one whole Concat/Sort<float,true>/Extract macro; input_blocks32 is normalization only",
        "sort_limitation": "cross-size merge cost is not established by a constant per-input-block rate",
        "simt_domain": "full-range dropless GATHER; logical INT32 thread accesses, not warp instructions or cache transactions",
        "source_launch_threads": 2048,
        "atomic_count": {"mechanism": "SetAtomicAdd<int32_t>; one DataCopyPad per active owner",
                         "payload_bytes_per_owner": experts * 4,
                         "shared_target": "all owners address the same expertCountTempGm array",
                         "unpriced": "128-byte transaction count/alignment and shared-line RMW serialization not inferred"},
        "excluded": ("SDK Sort merge expansion", "compiled SIMT warp/branch/address instructions",
                     "data-dependent expert-boundary UB accesses", "SIMT cache/coalescing policy"),
    }
