"""Executable resource stages for BF16 LightningIndexerV2.

[Role] Turn the access ledger into weighted reduction and TopK stages.
[Invariants] Stages conserve per-owner work and keep allocations live across
  their consumers. Exchange reads release the shared slot for reuse.
[Timeouts & Locks] Work scales with active tiles and owners; no I/O or locks.
[Side-Effect Policy] Returns immutable virtual resource programs and plans.
"""
from mindie_sim.schema.common import ModelingError, dtype_bytes

from ._bf16_access import (
    QUERY_TILE_ROWS, SCORE_ELEMENT_BYTES, _key_access_plan,
    _panel_selection_ops, _tile_sentinel, _weighted_vector_ops,
)
from ...mechanism import OperatorPlan, Work
from mindie_sim.prediction.operators.vector.mixed_core import read_role
from ...ownership import unit_group
from ...program import KernelProgram, Stage


def _split_loads(geometry):
    return "mte2_aic" in geometry.resource_units


def _resident_allocations(context, access):
    """Actual Q/K residency allocations, pinned to their physical owner."""
    allocations = []
    owner = int(access["owner"])
    capacity = max(1, int(getattr(context, "core_count", 1) or 1))
    if owner >= capacity:
        raise ModelingError("bf16_lightning_indexer_cache_owner_out_of_range")
    names = getattr(context, "pool_names", {})
    units = getattr(context, "pool_units", {})
    capacities = getattr(context, "pool_capacities", {})
    if "l1" in names:
        l1_units = unit_group(owner, capacity, units.get("l1", capacity))
        if access["q_allocation_bytes"]:
            allocations.append(context.allocate(
                "index_q_l1", "l1", int(access["q_allocation_bytes"]), l1_units))
        if access["k_allocation_bytes"]:
            allocations.append(context.allocate(
                "index_k_l1", "l1", int(access["k_allocation_bytes"]), l1_units))
    if access["l2_allocation_bytes"] and "l2" in names:
        size = int(access["l2_allocation_bytes"])
        if "l2" in capacities:
            size = min(size, int(capacities["l2"]))
        allocations.append(context.allocate("index_k_l2", "l2", size, (0,)))
    if access["exchange_gm_bytes"] and not access["exchange_grant_reused"]:
        gm_pool = ("gm" if "gm" in names else
                   next((name for name in names
                         if "workspace" in str(name)), None))
        if gm_pool is not None:
            allocations.append(context.allocate(
                "index_exchange_gm", gm_pool,
                int(access["exchange_gm_bytes"]), (0,)))
    if access["workspace_bytes"] and "ub" in names:
        aivs = int(access["aiv_units"])
        for sub in range(aivs):
            unit = (owner * aivs + sub,)
            if access["score_bytes"]:
                allocations.append(context.allocate(
                    f"index_score_ub_{sub}", "ub",
                    int(access["score_bytes"]), unit))
            if access["list_bytes"]:
                allocations.append(context.allocate(
                    f"index_lists_ub_{sub}", "ub",
                    int(access["list_bytes"]), unit))
            if access["exchange_receive_additional_bytes"]:
                allocations.append(context.allocate(
                    f"index_receive_ub_{sub}", "ub",
                    int(access["exchange_receive_additional_bytes"]), unit))
    return tuple(allocations)


def _placement(context, owner, roles, *, core_count, aiv_per_aic=1, aiv=False):
    method = getattr(context, "owned", None)
    if method is None:
        return {}
    if not aiv:
        return method(owner, core_count, roles)
    owners = core_count * aiv_per_aic
    merged = {}
    for sub in range(aiv_per_aic):
        part = method(owner * aiv_per_aic + sub, owners, roles)
        for name, units in part.items():
            merged.setdefault(name, [])
            merged[name].extend(units)
    return {name: tuple(units) for name, units in merged.items()}


def _build(op, geometry):
    metadata = dict(op.metadata)
    ledger = metadata.get("lightning_indexer_work")
    if (ledger is None or op.dtype != "bf16"
            or dict(op.matrix_flops).keys() != {"bf16"}
            or op.layout != "TND/PA_BBND" or "native_tensor_view" not in metadata):
        raise ModelingError("bf16_lightning_indexer_native_contract_required")
    spans = tuple(tuple(span) for span in ledger["sequence_partition"])
    compression = int(ledger["compression"])
    topk = int(ledger["topk"])
    heads, width, kv_heads = int(ledger["heads"]), int(ledger["head_dim"]), int(ledger["kv_heads"])
    owner = int(ledger.get("cache_owner", 0))
    core_count = max(1, int(geometry.compute_units or 1))
    if owner >= core_count:
        raise ModelingError("bf16_lightning_indexer_cache_owner_out_of_range")
    vector_units = int(geometry.resource_units.get("vector", 2 * core_count))
    aiv_per_aic = max(1, vector_units // core_count)
    key_row_bytes = int(kv_heads * width * dtype_bytes("bf16"))
    q_row_bytes = int(heads * width * dtype_bytes("bf16"))
    l1_capacity = int(geometry.buffers.get("l1", 0) or 0)
    declared_l2 = ledger.get("l2_capacity_bytes")
    l2_capacity = (min(int(declared_l2), int(geometry.buffers["l2"]))
                   if declared_l2 is not None and "l2" in geometry.buffers
                   else None)
    gm_capacity = (int(geometry.buffers["gm"])
                   if "gm" in geometry.buffers else None)
    access = _key_access_plan(
        spans, compression=compression, key_row_bytes=key_row_bytes,
        q_row_bytes=q_row_bytes, q_tile_rows=QUERY_TILE_ROWS,
        topk=topk, kv_heads=kv_heads, aiv_units=aiv_per_aic,
        ub_capacity=int(geometry.buffers.get("ub", 0) or 0),
        l1_capacity=l1_capacity, l2_capacity=l2_capacity, owner=owner,
        identity_prefix=f"{op.op_id}|layer={op.layer}|stage={op.stage}|phase={op.phase}",
        gm_capacity=gm_capacity, exchange_grant=ledger.get("exchange_grant"))
    exchange_bytes = sum(tile["exchange_bytes"] for tile in access["tiles"])
    merge_ops = sum(tile["merge_ops"] for tile in access["tiles"])
    lists = sum(tile["lists"] * (tile["q_end"] - tile["q_start"])
                for tile in access["tiles"]) * kv_heads
    comparisons = 0.0
    sentinel = 0.0
    for tile in access["tiles"]:
        for panel in tile["panels"]:
            comparisons += _panel_selection_ops(
                tile, panel, compression, topk, kv_heads)
        comparisons += tile["panel_merge_ops"]
        sentinel += _tile_sentinel(tile, compression, topk, kv_heads)
    weight_cast_ops = (int(ledger["query_rows"]) * heads
                       if ledger["weight_dtype"] != "fp32" else 0)
    weighted = float(ledger["weighted_head_reduce_ops"]) + weight_cast_ops
    vector = weighted + comparisons + sentinel + merge_ops
    query = int(access["q_bytes"])
    metadata_bytes = int(ledger["metadata_read_bytes"])
    weight = int(ledger["head_weight_bytes"])
    output = int(ledger["selection_output_bytes"])
    fixpipe = int(access["pairs"]) * heads * SCORE_ELEMENT_BYTES
    read = read_role(geometry.resource_units)
    split = read == "mte2_aic"
    if split:
        loads = {"mte2_aic": int(access["key_gm_bytes"]) + query + metadata_bytes,
                 "mte2_aiv": weight + exchange_bytes, "mte2": 0}
    else:
        loads = {"mte2": int(access["key_gm_bytes"]) + query + metadata_bytes
                 + weight + exchange_bytes}
    resources = {
        "hbm": int(access["key_hbm_bytes"]) + query + weight + metadata_bytes
        + output + 2 * exchange_bytes,
        "mte3": output + exchange_bytes, "vector": vector,
        "mte1": int(access["key_logical_bytes"]) + query,
        "fixpipe": fixpipe, **loads,
    }
    stage_defs, addresses, allocations = _index_stage_definitions(
        access, ledger, metadata_bytes, heads, q_row_bytes, key_row_bytes,
        compression, read, split, weight_cast_ops, kv_heads, width, topk)

    def program(context):
        return _lower_index_program(
            context, access, owner, aiv_per_aic, read, split, stage_defs, addresses)

    work = Work(dict(op.matrix_flops), resources)
    return OperatorPlan(work, program, _index_plan_metadata(
        access, geometry, ledger, owner, read, split, aiv_per_aic, l1_capacity,
        declared_l2, l2_capacity, query, allocations, lists, kv_heads, merge_ops,
        exchange_bytes))


def _index_stage_definitions(
    access, ledger, metadata_bytes, heads, q_row_bytes, key_row_bytes, compression, read, split,
    weight_cast_ops, kv_heads, width, topk,
):
    stage_defs = []
    addresses = {}
    sequence_spans = ledger.get("sequence_layout_spans", ())
    allocations = bool(access["l1_allocation_bytes"] or access["l2_allocation_bytes"]
                       or access["workspace_bytes"]
                       or (access["exchange_gm_bytes"]
                           and not access["exchange_grant_reused"]))
    reserve = "index_reserve" if allocations else None
    if reserve:
        stage_defs.append((reserve, {}, (), "acquire"))
    previous = reserve
    previous_exchange_read = None
    metadata_left = metadata_bytes
    weight_dtype = "fp32" if ledger["weight_dtype"] == "fp32" else "bf16"
    weight_element_bytes = int(dtype_bytes(weight_dtype))
    for tile_index, tile in enumerate(access["tiles"]):
        q_after = (previous,) if previous else ()
        weight_tile = tile["q_end"] - tile["q_start"]
        weight_tile_bytes = weight_tile * heads * weight_element_bytes
        metadata_tile = metadata_left if tile_index == 0 else 0
        metadata_left -= metadata_tile
        q_stage = f"query_load_{tile_index}"
        if sequence_spans:
            _, _, q_offset, kv_offset, _ = sequence_spans[tile["request"]]
            addresses[q_stage] = (q_offset + tile["q_start"]) * q_row_bytes
        q_work = {
            "hbm": tile["q_bytes"] + weight_tile_bytes + metadata_tile,
            read: tile["q_bytes"] + metadata_tile,
        }
        if split:
            q_work["mte2_aiv"] = weight_tile_bytes
        else:
            q_work["mte2"] = q_work.pop(read) + weight_tile_bytes
        stage_defs.append((q_stage, q_work, q_after, None))
        previous = q_stage
        for panel_index, panel in enumerate(tile["panels"]):
            load_name = f"key_load_{tile_index}_{panel_index}"
            if sequence_spans:
                addresses[load_name] = (kv_offset // compression + panel["key_start"]) * key_row_bytes
            stage_defs.append((load_name, {
                "hbm": panel["hbm_bytes"], read: panel["gm_bytes"],
            }, (previous,), None))
            qk_name = f"qk_weighted_reduce_{tile_index}_{panel_index}"
            qk_work = {
                "vector": _weighted_vector_ops(panel["pairs"], heads, kv_heads)
                + (weight_cast_ops if tile_index == 0 and panel_index == 0 else 0),
                "mte1": panel["logical_bytes"]
                + (tile["q_bytes"] if panel_index == 0 else 0),
                "fixpipe": panel["pairs"] * heads * SCORE_ELEMENT_BYTES,
                "matrix_bf16": 2 * panel["pairs"] * heads * width,
            }
            stage_defs.append((qk_name, qk_work,
                               (load_name, q_stage), None))
            update_name = f"topk_update_{tile_index}_{panel_index}"
            stage_defs.append((update_name, {
                "vector": _panel_selection_ops(
                    tile, panel, compression, topk, kv_heads),
            }, (qk_name,), None))
            previous = update_name
        exchange_tile = tile["exchange_bytes"]
        output_tile = tile["output_bytes"]
        if exchange_tile:
            write_name = f"exchange_write_{tile_index}"
            write_after = (previous,) if previous else ()
            if previous_exchange_read is not None:
                # One per-tile GM slot is reused; the next write waits for the
                # prior tile's read to complete.
                write_after = (*write_after, previous_exchange_read)
            stage_defs.append((write_name, {
                "hbm": exchange_tile, "mte3": exchange_tile,
            }, write_after, None))
            read_name = f"exchange_read_{tile_index}"
            read_work = {"hbm": exchange_tile}
            if split:
                read_work["mte2_aiv"] = exchange_tile
            else:
                read_work["mte2"] = exchange_tile
            stage_defs.append((read_name, read_work, (write_name,), None))
            previous = read_name
            previous_exchange_read = read_name
        merge_name = f"topk_merge_{tile_index}"
        stage_defs.append((merge_name, {
            "vector": _tile_sentinel(tile, compression, topk, kv_heads)
            + tile["merge_ops"] + tile["panel_merge_ops"],
        }, (previous,), None))
        previous = merge_name
        output_name = f"output_store_{tile_index}"
        stage_defs.append((output_name, {
            "hbm": output_tile, "mte3": output_tile,
        }, (previous,), None))
        previous = output_name
    release = "index_release" if allocations else None
    if release:
        stage_defs.append((release, {}, (previous,), "free"))
    return stage_defs, addresses, allocations


def _lower_index_program(context, access, owner, aiv_per_aic, read, split, stage_defs, addresses):
    names = getattr(context, "resource_names", {})
    if names and read not in names:
        raise ModelingError(f"bf16_lightning_indexer_requires_read_port:{read}")
    allocations = _resident_allocations(context, access)
    core = max(1, int(getattr(context, "core_count", 1) or 1))
    aic_roles = ["cube", "mte1", "fixpipe"]
    if read != "mte2_aiv":
        aic_roles.append(read)
    aiv_roles = ["vector", "mte3"]
    if split:
        aiv_roles.append("mte2_aiv")
    aic_place = _placement(context, owner, tuple(aic_roles), core_count=core)
    aiv_place = _placement(context, owner, tuple(aiv_roles),
                           core_count=core, aiv_per_aic=aiv_per_aic, aiv=True)
    generic_mte2_place = ({}
                          if split else
                          _placement(context, owner, ("mte2",),
                                     core_count=core))
    stages = []
    for name, amounts, after, action in stage_defs:
        values = dict(amounts)
        matrix = {"bf16": values.pop("matrix_bf16", 0)}
        work = Work({key: value for key, value in matrix.items() if value},
                    {key: value for key, value in values.items() if value})
        kwargs = {}
        if name in addresses:
            kwargs["metadata"] = {"gm_base_byte": addresses[name]}
        if action == "acquire":
            kwargs["acquire"] = allocations
        elif action == "free":
            kwargs["free"] = allocations
        if name.startswith("key_load"):
            kwargs["placement"] = aic_place
        elif name.startswith("query_load"):
            kwargs["placement"] = ({**aic_place, **aiv_place}
                                   if split else aic_place)
        elif name.startswith("qk_weighted_reduce"):
            kwargs["placement"] = {**aic_place, **aiv_place}
        elif name.startswith("exchange_write"):
            kwargs["placement"] = aiv_place
        elif name.startswith("exchange_read"):
            kwargs["placement"] = (aiv_place if split
                                   else generic_mte2_place)
        elif name.startswith("topk_update"):
            kwargs["placement"] = {**aiv_place, **generic_mte2_place}
        elif name.startswith(("topk_merge", "output_store")):
            kwargs["placement"] = aiv_place
        stages.append(Stage(name, context.bind(work), after=after, **kwargs))
    return KernelProgram(tuple(stages))


def _index_plan_metadata(
    access, geometry, ledger, owner, read, split, aiv_per_aic, l1_capacity, declared_l2,
    l2_capacity, query, allocations, lists, kv_heads, merge_ops, exchange_bytes,
):
    exchange_grant_name = (
        "none" if not exchange_bytes else
        access["exchange_grant"] or
        ("index_exchange_gm"
         if access["exchange_gm_bytes"] and "gm" in geometry.buffers
         else "external_gm_grant"))
    return {
        "execution_abstraction": "mechanism_phase_program",
        "native_dispatch_reconstructed": False,
        "internal_cache_reload_and_topk_exchange": "modeled",
        "mechanism": {
            "query_tile_rows": int(access["q_tile_rows"]),
            "cache_owner": owner,
            "read_port": read,
            "split_mte2": split,
            "aiv_per_aic": aiv_per_aic,
            "aiv_owner_units": tuple(
                owner * aiv_per_aic + sub for sub in range(aiv_per_aic)),
            "cache_levels": {
                "l1": {"capacity_bytes": l1_capacity,
                       "q_allocation_bytes": int(access["q_allocation_bytes"]),
                       "k_allocation_bytes": int(access["k_allocation_bytes"]),
                       "allocation_bytes": int(access["l1_allocation_bytes"]),
                       "policy": access["l1_policy"], "owner": owner},
                "l2": {"capacity_bytes": access["l2_capacity_bytes"],
                       "declared_capacity_bytes": declared_l2,
                       "allocation_bytes": int(access["l2_allocation_bytes"]),
                       "policy": access["l2_policy"],
                       "scope": ("engine_pool_shared" if l2_capacity is not None
                                 else "declared_policy_no_engine_pool")},
                "gm": {"capacity_bytes": access["exchange_gm_capacity_bytes"],
                       "slot_bytes": int(access["exchange_per_tile_bytes"]),
                       "allocation_bytes": int(access["exchange_gm_bytes"]),
                       "grant": (access["exchange_grant"] or
                                 ("index_exchange_gm"
                                  if access["exchange_gm_bytes"]
                                  else "external_gm_grant")),
                       "policy": "one_per_tile_slot_reused",
                       "scope": ("caller_grant" if access["exchange_grant_reused"]
                                 else "engine_pool_shared"
                                 if access["exchange_gm_bytes"]
                                 and "gm" in geometry.buffers
                                 else "declared_memory_grant")},
            },
            "key_access": {
                "logical_bytes": access["key_logical_bytes"],
                "gm_bytes": access["key_gm_bytes"],
                "hbm_bytes": access["key_hbm_bytes"],
                "q_bytes": query,
                "pairs": access["pairs"],
                "panels": access["panels"],
                "panel_rows": access["panel_rows"],
                "tiles": access["tiles"],
                "capacity_events": ("engine_acquire_free" if allocations
                                    else "none_no_resident_allocation"),
            },
            "key_transport_bytes": int(access["key_logical_bytes"]),
            "key_gm_bytes": int(access["key_gm_bytes"]),
            "key_hbm_bytes": int(access["key_hbm_bytes"]),
            "unique_key_bytes": int(ledger["unique_key_bytes"]),
            "key_hbm_reload_bytes": max(0, int(access["key_hbm_bytes"])
                                        - int(ledger["unique_key_bytes"])),
            "cache_capacity_bytes": l1_capacity,
            "l2_capacity_bytes": access["l2_capacity_bytes"],
            "l1_allocation_bytes": int(access["l1_allocation_bytes"]),
            "q_allocation_bytes": int(access["q_allocation_bytes"]),
            "k_allocation_bytes": int(access["k_allocation_bytes"]),
            "l2_allocation_bytes": int(access["l2_allocation_bytes"]),
            "local_topk_lists": lists,
            "topk_owner_units": tuple(
                owner * aiv_per_aic + sub for sub in range(aiv_per_aic)),
            "local_topk_exchange_kv_heads": kv_heads,
            "local_list_merge_ops": merge_ops,
            "exchange_bytes": exchange_bytes,
            "exchange_hbm_bytes": int(access["exchange_hbm_bytes"]),
            "exchange_storage": {
                "total_bytes": exchange_bytes,
                "per_tile_bytes": int(access["exchange_per_tile_bytes"]),
                "per_aiv_list_bytes": int(access["list_bytes"]),
                "per_aiv_write_bytes": (int(access["exchange_per_tile_bytes"])
                                        // max(1, aiv_per_aic)),
                "receive_bytes": int(access["exchange_receive_bytes"]),
                "receive_additional_bytes": int(
                    access["exchange_receive_additional_bytes"]),
                "receive_alias": access["exchange_receive_alias"],
                "receive_chunks": int(access["exchange_receive_chunks"]),
                "allocations": tuple(
                    f"index_lists_ub_{sub}" for sub in range(aiv_per_aic)),
                "gm_allocation": exchange_grant_name,
                "write_stages": ("exchange_write_<tile>",),
                "read_stage": "exchange_read_<tile>",
                "merge_stage": "topk_merge_<tile>",
                "output_stage": "output_store_<tile>",
                "release_stage": ("index_release" if allocations else "none"),
                "grant": exchange_grant_name,
                "grant_reused": bool(access["exchange_grant_reused"]),
                "capacity_feedback": {
                    "pool": "gm",
                    "capacity_bytes": access["exchange_gm_capacity_bytes"],
                    "slot_bytes": int(access["exchange_per_tile_bytes"]),
                    "fits": bool(access["exchange_gm_fits"]),
                    "allocation_bytes": int(access["exchange_gm_bytes"]),
                    "waits_for_grant": not bool(access["exchange_gm_fits"]),
                    "pool_declared": "gm" in geometry.buffers,
                    "requires_memory_grant": bool(
                        access["exchange_gm_bytes"]
                        and not access["exchange_grant_reused"]),
                },
                "hbm_round_trip_per_tile": 2 * int(
                    access["exchange_per_tile_bytes"]),
                "hbm_round_trip_total": 2 * int(access["exchange_total_bytes"]),
                "reuse": ("one per-tile GM slot; each write waits for the "
                          "previous tile's read; grant released after the last "
                          "read/output at index_release"),
            },
            "workspace": {
                "score_bytes_per_aiv": int(access["score_bytes"]),
                "list_bytes_per_aiv": int(access["list_bytes"]),
                "exchange_receive_bytes_per_aiv": int(
                    access["exchange_receive_bytes"]),
                "exchange_receive_additional_bytes_per_aiv": int(
                    access["exchange_receive_additional_bytes"]),
                "exchange_receive_alias": access["exchange_receive_alias"],
                "exchange_receive_chunks": int(
                    access["exchange_receive_chunks"]),
                "resident_bytes_per_aiv": int(access["workspace_bytes"]),
                "full_scores_if_materialized": int(access["full_scores_bytes"]),
                "full_lists_if_materialized": int(access["full_lists_bytes"]),
                "pool": ("ub" if "ub" in geometry.buffers else "unbound"),
                "reuse": ("score slot reused per panel and as the exchange "
                          "receive when it fits; list slot per 16-row tile"),
            },
            "vector_owner_partitions": aiv_per_aic,
            "source_basis": ("native MIX_AIC M tile plus paired AIV subblocks; "
                             "Q/K panel residency and ordered K panels are "
                             "explicit mechanisms"),
        },
    }
