"""A5 FusedInferAttentionScore q=1 owner route and chunked phase split.

The capable non-quant host for ``qkHeadDim != 128`` is
``FusedInferAttentionScoreTilingImpl``.  ``SetIsIFA`` turns gs-merge on, and
``CheckS1OutSplit`` returns false on gs-merge or pfa-merge before the L2
compare.  ``s1 * group < 64`` keeps pfa-merge, which disables flash decode, so
``ApplySplitStrategy`` runs ``SplitNBSeq`` only.  One owner is one
``(batch, KV head)`` and walks every ``sInner=128`` tile.  The deployed tiling
key is not in the capture.  ``Block Num`` is not this owner count.

``attention_v1.forward_fused_infer_attention`` splits a mixed batch only when
the machine supports ``CHUNKED_PREFILL_PHASE_SPLIT`` (A5) and the state is
``ChunkedPrefill`` with both a decode subgroup and a prefill subgroup.
``build_attn_state`` calls a step whose sequence lengths equal the scheduled
tokens ``PrefillNoCache`` (no block table).  A pure fresh prefill stays one
launch.  A context-0 prefill that shares a chunked step with a decode is still
paged, because that call passes ``is_prefill_no_cache=False``.  Batch-invariant
mode is not the ordinary caller and does not add a launch.
"""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from typing import Any

from .expressions import as_fraction

_IFA_S_INNER = 128
_PFA_MERGE_GS = 64
_DECODE_Q = 1
_ARCH_SPLIT = "A5"
_KNOWN_ARCH = frozenset({"A5", "A2"})
_UNSELECTED = "splitnbseq_when_a5_q1_dk_ne_128_else_unselected"


def arch_family(ctx: Any) -> str:
    return str(getattr(ctx.supply, "arch_family", "") or "")


def _chunked(operator: Any) -> bool:
    return int(operator.theta.find("chunked_prefill", 1) or 0) != 0


def attention_table_mode(ctx, operator, geometry, static, *, mla, compressed,
                         paged_table):
    """Whether this launch's block table is resident, and whether it is q=1."""
    requests = geometry.requests
    page_on = (
        (not mla) and (not compressed)
        and not (static["kv_dtype"] == "int8" and static["topk"] > 0)
        and int(operator.theta.find("page_attention", 0) or 0) == 1
        and bool(requests))
    fresh = bool(requests) and all(int(request.context) == 0 for request in requests)
    all_q1 = bool(requests) and all(int(request.q_len) == 1 for request in requests)
    if paged_table == "on":
        charge = page_on
    elif paged_table == "off":
        charge = False
    elif arch_family(ctx) in _KNOWN_ARCH:
        charge = page_on and not fresh
    else:
        # Undeclared arch keeps the historical q=1-only table.  It is not A2.
        charge = page_on and all_q1
    return charge, charge and all_q1


def _state(requests, chunked: bool) -> str:
    if all(int(request.context) == 0 for request in requests):
        return "prefill_no_cache"
    if all(int(request.q_len) <= _DECODE_Q for request in requests):
        return "decode_only"
    if chunked:
        return "chunked_prefill"
    return "prefill_cache_hit"


def _child(geometry, requests):
    return replace(geometry, requests=tuple(requests), local_batch=len(requests))


def partition_launches(ctx, operator, geometry):
    """Source launches.  A5 chunked mixed is two; every other step is one."""
    requests = tuple(geometry.requests)
    state = _state(requests, _chunked(operator))
    arch = arch_family(ctx)
    if arch == _ARCH_SPLIT and state == "chunked_prefill" and requests[0].q_len <= _DECODE_Q:
        first = next(index for index, request in enumerate(requests)
                     if int(request.q_len) > _DECODE_Q)
        decodes, prefills = requests[:first], requests[first:]
        if decodes and prefills:
            return ((_child(geometry, decodes), "on"),
                    (_child(geometry, prefills), "on"))
    if state == "prefill_no_cache":
        table = "off"
    elif arch in _KNOWN_ARCH:
        table = "on"
    else:
        table = "auto"
    return ((geometry, table),)


def ifa_owner_tail(ctx, spans, static, splits, cube, schedule, vector,
                   burst, q_bytes, block_bytes, fixpipe, query_rows, out_bytes):
    """Serial four-slot ring on the longest SplitNBSeq owner."""
    n2, cores, group = static["kv_heads"], static["cores"], static["group"]
    weights = [-( -int(span) // _IFA_S_INNER) for span in spans]
    target = Fraction(sum(weights) * n2, cores)
    buckets: list[list[int]] = [[]]
    current, started, core = Fraction(0), False, 0
    for span in spans:
        full, tail = divmod(int(span), _IFA_S_INNER)
        rows = (_IFA_S_INNER,) * full + ((tail,) if tail else ())
        for _head in range(n2):
            dif = target * (core + 1) - current
            if len(rows) - dif > dif and started:
                buckets.append([])
                core += 1
            buckets[-1].extend(rows)
            current += len(rows)
            started = True
    longest = max(buckets, key=len)
    names = ("cube", "vector", "hbm", "mte1", "mte2", "mte3", "fixpipe")
    rates = {pipe: Fraction(str(ctx.supply.effective(pipe))) for pipe in names}

    def elapsed(pipe, work, denom=1):
        rate = rates[pipe]
        return Fraction(work) / (rate * denom) if rate > 0 else Fraction(0)

    tiles = len(longest)
    cqk, cv, dv = static["cube_qk"], static["cube_v"], static["dv"]
    cube_rate = rates["cube"] / cores
    vector_rate = rates["vector"] / cores
    vec2, pipeline = group * (3 * static["ub_v"] + 10), Fraction(0)
    for slot in range(tiles + 3):
        aic = 2 * group * longest[slot] * cqk if slot < tiles else 0
        if 2 <= slot < tiles + 2:
            aic += 2 * group * longest[slot - 2] * cv
        aiv = 7 * group * longest[slot - 1] if 1 <= slot < tiles + 1 else 0
        if slot >= 3:
            aiv += vec2
        pipeline += max(
            Fraction(aic) / cube_rate if cube_rate else Fraction(0),
            Fraction(aiv) / vector_rate if vector_rate else Fraction(0))
    from .families import _declared_cube_rate
    scale = _declared_cube_rate(ctx, {"cube": Fraction(1)})
    occupancy = Fraction(1) if schedule <= 0 or cube <= 0 else min(
        Fraction(1), Fraction(cube) / Fraction(schedule))
    partial = (splits - 1) * query_rows * (4 * dv + 8)
    dma = Fraction(burst) + Fraction(q_bytes) * splits
    main = max(
        elapsed("cube", cube, scale * occupancy), elapsed("vector", vector),
        elapsed("mte2", dma), elapsed("mte1", dma), elapsed("mte3", partial),
        elapsed("fixpipe", fixpipe), elapsed("hbm", dma + partial + block_bytes))
    merge_vec = (splits - 1) * query_rows * (3 * dv + 9) + query_rows * (dv + 1)
    merge = max(elapsed("mte2", partial), elapsed("mte3", out_bytes),
                elapsed("hbm", partial + out_bytes), elapsed("vector", merge_vec))
    return core + 1, tiles, pipeline, max(Fraction(0), pipeline - main - merge)


def _lanes(geometry) -> dict[str, int]:
    lanes = geometry.token_lanes()
    return {
        "LANE_SCHEDULED": int(lanes["scheduled_sequences"]),
        "LANE_GATHERED": int(lanes["gathered_tokens"]),
        "LANE_VALID": int(lanes["valid_tokens"]),
        "FIA_QUERY_TOKENS": sum(int(request.q_len) for request in geometry.requests),
    }


def _blank_launch() -> dict[str, Fraction]:
    keys = ("HBM_BYTES", "SPLIT_PARTIAL_BYTES", "OUTPUT_BYTES", "OWNERS",
            "ACTIVE_CORES", "KV_SPLITS", "BLOCK_BYTES", "SELECTED",
            "PIPELINE_MS", "EXPOSED_MS", "S2_TILES", "DURATION_MS")
    return {key: Fraction(0) for key in keys}


def _launch_fields(prefix: str, fields: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in fields.items()}


def schedule_attention(ctx, operator, geometry, static, mla, accum):
    """Price one launch.  The no-FD route uses one split and the owner count."""
    query_rows = accum["query_rows"]
    physical_rows = accum.get("physical_query_rows", query_rows)
    io_act = accum.get("io_act", 2)
    tasks = accum["tasks"]
    cube_total = accum["cube_total"]
    max_rows = accum["max_rows"]
    max_kv = accum["max_kv"]
    span_by_request = accum["span_by_request"]
    route_q1 = accum["route_q1"]
    window = accum["window"]
    spans = [int(span_by_request[request]) for request in geometry.requests
             if request in span_by_request] if route_q1 else []
    split_count = ctx.policy.split_policy().splits(max(1, max_rows * max_kv))
    no_ranges = not any(request.query_ranges for request in geometry.requests)
    route = (
        (not mla) and arch_family(ctx) == _ARCH_SPLIT and route_q1 and window == 0
        and bool(spans) and static["dk"] != 128 and static["dv"] > 128
        and static["group"] < _PFA_MERGE_GS and no_ranges)
    effective = 1 if (route or static.get("packed_sparse")) else split_count
    cube_schedule = Fraction(0)
    if tasks:
        total_tasks = effective * tasks
        waves = -(-int(total_tasks) // static["cores"])
        per_task = max(Fraction(cube_total),
                       Fraction(accum["task_cube_max"]) * tasks) / total_tasks
        cube_schedule = waves * per_task * static["cores"]
    if static.get("packed_sparse"):
        cube_schedule = Fraction(cube_total)
    cube_rate = (Fraction(1) if cube_schedule <= 0 or cube_total <= 0
                 else min(Fraction(1), Fraction(cube_total) / cube_schedule))
    host_used = host_tiles = 0
    pipeline = exposed = Fraction(0)
    q_bytes = accum["q_bytes"]
    weights = accum["weights"]
    burst = accum["burst"]
    block_bytes = accum["block_bytes"]
    out_bytes = accum["out_bytes"]
    if route:
        host_used, host_tiles, pipeline, exposed = ifa_owner_tail(
            ctx, spans, static, effective, cube_total, cube_schedule,
            accum["vector_total"], burst, q_bytes, block_bytes,
            accum["fixpipe"], physical_rows, out_bytes)
    dv = static["dv"]
    partial = (0 if static.get("packed_sparse") else
               (effective - 1) * physical_rows * (4 * dv + 8))
    dma = Fraction(burst) + (Fraction(q_bytes) + Fraction(weights)) * effective
    merge_vec = ((effective - 1) * physical_rows * (3 * dv + 9)
                 + physical_rows * (dv + 1))
    packed = (static.get("kv_dtype") in {"int8", "fp8", "fp8_e4m3"}
              and static.get("topk", 0) > 0)
    hbm = (Fraction(accum["packed_hbm"]) + partial + Fraction(out_bytes) if packed
           else dma + partial + Fraction(block_bytes) + partial + Fraction(out_bytes))
    priced_tasks = max(1, tasks)
    active = host_used if route else min(effective * priced_tasks, static["cores"])
    launch = _blank_launch()
    launch.update({
        "HBM_BYTES": Fraction(hbm),
        "SPLIT_PARTIAL_BYTES": Fraction(2 * partial),
        "OUTPUT_BYTES": Fraction(out_bytes),
        "OWNERS": Fraction(host_used),
        "ACTIVE_CORES": Fraction(active),
        "KV_SPLITS": Fraction(effective),
        "BLOCK_BYTES": Fraction(block_bytes),
        "SELECTED": Fraction(1 if route else 0),
        "PIPELINE_MS": pipeline,
        "EXPOSED_MS": exposed,
        "S2_TILES": Fraction(host_tiles),
    })
    fields = {
        "EFFECTIVE_SPLITS": effective,
        "CUBE_SCHEDULE_WORK": cube_schedule,
        "HW_RATE_CUBE": cube_rate,
        "DMA_BYTES": dma,
        "PARTIAL_BYTES": Fraction(partial),
        "MERGE_VECTOR_OPS": Fraction(merge_vec),
        "HBM_BYTES": Fraction(hbm),
        "ACTIVE_CORES": active,
        "HOST_USED_AIC": host_used,
        "HOST_S2_TILES": host_tiles,
        "INTRA_OWNER_PIPELINE_MS": pipeline,
        "INTRA_OWNER_EXPOSED_MS": exposed,
        "HW_DEPENDENCY_MS": exposed,
        "D256_HOST_SELECTED": 1 if route else 0,
        "FIA_LAUNCH_COUNT": 1,
        "FIA_SERIAL_MS": Fraction(0),
        "FIA_PHYSICAL_ROWS": Fraction(accum.get("io_rows", 0)),
        "FIA_VALID_QUERY_TOKENS": Fraction(accum.get("valid_query_tokens", 0)),
        "FIA_IO_DTYPE_BYTES": Fraction(io_act),
        **_lanes(geometry),
        **_launch_fields("LAUNCH0", launch),
        **_launch_fields("LAUNCH1", _blank_launch()),
    }
    return fields


_ADD = (
    "pairs", "PAIRS", "PAIRS_PHYSICAL", "query_rows", "QUERY_ROWS",
    "kv_sweeps", "kv_positions", "KV_ROWS", "tiles", "KV_TILES",
    "query_tiles", "QUERY_TASKS", "KV_BURST_BYTES", "KV_LOGICAL_BYTES",
    "PACKED_HBM", "PAGE_TXNS", "Q_BYTES", "CUBE_FLOPS", "VECTOR_OPS",
    "FIXPIPE_BYTES", "OUT_BYTES", "ABSORB_FLOPS", "EXPAND_FLOPS",
    "WEIGHT_BYTES", "LOGICAL_FLOPS", "SELECTED_ROWS", "COMPRESSED_NEW_ROWS",
    "BLOCK_TABLE_ENTRIES", "BLOCK_TABLE_BYTES", "BLOCK_TABLE_LOOKUPS",
    "DMA_BYTES", "PARTIAL_BYTES", "MERGE_VECTOR_OPS", "HBM_BYTES", "flops",
    "FIA_PHYSICAL_ROWS", "FIA_IO_DTYPE_BYTES",
)
_LAUNCH_SUFFIX = (
    "HBM_BYTES", "SPLIT_PARTIAL_BYTES", "OUTPUT_BYTES", "OWNERS",
    "ACTIVE_CORES", "KV_SPLITS", "BLOCK_BYTES", "SELECTED",
    "PIPELINE_MS", "EXPOSED_MS", "S2_TILES", "DURATION_MS",
)


def _axis_coords(geometry, group: int):
    max_rows, max_kv = 1, 1
    for request in geometry.requests:
        max_rows = max(max_rows, int(request.selected_q_len) * group)
        max_kv = max(max_kv, int(request.kv_len))
    return max(1, max_rows), max(1, max_kv), max(1, max_rows * max_kv)


def combine_launches(geometry, metas, durations):
    """Serial launches keep their own bytes.  The ledger is their sum."""
    combined = dict(metas[0])
    for key in _ADD:
        combined[key] = sum((meta.get(key, 0) for meta in metas), Fraction(0))
    combined["CUBE_TASK_MAX"] = max(meta.get("CUBE_TASK_MAX", 0) for meta in metas)
    rows, kv, work = _axis_coords(geometry, int(metas[0]["group"]))
    combined["query_rows_max"] = combined["QUERY_ROWS_MAX"] = rows
    combined["max_kv_len"] = combined["MAX_KV_LEN"] = kv
    combined["kv_work"] = work
    # Launch 0 is the q<=1 subgroup.  Top-level owner fields name that launch.
    first, second = metas
    for key in _LAUNCH_SUFFIX:
        if key == "DURATION_MS":
            combined["LAUNCH0_DURATION_MS"] = durations[0]
            combined["LAUNCH1_DURATION_MS"] = durations[1]
        else:
            combined[f"LAUNCH0_{key}"] = first.get(f"LAUNCH0_{key}", Fraction(0))
            combined[f"LAUNCH1_{key}"] = second.get(f"LAUNCH0_{key}", Fraction(0))
    combined["HOST_USED_AIC"] = int(combined["LAUNCH0_OWNERS"])
    combined["HOST_S2_TILES"] = int(combined["LAUNCH0_S2_TILES"])
    combined["INTRA_OWNER_PIPELINE_MS"] = combined["LAUNCH0_PIPELINE_MS"]
    combined["INTRA_OWNER_EXPOSED_MS"] = combined["LAUNCH0_EXPOSED_MS"]
    combined["ACTIVE_CORES"] = int(combined["LAUNCH0_ACTIVE_CORES"])
    combined["EFFECTIVE_SPLITS"] = int(combined["LAUNCH0_KV_SPLITS"])
    combined["D256_HOST_SELECTED"] = int(combined["LAUNCH0_SELECTED"])
    # Each child duration already applied its floor and its owner tail.
    combined["HW_DEPENDENCY_MS"] = Fraction(0)
    combined["HW_RATE_CUBE"] = Fraction(1)
    combined["CUBE_SCHEDULE_WORK"] = Fraction(0)
    combined["FIA_LAUNCH_COUNT"] = 2
    combined["FIA_SERIAL_MS"] = durations[0] + durations[1]
    combined.update(_lanes(geometry))
    return combined


def bind_fia_launches(ctx, operator, geometry, state, metrics):
    forced = getattr(ctx, "_qg9_force_table", None)
    if forced:
        ctx._qg9_serial = None
        return metrics(ctx, operator, geometry, mla=False, paged_table=forced)
    parts = partition_launches(ctx, operator, geometry)
    if len(parts) == 1:
        ctx._qg9_serial = None
        geo, table = parts[0]
        return metrics(ctx, operator, geo, mla=False, paged_table=table)
    metas = []
    durations = []
    busy: dict[str, Fraction] = {}
    ctx._qg9_force_table = "on"
    try:
        for geo, _table in parts:
            metas.append(metrics(ctx, operator, geo, mla=False, paged_table="on"))
            cost = ctx.evaluate(geo, state, readiness=False)
            durations.append(as_fraction(cost.duration_ms))
            for pipe, time in cost.times:
                busy[pipe] = busy.get(pipe, Fraction(0)) + as_fraction(time)
    finally:
        ctx._qg9_force_table = None
    ctx._qg9_serial = (sum(durations, Fraction(0)), busy)
    return combine_launches(geometry, metas, durations)


def fia_program_duration(ctx):
    def program_duration(_geometry, _state):
        serial = getattr(ctx, "_qg9_serial", None)
        if not serial:
            return None
        duration, busy = serial
        return (duration, "", (), busy)
    return program_duration


def attention_notes(ctx, operator, static) -> tuple[tuple[str, str], ...]:
    paged = int(operator.theta.find("page_attention", 0) or 0) == 1
    arch = arch_family(ctx)
    shape = (paged and static["dk"] != 128 and static["dv"] > 128
             and static["group"] < _PFA_MERGE_GS)
    if paged and static["dk"] == 128:
        selected, applicability = "0", "unselected_dk_128_other_template"
    elif shape and arch == _ARCH_SPLIT:
        selected = "splitnbseq_when_a5_q1_pfa_merge"
        applicability = "FusedInferAttentionScoreTilingImpl_conditional_tiling_key_not_in_capture"
    else:
        selected, applicability = "0", _UNSELECTED
    notes = [
        ("family", "attention"),
        ("branch", static["branch"] or "dense"),
        ("group", str(static["group"])),
        ("dk", str(static["dk"])), ("dv", str(static["dv"])),
        ("dk_cube", str(static["cube_qk"])),
        ("dv_cube", str(static["cube_v"])),
        ("window", str(int(operator.theta.find("window", 0) or 0))),
        ("causal", str(int(operator.theta.find("causal", 1)))),
        ("page", str(static["page"])),
        ("block_table_index", "int32_getvalue" if paged else "absent"),
        ("block_table_scalar_rate", "unavailable"),
        ("hbm_service", "full_chip_unscaled_by_block_num"),
        ("balance_semantics", "raw_pre_service_roofline_diagnostic"),
        ("timing_selector_coordinates", "RIDGE_EFFECTIVE_TOKENS,VEC_EFFECTIVE_TOKENS"),
        ("analytical_kv_splits", "not_a_tiling_key"),
        ("tiling_key", "not_in_capture"),
        ("host_used_aic_is_not_block_num", "1"),
        ("d256_host_selected", selected),
        ("d256_host_applicability", applicability),
        ("arch_family", arch or "undeclared"),
    ]
    if shape and arch == _ARCH_SPLIT:
        notes += [
            ("intra_owner_pipeline", "simplified_bmm2_two_loops_later"),
            ("bmm2_issue_delay", "2"),
            ("s1_out_split", "false_for_q1_gs_merge"),
            ("flash_decode", "false_when_s1_times_group_below_64"),
            ("reconstruct_template", "false_when_dk_ne_128"),
            ("l2_selector", "not_consulted_for_q1_gs_merge"),
        ]
    return tuple(notes)
