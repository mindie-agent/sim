"""Synchronization, runtime-control, and communication L0 primitive lowerers.

[Role] Lower P12..P20 primitives (intracore/crosscore sync, async mutex, STARS,
  streams, AICPU, HCCL collective bands, direct peer windows, and host DMA).
[Invariants] Keep serialized delays separate from work and retain native units.
[Timeouts & Locks] Finite host-side lowering with no locks or device waits.
[Side-Effect Policy] Pure builders mutating only the caller-provided ``_Lowering``.
"""
from __future__ import annotations

from graphlib import TopologicalSorter

from mindie_sim.prediction.collectives.ports import RootComplexKey
from mindie_sim.prediction.collectives.transport import DEFAULT_MAX_PHYSICAL_CHUNKS
from mindie_sim.prediction.program.stages import PIPELINE_UNROLL_LIMIT, Stage
from mindie_sim.prediction.runtime.ring_buffer import RingBuffer, lower_generation
from mindie_sim.prediction.streams import apply_stream_owners_to_program, bind_stream_owners
from mindie_sim.schema.common import ModelingError, align_up
from mindie_sim.schema.device_input import StarsSpec

from .loader import _nonnegative_int as _int
from .schema import (
    _FramedCollectiveBand, _program_span_ms, _stars_plan_span_ms, frame_wire_bytes,
)

NS_TO_MS = 1e-6
US_TO_MS = 1e-3
MS_PER_SECOND = 1000.0
INTRACORE_EVENT_IDS = 8
INTRACORE_SYSTEM_EVENT_START = 6
INTER_BLOCK_IDS = {"aiv": 8, "aic": 9}
SUBBLOCK_BARRIER_ID = 10
PAIRED_EVENT_IDS = 16
INTRA_BLOCK_MODE = 4
INTRA_BLOCK_PAIRED_AIVS = 2
REVERSE_DEPTH = 15
HW_MUTEX_IDS = 32
CATLASS_MUTEX_BUDGET = 28
TOKEN_FRAME_PAYLOAD_BYTES = 480
TOKEN_FRAME_WIRE_BYTES = 512


def _ceil(value: int, unit: int) -> int:
    return (value + unit - 1) // unit


def _aicpu(b):
    host = b.delay("host_select", "host_select_ns")
    mode = b.get("mode")
    kernel_ns = b.value("aicpu_base_ns", mode=mode) + b.integer("batch", 1) * (
        b.value("aicpu_per_batch_ns", 0, mode=mode)
        + b.integer("candidates") * b.value("aicpu_per_candidate_ns", 0, mode=mode))
    kernel = b.delay("aicpu_kernel", after=(host,), ms=kernel_ns * NS_TO_MS)
    pub = b.delay("gm_pub", "gm_pub_ns", (kernel,))
    if b.get("eager_readback", True) and not b.get("aclgraph_cached", False):
        b.delay("eager_readback", "eager_d2h_sync_ns", (pub,))


def _collective(b):
    startup = b.value("startup_ns") * NS_TO_MS
    params = dict(b.cell.parameters)
    if b.arch35 and (params.get("payload_bytes_per_frame"), params.get("wire_bytes_per_frame")) == (
        TOKEN_FRAME_PAYLOAD_BYTES, TOKEN_FRAME_WIRE_BYTES,
    ):
        params.pop("payload_bytes_per_frame")
        params.pop("wire_bytes_per_frame")
    band = _FramedCollectiveBand.from_mapping({**params, **b.p, "startup_us": startup / US_TO_MS,
                                             "evidence_id": b.cell.evidence_id or "undeclared"})
    b.extra["collective_band"] = band
    rate = b.speed("link_bytes_per_s")
    payload = b.integer("payload_bytes", b.get("bytes", 0))
    if not band.matches_payload(payload) or not band.group_min <= b.integer("group_size", band.group_min, 1) <= band.group_max:
        b.gap("collective_outside_band")
    after = (b.delay("collective_startup", ms=startup),) if band.startup_scope == "collective" else ()
    wire_total = frames_total = 0
    for phase, peers in enumerate(b.get("phases", (b.get("peer_payloads", {"link": payload}),))):
        ready = (b.delay(f"phase:{phase}:startup", ms=startup, after=after),) if band.startup_scope == "phase" else after
        done = []
        for peer, amount in peers.items():
            name = f"phase:{phase}:peer:{peer}"
            dep = (b.delay(f"{name}:startup", ms=startup, after=ready),) if band.startup_scope == "hop" else ready
            wire, frames = band.wire_bytes_for(amount)
            wire_total, frames_total = wire_total + wire, frames_total + frames
            row = b.get("peer_resources", {}).get(peer, str(peer) if peer == "link" else f"link:{b.integer('die')}->{peer}")
            credit = band.credit_bytes or wire or 1
            if _ceil(wire, credit) > DEFAULT_MAX_PHYSICAL_CHUNKS:
                b.gap("collective_credit_window_too_small")
            for chunk, start in enumerate(range(0, wire, credit)):
                dep = (b.work(f"{name}:{chunk}", ((row, min(credit, wire - start), rate),), dep),)
            done.extend(dep)
        after = tuple(dict.fromkeys(done)) or ready
    b.meta.update(wire_bytes=wire_total, frames=frames_total, short_final_frame_policy=band.short_final_frame_policy)


def _peer(b):
    tokens = tuple(_int(size, "token_payload_bytes") for size in b.get("token_payload_bytes", ()))
    rank, target = b.integer("die"), b.integer("peer", 1)
    if target == rank:
        raise ModelingError("primitive_peer_self_invalid")
    explicit_edges, dp_edges = b.get("peer_edges"), (b.dp.ub_peer_edges if b.dp is not None else None)
    if explicit_edges is not None:
        edges = tuple((int(u), int(v)) for u, v in explicit_edges)
        if dp_edges is not None and not set(edges).issubset(set(dp_edges)):
            b.gap("direct_peer_edge_required")
    else:
        edges = dp_edges or ()
    if (dp_edges is not None or explicit_edges is not None) and (rank, target) not in edges:
        b.gap("direct_peer_edge_required")
    transport = b.p.get("transport", b.get("transport", "a5_ub_shmem_mte") if b.arch35 else
                        "a3_sio_hccs" if b.table.sku.startswith("Ascend910_") else "a2_hccs")
    if b.arch35 and transport == "a5_ub_shmem_mte":
        payload_frame = b.integer("token_payload_bytes_per_frame", TOKEN_FRAME_PAYLOAD_BYTES, 1)
        wire_frame = b.integer("token_wire_bytes_per_frame", TOKEN_FRAME_WIRE_BYTES, 1)
        policy = b.get("short_final_frame_policy", "pad_full_frame")
        wire = sum(frame_wire_bytes(size, payload_frame, wire_frame, policy)[0] for size in tokens)
    elif b.arch35 and "ccu" in transport.lower():
        wire = sum(align_up(size, TOKEN_FRAME_WIRE_BYTES) for size in tokens)
    else:
        wire = sum(tokens)
    status = b.integer("status_bytes")

    def token_delay(name, metric, after=(), **kwargs):
        return b.delay(name, after=after, ms=len(tokens) * b.value(metric, 0) * NS_TO_MS, **kwargs)

    frame = token_delay("per_token_frame", "frame_ns", (token_delay("pack", "pack_ns"),),
                        metadata={"wire_bytes": wire})
    rate = b.speed("peer_bytes_per_s")
    rows = (b.get("peer_resource", f"peer:{rank}->{target}"), b.get("endpoint_resource", f"peer_egress:d{rank}"))
    endpoint_rate = b.speed("endpoint_bytes_per_s", default=rate * MS_PER_SECOND * b.integer("fanout", max(1, sum(src == rank for src, _ in edges)), 1))
    transfer = b.work("transfer", ((rows[0], wire, rate), (rows[1], wire, endpoint_rate)), (frame,))
    sent = b.work("status", ((rows[0], status, rate), (rows[1], status, endpoint_rate)), (transfer,))
    token_delay("unpack", "unpack_ns", (token_delay("status_sync", "status_sync_ns", (sent,)),))
    b.meta.update(wire_bytes=wire + status, token_wire_bytes=wire, transport=transport)


def _dma(b):
    direction = b.get("direction", "d2h")
    namespace, die = b.get("namespace", ""), b.integer("die")
    root = RootComplexKey(b.integer("node"), b.get("root_complex_id", "0"), direction)
    rate = b.speed(f"host_{direction}_bytes_per_s")
    shared = b.speed(f"host_rc_{direction}_bytes_per_s", default=rate * MS_PER_SECOND)
    pinned = b.get("pinned", b.get("memory_kind", "pinned") == "pinned")
    setup = b.delay("dma_startup", "pinned_dma_startup_ns" if pinned else "pageable_dma_startup_ns")
    amount = b.integer("payload_bytes", b.get("bytes", 0))
    done = b.work("dma", ((f"{namespace}host_solo:{direction}:d{die}", amount, rate),
                          (root.resource_name(namespace), amount, shared)), (setup,))
    if direction == "d2h" and b.get("eager_readback", False):
        b.delay("eager_d2h_sync", "eager_d2h_sync_ns", (done,))


def _single_l0c(b):
    if b.get("enable_unit_flag", False) and b.integer("l0c_stages", b.get("l0c", 1), 1) != 1:
        raise ModelingError("primitive_unit_flag_requires_single_l0c_stage")


def _intracore(b):
    _single_l0c(b)
    event = b.integer("event_id")
    if event >= INTRACORE_EVENT_IDS:
        raise ModelingError("primitive_intracore_event_id_invalid")
    if event >= INTRACORE_SYSTEM_EVENT_START:
        b.warnings.append(f"primitive_intracore_event_id_system_reserved:{event}")
    unitflag = b.get("enable_unit_flag", False)
    src_pipe, dst_pipe = b.get("src_pipe"), b.get("dst_pipe")
    if (src_pipe is None) != (dst_pipe is None):
        raise ModelingError("primitive_intracore_requires_both_pipes")
    meta = {"event_id": event, "unit_flag": bool(unitflag)}
    if src_pipe is not None:
        meta.update(src_pipe=src_pipe, dst_pipe=dst_pipe)
    b.meta.update(meta)
    metric_base = "unitflag" if unitflag else "flag"
    edge = b.get("edge")
    if edge and edge not in b.cell.modes and f"{metric_base}_{edge}_ns" in b.cell.metrics:
        b.delay(metric_base, f"{metric_base}_{edge}_ns", metadata=meta)
    else:
        b.delay(metric_base, f"{metric_base}_ns", mode=edge, metadata=meta)


def _crosscore(b):
    mode = b.integer("mode", 2)
    kind = b.get("core_kind", "aiv" if mode == 1 else "aic")
    if mode not in (0, 1, 2, INTRA_BLOCK_MODE) or kind not in INTER_BLOCK_IDS or (mode == 1 and kind == "aic"):
        raise ModelingError("primitive_crosscore_scope_invalid")
    if mode == INTRA_BLOCK_MODE and not b.arch35:
        b.gap("crosscore_mode4_requires_dav_3510")
    key = f"mode0_{kind}_barrier" if mode == 0 else {
        1: "mode1_subblock", 2: "mode2_paired", INTRA_BLOCK_MODE: "mode4_intra_block",
    }[mode]
    count = b.integer("core_count", 1, 1)
    if mode == 0 and count > getattr(b.dp or b.table.provenance.silicon_fingerprint, f"{kind}_count"):
        raise ModelingError("primitive_crosscore_count_invalid")
    metric = f"{key}_{count}_ns" if mode == 0 and f"{key}_{count}_ns" in b.cell.metrics else f"{key}_ns"
    subblock = b.get("aiv_subblock")
    first_event = INTER_BLOCK_IDS[kind] if mode == 0 else SUBBLOCK_BARRIER_ID if mode == 1 else 0
    if mode == INTRA_BLOCK_MODE and subblock is not None:
        subblock = b.integer("aiv_subblock")
        if subblock >= INTRA_BLOCK_PAIRED_AIVS:
            raise ModelingError("primitive_crosscore_subblock_invalid")
        if kind == "aic":
            first_event = subblock * PAIRED_EVENT_IDS
    event = b.integer("event_id", first_event)
    reverse, first = b.get("with_reverse", False), b.integer("generation")
    reverse_id = b.integer("reverse_id", event ^ 1 if mode == INTRA_BLOCK_MODE else 1)
    limit = PAIRED_EVENT_IDS * INTRA_BLOCK_PAIRED_AIVS if mode == INTRA_BLOCK_MODE and kind == "aic" else PAIRED_EVENT_IDS
    if event >= limit or (reverse and mode in (2, INTRA_BLOCK_MODE) and (reverse_id == event or reverse_id >= limit)):
        raise ModelingError("primitive_crosscore_event_id_invalid")
    if mode == INTRA_BLOCK_MODE and kind == "aic":
        event_subblock = event // PAIRED_EVENT_IDS
        if ((subblock is not None and subblock != event_subblock)
                or (reverse and reverse_id // PAIRED_EVENT_IDS != event_subblock)):
            raise ModelingError("primitive_crosscore_subblock_invalid")
        subblock = event_subblock
    previous, credit = (), {}
    for index in range(first, first + b.integer("notifications", 1, 1)):
        group = max(0, index // REVERSE_DEPTH - 1)
        deps = previous + ((credit[group],) if reverse and index >= REVERSE_DEPTH and group in credit else ())
        meta = {"mode": mode, "scope": kind if mode == 0 else key, "core_count": count, "event_id": event}
        if mode == INTRA_BLOCK_MODE:
            meta.update(core_kind=kind, local_event_id=event % PAIRED_EVENT_IDS,
                        pipe=b.get("pipe", "PIPE_S"))
            if subblock is not None:
                meta["aiv_subblock"] = subblock
        latency_default = 0 if mode == INTRA_BLOCK_MODE else None
        previous = (b.delay(f"crosscore:{index}", after=deps, metadata=meta,
                            ms=b.value(metric, latency_default) * NS_TO_MS),)
        if reverse and (index + 1) % REVERSE_DEPTH == 0:
            reverse_meta = {**meta, "event_id": reverse_id}
            if mode == INTRA_BLOCK_MODE:
                reverse_meta["local_event_id"] = reverse_id % PAIRED_EVENT_IDS
            credit[index // REVERSE_DEPTH] = b.delay(f"reverse_credit:{index}", after=previous,
                                                     ms=b.value("reverse_ns", b.value(metric, latency_default)) * NS_TO_MS,
                                                     metadata=reverse_meta)
    b.meta.update(event_id=event, reverse_depth=REVERSE_DEPTH if reverse else 0)


def _mutex(b):
    if not b.arch35:
        b.gap("async_mutex_requires_dav_3510")
    _single_l0c(b)
    pools = {"l1a": "l1", "l1b": "l1", "l0a": "l0a", "l0b": "l0b", "l0c": "l0c"}
    depths = {name: b.integer(f"{name}_stages", b.get(name, 1), 1) for name in pools}
    total = sum(depths.values()) + 2 * bool(b.get("has_bias", False))
    frontend = b.get("frontend", "catlass")
    if frontend not in {"catlass", "hardware"}:
        raise ModelingError("primitive_mutex_frontend_invalid")
    limit = CATLASS_MUTEX_BUDGET if frontend == "catlass" else HW_MUTEX_IDS
    if total > limit:
        raise ModelingError("primitive_mutex_id_limit")
    owner = b.owner("aic")
    rings = {name: RingBuffer(allocation=name, pool=b.ctx.pool_names.get(pool, pool) if b.ctx else pool,
                              slot_bytes=b.integer(f"{name}_slot_bytes"), depth=depths[name],
                              die=b.integer("die"), owner=owner) for name, pool in pools.items()}
    steps = b.get("steps")
    count = len(steps) if steps is not None else b.integer("k_steps", 1, 1)
    if count > PIPELINE_UNROLL_LIMIT:
        raise ModelingError("primitive_mutex_window_too_large")
    tau, resident, hits = b.value("mutex_overhead_ns") * NS_TO_MS, {}, 0
    durations = {}
    for index, step in enumerate(steps if steps is not None else ({},) * count):
        events = {}
        for name, ring in rings.items():
            entry = step.get(name, {})
            slot = (name, ring.slot(index))
            identity = entry.get("addr"), entry.get("coord")
            reuse = name.startswith("l1") and None not in identity and resident.get(slot) == identity
            resident[slot] = identity
            hits += reuse
            work = {} if reuse else entry.get("work", {})
            dependency = {"l0a": ("l1a",), "l0b": ("l1b",), "l0c": ("l0a", "l0b")}.get(name, ())
            downstream = {"l1a": "l0a", "l1b": "l0b", "l0a": "l0c", "l0b": "l0c"}.get(name)
            wait = (f"{downstream}:g{index}:write:produce",) if downstream else ()
            events[name] = lower_generation(
                ring, index, "produce", ("consume",), producer_work=work,
                producer_placement={key: (owner,) for key in work}, visibility_latency_ms=tau,
                producer_extra_after=tuple(events[dep].publish for dep in dependency),
                consumer_extra_after={"consume": wait},
            )
            durations[events[name].write] = 0 if reuse else entry.get("duration_ms", 0)
    nodes = {stage.name: stage for ring in rings.values() for stage in ring.to_stages()}
    for name in TopologicalSorter({name: stage.after for name, stage in nodes.items()}).static_order():
        b.append(nodes[name], durations.get(name))
    b.extra["ring_buffers"] = tuple(rings.values())
    b.meta.update(mutex_ids=total, l1_reuse_hits=hits)


def _stars(b):
    declared = b.dp.stars_spec if b.dp is not None else None
    first = b.value(f"{b.get('owner_kind', 'aic')}_first_issue_ns", declared.first_issue_us * US_TO_MS / NS_TO_MS if declared else 0)
    interval = b.value("issue_interval_ns", declared.issue_interval_us * US_TO_MS / NS_TO_MS if declared else 0)
    spec = StarsSpec(b.get("queue_scope", declared.queue_scope if declared else "die"),
                     b.integer("queue_depth", declared.queue_depth if declared else 1, 1),
                     b.integer("lanes", declared.lanes if declared else 1, 1),
                     interval * NS_TO_MS / US_TO_MS, first * NS_TO_MS / US_TO_MS,
                     b.get("release_policy", declared.release_policy if declared else "descriptor_ready"),
                     b.cell.evidence_id or (declared.evidence_id if declared else "undeclared"))
    b.extra["stars_spec"] = spec
    die, ns = b.integer("die"), str(b.get("namespace", ""))
    default_host_ms = b.value("host_submit_ns", 0) * NS_TO_MS
    raw_descs = b.get("descriptors", 1)
    descs = (
        tuple((0, str(b.get("stream", "main")), default_host_ms, {}) for _ in range(max(1, raw_descs)))
        if isinstance(raw_descs, int) else
        tuple((int(d.get("replay", 0)), str(d.get("stream", "main")),
               float(d.get("host_submit_ns", default_host_ms / NS_TO_MS)) * NS_TO_MS,
               dict(d.get("work", {}))) for d in raw_descs)
    )
    b.stars_plan = (die, ns, descs)
    for _, _, _, work in descs:
        for row in work:
            if b.ctx is None or row not in b.ctx.rates:
                b.req_res.setdefault(row, (row, 1.0, "work", 1))
    b.delay("descriptor_issue", ms=_stars_plan_span_ms(spec, b.stars_plan, ctx=b.ctx))


def _streams(b):
    mode = b.get("mode", "disjoint")
    if mode not in {"disjoint", "shared_cap"}:
        raise ModelingError("primitive_stream_isolation_invalid")
    plan, program = b.get("stream_plan"), b.get("program")
    need_limit = b.get("set_limit", plan is not None or (
        "set_limit_ns" in b.cell.metrics and (b.role == "dual" or b.cell.band("set_limit_ns").p50 is not None)))
    setup = (b.delay("set_limit", "set_limit_ns", metadata={"isolation": mode}),) if need_limit else ()
    after = (b.delay("stream_event", "stream_event_ns", setup),)
    if plan is not None:
        if b.dp is None:
            b.gap("stream_datapath_required")
        owners = bind_stream_owners(plan, b.dp)[b.get("stream_id", "main")]
        b.meta.update(aic_owners=owners.aic, aiv_owners=owners.aiv, aic_limit=owners.aic_limit, aiv_limit=owners.aiv_limit)
        if program is not None:
            ctx = b.ctx
            if ctx is None:
                b.gap("stream_recipe_context_required")
            program = apply_stream_owners_to_program(
                program, owners, b.dp, resource_names=ctx.resource_names, rates=ctx.rates,
                resource_units=ctx.resource_units, quota_resources=b.get("quota_resources"),
                stream_id=b.get("stream_id", "main"),
            )
    if program is not None:
        span_ms = b.p["work_ms"] if "work_ms" in b.p else _program_span_ms(program, b.ctx)
        b.append(Stage("stream_work", after=after, program=program), span_ms)
    b.meta["isolation"] = mode
