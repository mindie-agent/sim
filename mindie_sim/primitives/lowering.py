"""Lower calibrated L0 primitives into the shared finite execution engine.

[Role] Bind primitive geometry, service bands, placement and causal dependencies.
[Invariants] Keep serialized delays separate from work and retain native units.
[Timeouts & Locks] Finite host-side lowering with no locks or device waits.
[Side-Effect Policy] Lowering previews memory state. Submission changes only the
  caller's engine and commits memory admission when its access becomes ready.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from graphlib import TopologicalSorter
from typing import Any

from mindie_sim.prediction.collectives.ports import RootComplexKey
from mindie_sim.prediction.collectives.transport import DEFAULT_MAX_PHYSICAL_CHUNKS
from mindie_sim.prediction.program.binding import RecipeContext
from mindie_sim.prediction.program.stages import PIPELINE_UNROLL_LIMIT, KernelProgram, Stage
from mindie_sim.prediction.runtime.l2_residency import AdmissionMemoryView, bind_memory_access
from mindie_sim.prediction.runtime.memory_physics import (
    CACHE_POLICY_DISABLE, CACHE_POLICY_NORMAL, DEFAULT_L2_GRANULE_BYTES, DEFAULT_SECTOR_BYTES,
    AddressRun, AddressRunPlan, BurstTurnRule, ConstraintTerm, CounterTerm,
    HbmBusSpec, MemoryAccessPattern, ReuseKey, VisibilityContract,
)
from mindie_sim.prediction.runtime.prepared_terms import VectorInstructionGroup, bind_vector_groups
from mindie_sim.prediction.runtime.ring_buffer import RingBuffer, lower_generation
from mindie_sim.prediction.runtime.tile_solver import DEFAULT_INST_BLOCK_DIM
from mindie_sim.prediction.streams import apply_stream_owners_to_program, bind_stream_owners
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError, align_up, dtype_bytes
from mindie_sim.schema.datapath import DatapathCapabilitySpec
from mindie_sim.schema.device_input import StarsSpec

from .loader import RateRole, _declared_mode, _nonnegative_int as _int, _optional_rate, _simulation_warnings
from .schema import (
    PRIMITIVE_IDS, PrimitiveLoweringResult, PrimitiveTable, ShortFramePolicy,
    _FramedCollectiveBand, _address_plan, frame_wire_bytes,
)

BYTES_PER_S_TO_MS = 1.0 / 1000.0
NS_TO_MS = 1e-6
US_TO_MS = 1e-3
MS_PER_SECOND = 1000.0
FRACTAL_BYTES = 512
FRACTAL_DIM = DEFAULT_INST_BLOCK_DIM
L0C_ACCUMULATOR_BYTES = 4
FIXPIPE_SPLIT_N_ELEMS = 32
PIPE_M_BARRIER_FRACTALS = 10
VECTOR_WIDTH_BYTES = 256
MX_SCALE_GROUP = 32
SORT_BLOCK_ELEMS = 32
SORTALL_LENGTHS = (64, 128, 256, 512, 1024, 2048)
MERGE_FAN_IN = 4
HISTOGRAM_TRUNK_ELEMS = 16384
HISTOGRAM_BINS = 256
SIMT_WARP_THREADS = 32
SIMT_THREAD_LIMITS = {"tla_dsl": 2048, "catlass": 1024}
INTRACORE_EVENT_IDS = 8
INTER_BLOCK_IDS = {"aiv": 8, "aic": 9}
SUBBLOCK_BARRIER_ID = 10
PAIRED_FLAG_MAX_ID = 7
REVERSE_DEPTH = 15
MAX_MUTEX_IDS = 28
TOKEN_FRAME_PAYLOAD_BYTES = 480
TOKEN_FRAME_WIRE_BYTES = 512
def _ceil(value: int, unit: int) -> int:
    return (value + unit - 1) // unit


def submit_lowered_primitive(engine, result, *, after=(), memory_state=None, pattern=None, name_prefix=""):
    """Submit the lowered DAG with live admission at its memory stage."""
    return result._start_lowered(engine, after, memory_state, pattern, name_prefix)


class _Lowering:
    def __init__(self, primitive_id, params, table, datapath, context, memory_state, role, allow):
        self.pid, self.p, self.table = primitive_id, params, table
        self.dp, self.ctx, self.state = datapath, context, memory_state
        self.role, self.allow = role, allow
        self.cell = table.cell(primitive_id)
        self.stages, self.times, self.meta, self.extra = [], {}, {}, {}
        self.warnings = list(table.warnings)
        self.live = None

    def gap(self, code):
        raise UnmodeledCapabilityError(capability=self.pid, entry="lower_primitive",
                                       code=code, message=f"{self.pid} requires {code}")

    def get(self, key, default=None):
        return self.p.get(key, self.cell.parameters.get(key, default))

    def integer(self, key, default=0, minimum=0):
        return _int(self.get(key, default), key, minimum)

    def value(self, metric, default=None, *, mode=None, pid=None):
        if mode is None and (pid is None or pid == self.pid):
            mode = _declared_mode(self.cell, self.p.get("mode"))
        value, warnings = _optional_rate(self.table, pid or self.pid, metric, mode, self.role, self.allow, default)
        self.warnings.extend(warnings)
        return value

    def speed(self, metric, **kwargs):
        rate = self.value(metric, **kwargs) * BYTES_PER_S_TO_MS
        if rate <= 0:
            self.gap(f"positive_rate_required:{metric}")
        return rate

    def owner(self, kind):
        owner = self.integer(f"{kind}_id", self.get("owner", 0))
        count = getattr(self.dp or self.table.provenance.silicon_fingerprint, f"{kind}_count")
        if owner >= count:
            raise ModelingError(f"primitive_owner_out_of_range:{kind}:{owner}")
        return owner

    def row(self, key):
        return key if self.ctx is None else self.ctx.resource_names.get(key, key)

    def append(self, stage, duration=None):
        ready = max((self.times[key] for key in stage.after), default=0.0)
        self.times[stage.name] = ready + (stage.latency if duration is None else duration)
        self.stages.append(stage)
        return stage.name

    def delay(self, name, metric=None, after=(), *, ms=None, metadata=None, mode=None):
        duration = self.value(metric, 0, mode=mode) * NS_TO_MS if ms is None else ms
        return self.append(Stage(name, after=after, latency=duration, keep_empty=True, metadata=metadata or {}))

    def work(self, name, rows, after=(), *, owner=None, metadata=None, scale_pipe=False):
        # Rows declare work and calibrated ceilings in the same native units.
        active = {self.row(role): (amount, rate) for role, amount, rate in rows if amount}
        if scale_pipe and self.ctx:
            for row, (amount, rate) in active.items():
                nominal = self.ctx.rates[row] / self.ctx.resource_units.get(row, self.ctx.resource_units.get("mte1", 1))
                active[row] = (amount * nominal / rate, nominal)
        return self.append(Stage(
            name, work={key: amount for key, (amount, _) in active.items()}, after=after,
            placement={key: (owner,) for key in active} if owner is not None else {},
            caps={key: rate for key, (_, rate) in active.items()}, coupled=len(active) > 1,
            metadata=metadata or {},
        ), max((amount / rate for amount, rate in active.values()), default=0.0))

    def route(self, src, dst, owner, *, flag=None):
        if self.dp is not None:
            self.dp.route(src, dst, self.get("rep", ""), self.get("owner_group", owner), self.get("tile"))
        elif flag is not None and not self.cell.parameters.get(flag, False):
            self.gap(f"datapath_route_unsupported:{src}->{dst}")

    def finish(self):
        stages = tuple(self.stages)
        result = PrimitiveLoweringResult(
            self.pid, stages, KernelProgram(stages), max(self.times.values(), default=0.0),
            warnings=tuple(dict.fromkeys(self.warnings)), metadata=tuple(self.meta.items()), **self.extra,
        )
        object.__setattr__(result, "_live_access", self.live)
        return result


def _memory(b, name, pipe, direction, owner, plan, copy_rate, after=(), *, atomic_rate=None, logical_size=None):
    die, version = b.integer("die"), b.integer("version")
    reuse = b.get("reuse_key")
    if isinstance(reuse, Mapping):
        reuse = ReuseKey(**reuse)
    visibility = b.get("visibility", VisibilityContract())
    if isinstance(visibility, Mapping):
        visibility = VisibilityContract(**visibility)
    logical = b.integer("logical_bytes", sum(run.rows * run.row_bytes for run in plan.runs) + plan.scale_bytes
                        if logical_size is None else logical_size)
    policy = CACHE_POLICY_DISABLE if atomic_rate else b.get("cache_policy", CACHE_POLICY_NORMAL)
    pattern = MemoryAccessPattern(
        b.get("tensor_id", name), version, die, owner, direction, pipe, logical, plan,
        reuse, b.get("reuse_distance_bytes"), visibility, policy,
    )
    read = b.speed("hbm_read_bytes_per_s", pid="P01_datacopy")
    write = b.speed("hbm_write_bytes_per_s", pid="P01_datacopy")
    alpha = b.value("hbm_rw_alpha", 0, pid="P01_datacopy")
    if atomic_rate is not None:
        read, write, alpha = min(read, atomic_rate), min(write, atomic_rate), 1.0
    hbm = b.row("hbm")
    bus = HbmBusSpec(die, read, write, alpha, *((
        f"{hbm}:r", f"{hbm}:w", f"{hbm}:ctr_r", f"{hbm}:ctr_w",
    ) if hbm != "hbm" else ()))
    l2 = b.speed("l2_die_bytes_per_s", pid="P05_l2_residency_reuse")
    rules = b.get("burst_turn_penalty", b.table.cell("P01_datacopy").parameters.get("burst_turn_penalty", ()))
    rules = () if b.role == "dual" else tuple(
        rule if isinstance(rule, BurstTurnRule) else
        BurstTurnRule(**{key: value for key, value in rule.items() if key != "domain"})
        if "min_row_bytes" in rule else
        BurstTurnRule(0, None, rule.get("min_stride_bytes", 0), rule.get("max_stride_bytes"),
                      rule["penalty_bytes"], rule.get("evidence_id", ""))
        for rule in rules
    )
    config = dict(hbm_bus=bus, burst_rules=rules, copy_hit_bytes_per_ms=copy_rate,
                  copy_miss_bytes_per_ms=b.get("copy_miss_bytes_per_ms", copy_rate), l2_bytes_per_ms=l2)
    if b.ctx is not None:
        bound = b.ctx.bind_access(pattern, b.state, **config)
    else:
        bound = bind_memory_access(pattern, b.state, **config, pipe_row_id=pipe, l2_row_id="l2")
    if atomic_rate is not None:
        ledger = bound.ledger
        service = ledger.dram_bus_transaction_bytes
        read_terms = bus.terms_for("read", service)
        terms = tuple(ConstraintTerm(term.row_id, term.ms_per_progress + next(
            (extra.ms_per_progress for extra in read_terms if extra.row_id == term.row_id), 0,
        )) for term in bound.constraints)
        bound = replace(bound, constraints=terms, deliveries=(*bound.deliveries, CounterTerm(bus.ctr_r, service)),
                        bind_config=None, ledger=replace(ledger, dram_sector_bytes=2 * ledger.dram_sector_bytes,
                                                       dram_bus_transaction_bytes=2 * service))
    # External visibility is resolved by the caller's ready events.
    b.append(replace(bound.to_stage(name), after=after),
             max((term.ms_per_progress for term in bound.constraints), default=0.0))
    b.live = (name, pattern, b.state, bus)
    b.extra["bound_access"] = bound
    b.meta.update(logical_bytes=logical, aligned_transfer_bytes=bound.ledger.aligned_transfer_bytes,
                  dram_sector_bytes=bound.ledger.dram_sector_bytes,
                  dram_service_bytes=bound.ledger.dram_bus_transaction_bytes)


def _datacopy(b):
    direction = b.get("direction", "write" if b.get("dst") == "gm" else "read")
    owner_kind = b.get("owner_kind", "aic" if b.get("dst") == "l1" else "aiv")
    pipe = b.get("pipe", f"mte2_{owner_kind}" if direction == "read" else "mte3")
    owner_kind = "aic" if pipe in {"mte2_aic", "fixpipe"} else "aiv"
    rate = (b.speed(f"{pipe}_hit_bytes_per_s_per_core") if pipe.startswith("mte2") else
            b.speed("mte3_store_bytes_per_s_per_aiv", pid="P04_mte3_store"))
    _memory(b, "datacopy", pipe, direction, b.owner(owner_kind), _address_plan(b.get, b.integer), rate)


def _load_data(b):
    owner, k = b.owner("aic"), b.integer("k", minimum=1)
    k0 = b.integer("k0", int(DEFAULT_SECTOR_BYTES / dtype_bytes(b.get("dtype", "bf16"))), 1)
    transpose = b.get("kappa_trans", b.get("transpose_factor", 1)) if b.get("ifTranspose", b.get("transpose", False)) else 1
    if not math.isfinite(transpose) or transpose < 1:
        raise ModelingError("primitive_load_data_transpose_factor_invalid")
    target, after = b.get("operand", "both").lower(), ()
    if target not in {"a", "b", "both", "l0a", "l0b"}:
        raise ModelingError("primitive_load_data_operand_invalid")
    for operand, dimension in (("a", "m"), ("b", "n")):
        if target not in {"both", operand, f"l0{operand}"}:
            continue
        amount = (_ceil(b.integer(dimension, minimum=1), FRACTAL_DIM) * _ceil(k, k0) * FRACTAL_BYTES
                  + b.integer(f"mx_scale_bytes_{operand}", b.get("mx_scale_bytes", 0)))
        rate = b.speed(f"l1_to_l0{operand}_bytes_per_s_per_aic")
        setup = b.delay(f"load_{operand}:setup", "setup_ns", after)
        after = (b.work(f"load_{operand}", (("mte1", amount * transpose, rate),), (setup,),
                        owner=owner, metadata={"fractal_bytes": amount}, scale_pipe=True),)
        b.meta[f"l0{operand}_bytes"] = amount


def _fixpipe(b):
    dst, owner = b.get("dst", "gm"), b.owner("aic")
    m, n = b.integer("m", minimum=1), b.integer("n", minimum=1)
    if dst not in {"gm", "ub", "l1"}:
        raise ModelingError("primitive_fixpipe_destination_invalid")
    b.route("l0c", dst, owner, flag=None if dst == "gm" else f"has_l0c_to_{dst}")
    width = dtype_bytes(b.get("dst_dtype", b.get("dtype", "bf16")))
    split = b.get("split_mode", "NO_SPLIT")
    if dst == "ub" and split not in {"NO_SPLIT", "SPLIT_N"}:
        raise ModelingError("primitive_fixpipe_split_invalid")
    granule = FIXPIPE_SPLIT_N_ELEMS if split == "SPLIT_N" else b.integer("ele_per_blk", int(DEFAULT_SECTOR_BYTES / width), 1)
    n_eff = align_up(n, granule) if dst == "ub" else n
    out, inp = math.ceil(m * n_eff * width), L0C_ACCUMULATOR_BYTES * m * n
    out_rate = b.speed("fixpipe_out_bytes_per_s_per_aic")
    in_rate = b.speed("fixpipe_in_bytes_per_s_per_aic")
    b.meta.update(n_eff=n_eff, output_bytes=out, input_l0c_bytes=inp)
    after = (b.delay("fixpipe:setup", "setup_ns"),)
    if dst == "gm":
        plan = _address_plan(b.get, b.integer, out)
        effective_rate = min(out_rate, plan.aligned_copy_bytes() * in_rate / inp)
        _memory(b, "fixpipe", "fixpipe", "write", owner, plan, effective_rate, after)
    else:
        b.work("fixpipe", (("fixpipe", out, min(out_rate, out * in_rate / inp)),), after, owner=owner)


def _mte3(b):
    dst, owner = b.get("dst", "gm"), b.owner("aiv")
    plan = _address_plan(b.get, b.integer)
    rate = b.speed("mte3_store_bytes_per_s_per_aiv")
    after = (b.delay("mte3:setup", "setup_ns"),)
    atomic = b.get("atomic", False)
    if dst == "l1" and not atomic:
        b.route("ub", "l1", owner, flag="has_ub_to_l1")
        b.work("mte3", (("mte3", plan.aligned_copy_bytes(), rate),), after, owner=owner)
    elif dst == "gm":
        if atomic:
            after = (b.delay("mte3:atomic", "atomic_contention_ns", after),)
        _memory(b, "mte3", "mte3", "write", owner, plan, rate, after,
                atomic_rate=b.speed("atomic_rmw_bytes_per_s") if atomic else None)
    else:
        b.gap(f"mte3_destination_unsupported:{dst}")


def _l2(b):
    die = b.integer("die")
    capacity = (b.dp.usable_bytes("l2") if b.dp is not None and "l2" in dict(b.dp.buffers) else
                b.integer("l2_bytes", b.table.provenance.silicon_fingerprint.l2_bytes))
    if b.state is not None:
        knee = b.state.effective_l2_bytes(die)
    else:
        granule = b.integer("granule_bytes", DEFAULT_L2_GRANULE_BYTES, 1)
        available = max(0, capacity - b.integer("reserved_bytes", b.get("l2_reserved_bytes", 0)))
        knee = granule * (available // (granule * max(1, b.integer("n_active_streams", 1))))
    bypass = b.get("cache_policy") == CACHE_POLICY_DISABLE
    work = b.integer("working_set_bytes", b.get("bytes", 0))
    pattern = b.get("pattern")
    hits = b.state.preview_access(pattern, pattern.runs.aligned_copy_bytes())[0] if pattern and b.state else b.integer("hit_bytes")
    hit = 0 if bypass or work > knee else min(work, hits)
    b.meta.update(effective_l2_bytes=knee, cache_bypass=bypass, l2_hit_bytes=hit)
    b.work("l2_reuse", (("l2", hit, b.speed("l2_die_bytes_per_s")),))


def _mmad(b):
    requested = b.get("mode", b.get("dtype", "bf16"))
    conversions = {"w4a8": "int8", "a8w4_mx": "mxfp8"}
    mode = requested if requested in b.cell.modes else conversions.get(requested, requested)
    b.warnings.extend(_simulation_warnings(b.cell, mode, b.role, b.allow))
    m, n, k = dims = tuple(b.integer(axis, minimum=1) for axis in ("m", "n", "k"))
    data = b.cell.modes[mode]
    wave = next((item for item in b.dp.subtile_wave if item.native_mode == mode), None) if b.dp is not None else None
    declared = b.dp.compute_arch.declared if b.dp is not None else None
    native = b.get("native_mode", conversions.get(requested, requested))
    if declared is not None and not declared.supports_native("matrix", native):
        b.gap(f"native_matrix_mode_required:{native}")
    defaults = (
        wave.independently_issuable_m if wave else data.get("m_iss", data.get("tile_m", FRACTAL_DIM)),
        wave.tile_n if wave else data.get("n_iss", data.get("tile_n", FRACTAL_DIM)),
        wave.tile_k if wave else data.get("k_iss", data.get("tile_k", FRACTAL_DIM)),
    )
    issued = 2 * math.prod(align_up(dim, b.integer(key, default, 1))
                           for dim, key, default in zip(dims, ("m_iss", "n_iss", "k_iss"), defaults))
    b.meta.update(issued_flops=issued, logical_flops=2 * m * n * k)
    after = ()
    prologue = b.get("prologue", requested if requested in conversions else "")
    if prologue == "w4a8":
        owner = b.owner("aiv")
        packed = _ceil(n * k, 2)
        read = b.work("unpack:packed_read", (("mte2_aiv", packed, b.speed("mte2_aiv_hit_bytes_per_s_per_core", pid="P01_datacopy")),), owner=owner)
        cast_rate = b.speed("cast_bytes_per_s_per_aiv", pid="P09_vec_reduce_cast")
        cast1 = b.work("unpack:cast_first", (("vec_alu", n * k, cast_rate),), (read,), owner=owner)
        barrier = b.delay("unpack:PipeBarrier<PIPE_V>", "unpack_barrier_ns", (cast1,), mode=mode)
        cast2 = b.work("unpack:cast_second", (("vec_alu", n * k, cast_rate),), (barrier,), owner=owner)
        after = (b.work("unpack:workspace_write", (("mte3", n * k, b.speed("mte3_store_bytes_per_s_per_aiv", pid="P04_mte3_store")),), (cast2,), owner=owner),)
    elif prologue == "a8w4_mx":
        amount = n * k + b.integer("mx_scale_bytes", n * _ceil(k, FRACTAL_DIM))
        after = (b.work("mx:decompress_b", (("mte1", amount, b.speed("l1_to_l0b_bytes_per_s_per_aic", pid="P02_load_data")),), owner=b.owner("aic")),)
    elif prologue:
        b.gap(f"mmad_prologue_unsupported:{prologue}")
    cube = b.work("mmad", (("cube", issued, b.speed("cube_only_rate", mode=mode)),), after,
                  owner=b.owner("aic"), metadata={"issued_flops": issued})
    if (m // FRACTAL_DIM) * (n // FRACTAL_DIM) < PIPE_M_BARRIER_FRACTALS:
        b.delay("PipeBarrier<PIPE_M>", "pipe_m_barrier_ns", (cube,))


def _vector(b):
    pipes = {"P07_vec_alu": "vec_alu", "P08_vec_sfu": "vec_sfu", "P09_vec_reduce_cast": "vec_reduce"}
    pipe = pipes[b.pid]
    dtype, owner = b.get("dtype", "bf16"), b.owner("aiv")
    lanes = b.integer("lanes", int(b.integer("vector_width_bytes", VECTOR_WIDTH_BYTES, 1) / dtype_bytes(dtype)), 1)
    elems, rows = b.integer("elements", b.get("n", 0)), b.integer("rows", 1)
    width = b.integer("reduction_dim", b.get("d", elems))
    raw = b.value("raw_latency_cycles", 0)
    if pipe == "vec_reduce" and width:
        depth = min(width, lanes, MX_SCALE_GROUP if b.get("mx_dynamic_quant") else lanes)
        raw += b.value("tree_base_cycles", 0) + b.value("tree_step_cycles", 0) * (depth - 1).bit_length()
    repeats = rows * _ceil(width if pipe == "vec_reduce" else elems, lanes)
    groups = b.get("groups")
    if groups is None:
        graph = [(pipe, pipe, repeats, raw)]
        if b.get("mx_dynamic_quant", False):
            if pipe != "vec_reduce" or not b.cell.parameters.get("mx_dynamic_quant_supported", False):
                b.gap("mx_dynamic_quant_required")
            scales = rows * _ceil(width, MX_SCALE_GROUP)
            graph = [("vabs", "vec_alu", repeats, 0), ("vreduce_max", pipe, scales, raw),
                     ("e8m0_scale", "vec_alu", _ceil(scales, lanes), 0), ("vmul", "vec_alu", repeats, 0),
                     ("vcast_pack", "vec_alu", repeats, 0)]
        elif pipe == "vec_reduce" and b.get("cast", False):
            graph.append(("vcast_pack", "vec_alu", repeats, 0))
        groups = tuple(VectorInstructionGroup(
            name, subpipe, count, b.integer("issue_slots", 1, 1), b.integer("initiation_cycles", 1, 1),
            cycles, (graph[index - 1][0],) if index else (), owner, b.get("last_issue_mark", True),
        ) for index, (name, subpipe, count, cycles) in enumerate(graph))
    else:
        groups = tuple(group if isinstance(group, VectorInstructionGroup) else
                       VectorInstructionGroup(**{"owner": owner, **group}) for group in groups)
    groups = tuple(replace(group, result_latency_cycles=0) if not group.count else group for group in groups)
    rates = {}
    for subpipe in {group.pipe for group in groups}:
        fallback = b.value(f"{subpipe}_cycles_per_s", 0)
        if not fallback and subpipe == "vec_alu" and pipe == "vec_reduce":
            fallback = b.value("cast_bytes_per_s_per_aiv") / (lanes * dtype_bytes(dtype))
        if not fallback:
            fallback = b.value(f"{subpipe}_ops_per_s_per_aiv") / lanes
        rates[subpipe] = fallback * BYTES_PER_S_TO_MS
    issue = b.value("vec_issue_cycles_per_s", b.value("issue_cycles_per_s", 0)) * BYTES_PER_S_TO_MS
    declared = b.dp.compute_arch.declared if b.dp is not None else None
    clock = b.get("clock_hz") or next((s.clock_hz for s in getattr(declared, "instruction_services", ())
                                     if s.name == "vec_issue" and s.clock_hz), None)
    if clock is None and any(group.result_latency_cycles for group in groups):
        b.gap("vector_clock_required")
    issue = issue or (clock * BYTES_PER_S_TO_MS if clock else max(rates.values(), default=1))
    bound = bind_vector_groups(groups, issue_rate_per_ms=issue, pipe_rates_per_ms=rates,
                               clock_hz=clock or 1, resource_names={key: b.row(key) for key in (*rates, "vec_issue")})
    limits = {b.row(key): value for key, value in {**rates, "vec_issue": issue}.items()}
    stages = tuple(replace(stage, caps={key: limits[key] for key in stage.work}) for stage in bound.stages)
    b.stages.extend(stages)
    b.times["vector"] = bound.span_ms
    b.extra["bound_vector"] = replace(bound, stages=stages)


def _sort(b):
    n, owner = b.integer("n"), b.owner("aiv")
    aggregate = b.speed("sort_elems_per_s")
    if b.get("algorithm", b.cell.parameters.get("sort_algorithm")) == "sort32_mrgsort4":
        if n not in SORTALL_LENGTHS:
            raise ModelingError("primitive_sortall_length_unsupported")
        groups, passes, after = n // SORT_BLOCK_ELEMS, 0, ()
        steps = [("Sort32", "vbitsort32", groups, SORT_BLOCK_ELEMS, {})]
        while groups > 1:
            fan_in = min(MERGE_FAN_IN, groups)
            groups = _ceil(groups, fan_in)
            passes += 1
            steps.append((f"MrgSort:{passes}", "vmrgsort4", groups, n // groups, {"fan_in": fan_in}))
        if passes % 2 == 0:
            steps.append(("copyback", "vec_alu", n, 1, {}))
        for name, pipe, count, width, meta in steps:
            rate = b.speed(f"{pipe}_per_s", default=aggregate * MS_PER_SECOND / width)
            done = b.work(name, ((pipe, count, rate),), after, owner=owner, metadata=meta)
            after = (b.delay(f"{name}:PipeBarrier<PIPE_V>", "pipe_v_barrier_ns", (done,)),)
        return
    if b.get("algorithm", b.cell.parameters.get("sort_algorithm")) != "masked_hist256":
        b.gap("sort_algorithm_required")
    after = ()
    for start in range(0, n, HISTOGRAM_TRUNK_ELEMS):
        width = min(n - start, HISTOGRAM_TRUNK_ELEMS)
        for phase, elems in (("high_histogram", width), ("threshold", HISTOGRAM_BINS),
                             ("masked_low_histogram", width), ("tie_prefix", width)):
            name = b.work(f"{phase}:{start}", (("vhist256", elems, aggregate),), after,
                          owner=owner, metadata={"bins": HISTOGRAM_BINS})
            after = (name,)
    splits = b.integer("splits", 1, 1)
    if splits > 1:
        io_bytes = b.integer("merge_bytes", splits * b.integer("k", minimum=1) * 2 * L0C_ACCUMULATOR_BYTES)
        read = b.work("split_merge:workspace_read", (("hbm", io_bytes, b.speed("hbm_read_bytes_per_s", pid="P01_datacopy")),), after)
        b.work("split_merge", (("vhist256", splits * b.integer("k", minimum=1), aggregate),), (read,), owner=owner)


def _simt(b):
    if b.dp is None or not b.dp.simt_ub_carveout_bytes or b.cell.parameters.get("supported") is False:
        b.gap("simt_ub_carveout_required")
    frontend = b.get("frontend", "tla_dsl")
    if frontend not in SIMT_THREAD_LIMITS:
        raise ModelingError("primitive_simt_frontend_invalid")
    dims = b.get("thread_block_dim")
    threads = b.integer("threads", math.prod(_int(dim, "thread_block_dim", 1) for dim in dims)
                        if dims is not None else SIMT_WARP_THREADS, 1)
    if threads > SIMT_THREAD_LIMITS[frontend]:
        raise ModelingError("primitive_simt_thread_limit")
    owner, warps = b.owner("aiv"), _ceil(threads, SIMT_WARP_THREADS)
    if "ub" in dict(b.dp.buffers) and b.integer("ub_bytes") > b.dp.usable_bytes("ub", owner, kernel="simt"):
        b.gap("simt_ub_capacity_exceeded")
    b.meta.update(active_warps=warps, simt_ub_carveout_bytes=b.dp.simt_ub_carveout_bytes)
    after = (b.delay("simt_launch", "simt_launch_ns"),)
    addresses = b.get("lane_addresses")
    if addresses is not None:
        width = b.integer("access_width", minimum=1)
        mask = b.integer("active_lane_mask", (1 << len(addresses)) - 1)
        plan = AddressRunPlan(tuple(AddressRun(_int(addr, "lane_address"), 1, width, width)
                                    for lane, addr in enumerate(addresses) if mask & (1 << lane)))
        # Canonical sector intervals deduplicate arbitrary coalesced lane accesses.
        sectors = AddressRunPlan(tuple(AddressRun(lo, 1, hi - lo, hi - lo) for lo, hi in plan.sector_intervals()))
        direction = b.get("direction", "read")
        pipe = "mte2_aiv" if direction == "read" else "mte3"
        metric, pid = (("mte2_aiv_hit_bytes_per_s_per_core", "P01_datacopy") if direction == "read" else
                       ("mte3_store_bytes_per_s_per_aiv", "P04_mte3_store"))
        _memory(b, "simt_memory", pipe, direction, owner, sectors, b.speed(metric, pid=pid), after,
                logical_size=len(plan.runs) * width)
        after = ("simt_memory",)
    done = b.work("simt", (("simt", warps * b.integer("warp_instructions", 1),
                            b.speed("simt_ops_per_s_per_aiv")),), after, owner=owner)
    b.delay("sync_threads", "syncthreads_ns", (done,))


def _single_l0c(b):
    if b.get("enable_unit_flag", False) and b.integer("l0c_stages", b.get("l0c", 1), 1) != 1:
        raise ModelingError("primitive_unit_flag_requires_single_l0c_stage")


def _intracore(b):
    _single_l0c(b)
    event = b.integer("event_id")
    if event >= INTRACORE_EVENT_IDS:
        raise ModelingError("primitive_intracore_event_id_invalid")
    unitflag = b.get("enable_unit_flag", False)
    b.delay("unitflag" if unitflag else "flag", "unitflag_ns" if unitflag else "flag_ns",
            mode=b.get("edge"), metadata={"event_id": event, "unit_flag": bool(unitflag)})


def _crosscore(b):
    mode, kind = b.integer("mode", 2), b.get("core_kind", "aic")
    if mode not in (0, 1, 2) or kind not in INTER_BLOCK_IDS:
        raise ModelingError("primitive_crosscore_scope_invalid")
    key = f"mode0_{kind}_barrier" if mode == 0 else "mode1_subblock" if mode == 1 else "mode2_paired"
    count = b.integer("core_count", 1, 1)
    if mode == 0 and count > getattr(b.dp or b.table.provenance.silicon_fingerprint, f"{kind}_count"):
        raise ModelingError("primitive_crosscore_count_invalid")
    metric = f"{key}_{count}_ns" if mode == 0 and f"{key}_{count}_ns" in b.cell.metrics else f"{key}_ns"
    event = b.integer("event_id", INTER_BLOCK_IDS[kind] if mode == 0 else SUBBLOCK_BARRIER_ID if mode == 1 else 0)
    reverse, first = b.get("with_reverse", False), b.integer("generation")
    reverse_id = b.integer("reverse_id", 1)
    if mode == 2 and (event > PAIRED_FLAG_MAX_ID or reverse and (reverse_id == event or reverse_id > PAIRED_FLAG_MAX_ID)):
        raise ModelingError("primitive_crosscore_event_id_invalid")
    previous, credit = (), {}
    for index in range(first, first + b.integer("notifications", 1, 1)):
        group = max(0, index // REVERSE_DEPTH - 1)
        deps = previous + ((credit[group],) if reverse and index >= REVERSE_DEPTH and group in credit else ())
        meta = {"mode": mode, "scope": kind if mode == 0 else key, "core_count": count, "event_id": event}
        previous = (b.delay(f"crosscore:{index}", metric, deps, metadata=meta),)
        if reverse and (index + 1) % REVERSE_DEPTH == 0:
            credit[index // REVERSE_DEPTH] = b.delay(f"reverse_credit:{index}", after=previous,
                                                     ms=b.value("reverse_ns", b.value(metric)) * NS_TO_MS,
                                                     metadata={**meta, "event_id": reverse_id})
    b.meta.update(event_id=event, reverse_depth=REVERSE_DEPTH if reverse else 0)


def _mutex(b):
    _single_l0c(b)
    pools = {"l1a": "l1", "l1b": "l1", "l0a": "l0a", "l0b": "l0b", "l0c": "l0c"}
    depths = {name: b.integer(f"{name}_stages", b.get(name, 1), 1) for name in pools}
    total = sum(depths.values()) + 2 * bool(b.get("has_bias", False))
    if total > MAX_MUTEX_IDS:
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
    b.delay("descriptor_issue", ms=max(b.value("host_submit_ns"), interval) * NS_TO_MS)


def _streams(b):
    mode = b.get("mode", "disjoint")
    if mode not in {"disjoint", "shared_cap"}:
        raise ModelingError("primitive_stream_isolation_invalid")
    setup = b.delay("set_limit", "set_limit_ns", metadata={"isolation": mode})
    after = (b.delay("stream_event", "stream_event_ns", (setup,)),)
    plan, program = b.get("stream_plan"), b.get("program")
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
        b.append(Stage("stream_work", after=after, program=program), b.get("work_ms", 0))
    b.meta["isolation"] = mode


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
    band = _FramedCollectiveBand.from_mapping({**b.cell.parameters, **b.p, "startup_us": startup / US_TO_MS,
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
    edges = b.dp.ub_peer_edges if b.dp else ()
    if edges and (rank, target) not in edges:
        b.gap("direct_peer_edge_required")
    payload_frame = b.integer("token_payload_bytes_per_frame", TOKEN_FRAME_PAYLOAD_BYTES, 1)
    wire_frame = b.integer("token_wire_bytes_per_frame", TOKEN_FRAME_WIRE_BYTES, 1)
    wire = sum(frame_wire_bytes(size, payload_frame, wire_frame)[0] for size in tokens)
    status = b.integer("status_bytes")
    pack = b.delay("pack", "pack_ns")
    frame = b.delay("per_token_frame", "frame_ns", (pack,), metadata={"wire_bytes": wire})
    rate = b.speed("peer_bytes_per_s")
    rows = (b.get("peer_resource", f"peer:{rank}->{target}"), b.get("endpoint_resource", f"peer_egress:d{rank}"))
    fanout = sum(src == rank for src, _ in edges)
    endpoint_rate = b.speed("endpoint_bytes_per_s", default=rate * MS_PER_SECOND * b.integer("fanout", max(1, fanout), 1))
    transfer = b.work("transfer", ((rows[0], wire, rate), (rows[1], wire, endpoint_rate)), (frame,))
    sent = b.work("status", ((rows[0], status, rate), (rows[1], status, endpoint_rate)), (transfer,))
    synced = b.delay("status_sync", "status_sync_ns", (sent,))
    b.delay("unpack", "unpack_ns", (synced,))
    b.meta.update(wire_bytes=wire + status, token_wire_bytes=wire)


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


_LOWERERS = dict(zip(PRIMITIVE_IDS, (
    _datacopy, _load_data, _fixpipe, _mte3, _l2, _mmad, _vector, _vector, _vector, _sort, _simt,
    _intracore, _crosscore, _mutex, _stars, _streams, _aicpu, _collective, _peer, _dma,
), strict=True))


def lower_primitive(
    primitive_id: str, params: Mapping[str, Any], table: PrimitiveTable, *,
    datapath: DatapathCapabilitySpec | None = None, context: RecipeContext | None = None,
    memory_state: AdmissionMemoryView | None = None, role: RateRole = "sim",
    allow_uncalibrated_declaration: bool = True,
) -> PrimitiveLoweringResult:
    """Lower a primitive and return executable stages plus their isolated span."""
    if role not in {"sim", "dual"} or not isinstance(params, Mapping):
        raise ModelingError("primitive_lowering_input_invalid")
    datapath = datapath if datapath is not None else getattr(context, "datapath", None)
    lowering = _Lowering(primitive_id, params, table, datapath, context, memory_state, role,
                         allow_uncalibrated_declaration)
    if primitive_id != "P06_mmad":
        selected = _declared_mode(lowering.cell, params.get("mode"))
        lowering.warnings.extend(_simulation_warnings(lowering.cell, selected, role, allow_uncalibrated_declaration))
    _LOWERERS[primitive_id](lowering)
    return lowering.finish()


__all__ = [
    "BYTES_PER_S_TO_MS", "NS_TO_MS", "US_TO_MS", "MS_PER_SECOND", "ShortFramePolicy",
    "PrimitiveLoweringResult", "frame_wire_bytes", "lower_primitive", "submit_lowered_primitive",
]
