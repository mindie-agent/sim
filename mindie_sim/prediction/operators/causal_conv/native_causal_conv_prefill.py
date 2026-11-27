"""Native arch35 causal convolution physical program.

[Role] Execute source host tiles, rolling/generic tokens and initial-state sync.
[Invariants] Every copy/vector stage owns its AIV; shared HBM is not divided.
  Parent readiness includes UB reservation and snapshot completion. Arithmetic
  and output casts retain their distinct input/output reuse dependencies.
[Timeouts & Locks] Bounded lazy token issue window; no locks or subprocesses.
[Side-Effect Policy] Pure programs on the caller engine; no device or data I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal, Mapping

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import OpSpec
from mindie_sim.prediction.engine import Engine as _ConvEngine, Resource as _ConvResource
from mindie_sim.prediction.instructions import REGISTER_WORK_ROLES
from mindie_sim.prediction.mechanism import Geometry, OperatorPlan, Work
from mindie_sim.prediction.ownership import unit_group
from mindie_sim.prediction.program import KernelProgram, PipelineProgram, Stage
from mindie_sim.prediction.operators.vector.mixed_core import read_role, split_reads
from mindie_sim.prediction.operators.causal_conv import native_conv_registers as registers
from .native_causal_conv_spec import (
    IMPLEMENTATION, MAX_WIDTH, PAIR_TOKENS, SOURCE_REFS,
    TOKEN_PIPELINE_SLOTS, UB_RESIDENT_BYTES, _ceil_div, _config_from_op,
    _dtype_width, _host_tiling, _parse_params, _resolve_cores_and_ub,
    host_tiling, make_op,
)

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"

SILU_OPS_PER_ELEMENT = 4
GENERIC_TAP_OPS_PER_ELEMENT = 3
CONV_COMPACT_MIN_LEN = 32
CONV_COMPACT_HEAD_TOKENS = 9
CONV_COMPACT_TAIL_MIN_TOKENS = 7
CONV_COMPACT_PERIOD = 4
CONV_COMPACT_MIN_PERIODS = 2
CONV_PROBE_LENGTHS = (20, 24, 28)
CONV_PERIOD_CACHE_MAX = 512
CONV_PERIOD_REL_TOL = 1e-12

__all__ = ["make_op", "build", "host_tiling", "source_ledger", "IMPLEMENTATION"]


def _work(*, vector=0, hbm=0, mte2=0, mte3=0):
    resources = {}
    for name, value in (("vector", vector), ("hbm", hbm),
                        ("mte2", mte2), ("mte3", mte3)):
        if value:
            resources[name] = float(value)
    return Work({}, resources)


def _add(left: Work, right: Work) -> Work:
    return left + right


def _sum(works) -> Work:
    total = Work()
    for work in works:
        total = total + work
    return total


def _find_seq(cfg, token):
    """FindVarlenSeqByToken and its exact number of scalar qsl reads."""
    left, right, reads = 0, cfg.batch, 0
    while left < right:
        mid = left + ((right - left) >> 1)
        reads += 1
        if token < cfg.qsl[mid + 1]:
            right = mid
        else:
            left = mid + 1
    return left, reads


def _cache_value(cfg, seq):
    """Known slot value, or None when the caller-bound tensor is not bound."""
    if not cfg.cache_index_present:
        return None
    return None if cfg.cache_indices is None else cfg.cache_indices[seq]


def _cache_skipped(cfg, cache_idx):
    if not cfg.cache_index_present or cache_idx is None:
        return False
    if cache_idx == cfg.pad_slot_id:
        return True
    return cfg.null_block_id is not None and cache_idx == cfg.null_block_id


@dataclass(frozen=True)
class _ChunkPlan:
    seq: int
    start: int
    end: int
    seq_start: int
    seq_len: int
    has_init: bool
    first: bool
    last: bool
    rolling: bool
    history: Work
    restore: Work
    state_load: Work
    state_store: Work
    per_token_load: int
    per_token_store: int
    compute: Work
    cast: Work

    @property
    def length(self):
        return self.end - self.start

    @property
    def state(self):
        return self.state_load + self.state_store


@dataclass(frozen=True)
class _BlockPlan:
    index: int
    token_tile: int
    base_index: int
    channel_start: int
    base_size: int
    token_start: int
    token_end: int
    weight_load: Work
    weight_convert: Work
    metadata: Work
    chunks: tuple[_ChunkPlan, ...]
    prefetch_load: Work
    prefetch_store: Work
    qsl_bytes: int
    cache_bytes: int
    his_bytes: int
    prefetch_his_bytes: int
    prefetch_cache_bytes: int
    prefetch_payload_bytes: int
    prefetch_copied_sequences: int


def _generic_token_vector(width, activation):
    act = SILU_OPS_PER_ELEMENT if activation == "silu" else 0
    # Generic RunSeq with bias: bias add, width casts, width MulAddDst
    # (multiply+add), activation and output cast.
    return 2 + GENERIC_TAP_OPS_PER_ELEMENT * width + act


def _chunk_plan(cfg, seq, cursor, tile_end, base_size):
    seq_start = cfg.qsl[seq]
    seq_len = cfg.qsl[seq + 1] - seq_start
    has_init = cfg.has_init(seq)
    history_count = cfg.history_count
    ring_start = MAX_WIDTH - cfg.width
    vector = ring_start * base_size
    read = 0
    for index in range(history_count):
        src = cursor - history_count + index
        if src >= seq_start:
            read += base_size * 2
        elif has_init:
            read += base_size * 2
        else:
            vector += base_size
    history = _work(vector=vector, hbm=read, mte2=read)
    if cfg.bias:
        restore = Work()
        compute = _work(vector=(_generic_token_vector(cfg.width, cfg.activation) - 1) * base_size)
        cast = _work(vector=base_size)
    else:
        restore = registers.restore(cfg.width, base_size).capacity_work(registers.CLASSES)
        compute = registers.rolling(cfg.width, base_size, cfg.activation).capacity_work(registers.CLASSES)
        cast = Work()
    state_vector = state_read = state_write = 0
    last = tile_end == cfg.qsl[seq + 1]
    if last and cfg.has_conv_states:
        if not cfg.bias:
            # WriteBackFnState's retained-x reload.  It runs before the
            # MTE2_MTE3 handshake and the state store; the two directions are
            # never a single simultaneous stage.
            if seq_len >= history_count:
                state_read += history_count * base_size * 2
            else:
                for pos in range(history_count):
                    relative = seq_len - history_count + pos
                    if relative >= 0:
                        state_read += base_size * 2
                    elif has_init:
                        state_read += base_size * 2
                    else:
                        state_vector += base_size
        state_write = history_count * base_size * 2
    state_load = _work(vector=state_vector, hbm=state_read, mte2=state_read)
    state_store = _work(hbm=state_write, mte3=state_write)
    return _ChunkPlan(
        seq=seq, start=cursor, end=tile_end, seq_start=seq_start,
        seq_len=seq_len, has_init=has_init, first=cursor == seq_start,
        last=last, rolling=not cfg.bias, history=history, restore=restore,
        state_load=state_load, state_store=state_store,
        per_token_load=base_size * 2, per_token_store=base_size * 2,
        compute=compute, cast=cast,
    )


def _block_chunks(cfg, host, block_index, base_size):
    base_dim = host["base_dim"]
    base_dim_count = host["base_dim_count"]
    token_tile = block_index // base_dim_count
    token_start = token_tile * host["token_block_size"]
    token_end = min(token_start + host["token_block_size"], cfg.cu_seqlen)
    explicit = bool(host["has_explicit_token_seq_ranges"])
    qsl_width = _dtype_width(cfg.dtypes.query_start_loc)
    cache_width = _dtype_width(cfg.dtypes.cache_indices)
    his_width = _dtype_width(cfg.dtypes.has_initial_state)

    qsl_bytes = cache_bytes = his_bytes = 0
    first_seq, reads = _find_seq(cfg, token_start)
    if not explicit:
        qsl_bytes += reads * qsl_width
    if explicit:
        end_seq = first_seq
        while end_seq < cfg.batch and cfg.qsl[end_seq] < token_end:
            end_seq += 1
    else:
        end_seq = cfg.batch
    chunks = []
    cursor = token_start
    seq = first_seq
    while cursor < token_end and seq < end_seq:
        qsl_bytes += 2 * qsl_width
        seq_start, seq_end = cfg.qsl[seq], cfg.qsl[seq + 1]
        if seq_end <= seq_start:
            seq += 1
            continue
        if cursor < seq_start:
            cursor = seq_start
        if cursor >= seq_end:
            seq += 1
            continue
        tile_end = min(token_end, seq_end)
        if tile_end <= cursor:
            seq += 1
            continue
        if cfg.cache_index_present:
            cache_bytes += cache_width
            if _cache_skipped(cfg, _cache_value(cfg, seq)):
                cursor = tile_end
                seq += 1
                continue
        if cfg.has_conv_states and cfg.has_init_flag:
            his_bytes += his_width
        chunks.append(_chunk_plan(cfg, seq, cursor, tile_end, base_size))
        cursor = tile_end
        seq += 1
    metadata = _work(hbm=qsl_bytes + cache_bytes + his_bytes,
                     mte2=qsl_bytes + cache_bytes + his_bytes)
    return token_start, token_end, metadata, tuple(chunks), qsl_bytes, cache_bytes, his_bytes


def _prefetch_block(cfg, host, token_tile, base_size):
    history_count = cfg.history_count
    his_width = _dtype_width(cfg.dtypes.has_initial_state)
    cache_width = _dtype_width(cfg.dtypes.cache_indices)
    his_bytes = cache_bytes = payload = copied = 0
    if cfg.has_init_workspace:
        owner_count = host["token_block_count"]
        for seq in range(token_tile, cfg.batch, owner_count):
            his_bytes += his_width
            if not cfg.has_init(seq):
                continue
            cache_idx = _cache_value(cfg, seq) if cfg.cache_index_present else seq
            if cfg.cache_index_present:
                cache_bytes += cache_width
            if _cache_skipped(cfg, cache_idx):
                continue
            payload += history_count * base_size * 2
            copied += 1
    load = _work(hbm=payload + his_bytes + cache_bytes,
                 mte2=payload + his_bytes + cache_bytes)
    store = _work(hbm=payload, mte3=payload)
    return load, store, his_bytes, cache_bytes, payload, copied


def _block_plan(cfg, host, block_index):
    base_dim = host["base_dim"]
    base_dim_count = host["base_dim_count"]
    token_tile = block_index // base_dim_count
    base_index = block_index % base_dim_count
    channel_start = base_index * base_dim
    base_size = min(base_dim, cfg.dim - channel_start)
    if base_size <= 0:
        raise ValueError("host tiling produced an empty channel tile")
    token_start, token_end, metadata, chunks, qsl_bytes, cache_bytes, his_bytes = \
        _block_chunks(cfg, host, block_index, base_size)
    prefetch_load, prefetch_store, prefetch_his, prefetch_cache, payload, copied = \
        _prefetch_block(cfg, host, token_tile, base_size)

    weight_bytes = (cfg.width + (1 if cfg.bias else 0)) * base_size * 2
    weight_load = _work(hbm=weight_bytes, mte2=weight_bytes)
    weight_vector = ((MAX_WIDTH - cfg.width) * base_size
                     + cfg.width * base_size + base_size)
    weight_convert = _work(vector=weight_vector)
    return _BlockPlan(
        index=block_index, token_tile=token_tile, base_index=base_index,
        channel_start=channel_start, base_size=base_size,
        token_start=token_start, token_end=token_end,
        weight_load=weight_load, weight_convert=weight_convert,
        metadata=metadata, chunks=chunks, prefetch_load=prefetch_load,
        prefetch_store=prefetch_store, qsl_bytes=qsl_bytes,
        cache_bytes=cache_bytes, his_bytes=his_bytes,
        prefetch_his_bytes=prefetch_his, prefetch_cache_bytes=prefetch_cache,
        prefetch_payload_bytes=payload, prefetch_copied_sequences=copied,
    )


def _prepare(cfg, host):
    if not host["valid"]:
        raise ValueError("causal_conv1d native FN host tiling is invalid for this geometry")
    block_dim = int(host["block_dim"])
    core_num = int(host["core_count"])
    if block_dim > core_num:
        raise ValueError("blockDim exceeds declared vector-unit ownership")
    if block_dim < int(host["grid_size"]):
        raise ValueError(
            "selected source tiling launches fewer blocks than token/dim tiles; "
            "output channels would not be covered exactly once")
    blocks = tuple(_block_plan(cfg, host, index) for index in range(block_dim))
    return blocks


def _block_total(block):
    total = block.weight_load + block.weight_convert + block.metadata
    total = total + block.prefetch_load + block.prefetch_store
    for chunk in block.chunks:
        total = total + chunk.history + chunk.restore + chunk.state_load + chunk.state_store
        total = total + (chunk.compute + chunk.cast).scaled(chunk.length)
        total = total + _work(hbm=chunk.per_token_load * chunk.length,
                             mte2=chunk.per_token_load * chunk.length)
        total = total + _work(hbm=chunk.per_token_store * chunk.length,
                              mte3=chunk.per_token_store * chunk.length)
    return total


def _sync_work(host):
    return _work(hbm=host["sync_workspace_bytes"]) if host["sync_workspace_bytes"] else Work()


def _placement(context, index, owners):
    return context.owned(index, owners,
                         ("vector", read_role(context.resource_names, aiv=0), "mte3",
                          *REGISTER_WORK_ROLES))


def _bind(context, work):
    return context.bind(split_reads(work, 0, context.resource_names))


def _scale_demand(base: Mapping[str, float], count: int = 1) -> dict[str, float]:
    if count == 1:
        return dict(base)
    return {key: value * count for key, value in base.items()}


def _emit_token_stages(
    pipeline: _TokenPipeline,
    engine,
    name: str,
    ready,
    owner: str,
    latest: dict[str, object],
    t: int,
    *,
    on_retire=None,
) -> list[object]:
    """Emit all stages for token index ``t`` preserving rolling/generic slot dependencies."""
    def sub(label, work, after, on_complete=None, key=None):
        event = engine.submit(
            f"{name}:tile{t}:{label}", work, after=tuple(after),
            placement=pipeline.placement, owner=owner, on_complete=on_complete,
        )
        latest[label if key is None else key] = event
        return event

    if pipeline.rolling:
        if t == 0:
            compute = sub("compute", _scale_demand(pipeline.compute), (ready,), key="compute_even")
            store = sub("store", _scale_demand(pipeline.store), (compute,), on_retire, key="store_even")
            ring = sub("ringp", {}, (compute,))
            return [compute, store, ring]
        if t % 2:
            k = (t - 1) // 2
            deps = (ready,) if t == 1 else (latest["load"],)
            if k == 1:
                deps += (latest["ringp"],)
            elif k >= 2:
                deps += (latest["ring%u" % (k % 2)],)
            tokens = min(PAIR_TOKENS, pipeline.length - t)
            load = sub("load", _scale_demand(pipeline.load, tokens), deps)
            cdeps = (load, latest["compute_even"])
            if t > PAIR_TOKENS:
                cdeps += (latest["store_odd"],)
            compute = sub("compute", _scale_demand(pipeline.compute), cdeps, key="compute_odd")
            store = sub("store", _scale_demand(pipeline.store), (compute, latest["store_even"]),
                        on_retire, key="store_odd")
            return [load, compute, store]
        k = (t - 2) // 2
        compute = sub(
            "compute", _scale_demand(pipeline.compute),
            (latest["load"], latest["compute_odd"], latest["store_even"]),
            key="compute_even",
        )
        store = sub("store", _scale_demand(pipeline.store), (compute, latest["store_odd"]),
                    on_retire, key="store_even")
        ring = sub("ring%u" % (k % 2), {}, (compute,))
        return [compute, store, ring]

    parity = "odd" if t % 2 else "even"
    prior = "even" if t % 2 else "odd"
    local: list[object] = []
    inputs = (ready,)
    if t:
        deps = (ready,) + ((latest["load"],) if t > 1 else ())
        if t >= 2:
            deps += (latest["cast_" + parity],)
        load = sub("load", _scale_demand(pipeline.load), deps)
        inputs = (load, latest["cast_" + prior])
        local.append(load)
    compute = sub("compute", _scale_demand(pipeline.compute), inputs)
    out_ready = (compute,) + ((latest["store_" + parity],) if t >= 2 else ())
    cast = sub("cast", _scale_demand(pipeline.cast), out_ready, key="cast_" + parity)
    store_ready = (cast,) + ((latest["store_" + prior],) if t else ())
    store = sub("store", _scale_demand(pipeline.store), store_ready, on_retire, key="store_" + parity)
    local.extend((compute, cast, store))
    return local


@lru_cache(maxsize=CONV_PERIOD_CACHE_MAX)
def _cached_conv_period4(
    rolling: bool,
    slots: int,
    load_t: tuple[tuple[str, float], ...],
    comp_t: tuple[tuple[str, float], ...],
    store_t: tuple[tuple[str, float], ...],
    cast_t: tuple[tuple[str, float], ...],
    res_spec: tuple[tuple[str, float, int, str, int], ...],
) -> float | None:
    """Measure and verify the exact 4-token steady-state period on a standalone probe engine."""
    resources = tuple(
        _ConvResource(rname, rate, units=units, policy=pol)
        for rname, rate, units, pol, _ in res_spec
    )
    placement = {
        rname: tuple(range(owned))
        for rname, _, units, _, owned in res_spec
        if owned > 0 and units > 1
    }
    d_load, d_comp, d_store, d_cast = dict(load_t), dict(comp_t), dict(store_t), dict(cast_t)
    times: list[float] = []
    for probe_len in CONV_PROBE_LENGTHS:
        eng = _ConvEngine(resources)
        pipe = _TokenPipeline(
            length=probe_len, rolling=rolling, load=d_load, compute=d_comp,
            store=d_store, cast=d_cast, placement=placement, slots=slots,
        )
        ev = pipe._start_literal(eng, "probe", eng.submit("r"), owner="")
        eng.advance(until=(ev,))
        times.append(float(ev.time))
    p1 = times[1] - times[0]
    p2 = times[2] - times[1]
    if p1 <= 0.0 or abs(p2 - p1) > CONV_PERIOD_REL_TOL * max(1.0, p1):
        return None
    return p1


@dataclass(frozen=True)
class _TokenPipeline:
    """Bounded lazy per-token schedule for RunSeqFnRolling / generic RunSeq.

    ``PipelineProgram`` chains only same-named stages, which cannot express
    "token t waits store t-2 while overlapping store t-1".  This small Program
    keeps the same bounded-issue discipline while compacting long periodic
    chunks (``length >= CONV_COMPACT_MIN_LEN``) into a callback-free static DAG
    of literal head/tail tokens plus one coupled steady-state stage.
    """

    length: int
    rolling: bool
    load: Mapping[str, float]
    compute: Mapping[str, float]
    store: Mapping[str, float]
    placement: Mapping[str, tuple[int, ...]]
    cast: Mapping[str, float] = field(default_factory=dict)
    slots: int = TOKEN_PIPELINE_SLOTS

    def _start_literal(self, engine, name, ready, *, owner=""):
        complete = engine.future(name + ":complete")
        latest: dict[str, object] = {}
        next_tile = 0
        finished = 0

        def retire(_):
            nonlocal finished
            finished += 1
            pump()

        def issue(t):
            nonlocal next_tile
            next_tile += 1
            local = _emit_token_stages(
                self, engine, name, ready, owner, latest, t, on_retire=retire,
            )
            if t == self.length - 1:
                engine.submit(f"{name}:finish", {}, after=tuple(local),
                              placement=self.placement, owner=owner,
                              on_complete=lambda _: engine.resolve(complete))

        def pump():
            while (next_tile < self.length
                   and next_tile - finished < self.slots):
                issue(next_tile)

        engine.submit(name + ":begin", after=(ready,), owner=owner,
                      on_complete=lambda _: pump())
        return complete

    def _start_compacted(self, engine, name, ready, *, owner="", mid_periods: int, period4: float):
        mid_tokens = CONV_COMPACT_PERIOD * mid_periods
        tail_start = CONV_COMPACT_HEAD_TOKENS + mid_tokens
        span_ms = mid_periods * period4
        mid_work: dict[str, float] = {}
        sources = (self.load, self.compute, self.store) + (() if self.rolling else (self.cast,))
        for src in sources:
            for key, value in src.items():
                if value:
                    mid_work[key] = mid_work.get(key, 0.0) + float(value) * mid_tokens
        caps = {key: (val / span_ms) for key, val in mid_work.items()} if span_ms > 0.0 else {}

        latest: dict[str, object] = {}
        for t in range(CONV_COMPACT_HEAD_TOKENS):
            _emit_token_stages(self, engine, name, ready, owner, latest, t)
        frontier = tuple(dict.fromkeys(latest.values()))
        mid_ev = engine.submit(
            f"{name}:steady_mid", mid_work, after=frontier,
            placement=self.placement, caps=caps, coupled=True, owner=owner,
        )
        for key in list(latest):
            latest[key] = mid_ev
        last_local: list[object] = [mid_ev]
        for t in range(tail_start, self.length):
            last_local = _emit_token_stages(self, engine, name, ready, owner, latest, t)
        return engine.submit(
            f"{name}:finish", {}, after=tuple(last_local),
            placement=self.placement, owner=owner,
        )

    def start(self, engine, name, ready, *, owner=""):
        if self.length < 0 or self.slots < 1:
            raise ValueError(
                "token_pipeline_requires_nonnegative_length_and_positive_slots")
        if not self.length:
            return engine.submit(
                f"{name}:empty", {}, after=(ready,),
                placement=self.placement, owner=owner,
            )
        if self.length >= CONV_COMPACT_MIN_LEN and self.slots >= TOKEN_PIPELINE_SLOTS:
            mid_periods = (
                self.length - (CONV_COMPACT_HEAD_TOKENS + CONV_COMPACT_TAIL_MIN_TOKENS)
            ) // CONV_COMPACT_PERIOD
            if mid_periods >= CONV_COMPACT_MIN_PERIODS:
                used_keys = sorted(set(self.load) | set(self.compute) | set(self.store) | set(self.cast))
                res_spec = tuple(
                    (
                        k,
                        float(engine.resources[k].rate),
                        int(engine.resources[k].units),
                        str(engine.resources[k].policy),
                        len(self.placement.get(k, ())),
                    )
                    for k in used_keys if k in engine.resources
                )
                period4 = _cached_conv_period4(
                    self.rolling,
                    self.slots,
                    tuple(sorted(self.load.items())),
                    tuple(sorted(self.compute.items())),
                    tuple(sorted(self.store.items())),
                    tuple(sorted(self.cast.items())),
                    res_spec,
                )
                if period4 is not None and period4 > 0.0:
                    return self._start_compacted(
                        engine, name, ready, owner=owner,
                        mid_periods=mid_periods, period4=period4,
                    )
        return self._start_literal(engine, name, ready, owner=owner)


def _chunk_kernel(context, chunk, placement):
    load = _bind(context, _work(hbm=chunk.per_token_load,
                               mte2=chunk.per_token_load))
    stages = [
        Stage("history", _bind(context, chunk.history), placement=placement),
        Stage("first_load", load, after=("history",), placement=placement),
    ]
    inputs = ("first_load",)
    if chunk.restore.resources or chunk.restore.matrix:
        # InitRingSeqSplit waits for history, then issues the current input.
        # RunSeqFnRolling restores partials before waiting for that input.
        stages.append(Stage("restore", _bind(context, chunk.restore),
                            after=("history",), placement=placement))
        inputs += ("restore",)
    pipeline = _TokenPipeline(
        length=chunk.length, rolling=chunk.rolling,
        load=load,
        compute=_bind(context, chunk.compute),
        cast=_bind(context, chunk.cast),
        store=_bind(context, _work(hbm=chunk.per_token_store,
                                 mte3=chunk.per_token_store)),
        placement=placement,
    )
    stages.append(Stage("pairs", after=inputs, program=pipeline))
    previous = "pairs"
    if chunk.state_load.resources or chunk.state_load.matrix:
        stages.append(Stage("state_load", _bind(context, chunk.state_load),
                            after=(previous,), placement=placement))
        previous = "state_load"
    if chunk.state_store.resources or chunk.state_store.matrix:
        stages.append(Stage("state_store", _bind(context, chunk.state_store),
                            after=(previous,), placement=placement))
        previous = "state_store"
    stages.append(Stage("drain", after=(previous,)))
    return KernelProgram(tuple(stages))


def _block_kernel(context, block, owners):
    placement = _placement(context, block.index, owners)
    stages = [
        Stage("weight_load", _bind(context, block.weight_load), placement=placement),
        Stage("weight_convert", _bind(context, block.weight_convert),
              after=("weight_load",), placement=placement),
        Stage("meta", _bind(context, block.metadata), after=("weight_convert",),
              placement=placement),
    ]
    previous = "meta"
    for index, chunk in enumerate(block.chunks):
        name = f"chunk{index}"
        stages.append(Stage(name, after=(previous,),
                            program=_chunk_kernel(context, chunk, placement)))
        previous = name
    stages.append(Stage("done", after=(previous,)))
    return KernelProgram(tuple(stages))


def _snapshot_program(context, cfg, host, block, owners):
    """Reuse the source's one temporary UB row only after its previous store."""
    placement = _placement(context, block.index, owners)
    first, stride = block.token_tile, host["token_block_count"]
    count = max(0, (cfg.batch - first - 1) // stride + 1)

    def sequence(index):
        seq = first + index * stride
        flag_bytes = _dtype_width(cfg.dtypes.has_initial_state)
        stages = [Stage("flag", _bind(context, _work(
            hbm=flag_bytes, mte2=flag_bytes)), placement=placement)]
        if not cfg.has_init(seq):
            return tuple(stages)
        previous = "flag"
        cache_idx = seq
        if cfg.cache_index_present:
            cache_idx = _cache_value(cfg, seq)
            cache_bytes = _dtype_width(cfg.dtypes.cache_indices)
            stages.append(Stage("cache", _bind(context, _work(
                hbm=cache_bytes, mte2=cache_bytes)),
                after=(previous,), placement=placement))
            previous = "cache"
        if _cache_skipped(cfg, cache_idx):
            return tuple(stages)
        size = block.base_size * 2
        for slot in range(cfg.history_count):
            read, write = f"read{slot}", f"write{slot}"
            stages.append(Stage(read, _bind(context, _work(hbm=size, mte2=size)),
                                after=(previous,), placement=placement))
            stages.append(Stage(write, _bind(context, _work(hbm=size, mte3=size)),
                                after=(read,), placement=placement))
            previous = write
        return tuple(stages)

    return PipelineProgram(count, 1, sequence)


def build(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    cfg = _config_from_op(op)
    owners, ub_bytes = _resolve_cores_and_ub(geometry, None, None)
    if ub_bytes < UB_RESIDENT_BYTES:
        raise ValueError(
            f"UB capacity {ub_bytes} is smaller than the source resident "
            f"buffers ({UB_RESIDENT_BYTES} bytes)")
    host = _host_tiling(cfg, owners, ub_bytes)
    blocks = _prepare(cfg, host)
    work = split_reads(_sum(_block_total(block) for block in blocks) + _sync_work(host),
                       0, geometry.resource_units)
    workspace = host["has_init_state_workspace"]

    def program(context):
        if int(context.resource_units.get("vector", context.core_count)) != owners:
            raise ValueError("execution vector ownership differs from build geometry")
        stages = []
        allocations = {}
        gm_workspace = None
        gm_grant = None
        if workspace and "workspace" in context.pool_names:
            gm_workspace = context.allocate(
                "native_conv_gm_workspace", "workspace",
                int(host["workspace_bytes"]),
                unit_group(0, owners,
                           context.pool_units.get("workspace", owners)))
            stages.append(Stage("gm_grant", acquire=(gm_workspace,)))
            gm_grant = "gm_grant"
        for block in blocks:
            items = ()
            if "ub" in context.pool_names:
                units = unit_group(block.index, owners,
                                   context.pool_units.get("ub", owners))
                items = (context.allocate(f"native_conv_ub{block.index}", "ub",
                                          UB_RESIDENT_BYTES, units),)
            allocations[block.index] = items
            stages.append(Stage(f"reserve{block.index}", acquire=items,
                                after=(gm_grant,) if gm_grant else ()))
        prefetch_tails = []
        if workspace:
            for block in blocks:
                name = f"prefetch{block.index}"
                stages.append(Stage(
                    name, after=(f"reserve{block.index}",),
                    program=_snapshot_program(context, cfg, host, block, owners)))
                prefetch_tails.append(name)
            stages.append(Stage("init_sync", _bind(context, _sync_work(host)),
                                after=tuple(prefetch_tails)))
            sync_name = "init_sync"
        else:
            sync_name = None
        body_names = []
        for block in blocks:
            name = f"body{block.index}"
            after = (sync_name,) if sync_name else (f"reserve{block.index}",)
            stages.append(Stage(name, after=after,
                                program=_block_kernel(context, block, owners)))
            release = f"release{block.index}"
            stages.append(Stage(release, after=(name,), free=allocations[block.index]))
            body_names.append(release)
        if gm_workspace is not None:
            stages.append(Stage("gm_release", after=tuple(body_names),
                                free=(gm_workspace,)))
            body_names.append("gm_release")
        stages.append(Stage("complete", after=tuple(body_names)))
        return KernelProgram(tuple(stages))

    return OperatorPlan(work, program)


def source_ledger(op_or_params, geometry) -> dict:
    """Return deterministic source accounting without executing the program."""
    if isinstance(op_or_params, OpSpec):
        cfg = _config_from_op(op_or_params)
    else:
        cfg = _parse_params(op_or_params)
    if geometry is None:
        raise ModelingError("causal_conv_requires_explicit_geometry")
    owners, ub_bytes = _resolve_cores_and_ub(geometry, None, None)
    if ub_bytes < UB_RESIDENT_BYTES:
        raise ValueError(
            f"UB capacity {ub_bytes} is smaller than the source resident "
            f"buffers ({UB_RESIDENT_BYTES} bytes)")
    host = _host_tiling(cfg, owners, ub_bytes)
    blocks = _prepare(cfg, host)
    work = _sum(_block_total(block) for block in blocks) + _sync_work(host)
    chunks = [chunk for block in blocks for chunk in block.chunks]
    output_bytes = sum(chunk.length * chunk.per_token_store for chunk in chunks)
    state_bytes = sum(chunk.state_store.resources.get("mte3", 0) for chunk in chunks)
    state_read_bytes = sum(chunk.state_load.resources.get("mte2", 0)
                           for chunk in chunks)
    prefetch_read = sum(block.prefetch_payload_bytes for block in blocks)
    prefetch_write = prefetch_read
    qsl_bytes = sum(block.qsl_bytes for block in blocks)
    cache_bytes = sum(block.cache_bytes + block.prefetch_cache_bytes for block in blocks)
    his_bytes = sum(block.his_bytes + block.prefetch_his_bytes for block in blocks)
    active = [seq for seq in range(cfg.batch)
              if cfg.qsl[seq + 1] > cfg.qsl[seq]
              and not _cache_skipped(
                  cfg, _cache_value(cfg, seq) if cfg.cache_index_present else seq)]
    return {
        "work": dict(sorted(split_reads(work, 0, geometry.resource_units).resources.items())),
        "host_tiling": host,
        "block_dim": host["block_dim"],
        "aiv_count": owners,
        "ub_bytes": ub_bytes,
        "ub_resident_bytes": UB_RESIDENT_BYTES,
        "workspace_bytes": host["workspace_bytes"],
        "sync_workspace_bytes": host["sync_workspace_bytes"],
        "snapshot_workspace_bytes": host["snapshot_workspace_bytes"],
        "prefetch_payload_read_bytes": prefetch_read,
        "prefetch_payload_write_bytes": prefetch_write,
        "prefetch_copied_channel_blocks": sum(
            block.prefetch_copied_sequences for block in blocks),
        "output_write_bytes": output_bytes,
        "state_read_bytes": state_read_bytes,
        "state_write_bytes": state_bytes,
        "metadata_qsl_bytes": qsl_bytes,
        "metadata_cache_bytes": cache_bytes,
        "metadata_has_initial_state_bytes": his_bytes,
        "chunk_count": len(chunks),
        "segment_count": sum(_ceil_div(chunk.length, PAIR_TOKENS)
                             for chunk in chunks),
        "active_sequences": tuple(active),
        "has_initial_state_flag": host["has_initial_state"],
        "has_conv_states": cfg.has_conv_states,
        "rolling_register_projection": {
            "math": "cann_default_ftz_true_intrinsic",
            "boundary": "source_capacity_projection_not_compiled_schedule",
            "setup_and_generic_bias": "analytical_vector_work",
        },
        "blocks": tuple({
            "index": block.index, "token_tile": block.token_tile,
            "base_index": block.base_index, "channel_start": block.channel_start,
            "base_size": block.base_size, "token_start": block.token_start,
            "token_end": block.token_end, "chunks": len(block.chunks),
        } for block in blocks),
    }
