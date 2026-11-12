"""Narrow program bridge for GDN gating only.

[Role] Execute the authoritative gdn_decode_programs.gating_plan program on the
common engine so its owner/order and BF16 return-queue dependencies survive
ordinary piecewise pricing.
[Invariants] The plan is built from the selected source geometry and queue
parameters; work and duration are never taken from target measurements. Cache
keys include the full active geometry and execution/queue configuration.
[Timeouts & Locks] Pure bounded resource scheduling; LRU-bounded by geometry.
[Side-Effect Policy] Builds private engines only; no global mutation or I/O.
"""
from __future__ import annotations

from functools import lru_cache
from fractions import Fraction
import math
from types import SimpleNamespace

from .engine import Engine, Resource
from .gdn_decode_programs import conv_plan, gating_plan
from .program import RecipeContext
from .piecewise.compiler import NativeProgram

def _queue_key(queue):
    return tuple(sorted((k, v) for k, v in queue.items()))


def _gating_execute(rates, units, queue, rows, heads):
    """Execute one actual gating plan; no fast-period logic."""
    resources = [Resource(name, float(rate), "work", "fair",
                          int(units.get(name, 1)))
                 for name, rate in rates.items()]
    engine = Engine(resources)
    names = {role: role for role in rates}
    context = RecipeContext(
        op=None, work={}, resource_names=names, rates=rates,
        core_count=max(1, int(units.get("vector", 1))),
        resource_units=units, pool_names={},
        store_queues=({"scalar_store_issue": dict(queue)} if queue else {}),
    )
    op = SimpleNamespace(
        local_shape=(int(rows), int(heads)),
        metadata={"beta_store": ({"resource": "scalar_store_issue",
                                  "from_hardware": True} if queue else {})},
    )
    geometry = SimpleNamespace(
        resource_units=units,
        compute_units=max(1, int(units.get("vector", 1))),
    )
    plan = gating_plan(op, geometry)
    ready = engine.submit("ready", release=0)
    event = plan.program(context).start(engine, "gdn_gating", ready)
    engine.advance(until=(event,))
    busy = tuple(sorted((name, Fraction.from_float(value))
                        for name, value in engine.busy_ms.items() if value))
    return Fraction.from_float(event.time - ready.time), busy


@lru_cache(maxsize=64)
def _duration_ms(rates_key, units_key, rows, heads, queue_key, floor_ms):
    rates = dict(rates_key)
    units = dict(units_key)
    queue = dict(queue_key) if queue_key is not None else None
    op = SimpleNamespace(
        local_shape=(int(rows), int(heads)),
        metadata={"beta_store": ({"resource": "scalar_store_issue",
                                  "from_hardware": True} if queue else {})},
    )
    geometry = SimpleNamespace(
        resource_units=units,
        compute_units=max(1, int(units.get("vector", 1))),
    )
    full_work = dict(gating_plan(op, geometry).work.resources)
    programs = int(units.get("vector", 1))
    tile_rows = 64
    tile_columns = 8
    iterations = (rows // (tile_rows * programs)
                  if programs > 0 and rows % (tile_rows * programs) == 0 else 0)
    stride_bytes = tile_rows * int(heads) * 2
    granule = int(queue.get("granule_bytes", 0) or 0) if queue else 0
    # The address phase repeats after `period` complete 64-row/8-head tile
    # iterations.  A period plan and its true remainder are executed directly;
    # only the repeated period count is multiplied.  For no queue an aligned
    # granule is period one.
    period = (1 if not queue else
              (granule // math.gcd(stride_bytes, granule)
               if granule > 0 else 0))
    # Only complete period cycles are certified.  A nonzero tail after a
    # multi-period prefix has a distinct post-warmup increment (the direct
    # full-engine control exposed it at k=9 for granule 4096), so it stays on
    # the original complete plan rather than being extrapolated.
    fast = (iterations > 1 and programs == int(units.get("vector", 1))
            and heads % tile_columns == 0 and period > 0
            and period <= 4 and iterations >= period
            and iterations % period == 0)
    if fast:
        cycles = iterations // period
        period_elapsed, period_busy = _gating_execute(
            rates, units, queue, tile_rows * programs * period, heads)
        elapsed = cycles * period_elapsed
        busy = {name: value * cycles for name, value in period_busy}
    else:
        elapsed, busy_tuple = _gating_execute(
            rates, units, queue, int(rows), int(heads))
        busy = dict(busy_tuple)
    busy = tuple(sorted((name, value) for name, value in busy.items() if value))
    if busy:
        peak = max(value for _, value in busy)
        dominant = next(name for name, value in busy if value == peak)
        tied = tuple(name for name, value in busy if value == peak)
    else:
        dominant, tied = "", ()
    if floor_ms:
        elapsed = max(elapsed, Fraction.from_float(float(floor_ms)))
    return elapsed, dominant, tied, busy, full_work

def gating_program_result(supply, rows: int, heads: int, *, queue=None):
    """Return the native plan duration in ms for this source geometry.

    ``queue=None`` means use the uniform queue carried by the real PipeSupply;
    a missing queue selects gating_plan's no-queue MTE3 BF16 path.  There is no
    cross-hardware default.
    """
    rates = tuple(sorted((str(k), float(v)) for k, v in supply.rates))
    units = tuple(sorted((str(k), int(v)) for k, v in supply.resource_units))
    if queue is None:
        declared = dict(supply.store_queues).get("scalar_store_issue")
        queue = dict(declared) if declared else None
    elif queue is False:
        queue = None
    if queue:
        queue = dict(queue)
        if "element_bytes" not in queue:
            queue["element_bytes"] = 2
    q = _queue_key(queue) if queue else None
    floor = dict(supply.invocations).get("vector", 0.0)
    return _duration_ms(rates, units, int(rows), int(heads), q, floor)


def gating_duration_ms(supply, rows: int, heads: int, *, queue=None):
    """Return only the native plan elapsed time in ms."""
    return gating_program_result(supply, rows, heads, queue=queue)[0]


@lru_cache(maxsize=64)
def _conv_duration_ms(rates_key, units_key, rows, width, lengths, kernel,
                      speculative, history, floor_ms):
    rates = dict(rates_key)
    units = dict(units_key)
    names = {role: role for role in rates}
    resources = [Resource(name, float(rate), "work", "fair",
                          int(units.get(name, 1))) for name, rate in rates.items()]
    engine = Engine(resources)
    context = RecipeContext(
        op=None, work={}, resource_names=names, rates=rates,
        core_count=max(1, int(units.get("vector", 1))),
        resource_units=units, pool_names={}, store_queues={},
    )
    op = SimpleNamespace(
        local_shape=(int(rows), int(width)), dtype="bf16",
        metadata={"physical_sequence_tokens": tuple(lengths),
                  "sequence_tokens": tuple(lengths),
                  "kernel_width": int(kernel), "speculative": bool(speculative),
                  "state_history_slots": int(history), "bias": False,
                  "cache_index_row_stride": 1, "cache_index_dtype": "int32"},
    )
    geometry = SimpleNamespace(resource_units=units,
                               compute_units=max(1, int(units.get("vector", 1))))
    plan = conv_plan(op, geometry)
    program = plan.program(context)
    ready = engine.submit("ready", release=0)
    event = program.start(engine, "gdn_conv", ready)
    engine.advance(until=(event,))
    elapsed = Fraction.from_float(event.time - ready.time)
    if floor_ms:
        elapsed = max(elapsed, Fraction.from_float(float(floor_ms)))
    busy = tuple(sorted((name, Fraction.from_float(value))
                        for name, value in engine.busy_ms.items() if value))
    if busy:
        peak = max(value for _, value in busy)
        dominant = next(name for name, value in busy if value == peak)
        tied = tuple(name for name, value in busy if value == peak)
    else:
        dominant, tied = "", ()
    return (elapsed, dominant, tied, busy, dict(plan.work.resources),
            NativeProgram(program, tuple(sorted(plan.work.resources.items())),
                          elapsed))


def conv_program_result(supply, rows: int, width: int, lengths, kernel: int,
                        speculative: bool, history: int):
    """Native conv owner/dependency program for the selected GDN caller."""
    rates = tuple(sorted((str(k), float(v)) for k, v in supply.rates))
    units = tuple(sorted((str(k), int(v)) for k, v in supply.resource_units))
    floor = dict(supply.invocations).get("vector", 0.0)
    return _conv_duration_ms(rates, units, int(rows), int(width),
                             tuple(int(q) for q in lengths), int(kernel),
                             bool(speculative), int(history), floor)

@lru_cache(maxsize=64)
def _recurrent_duration_ms(rates_key, units_key, lengths, heads, keys, dk, dv,
                           floor_ms, buffers_key, pool_units_key):
    from . import gdn_recurrent
    rates = dict(rates_key)
    units = dict(units_key)
    names = {role: role for role in rates}
    resources = [Resource(name, float(rate), "work", "fair",
                          int(units.get(name, 1))) for name, rate in rates.items()]
    pools = dict(buffers_key)
    pool_units = dict(pool_units_key)
    engine = Engine(resources, pools=pools, pool_units=pool_units)
    context = RecipeContext(
        op=None, work={}, resource_names=names, rates=rates,
        core_count=max(1, int(units.get("vector", 1))),
        resource_units=units,
        pool_names={name: name for name in pools},
        pool_units=pool_units, pool_capacities=pools,
        store_queues={},
    )
    op = SimpleNamespace(dtype="bf16", metadata={
        "sequence_tokens": tuple(int(q) for q in lengths),
        "heads": int(heads), "key_heads": int(keys),
        "dk": int(dk), "dv": int(dv), "state_dtype": "fp32",
    })
    geometry = SimpleNamespace(resource_units=units,
                               compute_units=max(1, int(units.get("vector", 1))))
    plan = gdn_recurrent.build(op, geometry)
    program = plan.program(context)
    ready = engine.submit("ready", release=0)
    event = program.start(engine, "gdn_recurrent", ready)
    engine.advance(until=(event,))
    elapsed = Fraction.from_float(event.time - ready.time)
    if floor_ms:
        elapsed = max(elapsed, Fraction.from_float(float(floor_ms)))
    busy = tuple(sorted((name, Fraction.from_float(value))
                        for name, value in engine.busy_ms.items() if value))
    if busy:
        peak = max(value for _, value in busy)
        dominant = next(name for name, value in busy if value == peak)
        tied = tuple(name for name, value in busy if value == peak)
    else:
        dominant, tied = "", ()
    return elapsed, dominant, tied, busy, dict(plan.work.resources)


def recurrent_program_result(supply, lengths, heads, keys, dk, dv):
    """Native arch35 FP32-state recurrent owner/dependency program."""
    rates = tuple(sorted((str(k), float(v)) for k, v in supply.rates))
    units = tuple(sorted((str(k), int(v)) for k, v in supply.resource_units))
    buffers = tuple(sorted((str(k), int(v)) for k, v in supply.buffers))
    pool_units = tuple(sorted((str(k), int(v)) for k, v in supply.pool_units))
    import json
    floor = dict(supply.invocations).get("vector", 0.0)
    return _recurrent_duration_ms(rates, units, tuple(int(q) for q in lengths),
                                  int(heads), int(keys), int(dk), int(dv), floor,
                                  buffers, pool_units)

@lru_cache(maxsize=32)
def _prefill_conv_duration_ms(rates_key, units_key, buffers_key, pool_units_key,
                              modes_key, lengths, width, kernel, state_rows,
                              draft_tokens, initial_state):
    from .operators import native_causal_conv_prefill
    rates = dict(rates_key)
    units = dict(units_key)
    modes = dict(modes_key)
    pools = dict(buffers_key)
    pool_units = dict(pool_units_key)
    names = {role: role for role in rates}
    resources = [Resource(name, float(rate), "work", "fair",
                          int(units.get(name, 1))) for name, rate in rates.items()]
    engine = Engine(resources, pools=pools, pool_units=pool_units)
    context = RecipeContext(
        op=None, work={}, resource_names=names, rates=rates,
        core_count=max(1, int(units.get("vector", 1))),
        resource_units=units, pool_names={name: name for name in pools},
        pool_units=pool_units, pool_capacities=pools, store_queues={},
        service_modes=modes,
    )
    op = native_causal_conv_prefill.make_op({
        "lengths": tuple(int(q) for q in lengths), "dim": int(width),
        "width": int(kernel), "dtype": "bf16", "activation": "silu",
        "bias": False, "metadata_placement": "device",
        "has_initial_state": tuple(bool(x) for x in initial_state),
        "cache_indices": tuple(range(1, len(lengths) + 1)),
        "state_rows": int(state_rows),
        "state_history_slots": int(kernel) - 1 + int(draft_tokens),
        "null_block_id": 0,
    })
    geometry = SimpleNamespace(
        resource_units=units, compute_units=max(1, int(units.get("vector", 1))),
        buffers=pools,
    )
    plan = native_causal_conv_prefill.build(op, geometry)
    program = plan.program(context)
    ready = engine.submit("ready", release=0)
    event = program.start(engine, "gdn_prefill_conv", ready)
    engine.advance(until=(event,))
    elapsed = Fraction.from_float(event.time - ready.time)
    busy = tuple(sorted((name, Fraction.from_float(value))
                        for name, value in engine.busy_ms.items() if value))
    if busy:
        peak = max(value for _, value in busy)
        dominant = next(name for name, value in busy if value == peak)
        tied = tuple(name for name, value in busy if value == peak)
    else:
        dominant, tied = "", ()
    return (elapsed, dominant, tied, busy, dict(plan.work.resources),
            NativeProgram(program, tuple(sorted(plan.work.resources.items())),
                          elapsed))


def prefill_conv_program_result(supply, lengths, width, kernel, state_rows,
                                draft_tokens=0, initial_state=()):
    """Native chunk-prefill causal convolution program (FN varlen)."""
    rates = tuple(sorted((str(k), float(v)) for k, v in supply.rates))
    units = tuple(sorted((str(k), int(v)) for k, v in supply.resource_units))
    buffers = tuple(sorted((str(k), int(v)) for k, v in supply.buffers))
    pool_units = tuple(sorted((str(k), int(v)) for k, v in supply.pool_units))
    modes = tuple(sorted((str(name), (str(role), float(rate)))
                         for name, (role, rate) in supply.service_modes))
    return _prefill_conv_duration_ms(
        rates, units, buffers, pool_units, modes, tuple(int(q) for q in lengths),
        int(width), int(kernel), int(state_rows), int(draft_tokens),
        tuple(bool(x) for x in initial_state))

@lru_cache(maxsize=32)
def _pf3_duration_ms(rates_key, units_key, buffers_key, pool_units_key,
                     modes_key, compute_units, peaks_key,
                     lengths, heads, keys, dk, dv, chunk):
    from .operators import gdn_prepare, gdn_state, gdn_output
    rates = dict(rates_key)
    units = dict(units_key)
    units.setdefault("cube", int(compute_units))
    buffers = dict(buffers_key)
    pool_units = dict(pool_units_key)
    modes = dict(modes_key)
    names = {role: role for role in set(rates) | set(modes)}
    resources = [Resource(name, float(rate), "work", "fair",
                          int(units.get(name, 1))) for name, rate in rates.items()]
    engine = Engine(resources, pools=buffers, pool_units=pool_units)
    context = RecipeContext(
        op=None, work={}, resource_names=names,
        rates={**rates, **{name: rate for name, (_, rate) in modes.items()}},
        core_count=max(1, int(compute_units)),
        resource_units=units, pool_names={name: name for name in buffers},
        pool_units=pool_units, pool_capacities=buffers, store_queues={},
        service_modes=modes, compute_peaks=dict(peaks_key),
    )
    lengths = tuple(int(q) for q in lengths)
    params = {"lengths": lengths, "heads": int(heads), "key_heads": int(keys),
              "dk": int(dk), "dv": int(dv), "chunk": int(chunk),
              "dtype": "bf16", "gate_dtype": "fp32", "use_exp2": True}
    prepare_op = gdn_prepare.make_op({**params,
        "use_qk_l2norm_in_kernel": True, "use_gate_in_kernel": False,
        "use_beta_sigmoid_in_kernel": False, "output_a": True,
        "output_beta": False})
    state_op = gdn_state.make_op({**params,
        "use_initial_state": True, "output_final_state": True,
        "state_v_first": True})
    output_op = gdn_output.make_op({**params, "state_v_first": True})
    geometry = SimpleNamespace(
        resource_units=units, compute_units=max(1, int(compute_units)),
        buffers=buffers,
    )
    plans = (gdn_prepare.build(prepare_op, geometry),
             gdn_state.build(state_op, geometry),
             gdn_output.build(output_op, geometry))
    work = {}
    for plan in plans:
        for key, value in plan.work.resources.items():
            work[key] = work.get(key, 0) + value
    # Uniform long sequences repeat one full pack per core.  The source
    # prepare/state/output pipelines are periodic once every core owns a pack,
    # so their elapsed and per-stage times are affine in the pack rounds.  Two
    # canonical runs (one and two packs per core) determine that line exactly;
    # work above is still built from the full geometry, never sampled.
    ratio = int(heads) // int(keys) if int(keys) else 0
    tpp = 3 if ratio == 3 else 4
    base_packs = int(heads) / float(tpp) if tpp else 0.0
    cores = int(compute_units)
    rounds = (cores / base_packs) if base_packs else 0.0
    stride = int(chunk) * int(round(rounds)) if rounds > 0 else 0
    # Ratio-4 packs do not preserve the per-resource busy split through the
    # same one-/two-pack affine (only elapsed/stages were invariant in the
    # direct full-engine controls), so they remain on the full path.  Ratio-3
    # is the source-compatible GDN GQA geometry with matched 3-task packs.
    fast = (len(lengths) == 1 and ratio == 3 and stride > 0
            and abs(rounds - round(rounds)) < 1e-12
            and lengths[0] % stride == 0 and lengths[0] // stride > 2)
    if fast:
        rounds_count = lengths[0] // stride
        one = _pf3_duration_ms(
            rates_key, units_key, buffers_key, pool_units_key, modes_key,
            compute_units, peaks_key, (int(stride),), heads, keys, dk, dv, chunk)
        two = _pf3_duration_ms(
            rates_key, units_key, buffers_key, pool_units_key, modes_key,
            compute_units, peaks_key, (int(stride) * 2,), heads, keys, dk, dv, chunk)
        delta = two[0] - one[0]
        elapsed = one[0] + (rounds_count - 1) * delta
        stage_ms = tuple(one[5][index] + (rounds_count - 1) *
                         (two[5][index] - one[5][index])
                         for index in range(len(one[5])))
        busy = tuple(sorted((name, value + (rounds_count - 1) *
                             (dict(two[3]).get(name, value) - value))
                            for name, value in one[3]))
        if busy:
            peak = max(value for _, value in busy)
            dominant = next(name for name, value in busy if value == peak)
            tied = tuple(name for name, value in busy if value == peak)
        else:
            dominant, tied = "", ()
        return elapsed, dominant, tied, busy, work, stage_ms
    ready = engine.submit("ready", release=0)
    previous = ready
    stage_ms = []
    for plan in plans:
        start = previous.time if previous.time is not None else engine.now
        program = plan.program(context)
        event = program.start(engine, "pf3", previous)
        engine.advance(until=(event,))
        stage_ms.append(Fraction.from_float(event.time - start))
        previous = event
    elapsed = Fraction.from_float(previous.time - ready.time)
    busy = tuple(sorted((name, Fraction.from_float(value))
                        for name, value in engine.busy_ms.items() if value))
    if busy:
        peak = max(value for _, value in busy)
        dominant = next(name for name, value in busy if value == peak)
        tied = tuple(name for name, value in busy if value == peak)
    else:
        dominant, tied = "", ()
    return elapsed, dominant, tied, busy, work, tuple(stage_ms)


def pf3_program_result(supply, lengths, heads, keys, dk=128, dv=128, chunk=64):
    """Three native PF3 ops: prepare -> state -> output, sequentially."""
    rates = tuple(sorted((str(k), float(v)) for k, v in supply.rates))
    units = tuple(sorted((str(k), int(v)) for k, v in supply.resource_units))
    buffers = tuple(sorted((str(k), int(v)) for k, v in supply.buffers))
    pool_units = tuple(sorted((str(k), int(v)) for k, v in supply.pool_units))
    modes = tuple(sorted((str(name), (str(role), float(rate)))
                         for name, (role, rate) in supply.service_modes))
    peaks = tuple(sorted((str(k), float(v)) for k, v in supply.compute_peaks))
    return _pf3_duration_ms(rates, units, buffers, pool_units, modes,
                            int(supply.compute_units), peaks,
                            tuple(int(q) for q in lengths),
                            int(heads), int(keys),
                            int(dk), int(dv), int(chunk))
