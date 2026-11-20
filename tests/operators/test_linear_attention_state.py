"""State ownership, checkpoint publication, and on-chip residency.

[Role] Exercise GDN and KDA state plans through their source and governed paths.
[Invariants] Each active sequence owns its work, state slots, and commit frontier.
[Timeouts & Locks] Bounded local planning and simulation.
[Side-Effect Policy] Read-only hardware profiles and in-memory state.
"""
from dataclasses import replace
from itertools import accumulate, groupby

import pytest

from _support.axes import REPRESENTATIVE_SKUS
from _support.hardware import machine
from _support.linear_cases import execute_one, gdn_decode_source_case as source_case, kda_hardware_case
from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.attention import AttentionPolicy
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.engine import Engine
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.family_kda import IMPLEMENTATION as GATES
from mindie_sim.prediction.family_kda_state import IMPLEMENTATION as STATE
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.gdn_decode_programs import CONV, conv_plan
from mindie_sim.prediction.mechanism import Geometry, PlannedRecipe
from mindie_sim.prediction.operators import gdn_state, kda_recurrent, kda_recurrent_arch22, kda_state
from mindie_sim.prediction.operators.contracts import OperatorPlanningContext
from mindie_sim.prediction.operators.kda_state_selection import resolve
from mindie_sim.prediction.operators.linear_attention import (
    LinearAttentionContract, analyze_state_residency, resolve_operator_plan, select_flash_kda_template,
)
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.recurrent import linear
from mindie_sim.prediction.results import OpOccurrence
from mindie_sim.prediction.scheduling import FragmentConductor
from mindie_sim.prediction.runtime.tile_solver import TileInfeasibleError
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.case import Software
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import IterationWork, OpSpec
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.workload import IterationPlan, RequestView
from mindie_sim.structure_subgraphs import subgraph_contract

pytestmark = pytest.mark.tier0

K = V = 128
CHUNK = 64
WIDTH = (2 * 2 + 6) * K
PREFILL = "gdn_prefill_arch35"


def params(lengths=(64,), heads=48, key_heads=16, **overrides):
    value = {
        "lengths": list(lengths), "heads": heads, "key_heads": key_heads,
        "dk": K, "dv": V, "chunk": CHUNK, "dtype": "bf16",
    }
    value.update(overrides)
    return value


def brute_force_owner(lengths, heads, cores):
    """Assign each flattened head task to a core with the contiguous ceiling."""
    total = len(lengths) * heads
    per_core = (total + cores - 1) // cores
    return [min(index // per_core, cores - 1) for index in range(total)]


def oracle_units(lengths, heads, cores):
    """Reconstruct contiguous units from the flattened head-to-core map."""
    owners = brute_force_owner(lengths, heads, cores)
    units = []
    for (core, seq), group in groupby(enumerate(owners), lambda row: (row[1], row[0] // heads)):
        local = [index % heads for index, _ in group]
        units.extend((core, seq, local[i], tuple(local[i:i+4])) for i in range(0, len(local), 4))
    return units


def oracle_work(value, cores):
    """Enumerate every head/chunk task and derive the physical ledger."""
    lengths = tuple(int(item) for item in value["lengths"])
    heads, key_heads = int(value["heads"]), int(value["key_heads"])
    use_initial = bool(value.get("use_initial_state", True))
    final = bool(value.get("output_final_state", True))
    gate_bf16 = str(value.get("gate_dtype", "bf16")) in {"bf16", "bfloat16"}
    exp2 = bool(value.get("use_exp2", False))
    group = heads // key_heads
    gate_bytes = 2 if gate_bf16 else 4
    chunks_per_sequence = [(length + CHUNK - 1) // CHUNK for length in lengths]
    matrix, mte1, fixpipe, mte2, mte3, vector = 0, 0, 0, 0, 0, 0
    local_mte2 = 0
    for _, sequence, _, hvs in oracle_units(lengths, heads, cores):
        length = lengths[sequence]
        chunks = chunks_per_sequence[sequence]
        kh_count = len({hv // group for hv in hvs})
        for chunk in range(chunks):
            valid = min(CHUNK, length - chunk * CHUNK)
            rows = (valid + 15) // 16 * 16
            first, last = chunk == 0, chunk == chunks - 1
            stage0 = use_initial or not first
            stage2 = final or not last
            for _ in hvs:
                if first and use_initial:
                    mte2 += K * V * 4
                    mte3 += K * V * 2
                    vector += K * V
                if stage0:
                    mte2 += 2 * valid * K + 2 * K * V
                    if valid < CHUNK:
                        local_mte2 += CHUNK * K * 2
                    matrix += 2 * rows * K * V
                    for tile_rows, tile_columns in ((rows, K), (K, V)):
                        mte1 += tile_rows * tile_columns * 2
                    fixpipe += valid * V * 4
                mte2 += 2 * valid * V
                mte3 += 2 * valid * V
                vector += 3 * valid * V
                if stage2:
                    mte2 += gate_bytes * valid
                    mte3 += 2 * valid * V
                    vector += valid * V
                    per_element = 2 + (1 if gate_bf16 else 0) + (1 if exp2 else 0)
                    vector += valid * per_element + per_element
                    for tile_rows, tile_columns in ((K, rows), (rows, V)):
                        mte1 += tile_rows * tile_columns * 2
                    fixpipe += K * V * 4
                    if valid < CHUNK:
                        local_mte2 += CHUNK * V * 2
                if first and not use_initial:
                    mte3 += K * V * 2
                    vector += K * V + valid * V
            if stage2:
                mte2 += kh_count * 2 * valid * K + len(hvs) * 2 * valid * V
                if valid < CHUNK:
                    local_mte2 += kh_count * CHUNK * K * 2
                matrix += len(hvs) * 2 * K * rows * V
                if first and use_initial:
                    mte2 += len(hvs) * K * V * 4
                if not last:
                    mte3 += len(hvs) * K * V * 2
                    vector += len(hvs) * 3 * K * V
                    if first and not use_initial:
                        vector += len(hvs) * K * V
                elif final:
                    mte3 += len(hvs) * K * V * 4
                    vector += len(hvs) * 2 * K * V
                    if first and not use_initial:
                        vector += len(hvs) * K * V
    return {
        "matrix": matrix, "mte1": mte1, "fixpipe": fixpipe,
        "mte2": mte2 + local_mte2, "mte3": mte3, "hbm": mte2 + mte3,
        "local_mte2": local_mte2, "vector": vector,
    }


def plan_ledger(value, cores=32):
    op = gdn_state.make_op(value)
    return gdn_state.build(op, Geometry(cores, {"vector": 2 * cores}, {})), op


def assert_work_matches(plan, expected):
    matrix = expected.pop("matrix")
    local_reads = expected.pop("local_mte2")
    assert plan.work.matrix == ({"bf16": matrix} if matrix else {})
    for role, work in expected.items():
        assert plan.work.resources.get(role, 0) == work
    resources = plan.work.resources
    assert resources["mte2"] + resources["mte3"] - local_reads == resources["hbm"]

@pytest.mark.parametrize("lengths,heads,key_heads,cores,flags", (
    ((64,), 48, 16, 32, {}), ((63, 1, 65), 32, 8, 32, {}),
    ((65,), 4, 1, 1, {"use_initial_state": False, "output_final_state": False}),
    ((193,), 7, 7, 4, {"gate_dtype": "fp32", "use_exp2": True}),
))
def test_gdn_state_ownership_and_enumerated_ledger(lengths, heads, key_heads, cores, flags):
    groups = gdn_state.head_ownership(lengths, heads, cores)
    actual = [(g.core, unit.sequence, unit.hv_begin,
               tuple(range(unit.hv_begin, unit.hv_begin + unit.heads)))
              for g in groups for unit in g.units]
    assert actual == oracle_units(lengths, heads, cores)
    value = params(lengths, heads, key_heads, **flags)
    assert_work_matches(plan_ledger(value, cores)[0], oracle_work(value, cores))


def test_gdn_contiguous_ceiling_changes_only_shared_key_loads():
    plans = [plan_ledger(params(), cores)[0].work for cores in (16, 32, 48)]
    groups = gdn_state.head_ownership([64], 48, 32)
    assert [(g.core, g.begin, g.end) for g in groups] == [(i, 2*i, 2*i+2) for i in range(24)]
    assert plans[1].matrix == {"bf16": 201_326_592}
    assert plans[1].resources["hbm"] == 16 * 704_768 + 8 * 721_152
    for left, right in zip(plans, plans[1:]):
        assert left.matrix == right.matrix
        delta = {role: right.resources[role] - count for role, count in left.resources.items()}
        assert {role: count for role, count in delta.items() if count} == {
            "hbm": 16 * CHUNK * K * 2, "mte2": 16 * CHUNK * K * 2}


def selected(lengths, stride, *, padded_tokens=None):
    parsed, views, _ = source_case(lengths=lengths)
    kernels = {"lowerings": {"gdn_decode": {
        "implementation": "vllm_ascend_gdn_decode",
        "options": {"cache_index_row_stride": stride}}}}
    parsed = replace(parsed, runtime=replace(parsed.runtime, kernels=kernels))
    ops, _ = Frontend().body(parsed.model, parsed.runtime, views, mode="decode", tokens=padded_tokens)
    return parsed, views, ops


@pytest.mark.parametrize("lengths,padding", (((4,) * 4, 32), ((1,) * 4, 16), ((4, 2, 3, 4), 32)))
def test_first_column_reads_only_one_int32_per_request(lengths, padding):
    parsed, views, ops = selected(lengths, 4, padded_tokens=padding)
    copy = next(op for op in ops if op.recipe == "state_indices_contiguous")
    assert copy.local_shape == (4, 1)
    assert copy.mte2_bytes == copy.mte3_bytes == 16
    meta = dict(copy.metadata)
    assert meta["source_layout"]["strides"] == (4, 1)
    assert meta["input_format"] == meta["output_format"] == "int32"
    conv = next(op for op in ops if op.recipe == CONV)
    assert copy.op_id in conv.deps
    assert dict(conv.metadata)["cache_index_row_stride"] == 4
    assert all(copy.op_id not in op.deps for op in ops if op.op_id != conv.op_id)
    engine, _ = execute_one(parsed, views, copy)
    assert engine.serviced["memory@r0:s0"] == 32
    assert engine.serviced["mte2@r0:s0"] + engine.serviced["mte2_aiv@r0:s0"] == 16


@pytest.mark.parametrize("lengths,stride", (((4,) * 4, 1), ((4,), 4), ((1,), 4)))
def test_contiguous_or_singleton_index_view_needs_no_materialization(lengths, stride):
    _, _, ops = selected(lengths, stride)
    assert not any(op.recipe == "state_indices_contiguous" for op in ops)


def _ops(parsed, views, mode, tokens):
    ops, _ = Frontend().body(parsed.model, parsed.runtime, views, mode=mode,
                             tokens=tokens)
    return ops


def _by_recipe(ops, recipe):
    return next(op for op in ops if op.recipe == recipe)


def _by_family(ops, family):
    return next(op for op in ops if op.family == family)


def test_conv_skips_null_slot_compute_and_keeps_only_active_work():
    parsed, _, _ = source_case(lengths=(1,), keys=2, heads=6)
    active = RequestView("a", 1, 4096, True, "decode")
    pad = RequestView("pad", 32, 0, False, "decode", padding=True)
    active_ops = _ops(parsed, (active,), "decode", 1)
    mixed_ops = _ops(parsed, (active, pad), "decode", 33)
    active_conv = _by_recipe(active_ops, "ascend_causal_conv1d_update")
    mixed_conv = _by_recipe(mixed_ops, "ascend_causal_conv1d_update")
    geometry = Geometry.from_devices(parsed.machine.dies)
    assert dict(conv_plan(active_conv, geometry).work.resources) == \
        dict(conv_plan(mixed_conv, geometry).work.resources)
    assert mixed_conv.local_shape == (1, WIDTH)
    assert mixed_conv.padded_shape == (33, WIDTH)
    assert dict(mixed_conv.metadata)["padding_slots"] == (1,)
    work = dict(conv_plan(mixed_conv, geometry).work.resources)
    assert work["mte2_aiv"] == (4 + 3 + 1) * WIDTH * 2
    assert work["mte3"] == (1 + 3) * WIDTH * 2
    assert work["vector"] == (4 + 1 + 17) * WIDTH

    all_pad = _ops(parsed, (pad,), "decode", 32)
    pad_conv = _by_recipe(all_pad, "ascend_causal_conv1d_update")
    pad_plan = conv_plan(pad_conv, geometry)
    assert dict(pad_plan.metadata)["empty_active_sequences"] is True
    assert pad_plan.work.resources == {}
    assert pad_conv.local_shape == (0, WIDTH)


def test_mixed_prefill_and_pad_decode_charge_no_pad_recurrence():
    parsed, _, _ = source_case((1,), keys=2, heads=6)
    parsed = replace(parsed, runtime=replace(parsed.runtime, kernels={"lowerings": {"gdn_prefill": PREFILL}}))
    views = (RequestView("pf", 64, 0, True, "prefill"),
             RequestView("pad", 32, 0, False, "decode", padding=True))
    ops = _ops(parsed, views, "decode", 96)
    scan = _by_family(ops, "recurrent_vector")
    conv = _by_family(ops, "causal_conv1d_fn")
    sm = dict(scan.metadata)
    assert sm["tokens"] == 0 and sm["gdn_state_effects"] == ()
    assert scan.local_shape[0] == 0 and scan.padded_shape[0] == 32
    assert dict(conv.metadata)["lengths"] == (32, 64)
    state = _by_family(ops, "gdn_state")
    assert dict(state.metadata)["gdn_state_contract"]["direction"] == "h"

@pytest.mark.parametrize("length,advance,padding", ((1, 1, False), (4, 2, False), (4, None, False), (32, None, True)))
def test_gdn_state_checkpoints_follow_request_acceptance(length, advance, padding):
    parsed, views, _ = source_case((length,), keys=2, heads=6)
    view = replace(views[0], padding=padding)
    if length == 4 and advance:
        view = replace(view, draft_proposal_tokens=3, draft_accepted_tokens=advance-1,
                       draft_advanced_tokens=advance)
    ops = _ops(parsed, (view,), "decode", length)
    scan = _by_family(ops, "recurrent_vector")
    meta = dict(scan.metadata)
    effects = meta["gdn_state_effects"]
    conv = _by_recipe(ops, CONV)
    if padding:
        assert meta["tokens"] == meta["state_checkpoints"] == 0
        assert effects == dict(conv.metadata)["gdn_state_effects"] == ()
        assert scan.local_shape[0] == conv.local_shape[0] == 0
        assert dict(conv.metadata)["state_write_bytes"] == 0
        return
    assert [(e["action"], e["version"]) for e in effects if e["action"] in {"read", "write"}] == [
        ("read", 4096), *(("write", 4096 + n) for n in range(1, length + 1))]
    contract = meta["gdn_state_contract"]
    decisions = [(e["action"], e["version"]) for e in effects if e["action"] in {"commit", "rollback"}]
    if advance is None:
        assert decisions == [] and contract["commit_versions"] == (None,)
        assert contract["commit_policy"] == "pending_acceptance_event"
    else:
        assert decisions == [("commit", 4096 + advance),
                             *(("rollback", 4096 + n) for n in range(advance + 1, length + 1))]
        assert contract["commit_versions"] == (4096 + advance,)
        if length == 1:
            assert contract["commit_policy"] == "token_commit"


def exact(lengths=(2, 4), accepted=(1, 4), *, packed=False):
    rows = [[17, 4, -1, -1], [31, 11, 27, 2]]
    return {"kind": "exact", "prefix": list(accumulate(lengths, initial=0)),
            "index_layout": "packed" if packed else "speculative",
            "index_row_stride": 0 if packed else 4,
            "state_indices": [17, 4, 31, 11, 27, 2] if packed else rows,
            "accepted_tokens": accepted, "state_capacity": 32}


@pytest.mark.parametrize("packed", (False, True))
def test_initial_state_uses_previous_acceptance_but_checkpoints_cover_all_current_queries(packed):
    before = resolve(exact(packed=packed), (2, 4))
    after = resolve(exact(accepted=(2, 1), packed=packed), (2, 4))
    assert before["initial_slots"] == (17, 2)
    assert after["initial_slots"] == (4, 31)
    assert before["checkpoint_slots"] == after["checkpoint_slots"] == ((17, 4), (31, 11, 27, 2))
    assert -1 not in {slot for row in before["checkpoint_slots"] for slot in row}


def test_exact_duplicate_slot_is_ordered_within_sequence_but_cross_sequence_alias_is_not():
    declaration = exact()
    declaration["state_indices"][0] = [17, 17, -1, -1]
    assert resolve(declaration, (2, 4))["checkpoint_slots"][0] == (17, 17)
    declaration["state_indices"][1][0] = 17
    with pytest.raises(ModelingError, match="cross_sequence_state_alias"):
        resolve(declaration, (2, 4))


def test_old_accepted_count_cannot_exceed_current_q_even_when_index_table_has_spare_columns():
    declaration = {"kind": "exact", "prefix": [0, 1], "index_layout": "speculative",
                   "index_row_stride": 8, "state_indices": [list(range(8))],
                   "accepted_tokens": [7], "state_capacity": 8}
    for module in (kda_recurrent_arch22, kda_recurrent):
        with pytest.raises(ModelingError, match="accepted_values_outside_sequence"):
            module.make_op({"lengths": (1,), "heads": 1, "key_heads": 1,
                           "state_selection": declaration})
    declaration["accepted_tokens"] = [1]
    assert resolve(declaration, (1,))["initial_slots"] == (0,)

@pytest.mark.parametrize("acceptance", (None, ((1, 2),)))
def test_kda_plan_conserves_verification_and_exposes_pending_or_committed_versions(acceptance):
    values = dict(lengths=(4,), heads=4, key_heads=4, state_contexts=(4096,))
    if acceptance is not None:
        values["state_acceptance"] = acceptance
    op = kda_recurrent.make_op(values, requests=("r",))
    control = kda_recurrent.make_op(dict(lengths=(4,), heads=4, key_heads=4))
    geometry = Geometry(2, {"vector": 4, "mte2_aic": 2, "mte2_aiv": 4, "mte3": 4}, {"ub": 256 * 1024}, {"ub": 4})
    plan = kda_recurrent.build(op, geometry, compact=True)
    assert plan.work == kda_recurrent.build(control, geometry, compact=True).work
    expected = [("read", 4096), *(("write", token) for token in range(4097, 4101))]
    if acceptance:
        expected += [("commit", 4098), ("rollback", 4099), ("rollback", 4100)]
    assert [(e["action"], e["version"]) for e in plan.metadata["state_effects"]] == expected
    contract = plan.metadata["state_contract"]
    assert contract["commit_versions"] == ((4098,) if acceptance else (None,))
    engine = Engine(())
    target, = contract["targets"]
    conductor = FragmentConductor(engine, state={target: 4096})
    fragments = kda_recurrent.state_fragments(op)
    candidate = conductor.submit(fragments["write"])
    conductor.advance(until=(candidate.event,))
    assert conductor.state.committed_version(target) == 4096
    assert 4100 in conductor.state.live_versions(target)
    for fragment in (*fragments["commits"], *fragments["rollbacks"]):
        inputs = {port.name: engine.submit(port.name) for port in fragment.ports.event_in}
        effect = conductor.submit(fragment, after=(candidate.event,), inputs=inputs)
        conductor.advance(until=(effect.event,))
    assert conductor.state.committed_version(target) == (4098 if acceptance else 4096)
    assert (4100 in conductor.state.live_versions(target)) is (acceptance is None)


def decode_ops(selection, tokens):
    model = load_model("kimi-k3")
    plan = next(layer for layer in model.layer_plans if layer.attention == "kda")
    lowerings = {"kda": selection} if selection else {}
    return linear(
        prefix="decode", plan=plan,
        views=(RequestView("r", tokens, 2048, False, "decode"),),
        stage=0, deps="input", model=model, parallel=ParallelSpec(tp=8),
        quant=model.quant, mode="decode", padded_tokens=tokens,
        policy=AttentionPolicy(recurrent_chunk=64),
        runtime=Software.from_mapping({"kernels": {"lowerings": lowerings}}),
    ).ops


def declaration(ops):
    return next(
        group for op in ops
        for group in dict(op.metadata).get("structure_subgraphs", ())
        if group["role"] == "kda_recurrent")


@pytest.mark.parametrize("tokens", [1, 2, 4])
def test_prefill_state_selection_preserves_decode_work_without_native_contract(tokens):
    ordinary, selected = decode_ops(None, tokens), decode_ops(STATE, tokens)
    assert tuple(replace(op, metadata=()) for op in selected) == tuple(
        replace(op, metadata=()) for op in ordinary)
    group = declaration(selected)
    assert not group["tensor_view"]["in_kernel_gate_source"]
    assert group["implementation"]["selection"] is None
    assert subgraph_contract(group) is None
    assert all(not op.source_boundary for op in selected)


def test_explicit_decode_gate_selection_keeps_its_declared_contract():
    group = declaration(decode_ops(GATES, 1))
    assert group["tensor_view"]["in_kernel_gate_source"]
    assert group["implementation"]["selection"] == GATES
    assert subgraph_contract(group) is not None


def kda_params(lengths=(64,), heads=2, **overrides):
    value = {
        "lengths": list(lengths), "heads": heads, "key_heads": heads,
        "dk": K, "dv": V, "chunk": CHUNK, "dtype": "bf16",
        "gate_dtype": "fp32", "state_dtype": "fp32",
    }
    value.update(overrides)
    return value


def execute(value, case, *, trace_limit=0):
    op = kda_state.make_op(value)
    parsed = parse_case(case)
    recipes = RecipeBook(recipes={op.recipe: PlannedRecipe(kda_state.build)})
    execution = Execution(parsed, CostModel(parsed), recipes=recipes,
                          trace_limit=trace_limit)
    plan = IterationPlan((RequestView("r", 1, 0, True, "prefill"),))
    iteration = IterationWork((op,), plan, 1, 1, 1, None, "prefill", "one")
    result = execution.add(iteration, 0, 0)
    execution.engine.advance(until=(result.completion,))
    return execution.engine


def tasks_for(lengths, heads, cores):
    total = len(lengths) * heads
    slots = 2 if total > cores else 1
    return [
        (core, tuple(tuple((core, slot, *divmod(index + slot, heads))
                           for slot in range(min(slots, total - index)))
                     for index in range(core * slots, total, cores * slots)))
        for core in range(min(cores, (total + slots - 1) // slots))
    ]


def kda_plan(value, cores, *, split_copy=False):
    op = kda_state.make_op(value)
    units = {"vector": 2 * cores}
    if split_copy:
        units.update(mte2_aic=cores, mte2_aiv=2 * cores)
    return kda_state.build(op, Geometry(cores, units, {})), op


@pytest.mark.parametrize("axis", ("vector", "mte2"))
def test_source_owner_supply_limits_service_to_its_declared_rate(axis):
    caps = dict.fromkeys(("memory", "mte2", "mte3", "mte1", "fixpipe", "vector"), 1e30)
    durations = []
    for supply in (1e6, 2e6):
        caps.update({axis: supply, **({"mte3": supply} if axis == "mte2" else {})})
        raw = kda_hardware_case(units=1, rates=caps, compute_bf16=1e15)
        durations.append(execute(kda_params((128,), heads=1), raw).now)
    assert durations[0] == pytest.approx(2 * durations[1], rel=1e-6)

def kda_ledger(lengths, heads):
    """Independent interval ledger for each FWD_H head stream."""
    totals = {role: 0 for role in ("mte2_aic", "mte2_aiv", "mte1", "mte3", "fixpipe", "vector")}
    state = K * V * 4
    snapshot = K * V * 2
    for length in lengths:
        totals["mte2_aiv"] += state
        totals["mte3"] += snapshot
        totals["vector"] += K * V
        for start in range(0, length, CHUNK):
            rows = min(CHUNK, length - start)
            more = start + rows < length
            key, value = rows * K * 2, rows * V * 2
            totals["mte2_aic"] += 2 * key + value + snapshot
            totals["mte2_aiv"] += 3 * value + key + 2 * state + K * 4
            totals["mte1"] += 2 * key + value + snapshot
            totals["mte3"] += 2 * value + key + state + more * snapshot
            totals["fixpipe"] += 2 * value + state
            totals["vector"] += 4 * rows * V + 2 * K + (2 + more) * K * V
    totals["hbm"] = sum(totals[role] for role in ("mte2_aic", "mte2_aiv", "mte3", "fixpipe"))
    return {role: amount * heads for role, amount in totals.items()}


@pytest.mark.parametrize("lengths,heads,cores", (((128,), 1, 32), ((64, 64), 1, 2), ((63, 1, 65), 3, 5)))
def test_kda_state_stride_ragged_work_and_stream_initialization(lengths, heads, cores):
    expected_owners = tasks_for(lengths, heads, cores)
    actual = [(group.core, tuple(tuple((t.core, t.slot, t.sequence, t.head) for t in run)
                                 for run in group.rounds))
              for group in kda_state.head_ownership(lengths, heads, cores)]
    assert actual == expected_owners
    plan, op = kda_plan(kda_params(lengths, heads), cores, split_copy=True)
    assert plan.work.matrix == {"bf16": 4 * sum(lengths) * heads * K * V}
    assert plan.work.resources == kda_ledger(lengths, heads)
    meta = dict(op.metadata)
    assert op.source_boundary == "kernel" and meta["source_phase"] == "FWD_H"
    assert meta["head_tasks"] == len(lengths) * heads
    assert (meta["internal_state_layout"], meta["public_state_layout"]) == ("K,V", "V,K")


@pytest.fixture
def planning_context():
    geom = Geometry.from_devices(machine(REPRESENTATIVE_SKUS["a5"]).dies)
    return OperatorPlanningContext(OpOccurrence(0, 0, 0, "state"), geom, geom.datapath)


@pytest.mark.parametrize("bt,counts,stored", ((64, (8, 64, 512), "bf16"), (32, (16, 128, 1024), "fp32")))
def test_resident_state_tile_is_constant_while_staged_bank_crosses_l2_knee(planning_context, bt, counts, stored):
    tile_bytes = 4 * 64 * 32
    bytes_per_chunk = 128 * 64 * (2 if stored == "bf16" else 4)
    capacity = counts[0] * bytes_per_chunk
    ub_sizes = set()
    for length, chunks in zip((512, 4096, 32768), counts, strict=True):
        contract = LinearAttentionContract(
            "gdn_prefill", (length,), 1, 1, 1, 64, 128, bt=bt, bv=64, bk=32,
            abi="gdn_chunk32" if bt == 32 else "", stored_chunk_state_dtype=stored,
            l2_capacity_bytes=capacity,
        )
        staged = analyze_state_residency(contract, planning_context)
        persistent = analyze_state_residency(contract, planning_context, persistent=True)
        assert staged.total_chunks == persistent.total_chunks == chunks
        assert staged.state_tile_bytes == persistent.state_tile_bytes == tile_bytes
        assert staged.chunk_bank_bytes == staged.compulsory_store_bytes == chunks * bytes_per_chunk
        assert persistent.persistent_ub_resident and persistent.chunk_bank_bytes == 0
        hits = capacity if length == 512 else 0
        assert staged.l2_hit_bytes == hits
        assert staged.dram_miss_bytes == staged.reload_bytes == chunks * bytes_per_chunk - hits
        assert staged.evicted_chunks == (0 if hits else chunks)
        ub_sizes.add(persistent.owner_ub_bytes)
    assert len(ub_sizes) == 1
    crowded = replace(contract, lengths=(512,), active_l2_streams=2)
    shared = analyze_state_residency(crowded, planning_context)
    assert shared.l2_hit_bytes == 0 and shared.dram_miss_bytes == capacity
    with pytest.raises(TileInfeasibleError):
        analyze_state_residency(replace(contract, bv=512, bk=256), planning_context)


@pytest.mark.parametrize("lengths,metadata,selected", (
    ((64, 128), {"has_cu": True}, "persistent"),
    ((64, 128), {"metadata_word_capacity": 1}, "staged"),
    ((512,), {"calibration_enabled": False}, "staged"),
    ((32768,), {"kda_template_request": "persistent"}, "persistent"),
))
def test_flash_kda_metadata_controls_route_and_saved_output_cache(planning_context, lengths, metadata, selected):
    route = {k: v for k, v in metadata.items() if k != "kda_template_request"}
    assert select_flash_kda_template(
        lengths, value_heads=2, query_heads=2, core_num=32,
        template_request=metadata.get("kda_template_request", "auto"), **route,
    ) == selected
    op = OpSpec("state", "kda_prefill", "prefill", 0, 0, (), (), (), (), (), recipe="kda_prefill",
                metadata=tuple(dict(lengths=lengths, hv=2, dk=64, dv=128,
                                    save_intermediate_state=True, **metadata).items()))
    plan = resolve_operator_plan(op, planning_context).recipe
    assert plan.metadata["template"] == selected
    assert plan.metadata["persistent_ub_resident"] is (selected == "persistent")
    assert plan.metadata["saved_entry_state_bytes"] == sum((n + 63)//64 for n in lengths) * 2*64*128*2
    assert plan.metadata.get("saved_outputs_l2_cache_disabled", False) is (selected == "staged")
