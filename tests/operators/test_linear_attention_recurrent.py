"""Linear recurrent kernels and verification/commit replay contracts.

[Role] Exercise output tiling, source recurrence, and governed RecoverSSM plans.
[Invariants] Preserve owner-local work, state traffic, and integer acceptance.
[Timeouts & Locks] Bounded local Engine programs and symbolic chunk plans.
[Side-Effect Policy] Read-only input fixtures and in-memory simulation.
"""
from collections import Counter
from dataclasses import replace
import json
import math
from pathlib import Path

import pytest

from _support.axes import REPRESENTATIVE_SKUS, SKUS
from _support.builders import linear_model, view
from _support.linear_cases import (
    caller, execute_one, gdn_decode_source_case, kda_hardware_case, lower_reference as _reference, packed_roles,
    recipe_context, recurrent_scan_context, sku_planning_context, stage_ancestors as _ancestors, stage_bytes as _bytes,
    stages_by_name as _stages, template_datapath,
)
from mindie_sim import graph_costs
from mindie_sim.catalog.speculative_state import MODE_CHECKPOINT, MODE_REPLAY, MODE_SINGLE, speculative_state_plan
from mindie_sim.prediction.operators.linear_attention.call_gdn import make_call
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.operators.linear_attention._gdn_decode_programs import CONV, GATING, gating_plan
from mindie_sim.prediction.operators.linear_attention.gdn_recurrent import (
    build as build_gdn, head_ownership, head_runs,
)
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.operators.linear_attention import gdn_output
from mindie_sim.prediction.operators.kda import kda_recurrent, kda_recurrent_membase
from mindie_sim.prediction.operators.linear_attention import (
    RECOVER_SSM_FAMILIES, build as build_governed, build_recover_ssm_commit, build_recover_ssm_verify,
    rank_one_transition, resolve_operator_plan,
)
from mindie_sim.prediction.program import RecipeContext
from mindie_sim.prediction.operators.recurrent.recover_ssm_commit import bind_declared_body_commits, bind_recover_ssm_commits
from mindie_sim.prediction.operators.recurrent.recover_ssm_work import commit_tokens_for_view
from mindie_sim.prediction.operators.recurrent.recurrent import _sum_valid_powers
from mindie_sim.prediction.operators.recurrent.recurrent_template import lower_recurrent_scan, recurrent_scan_classes
from mindie_sim.prediction.graph_analysis import verify_operator
from mindie_sim.schema.case import Software
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.workload import RequestView
from mindie_sim.structure_subgraphs import subgraph_contract
from mindie_sim.templates import dsl_catalog as dsl, enumerate_dsl_specs
from tools.kimi_kda_alignment import compare
from tools.model_scoreboard_kimi import prediction_packet, predict_worker

pytestmark = pytest.mark.tier0

ROOT = Path(__file__).resolve().parents[2]
BT, K, V = 64, 128, 128

def params(lengths=(64,), heads=12, key_heads=4, **overrides):
    return {"lengths": lengths, "heads": heads, "key_heads": key_heads,
            "dtype": "bf16", "gate_dtype": "fp32", "state_v_first": True,
            "use_exp2": True, "output_token_first": True, **overrides}

def transfer_oracle(values):
    """Enumerate grouped Q/K loads and disjoint output intervals."""
    heads, keys = values["heads"], values["key_heads"]
    ratio = heads // keys
    group = 3 if ratio == 3 else 4
    traffic = Counter()
    addresses = []
    token = 0
    for length in values["lengths"]:
        for start in range(0, length, BT):
            rows = min(BT, length - start)
            for first in range(0, heads, group):
                local = range(first, min(first + group, heads))
                qk_count = len({head // ratio for head in local})
                traffic["mte2"] += qk_count * 2 * rows * K * 2
                traffic["mte2"] += len(local) * (rows * (4 + V * 2) + K * V * 2)
                traffic["fixpipe"] += len(local) * 4 * (BT**2 + (BT + rows) * V)
                traffic["l1_write"] += len(local) * BT**2 * 2
                for row in range(token + start, token + start + rows):
                    addresses.append(((row * heads + first) * V, len(local) * V))
        token += length
    offset = 0
    for start, length in sorted(addresses):
        assert start == offset
        offset += length
    assert offset == token * heads * V
    return dict(mte2=traffic["mte2"], mte3=2 * offset + traffic["l1_write"],
                hbm=traffic["mte2"] + 2 * offset, fixpipe=traffic["fixpipe"])

@pytest.mark.parametrize("rows", (1, 15, 16, 17, 32, 33, 48, 63, 64))
def test_av_requested_tail_is_distinct_from_physical_bf16_fractal_work(rows):
    op = gdn_output.make_op(params(lengths=(rows,), heads=3, key_heads=1))
    work = gdn_output.build(op, Geometry(4, {"vector": 8})).work
    qk_outputs = BT * BT
    qh_outputs = BT * V
    av_outputs = rows * V
    requested = 3 * (qk_outputs * 2 * K + qh_outputs * 2 * K + av_outputs * 2 * rows)
    physical_av = sum(
        2 * 16**3 for _m in range(0, rows, 16)
        for _n in range(0, V, 16) for _k in range(0, rows, 16))
    physical = 3 * (qk_outputs * 2 * K + qh_outputs * 2 * K + physical_av)
    assert dict(op.metadata)["requested_matrix_flops"] == requested
    assert work.matrix == {"bf16": physical}

@pytest.mark.parametrize("lengths,heads,keys", (((17, 65), 8, 4), ((63, 1, 65), 5, 5), ((64,), 12, 3)))
def test_output_gm_tails_and_group_reuse_match_address_walk(lengths, heads, keys):
    values = params(lengths, heads, keys)
    op = gdn_output.make_op(values)
    expected = transfer_oracle(values)
    built = gdn_output.build(op, Geometry(4, {"vector": 8}))
    for resource, amount in expected.items():
        assert built.work.resources[resource] == amount
    raw = kda_hardware_case(units=4, rates={"read": 1e12, "write": 1e12})
    assert verify_operator(raw, op, gdn_output.build)["status"] == "passed"

def _scan(lengths, *, heads=3, dk=128, dv=128, units=4, compact=True, module=None,
          rates=None, contender=None, state_dtype="fp32", **flags):
    values = dict(lengths=lengths, heads=heads, key_heads=1, dk=dk, dv=dv, state_dtype=state_dtype, **flags)
    shape = (sum(lengths), heads, dk, dv)
    if module is None:
        op = OpSpec("scan", "recurrent_vector", "decode", 0, 0, (), shape, shape, shape, shape,
                    dtype="bf16", metadata=tuple(dict(values, sequence_tokens=lengths).items()))
    else:
        op = module.make_op(values, deps=("input",))
    physical = {"hbm": 1e12, "mte2_aiv": 1e12, "mte2_aic": 1, "mte2": 1,
                "vector": 1e12, "mte3": 1e12, **(rates or {})}
    units_by_pipe = {role: units if role != "hbm" else 1 for role in physical}
    aics, capacity = max(1, units // 2), 256 * 1024
    geom = Geometry(aics, units_by_pipe, {"ub": capacity}, {"ub": units})
    plan = (build_gdn if module is None else module.build)(op, geom, compact=compact)
    engine = Engine([Resource(role, rate, units=units_by_pipe[role]) for role, rate in physical.items()],
                    pools=geom.buffers, pool_units=geom.buffer_units, trace_limit=20000)
    ctx = RecipeContext(plan.work.apply(op), {}, {role: role for role in physical}, physical, aics,
                        resource_units=units_by_pipe, pool_names={"ub": "ub"}, pool_units=geom.buffer_units)
    ready = engine.submit("input", release=3)
    done = plan.lower(ctx).start(engine, "scan", ready)
    other = engine.submit("contender", contender, after=(ready,)) if contender else None
    engine.advance(horizon=2)
    assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    if other is not None:
        engine.advance(until=(other,))
    assert not engine._tasks and not engine._promises and not engine.allocations
    for role, amount in plan.work.resources.items():
        assert engine.serviced[role] == pytest.approx(amount + (contender or {}).get(role, 0))
    assert engine.serviced["mte2_aic"] == engine.serviced["mte2"] == 0
    return plan, engine, done.time - 3

@pytest.mark.parametrize("lengths,dk,dv,units", (
    ((1,), 16, 16, 1), ((4, 3), 128, 65, 4), ((2, 4, 3), 129, 97, 8), ((4, 4), 256, 129, 4),
))
def test_gdn_recurrent_payload_uses_real_values_and_padded_compute(lengths, dk, dv, units):
    plan, engine, _ = _scan(lengths, dk=dk, dv=dv, units=units)
    ak, av = math.ceil(dk / 16) * 16, math.ceil(dv / 16) * 16
    raw_step = 8192 // ak
    step = (raw_step if raw_step <= 256 else 255) // 16 * 16
    tiles = [min(step, dv - offset) for offset in range(0, dv, step)]
    tokens, owners = sum(lengths) * 3, len(lengths) * 3
    incoming = owners * dk * dv * 4 + tokens * (2 * dk + dv) * 2
    outgoing = tokens * (dk * dv * 4 + dv * 2)
    vector = owners * ak * dv + tokens * (6 * ak + av + sum((8 * ak + 1) * n for n in tiles))
    assert plan.work.resources == {
        "mte2_aiv": incoming, "mte3": outgoing, "vector": vector,
        "hbm": incoming + outgoing + tokens * (4 + 2 * len(tiles))}
    assert all(row["start_ms"] >= 3 for row in engine.trace if row["demand"])

@pytest.mark.parametrize("module", (None, kda_recurrent, kda_recurrent_membase), ids=("gdn", "kda", "staged-kda"))
@pytest.mark.parametrize("lengths,units,contender", (((4, 4), 1, None), ((4, 4, 1, 4), 4, {"mte3": 10000})))
def test_recurrent_compaction_conserves_owner_work(module, lengths, units, contender):
    args = dict(module=module, units=units, contender=contender)
    detailed, expanded, original = _scan(lengths, **args, compact=False)
    compact, grouped, duration = _scan(lengths, **args, compact=True)
    assert compact.work == detailed.work
    assert grouped.serviced == pytest.approx(expanded.serviced)
    if contender is None:
        assert duration == pytest.approx(original, rel=1e-9)
    assert grouped.completed_count <= expanded.completed_count
    if module is None:
        assert grouped.completed_count < expanded.completed_count
    runs = (length for assignment in head_ownership(lengths, 3, units)
            for length, count in head_runs(lengths, assignment) for _ in range(count))
    assert list(runs) == [n for n in lengths for _ in range(3)]

@pytest.mark.parametrize("lengths,heads,units,dv,state", (
    ((1,), 4, 1, 128, "fp32"), ((4, 1), 7, 4, 256, "bf16"),
    ((6,)*4, 4, 40, 128, "fp32"),
))
def test_kda_state_payload_and_owner_sequence_beta_reuse(lengths, heads, units, dv, state):
    args = dict(heads=heads, units=units, dv=dv, state_dtype=state)
    old, _, _ = _scan(lengths, module=kda_recurrent, **args)
    new, engine, _ = _scan(lengths, module=kda_recurrent_membase, **args)
    tokens = sum(lengths) * heads
    state_bytes = 128 * dv * (4 if state == "fp32" else 2)
    incoming = tokens * ((2 * 128 + dv + 128 + heads) * 2) + len(lengths) * heads * (state_bytes + 4 + 128 * 4)
    assert old.work.resources["mte2_aiv"] == incoming
    assert new.work.resources["mte3"] == old.work.resources["mte3"] == tokens * (state_bytes + dv * 2)
    reuse = heads - min(heads, units)
    assert old.work.resources["mte2_aiv"] - new.work.resources["mte2_aiv"] == tokens * 2 * reuse
    casts = sum(math.ceil(q * heads / 8) * 8 for q in lengths) * reuse
    assert new.work.resources["vector"] - old.work.resources["vector"] == tokens * 128 * dv - casts
    assert engine.peak_allocated["ub"] == new.metadata["kda_recurrent_schedule"]["ub_bytes_per_owner"]

@pytest.mark.parametrize("role", ("mte2_aiv", "mte3", "vector"))
def test_recurrent_service_changes_only_with_active_pipe_supply(role):
    results = [_scan((4,), module=kda_recurrent, heads=1, units=1, rates={role: rate})
               for rate in (1e6, 2e6)]
    assert results[0][2] > results[1][2]
    assert results[0][0].work == results[1][0].work
    assert results[0][1].serviced == pytest.approx(results[1][1].serviced)

def test_recurrent_output_queues_respect_ownership_and_dependency_order():
    _, engine, _ = _scan((4,), module=kda_recurrent, heads=1, units=1,
                         rates={"hbm": 1e30, "vector": 1e7, "mte2_aiv": 1e6, "mte3": 1e5})
    def event(suffix):
        return next(row for row in engine.trace if row["task"].endswith(":" + suffix))

    for first, second in (("attention_cast0_1", "attention_write0_0"),
                          ("attention_write0_0", "state_write0_0"), ("state_write0_0", "state_export0_2")):
        assert event(first)["end_ms"] <= event(second)["start_ms"]
    load, compute = event("state_load1"), event("recur0_0")
    assert max(load["start_ms"], compute["start_ms"]) < min(load["end_ms"], compute["end_ms"])

@pytest.mark.parametrize("dv", (128, 256))
@pytest.mark.parametrize("internal_beta", (False, True))
def test_source_replaces_fused_members_once_and_keeps_actual_producers(dv, internal_beta):
    _, ops = caller(dv=dv, options={"use_beta_sigmoid": internal_beta})
    by_recipe = {op.recipe: op for op in ops}
    kernel = by_recipe[kda_recurrent.IMPLEMENTATION]
    assert not ({"qk_l2norm", "delta_gates"} & by_recipe.keys())
    assert sum(op.family == "recurrent_scan" for op in ops) == 1
    packed, slices = packed_roles(ops)
    assert ("linear_beta_sigmoid" in by_recipe) is not internal_beta
    beta = packed if internal_beta else by_recipe["linear_beta_sigmoid"]
    assert kernel.deps == (by_recipe["short_convolution"].op_id,
                           by_recipe["linear_decay_b"].op_id, beta.op_id)
    assert by_recipe["linear_decay_b"].local_shape == (5, 256, 128)
    assert slices["g"] == (5, 2*dv)
    params = dict(kernel.metadata)["kda_recurrent_params"]
    assert params["beta_dtype"] == ("bf16" if internal_beta else "fp32")
    group, = dict(kernel.metadata)["structure_subgraphs"]
    assert subgraph_contract(group) is not None
    available = {"input"}
    for op in ops:
        assert set(op.deps) <= available
        available.add(op.op_id)

def test_physical_projection_padding_does_not_invent_recurrent_state_updates():
    case, ops = caller((1, 4), padded=16)
    by_recipe = {op.recipe: op for op in ops}
    _, slices = packed_roles(ops)
    assert all(shape[0] == 16 for shape in slices.values())
    assert by_recipe["linear_output"].local_shape[0] == 16
    native = by_recipe[kda_recurrent.IMPLEMENTATION]
    assert native.local_shape == (5, 2, 128, 128)
    assert dict(native.metadata)["kda_recurrent_params"]["lengths"] == (1, 4)
    graph = graph_costs(case, ops, input_ready_ms=7, isolated=False, external_inputs={"input": 7})
    call = next(row for row in graph["timeline"] if row["recipe"] == native.recipe)
    assert call["ready_ms"] >= 7 and call["program"] == "OperatorPlan"

@pytest.mark.parametrize("lengths,logical", (((1,), 1), ((4, 3), 2), ((8,) * 65, 4)))
def test_decode_source_binds_normalization_and_conserves_repeated_gate_work(lengths, logical):
    parsed, views, _ = gdn_decode_source_case(lengths, queue=True)
    lowerings = parsed.runtime.kernels["lowerings"]
    options = {**lowerings["gdn_decode"]["options"], "recurrent_implementation": "arch35_fp32"}
    kernels = {"lowerings": {"gdn_decode": dict(lowerings["gdn_decode"], options=options)}}
    ops, _ = Frontend().body(parsed.model, replace(parsed.runtime, kernels=kernels), views, mode="decode", machine=parsed.machine)
    by_recipe = {op.recipe: op for op in ops}
    assert {CONV, GATING, "decode_q_l2norm", "decode_k_l2norm", "ascend_recurrent_gdn_fp32_state"} <= by_recipe.keys()
    for role in ("decode_q_l2norm", "decode_k_l2norm"):
        engine, _ = execute_one(parsed, views, by_recipe[role])
        assert engine.serviced["vector@r0:s0"] > 0
    gate = by_recipe[GATING]
    gate = replace(gate, metadata=tuple(dict(dict(gate.metadata), logical_programs=logical).items()))
    work = gating_plan(gate, Geometry.from_devices(parsed.machine.dies)).work.resources
    engine, _ = execute_one(parsed, views, gate)
    assert engine.serviced["scalar_store_issue@r0:s0"] == pytest.approx(work["scalar_store_issue"]) == sum(lengths) * 12
    tiles = 2 * logical * math.ceil(sum(lengths) / (64 * logical))
    assert engine.serviced["vector@r0:s0"] == pytest.approx(work["vector"]) == tiles * (13 * 64 * 8 + 2 * 8)
    assert by_recipe["ascend_recurrent_gdn_fp32_state"].local_shape[0] == sum(lengths)

def test_serving_prefill_orders_state_gather_scan_and_publish():
    parsed, _, _ = gdn_decode_source_case((1,), keys=1, heads=3)
    runtime = replace(parsed.runtime, kernels={"lowerings": {"gdn_prefill": "gdn_prefill_arch35"}})
    views = (RequestView("new", 64, 0, False, "prefill"), RequestView("prior", 17, 32, True, "prefill"))
    ops, _ = Frontend().body(parsed.model, runtime, views, mode="prefill", machine=parsed.machine)
    by = {op.family: op for op in ops}
    assert by["state_gather"].op_id in by["gdn_state"].deps
    assert by["gdn_state"].op_id in by["state_scatter"].deps
    assert dict(by["gdn_prepare"].metadata)["dispatch"]["use_qk_l2norm_in_kernel"]
    assert not any(op.recipe.startswith("decode_") for op in ops)

@pytest.mark.parametrize("rows,keys,heads", ((1, 1, 3), (65, 2, 6)))
def test_scan_call_exports_real_rows_and_executes_three_source_launches(rows, keys, heads):
    call = make_call("scan_prefill", dict(batch=1, tokens=rows, k_heads=keys, v_heads=heads, k_dim=K, v_dim=V, chunk=BT))
    by = {op.recipe: op for op in call.ops}
    prepare, state, output = [next(op for op in call.ops if op.family == family)
                              for family in ("gdn_prepare", "gdn_state", "gdn_output")]
    assert prepare.op_id in state.deps and state.op_id in output.deps
    assert prepare.local_shape[-2] == rows and output.local_shape == (1, rows, heads, 128)
    assert dict(prepare.metadata)["output_beta"] is False
    assert not {"qkv_concatenate", "gather_recurrent_state", "scatter_recurrent_state"} & by.keys()
    assert call.ledger["exports"]["q_hat"]["shape"] == [1, rows, keys, 128]
    raw = json.loads((ROOT / "examples/calls/gdn-historical-device.json").read_text())
    graph = graph_costs(raw, call.ops, recipes=call.recipes, input_ready_ms=7)
    for row in graph["isolated"]["operators"]:
        assert row["declared_service"] == pytest.approx(row["serviced"])

def inputs():
    data = ROOT / "tests/fixtures/verification"
    blocks = json.loads((data / "kda-call-blocks.json").read_text())
    blocks["runs"] = blocks["runs"][:2]
    observed = json.loads((data / "kimi-native-descriptors.json").read_text())
    case = json.loads((ROOT / "examples/cases/kimi-k3-a5-current-checkpoint-conditional.json").read_text())
    packet = prediction_packet("kda", observed, case, {}, context=32768,
                               kda_tokens=8192, matrix_recipe="steady_matmul")
    return blocks, predict_worker(packet)

def test_closed_native_calls_reconcile_stage_sums_and_reject_missing_neighbors():
    blocks, prediction = inputs()
    result = compare(blocks, prediction)
    observed = sum(event["duration_ms"] for run in blocks["runs"] for event in run["events"])
    predicted = 2 * prediction["rows"][0]["prediction"]["per_call_ms"]
    assert result["matched_blocks"] == 2
    assert sum(stage["observed_activity_sum_ms"] for stage in result["stages"]) == pytest.approx(observed)
    assert sum(stage["predicted_activity_sum_ms"] for stage in result["stages"]) == pytest.approx(predicted)
    blocks["runs"][0]["semantic_boundary_anchors"]["following_output_projection"] = False
    assert compare(blocks, prediction)["coverage"] == {"observed_runs": 2, "matched": 1, "unpaired": 1}

def intent(owner="layer0", version="v0", requests=("r0", "r1")):
    return OpSpec(
        "verify", "recurrent_scan", "decode", 0, 0, requests,
        (2,), (2,), (2,), (2,), vector_ops=100.0, dtype="fp32",
        deps=("x",),
        metadata=(
            ("recover_ssm_intent", True), ("state_owner", owner),
            ("input_state_version", version), ("heads", 2), ("dk", 4),
            ("dv", 8), ("state_dtype", "fp32"),
            ("record_kg_dtype", "bf16"), ("record_bytes_per_token", 48.0),
        ))

def test_integer_commits_sum_and_publish_after_acceptance():
    ops = (intent(),)
    views = (view("r0", 6, advanced=3), view("r1", 6, advanced=4))
    bound = bind_recover_ssm_commits(
        ops, views=views, acceptance_events={"r0": "accept0", "r1": "accept1"},
        state_bindings={"layer0": "base0"}, runtime=None)
    commit = bound[0]
    assert commit.deps == ("x", "accept0", "accept1", "base0")
    metadata = dict(commit.metadata)
    assert metadata["committed_tokens"] == (("r0", 3), ("r1", 4))
    assert metadata["published_state_version"] == "v7"
    assert metadata["published_state_version"] != metadata["input_state_version"]
    assert commit.kernel_binding is not None
    assert commit.kernel_binding.kernel_id == "RecoverSSMStateCommit"
    assert metadata["acceptance_boundary"] == "sampling_acceptance"
    state = 2 * 4 * 8 * 4
    assert commit.vector_ops == 4 * 7 * 2 * 4 * 8
    assert commit.mte2_bytes == pytest.approx(2 * state + 7 * 48.0)
    assert commit.mte3_bytes == pytest.approx(2 * state)

def test_missing_acceptance_duplicate_owner_and_ambiguous_mean_fail():
    views = (view("r0", 6, advanced=3),)
    with pytest.raises(ModelingError, match="requires_acceptance_event"):
        bind_recover_ssm_commits((intent(requests=("r0",)),), views=views,
                                 acceptance_events={}, state_bindings={},
                                 runtime=None)
    with pytest.raises(ModelingError, match="duplicate_commit"):
        bind_recover_ssm_commits((intent(), intent()), views=views,
                                 acceptance_events={"r0": "a"},
                                 state_bindings={}, runtime=None)
    ambiguous = view("r0", 6)
    with pytest.raises(ModelingError, match="resolved_integer_advancement"):
        commit_tokens_for_view(ambiguous, Software(mtp_acceptance_length=3.5))
    with pytest.raises(ModelingError, match="resolved_integer_advancement"):
        commit_tokens_for_view(ambiguous, Software())
    with pytest.raises(ModelingError, match="advancement_out_of_range"):
        commit_tokens_for_view(view("r0", 4, advanced=6), Software())
    assert commit_tokens_for_view(view("r0", 1), Software(
        mtp_acceptance_length=3.5)) == 1

def test_pending_commit_is_rebound_without_duplication():
    pending = OpSpec(
        "commit", "elementwise", "decode", 0, 0, ("r0",), (1,), (1,), (1,), (1,),
        vector_ops=4 * 3 * 2 * 4 * 8, hbm_bytes=100.0, mte2_bytes=60.0,
        mte3_bytes=40.0, deps=("verify",),
        metadata=(("recover_ssm_pending_acceptance", True),
                  ("state_owner", "layer0"), ("input_state_version", "v1"),
                  ("heads", 2), ("dk", 4), ("dv", 8),
                  ("record_bytes_per_token", 48.0)))
    bound = bind_recover_ssm_commits(
        (pending,), views=(view("r0", 6, advanced=3),),
        acceptance_events={"r0": "accept"}, state_bindings={}, runtime=None)
    commit, = bound
    assert commit.deps == ("verify", "accept") and dict(commit.metadata)["published_state_version"] == "v4"
    op = intent(requests=("r0",))
    marked, = bind_declared_body_commits((op,), views=(view("r0", 6, advanced=2),))
    assert dict(marked.metadata)["acceptance_boundary"] == "declared_body_input"
    with pytest.raises(ModelingError, match="resolved_integer_advancement"):
        bind_declared_body_commits((op,), views=(view("r0", 6),))

def test_indexed_template_is_constant_time_and_binds_setup_correctly():
    ctx = recurrent_scan_context(tokens=6554, chunk=64, setup=0.25, minimum=0.5)
    program = lower_recurrent_scan(ctx)
    template = program.template
    assert template.indexed_panels == 103
    assert (template.setup_ms, template.minimum_ms) == (0.25, 0.5)
    assert template.tile_factory_calls <= 2
    full, tail = recurrent_scan_classes(recurrent_scan_context())
    assert tail.stages[-1].work["cube"] * 2 == pytest.approx(full.stages[-1].work["cube"])
    assert [stage.after for stage in tail.stages[1:]] == [("c1",), ("delta",)]
    tail_start = 6554 // 64
    assert template.index_of(0) == (0, tail_start, 0, 0, 0)
    assert template.index_of(tail_start) == (1, 103, 0, tail_start, 0)
    with pytest.raises(ModelingError, match="index_out_of_range"):
        template.index_of(103)

def test_closed_form_sums_match_the_small_finite_loop():
    for tokens in (0, 1, 3, 64, 65, 6554):
        for chunk in (1, 4, 64):
            for power in (2, 3):
                expected = sum(min(chunk, tokens - offset) ** power
                               for offset in range(0, tokens, chunk))
                assert _sum_valid_powers(tokens, chunk, power) == expected

def _governed_op(recipe, lengths, **metadata):
    return OpSpec("scan", recipe, "decode", 0, 0, (), (), (), (), (), recipe=recipe, dtype="bf16",
                  metadata=tuple(dict({"lengths": lengths, "hv": 2, "dk": 32, "dv": 64,
                                       "bv": 32, "bk": 16}, **metadata).items()))

@pytest.mark.parametrize("sku", SKUS)
@pytest.mark.parametrize("kind,gate_width", (("gdn", 1), ("kda", 32)))
def test_governed_decode_conserves_state_contractions_and_fp32_tile(sku, kind, gate_width):
    geom, ctx = sku_planning_context(sku, "scan")
    op = _governed_op(f"{kind}_recurrent", (1, 4))
    plan = resolve_operator_plan(op, ctx).recipe
    tokens, dk, dv = 10, 32, 64
    contraction = 2 * tokens * dk * dv
    assert plan.metadata["state_tile_bytes"] == 4 * 32 * 16
    assert plan.metadata["state_contraction_ops"] == contraction
    assert plan.work.resources["vector"] == 2 * contraction + tokens * (3 * dk * dv + gate_width + 2 * dv)
    state = 2 * 2 * 4 * dk * dv
    payload = tokens * ((2 * dk + dv) * 2 + gate_width * 4 + 2)
    checkpoint = tokens * dk * dv * 4 if kind == "kda" else state
    assert plan.work.resources["hbm"] == state + checkpoint + payload + tokens * 2 * dv
    assert build_governed(op, geom).work == plan.work

@pytest.mark.parametrize("sku", SKUS)
@pytest.mark.parametrize("state_dtype,element,total", (("fp32", 4, 262_144), ("bf16", 2, 131_072)))
@pytest.mark.parametrize("lengths,active_heads", (((1, 3), (2, 1, 1)), ((4,), (1, 1, 1, 1))))
def test_recurrent_kda_checkpoints_each_valid_token(sku, state_dtype, element, total, lengths, active_heads):
    geometry, ctx = sku_planning_context(sku, "scan")
    op = _governed_op("kda_recurrent", lengths, hv=1, dk=128, dv=128, state_dtype=state_dtype)
    plan = resolve_operator_plan(op, ctx).recipe
    tokens = next(stage.program for stage in plan.program(recipe_context(geometry, op)).stages if stage.name == "tokens")
    stored = 0
    for i, active in enumerate(active_heads):
        stages = {stage.name: stage for stage in tokens.tile(i)}
        transition = "key_dot_rank1_update_query_dot"
        if "key_dot" in stages:
            assert stages["key_dot"].after == ("decay",)
            transition = "query_dot"
        checkpoint = stages["publish_token_state"]
        assert checkpoint.after == (transition,) and checkpoint.metadata["token_index"] == i
        assert checkpoint.work["mte3"] == active * 128**2 * element
        stored += checkpoint.work["mte3"]
    assert plan.metadata["token_state_slots"] == 4 and plan.metadata["token_state_bytes"] == stored == total

DK, DV = 32, 64
FACTORS = {"B": (DK,), "dt": (), "A_diag": (DK,), "C_proj": (DK,)}
ABI_WIDTHS = (
    ("kda", False, (4*DV, 2*DK, 2*DK, 0)),
    ("gdn", False, (4*DV, 4*DK, 4, 0)),
    ("deltanet", False, (4*DV, 4*DK, 0, 0)),
    ("deltanet", True, (4*DV, 4*DK, 4, 0)),
    ("gla", False, (4*DV, 2*DK, 2*DK, 2*DV)),
    ("lightning", False, (4*DV, 4*DK, 4, 0)),
    ("retnet", False, (4*DV, 4*DK, 4, 0)),
    ("mamba", False, (4*DV, 4*DK, 4, 0)),
    ("mamba", True, (4*DV, 2*DK, 2, 2*DK)),
)

@pytest.mark.parametrize("family,gated,widths", ABI_WIDTHS)
def test_recover_verification_and_commit_cover_exact_abi_prefix(family, gated, widths):
    _, ctx = sku_planning_context(REPRESENTATIVE_SKUS["a5"], "scan")
    options = dict(gated=gated, factors=FACTORS if family == "mamba" else None)
    transition = rank_one_transition(family, DK, DV, **options)
    assert (transition.correction_field_bytes, transition.key_field_bytes,
            transition.decay_field_bytes, transition.extra_field_bytes) == widths
    if family == "gla":
        decay, _, _ = transition.reconstruction_rule.partition(" + ")
        assert transition.decay_kind == "separable" and decay == "g_v[:, None] * S * g_k[None, :]"
    record, state, heads = sum(widths), 2 * 4 * DK * DV, 2
    for q in (1, 4):
        verify = build_recover_ssm_verify(family, DK, DV, q=q, heads=heads, **options)
        assert verify.single_checkpoint is (q == 1)
        assert verify.state_read_bytes == state
        op = _governed_op("recover_ssm_verify", (q,), q=q, ssm_family=family, **options)
        plan = resolve_operator_plan(op, ctx).recipe
        phases = {p.phase_id: p for p in plan.pipeline_spec.phases}
        name, size = ("write_state_checkpoint", state) if q == 1 else ("write_records", q * heads * record)
        assert sum(w.amount for w in phases[name].work if w.resource == "hbm") == size
        if q > 1:
            assert verify.record_write_bytes == size
        for a in range(q + 1):
            boundaries = (1,)
            commit = build_recover_ssm_commit(family, DK, DV, q=q, a=a, heads=heads,
                                              base_version=8, boundary_checkpoints=boundaries, **options)
            op = _governed_op("recover_ssm_commit", (q,), q=q, a=a, ssm_family=family,
                              base_version=8, boundary_checkpoints=boundaries, **options)
            replay = resolve_operator_plan(op, ctx).recipe
            assert commit.published_version == replay.metadata["published_version"] == 8 + bool(a)
            assert commit.boundary_checkpoints == tuple(n for n in boundaries if n <= a)
            read, written = state * bool(a), state * (bool(a) + (q > 1 and a > 1))
            records = a * heads * record if q > 1 else 0
            replay_ops = (4 + (family == "gla")) * a * heads * DK * DV if q > 1 else 0
            assert (commit.state_read_bytes, commit.state_write_bytes, commit.record_read_bytes) == (read, written, records)
            assert commit.vector_ops == replay.work.resources.get("vector", 0) == replay_ops
            assert replay.work.resources.get("hbm", 0) == read + written + records + 4 * (q == a == 1)
            phases = {p.phase_id: p for p in replay.pipeline_spec.phases}
            if q == a == 1:
                assert tuple(phases) == ("read_acceptance", "read_checkpoint", "publish")
                assert phases["publish"].after == ("read_checkpoint",)
            elif a:
                assert phases["publish"].after == (f"replay_to_{a}",)
                assert replay.metadata["boundary_checkpoints"] == (1,)
                if a > 1:
                    assert phases[f"replay_to_{a}"].after == ("publish_boundary_1",)
            else:
                assert not phases and replay.metadata["launch_count"] == 0

@pytest.mark.parametrize("accept", (1.0, 1.5, True, -1, 5))
def test_recover_commit_requires_bounded_integer_acceptance(accept):
    with pytest.raises(ModelingError, match="requires_integer_acceptance"):
        build_recover_ssm_commit("gdn", DK, DV, q=4, a=accept)

def test_recover_selective_factors_and_boundary_checkpoints_are_validated():
    with pytest.raises(ModelingError, match="selective_rank_one_factors"):
        rank_one_transition("mamba", DK, DV, factors=dict(FACTORS, B=(DV,)))
    for checkpoints in ((0,), (2, 1), (1, 1), (5,)):
        with pytest.raises(ModelingError, match="boundary_checkpoints_invalid"):
            build_recover_ssm_commit("kda", DK, DV, q=4, a=2, boundary_checkpoints=checkpoints)

@pytest.mark.parametrize("accepted", (0, 3))
def test_partitioned_kda_commit_reloads_key_decay_pair_for_each_value_tile(accepted):
    _, ctx = sku_planning_context(REPRESENTATIVE_SKUS["a5"], "scan")
    dim, partitions = 128, 4
    op = _governed_op("recover_ssm_commit", (4,), abi="commit_recurrent_kda", hv=1,
                      q=4, a=accepted, dk=dim, dv=dim, row_partitions=partitions)
    plan = resolve_operator_plan(op, ctx).recipe
    assert plan.metadata["physical_record_read_bytes"] == accepted * (4 * dim + 6 * partitions * dim)
    assert plan.work.resources.get("vector", 0) == 4 * accepted * dim**2
    assert plan.metadata["launch_count"] == 1
    phases = plan.pipeline_spec.phases
    assert phases[0].phase_id == "read_acceptance"
    assert sum(item.amount for item in phases[0].work if item.resource == "hbm") == 4
    if not accepted:
        assert len(phases) == 1 and plan.work.resources["hbm"] == 4

@pytest.mark.parametrize("partitions", (1, 2))
def test_commit_zero_accept_and_partitioned_replay_bytes(partitions):
    direct = template_datapath(REPRESENTATIVE_SKUS["a5"])
    p = dsl.CommitRecurrentKdaProblem((0, 2), row_partitions=partitions, state_indices=(6, 7),
                                      replay_capacity=4, k_stride0=640)
    _, result = _reference(dsl.COMMIT_KDA, p, direct)
    stages = _stages(result)
    assert {name for name in stages if name.startswith("commit:0:")} == {
        f"commit:0:{row}:accepted_read" for row in range(0, 128, 128 // partitions)
    }
    assert "commit:1:0:accepted_read" in _ancestors(stages, "commit:1:0:checkpoint_read")
    assert _bytes(stages, ":checkpoint_read", "mte2_aiv") == _bytes(stages, ":checkpoint_write", "mte3") == 4 * 128**2
    replay = 4 * 128 + 6 * partitions * 128
    assert sum(_bytes(stages, f":replay_{tensor}:read", "mte2_aiv") for tensor in ("u", "k", "decay")) == 2 * replay
    assert dsl.commit_replay_bytes(partitions) == replay
    k, = (entry for entry in result.summary.read_obligations if entry.allocation_id == "replay_k")
    assert k.ranges == ((8960, 8960 + 2 * 128 * 2),)
    _, empty = _reference(dsl.COMMIT_KDA, replace(p, accepted=(0, 0)), direct)
    assert empty.summary.logical_transfer_bytes == 2 * partitions * 4
    assert {entry.allocation_id for entry in empty.summary.read_obligations} == {"accepted"}
    assert not empty.summary.write_obligations
    for budget, quota in ((8, 8), (23, 8), (24, 24), (27, 24), (28, 28), (31, 28), (32, 32)):
        spec, = enumerate_dsl_specs(dsl.COMMIT_KDA, p, direct, aic_budget=budget)
        assert (spec.aic_budget, spec.aiv_budget) == (quota, 2 * quota)

@pytest.mark.parametrize("family", RECOVER_SSM_FAMILIES)
def test_capacity_uses_single_checkpoint_or_full_verify_window(family):
    model = linear_model(family, layers=1)
    layer = model.layer_plans[0]
    state, record = 4 * 16**2 * 4, 4 * (4 * 16 + 2 * 16 * 2)
    def capacity(q, recover):
        return speculative_state_plan(model, layer, ParallelSpec(), query_tokens=q,
                                      recover_ssm=recover, capacity_padded=False)

    assert capacity(1, True) == capacity(1, False)
    assert capacity(1, True).mode == MODE_SINGLE
    if family != "kda":
        assert capacity(1, True).recurrent_bytes == state
        return
    expanded, replay = capacity(6, False), capacity(6, True)
    assert (expanded.mode, expanded.recurrent_bytes, expanded.replay_bytes) == (MODE_CHECKPOINT, 6 * state, 0)
    assert (replay.mode, replay.recurrent_bytes, replay.replay_bytes) == (MODE_REPLAY, state, 6 * record)
    assert expanded.conv_bytes == replay.conv_bytes == (4 + 5) * 64 * 2


def test_shard2_verify_lengths_fp32_ub_and_nonzero_compulsory_spec_work():
    from types import SimpleNamespace
    from _support.hardware import machine
    from mindie_sim.compose import compose_case
    from mindie_sim.export import build_export_bundle
    from mindie_sim.export.bundle import compose_export_case
    from mindie_sim.prediction.mechanism import Geometry
    from mindie_sim.prediction.operators.kda import kda_recurrent_membase
    from mindie_sim.prediction.operators.linear_attention import call_gdn, gdn_chunk32
    from mindie_sim.prediction.operators.linear_attention.common import (
        LinearAttentionContract, SYNC_SCRATCH_BYTES, _resolve_recover_ssm_accepted, analyze_state_residency,
    )
    from mindie_sim.prediction.operators.vector import gated_activation, ring_update
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.schema.workload import RequestView

    # 1. #408 regression: speculative decode_recurrent_scan honors q_len + draft_proposal_tokens (dur >= L*).
    raw, req = compose_export_case(
        "qwen3.8-flash-next", REPRESENTATIVE_SKUS["a5"],
        tp=1, dp=8, ep=8, prompt_tokens=2048, output_tokens=64,
        concurrency=16, mtp_proposal_tokens=3, feature_track="unmerged_pr",
    )
    bundle = build_export_bundle(raw, request=req, apply_target_envelope=False)
    scans = [r for r in bundle.operators if r.op_id.endswith(".decode_recurrent_scan")]
    assert scans and all(r.dur_us + 1e-4 >= r.l_star_lean_us > 0.0 for r in scans)
    convs = [r for r in bundle.operators if r.recipe == "ascend_causal_conv1d_update"]
    gates = [r for r in bundle.operators if r.recipe == "ascend_gdn_decode_gating"]
    assert convs and all((r.hbm_read_bytes + r.hbm_write_bytes) > 0.0 and r.dur_us + 1e-4 >= r.l_star_lean_us for r in convs)
    assert gates and all((r.hbm_read_bytes + r.hbm_write_bytes) > 0.0 and r.vector_ops > 0.0 for r in gates)

    # 2. FP32 (4B/elem) UB upcast on recurrent state residency & ceiling acceptance rounding.
    a5_geo = Geometry.from_devices(machine(REPRESENTATIVE_SKUS["a5"]).dies[:1])
    a3_geo = Geometry.from_devices(machine(REPRESENTATIVE_SKUS["a3"]).dies[:1])
    rec_c = LinearAttentionContract(
        "gdn_recurrent", (1,), 16, 16, 16, 128, 128, 64, 128, 128,
        "bf16", "fp32", "bf16", "bf16", "fp32",
    )
    res = analyze_state_residency(rec_c, a5_geo.datapath)
    expected_fp32_scratch = (2 * 128 + 128 + 1 + 1 + 128) * 4
    assert res.owner_ub_bytes == 128 * 128 * 4 + expected_fp32_scratch + SYNC_SCRATCH_BYTES
    # Also fits A3 192 KiB UB under FP32 upcast.
    assert analyze_state_residency(rec_c, a3_geo.datapath).owner_ub_bytes == res.owner_ub_bytes
    ctx_acc = SimpleNamespace(requests=(
        RequestView("r0", 4, 128, True, "decode", draft_proposal_tokens=3, draft_accepted_tokens=2),
        RequestView("r1", 4, 128, True, "decode", draft_proposal_tokens=3, draft_accepted_tokens=3),
    ))
    assert _resolve_recover_ssm_accepted({"committed_tokens": (("r0", 2), ("r1", 3))}, ctx_acc, 4) == (2, 3)

    # 3. Non-zero compulsory OpSpec work <= plan.work on gdn_chunk32, call_gdn, kda_membase, ring_update.
    c32_params = {"lengths": (64, 64), "heads": 8, "key_heads": 4, "dk": 128, "dv": 128, "chunk": 32}
    for ns in (gdn_chunk32.PREPARE, gdn_chunk32.STATE, gdn_chunk32.OUTPUT):
        op = ns.make_op(c32_params)
        plan = ns.build(op, a5_geo)
        assert op.hbm_bytes > 0 and op.mte2_bytes > 0 and op.mte3_bytes > 0 and op.vector_ops > 0 and op.flops > 0
        assert op.hbm_bytes <= plan.work.resources["hbm"] + 1e-6
        assert op.vector_ops <= plan.work.resources["vector"] + 1e-6
    gating_call = call_gdn.make_call("gating", {"rows": 16, "heads": 16})
    g_op = gating_call.ops[0]
    g_plan = gating_call.recipes.recipes[g_op.recipe].build(g_op, a5_geo)
    assert g_op.hbm_bytes > 0 and g_op.hbm_bytes <= g_plan.work.resources["hbm"]
    assert g_op.vector_ops > 0 and g_op.vector_ops <= g_plan.work.resources["vector"]
    kda_m_op = kda_recurrent_membase.make_op({"lengths": (4, 4), "heads": 8, "key_heads": 8, "dk": 128, "dv": 128})
    kda_m_plan = kda_recurrent_membase.build(kda_m_op, a3_geo)
    assert kda_m_op.hbm_bytes > 0 and kda_m_op.hbm_bytes <= kda_m_plan.work.resources["hbm"]
    ru_op = ring_update.make_ring_update_op(32, 8, 128)
    assert ru_op.mte2_bytes > 0 and ru_op.mte3_bytes > 0 and ru_op.vector_ops > 0
    assert dict(ru_op.metadata)["ub_working_bytes_per_elem"] == 4

    # 4. Structured activation_params + FP32 UB metadata on gated_activation.
    k31 = parse_case(compose_case("kimi-k3.1", REPRESENTATIVE_SKUS["a5"], tp=8, ep=8, layers=4))
    act_op = gated_activation.gated_activation(
        plan=k31.model.layer_plans[0], op_id="s0.l0.act", deps=("s0.l0.up",),
        rows=8, width=128, dtype="bf16", stage=0, requests=("r0",), phase="decode", recipe="swiglu",
    )
    ameta = dict(act_op.metadata)
    assert ameta["activation_compute_dtype"] == "fp32" and ameta["ub_working_bytes_per_elem"] == 4
    assert ameta["activation_params"] == {"activation": "situ", "beta": 4.0, "linear_beta": 25.0}


def test_gdn_decode_fp32_decay_record_size_and_mixed_prefill_recover_ssm_runtime() -> None:
    """GDN decode stamps recurrent_attention_kind='gdn' (1,028 B); mixed prefill forwards runtime."""
    from mindie_sim.compose import compose_case
    from mindie_sim.prediction.frontend import Frontend
    from mindie_sim.prediction.operators.recurrent.recover_ssm_work import (
        gdn_record_head_bytes,
        record_head_bytes,
    )
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.schema.workload import IterationPlan, RequestView

    # 1. GDN record is 1,028 B (u:512 + k:256 + g_fp32:4 + v:256) vs KDA 1,024 B.
    assert gdn_record_head_bytes(128, 128) == 1028
    assert record_head_bytes(128, 128) == 1024

    raw = compose_case(
        "qwen3.8-flash-next", REPRESENTATIVE_SKUS["a5"],
        layers=4, tp=1, dp=1, ep=1, mtp_proposal_tokens=3, feature_track="unmerged_pr",
    )
    case = parse_case(raw)
    rt_meta = dict(case.runtime.metadata)
    rt_meta["recover_ssm"] = True
    rec_rt = replace(case.runtime, metadata=tuple(rt_meta.items()))

    # 2. Pure speculative decode stamps recurrent_attention_kind == 'gdn'.
    dec_plan = IterationPlan((
        RequestView("d0", 4, 128, True, "decode", draft_proposal_tokens=3, draft_advanced_tokens=2),
    ))
    dec_work = Frontend()(dec_plan, case.model, rec_rt, case.machine)
    dec_scans = [op for op in dec_work.ops if op.op_id.endswith(".decode_recurrent_scan")]
    assert dec_scans and all(dict(op.metadata).get("recurrent_attention_kind") == "gdn" for op in dec_scans)

    # 3. Mixed prefill + speculative decode forwards runtime and emits recover_ssm_commit.
    mixed_plan = IterationPlan((
        RequestView("p0", 64, 64, False, "prefill"),
        RequestView("d1", 4, 128, True, "decode", draft_proposal_tokens=3, draft_advanced_tokens=2),
    ))
    mixed_work = Frontend()(mixed_plan, case.model, rec_rt, case.machine)
    commit_ops = [op for op in mixed_work.ops if "recover_ssm_commit" in op.op_id]
    assert commit_ops, "mixed prefill+verify batch must emit RecoverSSM commit op"
