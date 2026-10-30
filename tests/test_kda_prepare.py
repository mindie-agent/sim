"""Source-selected KDA GATE_PREPARE stage.

The oracle enumerates the source row/head intervals and the two score blocks
independently of ``kda_prepare.build``.  Engine controls cover paired ownership,
BF16/FP32 service separation, delayed input readiness, mixed sequence chunk
boundaries, work conservation and bounded active work.  The ordinary frontend
selects one GATE_PREPARE launch and the existing FWD_H binder composes after it.
"""
from dataclasses import replace

import pytest

from mindie_sim import graph_costs, target_graphs
from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.attention import AttentionPolicy
from mindie_sim.prediction.call import call_costs
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.family_kda import IMPLEMENTATION as IN_KERNEL_GATES
from mindie_sim.prediction.family_kda_prepare import IMPLEMENTATION
from mindie_sim.prediction.family_kda_state import IMPLEMENTATION as STATE_ONLY
from mindie_sim.prediction.mechanism import Geometry, PlannedRecipe
from mindie_sim.prediction.operator_analysis import (
    operator_costs, physical_signature, target_operators,
)
from mindie_sim.prediction.operators import kda_prepare, kda_state
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.recurrent import linear
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.verification import verify_operator
from mindie_sim.schema.case import Case, Software
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.model import WorkloadSpec
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.operators import IterationWork
from mindie_sim.schema.workload import IterationPlan, RequestView
from mindie_sim.structure_subgraphs import subgraph_contract

K, V, CHUNK = 128, 128, 64
SCORE_HEAD = 2 * 2 * 32 * (32 + 64) * 128
FP32_HEAD = 4 * 2 * 16 ** 3 + 2 * 2 * 32 ** 3
DIAG_HEAD = 4 * (16 * (119 + 119 + 14) + 16)


def params(lengths=(64,), heads=2, **overrides):
    value = {
        "lengths": list(lengths), "heads": heads, "key_heads": heads,
        "dk": K, "dv": V, "chunk": CHUNK, "dtype": "bf16",
        "gate_dtype": "fp32", "raw_g_dtype": "bf16", "beta_dtype": "fp32",
        "compact": False,
    }
    value.update(overrides)
    return value


def hardware_case(*, units=2, split_copy=False, bf16=1e14, fp32=2.5e13):
    base = {"memory": 1e30, "vector": 1e30, "mte1": 1e30,
            "fixpipe": 1e30, "mte3": 1e30, "reg_special_lanes": 1e30}
    units_map = {"vector": 2 * units, "reg_special_lanes": 2 * units}
    if split_copy:
        units_map.update(mte2_aic=units, mte2_aiv=2 * units)
        base.update(mte2_aic=1e30, mte2_aiv=1e30)
    else:
        base["mte2"] = 1e30
    return {
        "machine": {
            "name": "kda-prepare-control", "num_devices": 1,
            "device": {
                "memory_bytes": 64 * 1024 ** 3, "compute_units": units,
                "compute_peaks": {"bf16": bf16, "fp32": fp32},
                "rates": base, "resource_units": units_map,
            },
        },
        "model": {"name": "geometry", "layers": 1, "hidden_size": 8,
                  "num_heads": 1, "num_kv_heads": 1, "head_dim": 8,
                  "intermediate_size": 16, "vocab_size": 16},
        "workload": {"requests": [{"id": "r", "prompt_tokens": 1,
                                   "output_tokens": 1}]},
        "runtime": {},
    }


def execute(value, case, *, trace_limit=0):
    op = kda_prepare.make_op(value)
    parsed = parse_case(case)
    book = RecipeBook(recipes={op.recipe: PlannedRecipe(kda_prepare.build)})
    execution = Execution(parsed, CostModel(parsed), recipes=book,
                          trace_limit=trace_limit)
    plan = IterationPlan((RequestView("r", 1, 0, True, "prefill"),))
    iteration = IterationWork((op,), plan, 1, 1, 1, None, "prefill", "one")
    result = execution.add(iteration, 0, 0)
    execution.engine.advance(until=(result.completion,))
    return execution.engine


def oracle(lengths, heads):
    """Independent source interval arithmetic for GATE_PREPARE."""
    lengths = tuple(lengths)
    headchunks = sum(lengths) // CHUNK * heads
    rows = sum(lengths) * heads
    chunks = sum(lengths) // CHUNK
    gate_read = rows * 128 * 2 + len(lengths) * heads * (4 + 128 * 4 + 2 * 8)
    gate_write = rows * 128 * 4
    gate_vector = rows * 10 * 128 + chunks * heads * 128 + 2 * len(lengths) * heads
    gate_special = rows * 2 * 128 + len(lengths) * heads
    aic = headchunks * (57_344 + 24_576)
    aiv = gate_read + headchunks * (42_112 + 66_688 + 16_384)
    work = {
        "bf16": headchunks * SCORE_HEAD,
        "fp32": headchunks * FP32_HEAD,
        "mte2_aic": aic,
        "mte2_aiv": aiv,
        "mte2": aic + aiv,
        "mte3": gate_write + headchunks * (2 * 73_728 + 24_576 + 8_192),
        "fixpipe": headchunks * (24_576 + 12_288),
        "mte1": headchunks * (57_344 + 24_576),
        "vector": gate_vector + headchunks * (DIAG_HEAD + 64 * 64 * 7
            + 64 * 128 * 41 + 32 * 128 * 7 + 64 + 4096),
        "reg_special_lanes": gate_special + headchunks * (64 * 128 * 4 + 32 * 128),
        "score_fixpipe": headchunks * 24_576,
    }
    work["hbm"] = (aic + aiv + work["mte3"]
                   + headchunks * 12_288)
    return work, headchunks


def plan(value, cores=2, *, split=False):
    op = kda_prepare.make_op(value)
    units = {"vector": 2 * cores}
    if split:
        units.update(mte2_aic=cores, mte2_aiv=2 * cores)
    return kda_prepare.build(op, Geometry(cores, units, {})), op


def assert_work(plan_work, expected, *, split):
    assert plan_work.matrix == {"bf16": expected["bf16"], "fp32": expected["fp32"]}
    resources = plan_work.resources
    if split:
        assert resources["mte2_aic"] == expected["mte2_aic"]
        assert resources["mte2_aiv"] == expected["mte2_aiv"]
    else:
        assert resources["mte2"] == expected["mte2"]
    for name in ("mte3", "fixpipe", "mte1", "vector", "reg_special_lanes", "hbm"):
        assert resources[name] == expected[name]


# ---------------------------------------------------------------------------
# Source arithmetic and ownership.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lengths,heads", [
    ((64,), 2),
    ((8192,), 12),
    ((64, 128), 2),
])
def test_source_interval_work_is_conserved(lengths, heads):
    expected, headchunks = oracle(lengths, heads)
    assert headchunks == sum(lengths) // 64 * heads
    for split in (False, True):
        built, op = plan(params(lengths=lengths, heads=heads), 2, split=split)
        assert_work(built.work, expected, split=split)
        assert op.flops == expected["bf16"] + expected["fp32"]
        assert dict(op.metadata)["headchunks"] == headchunks
        assert dict(op.metadata)["gate_tasks"] == len(lengths) * heads


def test_retained_call_arithmetic_matches_verified_source_counts():
    value = params(lengths=(8192,), heads=12)
    built, op = plan(value, 32)
    meta = dict(op.metadata)
    assert meta["headchunks"] == 1536
    assert meta["paired_tasks"] == 768
    assert meta["small_fp32_gemms_per_head"] == 6
    assert built.work.matrix["bf16"] == 2_415_919_104
    assert built.work.matrix["fp32"] == 251_658_240
    assert meta["score_block_bf16_flops_per_head"] == (524_288, 1_048_576)


@pytest.mark.parametrize("lengths,heads,cores,expected_counts", [
    ((64,), 2, 2, (1,)),
    ((8192,), 12, 32, (24,) * 32),
    ((4096,), 4, 32, (4,) * 32),
])
def test_finite_paired_task_queue_and_stride(lengths, heads, cores, expected_counts):
    value = params(lengths=lengths, heads=heads)
    parsed = kda_prepare.parse_params(value)
    ownership = kda_prepare.task_ownership(parsed, cores)
    assert tuple(count for _core, count in ownership) == expected_counts
    assert sum(expected_counts) == parsed.paired_tasks
    assert parsed.paired_tasks == sum(lengths) // 64 * (heads // 2)
    assert parsed.metadata(cores)["paired_tasks_per_core"] == expected_counts


def test_gate_owner_loop_covers_more_heads_than_aivs_and_multiple_sequences():
    value = params(lengths=(128,), heads=66)
    parsed = kda_prepare.parse_params(value)
    assert parsed.metadata(32)["gate_active_aiv"] == 64
    assert kda_prepare.task_ownership(parsed, 32)[:3] == ((0, 3), (1, 3), (2, 2))
    built, op = plan(value, 2)
    report = verify_operator(hardware_case(units=2), op, kda_prepare.build,
                             input_ready_ms=0)
    assert report["status"] == "passed", report
    assert report["work_mismatches"] == {}
    multi = params(lengths=(64, 64), heads=12)
    assert kda_prepare.parse_params(multi).metadata(32)["gate_active_aiv"] == 24
    built, op = plan(multi, 2)
    report = verify_operator(hardware_case(units=2), op, kda_prepare.build)
    assert report["work_mismatches"] == {}


def test_gate_cumsum_uses_only_real_head_owners_and_global_barrier():
    value = params(lengths=(64,), heads=12)
    built, op = plan(value, 8)
    meta = dict(op.metadata)
    assert meta["gate_tasks"] == 12
    assert meta["gate_active_aiv"] == 12
    engine = execute(value, hardware_case(units=8), trace_limit=100000)
    gate = [row for row in engine.trace
            if ":gate:program:group" in row["task"]]
    groups = {row["task"].split(":gate:program:")[1].split(":")[0]
              for row in gate}
    assert len(groups) == 12
    gate_end = max(row["end_ms"] for row in gate)
    sync = next(row for row in engine.trace if row["task"].endswith(":syncall"))
    assert gate_end <= sync["start_ms"]
    assert engine.summary()["live_tasks"] == 0


def test_multi_sequence_chunk_boundaries_are_not_concatenated_state():
    value = params(lengths=(64, 128), heads=2)
    parsed = kda_prepare.parse_params(value)
    assert parsed.chunks_per_sequence == (1, 2)
    assert kda_prepare.sequence_chunk(0, parsed) == 0
    assert kda_prepare.sequence_chunk(1, parsed) == 1
    assert kda_prepare.sequence_chunk(2, parsed) == 1
    meta = dict(kda_prepare.make_op(value).metadata)
    assert meta["lengths"] == (64, 128)
    assert meta["total_chunks"] == 3
    assert meta["gate_tasks"] == 4


def test_tail_is_explicitly_rejected_not_padded():
    from mindie_sim.schema.common import ModelingError
    with pytest.raises(ModelingError, match="partial_chunk_tail"):
        kda_prepare.make_op(params(lengths=(65,)))


# ---------------------------------------------------------------------------
# Engine: readiness, separate precision service, bounded live work.
# ---------------------------------------------------------------------------

def test_delayed_input_and_all_work_are_served_once():
    value = params(lengths=(128,), heads=2)
    op = kda_prepare.make_op(value)
    expected, _ = oracle((128,), 2)
    for split in (False, True):
        report = verify_operator(hardware_case(units=2, split_copy=split),
                                 op, kda_prepare.build, input_ready_ms=4)
        assert report["status"] == "passed", report
        assert report["input_dependency_obeyed"] is True
        assert report["buffers_released"] is True
        assert report["work_mismatches"] == {}
        row = report["result"]
        assert row["serviced"]["matrix@r0:s0"] > 0
        assert row["serviced"]["memory@r0:s0"] == pytest.approx(expected["hbm"])
        assert row["live_tasks"] == 0
        assert not any(row["remaining_buffers"].values())


def test_bf16_and_ieee_fp32_matrix_service_are_independent():
    value = params(lengths=(512,), heads=2)
    fast_bf16 = execute(value, hardware_case(
        units=2, bf16=1e15, fp32=1e30))
    slow_bf16 = execute(value, hardware_case(
        units=2, bf16=1e12, fp32=1e30))
    assert slow_bf16.now > 10 * fast_bf16.now
    fast_fp32 = execute(value, hardware_case(
        units=2, bf16=1e30, fp32=1e15))
    slow_fp32 = execute(value, hardware_case(
        units=2, bf16=1e30, fp32=1e9))
    assert slow_fp32.now > 1.01 * fast_fp32.now
    # BF16-only service is not charged FP32 work and conversely.
    assert kda_prepare.headchunk_work().matrix == {
        "bf16": SCORE_HEAD, "fp32": FP32_HEAD}


def test_per_core_stage_names_are_static_and_previous_merge_serializes():
    engine = execute(params(lengths=(128,), heads=2),
                     hardware_case(units=1), trace_limit=100000)
    rows = [row for row in engine.trace
            if ":prepare:program:group0:tile" in row["task"]]
    suffixes = {row["task"].split(":tile", 1)[1].split(":", 1)[1]
                for row in rows}
    assert "score:load" in suffixes and "merge1_5:publish" in suffixes
    assert "diag0:program:block0:program:row2" in suffixes
    first_merge = next(row for row in rows
                       if row["task"].endswith(":tile0:task_done"))
    next_score = next(row for row in rows
                      if row["task"].endswith(":tile1:score:load"))
    assert first_merge["end_ms"] <= next_score["start_ms"]


def test_long_rows_keep_active_work_bounded_and_release_everything():
    short = execute(params(lengths=(512,), heads=2), hardware_case(units=2))
    long = execute(params(lengths=(4096,), heads=2), hardware_case(units=2))
    for engine in (short, long):
        summary = engine.summary()
        assert summary["live_tasks"] == 0
        assert summary["live_programs"] == 0
        assert not any(summary["live_allocated"].values())
    assert long.summary()["peak_live_tasks"] == short.summary()["peak_live_tasks"]
    assert long.completed_count > short.completed_count


def test_seed_and_final_exports_follow_their_actual_producers():
    case = hardware_case(units=1)
    case["machine"]["device"]["rates"]["vector"] = 1e11
    engine = execute(params(), case, trace_limit=100000)
    rows = {row["task"].split(":tile0:", 1)[1]: row for row in engine.trace
            if ":prepare:program:group0:tile0:" in row["task"]}
    for lane in (0, 1):
        diagonal = rows[f"diag{lane}:program:block3:program:row15"]
        seed = rows[f"seed{lane}:store"]
        assert seed["start_ms"] >= diagonal["end_ms"]
        assert rows["merge0_0:load"]["start_ms"] >= seed["end_ms"]
        assert rows[f"final{lane}:load"]["start_ms"] >= rows["merge1_5:publish"]["end_ms"]
    assert rows["merge1_0:load"]["start_ms"] >= rows["merge0_5:publish"]["end_ms"]


def test_special_function_supply_is_distinct_from_ordinary_vector_lanes():
    fast = hardware_case(units=1)
    slow = hardware_case(units=1)
    slow["machine"]["device"]["rates"]["reg_special_lanes"] = 1e8
    quick, limited = execute(params(), fast), execute(params(), slow)
    # Two source head-chunks, plus two raw-gate rows tasks. Exp/Div lanes
    # cannot be supplied by an arbitrarily fast ordinary vector resource.
    expected = 2 * (64 * 128 * 4 + 32 * 128) + 2 * 64 * 128 * 2 + 2
    assert limited.now >= expected / 1e8 * 1000
    assert limited.now > 100 * quick.now


def test_scoreboard_partitions_the_selected_prepare_stage_without_omissions():
    from mindie_sim.prediction.native_fusion import _groups
    from tools.model_scoreboard_kimi import kda_stage_predictions
    case = frontend_case()
    _, ops = next((group, ops) for group, ops in _groups(
        target_operators(case, phase="prefill", tokens=64))
        if group["role"] == "kda_chunk")
    ledger = {"stage_ledger": [
        {"name": name, "mapping_status": "semantic", "recognized_product_flops": [],
         "unresolved_work": []}
        for name in ("GATE_PREPARE", "POST_WU", "FWD_H", "FINALIZE")]}
    rows = kda_stage_predictions(
        case, ops, RecipeBook.from_mapping(case.runtime.kernels), ledger)
    first = rows[0]
    assert len(first["semantic_member_ids"]) == 1
    assert first["predicted_semantic_projection_ms"] > 0
    assert "GATE_PREPARE" in first["boundary"]
    assert {row["dtype"] for row in first["source_recognized_product_flops"]} == {"bf16", "fp32"}


# ---------------------------------------------------------------------------
# Ordinary frontend, composition with FWD_H, and capability regression.
# ---------------------------------------------------------------------------

def frontend_model():
    model = load_model("kimi-k3")
    layer = replace(model.layer_plans[0], linear_heads=2,
                    linear_key_heads=2, linear_value_heads=2,
                    intermediate_size=512)
    return replace(model, layers=1, layer_plans=(layer,), name="kda-prepare",
                   hidden_size=256, num_heads=2, num_kv_heads=2, head_dim=128,
                   intermediate_size=512, vocab_size=256)


def frontend_case(implementation=IMPLEMENTATION, *, prompt=64):
    model = frontend_model()
    hardware = Hardware.from_mapping({
        "name": "kda-prepare-frontend", "num_devices": 1,
        "device": {"memory_bytes": 64 * 1024 ** 3, "compute_units": 4,
                   "compute_peaks": {"bf16": 1e14, "fp32": 2.5e13},
                   "rates": {"memory": 1e12, "vector": 2e12, "mte2": 1e12,
                             "mte3": 1e12, "mte1": 1e12, "fixpipe": 1e12,
                             "reg_special_lanes": 1e12},
                   "resource_units": {"vector": 8, "reg_special_lanes": 8}},
    })
    workload = WorkloadSpec.from_mapping({"requests": [
        {"id": "r", "prompt_tokens": prompt, "output_tokens": 1}]})
    runtime = Software.from_mapping({"kernels": {"lowerings": {
        "kda": implementation}}})
    return Case(hardware, model, workload, runtime)


@pytest.mark.parametrize("sequences", [1, 2])
def test_ordinary_frontend_replaces_gate_and_local_math_once(sequences):
    case = frontend_case()
    ops = target_operators(case, phase="prefill", tokens=64, sequences=sequences)
    source = [op for op in ops if op.recipe == kda_prepare.RECIPE]
    scans = [op for op in ops if op.recipe == kda_state.RECIPE]
    assert len(source) == 1 and len(scans) == 1
    assert dict(source[0].metadata)["source_phase"] == "GATE_PREPARE"
    assert dict(source[0].metadata)["compute_gate_in_prepare"] is False
    assert dict(source[0].metadata)["beta_mode"] == (
        "external_beta_sigmoid_preserved")
    assert source[0].source_boundary == scans[0].source_boundary == "kernel"
    assert dict(source[0].metadata)["lengths"] == (64,) * sequences
    assert not any(op.recipe.startswith(("chunk_products", "triangular_solve"))
                   or op.recipe == "delta_gates" for op in ops)
    assert any(op.recipe == "linear_beta_sigmoid" for op in ops)
    assert any(op.recipe == "wy_transform0" for op in ops)
    row = operator_costs(case, (source[0],))["operators"][0]
    assert row["program"] == "OperatorPlan" and row["duration_ms"] > 0
    assert row["live_tasks"] == 0
    graph = graph_costs(case, target_graphs(
        case, phase="prefill", tokens=64, sequences=sequences, scope="model"), isolated=False)
    selected = [item for item in graph["timeline"]
                if item["recipe"] == kda_prepare.RECIPE]
    assert len(selected) == 1
    assert graph["execution"]["live_tasks"] == 0
    call = call_costs(case, (replace(source[0], deps=(), requests=()),),
                      invocations=2, mode="graph")
    assert call["single_call"]["source_boundaries"]["kernel"] == 1
    assert call["operator_occurrences"] == 2
    assert call["duration_ms"] > 0


def test_selected_recipe_is_label_independent():
    first = frontend_case()
    second = frontend_case()
    second = replace(second, model=replace(second.model, name="other"))
    one = next(op for op in target_operators(
        first, phase="prefill", tokens=64) if op.recipe == kda_prepare.RECIPE)
    two = next(op for op in target_operators(
        second, phase="prefill", tokens=64) if op.recipe == kda_prepare.RECIPE)
    assert physical_signature(one) == physical_signature(two)


def test_state_selection_does_not_certify_unmodeled_decode_gate_source():
    model = frontend_model()
    layer = model.layer_plans[0]
    runtime = Software.from_mapping({"kernels": {"lowerings": {
        "kda": STATE_ONLY}}})
    result = linear(
        prefix="decode", plan=layer,
        views=(RequestView("d", 1, 0, False, "decode"),), stage=0,
        deps="input", model=model, parallel=ParallelSpec(tp=8),
        quant=model.quant, mode="decode", padded_tokens=1,
        policy=AttentionPolicy(recurrent_chunk=64), runtime=runtime)
    declaration = next(
        decl for op in result.ops
        for decl in dict(op.metadata).get("structure_subgraphs", ())
        if decl["role"] == "kda_recurrent")
    assert declaration["tensor_view"]["in_kernel_gate_source"] is False
    assert declaration["implementation"]["selection"] is None
    assert subgraph_contract(declaration) is None

    # The explicit in-kernel-gate implementation preserves its original intent.
    runtime = Software.from_mapping({"kernels": {"lowerings": {
        "kda": IN_KERNEL_GATES}}})
    old = linear(
        prefix="old", plan=layer,
        views=(RequestView("d", 1, 0, False, "decode"),), stage=0,
        deps="input", model=model, parallel=ParallelSpec(tp=8),
        quant=model.quant, mode="decode", padded_tokens=1,
        policy=AttentionPolicy(recurrent_chunk=64), runtime=runtime)
    old_decl = next(
        decl for op in old.ops
        for decl in dict(op.metadata).get("structure_subgraphs", ())
        if decl["role"] == "kda_recurrent")
    assert old_decl["tensor_view"]["in_kernel_gate_source"] is True
    assert subgraph_contract(old_decl) is not None
