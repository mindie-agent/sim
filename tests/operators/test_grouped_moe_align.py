"""Grouped MoE alignment, count-list conversion and source init routing.

[Role] Exercise routing preparation through source builders and the Engine.
[Invariants] Preserve expert intervals, workspace bounds, work and ownership.
[Timeouts & Locks] Finite analytical execution with bounded active CTA streams.
[Side-Effect Policy] Read retained case inputs and simulate in memory.
"""
from collections import Counter, deque
from dataclasses import replace
from itertools import accumulate
import json
from math import ceil
from pathlib import Path

import pytest

from _support.axes import PHASES
from _support.hardware import hardware_case
from mindie_sim import graph_costs, operator_costs, target_graphs, target_operators
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.engine import ExecutionError
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.family_moe_init import apply, capabilities
from mindie_sim.prediction.mechanism import Geometry, PlannedRecipe, Work
from mindie_sim.prediction.moe_cann_fusion import absorb_init_routing
from mindie_sim.prediction.moe_init_instructions import source_operations
from mindie_sim.prediction.moe_init_mxfp8 import RECIPE, row_runs
from mindie_sim.prediction.operators import moe_align
from mindie_sim.prediction.program import RecipeBook, RecipeContext
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.verification import verify_operator
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import IterationWork
from mindie_sim.schema.workload import IterationPlan, RequestView
from mindie_sim.structure_contracts import predicted_contract
from tools.run_operator_cases import control_case, run

pytestmark = pytest.mark.tier0
ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION = "cuda_moe_align_block_size"
INDEX_BYTES = 4
WARP = 32
SORT_THREADS = 256
SORT_GRID_LIMIT = 65535
ALIGN_THREADS = 1024
SMALL_EXPERT_LIMIT = 64
INIT_WIDTH = 4096
INIT_EXPERTS = 288
INIT_TOP_K = 8
MX_GROUP = 32
SIMT_THREADS = 2048

def _params(rows, experts=4, block=16, **options):
    return dict(topk_ids=rows, num_experts=experts, block_size=block,
                implementation=IMPLEMENTATION, pad_sorted_ids=True) | options

def _filtered(rows, placement, block=16):
    return _params(rows, len(placement), block, expert_map=placement, ignore_invalid_experts=True)

def _ledger(small, size, valid, experts, capacity, blocks, stride, input_bytes, mapped):
    reads = size * (input_bytes + INDEX_BYTES * mapped)
    filled = capacity * INDEX_BYTES
    indices = blocks * INDEX_BYTES
    put = valid * INDEX_BYTES
    stages = {"init_sorted": {"hbm": filled, "mte3": filled},
              "clear_counts": {"vector": stride * experts if small else WARP * ceil(experts / WARP)},
              "histogram": {"hbm": reads, "mte2": reads, "vector": size}}
    if small:
        stages["prefix"] = {"hbm": INDEX_BYTES, "mte3": INDEX_BYTES, "vector": (stride + 2) * experts}
        stages["expert_ids"] = {"hbm": indices, "mte3": indices, "vector": blocks}
        stages["scatter"] = {"hbm": reads + put, "mte2": reads, "mte3": put, "vector": size}
    else:
        workspace = (experts + 2) * INDEX_BYTES
        scan_reads = (2 * experts + ALIGN_THREADS) * INDEX_BYTES
        stages["scan"] = {"hbm": workspace, "mte3": workspace}
        stages["expert_ids"] = {"hbm": indices + scan_reads, "mte2": scan_reads,
                                "mte3": indices, "vector": blocks}
        stages["sort"] = {"hbm": reads + 3 * put, "mte2": reads + put, "mte3": 2 * put, "vector": size}
    return stages

@pytest.mark.parametrize("params", [
    _params([[0, 1]]), _params([[0, 1]], pad_sorted_ids=False, topk_ids_dtype="int8"),
    _params([[2, 0], [2, 1], [2, 0]], topk_ids_dtype="int64"),
    _params([[0]] * 70, block=4, pad_sorted_ids=False),
    _params([[0]] * 32, 8), _params([[i % 8] for i in range(32)], 8),
    _params([[i % 64] for i in range(1023)], 64, 1),
    _params([[i % 64] for i in range(1024)], 64, 1),
    _params([[i % 65] for i in range(1023)], 65, 1),
    _params([[0]], 65, 1), _params([[0]], 64, 1),
    _params([[i % 992] for i in range(1024)], 992, 1),
    _filtered([[0, 1], [2, 3], [1, 0], [3, 2], [0, 2]], [1, -1, 2, 0], 8),
    _filtered([[0]] * 10 + [[1]] * 990, [0, -1, -1, -1], 32),
    _filtered([[0]] + [[1]] * 19999, [0] + [-1] * 64),
    _filtered([[i % 65] for i in range(4096)], [i // 2 if i % 2 == 0 else -1 for i in range(65)]),
], ids=[
    "sparse", "unpadded-byte", "repeated-wide", "thread-order", "skew", "uniform",
    "below-size-limit", "at-size-limit", "over-expert-limit", "large-sparse", "small-sparse",
    "max-experts", "remap", "small-residual-cycle", "large-residual-cycle", "filtered-sort",
])
def test_alignment_layout_scatter_and_work_follow_exact_expert_intervals(params):
    op = moe_align.make_op(params)
    meta = dict(op.metadata)
    analysis, dispatch = meta["analysis"], meta["dispatch"]
    experts, block = params["num_experts"], params["block_size"]
    ids = [expert for row in params["topk_ids"] for expert in row]
    placement = params.get("expert_map", tuple(range(experts)))
    members = [tuple(i for i, value in enumerate(ids) if placement[value] == expert)
               for expert in range(experts)]
    counts = tuple(map(len, members))
    padded = tuple(block * ceil(count / block) for count in counts)
    ends = tuple(accumulate(padded, initial=0))
    size, valid = len(ids), sum(counts)
    small = size < ALIGN_THREADS and experts <= SMALL_EXPERT_LIMIT
    stride = max(experts, WARP) if small else ALIGN_THREADS
    capacity = size + experts * (block - 1)
    if params["pad_sorted_ids"]:
        capacity = block * ceil(capacity / block)
    if size < experts:
        capacity = min(capacity, size * block)
    blocks, used = ceil(capacity / block), ends[-1] // block
    assert (meta["counts"], meta["padded_per_expert"], meta["cumsum"]) == (counts, padded, ends)
    assert (op.padded_shape, op.launch_shape) == ((ends[-1],), (capacity,))
    assert analysis["workspace_bytes"] == {
        "sorted_token_ids": capacity * INDEX_BYTES, "expert_ids": blocks * INDEX_BYTES,
        "num_tokens_post_pad": INDEX_BYTES, "cumsum_buffer": 0 if small else (experts + 1) * INDEX_BYTES,
    }
    assert (analysis["valid_entries"], analysis["sentinel"], analysis["used_blocks"]) == (valid, size, used)
    mapped = "expert_map" in params
    assert analysis["has_expert_map_kernel_arg"] == mapped and not analysis["expert_ids_post_gather"]
    # A witness for atomic scatter uses flat order. Small scatter uses thread-major order.
    layout = [size] * op.launch_shape[0]
    reported = {row["expert"]: row for row in moe_align.iter_blocks(op)}
    assert set(reported) == {expert for expert, count in enumerate(counts) if count}
    for expert, row in reported.items():
        indices = members[expert]
        start, count = row["start_row"], meta["counts"][expert]
        assert (start, row["first_block"], row["blocks"], row["padded_rows"]) == (
            ends[expert], ends[expert] // block, padded[expert] // block, padded[expert])
        ordered = sorted(indices, key=lambda index: (index % stride, index // stride)) if small else indices
        layout[start:start + count] = ordered
        assert tuple(layout[start + count:ends[expert + 1]]) == (size,) * (padded[expert] - count)
    assert sorted(value for value in layout if value != size) == [i for i, expert in enumerate(ids) if placement[expert] >= 0]
    assert layout[ends[-1]:] == [size] * (capacity - ends[-1])
    assert size != 70 or layout[:6] == [0, 32, 64, 1, 33, 65]
    residual = sorted(index for tid in range(stride) for index in range(used + tid, blocks, stride))
    assert residual == list(range(used, blocks))
    assert (analysis["residual_writers"], analysis["tail_blocks"], analysis["tail_blocks_written"],
            analysis["expert_ids_blocks_written"]) == (stride, len(residual), len(residual), blocks)
    assert dispatch["small_batch_expert_mode"] is small
    shared = ((stride + 2) * experts + 1 if small else WARP * ceil(experts / WARP)) * INDEX_BYTES
    assert dispatch["shared_bytes"] == analysis["align_shared_bytes"] == shared
    assert dispatch["grid"] == ((1,) if small else (2,))
    assert dispatch["threads"] == (SORT_THREADS + stride if small else ALIGN_THREADS)
    if small:
        assert dispatch["sort_grid"] is dispatch["sort_symbol"] is None
    else:
        grid = min(ceil(size / SORT_THREADS), SORT_GRID_LIMIT)
        assert dispatch["sort_grid"] == (1, grid)
        assert (dispatch["sort_block_threads"], dispatch["scan_lanes"]) == (SORT_THREADS, ALIGN_THREADS)
        ctas = Counter((index // SORT_THREADS) % grid for index in range(size))
        active_ctas = Counter((index // SORT_THREADS) % grid for indices in members for index in indices)
        assert analysis["sort_cta_ids"] == tuple(ctas[i] for i in range(grid))
        assert analysis["sort_cta_valid"] == tuple(active_ctas[i] for i in range(grid))
    dtype = params.get("topk_ids_dtype", "int32")
    width = {"int8": 1, "int32": INDEX_BYTES, "int64": 2 * INDEX_BYTES}[dtype]
    expected = _ledger(small, size, valid, experts, capacity, blocks, stride, width, mapped)
    assert {name: work.resources for name, work in moe_align.stage_work(op)} == expected
    assert dict(meta["work"]["stage_work"]) == expected
    geometry = Geometry(32, {}, {})
    plan = moe_align.build(op, geometry)
    totals = {role: sum(work.get(role, 0) for work in expected.values())
              for role in ("hbm", "mte2", "mte3", "vector")}
    assert plan.work.resources == totals and not plan.work.matrix
    children = moe_align.kernels(op)
    phases = ("small_batch_expert",) if small else ("large_align", "large_sort")
    assert tuple(dict(child.metadata)["kernel_phase"] for child in children) == dispatch["kernel_phases"] == phases
    assert [(child.op_id, child.deps) for child in children] == (
        [("align", ())] if small else [("align", ()), ("sort", ("align",))])
    assert sum((moe_align.build(child, geometry).work for child in children), start=Work()) == plan.work

@pytest.mark.parametrize("field,value,error", [
    ("implementation", "triton_moe_align", "implementation"), ("pad_sorted_ids", 1, "must_be_bool"),
    ("ignore_invalid_experts", 0, "must_be_bool"), ("num_experts", 0, "positive_int"),
    ("num_experts", 993, "below_1024"), ("block_size", 0, "positive_int"),
    ("phase", "other", "phase"), ("topk_ids_dtype", "float32", "unsupported_topk_ids_dtype"),
    ("topk_ids", [], "at_least_one_token"), ("topk_ids", [[]], "top_k_must_be_positive"),
    ("topk_ids", [[0, 1], [2]], "constant_top_k"), ("topk_ids", [[0, 4]], "below_num_experts"),
    ("topk_ids", [[0, -1]], "nonnegative"), ("expert_map", [0], "expert_map_length"),
    ("expert_map", [0, 1, 2, 4], "expert_map_local_id"), ("expert_map", [0, 1, 2, -2], "expert_map_values"),
    ("expert_map", [0, 1, 2, 3], "requires_native_layout"),
])
def test_invalid_alignment_inputs_raise_at_the_wrapper(field, value, error):
    with pytest.raises(ValueError, match=error):
        moe_align.make_op(_params([[0, 1]]) | {field: value})

@pytest.mark.parametrize("phase", (*PHASES, "mixed"))
def test_alignment_keeps_declared_phase(phase):
    assert moe_align.make_op(_params([[0, 1]], phase=phase)).phase == phase

def test_alignment_planners_reject_missing_or_incompatible_metadata():
    with pytest.raises(ValueError, match="params_must_be_mapping"):
        moe_align.make_op([])
    op = moe_align.make_op(_params([[0, 1]]))
    metadata = dict(op.metadata)
    for bad, reason in (
        (replace(op, recipe="other"), "selected_recipe"),
        (replace(op, metadata=tuple((metadata | {"dispatch": metadata["dispatch"] |
                                                 {"implementation": "other"}}).items())), "selected_implementation"),
        (replace(op, metadata=tuple((key, value) for key, value in op.metadata if key != "analysis")),
         "make_op_analysis"),
    ):
        with pytest.raises(ValueError, match=reason):
            moe_align.build(bad, Geometry(1, {}, {}))
        with pytest.raises(ValueError, match=reason):
            moe_align.kernels(bad)
    with pytest.raises(ValueError, match="parent_op"):
        moe_align.kernels(moe_align.kernels(op)[0])
    metadata["dispatch"] = metadata["dispatch"] | {"kernel_phases": ("unknown",)}
    with pytest.raises(ValueError, match="button_mismatch"):
        moe_align.build(replace(op, metadata=tuple(metadata.items())), Geometry(1, {}, {}))

def _execute_align(op, raw, *, scope="none", trace_limit=4000):
    case = parse_case(raw)
    recipes = RecipeBook(recipes={IMPLEMENTATION: PlannedRecipe(moe_align.build, moe_align.kernels)})
    execution = Execution(case, CostModel(case), recipes=recipes, trace_limit=trace_limit)
    plan = IterationPlan((RequestView("r", 1, 0, True, "prefill"),))
    work = IterationWork((op,), plan, 1, 1, 1, None, "prefill", "align", replay_scope=scope)
    scheduled = execution.add(work, 0, 0)
    execution.engine.advance(until=(scheduled.completion,))
    return execution.engine, scheduled

@pytest.mark.parametrize("large", [False, True])
@pytest.mark.parametrize("scope", ["none", "full"])
def test_alignment_launches_and_shared_pool_follow_physical_ctas(large, scope):
    op = moe_align.make_op(_filtered([[i % 4] for i in range(1024 if large else 2)], [1, -1, 2, 0]))
    shared = dict(op.metadata)["analysis"]["align_shared_bytes"]
    raw = control_case(units=4, split_copy=True)
    raw["machine"]["device"].update(launch_us=50, buffers={"shared": shared}, buffer_units={"shared": 4})
    engine, scheduled = _execute_align(op, raw, scope=scope)
    submits = [row for row in engine.trace if row["task"].endswith(":submit")]
    count = 2 if large and scope == "none" else 1
    assert [row["start_ms"] for row in submits] == [0.05 * i for i in range(count)]
    assert engine.serviced["host@r0:s0"] == pytest.approx(count * .05)
    assert scheduled.completion.time == pytest.approx(count * .05, abs=1e-3)
    summary = engine.summary()
    assert summary["peak_allocated"]["shared@r0:s0"] == shared
    assert summary["peak_allocated_per_unit"] == {
        f"shared@r0:s0:unit{i}": shared for i in range(2 if large else 1)}
    assert not any(summary["live_allocated"].values())
    assert summary["live_tasks"] == summary["live_programs"] == 0
    raw["machine"]["device"]["buffers"]["shared"] -= INDEX_BYTES
    with pytest.raises(ExecutionError, match="allocation_exceeds_capacity"):
        _execute_align(op, raw, scope=scope)

@pytest.mark.parametrize("units", [1, 4])
def test_alignment_ctas_share_only_their_physical_owner_slot(units):
    names = {"vector": "vector", "hbm": "memory", "mte2": "read", "mte3": "write"}
    resources = {"vector": 2 * units if units > 1 else 1, "mte2": units, "mte3": units}
    for size in (2, 1024):
        op = moe_align.make_op(_params([[i % 4] for i in range(size)]))
        child = moe_align.kernels(op)[0]
        context = RecipeContext(child, {}, names, dict.fromkeys(names.values(), 1e9), units,
                                resource_units=resources, pool_names={}, pool_units={})
        program = moe_align.build(child, Geometry(units, resources, {}, {})).program(context)
        assert len(program.stages) == (1 if size == 2 else 2)
        for index, cta in enumerate(program.stages):
            owner = index % units
            vector = (2 * owner, 2 * owner + 1) if units > 1 else (0,)
            for stage in cta.program.stages:
                assert stage.placement == {"vector": vector, "read": (owner,), "write": (owner,)}
                assert not stage.caps

def test_prefix_waits_for_both_source_groups_and_single_cta_rate_is_invariant():
    op = moe_align.make_op(_params([[0, 1]]))
    for units in (1, 2, 32):
        raw = control_case(units=units, split_copy=True)
        device = raw["machine"]["device"]
        device["rates"].update(memory=1e20, read=1e20, write=128000 * units, vector=1360000 * units)
        device["resource_units"].update(vector=units, mte2=units, mte3=units)
        device["launch_us"] = 0
        engine, scheduled = _execute_align(op, raw)
        leaves = {row["task"].rsplit(":", 1)[-1]: row for row in engine.trace}
        assert leaves["prefix"]["start_ms"] >= max(leaves[key]["end_ms"] for key in ("histogram", "init_sorted"))
        assert scheduled.completion.time == pytest.approx(1000 * (144 / 128000 + 136 / 1360000))

def test_sort_tiles_are_lazy_owned_streams_and_bound_live_engine_work():
    experts, owners = 128, 4
    op = moe_align.make_op(_filtered(
        [[(i * 7 + lane) % experts for lane in range(8)] for i in range(20000)],
        [i if i % 2 == 0 else -1 for i in range(experts)], 128))
    child = moe_align.kernels(op)[1]
    resources = {"vector": 2 * owners, "mte2": owners, "mte3": owners}
    roles = (*resources, "hbm")
    context = RecipeContext(child, {}, dict(zip(roles, roles)), dict.fromkeys(roles, 1e9),
                            owners, resource_units=resources, pool_names={}, pool_units={})
    program = moe_align.build(child, Geometry(owners, resources, {}, {})).program(context)
    streams = program.stages[0].program.programs
    grid = ceil(20000 * 8 / SORT_THREADS)
    assert len(streams) == owners
    assert [stream.count for stream in streams] == [len(range(i, grid, owners)) for i in range(owners)]
    billed = Counter()
    for owner, stream in enumerate(streams):
        for index in range(stream.count):
            tile, = stream.tile(index)
            billed.update(tile.work)
            assert tile.placement["vector"] == (2 * owner, 2 * owner + 1)
            assert not tile.caps
    assert billed == dict(moe_align.stage_work(op))["sort"].resources
    engine, _ = _execute_align(op, control_case(units=32), trace_limit=0)
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0
    assert engine.peak_tasks < SORT_THREADS and not engine.peak_allocated

def test_public_alignment_cases_conserve_work_and_identity_in_both_transfer_modes():
    cases = json.loads((ROOT / "tests/fixtures/operator_cases/moe_align.json").read_text())["cases"]
    result = run(moe_align, cases)
    assert result["status"] == "passed"
    assert {row["split"] for row in result["cases"]} == {"development", "heldout_geometry"}
    assert all(row["identity_invariant"] and not row["dispatch_mismatches"] for row in result["cases"])
    op = moe_align.make_op(_filtered([[i % 4] for i in range(1024)], [1, -1, 2, 0]))
    for split in (False, True):
        report = verify_operator(control_case(split_copy=split), op, moe_align.build, kernels=moe_align.kernels)
        assert report["status"] == "passed" and report["buffers_released"] and report["finite_completion"]
        assert all(not row["service_delta"] and row["live_tasks"] == 0 for row in report["result"]["kernels"])
        assert any(key.startswith("mte2@") for key in report["result"]["serviced"]) is split

@pytest.mark.parametrize("batch", [1, 4])
def test_count_list_scans_empty_experts_once_and_fused_consumer_waits(batch):
    path = ROOT / "examples/cases/longctx-decode-20261030/qwen35-35b-a3-tp4.json"
    raw = json.loads(path.read_text())
    raw["model"]["layers"] = 2
    raw["runtime"].setdefault("kernels", {})["lowerings"] = {
        "group_list_prefix": "vllm_ascend_group_list_prefix",
        "native_fusion": {"implementation": "declared_native_fusion",
                          "options": {"roles": ["grouped_gate_up_activation_quant"]}},
    }
    options = dict(phase="decode", tokens=2, sequences=batch, context=4096, scope="body")
    ops = target_graphs(raw, **options)[0]
    prefixes = [op for op in ops if op.recipe == "expert_group_list_cumsum"]
    experts = 256
    traffic = (2 * experts + 1) * 8
    assert len(prefixes) == raw["model"]["layers"]
    for scan in prefixes:
        assert (scan.local_shape, scan.hbm_bytes, dict(scan.metadata)["logical_integer_adds"]) == ((experts,), traffic, experts - 1)
        fused = next(op for op in ops if op.layer == scan.layer
                     and dict(op.metadata).get("fusion_role") == "grouped_gate_up_activation_quant")
        assert scan.op_id in fused.deps
        assert scan.op_id not in dict(fused.metadata)["fusion_semantic_ids"]
        down = next(op for op in ops if op.layer == scan.layer and op.recipe == "balanced_grouped_down")
        assert "group_list_producer" not in dict(down.metadata)
        report = operator_costs(raw, [scan], input_ready_ms=7)["operators"][0]
        assert report["serviced"] == {"memory@r0:s0": traffic}
        assert not any(report["service_before_input"].values()) and report["live_tasks"] == 0
    report = graph_costs(raw, {0: ops}, external_inputs={"analysis_input": 0}, isolated=False)
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    raw["runtime"]["kernels"]["service_regimes"] = [dict(id="scan-control", match={"recipe": prefixes[0].recipe},
        response={"fixed_us": 40}, provenance={"kind": "synthetic-control"})]
    inclusive = operator_costs(raw, prefixes[:1])["operators"][0]
    assert (inclusive["duration_ms"], inclusive["serviced"]["memory@r0:s0"]) == pytest.approx((.04, traffic))
    del raw["runtime"]["kernels"]["lowerings"]["group_list_prefix"]
    assert all(op.recipe != "expert_group_list_cumsum" for op in target_graphs(raw, **options)[0])

def _init_case(batch, *, selected=True, fused=False):
    path = ROOT / "examples/cases/high-concurrency-20261030/glm-flash-ws9"
    raw = json.loads((path / f"a5_glm53flash_tp8_c{batch}_p4k.json").read_text())
    bindings = raw["runtime"].setdefault("kernels", {}).setdefault("lowerings", {})
    if selected:
        bindings["moe_init_routing"] = RECIPE
    if fused:
        role = "grouped_gate_up_activation_quant"
        bindings["native_fusion"] = {"implementation": "declared_native_fusion", "options": {
            "roles": [role], "program_profiles": {role: "ascend_mxfp8_grouped_swiglu_quant"},
            "program_options": {role: {"l1_k": 512}},
        }}
    return raw

def _init_graph(raw, batch):
    return target_graphs(raw, scope="body", phase="decode", tokens=4, context=4096, sequences=batch)[0]

@pytest.mark.parametrize("batch", [16, 32, 64])
def test_init_replaces_pack_and_up_quantizer_and_conserves_expanded_outputs(batch):
    raw = _init_case(batch)
    baseline = {op.op_id: op for op in _init_graph(_init_case(batch, selected=False), batch)}
    ops = {op.op_id: op for op in _init_graph(raw, batch)}
    selected = [op for op in ops.values() if op.recipe == RECIPE]
    assert len(selected) == sum(key.endswith(".expert_up.quantize") for key in baseline) > 0
    assert [key for key in ops if key.endswith(".expert_down.quantize")] == [
        key for key in baseline if key.endswith(".expert_down.quantize")]
    assert not any(key.endswith(".expert_up.quantize") for key in ops)
    tokens, width, assignments = batch * 4, INIT_WIDTH, batch * 4 * INIT_TOP_K
    for op in selected:
        assert dict(op.metadata)["moe_init_config"] == dict(
            tokens=tokens, topk=INIT_TOP_K, width=width, experts=INIT_EXPERTS)
        up = ops[op.op_id.replace(".pack", ".expert_up")]
        assert up.deps == (op.op_id,)
        assert dict(up.metadata)["activation_scale_producer"] == op.op_id
        assert (up.flops, up.hbm_bytes) == (baseline[up.op_id].flops, baseline[up.op_id].hbm_bytes)
        contract = predicted_contract(op.to_dict())
        assert contract[2][0] == ((tokens, width), "bf16", "ND")
        assert contract[3][3] == ((assignments, width // MX_GROUP), "fp8_e8m0", "ND")
    row = operator_costs(raw, (selected[0],), input_ready_ms=7)["operators"][0]
    physical = row["operator"]
    program = physical["metadata"]["program_execution"]
    assert (program["quantized_input_rows"], program["assignments"]) == (tokens, assignments)
    assert (program["output_data_bytes"], program["output_scale_bytes"]) == (assignments * width, assignments * width // MX_GROUP)
    assert physical["mte2_bytes"] == 0 and physical["metadata"]["resource_work"]["mte2_aiv"] > tokens * width * 2
    assert row["live_tasks"] == 0 and not (row["service_before_input"] or any(row["remaining_buffers"].values()))
    assert all(abs(value) < 1e-6 for value in row["service_delta"].values())

@pytest.mark.parametrize("tokens,topk,units", [(3, 8, 4), (7, 5, 8), (9, 6, 4)])
def test_init_reloads_only_tokens_split_across_owner_intervals(tokens, topk, units):
    total = tokens * topk
    per = ceil(total / units)
    owners = [Counter(slot // topk for slot in range(start, min(start + per, total)))
              for start in range(0, total, per)]
    for owner, expected in enumerate(owners):
        assert list(row_runs(total, topk, units, owner)) == list(expected.items())
    assert sum(sum(counts.values()) for counts in owners) == total
    assert sum(map(len, owners)) == tokens + sum(start % topk != 0 for start in range(per, total, per))

def test_init_gates_and_fused_activation_dependency():
    caps = capabilities()
    assert (caps["input_dtypes"], caps["scale_group_size"]) == (("bf16",), MX_GROUP)
    assert "no_moe_transport_quantization" in caps["requires"]
    raw = _init_case(16, selected=False)
    ops = _init_graph(raw, 16)
    pack = next(op for op in ops if op.recipe == "route_pack")
    up = next(op for op in ops if op.recipe in {"grouped_gate_up", "balanced_grouped_gate_up"})
    up, binding = replace(up, deps=(pack.op_id,)), {caps["selector"]: {"implementation": RECIPE}}
    runtime = parse_case(raw).runtime
    with pytest.raises(ModelingError, match="width_multiple_of_256"):
        apply((pack, replace(up, local_shape=(*up.local_shape[:2], 100))), binding, runtime)
    with pytest.raises(ModelingError, match="complete_BF16_to_E4M3_MX32"):
        apply((replace(pack, dtype="fp16"), up), binding, runtime)
    quantizer = next(op for op in ops if op.op_id.endswith(".expert_up.quantize"))
    meta = dict(quantizer.metadata, quantization_scope="moe_dispatch_expert_input")
    quantizer = replace(quantizer, metadata=tuple(meta.items()))
    absorbed, retired = absorb_init_routing(pack=pack, quantizer=quantizer, ep=8)
    assert retired == quantizer.op_id and absorbed.vector_ops == pack.vector_ops + quantizer.vector_ops
    incompatible = replace(quantizer, metadata=tuple(dict(meta, activation_group_size=128).items()))
    for op, ep, owns_quant in ((quantizer, 1, False), (quantizer, 8, True), (incompatible, 8, False)):
        assert absorb_init_routing(pack=pack, quantizer=op, ep=ep, dispatch_owns_quant=owns_quant) == (pack, None)
    raw = _init_case(16, fused=True)
    ops = _init_graph(raw, 16)
    init = next(op for op in ops if op.recipe == RECIPE)
    up = next(op for op in ops if op.layer == init.layer and "fusion_program_profile" in dict(op.metadata))
    assert init.op_id in up.deps and not any(op.op_id.endswith(".expert_up.quantize") for op in ops)
    external = {dep: 11 for op in (init, up) for dep in op.deps if dep != init.op_id}
    report = graph_costs(raw, [init, up], external_inputs=external)
    rows = {row["op_id"]: row for row in report["timeline"]}
    assert rows[up.op_id]["ready_ms"] >= rows[init.op_id]["complete_ms"] >= rows[init.op_id]["ready_ms"] >= 11
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0

@pytest.mark.parametrize("axis", ["mte2_aiv", "mte3", "vector", "hbm"])
def test_init_duration_responds_to_each_declared_physical_rate(axis):
    raw = _init_case(64)
    op, case = next(op for op in _init_graph(raw, 64) if op.recipe == RECIPE), parse_case(raw)
    results = []
    for factor in (.5, 1, 2):
        dies = tuple(replace(die, rates=tuple((key, rate * factor if key == axis else rate)
                                             for key, rate in die.rates)) for die in case.machine.dies)
        scaled = replace(case, machine=replace(case.machine, dies=dies))
        results.append(operator_costs(scaled, (op,))["operators"][0]["duration_ms"])
    assert results[0] >= results[1] >= results[2] and results[0] > results[2]

def _source_program(op, services):
    raw = _init_case(16)
    raw["runtime"]["kernels"]["invocations"] = [{"match": {}, "startup_ms": 0, "minimum_ms": 0}]
    raw["machine"].setdefault("device", {})["compute_arch"] = {"instruction_services": {
        role: {"unit": "input_blocks32" if role.endswith("32_blocks") else "int32_thread_accesses",
               "owner_axis": "vector", "issue_ns": rate, "provenance": {"kind": "synthetic-control"}}
        for role, rate in services.items()}}
    case = parse_case(raw)
    execution = Execution(case, CostModel(case), recipes=RecipeBook.from_mapping(case.runtime.kernels))
    prepared, recipe = execution.prepare(op, 0)
    context = execution._context(prepared, 0, None, recipe=recipe)
    engine = execution.engine
    engine.trace = deque(maxlen=12000)
    end = execution.lower_program(context, recipe).start(engine, "source", engine.submit("ready", release=7))
    engine.advance(horizon=7)
    assert not any(engine.serviced.values())
    engine.advance(until=(end,))
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0
    assert not any(engine.allocated.values())
    meta = dict(prepared.metadata)["program_execution"]["source_operation_services"]
    return end.time - 7, engine, meta

@pytest.mark.parametrize("factor", [.5, 1, 2])
def test_sort_macros_serially_price_owner_zero_without_erasing_known_work(factor):
    op = next(op for op in _init_graph(_init_case(16), 16) if op.recipe == RECIPE)
    before, baseline, unavailable = _source_program(op, {})
    intervals = dict(source_concat32_blocks=2000 / factor, source_sort32_blocks=3000 / factor,
                     source_extract32_blocks=5000 / factor)
    after, engine, meta = _source_program(op, intervals)
    blocks = ceil(16 * 4 * INIT_TOP_K / WARP)
    assert after - before == pytest.approx(blocks * sum(intervals.values()) / 1e6)
    assert {key: engine.serviced[key] for key in baseline.serviced} == pytest.approx(baseline.serviced)
    assert any(row["task"].endswith("sort_known_lane_work") for row in baseline.trace)
    assert not any(row["task"].endswith("_macro") for row in baseline.trace)
    assert unavailable["sort"]["sort"]["unpriced_service_work"] == {"source_sort32_blocks": blocks}
    assert meta["sort"]["sort"]["priced_service_work"] == {"source_sort32_blocks": blocks}
    rows = [row for row in engine.trace if row["task"].endswith("_macro")]
    for row, (role, rate) in zip(rows, intervals.items(), strict=True):
        resource = role + "@r0:s0"
        assert (row["serviced"][resource], row["placement"][resource]) == (blocks, (0,))
        assert row["end_ms"] - row["start_ms"] == pytest.approx(blocks * rate / 1e6)
    post = next(row for row in engine.trace if row["task"].endswith("sort_post_lane_work"))
    assert all(left["end_ms"] <= right["start_ms"] for left, right in zip(rows, (*rows[1:], post)))

@pytest.mark.parametrize("assignments,units", [(15, 64), (65, 8), (1024, 40), (4099, 2)])
def test_simt_count_and_inverse_accesses_follow_thread_iterations(assignments, units):
    per = ceil(assignments / units)
    _, counts, inverse = source_operations(assignments, per, ceil(assignments / per), INIT_EXPERTS)
    for owner, (count, scatter) in enumerate(zip(counts, inverse)):
        rows = min(per, assignments - owner * per)
        visits = [i for thread in range(SIMT_THREADS) for i in range(thread, rows, SIMT_THREADS)]
        assert count.counts == {"source_simt_gm_load32": sum(2 + (i > 0) + (i < rows - 1) for i in visits)}
        assert scatter.counts == {"source_simt_gm_load32": rows, "source_simt_gm_store32": rows}

def test_simt_service_capacity_is_per_owner_and_keeps_payload():
    op = next(op for op in _init_graph(_init_case(16), 16) if op.recipe == RECIPE)
    meta = dict(op.metadata)
    meta["moe_init_config"] = dict(meta["moe_init_config"], tokens=3, topk=5)
    op = replace(op, metadata=tuple(meta.items()))
    _, baseline, unavailable = _source_program(op, {})
    load, store = "source_simt_gm_load32", "source_simt_gm_store32"
    _, engine, priced = _source_program(op, {load: 2000, store: 3000})
    assert tuple(engine.serviced[key + "@r0:s0"] for key in (load, store)) == pytest.approx((3 * 15, 15))
    assert {key: engine.serviced[key] for key in baseline.serviced} == pytest.approx(baseline.serviced)
    assert {row["placement"][load + "@r0:s0"] for row in engine.trace if load + "@r0:s0" in row["demand"]} == {
        (i,) for i in range(15)}
    assert priced["atomic_count"]["payload_bytes_per_owner"] == INIT_EXPERTS * INDEX_BYTES
    assert sum(row["counts"][load] for row in unavailable["count_by_owner"]) == 2 * 15
    assert not any("atomic_rmw" in key or "simt_warp" in key for key in engine.serviced)

@pytest.mark.parametrize("tokens", [5, 512])
def test_source_alignment_retains_payload_pack_and_orders_native_kernels(tokens):
    raw = hardware_case()
    raw["model"].update(hidden_size=130, ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32)
    ids = [(i % 2, 2) for i in range(tokens)]
    route = dict(scope="body", layer=0, phase="prefill", tokens=tokens, topk_ids=ids,
                 expert_rows=[ceil(tokens / 2), tokens // 2, tokens, 0])
    raw["runtime"] = {"routing": {"realizations": [route]}}
    options = dict(phase="prefill", tokens=tokens, scope="body")
    original = next(op for op in target_operators(raw, **options) if op.op_id == "s0.l0.pack")
    raw["runtime"]["kernels"] = {"lowerings": {
        "route_pack": {"implementation": IMPLEMENTATION, "options": {"block_size": 16, "pad_sorted_ids": True}},
        "route_unpack": {"implementation": "triton_moe_fused_mul_sum",
                         "options": {"compute_capability": 89, "weight_dtype": "bf16"}}}}
    graph = target_graphs(raw, **options)
    ops = {op.op_id.split(".", 2)[2]: op for op in graph[0] if op.op_id.startswith("s0.l0.")}
    align, pack, materialize, combine = (ops[key] for key in ("pack.align", "pack", "unpack.materialize", "unpack"))
    assert align.deps == ("s0.l0.topk",) and align.op_id in pack.deps
    analysis = dict(align.metadata)["analysis"]
    assert [row["expert"] for row in moe_align.iter_blocks(align)] == [0, 1, 2]
    index_read = INDEX_BYTES * (analysis["total_padded"] + analysis["used_blocks"] + 1)
    assert pack.mte2_bytes == original.mte2_bytes - tokens * 2 * INDEX_BYTES + index_read
    assert pack.mte3_bytes == original.mte3_bytes == tokens * 2 * (130 * 2 + INDEX_BYTES)
    assert pack.vector_ops == original.vector_ops + analysis["total_padded"]
    assert dict(pack.metadata)["pack_contract"]["sentinel_slots"] == analysis["total_padded"] - tokens * 2
    assert materialize.local_shape == (tokens, 2, 130)
    assert {ops["expert_down"].op_id, pack.op_id, ops["topk"].op_id} == set(materialize.deps)
    meta = dict(combine.metadata)
    assert combine.deps == (materialize.op_id, ops["topk"].op_id)
    assert (meta["last_tile_rows"], meta["last_tile_cols"], meta["output_write_bytes"]) == (
        (tokens - 1) % meta["block_m"] + 1, 130, tokens * 130 * 2)
    report = graph_costs(raw, graph, external_inputs={"analysis_input": 0})
    calls = [row for row in report["timeline"] if "pack.align/" in row["op_id"]]
    assert len(calls) == (2 if tokens * 2 >= ALIGN_THREADS else 1)
    assert len(calls) == 1 or calls[0]["id"] in calls[1]["predecessors"]
    route["topk_ids"] = ids[1:] + ids[:1]
    selected = next(op for op in target_operators(raw, **options) if op.family == "moe_align")
    assert dict(selected.metadata)["routing_realization"]["topk_ids"] == tuple(route["topk_ids"])
    route["topk_ids"] = [[2, 3]] * tokens
    with pytest.raises(ModelingError, match="must_agree_with_expert_rows"):
        target_operators(raw, **options)
    del route["topk_ids"]
    with pytest.raises(ModelingError, match="explicit_topk_ids"):
        target_operators(raw, **options)


@pytest.mark.parametrize("issued", [1, 8])
def test_balanced_capture_separates_logical_rows_from_issued_pack_and_scale_rows(issued):
    raw = hardware_case()
    raw["machine"].update(num_devices=2, links=[{"name": "fabric", "members": [0, 1], "capacity_bytes_per_ms": 1000}])
    raw["model"].update(hidden_size=32, num_heads=4, num_kv_heads=2, ffn="moe", num_experts=4, top_k=2,
                        moe_intermediate_size=32, quant=quant_preset("w8a8_int8").to_dict())
    transport = dict(kind="separate_quantized_payload", payload_dtype="int8", scale_dtype="fp32",
                     scale_granularity="row", histogram_dtype="int64")
    raw["runtime"] = dict(tp=2, ep=2, routing={"policy": "balanced"}, kernels={"moe_transport": transport})
    if issued > 1:
        raw["runtime"].update(graph_mode="auto", graph_capture_keys=[issued], graph_capture_mode="full")
    ops = {op.op_id.removeprefix("s0.l0."): op for op in target_graphs(raw, phase="decode", tokens=1, scope="body")[0]}
    gate = dict(ops["expert_up"].metadata)
    route = gate["routing_realization"]
    assert (route["valid_local_tokens"], route["padded_local_tokens"], sum(route["rank_assignment_rows"])) == (1, issued, 2 * issued)
    assert dict(ops["moe_histogram"].metadata)["source_assignments"] == issued
    assert ops["moe_transport_quantize"].mte3_bytes == issued * (32 + INDEX_BYTES)
    assert gate["activation_scale_bytes"] == issued * INDEX_BYTES
    assert tuple(ops[key].link_bytes for key in ("moe_payload_dispatch", "moe_scale_dispatch")) == (issued * 16, issued * 2)
