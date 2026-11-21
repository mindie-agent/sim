"""M2 Round-1 P1 review closure: physical, conservation and silicon anchors."""
from __future__ import annotations

from dataclasses import replace
import math
from types import SimpleNamespace

import pytest

from mindie_sim import operator_costs
from mindie_sim.backends.ascend import materialize
from mindie_sim.catalog.ops import gemm_op, vector_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.compose import compose_case
from mindie_sim.prediction.operators.attention.batch_prefix_attention import apply_attention_prefix_reads
from mindie_sim.prediction.call_builders import build_call
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.efficiency import analyze_network_efficiency
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.operators.matrix.matrix_mapping import matrix_geometry
from mindie_sim.prediction.operators.matrix.matrix_splitk import splitk_program
from mindie_sim.prediction.operators.matrix.matrix_steady import SteadyMatmul
from mindie_sim.prediction.operator_analysis import target_graphs, target_operators
from mindie_sim.prediction.physical_recipe_base import Phase, _phase_amounts
from mindie_sim.prediction.physical_recipe_registry import (
    ControlSyncRecipe, resolve_physical_recipe,
)
from mindie_sim.prediction.program import RecipeContext
from mindie_sim.prediction.resource_attribution import attribute_stage_dag
from mindie_sim.prediction.roofline_graph import (
    PhysicalWorkNode, RooflineTargets, critical_path_roofline,
    graph_from_occurrences,
)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.public.environment import ExecutionEnvironment
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.workload import IterationPlan, RequestView
from mindie_sim.sweep import sweep_topologies

MIB = 1024 * 1024
A5_SKU = "Ascend950DT_9582"
INDEX_ROW_BYTES = 256
INDEX_DIM = 128
INDEX_HEADS = 64
PREFILL_TOKENS = 4096
DEEPSEEK_LAYERS = 61
INDEX_QUERY_TILE = 128
INDEX_KEY_BYTES_PER_LAYER = PREFILL_TOKENS * INDEX_ROW_BYTES
INDEX_REFILL_ROWS_PER_LAYER = sum(
    INDEX_QUERY_TILE * (index + 1)
    for index in range(PREFILL_TOKENS // INDEX_QUERY_TILE)
)
ATOMIC_BYTES = 4
POOL_CONTEXT = 32768
POOL_SIZE = 4
POOL_UNIQUE_ROWS = POOL_CONTEXT // POOL_SIZE
QO_BYTES = 16 * MIB
SHARED_PREFIX_ROWS = 4096
QUERY_ROWS = 1024
SHARED_ROW_BYTES = 1024


def _a5_case(model="qwen3.5-35b-a3b", **kwargs):
    return parse_case(compose_case(
        model, hardware=A5_SKU, num_devices=8, ranks_per_node=8, **kwargs))


def _allreduce_op(world_size: int) -> OpSpec:
    payload = 128 * MIB
    volume = 2.0 * (world_size - 1) / world_size * payload
    return OpSpec(
        "probe.allreduce", "allreduce", "prefill", 0, 0, (), (1,), (1,), (1,), (1,),
        link_bytes=volume,
        metadata=(("group_axis", "tp"), ("steps", 2 * (world_size - 1))))


# --------------------------------------------------------------------------- #
# [P1 #1] index key reads are the physical tile-union, not score pairs.
# --------------------------------------------------------------------------- #
def test_deepseek_4k_prefill_index_scores_read_each_key_union_once():
    case = parse_case(compose_case(
        "deepseek-v3.2", hardware=A5_SKU, tp=8, ep=8, num_devices=8,
        ranks_per_node=8, prompt_tokens=PREFILL_TOKENS, output_tokens=2,
        concurrency=1))
    graphs = target_graphs(
        case, phase="prefill", tokens=PREFILL_TOKENS, context=0, sequences=1,
        scope="body")
    ops = next(iter(graphs.values())) if isinstance(graphs, dict) else graphs
    scores = [op for op in ops if op.recipe == "quant_lightning_indexer"]
    assert len(scores) == DEEPSEEK_LAYERS
    first = scores[0]
    by_id = {op.op_id: op for op in ops}
    operand_bytes = sum(by_id[dep].mte3_bytes for dep in first.deps[:2])
    # The fused native score+TopK call reads Q, the head weights and every
    # per-tile key refill on MTE2; the external HBM union is charged once.
    assert first.mte2_bytes - operand_bytes == pytest.approx(
        INDEX_REFILL_ROWS_PER_LAYER * INDEX_ROW_BYTES)
    assert first.hbm_bytes - operand_bytes - first.mte3_bytes == pytest.approx(
        INDEX_KEY_BYTES_PER_LAYER)
    # Per-query arithmetic is unchanged; only the external key fetch shrinks.
    assert first.flops == pytest.approx(
        2 * (PREFILL_TOKENS * (PREFILL_TOKENS + 1) / 2)
        * INDEX_HEADS * INDEX_DIM)
    assert sum(op.mte2_bytes - sum(by_id[dep].mte3_bytes for dep in op.deps[:2])
               for op in scores) == pytest.approx(
        DEEPSEEK_LAYERS * INDEX_REFILL_ROWS_PER_LAYER * INDEX_ROW_BYTES)
    assert sum(op.hbm_bytes - sum(by_id[dep].mte3_bytes for dep in op.deps[:2])
               - op.mte3_bytes for op in scores) == pytest.approx(
        DEEPSEEK_LAYERS * INDEX_KEY_BYTES_PER_LAYER)


# --------------------------------------------------------------------------- #
# [P1 #2] A cold pool scan charges every visited pool row.
# --------------------------------------------------------------------------- #
def test_pooled_indexer_cold_scan_charges_visited_pool_rows():
    case = parse_case(compose_case(
        "glm-5.3-flash", hardware="a5", tp=8, ep=8, num_devices=8,
        ranks_per_node=8, prompt_tokens=POOL_CONTEXT, output_tokens=2,
        concurrency=1))
    graphs = target_graphs(
        case, phase="decode", tokens=1, context=POOL_CONTEXT, sequences=1,
        scope="body")
    ops = next(iter(graphs.values())) if isinstance(graphs, dict) else graphs
    scores = next(op for op in ops if op.recipe == "cann_pool_key_indexer")
    metadata = dict(scores.metadata)
    assert metadata["scanned_pools"] == POOL_UNIQUE_ROWS
    assert metadata["pool_row_bytes"] == INDEX_ROW_BYTES
    pool_read = metadata["safe_reference_scanned_pools"] * INDEX_ROW_BYTES
    # The native PoolKeyIndexer envelope includes every visited pool row in
    # both its MTE2 and external-byte read ledgers.
    assert scores.mte2_bytes >= pool_read
    assert scores.hbm_bytes - scores.mte3_bytes >= pool_read
    assert not any(op.recipe == "index_pool_scores" for op in ops)
    topk = next(op for op in ops if op.recipe == "index_topk")
    # The native caller publishes expanded token indices; the adapter reads
    # them back for the retained selection boundary.
    assert topk.mte2_bytes > 0.0


# --------------------------------------------------------------------------- #
# [P1 #3] Shared-prefix replacement keeps query/output traffic.
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# [P1 #4] Private refills survive in MTE2 and split pipes are bound.
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# [P1 #5] The native PoolKeyIndexer declares its own score algebra.
# --------------------------------------------------------------------------- #
def test_weighted_query_scoring_is_vector_arithmetic_not_bf16_cube():
    case = parse_case(compose_case(
        "glm-5.3-flash", hardware="a5", tp=8, ep=8, num_devices=8,
        ranks_per_node=8, prompt_tokens=POOL_CONTEXT, output_tokens=2,
        concurrency=1))
    graphs = target_graphs(
        case, phase="decode", tokens=1, context=POOL_CONTEXT, sequences=1,
        scope="body")
    ops = next(iter(graphs.values())) if isinstance(graphs, dict) else graphs
    scores = next(op for op in ops if op.recipe == "cann_pool_key_indexer")
    metadata = dict(scores.metadata)
    scanned = metadata["safe_reference_scanned_pools"]
    plan = next(p for p in case.model.layer_plans if p.index_topk)
    # The native caller declares a per-query-head Cube dot product with the
    # same pool arithmetic; the TopK result stays inside the fused envelope.
    assert scores.flops == pytest.approx(
        2 * scanned * plan.index_heads * plan.index_dim)
    assert scores.vector_ops > 0.0
    assert not any(op.recipe in {"index_pool_scores", "index_pool_topk"}
                   for op in ops)


# --------------------------------------------------------------------------- #
# [P1 #6] Sampling is vector work; a control with payload is unmodeled.
# --------------------------------------------------------------------------- #
def test_sampling_and_acceptance_are_vector_families_and_control_rejects_work():
    sampling = OpSpec(
        "sample", "sampling", "decode", 0, 0, (), (128000,), (128000,),
        (128000,), (128000,), vector_ops=640000.0, hbm_bytes=1024000.0,
        mte2_bytes=512000.0, mte3_bytes=512000.0, recipe="sample_logits")
    recipe = resolve_physical_recipe(sampling)
    assert type(recipe).__name__ == "AIVTiledVectorAndGatherRecipe"
    assert recipe.submode(sampling) == "streaming"
    guard = SimpleNamespace(
        op=OpSpec("c", "barrier", "decode", 0, 0, (), (), (), (), ()),
        native_work={"mte2": 1.0}, work={}, minimum_ms=0.0)
    with pytest.raises(Exception, match="unmodeled_operator_mechanism"):
        ControlSyncRecipe().lower(guard)


# --------------------------------------------------------------------------- #
# [P1 #7] Each physical resource conserves its own work across phases.
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# [P1 #8] Split-K executes each FLOP once and charges GM partials.
# --------------------------------------------------------------------------- #
def _splitk_context(m=32, n=32, k=7168):
    raw = materialize({"backend": "ascend", "sku": A5_SKU,
                       "num_devices": 1, "ranks_per_node": 1})
    die = Hardware.from_mapping(raw).dies[0]
    op = gemm_op(op_id="probe", deps=(), stage=0, layer=0, requests=(),
                 m=m, n=n, k=k, weight_count=1, quant=QuantSpec("bf16"),
                 category="dense_mlp", phase="decode")
    roles = {role for role in dict(die.rates) if role != "link"}
    names = {role: role for role in roles}
    names["cube"] = "cube"
    rates = {role: float(dict(die.rates)[role]) / 1000 for role in roles}
    rates["cube"] = 1.0
    units = {role: die.units(role) for role in roles}
    units["cube"] = die.aic_count
    work = {"cube": 1000 * op.flops / die.peak("bf16"),
            "hbm": float(op.hbm_bytes), "mte2_aic": float(op.mte2_bytes),
            "mte3": float(op.mte3_bytes)}
    pools = {pool: int(size) for pool, size in dict(die.buffers).items()}
    context = RecipeContext(
        op, work, names, rates, die.aic_count,
        pool_names={pool: pool for pool in pools}, pool_capacities=pools,
        pool_units={pool: die.pool_units(pool) for pool in pools},
        resource_units=units, compute_peaks=dict(die.compute_peaks),
        service_modes=dict(die.service_modes),
        work_units=dict(die.work_units))
    return die, op, context


def _walk(program):
    if hasattr(program, "stages"):
        for stage in program.stages:
            if stage.program is not None:
                yield from _walk(stage.program)
            else:
                yield stage
    elif hasattr(program, "programs"):
        for child in program.programs:
            yield from _walk(child)


def test_splitk_issues_each_flop_once_and_charges_partial_traffic():
    die, op, context = _splitk_context()
    record = {}
    selected, _ = SteadyMatmul().select(context)
    geometry = matrix_geometry(context, selected)
    program = splitk_program(context, 8, recipe=selected, geometry=geometry,
                             record=record)
    issued = {}
    for stage in _walk(program):
        for key, value in stage.work.items():
            issued[key] = issued.get(key, 0.0) + value
    issued_flops = issued["cube"] * die.peak("bf16") / 1000.0
    assert issued_flops == pytest.approx(op.flops, rel=1e-12)
    assert record["split_k_traffic"]["partial_write_bytes"] == pytest.approx(
        4 * 8 * 32 * 32)
    assert record["split_k_traffic"]["partial_read_bytes"] == pytest.approx(
        4 * 8 * 32 * 32)
    assert record["split_k_traffic"]["merge_vector_ops"] == pytest.approx(
        (8 - 1) * 32 * 32)
    assert record["split_k_traffic"]["merge_latency_ms"] == 0.0
    assert record["split_k_tasks"] == 4 * 8


# --------------------------------------------------------------------------- #
# [P1 #9] Direct-mesh AllReduce follows (W-1) peer scaling.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("world_size,anchor_ms", ((4, 1.440), (8, 0.813)))
def test_direct_mesh_allreduce_matches_silicon_anchor(world_size, anchor_ms):
    case = parse_case(compose_case(
        "qwen3.5-35b-a3b", hardware=A5_SKU, tp=world_size, dp=1, ep=1, pp=1,
        layers=1, prompt_tokens=8, output_tokens=1, concurrency=1))
    row = operator_costs(case, (_allreduce_op(world_size),))["operators"][0]
    assert row["program"] == "TopologyCollectiveRecipe"
    assert abs(row["duration_ms"] - anchor_ms) / anchor_ms < 0.10
    assert row["duration_ms"] < anchor_ms * 1.10


# --------------------------------------------------------------------------- #
# [P1 #10] Public Link_Util stays within [0, 1] without clipping.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("world_size", (2, 4))
def test_public_link_utilization_uses_matching_physical_capacity(world_size):
    case = parse_case(compose_case(
        "qwen3.5-35b-a3b", hardware=A5_SKU, tp=world_size, dp=1, ep=1, pp=1,
        layers=1, prompt_tokens=8, output_tokens=1, concurrency=1))
    report = analyze_network_efficiency(
        case, phase="prefill", tokens=8, context=0, sequences=1, scope="body")
    utilization = float(report["network_link_util"])
    assert math.isfinite(utilization)
    assert 0.0 <= utilization <= 1.0 + 1e-9
    for op in report["operators"]:
        if op["link_bytes"] > 0:
            assert 0.0 <= op["link_util"] <= 1.0 + 1e-9


# --------------------------------------------------------------------------- #
# [P1 #11] Occurrence critical path keeps serial chains and multiplicity.
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# [P1 #12] Four-clock attribution keeps serial cube and memory time.
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# [P1 #13] Locked sweeps retain the complete baseline plan.
# --------------------------------------------------------------------------- #
def test_locked_sweep_reuses_one_plan_or_rejects_unsupported_comparison():
    case = parse_case(compose_case(
        "qwen3.5-35b-a3b", hardware=A5_SKU, tp=8, dp=1, ep=8, pp=1, layers=1,
        prompt_tokens=64, output_tokens=2, concurrency=1))
    with pytest.raises(Exception, match="fixed_plan_sweep_not_implemented"):
        sweep_topologies(case, num_devices=8, lock_program=True)
    report = sweep_topologies(
        case, num_devices=8, lock_parallel=True, lock_program=True)
    assert report["plan_selection"] == "fixed_parallel"
    assert report["program_selection"] == "fixed"
    rows = report["candidates"]
    assert rows and all(row.get("plan_identity") for row in rows)
    assert len({row["plan_identity"] for row in rows}) == 1
    assert len({row["b_retained"] for row in rows}) == 1


# --------------------------------------------------------------------------- #
# [P1 #14] Framework layer never performs whole-graph DBO microbatch splitting.
# --------------------------------------------------------------------------- #
def test_enable_dbo_builds_two_distinct_microbatch_partitions():
    from mindie_sim.public.errors import UnmodeledCapabilityError

    case = parse_case(compose_case(
        "qwen3-30b-a3b", hardware="a5", tp=8, ep=8, num_devices=8,
        ranks_per_node=8, prompt_tokens=64, output_tokens=2, concurrency=16))
    case = replace(case, runtime=replace(case.runtime, enable_dbo=True))
    views = tuple(RequestView(f"r{idx}", 1, 32, True, "decode") for idx in range(16))
    with pytest.raises(UnmodeledCapabilityError, match="unmodeled_in_op_dbo_pending_stage3"):
        Frontend()(IterationPlan(views), case.model, case.runtime, case.machine)


# --------------------------------------------------------------------------- #
# [P0 #15] aclnn single-call uses the one canonical native path.
# --------------------------------------------------------------------------- #
def test_aclnn_single_call_lowers_through_one_canonical_path():
    from mindie_sim import simulate_aclnn_call
    environment = ExecutionEnvironment.from_hardware(
        Hardware.from_mapping(materialize({"sku": A5_SKU})))
    declaration = {"builder": "aclnn",
                   "arguments": {"api": "aclnnMatmul", "m": 2048, "n": 4096,
                                 "k": 4096, "dtype": "bf16",
                                 "phase": "prefill"}}
    report = simulate_aclnn_call(environment, declaration)
    assert report["schema"] == "mindie-sim.call-costs/1"
    assert report["replay_scope"] == "none"
    row = report["timeline"][0]
    assert row["program"] == "SteadyMatmul"
    assert "piecewise" not in str(report).lower()
    second = simulate_aclnn_call(environment, declaration)
    assert second["duration_ms"] == pytest.approx(report["duration_ms"])
    assert build_call(declaration) is not None


# --------------------------------------------------------------------------- #
# [P0] No cross-call execution result memoization.
# --------------------------------------------------------------------------- #
def test_efficiency_evaluation_is_not_served_from_a_cross_call_result_cache():
    import mindie_sim.prediction.efficiency as efficiency
    assert not hasattr(efficiency, "_STRUCTURAL_MEMO")
    assert not hasattr(efficiency, "_EFFICIENCY_MEMO")
    case = parse_case(compose_case(
        "qwen3.5-35b-a3b", hardware=A5_SKU, tp=8, dp=1, ep=8, pp=1, layers=1,
        prompt_tokens=32, output_tokens=2, concurrency=1))
    calls = []
    original = efficiency._lower_and_run_graph

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    efficiency._lower_and_run_graph = counting
    try:
        for _ in range(2):
            efficiency.analyze_network_efficiency(
                case, phase="decode", tokens=1, context=64, sequences=1,
                scope="body")
    finally:
        efficiency._lower_and_run_graph = original
    assert len(calls) == 2
