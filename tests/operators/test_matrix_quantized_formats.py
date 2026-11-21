"""Packed INT4 weights and INT8 activation matrix schedules.

[Role] Verify gate/up fusion, down projection, and MSD owner placement.
[Invariants] Independent row, byte, FLOP and timing controls conserve work.
[Timeouts & Locks] Bounded local graph and Engine execution without locks.
[Side-Effect Policy] Read committed examples and modify only local test inputs.
"""
from dataclasses import replace
import json
import math
from pathlib import Path

import pytest

from mindie_sim import graph_costs, predict
from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.operators.hyper_connections.native_fusion import _groups, lower_native_fusions
from mindie_sim.prediction.operator_analysis import target_graphs, target_operators
from mindie_sim.prediction.program import RecipeContext
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.transforms import lower_implementations
from mindie_sim.prediction.operators.mx_quant_matrix.w4a8_down import RECIPE as DOWN, lower_op
from mindie_sim.prediction.operators.mx_quant_matrix.w4a8_matrix import matrix_program, owner_rectangles
from mindie_sim.prediction.operators.mx_quant_matrix.w4a8_schedule import (
    down_schedule, gate_schedule, source_row_partition, strided_rows, total_work,
)
from mindie_sim.schema.common import ModelingError

EXAMPLES = Path(__file__).resolve().parents[2] / "examples/source-families"
GATE = "ascend_w4a8_msd"
ROLE = "grouped_gate_up_activation_quant"
ROWS = (3, 0, 3, 2)
SPARSE_ROWS = (1,) * 8 + (0,) * 248
pytestmark = pytest.mark.tier0


def case_input(down=False, *, down_binding=True, gate_profile=True, routing=None):
    path = EXAMPLES / ("w4a8-down-msd-control.json" if down else "w4a8-msd-control.json")
    raw = json.loads(path.read_text())
    lowerings = raw["runtime"]["kernels"].get("lowerings", {})
    if not down_binding:
        lowerings.pop("grouped_gemm")
    if not gate_profile:
        lowerings.pop("native_fusion")
    if routing is not None:
        raw["runtime"]["routing"] = routing
    return raw


def ops(case, *, fusion=True, tokens=4):
    return target_operators(case, phase="prefill", tokens=tokens, resource_fusion=fusion)


def pick(operators, recipe):
    return next(op for op in operators if op.recipe == recipe)


def run(case, op, ready=0, **options):
    return graph_costs(case, [op], external_inputs=dict.fromkeys(op.deps, ready), **options)


def phase_work(rows, n, k, *, down, aivs):
    """Account for each GM read/write and the per-owner assist reloads."""
    m, active = sum(rows), sum(count > 0 for count in rows)
    labels = [bank for bank, count in enumerate(rows) for _ in range(count)]
    quotient, remainder = divmod(m, min(m, aivs))
    boundaries = [i * quotient + min(i, remainder) for i in range(min(m, aivs) + 1)]
    assist = active if down else sum(len(set(labels[i:j])) for i, j in zip(boundaries, boundaries[1:]))
    packed, encoded = n * k * active / 2, 8 * n * active
    metadata = 4 * len(rows) if down else 0
    pre_read = m * k + 8 * metadata
    matrix_read, matrix_write = m * k + packed + encoded, 4 * m * n
    post_read = matrix_write + 4 * n * assist + 4 * m
    out_write = 2 * m * n if down else m * (n // 2) + 4 * m
    reads, writes = pre_read + matrix_read + post_read, m * k + out_write
    vector = 6.5 * m * k + 8 * m * n + (m * n + 2 * metadata if down else 10 * m * (n // 2) + m)
    return dict(flops=4 * m * n * k, mte2_bytes=reads, mte3_bytes=writes,
                hbm_bytes=reads + writes + matrix_write, vector_ops=vector,
                weight_bytes=packed, encoded_scale_bytes=encoded)


def assert_drained(report):
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0


@pytest.mark.parametrize("down,recipe,k", [(False, GATE, 32), (True, DOWN, 64)])
def test_phase_work_and_release_match_independent_arithmetic(down, recipe, k):
    case = parse_case(case_input(down, gate_profile=not down))
    op = pick(ops(case), recipe)
    expected = phase_work(ROWS, 64, k, down=down, aivs=8)
    report, late = (run(case, op, t, include_submission=True, isolated=True) for t in (0, 5))
    execution = report["execution"]
    applied = report["isolated"]["operators"][0]["operator"]
    for field in ("mte2_bytes", "mte3_bytes", "hbm_bytes", "vector_ops"):
        assert applied[field] == pytest.approx(expected[field])
    for field in ("flops", "vector_ops", "hbm_bytes"):
        assert execution["work"][field] == pytest.approx(expected[field])
    assert applied["matrix_flops"] == {"int4": expected["flops"]}
    assert execution["recipes"] == {recipe: 1}
    assert execution["serviced"]["matrix@r0:s0"] == pytest.approx(1000 * expected["flops"] / 500_000_000)
    assert_drained(report)
    assert report["timeline"][0]["ready_ms"] == 0
    assert late["timeline"][0]["ready_ms"] == 5
    assert late["timeline"][0]["complete_ms"] > 5
    assert late["timeline"][0]["complete_ms"] - report["timeline"][0]["complete_ms"] == pytest.approx(5)
    if down:
        ledger = dict(op.metadata)["w4a8_msd_down_ledger"]["matrix"]
        for field in ("weight_bytes", "encoded_scale_bytes"):
            assert ledger[field] == pytest.approx(expected[field])

    missing = case_input(down, gate_profile=not down)
    del missing["machine"]["device"]["compute_peaks"]["int4"]
    with pytest.raises(ModelingError, match="compute_precision_rate_required:int4"):
        run(parse_case(missing), op, include_submission=True)


def test_down_binding_preserves_semantic_identity_and_input_quantizer():
    operators = ops(case_input(True), fusion=False)
    down = pick(operators, DOWN)
    quant = next(op for op in operators if op.op_id == down.deps[0])
    semantic = pick(ops(case_input(True, down_binding=False), fusion=False), "grouped_down")
    assert down.source_boundary == "kernel" and down.dtype == "int4"
    assert dict(down.matrix_flops) == {"int4": 4 * 8 * 64 * 64}
    metadata = dict(down.metadata)
    assert metadata["output_format"] == "bf16" and not metadata["w4a8_msd_down"]["output_quantization"]
    assert quant.op_id == down.op_id + ".quantize"
    assert (quant.family, quant.recipe, quant.deps) == ("quantize", "dynamic_quantize", ("s0.l0.expert_act",))
    assert not any(op.family == "quantize" and down.op_id in op.deps for op in operators)
    for field in ("op_id", "family", "phase", "stage", "layer", "requests", "semantic_shape", "stream"):
        assert getattr(down, field) == getattr(semantic, field)
    assert all(metadata[key] == value for key, value in semantic.metadata if key != "compute_format")


@pytest.mark.parametrize("recipe,down,k", [(GATE, False, 32), (DOWN, True, 64)])
def test_empty_bank_has_no_matrix_reads_and_balancing_conserves_rows(recipe, down, k):
    raw = case_input(down, gate_profile=not down)
    uneven = pick(ops(raw), recipe)
    balanced_raw = case_input(down, gate_profile=not down, routing={"policy": "balanced"})
    balanced = pick(ops(balanced_raw), recipe)
    key = "w4a8_msd_down" if down else "w4a8_msd"
    config = dict(balanced.metadata)[key]
    assert config["expert_rows"] == [2, 2, 2, 2] and config["active_groups"] == 4
    assert dict(uneven.metadata)[key]["expert_rows"] == list(ROWS)
    assert balanced.flops == uneven.flops == 4 * 8 * 64 * k
    extra_bank = 0.5 * 64 * k + 8 * 64
    if down:
        assert config["routing_basis"] == "forced_balanced_integer_counts"
        a, b = (run(case, op, isolated=True)["isolated"]["operators"][0]["operator"]
                for case, op in ((raw, uneven), (balanced_raw, balanced)))
        for field in ("hbm_bytes", "mte2_bytes"):
            assert b[field] - a[field] == pytest.approx(extra_bank + 4 * 64)
        assert b["vector_ops"] == pytest.approx(a["vector_ops"])
    else:
        assert balanced.mte2_bytes - uneven.mte2_bytes == pytest.approx(extra_bank)
        raw["runtime"]["routing"] = {"realizations": [{
            "scope": "body", "layer": 0, "phase": "prefill", "tokens": 1, "expert_rows": [1, 1, 0, 0]}]}
        small = pick(ops(raw, tokens=1), GATE)
        assert dict(small.metadata)[key]["m"] == 2 and small.flops == 4 * 2 * 64 * 32
        assert small.mte2_bytes == phase_work((1, 1, 0, 0), 64, 32, down=False, aivs=2)["mte2_bytes"]
        assert small.mte2_bytes < phase_work((1, 1, 1, 1), 64, 32, down=False, aivs=2)["mte2_bytes"]


@pytest.mark.parametrize("down,recipe,write", [(False, GATE, 8 * 32 * 2 + 4 * 8), (True, DOWN, 8 * 64 * 3)])
def test_fixpipe_publication_preserves_gm_traffic(down, recipe, write):
    raw = case_input(down, gate_profile=not down)
    before = run(raw, pick(ops(raw), recipe), isolated=True)
    raw["machine"]["device"]["rates"]["fixpipe"] = 1e9
    op = pick(ops(raw), recipe)
    after = run(raw, op, isolated=True)
    assert after["execution"]["serviced"]["fixpipe@r0:s0"] == pytest.approx(4 * 8 * 64)
    assert after["isolated"]["operators"][0]["operator"]["mte3_bytes"] == write
    if down:
        assert "fixpipe@r0:s0" not in before["execution"]["serviced"]
        assert after["execution"]["work"]["hbm_bytes"] == before["execution"]["work"]["hbm_bytes"]


@pytest.mark.parametrize("fault,code", [
    ("wrong_projection", "unsupported_recipe"), ("column_tail", "n_multiple64"),
    ("inner_tail", "divisible_by_64"), ("int8_output", "requires_bf16_output"),
])
def test_down_rejects_incompatible_projection(fault, code):
    semantic = ops(case_input(True, down_binding=False), fusion=False)
    op = pick(semantic, "grouped_gate_up" if fault == "wrong_projection" else "grouped_down")
    if fault in ("column_tail", "inner_tail"):
        m, n, k = (8, 32, 64) if fault == "column_tail" else (8, 64, 96)
        op = replace(op, local_shape=(m, n, k), semantic_shape=(m, n, k), flops=2 * m * n * k)
    if fault == "int8_output":
        op = replace(op, metadata=tuple({**dict(op.metadata), "output_format": "int8"}.items()))
    with pytest.raises(ModelingError, match=code):
        lower_op(op)


@pytest.mark.parametrize("down,code", [
    (False, "w4a8_msd_requires_explicit_integer_expert_rows"),
    (True, "rejects_probability_moment_routing"),
])
def test_source_program_requires_integer_expert_rows(down, code):
    raw = case_input(down)
    raw["runtime"].pop("routing")
    with pytest.raises(ModelingError, match=code):
        ops(raw, fusion=not down)


def test_gate_fusion_replaces_member_work_and_preserves_down_dependency():
    case = parse_case(case_input())
    declaration, members = next((d, m) for d, m in _groups(ops(case, fusion=False)) if d["role"] == ROLE)
    operators = ops(case)
    fused, = [op for op in operators if op.family == "native_fusion"]
    metadata = dict(fused.metadata)
    assert dict(members[0].metadata)["routing_realization"]["expert_rows"] == ROWS
    assert members[0].local_shape == (8, 64, 32)
    assert fused.recipe == GATE and fused.requests == members[-1].requests
    assert metadata["fusion_semantic_ids"] == [op.op_id for op in members]
    assert metadata["fusion_group_key"] == declaration["key"]
    assert metadata["w4a8_msd"]["expert_rows"] == list(ROWS) and metadata["w4a8_msd"]["m"] == 8
    assert metadata["fusion_boundary"]["traffic"]["eliminated_reads"] == 0
    expected = phase_work(ROWS, 64, 32, down=False, aivs=2)
    for field in ("flops", "vector_ops", "hbm_bytes", "mte2_bytes", "mte3_bytes"):
        assert getattr(fused, field) == pytest.approx(expected[field])
    assert fused.vector_ops != pytest.approx(sum(op.vector_ops for op in members) + expected["vector_ops"])
    assert pick(operators, "grouped_down").deps == (fused.op_id,)


@pytest.mark.parametrize("fault,code", [
    ("external_gate", "intermediate_external_outputs"), ("block_scale", "one_fp32_output_scale"),
])
def test_gate_fusion_rejects_incompatible_materialization(fault, code):
    case = parse_case(case_input())
    semantic = ops(case, fusion=False)
    _, members = next((d, m) for d, m in _groups(semantic) if d["role"] == ROLE)
    if fault == "external_gate":
        semantic = tuple(replace(op, deps=(*op.deps, members[0].op_id))
                         if op.recipe == "grouped_down" else op for op in semantic)
    else:
        quant = members[-1]
        metadata = dict(quant.metadata, activation_group_size=16, activation_scale_shape=(8, 2),
                        activation_scale_bytes=64)
        changed = replace(quant, metadata=tuple(metadata.items()), mte3_bytes=quant.mte3_bytes + 32,
                          hbm_bytes=quant.hbm_bytes + 32)
        semantic = tuple(changed if op.op_id == quant.op_id else op for op in semantic)
    with pytest.raises(ModelingError, match=code):
        lower_native_fusions(semantic, case.runtime)


@pytest.mark.parametrize("down", (False, True))
def test_public_prediction_executes_the_selected_programs(down):
    raw = case_input(down)
    gate = pick(ops(raw), GATE)
    assert gate.family == "native_fusion" and dict(gate.metadata)["fusion_role"] == ROLE
    report = predict(raw)
    assert report["status"] == "modeled"
    assert report["metrics"]["ttft_ms"]["mean"] > 0 and report["metrics"]["tpot_ms"]["mean"] > 0
    assert report["execution"]["recipes"][GATE] == 3
    if down:
        assert report["execution"]["recipes"][DOWN] == 3
    assert_drained(report)


def test_binding_waits_for_normal_non_bf16_producer_and_requires_activation_lowering():
    raw = case_input(True, gate_profile=False)
    raw["model"]["activation_dtype"] = "fp32"
    operators = ops(raw, fusion=False)
    down = pick(operators, DOWN)
    quant = next(op for op in operators if op.op_id == down.deps[0])
    assert quant.dtype == "fp32"
    assert quant.mte2_bytes == 8 * 64 * 4
    assert quant.mte3_bytes == 8 * 64 + 8 * 4
    assert len([op for op in operators if op.op_id == quant.op_id]) == 1
    raw["runtime"]["lowering_passes"] = []
    with pytest.raises(ModelingError, match="implementation_requires_activation_quantization"):
        ops(raw, fusion=False)


def test_cross_dp_balanced_source_consumes_each_local_expert_histogram():
    raw = case_input(True, gate_profile=False, routing={"policy": "balanced"})
    raw["machine"]["num_devices"] = 2
    raw["machine"]["links"][0]["members"] = [0, 1]
    raw["runtime"].update(dp=2, ep=2)
    case = parse_case(raw)
    graphs = target_graphs(case, phase="prefill", tokens=4, scope="body")
    assert set(graphs) == {0, 1}
    for operators in graphs.values():
        down = pick(operators, DOWN)
        config = dict(down.metadata)["w4a8_msd_down"]
        assert config["expert_rows"] == [4, 4]
        quant = next(op for op in operators if op.op_id == down.deps[0])
        assert quant.local_shape == (8, 64)
        assert down.flops == 4 * 8 * 64 * 64
    report = graph_costs(case, graphs, external_inputs={"analysis_input": 0})
    assert_drained(report)
    assert report["execution"]["recipes"][DOWN] == 2


def test_draft_scope_binding_keeps_request_identity_and_actual_source_dtype():
    raw = case_input(True, down_binding=False, gate_profile=False)
    semantic = pick(ops(raw, fusion=False), "grouped_down")
    metadata = dict(semantic.metadata)
    metadata.pop("activation_producer")
    draft = replace(
        semantic, layer=-1, phase="draft", requests=("draft:r0",),
        deps=("draft:activation",), metadata=tuple(metadata.items()))
    runtime = parse_case(case_input(True, gate_profile=False)).runtime
    operators = lower_implementations((draft,), runtime, "fp16")
    down, quant = pick(operators, DOWN), next(op for op in operators if op.family == "quantize")
    assert down.layer == quant.layer == -1
    assert down.phase == quant.phase == "draft"
    assert down.requests == quant.requests == ("draft:r0",)
    assert quant.dtype == "fp16" and quant.deps == ("draft:activation",)
    assert down.deps == (quant.op_id,)


def test_source_phases_use_existing_split_read_pipes_without_capacity_aliases():
    raw = case_input(True, gate_profile=False)
    raw["machine"]["device"]["rates"].update(mte2_aic=1e9, mte2_aiv=2e9)
    case = parse_case(raw)
    down = pick(ops(raw, fusion=False), DOWN)
    serviced = run(case, down)["execution"]["serviced"]
    m, n, k, active = 8, 64, 64, 3
    assert serviced["mte2_aic@r0:s0"] == pytest.approx(
        m * k + 0.5 * n * k * active + 8 * n * active)
    assert serviced["mte2_aiv@r0:s0"] == pytest.approx(
        m * k + 4 * m * n + 4 * n * active + 4 * m + 4 * 4 * 8)
    assert "mte2@r0:s0" not in serviced
    split_service = dict(serviced)
    raw["machine"]["device"]["rates"]["mte2"] = 3e9
    serviced = run(raw, pick(ops(raw, fusion=False), DOWN))["execution"]["serviced"]
    assert serviced.get("mte2@r0:s0", 0) == 0
    assert {key: value for key, value in serviced.items() if value} == pytest.approx(
        {key: value for key, value in split_service.items() if value})
    read_bytes = phase_work(ROWS, n, k, down=True, aivs=8)["mte2_bytes"]
    assert serviced["mte2_aic@r0:s0"] + serviced["mte2_aiv@r0:s0"] == read_bytes
    for role in ("mte2_aic", "mte2_aiv"):
        del raw["machine"]["device"]["rates"][role]
    generic = run(raw, pick(ops(raw, fusion=False), DOWN))
    assert generic["execution"]["serviced"]["mte2@r0:s0"] == read_bytes

def _schedule(rows=(1, 0, 3), n=320, k=640, cores=1, down=False):
    factory = down_schedule if down else gate_schedule
    return factory(sum(rows), n, k, rows, aic_owners=cores, aiv_owners=2 * cores,
                   fixpipe=True)

def _run(schedule, *, compact=True, modes=False, operand_rate=128):
    cores = schedule.aic_owners
    op = gemm_op(op_id="matrix", deps=(), stage=0, layer=0, requests=(),
                 m=schedule.m, n=schedule.n, k=schedule.k, quant=QuantSpec("bf16"),
                 category="dense_mlp", phase="decode")
    rates = dict(cube=cores, hbm=1e12, mte2_aic=512 * cores, mte2_aiv=512 * cores,
                 mte1=operand_rate * cores, mte3=1e9, fixpipe=1e9, vector=1e9)
    units = {key: 1 if key == "hbm" else (2 * cores if key in
             {"vector", "mte2_aiv", "mte3"} else cores) for key in rates}
    pools = {"l1": 512 * 1024, "l0a": 64 * 1024,
             "l0b": 64 * 1024, "l0c": 128 * 1024}
    modes = ({"mte1_l0a": ("mte1", 2000 * rates["mte1"]),
              "mte1_l0b": ("mte1", 1000 * rates["mte1"])} if modes else {})
    context = RecipeContext(
        op, {}, {key: key for key in rates}, rates, cores,
        pool_names={key: key for key in pools}, pool_capacities=pools,
        pool_units={key: cores for key in pools}, resource_units=units,
        compute_peaks={"int4": 1e9 * cores}, service_modes=modes)
    engine = Engine([Resource(key, value, units=units[key]) for key, value in rates.items()],
                    pools=pools, pool_units={key: cores for key in pools}, trace_limit=300)
    done = matrix_program(context, schedule, compact=compact).start(
        engine, "matrix", engine.submit("ready", release=7))
    engine.advance(horizon=7)
    assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    assert not any(engine.allocated.values())
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0
    assert all(value <= pools[pool] for (pool, _), value in engine.peak_unit_allocated.items())
    return done.time - 7, engine

@pytest.mark.parametrize("rows,n,cores,down", [
    ((1, 0, 3), 320, 1, False), ((1,) * 8 + (0,) * 248, 512, 20, False),
    ((3, 0, 129, 2), 320, 4, False), ((1,) * 8 + (0,) * 248, 6144, 20, True),
])
def test_source_rectangles_preserve_expert_boundaries_and_rolling_owners(rows, n, cores, down):
    schedule = _schedule(rows, n, cores=cores, down=down)
    expected = [[] for _ in range(cores)]
    ordinal = 0
    for rows in rows:
        for m in range(0, rows * 2, schedule.base_m):
            for column in range(0, n, schedule.base_n):
                expected[ordinal % cores].append((min(schedule.base_m, rows * 2 - m),
                                                   min(schedule.base_n, n - column)))
                ordinal += 1
    assert [list(owner_rectangles(schedule, owner)) for owner in range(cores)] == expected

@pytest.mark.parametrize("down,modes,k", [(False, False, 128), (False, True, 640),
                                        (True, True, 512), (True, False, 768)])
def test_steady_and_detailed_match_physical_work_and_serial_owner_boundary(down, modes, k):
    schedule = _schedule(k=k, down=down)
    detailed_ms, detailed = _run(schedule, compact=False, modes=modes)
    compact_ms, compact = _run(schedule, modes=modes)
    assert compact.serviced == pytest.approx(detailed.serviced)
    assert compact_ms == pytest.approx(detailed_ms, abs=1e-7)
    matrix = total_work(schedule.matrix)
    assert compact.serviced["hbm"] == pytest.approx(matrix.resources["hbm"])
    assert compact.serviced["mte2_aic"] == pytest.approx(matrix.resources["mte2"])
    assert compact.serviced["vector"] == pytest.approx(matrix.resources["vector"])
    assert compact.serviced["fixpipe"] == pytest.approx(4 * schedule.m * schedule.n)
    assert compact.serviced["mte3"] == 0
    # Directional MTE1 services A at twice the rate of B.
    a = b = 0
    for owner in range(schedule.aic_owners):
        for rows, cols in owner_rectangles(schedule, owner):
            a += 0.5 * rows * k * math.ceil(cols / 128)
            b += 0.5 * cols * k * math.ceil(rows / min(schedule.base_m, 128))
    assert compact.serviced["mte1"] == pytest.approx((a / 2 if modes else a) + b)

def test_owner_tiles_remain_independent_bank_loads_and_k_compression_is_bounded():
    times = []
    for rate in (64, 128, 256):
        elapsed, engine = _run(_schedule(cores=4), operand_rate=rate)
        times.append(elapsed)
        assert engine.serviced["hbm"] > 0
    assert times[0] > times[1] > times[2]
    _, small = _run(_schedule(k=512, cores=4))
    _, large = _run(_schedule(k=256 * 1_000_000, cores=4))
    assert large.completed_count == small.completed_count
    assert large.peak_tasks == small.peak_tasks
    assert large.serviced["fixpipe"] == small.serviced["fixpipe"]
    assert large.serviced["vector"] == small.serviced["vector"]

def test_group_count_bulk_copy_includes_idle_aivs_without_256_active_weight_banks():
    schedule = _schedule((1,) * 8 + (0,) * 248, n=6144, k=256, cores=20, down=True)
    pre = total_work(schedule.pre)
    assert pre.resources["mte2"] == 8 * 256 + 40 * 256 * 8
    assert pre.resources["vector"] == 6.5 * 8 * 256 + 40 * 2 * 256
    _, engine = _run(schedule, modes=True)
    assert engine.serviced["mte2_aic"] == pytest.approx(
        8 * 0.5 * 6144 * 256 + 96 * 8 * 512 + 96 * 256)
    assert engine.serviced["mte1"] > 0

def test_mixed_fusion_keeps_independent_router_role():
    raw = case_input()
    raw["runtime"]["kernels"]["lowerings"]["native_fusion"]["options"]["roles"] = [ROLE, "router_selection"]
    _, members = next((d, m) for d, m in _groups(ops(raw, fusion=False)) if d["role"] == "router_selection")
    by_role = {dict(op.metadata).get("fusion_role"): op for op in ops(raw) if op.family == "native_fusion"}
    assert set(by_role) == {ROLE, "router_selection"}
    assert by_role[ROLE].recipe == GATE
    router = by_role["router_selection"]
    assert router.recipe == "declared_native_fusion"
    assert router.hbm_bytes == pytest.approx(sum(op.hbm_bytes for op in members))


@pytest.mark.parametrize("rows,n,k,down,m_base,n_base,tiles,read", [
    (SPARSE_ROWS, 512, 6144, False, 128, 256, 16, 8 * (256 * 6144 + 8 * 512 + 2 * 6144)),
    ((8,) + (0,) * 255, 512, 6144, False, 128, 256, 2, 256 * 6144 + 8 * 512 + 16 * 6144),
    ((3, 0, 129, 2), 320, 256, False, 128, 256, 10, None),
    (SPARSE_ROWS, 6144, 256, True, 64, 512, 96, 8 * (3072 * 256 + 8 * 6144 + 12 * 256)),
    ((33, 0, 97, 126), 640, 256, True, 128, 256, 15, None),
])
def test_packed_weights_and_cube_work_follow_real_expert_rectangles(rows, n, k, down, m_base, n_base, tiles, read):
    schedule = _schedule(rows, n, k, cores=20, down=down)
    assert (schedule.base_m, schedule.base_n) == (m_base, n_base)
    rectangles = [(min(2 * count - row, m_base), min(n - col, n_base))
                  for count in rows for row in range(0, 2 * count, m_base) for col in range(0, n, n_base)]
    assert schedule.tile_count == len(rectangles) == tiles
    assert schedule.expert_scan_count == len(rows) and schedule.active_experts == sum(r > 0 for r in rows)
    assert schedule.zero_bank_offsets == tuple(i * n * k for i, r in enumerate(rows) if not r)
    matrix = {record.owner: record for record in schedule.matrix}
    for owner in range(min(20, len(rectangles))):
        owned = rectangles[owner::20]
        assert matrix[owner].tiles == len(owned)
        assert matrix[owner].work.matrix["int4"] == sum(2 * m * width * k for m, width in owned)
        assert matrix[owner].work.resources["hbm"] >= sum(width * k / 2 for _, width in owned)
    assert total_work(schedule.matrix).matrix["int4"] == 4 * sum(rows) * n * k
    if read is not None:
        assert total_work(schedule.matrix).resources["mte2"] == pytest.approx(read)
    if (n, down) == (320, False):
        tail = schedule.experts[2]
        assert (tail.double_rows, tail.m_tiles, tail.last_m_tile, tail.last_n_tile) == (258, 3, 2, 64)
        labels = [bank for bank, count in enumerate(rows) for _ in range(count)]
        q, r = divmod(sum(rows), 40)
        expected = {i: len(set(labels[i*q+min(i,r):(i+1)*q+min(i+1,r)])) for i in range(40)}
        assert {record.owner: record.assist_loads for record in schedule.post_dequant} == expected
        assert sum(expected.values()) > schedule.active_experts
    if down and n == 640:
        assert schedule.experts[2].last_m_tile == 66 and schedule.experts[0].last_n_tile == 128


@pytest.mark.parametrize("cores,rows_per_owner", [(20, (1,) * 8), (2, (2,) * 4)])
def test_gate_aiv_partitions_conserve_rows(cores, rows_per_owner):
    schedule = _schedule(SPARSE_ROWS, 512, 6144, cores)
    for records in (schedule.pre, schedule.post_dequant):
        assert [(r.owner, r.rows) for r in records] == list(enumerate(rows_per_owner))
    assert not any(r.assist_loads for r in schedule.pre)
    if cores == 20:
        assert source_row_partition(8, 40) == tuple((i, i, 1) for i in range(8))
        assert strided_rows(8, 40) == tuple(enumerate(rows_per_owner))


def test_down_aiv_assist_reloads_and_idle_group_list_reads():
    schedule = _schedule(SPARSE_ROWS, 6144, 256, cores=20, down=True)
    assert schedule.aivs_per_aic == 2
    tiles = {r.owner: r.tiles for r in schedule.matrix}
    assists = {r.owner: r.assist_loads for r in schedule.post_dequant}
    active = {r.owner: r.tiles for r in schedule.post_dequant}
    for owner, count in tiles.items():
        a, b = 2 * owner, 2 * owner + 1
        assert assists[a] == assists[b] == count
        assert active[a] > 0 and active[b] > 0 and active[a] + active[b] == count
    assert sum(assists.values()) == 2 * schedule.tile_count and sum(active.values()) == schedule.tile_count
    assert total_work(schedule.post_dequant).resources["vector"] == 3 * 16 * 6144
    assert len(schedule.pre) == 40 and sum(r.rows > 0 for r in schedule.pre) == 8
    for record in schedule.pre[8:]:
        assert record.rows == 0 and record.work.resources == {"hbm": 256 * 8, "mte2": 256 * 8, "mte3": 0, "vector": 512}
    pre = total_work(schedule.pre).resources
    assert pre["hbm"] == 2 * 8 * 256 + 40 * 256 * 8 and pre["vector"] == 6.5 * 8 * 256 + 40 * 512
    single = down_schedule(8, 64, 64, ROWS, aic_owners=4, aiv_owners=8)
    assert single.aivs_per_aic == 1
    assert total_work(single.post_dequant).resources["hbm"] == 4 * 8 * 64 + 4 * 64 * 3 + 4 * 8


def test_large_declared_gate_geometry_conserves_work_under_contention():
    raw = case_input()
    raw["machine"]["device"].update(compute_units=20, resource_units={
        "cube": 20, "vector": 40, "mte2_aic": 20, "mte2_aiv": 40, "mte3": 40})
    case = parse_case(raw)
    op = pick(ops(case), GATE)
    metadata = dict(op.metadata)
    metadata["w4a8_msd"] = dict(metadata["w4a8_msd"], m=8, n=512, k=6144, expert_rows=list(SPARSE_ROWS),
        active_groups=8, expert_scan_count=256, zero_expert_indices=list(range(8, 256)),
        zero_bank_offsets_int4_elements=[i * 512 * 6144 for i in range(8, 256)])
    op = replace(op, local_shape=(8, 512, 6144), metadata=tuple(metadata.items()))
    schedule = _schedule(SPARSE_ROWS, 512, 6144, cores=20)
    expected = total_work(schedule.pre, schedule.matrix, schedule.post_dequant, schedule.swiglu, schedule.quantize)
    report, late = (run(case, op, t, include_submission=True, isolated=True) for t in (0, 5))
    applied = report["isolated"]["operators"][0]["operator"]
    assert report["execution"]["recipes"] == {GATE: 1}
    assert applied["flops"] == expected.matrix["int4"] == 4 * 8 * 512 * 6144
    for field, role in (("hbm_bytes", "hbm"), ("mte3_bytes", "mte3"), ("vector_ops", "vector")):
        assert applied[field] == pytest.approx(expected.resources[role])
    assert late["timeline"][0]["ready_ms"] == 5 and report["timeline"][0]["ready_ms"] == 0
    assert late["timeline"][0]["complete_ms"] - report["timeline"][0]["complete_ms"] == pytest.approx(5)
    twin = replace(op, op_id=op.op_id + ".twin")
    shared = graph_costs(case, [op, twin], external_inputs=dict.fromkeys(op.deps, 0),
                         include_submission=True, isolated=True)
    assert shared["execution"]["work"]["flops"] == 2 * expected.matrix["int4"]
    assert shared["execution"]["work"]["hbm_bytes"] == 2 * expected.resources["hbm"]
    for calls, current in ((1, report), (2, shared)):
        assert current["execution"]["serviced"]["matrix@r0:s0"] == pytest.approx(
            calls * 1000 * expected.matrix["int4"] / 500_000_000)
        assert_drained(current)

def _source_down_output_oracle(rows, n, k, cores):
    """Execute the source's per-core loops, including persistent vecCount."""
    avg_m = sum(rows) // len(rows)
    base_m, base_n = (64, 512) if avg_m < 32 else (128, 256)
    ratio = 1 if avg_m <= 128 and len(rows) <= 4 and k <= 8192 and n <= 8192 else 2
    output_elements = {}
    for core in range(cores):
        pre_count, vec_count = 0, 0
        for count in rows:
            if not count:
                continue
            block_n = (n + base_n - 1) // base_n
            block_m = (2 * count + base_m - 1) // base_m
            cur_count = pre_count + block_n * block_m
            cur_block = core if core >= pre_count else core + cores
            while cur_block < cur_count:
                m_idx, n_idx = divmod(cur_block - pre_count, block_n)
                cube_m = min(base_m // 2, count - m_idx * (base_m // 2))
                cube_n = min(base_n, n - n_idx * base_n)
                step = min(8192 // base_n, cube_m)
                for offset in range(0, cube_m, step):
                    vec_count += 1
                    sub = vec_count % ratio
                    physical = 2 * core + sub
                    elements = min(step, cube_m - offset) * cube_n
                    output_elements[physical] = output_elements.get(physical, 0) + elements
                cur_block += cores
            pre_count = cur_count % cores
    return output_elements

@pytest.mark.parametrize("rows,n,k,cores", [
    ((33,) + (0,) * 255, 512, 256, 1),
    ((17, 0, 65, 3) + (0,) * 5, 576, 256, 3),
    ((129, 0, 2, 3), 640, 256, 2),
    ((1,) * 8 + (0,) * 248, 6144, 256, 20),
])
def test_down_ub_chunk_owner_matches_independent_source_loop(rows, n, k, cores):
    schedule = down_schedule(sum(rows), n, k, rows,
                             aic_owners=cores, aiv_owners=2 * cores)
    expected = _source_down_output_oracle(rows, n, k, cores)
    assert {r.owner: r.work.resources["mte3"] / 2 for r in schedule.output} == expected
    assert sum(expected.values()) == sum(rows) * n
    if sum(rows) == 33:
        # The two full 16-row chunks and the tail alternate 1, 0, 1.
        assert expected == {0: 16 * n, 1: 17 * n}

def test_public_source_selection_rejects_incompatible_vector_inventory():
    raw = case_input()
    raw["machine"]["device"]["resource_units"]["vector"] = 4
    case = parse_case(raw)
    op = pick(ops(case), GATE)
    with pytest.raises(ValueError, match="two_physical_aivs_per_aic"):
        graph_costs(case, [op], external_inputs={dep: 0 for dep in op.deps})
