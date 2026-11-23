"""Governed ``rotary`` operator family registration, work conservation, and replay.

[Role] Verify ``mindie_sim.prediction.operators.rotary`` executes live through
  ``FamilyRegistry``, ``RecipeBook``, and ``PredictionSpine`` across Ascend
  architectures (``a2``, ``a3``, ``a5``).
[Invariants] Standalone shape-only rotary ops derive conserved rotate-half
  vector/MTE2/MTE3/HBM work; pre-declared traffic ops preserve declared bytes;
  partial rotations preserve sliced traffic, sine sign and caller order;
  unsupported ABIs and missing ``gm->ub`` datapaths fail with ``ModelingError``.
[Timeouts & Locks] Bounded local graph execution. No locks.
[Side-Effect Policy] Reads packaged JSON cases. No input or device mutations.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import replace
from pathlib import Path

import pytest

from _support.axes import REPRESENTATIVE_SKUS, SKUS
from _support.hardware import isa_of

import mindie_sim.api as sdk_api
from mindie_sim import graph_costs, target_graphs
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.operators import builtin_family_registry
from mindie_sim.prediction.operators.contracts import OperatorPlanningContext
from mindie_sim.prediction.operators.rotary import (
    register_family as register_rotary_family,
    resolve_operator_plan as resolve_rotary_plan,
)
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.results import LoweringRequest, OpOccurrence, RankOps
from mindie_sim.prediction.spine import PredictionSpine
from mindie_sim.prediction.streams import MAIN_STREAM
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec

pytestmark = pytest.mark.tier0

ROOT = Path(__file__).resolve().parents[2]
SOURCE_CASES = ROOT / "examples/cases/high-concurrency-20261030/v41-source-conditional"


def _rotary_op(op_id: str = "l0.rope") -> OpSpec:
    shape = (8, 16, 64)
    return OpSpec(
        op_id=op_id,
        layer=0,
        stage=0,
        family="rotary",
        recipe="rotary_position_embedding",
        phase="decode",
        requests=("r0",),
        dtype="bf16",
        local_shape=shape,
        semantic_shape=shape,
        padded_shape=shape,
        launch_shape=shape,
        stream=MAIN_STREAM,
    )


def _case_for(sku: str):
    return sdk_api.compose_case(
        "qwen3-32b", hardware=sku, options={"dp": 1, "tp": 1, "pp": 1, "layers": 1}
    )


def test_rotary_registered_in_builtin_family_registry():
    registry = builtin_family_registry()
    assert registry.lookup(_rotary_op()) == register_rotary_family()


@pytest.mark.parametrize("sku", SKUS)
def test_rotary_work_conservation_and_live_spine_execution(sku):
    registry = builtin_family_registry()
    case = _case_for(sku)
    geom = Geometry.from_devices(case.machine.dies)
    book = RecipeBook.from_mapping(case.runtime.kernels, family_registry=registry)

    raw_op = _rotary_op("rope_shape_only")
    prepared_op, recipe = book.prepare(raw_op, geom)
    assert getattr(recipe, "traffic_rewrite_capable", False) is True

    elems = 8 * 16 * 64
    elem_bytes = 2
    coeff_bytes = 2 * 8 * 64 * elem_bytes
    expected_read = elems * elem_bytes + coeff_bytes
    expected_write = elems * elem_bytes
    assert prepared_op.vector_ops == pytest.approx(float(elems * 3.5))
    assert prepared_op.mte2_bytes == pytest.approx(float(expected_read))
    assert prepared_op.mte3_bytes == pytest.approx(float(expected_write))
    assert prepared_op.hbm_bytes == pytest.approx(float(expected_read + expected_write))

    traffic_op = replace(
        raw_op,
        op_id="rope_declared",
        hbm_bytes=123456.0,
        mte2_bytes=80000.0,
        mte3_bytes=43456.0,
        vector_ops=65536.0,
    )
    spine = PredictionSpine(families=registry)
    step = spine.lower(case, LoweringRequest(declared_ops=(RankOps(0, (traffic_op,)),)))
    assert len(step.operators) == 1
    assert step.operators[0].implementation == "rotary_position_embedding"
    assert step.operators[0].descriptor is not None
    assert step.operators[0].descriptor.sources[0].source_sha256.startswith("5160d08c")
    assert dict(step.operators[0].recipe.metadata)["arch_id"] == isa_of(sku)

    ev = spine.run_step(step)
    assert ev.span_ms > 0.0
    assert len(ev.operators) == 1
    assert ev.operators[0].implementation == "rotary_position_embedding"
    assert len(ev.pipelines) == 1

    # Zero-token rotary (e.g. inactive indexer layer) conserves 0 work without failing.
    zero_op = replace(
        raw_op,
        op_id="rope_zero_tokens",
        local_shape=(0, 128),
        semantic_shape=(0, 128),
        padded_shape=(0, 128),
        launch_shape=(0, 128),
    )
    zero_prepared, _ = book.prepare(zero_op, geom)
    assert zero_prepared.hbm_bytes == 0.0 and zero_prepared.vector_ops == 0.0


def test_rotary_rejects_unsupported_abi_and_missing_datapath():
    case = _case_for(REPRESENTATIVE_SKUS["a2"])
    geom = Geometry.from_devices(case.machine.dies)
    assert geom.datapath is not None

    bad_abi_ctx = OperatorPlanningContext(
        occurrence=OpOccurrence(0, 0, 0, "rope_bad_abi"),
        geometry=geom,
        datapath=geom.datapath,
        requested_abi="aclnnUnsupportedRotary",
    )
    with pytest.raises(ModelingError, match="rotary_abi_unsupported"):
        resolve_rotary_plan(_rotary_op("rope_bad_abi"), bad_abi_ctx)

    no_ub_dp = replace(
        geom.datapath,
        usable_paths=tuple(
            p
            for p in geom.datapath.usable_paths
            if not (p.source == "gm" and p.target == "ub")
        ),
    )
    no_ub_ctx = OperatorPlanningContext(
        occurrence=OpOccurrence(0, 0, 0, "rope_no_ub"),
        geometry=geom,
        datapath=no_ub_dp,
    )
    with pytest.raises(ModelingError, match="rotary_datapath_missing_gm_ub"):
        resolve_rotary_plan(_rotary_op("rope_no_ub"), no_ub_ctx)


def test_rotary_bsnd_4d_abi_domain_picklability_and_hbm_split():
    registry = builtin_family_registry()
    case = _case_for(REPRESENTATIVE_SKUS["a5"])
    geom = Geometry.from_devices(
        case.machine.dies,
        links=case.machine.links,
        transfer_channels=case.machine.transfer_channels,
    )
    assert geom.datapath is not None

    # 4D BSND [1, 8, 16, 64]: tokens = 1 * 8 = 8 (not shape[0] == 1).
    bsnd_op = replace(
        _rotary_op("rope_bsnd"),
        local_shape=(1, 8, 16, 64),
        semantic_shape=(1, 8, 16, 64),
        padded_shape=(1, 8, 16, 64),
        launch_shape=(1, 8, 16, 64),
    )
    ctx_ok = OperatorPlanningContext(
        occurrence=OpOccurrence(0, 0, 0, "rope_bsnd"),
        geometry=geom,
        datapath=geom.datapath,
        requested_abi="aclnnRotaryPositionEmbedding",
    )
    resolved = resolve_rotary_plan(bsnd_op, ctx_ok)
    elems = 1 * 8 * 16 * 64
    coeff_bytes = 2 * (1 * 8) * 64 * 2
    tensor_bytes = elems * 2
    assert resolved.recipe.work.resources["mte2"] == pytest.approx(
        float(tensor_bytes + coeff_bytes)
    )
    assert resolved.recipe.work.resources["mte3"] == pytest.approx(float(tensor_bytes))
    assert resolved.recipe.work.resources["vector"] == pytest.approx(float(elems * 3.5))

    # Descriptor HBM read/write split matches io_weight (write == mte3, read == total - write).
    assert resolved.descriptor is not None
    load_hbm = next(
        w.amount for w in resolved.descriptor.phases[0].work if w.resource == "hbm"
    )
    store_hbm = next(
        w.amount for w in resolved.descriptor.phases[2].work if w.resource == "hbm"
    )
    assert store_hbm == pytest.approx(float(tensor_bytes))
    assert load_hbm == pytest.approx(float(tensor_bytes + coeff_bytes))

    # Dual-tensor aclnnApplyRotaryPosEmb must be rejected by single-tensor rotary.
    ctx_dual = replace(ctx_ok, requested_abi="aclnnApplyRotaryPosEmb")
    with pytest.raises(ModelingError, match="rotary_abi_unsupported"):
        resolve_rotary_plan(bsnd_op, ctx_dual)

    # Misspelled explicit recipe ID must not be hijacked by FamilyRegistry or run at 0 ms.
    typo_op = replace(bsnd_op, recipe="rotary_misspelled")
    assert registry.match(typo_op) is None
    with pytest.raises(ModelingError, match="unregistered_operator_family"):
        registry.lookup(typo_op)
    book = RecipeBook.from_mapping(case.runtime.kernels, family_registry=registry)
    with pytest.raises(
        ModelingError, match="unregistered_operator_recipe:rotary:rotary_misspelled"
    ):
        book.select(typo_op)
    with pytest.raises(
        ModelingError, match="unregistered_operator_recipe:rotary:rotary_misspelled"
    ):
        graph_costs(case, (typo_op,), recipes=book)

    # Plans and evaluation results also cross process boundaries.
    plan_roundtrip = pickle.loads(pickle.dumps(resolved.recipe))
    assert plan_roundtrip.work == resolved.recipe.work
    spine = PredictionSpine(families=registry)
    step = spine.lower(case, LoweringRequest(declared_ops=(RankOps(0, (bsnd_op,)),)))
    ev = spine.run_step(step)
    assert ev.operators[0].occurrence.rank == 0
    assert any(
        item.resource == "vector" and item.amount > 0.0 for item in ev.operators[0].work
    )
    assert pickle.loads(pickle.dumps(ev)) == ev


def _source_rotations(batch=1, *, width=512, heads=64):
    raw = json.loads((ROOT / "examples/source-families/v41-mqsmla-source-control.json").read_text())
    raw["model"]["layers"] = 1
    raw["model"]["hf_config"]["text_config"].update(head_dim=width, num_attention_heads=heads)
    lowerings = raw["runtime"]["kernels"]["lowerings"]
    lowerings["v41_rotary"] = "v41_partial_rotary"
    del lowerings["mqsmla_attention"]
    case = parse_case(raw)
    ops = target_graphs(case, scope="body", phase="decode", tokens=1, sequences=batch, context=129)[0]
    return case, {op.recipe: op for op in ops if op.family == "rotary"}


def _rotary_plan(case, op):
    geometry = Geometry.from_devices(case.machine.dies[:1])
    ctx = OperatorPlanningContext(OpOccurrence(0, op.stage, 0, op.op_id), geometry, geometry.datapath)
    return resolve_rotary_plan(op, ctx).recipe


def test_q_kv_and_inverse_have_three_independent_native_boundaries():
    case, rotations = _source_rotations()
    ops = [rotations[name] for name in ("q_rotary", "kv_rotary", "output_inverse_rotary")]
    assert len({op.op_id for op in ops}) == 3
    assert all(op.source_boundary == "kernel" and len(op.deps) == 1 for op in ops)
    assert ops[0].deps[0].endswith(".q_b") and ops[1].deps[0].endswith(".kv_norm")
    assert len({op.deps for op in ops}) == 3
    # BF16 slices and FP32 coefficients retain their distinct GM issue and HBM totals.
    work = [_rotary_plan(case, op).work.resources for op in ops]
    assert [entry["mte2"] for entry in work] == [40960, 640, 40960]
    assert [entry["mte3"] for entry in work] == [8192, 128, 8192]
    assert [entry["hbm"] for entry in work] == [16896, 768, 16896]
    assert work[0]["vector"] == work[2]["vector"]
    assert dict(ops[2].metadata)["native_tensor_view"]["sine_sign"] == "caller_negated"


def test_nope_width_does_not_create_inplace_slice_traffic_or_vector_work():
    cases = [_source_rotations(16, width=width, heads=32) for width in (128, 512)]
    short, long = [ops["q_rotary"] for _, ops in cases]
    assert short.hbm_bytes == long.hbm_bytes == 139264
    assert short.vector_ops == long.vector_ops
    assert short.local_shape == (16, 32, 128) and long.local_shape == (16, 32, 512)
    assert dict(long.metadata)["native_tensor_view"]["partial_slice"] == (448, 512)


def test_batch_scaling_and_coefficient_reuse_are_explicit():
    first, last = [_source_rotations(batch)[1]["q_rotary"] for batch in (1, 16)]
    assert last.hbm_bytes == 16 * first.hbm_bytes
    assert last.vector_ops == 16 * first.vector_ops
    work = dict(last.metadata)["partial_rotary_work"]
    assert work["coefficient_bytes"] == 16 * 4 * 2 * 64 * 4
    assert work["vector_lane_operations"]["coefficient_broadcast"] == 16 * 2 * 63 * 64
    assert not work["native_owner_split_and_reload_not_selected"]
    assert work["ub_bytes_per_owner"] == 1024


def test_public_caller_replaces_combined_rotation_and_orders_inverse_before_output_projection():
    raw = json.loads(
        (ROOT / "examples/source-families/v41-mqsmla-source-control.json").read_text(
            encoding="utf-8"
        )
    )
    raw["model"]["layers"] = 3
    probe = dict(scope="body", phase="decode", tokens=1, sequences=2, context=129)
    original = target_graphs(raw, **probe)
    raw["runtime"]["kernels"]["lowerings"]["v41_rotary"] = "v41_partial_rotary"
    selected = target_graphs(raw, **probe)
    for rank, ops in selected.items():
        before = original[rank]
        assert sum(op.flops for op in ops) == sum(op.flops for op in before)
        assert not any(op.recipe == "rotary" for op in ops)
        for layer in range(3):
            by_recipe = {op.recipe: op for op in ops if op.layer == layer}
            q, kv, inverse = (
                by_recipe[name]
                for name in ("q_rotary", "kv_rotary", "output_inverse_rotary")
            )
            assert q.deps == (by_recipe["q_b"].op_id,)
            assert kv.deps == (by_recipe["kv_norm"].op_id,)
            assert by_recipe["kv_store"].deps == (kv.op_id,)
            wo_a = by_recipe["grouped_output_a"]
            quantize = next(op for op in ops if op.op_id == wo_a.deps[0])
            assert quantize.family == "quantize" and quantize.deps == (inverse.op_id,)
            assert all(op.source_boundary == "kernel" for op in (q, kv, inverse))
            attention = [
                op for op in ops if op.layer == layer and op.family == "attention"
            ]
            assert attention and set(inverse.deps) == {op.op_id for op in attention}
            assert all(q.op_id in op.deps for op in attention)
    result = graph_costs(
        raw,
        selected,
        input_ready_ms=7,
        external_inputs={"analysis_input": 7},
        isolated=False,
    )
    assert result["duration_ms"] > 0
    assert (
        result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
    )
    assert all(
        abs(value) < 1e-7 for value in result["execution"]["live_allocated"].values()
    )
    rotations = [
        row
        for row in result["timeline"]
        if row["recipe"] in ("q_rotary", "kv_rotary", "output_inverse_rotary")
    ]
    assert len(rotations) == 3 * 3 * len(selected)
    assert all(row["ready_ms"] >= 7 for row in rotations)


def test_regbase_ab_bn_split_and_kv_tail_are_geometry_derived():
    case, rotations = _source_rotations(2)
    q, kv = (rotations[name] for name in ("q_rotary", "kv_rotary"))
    assert _rotary_plan(case, q).work.resources["mte2"] == 49152
    assert _rotary_plan(case, kv).work.resources["mte2"] == 1280
    geom = Geometry.from_devices(case.machine.dies[:1])
    book = RecipeBook.from_mapping(case.runtime.kernels)
    qplan, kvplan = (book.select(op).build(op, geom) for op in (q, kv))
    qt = dict(qplan.metadata["host_tiler"])
    kt = dict(kvplan.metadata["host_tiler"])
    assert (
        qt["class"],
        qt["layout"],
        qt["owners"],
        qt["block_num_n"],
        qt["ub_bytes_owner"],
    ) == ("RopeRegBaseTilingClassAB", "SBND", 64, 32, 2048)
    assert (kt["class"], kt["layout"], kt["owners"], kt["ub_bytes_owner"]) == (
        "RopeRegBaseTilingClassAAndB",
        "NO_BROADCAST",
        2,
        1536,
    )
    assert qplan.work.resources["mte2_aiv"] == 49152
    assert kvplan.work.resources["mte2_aiv"] == 1280


@pytest.mark.parametrize(
    "path", sorted(SOURCE_CASES.glob("*.json")), ids=lambda path: path.stem
)
def test_complete_source_variant_preserves_geometry_and_prepares_real_programs(path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    selected = raw["runtime"]["metadata"]["source_scenario"]
    full_case = parse_case(raw)
    assert full_case.model.layers == 40
    assert (
        full_case.model.quant.for_category("routed_expert").expert_weight_format
        == "mxfp4"
    )
    raw["model"]["layers"] = 2
    if "body_layers" in raw["model"]:
        raw["model"]["body_layers"] = 2
    if "layer_plans" in raw["model"]:
        raw["model"]["layer_plans"] = raw["model"]["layer_plans"][:2]
    case = parse_case(raw)
    graphs = target_graphs(case, **selected["conditions"]["single_active_rank0"])
    assert len(graphs) == 8
    rank0 = graphs[0]
    attention = [
        op for op in rank0 if op.recipe == "cannbot_mixed_quant_sparse_flash_mla"
    ]
    assert len(attention) == 2
    assert {op.local_shape[0] for op in attention} == {
        selected["observed_rank0_geometry"]["query_rows"]
    }
    assert (
        len([op for op in rank0 if dict(op.metadata).get("fusion_role") == "hc_pre"])
        == 4
    )
    assert (
        len([op for op in rank0 if dict(op.metadata).get("fusion_role") == "hc_post"])
        == 4
    )
    grouped = next(op for op in rank0 if op.family == "grouped_gemm")
    book = RecipeBook.from_mapping(case.runtime.kernels)
    prepared, _ = book.select(grouped).prepare(
        grouped, Geometry.from_devices(case.machine.dies[:1])
    )
    execution = dict(prepared.metadata)["program_execution"]
    assert execution["source"]["selector"].startswith("MXA8W4_NZNK/")
    assert prepared.flops > 0
    for rank in range(1, 8):
        assert not any(
            op.recipe == "cannbot_mixed_quant_sparse_flash_mla" for op in graphs[rank]
        )
        assert len([op for op in graphs[rank] if op.recipe == "expert_combine"]) == 2
