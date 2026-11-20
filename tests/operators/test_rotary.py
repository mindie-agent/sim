"""Governed ``rotary`` operator family registration, work conservation, and replay.

[Role] Verify ``mindie_sim.prediction.operators.rotary`` executes live through
  ``FamilyRegistry``, ``RecipeBook``, and ``PredictionSpine`` across Ascend
  architectures (``a2``, ``a3``, ``a5``).
[Invariants] Standalone shape-only rotary ops derive conserved rotate-half
  vector/MTE2/MTE3/HBM work; pre-declared traffic ops preserve declared bytes;
  unsupported ABIs and missing ``gm->ub`` datapaths fail with ``ModelingError``.
[Timeouts & Locks] Fast Tier-0 in-memory tests (<3s); no I/O or locks.
[Side-Effect Policy] Zero filesystem writes or global state mutations.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

import mindie_sim.api as sdk_api
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.operators import builtin_family_registry
from mindie_sim.prediction.operators.contracts import OperatorPlanningContext
from mindie_sim.prediction.operators.rotary import (
    register_family as register_rotary_family,
    resolve_operator_plan as resolve_rotary_plan,
)
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.results import OpOccurrence
from mindie_sim.prediction.spine import PredictionSpine
from mindie_sim.prediction.streams import MAIN_STREAM
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from mindie_sim.stack_registry import LoweringRequest, RankOps

pytestmark = pytest.mark.tier0

SKU_ARCH_CASES = (
    ("a2", "Ascend910B2", "npu_arch_2201"),
    ("a3", "Ascend910_9382", "npu_arch_2201"),
    ("a5", "Ascend950DT_9581", "npu_arch_3510"),
)


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


@pytest.mark.parametrize("tier,sku,expected_arch", SKU_ARCH_CASES)
def test_rotary_work_conservation_and_live_spine_execution(tier, sku, expected_arch):
    del tier
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
    step = spine.lower(
        case, LoweringRequest(declared_ops=(RankOps(0, (traffic_op,)),))
    )
    assert len(step.operators) == 1
    assert step.operators[0].implementation == "rotary_position_embedding"
    assert step.operators[0].descriptor is not None
    assert step.operators[0].descriptor.sources[0].source_sha256.startswith("5160d08c")
    assert dict(step.operators[0].recipe.metadata)["arch_id"] == expected_arch

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
    case = _case_for("Ascend910B2")
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
            p for p in geom.datapath.usable_paths
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
    import pickle

    registry = builtin_family_registry()
    case = _case_for("Ascend950DT_9581")
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
    assert resolved.recipe.work.resources["mte2"] == pytest.approx(float(tensor_bytes + coeff_bytes))
    assert resolved.recipe.work.resources["mte3"] == pytest.approx(float(tensor_bytes))
    assert resolved.recipe.work.resources["vector"] == pytest.approx(float(elems * 3.5))

    # Descriptor HBM read/write split matches io_weight (write == mte3, read == total - write).
    assert resolved.descriptor is not None
    load_hbm = next(w.amount for w in resolved.descriptor.phases[0].work if w.resource == "hbm")
    store_hbm = next(w.amount for w in resolved.descriptor.phases[2].work if w.resource == "hbm")
    assert store_hbm == pytest.approx(float(tensor_bytes))
    assert load_hbm == pytest.approx(float(tensor_bytes + coeff_bytes))

    # Dual-tensor aclnnApplyRotaryPosEmb must be rejected by single-tensor rotary.
    ctx_dual = replace(ctx_ok, requested_abi="aclnnApplyRotaryPosEmb")
    with pytest.raises(ModelingError, match="rotary_abi_unsupported"):
        resolve_rotary_plan(bsnd_op, ctx_dual)

    # Misspelled explicit recipe ID must not be hijacked by FamilyRegistry or run at 0 ms.
    from mindie_sim.prediction.graph_analysis import graph_costs

    typo_op = replace(bsnd_op, recipe="rotary_misspelled")
    assert registry.match(typo_op) is None
    with pytest.raises(ModelingError, match="unregistered_operator_family"):
        registry.lookup(typo_op)
    book = RecipeBook.from_mapping(case.runtime.kernels, family_registry=registry)
    with pytest.raises(ModelingError, match="unregistered_operator_recipe:rotary:rotary_misspelled"):
        book.select(typo_op)
    with pytest.raises(ModelingError, match="unregistered_operator_recipe:rotary:rotary_misspelled"):
        graph_costs(case, (typo_op,), recipes=book)

    # OperatorPlan and StepEvaluation must pickle cleanly under multiprocessing spawn.
    plan_roundtrip = pickle.loads(pickle.dumps(resolved.recipe))
    assert plan_roundtrip.work == resolved.recipe.work
    spine = PredictionSpine(families=registry)
    step = spine.lower(case, LoweringRequest(declared_ops=(RankOps(0, (bsnd_op,)),)))
    ev = spine.run_step(step)
    assert ev.operators[0].occurrence.rank == 0
    assert any(item.resource == "vector" and item.amount > 0.0 for item in ev.operators[0].work)
    assert pickle.loads(pickle.dumps(ev)) == ev
