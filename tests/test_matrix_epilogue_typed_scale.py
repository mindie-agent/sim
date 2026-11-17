"""Typed ABI scale storage must reach the common INT8 matrix epilogue.

The product W8A8 QBMM record currently carries the catalog checkpoint default
(FP32 scale plus offset).  The captured ABI has four inputs: INT8 activation,
INT8 FRACTAL_NZ weight, BF16 weight scale, FP32 per-token scale.  T12's typed
input slots are the authority.  The tests below use one real product case and
the exact captured byte counts from T18; they do not read a target duration.
"""
from dataclasses import replace
import json
import math
from pathlib import Path

import pytest

#: A5 sustainable DRAM service is declared at 0.8316 of the raw HBM peak.
A5_DRAM_SERVICE_SCALE = 1 / 0.8316

from mindie_sim import graph_costs, operator_costs, target_graphs
from mindie_sim.prediction.matrix_epilogue import (
    MatrixEpilogue, plan_matrix_epilogue, reconcile_native_scale_storage,
    reconcile_scale_storage,
)
from mindie_sim.prediction.matrix_steady import SteadyMatmul
from mindie_sim.prediction.native_binding import bind_native_call, declare_native_call
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.schema.common import ModelingError

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "examples/cases/high-concurrency-20261030/new9/qwen36-a5-tp2-c1.json"
CAPTURED_INPUT_BYTES = 44_650_512
CAPTURED_WEIGHT_SCALE_BYTES = 10_240
DEFAULT_COLUMN_SCALE_BYTES = 40_960
OP_ID = "s0.l0.down"


def _case_and_op():
    raw = json.loads(CASE.read_text())
    raw["model"]["layers"] = 1
    ops = target_graphs(raw, scope="body", phase="decode", tokens=4,
                        sequences=1, context=4096)[0]
    return raw, next(op for op in ops if op.op_id == OP_ID)


def _legacy_fp32_offset(op):
    """Reconstruct the declared FP32 scale+offset record from real dimensions.

    The product case serialized the checkpoint default; T18 measured that gap.
    Building it explicitly keeps this regression independent of a later input
    generator fix while preserving the real M/N/K/group values.
    """
    metadata = dict(op.metadata)
    m, n, k = (int(value) for value in op.local_shape)
    groups = int(metadata.get("matrix_groups", 1))
    payload = groups * n * k
    column = groups * n * 8
    row = m * 4
    output = m * n * 2
    hbm = m * k + payload + column + row + output
    metadata.update({
        "matrix_epilogue": {
            "kind": "row_column_scale", "row_scale_bytes": row,
            "column_scale_bytes": column, "operations_per_output": 4,
        },
        "packed_weight_bytes": payload + column,
        "weight_bytes": payload + column,
        "weight_scale_format": "fp32", "weight_scale_bytes": 4.0,
        "weight_scale_storage_format": "fp32", "weight_scale_storage_bytes": 4.0,
        "activation_scale_format": "fp32", "activation_scale_bytes": row,
        "int8_weight_metadata": "scale_offset",
        "column_metadata_policy": "retains_declared_scale_and_offset_storage_traffic",
    })
    return replace(op, hbm_bytes=hbm, mte2_bytes=hbm - output,
                   mte3_bytes=output, vector_ops=4 * m * n,
                   metadata=tuple(metadata.items()))


def _declared(op, dtypes=("int8", "int8", "bf16", "fp32")):
    return declare_native_call(
        op,
        api={"namespace": "cann_kernel", "entry": "QuantBatchMatmulV3"},
        source={"repository": "mindie-sim-data", "path": "native_qbmm_row120",
                "sha256": "0" * 64, "symbol": "QuantBatchMatmulV3",
                "revision": "20261030", "state": "task_artifact"},
        typed={"attrs": {"input_dtypes": list(dtypes),
                         "input_formats": ["ND", "FRACTAL_NZ", "ND", "ND"],
                         "output_formats": ["ND"]}},
        kernels=[{"name": "QuantBatchMatmulV3", "role": "cube",
                   "evidence": "source_declared"}],
    )


def test_default_fp32_scale_offset_cannot_stand_for_typed_bf16_scale():
    _raw, op = _case_and_op()
    legacy = _legacy_fp32_offset(op)
    # The declared path alone is internally consistent and keeps the bytes.
    declared_only = MatrixEpilogue.from_op(legacy)
    assert declared_only.row_scale_bytes == 16
    assert declared_only.column_scale_bytes == DEFAULT_COLUMN_SCALE_BYTES
    assert legacy.hbm_bytes == 44_681_232
    # T12's real four-slot ABI makes the mismatch explicit instead of silently
    # pricing the default FP32 scale/offset.
    with pytest.raises(ModelingError, match="typed_scale_representation_mismatch"):
        MatrixEpilogue.from_op(_declared(legacy))


def test_reconciled_typed_abi_matches_capture_and_operator_native_work():
    raw, op = _case_and_op()
    legacy = _legacy_fp32_offset(op)
    fixed = reconcile_native_scale_storage(_declared(legacy))
    assert fixed.hbm_bytes == CAPTURED_INPUT_BYTES
    assert fixed.mte2_bytes == CAPTURED_INPUT_BYTES - 40_960
    assert fixed.mte3_bytes == 40_960
    metadata = dict(fixed.metadata)
    epilogue = MatrixEpilogue.from_op(fixed)
    assert epilogue.row_scale_bytes == 16
    assert epilogue.column_scale_bytes == CAPTURED_WEIGHT_SCALE_BYTES
    assert metadata["weight_scale_format"] == "bf16"
    assert metadata["weight_scale_bytes"] == 2.0
    assert metadata["int8_weight_metadata"] == "scale_only"
    assert metadata["packed_weight_bytes"] == 44_574_720
    assert metadata["native_scale_reconciliation"]["hbm_delta_bytes"] == -30_720
    assert metadata["native_scale_reconciliation"]["offset"] == \
        "not_present_in_four_input_abi"
    # Idempotent reconciliation; no target duration is read or stored.
    assert reconcile_native_scale_storage(fixed) == fixed

    tiled = operator_costs(raw, (fixed,), input_ready_ms=0)["operators"][0]
    steady = operator_costs(
        raw, (fixed,),
        recipes=RecipeBook(recipes={"dense_down": SteadyMatmul()}),
        input_ready_ms=0)["operators"][0]
    for row in (tiled, steady):
        assert row["declared_service"]["memory@r0:s0"] == pytest.approx(
            CAPTURED_INPUT_BYTES)
        assert row["serviced"]["memory@r0:s0"] == pytest.approx(
            CAPTURED_INPUT_BYTES, rel=1e-12)
        assert row["operator"]["hbm_bytes"] == CAPTURED_INPUT_BYTES
    # Tiled and Steady are allowed to be different explicit programs, but the
    # same physical input boundary must not be reported as the same plan.
    assert tiled["program"] == "TiledMatmul"
    assert steady["program"] == "SteadyMatmul"
    assert tiled["program"] != steady["program"]

    # A native binding without a real selected plan must not fabricate a
    # physical work view.  The real native/operator/cohort plan wiring belongs
    # to T12; this file owns the typed scale representation only.
    declared = _declared(fixed)
    binding = bind_native_call(declared)
    assert binding.physical is None
    assert binding.typed["attrs"]["input_dtypes"] == [
        "int8", "int8", "bf16", "fp32"]



def test_absent_typed_abi_keeps_the_declared_scale_record():
    _raw, op = _case_and_op()
    legacy = _legacy_fp32_offset(op)
    assert reconcile_native_scale_storage(legacy) is legacy
    assert MatrixEpilogue.from_op(legacy).column_scale_bytes == \
        DEFAULT_COLUMN_SCALE_BYTES


def test_reconciliation_normalizes_stale_metadata_without_repricing_bytes():
    _raw, op = _case_and_op()
    fixed = reconcile_native_scale_storage(_declared(_legacy_fp32_offset(op)))
    stale_metadata = dict(fixed.metadata)
    stale_metadata.update({"weight_scale_format": "fp32",
                           "weight_scale_bytes": 4.0,
                           "int8_weight_metadata": "scale_offset"})
    stale = replace(fixed, metadata=tuple(stale_metadata.items()))
    restored = reconcile_native_scale_storage(_declared(stale))
    assert restored.hbm_bytes == fixed.hbm_bytes
    assert dict(restored.metadata)["weight_scale_format"] == "bf16"
    assert dict(restored.metadata)["weight_scale_bytes"] == 2.0
    assert dict(restored.metadata)["int8_weight_metadata"] == "scale_only"


def test_row_runtime_representation_is_part_of_the_same_typed_contract():
    _raw, op = _case_and_op()
    fixed = reconcile_native_scale_storage(_declared(_legacy_fp32_offset(op)))
    stale_metadata = dict(fixed.metadata)
    stale_metadata.update({"activation_scale_format": "bf16",
                           "activation_scale_bytes": 8})
    stale = replace(fixed, metadata=tuple(stale_metadata.items()))
    # The byte total is already right, but the row consumer would read BF16/8 B
    # instead of the typed FP32/16 B slot.  Read-only parsing must reject it.
    with pytest.raises(ModelingError, match="typed_scale_representation_mismatch"):
        MatrixEpilogue.from_op(_declared(stale))
    restored = reconcile_native_scale_storage(_declared(stale))
    metadata = dict(restored.metadata)
    assert restored.hbm_bytes == fixed.hbm_bytes
    assert restored.mte2_bytes == fixed.mte2_bytes
    assert metadata["activation_scale_format"] == "fp32"
    assert metadata["activation_scale_bytes"] == 16
    assert metadata["activation_scale_shape"] == (4,)
    assert metadata["matrix_epilogue"]["row_scale_bytes"] == 16
    assert metadata["native_scale_reconciliation"]["hbm_delta_bytes"] == 0
    assert reconcile_native_scale_storage(restored) == restored


def test_typed_scale_formula_on_a_legal_tail_geometry():
    _raw, op = _case_and_op()
    fixed = reconcile_native_scale_storage(_declared(_legacy_fp32_offset(op)))
    m, n, k = 5, 7, 11
    groups = 1
    payload = groups * n * k
    row, column = m * 4, groups * n * 2
    output = m * n * 2
    hbm = m * k + payload + column + row + output
    metadata = dict(fixed.metadata)
    metadata.update({
        "matrix_epilogue": {"kind": "row_column_scale",
                            "row_scale_bytes": row,
                            "column_scale_bytes": column,
                            "operations_per_output": 4},
        "packed_weight_bytes": payload + column,
        "weight_bytes": payload + column,
        "weight_scale_format": "bf16", "weight_scale_bytes": 2.0,
        "activation_scale_format": "fp32", "activation_scale_bytes": row,
        "activation_scale_shape": (m,), "weight_scale_shape": (n,),
        "int8_weight_metadata": "scale_only",
    })
    tail = replace(fixed, local_shape=(m, n, k), semantic_shape=(m, n, k),
                   launch_shape=(m, n, k), padded_shape=(m, n, k),
                   hbm_bytes=hbm, mte2_bytes=hbm - output,
                   mte3_bytes=output, vector_ops=4 * m * n,
                   metadata=tuple(metadata.items()))
    result = MatrixEpilogue.from_op(_declared(tail))
    assert (result.row_scale_bytes, result.column_scale_bytes) == (20, 14)
    assert reconcile_native_scale_storage(_declared(tail)) == _declared(tail)
    # The same epilogue split is used for a non-tile-aligned output tail.
    operands = result.operands(tail)
    assert operands == (tail.mte2_bytes - (payload + column) - row,
                        payload)


def test_real_quantize_producer_feeds_the_corrected_matrix_plan():
    raw = json.loads(CASE.read_text())
    raw["model"]["layers"] = 1
    quant = raw["model"]["quant"]
    quant.update({"runtime_weight_scale_format": "bf16",
                  "runtime_weight_scale_bytes": 2,
                  "runtime_activation_scale_format": "fp32",
                  "runtime_int8_weight_metadata": "scale_only"})
    ops = target_graphs(raw, scope="body", phase="decode", tokens=4,
                        sequences=1, context=4096)[0]
    matrix = next(op for op in ops if op.op_id == OP_ID)
    producer = next(op for op in ops if op.op_id in matrix.deps)
    producer_meta = dict(producer.metadata)
    matrix_meta = dict(matrix.metadata)
    assert producer.family == "quantize"
    assert producer_meta["output_format"] == "int8"
    assert producer_meta["activation_scale_format"] == "fp32"
    assert matrix_meta["activation_producer"] == "explicit_quantization"
    assert matrix_meta["activation_scale_shape"] == (4,)
    assert matrix_meta["matrix_epilogue"]["column_scale_bytes"] == 10_240
    # One joint graph keeps the producer dependency; do not price the matrix
    # through a deps-stripped cohort or separate isolated calls.
    joint = graph_costs(raw, (producer, matrix), input_ready_ms=0,
                        external_inputs={producer.deps[0]: 0})
    timeline = {row["op_id"]: row for row in joint["timeline"]}
    assert matrix.op_id in timeline and producer.op_id in timeline
    assert timeline[matrix.op_id]["predecessors"] == [
        f"r0:{producer.op_id}"]
    assert timeline[matrix.op_id]["ready_ms"] == pytest.approx(
        timeline[producer.op_id]["complete_ms"])
    # The matrix payload stays byte-exact; the vector quantize producer is
    # normalized by the declared sustainable DRAM service ceiling.
    assert joint["execution"]["serviced"]["memory@r0:s0"] == pytest.approx(
        matrix.hbm_bytes + producer.hbm_bytes * A5_DRAM_SERVICE_SCALE, rel=1e-12)
    assert matrix.hbm_bytes == CAPTURED_INPUT_BYTES


def test_multidimensional_row_scale_shape_cannot_be_accepted_as_is():
    _raw, op = _case_and_op()
    fixed = reconcile_native_scale_storage(_declared(_legacy_fp32_offset(op)))
    stale_metadata = dict(fixed.metadata)
    stale_metadata["activation_scale_shape"] = (4, 2)
    stale = replace(fixed, metadata=tuple(stale_metadata.items()))
    # Four FP32 row scales cannot be represented as (4,2); the byte total is
    # unchanged, so a byte-only check would wrongly accept this record.
    with pytest.raises(ModelingError, match="typed_scale_representation_mismatch"):
        MatrixEpilogue.from_op(_declared(stale))
    restored = reconcile_native_scale_storage(_declared(stale))
    metadata = dict(restored.metadata)
    assert restored.hbm_bytes == fixed.hbm_bytes
    assert restored.mte2_bytes == fixed.mte2_bytes
    assert metadata["activation_scale_shape"] == (4,)
    assert metadata["activation_scale_bytes"] == 16
    assert math.prod(metadata["activation_scale_shape"]) == 4
    assert metadata["activation_scale_bytes"] == 4 * 4
    assert metadata["native_scale_reconciliation"]["hbm_delta_bytes"] == 0
    assert reconcile_native_scale_storage(restored) == restored


def test_bound_producer_scale_representation_cannot_be_relabelled():
    raw = json.loads(CASE.read_text())
    raw["model"]["layers"] = 1
    quant = raw["model"]["quant"]
    quant.update({"runtime_weight_scale_format": "bf16",
                  "runtime_weight_scale_bytes": 2,
                  "runtime_activation_scale_format": "fp32",
                  "runtime_int8_weight_metadata": "scale_only"})
    ops = target_graphs(raw, scope="body", phase="decode", tokens=4,
                        sequences=1, context=4096)[0]
    matrix = next(op for op in ops if op.op_id == OP_ID)
    producer = next(op for op in ops if op.op_id in matrix.deps)
    declared = reconcile_native_scale_storage(_declared(matrix), producer=producer)
    assert MatrixEpilogue.from_op(declared, producer=producer).row_scale_bytes == 16
    assert reconcile_native_scale_storage(declared, producer=producer) == declared
    bad_metadata = dict(producer.metadata)
    bad_metadata.update({"activation_scale_format": "bf16",
                         "activation_scale_bytes": 8,
                         "activation_scale_shape": (4, 2)})
    bad = replace(producer, metadata=tuple(bad_metadata.items()))
    with pytest.raises(ModelingError, match="producer_scale_conflict"):
        MatrixEpilogue.from_op(declared, producer=bad)
    with pytest.raises(ModelingError, match="producer_scale_conflict"):
        reconcile_native_scale_storage(declared, producer=bad)


def test_missing_runtime_shapes_are_generated_not_treated_as_consistent():
    _raw, op = _case_and_op()
    fixed = reconcile_native_scale_storage(_declared(_legacy_fp32_offset(op)))
    stale_metadata = dict(fixed.metadata)
    stale_metadata.pop("activation_scale_shape", None)
    stale_metadata.pop("weight_scale_shape", None)
    stale = replace(fixed, metadata=tuple(stale_metadata.items()))
    with pytest.raises(ModelingError, match="typed_scale_representation_mismatch"):
        MatrixEpilogue.from_op(_declared(stale))
    restored = reconcile_native_scale_storage(_declared(stale))
    metadata = dict(restored.metadata)
    assert metadata["activation_scale_shape"] == (4,)
    assert metadata["weight_scale_shape"] == (5120,)
    assert restored.hbm_bytes == fixed.hbm_bytes
    assert restored.mte2_bytes == fixed.mte2_bytes
    assert reconcile_native_scale_storage(restored) == restored


def test_graph_helper_normalizes_same_object_and_detects_producer_conflict():
    raw = json.loads(CASE.read_text())
    raw["model"]["layers"] = 1
    quant = raw["model"]["quant"]
    quant.update({"runtime_weight_scale_format": "bf16",
                  "runtime_weight_scale_bytes": 2,
                  "runtime_activation_scale_format": "fp32",
                  "runtime_int8_weight_metadata": "scale_only"})
    ops = target_graphs(raw, scope="body", phase="decode", tokens=4,
                        sequences=1, context=4096)[0]
    matrix = next(op for op in ops if op.op_id == OP_ID)
    producer = next(op for op in ops if op.op_id in matrix.deps)
    declared = _declared(matrix)
    prepared = reconcile_scale_storage((producer, declared))
    assert prepared[0] == producer
    normalized, epilogue = plan_matrix_epilogue(prepared[1], producer=producer)
    metadata = dict(normalized.metadata)
    assert metadata["activation_scale_shape"] == (4,)
    assert metadata["weight_scale_shape"] == (5120,)
    assert normalized.hbm_bytes == CAPTURED_INPUT_BYTES
    assert epilogue == MatrixEpilogue(16, 10_240, 4)
    inputs, packed = epilogue.operands(normalized)
    assert packed == 44_564_480
    assert inputs == normalized.mte2_bytes - metadata["packed_weight_bytes"] - 16
    # The normalized object, not a manual scalar/plan, enters the public
    # graph preparation and the same Tiled program.
    joint = graph_costs(raw, (producer, normalized), input_ready_ms=0,
                        external_inputs={producer.deps[0]: 0})
    timeline = {row["op_id"]: row for row in joint["timeline"]}
    assert timeline[normalized.op_id]["program"] == "TiledMatmul"
    assert timeline[normalized.op_id]["predecessors"] == [f"r0:{producer.op_id}"]
    assert joint["execution"]["serviced"]["memory@r0:s0"] == pytest.approx(
        normalized.hbm_bytes + producer.hbm_bytes * A5_DRAM_SERVICE_SCALE,
        rel=1e-12)
    bad_metadata = dict(producer.metadata)
    bad_metadata.update({"activation_scale_bytes": 8,
                         "activation_scale_format": "bf16"})
    bad = replace(producer, metadata=tuple(bad_metadata.items()))
    with pytest.raises(ModelingError, match="producer_scale_conflict"):
        reconcile_scale_storage((bad, declared))
