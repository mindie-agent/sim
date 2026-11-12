"""Checkpoint norm gamma storage does not override the actual BF16 kernel.

[Role]
  Separate three quantities for a normalized layer: the checkpoint gamma
  storage (F32 in the signed GLM deployments), the loaded/kernel gamma format
  (BF16 in the retained RmsNorm/AddRmsNormBias ABI), and the activation and
  output formats (BF16).  The disk format is provenance, not a per-forward
  price.
[Invariants]
  * A stored F32 norm gamma is reported as 24576 B for 6144 elements; the
    deployed BF16 gamma read is 12288 B; both are reported separately.
  * The native and piecewise norm declarations charge one BF16 gamma read,
    once per launch, in addition to the activation read and output write.
  * F32 storage alone never changes native or piecewise work; only an explicit
    runtime ABI declaration can select another loaded format.
  * Shared-head, Engram and hyper-connection norms are not silently promoted
    by the generic layer-norm storage declaration.
  * Absent a norm declaration, the existing BF16 actual is preserved.
[Timeouts & Locks]
  One 78-layer GLM body construction plus two one-token forwards; no device,
  filesystem writes or locks.
[Side-Effect Policy]
  Read-only checked-in GLM case; immutable records only.
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path

import pytest

from mindie_sim.catalog.types import QuantSpec
from mindie_sim.schema.common import ModelingError
from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.piecewise.compiler import PipeSupply
from mindie_sim.prediction.piecewise.geometry import RequestGeometry, WorkloadGeometry
from mindie_sim.prediction.piecewise.model_pipeline import assemble_model
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.subgraphs.instantiate import FeatureConfig

ROOT = Path(__file__).resolve().parents[1]
GLM = ROOT / "examples/cases/glm-5.3.json"
WIDTH = 6144


def _quant(declared: bool, *, stored: str = "fp32") -> QuantSpec:
    case = parse_case(json.loads(GLM.read_text(encoding="utf-8")))
    base = case.model.quant
    if not declared:
        return replace(base, category_overrides={}, exclusions=(*base.exclusions, "norm"))
    return replace(
        base, exclusions=(*base.exclusions, "norm"),
        category_overrides={"norm": (("weight_format", stored),
                                     ("activation_format", "fp32"))},
    )


def _case(quant: QuantSpec):
    raw = json.loads(GLM.read_text(encoding="utf-8"))
    case = parse_case(raw)
    return replace(case, model=replace(case.model, quant=quant))


@lru_cache(maxsize=8)
def _native_norm(quant: QuantSpec):
    case = _case(quant)
    ops = target_operators(case, phase="prefill", tokens=1, context=0,
                           resource_fusion=False)
    return next(op for op in ops if op.op_id.endswith(".norm1")
                and op.local_shape == (1, WIDTH))


@lru_cache(maxsize=8)
def _piecewise_norm(quant: QuantSpec):
    case = _case(quant)
    features = FeatureConfig.from_runtime(case.runtime, case.model)
    pipeline = assemble_model(case.model, supply=PipeSupply.from_hardware(case.machine),
                              features=features)
    prediction = pipeline.forward(
        WorkloadGeometry((RequestGeometry(1, 0, phase="prefill"),), phase="prefill"),
        fusion=False)
    return next(row for row in prediction.rows
                if row.layer == 0 and row.kind == "N0"
                and row.cost.name == "rmsnorm")


def test_stored_fp32_gamma_is_provenance_not_a_fp32_kernel_price():
    quant = _quant(True, stored="fp32")
    assert dict(quant.norm_storage) == {"norm": "fp32"}
    assert dict(quant.norm_runtime) == {}
    assert quant.norm_gamma_format("norm") == "bf16"
    assert quant.norm_gamma_format("norm", source=True) == "fp32"
    assert quant.norm_gamma_bytes(WIDTH, "norm") == 12288.0
    assert quant.norm_gamma_bytes(WIDTH, "norm", source=True) == 24576.0
    assert quant.format_for("norm") == "bf16"
    assert quant.format_for("norm", source=True) == "fp32"
    assert quant.source_format_for("norm") == "fp32"
    assert quant.matrix_payload_bytes(1, WIDTH, "norm") == 12288.0
    assert quant.matrix_payload_bytes(1, WIDTH, "norm", source=True) == 24576.0


def test_norm_storage_round_trips_and_rejects_arbitrary_roles_or_formats():
    quant = _quant(True, stored="fp32")
    restored = QuantSpec(**json.loads(json.dumps(quant.to_dict())))
    assert restored.norm_storage == quant.norm_storage
    assert restored.norm_runtime == quant.norm_runtime
    assert restored.norm_gamma_bytes(WIDTH, "attn_norm", source=True) == 24576.0
    with pytest.raises(ModelingError):
        QuantSpec(name="bad", norm_storage={"arbitrary_norm": "fp32"})
    with pytest.raises(ModelingError):
        QuantSpec(name="bad", norm_storage={"norm": "float64"})


def test_native_and_piecewise_charge_one_bf16_gamma_once():
    quant = _quant(True, stored="fp32")
    native = _native_norm(quant)
    piece = _piecewise_norm(quant)
    assert native.dtype == "bf16"
    assert dict(native.metadata)["norm_gamma_runtime_format"] == "bf16"
    assert dict(native.metadata)["norm_gamma_storage_format"] == "fp32"
    assert dict(native.metadata)["norm_gamma_runtime_bytes"] == 12288.0
    assert dict(native.metadata)["norm_gamma_storage_bytes"] == 24576.0
    assert native.mte2_bytes == 24576.0       # activation 12288 + gamma 12288
    assert native.mte3_bytes == 12288.0
    assert native.hbm_bytes == 36864.0
    assert native.vector_ops > 0
    work = dict(piece.cost.work)
    assert float(work["mte2"]) == 24576.0
    assert float(work["mte3"]) == 12288.0
    assert float(work["hbm"]) == 36864.0
    assert float(work["vector"]) > 0
    derived = dict(piece.cost.derived)
    assert float(_float(derived["gamma_bytes"])) == 12288.0
    assert float(_float(derived["gamma_storage_bytes"])) == 24576.0
    assert _float(derived["ub_row_bytes"]) == 12288.0   # activation only
    assert _float(derived["ub_write_row_bytes"]) == 12288.0
    assert piece.cost.name == "rmsnorm"


def _float(value):
    return float(value) if not hasattr(value, "numerator") else float(value)


def test_disk_fp32_does_not_change_actual_work():
    fp32 = _quant(True, stored="fp32")
    bf16 = _quant(True, stored="bf16")
    native_fp32, native_bf16 = _native_norm(fp32), _native_norm(bf16)
    assert native_fp32.mte2_bytes == native_bf16.mte2_bytes
    assert native_fp32.mte3_bytes == native_bf16.mte3_bytes
    assert native_fp32.hbm_bytes == native_bf16.hbm_bytes
    assert native_fp32.vector_ops == native_bf16.vector_ops
    assert dict(native_fp32.metadata)["norm_gamma_storage_bytes"] == 24576.0
    assert dict(native_bf16.metadata)["norm_gamma_storage_bytes"] == 12288.0
    piece_fp32, piece_bf16 = _piecewise_norm(fp32), _piecewise_norm(bf16)
    assert dict(piece_fp32.cost.work) == dict(piece_bf16.cost.work)
    assert _float(dict(piece_fp32.cost.derived)["gamma_storage_bytes"]) == 24576.0
    assert _float(dict(piece_bf16.cost.derived)["gamma_storage_bytes"]) == 12288.0


@lru_cache(maxsize=8)
def _piecewise_fused(quant: QuantSpec):
    case = _case(quant)
    features = FeatureConfig.from_runtime(case.runtime, case.model)
    pipeline = assemble_model(case.model, supply=PipeSupply.from_hardware(case.machine),
                              features=features)
    prediction = pipeline.forward(
        WorkloadGeometry((RequestGeometry(1, 0, phase="prefill"),), phase="prefill"),
        fusion=True)
    rows = [row for row in prediction.rows if row.layer == 0 and row.kind == "N0"]
    fused = next(row for row in rows if row.cost.name == "fused:add_rmsnorm_quant")
    residual = next(row for row in rows if row.cost.name == "residual_add")
    return fused, residual


def test_fused_norm_path_keeps_exactly_one_bf16_gamma_read():
    fp32 = _quant(True, stored="fp32")
    bf16 = _quant(True, stored="bf16")
    fused_fp32, residual_fp32 = _piecewise_fused(fp32)
    fused_bf16, residual_bf16 = _piecewise_fused(bf16)
    assert dict(fused_fp32.cost.work) == dict(fused_bf16.cost.work)
    fused_work, residual_work = dict(fused_fp32.cost.work), dict(residual_fp32.cost.work)
    # Fusion removes the internal residual output but keeps the activation
    # read and exactly one 12288 B gamma read.
    assert float(fused_work["mte2"]) - float(residual_work["mte2"]) == 12288.0
    assert float(fused_work["hbm"]) - float(residual_work["hbm"]) == 24576.0
    assert float(fused_work["mte3"]) - float(residual_work["mte3"]) == 12288.0
    assert float(fused_work["vector"]) > 0
    assert float(residual_work["mte2"]) == 24576.0


def test_generic_declaration_does_not_promote_shared_or_engram_norms():
    quant = _quant(True, stored="fp32")
    for role in ("attn_norm", "ffn_norm", "kv_a", "q_a", "indexer_k", "final"):
        assert quant.norm_gamma_format(role) == "bf16"
        assert quant.norm_gamma_format(role, source=True) == "fp32"
    for role in ("shared_head", "enorm", "hnorm"):
        assert quant.norm_gamma_format(role) == "bf16"
        assert quant.norm_gamma_format(role, source=True) == "bf16"


def test_heldout_width_scales_only_the_source_gamma_component():
    quant = _quant(True, stored="fp32")
    assert quant.norm_gamma_bytes(512, "kv_a") == 1024.0
    assert quant.norm_gamma_bytes(512, "kv_a", source=True) == 2048.0
    assert quant.norm_gamma_bytes(6144, "attn_norm") == 12288.0
    assert quant.norm_gamma_bytes(6144, "attn_norm", source=True) == 24576.0


def test_absent_declaration_preserves_existing_bf16_actual():
    quant = _quant(False)
    assert not quant.norm_gamma_declared("norm")
    native = _native_norm(quant)
    piece = _piecewise_norm(quant)
    assert dict(native.metadata)["norm_gamma_runtime_format"] == "bf16"
    assert native.mte2_bytes == 24576.0
    assert float(dict(piece.cost.work)["mte2"]) == 24576.0
    assert float(_float(dict(piece.cost.derived)["gamma_bytes"])) == 12288.0
