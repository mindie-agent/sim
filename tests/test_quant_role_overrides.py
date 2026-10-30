"""Role-specific representation must agree across execution and residency."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.catalog.weights import layer_weight_terms
from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.transforms import activation_quantization


def quant_plan():
    return QuantSpec(
        "mixed", weight_block=(1, 32), expert_block=32,
        scale_format="fp8_e8m0", scale_bytes=1,
        activation_group_size=32, activation_scale_format="fp8_e8m0",
        activation_scale_pack=2,
        category_overrides={
            "projection": {"weight_format": "fp8_e4m3", "activation_format": "fp8_e4m3"},
            "projection.gate": {"weight_format": "bf16", "activation_format": "bf16"},
            "shared_expert": {"weight_format": "mxfp4", "activation_format": "fp8_e4m3"},
        },
    )


def test_role_inheritance_retains_independent_scale_geometry_after_json_roundtrip():
    quant = quant_plan()
    restored = QuantSpec(**json.loads(json.dumps(quant.to_dict())))
    for plan in (quant, restored):
        # Five rows, three scale groups per row; FP4 row padding is separate.
        assert plan.matrix_payload_bytes(5, 65, "projection.q") == 325
        assert plan.matrix_bytes(5, 65, "projection.q") == 340
        assert plan.matrix_bytes(5, 65, "projection.gate") == 650
        assert plan.matrix_payload_bytes(5, 65, "shared_expert") == 165
        assert plan.matrix_bytes(5, 65, "shared_expert") == 180
        assert plan.matrix_bytes(5, 65, "shared_expert", source=True) == 180
    assert quant == restored
    assert replace(quant, exclusions=("projection",)).matrix_bytes(5, 65, "projection.q") == 650


def test_fp4_preserves_declared_fp8_payload_and_real_mx_quantization_producer():
    op = gemm_op(
        op_id="shared", deps=(), stage=0, layer=0, requests=(),
        m=3, n=5, k=65, quant=quant_plan(), category="shared_expert", phase="prefill",
    )
    meta = dict(op.metadata)
    assert op.dtype == "bf16"  # Explicit expanded compute; no FP4 peak inferred.
    assert op.flops == 1950
    assert meta["weight_format"] == "mxfp4"
    assert meta["activation_format"] == "fp8_e4m3"
    assert meta["activation_scale_shape"] == (3, 2, 2)
    assert meta["activation_scale_bytes"] == 12
    assert op.mte2_bytes == 195 + 12 + 180
    assert op.mte3_bytes == 30
    assert op.hbm_bytes == 417
    quantize, matrix = activation_quantization((op,), "bf16")
    assert matrix.deps == (quantize.op_id,)
    assert quantize.mte2_bytes == 390
    assert quantize.mte3_bytes == 207
    assert quantize.vector_ops == 798
    assert dict(quantize.metadata)["activation_scale_format"] == "fp8_e8m0"


def test_checkpoint_roles_use_the_same_representation_in_ops_and_weight_terms():
    path = Path(__file__).parents[1] / "examples/cases/kimi-k3-a5-current-checkpoint-conditional.json"
    raw = json.loads(path.read_text())
    raw["model"]["layers"] = 4  # KDA+dense, KDA+MoE and MLA+MoE.
    case = parse_case(raw)
    restored = parse_case({**raw, "model": case.model.to_dict()})
    assert case.model.quant == restored.model.quant
    ops = target_operators(restored, phase="decode", tokens=1, context=32)
    expected = {
        "kda.q": ("kda_q", "fp8_e4m3"), "kda.k": ("kda_k", "fp8_e4m3"),
        "kda.v": ("kda_v", "fp8_e4m3"), "kda.g": ("kda_g", "bf16"),
        "kda.b": ("kda_b", "bf16"), "kda.f_a": ("kda_f_a", "bf16"),
        "kda.f_b": ("kda_f_b", "bf16"), "kda.o": ("kda_o", "bf16"),
        "mla.q_a": ("q_a", "fp8_e4m3"), "mla.q_b": ("q_b", "fp8_e4m3"),
        "mla.kv_a": ("kv_a", "fp8_e4m3"), "mla.g": ("mla_g", "bf16"),
        "mla.o": ("o_proj", "bf16"),
    }
    seen = set()
    for op in ops:
        meta = dict(op.metadata)
        category = meta.get("quant_category")
        if category not in expected:
            continue
        term_name, fmt = expected[category]
        terms = layer_weight_terms(case.model.layer_plans[op.layer], case.model)
        weight = next(term for term in terms if term.name == term_name)
        assert meta["weight_format"] == fmt
        assert meta["packed_weight_bytes"] == term_bytes(
            weight, case.model.quant, case.runtime.parallel)
        if fmt == "fp8_e4m3":
            assert meta["activation_producer"] == "explicit_quantization"
        seen.add(category)
    assert seen == expected.keys()
    matrices = [op for op in ops if dict(op.metadata).get("quant_category") == "shared_expert"]
    assert matrices
    assert all(dict(op.metadata)["weight_format"] == "mxfp4" for op in matrices)
    assert all(dict(op.metadata)["activation_format"] == "fp8_e4m3" for op in matrices)
